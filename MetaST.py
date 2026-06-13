# =============================================================================
# IMPORTS
# =============================================================================

# Standard library
import time
import math
import re
import random
import bisect
import ast
import json
import copy
import pickle
import inspect
import collections
import os, json, hashlib
from pathlib import Path
from collections import defaultdict, deque, Counter
from contextlib import nullcontext
from functools import lru_cache
from typing import Optional, Sequence, Tuple, Dict, List, Any
from concurrent.futures import ThreadPoolExecutor

# Scientific computing
import numpy as np
import pandas as pd
import scipy
from scipy.sparse import csr_matrix, vstack
from scipy.optimize import nnls
from scipy.spatial import cKDTree

# Machine learning
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Geospatial
import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString, MultiLineString, Point
from shapely.strtree import STRtree
from shapely.geometry.base import BaseGeometry
from shapely import wkb as shapely_wkb


# Visualization 
from folium.plugins import HeatMap
from pyrosm import OSM
from tqdm import tqdm

# PyTorch core
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# PyTorch AMP
try:
    from torch.amp import GradScaler, autocast
except ImportError:
    from torch.cuda.amp import GradScaler, autocast

# PyTorch Geometric
from torch_geometric.data import Data
from torch_sparse import SparseTensor


import LinkETA as _linketa_mod
from LinkETA import one_epoch, lam_g_at
from loss_functions import (
    link_loss_huber_macro, route_loss_mape,
    link_metrics_micro, route_metrics_pooled, finalize_pct_metrics,
    selection_metric, L_MIN_METRIC, T_MIN_METRIC,
    compute_global_losses_and_metrics,
    # [2026-06-06] Controlled-backbone alignment: MetaST now optimizes the SAME
    # LinkETA backbone losses as MetaTTE / MetaTraffic / base LinkETA so the
    # backbone confers no per-method advantage (fair comparison). Only the
    # meta-strategy (FOMAML + STMemory clustering) differs.
    link_eta_in_seconds, link_loss_huber_log_lenweighted, route_loss_huber_log,
)
from checkpoint import CheckpointManager, write_training_history


# =============================================================================
# [2026-06-06] Canonical LinkETA speed loss vs sp_actual (mirrors base
# LinkETA one_epoch). Shared by every forward+loss site in this file so the
# speed-supervision target is identical across MetaST / MetaTTE / MetaTraffic.
#   sp_actual = link_len / (link_tt · v_ff), clamped to r_max; rows pinned at the
#   r_max floor by Stage-1 redistribution are excluded from the speed loss.
# =============================================================================
def _backbone_speed_loss(sp_pred, seq_link_len, seq_true_tt, vff_seq, seq_mask,
                         vff_seq_clean=None, r_max=3.0):
    if vff_seq_clean is None:
        vff_seq_clean = vff_seq
    sp_actual = (seq_link_len / seq_true_tt.clamp_min(1e-3)) / vff_seq.clamp_min(1.0)
    sp_actual = sp_actual.clamp(0.0, r_max)
    sp_actual_clean = (seq_link_len / seq_true_tt.clamp_min(1e-3)) / vff_seq_clean.clamp_min(1.0)
    was_pinned = sp_actual_clean >= (r_max - 1e-3)
    sp_w_actual = (
        seq_mask
        & (seq_true_tt >= 1.0)
        & (seq_link_len >= 10.0)
        & (~was_pinned)
    ).float()
    denom = sp_w_actual.sum().clamp_min(1.0)
    per = F.smooth_l1_loss(sp_pred, sp_actual, reduction="none", beta=0.1)
    return (per * sp_w_actual).sum() / denom

# ---------------------------------------------------------------------------
# Module-level sentinels for MetaST-only globals
# (set by set_globals_from_core; read by _route_cluster_id_from_globals, etc.)
# ---------------------------------------------------------------------------
model = None                 # will hold LinkETAWithSTMemory once created
EDGE_CLUSTER_LABELS = None
CORE = None                  # [MetaETA stateless] active core; passed to one_epoch(CORE, ...)


def extract_edge_patterns_from_profile(
    traffic_profile: pd.DataFrame,
    edges_gdf: pd.DataFrame,
    speed_prefix: str = "median_speed",
    fallback_vff: float = 10.0,
    normalize_per_edge: bool = True,
) -> pd.DataFrame:
    """
    Derive 24-hour speed-ratio pattern vectors from the precomputed
    link_traffic_profile.parquet — no raw trip reprocessing needed.

    Steps:
      1. Average median_speed across all 7 days-of-week for each (edge, hour)
         using the first 5-min slot (slot 0) as the representative speed.
      2. Divide by free-flow speed → speed_ratio = v_obs / v_ff.
      3. Optionally normalize per-edge (emphasises diurnal shape over magnitude).

    Args:
        traffic_profile: DataFrame with columns
            [way_id, day_of_week, hour_of_day, median_speed_0, ...]
            One row per (way_id, dow, hod) — 168 rows per edge.
        edges_gdf: Edge GeoDataFrame with [way_id|id|edge_id, edge_idx, v_ff].
        speed_prefix: Which dynamic metric to use (default "median_speed").
        fallback_vff: Fallback free-flow speed (m/s) for edges missing v_ff.
        normalize_per_edge: If True, divide each 24-dim vector by its mean
            so the pattern captures shape, not magnitude.

    Returns:
        DataFrame indexed by edge_idx with columns h0..h23 (float32).
    """
    tp = traffic_profile.copy()

    # ---- resolve way_id column in edges_gdf ----
    eid_col = None
    for c in ["way_id", "id", "edge_id", "osmid"]:
        if c in edges_gdf.columns:
            eid_col = c
            break
    if eid_col is None:
        raise ValueError("edges_gdf has no recognisable way-id column (way_id/id/edge_id/osmid)")

    # Build way_id → edge_idx mapping
    id2edge = dict(
        zip(
            edges_gdf[eid_col].astype(str),
            edges_gdf["edge_idx"].astype(int),
        )
    )

    tp["edge_idx"] = tp["way_id"].astype(str).map(id2edge)
    tp = tp.dropna(subset=["edge_idx"])
    tp["edge_idx"] = tp["edge_idx"].astype(int)

    # ---- representative speed per (edge, hour): slot-0 of speed_prefix ----
    speed_col = f"{speed_prefix}_0"
    if speed_col not in tp.columns:
        raise ValueError(f"Column '{speed_col}' not found in traffic_profile. "
                         f"Available: {[c for c in tp.columns if speed_prefix in c]}")

    tp["_speed"] = pd.to_numeric(tp[speed_col], errors="coerce").fillna(0.0)

    # Average across all 7 days-of-week → one value per (edge, hour)
    avg = tp.groupby(["edge_idx", "hour_of_day"])["_speed"].mean().reset_index()
    avg.columns = ["edge_idx", "hour", "avg_speed"]

    # ---- free-flow speed per edge ----
    if "v_ff" in edges_gdf.columns:
        vff_map = dict(
            zip(
                edges_gdf["edge_idx"].astype(int),
                pd.to_numeric(edges_gdf["v_ff"], errors="coerce"),
            )
        )
    else:
        vff_map = {}

    avg["v_ff"] = avg["edge_idx"].map(vff_map).fillna(fallback_vff).clip(lower=1.0)

    # ---- speed ratio: v_obs / v_ff  (0 speed → ratio 0, filled later) ----
    avg["speed_ratio"] = (avg["avg_speed"] / avg["v_ff"]).clip(0.0, 5.0)

    # ---- pivot to 24-hour vector ----
    pivot = avg.pivot(index="edge_idx", columns="hour", values="speed_ratio")

    for h in range(24):
        if h not in pivot.columns:
            pivot[h] = np.nan
    pivot = pivot[list(range(24))]
    pivot.columns = [f"h{h}" for h in range(24)]

    # Fill missing hours: edge mean → global mean
    edge_means = pivot.mean(axis=1)
    global_mean = pivot.stack().mean()
    for col in pivot.columns:
        pivot[col] = pivot[col].fillna(edge_means)
    pivot = pivot.fillna(global_mean if pd.notna(global_mean) else 1.0)

    # Replace pure-zero rows (edges with no traffic at all) with global mean
    zero_mask = (pivot.sum(axis=1) == 0.0)
    if zero_mask.any():
        gm = global_mean if pd.notna(global_mean) else 1.0
        pivot.loc[zero_mask] = gm

    # ---- optional shape normalisation ----
    if normalize_per_edge:
        row_mean = pivot.mean(axis=1).replace(0.0, 1.0)
        pivot = pivot.div(row_mean, axis=0)

    print(f"[extract_edge_patterns_from_profile] {len(pivot)} edges, "
          f"speed_col={speed_col}, normalize={normalize_per_edge}")
    return pivot.astype(np.float32)

def cluster_edge_patterns(
    pattern_dfs: Dict[str, pd.DataFrame],
    n_clusters: int,
    random_state: int = 42,
) -> Tuple[KMeans, StandardScaler, Dict[str, pd.Series]]:
    """
    Cluster edge patterns across multiple cities using KMeans.

    Assumes each pattern_df is already per-edge normalized (shape-first).
    Then applies StandardScaler across pooled edges before KMeans.

    Returns:
      kmeans, scaler, cluster_labels dict(city -> Series(edge_idx -> cluster_id))
    """
    # Pool all patterns
    all_patterns = []
    edge_city_idx = []

    for city_name, pdf in pattern_dfs.items():
        if pdf is None or len(pdf) == 0:
            continue

        # Force expected columns and stable ordering
        expected_cols = [f"h{h}" for h in range(24)]
        missing = [c for c in expected_cols if c not in pdf.columns]
        if missing:
            raise ValueError(f"{city_name} pattern_df missing columns: {missing}")

        pdf2 = pdf[expected_cols].copy()
        patterns = pdf2.values
        all_patterns.append(patterns)

        for eidx in pdf2.index:
            edge_city_idx.append((city_name, int(eidx)))

    if len(all_patterns) == 0:
        raise ValueError("No patterns provided to cluster_edge_patterns().")

    all_patterns = np.vstack(all_patterns)
    all_patterns = np.nan_to_num(all_patterns, nan=1.0)

    # Standardize across pooled edges (per-hour features)
    scaler = StandardScaler()
    patterns_scaled = scaler.fit_transform(all_patterns)

    # KMeans
    kmeans = KMeans(n_clusters=int(n_clusters), random_state=random_state, n_init=20)
    labels = kmeans.fit_predict(patterns_scaled)

    # Map back to per-city Series
    cluster_labels = {city: {} for city in pattern_dfs.keys()}
    for i, (city_name, edge_idx) in enumerate(edge_city_idx):
        cluster_labels[city_name][edge_idx] = int(labels[i])

    cluster_labels = {
        city: pd.Series(d, name="cluster_id", dtype="int64")
        for city, d in cluster_labels.items()
    }

    return kmeans, scaler, cluster_labels

def choose_k_silhouette(
    source_pattern_dfs: Dict[str, pd.DataFrame],
    k_min: int = 4,
    k_max: int = 10,
    random_state: int = 42,
) -> Tuple[int, pd.DataFrame]:
    """
    Choose K using silhouette score on pooled source-city patterns.

    Returns:
      best_k, scores_df (columns: k, silhouette)
    """
    expected_cols = [f"h{h}" for h in range(24)]
    pooled = []

    for city, pdf in source_pattern_dfs.items():
        if pdf is None or len(pdf) == 0:
            continue
        pooled.append(pdf[expected_cols].values)

    if len(pooled) == 0:
        raise ValueError("No source patterns provided to choose_k_silhouette().")

    X = np.vstack(pooled)
    X = np.nan_to_num(X, nan=1.0)

    # Standardize before silhouette + kmeans
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    n = Xs.shape[0]
    k_max_eff = min(int(k_max), max(2, n - 1))
    k_min_eff = max(2, int(k_min))
    if k_min_eff > k_max_eff:
        raise ValueError(f"Not enough samples for silhouette: n={n}, need at least k in [2..n-1].")

    rows = []
    best_k = None
    best_score = -1e9

    for k in range(k_min_eff, k_max_eff + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=20)
        labels = km.fit_predict(Xs)

        # Silhouette requires at least 2 clusters actually used
        if len(np.unique(labels)) < 2:
            continue

        s = silhouette_score(Xs, labels, metric="euclidean")
        rows.append({"k": k, "silhouette": float(s)})

        if s > best_score:
            best_score = s
            best_k = k

    scores_df = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)

    if best_k is None:
        raise ValueError("Silhouette failed to select a K (degenerate clustering).")
    
    
    return int(best_k), scores_df

def assign_target_clusters(
    target_patterns: pd.DataFrame,
    kmeans: KMeans,
    scaler: StandardScaler,
) -> pd.Series:
    """
    Assign cluster labels to target city edges using trained KMeans.
    
    Args:
        target_patterns: Pattern DataFrame for target city
        kmeans: Fitted KMeans from source cities
        scaler: Fitted StandardScaler from source cities
    
    Returns:
        Series(edge_idx -> cluster_id)
    """
    patterns = target_patterns.values
    patterns = np.nan_to_num(patterns, nan=1.0)
    patterns_scaled = scaler.transform(patterns)
    labels = kmeans.predict(patterns_scaled)
    
    return pd.Series(labels, index=target_patterns.index, name='cluster_id')


def build_edge_cluster_tensor(
    cluster_labels: pd.Series,
    n_edges: int,
    default_cluster: int = 0,
) -> torch.Tensor:
    """
    Build a tensor mapping edge_idx -> cluster_id.
    
    Args:
        cluster_labels: Series(edge_idx -> cluster_id)
        n_edges: Total number of edges
        default_cluster: Default cluster for edges without patterns
    
    Returns:
        LongTensor of shape [n_edges] with cluster ids
    """
    labels = torch.full((n_edges,), default_cluster, dtype=torch.long)
    for edge_idx, cluster_id in cluster_labels.items():
        if 0 <= edge_idx < n_edges:
            labels[edge_idx] = int(cluster_id)
    return labels


# ============================================================
# Pattern Visualization Helper (optional, for debugging)
# ============================================================

def visualize_cluster_patterns(
    pattern_df: pd.DataFrame,
    cluster_labels: pd.Series,
    n_clusters: int = None,
):
    """
    Visualize average pattern per cluster.
    """
    import matplotlib.pyplot as plt
    
    df = pattern_df.copy()
    df['cluster'] = df.index.map(cluster_labels)
    df = df.dropna(subset=['cluster'])
    
    fig, axes = plt.subplots(1, n_clusters, figsize=(4*n_clusters, 3), sharey=True)
    if n_clusters == 1:
        axes = [axes]
    
    hours = list(range(24))
    for c in range(n_clusters):
        ax = axes[c]
        subset = df[df['cluster'] == c]
        if len(subset) == 0:
            continue
        
        pattern_cols = [f'h{h}' for h in hours]
        mean_pattern = subset[pattern_cols].mean().values
        std_pattern = subset[pattern_cols].std().values
        
        ax.plot(hours, mean_pattern, 'b-', linewidth=2)
        ax.fill_between(hours, mean_pattern - std_pattern, mean_pattern + std_pattern, alpha=0.3)
        ax.set_title(f'Cluster {c} (n={len(subset)})')
        ax.set_xlabel('Hour of Day')
        if c == 0:
            ax.set_ylabel('Speed Ratio')
        ax.set_xlim(0, 23)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# ============================================================
# Multi-City Pattern Clustering (Run after all cities processed)
# ============================================================
# This cell should be run ONCE after you have extracted patterns
# from all SOURCE cities. 
#
# Workflow:
# 1. Load patterns from source cities (Porto, San Francisco)
# 2. Cluster across source cities
# 3. Assign labels to all cities (including target: Busan)
# 4. Save the clustering artifacts
# ============================================================

def run_metast_clustering_pipeline(
    source_pattern_dfs: Dict[str, pd.DataFrame],
    target_pattern_dfs: Dict[str, pd.DataFrame],
    n_edges_per_city: Dict[str, int],
    n_clusters = "auto",
    k_min: int = 2,
    k_max: int = 10,
):
    """
    Full pattern clustering pipeline with silhouette-based K selection on source cities.

    If n_clusters == "auto" or None:
      - choose K via silhouette on pooled source patterns in [k_min, k_max]
      - then fit KMeans on pooled source patterns
      - assign target labels with same scaler + kmeans
      - build per-city edge_cluster_tensors (size = total graph edges)
    """
    if n_clusters is None or n_clusters == "auto":
        best_k, sil_df = choose_k_silhouette(source_pattern_dfs, k_min=k_min, k_max=k_max)
        print("Silhouette scores by K:")
        print(sil_df)
        n_clusters = best_k
        print(f"Chosen K via silhouette: {n_clusters}")

    n_clusters = int(n_clusters)
    G_PATTERN_CLUSTERS = n_clusters

    print(f"✅ G_PATTERN_CLUSTERS set to {G_PATTERN_CLUSTERS}")

    n_clusters = int(n_clusters)
    print(f"Clustering with K={n_clusters} clusters across {len(source_pattern_dfs)} source cities")

    # 1) Fit clustering on source cities
    kmeans, scaler, source_labels = cluster_edge_patterns(
        source_pattern_dfs,
        n_clusters=n_clusters,
    )

    print("Source city cluster distributions:")
    for city, labels in source_labels.items():
        if labels is None or len(labels) == 0:
            print(f"  {city}: empty")
        else:
            print(f"  {city}: {dict(labels.value_counts().sort_index())}")

    # 2) Assign labels to target cities
    target_labels = {}
    for city, patterns in target_pattern_dfs.items():
        target_labels[city] = assign_target_clusters(patterns, kmeans, scaler)
        print(f"  {city} (target): {dict(target_labels[city].value_counts().sort_index())}")

    # 3) Build tensors for all cities
    all_labels = {**source_labels, **target_labels}
    edge_cluster_tensors = {}

    for city, labels in all_labels.items():
        n_edges = int(n_edges_per_city[city])
        edge_cluster_tensors[city] = build_edge_cluster_tensor(labels, n_edges, default_cluster=0)
        print(f"  {city} tensor: shape={tuple(edge_cluster_tensors[city].shape)}")

    return {
        "kmeans": kmeans,
        "scaler": scaler,
        "source_labels": source_labels,
        "target_labels": target_labels,
        "edge_cluster_tensors": edge_cluster_tensors,
        "n_clusters": n_clusters,
    }

def get_n_edges_from_core(core):
    """Extract number of edges from a core dictionary."""
    edges_vff = core['graph']['edges_vff']
    return int(edges_vff['edge_idx'].max()) + 1


class STMemory(nn.Module):
    """
    MetaST Spatial-Temporal Memory.

    Paper-faithful mechanics:
    - Query: v = W_l h + b_l
    - Attention: softmax(<v, M(g)>)  (no 1/sqrt(d) scaling in the paper)
    - Read: z = sum_g p(g) M(g)
    - Loss: cross-entropy between attention logits and fixed cluster labels
    """
    def __init__(self, hidden_dim: int, d_mem: int = 8, G: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.d_mem = d_mem
        self.G = G

        self.query_proj = nn.Linear(hidden_dim, d_mem)
        self.prototypes = nn.Parameter(0.02 * torch.randn(G, d_mem))

    def forward(
        self,
        h_seq: torch.Tensor,                 # [B, L, hidden_dim]
        seq_mask: torch.Tensor,              # [B, L] bool
        cluster_labels: Optional[torch.Tensor] = None,  # [B, L] long in [0..G-1]
    ):
        B, L, D = h_seq.shape

        # Eq. 9
        v = self.query_proj(h_seq)  # [B, L, d_mem]

        # Eq. 10 (paper form, no sqrt scaling)
        logits = torch.einsum("bld,gd->blg", v, self.prototypes)  # [B, L, G]
        att = F.softmax(logits, dim=-1)  # [B, L, G]

        # Eq. 11
        z = torch.einsum("blg,gd->bld", att, self.prototypes)  # [B, L, d_mem]

        # Eq. 13 (cross-entropy with fixed labels)
        mem_loss = None
        if cluster_labels is not None:
            m = seq_mask.bool()
            if m.any():
                logits_flat = logits.view(-1, self.G)
                labels_flat = cluster_labels.view(-1)
                mask_flat = m.view(-1)

                ce = F.cross_entropy(logits_flat, labels_flat, reduction="none")
                mem_loss = ce[mask_flat].mean()
            else:
                mem_loss = torch.zeros((), device=h_seq.device)

        return z, mem_loss, att


class LinkETAWithSTMemory(nn.Module):
    """
    Wrapper for LinkETA with MetaST memory.

    Paper-faithful fusion:
    - Eq. 12 uses tanh(W'_n [h; z] + b'_n)
    Here:
    - fuse([h_seq; z_mem]) -> hidden_dim with a single Linear + Tanh
    """
    def __init__(self, base_model: nn.Module, mem: STMemory):
        super().__init__()
        self.base = base_model
        self.mem = mem

        self.last_mem_loss = None
        self.last_att = None

        hidden_dim = base_model.hidden_dim
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + mem.d_mem, hidden_dim),
            nn.Tanh(),
        )

    def forward(
        self,
        x_seq_norm,
        hop_idx,
        route_flag_nodes,
        masks_per_rel,
        route_seq_idx,
        seq_link_len,
        vff_seq_b,
        seq_mask,
        node_edge_idx: Optional[torch.Tensor] = None,
        route_cluster_id: Optional[torch.Tensor] = None,  # [B, L] fixed labels for this city
        compute_mem_loss: bool = True,
        return_mem_loss: bool = False,
        **kwargs
    ):
        B, M, _ = x_seq_norm.size()
        L = route_seq_idx.size(1)

        x_seq_norm = torch.nan_to_num(x_seq_norm)

        # 1) Base encoders (unchanged)
        h = self.base.input_proj(x_seq_norm)
        h_dyn = self.base._encode_dyn_slots(x_seq_norm)
        h = h + h_dyn
        h = self.base.inp_dropout(h)
        h = self.base.route_enc(h, hop_idx, route_flag_nodes.long())

        for layer in self.base.layers:
            h = layer(h, masks_per_rel)  # [B, M, D]

        # 2) Gather sequence representation h_seq
        D = h.size(-1)
        gather_idx = route_seq_idx.unsqueeze(-1).expand(-1, -1, D)
        h_seq = torch.gather(h, 1, gather_idx)  # [B, L, D]

        # 3) Memory (always used), loss only if labels given and compute_mem_loss True
        cluster_labels = route_cluster_id if (compute_mem_loss and route_cluster_id is not None) else None
        z_mem, mem_loss, att = self.mem(h_seq, seq_mask, cluster_labels)

        self.last_mem_loss = mem_loss
        self.last_att = att

        # Eq. 12-style fusion
        h_seq_fused = self.fusion(torch.cat([h_seq, z_mem], dim=-1))  # [B, L, D]

        # 4) Heads (unchanged)
        enc_t = h_seq_fused.transpose(1, 2)  # [B, D, L]

        speed_1h_raw = self.base.speed_head(enc_t).squeeze(1)
        logcnt_1h = self.base.count_head(enc_t).squeeze(1)
        r_max = kwargs.get("r_max", 3.0)
        sp_hat = torch.sigmoid(speed_1h_raw) * r_max

        seq_link_len = torch.nan_to_num(seq_link_len).clamp_min(1e-3)
        vff_seq_b = torch.nan_to_num(vff_seq_b).clamp_min(1.0)
        eta_base = seq_link_len / (sp_hat.clamp_min(1e-3) * vff_seq_b)

        # 5) Log-space residual  (canonical: 2.0 * tanh → delta ∈ [-2,2])
        res_raw = self.base.residual_head(enc_t).squeeze(1)
        res_raw = torch.nan_to_num(res_raw)
        delta = 2.0 * torch.tanh(res_raw)                              # [B,L]
        eta_phys = eta_base * torch.exp(delta)

        # 6) Route-level calibration
        seq_mask_f = seq_mask.float()
        route_repr = (h_seq_fused * seq_mask_f.unsqueeze(-1)).sum(dim=1) / \
                     seq_mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
        g_raw = self.base.route_calib(route_repr).squeeze(-1)
        g = 1.0 + self.base.g_range * torch.tanh(g_raw)

        # 7) Final ETA
        link_eta_seq = eta_phys * g.unsqueeze(-1)
        link_eta_seq = link_eta_seq * seq_mask_f
        trip_eta = link_eta_seq.sum(dim=1)

        # Return 7 values (same as canonical LinkETA) + optional mem_loss
        if kwargs.get("return_parts", True):
            out = (
                link_eta_seq,  # [B,L]
                trip_eta,      # [B]
                sp_hat,        # [B,L]
                logcnt_1h,     # [B,L]
                g,             # [B]
                delta,         # [B,L]
                eta_phys,      # [B,L]
            )
        else:
            out = (link_eta_seq, trip_eta, sp_hat, logcnt_1h, g, delta, eta_phys)

        if return_mem_loss:
            return out + (mem_loss,)

        return out

def infer_hidden_dim_from_model(model):
    """Get hidden_dim from LinkETA model."""
    return model.hidden_dim


def infer_x_feat_dim_from_loader(loader):
    """Get input feature dimension from data loader."""
    b = next(iter(loader))
    return b["x_seq"].shape[-1]


G_PATTERN_CLUSTERS = 4          # paper Section 5.1.4: "cluster … to 4 groups"
def create_metast_model(
    base_model: nn.Module,
    edge_cluster_labels: torch.Tensor = None,   # vestigial, kept for back-compat
    G: int = G_PATTERN_CLUSTERS,
    d_mem: int = 8,
    device: torch.device = None,
):
    # NOTE: edge_cluster_labels is no longer used here — cluster labels
    # are passed per-forward-call via route_cluster_id (looked up from
    # the module-level EDGE_CLUSTER_LABELS global).
    hidden_dim = infer_hidden_dim_from_model(base_model)

    mem = STMemory(
        hidden_dim=hidden_dim,
        d_mem=d_mem,
        G=G,
    )

    wrapped = LinkETAWithSTMemory(
        base_model=base_model,
        mem=mem,
    )

    if device is not None:
        wrapped = wrapped.to(device)

    return wrapped


def _route_cluster_id_from_globals(node_edge_idx, route_seq_idx):
    if EDGE_CLUSTER_LABELS is None:
        return None
    edge_seq = torch.gather(node_edge_idx, 1, route_seq_idx)
    edge_seq = edge_seq.clamp(0, EDGE_CLUSTER_LABELS.size(0) - 1)
    return EDGE_CLUSTER_LABELS[edge_seq].long()

# ---------------------------------------------------------------------
# 0) REQUIRED: swap globals by core  (Problem-B reconciliation)
# ---------------------------------------------------------------------
#   activate_core() (LinkETA_Fixed) sets the globals that one_epoch()
#   reads from its own module namespace.  set_globals_from_core() calls
#   it, then layers on the extras that only MetaST functions need.
# ---------------------------------------------------------------------
def set_globals_from_core(core, keep_model=True):
    """
    Swap all module-level globals to *core* so that both LinkETA-side
    functions (one_epoch, lam_g_at) and MetaST-side functions
    (one_epoch_query_backward_with_mem, inner_adapt_steps, …) operate
    on the correct city.

    Parameters
    ----------
    core : dict
        City core built by ``linketa_get_core_components``.
    keep_model : bool
        If True the current ``model`` is preserved (MetaST wraps a
        shared model around per-city data).  If False the per-core
        base model is restored.
    """
    # --- MetaST-side globals ---
    global device, model, optimizer, scheduler
    global train_loader, val_loader, test_loader
    global train_ds, val_ds, test_ds
    global inst, INST_FEAT_COLS, STATIC_FEAT_COLS, TIME_FEAT_COLS, DYN_FEAT_COLS
    global edges_vff, rel_edges, vff_vec, EDGE_CLUSTER_LABELS
    global DYN_STORE, CFG
    global H_SLICES, BIN_SECONDS, L_MIN, T_MIN
    global SCALER_MEAN, SCALER_STD, DYN_IDX
    global HIST, num_epochs, clip_norm, ckpt
    global CORE

    # [MetaETA stateless] LinkETA's one_epoch now takes `core` explicitly (no
    # activate_core / LinkETA module globals). We keep a handle to the active
    # core and make sure core["model"] points at the model one_epoch must run.
    CORE = core
    if keep_model:
        # `model` holds the MetaST-wrapped model from a previous assignment
        # (e.g. main.py: MetaST.model = wrapped_model). Run THAT model.
        core["model"] = model
    else:
        model = core["model"]

    device = core["device"]

    # loaders / datasets
    train_loader = core["loaders"]["train"]
    val_loader   = core["loaders"]["val"]
    test_loader  = core["loaders"]["test"]
    train_ds     = core["datasets"]["train"]
    val_ds       = core["datasets"]["val"]
    test_ds      = core["datasets"]["test"]

    # tables / column lists
    inst             = core["inst_table"]
    INST_FEAT_COLS   = core["inst_feat_cols"]
    STATIC_FEAT_COLS = core["static_cols"]
    TIME_FEAT_COLS   = core["time_cols"]
    DYN_FEAT_COLS    = core["dyn_cols"]

    # graph (also in LinkETA via activate_core; kept here for MetaST funcs)
    edges_vff = core["graph"]["edges_vff"]
    rel_edges = core["graph"]["rel_edges"]
    vff_vec   = core["graph"]["vff_vec"]

    # MetaST-only: per-city edge cluster labels
    EDGE_CLUSTER_LABELS = core["graph"].get("edge_cluster_labels", None)

    # dyn + cfg
    DYN_STORE = core["dyn_store"]
    CFG       = core["cfg"]

    # constants
    H_SLICES    = core["constants"]["H_SLICES"]
    BIN_SECONDS = core["constants"]["BIN_SECONDS"]
    L_MIN       = core["constants"]["L_MIN"]
    T_MIN       = core["constants"]["T_MIN"]

    # scalers (also in LinkETA via activate_core)
    SCALER_MEAN = core["scalers"]["mean"]
    SCALER_STD  = core["scalers"]["std"]
    DYN_IDX     = core["scalers"]["dyn_idx"]

    # misc
    optimizer  = core.get("optimizer", None)
    scheduler  = core.get("scheduler", None)
    HIST       = core.get("history", None)
    num_epochs = core.get("num_epochs", None)
    clip_norm  = core.get("clip_norm", None)
    ckpt       = core.get("checkpoint_manager", None)


# -----------------------------
# State helpers
# -----------------------------
def set_requires_grad(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag

def one_epoch_query_backward_with_mem(loader, epoch=0, w_speed=0.5, w_link=0.5, w_route=0.5, lambda_mem=1e-4):
    """
    Query (outer) loop: compute loss on validation set with memory clustering loss.
    
    MetaST-faithful changes:
    1. Use train mode (consistent with inner loop)
    2. Do NOT use force_physics (match inference behavior)
    3. Pass node_edge_idx for cluster label lookup
    4. Properly include clustering loss in outer objective
    """
    model.train()  # FIX: Use train mode for consistency with inner loop
    model.zero_grad(set_to_none=True)

    # accumulators (same keys as your one_epoch return)
    sum_link_loss = sum_route_loss = 0.0
    sum_mae_link = sum_mape_link = 0.0
    sum_sse_link = 0.0
    N_links = N_links_mape = 0

    sum_mae_route = sum_mape_route = 0.0
    sum_sse_route = 0.0
    N_routes_loss = N_routes_metrics = N_routes_mape = 0

    sum_speed_mae = sum_vol_mae = 0.0
    N_speed = N_vol = 0

    n_batches = 0
    total_final_loss = 0.0

    # No AMP here for stability (you can add it later)
    for batch in loader:
        n_batches += 1

        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        node_edge_idx = batch["node_edge_idx"]
        x_seq_nodes = batch["x_seq"]
        route_flag_nodes = batch["route_flag"]
        hop_idx = batch["hop_idx"]
        masks_per_rel = [m.to(device) for m in batch["masks_per_rel"]]

        route_seq_idx = batch["route_seq_idx"]
        seq_true_tt = torch.nan_to_num(batch["seq_true_tt"], nan=0.0)
        seq_link_len = torch.nan_to_num(batch["seq_link_len"], nan=0.0).clamp_min(1e-3)
        seq_mask = batch["seq_mask"].bool()
        trip_tt = torch.nan_to_num(batch["trip_tt"], nan=0.0)

        sp_true_node = torch.nan_to_num(batch["sp_ratio_agg"], nan=0.0)
        lc_true_node = torch.nan_to_num(batch["logcnt1h_node"], nan=0.0)
        sp_w_node = torch.nan_to_num(batch["speed_w_node"], nan=0.0)

        gather_for_seq = lambda node_vals: torch.gather(node_vals, 1, route_seq_idx)
        sp_true = gather_for_seq(sp_true_node)
        lc_true = gather_for_seq(lc_true_node)
        sp_w    = gather_for_seq(sp_w_node)

        vff_seq_b = vff_vec[node_edge_idx]
        vff_seq_b = torch.gather(vff_seq_b, 1, route_seq_idx)

        x_seq_norm = (x_seq_nodes - SCALER_MEAN.view(1, 1, -1)) / SCALER_STD.view(1, 1, -1)

        # masks: seq_mask = all valid links; sp_mask = links with speed data
        # (canonical: NO is_stub filtering — that's done inside metric functions)
        sp_mask = (sp_w > 0) & seq_mask

        Bsz = seq_true_tt.size(0)

        # Model returns 7 + mem_loss (canonical 7-tuple + MetaST mem_loss)
        link_eta_seq, trip_eta, sp_pred, lc_pred, g, delta, eta_phys_seq, mem_loss = model(
            x_seq_norm,
            hop_idx,
            route_flag_nodes,
            masks_per_rel,
            route_seq_idx,
            seq_link_len,
            vff_seq_b,
            seq_mask,
            node_edge_idx=node_edge_idx,
            route_cluster_id=_route_cluster_id_from_globals(node_edge_idx, route_seq_idx),
            compute_mem_loss=True,
            return_mem_loss=True,
            r_max=3.0,
            return_parts=True,
        )

        link_eta_seq = torch.nan_to_num(link_eta_seq, nan=0.0)
        trip_eta = torch.nan_to_num(trip_eta, nan=0.0)
        sp_pred = torch.nan_to_num(sp_pred, nan=0.0)
        lc_pred = torch.nan_to_num(lc_pred, nan=0.0)

        # [2026-06-06] Controlled backbone losses (identical to base LinkETA):
        #   link/route log1p-Huber, speed vs sp_actual, no lam_g g-reg.
        pred_seq = link_eta_in_seconds(link_eta_seq, seq_link_len)
        seq_true = seq_true_tt

        loss_link  = link_loss_huber_log_lenweighted(
            pred_seq, seq_true, seq_mask, link_len=seq_link_len, delta=0.3)
        loss_route = route_loss_huber_log(pred_seq, trip_tt, seq_mask, delta=0.3)

        loss_speed = _backbone_speed_loss(
            sp_pred, seq_link_len, seq_true_tt, vff_seq_b, seq_mask)

        if seq_mask.any():
            loss_vol = F.smooth_l1_loss(lc_pred[seq_mask], lc_true[seq_mask])
        else:
            loss_vol = torch.zeros((), device=device)

        # Base final loss: 3-task {speed,link,route} (DWA-managed) + fixed 0.10*vol
        final_loss = (
            w_speed * loss_speed +
            w_link * loss_link +
            w_route * loss_route +
            0.10 * loss_vol
        )

        # MetaST addition: memory clustering loss (the ONLY method-specific term)
        if mem_loss is not None:
            final_loss = final_loss + lambda_mem * mem_loss

        # backward (normalize by number of batches)
        denom = max(len(loader), 1)
        (final_loss / denom).backward()

        total_final_loss += float(final_loss.detach().cpu().item())

        # ---------- metrics (canonical: use seq_mask, no-grad) ----------
        with torch.no_grad():
            link_stats = link_metrics_micro(pred_seq, seq_true, seq_mask, link_len=seq_link_len,
                                            L_min=L_MIN_METRIC, T_min=T_MIN_METRIC)
            route_stats = route_metrics_pooled(pred_seq, trip_tt, seq_mask)

        sum_link_loss += float(loss_link.detach().cpu().item()) * Bsz
        sum_route_loss += float(loss_route.detach().cpu().item()) * Bsz
        N_routes_loss += Bsz

        n_links = int(link_stats["n_links"])
        if n_links > 0:
            sum_mae_link += link_stats["mae"] * n_links
            sum_sse_link += (link_stats["rmse"] ** 2) * n_links
            n_m = int(link_stats.get("n_mape", 0))
            if n_m > 0 and not math.isnan(link_stats["mape"]):
                sum_mape_link += link_stats["mape"] * n_m
                N_links_mape += n_m
            N_links += n_links

        n_routes = int(route_stats["n_routes"])
        if n_routes > 0:
            sum_mae_route += route_stats["mae"] * n_routes
            sum_sse_route += (route_stats["rmse"] ** 2) * n_routes
            n_mr = int(route_stats.get("n_mape", 0))
            if n_mr > 0 and not math.isnan(route_stats["mape"]):
                sum_mape_route += route_stats["mape"] * n_mr
                N_routes_mape += n_mr
            N_routes_metrics += n_routes

        if sp_mask.any():
            sp_abs = (sp_pred[sp_mask] - sp_true[sp_mask]).abs().mean().item()
            sum_speed_mae += sp_abs * int(sp_mask.sum().item())
            N_speed += int(sp_mask.sum().item())

        if seq_mask.any():
            vol_abs = (lc_pred[seq_mask] - lc_true[seq_mask]).abs().mean().item()
            n_valid = int(seq_mask.sum().item())
            sum_vol_mae += vol_abs * n_valid
            N_vol += n_valid

    metrics = {
        'link_loss': sum_link_loss / max(N_routes_loss, 1),
        'route_loss': sum_route_loss / max(N_routes_loss, 1),
        'link_mae': sum_mae_link / max(N_links, 1),
        'link_rmse': math.sqrt(sum_sse_link / max(N_links, 1)) if N_links > 0 else float("nan"),
        'link_mape': (sum_mape_link / max(N_links_mape, 1)) if N_links_mape > 0 else float('nan'),
        'route_mae': sum_mae_route / max(N_routes_metrics, 1),
        'route_rmse': math.sqrt(sum_sse_route / max(N_routes_metrics, 1)) if N_routes_metrics > 0 else float("nan"),
        'route_mape': (sum_mape_route / max(N_routes_mape, 1)) if N_routes_mape > 0 else float('nan'),
        'speed_mae': (sum_speed_mae / max(N_speed, 1)) if N_speed else float('nan'),
        'vol_mae': (sum_vol_mae / max(N_vol, 1)) if N_vol else float('nan'),
    }

    mean_final_loss = total_final_loss / max(n_batches, 1)

    # grads are on model params now
    grads = {}
    for n, p in model.named_parameters():
        if p.grad is not None:
            grads[n] = p.grad.detach().cpu().clone()

    return metrics, mean_final_loss, grads


# -----------------------------
# 3) Formatting prints
# -----------------------------
def fmt_metrics(tag, m):
    # Prints the key things you requested; ignores missing keys safely.
    def g(key, default=float("nan")):
        return m.get(key, default)

    return (
        f"{tag} | "
        f"Link: loss={g('link_loss'):.4f} mae={g('link_mae'):.4f} rmse={g('link_rmse'):.4f} mape={g('link_mape'):.4f}  ||  "
        f"Route: loss={g('route_loss'):.4f} mae={g('route_mae'):.4f} rmse={g('route_rmse'):.4f} mape={g('route_mape'):.4f}  ||  "
        f"Traffic: speed_mae={g('speed_mae'):.4f} vol_mae={g('vol_mae'):.4f}"
    )


def clone_state_cpu(m):
    return {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}

def load_state_cpu(m, state_cpu, device):
    m.load_state_dict({k: v.to(device) for k, v in state_cpu.items()}, strict=True)

def grads_add_inplace(acc, g):
    for k, v in g.items():
        if k in acc:
            acc[k] += v
        else:
            acc[k] = v.clone()
    return acc

def grads_scale_inplace(g, s):
    for k in g:
        g[k] *= s
    return g

def apply_update(meta_state_cpu, grads_cpu, outer_lr):
    for k, g in grads_cpu.items():
        if k in meta_state_cpu:
            meta_state_cpu[k] = meta_state_cpu[k] - outer_lr * g
    return meta_state_cpu

def inner_adapt_steps(meta_state_cpu, core, inner_steps=5, inner_lr=1e-5, w_speed=0.5, w_link=0.5, w_route=0.5):
    """
    Inner adaptation loop: K gradient steps on support set.

    Paper-faithful:
    - K gradient steps (batches) on support
    - no clustering loss in inner objective
    - memory params are shared and not adapted in inner loop
    """
    global optimizer
    set_globals_from_core(core)

    load_state_cpu(model, meta_state_cpu, device)
    model.to(device)
    model.train()

    # Freeze memory only (shared module, not task-adapted)
    if hasattr(model, "mem"):
        set_requires_grad(model.mem, False)

    inner_opt = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=inner_lr
    )

    step_count = 0
    train_iter = iter(train_loader)

    for step in range(inner_steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        node_edge_idx = batch["node_edge_idx"]
        x_seq_nodes = batch["x_seq"]
        route_flag_nodes = batch["route_flag"]
        hop_idx = batch["hop_idx"]
        masks_per_rel = [m.to(device) for m in batch["masks_per_rel"]]

        route_seq_idx = batch["route_seq_idx"]
        seq_true_tt = torch.nan_to_num(batch["seq_true_tt"], nan=0.0)
        seq_link_len = torch.nan_to_num(batch["seq_link_len"], nan=0.0).clamp_min(1e-3)
        seq_mask = batch["seq_mask"].bool()
        trip_tt = torch.nan_to_num(batch["trip_tt"], nan=0.0)

        sp_true_node = torch.nan_to_num(batch["sp_ratio_agg"], nan=0.0)
        lc_true_node = torch.nan_to_num(batch["logcnt1h_node"], nan=0.0)
        sp_w_node = torch.nan_to_num(batch["speed_w_node"], nan=0.0)

        gather_for_seq = lambda node_vals: torch.gather(node_vals, 1, route_seq_idx)
        sp_true = gather_for_seq(sp_true_node)
        lc_true = gather_for_seq(lc_true_node)
        sp_w    = gather_for_seq(sp_w_node)

        vff_seq_b = vff_vec[node_edge_idx]
        vff_seq_b = torch.gather(vff_seq_b, 1, route_seq_idx)

        x_seq_norm = (x_seq_nodes - SCALER_MEAN.view(1, 1, -1)) / SCALER_STD.view(1, 1, -1)

        # masks: canonical — no is_stub filtering in losses
        sp_mask = (sp_w > 0) & seq_mask

        # Forward: 7 values, no clustering loss in inner loop
        link_eta_seq, trip_eta, sp_pred, lc_pred, g, delta, eta_phys_seq = model(
            x_seq_norm,
            hop_idx,
            route_flag_nodes,
            masks_per_rel,
            route_seq_idx,
            seq_link_len,
            vff_seq_b,
            seq_mask,
            node_edge_idx=node_edge_idx,
            compute_mem_loss=False,
            r_max=3.0,
            return_parts=True,
        )

        link_eta_seq = torch.nan_to_num(link_eta_seq, nan=0.0)
        trip_eta = torch.nan_to_num(trip_eta, nan=0.0)
        sp_pred = torch.nan_to_num(sp_pred, nan=0.0)
        lc_pred = torch.nan_to_num(lc_pred, nan=0.0)

        # [2026-06-06] Controlled backbone losses (identical to base LinkETA):
        #   link/route log1p-Huber, speed vs sp_actual, no lam_g g-reg.
        pred_seq = link_eta_in_seconds(link_eta_seq, seq_link_len)
        seq_true = seq_true_tt

        loss_link = link_loss_huber_log_lenweighted(
            pred_seq, seq_true, seq_mask, link_len=seq_link_len, delta=0.3)
        loss_route = route_loss_huber_log(pred_seq, trip_tt, seq_mask, delta=0.3)

        loss_speed = _backbone_speed_loss(
            sp_pred, seq_link_len, seq_true_tt, vff_seq_b, seq_mask)

        if seq_mask.any():
            loss_vol = F.smooth_l1_loss(lc_pred[seq_mask], lc_true[seq_mask])
        else:
            loss_vol = torch.zeros((), device=device)

        # 3-task {speed,link,route} + fixed 0.10*vol (matches base LinkETA)
        inner_loss = (
            w_speed * loss_speed +
            w_link * loss_link +
            w_route * loss_route +
            0.10 * loss_vol
        )

        inner_opt.zero_grad(set_to_none=True)
        inner_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        inner_opt.step()

        step_count += 1

    # Full support-set metrics after adaptation
    # NOTE: one_epoch() (LinkETA_Fixed) does NOT pass route_cluster_id to the
    # model forward, so clustering loss is not computed here.  This is fine:
    # we are in eval-only mode (train=False, no_grad) and the memory readout
    # z_mem is still applied to predictions — only the auxiliary clustering
    # loss term is absent, which does not affect evaluation metrics.
    model.eval()
    with torch.no_grad():
        support_metrics = one_epoch(CORE, train_loader, epoch=0, train=False, w_link=w_link, w_route=w_route)
    support_metrics["inner_steps"] = step_count
    model.train()

    adapted_state_cpu = clone_state_cpu(model)

    # Restore memory grads for outer/meta stage and future usage
    if hasattr(model, "mem"):
        set_requires_grad(model.mem, True)

    return adapted_state_cpu, support_metrics

def meta_train_metast(
    source_cores,
    source_names,
    meta_iters=20_000,
    inner_steps=5,
    inner_lr=1e-5,
    outer_lr=1e-5,
    w_speed=0.5,           # [2026-06-06] DWA-managed speed weight (3-task)
    w_link=0.5,
    w_route=0.5,
    lambda_mem=1e-4,       # paper γ = 10⁻⁴
    seed=42,
    enable_metrics: bool = False,
    log_every: int = 10,
    patience: int = None,          # early stopping on meta query loss (avg over cities)
    min_delta: float = 0.0,
    max_query_batches: int = None,  # None = full val_loader (recommended)
    dwa_T: float = 2.0,           # DWA temperature (0 = disable DWA, use fixed w_link/w_route)
    ckpt_dir: str = "ckpts_MetaST",  # directory to save best/last checkpoints
    save_every: int = 5,             # [2026-06-06] interval theta0 ckpts (0=off)
):
    """
    MetaST meta-training (FOMAML inner loop) with optional detailed per-city metrics logging.

    Inner loop:
      - K (=inner_steps) gradient steps on support batches (train_loader)
      - no clustering loss in support objective
      - adapt all params except mem.*
      - create_graph=False: inner gradients are detached (first-order
        approximation — FOMAML)

    Outer loop:
      - query loss on val_loader + gamma*mem_loss (clustering)
      - FOMAML gradient: qloss.backward() computes ∂L_query/∂θ' only
        (the gradient w.r.t. the adapted parameters), which is used
        as the meta-gradient for θ₀ — no second-order terms.

    When enable_metrics=True, prints per-meta-iter:
      - Support(K batches aggregate) metrics for each city
      - Query(val aggregate) metrics for each city
    """

    import math
    random.seed(seed)
    torch.manual_seed(seed)

    # functional_call import
    try:
        from torch.func import functional_call
    except Exception:
        from torch.nn.utils.stateless import functional_call

    # outer optimizer on theta0 params (the real model params)
    meta_opt = torch.optim.Adam(model.parameters(), lr=outer_lr)

    def _to_device(batch):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        return batch

    def _route_cluster_id(batch):
        # labels are per-edge (EDGE_CLUSTER_LABELS[edge_id] = cluster_id)
        if EDGE_CLUSTER_LABELS is None:
            return None
        node_edge_idx = batch["node_edge_idx"]                 # [B, M]
        route_seq_idx = batch["route_seq_idx"]                 # [B, L]
        edge_seq = torch.gather(node_edge_idx, 1, route_seq_idx)  # [B, L]
        edge_seq = edge_seq.clamp(0, EDGE_CLUSTER_LABELS.size(0) - 1)
        return EDGE_CLUSTER_LABELS[edge_seq].long()            # [B, L]

    # --------- metric accumulator (uses your existing metric funcs) ----------
    def _acc_init():
        return dict(
            sum_link_loss=0.0, sum_route_loss=0.0, N_loss=0,

            sum_mae_link=0.0, sum_sse_link=0.0, sum_mape_link=0.0, N_links=0, N_links_mape=0,
            sum_mae_route=0.0, sum_sse_route=0.0, sum_mape_route=0.0, N_routes=0, N_routes_mape=0,

            sum_speed_abs=0.0, N_speed=0,
            sum_vol_abs=0.0,   N_vol=0,

            sum_mem_loss=0.0, N_mem=0,
            # [2026-06-07] WAPE/MdAPE/P90 pooling
            sum_abs_link=0.0, sum_true_link=0.0, ape_link=[],
            sum_abs_route=0.0, sum_true_route=0.0, ape_route=[],
        )

    def _acc_update(acc, loss_link, loss_route, pred_seq, seq_true, seq_mask, seq_link_len,
                    trip_tt,
                    sp_pred, sp_true, sp_mask,
                    lc_pred, lc_true,
                    mem_loss_val,
                    batch_size: int):

        acc["sum_link_loss"] += float(loss_link.detach().cpu().item()) * batch_size
        acc["sum_route_loss"] += float(loss_route.detach().cpu().item()) * batch_size
        acc["N_loss"] += batch_size

        with torch.no_grad():
            # canonical: use seq_mask for metrics (same as LinkETA_Fixed.py)
            link_stats = link_metrics_micro(
                pred_seq.detach(), seq_true.detach(), seq_mask.detach(),
                link_len=seq_link_len.detach(),
                L_min=L_MIN_METRIC, T_min=T_MIN_METRIC
            )
            route_stats = route_metrics_pooled(
                pred_seq.detach(), trip_tt.detach(), seq_mask.detach()
            )

            n_links = int(link_stats.get("n_links", 0))
            if n_links > 0:
                acc["sum_mae_link"] += float(link_stats["mae"]) * n_links
                acc["sum_sse_link"] += float(link_stats["rmse"] ** 2) * n_links
                acc["N_links"] += n_links

                n_m = int(link_stats.get("n_mape", 0))
                if n_m > 0 and (not math.isnan(link_stats.get("mape", float("nan")))):
                    acc["sum_mape_link"] += float(link_stats["mape"]) * n_m
                    acc["N_links_mape"] += n_m

            n_routes = int(route_stats.get("n_routes", 0))
            if n_routes > 0:
                acc["sum_mae_route"] += float(route_stats["mae"]) * n_routes
                acc["sum_sse_route"] += float(route_stats["rmse"] ** 2) * n_routes
                acc["N_routes"] += n_routes

                n_mr = int(route_stats.get("n_mape", 0))
                if n_mr > 0 and (not math.isnan(route_stats.get("mape", float("nan")))):
                    acc["sum_mape_route"] += float(route_stats["mape"]) * n_mr
                    acc["N_routes_mape"] += n_mr

            # [2026-06-07] pool WAPE/MdAPE/P90 building blocks
            acc["sum_abs_link"] += float(link_stats.get("sum_abs", 0.0))
            acc["sum_true_link"] += float(link_stats.get("sum_true", 0.0))
            _av = link_stats.get("ape_vals")
            if _av is not None and _av.numel() > 0:
                acc["ape_link"].append(_av)
            acc["sum_abs_route"] += float(route_stats.get("sum_abs", 0.0))
            acc["sum_true_route"] += float(route_stats.get("sum_true", 0.0))
            _avr = route_stats.get("ape_vals")
            if _avr is not None and _avr.numel() > 0:
                acc["ape_route"].append(_avr)

            # speed MAE
            if sp_mask is not None and sp_mask.any():
                acc["sum_speed_abs"] += float((sp_pred.detach()[sp_mask] - sp_true.detach()[sp_mask]).abs().sum().cpu().item())
                acc["N_speed"] += int(sp_mask.sum().cpu().item())

            # vol MAE — canonical: use seq_mask
            if seq_mask is not None and seq_mask.any():
                acc["sum_vol_abs"] += float((lc_pred.detach()[seq_mask] - lc_true.detach()[seq_mask]).abs().sum().cpu().item())
                acc["N_vol"] += int(seq_mask.sum().cpu().item())

            if mem_loss_val is not None:
                acc["sum_mem_loss"] += float(mem_loss_val.detach().cpu().item())
                acc["N_mem"] += 1

    def _acc_finalize(acc):
        out = {}

        out["link_loss"] = acc["sum_link_loss"] / max(acc["N_loss"], 1)
        out["route_loss"] = acc["sum_route_loss"] / max(acc["N_loss"], 1)

        out["link_mae"] = acc["sum_mae_link"] / max(acc["N_links"], 1)
        out["link_rmse"] = math.sqrt(acc["sum_sse_link"] / max(acc["N_links"], 1)) if acc["N_links"] > 0 else float("nan")
        out["link_mape"] = acc["sum_mape_link"] / max(acc["N_links_mape"], 1) if acc["N_links_mape"] > 0 else float("nan")

        out["route_mae"] = acc["sum_mae_route"] / max(acc["N_routes"], 1)
        out["route_rmse"] = math.sqrt(acc["sum_sse_route"] / max(acc["N_routes"], 1)) if acc["N_routes"] > 0 else float("nan")
        out["route_mape"] = acc["sum_mape_route"] / max(acc["N_routes_mape"], 1) if acc["N_routes_mape"] > 0 else float("nan")

        out["speed_mae"] = (acc["sum_speed_abs"] / max(acc["N_speed"], 1)) if acc["N_speed"] > 0 else float("nan")
        out["vol_mae"] = (acc["sum_vol_abs"] / max(acc["N_vol"], 1)) if acc["N_vol"] > 0 else float("nan")

        out["mem_loss"] = (acc["sum_mem_loss"] / max(acc["N_mem"], 1)) if acc["N_mem"] > 0 else float("nan")
        # [2026-06-07] pooled robust % metrics
        _lp = finalize_pct_metrics(acc["sum_abs_link"], acc["sum_true_link"], acc["ape_link"])
        _rp = finalize_pct_metrics(acc["sum_abs_route"], acc["sum_true_route"], acc["ape_route"])
        out["link_wape"], out["link_mdape"], out["link_p90"] = _lp["wape"], _lp["mdape"], _lp["ape_p90"]
        out["route_wape"], out["route_mdape"], out["route_p90"] = _rp["wape"], _rp["mdape"], _rp["ape_p90"]
        return out

    # --------- forward + loss on a batch (stateless) ----------
    def _forward_losses_and_parts(state, buffers, batch, compute_mem_loss, use_cluster_labels, epoch_for_lam=0):
        node_edge_idx = batch["node_edge_idx"]
        x_seq_nodes = batch["x_seq"]
        route_flag_nodes = batch["route_flag"]
        hop_idx = batch["hop_idx"]
        masks_per_rel = [m.to(device) for m in batch["masks_per_rel"]]

        route_seq_idx = batch["route_seq_idx"]
        seq_true_tt = torch.nan_to_num(batch["seq_true_tt"], nan=0.0)
        seq_link_len = torch.nan_to_num(batch["seq_link_len"], nan=0.0).clamp_min(1e-3)
        seq_mask = batch["seq_mask"].bool()
        trip_tt = torch.nan_to_num(batch["trip_tt"], nan=0.0)

        sp_true_node = torch.nan_to_num(batch["sp_ratio_agg"], nan=0.0)
        lc_true_node = torch.nan_to_num(batch["logcnt1h_node"], nan=0.0)
        sp_w_node = torch.nan_to_num(batch["speed_w_node"], nan=0.0)

        gather_for_seq = lambda node_vals: torch.gather(node_vals, 1, route_seq_idx)
        sp_true = gather_for_seq(sp_true_node)
        lc_true = gather_for_seq(lc_true_node)
        sp_w    = gather_for_seq(sp_w_node)

        vff_seq_b = vff_vec[node_edge_idx]
        vff_seq_b = torch.gather(vff_seq_b, 1, route_seq_idx)

        x_seq_norm = (x_seq_nodes - SCALER_MEAN.view(1, 1, -1)) / SCALER_STD.view(1, 1, -1)

        # masks: canonical — no is_stub in losses
        sp_mask = (sp_w > 0) & seq_mask

        route_cluster = _route_cluster_id(batch) if use_cluster_labels else None

        args = (
            x_seq_norm,
            hop_idx,
            route_flag_nodes,
            masks_per_rel,
            route_seq_idx,
            seq_link_len,
            vff_seq_b,
            seq_mask,
        )
        kwargs = dict(
            route_cluster_id=route_cluster,
            compute_mem_loss=compute_mem_loss,
            r_max=3.0,
            return_parts=True,
            return_mem_loss=True,
        )

        out = functional_call(model, {**state, **buffers}, args=args, kwargs=kwargs)
        # 7 canonical values + mem_loss
        (link_eta_seq, trip_eta, sp_pred, lc_pred, g, delta, eta_phys_seq, mem_loss) = out

        link_eta_seq = torch.nan_to_num(link_eta_seq, nan=0.0)
        trip_eta = torch.nan_to_num(trip_eta, nan=0.0)
        sp_pred = torch.nan_to_num(sp_pred, nan=0.0)
        lc_pred = torch.nan_to_num(lc_pred, nan=0.0)

        # [2026-06-06] Controlled backbone losses (identical to base LinkETA):
        #   link/route log1p-Huber, speed vs sp_actual, no lam_g g-reg.
        pred_seq = link_eta_in_seconds(link_eta_seq, seq_link_len)
        seq_true = seq_true_tt

        loss_link  = link_loss_huber_log_lenweighted(
            pred_seq, seq_true, seq_mask, link_len=seq_link_len, delta=0.3)
        loss_route = route_loss_huber_log(pred_seq, trip_tt, seq_mask, delta=0.3)

        loss_speed = _backbone_speed_loss(
            sp_pred, seq_link_len, seq_true_tt, vff_seq_b, seq_mask)

        if seq_mask.any():
            loss_vol = F.smooth_l1_loss(lc_pred[seq_mask], lc_true[seq_mask])
        else:
            loss_vol = torch.zeros((), device=device)

        # 3-task {speed,link,route} (DWA-managed) + fixed 0.10*vol (matches base LinkETA)
        task_loss = (
            w_speed * loss_speed +
            w_link * loss_link +
            w_route * loss_route +
            0.10 * loss_vol
        )

        # MetaST addition: memory clustering loss (the ONLY method-specific term)
        if compute_mem_loss and (mem_loss is not None):
            task_loss = task_loss + lambda_mem * mem_loss

        return dict(
            task_loss=task_loss,
            loss_speed=loss_speed,
            loss_link=loss_link,
            loss_route=loss_route,
            pred_seq=pred_seq,
            seq_true=seq_true,
            seq_mask=seq_mask,
            seq_link_len=seq_link_len,
            trip_tt=trip_tt,
            sp_pred=sp_pred,
            sp_true=sp_true,
            sp_mask=sp_mask,
            lc_pred=lc_pred,
            lc_true=lc_true,
            mem_loss=mem_loss,
        )

    # ---- DWA (Dynamic Weight Averaging) — [2026-06-06] 3-task: speed, link, route ----
    _dwa_hist_speed = deque(maxlen=2)
    _dwa_hist_link  = deque(maxlen=2)
    _dwa_hist_route = deque(maxlen=2)
    _dwa_enabled = (dwa_T > 0)
    # [2026-06-06] Normalize pre-DWA weights to sum 1.0 (DWA softmax also sums to
    # 1.0) so there is no discontinuity at DWA kick-in, and all methods start DWA
    # from uniform weights (fair comparison).
    _w0 = w_speed + w_link + w_route
    w_speed, w_link, w_route = w_speed / _w0, w_link / _w0, w_route / _w0
    w_speed_init, w_link_init, w_route_init = w_speed, w_link, w_route   # remember caller's defaults

    print("Starting MetaST meta-training (FOMAML — first-order; optional metrics logging)")
    print(f"  Source cities: {source_names}")
    print(f"  Max meta iters: {meta_iters}")
    print(f"  Inner steps: {inner_steps}")
    print(f"  Inner LR: {inner_lr}")
    print(f"  Outer LR (Adam): {outer_lr}")
    print(f"  Mem loss weight (gamma): {lambda_mem}")
    print(f"  DWA: {'ON (T={:.1f})'.format(dwa_T) if _dwa_enabled else 'OFF (fixed w_link={:.2f}, w_route={:.2f})'.format(w_link, w_route)}")
    print(f"  Gradient order: FOMAML (first-order, create_graph=False)")
    print(f"  enable_metrics={enable_metrics} | log_every={log_every} | max_query_batches={max_query_batches}")
    if patience is not None:
        print(f"  Early stopping: patience={patience}, min_delta={min_delta}")
    print("=" * 100)

    best_loss = float("inf")
    best_state = None
    best_meta_iter = None       # [2026-06-07] iter of min meta_loss (selected model)
    best_summary = None         # [2026-06-07] per-city query metrics @ that iter
    bad = 0
    _hist_rows = []   # [2026-06-06] per-source val metrics for training_history.csv

    # ---- Checkpoint manager for saving to disk ----
    _ckpt_mgr = CheckpointManager(dirpath=ckpt_dir, max_keep=3)
    print(f"  Checkpoint dir: {ckpt_dir}")

    for it in range(1, meta_iters + 1):
        # we only do the expensive per-city metric aggregation when we actually log
        do_log = enable_metrics and ((it % log_every == 0) or (it <= 3))

        # ---- DWA: recompute task weights from loss-change ratios (3-task) ----
        if (_dwa_enabled and len(_dwa_hist_speed) >= 2
                and len(_dwa_hist_link) >= 2 and len(_dwa_hist_route) >= 2):
            r_speed = float(_dwa_hist_speed[-1] / (_dwa_hist_speed[-2] + 1e-8))
            r_link  = float(_dwa_hist_link[-1]  / (_dwa_hist_link[-2]  + 1e-8))
            r_route = float(_dwa_hist_route[-1] / (_dwa_hist_route[-2] + 1e-8))
            e_speed = math.exp(r_speed / dwa_T)
            e_link  = math.exp(r_link  / dwa_T)
            e_route = math.exp(r_route / dwa_T)
            s = e_speed + e_link + e_route
            w_speed = e_speed / s
            w_link  = e_link  / s
            w_route = e_route / s
        # (else: keep current weights — initial values or last DWA values)

        meta_opt.zero_grad(set_to_none=True)
        meta_loss_detached = 0.0

        # DWA: accumulate per-city average query losses for history update
        _dwa_sum_speed = 0.0
        _dwa_sum_link  = 0.0
        _dwa_sum_route = 0.0
        _dwa_n_cities  = 0

        # [2026-06-07] per-city query metrics for THIS iter (selected-model summary)
        cur_iter_metrics = {}
        for core, name in zip(source_cores, source_names):
            set_globals_from_core(core, keep_model=True)
            model.train()

            params0 = {k: v for k, v in model.named_parameters()}
            buffs0  = {k: v for k, v in model.named_buffers()}

            # adapt everything except memory params
            adapt_names = [n for n in params0.keys() if not n.startswith("mem.")]
            fast = dict(params0)

            # ----- inner loop (support) -----
            support_acc = _acc_init() if do_log else None
            train_iter = iter(train_loader)

            for s in range(inner_steps):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    batch = next(train_iter)

                batch = _to_device(batch)

                pack = _forward_losses_and_parts(
                    state=fast, buffers=buffs0, batch=batch,
                    compute_mem_loss=False, use_cluster_labels=False, epoch_for_lam=0
                )

                support_loss = pack["task_loss"]

                # gradient step on fast params (create_graph=False → FOMAML first-order)
                grads = torch.autograd.grad(
                    support_loss,
                    [fast[n] for n in adapt_names],
                    create_graph=False,
                )
                for n, g in zip(adapt_names, grads):
                    fast[n] = fast[n] - inner_lr * g

                # optional support metrics aggregate across these K batches
                if do_log:
                    Bsz = int(pack["seq_true"].size(0))
                    _acc_update(
                        support_acc,
                        pack["loss_link"], pack["loss_route"],
                        pack["pred_seq"], pack["seq_true"], pack["seq_mask"], pack["seq_link_len"],
                        pack["trip_tt"],
                        pack["sp_pred"], pack["sp_true"], pack["sp_mask"],
                        pack["lc_pred"], pack["lc_true"],
                        mem_loss_val=None,
                        batch_size=Bsz
                    )

            # ----- query loop (val) -----
            # [2026-06-07] Always accumulate query metrics (cheap: reuses the query
            # forward already run for qloss) so the selected-model summary can
            # snapshot BOTH cities at the argmin-meta_loss iteration.
            query_acc = _acc_init()
            qloss_sum = 0.0
            qcount = 0

            _city_qloss_speed = 0.0
            _city_qloss_link  = 0.0
            _city_qloss_route = 0.0
            _city_qcount = 0

            for b_idx, batch in enumerate(val_loader):
                if (max_query_batches is not None) and (b_idx >= max_query_batches):
                    break

                batch = _to_device(batch)
                pack = _forward_losses_and_parts(
                    state=fast, buffers=buffs0, batch=batch,
                    compute_mem_loss=True, use_cluster_labels=True, epoch_for_lam=0
                )

                qloss_sum = qloss_sum + pack["task_loss"]
                qcount += 1

                # DWA: track per-city query speed/link/route losses (always, not just when logging)
                _city_qloss_speed += float(pack["loss_speed"].detach().cpu().item())
                _city_qloss_link  += float(pack["loss_link"].detach().cpu().item())
                _city_qloss_route += float(pack["loss_route"].detach().cpu().item())
                _city_qcount += 1

                Bsz = int(pack["seq_true"].size(0))
                _acc_update(
                    query_acc,
                    pack["loss_link"], pack["loss_route"],
                    pack["pred_seq"], pack["seq_true"], pack["seq_mask"], pack["seq_link_len"],
                    pack["trip_tt"],
                    pack["sp_pred"], pack["sp_true"], pack["sp_mask"],
                    pack["lc_pred"], pack["lc_true"],
                    mem_loss_val=pack["mem_loss"],
                    batch_size=Bsz
                )

            qloss = qloss_sum / max(qcount, 1)
            meta_loss_detached += float(qloss.detach().cpu().item())

            # DWA: track this city's average query speed/link/route losses
            _dwa_sum_speed += _city_qloss_speed / max(_city_qcount, 1)
            _dwa_sum_link  += _city_qloss_link  / max(_city_qcount, 1)
            _dwa_sum_route += _city_qloss_route / max(_city_qcount, 1)
            _dwa_n_cities  += 1

            # FOMAML: backward through qloss → fast → θ₀
            # With create_graph=False in the inner loop, inner gradients
            # are detached, so fast[n] = θ₀ - lr*g_detached (constant offset).
            # .backward() computes ∂L_query/∂θ' and maps it 1:1 to θ₀ grads
            # (first-order approximation, no second-order Hessian terms).
            (qloss / max(len(source_cores), 1)).backward()

            # [2026-06-07] Finalize query metrics EVERY iter so the selected-model
            # summary can snapshot both cities at the argmin-meta_loss iteration.
            qry_m = _acc_finalize(query_acc)
            cur_iter_metrics[name] = qry_m

            # per-city logging
            if do_log:
                sup_m = _acc_finalize(support_acc)
                print(f"MetaIter {it:04d} | City={name} | DWA w_speed={w_speed:.3f} w_link={w_link:.3f} w_route={w_route:.3f}")
                print("  " + fmt_metrics("Support(K steps aggregate)", sup_m) + f"  ||  mem_loss=nan")
                print("  " + fmt_metrics("Query(val aggregate)", qry_m) + f"  ||  mem_loss={qry_m.get('mem_loss', float('nan')):.6f}")
                # [2026-06-06] per-source val metrics for training_history.csv
                _hist_rows.append({
                    "meta_iter": it, "city": name,
                    "link_mae": qry_m.get("link_mae"), "link_rmse": qry_m.get("link_rmse"),
                    "link_mape": qry_m.get("link_mape"),
                    "route_mae": qry_m.get("route_mae"), "route_rmse": qry_m.get("route_rmse"),
                    "route_mape": qry_m.get("route_mape"),
                    "link_wape": qry_m.get("link_wape"), "link_mdape": qry_m.get("link_mdape"),
                    "link_p90": qry_m.get("link_p90"),
                    "route_wape": qry_m.get("route_wape"), "route_mdape": qry_m.get("route_mdape"),
                    "route_p90": qry_m.get("route_p90"),   # [2026-06-07] full metric parity
                })

        # Grads have been accumulated by .backward() calls (already /nC per city)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        meta_opt.step()

        nC = max(len(source_cores), 1)
        meta_loss_detached = meta_loss_detached / nC

        # ---- DWA: update loss history for next iteration's weight computation ----
        if _dwa_enabled and _dwa_n_cities > 0:
            _dwa_hist_speed.append(_dwa_sum_speed / _dwa_n_cities)
            _dwa_hist_link.append(_dwa_sum_link   / _dwa_n_cities)
            _dwa_hist_route.append(_dwa_sum_route / _dwa_n_cities)

        if (not enable_metrics) and ((it % log_every == 0) or (it <= 3)):
            print(f"MetaIter {it:04d} | MetaLoss(detached avg over cities)={meta_loss_detached:.6f} | DWA w_speed={w_speed:.3f} w_link={w_link:.3f} w_route={w_route:.3f}")

        # early stopping on meta query loss (detached)
        if meta_loss_detached < best_loss - min_delta:
            best_loss = meta_loss_detached
            best_meta_iter = it
            # [2026-06-07] snapshot both cities' query metrics at the selected iter
            best_summary = {c: dict(m) for c, m in cur_iter_metrics.items()}
            best_state = clone_state_cpu(model)
            bad = 0
            # Save best checkpoint to disk
            _ckpt_mgr.save_best(
                model=model, optimizer=meta_opt, scheduler=None,
                epoch=it, step=it, best_metric=best_loss,
                extra={"method": "MetaST", "meta_iter": it},
            )
        else:
            bad += 1

        # Save last checkpoint every iteration
        _ckpt_mgr.save_last(
            model=model, optimizer=meta_opt, scheduler=None,
            epoch=it, step=it, best_metric=best_loss,
            extra={"method": "MetaST", "meta_iter": it},
        )

        # [2026-06-06] interval theta0 checkpoint for the meta-iter sweep
        if save_every and (it % save_every == 0):
            _ckpt_mgr.save(
                os.path.join(ckpt_dir, f"theta0_iter{it}.ckpt"),
                model=model, optimizer=None, scheduler=None,
                epoch=it, step=it, best_metric=best_loss,
                extra={"method": "MetaST", "meta_iter": it},
            )

        if patience is not None and bad >= patience:
            print(f"Early stopping at MetaIter {it:04d} | best MetaLoss={best_loss:.6f}")
            if best_state is not None:
                load_state_cpu(model, best_state, device)
            break

    # [2026-06-06] Persist per-source training history + best-iter summary.
    write_training_history(ckpt_dir, _hist_rows, method="MetaST",
                           best_summary=best_summary, best_meta_iter=best_meta_iter)

    print("=" * 100)
    print(f"Meta-training complete. Best MetaLoss: {best_loss:.6f}")
    print(f"Checkpoints saved to: {ckpt_dir}/")
    return clone_state_cpu(model)
