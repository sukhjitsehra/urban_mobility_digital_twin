# =============================================================================
# LinkETA - refactored from LinkETA_Fixed.py.
#
# Stage 2: All module-level setup (data prep, scalers, splits, edge graph,
#          DYN_STORE, datasets, model, optimizer, loaders, ckpt) now lives
#          inside linketa_get_core_components(edges_gdf, df_all).
# Stage 3: Bundle save/load/verify is no longer auto-run at import; it is
#          gated behind an explicit save_bundle(core, out_dir) call.
# Stage 4: run_training(core) reads model/optimizer/loaders/etc. from the
#          core dict instead of module globals.
#
# Result: importing this module has no side effects. All work happens when
# main.py calls linketa_get_core_components(edges_gdf, df_all).
# =============================================================================

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
import numpy as np
import pandas as pd
import scipy
from scipy.sparse import csr_matrix, vstack
from scipy.optimize import nnls
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString, MultiLineString, Point
from shapely.strtree import STRtree
from shapely.geometry.base import BaseGeometry
from shapely import wkb as shapely_wkb
from folium.plugins import HeatMap
from pyrosm import OSM
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Data
from torch_sparse import SparseTensor
from checkpoint import CheckpointManager
from loss_functions import (
    _TINY, L_MIN_METRIC, T_MIN_METRIC,
    _huber_per_element, selection_metric, composite_loss, build_seq_mask,
    link_metrics_micro, route_metrics_pooled,
    finalize_pct_metrics,
    link_eta_in_seconds,
    link_loss_huber_log_lenweighted,
    route_loss_huber_log,
    compute_global_losses_and_metrics_linketa as compute_global_losses_and_metrics,
)
from pathlib import Path
from shapely.geometry.base import BaseGeometry
from shapely import wkb as shapely_wkb
import pandas.testing as pdt

# Default checkpoint subdir name. main.py overrides this per (model, city)
# before calling linketa_get_core_components.
CKPT_DIRNAME = "ckpts_LinkETA"

USE_AMP = True  # [2026-04-20] Disabled to avoid FP16 overflow on Daegu's 15km link outliers (see update_log_LinkETA_Fixed.md)

# =============================================================================
# [MetaETA parity] Module-level CFG + schema constants.
#
# LinkETA.py was refactored to be side-effect-free: CFG used to be created
# *inside* linketa_get_core_components and published to globals at call time.
# But the MetaETA consumers (main.py / MetaST / MetaTTE / MetaTraffic /
# evaluator) do `from LinkETA import CFG` AT IMPORT, and the module-level
# classes (EdgeDynStore, TripDatasetLinkETA, LinkETA) read CFG / H_SLICES /
# BIN_SECONDS via global lookup. So CFG and these constants now live at module
# level as plain constants. This costs nothing at import (no data is loaded, no
# model is built) — the heavy work still happens only inside the explicitly
# called functions. CFG is the SAME for every city (one fixed feature schema),
# so a shared module-level constant is safe; per-model state stays in `core`.
# =============================================================================
CFG = dict(
    way_id_col_df="way_id",
    time_col_df="departure_time",
    way_id_col_edges_candidates=["way_id", "id", "edge_id", "osmid"],
    length_col_edges="length",
    speed_is_kmh=False,
    slots_per_hour=12,
    dyn_prefixes=[
        "median_speed", "mean_speed", "min_speed",
        "max_speed", "count_vehicle"
    ],
    static_cols=[
        "length", "lanes", "width_m", "speed_limit_ms",
        "hw_living_street", "hw_motorway", "hw_motorway_link", "hw_pedestrian",
        "hw_primary", "hw_primary_link", "hw_residential", "hw_secondary",
        "hw_secondary_link", "hw_service", "hw_tertiary", "hw_tertiary_link",
        "has_traffic_signal", "link_len_m",
    ],
    time_cols=[
        "day_of_week", "month", "day_of_month", "day_of_year",
        "tod_sin", "tod_cos", "dow_sin", "dow_cos",
    ],
    hidden_dim=32,
    heads=8,
    num_layers=1,
    max_route_len=1052,
    high_order_topk=5,
    speed_metric_index=0,
    K_first=5,
    K_conflict=5,
    K_high=5,
    max_nodes_local=360,
)

# Recency-window constants for TripDatasetLinkETA (read via global lookup).
# 12 slices of 5 minutes = the last hour, slot 0 = most recent.
H_SLICES = 12
BIN_SECONDS = 5 * 60

# DYN_IDX is NOT defined at module level on purpose: LinkETA._encode_dyn_slots
# guards on `"DYN_IDX" not in globals()` (returns zeros = dyn disabled) until a
# loader publishes it. init_linketa() / linketa_get_core_components() publish it
# via globals()["DYN_IDX"] = ... once the feature schema is known.

# Precompute-load state (set by init_linketa). _linketa_initialized lets callers
# detect that a precomputed base has been loaded; the actual per-model state is
# RETURNED in the `base` dict, not stashed in globals (stateless design).
PRECOMPUTED_DIR = None
CHECKPOINT_PATH = None
_linketa_initialized = False

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def guess_edges_id_col(edges_gdf: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in edges_gdf.columns:
            return c
    return None

def ensure_edge_idx_and_uv(edges_gdf: pd.DataFrame) -> pd.DataFrame:
    g = edges_gdf.copy()
    if "edge_idx" not in g.columns:
        g = g.reset_index(drop=True)
        g["edge_idx"] = np.arange(len(g), dtype=int)

    if ("u" not in g.columns) or ("v" not in g.columns):
        def _ends(geom):
            if isinstance(geom, LineString):
                s = geom.coords[0]; e = geom.coords[-1]
            elif isinstance(geom, MultiLineString):
                s = list(geom.geoms[0].coords)[0]
                e = list(geom.geoms[-1].coords)[-1]
            else:
                return None, None
            s = (round(s[0], 6), round(s[1], 6))
            e = (round(e[0], 6), round(e[1], 6))
            return s, e

        starts, ends = zip(*[_ends(ge) for ge in g["geometry"]])
        g["start_xy"] = starts
        g["end_xy"] = ends
        pts = pd.unique(list(g["start_xy"]) + list(g["end_xy"]))
        coord2nid = {pt: i for i, pt in enumerate(pts)}
        g["u"] = g["start_xy"].map(coord2nid)
        g["v"] = g["end_xy"].map(coord2nid)
        g = g.drop(columns=["start_xy", "end_xy"])
    return g

def ensure_static_feature_columns(edges_gdf, df_all, candidates, edges_id_col, way_id_df_col):
    """Merge/fill static columns onto edges_gdf; map df_all->edge_idx; ensure meters for length."""
    g = edges_gdf.copy()
    if "edge_idx" not in g.columns:
        g = g.reset_index(drop=True)
        g["edge_idx"] = g.index

    # map df_all -> edge_idx
    id2edge = dict(zip(g[edges_id_col].tolist(), g["edge_idx"].tolist()))
    if "edge_idx" not in df_all.columns:
        df_all = df_all.copy()
        df_all["edge_idx"] = df_all[way_id_df_col].map(id2edge)

    # length (full edge length in meters) - always available in df_all
    if "length" not in g.columns:
        tmp = (
            df_all.dropna(subset=["edge_idx", "length"])
            .groupby("edge_idx")["length"]
            .first()
        )
        g["length"] = g["edge_idx"].map(tmp)
    g["length"] = pd.to_numeric(g["length"], errors="coerce").fillna(0.0)

    # speed_limit_ms
    if "speed_limit_ms" not in g.columns:
        src = None
        if "speed_limit_ms" in g.columns:
            src = ("edges", "speed_limit_ms")
        elif "speed_limit_ms" in df_all.columns:
            src = ("df", "speed_limit_ms")
        elif "speed_limit_kmh" in g.columns:
            src = ("edges_kmh", "speed_limit_kmh")
        elif "speed_limit_kmh" in df_all.columns:
            src = ("df_kmh", "speed_limit_kmh")
        if src is None:
            g["speed_limit_ms"] = 0.0
        else:
            if src[0] == "edges":
                g["speed_limit_ms"] = pd.to_numeric(g["speed_limit_ms"], errors="coerce")
            elif src[0] == "df":
                tmp = (
                    df_all.dropna(subset=["edge_idx", "speed_limit_ms"])
                    .groupby("edge_idx")["speed_limit_ms"]
                    .first()
                )
                g["speed_limit_ms"] = g["edge_idx"].map(tmp)
            elif src[0] == "edges_kmh":
                g["speed_limit_ms"] = pd.to_numeric(g["speed_limit_kmh"], errors="coerce") / 3.6
            elif src[0] == "df_kmh":
                tmp = (
                    df_all.dropna(subset=["edge_idx", "speed_limit_kmh"])
                    .groupby("edge_idx")["speed_limit_kmh"]
                    .first() / 3.6
                )
                g["speed_limit_ms"] = g["edge_idx"].map(tmp)
    g["speed_limit_ms"] = pd.to_numeric(g["speed_limit_ms"], errors="coerce").fillna(0.0)

    # other static
    for col in candidates:
        if col in ("length", "speed_limit_ms"):
            continue
        if col not in g.columns:
            if col in df_all.columns:
                tmp = df_all.dropna(subset=["edge_idx"]).groupby("edge_idx")[col].first()
                g[col] = g["edge_idx"].map(tmp)
            else:
                g[col] = 0.0
        if g[col].dtype == "bool":
            g[col] = g[col].astype(int)
        else:
            g[col] = pd.to_numeric(g[col], errors="coerce").fillna(0.0)

    if "lanes" in g.columns:
        g["lanes"] = pd.to_numeric(g["lanes"], errors="coerce").fillna(0.0)

    actual = []
    for c in candidates:
        if c in g.columns:
            actual.append(c)
        else:
            g[c] = 0.0
            actual.append(c)

    return g, actual, df_all

def _bearing_toward_end(geom):
    if isinstance(geom, LineString):
        x0, y0 = geom.coords[0]; x1, y1 = geom.coords[-1]
    elif isinstance(geom, MultiLineString):
        x0, y0 = list(geom.geoms[0].coords)[0]
        x1, y1 = list(geom.geoms[-1].coords)[-1]
    else:
        return np.nan
    return math.atan2(y1 - y0, x1 - x0)

def _ang_diff(a, b):
    return (b - a + math.pi) % (2 * math.pi) - math.pi

def add_time_features(df_all: pd.DataFrame):
    df_all = df_all.copy()
    df_all['departure_time'] = pd.to_datetime(df_all['departure_time'], errors='coerce')
    df_all = df_all.dropna(subset=['departure_time'])

    _tod = df_all['departure_time'].dt.hour + df_all['departure_time'].dt.minute / 60.0
    _tod_rad = 2.0 * np.pi * (_tod / 24.0)
    df_all['tod_sin'] = np.sin(_tod_rad).astype('float32')
    df_all['tod_cos'] = np.cos(_tod_rad).astype('float32')

    _dow = df_all['departure_time'].dt.dayofweek
    _dow_rad = 2.0 * np.pi * (_dow / 7.0)
    df_all['dow_sin'] = np.sin(_dow_rad).astype('float32')
    df_all['dow_cos'] = np.cos(_dow_rad).astype('float32')
    return df_all

def cols_if_exist(df: pd.DataFrame, names):
    return [c for c in names if c in df.columns]



def fit_scaler(inst, df_train, INST_FEAT_COLS):
    inst_indexed = inst if isinstance(inst.index, pd.MultiIndex) \
        else inst.set_index(['TRIP_ID', 'edge_idx'])

    train_inst_idx = pd.MultiIndex.from_frame(
        df_train[['TRIP_ID', 'edge_idx']].drop_duplicates()
    )
    train_inst = inst_indexed.loc[inst_indexed.index.intersection(train_inst_idx)]

    feat_mean = train_inst[INST_FEAT_COLS].mean().astype('float32').fillna(0.0)
    # [2026-04-20] Clamp tiny stds (not just exact 0) — e.g. Daegu's width_m=3.4 constant
    # gives std≈8.9e-16 (float residual) which would blow up normalization to ~1e15.
    feat_std = train_inst[INST_FEAT_COLS].std().astype('float32').fillna(1.0)
    feat_std = feat_std.where(feat_std > 1e-6, 1.0)

    return (torch.tensor(feat_mean.values, dtype=torch.float32),
            torch.tensor(feat_std.values, dtype=torch.float32))


def temporal_split_by_trip(df_all):
    first_dep = df_all.groupby('TRIP_ID')['departure_time'].min().rename('first_dep').reset_index()
    df_all = df_all.merge(first_dep, on='TRIP_ID', how='left')

    min_date, max_date = df_all['first_dep'].min(), df_all['first_dep'].max()
    total_days = max((max_date - min_date).days, 1)
    train_cut = min_date + pd.Timedelta(days=int(0.6 * total_days))
    val_cut = min_date + pd.Timedelta(days=int(0.8 * total_days))
    gap = pd.Timedelta(days=min(7, max(1, total_days // 20)))

    df_train = df_all[df_all['first_dep'] <= train_cut].copy()
    df_val = df_all[(df_all['first_dep'] > train_cut + gap) & (df_all['first_dep'] <= val_cut)].copy()
    df_test = df_all[df_all['first_dep'] > val_cut].copy()
    for _df in (df_train, df_val, df_test):
        if 'first_dep' in _df.columns:
            _df.drop(columns='first_dep', inplace=True)
    return df_train, df_val, df_test

def split_stats(df, name):
    print(f"{name}:")
    print("  rows:", len(df))
    print("  unique TRIP_ID:", df['TRIP_ID'].nunique())
    print("  time range:", df['departure_time'].min(), "->", df['departure_time'].max())



def speeds_to_tt(slot_block_12xP: np.ndarray,
                 length_m: float,
                 speed_metric_index=0,
                 speed_is_kmh=False,
                 eps=1e-3) -> np.ndarray:
    v = slot_block_12xP[:, speed_metric_index]
    if speed_is_kmh:
        v = v / 3.6
    v = np.maximum(v, eps)
    return float(length_m) / v

def build_first_order_relations_with_uturn(edges_gdf: pd.DataFrame):
    """
    Build first-order and conflict relations + U-turns.

    Returns a dict of 6 relations, each as a 2×E LongTensor:
      - upstream   : (i -> j) where j ends at u_i
      - downstream : (i -> j) where j starts at v_i
      - left_turn  : conflicts at v_i classified as left (inbound->inbound)
      - right_turn : conflicts at v_i classified as right (inbound->inbound)
      - opposite   : conflicts at v_i ~180° opposite (inbound->inbound)
      - u_turn     : (i -> j) where i ends at v_i and j starts at v_i, angle ≈ 180° (inbound->outbound)
    """
    g = edges_gdf.copy()
    if ("u" not in g.columns) or ("v" not in g.columns):
        raise ValueError("edges_gdf must have u,v columns before building relations.")
    if "edge_idx" not in g.columns:
        raise ValueError("edges_gdf must have edge_idx before building relations.")

    # bearing along edge direction
    g["bearing_in"] = g["geometry"].apply(_bearing_toward_end)

    out_by_u = g.groupby("u")["edge_idx"].apply(list).to_dict()  # edges starting at u
    in_by_v  = g.groupby("v")["edge_idx"].apply(list).to_dict()  # edges ending at v

    bmap = dict(zip(g["edge_idx"], g["bearing_in"]))

    up_src, up_dst = [], []
    dn_src, dn_dst = [], []
    lt_src, lt_dst = [], []
    rt_src, rt_dst = [], []
    op_src, op_dst = [], []
    ut_src, ut_dst = [], []

    SAME_THR   = math.radians(30)   # your original ±30°
    OPP_THR    = math.radians(30)   # your original ±30° around 180°
    UTURN_THR  = math.radians(20.0) # a bit tighter for U-turn

    # --- 1) First-order + conflict relations (Opposite same as your original) ---
    for row in g.itertuples(index=False):
        i = row.edge_idx
        u, v = row.u, row.v
        theta_i = bmap[i]

        # upstream: edges ending at u
        for j in in_by_v.get(u, []):
            if j != i:
                up_src.append(i); up_dst.append(j)

        # downstream: edges starting at v
        for j in out_by_u.get(v, []):
            if j != i:
                dn_src.append(i); dn_dst.append(j)

        # conflicts: other inbound edges to v
        for j in in_by_v.get(v, []):
            if j == i:
                continue
            theta_j = bmap[j]
            
            d = _ang_diff(theta_j, theta_i)

            # same direction (ignore)
            if abs(d) <= SAME_THR:
                continue

            # Opposing: near 180° (your original rule)
            if abs(abs(d) - math.pi) <= OPP_THR:
                op_src.append(i); op_dst.append(j)
                continue

            # Left vs right based on sign
            if 0 < d < (math.pi - OPP_THR):
                lt_src.append(i); lt_dst.append(j)
            elif -(math.pi - OPP_THR) < d < 0:
                rt_src.append(i); rt_dst.append(j)
            else:
                # near ±π weird edge cases; fall back to sign
                if d > 0:
                    lt_src.append(i); lt_dst.append(j)
                else:
                    rt_src.append(i); rt_dst.append(j)

    # --- 2) U-turns: inbound (u_i->v_i) followed by outbound from v_i, angle ≈ 180° ---
    for row in g.itertuples(index=False):
        i = row.edge_idx
        v_i = row.v
        theta_i = bmap[i]

        # candidate next edges: all edges starting at v_i
        for j in out_by_u.get(v_i, []):
            if j == i:
                continue
            theta_j = bmap[j]
            d = _ang_diff(theta_i, theta_j)

            # angle ≈ 180° with tighter tolerance
            if abs(abs(d) - math.pi) <= UTURN_THR:
                ut_src.append(i)
                ut_dst.append(j)

    # --- 3) Remove U-turn pairs from Opposite so they don’t double-count ---
    ut_pairs = set(zip(ut_src, ut_dst))
    op_src2, op_dst2 = [], []
    for s, d in zip(op_src, op_dst):
        if (s, d) not in ut_pairs:
            op_src2.append(s)
            op_dst2.append(d)

    return {
        "upstream":   torch.tensor([up_src, up_dst], dtype=torch.long),
        "downstream": torch.tensor([dn_src, dn_dst], dtype=torch.long),
        "left_turn":  torch.tensor([lt_src, lt_dst], dtype=torch.long),
        "right_turn": torch.tensor([rt_src, rt_dst], dtype=torch.long),
        "opposite":   torch.tensor([op_src2, op_dst2], dtype=torch.long),
        "u_turn":     torch.tensor([ut_src, ut_dst], dtype=torch.long),
    }


def _build_src2dst(edge_index: torch.LongTensor):
    d = defaultdict(list)
    s = edge_index[0].tolist()
    t = edge_index[1].tolist()
    for u, v in zip(s, t):
        d[u].append(v)
    return d

def _bfs_hops_global(src2dst_full: dict, start_edge: int, cap: int = 512) -> dict:
    dist = {start_edge: 0}
    q = deque([start_edge])
    while q:
        u = q.popleft()
        du = dist[u]
        for v in src2dst_full.get(u, []):
            if v not in dist:
                d = du + 1
                dist[v] = d
                if d < cap:
                    q.append(v)
    return dist

def rel_info(name, ei):
    print(f"{name:10s} | num_relations: {ei.size(1)}")

class EdgeDynStore:
    """Time-aligned 12×P lookup for any (edge_idx, t_ref) with optional mean-window."""
    def __init__(self, df_all: pd.DataFrame, dyn_cols_12xP: List[str],
                 edges_gdf: pd.DataFrame, way_id_col_df: str, way_id_col_edges: str, time_col: str):
        self.dyn_cols = dyn_cols_12xP
        self.time_col = time_col

        id2edge = dict(zip(edges_gdf[way_id_col_edges].tolist(), edges_gdf["edge_idx"].tolist()))
        df = df_all.dropna(subset=[time_col]).copy()
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df = df.dropna(subset=[time_col])

        if "edge_idx" not in df.columns:
            df["edge_idx"] = df[way_id_col_df].map(id2edge)
        df = df.dropna(subset=["edge_idx"])
        df["edge_idx"] = df["edge_idx"].astype(int)

        for c in self.dyn_cols:
            if c not in df.columns:
                df[c] = 0.0

        P = max(1, len(CFG["dyn_prefixes"]))
        self.idx: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

        for eidx, grp in df.groupby("edge_idx"):
            g = grp.sort_values(time_col)
            mat = g[self.dyn_cols].to_numpy(dtype="float32")
            arr = mat.reshape(len(g), 12, P).astype("float32")

            if np.isnan(arr).any():
                for r in range(arr.shape[0]):
                    for j in range(arr.shape[2]):
                        col = arr[r, :, j]
                        if np.isnan(col).any():
                            mu = np.nanmean(col)
                            if not np.isfinite(mu): mu = 0.0
                            col[np.isnan(col)] = mu
                        arr[r, :, j] = col

            t64 = g[time_col].astype("int64").to_numpy()
            self.idx[int(eidx)] = (t64, arr)

    def get(self, edge_idx: int, t_ref: pd.Timestamp, max_gap_seconds=1800) -> torch.Tensor:
        """Get dynamic features for edge at t_ref. Only uses PAST records to avoid future leakage."""
        if edge_idx not in self.idx:
            return torch.zeros(12, len(CFG["dyn_prefixes"]), dtype=torch.float32)
        t64, arr = self.idx[edge_idx]
        tref = np.int64(pd.Timestamp(t_ref).value)
        pos = bisect.bisect_right(t64, tref)
        
        # Only use records BEFORE t_ref (no future leakage)
        if pos > 0:
            idx = pos - 1
            gap = (int(tref) - int(t64[idx])) / 1e9
            if gap <= max_gap_seconds:
                return torch.tensor(arr[idx], dtype=torch.float32)
        
        # No valid past record within gap -> return zeros
        return torch.zeros(12, len(CFG["dyn_prefixes"]), dtype=torch.float32)

    def get_mean_window(self, edge_idx: int, t_ref: pd.Timestamp,
                        window_seconds: int = 1800,
                        max_gap_seconds: int = 1800,
                        fallback_to_nearest: bool = True) -> torch.Tensor:
        if edge_idx not in self.idx:
            return torch.zeros(12, len(CFG["dyn_prefixes"]), dtype=torch.float32)
        t64, arr = self.idx[edge_idx]
        tref = np.int64(pd.Timestamp(t_ref).value)

        lo = tref - int(window_seconds * 1e9)
        hi = tref
        left = bisect.bisect_left(t64, lo)
        right = bisect.bisect_right(t64, hi)
        if right > left:
            block_mean = arr[left:right].mean(axis=0)
            return torch.tensor(block_mean, dtype=torch.float32)
        if fallback_to_nearest:
            return self.get(edge_idx, t_ref, max_gap_seconds=max_gap_seconds)
        return torch.zeros(12, len(CFG["dyn_prefixes"]), dtype=torch.float32)

def _compute_time_feature_vector(t_ref: pd.Timestamp, cols: List[str]) -> Dict[str, float]:
    t_ref = pd.Timestamp(t_ref)
    out = {}
    if "tod_sin" in cols or "tod_cos" in cols:
        tod = t_ref.hour + t_ref.minute / 60.0
        tod_rad = 2 * np.pi * (tod / 24.0)  
        if "tod_sin" in cols: out["tod_sin"] = float(np.sin(tod_rad))
        if "tod_cos" in cols: out["tod_cos"] = float(np.cos(tod_rad))
    if "dow_sin" in cols or "dow_cos" in cols or "day_of_week" in cols:
        dow = t_ref.dayofweek
        dow_rad = 2 * np.pi * (dow / 7.0)
        if "day_of_week" in cols: out["day_of_week"] = float(dow)
        if "dow_sin" in cols: out["dow_sin"] = float(np.sin(dow_rad))
        if "dow_cos" in cols: out["dow_cos"] = float(np.cos(dow_rad))
    if "month" in cols: out["month"] = float(t_ref.month)
    if "day_of_month" in cols: out["day_of_month"] = float(t_ref.day)
    if "day_of_year" in cols: out["day_of_year"] = float(t_ref.dayofyear)
    return out


# =============================================================================
# [2026-06-07] Off-route-neighbour group cache (ADDITIVE).
# TripDatasetLinkETA._build_groups() is the expensive per-trip work (neighbour
# sampling + dense [M,M] relation masks + BFS + dynamic windows). These helpers
# let us serialise the built `self.groups` to disk and reload it, keyed by a
# fingerprint of everything that affects the build. The build logic itself is
# UNCHANGED; a cache miss / stale fingerprint / any error => build as usual.
# Masks are stored SPARSE (COO indices) since dense [M,M] x R is ~1 MB/trip.
# =============================================================================
_DATASET_CACHE_SCHEMA = 2  # bump if the per-trip dict layout changes


def _dataset_cache_fingerprint(df_all, K_first, K_conflict, K_high,
                               max_nodes, seed, rel_edges_list, dyn_feat_cols):
    """Stable hash of everything that changes the built groups."""
    import hashlib
    h = hashlib.md5()
    # [2026-06-07] hash TRIP_IDs as STRINGS (dtype-agnostic). Using to_numeric here
    # silently dropped non-numeric IDs (e.g. Chengdu) -> empty -> identical hash for
    # every split -> val/test wrongly loaded train's groups. Strings fix that.
    tids = sorted(df_all["TRIP_ID"].astype(str).unique().tolist())
    h.update(("\n".join(tids)).encode())
    h.update(("|ntrips=%d|" % len(tids)).encode())
    h.update(str((int(K_first), int(K_conflict), int(K_high),
                  int(max_nodes), int(seed))).encode())
    for t in (rel_edges_list or []):
        try:
            s = int(t.sum().item()) if t.numel() else 0
            h.update(str((tuple(t.shape), s)).encode())
        except Exception:
            h.update(b"relerr")
    h.update("|".join(map(str, dyn_feat_cols or [])).encode())
    return h.hexdigest()[:16]


def _serialize_groups(groups):
    """Replace each trip's dense [M,M] relation masks with sparse (M, COO idx)."""
    out = []
    for g in groups:
        gg = dict(g)
        masks = gg.get("masks")
        if masks is not None:
            sp = []
            for m in masks:
                mm = m.bool()
                M = int(mm.shape[0])
                idx = mm.nonzero(as_tuple=False).to(torch.int32)  # [K, 2]
                sp.append((M, idx))
            gg["masks"] = ("sparse", sp)
        out.append(gg)
    return out


def _deserialize_groups(payload):
    """Rebuild dense [M,M] bool masks from the sparse (M, COO idx) form."""
    out = []
    for gg in payload:
        g = dict(gg)
        masks = g.get("masks")
        if isinstance(masks, tuple) and len(masks) == 2 and masks[0] == "sparse":
            dense = []
            for (M, idx) in masks[1]:
                d = torch.zeros(M, M, dtype=torch.bool)
                if idx.numel():
                    ii = idx.long()
                    d[ii[:, 0], ii[:, 1]] = True
                dense.append(d)
            g["masks"] = dense
        out.append(g)
    return out


class TripDatasetLinkETA(Dataset):
    """
    New dataset for LinkETA with:
      - Local per-trip graph over nodes = (route occurrences + off-route neighbours PER occurrence).
      - t_ref for off-route neighbours anchored at the time of the route edge they are attached to (Change 5).
      - Dynamic 12xP added via EdgeDynStore.get_mean_window within 30min window.
      - BFS-hop + route_flag encodings (Change 2).
      - Per-sequence route_seq_idx for sequence-level physics/neural (Change 8).
    """

    def __init__(
        self,
        df_all: pd.DataFrame,
        edges_gdf: pd.DataFrame,
        inst_table: pd.DataFrame,
        rel_edges_list: List[torch.Tensor],
        static_feat_cols: List[str],
        time_feat_cols: List[str],
        dyn_feat_cols: List[str],
        dyn_store: EdgeDynStore,
        center_idx: int = 0,
        K_first: int = 10,
        K_conflict: int = 5,
        K_high: int = 3,
        use_cache: bool = False,      # [2026-06-07] opt-in off-route-neighbour disk cache
        cache_dir: str = None,        # where groups_<fp>.pt lives (per city)
        cache_seed: int = 42,         # seeds neighbour sampling so the cache is reproducible
    ):
        super().__init__()
        self.df_all = df_all
        self.edges_gdf = edges_gdf
        self.inst_table = inst_table
        self.static_feat_cols = static_feat_cols or []
        self.time_feat_cols = time_feat_cols or []
        self.dyn_feat_cols = dyn_feat_cols or []
        self.dyn_store = dyn_store
        self.sort_col = CFG["time_col_df"] if CFG["time_col_df"] in df_all.columns else "enter_ts"

        self.P = len(CFG["dyn_prefixes"])
        self.idx_speed = CFG["dyn_prefixes"].index("median_speed") \
            if "median_speed" in CFG["dyn_prefixes"] else 0
        self.idx_count = CFG["dyn_prefixes"].index("count_vehicle") \
            if "count_vehicle" in CFG["dyn_prefixes"] else None

        # Free-flow v_ff
        vff_map = dict(zip(edges_gdf['edge_idx'], pd.to_numeric(edges_gdf.get('v_ff', np.nan), errors='coerce')))
        self.edge2vff = {int(k): float(v) if pd.notna(v) else 10.0 for k, v in vff_map.items()}

        if not (isinstance(inst_table.index, pd.MultiIndex) and inst_table.index.names[:2] == ['TRIP_ID', 'edge_idx']):
            inst_table = inst_table.set_index(['TRIP_ID', 'edge_idx']).sort_index()
        self.inst_table = inst_table

        cols = list(self.inst_table.columns)
        self.static_idx = [cols.index(c) for c in self.static_feat_cols if c in cols]
        self.time_idx = [cols.index(c) for c in self.time_feat_cols if c in cols]
        self.dyn_idx = [cols.index(c) for c in self.dyn_feat_cols if c in cols]

        # precompute recency weights for speed1h/logcnt1h
        tau = 600.0 / max(math.log(2.0), 1e-6)  # half-life 600s
        idx = torch.arange(H_SLICES, dtype=torch.float32)
        ages = (idx - float(center_idx)).abs() * float(BIN_SECONDS)
        self.w_rec = torch.exp(-ages / max(tau, 1.0))

        # relation dicts
        # rel_edges_list order: [self_edge, downstream, upstream, left, opposite, right, high, uturn]
        self.rel_src2dst_list = [_build_src2dst(ei) for ei in rel_edges_list]

        # maps for neighbour sampling (by edge_idx)
        down_map = self.rel_src2dst_list[1]
        up_map = self.rel_src2dst_list[2]
        lt_map = self.rel_src2dst_list[3]
        op_map = self.rel_src2dst_list[4]
        rt_map = self.rel_src2dst_list[5]
        high_map = self.rel_src2dst_list[6]

        self.first_map = defaultdict(set)
        for k in (down_map.keys() | up_map.keys()):
            self.first_map[k] |= set(down_map.get(k, []))
            self.first_map[k] |= set(up_map.get(k, []))

        self.conf_map = defaultdict(set)
        for k in (lt_map.keys() | op_map.keys() | rt_map.keys()):
            self.conf_map[k] |= set(lt_map.get(k, []))
            self.conf_map[k] |= set(op_map.get(k, []))
            self.conf_map[k] |= set(rt_map.get(k, []))

        self.high_map = defaultdict(set)
        for k, v in high_map.items():
            self.high_map[k] |= set(v)

        self.K_first = int(K_first)
        self.K_conflict = int(K_conflict)
        self.K_high = int(K_high)

        # BFS hop graph: use downstream adjacency (Change 2)
        self.hop_src2dst_full = self.rel_src2dst_list[1]

        # build per-trip groups
        # [2026-06-07] Optional disk cache (additive). Cache hit -> load; otherwise
        # build with the UNCHANGED logic and (optionally) save. Any error -> build.
        self.groups = []
        self._use_cache = bool(use_cache)
        self._cache_dir = cache_dir
        self._cache_seed = cache_seed
        self._cache_fp = None
        if self._use_cache and cache_dir is not None:
            self._cache_fp = _dataset_cache_fingerprint(
                df_all, self.K_first, self.K_conflict, self.K_high,
                CFG.get("max_nodes_local", 360), cache_seed,
                rel_edges_list, self.dyn_feat_cols)
        if not self._try_load_groups_cache():
            if self._use_cache and cache_seed is not None:
                # Seed neighbour sampling (random.shuffle) for a reproducible cache,
                # restoring global RNG afterwards so nothing else is perturbed.
                _st = random.getstate()
                random.seed(int(cache_seed))
                try:
                    self._build_groups()
                finally:
                    random.setstate(_st)
            else:
                self._build_groups()
            self._try_save_groups_cache()

    def _build_groups(self):
        df_small = self.df_all[["TRIP_ID", "edge_idx", "link_len_m", "link_travel_time_s",
                                "trip_travel_time_s", self.sort_col]].copy()
        for c in ('link_len_m', 'link_travel_time_s', 'trip_travel_time_s'):
            if c in df_small.columns:
                df_small[c] = pd.to_numeric(df_small[c], errors='coerce')

        for trip_id, grp in df_small.groupby('TRIP_ID', sort=False):
            grp = grp.sort_values(self.sort_col)
            edge_seq_on = pd.to_numeric(grp['edge_idx'], errors='coerce').dropna().astype(int).to_numpy()
            if edge_seq_on.size == 0:
                continue
            t_arr_on = pd.to_datetime(grp[self.sort_col].to_numpy())
            L_on = len(edge_seq_on)

            link_len_np = pd.to_numeric(grp['link_len_m'], errors='coerce').fillna(5.0).to_numpy()
            true_tt_np = pd.to_numeric(grp['link_travel_time_s'], errors='coerce').fillna(0.0).to_numpy()

            # trip-level label: prefer trip_travel_time_s if available, else sum of link_tt
            t_val = float(grp['trip_travel_time_s'].iloc[0]) \
                if 'trip_travel_time_s' in grp.columns and pd.notna(grp['trip_travel_time_s'].iloc[0]) \
                else float(np.nansum(true_tt_np))
            if not (np.isfinite(t_val) and t_val > 0):
                t_val = float(np.nansum(true_tt_np))
            trip_tt = torch.tensor(t_val, dtype=torch.float32)
            trip_tt_label = torch.tensor(float(np.nansum(true_tt_np)), dtype=torch.float32)

            # build nodes (route occurrences + off-route neighbors per occurrence) (Change 5 & Option 1)
            max_nodes = int(CFG.get("max_nodes_local", 360))
            node_edge_idx = []
            node_tref = []
            route_flag = []

            neigh_parent = []  # which route index this node is attached to (for neighbors)

            for k in range(L_on):
                e = int(edge_seq_on[k])
                t_ref = t_arr_on[k]

                node_edge_idx.append(e)
                node_tref.append(t_ref)
                route_flag.append(True)
                neigh_parent.append(k)

            for k in range(L_on):
                if len(node_edge_idx) >= max_nodes:
                    break

                e = int(edge_seq_on[k])
                t_ref = t_arr_on[k]

                cand_first = set(self.first_map.get(e, set()))
                cand_conf  = set(self.conf_map.get(e, set()))
                cand_high  = set(self.high_map.get(e, set()))

                # don’t include the route edge itself as neighbour
                cand_first.discard(e)
                cand_conf.discard(e)
                cand_high.discard(e)

                def sample_set(S, K):
                    if K <= 0 or not S:
                        return []
                    Ls = list(S)
                    random.shuffle(Ls)
                    return Ls[:K] if len(Ls) > K else Ls

                nbr_first = sample_set(cand_first, self.K_first)
                nbr_conf  = sample_set(cand_conf,  self.K_conflict)
                nbr_high  = sample_set(cand_high,  self.K_high)
                nbrs = nbr_first + nbr_conf + nbr_high

                for e_n in nbrs:
                    if len(node_edge_idx) >= max_nodes:
                        break
                    node_edge_idx.append(int(e_n))
                    node_tref.append(t_ref)      # t_ref = time when we hit route edge k (Change 5)
                    route_flag.append(False)
                    neigh_parent.append(k)

            M = len(node_edge_idx)
            node_edge_idx = np.array(node_edge_idx, dtype=int)
            route_flag = torch.tensor(route_flag, dtype=torch.bool)
            # neigh_parent = np.array(neigh_parent, dtype=int)
            neigh_parent_tensor = torch.tensor(neigh_parent, dtype=torch.long)

            # route sequence mapping (Change 8)
            route_seq_idx = []
            seq_true_tt = []
            seq_link_len = []
            seq_mask = []

            for k in range(L_on):
                # NOTE: Nodes are added in TWO passes:
                #   Pass 1: ALL route nodes [route0, route1, ..., routeN]
                #   Pass 2: ALL neighbor nodes grouped by their parent route index
                # So node order is: [all route nodes, then all neighbors], NOT interleaved.
                # neigh_parent[j] tracks which route index node j is attached to.
                node_pos = None
                # node index is where neigh_parent == k AND route_flag == True
                for j in range(M):
                    if neigh_parent[j] == k and route_flag[j]:
                        node_pos = j
                        break
                if node_pos is None:
                    continue
                route_seq_idx.append(node_pos)
                seq_true_tt.append(float(true_tt_np[k]))
                seq_link_len.append(float(link_len_np[k]))
                seq_mask.append(True)

            if len(route_seq_idx) == 0:
                continue

            route_seq_idx = torch.tensor(route_seq_idx, dtype=torch.long)
            seq_true_tt = torch.tensor(seq_true_tt, dtype=torch.float32)
            seq_link_len = torch.tensor(seq_link_len, dtype=torch.float32)
            seq_mask = torch.tensor(seq_mask, dtype=torch.bool)

            # x_seq from inst_table
            try:
                sub = self.inst_table.xs(trip_id, level=0, drop_level=True)
                rows = sub.reindex(node_edge_idx, fill_value=0.0).to_numpy(dtype='float32', copy=False)
            except KeyError:
                Fdim = self.inst_table.shape[1]
                rows = np.zeros((M, Fdim), dtype='float32')
            x_seq = torch.tensor(rows, dtype=torch.float32)

            # OFF-ROUTE: fill full features (static + time + dynamic-window) (Change 5)
            # route nodes already have inst features; we enrich both route & neighbours with dynamic-window + time
            cols_all = self.inst_table.columns.tolist()
            for j in range(M):
                e = int(node_edge_idx[j])
                t_ref = node_tref[j]

                # time features
                if self.time_feat_cols:
                    tf = _compute_time_feature_vector(t_ref, self.time_feat_cols)
                    for col_name, val in tf.items():
                        if col_name in cols_all:
                            idx_col = cols_all.index(col_name)
                            x_seq[j, idx_col] = float(val)

                # dynamic window (mean over last 30min, per Change 5)
                if self.dyn_feat_cols:
                    block = self.dyn_store.get_mean_window(e, t_ref)  # [12,P]
                    flat = block.reshape(-1).cpu().numpy().astype("float32")
                    m = min(len(flat), len(self.dyn_idx))
                    if m > 0:
                        x_seq[j, self.dyn_idx[:m]] = torch.from_numpy(flat[:m])

            # recency features for speed1h/logcnt1h (keep as original style)
            speed_vals = torch.zeros(M, H_SLICES)
            count_vals = torch.zeros(M, H_SLICES) if self.idx_count is not None else None

            for j in range(M):
                e = int(node_edge_idx[j])
                t_ref = node_tref[j]
                block = self.dyn_store.get(e, t_ref)
                v = block[:, self.idx_speed]
                if CFG["speed_is_kmh"]:
                    v = v / 3.6
                speed_vals[j] = v
                if self.idx_count is not None:
                    count_vals[j] = block[:, self.idx_count]

            if self.idx_count is not None:
                valid = (count_vals > 0).float()
                w = (self.w_rec.view(1, -1) * valid)
            else:
                w = self.w_rec.view(1, -1).expand_as(speed_vals)

            den = w.sum(dim=1).clamp_min(1.0)
            vff_seq_l = torch.tensor([self.edge2vff.get(int(e), 10.0) for e in node_edge_idx],
                                     dtype=torch.float32)

            sp_ratio_agg = (speed_vals * w).sum(dim=1) / den
            sp_ratio_agg = (sp_ratio_agg / vff_seq_l.clamp_min(1.0)).clamp(0.0, 3.5)

            if self.idx_count is not None:
                logcnt1h_node = torch.log1p((count_vals * w).sum(dim=1))
            else:
                logcnt1h_node = torch.zeros_like(sp_ratio_agg)

            speed_w_node = den

            # adjacency masks over nodes via global src2dst dicts (Change 1 & 3)
            edge_idx_to_positions = defaultdict(list)
            for j, e in enumerate(node_edge_idx):
                edge_idx_to_positions[int(e)].append(j)

            masks = []
            for src2dst in self.rel_src2dst_list:
                mat = torch.zeros(M, M, dtype=torch.bool)
                for s in set(node_edge_idx.tolist()):
                    pos_s_list = edge_idx_to_positions[int(s)]
                    for d in src2dst.get(int(s), []):
                        pos_d_list = edge_idx_to_positions.get(int(d), [])
                        for i_s in pos_s_list:
                            for i_d in pos_d_list:
                                if i_s != i_d:
                                    mat[i_s, i_d] = True
                masks.append(mat)

            # BFS hop distances (Change 2)
            start_edge = int(edge_seq_on[0])
            dist_full = _bfs_hops_global(self.hop_src2dst_full, start_edge, cap=512)
            hop_idx = torch.tensor([dist_full.get(int(e), 512) for e in node_edge_idx], dtype=torch.long)

            # store group
            self.groups.append(dict(
                trip_id=int(trip_id),
                node_edge_idx=torch.tensor(node_edge_idx, dtype=torch.long),
                x_seq=x_seq,
                route_flag=route_flag,
                hop_idx=hop_idx,
                masks=masks,
                sp_ratio_agg=sp_ratio_agg,
                logcnt1h_node=logcnt1h_node,
                speed_w_node=speed_w_node,
                route_seq_idx=route_seq_idx,
                seq_true_tt=seq_true_tt,
                seq_link_len=seq_link_len,
                seq_mask=seq_mask,
                trip_tt=trip_tt,
                trip_tt_label=trip_tt_label,
                neigh_parent=neigh_parent_tensor,
            ))

    # ------------------------------------------------------------------
    # [2026-06-07] Off-route-neighbour disk cache (load/save). Robust: any
    # failure or fingerprint/schema mismatch falls back to a normal build.
    # ------------------------------------------------------------------
    def _cache_path(self):
        return os.path.join(self._cache_dir, f"groups_{self._cache_fp}.pt")

    def _try_load_groups_cache(self):
        if not (self._use_cache and self._cache_dir and self._cache_fp):
            return False
        path = self._cache_path()
        if not os.path.isfile(path):
            print(f"[dataset-cache] MISS {path} -> building")
            return False
        try:
            pkg = torch.load(path, map_location="cpu", weights_only=False)
            if (pkg.get("schema") != _DATASET_CACHE_SCHEMA
                    or pkg.get("fingerprint") != self._cache_fp):
                print(f"[dataset-cache] STALE {path} "
                      f"(schema/fp mismatch) -> rebuilding")
                return False
            self.groups = _deserialize_groups(pkg["groups"])
            print(f"[dataset-cache] HIT  {path}  ({len(self.groups)} trips)")
            return True
        except Exception as e:
            print(f"[dataset-cache] load failed ({e}) -> rebuilding")
            return False

    def _try_save_groups_cache(self):
        if not (self._use_cache and self._cache_dir and self._cache_fp):
            return
        try:
            os.makedirs(self._cache_dir, exist_ok=True)
            path = self._cache_path()
            tmp = path + ".tmp"
            torch.save({"schema": _DATASET_CACHE_SCHEMA,
                        "fingerprint": self._cache_fp,
                        "groups": _serialize_groups(self.groups)}, tmp)
            os.replace(tmp, path)
            print(f"[dataset-cache] SAVED {path}  ({len(self.groups)} trips)")
        except Exception as e:
            print(f"[dataset-cache] save failed ({e}) -> continuing without cache")

    def __len__(self): return len(self.groups)
    def __getitem__(self, i): return self.groups[i]

def collate_trips_linketa(batch):
    B = len(batch)
    M_list = [g["node_edge_idx"].numel() for g in batch]
    L_list = [g["route_seq_idx"].numel() for g in batch]
    Mmax = max(M_list)
    Lmax = max(L_list)

    in_dim = batch[0]["x_seq"].size(1)

    node_edge_idx = torch.zeros(B, Mmax, dtype=torch.long)
    x_seq = torch.zeros(B, Mmax, in_dim, dtype=torch.float32)
    route_flag = torch.zeros(B, Mmax, dtype=torch.bool)
    hop_idx = torch.zeros(B, Mmax, dtype=torch.long)
    sp_ratio_agg = torch.zeros(B, Mmax, dtype=torch.float32)
    logcnt1h_node = torch.zeros(B, Mmax, dtype=torch.float32)
    speed_w_node = torch.zeros(B, Mmax, dtype=torch.float32)

    masks_per_rel = [
        torch.zeros(B, Mmax, Mmax, dtype=torch.bool)
        for _ in range(len(batch[0]["masks"]))
    ]

    route_seq_idx = torch.zeros(B, Lmax, dtype=torch.long)
    seq_true_tt = torch.zeros(B, Lmax, dtype=torch.float32)
    seq_link_len = torch.zeros(B, Lmax, dtype=torch.float32)
    seq_mask = torch.zeros(B, Lmax, dtype=torch.bool)
    trip_tt = torch.zeros(B, dtype=torch.float32)
    trip_tt_label = torch.zeros(B, dtype=torch.float32)
    trip_ids = []  # Store as list since TRIP_IDs may exceed int64 range

    for i, g in enumerate(batch):
        M = M_list[i]
        node_edge_idx[i, :M] = g["node_edge_idx"]
        x_seq[i, :M] = g["x_seq"]
        route_flag[i, :M] = g["route_flag"]
        hop_idx[i, :M] = g["hop_idx"]
        sp_ratio_agg[i, :M] = g["sp_ratio_agg"]
        logcnt1h_node[i, :M] = g["logcnt1h_node"]
        speed_w_node[i, :M] = g["speed_w_node"]
        for r in range(len(masks_per_rel)):
            masks_per_rel[r][i, :M, :M] = g["masks"][r]

        L = L_list[i]
        route_seq_idx[i, :L] = g["route_seq_idx"]
        seq_true_tt[i, :L] = g["seq_true_tt"]
        seq_link_len[i, :L] = g["seq_link_len"]
        seq_mask[i, :L] = g["seq_mask"]
        trip_tt[i] = g["trip_tt"]
        trip_tt_label[i] = g["trip_tt_label"]
        trip_ids.append(g["trip_id"])  # Append to list instead of tensor assignment

    return dict(
        node_edge_idx=node_edge_idx,
        x_seq=x_seq,
        route_flag=route_flag,
        hop_idx=hop_idx,
        masks_per_rel=masks_per_rel,
        sp_ratio_agg=sp_ratio_agg,
        logcnt1h_node=logcnt1h_node,
        speed_w_node=speed_w_node,
        route_seq_idx=route_seq_idx,
        seq_true_tt=seq_true_tt,
        seq_link_len=seq_link_len,
        seq_mask=seq_mask,
        trip_tt=trip_tt,
        trip_tt_label=trip_tt_label,
        trip_ids=trip_ids,
    )

def inspect_offroute_neighbours_for_trip(ds: TripDatasetLinkETA, idx: int = 0, max_k: int = 10):
    """
    For a given trip index in the dataset, print the on-route edges and their off-route neighbours.
    """
    g = ds[idx]
    node_edge_idx = g["node_edge_idx"].cpu().numpy()
    route_seq_idx = g["route_seq_idx"].cpu().numpy()
    route_flag = g["route_flag"].cpu().numpy()
    neigh_parent = g["neigh_parent"].cpu().numpy()

    print(f"Trip index={idx}, trip_id={g['trip_id']}")
    print(f"Total nodes={len(node_edge_idx)}, on-route positions={len(route_seq_idx)}")

    for k, node_idx in enumerate(route_seq_idx[:max_k]):
        route_edge = node_edge_idx[node_idx]
        mask = (neigh_parent == k) & (~route_flag)
        neigh_nodes = np.where(mask)[0]
        neigh_edges = node_edge_idx[neigh_nodes]

        print(f"Route pos {k:3d} | edge_idx={route_edge} | num_neighbours={len(neigh_edges)}")
        if len(neigh_edges) > 0:
            print("   neighbours:", neigh_edges.tolist())

class RelationalTransformerLayer(nn.Module):
    def __init__(self, dim, heads, num_relations, dropout=0.3):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dk = dim // heads
        self.R = num_relations

        self.to_q = nn.ModuleList([nn.Linear(dim, dim, bias=True) for _ in range(self.R)])
        self.to_k = nn.ModuleList([nn.Linear(dim, dim, bias=True) for _ in range(self.R)])
        self.to_v = nn.ModuleList([nn.Linear(dim, dim, bias=True) for _ in range(self.R)])

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, masks: List[torch.Tensor]):
        """
        x: [B,M,D]; masks: list of R tensors [B,M,M] bool
        """
        B, M, D = x.shape
        agg = torch.zeros(B, M, self.heads, self.dk, device=x.device, dtype=x.dtype)

        for r in range(self.R):
            q = self.to_q[r](x).view(B, M, self.heads, self.dk)
            k = self.to_k[r](x).view(B, M, self.heads, self.dk)
            v = self.to_v[r](x).view(B, M, self.heads, self.dk)

            scores = torch.einsum("bmhd,bnhd->bhmn", q, k) / math.sqrt(self.dk)
            mask = masks[r].unsqueeze(1)  # [B,1,M,M]
            scores = scores.masked_fill(~mask, float("-inf"))
            alpha = F.softmax(scores, dim=-1)
            alpha = torch.where(torch.isfinite(alpha), alpha, torch.zeros_like(alpha))
            alpha = self.dropout(alpha)
            out_r = torch.einsum("bhmn,bnhd->bmhd", alpha, v)
            agg += out_r

        y = x + agg.reshape(B, M, D) / (self.R * self.heads)
        return y

class RouteAwareEncodings(nn.Module):
    def __init__(self, max_hop=512, dim=32):
        super().__init__()
        self.pos_emb = nn.Embedding(max_hop, dim)
        self.route_emb = nn.Embedding(2, dim)

    def forward(self, h, hop_idx, route_flag):
        hop_idx = hop_idx.clamp_min(0).clamp_max(self.pos_emb.num_embeddings - 1)
        route_id = route_flag.clamp(0, 1)
        return h + self.pos_emb(hop_idx) + self.route_emb(route_id)

class RouteSeqPosEncoding(nn.Module):
    """
    Positional embedding for the index along the route (0..L-1).
    This is *in addition* to hop_idx / route_flag encodings.
    """
    def __init__(self, max_len: int, dim: int):
        super().__init__()
        self.pos_emb = nn.Embedding(max_len, dim)

    def forward(self, h_seq: torch.Tensor, seq_mask: torch.Tensor) -> torch.Tensor:
        """
        h_seq: [B,L,D], seq_mask: [B,L] (unused but kept for API symmetry)
        """
        B, L, D = h_seq.shape
        # Positions 0..L-1 (shared across batch)
        pos = torch.arange(L, device=h_seq.device).unsqueeze(0).expand(B, L)  # [B,L]
        pos = pos.clamp_max(self.pos_emb.num_embeddings - 1)
        return h_seq + self.pos_emb(pos)

class LinkETA(nn.Module):
    """
    [2026-05-01 Stage 3] sp_hat-anchored ETA with TIGHT residual + route-calib.

    Three heads, one anchored prediction:
      - speed_head    -> sp_hat ∈ [0, r_max=3]   (PRIMARY: per-link traffic state)
      - residual_head -> delta  ∈ [-0.2, 0.2]    (per-link residual; multiplier exp(delta) ∈ [0.82, 1.22])
      - route_calib   -> g      ∈ [0.9, 1.1]     (per-trip scale; ±10 % global multiplier)

    Forward composition:
        eta_base   = seq_link_len / (sp_hat · vff_seq)        # bounded ETA in seconds
        link_eta   = eta_base · exp(delta) · g.unsqueeze(-1)  # tight residual adjustments
        trip_eta   = Σ link_eta over the route mask

    History:
      - 2026-04-28 (A1): residual_head and route_calib were REMOVED because the
        labels were unbounded (~10 % of rows had sp_actual >> 3). With wide
        residuals (delta ∈ [-2, 2], multiplier exp(delta) ∈ [0.14, 7.4]) the
        model could route any prediction past the speed bound to fit absurd
        labels — three multiplicative knobs were unidentifiable.
      - 2026-05-01 (Stage 3): residual_head and route_calib are RE-ENABLED but
        with much tighter ranges (delta_range=0.2, g_range=0.1). After Stage 1
        redistribution every supervised row has sp_actual <= r_max, and after
        the Stage 2 target swap sp_hat is anchored by direct supervision
        against per-link sp_actual (not the historical proxy). With sp_hat
        pinned, delta/g have a well-defined small-residual role: per-link
        adjustments and per-trip scale that sp_hat alone can't express.
        Tight bounds preserve the speed bound on link_eta, and zero-init on
        the last conv/linear layers means delta=0 and g=1 at step 0.

    Initialization:
      - speed_head   last conv: zero weight + bias  ⇒ sp_hat = 0.5·r_max ≈ 1.5
      - residual_head last conv: zero weight + bias ⇒ delta = 0
      - route_calib   last linear: zero weight + bias ⇒ g = 1.0
      - count_head   last conv: zero weight + bias  ⇒ logcnt_1h = 0
      At step 0: link_eta = link_len / (1.5 · vff) — sane warm start near
      free-flow time, no chaotic init.

    Interface (unchanged):
      forward(x_seq_nodes, hop_idx, route_flag_nodes, masks_per_rel,
              route_seq_idx, seq_link_len, vff_seq, seq_mask,
              r_max=3.0, return_parts=True)
      returns 7-tuple:
        (link_eta_seq, trip_eta, sp_hat, logcnt_1h, g, delta, eta_phys)
    """
    def __init__(self, in_dim, hidden_dim=32, heads=8, relations=8,
                 num_layers=1, max_route_len=1052, num_experts=3, dropout=0.1,
                 input_dropout=0.1):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.relations = relations

        # ------------------------------------------------------------------
        # 1) Input projection + route-aware positional encodings
        # ------------------------------------------------------------------
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.inp_dropout = nn.Dropout(input_dropout)
        self.route_enc = RouteAwareEncodings(max_hop=512, dim=hidden_dim)

        # ------------------------------------------------------------------
        # 2) Temporal encoder for 12×P dynamic slots per edge
        # ------------------------------------------------------------------
        # We assume CFG["dyn_prefixes"] and 12 slots (slots_per_hour) like in your pipeline.
        self.P = len(CFG["dyn_prefixes"])
        self.H_SLICES = CFG.get("slots_per_hour", 12)

        # Conv1d expects [batch, channels, seq_len] = [B*M, P, 12]
        self.dyn_encoder = nn.Sequential(
            nn.Conv1d(self.P, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.dyn_pool = nn.AdaptiveAvgPool1d(1)  # pool over time -> [B*M, hidden_dim, 1]
        # Gating so we can start near the original behaviour
        self.dyn_gate = nn.Parameter(torch.tensor(0.0))

        # ------------------------------------------------------------------
        # 3) Relational "transformer" layers over local trip graph
        # ------------------------------------------------------------------
        self.layers = nn.ModuleList([
            RelationalTransformerLayer(hidden_dim, heads, relations, dropout=dropout)
            for _ in range(num_layers)
        ])

        # ------------------------------------------------------------------
        # 4) Aux heads: speed ratio & volume (per sequence position)
        # ------------------------------------------------------------------
        self.speed_head = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, 1, 1),
        )
        self.count_head = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, 1, 1),
        )

        # ------------------------------------------------------------------
        # 5) [2026-05-01 Stage 3] Log-space residual head — RE-ENABLED with TIGHT bounds.
        #    Previously disabled in A1 because labels were unbounded and the wide
        #    delta range (multiplier 0.14..7.4) let the model bypass the speed bound.
        #    After Stage 1 (redistribution) labels are bounded; after Stage 2
        #    (target swap) sp_hat is anchored by direct supervision. With sp_hat
        #    pinned, delta does its proper job: small per-link residual.
        #    Range: delta = self.delta_range · tanh(...) with delta_range = 0.2
        #    ⇒ multiplier exp(delta) ∈ [exp(-0.2), exp(0.2)] ≈ [0.82, 1.22].
        # ------------------------------------------------------------------
        self.residual_head = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, 1, 1),
        )
        # [2026-04-28 A1 OLD — DISABLED]
        # (residual_head was removed entirely; old block kept here for trace)

        # ------------------------------------------------------------------
        # 6) [2026-05-01 Stage 3] Route-level calibrator g — RE-ENABLED with TIGHT range.
        #    Previously disabled in A1 because g_range=0.5 (multiplier 0.5..1.5)
        #    was wide enough to undermine the speed bound. New g_range=0.1 keeps
        #    g ∈ [0.9, 1.1] — a ±10 % per-trip multiplier for systematic offsets
        #    (driver style, weather, time-of-day not captured by sp_hat features).
        # ------------------------------------------------------------------
        self.route_calib = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # [2026-04-28 A1 OLD] self.g_range = 0.5  (replaced; too wide)
        # [2026-05-01 Stage 3 OLD] tight bounds — kept here for rollback:
        # self.g_range     = 0.1   # ±10 %
        # self.delta_range = 0.2   # ±20 % via exp(delta)
        # [2026-05-01 Step 6, Change 3] Loosened bounds.
        # The training run on SF-corrected hit a convergence floor with train ≈ val,
        # signalling unused generalization headroom. The tight bounds were clamping
        # the model on hard / long trips where gradients vanished at the boundary.
        # Loosening lets the model apply larger corrections; if any overfit shows
        # (train ≪ val), revert by swapping the constants below.
        self.g_range     = 0.3   # ±30 % global multiplicative correction
        self.delta_range = 0.4   # ±40 % per-link correction via exp(delta)

        # ------------------------------------------------------------------
        # Zero-init last layer of EVERY head so the model starts at:
        #   sp_hat = 0.5 · r_max ≈ 1.5,  delta = 0,  g = 1.0,  logcnt = 0
        # ⇒ link_eta = link_len / (1.5 · vff) at step 0 (free-flow warm start).
        # ------------------------------------------------------------------
        nn.init.zeros_(self.speed_head[-1].weight)
        nn.init.zeros_(self.speed_head[-1].bias)
        nn.init.zeros_(self.count_head[-1].weight)
        nn.init.zeros_(self.count_head[-1].bias)
        nn.init.zeros_(self.residual_head[-1].weight)   # [2026-05-01 Stage 3]
        nn.init.zeros_(self.residual_head[-1].bias)     # [2026-05-01 Stage 3]
        nn.init.zeros_(self.route_calib[-1].weight)     # [2026-05-01 Stage 3]
        nn.init.zeros_(self.route_calib[-1].bias)       # [2026-05-01 Stage 3]

    def _encode_dyn_slots(self, x_seq_nodes: torch.Tensor) -> torch.Tensor:
        """
        Build a dynamic embedding for each node from the dynamic 12×P features.

        x_seq_nodes: [B, M, in_dim]
        Uses global DYN_IDX (indices of all dynamic columns) if available.

        Returns: dyn_emb [B, M, hidden_dim]
        """
        B, M, _ = x_seq_nodes.shape

        # NOTE: In original code, this line actually should be:
        # B, M, _ = x_seq_nodes.shape
        # but we keep the body exactly as provided by the user, including this.

        # If DYN_IDX isn't defined or has unexpected length, fall back to zeros.
        if "DYN_IDX" not in globals():
            return torch.zeros(B, M, self.hidden_dim, device=x_seq_nodes.device, dtype=x_seq_nodes.dtype)

        dyn_idx = DYN_IDX
        num_dyn = dyn_idx.numel()
        expected_dyn = self.H_SLICES * self.P
        if num_dyn != expected_dyn:
            # Safety: shapes don't match the assumed 12×P layout
            return torch.zeros(B, M, self.hidden_dim, device=x_seq_nodes.device, dtype=x_seq_nodes.dtype)

        # Gather dynamic columns: [B, M, num_dyn]
        x_dyn = x_seq_nodes[..., dyn_idx]  # indexing last dim with long tensor

        # Reshape to [B*M, H_SLICES, P] then to [B*M, P, H_SLICES] for Conv1d
        BM = B * M
        x_dyn = x_dyn.reshape(BM, self.H_SLICES, self.P).permute(0, 2, 1)  # [BM, P, 12]

        # Conv + pooling over time
        dyn_feat = self.dyn_encoder(x_dyn)              # [BM, hidden_dim, 12]
        dyn_feat = self.dyn_pool(dyn_feat).squeeze(-1)  # [BM, hidden_dim]
        dyn_feat = dyn_feat.view(B, M, self.hidden_dim) # [B, M, hidden_dim]

        # Gate it so we start near zero contribution
        gate = torch.sigmoid(self.dyn_gate)             # scalar in (0,1)
        return gate * dyn_feat

    def forward(self,
                x_seq_nodes,       # [B,M,in_dim]   (local nodes)
                hop_idx,           # [B,M]
                route_flag_nodes,  # [B,M]
                masks_per_rel,     # list len=R of [B,M,M] bool
                route_seq_idx,     # [B,L] indices into local nodes (sequence view)
                seq_link_len,      # [B,L] meters
                vff_seq,           # [B,L] m/s (free-flow speed)
                seq_mask,          # [B,L] bool (valid positions)
                # [2026-04-28 A1] r_max default 2.8 -> 2.0 (bounded predicted speed = r_max * vff)
                # [2026-05-01 Stage 2/3] r_max default 2.0 -> 3.0 (matches the
                # statistical bound from SF q99 of sp_actual on real driving,
                # and the r_max used in prepare_corrected_dataset.py / Stage 1).
                r_max=3.0,
                return_parts=True):

        B, M, _ = x_seq_nodes.size()
        L = route_seq_idx.size(1)

        x_seq_nodes = torch.nan_to_num(x_seq_nodes)

        # --------------------------------------------------------------
        # 1) Node encodings: input projection + dyn embedding + route/hop enc
        # --------------------------------------------------------------
        h = self.input_proj(x_seq_nodes)                  # [B,M,D]
        h_dyn = self._encode_dyn_slots(x_seq_nodes)       # [B,M,D]
        h = h + h_dyn                                     # fuse dynamic temporal info

        h = self.inp_dropout(h)
        h = self.route_enc(h, hop_idx, route_flag_nodes.long())  # [B,M,D]

        # Relational graph "transformer" layers
        for layer in self.layers:
            h = layer(h, masks_per_rel)  # [B,M,D]

        # --------------------------------------------------------------
        # 2) Sequence view along the route (gather by route_seq_idx)
        # --------------------------------------------------------------
        D = h.size(-1)
        gather_idx = route_seq_idx.unsqueeze(-1).expand(-1, -1, D)  # [B,L,D]
        h_seq = torch.gather(h, 1, gather_idx)                      # [B,L,D]

        # enc_t: [B,D,L] for Conv1d-based heads
        enc_t = h_seq.transpose(1, 2)  # [B,D,L]

        # --------------------------------------------------------------
        # 3) Traffic auxiliaries: speed ratio (sp_hat) & volume (logcnt_1h)
        # --------------------------------------------------------------
        speed_1h_raw = self.speed_head(enc_t).squeeze(1)  # [B,L]
        logcnt_1h = self.count_head(enc_t).squeeze(1)     # [B,L]
        sp_hat = torch.sigmoid(speed_1h_raw) * r_max      # [B,L], ratio vs v_ff

        # --------------------------------------------------------------
        # 4) [2026-05-01 Stage 3] sp_hat-anchored ETA with TIGHT delta + g
        # --------------------------------------------------------------
        seq_link_len = torch.nan_to_num(seq_link_len).clamp_min(1e-3)  # meters
        vff_seq      = torch.nan_to_num(vff_seq).clamp_min(1.0)        # m/s
        seq_mask_f   = seq_mask.float()

        # [2026-04-28 A1 OLD — speed-only path (no delta, no g):]
        # v_pred = (sp_hat.clamp_min(1e-3) * vff_seq).clamp_min(0.5)
        # link_eta_seq = seq_link_len / v_pred
        # link_eta_seq = link_eta_seq * seq_mask_f
        # delta = torch.zeros_like(sp_hat)
        # g     = torch.ones(B, device=link_eta_seq.device, dtype=link_eta_seq.dtype)
        # eta_phys = link_eta_seq
        # trip_eta = link_eta_seq.sum(dim=1)

        # eta_base = length / (bounded speed)
        v_pred   = (sp_hat.clamp_min(1e-3) * vff_seq).clamp_min(0.5)   # [B,L] m/s
        eta_base = seq_link_len / v_pred                                # [B,L] seconds

        # Per-link residual: tight ±0.2 in log-space (multiplier exp(delta) ∈ [0.82, 1.22])
        res_raw = self.residual_head(enc_t).squeeze(1)                  # [B,L]
        res_raw = torch.nan_to_num(res_raw)
        delta   = self.delta_range * torch.tanh(res_raw)                # [B,L]
        # [2026-04-28 A1 OLD] delta = 2.0 * torch.tanh(res_raw)         # delta_range was 2.0 (too wide)

        eta_phys = eta_base * torch.exp(delta)                          # [B,L], positive

        # Per-trip scale g: tight ±0.1 (multiplier ∈ [0.9, 1.1])
        route_repr = (h_seq * seq_mask_f.unsqueeze(-1)).sum(dim=1) / \
                     seq_mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)  # [B,D]
        g_raw = self.route_calib(route_repr).squeeze(-1)                 # [B]
        g     = 1.0 + self.g_range * torch.tanh(g_raw)                   # [B]
        # [2026-04-28 A1 OLD] g_range was 0.5 -> g ∈ [0.5, 1.5] (too wide)

        link_eta_seq = eta_phys * g.unsqueeze(-1)                        # [B,L]
        link_eta_seq = link_eta_seq * seq_mask_f                         # mask invalid

        trip_eta = link_eta_seq.sum(dim=1)                               # [B]

        if return_parts:
            return (
                link_eta_seq,  # [B,L]  ETA in seconds (= length / v_pred)
                trip_eta,      # [B]
                sp_hat,        # [B,L]  predicted speed ratio
                logcnt_1h,     # [B,L]
                g,             # [B]    always 1.0 after A1 (ABI placeholder)
                delta,         # [B,L]  always 0.0 after A1 (ABI placeholder)
                eta_phys,      # [B,L]  = link_eta_seq (no residual or g multiplier)
            )

        return link_eta_seq, trip_eta, sp_hat, logcnt_1h, g, delta, eta_phys

DWA_T = 2.0


def lam_g_at(epoch):
    return 0.02 + (0.06 * min(epoch / 10.0, 1.0))

def vff_noise_range(epoch):
    return 0.10 if epoch < 8 else 0.04

def _max_L(ds):
    return max((g["route_seq_idx"].numel() for g in ds.groups), default=0)

clip_norm = 1.0

num_epochs = 100

WARMUP_EPOCHS = 1


def combined_metric(va):
    return selection_metric(va.get('route_mape', float('nan')),
                            va.get('link_mape', float('nan')),
                            w_route=0.7, w_link=0.3)

global_step = 0

def _init_metric_accumulator():
    return {
        "link_loss_sum": 0.0,
        "route_loss_sum": 0.0,
        "link_mae_sum": 0.0,
        "link_sse_sum": 0.0,
        "link_mape_sum": 0.0,
        "route_mae_sum": 0.0,
        "route_sse_sum": 0.0,
        "route_mape_sum": 0.0,
        "total_links": 0,
        "total_links_mape": 0,
        "total_routes": 0,
        "total_routes_mape": 0,
    }

def _accumulate_batch_metrics(agg: dict, stats: dict):
    """
    stats is the output of compute_global_losses_and_metrics for one batch.
    We accumulate weighted sums so final dataset-level metrics are exact.
    """
    b_n_links       = int(stats.get("n_links", 0))
    b_n_links_mape  = int(stats.get("n_links_mape", 0))
    b_n_routes      = int(stats.get("n_routes", 0))
    b_n_routes_mape = int(stats.get("n_routes_mape", 0))

    def as_float(x):
        if isinstance(x, torch.Tensor):
            return float(x.item())
        return float(x)

    # link + route "losses" are averages over routes -> weight by #routes
    if b_n_routes > 0:
        agg["link_loss_sum"]  += as_float(stats["link_loss"])  * b_n_routes
        agg["route_loss_sum"] += as_float(stats["route_loss"]) * b_n_routes

    # link metrics -> weight by number of links
    if b_n_links > 0:
        agg["link_mae_sum"] += as_float(stats["link_mae"]) * b_n_links
        agg["link_sse_sum"] += (as_float(stats["link_rmse"]) ** 2) * b_n_links

    if b_n_links_mape > 0:
        m_link = as_float(stats["link_mape"])
        if not math.isnan(m_link):
            agg["link_mape_sum"] += m_link * b_n_links_mape

    # route metrics -> weight by number of routes
    if b_n_routes > 0:
        agg["route_mae_sum"] += as_float(stats["route_mae"]) * b_n_routes
        agg["route_sse_sum"] += (as_float(stats["route_rmse"]) ** 2) * b_n_routes

    if b_n_routes_mape > 0:
        m_route = as_float(stats["route_mape"])
        if not math.isnan(m_route):
            agg["route_mape_sum"] += m_route * b_n_routes_mape

    agg["total_links"]       += b_n_links
    agg["total_links_mape"]  += b_n_links_mape
    agg["total_routes"]      += b_n_routes
    agg["total_routes_mape"] += b_n_routes_mape

def _finalize_metric_accumulator(agg: dict) -> dict:
    n_links       = agg["total_links"]
    n_links_mape  = agg["total_links_mape"]
    n_routes      = agg["total_routes"]
    n_routes_mape = agg["total_routes_mape"]

    if n_routes > 0:
        link_loss  = agg["link_loss_sum"]  / n_routes
        route_loss = agg["route_loss_sum"] / n_routes
    else:
        link_loss = route_loss = float("nan")

    if n_links > 0:
        link_mae  = agg["link_mae_sum"] / n_links
        link_rmse = math.sqrt(agg["link_sse_sum"] / n_links)
    else:
        link_mae = link_rmse = float("nan")

    if n_links_mape > 0:
        link_mape = agg["link_mape_sum"] / n_links_mape
    else:
        link_mape = float("nan")

    if n_routes > 0:
        route_mae  = agg["route_mae_sum"] / n_routes
        route_rmse = math.sqrt(agg["route_sse_sum"] / n_routes)
    else:
        route_mae = route_rmse = float("nan")

    if n_routes_mape > 0:
        route_mape = agg["route_mape_sum"] / n_routes_mape
    else:
        route_mape = float("nan")

    return dict(
        link_loss=link_loss,
        link_mae=link_mae,
        link_rmse=link_rmse,
        link_mape=link_mape,
        route_loss=route_loss,
        route_mae=route_mae,
        route_rmse=route_rmse,
        route_mape=route_mape,
        n_links=int(n_links),
        n_routes=int(n_routes),
    )


EVAL_TAG     = "step6_logHuber_lenW_loosenedHeads"

def _print_eval_diagnostics(trip_df: pd.DataFrame,
                            link_df: pd.DataFrame,
                            tag: str) -> None:
    """Self-contained worst-trip / worst-link / per-bucket analysis."""
    print(f"\n{'='*78}\n  EVAL DIAGNOSTICS [{tag}]\n{'='*78}")

    # ---- per-trip ----
    df = trip_df.copy()
    df["abs_err"] = (df["y_pred"] - df["y_true"]).abs()
    df["sq_err"]  = (df["y_pred"] - df["y_true"]) ** 2
    df["rel_err"] = df["abs_err"] / df["y_true"].clip(lower=1.0)
    df["signed_err"] = df["y_pred"] - df["y_true"]

    print("\n--- Top 20 worst trips by |error| (MAE drivers) ---")
    print(df.nlargest(20, "abs_err")[
        ["TRIP_ID","y_true","y_pred","abs_err","rel_err","route_len_m","target_source"]
    ].to_string(index=False))

    print("\n--- Top 20 worst trips by squared error (RMSE drivers) ---")
    print(df.nlargest(20, "sq_err")[
        ["TRIP_ID","y_true","y_pred","abs_err","sq_err","rel_err","route_len_m"]
    ].to_string(index=False))

    n = len(df)
    total_abs = float(df["abs_err"].sum())
    total_sq  = float(df["sq_err"].sum())
    if total_abs > 0 and total_sq > 0:
        print("\n--- Concentration of trip-level error ---")
        for k_pct in [1, 5, 10, 25]:
            k = max(1, int(n * k_pct / 100))
            top_abs = df.nlargest(k, "abs_err")["abs_err"].sum() / total_abs
            top_sq  = df.nlargest(k, "sq_err")["sq_err"].sum()  / total_sq
            print(f"  Top {k_pct:>2}% ({k:>4}/{n} trips): "
                  f"{top_abs*100:5.1f}% of total |err|, "
                  f"{top_sq*100:5.1f}% of total sq_err")

    # ---- per-bucket ----
    bins   = [0, 60, 180, 600, 1800, 3600, np.inf]
    labels = ["0–1m","1–3m","3–10m","10–30m","30–60m","60m+"]
    df["bucket"] = pd.cut(df["y_true"], bins=bins, labels=labels, include_lowest=True)
    print("\n--- Per-bucket trip error breakdown ---")
    print(df.groupby("bucket", observed=True).agg(
        n        = ("abs_err", "size"),
        mae      = ("abs_err", "mean"),
        rmse     = ("sq_err",  lambda s: float(np.sqrt(s.mean()))),
        mape     = ("rel_err", "mean"),
        p95_err  = ("abs_err", lambda s: float(s.quantile(0.95))),
        max_err  = ("abs_err", "max"),
        bias     = ("signed_err", "mean"),
    ).round(3).to_string())

    # ---- per-link ----
    if len(link_df) > 0:
        ldf = link_df.copy()
        ldf["abs_err"] = (ldf["pred_tt_link"] - ldf["true_tt_link"]).abs()
        ldf["sq_err"]  = (ldf["pred_tt_link"] - ldf["true_tt_link"]) ** 2

        print("\n--- Top 20 worst links by |error| (link-MAE drivers) ---")
        print(ldf.nlargest(20, "abs_err")[
            ["TRIP_ID","true_tt_link","pred_tt_link","abs_err"]
        ].to_string(index=False))

        # link-bucket by true_tt
        lbins   = [0, 5, 15, 60, 300, np.inf]
        llabels = ["0-5s","5-15s","15-60s","1-5m","5m+"]
        ldf["bucket"] = pd.cut(ldf["true_tt_link"], bins=lbins, labels=llabels, include_lowest=True)
        print("\n--- Per-bucket link error breakdown ---")
        print(ldf.groupby("bucket", observed=True).agg(
            n        = ("abs_err", "size"),
            mae      = ("abs_err", "mean"),
            rmse     = ("sq_err",  lambda s: float(np.sqrt(s.mean()))),
            p95_err  = ("abs_err", lambda s: float(s.quantile(0.95))),
            max_err  = ("abs_err", "max"),
        ).round(3).to_string())

    print(f"\n{'='*78}\n")

def _jsonable(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.ndarray,)):
        return x.tolist()
    if isinstance(x, (torch.Tensor,)):
        return x.detach().cpu().tolist()
    if isinstance(x, (set,)):
        return sorted(list(x))
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x

def _df_to_parquet_with_optional_geometry(df: pd.DataFrame, path: str, geometry_col: str = "geometry"):
    """
    Saves df to parquet.
    If geometry column exists, stores it as geometry_wkb bytes and drops geometry.
    This makes it deterministic and reloadable without geopandas.
    """
    out = df.copy()
    if geometry_col in out.columns:
        def to_wkb(g):
            if g is None or (isinstance(g, float) and np.isnan(g)):
                return None
            if isinstance(g, BaseGeometry):
                return g.wkb
            # If already bytes (unlikely), keep as is:
            if isinstance(g, (bytes, bytearray)):
                return bytes(g)
            return None

        out["geometry_wkb"] = out[geometry_col].apply(to_wkb)
        out = out.drop(columns=[geometry_col])

    path = str(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)

def _parquet_to_df_with_optional_geometry(path: str, geometry_col_out: str = "geometry") -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "geometry_wkb" in df.columns:
        def from_wkb(b):
            if b is None or (isinstance(b, float) and np.isnan(b)):
                return None
            return shapely_wkb.loads(b)
        df[geometry_col_out] = df["geometry_wkb"].apply(from_wkb)
        df = df.drop(columns=["geometry_wkb"])
    return df

def _stable_df_fingerprint(df: pd.DataFrame, sort_cols: Optional[List[str]] = None) -> str:
    """
    Stable fingerprint based on pandas hashing (content + order after sorting).
    This is strict: any value/dtype/order difference changes the hash.
    """
    x = df.copy()
    if sort_cols:
        existing = [c for c in sort_cols if c in x.columns]
        if existing:
            x = x.sort_values(existing, kind="mergesort")
    x = x.reset_index(drop=True)

    # Hash each row; include index to preserve ordering.
    h = pd.util.hash_pandas_object(x, index=True).values
    return hashlib.sha256(h.tobytes()).hexdigest()

REL_NAMES_DEFAULT = ["self", "downstream", "upstream", "left_turn", "opposite", "right_turn", "high_order", "u_turn"]

def save_linketa_bundle(
    out_dir: str,
    *,
    edges_vff: pd.DataFrame,
    rel_edges: List[torch.Tensor],
    vff_vec: torch.Tensor,
    cfg: Dict[str, Any],
    inst_feat_cols: List[str],
    static_cols: List[str],
    time_cols: List[str],
    dyn_cols: List[str],
    scaler_mean: torch.Tensor,
    scaler_std: torch.Tensor,
    dyn_idx: torch.Tensor,
    traffic_profile_df: pd.DataFrame,
    traffic_profile_path_name: str = "traffic_profile.parquet",
    edges_vff_name: str = "edges_vff.parquet",
    rel_name: str = "relations.pt",
    vff_name: str = "vff_vec.pt",
    schema_name: str = "feature_schema.json",
    cfg_name: str = "cfg.json",
    scaler_mean_name: str = "scaler_mean.pt",
    scaler_std_name: str = "scaler_std.pt",
    dyn_idx_name: str = "dyn_idx.pt",
    rel_names: Optional[List[str]] = None,
    save_dyn_store_idx: bool = False,
    dyn_store_obj: Optional[Any] = None,  # pass DYN_STORE if save_dyn_store_idx=True
):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1) edges_vff.parquet
    _df_to_parquet_with_optional_geometry(edges_vff, out / edges_vff_name, geometry_col="geometry")

    # 2) relations.pt (store named tensors + order)
    names = rel_names if rel_names is not None else REL_NAMES_DEFAULT
    if len(names) != len(rel_edges):
        raise ValueError(f"rel_names length {len(names)} != rel_edges length {len(rel_edges)}")

    rel_pack = {
        "names": names,
        "rel_edges": {n: t.detach().cpu().contiguous() for n, t in zip(names, rel_edges)},
    }
    torch.save(rel_pack, out / rel_name)

    # 3) vff_vec.pt
    torch.save(vff_vec.detach().cpu().contiguous(), out / vff_name)

    # 4) feature_schema.json
    schema = {
        "INST_FEAT_COLS": inst_feat_cols,
        "STATIC_FEAT_COLS": static_cols,
        "TIME_FEAT_COLS": time_cols,
        "DYN_FEAT_COLS": dyn_cols,
        "dyn_prefixes": list(cfg.get("dyn_prefixes", [])),
        "slots_per_hour": int(cfg.get("slots_per_hour", 12)),
    }
    with open(out / schema_name, "w", encoding="utf-8") as f:
        json.dump(_jsonable(schema), f, indent=2, sort_keys=True)

    # 5) cfg.json
    with open(out / cfg_name, "w", encoding="utf-8") as f:
        json.dump(_jsonable(cfg), f, indent=2, sort_keys=True)

    # 6) scalers
    torch.save(scaler_mean.detach().cpu().contiguous(), out / scaler_mean_name)
    torch.save(scaler_std.detach().cpu().contiguous(), out / scaler_std_name)
    torch.save(dyn_idx.detach().cpu().contiguous(), out / dyn_idx_name)

    # 7) traffic_profile.parquet (canonical dynamic base)
    # Save exactly as given; to maximize reproducibility, sort by (edge, time) if columns exist.
    tp = traffic_profile_df.copy()
    sort_cols = [cfg.get("way_id_col_df", "way_id"), cfg.get("time_col_df", "departure_time")]
    existing = [c for c in sort_cols if c in tp.columns]
    if existing:
        tp = tp.sort_values(existing, kind="mergesort").reset_index(drop=True)
    tp.to_parquet(out / traffic_profile_path_name, index=False)

    # 8) optional dyn_store_idx.pt (fast startup, larger disk)
    if save_dyn_store_idx:
        if dyn_store_obj is None or not hasattr(dyn_store_obj, "idx"):
            raise ValueError("save_dyn_store_idx=True requires dyn_store_obj=DYN_STORE")

        # Convert to pure CPU numpy/tensors for portability
        packed = {}
        for eidx, (t64, arr) in dyn_store_obj.idx.items():
            packed[int(eidx)] = {
                "t64": np.asarray(t64, dtype=np.int64),
                "arr": np.asarray(arr, dtype=np.float32),
            }
        torch.save(packed, out / "dyn_store_idx.pt")

    # 9) write fingerprints so you can verify quickly later
    meta = {
        "edges_vff_fingerprint": _stable_df_fingerprint(
            _parquet_to_df_with_optional_geometry(out / edges_vff_name), sort_cols=["edge_idx"]
        ),
        "traffic_profile_fingerprint": _stable_df_fingerprint(
            pd.read_parquet(out / traffic_profile_path_name),
            sort_cols=[cfg.get("way_id_col_df", "way_id"), cfg.get("time_col_df", "departure_time")]
        ),
        "rel_names": names,
        "num_relations": len(names),
        "vff_vec_shape": list(vff_vec.detach().cpu().shape),
        "scaler_mean_shape": list(scaler_mean.detach().cpu().shape),
        "scaler_std_shape": list(scaler_std.detach().cpu().shape),
        "dyn_idx_shape": list(dyn_idx.detach().cpu().shape),
    }
    with open(out / "bundle_meta.json", "w", encoding="utf-8") as f:
        json.dump(_jsonable(meta), f, indent=2, sort_keys=True)

    print(f"Saved LinkETA bundle to: {out.resolve()}")
    return str(out.resolve())

def load_linketa_bundle(in_dir: str) -> Dict[str, Any]:
    p = Path(in_dir)

    edges_vff = _parquet_to_df_with_optional_geometry(p / "edges_vff.parquet")

    rel_pack = torch.load(p / "relations.pt", map_location="cpu")
    rel_names = rel_pack["names"]
    rel_edges_dict = rel_pack["rel_edges"]

    vff_vec = torch.load(p / "vff_vec.pt", map_location="cpu")

    with open(p / "feature_schema.json", "r", encoding="utf-8") as f:
        schema = json.load(f)

    with open(p / "cfg.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    scaler_mean = torch.load(p / "scaler_mean.pt", map_location="cpu")
    scaler_std = torch.load(p / "scaler_std.pt", map_location="cpu")
    dyn_idx = torch.load(p / "dyn_idx.pt", map_location="cpu")

    traffic_profile = pd.read_parquet(p / "traffic_profile.parquet")

    bundle = dict(
        edges_vff=edges_vff,
        rel_names=rel_names,
        rel_edges_dict=rel_edges_dict,
        vff_vec=vff_vec,
        schema=schema,
        cfg=cfg,
        scaler_mean=scaler_mean,
        scaler_std=scaler_std,
        dyn_idx=dyn_idx,
        traffic_profile=traffic_profile,
        meta=(json.load(open(p / "bundle_meta.json", "r", encoding="utf-8")) if (p / "bundle_meta.json").exists() else None),
        dyn_store_idx=(torch.load(p / "dyn_store_idx.pt", map_location="cpu") if (p / "dyn_store_idx.pt").exists() else None),
        in_dir=str(p.resolve()),
    )
    return bundle

def verify_linketa_bundle_exact(
    bundle: Dict[str, Any],
    *,
    edges_vff_ref: pd.DataFrame,
    rel_edges_ref: List[torch.Tensor],
    vff_vec_ref: torch.Tensor,
    cfg_ref: Dict[str, Any],
    inst_feat_cols_ref: List[str],
    static_cols_ref: List[str],
    time_cols_ref: List[str],
    dyn_cols_ref: List[str],
    scaler_mean_ref: torch.Tensor,
    scaler_std_ref: torch.Tensor,
    dyn_idx_ref: torch.Tensor,
    traffic_profile_ref: pd.DataFrame,
    strict_df: bool = True,
) -> bool:
    ok = True

    # ---- 1) relations tensors
    names = bundle["rel_names"]
    loaded_rel = bundle["rel_edges_dict"]
    if len(names) != len(rel_edges_ref):
        raise AssertionError("Relation count mismatch")

    for n, t_ref in zip(names, rel_edges_ref):
        t_load = loaded_rel[n]
        if not torch.equal(t_load, t_ref.detach().cpu()):
            raise AssertionError(f"Relation tensor mismatch: {n}")

    # ---- 2) vff_vec
    if not torch.equal(bundle["vff_vec"], vff_vec_ref.detach().cpu()):
        raise AssertionError("vff_vec mismatch")

    # ---- 3) scalers
    if not torch.equal(bundle["scaler_mean"], scaler_mean_ref.detach().cpu()):
        raise AssertionError("SCALER_MEAN mismatch")
    if not torch.equal(bundle["scaler_std"], scaler_std_ref.detach().cpu()):
        raise AssertionError("SCALER_STD mismatch")
    if not torch.equal(bundle["dyn_idx"], dyn_idx_ref.detach().cpu()):
        raise AssertionError("DYN_IDX mismatch")

    # ---- 4) feature schema
    sch = bundle["schema"]
    if sch["INST_FEAT_COLS"] != inst_feat_cols_ref:
        raise AssertionError("INST_FEAT_COLS mismatch")
    if sch["STATIC_FEAT_COLS"] != static_cols_ref:
        raise AssertionError("STATIC_FEAT_COLS mismatch")
    if sch["TIME_FEAT_COLS"] != time_cols_ref:
        raise AssertionError("TIME_FEAT_COLS mismatch")
    if sch["DYN_FEAT_COLS"] != dyn_cols_ref:
        raise AssertionError("DYN_FEAT_COLS mismatch")

    # ---- 5) cfg (verify only stable keys - mix_mode/mix_phys_weight removed in Fixed version)
    cfg_loaded = bundle["cfg"]
    stable_keys = [
        "way_id_col_df", "time_col_df", "dyn_prefixes", "slots_per_hour",
        "hidden_dim", "heads", "num_layers", "max_route_len",
        "high_order_topk", "speed_metric_index", "K_first", "K_conflict", "K_high",
        "max_nodes_local", "length_col_edges", "speed_is_kmh"
    ]
    for k in stable_keys:
        if k in cfg_ref and k in cfg_loaded:
            if _jsonable(cfg_ref[k]) != _jsonable(cfg_loaded[k]):
                raise AssertionError(f"CFG mismatch at key={k}")

    # ---- 6) edges_vff dataframe
    edges_loaded = bundle["edges_vff"]
    # Use same serialization transform as in save() for a fair compare
    edges_ref_norm = edges_vff_ref.copy()
    if "geometry" in edges_ref_norm.columns:
        edges_ref_norm["geometry_wkb"] = edges_ref_norm["geometry"].apply(lambda g: None if g is None else g.wkb)
        edges_ref_norm = edges_ref_norm.drop(columns=["geometry"])

    edges_loaded_norm = edges_loaded.copy()
    if "geometry" in edges_loaded_norm.columns:
        edges_loaded_norm["geometry_wkb"] = edges_loaded_norm["geometry"].apply(lambda g: None if g is None else g.wkb)
        edges_loaded_norm = edges_loaded_norm.drop(columns=["geometry"])

    # Sort by edge_idx if present for stable equality
    sort_cols = ["edge_idx"] if "edge_idx" in edges_ref_norm.columns and "edge_idx" in edges_loaded_norm.columns else None
    if sort_cols:
        edges_ref_norm = edges_ref_norm.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
        edges_loaded_norm = edges_loaded_norm.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    if strict_df:
        pdt.assert_frame_equal(edges_loaded_norm, edges_ref_norm, check_dtype=True, check_exact=True)
    else:
        # relaxed: allow float tolerance
        pdt.assert_frame_equal(edges_loaded_norm, edges_ref_norm, check_dtype=False, check_exact=False, atol=0, rtol=0)

    # ---- 7) traffic_profile dataframe
    tp_loaded = bundle["traffic_profile"].copy()
    tp_ref = traffic_profile_ref.copy()

    # Sort deterministically (same as save)
    way_col = cfg_ref.get("way_id_col_df", "way_id")
    time_col = cfg_ref.get("time_col_df", "departure_time")
    sort_cols = [c for c in [way_col, time_col] if c in tp_ref.columns and c in tp_loaded.columns]
    if sort_cols:
        tp_ref = tp_ref.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
        tp_loaded = tp_loaded.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    if strict_df:
        pdt.assert_frame_equal(tp_loaded, tp_ref, check_dtype=True, check_exact=True)
    else:
        pdt.assert_frame_equal(tp_loaded, tp_ref, check_dtype=False, check_exact=False)

    print("Verification PASSED: loaded bundle matches in-memory pipeline artifacts exactly.")
    return True

def build_traffic_profile_df(df_all_local: pd.DataFrame, cfg: Dict[str, Any], dyn_12xP_cols: List[str]) -> pd.DataFrame:
    way_col = cfg.get("way_id_col_df", "way_id")
    time_col = cfg.get("time_col_df", "departure_time")

    keep = []
    for c in [way_col, time_col, "TRIP_ID"]:
        if c in df_all_local.columns:
            keep.append(c)

    keep += [c for c in dyn_12xP_cols if c in df_all_local.columns]

    tp = df_all_local[keep].copy()

    # Ensure types are stable
    if way_col in tp.columns:
        tp[way_col] = tp[way_col].astype(str)
    if time_col in tp.columns:
        tp[time_col] = pd.to_datetime(tp[time_col], errors="coerce")
        tp = tp.dropna(subset=[time_col])

    # Fill missing dynamic cols deterministically
    for c in dyn_12xP_cols:
        if c not in tp.columns:
            tp[c] = 0.0

    # Sort for deterministic reload -> deterministic EdgeDynStore.idx
    sort_cols = [c for c in [way_col, time_col] if c in tp.columns]
    if sort_cols:
        tp = tp.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    return tp


# =============================================================================
# [MetaETA parity] Precomputed-data fast path (ported from LinkETA_Fixed).
#
# These module-level helpers let LinkETA.py LOAD a precomputed city bundle +
# checkpoint and reuse it across reruns (fast), instead of rebuilding the whole
# pipeline from raw data every time. They are STATELESS: init_linketa RETURNS a
# `base` dict (per-model state) rather than stashing it in module globals; only
# the shared feature-schema constants (CFG, DYN_IDX, H_SLICES, BIN_SECONDS) are
# published to globals because the model/dataset classes read them by global
# lookup. linketa_get_core_components(df, edges, base=base) consumes `base`.
# =============================================================================

def select_dynamic_12x5_columns_full(df_all: pd.DataFrame) -> List[str]:
    """Ensure all 12xP dynamic columns exist, suffix 0 = most recent, 11 = oldest."""
    order = list(range(0, 12))
    cols = [f"{p}_{s}" for s in order for p in CFG["dyn_prefixes"]]
    for c in cols:
        if c not in df_all.columns:
            df_all[c] = 0.0
    return cols


def select_dynamic_12x5_columns(df_all: pd.DataFrame) -> List[str]:
    """k=0 most recent; last hour => slots 0..11 (0 is latest)."""
    order = list(range(0, 12))
    cols = [f"{p}_{s}" for s in order for p in CFG["dyn_prefixes"]]
    return cols


def dyn_indices(INST_FEAT_COLS):
    prefixes = CFG["dyn_prefixes"]
    DYN_COLS = [c for c in INST_FEAT_COLS if any(str(c).startswith(f"{p}_") for p in prefixes)]
    return torch.tensor([INST_FEAT_COLS.index(c) for c in DYN_COLS], dtype=torch.long)


def merge_traffic_profile(df_all: pd.DataFrame, traffic_profile: pd.DataFrame = None) -> pd.DataFrame:
    """
    Merge precomputed traffic features (12xP dynamic stats) onto df_all by
    (way_id, day_of_week, hour_of_day) derived from departure_time. Replaces
    on-the-fly dynamic feature computation. `traffic_profile` must be supplied
    (e.g. base["traffic_profile"]); if None this is a no-op (stateless: no
    module-global fallback).
    """
    if traffic_profile is None:
        print("[merge_traffic_profile] No traffic profile supplied, skipping merge")
        return df_all
    df = df_all.copy()
    if 'departure_time' not in df.columns:
        print("[merge_traffic_profile] WARNING: departure_time not found")
        return df
    df['departure_time'] = pd.to_datetime(df['departure_time'], errors='coerce')
    df['_merge_dow'] = df['departure_time'].dt.dayofweek.astype(int)
    df['_merge_hour'] = df['departure_time'].dt.hour.astype(int)
    df['_merge_way_id'] = pd.to_numeric(df['way_id'], errors='coerce').astype('Int64')

    tp = traffic_profile.copy()
    tp['_merge_way_id'] = pd.to_numeric(tp['way_id'], errors='coerce').astype('Int64')
    if 'day_of_week' not in tp.columns and 'departure_time' in tp.columns:
        tp['departure_time'] = pd.to_datetime(tp['departure_time'], errors='coerce')
        tp['_merge_dow'] = tp['departure_time'].dt.dayofweek.astype(int)
        tp['_merge_hour'] = tp['departure_time'].dt.hour.astype(int)
    else:
        tp['_merge_dow'] = tp['day_of_week'].astype(int)
        tp['_merge_hour'] = tp['hour_of_day'].astype(int)

    dyn_cols = [c for c in tp.columns if any(c.startswith(p) for p in CFG["dyn_prefixes"])]
    merge_cols = ['_merge_way_id', '_merge_dow', '_merge_hour'] + dyn_cols
    tp_merge = tp[merge_cols].drop_duplicates(subset=['_merge_way_id', '_merge_dow', '_merge_hour'])

    n_before = len(df)
    df = df.merge(tp_merge, on=['_merge_way_id', '_merge_dow', '_merge_hour'], how='left', suffixes=('', '_tp'))
    for col in dyn_cols:
        if f"{col}_tp" in df.columns:
            df[col] = df[f"{col}_tp"].fillna(df.get(col, 0.0))
            df.drop(columns=[f"{col}_tp"], inplace=True, errors='ignore')
    df.drop(columns=['_merge_dow', '_merge_hour', '_merge_way_id'], inplace=True, errors='ignore')
    for col in dyn_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)
    matched = df[dyn_cols[0]].notna().sum() if dyn_cols else 0
    print(f"[merge_traffic_profile] Merged traffic features for {matched}/{n_before} rows")
    return df


def build_instance_features(df_all, edges_static=None, traffic_profile=None, inst_feat_cols=None):
    """
    Inference-aware instance feature table. If edges_static is given, static
    features are merged by way_id; if traffic_profile is given, dynamic features
    are merged by (way_id, dow, hour). If inst_feat_cols is given, the result is
    reindexed to EXACTLY that schema (order + count) so it lines up with the
    precomputed scaler/checkpoint. Stateless: no module-global fallbacks.
    """
    all_candidate_static = [
        'length', 'lanes', 'width_m', 'speed_limit_ms',
        'hw_living_street', 'hw_motorway', 'hw_motorway_link', 'hw_pedestrian',
        'hw_primary', 'hw_primary_link', 'hw_residential', 'hw_secondary',
        'hw_secondary_link', 'hw_service', 'hw_tertiary', 'hw_tertiary_link',
        'has_traffic_signal', 'link_len_m',
    ]
    existing_static = [c for c in all_candidate_static if c in df_all.columns]
    if len(existing_static) < 5 and edges_static is not None:
        static_cols_in_edges = [c for c in all_candidate_static if c in edges_static.columns]
        if static_cols_in_edges and 'id' in edges_static.columns:
            edges_static_df = edges_static[['id'] + static_cols_in_edges].copy()
            edges_static_df = edges_static_df.rename(columns={'id': 'way_id'})
            edges_static_df['way_id'] = pd.to_numeric(edges_static_df['way_id'], errors='coerce').astype('Int64')
            df_all = df_all.copy()
            df_all['way_id'] = pd.to_numeric(df_all['way_id'], errors='coerce').astype('Int64')
            df_all = df_all.merge(edges_static_df, on='way_id', how='left', suffixes=('', '_static'))
            for col in static_cols_in_edges:
                if col not in df_all.columns and f"{col}_static" in df_all.columns:
                    df_all[col] = df_all[f"{col}_static"]
            print(f"[build_instance_features] Merged {len(static_cols_in_edges)} static features from edges_static")

    dyn_cols_check = [f"median_speed_{i}" for i in range(12)]
    existing_dyn = [c for c in dyn_cols_check if c in df_all.columns and df_all[c].abs().sum() > 0]
    if len(existing_dyn) < 6:
        df_all = merge_traffic_profile(df_all, traffic_profile)

    if 'edge_idx' not in df_all.columns and 'way_id' in df_all.columns:
        df_all['edge_idx'] = df_all['way_id']

    static_feat_cols = cols_if_exist(df_all, all_candidate_static)
    edge_static = (df_all.groupby('edge_idx')[static_feat_cols].first().fillna(0.0)
                   if static_feat_cols else pd.DataFrame(index=df_all['edge_idx'].unique()))

    CAL_TIME_COLS_ALL = ["day_of_week", "month", "day_of_month", "day_of_year",
                         "tod_sin", "tod_cos", "dow_sin", "dow_cos"]
    time_feat_cols = cols_if_exist(df_all, CAL_TIME_COLS_ALL)
    dyn_feat_cols = select_dynamic_12x5_columns_full(df_all)

    BASE_COLS = cols_if_exist(df_all, ["TRIP_ID", "edge_idx"] + time_feat_cols + dyn_feat_cols)
    sort_cols = ['TRIP_ID', 'enter_ts'] if 'enter_ts' in df_all.columns else ['TRIP_ID', 'departure_time']
    inst_raw = (
        df_all.sort_values(sort_cols)
              .drop_duplicates(subset=['TRIP_ID', 'edge_idx'], keep='first')
              [BASE_COLS]
              .set_index(['TRIP_ID', 'edge_idx'])
              .fillna(0.0)
    )
    inst = inst_raw.join(edge_static, on='edge_idx', how='left').fillna(0.0)
    INST_FEAT_COLS_local = list(inst.columns)

    if inst_feat_cols is not None:
        missing = [c for c in inst_feat_cols if c not in INST_FEAT_COLS_local]
        if missing:
            print(f"[build_instance_features] padding {len(missing)} missing schema cols with 0.0")
            for col in missing:
                inst[col] = 0.0
        inst = inst.reindex(columns=inst_feat_cols, fill_value=0.0)
        INST_FEAT_COLS_local = list(inst.columns)

    return inst, INST_FEAT_COLS_local, static_feat_cols, time_feat_cols, dyn_feat_cols


def init_linketa(precomputed_dir: str, checkpoint_path: str = None) -> Dict[str, Any]:
    """
    Load a precomputed city bundle + base-LinkETA checkpoint and build a ready
    model. Returns a `base` dict (stateless) consumed by
    linketa_get_core_components(df, edges, base=base). Publishes only the shared
    feature-schema constants (DYN_IDX) to globals (the model reads DYN_IDX by
    global lookup); CFG / H_SLICES / BIN_SECONDS are already module-level.

    The bundle layout matches what is on disk under meta_learning_data/<city>/:
      feature_schema.json, dyn_idx.pt, edges_vff.parquet, relations.pt,
      scaler_mean.pt, scaler_std.pt, vff_vec.pt, link_traffic_profile.parquet,
      and ckpts_LinkETA_<city>/best.ckpt (passed as checkpoint_path).
    """
    global PRECOMPUTED_DIR, CHECKPOINT_PATH, _linketa_initialized
    PRECOMPUTED_DIR = precomputed_dir
    CHECKPOINT_PATH = checkpoint_path

    with open(os.path.join(precomputed_dir, "feature_schema.json"), "r") as f:
        schema = json.load(f)
    STATIC_FEAT_COLS = schema["STATIC_FEAT_COLS"]
    TIME_FEAT_COLS   = schema["TIME_FEAT_COLS"]
    DYN_FEAT_COLS    = schema["DYN_FEAT_COLS"]
    INST_FEAT_COLS   = schema["INST_FEAT_COLS"]

    DYN_IDX = torch.load(os.path.join(precomputed_dir, "dyn_idx.pt"), map_location="cpu")

    _edges_df = pd.read_parquet(os.path.join(precomputed_dir, "edges_vff.parquet"))
    if "geometry_wkb" in _edges_df.columns:
        _edges_df["geometry"] = _edges_df["geometry_wkb"].apply(shapely_wkb.loads)
        edges_vff_b = gpd.GeoDataFrame(_edges_df, geometry="geometry", crs="EPSG:4326").drop(columns=["geometry_wkb"])
    else:
        edges_vff_b = _edges_df

    _rel_raw = torch.load(os.path.join(precomputed_dir, "relations.pt"), map_location="cpu")
    if isinstance(_rel_raw, dict) and 'rel_edges' in _rel_raw:
        names = _rel_raw.get('names', REL_NAMES_DEFAULT)
        rel_edges_b = [_rel_raw['rel_edges'][n] for n in names]
    elif isinstance(_rel_raw, dict):
        rel_edges_b = [_rel_raw[n] for n in REL_NAMES_DEFAULT]
    else:
        rel_edges_b = _rel_raw

    SCALER_MEAN = torch.load(os.path.join(precomputed_dir, "scaler_mean.pt"), map_location="cpu")
    SCALER_STD  = torch.load(os.path.join(precomputed_dir, "scaler_std.pt"),  map_location="cpu")
    vff_vec_b   = torch.load(os.path.join(precomputed_dir, "vff_vec.pt"),     map_location="cpu")

    _tp_path = os.path.join(precomputed_dir, "link_traffic_profile.parquet")
    TRAFFIC_PROFILE = pd.read_parquet(_tp_path) if os.path.exists(_tp_path) else None
    if TRAFFIC_PROFILE is not None:
        print(f"[LinkETA] Loaded traffic profile: {TRAFFIC_PROFILE.shape}")
    else:
        print(f"[LinkETA] WARNING: Traffic profile not found at {_tp_path}")

    # publish schema constant the model reads by global lookup
    globals()["DYN_IDX"] = DYN_IDX

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vff_vec_b = vff_vec_b.to(device)
    IN_DIM = len(INST_FEAT_COLS)

    model = LinkETA(
        in_dim=IN_DIM,
        hidden_dim=CFG["hidden_dim"],
        heads=CFG["heads"],
        relations=len(rel_edges_b),
        num_layers=CFG["num_layers"],
        max_route_len=CFG.get("max_route_len", 1052),
        num_experts=3,
        dropout=0.3,
        input_dropout=0.2,
    ).to(device)

    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"[LinkETA] Checkpoint not found: {checkpoint_path}\n"
            f"Please provide a valid checkpoint_path when calling init_linketa()."
        )
    _pkg = torch.load(checkpoint_path, map_location=device)
    _state = _pkg["model"] if isinstance(_pkg, dict) and "model" in _pkg else _pkg
    model.load_state_dict(_state, strict=True)
    print(f"[LinkETA] Loaded checkpoint from: {checkpoint_path}")
    print(f"[LinkETA] init_linketa done: IN_DIM={IN_DIM}, relations={len(rel_edges_b)}, device={device}")

    _linketa_initialized = True

    base = dict(
        model=model,
        device=device,
        in_dim=IN_DIM,
        edges_id_col="id",
        precomputed_dir=precomputed_dir,   # [2026-06-07] used to locate dataset_cache/
        scalers=dict(mean=SCALER_MEAN, std=SCALER_STD, dyn_idx=DYN_IDX),
        graph=dict(edges_vff=edges_vff_b, rel_edges=rel_edges_b, vff_vec=vff_vec_b),
        traffic_profile=TRAFFIC_PROFILE,
        static_feat_cols=STATIC_FEAT_COLS,
        time_feat_cols=TIME_FEAT_COLS,
        dyn_feat_cols=DYN_FEAT_COLS,
        inst_feat_cols=INST_FEAT_COLS,
    )
    return base


# =============================================================================
# linketa_get_core_components - builds the entire LinkETA pipeline.
#   - base is None (default): heavy build from raw edges_gdf + df_all (slow;
#     used to CREATE a city's precomputed bundle / train a base from scratch).
#   - base given (from init_linketa): FAST path that reuses the precomputed
#     edges_vff / relations / vff_vec / scalers / model and only builds the
#     per-city datasets, loaders, optimizer & scheduler.
# Arg order is (df_all, edges_gdf) to match the MetaETA consumers.
# =============================================================================
# =============================================================================
# [MetaETA stateless] Module-level one_epoch / eval_linketa.
# Relocated out of linketa_get_core_components so the MetaETA consumers can
# `from LinkETA import one_epoch` and drive ANY core explicitly:
#     one_epoch(core, loader, epoch=0, train=True/False)
# They read model/optimizer/scalers/graph from `core`, so swapping a wrapped
# model is just `core["model"] = wrapped` before the call -- no module globals,
# safe to juggle multiple cores/cities. (The nested closures inside the heavy
# builder are kept intact and unused by these.)
# =============================================================================
def one_epoch(core, loader, epoch=0, train=True,
              w_speed=1/3, w_link=1/3, w_route=1/3):
    # [MetaETA stateless] read all per-model state from `core` (no module globals).
    model       = core["model"]
    device      = core["device"]
    optimizer   = core["optimizer"]
    clip_norm   = core.get("clip_norm", 1.0)
    SCALER_MEAN = core["scalers"]["mean"].to(device)
    SCALER_STD  = core["scalers"]["std"].to(device)
    vff_vec     = core["graph"]["vff_vec"].to(device)
    # model._encode_dyn_slots reads DYN_IDX by global lookup; keep it in sync
    # with THIS core (same value for every city, so no cross-contamination).
    globals()["DYN_IDX"] = core["scalers"]["dyn_idx"]
    if core.get("scaler") is None:
        core["scaler"] = torch.amp.GradScaler('cuda', enabled=USE_AMP)
    scaler = core["scaler"]
    # [2026-05-01 Stage 5 OLD signature]
    # def one_epoch(loader, epoch=0, train=True, w_link=0.5, w_route=0.5):
    # -> expanded to take w_speed too so DWA can balance all three primary losses.
    model.train() if train else model.eval()

    sum_speed_loss = 0.0     # [2026-05-01 Stage 5] track loss_speed for DWA history
    sum_link_loss = sum_route_loss = 0.0
    sum_mae_link = sum_mape_link = 0.0
    sum_sse_link = 0.0
    N_links = N_links_mape = 0
    # [2026-06-07] WAPE/MdAPE/P90 pooling (link)
    sum_abs_link = sum_true_link = 0.0
    ape_link_list = []

    sum_mae_route = sum_mape_route = 0.0
    sum_sse_route = 0.0
    N_routes_loss = N_routes_metrics = N_routes_mape = 0
    # [2026-06-07] WAPE/MdAPE/P90 pooling (route)
    sum_abs_route = sum_true_route = 0.0
    ape_route_list = []

    sum_speed_mae = sum_vol_mae = 0.0
    N_speed = N_vol = 0
    # [2026-05-01 Diagnostic] track speed MAE against the actual training target
    # (sp_actual), not the historical proxy. The legacy sum_speed_mae is kept
    # alongside so we can see both numbers per epoch.
    sum_speed_mae_actual = 0.0
    N_speed_actual = 0

    for batch in loader:
        # device move
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        # ----------------------------------------------------------------
        # Batch tensors (see TERMINOLOGY NOTE at top of file):
        #   node_edge_idx   [B, M]        - edge index for each position in local graph
        #   x_seq_nodes     [B, M, in_dim]- feature vectors for all positions (route + neighbors)
        #   route_flag_nodes[B, M]        - True=on-route, False=neighbor
        #   hop_idx         [B, M]        - BFS hop distance from route start
        #   masks_per_rel   list[R] of [B,M,M] - boolean adjacency masks per relation
        #   route_seq_idx   [B, L]        - indices into M positions for route sequence
        #   seq_true_tt     [B, L]        - ground-truth link travel time (seconds)
        #   seq_link_len    [B, L]        - traversed length (meters) per route link
        #   seq_mask        [B, L]        - boolean mask for valid route positions
        #   trip_tt         [B]           - ground-truth total trip travel time (seconds)
        # ----------------------------------------------------------------
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

        # Per-position speed/volume features -> gathered to route sequence
        #   sp_true_node [B, M] - observed speed ratio (speed/vff) per position [HISTORICAL PROXY]
        #   lc_true_node [B, M] - log-count of vehicles per position
        #   sp_w_node    [B, M] - weight/confidence for speed observation
        # [2026-05-01 Stage 2] sp_true_node was previously the speed-loss target.
        # We still gather it (kept available for diagnostics), but the speed-loss
        # target is now sp_actual computed from the (corrected) per-trip labels —
        # see the new sp_actual / sp_w_actual block below.
        sp_true_node = torch.nan_to_num(batch["sp_ratio_agg"], nan=0.0)
        lc_true_node = torch.nan_to_num(batch["logcnt1h_node"], nan=0.0)
        sp_w_node = torch.nan_to_num(batch["speed_w_node"], nan=0.0)

        gather_for_seq = lambda node_vals: torch.gather(
            node_vals, 1, route_seq_idx
        )

        # Gather to route sequence: [B, M] -> [B, L]
        sp_true = gather_for_seq(sp_true_node)     # [B,L] HISTORICAL proxy (no longer used as loss target)
        lc_true = gather_for_seq(lc_true_node)     # [B,L] log-count at route positions
        sp_w = gather_for_seq(sp_w_node)           # [B,L] HISTORICAL weight (no longer used as loss weight)

        # vff_seq_b [B, L] - free-flow speed (m/s) for each link in route
        vff_seq_b = vff_vec[node_edge_idx]         # [B, M] vff for all positions
        vff_seq_b = torch.gather(vff_seq_b, 1, route_seq_idx)  # [B, L] gather route only
        # [2026-05-01 Stage 2.1] Keep a CLEAN copy of v_ff (pre-noise) so we can
        # detect rows that were pinned at the r_max floor by Stage 1 redistribution.
        # Pinned rows satisfy link_tt = link_len / (r_max · v_ff_clean) exactly,
        # so sp_actual evaluated against vff_seq_clean equals R_MAX_TARGET on the
        # nose — these rows must be excluded from the speed loss because the
        # features cannot disambiguate them from clean rows and their target is
        # synthetic (a redistribution boundary, not a measurement).
        vff_seq_clean = vff_seq_b.clone()
        if train:
            rng = vff_noise_range(epoch)
            if rng > 0:
                noise = 1.0 + rng * (2.0 * torch.rand_like(vff_seq_b) - 1.0)
                vff_seq_b = vff_seq_b * noise

        # ----------------------------------------------------------------
        # [2026-05-01 Stage 2] Per-link sp_actual TARGET, derived from the
        # (already-redistributed) per-link labels. After Stage 1 every row's
        # sp_actual <= r_max by construction, so the bounded sp_hat can fit
        # it exactly. This replaces the historical-proxy target above.
        #     sp_actual    = link_len_m / (link_tt · v_ff)        # ratio
        #     sp_w_actual  = trust mask: 1 where label is physically credible
        # ----------------------------------------------------------------
        R_MAX_TARGET = 3.0   # must match the r_max used in prepare_corrected_dataset.py
        sp_actual = (seq_link_len / seq_true_tt.clamp_min(1e-3)) / vff_seq_b.clamp_min(1.0)
        sp_actual = sp_actual.clamp(0.0, R_MAX_TARGET)

        # [2026-05-01 Stage 2.1] Detect rows pinned at the floor by Stage 1.
        # Use vff_seq_clean (pre-noise) so the pinned condition is exact:
        # sp_actual_clean == R_MAX_TARGET to within float epsilon.
        sp_actual_clean = (
            seq_link_len / seq_true_tt.clamp_min(1e-3)
        ) / vff_seq_clean.clamp_min(1.0)
        was_pinned = sp_actual_clean >= (R_MAX_TARGET - 1e-3)

        sp_w_actual = (
            seq_mask
            & (seq_true_tt  >= 1.0)
            & (seq_link_len >= 10.0)
            & (~was_pinned)        # exclude Stage-1 pinned rows from speed loss
        ).float()

        # normalize x_seq_nodes
        x_seq_norm = (x_seq_nodes - SCALER_MEAN.view(1, 1, -1)) / SCALER_STD.view(1, 1, -1)

        # masks: seq_mask = all valid links; sp_mask = legacy mask (kept for compatibility / diagnostics)
        sp_mask = (sp_w > 0) & seq_mask

        Bsz = seq_true_tt.size(0)

        with torch.amp.autocast('cuda', enabled=USE_AMP):
            # Model returns 7 values: link_eta_seq, trip_eta, sp_hat, logcnt_1h, g, delta, eta_phys
            link_eta_seq, trip_eta, sp_pred, lc_pred, g, delta, eta_phys_seq = model(
                x_seq_norm,
                hop_idx,
                route_flag_nodes,
                masks_per_rel,
                route_seq_idx,
                seq_link_len,
                vff_seq_b,
                seq_mask,
                # [2026-04-28 A1] r_max 3.0 -> 2.0
                # [2026-05-01 Stage 2/3] r_max 2.0 -> 3.0 to match SF-derived
                # statistical bound + the r_max used in Stage 1 redistribution.
                # Now consistent: data is clipped at sp_actual <= 3, model
                # output is bounded at sp_hat <= 3, target sp_actual <= 3.
                r_max=R_MAX_TARGET,
                return_parts=True,
            )

            link_eta_seq = torch.nan_to_num(link_eta_seq, nan=0.0)
            trip_eta = torch.nan_to_num(trip_eta, nan=0.0)
            sp_pred = torch.nan_to_num(sp_pred, nan=0.0)
            lc_pred = torch.nan_to_num(lc_pred, nan=0.0)

            # [2026-04-28 A1+A2] Funnel model output through the seconds-helper.
            # After A1 this is a no-op, but it keeps a single canonical path so
            # any future swap to e.g. log1p-target heads only changes one place.
            pred_seq = link_eta_in_seconds(link_eta_seq, seq_link_len)
            seq_true = seq_true_tt.clamp_min(0.0)  # [2026-04-28 A2] guard for log1p (a few rows have negative TT)

            # ---------- main link & route losses (use ALL valid links) ----------
            # [2026-04-28 A2] delta=1.0 -> 0.3 (Huber operates in log1p space now)
            # [2026-05-01 Step 6 OLD — replaced for long-trip-tail fix:
            # loss_link  = link_loss_huber_macro(pred_seq, seq_true, seq_mask, delta=0.3)
            # loss_route = route_loss_mape(pred_seq, trip_tt, seq_mask)
            # ]
            # New training losses:
            #   - link: length-weighted log1p-Huber so long links drive the gradient
            #     (matches what dominates Route MAE).
            #   - route: log1p-Huber so long-trip outliers no longer get a vanishing
            #     1/trip_tt MAPE gradient.
            loss_link  = link_loss_huber_log_lenweighted(
                pred_seq, seq_true, seq_mask,
                link_len=seq_link_len,
                delta=0.3,
            )
            loss_route = route_loss_huber_log(pred_seq, trip_tt, seq_mask, delta=0.3)

            # ---------- speed loss : PRIMARY supervision (Stage 2 target swap) ----------
            # [2026-05-01 Stage 2] Target swapped from sp_ratio_agg (historical proxy)
            # to sp_actual (per-link actual ratio derived from corrected labels).
            # Weight swapped from sp_w (historical confidence) to sp_w_actual (trust mask).
            # SmoothL1 in ratio-space with beta=0.1 (≈10 % of free-flow as transition point).
            denom = sp_w_actual.sum().clamp_min(1.0)
            per = F.smooth_l1_loss(sp_pred, sp_actual, reduction='none', beta=0.1)
            loss_speed = (per * sp_w_actual).sum() / denom

            # [2026-05-01 Stage 2 OLD — historical-proxy target, DISABLED]
            # if sp_mask.any():
            #     per = F.smooth_l1_loss(sp_pred[sp_mask], sp_true[sp_mask], reduction='none')
            #     w = sp_w[sp_mask]
            #     loss_speed = (w * per).sum() / w.sum().clamp_min(1.0)
            # else:
            #     loss_speed = torch.zeros((), device=device)

            # ---------- volume auxiliary loss (use ALL valid links) ----------

            if seq_mask.any():
                loss_vol = F.smooth_l1_loss(lc_pred[seq_mask], lc_true[seq_mask])
            else:
                loss_vol = torch.zeros((), device=device)


            # [2026-04-28 A1] g calibration removed; this term is always 0 now.
            # ---------- calibration penalty on g (DISABLED) ----------
            # lam_g = lam_g_at(epoch)

            # ---------- final loss ----------
            # [2026-05-01 Stage 5] All three primary losses (speed, link, route)
            # are now under DWA. loss_vol stays a fixed-weight auxiliary.
            final_loss = (
                w_speed * loss_speed +
                w_link  * loss_link  +
                w_route * loss_route +
                0.10    * loss_vol
            )
            # [2026-04-28 A1 OLD] hardcoded weights for loss_speed (was 0.05, then 0.30):
            # final_loss = (
            #     w_link  * loss_link  +
            #     w_route * loss_route +
            #     0.30    * loss_speed +
            #     0.10    * loss_vol
            #     # + lam_g * (g - 1.0).abs().mean()   # [2026-04-28 A1] removed
            # )

        if train:
            optimizer.zero_grad(set_to_none=True)
            if USE_AMP:
                scaler.scale(final_loss).backward()
                if clip_norm is not None:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                final_loss.backward()
                if clip_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                optimizer.step()

        # ---------- metrics (using provided functions) ----------
        # link_metrics_micro uses seq_mask for MAE/RMSE but internally filters for MAPE
        link_stats = link_metrics_micro(pred_seq, seq_true, seq_mask,
                                        link_len=seq_link_len,
                                        L_min=L_MIN_METRIC,
                                        T_min=T_MIN_METRIC)
        route_stats = route_metrics_pooled(pred_seq, trip_tt, seq_mask)

        sum_speed_loss += float(loss_speed.item()) * Bsz   # [2026-05-01 Stage 5] new
        sum_link_loss  += float(loss_link.item()) * Bsz
        sum_route_loss += float(loss_route.item()) * Bsz
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
        # [2026-06-07] pool WAPE/MdAPE/P90 building blocks (link)
        sum_abs_link += float(link_stats.get("sum_abs", 0.0))
        sum_true_link += float(link_stats.get("sum_true", 0.0))
        _av = link_stats.get("ape_vals")
        if _av is not None and _av.numel() > 0:
            ape_link_list.append(_av)

        n_routes = int(route_stats["n_routes"])
        if n_routes > 0:
            sum_mae_route += route_stats["mae"] * n_routes
            sum_sse_route += (route_stats["rmse"] ** 2) * n_routes
            n_mr = int(route_stats.get("n_mape", 0))
            if n_mr > 0 and not math.isnan(route_stats["mape"]):
                sum_mape_route += route_stats["mape"] * n_mr
                N_routes_mape += n_mr
            N_routes_metrics += n_routes
        # [2026-06-07] pool WAPE/MdAPE/P90 building blocks (route)
        sum_abs_route += float(route_stats.get("sum_abs", 0.0))
        sum_true_route += float(route_stats.get("sum_true", 0.0))
        _avr = route_stats.get("ape_vals")
        if _avr is not None and _avr.numel() > 0:
            ape_route_list.append(_avr)

        if sp_mask.any():
            # [LEGACY metric — sp_pred vs HISTORICAL proxy sp_ratio_agg, kept for continuity]
            sp_abs = (sp_pred[sp_mask] - sp_true[sp_mask]).abs().mean().item()
            sum_speed_mae += sp_abs * int(sp_mask.sum().item())
            N_speed += int(sp_mask.sum().item())

        # [2026-05-01 Diagnostic] speed MAE against the ACTUAL loss target (sp_actual).
        # This is the number that tells you whether sp_hat is converging to the
        # supervision target. The legacy speed_mae above is against the historical
        # proxy and naturally non-zero — don't read it as "model not learning."
        sp_actual_mask = sp_w_actual.bool()
        if sp_actual_mask.any():
            sp_abs_actual = (sp_pred[sp_actual_mask] - sp_actual[sp_actual_mask]).abs().mean().item()
            n_act = int(sp_actual_mask.sum().item())
            sum_speed_mae_actual += sp_abs_actual * n_act
            N_speed_actual += n_act

        if seq_mask.any():
            vol_abs = (lc_pred[seq_mask] - lc_true[seq_mask]).abs().mean().item()
            n_valid = int(seq_mask.sum().item())
            sum_vol_mae += vol_abs * n_valid
            N_vol += n_valid

    _lpct = finalize_pct_metrics(sum_abs_link, sum_true_link, ape_link_list)
    _rpct = finalize_pct_metrics(sum_abs_route, sum_true_route, ape_route_list)
    return {
        'speed_loss': sum_speed_loss / max(N_routes_loss, 1),   # [2026-05-01 Stage 5] for DWA
        'link_loss':  sum_link_loss  / max(N_routes_loss, 1),
        'route_loss': sum_route_loss / max(N_routes_loss, 1),
        'link_mae':   sum_mae_link   / max(N_links, 1),
        'link_rmse':  math.sqrt(sum_sse_link / max(N_links, 1)) if N_links > 0 else float("nan"),
        'link_mape':  (sum_mape_link / max(N_links_mape, 1)) if N_links_mape > 0 else float('nan'),
        'link_wape':  _lpct['wape'], 'link_mdape': _lpct['mdape'], 'link_p90': _lpct['ape_p90'],  # [2026-06-07]
        'route_mae':  sum_mae_route / max(N_routes_metrics, 1),
        'route_rmse': math.sqrt(sum_sse_route / max(N_routes_metrics, 1)) if N_routes_metrics > 0 else float("nan"),
        'route_mape': (sum_mape_route / max(N_routes_mape, 1)) if N_routes_mape > 0 else float('nan'),
        'route_wape': _rpct['wape'], 'route_mdape': _rpct['mdape'], 'route_p90': _rpct['ape_p90'],  # [2026-06-07]
        'speed_mae':         (sum_speed_mae        / max(N_speed,        1)) if N_speed        else float('nan'),
        'speed_mae_actual':  (sum_speed_mae_actual / max(N_speed_actual, 1)) if N_speed_actual else float('nan'),  # [2026-05-01 Diagnostic]
        'vol_mae':    (sum_vol_mae / max(N_vol, 1)) if N_vol else float('nan'),
    }




def eval_linketa(core,
                 ckpt_path: str = None,
                 label: str = "LinkETA",
                 split: str = "test"):
    """
    Evaluation loop for LinkETA model (Fixed version).
    
    Model forward() returns 7 values:
        (link_eta_seq, trip_eta, sp_hat, logcnt_1h, g, delta, eta_phys)
    
    Args:
        core: Dictionary containing model, device, loaders, scalers, etc.
        ckpt_path: Optional path to checkpoint file
        label: Label for output DataFrames
        split: Which data split to evaluate ("train", "val", or "test")
    
    Returns:
        metrics: Dictionary with link/route losses, MAE, RMSE, MAPE, speed/vol MAE
        trip_df: DataFrame with per-trip predictions
        link_df: DataFrame with per-link predictions
    """
    device = core["device"]
    model  = core["model"].to(device)
    globals()["DYN_IDX"] = core["scalers"]["dyn_idx"]

    # pick loader
    if "loaders" in core and split in core["loaders"]:
        loader = core["loaders"][split]
    else:
        key = f"{split}_loader"
        if key in core:
            loader = core[key]
        else:
            raise KeyError(f"No DataLoader found for split='{split}' in core.")

    # v_ff vector and feature scaler
    vff_vec_local = core["graph"]["vff_vec"].to(device)
    scaler_mean   = core["scalers"]["mean"].to(device)
    scaler_std    = core["scalers"]["std"].to(device)

    # optional ckpt
    if ckpt_path is not None:
        pkg = torch.load(ckpt_path, map_location=device)
        state = pkg["model"] if isinstance(pkg, dict) and "model" in pkg else pkg
        model.load_state_dict(state, strict=True)

    model.eval()

    # metric accumulator (link+route)
    agg = _init_metric_accumulator()

    # extras: speed/volume
    # [2026-05-01 Stage 2.1] We now track BOTH the historical-proxy speed MAE
    # (Speed[hist], vs sp_ratio_agg) AND the Stage-2 actual-target speed MAE
    # (Speed[actual], vs sp_actual). The actual-target metric matches the
    # training loss target and excludes Stage-1-pinned rows.
    sum_speed_mae = sum_vol_mae = 0.0
    N_speed = N_vol = 0
    sum_speed_mae_actual = 0.0
    N_speed_actual = 0

    trip_rows = []
    link_rows = []

    for batch in loader:
        # move tensors
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        node_edge_idx    = batch["node_edge_idx"]       # [B,M]
        x_seq_nodes      = batch["x_seq"]               # [B,M,F]
        route_flag_nodes = batch["route_flag"].bool()   # [B,M]
        hop_idx          = batch["hop_idx"].long()      # [B,M]
        masks_per_rel    = [m.to(device) for m in batch["masks_per_rel"]]

        route_seq_idx = batch["route_seq_idx"].long()   # [B,L]
        seq_true_tt   = torch.nan_to_num(batch["seq_true_tt"], nan=0.0)      # [B,L]
        seq_link_len  = torch.nan_to_num(batch["seq_link_len"], nan=0.0)     # [B,L]
        seq_link_len  = seq_link_len.clamp_min(1e-3)
        seq_mask      = batch["seq_mask"].bool()        # [B,L]
        trip_tt       = torch.nan_to_num(batch["trip_tt"], nan=0.0)          # [B]

        # per-node recency features -> sequence
        sp_true_node = torch.nan_to_num(batch["sp_ratio_agg"],   nan=0.0)    # [B,M]
        lc_true_node = torch.nan_to_num(batch["logcnt1h_node"],  nan=0.0)    # [B,M]
        sp_w_node    = torch.nan_to_num(batch["speed_w_node"],   nan=0.0)    # [B,M]

        gather_for_seq = lambda node_vals: torch.gather(
            node_vals, 1, route_seq_idx
        )

        sp_true = gather_for_seq(sp_true_node)   # [B,L]
        lc_true = gather_for_seq(lc_true_node)   # [B,L]
        sp_w    = gather_for_seq(sp_w_node)      # [B,L]

        # v_ff per node -> per-sequence
        vff_node = vff_vec_local[node_edge_idx]     # [B,M]
        vff_seq  = gather_for_seq(vff_node)         # [B,L]

        # normalize features
        x_seq_norm = (x_seq_nodes - scaler_mean.view(1, 1, -1)) / scaler_std.view(1, 1, -1)
        x_seq_norm = torch.nan_to_num(x_seq_norm)

        # forward - Model returns 7 values (no mix/eta_direct, no force_physics)
        with torch.amp.autocast('cuda', enabled=USE_AMP):
            (
                link_eta_seq,   # [B,L]
                trip_eta,       # [B]
                sp_pred,        # [B,L]
                lc_pred,        # [B,L]
                g,              # [B]
                delta,          # [B,L]
                eta_phys_seq,   # [B,L]
            ) = model(
                x_seq_norm,
                hop_idx,
                route_flag_nodes,
                masks_per_rel,
                route_seq_idx,
                seq_link_len,
                vff_seq,
                seq_mask,
                # [2026-04-28 A1] r_max 2.8 -> 2.0 (consistent with training)
                # [2026-05-01 Stage 2/3] r_max 2.0 -> 3.0 (matches Stage 1 redistribution)
                r_max=3.0,
                return_parts=True,
            )

            link_eta_seq = torch.nan_to_num(link_eta_seq, nan=0.0)
            trip_eta     = torch.nan_to_num(trip_eta,     nan=0.0)
            sp_pred      = torch.nan_to_num(sp_pred,      nan=0.0)
            lc_pred      = torch.nan_to_num(lc_pred,      nan=0.0)

            # [2026-04-28 A1+A2] Same seconds-helper as training path.
            pred_seq = link_eta_in_seconds(link_eta_seq, seq_link_len)
            seq_true = seq_true_tt.clamp_min(0.0)

            # ---- canonical metrics / losses for this batch ----
            # [2026-04-28 A2] delta_huber 1.0 -> 0.3 (log-space)
            stats = compute_global_losses_and_metrics(
                pred_seq=pred_seq,
                trip_tt=trip_tt,
                seq_true=seq_true,
                seq_mask=seq_mask,
                seq_link_len=seq_link_len,
                delta_huber=0.3,
            )

        # accumulate link/route metrics
        _accumulate_batch_metrics(agg, stats)

        # ---------- speed / volume MAE ----------
        # Speed[hist] : legacy proxy target (sp_ratio_agg). Kept for continuity.
        sp_mask = (sp_w > 0) & seq_mask
        if sp_mask.any():
            sp_abs = (sp_pred[sp_mask] - sp_true[sp_mask]).abs().mean().item()
            n_sp   = int(sp_mask.sum().item())
            sum_speed_mae += sp_abs * n_sp
            N_speed       += n_sp

        # [2026-05-01 Stage 2.1] Speed[actual] : Stage-2 training target.
        # No noise injection in eval, so vff_seq is already clean — single pass.
        # Trust mask matches one_epoch: trip-credible link, len >= 10m, tt >= 1s,
        # and Stage-1 pinned rows excluded (synthetic targets).
        R_MAX_TARGET = 3.0
        sp_actual = (seq_link_len / seq_true_tt.clamp_min(1e-3)) / vff_seq.clamp_min(1.0)
        sp_actual = sp_actual.clamp(0.0, R_MAX_TARGET)
        was_pinned = sp_actual >= (R_MAX_TARGET - 1e-3)
        sp_w_actual = (
            seq_mask
            & (seq_true_tt  >= 1.0)
            & (seq_link_len >= 10.0)
            & (~was_pinned)
        )
        if sp_w_actual.any():
            sp_abs_act = (sp_pred[sp_w_actual] - sp_actual[sp_w_actual]).abs().mean().item()
            n_sp_act   = int(sp_w_actual.sum().item())
            sum_speed_mae_actual += sp_abs_act * n_sp_act
            N_speed_actual       += n_sp_act

        if seq_mask.any():
            vol_abs = (lc_pred[seq_mask] - lc_true[seq_mask]).abs().mean().item()
            n_vol   = int(seq_mask.sum().item())
            sum_vol_mae += vol_abs * n_vol
            N_vol       += n_vol

        # ---------- per-trip rows ----------
        route_true_from_links = (seq_true_tt * seq_mask.float()).sum(dim=1)  # [B]
        yR = torch.where(trip_tt > 0, trip_tt, route_true_from_links)
        pR = (pred_seq * seq_mask.float()).sum(dim=1)  # equals trip_eta by construction

        trip_ids     = batch["trip_ids"]
        # Handle trip_ids - may be list or tensor
        if isinstance(trip_ids, torch.Tensor):
            trip_ids_np = trip_ids.detach().cpu().numpy()
        else:
            trip_ids_np = np.array(trip_ids)
        y_true_np    = yR.detach().cpu().numpy()
        y_pred_np    = pR.detach().cpu().numpy()
        route_len_np = (seq_link_len * seq_mask.float()).sum(dim=1).detach().cpu().numpy()
        g_np         = g.detach().cpu().numpy()
        src_tag_np   = np.where(trip_tt.detach().cpu().numpy() > 0, "trip_tt", "sum_links")

        for i in range(len(trip_ids_np)):
            trip_rows.append((
                label,
                int(trip_ids_np[i]) if np.isscalar(trip_ids_np[i]) else int(trip_ids_np[i]),
                float(y_true_np[i]),
                float(y_pred_np[i]),
                src_tag_np[i],
                float(route_len_np[i]),
                float(g_np[i]),
            ))

        # ---------- per-link rows (for BA, etc.) ----------
        pred_seq_cpu = pred_seq.detach().cpu()
        seq_true_cpu = seq_true_tt.detach().cpu()
        mask_cpu     = seq_mask.detach().cpu()

        B0, L0 = pred_seq_cpu.shape
        for b in range(B0):
            tid = int(trip_ids_np[b]) if np.isscalar(trip_ids_np[b]) else int(trip_ids_np[b])
            for l in range(L0):
                if not bool(mask_cpu[b, l]):
                    continue
                link_rows.append((
                    label,
                    tid,
                    float(seq_true_cpu[b, l].item()),
                    float(pred_seq_cpu[b, l].item()),
                ))

    # ---- finalize metrics ----
    core_metrics = _finalize_metric_accumulator(agg)

    if N_speed > 0:
        speed_mae = sum_speed_mae / N_speed
    else:
        speed_mae = float("nan")

    if N_speed_actual > 0:
        speed_mae_actual = sum_speed_mae_actual / N_speed_actual
    else:
        speed_mae_actual = float("nan")

    if N_vol > 0:
        vol_mae = sum_vol_mae / N_vol
    else:
        vol_mae = float("nan")

    metrics = dict(core_metrics)
    metrics.update({
        "speed_mae":         speed_mae,           # Speed[hist]   - legacy proxy
        "speed_mae_actual":  speed_mae_actual,    # Speed[actual] - Stage-2 target
        "vol_mae":           vol_mae,
    })

    trip_df = pd.DataFrame(
        trip_rows,
        columns=["model", "TRIP_ID", "y_true", "y_pred",
                 "target_source", "route_len_m", "gain_g"],
    )
    link_df = pd.DataFrame(
        link_rows,
        columns=["model", "TRIP_ID", "true_tt_link", "pred_tt_link"],
    )

    return metrics, trip_df, link_df




def _linketa_core_from_base(df_all, edges_gdf, base, cache_datasets=False, cache_seed=42):
    """
    FAST core builder: reuse a precomputed `base` (from init_linketa) and only
    build THIS city's datasets / loaders / optimizer / scheduler. No relation
    rebuild, no v_ff fit, no high-order edge mining -> fast reruns. Stateless:
    all reused state comes from `base`; only DYN_IDX (a schema constant the model
    reads by global lookup) is published to globals.
    """
    model       = base["model"]
    device      = base["device"]
    edges_vff_b = base["graph"]["edges_vff"]
    rel_edges_b = base["graph"]["rel_edges"]
    vff_vec_b   = base["graph"]["vff_vec"]
    SCALER_MEAN = base["scalers"]["mean"]
    SCALER_STD  = base["scalers"]["std"]
    DYN_IDX_b   = base["scalers"]["dyn_idx"]
    STATIC_FEAT_COLS = base["static_feat_cols"]
    TIME_FEAT_COLS   = base["time_feat_cols"]
    DYN_FEAT_COLS    = base["dyn_feat_cols"]
    INST_FEAT_COLS   = base["inst_feat_cols"]
    edges_id_col     = base.get("edges_id_col", "id")
    traffic_profile  = base.get("traffic_profile")

    globals()["DYN_IDX"] = DYN_IDX_b  # model._encode_dyn_slots reads via global lookup

    # map df_all -> edge_idx using the precomputed edges. ensure_static_feature_columns
    # matches id<->way_id by RAW dtype, so align both as strings first (the heavy path
    # does the same external str-cast before calling it). Without this an int/str
    # mismatch maps every row to NaN and dropna() empties the dataframe.
    edges_for_map = edges_vff_b.copy()
    edges_for_map[edges_id_col] = edges_for_map[edges_id_col].astype(str)
    df_all = df_all.copy()
    df_all[CFG["way_id_col_df"]] = df_all[CFG["way_id_col_df"]].astype(str)
    edges_local, _static_mapped, df_all = ensure_static_feature_columns(
        edges_for_map, df_all, CFG["static_cols"], edges_id_col, CFG["way_id_col_df"]
    )
    df_all = df_all.dropna(subset=["edge_idx"])
    df_all["edge_idx"] = df_all["edge_idx"].astype(int)

    if 'tod_sin' not in df_all.columns:
        df_all = add_time_features(df_all)

    df_all = merge_traffic_profile(df_all, traffic_profile)

    inst, INST_FEAT_COLS_local, _, _, _ = build_instance_features(
        df_all, edges_static=edges_vff_b, traffic_profile=traffic_profile,
        inst_feat_cols=INST_FEAT_COLS,
    )

    dyn_12x5_cols = select_dynamic_12x5_columns(df_all)
    DYN_STORE = EdgeDynStore(
        df_all, dyn_12x5_cols, edges_vff_b,
        way_id_col_df=CFG["way_id_col_df"],
        way_id_col_edges=edges_id_col,
        time_col=CFG["time_col_df"],
    )

    df_train, df_val, df_test = temporal_split_by_trip(df_all)

    # [2026-06-07] optional off-route-neighbour disk cache under <precomputed_dir>/dataset_cache/
    _cache_dir = (os.path.join(base.get("precomputed_dir", "."), "dataset_cache")
                  if cache_datasets else None)
    _ds_kwargs = dict(
        edges_gdf=edges_vff_b,
        inst_table=inst,
        rel_edges_list=rel_edges_b,
        static_feat_cols=STATIC_FEAT_COLS,
        time_feat_cols=TIME_FEAT_COLS,
        dyn_feat_cols=DYN_FEAT_COLS,
        dyn_store=DYN_STORE,
        K_first=CFG["K_first"],
        K_conflict=CFG["K_conflict"],
        K_high=CFG["K_high"],
        use_cache=cache_datasets,
        cache_dir=_cache_dir,
        cache_seed=cache_seed,
    )
    train_ds = TripDatasetLinkETA(df_train, **_ds_kwargs)
    val_ds   = TripDatasetLinkETA(df_val,   **_ds_kwargs)
    test_ds  = TripDatasetLinkETA(df_test,  **_ds_kwargs)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True,
                              collate_fn=collate_trips_linketa, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False,
                              collate_fn=collate_trips_linketa, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=16, shuffle=False,
                              collate_fn=collate_trips_linketa, num_workers=0)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    SCALER_MEAN_d = SCALER_MEAN.to(device)
    SCALER_STD_d  = SCALER_STD.to(device)
    DYN_IDX_d     = DYN_IDX_b.to(device)
    vff_vec_d     = vff_vec_b.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=3e-6
    )
    HIST = {
        "train": {"speed_loss": [], "route_loss": [], "link_loss": [], "route_mape": [],
                  "link_mape": [], "speed_mae": [], "speed_mae_actual": [], "vol_mae": []},
        "val":   {"speed_loss": [], "route_loss": [], "link_loss": [], "route_mape": [],
                  "link_mape": [], "speed_mae_actual": [], "speed_mae": [], "vol_mae": []},
    }
    ckpt = CheckpointManager(dirpath=CKPT_DIRNAME, max_keep=3)

    print(f"[LinkETA] (fast/base) Device: {device}, "
          f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M, "
          f"batches train/val/test: {len(train_loader)}/{len(val_loader)}/{len(test_loader)}")

    core = dict(
        device=device,
        model=model,
        loaders=dict(train=train_loader, val=val_loader, test=test_loader),
        datasets=dict(train=train_ds, val=val_ds, test=test_ds),
        inst_table=inst,
        inst_feat_cols=INST_FEAT_COLS_local,
        static_cols=STATIC_FEAT_COLS,
        time_cols=TIME_FEAT_COLS,
        dyn_cols=DYN_FEAT_COLS,
        graph=dict(edges_vff=edges_vff_b, rel_edges=rel_edges_b, vff_vec=vff_vec_d),
        dyn_store=DYN_STORE,
        cfg=CFG,
        constants=dict(H_SLICES=H_SLICES, BIN_SECONDS=BIN_SECONDS, L_MIN=10.0, T_MIN=1.5),
        scalers=dict(mean=SCALER_MEAN_d, std=SCALER_STD_d, dyn_idx=DYN_IDX_d),
        optimizer=optimizer,
        scheduler=scheduler,
        history=HIST,
        num_epochs=100,
        clip_norm=1.0,
        checkpoint_manager=ckpt,
        scaler=None,  # one_epoch lazily creates a GradScaler here on first train call
    )
    return core


def linketa_get_core_components(df_all, edges_gdf, base=None, cache_datasets=False, cache_seed=42):
    if base is not None:
        return _linketa_core_from_base(df_all, edges_gdf, base,
                                       cache_datasets=cache_datasets, cache_seed=cache_seed)
    try:
        from torch.amp import GradScaler, autocast
    except ImportError:
        from torch.cuda.amp import GradScaler, autocast
    # ---- nested helpers moved from module level (closure-captured) ---

    def select_dynamic_12x5_columns_full(df_all: pd.DataFrame) -> List[str]:
        """
        Ensure all 12×P dynamic columns exist, with suffix 0 = most recent, 11 = oldest.
        """
        order = list(range(0, 12))  # 0..11
        cols = [f"{p}_{s}" for s in order for p in CFG["dyn_prefixes"]]
        for c in cols:
            if c not in df_all.columns:
                df_all[c] = 0.0
        return cols

    def build_instance_features(df_all):
        # Use canonical lists from CFG for consistent feature dimensions
        static_feat_cols = list(CFG["static_cols"])
        time_feat_cols = list(CFG["time_cols"])
        
        # Force all static columns to exist (fill missing with 0.0)
        for col in static_feat_cols:
            if col not in df_all.columns:
                df_all[col] = 0.0
        
        edge_static = (df_all.groupby('edge_idx')[static_feat_cols].first().fillna(0.0)
                       if static_feat_cols else pd.DataFrame(index=df_all['edge_idx'].unique()))

        dyn_feat_cols = select_dynamic_12x5_columns_full(df_all)

        BASE_COLS = cols_if_exist(df_all, ["TRIP_ID", "edge_idx"] + time_feat_cols + dyn_feat_cols)
        sort_cols = ['TRIP_ID', 'enter_ts'] if 'enter_ts' in df_all.columns else ['TRIP_ID', 'departure_time']

        inst_raw = (
            df_all.sort_values(sort_cols)
                  .drop_duplicates(subset=['TRIP_ID', 'edge_idx'], keep='first')
                  [BASE_COLS]
                  .set_index(['TRIP_ID', 'edge_idx'])
                  .fillna(0.0)
        )
        inst = inst_raw.join(edge_static, on='edge_idx', how='left').fillna(0.0)
        INST_FEAT_COLS = list(inst.columns)
        return inst, INST_FEAT_COLS, static_feat_cols, time_feat_cols, dyn_feat_cols

    def dyn_indices(INST_FEAT_COLS):
        prefixes = CFG["dyn_prefixes"]
        DYN_COLS = [c for c in INST_FEAT_COLS if any(str(c).startswith(f"{p}_") for p in prefixes)]
        return torch.tensor([INST_FEAT_COLS.index(c) for c in DYN_COLS], dtype=torch.long)

    def discover_slot_indices(df_all: pd.DataFrame, base_prefix: Optional[str] = "mean_speed") -> List[int]:
        prefixes = [base_prefix] if base_prefix is not None else []
        if not prefixes or not any(any(c.startswith(f"{p}_") for c in df_all.columns) for p in prefixes):
            prefixes = [p for p in CFG["dyn_prefixes"] if any(c.startswith(f"{p}_") for c in df_all.columns)]
        idxs = set()
        for p in prefixes:
            pat = re.compile(rf"^{re.escape(p)}_(\d+)$")
            for c in df_all.columns:
                m = pat.match(c)
                if m:
                    try:
                        idxs.add(int(m.group(1)))
                    except ValueError:
                        pass
        return sorted(idxs)

    def select_dynamic_12x5_columns(df_all: pd.DataFrame) -> List[str]:
        """
        k=0 is most recent; we want the last hour => slots 0..11.
        Order newest->oldest: 0..11, and index 0 in the 12×P block is the latest slot.
        """
        order = list(range(0, 12))  # 0..11
        cols = []
        for s in order:
            for p in CFG["dyn_prefixes"]:
                cols.append(f"{p}_{s}")
        print("Using dynamic slots 0..11 (0 is latest).")
        return cols

    def build_high_order_edge_index_dueta(
        df_all: pd.DataFrame,
        edges_gdf: pd.DataFrame,
        top_k: int,
        way_id_col_df: str,
        way_id_col_edges: str,
        length_col: str,
        slots_per_hour: int,
        speed_metric_index: int,
        speed_is_kmh: bool,
        exclude_first_order: Optional[torch.LongTensor] = None,
    ) -> torch.LongTensor:
        id2edge = dict(zip(edges_gdf[way_id_col_edges].tolist(), edges_gdf["edge_idx"].tolist()))
        idxs = discover_slot_indices(df_all, base_prefix="mean_speed")
        last1h = sorted(idxs[-slots_per_hour:])
        if len(last1h) == 0:
            raise ValueError(
                "No dynamic slot columns found. Expected columns like "
                f"'median_speed_<k>' or {CFG['dyn_prefixes']} with numeric suffix."
            )
        flat_cols_12x5 = sum([[f"{p}_{s}" for p in SLOT_PREFIXES] for s in last1h], [])

        # edge length per edge_idx
        if length_col in edges_gdf.columns:
            edge_len = dict(zip(edges_gdf["edge_idx"], edges_gdf[length_col]))
        else:
            tmp = df_all.dropna(subset=[way_id_col_df, length_col]).copy()
            tmp["edge_idx"] = tmp[way_id_col_df].map(id2edge)
            edge_len = tmp.dropna(subset=["edge_idx"]).groupby("edge_idx")[length_col].first().to_dict()

        corr_accum: Dict[Tuple[int, int], float] = {}
        time_key = CFG["time_col_df"] if CFG["time_col_df"] in df_all.columns else "enter_ts"
        grouped = df_all.sort_values(time_key).groupby("TRIP_ID", sort=False)

        for _, grp in grouped:
            es = grp[way_id_col_df].map(id2edge)
            grp = grp[pd.notna(es)]
            es = grp[way_id_col_df].map(id2edge).astype(int).tolist()
            if len(es) < 3:
                continue
            slot_block = grp[flat_cols_12x5].to_numpy().reshape(len(grp), len(last1h), len(SLOT_PREFIXES))  # [R,12,5]

            L = len(es)
            for i in range(L):
                ei = es[i]
                if ei not in edge_len:
                    continue
                tt_i = speeds_to_tt(slot_block[i], edge_len[ei], speed_metric_index, speed_is_kmh)
                for d in (2, 3, 4, 5):
                    for j_idx in (i - d, i + d):
                        if 0 <= j_idx < L:
                            ej = es[j_idx]
                            if ej not in edge_len:
                                continue
                            tt_j = speeds_to_tt(slot_block[j_idx], edge_len[ej], speed_metric_index, speed_is_kmh)
                            if np.allclose(tt_i, tt_i[0]) or np.allclose(tt_j, tt_j[0]):
                                rho = 0.0
                            else:
                                rho = float(np.corrcoef(tt_i, tt_j)[0, 1])
                                if not np.isfinite(rho):
                                    rho = 0.0
                            corr_accum[(ei, ej)] = corr_accum.get((ei, ej), 0.0) + rho

        first_pairs = set()
        if exclude_first_order is not None and exclude_first_order.numel() > 0:
            r = exclude_first_order[0].cpu().tolist()
            c = exclude_first_order[1].cpu().tolist()
            first_pairs = set(zip(r, c))

        by_src: Dict[int, List[Tuple[int, float]]] = {}
        for (src, dst), sc in corr_accum.items():
            if (src, dst) in first_pairs:
                continue
            by_src.setdefault(src, []).append((dst, sc))

        srcs, dsts = [], []
        for src, items in by_src.items():
            items.sort(key=lambda t: t[1], reverse=True)
            taken, seen = 0, set()
            for dst, sc in items:
                if dst == src or dst in seen:
                    continue
                srcs.append(src); dsts.append(dst); seen.add(dst); taken += 1
                if taken >= top_k:
                    break
        return torch.tensor([srcs, dsts], dtype=torch.long)

    def _dwa_weights():
        """
        [2026-05-01 Stage 5] Returns 3-tuple (w_speed, w_link, w_route).

        Old 2-way version (kept as comment for reference):
            if len(_dwa_hist_link) < 2 or len(_dwa_hist_route) < 2:
                return 0.5, 0.5
            r1 = float(_dwa_hist_link [-1] / (_dwa_hist_link [-2] + 1e-8))
            r2 = float(_dwa_hist_route[-1] / (_dwa_hist_route[-2] + 1e-8))
            w1 = math.exp(r1 / DWA_T); w2 = math.exp(r2 / DWA_T)
            s = w1 + w2
            return (w1 / s, w2 / s)
        """
        if (len(_dwa_hist_speed) < 2 or
            len(_dwa_hist_link)  < 2 or
            len(_dwa_hist_route) < 2):
            # Warm-up: uniform weights until two epochs of history exist
            return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)

        r_speed = float(_dwa_hist_speed[-1] / (_dwa_hist_speed[-2] + 1e-8))
        r_link  = float(_dwa_hist_link [-1] / (_dwa_hist_link [-2] + 1e-8))
        r_route = float(_dwa_hist_route[-1] / (_dwa_hist_route[-2] + 1e-8))
        w_speed = math.exp(r_speed / DWA_T)
        w_link  = math.exp(r_link  / DWA_T)
        w_route = math.exp(r_route / DWA_T)
        s = w_speed + w_link + w_route
        return (w_speed / s, w_link / s, w_route / s)

    CFG = dict(
        way_id_col_df="way_id",
        time_col_df="departure_time",
        way_id_col_edges_candidates=["way_id", "id", "edge_id", "osmid"],
        length_col_edges="length",
        speed_is_kmh=False,
        slots_per_hour=12,
        dyn_prefixes=[
            "median_speed", "mean_speed", "min_speed",
            "max_speed", "count_vehicle"
        ],
        static_cols=[
            "length", "lanes", "width_m", "speed_limit_ms",
            "hw_living_street", "hw_motorway", "hw_motorway_link", "hw_pedestrian",
            "hw_primary", "hw_primary_link", "hw_residential", "hw_secondary",
            "hw_secondary_link", "hw_service", "hw_tertiary", "hw_tertiary_link",
            "has_traffic_signal", "link_len_m",
        ],
        time_cols=[
            "day_of_week", "month", "day_of_month", "day_of_year",
            "tod_sin", "tod_cos", "dow_sin", "dow_cos",
        ],
        hidden_dim=32,
        heads=8,
        num_layers=1,
        max_route_len=1052,
        high_order_topk=5,
        speed_metric_index=0,
        K_first=5,
        K_conflict=5,
        K_high=5,
        max_nodes_local=360,
    )
    # Publish CFG to module globals so module-level classes (EdgeDynStore,
    # TripDatasetLinkETA, LinkETA) — which read CFG by global lookup — can find
    # it. Mirrors LinkETA_Fixed's original module-level state exactly.
    globals()["CFG"] = CFG
    set_seed(42)
    print("CFG keys:", list(CFG.keys()))
    print("Random seeds set. CUDA available:", torch.cuda.is_available())
    edges_local = ensure_edge_idx_and_uv(edges_gdf)
    edges_id_col = guess_edges_id_col(edges_local, CFG["way_id_col_edges_candidates"])
    assert edges_id_col is not None, "No id column found in edges_gdf."
    edges_local = edges_local.copy()
    edges_local[edges_id_col] = edges_local[edges_id_col].astype(str)
    df_all_local = df_all.copy()
    df_all_local[CFG["way_id_col_df"]] = df_all_local[CFG["way_id_col_df"]].astype(str)
    edges_local, STATIC_FEAT_COLS, df_all_local = ensure_static_feature_columns(
        edges_local, df_all_local,
        CFG["static_cols"], edges_id_col, CFG["way_id_col_df"]
    )
    CFG["static_cols"] = STATIC_FEAT_COLS
    id2edge = dict(zip(edges_local[edges_id_col], edges_local["edge_idx"]))
    df_all_local["edge_idx"] = df_all_local[CFG["way_id_col_df"]].map(id2edge)
    df_all_local = df_all_local.dropna(subset=["edge_idx"])
    df_all_local["edge_idx"] = df_all_local["edge_idx"].astype(int)
    print("edges_local shape:", edges_local.shape)
    print("df_all_local shape:", df_all_local.shape)
    print("Static feature columns on edges:", STATIC_FEAT_COLS[:10], "...")
    print("Unique edges in df_all_local:", df_all_local["edge_idx"].nunique())
    df_all_local = add_time_features(df_all_local)
    if 'link_len_m' not in df_all_local.columns:
        df_all_local['link_len_m'] = np.nan
    df_all_local['length'] = df_all_local.get('length', np.nan)
    df_all_local['link_len_m'] = df_all_local['link_len_m'].fillna(df_all_local['length']).fillna(5.0)
    df_all_local.loc[df_all_local['link_len_m'] <= 0, 'link_len_m'] = 5.0
    print("Temporal features added:", [c for c in ['tod_sin','tod_cos','dow_sin','dow_cos'] if c in df_all_local.columns])
    print("df_all_local rows after dropping invalid times:", len(df_all_local))
    print("link_len_m stats:", df_all_local['link_len_m'].describe())
    inst, INST_FEAT_COLS, STATIC_FEAT_COLS, TIME_FEAT_COLS, DYN_FEAT_COLS = build_instance_features(df_all_local)
    print("Instance table shape:", inst.shape)
    print("Number of instance features:", len(INST_FEAT_COLS))
    print("Example INST_FEAT_COLS[:10]:", INST_FEAT_COLS[:10])
    df_train, df_val, df_test = temporal_split_by_trip(df_all_local)
    SCALER_MEAN, SCALER_STD = fit_scaler(inst, df_train, INST_FEAT_COLS)
    DYN_IDX = dyn_indices(INST_FEAT_COLS)
    globals()["DYN_IDX"] = DYN_IDX  # LinkETA._encode_dyn_slots reads via global lookup
    split_stats(df_train, "Train")
    split_stats(df_val, "Val")
    split_stats(df_test, "Test")
    print("Scaler mean shape:", SCALER_MEAN.shape)
    print("Scaler std shape:", SCALER_STD.shape)
    SLOT_PREFIXES = CFG["dyn_prefixes"]
    s = df_train[['edge_idx', 'link_len_m', 'link_travel_time_s']].copy()
    s['link_len_m'] = pd.to_numeric(s['link_len_m'], errors='coerce')
    s['link_travel_time_s'] = pd.to_numeric(s['link_travel_time_s'], errors='coerce')
    s = s.dropna()
    s = s[s['link_travel_time_s'] > 0.1]
    s['v_obs'] = s['link_len_m'] / s['link_travel_time_s']
    v95 = s.groupby('edge_idx')['v_obs'].quantile(0.95).clip(lower=5.0, upper=35.0)
    eg = edges_local.copy()
    eg = eg.drop_duplicates(subset=['edge_idx']).set_index('edge_idx', drop=False)
    if 'speed_limit_ms' in eg.columns:
        sl = pd.to_numeric(eg['speed_limit_ms'], errors='coerce')
        sl = sl.where((sl >= 5.0) & (sl <= 40.0))
        eg['v_ff'] = sl
    else:
        eg['v_ff'] = np.nan
    eg['v_ff'] = eg['v_ff'].fillna(v95).fillna(10.0).clip(lower=5.0, upper=35.0)
    edges_vff = eg.reset_index(drop=True)
    edges_sorted = edges_vff.sort_values('edge_idx')
    vff_vec = torch.tensor(edges_sorted['v_ff'].to_numpy(), dtype=torch.float32)
    rels = build_first_order_relations_with_uturn(edges_vff)
    edge_index_up = rels["upstream"]
    edge_index_down = rels["downstream"]
    edge_index_lt = rels["left_turn"]
    edge_index_rt = rels["right_turn"]
    edge_index_op = rels["opposite"]
    edge_index_ut = rels["u_turn"]
    first_order_all = torch.cat(
        [edge_index_up, edge_index_down, edge_index_lt, edge_index_rt, edge_index_op],
        dim=1
    )
    edge_index_high = build_high_order_edge_index_dueta(
        df_all=df_train,
        edges_gdf=edges_vff,
        top_k=CFG["high_order_topk"],
        way_id_col_df=CFG["way_id_col_df"],
        way_id_col_edges=edges_id_col,
        length_col=CFG["length_col_edges"] if CFG["length_col_edges"] in edges_vff.columns else "length",
        slots_per_hour=CFG["slots_per_hour"],
        speed_metric_index=CFG["speed_metric_index"],
        speed_is_kmh=CFG["speed_is_kmh"],
        exclude_first_order=first_order_all,
    )
    N_edges = int(edges_vff["edge_idx"].max()) + 1
    self_edge_index = torch.tensor([list(range(N_edges)), list(range(N_edges))], dtype=torch.long)
    rel_edges = [
        self_edge_index,
        edge_index_down,
        edge_index_up,
        edge_index_lt,
        edge_index_op,
        edge_index_rt,
        edge_index_high,
        edge_index_ut,
    ]
    for name in ["self", "downstream", "upstream", "left_turn", "opposite", "right_turn", "high_order", "u_turn"]:
        if name == "self":
            rel_info(name, self_edge_index)
        elif name == "downstream":
            rel_info(name, edge_index_down)
        elif name == "upstream":
            rel_info(name, edge_index_up)
        elif name == "left_turn":
            rel_info(name, edge_index_lt)
        elif name == "right_turn":
            rel_info(name, edge_index_rt)
        elif name == "opposite":
            rel_info(name, edge_index_op)
        elif name == "high_order":
            rel_info(name, edge_index_high)
        elif name == "u_turn":
            rel_info(name, edge_index_ut)
    print("v_ff stats:", edges_vff["v_ff"].describe())
    dyn_12x5_cols = select_dynamic_12x5_columns(df_all_local)
    DYN_STORE = EdgeDynStore(
        df_all_local, dyn_12x5_cols, edges_vff,
        way_id_col_df=CFG["way_id_col_df"],
        way_id_col_edges=edges_id_col,
        time_col=CFG["time_col_df"]
    )
    print("Number of edges with dynamic data:", len(DYN_STORE.idx))
    if DYN_STORE.idx:
        sample_edge = next(iter(DYN_STORE.idx.keys()))
        t64, arr = DYN_STORE.idx[sample_edge]
        print(f"Sample edge_idx={sample_edge}, num time points={len(t64)}, block shape={arr.shape}")
    H_SLICES, BIN_SECONDS = 12, 5 * 60
    globals()["H_SLICES"] = H_SLICES        # TripDatasetLinkETA reads via global lookup
    globals()["BIN_SECONDS"] = BIN_SECONDS
    L_MIN, T_MIN = 10.0, 1.5
    train_ds = TripDatasetLinkETA(
        df_train, edges_vff, inst,
        rel_edges,
        static_feat_cols=STATIC_FEAT_COLS,
        time_feat_cols=TIME_FEAT_COLS,
        dyn_feat_cols=DYN_FEAT_COLS,
        dyn_store=DYN_STORE,
        K_first=CFG["K_first"],
        K_conflict=CFG["K_conflict"],
        K_high=CFG["K_high"],
    )
    val_ds = TripDatasetLinkETA(
        df_val, edges_vff, inst,
        rel_edges,
        static_feat_cols=STATIC_FEAT_COLS,
        time_feat_cols=TIME_FEAT_COLS,
        dyn_feat_cols=DYN_FEAT_COLS,
        dyn_store=DYN_STORE,
        K_first=CFG["K_first"],
        K_conflict=CFG["K_conflict"],
        K_high=CFG["K_high"],
    )
    test_ds = TripDatasetLinkETA(
        df_test, edges_vff, inst,
        rel_edges,
        static_feat_cols=STATIC_FEAT_COLS,
        time_feat_cols=TIME_FEAT_COLS,
        dyn_feat_cols=DYN_FEAT_COLS,
        dyn_store=DYN_STORE,
        K_first=CFG["K_first"],
        K_conflict=CFG["K_conflict"],
        K_high=CFG["K_high"],
    )
    print("Num trips in train_ds:", len(train_ds))
    print("Num trips in val_ds:", len(val_ds))
    print("Num trips in test_ds:", len(test_ds))
    if len(train_ds) > 0:
        g0 = train_ds[0]
        print("\nExample group (first train trip):")
        print("  trip_id:", g0["trip_id"])
        print("  total nodes (route + neighbours):", g0["node_edge_idx"].shape[0])
        print("  route length (L_on):", g0["route_seq_idx"].shape[0])

        # ---- NEW: detailed table of nodes & features ----
        node_edge_idx = g0["node_edge_idx"].cpu().numpy()
        route_flag = g0["route_flag"].cpu().numpy()
        neigh_parent = g0["neigh_parent"].cpu().numpy()
        x_seq = g0["x_seq"].cpu().numpy()  # shape [M, F]

        # Build a DataFrame with features
        max_nodes_show = 30  # so it doesn't spam your console
        M = min(len(node_edge_idx), max_nodes_show)

        df_nodes = pd.DataFrame(x_seq[:M, :], columns=INST_FEAT_COLS)
        df_nodes.insert(0, "node_idx", np.arange(M))
        df_nodes.insert(1, "edge_idx", node_edge_idx[:M])
        df_nodes.insert(2, "is_route", route_flag[:M])
        df_nodes.insert(3, "route_parent_pos", neigh_parent[:M])

        print("\nFirst", M, "nodes (route + neighbours) with features:")
        try:
            from IPython.display import display
            display(df_nodes)
        except Exception:
            print(df_nodes)

        # Optional: separate on-route vs off-route views
        df_on = df_nodes[df_nodes["is_route"]].copy()
        df_off = df_nodes[~df_nodes["is_route"]].copy()

        print("\nOn-route nodes (subset):")
        try:
            display(df_on.head(10))
        except Exception:
            print(df_on.head(10))

        print("\nOff-route neighbour nodes (subset):")
        try:
            display(df_off.head(10))
        except Exception:
            print(df_off.head(10))
    if len(train_ds) > 0:
        inspect_offroute_neighbours_for_trip(train_ds, idx=0, max_k=10)
        if len(train_ds) > 1:
            inspect_offroute_neighbours_for_trip(train_ds, idx=1, max_k=10)
    _dwa_hist_speed = deque(maxlen=2)   # [2026-05-01 Stage 5] NEW
    _dwa_hist_link  = deque(maxlen=2)
    _dwa_hist_route = deque(maxlen=2)
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)
    print("DWA temperature:", DWA_T)
    print("AMP enabled:", USE_AMP)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vff_vec = vff_vec.to(device)
    IN_DIM = len(INST_FEAT_COLS)
    max_route_len = int(max(_max_L(train_ds), _max_L(val_ds), _max_L(test_ds)) + 16)
    print("IN_DIM:", IN_DIM)
    print("max_route_len:", max_route_len)
    IN_DIM = inst.shape[1]
    model = LinkETA(
        in_dim=IN_DIM,
        hidden_dim=CFG["hidden_dim"],
        heads=CFG["heads"],
        relations=len(rel_edges),
        num_layers=CFG["num_layers"],
        max_route_len=max_route_len,
        num_experts=3,
        dropout=0.3,
        input_dropout=0.2,
    ).to(device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mu_link_tt = float(pd.to_numeric(df_train["link_travel_time_s"], errors="coerce").dropna().mean())
    if not np.isfinite(mu_link_tt) or mu_link_tt <= 0:
        mu_link_tt = 10.0  # kept only for potential future use
    train_loader = DataLoader(
        train_ds, batch_size=16, shuffle=True,
        collate_fn=collate_trips_linketa, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=16, shuffle=False,
        collate_fn=collate_trips_linketa, num_workers=0
    )
    test_loader = DataLoader(
        test_ds, batch_size=16, shuffle=False,
        collate_fn=collate_trips_linketa, num_workers=0
    )
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    SCALER_MEAN = SCALER_MEAN.to(device)
    SCALER_STD = SCALER_STD.to(device)
    DYN_IDX = DYN_IDX.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=3e-6
    )
    HIST = {
        "train": {
            "speed_loss": [],                                    # [2026-05-01 Stage 5] new
            "route_loss": [], "link_loss": [],
            "route_mape": [], "link_mape": [],
            "speed_mae": [], "speed_mae_actual": [],             # [2026-05-01 Diagnostic] both targets
            "vol_mae": []
        },
        "val": {
            "speed_loss": [],                                    # [2026-05-01 Stage 5] new
            "route_loss": [], "link_loss": [],
            "route_mape": [], "link_mape": [],
            "speed_mae_actual": [],                              # [2026-05-01 Diagnostic] (extra key here only)
            "speed_mae": [], "vol_mae": []
        }
    }
    print("Device:", device)
    print("Model parameters:", sum(p.numel() for p in model.parameters()) / 1e6, "M")
    print("Train batches:", len(train_loader),
          "Val batches:", len(val_loader),
          "Test batches:", len(test_loader))
    print("x_seq_nodes dim:", IN_DIM)
    print("input_proj in_features:", model.input_proj.in_features)
    ckpt = CheckpointManager(dirpath=CKPT_DIRNAME, max_keep=3)

    # ---- assemble core dict --------------------------------------------
    # ---- nested helpers (capture state above via closure) -----------

    def one_epoch(loader, epoch=0, train=True,
                  w_speed=1/3, w_link=1/3, w_route=1/3):
        # [2026-05-01 Stage 5 OLD signature]
        # def one_epoch(loader, epoch=0, train=True, w_link=0.5, w_route=0.5):
        # -> expanded to take w_speed too so DWA can balance all three primary losses.
        model.train() if train else model.eval()

        sum_speed_loss = 0.0     # [2026-05-01 Stage 5] track loss_speed for DWA history
        sum_link_loss = sum_route_loss = 0.0
        sum_mae_link = sum_mape_link = 0.0
        sum_sse_link = 0.0
        N_links = N_links_mape = 0

        sum_mae_route = sum_mape_route = 0.0
        sum_sse_route = 0.0
        N_routes_loss = N_routes_metrics = N_routes_mape = 0

        sum_speed_mae = sum_vol_mae = 0.0
        N_speed = N_vol = 0
        # [2026-05-01 Diagnostic] track speed MAE against the actual training target
        # (sp_actual), not the historical proxy. The legacy sum_speed_mae is kept
        # alongside so we can see both numbers per epoch.
        sum_speed_mae_actual = 0.0
        N_speed_actual = 0

        for batch in loader:
            # device move
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            # ----------------------------------------------------------------
            # Batch tensors (see TERMINOLOGY NOTE at top of file):
            #   node_edge_idx   [B, M]        - edge index for each position in local graph
            #   x_seq_nodes     [B, M, in_dim]- feature vectors for all positions (route + neighbors)
            #   route_flag_nodes[B, M]        - True=on-route, False=neighbor
            #   hop_idx         [B, M]        - BFS hop distance from route start
            #   masks_per_rel   list[R] of [B,M,M] - boolean adjacency masks per relation
            #   route_seq_idx   [B, L]        - indices into M positions for route sequence
            #   seq_true_tt     [B, L]        - ground-truth link travel time (seconds)
            #   seq_link_len    [B, L]        - traversed length (meters) per route link
            #   seq_mask        [B, L]        - boolean mask for valid route positions
            #   trip_tt         [B]           - ground-truth total trip travel time (seconds)
            # ----------------------------------------------------------------
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

            # Per-position speed/volume features -> gathered to route sequence
            #   sp_true_node [B, M] - observed speed ratio (speed/vff) per position [HISTORICAL PROXY]
            #   lc_true_node [B, M] - log-count of vehicles per position
            #   sp_w_node    [B, M] - weight/confidence for speed observation
            # [2026-05-01 Stage 2] sp_true_node was previously the speed-loss target.
            # We still gather it (kept available for diagnostics), but the speed-loss
            # target is now sp_actual computed from the (corrected) per-trip labels —
            # see the new sp_actual / sp_w_actual block below.
            sp_true_node = torch.nan_to_num(batch["sp_ratio_agg"], nan=0.0)
            lc_true_node = torch.nan_to_num(batch["logcnt1h_node"], nan=0.0)
            sp_w_node = torch.nan_to_num(batch["speed_w_node"], nan=0.0)

            gather_for_seq = lambda node_vals: torch.gather(
                node_vals, 1, route_seq_idx
            )

            # Gather to route sequence: [B, M] -> [B, L]
            sp_true = gather_for_seq(sp_true_node)     # [B,L] HISTORICAL proxy (no longer used as loss target)
            lc_true = gather_for_seq(lc_true_node)     # [B,L] log-count at route positions
            sp_w = gather_for_seq(sp_w_node)           # [B,L] HISTORICAL weight (no longer used as loss weight)

            # vff_seq_b [B, L] - free-flow speed (m/s) for each link in route
            vff_seq_b = vff_vec[node_edge_idx]         # [B, M] vff for all positions
            vff_seq_b = torch.gather(vff_seq_b, 1, route_seq_idx)  # [B, L] gather route only
            # [2026-05-01 Stage 2.1] Keep a CLEAN copy of v_ff (pre-noise) so we can
            # detect rows that were pinned at the r_max floor by Stage 1 redistribution.
            # Pinned rows satisfy link_tt = link_len / (r_max · v_ff_clean) exactly,
            # so sp_actual evaluated against vff_seq_clean equals R_MAX_TARGET on the
            # nose — these rows must be excluded from the speed loss because the
            # features cannot disambiguate them from clean rows and their target is
            # synthetic (a redistribution boundary, not a measurement).
            vff_seq_clean = vff_seq_b.clone()
            if train:
                rng = vff_noise_range(epoch)
                if rng > 0:
                    noise = 1.0 + rng * (2.0 * torch.rand_like(vff_seq_b) - 1.0)
                    vff_seq_b = vff_seq_b * noise

            # ----------------------------------------------------------------
            # [2026-05-01 Stage 2] Per-link sp_actual TARGET, derived from the
            # (already-redistributed) per-link labels. After Stage 1 every row's
            # sp_actual <= r_max by construction, so the bounded sp_hat can fit
            # it exactly. This replaces the historical-proxy target above.
            #     sp_actual    = link_len_m / (link_tt · v_ff)        # ratio
            #     sp_w_actual  = trust mask: 1 where label is physically credible
            # ----------------------------------------------------------------
            R_MAX_TARGET = 3.0   # must match the r_max used in prepare_corrected_dataset.py
            sp_actual = (seq_link_len / seq_true_tt.clamp_min(1e-3)) / vff_seq_b.clamp_min(1.0)
            sp_actual = sp_actual.clamp(0.0, R_MAX_TARGET)

            # [2026-05-01 Stage 2.1] Detect rows pinned at the floor by Stage 1.
            # Use vff_seq_clean (pre-noise) so the pinned condition is exact:
            # sp_actual_clean == R_MAX_TARGET to within float epsilon.
            sp_actual_clean = (
                seq_link_len / seq_true_tt.clamp_min(1e-3)
            ) / vff_seq_clean.clamp_min(1.0)
            was_pinned = sp_actual_clean >= (R_MAX_TARGET - 1e-3)

            sp_w_actual = (
                seq_mask
                & (seq_true_tt  >= 1.0)
                & (seq_link_len >= 10.0)
                & (~was_pinned)        # exclude Stage-1 pinned rows from speed loss
            ).float()

            # normalize x_seq_nodes
            x_seq_norm = (x_seq_nodes - SCALER_MEAN.view(1, 1, -1)) / SCALER_STD.view(1, 1, -1)

            # masks: seq_mask = all valid links; sp_mask = legacy mask (kept for compatibility / diagnostics)
            sp_mask = (sp_w > 0) & seq_mask

            Bsz = seq_true_tt.size(0)

            with torch.amp.autocast('cuda', enabled=USE_AMP):
                # Model returns 7 values: link_eta_seq, trip_eta, sp_hat, logcnt_1h, g, delta, eta_phys
                link_eta_seq, trip_eta, sp_pred, lc_pred, g, delta, eta_phys_seq = model(
                    x_seq_norm,
                    hop_idx,
                    route_flag_nodes,
                    masks_per_rel,
                    route_seq_idx,
                    seq_link_len,
                    vff_seq_b,
                    seq_mask,
                    # [2026-04-28 A1] r_max 3.0 -> 2.0
                    # [2026-05-01 Stage 2/3] r_max 2.0 -> 3.0 to match SF-derived
                    # statistical bound + the r_max used in Stage 1 redistribution.
                    # Now consistent: data is clipped at sp_actual <= 3, model
                    # output is bounded at sp_hat <= 3, target sp_actual <= 3.
                    r_max=R_MAX_TARGET,
                    return_parts=True,
                )

                link_eta_seq = torch.nan_to_num(link_eta_seq, nan=0.0)
                trip_eta = torch.nan_to_num(trip_eta, nan=0.0)
                sp_pred = torch.nan_to_num(sp_pred, nan=0.0)
                lc_pred = torch.nan_to_num(lc_pred, nan=0.0)

                # [2026-04-28 A1+A2] Funnel model output through the seconds-helper.
                # After A1 this is a no-op, but it keeps a single canonical path so
                # any future swap to e.g. log1p-target heads only changes one place.
                pred_seq = link_eta_in_seconds(link_eta_seq, seq_link_len)
                seq_true = seq_true_tt.clamp_min(0.0)  # [2026-04-28 A2] guard for log1p (a few rows have negative TT)

                # ---------- main link & route losses (use ALL valid links) ----------
                # [2026-04-28 A2] delta=1.0 -> 0.3 (Huber operates in log1p space now)
                # [2026-05-01 Step 6 OLD — replaced for long-trip-tail fix:
                # loss_link  = link_loss_huber_macro(pred_seq, seq_true, seq_mask, delta=0.3)
                # loss_route = route_loss_mape(pred_seq, trip_tt, seq_mask)
                # ]
                # New training losses:
                #   - link: length-weighted log1p-Huber so long links drive the gradient
                #     (matches what dominates Route MAE).
                #   - route: log1p-Huber so long-trip outliers no longer get a vanishing
                #     1/trip_tt MAPE gradient.
                loss_link  = link_loss_huber_log_lenweighted(
                    pred_seq, seq_true, seq_mask,
                    link_len=seq_link_len,
                    delta=0.3,
                )
                loss_route = route_loss_huber_log(pred_seq, trip_tt, seq_mask, delta=0.3)

                # ---------- speed loss : PRIMARY supervision (Stage 2 target swap) ----------
                # [2026-05-01 Stage 2] Target swapped from sp_ratio_agg (historical proxy)
                # to sp_actual (per-link actual ratio derived from corrected labels).
                # Weight swapped from sp_w (historical confidence) to sp_w_actual (trust mask).
                # SmoothL1 in ratio-space with beta=0.1 (≈10 % of free-flow as transition point).
                denom = sp_w_actual.sum().clamp_min(1.0)
                per = F.smooth_l1_loss(sp_pred, sp_actual, reduction='none', beta=0.1)
                loss_speed = (per * sp_w_actual).sum() / denom

                # [2026-05-01 Stage 2 OLD — historical-proxy target, DISABLED]
                # if sp_mask.any():
                #     per = F.smooth_l1_loss(sp_pred[sp_mask], sp_true[sp_mask], reduction='none')
                #     w = sp_w[sp_mask]
                #     loss_speed = (w * per).sum() / w.sum().clamp_min(1.0)
                # else:
                #     loss_speed = torch.zeros((), device=device)

                # ---------- volume auxiliary loss (use ALL valid links) ----------

                if seq_mask.any():
                    loss_vol = F.smooth_l1_loss(lc_pred[seq_mask], lc_true[seq_mask])
                else:
                    loss_vol = torch.zeros((), device=device)


                # [2026-04-28 A1] g calibration removed; this term is always 0 now.
                # ---------- calibration penalty on g (DISABLED) ----------
                # lam_g = lam_g_at(epoch)

                # ---------- final loss ----------
                # [2026-05-01 Stage 5] All three primary losses (speed, link, route)
                # are now under DWA. loss_vol stays a fixed-weight auxiliary.
                final_loss = (
                    w_speed * loss_speed +
                    w_link  * loss_link  +
                    w_route * loss_route +
                    0.10    * loss_vol
                )
                # [2026-04-28 A1 OLD] hardcoded weights for loss_speed (was 0.05, then 0.30):
                # final_loss = (
                #     w_link  * loss_link  +
                #     w_route * loss_route +
                #     0.30    * loss_speed +
                #     0.10    * loss_vol
                #     # + lam_g * (g - 1.0).abs().mean()   # [2026-04-28 A1] removed
                # )

            if train:
                optimizer.zero_grad(set_to_none=True)
                if USE_AMP:
                    scaler.scale(final_loss).backward()
                    if clip_norm is not None:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    final_loss.backward()
                    if clip_norm is not None:
                        nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                    optimizer.step()

            # ---------- metrics (using provided functions) ----------
            # link_metrics_micro uses seq_mask for MAE/RMSE but internally filters for MAPE
            link_stats = link_metrics_micro(pred_seq, seq_true, seq_mask,
                                            link_len=seq_link_len,
                                            L_min=L_MIN_METRIC,
                                            T_min=T_MIN_METRIC)
            route_stats = route_metrics_pooled(pred_seq, trip_tt, seq_mask)

            sum_speed_loss += float(loss_speed.item()) * Bsz   # [2026-05-01 Stage 5] new
            sum_link_loss  += float(loss_link.item()) * Bsz
            sum_route_loss += float(loss_route.item()) * Bsz
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
                # [LEGACY metric — sp_pred vs HISTORICAL proxy sp_ratio_agg, kept for continuity]
                sp_abs = (sp_pred[sp_mask] - sp_true[sp_mask]).abs().mean().item()
                sum_speed_mae += sp_abs * int(sp_mask.sum().item())
                N_speed += int(sp_mask.sum().item())

            # [2026-05-01 Diagnostic] speed MAE against the ACTUAL loss target (sp_actual).
            # This is the number that tells you whether sp_hat is converging to the
            # supervision target. The legacy speed_mae above is against the historical
            # proxy and naturally non-zero — don't read it as "model not learning."
            sp_actual_mask = sp_w_actual.bool()
            if sp_actual_mask.any():
                sp_abs_actual = (sp_pred[sp_actual_mask] - sp_actual[sp_actual_mask]).abs().mean().item()
                n_act = int(sp_actual_mask.sum().item())
                sum_speed_mae_actual += sp_abs_actual * n_act
                N_speed_actual += n_act

            if seq_mask.any():
                vol_abs = (lc_pred[seq_mask] - lc_true[seq_mask]).abs().mean().item()
                n_valid = int(seq_mask.sum().item())
                sum_vol_mae += vol_abs * n_valid
                N_vol += n_valid

        return {
            'speed_loss': sum_speed_loss / max(N_routes_loss, 1),   # [2026-05-01 Stage 5] for DWA
            'link_loss':  sum_link_loss  / max(N_routes_loss, 1),
            'route_loss': sum_route_loss / max(N_routes_loss, 1),
            'link_mae':   sum_mae_link   / max(N_links, 1),
            'link_rmse':  math.sqrt(sum_sse_link / max(N_links, 1)) if N_links > 0 else float("nan"),
            'link_mape':  (sum_mape_link / max(N_links_mape, 1)) if N_links_mape > 0 else float('nan'),
            'route_mae':  sum_mae_route / max(N_routes_metrics, 1),
            'route_rmse': math.sqrt(sum_sse_route / max(N_routes_metrics, 1)) if N_routes_metrics > 0 else float("nan"),
            'route_mape': (sum_mape_route / max(N_routes_mape, 1)) if N_routes_mape > 0 else float('nan'),
            'speed_mae':         (sum_speed_mae        / max(N_speed,        1)) if N_speed        else float('nan'),
            'speed_mae_actual':  (sum_speed_mae_actual / max(N_speed_actual, 1)) if N_speed_actual else float('nan'),  # [2026-05-01 Diagnostic]
            'vol_mae':    (sum_vol_mae / max(N_vol, 1)) if N_vol else float('nan'),
        }


    def eval_linketa(core,
                     ckpt_path: str = None,
                     label: str = "LinkETA",
                     split: str = "test"):
        """
        Evaluation loop for LinkETA model (Fixed version).
        
        Model forward() returns 7 values:
            (link_eta_seq, trip_eta, sp_hat, logcnt_1h, g, delta, eta_phys)
        
        Args:
            core: Dictionary containing model, device, loaders, scalers, etc.
            ckpt_path: Optional path to checkpoint file
            label: Label for output DataFrames
            split: Which data split to evaluate ("train", "val", or "test")
        
        Returns:
            metrics: Dictionary with link/route losses, MAE, RMSE, MAPE, speed/vol MAE
            trip_df: DataFrame with per-trip predictions
            link_df: DataFrame with per-link predictions
        """
        device = core["device"]
        model  = core["model"].to(device)

        # pick loader
        if "loaders" in core and split in core["loaders"]:
            loader = core["loaders"][split]
        else:
            key = f"{split}_loader"
            if key in core:
                loader = core[key]
            else:
                raise KeyError(f"No DataLoader found for split='{split}' in core.")

        # v_ff vector and feature scaler
        vff_vec_local = core.get("graph", {}).get("vff_vec", vff_vec).to(device)
        scaler_mean   = core.get("scalers", {}).get("mean", SCALER_MEAN).to(device)
        scaler_std    = core.get("scalers", {}).get("std", SCALER_STD).to(device)

        # optional ckpt
        if ckpt_path is not None:
            pkg = torch.load(ckpt_path, map_location=device)
            state = pkg["model"] if isinstance(pkg, dict) and "model" in pkg else pkg
            model.load_state_dict(state, strict=True)

        model.eval()

        # metric accumulator (link+route)
        agg = _init_metric_accumulator()

        # extras: speed/volume
        # [2026-05-01 Stage 2.1] We now track BOTH the historical-proxy speed MAE
        # (Speed[hist], vs sp_ratio_agg) AND the Stage-2 actual-target speed MAE
        # (Speed[actual], vs sp_actual). The actual-target metric matches the
        # training loss target and excludes Stage-1-pinned rows.
        sum_speed_mae = sum_vol_mae = 0.0
        N_speed = N_vol = 0
        sum_speed_mae_actual = 0.0
        N_speed_actual = 0

        trip_rows = []
        link_rows = []

        for batch in loader:
            # move tensors
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            node_edge_idx    = batch["node_edge_idx"]       # [B,M]
            x_seq_nodes      = batch["x_seq"]               # [B,M,F]
            route_flag_nodes = batch["route_flag"].bool()   # [B,M]
            hop_idx          = batch["hop_idx"].long()      # [B,M]
            masks_per_rel    = [m.to(device) for m in batch["masks_per_rel"]]

            route_seq_idx = batch["route_seq_idx"].long()   # [B,L]
            seq_true_tt   = torch.nan_to_num(batch["seq_true_tt"], nan=0.0)      # [B,L]
            seq_link_len  = torch.nan_to_num(batch["seq_link_len"], nan=0.0)     # [B,L]
            seq_link_len  = seq_link_len.clamp_min(1e-3)
            seq_mask      = batch["seq_mask"].bool()        # [B,L]
            trip_tt       = torch.nan_to_num(batch["trip_tt"], nan=0.0)          # [B]

            # per-node recency features -> sequence
            sp_true_node = torch.nan_to_num(batch["sp_ratio_agg"],   nan=0.0)    # [B,M]
            lc_true_node = torch.nan_to_num(batch["logcnt1h_node"],  nan=0.0)    # [B,M]
            sp_w_node    = torch.nan_to_num(batch["speed_w_node"],   nan=0.0)    # [B,M]

            gather_for_seq = lambda node_vals: torch.gather(
                node_vals, 1, route_seq_idx
            )

            sp_true = gather_for_seq(sp_true_node)   # [B,L]
            lc_true = gather_for_seq(lc_true_node)   # [B,L]
            sp_w    = gather_for_seq(sp_w_node)      # [B,L]

            # v_ff per node -> per-sequence
            vff_node = vff_vec_local[node_edge_idx]     # [B,M]
            vff_seq  = gather_for_seq(vff_node)         # [B,L]

            # normalize features
            x_seq_norm = (x_seq_nodes - scaler_mean.view(1, 1, -1)) / scaler_std.view(1, 1, -1)
            x_seq_norm = torch.nan_to_num(x_seq_norm)

            # forward - Model returns 7 values (no mix/eta_direct, no force_physics)
            with torch.amp.autocast('cuda', enabled=USE_AMP):
                (
                    link_eta_seq,   # [B,L]
                    trip_eta,       # [B]
                    sp_pred,        # [B,L]
                    lc_pred,        # [B,L]
                    g,              # [B]
                    delta,          # [B,L]
                    eta_phys_seq,   # [B,L]
                ) = model(
                    x_seq_norm,
                    hop_idx,
                    route_flag_nodes,
                    masks_per_rel,
                    route_seq_idx,
                    seq_link_len,
                    vff_seq,
                    seq_mask,
                    # [2026-04-28 A1] r_max 2.8 -> 2.0 (consistent with training)
                    # [2026-05-01 Stage 2/3] r_max 2.0 -> 3.0 (matches Stage 1 redistribution)
                    r_max=3.0,
                    return_parts=True,
                )

                link_eta_seq = torch.nan_to_num(link_eta_seq, nan=0.0)
                trip_eta     = torch.nan_to_num(trip_eta,     nan=0.0)
                sp_pred      = torch.nan_to_num(sp_pred,      nan=0.0)
                lc_pred      = torch.nan_to_num(lc_pred,      nan=0.0)

                # [2026-04-28 A1+A2] Same seconds-helper as training path.
                pred_seq = link_eta_in_seconds(link_eta_seq, seq_link_len)
                seq_true = seq_true_tt.clamp_min(0.0)

                # ---- canonical metrics / losses for this batch ----
                # [2026-04-28 A2] delta_huber 1.0 -> 0.3 (log-space)
                stats = compute_global_losses_and_metrics(
                    pred_seq=pred_seq,
                    trip_tt=trip_tt,
                    seq_true=seq_true,
                    seq_mask=seq_mask,
                    seq_link_len=seq_link_len,
                    delta_huber=0.3,
                )

            # accumulate link/route metrics
            _accumulate_batch_metrics(agg, stats)

            # ---------- speed / volume MAE ----------
            # Speed[hist] : legacy proxy target (sp_ratio_agg). Kept for continuity.
            sp_mask = (sp_w > 0) & seq_mask
            if sp_mask.any():
                sp_abs = (sp_pred[sp_mask] - sp_true[sp_mask]).abs().mean().item()
                n_sp   = int(sp_mask.sum().item())
                sum_speed_mae += sp_abs * n_sp
                N_speed       += n_sp

            # [2026-05-01 Stage 2.1] Speed[actual] : Stage-2 training target.
            # No noise injection in eval, so vff_seq is already clean — single pass.
            # Trust mask matches one_epoch: trip-credible link, len >= 10m, tt >= 1s,
            # and Stage-1 pinned rows excluded (synthetic targets).
            R_MAX_TARGET = 3.0
            sp_actual = (seq_link_len / seq_true_tt.clamp_min(1e-3)) / vff_seq.clamp_min(1.0)
            sp_actual = sp_actual.clamp(0.0, R_MAX_TARGET)
            was_pinned = sp_actual >= (R_MAX_TARGET - 1e-3)
            sp_w_actual = (
                seq_mask
                & (seq_true_tt  >= 1.0)
                & (seq_link_len >= 10.0)
                & (~was_pinned)
            )
            if sp_w_actual.any():
                sp_abs_act = (sp_pred[sp_w_actual] - sp_actual[sp_w_actual]).abs().mean().item()
                n_sp_act   = int(sp_w_actual.sum().item())
                sum_speed_mae_actual += sp_abs_act * n_sp_act
                N_speed_actual       += n_sp_act

            if seq_mask.any():
                vol_abs = (lc_pred[seq_mask] - lc_true[seq_mask]).abs().mean().item()
                n_vol   = int(seq_mask.sum().item())
                sum_vol_mae += vol_abs * n_vol
                N_vol       += n_vol

            # ---------- per-trip rows ----------
            route_true_from_links = (seq_true_tt * seq_mask.float()).sum(dim=1)  # [B]
            yR = torch.where(trip_tt > 0, trip_tt, route_true_from_links)
            pR = (pred_seq * seq_mask.float()).sum(dim=1)  # equals trip_eta by construction

            trip_ids     = batch["trip_ids"]
            # Handle trip_ids - may be list or tensor
            if isinstance(trip_ids, torch.Tensor):
                trip_ids_np = trip_ids.detach().cpu().numpy()
            else:
                trip_ids_np = np.array(trip_ids)
            y_true_np    = yR.detach().cpu().numpy()
            y_pred_np    = pR.detach().cpu().numpy()
            route_len_np = (seq_link_len * seq_mask.float()).sum(dim=1).detach().cpu().numpy()
            g_np         = g.detach().cpu().numpy()
            src_tag_np   = np.where(trip_tt.detach().cpu().numpy() > 0, "trip_tt", "sum_links")

            for i in range(len(trip_ids_np)):
                trip_rows.append((
                    label,
                    int(trip_ids_np[i]) if np.isscalar(trip_ids_np[i]) else int(trip_ids_np[i]),
                    float(y_true_np[i]),
                    float(y_pred_np[i]),
                    src_tag_np[i],
                    float(route_len_np[i]),
                    float(g_np[i]),
                ))

            # ---------- per-link rows (for BA, etc.) ----------
            pred_seq_cpu = pred_seq.detach().cpu()
            seq_true_cpu = seq_true_tt.detach().cpu()
            mask_cpu     = seq_mask.detach().cpu()

            B0, L0 = pred_seq_cpu.shape
            for b in range(B0):
                tid = int(trip_ids_np[b]) if np.isscalar(trip_ids_np[b]) else int(trip_ids_np[b])
                for l in range(L0):
                    if not bool(mask_cpu[b, l]):
                        continue
                    link_rows.append((
                        label,
                        tid,
                        float(seq_true_cpu[b, l].item()),
                        float(pred_seq_cpu[b, l].item()),
                    ))

        # ---- finalize metrics ----
        core_metrics = _finalize_metric_accumulator(agg)

        if N_speed > 0:
            speed_mae = sum_speed_mae / N_speed
        else:
            speed_mae = float("nan")

        if N_speed_actual > 0:
            speed_mae_actual = sum_speed_mae_actual / N_speed_actual
        else:
            speed_mae_actual = float("nan")

        if N_vol > 0:
            vol_mae = sum_vol_mae / N_vol
        else:
            vol_mae = float("nan")

        metrics = dict(core_metrics)
        metrics.update({
            "speed_mae":         speed_mae,           # Speed[hist]   - legacy proxy
            "speed_mae_actual":  speed_mae_actual,    # Speed[actual] - Stage-2 target
            "vol_mae":           vol_mae,
        })

        trip_df = pd.DataFrame(
            trip_rows,
            columns=["model", "TRIP_ID", "y_true", "y_pred",
                     "target_source", "route_len_m", "gain_g"],
        )
        link_df = pd.DataFrame(
            link_rows,
            columns=["model", "TRIP_ID", "true_tt_link", "pred_tt_link"],
        )

        return metrics, trip_df, link_df


    core = dict(
        device=device,
        model=model,
        loaders=dict(train=train_loader, val=val_loader, test=test_loader),
        datasets=dict(train=train_ds, val=val_ds, test=test_ds),
        inst_table=inst,
        inst_feat_cols=INST_FEAT_COLS,
        static_cols=STATIC_FEAT_COLS,
        time_cols=TIME_FEAT_COLS,
        dyn_cols=DYN_FEAT_COLS,
        graph=dict(
            edges_vff=edges_vff,
            rel_edges=rel_edges,
            vff_vec=vff_vec,
        ),
        dyn_store=DYN_STORE,
        cfg=CFG,
        constants=dict(
            H_SLICES=H_SLICES,
            BIN_SECONDS=BIN_SECONDS,
            L_MIN=L_MIN,
            T_MIN=T_MIN,
        ),
        scalers=dict(
            mean=SCALER_MEAN,
            std=SCALER_STD,
            dyn_idx=DYN_IDX,
        ),
        optimizer=optimizer,
        scheduler=scheduler,
        history=HIST,
        num_epochs=num_epochs,
        clip_norm=clip_norm,
        checkpoint_manager=ckpt,
        one_epoch=one_epoch,
        _dwa_weights=_dwa_weights,
        _dwa_hist_speed=_dwa_hist_speed,
        _dwa_hist_link=_dwa_hist_link,
        _dwa_hist_route=_dwa_hist_route,
        eval_linketa=eval_linketa,
    )
    return core
    core["df_all_local"] = df_all_local
    core["dyn_12x5_cols"] = dyn_12x5_cols


def run_training(core):
    """
    [MetaETA: TRAINING-ONLY — not used by the meta runtime path]
    --------------------------------------------------------------------------
    This is the standalone base-LinkETA trainer. MetaST / MetaTTE / MetaTraffic
    do NOT call it: they load a pretrained base via init_linketa() and adapt it.
    It is retained (not deleted) only so a base checkpoint can be (re)generated
    from scratch for a city that does not yet have one. Safe to ignore for the
    meta workflow. NOTE: it predates the stateless module-level one_epoch and
    uses core["one_epoch"] (the heavy builder's nested closure) — only valid for
    a core built WITHOUT a precomputed base.
    --------------------------------------------------------------------------
    Run LinkETA training using objects packaged in `core`.

    All model/optimizer/loader/ckpt state is read from `core` (built by
    linketa_get_core_components). No module globals are required.
    """
    # ---- unpack core into locals (replaces former `global` statements) ----
    model         = core["model"]
    optimizer     = core["optimizer"]
    scheduler     = core["scheduler"]
    train_loader  = core["loaders"]["train"]
    val_loader    = core["loaders"]["val"]
    test_loader   = core["loaders"]["test"]
    HIST          = core["history"]
    num_epochs    = core["num_epochs"]
    clip_norm     = core["clip_norm"]
    ckpt          = core["checkpoint_manager"]
    one_epoch         = core["one_epoch"]
    _dwa_weights      = core["_dwa_weights"]
    _dwa_hist_speed   = core["_dwa_hist_speed"]
    _dwa_hist_link    = core["_dwa_hist_link"]
    _dwa_hist_route   = core["_dwa_hist_route"]
    eval_linketa  = core["eval_linketa"]


    # NOTE: core is accepted for interface compatibility with other models,
    # but this training loop continues to use the already-constructed
    # global objects (model, loaders, optimizer, etc.), exactly as before.

    best_combined, best_state = float('inf'), None
    patience, since_best = 15, 0

    tr_last = va_last = None

    for epoch in range(num_epochs):
        # [2026-05-01 Stage 5] DWA now returns 3-tuple over (speed, link, route).
        wS, wL, wR = _dwa_weights()
        # [2026-05-01 Stage 5 OLD]  wL, wR = _dwa_weights()

        tr = one_epoch(train_loader, epoch=epoch, train=True,
                       w_speed=wS, w_link=wL, w_route=wR)
        va = one_epoch(val_loader,   epoch=epoch, train=False,
                       w_speed=wS, w_link=wL, w_route=wR)

        tr_last, va_last = tr, va

        # [2026-05-01 Stage 5] track speed_loss history alongside link/route
        _dwa_hist_speed.append(tr['speed_loss'])
        _dwa_hist_link .append(tr['link_loss'])
        _dwa_hist_route.append(tr['route_loss'])

        # [2026-05-01 Diagnostic] track both speed_mae (vs historical proxy) and
        # speed_mae_actual (vs the actual sp_actual loss target) per epoch.
        for k in ["speed_loss", "route_loss", "link_loss", "route_mape", "link_mape",
                  "speed_mae", "speed_mae_actual", "vol_mae"]:
            train_lst = HIST["train"].setdefault(k, [])
            val_lst   = HIST["val"  ].setdefault(k, [])
            train_lst.append(tr.get(k, float("nan")))
            val_lst.append  (va.get(k, float("nan")))

        comb = combined_metric(va)
        scheduler.step(comb)

        improved = comb < best_combined - 1e-6
        if improved:
            best_combined = comb
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            since_best = 0
            ckpt.save_best(
                model=model, optimizer=optimizer, scheduler=scheduler,
                epoch=epoch, step=global_step, best_metric=float(best_combined),
                extra={"val_metrics": va}
            )
        else:
            since_best += 1

        ckpt.save_last(
            model=model, optimizer=optimizer, scheduler=scheduler,
            epoch=epoch, step=global_step, best_metric=float(best_combined),
            extra={"val_metrics": va}
        )

        print(
            f"Epoch {epoch:02d} | "
            f"Train: Link {tr['link_loss']:.4f}/{tr['link_mae']:.3f} (MAPE={tr['link_mape']:.3f}) (RMSE={tr['link_rmse']:.3f})  "
            f"Route {tr['route_loss']:.4f}/{tr['route_mae']:.3f} (MAPE={tr['route_mape']:.3f}) (RMSE={tr['route_rmse']:.3f})  "
            f"Speed[actual]={tr['speed_mae_actual']:.4f} Speed[hist]={tr['speed_mae']:.4f} Vol={tr['vol_mae']:.4f}  ||  "
            f"Val: Link {va['link_loss']:.4f}/{va['link_mae']:.3f} (MAPE={va['link_mape']:.3f}) (RMSE={va['link_rmse']:.3f})  "
            f"Route {va['route_loss']:.4f}/{va['route_mae']:.3f} (MAPE={va['route_mape']:.3f}) (RMSE={va['route_rmse']:.3f})  "
            f"| DWA wS={wS:.3f} wL={wL:.3f} wR={wR:.3f}  | Combined={comb:.3f}"
        )
        
        
        if since_best >= patience:
            print("Early stopping triggered.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final test
    # [2026-05-01 Stage 5] 3-way DWA
    wS, wL, wR = _dwa_weights()
    te = one_epoch(test_loader, train=False,
                   w_speed=wS, w_link=wL, w_route=wR)
    

    print("— Final Test —")
    print(f"Link:  Huber={te['link_loss']:.4f} | MAE={te['link_mae']:.4f} | RMSE={te['link_rmse']:.4f} | MAPE={te['link_mape']:.4f}")
    print(f"Route: MAPE_Loss={te['route_loss']:.4f} | MAE={te['route_mae']:.4f} | RMSE={te['route_rmse']:.4f} | MAPE={te['route_mape']:.4f}")
    print(f"Traffic(1h): SpeedMAE={te['speed_mae']:.4f} | VolMAE={te['vol_mae']:.4f}")

    # Persist training history + final summary alongside the checkpoints.
    hist_path    = os.path.join(ckpt.dir, "training_history.json")
    summary_path = os.path.join(ckpt.dir, "training_summary.json")
    with open(hist_path, "w", encoding="utf-8") as _f:
        json.dump(HIST, _f, indent=2, default=str)
    with open(summary_path, "w", encoding="utf-8") as _f:
        json.dump({"test": te, "best_selector": float(best_combined),
                   "train_last": tr_last, "val_last": va_last}, _f, indent=2, default=str)
    print(f"[history] {hist_path}")
    print(f"[summary] {summary_path}")

    return {
        "train_last": tr_last,
        "val_last": va_last,
        "test": te,
        "best_selector": float(best_combined),
        "history": HIST,
        "best_state": best_state,
    }



# =============================================================================
# save_bundle - explicit, opt-in wrapper around the bundle save/load/verify
# block that used to auto-run at import in LinkETA_Fixed.py.
# =============================================================================
def save_bundle(core, out_dir="precomputed_linketa_bundle"):
    """
    [MetaETA: BUNDLE-BUILD ONLY — not used by the meta runtime path]
    The precomputed bundles already exist on disk under meta_learning_data/<city>/,
    so the meta workflow never calls this. Retained only to (re)generate a bundle
    for a new city. Safe to ignore for the meta workflow.
    --------------------------------------------------------------------------
    Save the LinkETA precomputed bundle for a built `core`.

    No-op unless explicitly called. Mirrors the body that previously ran
    automatically at import time at the bottom of LinkETA_Fixed.py.
    """
    cfg            = core["cfg"]
    edges_vff      = core["graph"]["edges_vff"]
    rel_edges      = core["graph"]["rel_edges"]
    vff_vec        = core["graph"]["vff_vec"]
    inst_feat_cols = core["inst_feat_cols"]
    static_cols    = core["static_cols"]
    time_cols      = core["time_cols"]
    dyn_cols       = core["dyn_cols"]
    SCALER_MEAN    = core["scalers"]["mean"]
    SCALER_STD     = core["scalers"]["std"]
    DYN_IDX        = core["scalers"]["dyn_idx"]
    DYN_STORE      = core["dyn_store"]
    df_all_local   = core.get("df_all_local")
    dyn_12x5_cols  = core.get("dyn_12x5_cols")
    if df_all_local is None or dyn_12x5_cols is None:
        raise RuntimeError("core lacks df_all_local / dyn_12x5_cols; rebuild via linketa_get_core_components.")

    traffic_profile_df = build_traffic_profile_df(df_all_local, CFG, dyn_12x5_cols)
    bundle_dir = save_linketa_bundle(
        "precomputed_linketa_bundle",
        edges_vff=edges_vff,
        rel_edges=rel_edges,
        vff_vec=vff_vec,
        cfg=CFG,
        inst_feat_cols=INST_FEAT_COLS,
        static_cols=STATIC_FEAT_COLS,
        time_cols=TIME_FEAT_COLS,
        dyn_cols=DYN_FEAT_COLS,
        scaler_mean=SCALER_MEAN,
        scaler_std=SCALER_STD,
        dyn_idx=DYN_IDX,
        traffic_profile_df=traffic_profile_df,
        save_dyn_store_idx=False,
        dyn_store_obj=DYN_STORE,
    )
    bundle = load_linketa_bundle(bundle_dir)
    verify_linketa_bundle_exact(
        bundle,
        edges_vff_ref=edges_vff,
        rel_edges_ref=rel_edges,
        vff_vec_ref=vff_vec,
        cfg_ref=CFG,
        inst_feat_cols_ref=INST_FEAT_COLS,
        static_cols_ref=STATIC_FEAT_COLS,
        time_cols_ref=TIME_FEAT_COLS,
        dyn_cols_ref=DYN_FEAT_COLS,
        scaler_mean_ref=SCALER_MEAN,
        scaler_std_ref=SCALER_STD,
        dyn_idx_ref=DYN_IDX,
        traffic_profile_ref=traffic_profile_df,
        strict_df=True,
    )
