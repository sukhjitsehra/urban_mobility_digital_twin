"""
MetaTraffic: Cross-City Traffic Profile Prediction via Relational GAT
=====================================================================
Generates per-link traffic profiles from static road attributes, temporal
features, and road-network graph structure.  The predicted traffic profiles
feed directly into LinkETA (or any TTE model) as the 60-dim dynamic input.

Design (v2.0 — per-trip traffic prediction, 2026-02-21):
  Phase 1 — encode_structure(static_feats, rel_coo):
    1. Z-normalize static features (registered buffers)    [C6]
    2. Project static_31 → D  (no time in GAT)             [C2]
    3. Relational GAT over full road graph → h_struct [N,D] [C1]
  Phase 2 — predict_traffic(h_struct, static, time, edge_idx):
    4. param_gen(h_struct[edges]) → per-link W_i [39,60], b_i [60]  [C3]
    5. predicted_traffic = W_i · [static_31, time_8] + b_i (60 dims) [C4]
  Training:
    6. Per-trip (day, hour) for ground-truth traffic lookup  [C5]
    7. Joint loss: L_TTE + λ · L_traffic_reconstruction     (unchanged)

Key properties:
  - Per-link weights (W_i) are *generated* by a shared network, not stored
  - Time-aware: time features enter the GAT → W_i changes with time-of-day
  - Model-agnostic: outputs a plain 60-dim tensor per link
  - Verifiable: traffic prediction has its own reconstruction loss
  - Transfers to unseen target cities via the shared generator weights

Parameters (approximate):
  - MetaTraffic encoder + generator: ~28K params
  - LinkETA (shared, unchanged): ~55K params
"""

# =============================================================================
# IMPORTS
# =============================================================================
import os
import math
import copy
import time
import random
import warnings
import hashlib
from typing import List, Optional, Dict, Tuple
from collections import deque

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import LinkETA as _linketa
from LinkETA import (
    LinkETA,
    collate_trips_linketa,
    TripDatasetLinkETA,
    lam_g_at,
    vff_noise_range,
    CFG,
    USE_AMP,
)
from loss_functions import (
    # [MetaETA] MetaTraffic (our method) uses the NEW log1p-space losses to match
    # the new LinkETA base: length-weighted log-Huber (link) + log-Huber (route).
    link_eta_in_seconds,
    link_loss_huber_log_lenweighted,
    route_loss_huber_log,
    link_metrics_micro,
    route_metrics_pooled,
    finalize_pct_metrics,
    L_MIN_METRIC,
    T_MIN_METRIC,
)
from checkpoint import CheckpointManager, write_training_history


# =============================================================================
# MODULE-LEVEL GLOBALS  (set by caller before training)
# =============================================================================
device: Optional[torch.device] = None
USE_AMP_MT = False
scaler_amp = None   # GradScaler


def _scaler_fingerprint(city_scaler_mean: Dict[str, torch.Tensor],
                        city_scaler_std: Dict[str, torch.Tensor]) -> str:
    """Hash scaler statistics to detect checkpoint/data mismatches."""
    parts = []
    for ck in sorted(city_scaler_mean.keys()):
        m = city_scaler_mean[ck].detach().cpu().float().numpy()
        s = city_scaler_std[ck].detach().cpu().float().numpy()
        parts.append(f"{ck}:{m[:5].tobytes().hex()}{s[:5].tobytes().hex()}")
    return hashlib.md5("|".join(parts).encode()).hexdigest()[:12]


# =============================================================================
# UTILITY: Scatter-softmax for sparse GAT (AMP-safe)
# =============================================================================
def _scatter_softmax(scores: torch.Tensor, index: torch.Tensor,
                     dim_size: int) -> torch.Tensor:
    """
    Softmax over groups defined by ``index``.
    All computations in float32 for AMP compatibility.
    """
    squeeze = False
    if scores.dim() == 1:
        scores = scores.unsqueeze(-1)
        squeeze = True

    orig_dtype = scores.dtype
    scores = scores.float()
    H = scores.shape[1]
    idx_exp = index.unsqueeze(-1).expand(-1, H)

    max_val = torch.full((dim_size, H), float("-inf"),
                         device=scores.device, dtype=torch.float32)
    max_val.scatter_reduce_(0, idx_exp, scores, reduce="amax",
                            include_self=True)
    scores_shifted = scores - max_val[index]
    exp_s = scores_shifted.exp()

    sum_exp = torch.zeros(dim_size, H, device=scores.device, dtype=torch.float32)
    sum_exp.scatter_add_(0, idx_exp, exp_s)

    result = exp_s / (sum_exp[index] + 1e-16)
    result = result.to(orig_dtype)
    if squeeze:
        result = result.squeeze(-1)
    return result


# =============================================================================
# UTILITY: Extract feature tensors from DataFrames
# =============================================================================
def extract_static_features_tensor(
    edges_gdf: pd.DataFrame,
    static_cols: list,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """[N_edges, F] tensor of static road attributes indexed by edge_idx."""
    n_edges = int(edges_gdf["edge_idx"].max()) + 1
    feats = torch.zeros(n_edges, len(static_cols), dtype=torch.float32, device=device)
    edge_idx = edges_gdf["edge_idx"].values
    for i, col in enumerate(static_cols):
        if col in edges_gdf.columns:
            vals = pd.to_numeric(edges_gdf[col], errors="coerce").fillna(0.0).values
            feats[edge_idx, i] = torch.tensor(vals, dtype=torch.float32, device=device)
    return feats


def extract_traffic_profile_tensor(
    traffic_profile: pd.DataFrame,
    edges_gdf: pd.DataFrame,
    dyn_prefixes: list,
    n_slots: int = 12,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Build a [N_edges, 7, 24, 60] tensor preserving day_of_week × hour_of_day
    temporal context from the traffic profile parquet.

    Each edge has 168 rows (7 days × 24 hours) in the parquet.  We place each
    row into feats[edge_idx, day_of_week, hour_of_day, :] so that the training
    loop can look up the correct time-slice as ground truth.
    """
    n_edges = int(edges_gdf["edge_idx"].max()) + 1
    n_feats = n_slots * len(dyn_prefixes)
    feats = torch.zeros(n_edges, 7, 24, n_feats, dtype=torch.float32, device=device)

    # OLD: grouped-by-prefix ordering (WRONG — mismatched DataLoader)
    # dyn_cols = []
    # for prefix in dyn_prefixes:
    #     for s in range(n_slots):
    #         dyn_cols.append(f"{prefix}_{s}")

    # NEW: interleaved-by-slot ordering (matches DL DYN_FEAT_COLS)
    dyn_cols = []
    for s in range(n_slots):
        for prefix in dyn_prefixes:
            dyn_cols.append(f"{prefix}_{s}")

    id_col = "way_id" if "way_id" in traffic_profile.columns else "id"
    eid_col = "way_id" if "way_id" in edges_gdf.columns else "id"

    present_cols = [c for c in dyn_cols if c in traffic_profile.columns]

    # Keep all rows — one per (edge, day_of_week, hour_of_day)
    tp = traffic_profile[[id_col, "day_of_week", "hour_of_day"] + present_cols].copy()
    tp[id_col] = tp[id_col].astype(str)

    edge_lookup = edges_gdf[[eid_col, "edge_idx"]].copy()
    edge_lookup[eid_col] = edge_lookup[eid_col].astype(str)

    merged = tp.merge(
        edge_lookup, left_on=id_col, right_on=eid_col, how="inner",
    )
    matched = len(merged)

    if matched > 0:
        edge_indices = merged["edge_idx"].values.astype(np.int64)
        day_indices  = merged["day_of_week"].values.astype(np.int64)
        hour_indices = merged["hour_of_day"].values.astype(np.int64)

        vals_np = merged[present_cols].values.astype(np.float32)
        np.nan_to_num(vals_np, copy=False)

        # If some dyn_cols are missing from profile, pad with zeros
        if len(present_cols) < len(dyn_cols):
            full = np.zeros((matched, len(dyn_cols)), dtype=np.float32)
            col_map = {c: i for i, c in enumerate(dyn_cols)}
            for j, c in enumerate(present_cols):
                full[:, col_map[c]] = vals_np[:, j]
            vals_np = full

        feats[edge_indices, day_indices, hour_indices] = torch.tensor(
            vals_np, dtype=torch.float32, device=device
        )

    n_unique = merged["edge_idx"].nunique() if matched > 0 else 0
    print(f"[MetaTraffic] Traffic profile: {len(traffic_profile):,} rows "
          f"→ matched {matched} ({n_unique} edges) → tensor [{n_edges}, 7, 24, {n_feats}]")
    return feats


# =============================================================================
# COMPONENT 1: Relational GAT Layer (sparse COO, multi-head)
# =============================================================================
class RelationalGATLayer(nn.Module):
    """
    Single-layer relational Graph Attention over the full road network.
    Operates on sparse COO adjacency — not dense masks.

    Input:  h [N, D]  +  rel_coo (list of R COO [2, E_r])
    Output: h' [N, D]  (residual connection)
    """

    def __init__(self, dim: int, num_relations: int = 8,
                 n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.dk = dim // n_heads
        self.R = num_relations

        # Per-relation Q/K projections for attention
        self.W_q = nn.ParameterList([
            nn.Parameter(torch.randn(n_heads, dim, self.dk) * 0.02)
            for _ in range(num_relations)
        ])
        self.W_k = nn.ParameterList([
            nn.Parameter(torch.randn(n_heads, dim, self.dk) * 0.02)
            for _ in range(num_relations)
        ])

        # Per-relation V projection
        self.W_v = nn.ParameterList([
            nn.Parameter(torch.randn(n_heads, dim, self.dk) * 0.02)
            for _ in range(num_relations)
        ])

        # Output projection
        self.out_proj = nn.Linear(dim, dim)
        self.layer_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, rel_coo: List[torch.Tensor]) -> torch.Tensor:
        """
        h:       [N, D]
        rel_coo: list of R tensors, each [2, E_r] (src, dst) COO indices
        returns: [N, D]
        """
        N, D = h.shape
        agg = torch.zeros(N, self.n_heads, self.dk, device=h.device, dtype=h.dtype)
        count = torch.zeros(N, 1, 1, device=h.device, dtype=h.dtype)

        for r in range(self.R):
            if rel_coo[r].numel() == 0:
                continue
            src = rel_coo[r][0]  # [E_r]
            dst = rel_coo[r][1]  # [E_r]
            E_r = src.shape[0]

            h_src = h[src]  # [E_r, D]
            h_dst = h[dst]  # [E_r, D]

            # Multi-head Q, K, V
            # q = h_dst @ W_q[r]  → [E_r, n_heads, dk]
            q = torch.einsum("ed,hdo->eho", h_dst, self.W_q[r])
            k = torch.einsum("ed,hdo->eho", h_src, self.W_k[r])
            v = torch.einsum("ed,hdo->eho", h_src, self.W_v[r])

            # Attention scores
            att_logits = (q * k).sum(-1) / math.sqrt(self.dk)  # [E_r, n_heads]

            # Softmax per destination node
            att_w = _scatter_softmax(att_logits, dst, N)  # [E_r, n_heads]

            # Weighted messages
            msg = v * att_w.unsqueeze(-1)  # [E_r, n_heads, dk]

            # Scatter-add to destination nodes
            dst_exp = dst.unsqueeze(-1).unsqueeze(-1).expand(-1, self.n_heads, self.dk)
            agg.scatter_add_(0, dst_exp, msg.to(agg.dtype))
            count.scatter_add_(
                0,
                dst.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 1),
                torch.ones(E_r, 1, 1, device=h.device, dtype=h.dtype),
            )

        # Average over contributing relations + heads
        safe_count = count.clamp_min(1.0)
        agg = agg / safe_count
        agg_flat = agg.reshape(N, D)

        out = self.out_proj(self.dropout(agg_flat))
        return self.layer_norm(h + out)


# =============================================================================
# COMPONENT 2: Direct Parameter Generator (simplified — no factorisation)
# =============================================================================
class DirectParamGenerator(nn.Module):
    """
    Generates per-link weight matrix W_i [d_in, d_out] and bias b_i [d_out]
    from meta-knowledge h_meta_i [meta_dim] via direct linear projection.

    W_gen: Linear(meta_dim, d_in * d_out) → reshape to [d_in, d_out]
    b_gen: Linear(meta_dim, d_out)

    For each link, the single matrix multiply  θ·h_i → 2100 numbers
    is reshaped into that link's personal W_i [39, 60].
    """

    # OLD: def __init__(self, meta_dim: int = 32, d_in: int = 35, d_out: int = 60):
    def __init__(self, meta_dim: int = 32, d_in: int = 39, d_out: int = 60):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out

        # Direct projection: h[D] → d_in*d_out numbers → reshape to W[d_in, d_out]
        self.W_gen = nn.Linear(meta_dim, d_in * d_out)

        # Bias generator: h[D] → b[d_out]
        self.b_gen = nn.Linear(meta_dim, d_out)

    def forward(self, h_meta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        h_meta: [N, meta_dim]
        Returns:
            W: [N, d_in, d_out]  per-link weight matrices
            b: [N, d_out]        per-link biases
        """
        N = h_meta.shape[0]
        W = self.W_gen(h_meta).view(N, self.d_in, self.d_out)  # [N, d_in, 60]
        b = self.b_gen(h_meta)                                  # [N, 60]
        return W, b


# =============================================================================
# MAIN MODEL: MetaTraffic
# =============================================================================
class MetaTrafficModel(nn.Module):
    """
    Full MetaTraffic module:
      1. Projects static_31 → D
      2. Relational GAT (2 layers) → h_struct [N, D]
      3. Direct param generator → W_i [39, 60], b_i [60]
      4. traffic_i = W_i · [static_i, time_i] + b_i   [M, 60]

    Both the encoder and generator are shared across all links/cities.
    Per-link individuality arises from different inputs and graph neighborhoods.
    """

    def __init__(
        self,
        # OLD: static_dim: int = 27,
        static_dim: int = 31,
        time_dim: int = 8,
        meta_dim: int = 32,
        traffic_dim: int = 60,
        num_relations: int = 8,
        gat_heads: int = 4,
        gat_layers: int = 2,
        gat_dropout: float = 0.1,
        generator_k: int = 16,
        refine_hidden: int = 128,
    ):
        super().__init__()
        self.static_dim = static_dim
        self.time_dim = time_dim
        self.input_dim = static_dim + time_dim  # 39 (was 35)
        self.meta_dim = meta_dim
        self.traffic_dim = traffic_dim
        # v3.2: Removed unused n_feature_groups / slots_per_group attributes
        # (were never referenced by any method)

        # --- v2.0 C2: GAT input_proj is now static-only (31→D, was 35→D) ---
        # OLD: self.input_proj = nn.Linear(self.input_dim, meta_dim)  # 35→D
        self.input_proj = nn.Linear(static_dim, meta_dim)  # C2: 31→D

        # --- v2.0 C6: Static feature z-normalization buffers ---
        # v3.1: These buffers hold the *active* city's stats.
        #        Per-city stats live in _per_city_norm (plain dict, not saved).
        # NOTE (v3.2): _per_city_norm is NOT serialized in state_dict().
        #   After loading a checkpoint, call init_static_normalization() to
        #   re-populate it before calling set_city_norm().
        self.register_buffer('static_mean', torch.zeros(static_dim))
        self.register_buffer('static_std', torch.ones(static_dim))
        self._per_city_norm: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

        # Relational GAT layers
        self.gat_layers = nn.ModuleList([
            RelationalGATLayer(
                dim=meta_dim,
                num_relations=num_relations,
                n_heads=gat_heads,
                dropout=gat_dropout,
            )
            for _ in range(gat_layers)
        ])

        # Direct parameter generator: h_struct → (W_i, b_i) per link
        # No factorisation — single Linear(D, 39*60) → reshape to W[39,60]
        self.param_gen = DirectParamGenerator(
            meta_dim=meta_dim,
            d_in=self.input_dim,    # 39
            d_out=traffic_dim,      # 60
        )

        # Non-linear refinement MLP: transforms the linear per-link
        # output into a more expressive traffic prediction.
        # This is shared across all links (per-link specialization
        # comes from the factored W_i parameters above).
        self.refine = nn.Sequential(
            nn.LayerNorm(traffic_dim),
            nn.Linear(traffic_dim, refine_hidden),
            nn.GELU(),
            nn.Linear(refine_hidden, refine_hidden),
            nn.GELU(),
            nn.Linear(refine_hidden, traffic_dim),
        )

        # --- Output activation: sigmoid + learnable per-feature scale/shift ---
        # Bounds predictions to a physically plausible range.
        # sigmoid(raw) * |output_scale| + output_shift
        #   → maps output to [output_shift, output_shift + |output_scale|]
        # Initialized with conservative defaults; call init_output_range()
        # with actual traffic profile data before training for best results.
        self.output_scale = nn.Parameter(torch.ones(traffic_dim))   # [60]
        self.output_shift = nn.Parameter(torch.zeros(traffic_dim))  # [60]

        # --- Structural cluster prototype bank (populated after meta-training) ---
        # proto_W: [K, S, d_in, d_out]  mean W_i per (k,s) from source cities
        # proto_b: [K, S, d_out]         mean b_i per (k,s) from source cities
        # target_k/s_labels: [N_target]  cluster tags for current target city
        self.register_buffer('proto_W', None)
        self.register_buffer('proto_b', None)
        self.register_buffer('target_k_labels', None)
        self.register_buffer('target_s_labels', None)

        # Gated interpolation:  W = α·proto_W + (1-α)·W_gen
        # v3.2 FIX: Initialize at 2.0 (sigmoid → 0.88) so the model starts by
        # trusting source prototypes. This lets the target adaptation learn to
        # reduce α if the model's own generation becomes better.
        # OLD: torch.tensor(0.0)  # sigmoid → 0.5 — barely biased, never moved
        self.proto_alpha_logit = nn.Parameter(torch.tensor(2.0))  # sigmoid → 0.88

    # -----------------------------------------------------------------
    # v2.0 C1: OLD forward() — COMMENTED OUT
    #   (replaced by encode_structure + predict_traffic below)
    # -----------------------------------------------------------------
    # def forward(self, static_feats, time_feats, rel_coo):
    #     x_input = torch.cat([static_feats, time_feats], dim=-1)  # [N, 26]
    #     h = self.input_proj(x_input)          # [N, meta_dim]
    #     for gat_layer in self.gat_layers:
    #         h = gat_layer(h, rel_coo)
    #     h_meta = h
    #     W_i, b_i = self.param_gen(h_meta)     # [N, 26, 60], [N, 60]
    #     traffic_raw = torch.bmm(
    #         W_i.transpose(1, 2), x_input.unsqueeze(-1)
    #     ).squeeze(-1) + b_i                   # [N, 60]
    #     raw = traffic_raw + self.refine(traffic_raw)
    #     predicted_traffic = (
    #         torch.sigmoid(raw) * self.output_scale.abs() + self.output_shift
    #     )
    #     return predicted_traffic, h_meta

    # =================================================================
    # v2.0 C1: Phase 1 — Time-invariant structural encoding
    # =================================================================
    def encode_structure(
        self,
        static_feats: torch.Tensor,    # [N, 31]
        rel_coo: List[torch.Tensor],   # list of R COO [2, E_r]
    ) -> torch.Tensor:
        """
        Encode road-network structure into per-node embeddings.
        Uses ONLY static features (no time) — called once per batch.

        Returns:
            h_struct: [N, meta_dim]  structural embeddings for all nodes
        """
        # C6: z-normalize static features before projection
        s_norm = (static_feats - self.static_mean) / self.static_std.clamp_min(1e-6)

        # C2: input_proj is now Linear(31, D), static-only
        h = self.input_proj(s_norm)  # [N, meta_dim]

        # Relational GAT over road graph (time-invariant)
        for gat_layer in self.gat_layers:
            h = gat_layer(h, rel_coo)  # [N, meta_dim]

        return h  # [N, meta_dim]

    # =================================================================
    # v2.0 C1, C3: Phase 2 — Predict traffic for specific edges
    # =================================================================
    def predict_traffic(
        self,
        h_struct: torch.Tensor,        # [N, D] from encode_structure
        static_feats: torch.Tensor,    # [N, 31] raw (NOT z-normalised)
        time_feats: torch.Tensor,      # [M, 8] per-query time features
        edge_indices: torch.Tensor,    # [M] edge index for each query
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict traffic at specific edges with their per-node time context.

        Generates W_i, b_i only for *unique* edges in ``edge_indices``
        (C3 — avoids 100× waste of predicting all N), then applies the
        per-link transform using each query's own time features (C4).

        Returns:
            predicted_traffic: [M, 60]
            h_unique:          [U, D]  (embeddings for unique edges)
        """
        # De-duplicate: generate W/b only for unique edges
        unique_edges, inv = torch.unique(edge_indices, return_inverse=True)
        h_u = h_struct[unique_edges]           # [U, D]
        W_u, b_u = self.param_gen(h_u)         # [U, d_in, 60], [U, 60]

        # ── Structural cluster prototype injection (gated) ───────────
        # When a prototype bank is loaded (post meta-training) and
        # target cluster labels are set, interpolate between the source
        # prototype and the model's own generation via a learnable gate:
        #   W = α·proto_W[k,s] + (1-α)·W_gen
        # α = sigmoid(proto_alpha_logit), initialized at 0.5.
        if self.proto_W is not None and self.target_k_labels is not None:
            k_u = self.target_k_labels[unique_edges]   # [U]
            s_u = self.target_s_labels[unique_edges]   # [U]
            alpha = torch.sigmoid(self.proto_alpha_logit)  # scalar in (0,1)
            W_u = alpha * self.proto_W[k_u, s_u] + (1 - alpha) * W_u  # [U, d_in, 60]
            b_u = alpha * self.proto_b[k_u, s_u] + (1 - alpha) * b_u  # [U, 60]

        # Expand to all M queries via inverse map
        W_m = W_u[inv]                         # [M, d_in, 60]
        b_m = b_u[inv]                         # [M, 60]

        # Per-query input: [static[edge_i], time_i]
        # OLD (v3.0): raw static features → scale mismatch (length ~3000 vs time ~1)
        # s_m = static_feats[edge_indices]       # [M, 31]
        # x_input = torch.cat([s_m, time_feats], dim=-1)  # [M, 39]
        # v3.1: z-normalize static features in predict_traffic too,
        #        matching encode_structure's normalization, so W_i sees
        #        features at comparable scales (~0-mean, ~1-std).
        s_m_raw = static_feats[edge_indices]   # [M, 31]
        s_m = (s_m_raw - self.static_mean) / self.static_std.clamp_min(1e-6)  # [M, 31]
        x_input = torch.cat([s_m, time_feats], dim=-1)  # [M, 39]

        # Per-query linear transform
        traffic_raw = torch.bmm(
            W_m.transpose(1, 2),               # [M, 60, d_in]
            x_input.unsqueeze(-1),             # [M, d_in, 1]
        ).squeeze(-1) + b_m                    # [M, 60]

        # Non-linear refinement (shared MLP, unchanged)
        raw = traffic_raw + self.refine(traffic_raw)  # residual

        # Sigmoid + per-feature scale/shift (unchanged)
        predicted_traffic = (
            torch.sigmoid(raw) * self.output_scale.abs() + self.output_shift
        )

        return predicted_traffic, h_u

    # =================================================================
    # v2.0: Backward-compatible forward() — delegates to two phases
    # =================================================================
    def forward(
        self,
        static_feats: torch.Tensor,    # [N, 31]
        time_feats: torch.Tensor,      # [M, 8]  (per-query)
        rel_coo: List[torch.Tensor],   # list of R COO [2, E_r]
        edge_indices: Optional[torch.Tensor] = None,  # [M] or None → all N
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode_structure → predict_traffic.

        When ``edge_indices`` is None (backward compat), predicts for ALL
        N edges — equivalent to the old forward().  When provided, predicts
        only for the requested edges (v2.0 efficient path).
        """
        h_struct = self.encode_structure(static_feats, rel_coo)
        if edge_indices is None:
            edge_indices = torch.arange(
                static_feats.shape[0], device=static_feats.device)
        predicted, h_meta = self.predict_traffic(
            h_struct, static_feats, time_feats, edge_indices)
        return predicted, h_meta

    @torch.no_grad()
    def init_output_range(
        self,
        traffic_tensors: Dict[str, torch.Tensor],
        margin: float = 0.1,
    ):
        """
        Initialize output_scale and output_shift from actual traffic profile
        data so that sigmoid(0→1) maps to [feat_min, feat_max] per feature.

        Args:
            traffic_tensors: dict of {city_key: [N, 7, 24, 60]} raw profiles
            margin: fractional margin beyond observed min/max (default 10%)
        """
        all_vals = []  # collect [K, 60] slices with actual data
        for ck, tp in traffic_tensors.items():
            # Flatten to [N*7*24, 60]
            flat = tp.reshape(-1, tp.shape[-1])
            # Keep only rows with at least one non-zero feature
            has_data = (flat.abs().sum(-1) > 0)
            if has_data.any():
                all_vals.append(flat[has_data].float().cpu())

        if not all_vals:
            print("[MetaTraffic] WARNING: no traffic data found, keeping default output range")
            return

        combined = torch.cat(all_vals, dim=0)  # [total_rows, 60]

        # Per-feature min/max (robust: use 1st/99th percentile to avoid outliers)
        feat_lo = torch.quantile(combined, 0.01, dim=0)  # [60]
        feat_hi = torch.quantile(combined, 0.99, dim=0)  # [60]

        feat_range = (feat_hi - feat_lo).clamp_min(0.1)  # avoid zero range
        feat_lo_m  = feat_lo - margin * feat_range        # add margin below
        feat_range_m = feat_range * (1 + 2 * margin)      # expand range

        self.output_shift.data.copy_(feat_lo_m)
        self.output_scale.data.copy_(feat_range_m)

        print(f"[MetaTraffic] Output range initialized from {len(traffic_tensors)} cities "
              f"({combined.shape[0]:,} data rows)")
        print(f"  output_shift (min per feat): mean={feat_lo_m.mean():.3f}, "
              f"range=[{feat_lo_m.min():.3f}, {feat_lo_m.max():.3f}]")
        print(f"  output_scale (range per feat): mean={feat_range_m.mean():.3f}, "
              f"range=[{feat_range_m.min():.3f}, {feat_range_m.max():.3f}]")


    # =================================================================
    # v3.1: Per-city static-feature normalisation
    # =================================================================
    @torch.no_grad()
    def init_static_normalization(
        self,
        static_tensors: Dict[str, torch.Tensor],
    ):
        """
        Compute **per-city** static_mean / static_std and store in
        ``_per_city_norm``.  The registered buffers are set to the first
        city's stats as a sensible default.

        Args:
            static_tensors: dict of {city_key: [N_city, 31]} raw static features
        """
        self._per_city_norm.clear()
        first_ck = None
        for ck, sf in static_tensors.items():
            sf_f = sf.float().cpu()
            mu = sf_f.mean(dim=0)
            sigma = sf_f.std(dim=0).clamp_min(1e-6)
            self._per_city_norm[ck] = (mu, sigma)
            if first_ck is None:
                first_ck = ck

        # Default buffers = first city (overwritten by set_city_norm)
        if first_ck is not None:
            self.static_mean.copy_(self._per_city_norm[first_ck][0])
            self.static_std.copy_(self._per_city_norm[first_ck][1])

        print(f"[MetaTraffic] Per-city static normalization registered for "
              f"{list(static_tensors.keys())}")
        for ck in static_tensors:
            mu, sigma = self._per_city_norm[ck]
            n = static_tensors[ck].shape[0]
            print(f"  {ck}: {n:,} edges  "
                  f"mean=[{mu.min():.4f}, {mu.max():.4f}]  "
                  f"std=[{sigma.min():.4f}, {sigma.max():.4f}]")

    @torch.no_grad()
    def register_city_norm(self, city_key: str, static_feats: torch.Tensor):
        """Register normalisation stats for a single city (e.g. target)."""
        sf_f = static_feats.float().cpu()
        mu = sf_f.mean(dim=0)
        sigma = sf_f.std(dim=0).clamp_min(1e-6)
        self._per_city_norm[city_key] = (mu, sigma)
        print(f"[MetaTraffic] Registered norm for '{city_key}' "
              f"({static_feats.shape[0]:,} edges)")

    @torch.no_grad()
    def set_city_norm(self, city_key: str):
        """Copy per-city stats into the active buffers."""
        if city_key not in self._per_city_norm:
            raise KeyError(f"No normalisation stats for '{city_key}'. "
                           f"Call register_city_norm first.")
        mu, sigma = self._per_city_norm[city_key]
        self.static_mean.copy_(mu)
        self.static_std.copy_(sigma)

    def validate_city_norm_ready(self, expected_keys: List[str]):
        """Ensure per-city normalization stats exist for all required cities."""
        missing = [k for k in expected_keys if k not in self._per_city_norm]
        if missing:
            raise RuntimeError(
                "Missing per-city normalization stats for: "
                f"{missing}. Call register_city_norm for all cities first."
            )

    # =================================================================
    # Structural Cluster Prototype Bank — explicit (k,s) transfer
    # =================================================================
    @torch.no_grad()
    def build_prototype_bank(
        self,
        city_static_feats: Dict[str, torch.Tensor],
        city_rel_coo: Dict[str, List[torch.Tensor]],
        city_clusters: Dict[str, dict],
        K: int = 7,
        S: int = 9,
        city_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Build (k,s) prototype bank from source-city per-link parameters.

        After meta-training, runs the encoder + param generator on each
        source city to obtain W_i [d_in, d_out] and b_i [d_out] per link.
        Groups by structural cluster tag (k, s) and averages (optionally
        weighted by city similarity) to create a prototype bank.

        Args:
            city_static_feats: {city_key: [N_city, 31]}
            city_rel_coo:      {city_key: list of R COO tensors}
            city_clusters:     {city_key: dict with 'k_labels', 's_labels'}
            K: number of physical clusters (default 6)
            S: number of topology clusters (default 14)
            city_weights: {city_key: float} cosine-similarity weight per source city.
                          If None, all cities get equal weight (backward compat).
        """
        dev = next(self.parameters()).device
        d_in, d_out = self.input_dim, self.traffic_dim

        proto_W_sum = torch.zeros(K * S, d_in * d_out)
        proto_b_sum = torch.zeros(K * S, d_out)
        # OLD: proto_count = torch.zeros(K * S)  # equal count-based averaging
        proto_weight_sum = torch.zeros(K * S)  # v2: weighted sum denominator

        self.eval()
        total_links = 0
        for ck in city_static_feats:
            # v3.1: per-city normalization
            if ck in self._per_city_norm:
                self.set_city_norm(ck)

            # v2: city-similarity weight (default 1.0 for equal weighting)
            w_city = city_weights.get(ck, 1.0) if city_weights is not None else 1.0

            sf = city_static_feats[ck].to(dev)
            rc = [r.to(dev) for r in city_rel_coo[ck]]
            k_lab = city_clusters[ck]['k_labels'].long()
            s_lab = city_clusters[ck]['s_labels'].long()
            N = sf.shape[0]
            total_links += N

            # Forward through encoder + param generator
            h_struct = self.encode_structure(sf, rc)
            W_all, b_all = self.param_gen(h_struct)    # [N, d_in, d_out], [N, d_out]

            # OLD: Vectorized accumulation by flat (k,s) index (equal weighting)
            # flat_idx = (k_lab * S + s_lab).cpu()       # [N]
            # proto_W_sum.index_add_(0, flat_idx, W_all.cpu().reshape(N, -1))
            # proto_b_sum.index_add_(0, flat_idx, b_all.cpu())
            # proto_count.index_add_(0, flat_idx, torch.ones(N))

            # v2: City-weighted accumulation — each link's W_i/b_i is scaled
            # by the similarity weight of its source city
            flat_idx = (k_lab * S + s_lab).cpu()       # [N]
            proto_W_sum.index_add_(0, flat_idx, W_all.cpu().reshape(N, -1) * w_city)
            proto_b_sum.index_add_(0, flat_idx, b_all.cpu() * w_city)
            proto_weight_sum.index_add_(0, flat_idx, torch.full((N,), w_city))

        # Average — only populated slots (weighted)
        # OLD: mask = proto_count > 0
        mask = proto_weight_sum > 0
        proto_W_avg = torch.zeros_like(proto_W_sum)
        proto_b_avg = torch.zeros_like(proto_b_sum)
        # OLD: proto_W_avg[mask] = proto_W_sum[mask] / proto_count[mask].unsqueeze(-1)
        # OLD: proto_b_avg[mask] = proto_b_sum[mask] / proto_count[mask].unsqueeze(-1)
        proto_W_avg[mask] = proto_W_sum[mask] / proto_weight_sum[mask].unsqueeze(-1)
        proto_b_avg[mask] = proto_b_sum[mask] / proto_weight_sum[mask].unsqueeze(-1)

        # Reshape and store as buffers
        proto_W = proto_W_avg.reshape(K, S, d_in, d_out).to(dev)
        proto_b = proto_b_avg.reshape(K, S, d_out).to(dev)
        self.register_buffer('proto_W', proto_W)
        self.register_buffer('proto_b', proto_b)

        populated = mask.sum().item()
        print(f"\n[MetaTraffic] Prototype bank built from {len(city_static_feats)} "
              f"source cities ({total_links:,} links)")
        if city_weights is not None:
            for ck, w in city_weights.items():
                print(f"  City weight: {ck} = {w:.4f}")
        else:
            print(f"  City weights: equal (no similarity weighting)")
        print(f"  Populated (k,s) slots: {populated}/{K*S}")
        print(f"  proto_W: {list(proto_W.shape)}, "
              f"proto_b: {list(proto_b.shape)}")
        print(f"  W norm range: [{proto_W[mask.reshape(K,S)].norm(dim=(-2,-1)).min():.4f}, "
              f"{proto_W[mask.reshape(K,S)].norm(dim=(-2,-1)).max():.4f}]")

    def set_target_clusters(
        self,
        k_labels: torch.Tensor,
        s_labels: torch.Tensor,
    ):
        """
        Set target-city cluster assignments for prototype lookup.

        Args:
            k_labels: [N_target] physical cluster index (0..K-1)
            s_labels: [N_target] topology cluster index (0..S-1)
        """
        dev = next(self.parameters()).device
        self.register_buffer('target_k_labels', k_labels.long().to(dev))
        self.register_buffer('target_s_labels', s_labels.long().to(dev))
        print(f"[MetaTraffic] Target clusters set: {k_labels.shape[0]} links, "
              f"K={k_labels.max().item()+1}, S={s_labels.max().item()+1}")

    def clear_prototypes(self):
        """Clear prototype bank and target cluster assignments."""
        self.register_buffer('proto_W', None)
        self.register_buffer('proto_b', None)
        self.register_buffer('target_k_labels', None)
        self.register_buffer('target_s_labels', None)


# =============================================================================
# LOSS: Traffic reconstruction  (RMSE on z-score normalised features)
# =============================================================================
def compute_traffic_loss(
    predicted_traffic: torch.Tensor,  # [N, 60]
    real_traffic_4d: torch.Tensor,    # [N, 7, 24, 60]
    day_of_week: int,                 # 0-6
    hour_of_day: int,                 # 0-23
    mask: Optional[torch.Tensor] = None,  # [N] bool — optional external mask
    dyn_scaler_mean: Optional[torch.Tensor] = None,  # [60]
    dyn_scaler_std: Optional[torch.Tensor] = None,   # [60]
) -> torch.Tensor:
    """
    RMSE between predicted traffic and real traffic at the given time slot.
    Features are z-score normalised before comparison so that speeds (~0-50)
    and vehicle counts (~0-5000) contribute equally.

    Returns scalar RMSE (sqrt of MSE on normalised features).
    """
    real_slice = real_traffic_4d[:, day_of_week, hour_of_day, :]  # [N, 60]
    # Only compare edges that have real data at this time slot
    slot_mask = (real_slice.abs().sum(dim=-1) > 0)  # [N]
    if mask is not None:
        slot_mask = slot_mask & mask
    if not slot_mask.any():
        return torch.zeros((), device=predicted_traffic.device)

    pred_sel = predicted_traffic[slot_mask]  # [K, 60]
    real_sel = real_slice[slot_mask]          # [K, 60]

    # Z-score normalise so speed and count features contribute equally
    if dyn_scaler_std is not None:
        std = dyn_scaler_std.view(1, -1).clamp_min(1e-6)
        mean = dyn_scaler_mean.view(1, -1) if dyn_scaler_mean is not None else 0.0
        pred_sel = (pred_sel - mean) / std
        real_sel = (real_sel - mean) / std

    mse = F.mse_loss(pred_sel, real_sel)
    return torch.sqrt(mse + 1e-8)  # RMSE


# =============================================================================
# DEPRECATED HELPER: Build time feature tensor for all edges
# v2.0 C7: This function embodied the batch-averaging flaw.  Kept as a
#          deprecated wrapper for backward compatibility / diagnostics.
# =============================================================================
def build_time_features_for_batch(
    batch: dict,
    time_col_indices: List[int],
    n_edges: int,
    device: torch.device,
) -> torch.Tensor:
    """DEPRECATED (v2.0 C7) — use per-node time from x_seq instead."""
    warnings.warn(
        "build_time_features_for_batch is deprecated (v2.0 C7). "
        "Per-node time features are now extracted directly from x_seq.",
        DeprecationWarning, stacklevel=2,
    )
    # --- original body kept as comments for reference ---
    # x_seq = batch["x_seq"]                 # [B, M, 86]
    # time_data = x_seq[:, :, time_col_indices]  # [B, M, 8]
    # mask = batch.get("seq_mask", None)
    # if mask is not None:
    #     route_idx = batch["route_seq_idx"]  # [B, L]
    #     B, L = route_idx.shape
    #     time_route = torch.zeros(B, L, len(time_col_indices), device=x_seq.device)
    #     for t_i, col_i in enumerate(time_col_indices):
    #         col_vals = x_seq[:, :, col_i]  # [B, M]
    #         time_route[:, :, t_i] = torch.gather(col_vals, 1, route_idx)
    #     seq_mask = mask.bool()
    #     time_route[~seq_mask] = 0.0
    #     n_valid = seq_mask.float().sum()
    #     if n_valid > 0:
    #         avg_time = time_route.sum(dim=(0, 1)) / n_valid  # [8]
    #     else:
    #         avg_time = torch.zeros(len(time_col_indices), device=device)
    # else:
    #     avg_time = time_data.mean(dim=(0, 1))  # [8]
    # time_tensor = avg_time.unsqueeze(0).expand(n_edges, -1)  # [N, 8]
    # return time_tensor
    x_seq = batch["x_seq"]
    time_data = x_seq[:, :, time_col_indices]
    avg_time = time_data.mean(dim=(0, 1))
    return avg_time.unsqueeze(0).expand(n_edges, -1)


# =============================================================================
# DEPRECATED HELPER: Extract day_of_week & hour_of_day from a batch (averaged)
# v2.0 C7: Replaced by extract_per_trip_time_slot.
# =============================================================================
def extract_batch_time_slot(
    batch: dict,
    time_col_indices: List[int],
) -> Tuple[int, int]:
    """DEPRECATED (v2.0 C7) — use extract_per_trip_time_slot instead."""
    warnings.warn(
        "extract_batch_time_slot is deprecated (v2.0 C7). "
        "Use extract_per_trip_time_slot for per-trip (day, hour).",
        DeprecationWarning, stacklevel=2,
    )
    # --- original body kept as comments ---
    # x_seq = batch["x_seq"]
    # dow_col = time_col_indices[0]; sin_col = time_col_indices[4]; cos_col = time_col_indices[5]
    # mask = batch.get("seq_mask", None)
    # if mask is not None:
    #     route_idx = batch["route_seq_idx"]
    #     dow_route = torch.gather(x_seq[:, :, dow_col], 1, route_idx)
    #     sin_route = torch.gather(x_seq[:, :, sin_col], 1, route_idx)
    #     cos_route = torch.gather(x_seq[:, :, cos_col], 1, route_idx)
    #     seq_mask = mask.bool()
    #     n_valid = seq_mask.float().sum().clamp_min(1.0)
    #     avg_dow = (dow_route * seq_mask.float()).sum() / n_valid
    #     avg_sin = (sin_route * seq_mask.float()).sum() / n_valid
    #     avg_cos = (cos_route * seq_mask.float()).sum() / n_valid
    # else:
    #     avg_dow = x_seq[:, :, dow_col].mean()
    #     avg_sin = x_seq[:, :, sin_col].mean()
    #     avg_cos = x_seq[:, :, cos_col].mean()
    # day = int(avg_dow.round().item()) % 7
    # hour = int(round(math.atan2(avg_sin.item(), avg_cos.item()) * 24 / (2*math.pi))) % 24
    # return day, hour
    x_seq = batch["x_seq"]
    avg_dow = x_seq[:, :, time_col_indices[0]].mean()
    avg_sin = x_seq[:, :, time_col_indices[4]].mean()
    avg_cos = x_seq[:, :, time_col_indices[5]].mean()
    day = int(avg_dow.round().item()) % 7
    hour = int(round(math.atan2(avg_sin.item(), avg_cos.item()) * 24 / (2 * math.pi))) % 24
    return day, hour


# =============================================================================
# v2.0 C5: Per-trip time slot extraction
# =============================================================================
def extract_per_trip_time_slot(
    x_seq: torch.Tensor,            # [B, M, 86] raw features
    route_seq_idx: torch.Tensor,    # [B, L]
    seq_mask: torch.Tensor,         # [B, L] bool
    time_col_indices: List[int],
) -> Tuple[List[int], List[int]]:
    """
    Extract per-trip (day_of_week, hour_of_day) from each trip's first
    valid route node.  Returns two lists of length B.

    Within a single trip all route nodes have very similar timestamps
    (a trip spans minutes, not hours), so the first valid node is
    a faithful representative of the trip's time.
    """
    B = x_seq.shape[0]
    dow_col = time_col_indices[0]   # day_of_week
    sin_col = time_col_indices[4]   # tod_sin
    cos_col = time_col_indices[5]   # tod_cos

    days: List[int] = []
    hours: List[int] = []
    for b in range(B):
        mask_b = seq_mask[b].bool()
        if not mask_b.any():
            days.append(0); hours.append(0)
            continue
        first_pos = mask_b.nonzero(as_tuple=True)[0][0].item()
        node_pos = route_seq_idx[b, first_pos].item()

        dow = x_seq[b, node_pos, dow_col].item()
        s   = x_seq[b, node_pos, sin_col].item()
        c   = x_seq[b, node_pos, cos_col].item()

        day = int(round(dow)) % 7
        hour = int(round(math.atan2(s, c) * 24 / (2 * math.pi))) % 24
        days.append(day)
        hours.append(hour)
    return days, hours


# =============================================================================
# v2.0 C5: Per-trip traffic reconstruction loss
# =============================================================================
def compute_per_trip_traffic_loss(
    pred_traffic_batch: torch.Tensor,       # [B, M, 60] predictions
    node_edge_idx: torch.Tensor,            # [B, M] edge index per position
    real_traffic_4d: torch.Tensor,          # [N, 7, 24, 60]
    trip_days: List[int],                   # [B] day_of_week per trip
    trip_hours: List[int],                  # [B] hour_of_day per trip
    route_seq_idx: torch.Tensor,            # [B, L]
    seq_mask: torch.Tensor,                 # [B, L] bool
    traffic_observed_mask: torch.Tensor,    # [N] bool
    dyn_scaler_mean: Optional[torch.Tensor] = None,  # [60]
    dyn_scaler_std: Optional[torch.Tensor] = None,   # [60]
    n_edges: Optional[int] = None,
) -> torch.Tensor:
    """
    Per-trip traffic reconstruction loss (v2.0 C5).

    Each trip's predictions are compared against real traffic at that
    trip's actual (day_of_week, hour_of_day), not a batch-averaged time.

    Returns scalar RMSE averaged across trips with valid data.
    """
    B = pred_traffic_batch.shape[0]
    dev = pred_traffic_batch.device
    if n_edges is None:
        n_edges = real_traffic_4d.shape[0]

    loss_sum = torch.zeros((), device=dev)
    n_valid_trips = 0

    for b in range(B):
        mask_b = seq_mask[b].bool()
        if not mask_b.any():
            continue

        day_b = trip_days[b]
        hour_b = trip_hours[b]

        # Route-node positions (within the M-length node sequence)
        valid_pos = route_seq_idx[b][mask_b]  # [L_valid]

        # Edge indices for valid route positions
        route_edge_idx = node_edge_idx[b, valid_pos].clamp(0, n_edges - 1)

        # Predicted traffic for those positions
        pred_route = pred_traffic_batch[b, valid_pos]  # [L_valid, 60]

        # Ground truth at this trip's time
        real_slice = real_traffic_4d[route_edge_idx, day_b, hour_b, :]  # [L_valid, 60]

        # Keep only edges that have real data
        has_data = (
            (real_slice.abs().sum(-1) > 0)
            & traffic_observed_mask[route_edge_idx]
        )
        if not has_data.any():
            continue

        pred_sel = pred_route[has_data]
        real_sel = real_slice[has_data]

        # Z-normalise so speed and count features contribute equally
        if dyn_scaler_std is not None:
            std  = dyn_scaler_std.view(1, -1).clamp_min(1e-6)
            mean = dyn_scaler_mean.view(1, -1) if dyn_scaler_mean is not None else 0.0
            pred_sel = (pred_sel - mean) / std
            real_sel = (real_sel - mean) / std

        mse_b = F.mse_loss(pred_sel, real_sel)
        loss_sum = loss_sum + torch.sqrt(mse_b + 1e-8)
        n_valid_trips += 1

    if n_valid_trips == 0:
        return torch.zeros((), device=dev)
    return loss_sum / n_valid_trips


# =============================================================================
# TRAINING: MetaTraffic + LinkETA joint one epoch
# =============================================================================
def metatraffic_one_epoch(
    loader: DataLoader,
    meta_traffic_model: MetaTrafficModel,
    linketa_model: LinkETA,
    optimizer: torch.optim.Optimizer,
    static_feats: torch.Tensor,       # [N, 31] for current city
    traffic_real: torch.Tensor,        # [N, 7, 24, 60] real traffic (temporal)
    rel_coo: List[torch.Tensor],       # COO relations for current city
    time_col_indices: List[int],       # indices of time columns in 86-dim input
    dyn_col_indices: torch.Tensor,     # indices of dynamic columns in 86-dim input
    scaler_mean: torch.Tensor,         # [86]
    scaler_std: torch.Tensor,          # [86]
    vff_vec_city: torch.Tensor,        # [N]
    epoch: int = 0,
    train: bool = True,
    # [2026-06-06] 4-task DWA: speed/link/route/traffic (matches base LinkETA's
    # {speed,link,route} DWA + traffic added as the 4th task).
    w_speed: float = 0.5,                 # DWA-managed speed weight
    w_link: float = 0.5,
    w_route: float = 0.5,
    # OLD: lambda_traffic: float = 0.5,   # fixed weight — replaced by DWA-managed w_traffic
    w_traffic: float = 0.5,               # DWA-managed traffic weight (symmetric with w_link, w_route)
    clip_norm: float = 10.0,
    city_name: str = "",
) -> dict:
    """
    One epoch of joint MetaTraffic + LinkETA training (v2.0).

    For each batch:
      0. encode_structure() once → h_struct [N, D]         (C1)
      1. Per-node predict_traffic() for batch edges only    (C3, C4)
      2. Inject predictions into x_seq[..., dyn]
      3. LinkETA standard forward pass                     (unchanged)
      4. Per-trip traffic loss at trip's actual (day, hour)  (C5)
         + L_TTE → final loss

    Pure MAML: always uses MetaTraffic predictions (no blending with real traffic).

    Returns dict of metrics.
    """
    # v3.1: activate per-city normalization if available
    if city_name and city_name in meta_traffic_model._per_city_norm:
        meta_traffic_model.set_city_norm(city_name)

    meta_traffic_model.train() if train else meta_traffic_model.eval()
    linketa_model.train() if train else linketa_model.eval()

    dev = static_feats.device
    n_edges = static_feats.shape[0]

    # Global mask: edges that have *any* traffic data across all time slots
    traffic_observed_mask = (traffic_real.abs().sum(dim=(-1, -2, -3)) > 0)  # [N]

    # Dynamic-feature normalisation parameters for traffic loss
    dyn_scaler_mean = scaler_mean[dyn_col_indices]  # [60]
    dyn_scaler_std  = scaler_std[dyn_col_indices]   # [60]

    # Accumulators
    sum_link_loss = sum_route_loss = sum_traffic_rmse = 0.0
    sum_mae_link = sum_mape_link = sum_sse_link = 0.0
    N_links = N_links_mape = 0
    sum_mae_route = sum_mape_route = sum_sse_route = 0.0
    N_routes_loss = N_routes_metrics = N_routes_mape = 0
    sum_speed_mae = sum_vol_mae = 0.0
    N_speed = N_vol = 0
    n_batches_with_traffic = 0
    # [2026-06-07] WAPE/MdAPE/P90 pooling (link + route)
    sum_abs_link = sum_true_link = 0.0
    ape_link_list = []
    sum_abs_route = sum_true_route = 0.0
    ape_route_list = []
    # [2026-06-06] speed loss (for DWA) + speed MAE vs the actual target sp_actual
    sum_speed_loss = 0.0
    sum_speed_mae_actual = 0.0
    N_speed_actual = 0

    for batch in loader:
        # Move batch to device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(dev)

        # ── 0. Encode road structure (once per batch, C1) ─────────────
        # OLD (averaged time + all-N prediction):
        #   batch_day, batch_hour = extract_batch_time_slot(batch, time_col_indices)
        #   time_tensor = build_time_features_for_batch(batch, time_col_indices, n_edges, dev)
        #   predicted_traffic, h_meta = meta_traffic_model(static_feats, time_tensor, rel_coo)
        #   predicted_traffic = torch.nan_to_num(predicted_traffic, ...)
        #   node_edge_idx = batch["node_edge_idx"]
        #   x_seq = batch["x_seq"].clone()
        #   flat_edge_idx = node_edge_idx.clamp(0, n_edges - 1)
        #   pred_traffic_batch = predicted_traffic[flat_edge_idx]
        #   x_seq[:, :, dyn_col_indices] = pred_traffic_batch.to(x_seq.dtype)
        with torch.amp.autocast('cuda', enabled=USE_AMP_MT):
            h_struct = meta_traffic_model.encode_structure(
                static_feats, rel_coo)                       # [N, meta_dim]

        # ── 1. Per-node traffic prediction (C1, C3, C4) ──────────────
        node_edge_idx = batch["node_edge_idx"]               # [B, M]
        x_seq = batch["x_seq"].clone()                        # [B, M, 86]
        B, M, _ = x_seq.shape

        flat_edge_idx = node_edge_idx.reshape(-1).clamp(0, n_edges - 1)    # [B*M]
        flat_time = x_seq.reshape(B * M, -1)[:, time_col_indices]          # [B*M, 8]

        with torch.amp.autocast('cuda', enabled=USE_AMP_MT):
            predicted_traffic_flat, _ = meta_traffic_model.predict_traffic(
                h_struct, static_feats, flat_time, flat_edge_idx
            )  # [B*M, 60]

        # Safety: clamp extreme values from early random weights
        predicted_traffic_flat = torch.nan_to_num(
            predicted_traffic_flat, nan=0.0, posinf=1e4, neginf=-1e4)
        pred_traffic_batch = predicted_traffic_flat.reshape(B, M, -1)  # [B, M, 60]

        # ── 2. Inject MetaTraffic predictions into batch features ─────
        x_seq[:, :, dyn_col_indices] = pred_traffic_batch.to(x_seq.dtype)

        # ── 3. Standard LinkETA forward ──────────────────────────────
        masks_per_rel = [m.to(dev) for m in batch["masks_per_rel"]]
        route_seq_idx = batch["route_seq_idx"]
        seq_true_tt = torch.nan_to_num(batch["seq_true_tt"], nan=0.0)
        seq_link_len = torch.nan_to_num(batch["seq_link_len"], nan=0.0).clamp_min(1e-3)
        seq_mask = batch["seq_mask"].bool()
        trip_tt = torch.nan_to_num(batch["trip_tt"], nan=0.0)
        hop_idx = batch["hop_idx"]
        route_flag_nodes = batch["route_flag"]

        sp_true_node = torch.nan_to_num(batch["sp_ratio_agg"], nan=0.0)
        lc_true_node = torch.nan_to_num(batch["logcnt1h_node"], nan=0.0)
        sp_w_node = torch.nan_to_num(batch["speed_w_node"], nan=0.0)

        gather_for_seq = lambda node_vals: torch.gather(node_vals, 1, route_seq_idx)
        sp_true = gather_for_seq(sp_true_node)
        lc_true = gather_for_seq(lc_true_node)
        sp_w = gather_for_seq(sp_w_node)

        vff_seq_b = vff_vec_city[node_edge_idx]
        vff_seq_b = torch.gather(vff_seq_b, 1, route_seq_idx)
        # [2026-06-06] CLEAN (pre-noise) v_ff copy for sp_actual pinned-row detection
        vff_seq_clean = vff_seq_b.clone()
        if train:
            rng = vff_noise_range(epoch)
            if rng > 0:
                noise = 1.0 + rng * (2.0 * torch.rand_like(vff_seq_b) - 1.0)
                vff_seq_b = vff_seq_b * noise

        # [2026-06-06] Per-link sp_actual TARGET (mirrors base LinkETA one_epoch):
        #   sp_actual = link_len / (link_tt · v_ff), clamped to r_max; rows pinned at
        #   the r_max floor by Stage-1 redistribution are excluded from the speed loss.
        R_MAX_TARGET = 3.0
        sp_actual = (seq_link_len / seq_true_tt.clamp_min(1e-3)) / vff_seq_b.clamp_min(1.0)
        sp_actual = sp_actual.clamp(0.0, R_MAX_TARGET)
        sp_actual_clean = (seq_link_len / seq_true_tt.clamp_min(1e-3)) / vff_seq_clean.clamp_min(1.0)
        was_pinned = sp_actual_clean >= (R_MAX_TARGET - 1e-3)
        sp_w_actual = (
            seq_mask
            & (seq_true_tt >= 1.0)
            & (seq_link_len >= 10.0)
            & (~was_pinned)
        ).float()

        # Normalize input (using the modified x_seq with predicted traffic)
        x_seq_norm = (x_seq - scaler_mean.view(1, 1, -1)) / scaler_std.view(1, 1, -1)

        sp_mask = (sp_w > 0) & seq_mask
        Bsz = seq_true_tt.size(0)

        with torch.amp.autocast('cuda', enabled=USE_AMP_MT):
            link_eta_seq, trip_eta, sp_pred, lc_pred, g, delta, eta_phys = linketa_model(
                x_seq_norm, hop_idx, route_flag_nodes, masks_per_rel,
                route_seq_idx, seq_link_len, vff_seq_b, seq_mask,
                r_max=3.0, return_parts=True,
            )

            link_eta_seq = torch.nan_to_num(link_eta_seq, nan=0.0)
            trip_eta = torch.nan_to_num(trip_eta, nan=0.0)
            sp_pred = torch.nan_to_num(sp_pred, nan=0.0)
            lc_pred = torch.nan_to_num(lc_pred, nan=0.0)

            # ── 4. Losses — NEW log1p-space (consistent with new LinkETA base) ──
            pred_seq = link_eta_in_seconds(link_eta_seq, seq_link_len)
            loss_link = link_loss_huber_log_lenweighted(
                pred_seq, seq_true_tt, seq_mask, link_len=seq_link_len, delta=0.3)
            loss_route = route_loss_huber_log(pred_seq, trip_tt, seq_mask, delta=0.3)

            # [2026-06-06] speed loss vs sp_actual (PRIMARY supervision, matches base LinkETA).
            # OLD (historical proxy): smooth_l1(sp_pred[sp_mask], sp_true[sp_mask]) weighted by sp_w.
            denom_sp = sp_w_actual.sum().clamp_min(1.0)
            per_sp = F.smooth_l1_loss(sp_pred, sp_actual, reduction='none', beta=0.1)
            loss_speed = (per_sp * sp_w_actual).sum() / denom_sp

            if seq_mask.any():
                loss_vol = F.smooth_l1_loss(lc_pred[seq_mask], lc_true[seq_mask])
            else:
                loss_vol = torch.zeros((), device=dev)

            # Traffic reconstruction loss — per-trip (v2.0 C5)
            # OLD: loss_traffic = compute_traffic_loss(
            #     predicted_traffic, traffic_real,
            #     batch_day, batch_hour, traffic_observed_mask, ...)
            trip_days, trip_hours = extract_per_trip_time_slot(
                batch["x_seq"], route_seq_idx, seq_mask, time_col_indices)
            loss_traffic = compute_per_trip_traffic_loss(
                pred_traffic_batch, node_edge_idx, traffic_real,
                trip_days, trip_hours, route_seq_idx, seq_mask,
                traffic_observed_mask,
                dyn_scaler_mean=dyn_scaler_mean,
                dyn_scaler_std=dyn_scaler_std,
                n_edges=n_edges,
            )

            # [2026-06-06] 4-task DWA loss (speed/link/route/traffic) + fixed-weight vol.
            #   Matches base LinkETA's {speed,link,route} DWA + 0.10·vol, with traffic
            #   added as the 4th DWA task. The lam_g·|g-1| g-reg is DROPPED (base removed
            #   it in A1; g is still bounded by g_range in the model forward).
            # OLD: w_link·link + w_route·route + 0.05·speed + 0.10·vol
            #      + lam_g·|g-1| + w_traffic·traffic
            final_loss = (
                w_speed * loss_speed
                + w_link * loss_link
                + w_route * loss_route
                + w_traffic * loss_traffic
                + 0.10 * loss_vol
            )

        if train:
            optimizer.zero_grad(set_to_none=True)
            if USE_AMP_MT and scaler_amp is not None:
                scaler_amp.scale(final_loss).backward()
                scaler_amp.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    list(meta_traffic_model.parameters()) + list(linketa_model.parameters()),
                    clip_norm,
                )
                scaler_amp.step(optimizer)
                scaler_amp.update()
            else:
                final_loss.backward()
                nn.utils.clip_grad_norm_(
                    list(meta_traffic_model.parameters()) + list(linketa_model.parameters()),
                    clip_norm,
                )
                optimizer.step()

        # ── 5. Metrics (from loss_functions.py) ────────────────────
        link_stats = link_metrics_micro(
            link_eta_seq, seq_true_tt, seq_mask,
            link_len=seq_link_len, L_min=L_MIN_METRIC, T_min=T_MIN_METRIC,
        )
        route_stats = route_metrics_pooled(link_eta_seq, trip_tt, seq_mask)

        sum_link_loss += float(loss_link.item()) * Bsz
        sum_route_loss += float(loss_route.item()) * Bsz
        sum_speed_loss += float(loss_speed.item()) * Bsz   # [2026-06-06] for DWA
        traffic_val = float(loss_traffic.item())
        if not math.isnan(traffic_val) and traffic_val > 0:
            sum_traffic_rmse += traffic_val * Bsz
            n_batches_with_traffic += Bsz
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

        # [2026-06-07] pool WAPE/MdAPE/P90 building blocks (link + route)
        sum_abs_link += float(link_stats.get("sum_abs", 0.0))
        sum_true_link += float(link_stats.get("sum_true", 0.0))
        _av = link_stats.get("ape_vals")
        if _av is not None and _av.numel() > 0:
            ape_link_list.append(_av)
        sum_abs_route += float(route_stats.get("sum_abs", 0.0))
        sum_true_route += float(route_stats.get("sum_true", 0.0))
        _avr = route_stats.get("ape_vals")
        if _avr is not None and _avr.numel() > 0:
            ape_route_list.append(_avr)

        if sp_mask.any():
            sum_speed_mae += (sp_pred[sp_mask] - sp_true[sp_mask]).abs().mean().item() * int(sp_mask.sum().item())
            N_speed += int(sp_mask.sum().item())
        # [2026-06-06] speed MAE vs the ACTUAL loss target (sp_actual)
        sp_actual_mask = sp_w_actual.bool()
        if sp_actual_mask.any():
            n_act = int(sp_actual_mask.sum().item())
            sum_speed_mae_actual += (sp_pred[sp_actual_mask] - sp_actual[sp_actual_mask]).abs().mean().item() * n_act
            N_speed_actual += n_act
        if seq_mask.any():
            n_valid = int(seq_mask.sum().item())
            sum_vol_mae += (lc_pred[seq_mask] - lc_true[seq_mask]).abs().mean().item() * n_valid
            N_vol += n_valid

    _lpct = finalize_pct_metrics(sum_abs_link, sum_true_link, ape_link_list)
    _rpct = finalize_pct_metrics(sum_abs_route, sum_true_route, ape_route_list)
    return {
        'link_loss': sum_link_loss / max(N_routes_loss, 1),
        'route_loss': sum_route_loss / max(N_routes_loss, 1),                # [2026-06-06]
        'speed_loss': sum_speed_loss / max(N_routes_loss, 1),               # [2026-06-06] for DWA
        'traffic_rmse': (sum_traffic_rmse / max(n_batches_with_traffic, 1))
                        if n_batches_with_traffic > 0 else float('nan'),
        'link_mae': sum_mae_link / max(N_links, 1),
        'link_rmse': math.sqrt(sum_sse_link / max(N_links, 1)) if N_links > 0 else float("nan"),
        'link_mape': (sum_mape_link / max(N_links_mape, 1)) if N_links_mape > 0 else float('nan'),
        'link_wape': _lpct['wape'], 'link_mdape': _lpct['mdape'], 'link_p90': _lpct['ape_p90'],  # [2026-06-07]
        'route_mae': sum_mae_route / max(N_routes_metrics, 1),
        'route_rmse': math.sqrt(sum_sse_route / max(N_routes_metrics, 1)) if N_routes_metrics > 0 else float("nan"),
        'route_mape': (sum_mape_route / max(N_routes_mape, 1)) if N_routes_mape > 0 else float('nan'),
        'route_wape': _rpct['wape'], 'route_mdape': _rpct['mdape'], 'route_p90': _rpct['ape_p90'],  # [2026-06-07]
        'speed_mae': (sum_speed_mae / max(N_speed, 1)) if N_speed else float('nan'),
        'speed_mae_actual': (sum_speed_mae_actual / max(N_speed_actual, 1)) if N_speed_actual else float('nan'),  # [2026-06-06]
        'vol_mae': (sum_vol_mae / max(N_vol, 1)) if N_vol else float('nan'),
    }


# =============================================================================
# FORMATTING
# =============================================================================
def fmt_metrics(m: dict, city: str = "") -> str:
    prefix = f"[{city}] " if city else ""
    parts = []
    for k in ["link_mape", "route_mape", "traffic_rmse",
              "link_mae", "route_mae", "link_rmse", "route_rmse"]:
        v = m.get(k, float("nan"))
        if not math.isnan(v):
            parts.append(f"{k}={v:.4f}")
    return prefix + "  ".join(parts)


# =============================================================================
# STATE HELPERS  (clone / load for two-model setup)
# =============================================================================
def clone_dual_state_cpu(mt_model: MetaTrafficModel, le_model: LinkETA) -> dict:
    """Save both models' state dicts to CPU."""
    return {
        "meta_traffic": {k: v.detach().cpu().clone()
                         for k, v in mt_model.state_dict().items()},
        "linketa": {k: v.detach().cpu().clone()
                    for k, v in le_model.state_dict().items()},
    }


def load_dual_state_cpu(mt_model: MetaTrafficModel, le_model: LinkETA,
                        state: dict, dev: torch.device):
    """Restore both models from a CPU state dict."""
    mt_model.load_state_dict(
        {k: v.to(dev) for k, v in state["meta_traffic"].items()}, strict=True)
    le_model.load_state_dict(
        {k: v.to(dev) for k, v in state["linketa"].items()}, strict=True)


# =============================================================================
# TRAINING LOOP: Source Cities  (MAML / FOMAML)
# =============================================================================
def train_metatraffic_source(
    source_cores: Dict[str, dict],       # {city_key: core_dict}
    meta_traffic_model: MetaTrafficModel,
    linketa_model: LinkETA,
    city_static_feats: Dict[str, torch.Tensor],    # {city_key: [N_city, 31]}
    city_traffic_real: Dict[str, torch.Tensor],     # {city_key: [N_city, 7, 24, 60]}
    city_rel_coo: Dict[str, List[torch.Tensor]],    # {city_key: list of R COO}
    city_scaler_mean: Dict[str, torch.Tensor],       # {city_key: [86]}
    city_scaler_std: Dict[str, torch.Tensor],        # {city_key: [86]}
    city_vff: Dict[str, torch.Tensor],               # {city_key: [N_city]}
    city_dyn_idx: Dict[str, torch.Tensor],           # {city_key: LongTensor of dyn indices}
    city_time_idx: Dict[str, List[int]],             # {city_key: list of time col indices}
    meta_iters: int = 200,
    inner_steps: int = 10,
    inner_lr: float = 1e-4,
    outer_lr: float = 1e-4,
    w_link: float = 0.5,
    w_route: float = 0.5,
    lambda_traffic: float = 0.5,
    patience: int = 50,
    log_every: int = 10,
    ckpt_dir: str = "ckpts_MetaTraffic",
    dwa_T: float = 2.0,
    max_query_batches: int = 5,
    clip_inner: float = 10.0,
    seed: int = 42,
    enable_metrics: bool = True,
    save_every: int = 5,   # [2026-06-06] interval theta0 ckpts (theta0_iter{it}.ckpt; 0=off)
) -> dict:
    """
    MAML (FOMAML) meta-training of MetaTraffic + LinkETA on source cities.

    Matches the diagram:
      ┌─ Meta Iteration ──────────────────────────────────────────┐
      │  For each source city (sequentially):                     │
      │    Inner Loop (Support) — K SGD steps on train batches    │
      │      → Adapted parameters θ'_city                        │
      │    Outer Loop (Query) — evaluate θ'_city on val batches   │
      │      → Compute FOMAML gradients                          │
      │  Aggregate gradients across cities                        │
      │  Meta-optimizer (Adam) step on θ₀                        │
      └──────────────────────────────────────────────────────────┘

    Pure MAML: always uses MetaTraffic predictions (no blending).

    Returns dict with meta_state + history.
    """
    random.seed(seed)
    torch.manual_seed(seed)

    try:
        from torch.func import functional_call
    except Exception:
        from torch.nn.utils.stateless import functional_call

    dev = next(meta_traffic_model.parameters()).device
    source_keys = list(source_cores.keys())

    # ---- Meta-optimizer on θ₀ (the real model parameters) ----
    meta_opt = torch.optim.Adam(
        list(meta_traffic_model.parameters()) + list(linketa_model.parameters()),
        lr=outer_lr,
    )

    # ---- DWA (Dynamic Weight Averaging) — [2026-06-06] 4 tasks: speed, link, route, traffic ----
    # Matches base LinkETA's {speed,link,route} DWA + traffic added as the 4th task.
    # OLD: 3-task DWA (link, route, traffic; speed was a fixed 0.05 auxiliary).
    _dwa_hist_speed = deque(maxlen=2)
    _dwa_hist_link = deque(maxlen=2)
    _dwa_hist_route = deque(maxlen=2)
    _dwa_hist_traffic = deque(maxlen=2)
    _dwa_enabled = (dwa_T > 0)
    # Pre-DWA weights sum to 1.0 (softmax outputs sum to 1.0) to avoid a
    # discontinuity when DWA activates.
    w_speed = 1.0 / 4.0
    w_link = 1.0 / 4.0
    w_route = 1.0 / 4.0
    w_traffic = 1.0 / 4.0

    # ---- Persistent train iterators per city ----
    train_iters = {}
    for ck in source_keys:
        train_iters[ck] = iter(source_cores[ck]["loaders"]["train"])

    def _next_batch(ck):
        try:
            batch = next(train_iters[ck])
        except StopIteration:
            train_iters[ck] = iter(source_cores[ck]["loaders"]["train"])
            batch = next(train_iters[ck])
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(dev)
        return batch

    # ---- Metric accumulator (reused from MetaST pattern) ----
    def _acc_init():
        return dict(
            sum_link_loss=0.0, sum_route_loss=0.0, sum_traffic_rmse=0.0,
            N_loss=0,
            sum_mae_link=0.0, sum_sse_link=0.0, sum_mape_link=0.0,
            N_links=0, N_links_mape=0,
            sum_mae_route=0.0, sum_sse_route=0.0, sum_mape_route=0.0,
            N_routes=0, N_routes_mape=0,
            N_traffic=0,
            # [2026-06-07] WAPE/MdAPE/P90 pooling
            sum_abs_link=0.0, sum_true_link=0.0, ape_link=[],
            sum_abs_route=0.0, sum_true_route=0.0, ape_route=[],
        )

    def _acc_update(acc, pack, batch_size):
        acc["sum_link_loss"] += float(pack["loss_link"].detach()) * batch_size
        acc["sum_route_loss"] += float(pack["loss_route"].detach()) * batch_size
        tval = float(pack["loss_traffic"].detach())
        if not math.isnan(tval) and tval > 0:
            acc["sum_traffic_rmse"] += tval * batch_size
            acc["N_traffic"] += batch_size
        acc["N_loss"] += batch_size

        with torch.no_grad():
            ls = link_metrics_micro(
                pack["pred_seq"].detach(), pack["seq_true"].detach(),
                pack["seq_mask"].detach(), link_len=pack["seq_link_len"].detach(),
                L_min=L_MIN_METRIC, T_min=T_MIN_METRIC,
            )
            rs = route_metrics_pooled(
                pack["pred_seq"].detach(), pack["trip_tt"].detach(),
                pack["seq_mask"].detach(),
            )
            n_l = int(ls.get("n_links", 0))
            if n_l > 0:
                acc["sum_mae_link"] += float(ls["mae"]) * n_l
                acc["sum_sse_link"] += float(ls["rmse"] ** 2) * n_l
                acc["N_links"] += n_l
                n_m = int(ls.get("n_mape", 0))
                if n_m > 0 and not math.isnan(ls.get("mape", float("nan"))):
                    acc["sum_mape_link"] += float(ls["mape"]) * n_m
                    acc["N_links_mape"] += n_m

            n_r = int(rs.get("n_routes", 0))
            if n_r > 0:
                acc["sum_mae_route"] += float(rs["mae"]) * n_r
                acc["sum_sse_route"] += float(rs["rmse"] ** 2) * n_r
                acc["N_routes"] += n_r
                n_mr = int(rs.get("n_mape", 0))
                if n_mr > 0 and not math.isnan(rs.get("mape", float("nan"))):
                    acc["sum_mape_route"] += float(rs["mape"]) * n_mr
                    acc["N_routes_mape"] += n_mr

            # [2026-06-07] pool WAPE/MdAPE/P90 building blocks
            acc["sum_abs_link"] += float(ls.get("sum_abs", 0.0))
            acc["sum_true_link"] += float(ls.get("sum_true", 0.0))
            _av = ls.get("ape_vals")
            if _av is not None and _av.numel() > 0:
                acc["ape_link"].append(_av)
            acc["sum_abs_route"] += float(rs.get("sum_abs", 0.0))
            acc["sum_true_route"] += float(rs.get("sum_true", 0.0))
            _avr = rs.get("ape_vals")
            if _avr is not None and _avr.numel() > 0:
                acc["ape_route"].append(_avr)

    def _acc_finalize(acc):
        o = {}
        o["link_loss"] = acc["sum_link_loss"] / max(acc["N_loss"], 1)
        o["route_loss"] = acc["sum_route_loss"] / max(acc["N_loss"], 1)
        o["traffic_rmse"] = (acc["sum_traffic_rmse"] / max(acc["N_traffic"], 1)
                             ) if acc["N_traffic"] > 0 else float("nan")
        o["link_mae"] = acc["sum_mae_link"] / max(acc["N_links"], 1)
        o["link_rmse"] = (math.sqrt(acc["sum_sse_link"] / max(acc["N_links"], 1))
                          if acc["N_links"] > 0 else float("nan"))
        o["link_mape"] = (acc["sum_mape_link"] / max(acc["N_links_mape"], 1)
                          if acc["N_links_mape"] > 0 else float("nan"))
        o["route_mae"] = acc["sum_mae_route"] / max(acc["N_routes"], 1)
        o["route_rmse"] = (math.sqrt(acc["sum_sse_route"] / max(acc["N_routes"], 1))
                           if acc["N_routes"] > 0 else float("nan"))
        o["route_mape"] = (acc["sum_mape_route"] / max(acc["N_routes_mape"], 1)
                           if acc["N_routes_mape"] > 0 else float("nan"))
        # [2026-06-07] pooled robust % metrics
        _lp = finalize_pct_metrics(acc["sum_abs_link"], acc["sum_true_link"], acc["ape_link"])
        _rp = finalize_pct_metrics(acc["sum_abs_route"], acc["sum_true_route"], acc["ape_route"])
        o["link_wape"], o["link_mdape"], o["link_p90"] = _lp["wape"], _lp["mdape"], _lp["ape_p90"]
        o["route_wape"], o["route_mdape"], o["route_p90"] = _rp["wape"], _rp["mdape"], _rp["ape_p90"]
        return o

    # ------------------------------------------------------------------
    # Stateless forward + loss  (works with functional_call on both models)
    # ------------------------------------------------------------------
    def _forward_losses(mt_state, mt_buffs, le_state, le_buffs,
                        batch, ck, meta_iter=0):
        """
        Full MetaTraffic → inject → LinkETA → losses  in a stateless way.

        Parameters are passed explicitly so torch.autograd.grad can
        differentiate w.r.t. the fast-weight tensors.

        Pure MAML: always uses MetaTraffic predictions (no blending).

        Returns dict with task_loss and all component tensors.
        """
        # v3.1: inject per-city normalization stats into buffers
        if ck in meta_traffic_model._per_city_norm:
            mu, sigma = meta_traffic_model._per_city_norm[ck]
            mt_buffs = dict(mt_buffs)  # shallow copy to avoid mutating shared ref
            mt_buffs['static_mean'] = mu.to(dev)
            mt_buffs['static_std'] = sigma.to(dev)

        static_feats = city_static_feats[ck]
        traffic_real = city_traffic_real[ck]
        rel_coo      = city_rel_coo[ck]
        scaler_mean  = city_scaler_mean[ck]
        scaler_std   = city_scaler_std[ck]
        vff_city     = city_vff[ck]
        dyn_idx      = city_dyn_idx[ck]
        time_idx     = city_time_idx[ck]
        n_edges      = static_feats.shape[0]

        # 0) Per-node time and edge indices from batch (v2.0 C4)
        # OLD:
        #   batch_day, batch_hour = extract_batch_time_slot(batch, time_idx)
        #   time_tensor = build_time_features_for_batch(batch, time_idx, n_edges, dev)
        #   predicted_traffic, _h = functional_call(
        #       meta_traffic_model, {**mt_state, **mt_buffs},
        #       args=(static_feats, time_tensor, rel_coo))
        #   predicted_traffic = torch.nan_to_num(...)
        #   node_edge_idx = batch["node_edge_idx"]
        #   x_seq = batch["x_seq"].clone()
        #   flat_edge_idx = node_edge_idx.clamp(0, n_edges - 1)
        #   pred_batch = predicted_traffic[flat_edge_idx]
        #   x_seq[:, :, dyn_idx] = pred_batch.to(x_seq.dtype)
        node_edge_idx = batch["node_edge_idx"]               # [B, M]
        x_seq = batch["x_seq"].clone()                        # [B, M, 86]
        B_fl, M_fl, _ = x_seq.shape

        flat_edge_idx = node_edge_idx.reshape(-1).clamp(0, n_edges - 1)  # [B*M]
        flat_time = x_seq.reshape(B_fl * M_fl, -1)[:, time_idx]          # [B*M, 8]

        # 1) MetaTraffic forward (stateless) — encode + predict (C1, C3)
        predicted_traffic_flat, _h = functional_call(
            meta_traffic_model,
            {**mt_state, **mt_buffs},
            args=(static_feats, flat_time, rel_coo, flat_edge_idx),
        )
        predicted_traffic_flat = torch.nan_to_num(
            predicted_traffic_flat, nan=0.0, posinf=1e4, neginf=-1e4)

        pred_traffic_batch = predicted_traffic_flat.reshape(B_fl, M_fl, -1)

        # 2) Inject MetaTraffic predictions into batch (C4)
        x_seq[:, :, dyn_idx] = pred_traffic_batch.to(x_seq.dtype)

        # 3) Prepare LinkETA inputs
        masks_per_rel = [m.to(dev) for m in batch["masks_per_rel"]]
        route_seq_idx = batch["route_seq_idx"]
        seq_true_tt = torch.nan_to_num(batch["seq_true_tt"], nan=0.0)
        seq_link_len = torch.nan_to_num(batch["seq_link_len"], nan=0.0).clamp_min(1e-3)
        seq_mask = batch["seq_mask"].bool()
        trip_tt = torch.nan_to_num(batch["trip_tt"], nan=0.0)
        hop_idx = batch["hop_idx"]
        route_flag = batch["route_flag"]

        sp_true_node = torch.nan_to_num(batch["sp_ratio_agg"], nan=0.0)
        lc_true_node = torch.nan_to_num(batch["logcnt1h_node"], nan=0.0)
        sp_w_node = torch.nan_to_num(batch["speed_w_node"], nan=0.0)
        _gather = lambda v: torch.gather(v, 1, route_seq_idx)
        sp_true = _gather(sp_true_node)
        lc_true = _gather(lc_true_node)
        sp_w = _gather(sp_w_node)

        vff_seq_b = vff_city[node_edge_idx]
        vff_seq_b = torch.gather(vff_seq_b, 1, route_seq_idx)

        # [2026-06-06] Per-link sp_actual TARGET (no v_ff noise in the inner loop, so
        #   clean == noisy). Mirrors base LinkETA one_epoch's speed supervision.
        R_MAX_TARGET = 3.0
        sp_actual = (seq_link_len / seq_true_tt.clamp_min(1e-3)) / vff_seq_b.clamp_min(1.0)
        sp_actual = sp_actual.clamp(0.0, R_MAX_TARGET)
        was_pinned = sp_actual >= (R_MAX_TARGET - 1e-3)
        sp_w_actual = (
            seq_mask
            & (seq_true_tt >= 1.0)
            & (seq_link_len >= 10.0)
            & (~was_pinned)
        ).float()

        x_seq_norm = (x_seq - scaler_mean.view(1, 1, -1)) / scaler_std.view(1, 1, -1)
        sp_mask = (sp_w > 0) & seq_mask

        # 4) LinkETA forward (stateless)
        le_out = functional_call(
            linketa_model,
            {**le_state, **le_buffs},
            args=(x_seq_norm, hop_idx, route_flag, masks_per_rel,
                  route_seq_idx, seq_link_len, vff_seq_b, seq_mask),
            kwargs=dict(r_max=3.0, return_parts=True),
        )
        link_eta_seq, trip_eta, sp_pred, lc_pred, g, delta, eta_phys = le_out

        link_eta_seq = torch.nan_to_num(link_eta_seq, nan=0.0)
        sp_pred = torch.nan_to_num(sp_pred, nan=0.0)
        lc_pred = torch.nan_to_num(lc_pred, nan=0.0)

        # 5) Losses
        pred_seq = link_eta_in_seconds(link_eta_seq, seq_link_len)
        loss_link = link_loss_huber_log_lenweighted(
            pred_seq, seq_true_tt, seq_mask, link_len=seq_link_len, delta=0.3)
        loss_route = route_loss_huber_log(pred_seq, trip_tt, seq_mask, delta=0.3)

        # [2026-06-06] speed loss vs sp_actual (PRIMARY supervision, matches base LinkETA).
        # OLD (historical proxy): smooth_l1(sp_pred[sp_mask], sp_true[sp_mask]) weighted by sp_w.
        denom_sp = sp_w_actual.sum().clamp_min(1.0)
        per_sp = F.smooth_l1_loss(sp_pred, sp_actual, reduction="none", beta=0.1)
        loss_speed = (per_sp * sp_w_actual).sum() / denom_sp

        if seq_mask.any():
            loss_vol = F.smooth_l1_loss(lc_pred[seq_mask], lc_true[seq_mask])
        else:
            loss_vol = torch.zeros((), device=dev)

        dyn_scaler_mean = scaler_mean[dyn_idx]
        dyn_scaler_std = scaler_std[dyn_idx]
        traffic_mask = (traffic_real.abs().sum(dim=(-1, -2, -3)) > 0)
        # v2.0 C5: Per-trip traffic loss
        # OLD: loss_traffic = compute_traffic_loss(
        #     predicted_traffic, traffic_real, batch_day, batch_hour,
        #     traffic_mask, dyn_scaler_mean=..., dyn_scaler_std=...)
        trip_days, trip_hours = extract_per_trip_time_slot(
            batch["x_seq"], route_seq_idx, seq_mask, time_idx)
        loss_traffic = compute_per_trip_traffic_loss(
            pred_traffic_batch, node_edge_idx, traffic_real,
            trip_days, trip_hours, route_seq_idx, seq_mask,
            traffic_mask,
            dyn_scaler_mean=dyn_scaler_mean,
            dyn_scaler_std=dyn_scaler_std,
            n_edges=n_edges,
        )

        # [2026-06-06] 4-task DWA loss (speed/link/route/traffic) + fixed-weight vol.
        #   Matches base LinkETA's {speed,link,route} DWA + 0.10·vol, with traffic the
        #   4th DWA task. lam_g·|g-1| g-reg DROPPED (base removed it in A1; g still
        #   bounded by g_range in the model forward).
        # OLD: w_link·link + w_route·route + 0.05·speed + 0.10·vol
        #      + lam_g·|g-1| + w_traffic·traffic
        task_loss = (
            w_speed * loss_speed
            + w_link * loss_link
            + w_route * loss_route
            + w_traffic * loss_traffic
            + 0.10 * loss_vol
        )

        return dict(
            task_loss=task_loss,
            loss_link=loss_link,
            loss_route=loss_route,
            loss_speed=loss_speed,
            loss_traffic=loss_traffic,
            pred_seq=link_eta_seq,
            seq_true=seq_true_tt,
            seq_mask=seq_mask,
            seq_link_len=seq_link_len,
            trip_tt=trip_tt,
        )

    # ------------------------------------------------------------------
    # Print header
    # ------------------------------------------------------------------
    mt_n = sum(p.numel() for p in meta_traffic_model.parameters())
    le_n = sum(p.numel() for p in linketa_model.parameters())
    print(f"\n{'='*80}")
    print(f"  MetaTraffic + LinkETA  —  MAML (FOMAML) Meta-Training")
    print(f"{'='*80}")
    print(f"  Source cities: {source_keys}")
    print(f"  MetaTraffic params: {mt_n:,}  |  LinkETA params: {le_n:,}  |  Total: {mt_n+le_n:,}")
    print(f"  Meta iters: {meta_iters}  |  Inner steps (K): {inner_steps}  |  Inner LR: {inner_lr}")
    print(f"  Outer LR (Adam): {outer_lr}  |  λ_traffic(init): {lambda_traffic}")
    print(f"  Inner grad clip: {clip_inner}")
    print(f"  DWA (4-task: speed/link/route/traffic): {'ON (T={:.1f})'.format(dwa_T) if _dwa_enabled else 'OFF'}")
    print(f"  max_query_batches: {max_query_batches}  |  patience: {patience}")
    print(f"  Losses: speed=SmoothL1(sp_actual), link/route=log1p-Huber, traffic=RMSE(z-norm); +0.10*vol")
    print(f"{'='*80}\n")

    best_loss = float("inf")
    best_state = None
    best_meta_iter = None       # [2026-06-07] iter of min meta_loss_avg (selected model)
    best_summary = None         # [2026-06-07] per-city query metrics @ that iter
    bad = 0
    history = {"meta_loss": [], "support": [], "query": []}

    # ---- Checkpoint manager for saving to disk ----
    _combined_model = _CombinedModel(meta_traffic_model, linketa_model)
    _ckpt_mgr = CheckpointManager(dirpath=ckpt_dir, max_keep=3)
    scaler_hash = _scaler_fingerprint(city_scaler_mean, city_scaler_std)
    print(f"  Scaler fingerprint: {scaler_hash}")
    print(f"  Checkpoint dir: {ckpt_dir}")

    for it in range(1, meta_iters + 1):
        do_log = enable_metrics and ((it % log_every == 0) or (it <= 3))

        # ---- DWA: recompute task weights (4-task: speed, link, route, traffic) ----
        # OLD: 2-task DWA (link, route only)
        # if _dwa_enabled and len(_dwa_hist_link) >= 2:
        #     r_link = float(_dwa_hist_link[-1] / (_dwa_hist_link[-2] + 1e-8))
        #     r_route = float(_dwa_hist_route[-1] / (_dwa_hist_route[-2] + 1e-8))
        #     e_l = math.exp(r_link / dwa_T)
        #     e_r = math.exp(r_route / dwa_T)
        #     s = e_l + e_r
        #     w_link = e_l / s
        #     w_route = e_r / s
        # [2026-06-06] 4-task DWA — softmax over (speed, link, route, traffic) ratios
        if _dwa_enabled and len(_dwa_hist_link) >= 2:
            r_speed = float(_dwa_hist_speed[-1] / (_dwa_hist_speed[-2] + 1e-8))
            r_link = float(_dwa_hist_link[-1] / (_dwa_hist_link[-2] + 1e-8))
            r_route = float(_dwa_hist_route[-1] / (_dwa_hist_route[-2] + 1e-8))
            r_traffic = float(_dwa_hist_traffic[-1] / (_dwa_hist_traffic[-2] + 1e-8))
            # Clamp DWA ratios to prevent exp() overflow when a task loss drops
            #   near-zero between intervals
            _DWA_RATIO_MAX = 10.0
            r_speed = max(min(r_speed, _DWA_RATIO_MAX), -_DWA_RATIO_MAX)
            r_link = max(min(r_link, _DWA_RATIO_MAX), -_DWA_RATIO_MAX)
            r_route = max(min(r_route, _DWA_RATIO_MAX), -_DWA_RATIO_MAX)
            r_traffic = max(min(r_traffic, _DWA_RATIO_MAX), -_DWA_RATIO_MAX)
            e_s = math.exp(r_speed / dwa_T)
            e_l = math.exp(r_link / dwa_T)
            e_r = math.exp(r_route / dwa_T)
            e_t = math.exp(r_traffic / dwa_T)
            s = e_s + e_l + e_r + e_t
            w_speed = e_s / s
            w_link = e_l / s
            w_route = e_r / s
            w_traffic = e_t / s

        meta_opt.zero_grad(set_to_none=True)
        meta_loss_sum = 0.0

        # DWA accumulators (4-task)
        _dwa_sum_speed = 0.0
        _dwa_sum_link = 0.0
        _dwa_sum_route = 0.0
        _dwa_sum_traffic = 0.0
        _dwa_n = 0

        # θ₀ parameter snapshots (references to actual params)
        mt_params0 = dict(meta_traffic_model.named_parameters())
        mt_buffs0 = dict(meta_traffic_model.named_buffers())
        le_params0 = dict(linketa_model.named_parameters())
        le_buffs0 = dict(linketa_model.named_buffers())

        # Collect all param names
        mt_names = list(mt_params0.keys())
        le_names = list(le_params0.keys())
        all_names = [("mt", n) for n in mt_names] + [("le", n) for n in le_names]

        # Gradient accumulator (zero)
        grads_acc = {}
        for tag, n in all_names:
            src = mt_params0 if tag == "mt" else le_params0
            grads_acc[(tag, n)] = torch.zeros_like(src[n])

        # [2026-06-07] per-city query metrics for THIS iter (selected-model summary)
        cur_iter_metrics = {}
        for ck in source_keys:
            meta_traffic_model.train()
            linketa_model.train()

            # ── Clone fast params for this city ──
            mt_fast = {n: mt_params0[n].clone() for n in mt_names}
            le_fast = {n: le_params0[n].clone() for n in le_names}

            # ── Inner Loop (Support): K SGD steps ──
            support_acc = _acc_init() if do_log else None

            for _s in range(inner_steps):
                batch = _next_batch(ck)

                pack = _forward_losses(mt_fast, mt_buffs0, le_fast, le_buffs0,
                                       batch, ck, meta_iter=it)

                # NaN / Inf guard — skip this step if loss is bad
                _loss_val = pack["task_loss"]
                if torch.isnan(_loss_val) or torch.isinf(_loss_val):
                    continue

                # Gradient w.r.t. all fast params
                fast_vals = [mt_fast[n] for n in mt_names] + \
                            [le_fast[n] for n in le_names]
                grads = torch.autograd.grad(
                    _loss_val, fast_vals,
                    create_graph=False, retain_graph=False,
                    allow_unused=True,   # proto_alpha_logit unused when proto_active=False
                )

                # ---- Clip inner-loop gradients by global norm ----
                grad_norms_sq = sum(
                    g.detach().norm().square() for g in grads if g is not None
                )
                total_norm = torch.sqrt(grad_norms_sq + 1e-12)
                clip_coef = clip_inner / (total_norm + 1e-6)
                if clip_coef < 1.0:
                    grads = tuple(
                        g * clip_coef if g is not None else g for g in grads
                    )

                # SGD update on fast params (skip None grads from unused params)
                idx = 0
                for n in mt_names:
                    if grads[idx] is not None:
                        mt_fast[n] = mt_fast[n] - inner_lr * grads[idx]
                    idx += 1
                for n in le_names:
                    if grads[idx] is not None:
                        le_fast[n] = le_fast[n] - inner_lr * grads[idx]
                    idx += 1

                if do_log:
                    _acc_update(support_acc, pack, pack["seq_true"].size(0))

            # ── Outer Loop (Query): evaluate adapted params on val ──
            # [2026-06-07] Always accumulate query metrics (cheap: reuses the query
            # forward already run for qloss) so the selected-model summary can
            # snapshot BOTH cities at the argmin-meta_loss iteration.
            query_acc = _acc_init()
            qloss_sum = 0.0
            qcount = 0
            _city_qs = 0.0   # [2026-06-06] speed query-loss accumulator (4-task DWA)
            _city_ql = 0.0
            _city_qr = 0.0
            _city_qt = 0.0
            _city_qn = 0

            val_loader = source_cores[ck]["loaders"]["val"]
            for b_idx, batch in enumerate(val_loader):
                if max_query_batches is not None and b_idx >= max_query_batches:
                    break
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(dev)

                pack = _forward_losses(mt_fast, mt_buffs0, le_fast, le_buffs0,
                                       batch, ck, meta_iter=it)
                qloss_sum = qloss_sum + pack["task_loss"]
                qcount += 1

                _city_qs += float(pack["loss_speed"].detach())
                _city_ql += float(pack["loss_link"].detach())
                _city_qr += float(pack["loss_route"].detach())
                _city_qt += float(pack["loss_traffic"].detach())
                _city_qn += 1

                _acc_update(query_acc, pack, pack["seq_true"].size(0))

            qloss = qloss_sum / max(qcount, 1)

            # NaN / Inf guard — skip city if query loss is bad
            if torch.isnan(qloss) or torch.isinf(qloss):
                if do_log:
                    print(f"  MetaIter {it:04d} | {ck} | SKIPPED (query loss NaN/Inf)")
                continue

            meta_loss_sum += float(qloss.detach())

            _dwa_sum_speed += _city_qs / max(_city_qn, 1)
            _dwa_sum_link += _city_ql / max(_city_qn, 1)
            _dwa_sum_route += _city_qr / max(_city_qn, 1)
            _dwa_sum_traffic += _city_qt / max(_city_qn, 1)
            _dwa_n += 1

            # ── FOMAML gradients: ∇_θ' L_query, mapped back to θ₀ ──
            fast_vals = [mt_fast[n] for n in mt_names] + \
                        [le_fast[n] for n in le_names]
            qgrads = torch.autograd.grad(
                qloss, fast_vals,
                create_graph=False, retain_graph=False,
                allow_unused=True,   # proto_alpha_logit unused when proto_active=False
            )

            # Check for NaN in query gradients
            has_nan = any(torch.isnan(g).any() for g in qgrads if g is not None)
            if has_nan:
                if do_log:
                    print(f"  MetaIter {it:04d} | {ck} | SKIPPED (NaN in query grads)")
                continue

            idx = 0
            for n in mt_names:
                if qgrads[idx] is not None:
                    grads_acc[("mt", n)] = grads_acc[("mt", n)] + qgrads[idx]
                idx += 1
            for n in le_names:
                if qgrads[idx] is not None:
                    grads_acc[("le", n)] = grads_acc[("le", n)] + qgrads[idx]
                idx += 1

            # [2026-06-07] Finalize query metrics EVERY iter so the selected-model
            # summary can snapshot both cities at the argmin-meta_loss iteration.
            qry_m = _acc_finalize(query_acc)
            cur_iter_metrics[ck] = qry_m

            # Per-city logging
            if do_log:
                sup_m = _acc_finalize(support_acc)
                print(f"  MetaIter {it:04d} | {ck} | DWA w_speed={w_speed:.3f} w_link={w_link:.3f} w_route={w_route:.3f} w_traffic={w_traffic:.3f}")
                print(f"    Support(K={inner_steps}): {fmt_metrics(sup_m, city=ck)}")
                print(f"    Query(val):    {fmt_metrics(qry_m, city=ck)}")
                # [2026-06-06] per-source val metrics for training_history.csv
                history.setdefault("source_rows", []).append({
                    "meta_iter": it, "city": ck,
                    "link_mae": qry_m.get("link_mae"), "link_rmse": qry_m.get("link_rmse"),
                    "link_mape": qry_m.get("link_mape"),
                    "route_mae": qry_m.get("route_mae"), "route_rmse": qry_m.get("route_rmse"),
                    "route_mape": qry_m.get("route_mape"),
                    "link_wape": qry_m.get("link_wape"), "link_mdape": qry_m.get("link_mdape"),
                    "link_p90": qry_m.get("link_p90"),
                    "route_wape": qry_m.get("route_wape"), "route_mdape": qry_m.get("route_mdape"),
                    "route_p90": qry_m.get("route_p90"),   # [2026-06-07] full metric parity
                })

        # ── Average grads over cities → assign to θ₀ → Adam step ──
        nC = max(len(source_keys), 1)
        # v3.2 FIX: Average FOMAML gradients over cities (was raw sum → doubled effective LR)
        for n, p in meta_traffic_model.named_parameters():
            p.grad = grads_acc[("mt", n)] / nC
        for n, p in linketa_model.named_parameters():
            p.grad = grads_acc[("le", n)] / nC

        torch.nn.utils.clip_grad_norm_(
            list(meta_traffic_model.parameters()) + list(linketa_model.parameters()),
            10.0,
        )
        meta_opt.step()

        meta_loss_avg = meta_loss_sum / nC
        history["meta_loss"].append(meta_loss_avg)

        # DWA history (4-task)
        if _dwa_enabled and _dwa_n > 0:
            _dwa_hist_speed.append(_dwa_sum_speed / _dwa_n)
            _dwa_hist_link.append(_dwa_sum_link / _dwa_n)
            _dwa_hist_route.append(_dwa_sum_route / _dwa_n)
            _dwa_hist_traffic.append(_dwa_sum_traffic / _dwa_n)

        if not enable_metrics and ((it % log_every == 0) or (it <= 3)):
            print(f"MetaIter {it:04d} | MetaLoss={meta_loss_avg:.6f} "
                  f"| DWA w_speed={w_speed:.3f} w_link={w_link:.3f} w_route={w_route:.3f} w_traffic={w_traffic:.3f}")

        # Early stopping
        if meta_loss_avg < best_loss:
            best_loss = meta_loss_avg
            best_meta_iter = it
            # [2026-06-07] snapshot both cities' query metrics at the selected iter
            best_summary = {c: dict(m) for c, m in cur_iter_metrics.items()}
            best_state = clone_dual_state_cpu(meta_traffic_model, linketa_model)
            bad = 0
            # Save best checkpoint to disk
            _ckpt_mgr.save_best(
                model=_combined_model, optimizer=meta_opt, scheduler=None,
                epoch=it, step=it, best_metric=best_loss,
                extra={"method": "MetaTraffic", "meta_iter": it,
                       "scaler_hash": scaler_hash},
            )
        else:
            bad += 1

        # Save last checkpoint every iteration
        _ckpt_mgr.save_last(
            model=_combined_model, optimizer=meta_opt, scheduler=None,
            epoch=it, step=it, best_metric=best_loss,
            extra={"method": "MetaETA", "meta_iter": it,
                   "scaler_hash": scaler_hash},
        )

        # [2026-06-06] interval theta0 checkpoint for the meta-iter sweep
        if save_every and (it % save_every == 0):
            _ckpt_mgr.save(
                os.path.join(ckpt_dir, f"theta0_iter{it}.ckpt"),
                model=_combined_model, optimizer=None, scheduler=None,
                epoch=it, step=it, best_metric=best_loss,
                extra={"method": "MetaETA", "meta_iter": it, "scaler_hash": scaler_hash},
            )

        if bad >= patience:
            print(f"\nEarly stopping at MetaIter {it} | best MetaLoss={best_loss:.6f}")
            if best_state is not None:
                load_dual_state_cpu(meta_traffic_model, linketa_model, best_state, dev)
            break

    print(f"\n{'='*80}")
    print(f"  Meta-training complete.  Best MetaLoss: {best_loss:.6f}")
    print(f"  Checkpoints saved to: {ckpt_dir}/")
    print(f"{'='*80}")

    # [2026-06-07] Persist per-source training history (CSV) + SELECTED-MODEL
    # summary: both cities' query metrics at the iteration with min meta_loss_avg
    # (the deployed theta_0), instead of per-city argmin route_mape.
    write_training_history(ckpt_dir, history.get("source_rows", []), method="MetaETA",
                           best_summary=best_summary, best_meta_iter=best_meta_iter)

    final_state = clone_dual_state_cpu(meta_traffic_model, linketa_model)
    history["meta_state"] = final_state
    return history


# =============================================================================
# HELPER: Combined model wrapper for checkpointing
# =============================================================================
class _CombinedModel(nn.Module):
    """Thin wrapper to save/load MetaTraffic + LinkETA as one state_dict."""

    def __init__(self, meta_traffic: MetaTrafficModel, linketa: LinkETA):
        super().__init__()
        self.meta_traffic = meta_traffic
        self.linketa = linketa

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Use meta_traffic and linketa separately.")


# =============================================================================
# CITY-SIMILARITY WEIGHTING: Cosine similarity → softmax weights
# =============================================================================
def compute_city_similarity_weights(
    source_descriptors: Dict[str, torch.Tensor],
    target_descriptor: torch.Tensor,
    temperature: float = 1.0,
) -> Dict[str, float]:
    """
    Compute per-source-city weights based on cosine similarity to the target.

    Uses city-level graph descriptors (from structural_clustering.py's
    compute_city_descriptor) to find which source cities are structurally
    more similar to the target.

    Args:
        source_descriptors: {city_key: descriptor tensor [D]}
        target_descriptor:  target descriptor tensor [D]
        temperature: softmax temperature (default 1.0). Lower = sharper.

    Returns:
        {city_key: float} normalized weights (sum to 1.0).
    """
    # v3.2 FIX: z-normalize descriptors per-feature across all cities
    # before cosine similarity. Without this, features with huge magnitude
    # (e.g. avg_path_length ~4000) dominate the direction, making all
    # cities appear equally similar (cosine ≈ 1.0 for everyone).
    all_keys = list(source_descriptors.keys())
    all_descs = torch.stack([source_descriptors[k].float() for k in all_keys]
                            + [target_descriptor.float()])  # [nC+1, D]
    mu = all_descs.mean(0)            # [D]
    std = all_descs.std(0).clamp_min(1e-8)  # [D]
    normed = (all_descs - mu) / std   # [nC+1, D]  z-scored per feature

    target = normed[-1]               # last row = target
    target_norm = target / target.norm().clamp_min(1e-8)

    similarities = {}
    for i, ck in enumerate(all_keys):
        src = normed[i]
        src_norm = src / src.norm().clamp_min(1e-8)
        cos_sim = float(torch.dot(src_norm, target_norm))
        similarities[ck] = cos_sim

    # Softmax over similarities with temperature
    keys = list(similarities.keys())
    sims = torch.tensor([similarities[k] for k in keys])
    weights = torch.softmax(sims / temperature, dim=0)

    result = {k: float(weights[i]) for i, k in enumerate(keys)}

    # Sort by weight (descending) for readability
    ranked = sorted(result.items(), key=lambda x: x[1], reverse=True)
    closest_city, closest_weight = ranked[0]

    print(f"\n{'─'*60}")
    print(f"[MetaTraffic] City-similarity weights (cosine sim → softmax)")
    print(f"{'─'*60}")
    print(f"  {'City':<15} {'Cosine Sim':>12} {'Weight':>10}  ")
    print(f"  {'─'*42}")
    for ck, w in ranked:
        marker = " ← closest" if ck == closest_city else ""
        print(f"  {ck:<15} {similarities[ck]:>+12.4f} {w:>10.4f}{marker}")
    print(f"  {'─'*42}")
    print(f"  Closest to target: {closest_city} "
          f"(weight={closest_weight:.4f}, "
          f"cosine_sim={similarities[closest_city]:+.4f})")
    print(f"  NOTE: weights are NOT equal — derived from structural similarity")
    print(f"{'─'*60}\n")

    return result
