#!/usr/bin/env python
# coding: utf-8
"""
MetaETA.py - whole-city-graph ETA by propagating a few trips' observed speed
ratios over the relation graph.  Implements the architecture in
MetaETA_ARCHITECTURE.txt PART 7 (the corrected, leaner design).

================================================================================
STAGE 1 (this file): the MODEL + its OWN standalone episodic training (NO MAML).
  Goal: validate that the architecture learns on a single city. If route MAPE
  trains down here, the architecture works; STAGE 2 (separate) then wraps this
  same model in the MetaTraffic-style MAML/episodic-across-cities setup.

WHAT IS REUSED FROM THE EXISTING CODE (as-is, same variable names):
  - init_linketa(...)            -> the per-city bundle loader (rel_edges, vff_vec,
                                    scalers, device).  We use its `base` dict and
                                    ignore base["model"] (we build our own).
  - CFG, set_seed                -> from LinkETA (same feature schema / seeding).
  - _scatter_softmax             -> from MetaTraffic (AMP-safe sparse softmax).
  - loss_functions.*             -> the SAME losses/metrics as base LinkETA:
       link_loss_huber_log_lenweighted, route_loss_huber_log,
       link_metrics_micro, route_metrics_pooled, finalize_pct_metrics,
       selection_metric, link_eta_in_seconds.
  - checkpoint.CheckpointManager -> same best/last ckpt mechanism.
  - DWA over (speed, link, route)-> same 3-task balancing idea as LinkETA one_epoch.

WHY THE PER-TRIP LinkETA DATASET/one_epoch ARE *NOT* REUSED VERBATIM:
  LinkETA builds a per-trip LOCAL graph (route + sampled off-route neighbours) and
  processes each trip in isolation.  MetaETA is a CROSS-TRIP, WHOLE-GRAPH model:
  support trips supply observed sp_actual on the links they drove; Stage 2 runs
  ONCE per (regime) over the FULL relation graph; query trips then just gather.
  A per-trip local graph has nowhere to hold "what other trips observed", so the
  episode builder + sparse whole-graph attention here REPLACE TripDatasetLinkETA /
  collate / the LinkETA forward.  Everything else (losses, metrics, DWA, ckpt) is
  the existing machinery.

PART-7 MAPPING:
  Stage 0' read regime-specific sp_actual ......... build_regime_observations()
  Stage 1' regime emb + reachability r_i .......... MetaETAModel input + _reach_hops()
  Stage 2' ONE sparse spatial-relational pass / regime, bounded depth + JK ... MetaETAModel
  Stage 3' per-trip ETA, static-only tight delta, TARGET-ANCHORED g_tau ...... _stage3_eta()
  Losses   L_speed(HIDDEN only) + L_link(UNOBSERVED only) + L_route, DWA ...... one_episode()
  No-leakage: hidden links never see their own value; query scored on unobserved.

STAGE-1 SIMPLIFYING DECISIONS (flagged to the user; easy to revisit):
  D1. regime = 5 coarse buckets from departure hour + weekend (see time_regime()).
  D2. one regime per EPISODE (matches "run Stage 2 once per regime").
  D3. masking = uniform-random hide fraction p ~ U(p_lo, p_hi); the DISTANCE-
      stratified mask + curriculum from PART 7 are deferred to a later pass.
  D4. observed links use the MEASURED ratio (hard switch, A5); the confidence
      blend is deferred. g_tau applied to PREDICTED links only.
  D5. static features = static_feats_31.pt (same tensor MetaTraffic uses).

Run (STAGE-1 standalone, heavy -> run yourself):
  conda activate ETA_V2
  set PYTHONIOENCODING=utf-8   (Windows)
  python MetaETA.py --city porto --epochs 40                 # all trips (default)
  python MetaETA.py --city porto --epochs 40 --trips 2000    # use only 2000 trips
Quick logic check (no data, synthetic graph):
  python MetaETA.py --smoke
"""

import os
import math
import json
import time
import random
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

# ---- reuse existing project machinery (as-is) -------------------------------
from LinkETA import init_linketa, CFG, set_seed
from MetaTraffic import _scatter_softmax
from loss_functions import (
    link_loss_huber_log_lenweighted,
    route_loss_huber_log,
    link_metrics_micro,
    route_metrics_pooled,
    finalize_pct_metrics,
    selection_metric,
    link_eta_in_seconds,
    L_MIN_METRIC,
    T_MIN_METRIC,
)
from checkpoint import CheckpointManager


# =============================================================================
# Constants (kept aligned with base LinkETA)
# =============================================================================
R_MAX_TARGET = 3.0           # sp_hat in (0, R_MAX]; matches base LinkETA r_max
DEFAULT_REGIME_K = 7         # data-driven regimes (KMeans over the weekly clock);
                             # congestion-ordered (0=slowest); validate k per run.

# per-city data layout (mirrors main.py CITY_CONFIGS; relative to this file)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BASE_DIR, "meta_learning_data")


def city_paths(city: str):
    d = os.path.join(DATA_ROOT, city)
    return dict(
        precomputed_dir=d,
        trips_path=os.path.join(d, f"df_all_{city}.csv"),
        checkpoint_path=os.path.join(d, f"ckpts_LinkETA_{city}", "best.ckpt"),
        static_feats_31=os.path.join(d, "static_feats_31.pt"),
        ckpt_out=os.path.join(d, f"ckpts_MetaETA_standalone_{city}"),
    )


# =============================================================================
# Time regime (DATA-DRIVEN): cluster the weekly clock (7 dow x 24 hour) by its
# congestion signature, like make_traffic_profile.py.  Each (dow,hour) cell ->
# one regime id, ORDERED BY CONGESTION (0 = slowest / most congested) so the
# regime embedding means "severity", coherent across cities for transfer.
# Built from the training city (Stage 1) / pooled sources (Stage 2) and applied
# to ANY city/target purely by wall-clock — needs no target traffic data.
# =============================================================================
def _z(a):
    a = np.asarray(a, dtype=np.float64)
    return (a - a.mean()) / (a.std() + 1e-9)


def build_regime_map(trips, vff_np, k=DEFAULT_REGIME_K, seed=42):
    """Return (regime_grid[7,24] int congestion-ordered, k_eff).
    Cell signature = mean sp_actual of the links of trips departing in that
    (dow,hour) cell.  Unobserved cells inherit the nearest observed cell on the
    circular clock, so any wall-clock (incl. a target's) maps to a regime."""
    cell_sum = np.zeros((7, 24)); cell_cnt = np.zeros((7, 24))
    for t in trips:
        d, h = t["dow"], t["hour"]
        e = t["edges"].numpy(); ln = t["lens"].numpy(); tt = t["tts"].numpy()
        valid = (tt >= 1.0) & (ln >= 10.0)
        if not valid.any():
            continue
        vff_e = np.maximum(vff_np[e[valid]], 1.0)
        sp = np.clip((ln[valid] / np.maximum(tt[valid], 1e-3)) / vff_e, 0.0, R_MAX_TARGET)
        cell_sum[d, h] += sp.sum(); cell_cnt[d, h] += sp.size
    observed = cell_cnt > 0
    if int(observed.sum()) < 2:
        return np.zeros((7, 24), dtype=int), 1          # degenerate: single regime
    s = cell_sum / np.maximum(cell_cnt, 1.0)
    dows, hrs = np.where(observed)
    feats = np.column_stack([
        _z(s[observed]),
        np.sin(2 * np.pi * hrs / 24), np.cos(2 * np.pi * hrs / 24),
        np.sin(2 * np.pi * dows / 7), np.cos(2 * np.pi * dows / 7),
    ])
    k_eff = int(min(k, len(feats)))
    labels = KMeans(n_clusters=k_eff, random_state=seed, n_init=10).fit_predict(feats)
    # congestion-order clusters: 0 = slowest (lowest mean sp_actual)
    means = np.array([s[observed][labels == c].mean() for c in range(k_eff)])
    remap = {c: i for i, c in enumerate(np.argsort(means))}
    grid = np.full((7, 24), -1, dtype=int)
    for (d, h), lab in zip(zip(dows.tolist(), hrs.tolist()), labels):
        grid[d, h] = remap[int(lab)]
    obs_cells = list(zip(dows.tolist(), hrs.tolist()))
    for d in range(7):
        for h in range(24):
            if grid[d, h] >= 0:
                continue
            best, bdist = obs_cells[0], 1e9
            for (od, oh) in obs_cells:
                dist = min(abs(d - od), 7 - abs(d - od)) * 24 + min(abs(h - oh), 24 - abs(h - oh))
                if dist < bdist:
                    bdist, best = dist, (od, oh)
            grid[d, h] = grid[best[0], best[1]]
    return grid, k_eff


def regime_of(grid, ts):
    """Assign a wall-clock timestamp to a regime via the derived grid."""
    if ts is None or pd.isna(ts):
        return 0
    return int(grid[ts.dayofweek, ts.hour])


# =============================================================================
# City container: whole-graph tensors + per-trip routes (NO local graphs).
# =============================================================================
class MetaETACity:
    def __init__(self, x, vff, rel_coo, trips, device, regime_grid=None, num_regimes=None):
        self.regime_grid = regime_grid
        self.x = x                  # [N, F] normalised static features
        self.vff = vff              # [N]    free-flow speed (m/s)
        self.rel_coo = rel_coo      # list of R [2, E_r] COO (src, dst)
        self.trips = trips          # list of dicts (see _build_trips)
        self.N = x.size(0)
        self.F = x.size(1)
        self.device = device
        # combined undirected adjacency over upstream+downstream for hop-BFS
        # rel order = [self, downstream, upstream, left, opposite, right, high, uturn]
        dn, up = rel_coo[1], rel_coo[2]
        und_src = torch.cat([dn[0], dn[1], up[0], up[1]])
        und_dst = torch.cat([dn[1], dn[0], up[1], up[0]])
        self._und_src = und_src
        self._und_dst = und_dst
        # bucket trip indices by regime for episode sampling
        self.by_regime = defaultdict(list)
        for i, t in enumerate(trips):
            self.by_regime[t["regime"]].append(i)
        self.num_regimes = int(num_regimes) if num_regimes is not None \
            else (max((t["regime"] for t in trips), default=0) + 1)


def _build_trips(df_all: pd.DataFrame, vff_np: np.ndarray):
    """Per-trip route records from df_all (mirrors TripDatasetLinkETA's df_small
    extraction, MINUS the off-route-neighbour local graph)."""
    sort_col = CFG["time_col_df"] if CFG["time_col_df"] in df_all.columns else "enter_ts"
    cols = ["TRIP_ID", "edge_idx", "link_len_m", "link_travel_time_s",
            "trip_travel_time_s", sort_col]
    df = df_all[[c for c in cols if c in df_all.columns]].copy()
    for c in ("link_len_m", "link_travel_time_s", "trip_travel_time_s"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df[sort_col] = pd.to_datetime(df[sort_col], errors="coerce")

    trips = []
    for trip_id, grp in df.groupby("TRIP_ID", sort=False):
        grp = grp.sort_values(sort_col)
        edges = pd.to_numeric(grp["edge_idx"], errors="coerce").dropna().astype(int).to_numpy()
        if edges.size == 0:
            continue
        lens = pd.to_numeric(grp["link_len_m"], errors="coerce").fillna(5.0).to_numpy()
        tts = pd.to_numeric(grp["link_travel_time_s"], errors="coerce").fillna(0.0).to_numpy()
        t0 = grp[sort_col].iloc[0]
        # trip-level label: prefer trip_travel_time_s, else sum of link tt
        if "trip_travel_time_s" in grp.columns and pd.notna(grp["trip_travel_time_s"].iloc[0]) \
                and float(grp["trip_travel_time_s"].iloc[0]) > 0:
            trip_tt = float(grp["trip_travel_time_s"].iloc[0])
        else:
            trip_tt = float(np.nansum(tts))
        if not (np.isfinite(trip_tt) and trip_tt > 0):
            continue
        if pd.isna(t0):
            dow, hour = 0, 12
        else:
            dow, hour = int(t0.dayofweek), int(t0.hour)
        trips.append(dict(
            trip_id=int(trip_id) if str(trip_id).isdigit() else trip_id,
            edges=torch.tensor(edges, dtype=torch.long),
            lens=torch.tensor(lens, dtype=torch.float32),
            tts=torch.tensor(tts, dtype=torch.float32),
            trip_tt=float(trip_tt),
            t0=t0, dow=dow, hour=hour, regime=0,   # regime assigned after map is built
        ))
    return trips


def load_city(city: str, device, max_trips=None, regime_k=DEFAULT_REGIME_K) -> MetaETACity:
    p = city_paths(city)
    print(f"[MetaETA] loading city '{city}' from {p['precomputed_dir']}")
    # reuse the LinkETA bundle loader as-is (gives rel_edges, vff_vec, scalers, device)
    base = init_linketa(precomputed_dir=p["precomputed_dir"],
                        checkpoint_path=p["checkpoint_path"])
    device = base["device"]
    vff = base["graph"]["vff_vec"].to(device).float()        # [N]
    rel_coo = [r.to(device).long() for r in base["graph"]["rel_edges"]]
    N = int(vff.numel())

    # static features [N, 31] (same tensor MetaTraffic uses)
    if not os.path.exists(p["static_feats_31"]):
        raise FileNotFoundError(
            f"{p['static_feats_31']} not found. Run structural_clustering.py first "
            f"(it builds static_feats_31.pt), as main.py's MetaTraffic path does.")
    sf = torch.load(p["static_feats_31"], map_location="cpu", weights_only=False)
    x = sf["features"].float()                               # [N, 31]
    if x.size(0) != N:
        # align to graph N (pad/truncate defensively)
        xx = torch.zeros(N, x.size(1))
        m = min(N, x.size(0)); xx[:m] = x[:m]; x = xx
    # per-column z-norm (self-contained for a single city)
    mu = x.mean(0, keepdim=True)
    sd = x.std(0, keepdim=True).clamp_min(1e-6)
    x = ((x - mu) / sd).to(device)

    df_all = pd.read_csv(p["trips_path"])
    if max_trips is not None:
        keep = df_all["TRIP_ID"].drop_duplicates().sample(
            n=min(max_trips, df_all["TRIP_ID"].nunique()), random_state=42)
        df_all = df_all[df_all["TRIP_ID"].isin(keep)]

    # df_all identifies links by way_id; map to internal edge_idx exactly like
    # LinkETA.ensure_static_feature_columns (id2edge from the edges table).
    edges_vff = base["graph"]["edges_vff"]
    edges_id_col = base.get("edges_id_col", "id")
    if "edge_idx" not in edges_vff.columns:
        edges_vff = edges_vff.copy(); edges_vff["edge_idx"] = np.arange(len(edges_vff))
    way_col = CFG["way_id_col_df"] if CFG["way_id_col_df"] in df_all.columns else \
        next((c for c in ["way_id", "id", "osmid", "edge_id"] if c in df_all.columns), None)
    if way_col is None:
        raise KeyError(f"df_all has no way-id column (looked for way_id/id/osmid/edge_id)")
    id2edge = dict(zip(edges_vff[edges_id_col].tolist(), edges_vff["edge_idx"].tolist()))
    df_all["edge_idx"] = df_all[way_col].map(id2edge)
    if int(df_all["edge_idx"].notna().sum()) == 0:        # dtype mismatch -> str keys
        id2edge = {str(k): v for k, v in id2edge.items()}
        df_all["edge_idx"] = df_all[way_col].astype(str).map(id2edge)
    n0 = len(df_all)
    df_all = df_all.dropna(subset=["edge_idx"]).copy()
    df_all["edge_idx"] = df_all["edge_idx"].astype(int)
    if len(df_all) < n0:
        print(f"[MetaETA] dropped {n0 - len(df_all)} rows with unmapped {way_col}")
    if len(df_all) == 0:
        raise RuntimeError(f"No rows mapped {way_col}->edge_idx "
                           f"(edges '{edges_id_col}' vs df '{way_col}' dtype mismatch?)")

    vff_np = vff.detach().cpu().numpy()
    trips = _build_trips(df_all, vff_np)
    regime_grid, k_eff = build_regime_map(trips, vff_np, k=regime_k)
    for t in trips:
        t["regime"] = int(regime_grid[t["dow"], t["hour"]])
    counts = np.bincount([t["regime"] for t in trips], minlength=k_eff).tolist()
    print(f"[MetaETA] city '{city}': N={N} edges, {len(trips)} trips, "
          f"x={tuple(x.shape)}, relations={len(rel_coo)}")
    print(f"[MetaETA] regimes (k={k_eff}, congestion-ordered 0=slowest): "
          f"trips/regime = {counts}")
    return MetaETACity(x, vff, rel_coo, trips, device, regime_grid, k_eff)


# =============================================================================
# Stage 0' : regime-specific observations from a SUPPORT set of trips.
#   o_true[N]  = mean sp_actual over the support links observed in this regime
#   b[N]       = observed flag
#   plus the raw support observations (edges, tts, lens) for g_tau.
# =============================================================================
def compute_sp_actual(lens, tts, vff_e):
    sp = (lens / tts.clamp_min(1e-3)) / vff_e.clamp_min(1.0)
    return sp.clamp(0.0, R_MAX_TARGET)


def build_regime_observations(city: MetaETACity, support_trips):
    """Vectorised: one cat + one scatter over ALL support link-observations
    (so conditioning on thousands of trips is cheap)."""
    N, dev = city.N, city.device
    if not support_trips:
        return torch.zeros(N, device=dev), torch.zeros(N, dtype=torch.bool, device=dev), None
    all_e = torch.cat([t["edges"] for t in support_trips]).to(dev)
    all_ln = torch.cat([t["lens"] for t in support_trips]).to(dev)
    all_tt = torch.cat([t["tts"] for t in support_trips]).to(dev)
    valid = (all_tt >= 1.0) & (all_ln >= 10.0)
    e, ln, tt = all_e[valid], all_ln[valid], all_tt[valid]
    # GEOMETRIC (log-space) mean of sp_actual: robust to heavy-tailed link times
    # (diag-floors: geo-mean floor 0.29 vs arithmetic 0.38 on porto) and consistent
    # with the log-space link/route losses.  Geo-mean, NOT median.
    sum_log = torch.zeros(N, device=dev); cnt = torch.zeros(N, device=dev)
    if e.numel() > 0:
        sp = compute_sp_actual(ln, tt, city.vff[e]).clamp_min(1e-3)
        sum_log.scatter_add_(0, e, torch.log(sp))
        cnt.scatter_add_(0, e, torch.ones_like(sp))
    b = (cnt > 0)
    o_true = torch.where(b, torch.exp(sum_log / cnt.clamp_min(1.0)), torch.zeros_like(sum_log))
    obs = dict(edges=e, tts=tt, lens=ln) if e.numel() > 0 else None
    return o_true, b, obs


def _reach_hops(city: MetaETACity, b_observed, max_hops=6):
    """Multi-source BFS hop distance to the nearest observed link (undirected
    up+down adjacency).  Returns reachability feature exp(-hops/scale) in [0,1]."""
    N, dev = city.N, city.device
    reached = b_observed.clone()
    dist = torch.full((N,), float(max_hops + 1), device=dev)
    dist[reached] = 0.0
    src, dst = city._und_src, city._und_dst
    for h in range(1, max_hops + 1):
        agg = torch.zeros(N, device=dev)
        agg.scatter_add_(0, dst, reached[src].float())
        newly = (agg > 0) & (~reached)
        if not newly.any():
            break
        dist[newly] = float(h)
        reached = reached | newly
    return torch.exp(-dist / 2.0)            # [N] in (0,1], 1 at observed links


# =============================================================================
# Stage 2' : ONE sparse spatial-relational pass per regime (NO temporal gates).
# =============================================================================
class MetaETAGATLayer(nn.Module):
    """Sparse relational attention (mirrors MetaTraffic.RelationalGATLayer) PLUS a
    per-relation learnable bias beta_r (PART 7 Stage 2': score = content + beta_r).
    Convention kept identical: destination node aggregates from source neighbours,
    softmax per destination."""

    def __init__(self, dim, num_relations=8, n_heads=4, dropout=0.1):
        super().__init__()
        self.dim, self.n_heads, self.dk, self.R = dim, n_heads, dim // n_heads, num_relations
        self.W_q = nn.ParameterList([nn.Parameter(torch.randn(n_heads, dim, self.dk) * 0.02) for _ in range(num_relations)])
        self.W_k = nn.ParameterList([nn.Parameter(torch.randn(n_heads, dim, self.dk) * 0.02) for _ in range(num_relations)])
        self.W_v = nn.ParameterList([nn.Parameter(torch.randn(n_heads, dim, self.dk) * 0.02) for _ in range(num_relations)])
        self.beta = nn.Parameter(torch.zeros(num_relations, n_heads))   # per-relation bias
        self.out_proj = nn.Linear(dim, dim)
        self.layer_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, rel_coo):
        N, D = h.shape
        agg = torch.zeros(N, self.n_heads, self.dk, device=h.device, dtype=h.dtype)
        count = torch.zeros(N, 1, 1, device=h.device, dtype=h.dtype)
        for r in range(self.R):
            if rel_coo[r].numel() == 0:
                continue
            src, dst = rel_coo[r][0], rel_coo[r][1]
            E_r = src.shape[0]
            q = torch.einsum("ed,hdo->eho", h[dst], self.W_q[r])
            k = torch.einsum("ed,hdo->eho", h[src], self.W_k[r])
            v = torch.einsum("ed,hdo->eho", h[src], self.W_v[r])
            att_logits = (q * k).sum(-1) / math.sqrt(self.dk) + self.beta[r].unsqueeze(0)
            att_w = _scatter_softmax(att_logits, dst, N)
            msg = v * att_w.unsqueeze(-1)
            dst_exp = dst.unsqueeze(-1).unsqueeze(-1).expand(-1, self.n_heads, self.dk)
            agg.scatter_add_(0, dst_exp, msg.to(agg.dtype))
            count.scatter_add_(0, dst.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 1),
                               torch.ones(E_r, 1, 1, device=h.device, dtype=h.dtype))
        agg = (agg / count.clamp_min(1.0)).reshape(N, D)
        out = self.out_proj(self.dropout(agg))
        return self.layer_norm(h + out)


class MetaETAModel(nn.Module):
    """Whole-graph speed-ratio predictor + static-only residual.
    forward(x, o, b, reach, regime_id, rel_coo) -> sp_hat[N], delta[N]."""

    def __init__(self, static_dim, dim=32, n_heads=4, num_relations=8,
                 num_layers=2, dropout=0.1, delta_range=0.1, num_regimes=DEFAULT_REGIME_K):
        super().__init__()
        self.dim = dim
        self.delta_range = delta_range
        # input: [static(F), o, b, reach] -> dim ; + regime embedding
        self.input_proj = nn.Linear(static_dim + 3, dim)
        self.regime_emb = nn.Embedding(num_regimes, dim)
        self.layers = nn.ModuleList([
            MetaETAGATLayer(dim, num_relations, n_heads, dropout) for _ in range(num_layers)
        ])
        # [A4] jumping-knowledge: combine h^(0..L) so deep layers can't over-smooth
        self.jk = nn.Linear(dim * (num_layers + 1), dim)
        self.speed_head = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 1))
        # [A3] delta from STATIC features ONLY -> cannot carry congestion level
        self.residual_head = nn.Sequential(nn.Linear(static_dim, dim), nn.ReLU(), nn.Linear(dim, 1))
        # warm start: sp_hat = 0.5*r_max ~ free-flow-ish; delta = 0
        nn.init.zeros_(self.speed_head[-1].weight); nn.init.zeros_(self.speed_head[-1].bias)
        nn.init.zeros_(self.residual_head[-1].weight); nn.init.zeros_(self.residual_head[-1].bias)

    def forward(self, x, o, b, reach, regime_id, rel_coo, r_max=R_MAX_TARGET):
        feat = torch.cat([x, o.unsqueeze(-1), b.float().unsqueeze(-1), reach.unsqueeze(-1)], dim=-1)
        h = self.input_proj(feat) + self.regime_emb(
            torch.full((1,), int(regime_id), device=x.device, dtype=torch.long))
        hs = [h]
        for layer in self.layers:
            h = layer(h, rel_coo)
            hs.append(h)
        h_jk = self.jk(torch.cat(hs, dim=-1))
        sp_hat = torch.sigmoid(self.speed_head(h_jk).squeeze(-1)) * r_max     # [N]
        delta = self.delta_range * torch.tanh(self.residual_head(x).squeeze(-1))  # [N]
        return sp_hat, delta


# =============================================================================
# Stage 3' : per-trip ETA with measured-on-observed + target-anchored g_tau.
# =============================================================================
def _stage3_eta(city, sp_hat, delta, o_meas, b_input, b_support, query_trips, obs):
    """Assemble padded [B,L] route tensors and compute link/trip ETA.
    Returns pred_seq[B,L], seq_true[B,L], seq_link_len[B,L], trip_tt[B],
            seq_mask[B,L], score_mask[B,L] (UNOBSERVED links only)."""
    dev = city.device
    B = len(query_trips)
    Lmax = max(t["edges"].numel() for t in query_trips)

    edge_idx = torch.zeros(B, Lmax, dtype=torch.long, device=dev)
    seq_link_len = torch.zeros(B, Lmax, device=dev)
    seq_true = torch.zeros(B, Lmax, device=dev)
    seq_mask = torch.zeros(B, Lmax, dtype=torch.bool, device=dev)
    trip_tt = torch.zeros(B, device=dev)
    for i, t in enumerate(query_trips):
        L = t["edges"].numel()
        edge_idx[i, :L] = t["edges"].to(dev)
        seq_link_len[i, :L] = t["lens"].to(dev)
        seq_true[i, :L] = t["tts"].to(dev)
        seq_mask[i, :L] = True
        trip_tt[i] = t["trip_tt"]
    seq_link_len = seq_link_len.clamp_min(1e-3)

    vff_seq = city.vff[edge_idx]
    sp_hat_seq = sp_hat[edge_idx]
    delta_seq = delta[edge_idx]
    o_seq = o_meas[edge_idx]
    b_in_seq = b_input[edge_idx]          # observed AND not hidden -> use measured
    b_sup_seq = b_support[edge_idx]       # observed by support (pre-hide) -> exclude from L_link

    # A5/D4: measured ratio where observed-not-hidden, else predicted sp_hat
    s_used = torch.where(b_in_seq, o_seq.clamp_min(1e-3), sp_hat_seq.clamp_min(1e-3))
    pred_mask = ~b_in_seq                 # links that used sp_hat -> get g_tau

    # [A3] TARGET-ANCHORED g_tau from support-observed links (detached predictor)
    g_tau = torch.ones((), device=dev)
    if obs is not None and obs["edges"].numel() > 0:
        e = obs["edges"]
        base_obs = (obs["lens"] / (sp_hat[e].detach().clamp_min(1e-3) * city.vff[e].clamp_min(1.0))) \
            * torch.exp(delta[e].detach())
        denom = base_obs.sum().clamp_min(1e-6)
        g_tau = (obs["tts"].sum() / denom).clamp(0.5, 2.0)

    # delta corrects PREDICTED links only; observed links keep their measured
    # geo-mean value untouched (else the residual drags the good estimate off and
    # the per-link errors compound at the route level -> the epoch-wise drift).
    delta_eff = torch.where(pred_mask, delta_seq, torch.zeros_like(delta_seq))
    eta_base = (seq_link_len / (s_used * vff_seq.clamp_min(1.0))) * torch.exp(delta_eff)
    link_eta = torch.where(pred_mask, eta_base * g_tau, eta_base)
    link_eta = link_eta * seq_mask.float()
    pred_seq = link_eta_in_seconds(link_eta, seq_link_len)

    score_mask = seq_mask & (~b_sup_seq)  # [B1] UNOBSERVED links only
    return pred_seq, seq_true, seq_link_len, trip_tt, seq_mask, score_mask


# =============================================================================
# Condition ONE regime: build observations, hide, propagate -> sp_hat/delta.
# Stage 2 is run ONCE here and reused by every query trip in the regime (doc-faithful).
# =============================================================================
def condition_regime(city, model, support_trips, regime, hide_p=0.0, train=False):
    dev = city.device
    o_true, b_support, obs = build_regime_observations(city, support_trips)
    if obs is None or not bool(b_support.any()):
        return None
    b_input = b_support.clone()
    hidden = torch.zeros_like(b_support)
    if train and hide_p > 0:                       # [D3] uniform-random hiding
        obs_idx = b_support.nonzero(as_tuple=False).squeeze(-1)
        if obs_idx.numel() > 1:
            n_hide = max(1, int(hide_p * obs_idx.numel()))
            perm = obs_idx[torch.randperm(obs_idx.numel(), device=dev)[:n_hide]]
            b_input[perm] = False
            hidden[perm] = True
    o_input = torch.where(b_input, o_true, torch.zeros_like(o_true))  # hidden -> looks unobserved
    reach = _reach_hops(city, b_input)
    sp_hat, delta = model(city.x, o_input, b_input, reach, regime, city.rel_coo)
    return dict(sp_hat=sp_hat, delta=delta, o_true=o_true, b_support=b_support,
                b_input=b_input, hidden=hidden, obs=obs)


def one_episode(city, model, regime, support_trips, query_trips,
                p_lo=0.3, p_hi=0.6, train=True):
    hide_p = random.uniform(p_lo, p_hi) if train else 0.0
    ctx = condition_regime(city, model, support_trips, regime, hide_p=hide_p, train=train)
    if ctx is None:
        return None
    sp_hat, delta, hidden = ctx["sp_hat"], ctx["delta"], ctx["hidden"]

    # ---- L_speed : HIDDEN links only (no-leakage) ----
    if hidden.any():
        loss_speed = F.smooth_l1_loss(sp_hat[hidden], ctx["o_true"][hidden], beta=0.1)
    else:
        loss_speed = torch.zeros((), device=city.device)

    # ---- Stage 3 + L_link / L_route on query trips ----
    pred_seq, seq_true, seq_link_len, trip_tt, seq_mask, score_mask = _stage3_eta(
        city, sp_hat, delta, ctx["o_true"], ctx["b_input"], ctx["b_support"], query_trips, ctx["obs"])

    loss_link = link_loss_huber_log_lenweighted(pred_seq, seq_true.clamp_min(0.0),
                                                score_mask, link_len=seq_link_len, delta=0.3)
    loss_route = route_loss_huber_log(pred_seq, trip_tt, seq_mask, delta=0.3)

    return dict(loss_speed=loss_speed, loss_link=loss_link, loss_route=loss_route,
                pred_seq=pred_seq, seq_true=seq_true, seq_link_len=seq_link_len,
                trip_tt=trip_tt, seq_mask=seq_mask)


# =============================================================================
# DWA over (speed, link, route)  (same idea as LinkETA one_epoch)
# =============================================================================
def dwa_weights(hist, T=2.0):
    keys = ["speed", "link", "route"]
    if any(len(hist[k]) < 2 for k in keys):
        return {k: 1.0 / 3 for k in keys}
    r = torch.tensor([hist[k][-1] / (hist[k][-2] + 1e-8) for k in keys])
    w = torch.softmax(r / T, dim=0)
    return {k: float(w[i]) for i, k in enumerate(keys)}


# =============================================================================
# Train / eval loops
# =============================================================================
def temporal_split(trips, fr=(0.7, 0.15, 0.15)):
    order = sorted(range(len(trips)), key=lambda i: (trips[i]["t0"] is pd.NaT, trips[i]["t0"]))
    n = len(order)
    a, b = int(fr[0] * n), int((fr[0] + fr[1]) * n)
    return order[:a], order[a:b], order[b:]


def _freeflow_route_mape(city, query_idx):
    """Model-free reference: every link at free-flow (sp=1 -> eta=len/vff)."""
    apes = []
    for i in query_idx:
        t = city.trips[i]
        e = t["edges"].to(city.device); ln = t["lens"].to(city.device)
        ff = float((ln / city.vff[e].clamp_min(1.0)).sum())
        apes.append(abs(ff - t["trip_tt"]) / max(t["trip_tt"], 1e-6))
    return float(np.mean(apes)) if apes else float("nan")


def run_eval(city, model, support_idx, query_idx, qbatch=32, eval_support=1_000_000):
    """Stage 2 computed ONCE per regime (doc-faithful), Stage 3 per query batch.
    eval_support caps how many support trips condition each regime (coverage knob).
    Also reports 'coverage' = fraction of query links observed by the support."""
    model.eval()
    sums = defaultdict(float); counts = defaultdict(int)
    sum_abs_r = sum_true_r = 0.0; ape_r = []
    cov_obs = cov_tot = 0
    with torch.no_grad():
        for regime in range(city.num_regimes):
            q_all = [i for i in query_idx if city.trips[i]["regime"] == regime]
            sup = [i for i in support_idx if city.trips[i]["regime"] == regime]
            if not q_all or len(sup) < 2:
                continue
            sup_trips = [city.trips[i] for i in sup[:eval_support]]
            ctx = condition_regime(city, model, sup_trips, regime, hide_p=0.0, train=False)
            if ctx is None:
                continue
            for s in range(0, len(q_all), qbatch):
                qb = [city.trips[i] for i in q_all[s:s + qbatch]]
                pred_seq, seq_true, seq_link_len, trip_tt, seq_mask, score_mask = _stage3_eta(
                    city, ctx["sp_hat"], ctx["delta"], ctx["o_true"],
                    ctx["b_input"], ctx["b_support"], qb, ctx["obs"])
                lm = link_metrics_micro(pred_seq, seq_true.clamp_min(0.0), seq_mask,
                                        link_len=seq_link_len, L_min=L_MIN_METRIC, T_min=T_MIN_METRIC)
                rm = route_metrics_pooled(pred_seq, trip_tt, seq_mask)
                if int(lm["n_links"]) > 0 and int(lm.get("n_mape", 0)) > 0 and not math.isnan(lm["mape"]):
                    sums["link_mape"] += lm["mape"] * int(lm["n_mape"]); counts["link_mape"] += int(lm["n_mape"])
                if int(rm["n_routes"]) > 0 and int(rm.get("n_mape", 0)) > 0 and not math.isnan(rm["mape"]):
                    sums["route_mape"] += rm["mape"] * int(rm["n_mape"]); counts["route_mape"] += int(rm["n_mape"])
                if int(rm["n_routes"]) > 0:
                    sums["route_mae"] += rm["mae"] * int(rm["n_routes"]); counts["route_mae"] += int(rm["n_routes"])
                sum_abs_r += float(rm.get("sum_abs", 0.0)); sum_true_r += float(rm.get("sum_true", 0.0))
                _avr = rm.get("ape_vals")
                if _avr is not None and _avr.numel() > 0:
                    ape_r.append(_avr)
                cov_tot += int(seq_mask.sum()); cov_obs += int((seq_mask & ~score_mask).sum())
    rpct = finalize_pct_metrics(sum_abs_r, sum_true_r, ape_r)
    return {
        "route_mape": sums["route_mape"] / max(counts["route_mape"], 1) if counts["route_mape"] else float("nan"),
        "link_mape": sums["link_mape"] / max(counts["link_mape"], 1) if counts["link_mape"] else float("nan"),
        "route_mae": sums["route_mae"] / max(counts["route_mae"], 1) if counts["route_mae"] else float("nan"),
        "route_wape": rpct["wape"],
        "coverage": cov_obs / max(cov_tot, 1),
    }


def train(city, model, args):
    dev = city.device
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)
    os.makedirs(city_paths(args.city)["ckpt_out"], exist_ok=True) if args.city != "__smoke__" else None
    ckpt = CheckpointManager(city_paths(args.city)["ckpt_out"]) if args.city != "__smoke__" else None

    train_idx, val_idx, test_idx = temporal_split(city.trips)
    print(f"[MetaETA] split: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
    print(f"[MetaETA] free-flow baseline (val) route_mape = {_freeflow_route_mape(city, val_idx):.4f}")
    by_reg_train = defaultdict(list)
    for i in train_idx:
        by_reg_train[city.trips[i]["regime"]].append(i)

    hist = {"speed": [], "link": [], "route": []}
    best_sel, since_best, best_state = float("inf"), 0, None

    for epoch in range(args.epochs):
        model.train()
        w = dwa_weights(hist)
        ep = defaultdict(float); nb = 0
        regimes = [r for r in range(city.num_regimes) if len(by_reg_train[r]) >= (args.K + 2)]
        random.shuffle(regimes)
        for regime in regimes:
            pool = by_reg_train[regime][:]
            for _ in range(args.steps_per_regime):
                random.shuffle(pool)
                sup = [city.trips[i] for i in pool[:args.K]]
                qb = [city.trips[i] for i in pool[args.K:args.K + args.qbatch]]
                if len(qb) == 0:
                    continue
                parts = one_episode(city, model, regime, sup, qb,
                                    p_lo=args.p_lo, p_hi=args.p_hi, train=True)
                if parts is None:
                    continue
                loss = (w["speed"] * parts["loss_speed"]
                        + w["link"] * parts["loss_link"]
                        + w["route"] * parts["loss_route"])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                ep["speed"] += float(parts["loss_speed"]); ep["link"] += float(parts["loss_link"])
                ep["route"] += float(parts["loss_route"]); nb += 1
        if nb == 0:
            print(f"[MetaETA] epoch {epoch}: no usable episodes (need >= K+2 trips/regime)"); continue
        for k in hist:
            hist[k].append(ep[k] / nb)

        va = run_eval(city, model, train_idx, val_idx, qbatch=args.qbatch,
                      eval_support=args.eval_support)
        sel = selection_metric(va.get("route_mape"), va.get("link_mape"))
        sched.step(sel if math.isfinite(sel) else 1e9)
        print(f"Epoch {epoch:02d} | train L_speed={hist['speed'][-1]:.4f} "
              f"L_link={hist['link'][-1]:.4f} L_route={hist['route'][-1]:.4f} "
              f"| DWA wS={w['speed']:.2f} wL={w['link']:.2f} wR={w['route']:.2f} "
              f"| val route_mape={va['route_mape']:.4f} link_mape={va['link_mape']:.4f} "
              f"route_mae={va['route_mae']:.2f} cov={va['coverage']:.2f} | sel={sel:.4f}")

        if sel < best_sel - 1e-6:
            best_sel, since_best = sel, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if ckpt is not None:
                ckpt.save_best(model=model, optimizer=opt, scheduler=sched, epoch=epoch,
                               step=epoch, best_metric=float(best_sel), extra={"val": va})
        else:
            since_best += 1
        if ckpt is not None:
            ckpt.save_last(model=model, optimizer=opt, scheduler=sched, epoch=epoch,
                           step=epoch, best_metric=float(best_sel), extra={"val": va})
        if since_best >= args.patience:
            print("Early stopping."); break

    if best_state is not None:
        model.load_state_dict(best_state)
    te = run_eval(city, model, train_idx, test_idx, qbatch=args.qbatch,
                  eval_support=args.eval_support)
    print(f"-- Final Test -- route_mape={te['route_mape']:.4f} link_mape={te['link_mape']:.4f} "
          f"route_mae={te['route_mae']:.2f} route_wape={te['route_wape']:.4f} cov={te['coverage']:.2f}")
    return te


# =============================================================================
# Synthetic smoke test (no data) — validates shapes / backward / no-NaN.
# =============================================================================
def _smoke():
    set_seed(0)
    dev = torch.device("cpu")
    N = 200
    # ring + random downstream edges
    ring_s = torch.arange(N); ring_d = (ring_s + 1) % N
    rnd_s = torch.randint(0, N, (300,)); rnd_d = torch.randint(0, N, (300,))
    dn = torch.stack([torch.cat([ring_s, rnd_s]), torch.cat([ring_d, rnd_d])])
    up = torch.stack([dn[1], dn[0]])
    empty = torch.zeros(2, 0, dtype=torch.long)
    rel_coo = [torch.stack([torch.arange(N), torch.arange(N)]), dn, up, empty, empty, empty, empty, empty]
    x = torch.randn(N, 31)
    vff = torch.rand(N) * 10 + 5
    # synthetic trips: random walks along downstream ring, plausible tts
    trips = []
    for k in range(120):
        L = random.randint(4, 12)
        start = random.randint(0, N - 1)
        edges = [(start + j) % N for j in range(L)]
        edges_t = torch.tensor(edges)
        lens = torch.rand(L) * 100 + 20
        sp = torch.rand(L) * 0.6 + 0.4
        tts = lens / (sp * vff[edges_t])
        ts = pd.Timestamp("2020-01-01") + pd.Timedelta(hours=random.randint(0, 167))
        trips.append(dict(trip_id=k, edges=edges_t, lens=lens, tts=tts,
                          trip_tt=float(tts.sum()), t0=ts,
                          dow=int(ts.dayofweek), hour=int(ts.hour), regime=0))
    grid, k_eff = build_regime_map(trips, vff.numpy(), k=4)
    for t in trips:
        t["regime"] = int(grid[t["dow"], t["hour"]])
    city = MetaETACity(x, vff, rel_coo, trips, dev, grid, k_eff)
    model = MetaETAModel(static_dim=31, dim=32, n_heads=4, num_layers=2, num_regimes=k_eff)
    print(f"[smoke] params={sum(p.numel() for p in model.parameters())/1e3:.1f}K")

    class A: pass
    args = A(); args.city = "__smoke__"; args.epochs = 2; args.lr = 1e-3
    args.K = 8; args.qbatch = 16; args.steps_per_regime = 3
    args.p_lo, args.p_hi, args.patience = 0.3, 0.6, 5
    args.eval_support = 10 ** 6
    args.weight_decay = 1e-4
    te = train(city, model, args)
    assert all(math.isfinite(v) or math.isnan(v) for v in te.values()), te
    print("[smoke] OK — forward/backward/eval run end to end.")


def diag_floors(city, train_idx, val_idx):
    """Model-free transductive floors on val, to separate 'reconstruction bias'
    from 'irreducible trip-to-trip variance'.  Predicts each val-trip link from
    train trips' per-(edge,regime) history, then sums to a trip ETA.
      FREE-FLOW   : len/vff
      MEAN-TIME   : arithmetic mean of observed link TIMEs   (time-space avg)
      MEAN-RATIO  : len/(mean sp_actual * vff)               (ratio-space avg = what the model uses)
      MEDIAN-TIME : median observed link time                (robust)
    Unobserved links fall back to free-flow in every variant."""
    N, R = city.N, city.num_regimes
    vff = city.vff.detach().cpu().numpy()
    sum_tt = np.zeros((R, N)); cnt = np.zeros((R, N)); sum_sp = np.zeros((R, N))
    times = defaultdict(list)  # (regime,edge) -> list of link times (for median)
    for i in train_idx:
        t = city.trips[i]; rg = t["regime"]
        e = t["edges"].numpy(); ln = t["lens"].numpy(); tt = t["tts"].numpy()
        v = (tt >= 1.0) & (ln >= 10.0)
        e, ln, tt = e[v], ln[v], tt[v]
        np.add.at(sum_tt[rg], e, tt); np.add.at(cnt[rg], e, 1.0)
        sp = np.clip((ln / np.maximum(tt, 1e-3)) / np.maximum(vff[e], 1.0), 0, R_MAX_TARGET)
        np.add.at(sum_sp[rg], e, sp)
        for ee, tx in zip(e.tolist(), tt.tolist()):
            times[(rg, ee)].append(tx)
    mean_tt = np.where(cnt > 0, sum_tt / np.maximum(cnt, 1), np.nan)
    mean_sp = np.where(cnt > 0, sum_sp / np.maximum(cnt, 1), np.nan)
    med = {k: float(np.median(v)) for k, v in times.items()}
    geo = {k: float(np.exp(np.mean(np.log(np.maximum(np.asarray(v), 1e-3))))) for k, v in times.items()}
    ff_a, mt_a, rt_a, md_a, gm_a = [], [], [], [], []
    cov_obs = cov_tot = 0
    for i in val_idx:
        t = city.trips[i]; rg = t["regime"]
        e = t["edges"].numpy(); ln = t["lens"].numpy()
        ve = np.maximum(vff[e], 1.0); ff_link = ln / ve
        obs = cnt[rg][e] > 0
        cov_obs += int(obs.sum()); cov_tot += len(e)
        pmt = np.where(obs, np.nan_to_num(mean_tt[rg][e]), ff_link)
        pms = np.where(obs, ln / (np.nan_to_num(mean_sp[rg][e], nan=1.0) * ve), ff_link)
        pmd = np.array([med.get((rg, int(ee)), fl) for ee, fl in zip(e, ff_link)])
        pgm = np.array([geo.get((rg, int(ee)), fl) for ee, fl in zip(e, ff_link)])
        true = max(t["trip_tt"], 1e-6)
        ff_a.append(abs(ff_link.sum() - true) / true)
        mt_a.append(abs(pmt.sum() - true) / true)
        rt_a.append(abs(pms.sum() - true) / true)
        md_a.append(abs(pmd.sum() - true) / true)
        gm_a.append(abs(pgm.sum() - true) / true)
    print(f"[diag-floors] val coverage = {cov_obs / max(cov_tot, 1):.3f}")
    print(f"[diag-floors] FREE-FLOW    route MAPE = {np.mean(ff_a):.4f}")
    print(f"[diag-floors] MEAN-RATIO   route MAPE = {np.mean(rt_a):.4f}   (what the model's observed path uses)")
    print(f"[diag-floors] MEAN-TIME    route MAPE = {np.mean(mt_a):.4f}   (avg link TIME instead of ratio)")
    print(f"[diag-floors] GEO-MEAN     route MAPE = {np.mean(gm_a):.4f}   (log-space mean; robust, not a median)")
    print(f"[diag-floors] MEDIAN-TIME  route MAPE = {np.mean(md_a):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="porto")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", dest="weight_decay", type=float, default=1e-4)
    ap.add_argument("--K", type=int, default=512, help="support trips per TRAIN episode (coverage)")
    ap.add_argument("--qbatch", type=int, default=32, help="query trips per step")
    ap.add_argument("--eval-support", dest="eval_support", type=int, default=1_000_000,
                    help="max support trips per regime at EVAL (default: all train trips)")
    ap.add_argument("--steps-per-regime", dest="steps_per_regime", type=int, default=20)
    ap.add_argument("--p-lo", dest="p_lo", type=float, default=0.3)
    ap.add_argument("--p-hi", dest="p_hi", type=float, default=0.6)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--trips", "--max-trips", dest="max_trips", type=int, default=None,
                    help="number of trips to use (default: ALL trips)")
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--regime-k", dest="regime_k", type=int, default=DEFAULT_REGIME_K,
                    help="number of data-driven regimes (KMeans over the weekly clock)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--diag-floors", dest="diag_floors", action="store_true",
                    help="report model-free transductive floors (no training) and exit")
    args = ap.parse_args()

    if args.smoke:
        _smoke(); return

    if args.diag_floors:
        set_seed(42)
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        city = load_city(args.city, dev, max_trips=args.max_trips, regime_k=args.regime_k)
        tr, va, te = temporal_split(city.trips)
        diag_floors(city, tr, va)
        return

    set_seed(42)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    city = load_city(args.city, dev, max_trips=args.max_trips, regime_k=args.regime_k)
    model = MetaETAModel(static_dim=city.F, dim=args.dim, n_heads=4,
                         num_relations=len(city.rel_coo), num_layers=args.layers,
                         num_regimes=city.num_regimes).to(dev)
    print(f"[MetaETA] model params={sum(p.numel() for p in model.parameters())/1e6:.3f}M")
    train(city, model, args)


if __name__ == "__main__":
    main()
