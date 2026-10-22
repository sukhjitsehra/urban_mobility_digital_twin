import pandas as pd
import ast
import requests
import numpy as np
import json
from scipy.sparse import csr_matrix, vstack
from scipy.optimize import nnls
import scipy
import os, time
from tqdm import tqdm
import networkx as nx
from scipy.sparse import csr_matrix
from shapely.geometry import LineString, MultiLineString, Point
from shapely.strtree import STRtree
from scipy.spatial import cKDTree

from folium.plugins import HeatMap
from concurrent.futures import ThreadPoolExecutor
import geopandas as gpd
from collections import defaultdict
from shapely.geometry import LineString
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import math
from torch_geometric.data import Data
from torch_sparse import SparseTensor
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp import autocast, GradScaler
try:
    from torch.amp import GradScaler, autocast
except Exception:
    from torch.cuda.amp import GradScaler, autocast
from torch import amp
from torch import nn, Tensor
import torch.nn.functional as F
import torch
from contextlib import nullcontext
from collections import defaultdict
import inspect
from collections import Counter
from typing import Optional
import os, time, random
import math, bisect, re, collections
from typing import List, Dict, Tuple, Optional, Sequence
from functools import lru_cache
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from shapely.geometry import LineString, MultiLineString
from torch.cuda.amp import autocast, GradScaler
from loss_functions import (
    link_loss_huber_macro, route_loss_mape,
    link_metrics_micro, route_metrics_pooled,
    selection_metric, L_MIN_METRIC, T_MIN_METRIC,
)
from checkpoint import CheckpointManager

CKPT_DIRNAME = "ckpts_DuETA"

NUM_EPOCHS = 100
PATIENCE = 30
DELTA_HUBER = 5.0

CFG = dict(
    way_id_col_df="way_id",
    time_col_df="departure_time",
    way_id_col_edges_candidates=["way_id","id","edge_id","osmid"],
    length_col_edges="length",
    speed_is_kmh=False,
    slots_per_hour=12,
    dyn_prefixes=["median_speed","mean_speed","min_speed","max_speed","count_vehicle"],
    static_cols=[
        "length","lanes","width_m","speed_limit_ms","has_traffic_signal"
    ],
    hidden_dim=32, heads=8, num_layers=1, max_route_nodes=360,
    high_order_topk=5, speed_metric_index=0,
)

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
            s = (round(s[0],6), round(s[1],6))
            e = (round(e[0],6), round(e[1],6))
            return s,e

        starts, ends = zip(*[_ends(ge) for ge in g["geometry"]])
        g["start_xy"] = starts; g["end_xy"] = ends
        pts = pd.unique(list(g["start_xy"]) + list(g["end_xy"]))
        coord2nid = {pt:i for i,pt in enumerate(pts)}
        g["u"] = g["start_xy"].map(coord2nid)
        g["v"] = g["end_xy"].map(coord2nid)
        g = g.drop(columns=["start_xy","end_xy"])
    return g

def ensure_static_feature_columns(edges_gdf, df_all, candidates, edges_id_col, way_id_df_col):
    g = edges_gdf.copy()

    if "edge_idx" not in g.columns:
        g = g.reset_index(drop=True)
        g["edge_idx"] = g.index

    id2edge = dict(zip(g[edges_id_col].tolist(), g["edge_idx"].tolist()))
    if "edge_idx" not in df_all.columns:
        df_all = df_all.copy()
        df_all["edge_idx"] = df_all[way_id_df_col].map(id2edge)

    if "length" not in g.columns:
        src = None
        for cand in ["length", "length_m_osm"]:
            if cand in g.columns:
                src=("edges", cand); break
        if src is None:
            for cand in ["link_len_m", "length"]:
                if cand in df_all.columns:
                    src=("df", cand); break

        if src is None:
            if hasattr(g, "crs") and g.crs and getattr(g.crs, "is_geographic", False):
                g["length"] = 0.0
            else:
                g["length"] = g.geometry.length
        else:
            if src[0] == "edges":
                g["length"] = pd.to_numeric(g[src[1]], errors="coerce")
            else:
                tmp = (
                    df_all.dropna(subset=["edge_idx", src[1]])
                          .groupby("edge_idx")[src[1]]
                          .first()
                )
                g["length"] = g["edge_idx"].map(tmp)
    g["length"] = pd.to_numeric(g["length"], errors="coerce").fillna(0.0)

    if "speed_limit_ms" not in g.columns:
        src = None
        if "speed_limit_ms" in g.columns: src=("edges","speed_limit_ms")
        elif "speed_limit_ms" in df_all.columns: src=("df","speed_limit_ms")
        elif "speed_limit_kmh" in g.columns: src=("edges_kmh","speed_limit_kmh")
        elif "speed_limit_kmh" in df_all.columns: src=("df_kmh","speed_limit_kmh")
        if src is None:
            g["speed_limit_ms"] = 0.0
        else:
            if src[0] == "edges":
                g["speed_limit_ms"] = pd.to_numeric(g["speed_limit_ms"], errors="coerce")
            elif src[0] == "df":
                tmp = (
                    df_all.dropna(subset=["edge_idx","speed_limit_ms"])
                          .groupby("edge_idx")["speed_limit_ms"].first()
                )
                g["speed_limit_ms"] = g["edge_idx"].map(tmp)
            elif src[0] == "edges_kmh":
                g["speed_limit_ms"] = pd.to_numeric(g["speed_limit_kmh"], errors="coerce")/3.6
            elif src[0] == "df_kmh":
                tmp = (
                    df_all.dropna(subset=["edge_idx","speed_limit_kmh"])
                          .groupby("edge_idx")["speed_limit_kmh"].first()/3.6
                )
                g["speed_limit_ms"] = g["edge_idx"].map(tmp)
    g["speed_limit_ms"] = pd.to_numeric(g["speed_limit_ms"], errors="coerce").fillna(0.0)

    for col in candidates:
        if col in ("length","speed_limit_ms"):
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

def route_order_key(df: pd.DataFrame) -> pd.Series:
    if "enter_ts" not in df.columns:
        raise KeyError("route_order_key: 'enter_ts' column is required to order a route.")
    e = df["enter_ts"]
    if pd.api.types.is_numeric_dtype(e):
        return pd.to_numeric(e, errors="coerce")
    return pd.to_datetime(e, errors="coerce").astype("int64").astype("float64")


def _bearing_toward_end(geom):
    if isinstance(geom, LineString):
        x0,y0 = geom.coords[0]; x1,y1 = geom.coords[-1]
    elif isinstance(geom, MultiLineString):
        x0,y0 = list(geom.geoms[0].coords)[0]
        x1,y1 = list(geom.geoms[-1].coords)[-1]
    else:
        return np.nan
    return math.atan2(y1 - y0, x1 - x0)

def _ang_diff(a, b):
    return (b - a + math.pi) % (2*math.pi) - math.pi

def build_first_order_relations_movement_exact(edges_gdf: pd.DataFrame):
    g = edges_gdf.copy()
    if ("u" not in g.columns) or ("v" not in g.columns):
        raise ValueError("edges_gdf must have u,v columns before building first-order relations.")
    if "edge_idx" not in g.columns:
        raise ValueError("edges_gdf must have edge_idx before building first-order relations.")

    g["bearing_in"] = g["geometry"].apply(_bearing_toward_end)

    out_by_u = g.groupby("u")["edge_idx"].apply(list).to_dict()
    in_by_v  = g.groupby("v")["edge_idx"].apply(list).to_dict()
    bmap     = dict(zip(g["edge_idx"], g["bearing_in"]))

    up_src, up_dst = [], []
    dn_src, dn_dst = [], []
    cl_src, cl_dst = [], []
    ct_src, ct_dst = [], []
    cr_src, cr_dst = [], []

    SAME_THR = math.radians(30)
    OPP_THR  = math.radians(30)

    for row in g.itertuples(index=False):
        i = row.edge_idx
        u, v = row.u, row.v
        theta_i = bmap[i]

        for j in in_by_v.get(u, []):
            if j != i:
                up_src.append(i); up_dst.append(j)

        for j in out_by_u.get(v, []):
            if j != i:
                dn_src.append(i); dn_dst.append(j)

        for j in in_by_v.get(v, []):
            if j == i:
                continue
            theta_j = bmap[j]
            d = _ang_diff(theta_i, theta_j)

            if abs(d) <= SAME_THR:
                continue
            if abs(abs(d) - math.pi) <= OPP_THR:
                ct_src.append(i); ct_dst.append(j)
            elif 0 < d < (math.pi - OPP_THR):
                cl_src.append(i); cl_dst.append(j)
            elif -(math.pi - OPP_THR) < d < 0:
                cr_src.append(i); cr_dst.append(j)
            else:
                (cl_src if d > 0 else cr_src).append(i)
                (cl_dst if d > 0 else cr_dst).append(j)

    return {
        "upstream":         torch.tensor([up_src, up_dst], dtype=torch.long),
        "downstream":       torch.tensor([dn_src, dn_dst], dtype=torch.long),
        "conflict_left":    torch.tensor([cl_src, cl_dst], dtype=torch.long),
        "conflict_through": torch.tensor([ct_src, ct_dst], dtype=torch.long),
        "conflict_right":   torch.tensor([cr_src, cr_dst], dtype=torch.long),
    }

SLOT_PREFIXES = CFG["dyn_prefixes"]

def discover_slot_indices(df_all: pd.DataFrame, base_prefix: Optional[str]="mean_speed") -> List[int]:
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

def pick_last2h_slots(idxs: List[int], slots_per_hour: int) -> List[int]:
    if not idxs:
        return []
    need = 2 * slots_per_hour
    return list(sorted(idxs[-need:]))

def speeds_to_tt(slot_block_24x5: np.ndarray, length_m: float, speed_metric_index=0, speed_is_kmh=False, eps=1e-3) -> np.ndarray:
    v = slot_block_24x5[:, speed_metric_index]
    if speed_is_kmh:
        v = v / 3.6
    v = np.maximum(v, eps)
    return float(length_m) / v

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
    exclude_first_order: Optional[torch.LongTensor]=None
) -> torch.LongTensor:
    id2edge = dict(zip(edges_gdf[way_id_col_edges].tolist(), edges_gdf["edge_idx"].tolist()))
    idxs = discover_slot_indices(df_all, base_prefix="mean_speed")
    last1h = sorted(idxs[:slots_per_hour])
    if len(last1h) == 0:
        raise ValueError(
            "No dynamic slot columns found. Expected columns like "
            f"'median_speed_<k>' (or {CFG['dyn_prefixes']}) with numeric suffix. "
            "Please ensure df_all has these columns."
        )
    flat_cols_12x5 = sum([[f"{p}_{s}" for p in SLOT_PREFIXES] for s in last1h], [])


    if length_col in edges_gdf.columns:
        edge_len = dict(zip(edges_gdf["edge_idx"], edges_gdf[length_col]))
    else:
        tmp = df_all.dropna(subset=[way_id_col_df,length_col]).copy()
        tmp["edge_idx"] = tmp[way_id_col_df].map(id2edge)
        edge_len = tmp.dropna(subset=["edge_idx"]).groupby("edge_idx")[length_col].first().to_dict()

    corr_accum: Dict[Tuple[int,int], float] = {}
    _ordered = df_all.assign(_route_order=route_order_key(df_all))
    grouped = _ordered.sort_values(["TRIP_ID", "_route_order"], kind="mergesort").groupby("TRIP_ID", sort=False)

    for tid, grp in grouped:
        es = grp[way_id_col_df].map(id2edge)
        grp = grp[pd.notna(es)]
        es = grp[way_id_col_df].map(id2edge).astype(int).tolist()
        if len(es) < 3:
            continue
        slot_block = grp[flat_cols_12x5].to_numpy().reshape(len(grp), len(last1h), len(SLOT_PREFIXES))

        L = len(es)
        for i in range(L):
            ei = es[i]
            if ei not in edge_len:
                continue
            tt_i = speeds_to_tt(slot_block[i], edge_len[ei], speed_metric_index, speed_is_kmh)
            for d in (2,3,4,5):
                for j_idx in (i-d, i+d):
                    if 0 <= j_idx < L:
                        ej = es[j_idx]
                        if ej not in edge_len:
                            continue
                        tt_j = speeds_to_tt(slot_block[j_idx], edge_len[ej], speed_metric_index, speed_is_kmh)
                        if np.allclose(tt_i, tt_i[0]) or np.allclose(tt_j, tt_j[0]):
                            rho = 0.0
                        else:
                            rho = float(np.corrcoef(tt_i, tt_j)[0,1])
                            if not np.isfinite(rho): rho = 0.0
                        corr_accum[(ei,ej)] = corr_accum.get((ei,ej), 0.0) + rho

    first_pairs = set()
    if exclude_first_order is not None and exclude_first_order.numel() > 0:
        r = exclude_first_order[0].cpu().tolist(); c = exclude_first_order[1].cpu().tolist()
        first_pairs = set(zip(r,c))

    by_src: Dict[int,List[Tuple[int,float]]] = {}
    for (src,dst),sc in corr_accum.items():
        if (src,dst) in first_pairs:
            continue
        by_src.setdefault(src, []).append((dst, sc))

    srcs, dsts = [], []
    for src, items in by_src.items():
        items.sort(key=lambda t:t[1], reverse=True)
        taken, seen = 0, set()
        for dst, sc in items:
            if dst == src or dst in seen:
                continue
            srcs.append(src); dsts.append(dst); seen.add(dst); taken += 1
            if taken >= top_k:
                break
    return torch.tensor([srcs, dsts], dtype=torch.long)


def select_dynamic_12x5_columns(df_all: pd.DataFrame) -> List[str]:
    order = list(range(11, -1, -1))
    cols = []
    for s in order:
        for p in CFG["dyn_prefixes"]:
            cols.append(f"{p}_{s}")
    print("Using slots k=11..0 (0 is latest). v_last will point to k=0.")
    return cols


class DynamicProjector(nn.Module):
    def __init__(self, hidden_dim_dyn=32):
        super().__init__()
        self.proj = nn.Linear(6*len(CFG["dyn_prefixes"]), hidden_dim_dyn)

    def forward(self, dyn_slots_12x5: torch.Tensor):
        B,M,S,P = dyn_slots_12x5.shape
        x = dyn_slots_12x5.reshape(B,M,S*P)
        return self.proj(x)


class RelationalTransformerLayer(nn.Module):
    def __init__(self, dim, heads, num_relations, dropout=0.1):
        super().__init__()
        self.dim=dim; self.heads=heads; self.dk = dim; self.R = num_relations
        self.to_q = nn.ModuleList([nn.Linear(dim, heads * dim, bias=True) for _ in range(self.R)])
        self.to_k = nn.ModuleList([nn.Linear(dim, heads * dim, bias=True) for _ in range(self.R)])
        self.to_v = nn.ModuleList([nn.Linear(dim, heads * dim, bias=True) for _ in range(self.R)])

        self.dropout  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, local_adjs: List[torch.Tensor]):
        B,M,D = x.shape
        agg = torch.zeros(B,M,self.heads,self.dk, device=x.device, dtype=x.dtype)
        for r in range(self.R):
            q = self.to_q[r](x).view(B,M,self.heads,self.dk)
            k = self.to_k[r](x).view(B,M,self.heads,self.dk)
            v = self.to_v[r](x).view(B,M,self.heads,self.dk)
            scores = torch.einsum('bmhd,bnhd->bhmn', q, k) / math.sqrt(self.dk)
            mask = (local_adjs[r]).unsqueeze(1)
            scores = scores.masked_fill(~mask, float('-inf'))
            alpha = F.softmax(scores, dim=-1)
            alpha = torch.where(torch.isfinite(alpha), alpha, torch.zeros_like(alpha))
            alpha = self.dropout(alpha)
            out_r = torch.einsum('bhmn,bnhd->bmhd', alpha, v)
            agg += out_r
        y = x + agg.sum(dim=2) / (self.R * self.heads)
        return y

def hop_distance_on_edge_graph(local_edges: List[int],
                               edge_index_downstream: torch.LongTensor,
                               start_edge: int) -> Dict[int,int]:
    rows = edge_index_downstream[0].cpu().tolist()
    cols = edge_index_downstream[1].cpu().tolist()
    adj = collections.defaultdict(list)
    for u,v in zip(rows, cols): adj[u].append(v)
    dist = {e: math.inf for e in local_edges}
    if start_edge not in dist and local_edges:
        start_edge = local_edges[0]
    dq = collections.deque([start_edge]); dist[start_edge]=0
    while dq:
        u = dq.popleft()
        for v in adj.get(u,[]):
            if v in dist and dist[v] > dist[u]+1:
                dist[v]=dist[u]+1; dq.append(v)
    for k,v in list(dist.items()):
        if not math.isfinite(v): dist[k]=256
    return dist

class RouteAwareEncodings(nn.Module):
    def __init__(self, max_hop=512, dim=32):
        super().__init__()
        self.pos_emb   = nn.Embedding(max_hop, dim)
        self.route_emb = nn.Embedding(2, dim)
    def forward(self, h, hop_idx, route_id):
        hop_idx = hop_idx.clamp_min(0).clamp_max(self.pos_emb.num_embeddings-1)
        route_id = route_id.clamp(0,1)
        return h + self.pos_emb(hop_idx) + self.route_emb(route_id)

import torch as _torch_alias_for_scope_guard
import math as _math_alias_for_scope_guard
from collections import deque

def _bfs_hops(local_nodes: list, src2dst: dict, start_edge: int, cap: int = 256) -> dict:
    dist = {e: math.inf for e in local_nodes}
    if start_edge not in dist:
        start_edge = local_nodes[0]
    q = deque([start_edge])
    dist[start_edge] = 0
    local_set = set(local_nodes)
    while q:
        u = q.popleft()
        for v in src2dst.get(u, []):
            if v in local_set and dist[v] > dist[u] + 1:
                dist[v] = dist[u] + 1
                q.append(v)
    for k, v in dist.items():
        if not math.isfinite(v):
            dist[k] = cap
    return dist

def _bfs_hops_global(src2dst_full: dict, start_edge: int, cap: int = 512) -> dict:
    from collections import deque
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

def induce_local_subgraph_fast(
    route_edge_seq: list,
    rel_src2dst_list: list,
    hop_src2dst_full: dict,
    max_nodes: int = 360
):
    on_route = list(dict.fromkeys(route_edge_seq))
    local_set = set(on_route)

    for ei in on_route:
        for src2dst in rel_src2dst_list:
            for j in src2dst.get(ei, []):
                if j not in local_set and len(local_set) < max_nodes:
                    local_set.add(j)

    off_route  = [e for e in local_set if e not in on_route]
    local_nodes = on_route + off_route
    M = len(local_nodes)
    node2local = {g: i for i, g in enumerate(local_nodes)}

    route_flag = torch.zeros(M, dtype=torch.long)
    route_flag[:len(on_route)] = 1

    masks = []
    for src2dst in rel_src2dst_list:
        mat = torch.zeros(M, M, dtype=torch.bool)
        for s in local_nodes:
            si = node2local[s]
            for d in src2dst.get(s, []):
                di = node2local.get(d, None)
                if di is not None:
                    mat[si, di] = True
        masks.append(mat)

    start_edge = on_route[0]
    dist_full = _bfs_hops_global(hop_src2dst_full, start_edge, cap=512)
    hop_idx = torch.tensor([dist_full.get(int(e), 512) for e in local_nodes], dtype=torch.long)

    return {
        "local_nodes": torch.tensor(local_nodes, dtype=torch.long),
        "route_flag":  route_flag,
        "hop_idx":     hop_idx,
        "masks":       masks
    }


class DuETA(nn.Module):
    def __init__(self,
                 in_dim_static: int,
                 num_links: int,
                 num_road_types: int,
                 num_junction_types: int,
                 hidden_dim: int = CFG["hidden_dim"],
                 heads: int = CFG["heads"],
                 num_layers: int = CFG["num_layers"],
                 num_relations: int = 6,
                 hidden_dim_dyn: int = 32,
                 max_hop: int = 512,
                 use_prior: bool = False,
                 enforce_positive: bool = True,
                 use_len_seq: bool = False):
        super().__init__()
        self.static_proj  = nn.Linear(in_dim_static, hidden_dim)
        self.link_emb = nn.Embedding(num_links, hidden_dim)
        self.road_type_emb = nn.Embedding(num_road_types, hidden_dim)
        self.junction_emb = nn.Embedding(num_junction_types, hidden_dim)
        self.len_proj = nn.Linear(1, hidden_dim) if use_len_seq else None
        if use_len_seq:
            self.register_buffer("len_mu", torch.tensor(0.0))
            self.register_buffer("len_sd", torch.tensor(1.0))
        self.dynamic_proj = DynamicProjector(hidden_dim_dyn=hidden_dim)

        self.route_enc = RouteAwareEncodings(max_hop=max_hop, dim=hidden_dim)

        self.layers = nn.ModuleList([
            RelationalTransformerLayer(hidden_dim, heads, num_relations, dropout=0.0)
            for _ in range(num_layers)
        ])

        self.conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.mlp  = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.prior_alpha = nn.Parameter(torch.tensor(1.0))
        self.prior_beta  = nn.Parameter(torch.tensor(0.0))
        self.route_scale = nn.Parameter(torch.tensor(1.0))
        self.route_bias  = nn.Parameter(torch.tensor(0.0))

        self.use_prior = use_prior
        self.enforce_positive = enforce_positive


    def forward(self,
            static_feats,
            dyn_slots_12x5,
            hop_idx,
            route_flag,
            masks_per_relation,
            prior_link_eta=None,
            route_seq_idx=None,
            seq_mask=None,
            seq_len_m=None,
            node_edge_idx=None,
            road_type_idx=None,
            junction_idx=None
            ):
        hs = self.static_proj(static_feats)
        hd = self.dynamic_proj(dyn_slots_12x5)
        h  = hs + hd + self.link_emb(node_edge_idx) + self.road_type_emb(road_type_idx) + self.junction_emb(junction_idx)

        h = self.route_enc(h, hop_idx, route_flag)

        for layer in self.layers:
            h = layer(h, masks_per_relation)


        assert route_seq_idx is not None and seq_mask is not None, \
            "route_seq_idx and seq_mask must be provided for sequence aggregator."
        B, L = route_seq_idx.shape
        D = h.size(-1)

        gather_idx = route_seq_idx.unsqueeze(-1).expand(-1, -1, D)
        h_seq = torch.gather(h, 1, gather_idx)
        if self.len_proj is not None and seq_len_m is not None:
            z = (torch.log1p(seq_len_m.clamp_min(0.0)) - self.len_mu) / self.len_sd
            h_seq = h_seq + self.len_proj(z.unsqueeze(-1)) * seq_mask.unsqueeze(-1).to(h_seq.dtype)

        prior_seq = None
        if prior_link_eta is not None:
            prior_seq = torch.gather(prior_link_eta, 1, route_seq_idx)

        x = self.conv(h_seq.transpose(1, 2)).transpose(1, 2)
        delta_seq = self.mlp(x).squeeze(-1)

        if self.use_prior and (prior_seq is not None):
            raw_seq = self.prior_alpha * prior_seq + delta_seq + self.prior_beta
        else:
            raw_seq = delta_seq

        if self.enforce_positive:
            link_eta_seq = F.softplus(raw_seq) + 1e-6
        else:
            link_eta_seq = raw_seq

        link_eta_seq = link_eta_seq * seq_mask.float()

        trip_eta = link_eta_seq.sum(dim=1)

        return link_eta_seq, trip_eta

FAST_COLLATE = True
NO_FUTURE_FALLBACK = True
HIGH_ORDER_TRAIN_ONLY = True
DRIVEN_LEN = True
if os.environ.get("DUETA_LEGACY", "0") == "1":
    NO_FUTURE_FALLBACK = HIGH_ORDER_TRAIN_ONLY = DRIVEN_LEN = False
print(f"[DuETA] NO_FUTURE_FALLBACK={NO_FUTURE_FALLBACK} HIGH_ORDER_TRAIN_ONLY={HIGH_ORDER_TRAIN_ONLY} "
      f"DRIVEN_LEN={DRIVEN_LEN} FAST_COLLATE={FAST_COLLATE}")
_EDGE_ARRAYS = None
_TRIP_CACHE = {}
SPEED_HISTORY = None


def _set_edge_arrays(edges_gdf: pd.DataFrame, cols: List[str]):
    global _EDGE_ARRAYS
    e = edges_gdf.set_index("edge_idx").sort_index()
    if not np.array_equal(e.index.to_numpy(), np.arange(len(e))):
        raise ValueError("edge_idx must be 0..N-1 for FAST_COLLATE")
    _EDGE_ARRAYS = (id(edges_gdf),
                    e[cols].fillna(0.0).to_numpy(dtype=float),
                    e["length"].to_numpy(dtype=float),
                    e["speed_limit_ms"].to_numpy(dtype=float),
                    e["road_type_idx"].to_numpy(dtype=np.int64),
                    e["junction_idx"].to_numpy(dtype=np.int64))
    _TRIP_CACHE.clear()


def gather_static_features(local_nodes: List[int], edges_gdf: pd.DataFrame, cols: List[str]) -> torch.Tensor:
    if FAST_COLLATE and _EDGE_ARRAYS is not None and _EDGE_ARRAYS[0] == id(edges_gdf):
        return torch.tensor(_EDGE_ARRAYS[1][np.asarray(local_nodes, dtype=np.int64)], dtype=torch.float)
    sub = edges_gdf.set_index("edge_idx").reindex(local_nodes)
    Xs = sub[cols].fillna(0.0).to_numpy(dtype=float)
    return torch.tensor(Xs, dtype=torch.float)

def select_dyn_12x5_cols(df_all: pd.DataFrame) -> List[str]:
    return select_dynamic_12x5_columns(df_all)

class EdgeDynStore:
    def __init__(self, df_all: pd.DataFrame, dyn_cols_12x5: List[str],
                 edges_gdf: pd.DataFrame, way_id_col_df: str, way_id_col_edges: str, time_col: str):
        self.dyn_cols = dyn_cols_12x5
        self.time_col = time_col
        id2edge = dict(zip(edges_gdf[way_id_col_edges].tolist(), edges_gdf["edge_idx"].tolist()))
        df = df_all.dropna(subset=[time_col]).copy()
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df = df.dropna(subset=[time_col])
        if "edge_idx" not in df.columns:
            df["edge_idx"] = df[way_id_col_df].map(id2edge)
        df = df.dropna(subset=["edge_idx"])
        df["edge_idx"] = df["edge_idx"].astype(int)
        self.idx: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        for eidx, grp in df.groupby("edge_idx"):
            g = grp.sort_values(time_col)
            arr = g[self.dyn_cols].to_numpy().reshape(len(g), 12, len(CFG["dyn_prefixes"]))
            t64 = g[time_col].astype("int64").to_numpy()
            if np.isnan(arr).any():
                for r in range(arr.shape[0]):
                    for j in range(arr.shape[2]):
                        col = arr[r,:,j]
                        if np.isnan(col).any():
                            mu = np.nanmean(col)
                            if not np.isfinite(mu): mu = 0.0
                            col[np.isnan(col)] = mu
                        arr[r,:,j] = col
            self.idx[eidx] = (t64, arr)


    def get(self, edge_idx: int, t_ref: pd.Timestamp, max_gap_seconds=1800) -> torch.Tensor:
        if edge_idx not in self.idx:
            return torch.zeros(12, len(CFG["dyn_prefixes"]), dtype=torch.float)
        t64, arr = self.idx[edge_idx]
        tref = np.int64(pd.Timestamp(t_ref).value)

        pos = bisect.bisect_right(t64, tref)
        cand = None
        if pos > 0:
            idx = pos - 1
            gap = (int(tref) - int(t64[idx])) / 1e9
            if gap <= max_gap_seconds:
                cand = idx
        if cand is None and pos < len(t64) and not NO_FUTURE_FALLBACK:
            idx = pos
            gap = (int(t64[idx]) - int(tref)) / 1e9
            if gap <= max_gap_seconds:
                cand = idx

        if cand is None:
            return torch.zeros(12, len(CFG["dyn_prefixes"]), dtype=torch.float)
        return torch.tensor(arr[cand], dtype=torch.float)


PRIOR_FALLBACK_V = 13.9
EDGE_PRIOR_MAP = None

class TripDatasetDuETA(torch.utils.data.Dataset):
    def __init__(self, df_all: pd.DataFrame, edges_gdf: pd.DataFrame,
                 way_id_col_df: str, way_id_col_edges: str, time_col: str):
        self.time_col = time_col
        self.id2edge = dict(zip(edges_gdf[way_id_col_edges].tolist(), edges_gdf["edge_idx"].tolist()))
        self.trips: List[Tuple[int,pd.DataFrame]] = []
        _order = route_order_key(df_all)
        for tid, grp in df_all.groupby("TRIP_ID"):
            g = grp.loc[_order.loc[grp.index].sort_values(kind="mergesort").index].copy()
            if "edge_idx" not in g.columns:
                g["edge_idx"] = g[way_id_col_df].map(self.id2edge)
            g = g.dropna(subset=["edge_idx"])
            if len(g)==0: continue
            g["edge_idx"] = g["edge_idx"].astype(int)
            self.trips.append((tid, g))

    def __len__(self): return len(self.trips)
    def __getitem__(self, i):
        tid, g = self.trips[i]
        edge_seq = torch.tensor(g["edge_idx"].tolist(), dtype=torch.long)
        true_tt  = torch.tensor(g["link_travel_time_s"].to_numpy(), dtype=torch.float)
        if "trip_travel_time_s" in g.columns:
            trip_tt = torch.tensor(float(g["trip_travel_time_s"].iloc[0]), dtype=torch.float)
        else:
            trip_tt = torch.tensor(float(g["link_travel_time_s"].sum()), dtype=torch.float)
        t_ref    = pd.to_datetime(g[self.time_col].iloc[0])
        return dict(TRIP_ID=tid, df=g, edge_seq=edge_seq, true_tt=true_tt, trip_tt=trip_tt, t_ref=t_ref)

def collate_trips_dueta(batch: List[dict],
                        rel_edges_list_global: List[torch.LongTensor],
                        edges_gdf: pd.DataFrame,
                        dyn_store: EdgeDynStore,
                        static_cols: List[str],
                        device: Optional[torch.device]=None) -> dict:
    device = device or torch.device("cpu")
    B = len(batch)

    locals_static=[]; locals_dyn=[]; locals_hop=[]; locals_flag=[]; locals_len = []
    locals_true=[]; locals_trip=[]; locals_prior=[]
    locals_masks=[[] for _ in range(6)]
    Ms=[]
    locals_tripids=[]
    locals_mult =[]
    locals_seq_idx = []
    locals_seq_true = []
    locals_seq_len = []
    node_edge_idx = []
    PRIOR_FALLBACK_V = 13.9
    V_MIN, V_MAX = 1.0, 60.0

    for b in batch:
        _key = (int(b["TRIP_ID"]), int(pd.Timestamp(b["t_ref"]).value)) if FAST_COLLATE else None
        _it = _TRIP_CACHE.get(_key) if _key is not None else None
        if _it is None:
            edge_seq = b["edge_seq"].tolist()
            t_ref    = b["t_ref"]

            local = _cached_local_for_route(tuple(edge_seq))
            local_nodes = local["local_nodes"].tolist()
            route_flag  = local["route_flag"]
            hop_idx     = local["hop_idx"]
            masks       = local["masks"]

            Xs = gather_static_features(local_nodes, edges_gdf, static_cols)
            t0 = pd.to_datetime(b["t_ref"])
            tod = t0.hour + t0.minute/60.0
            tod_sin = math.sin(2*math.pi * tod/24.0)
            tod_cos = math.cos(2*math.pi * tod/24.0)
            is_weekend = 1 if t0.dayofweek >= 5 else 0
            from collections import Counter

            cnt = Counter(edge_seq)
            route_mult_vec = torch.tensor([cnt.get(int(e), 0) for e in local_nodes],
                                            dtype=torch.float)

            ctx = torch.tensor([tod_sin, tod_cos, is_weekend], dtype=torch.float).repeat(len(local_nodes), 1)
            Xs = torch.cat([Xs, ctx], dim=1)
            edge2time = dict(zip(
                b["df"]["edge_idx"].tolist(),
                pd.to_datetime(b["df"][dyn_store.time_col]).tolist()
            ))
            assert SPEED_HISTORY is not None and SPEED_HISTORY.bin_seconds == 600, "SPEED_HISTORY not set"
            assert CFG["dyn_prefixes"] == ["median_speed", "mean_speed", "min_speed", "max_speed", "count_vehicle"]
            _way = edges_gdf.set_index("edge_idx")[guess_edges_id_col(edges_gdf, CFG["way_id_col_edges_candidates"])]
            _rows = []
            end = pd.Timestamp(b["t_ref"]).value / 1e9
            for e in local_nodes:
                pic = SPEED_HISTORY.slots(int(_way.at[int(e)]), end, int(b["TRIP_ID"]))[:6][::-1]
                _rows.append(torch.tensor(np.ascontiguousarray(pic), dtype=torch.float))
            dyn = torch.stack(_rows, dim=0)

            true_local = torch.zeros(len(local_nodes), dtype=torch.float)
            e2pos = {int(e): i for i, e in enumerate(local_nodes)}

            _sum, _cnt = {}, {}
            for e, y in zip(b["df"]["edge_idx"].tolist(), b["df"]["link_travel_time_s"].tolist()):
                if int(e) in e2pos:
                    _sum[int(e)] = _sum.get(int(e), 0.0) + float(y)
                    _cnt[int(e)] = _cnt.get(int(e), 0) + 1
            for e in _sum:
                true_local[e2pos[e]] = _sum[e] / _cnt[e]


            if FAST_COLLATE and _EDGE_ARRAYS is not None and _EDGE_ARRAYS[0] == id(edges_gdf):
                _ln = np.asarray(local_nodes, dtype=np.int64)
                L_np, v_lim_np = _EDGE_ARRAYS[2][_ln], _EDGE_ARRAYS[3][_ln]
            else:
                sub = edges_gdf.set_index("edge_idx").reindex(local_nodes)[["length","speed_limit_ms"]]
                L_np = sub["length"].to_numpy(dtype=float)
                v_lim_np = sub["speed_limit_ms"].to_numpy(dtype=float)
            v_lim_np = np.where(np.isfinite(v_lim_np) & (v_lim_np > 0.0), v_lim_np, 13.9)

            L_t    = torch.tensor(L_np,    dtype=torch.float)
            v_limit= torch.tensor(v_lim_np,dtype=torch.float)

            pos_seq = [e2pos[int(e)] for e in b["df"]["edge_idx"].tolist()]
            seq_true = torch.tensor(b["df"]["link_travel_time_s"].to_numpy(), dtype=torch.float)
            seq_len  = L_t[pos_seq]
            if DRIVEN_LEN and "link_len_m" in b["df"].columns:
                _drv = torch.tensor(pd.to_numeric(b["df"]["link_len_m"], errors="coerce").to_numpy(dtype=float),
                                    dtype=torch.float)
                _drv = torch.where(torch.isfinite(_drv), _drv, seq_len)
                seq_len = torch.where(_drv > 0, _drv, torch.full_like(_drv, 5.0))

            v_last = dyn[:, -1, CFG["speed_metric_index"]]
            if CFG["speed_is_kmh"]:
                v_last = v_last / 3.6

            dyn_zero = (dyn.abs().sum(dim=(1,2)) == 0)
            v_floor  = 3.0
            bad_dyn  = dyn_zero | (v_last < v_floor)

            v_eff = torch.where(bad_dyn, v_limit, torch.maximum(v_last, 0.5 * v_limit))

            prior_dyn = L_t / v_eff
            prior_lim = L_t / v_limit

            prior_local_t = torch.minimum(prior_dyn, 3.0 * prior_lim)

            _it = dict(Xs=Xs, dyn=dyn, hop_idx=hop_idx, route_flag=route_flag,
                       masks=[m.nonzero(as_tuple=False).to(torch.int16) for m in masks] if FAST_COLLATE else masks,
                       true_local=true_local, trip_tt=b["trip_tt"], prior=prior_local_t, M=len(local_nodes),
                       L_t=L_t, tid=int(b["TRIP_ID"]), mult=route_mult_vec,
                       seq_idx=torch.tensor(pos_seq, dtype=torch.long), seq_true=seq_true, seq_len=seq_len,
                       node_edge_idx=torch.tensor(local_nodes, dtype=torch.long))
            if _key is not None:
                _TRIP_CACHE[_key] = _it

        locals_static.append(_it["Xs"])
        locals_dyn.append(_it["dyn"])
        locals_hop.append(_it["hop_idx"])
        locals_flag.append(_it["route_flag"])
        for r in range(6): locals_masks[r].append(_it["masks"][r])
        locals_true.append(_it["true_local"])
        locals_trip.append(_it["trip_tt"])
        locals_prior.append(_it["prior"])
        Ms.append(_it["M"])
        locals_len.append(_it["L_t"])
        locals_tripids.append(_it["tid"])
        locals_mult.append(_it["mult"])
        locals_seq_idx.append(_it["seq_idx"])
        locals_seq_true.append(_it["seq_true"])
        locals_seq_len.append(_it["seq_len"])
        node_edge_idx.append(_it["node_edge_idx"])

    out_device = device
    if FAST_COLLATE:
        device = torch.device("cpu")
    Mmax = max(Ms); S = locals_static[0].shape[1]
    Mseq = max((len(s) for s in locals_seq_idx), default=0)
    static_feats = torch.zeros(B, Mmax, S, dtype=torch.float, device=device)
    dyn_slots    = torch.zeros(B, Mmax, 6, len(CFG["dyn_prefixes"]), dtype=torch.float, device=device)
    hop_idx      = torch.zeros(B, Mmax, dtype=torch.long, device=device)
    route_flag   = torch.zeros(B, Mmax, dtype=torch.long, device=device)
    true_tt_mask = torch.zeros(B, Mmax, dtype=torch.float, device=device)
    prior_eta    = torch.zeros(B, Mmax, dtype=torch.float, device=device)
    link_len = torch.zeros(B, Mmax, dtype=torch.float, device=device)
    trip_tt      = torch.stack(locals_trip).to(device)
    trip_ids     = torch.tensor(locals_tripids, dtype=torch.long, device=device)
    route_mult = torch.zeros(B, Mmax, dtype=torch.float, device=device)
    route_seq_idx = torch.zeros(B, Mseq, dtype=torch.long, device=device)
    seq_true_tt   = torch.zeros(B, Mseq, dtype=torch.float, device=device)
    seq_link_len  = torch.zeros(B, Mseq, dtype=torch.float, device=device)
    seq_mask      = torch.zeros(B, Mseq, dtype=torch.bool,  device=device)

    masks_per_rel = [torch.zeros(B, Mmax, Mmax, dtype=torch.bool, device=device) for _ in range(6)]
    node_edge_idx = pad_sequence(node_edge_idx, batch_first=True).to(device)
    if FAST_COLLATE and _EDGE_ARRAYS is not None and _EDGE_ARRAYS[0] == id(edges_gdf):
        road_type_idx = torch.as_tensor(_EDGE_ARRAYS[4], dtype=torch.long).to(device)[node_edge_idx]
        junction_idx = torch.as_tensor(_EDGE_ARRAYS[5], dtype=torch.long).to(device)[node_edge_idx]
    else:
        road_type_idx = torch.as_tensor(edges_gdf.set_index("edge_idx").sort_index()["road_type_idx"].to_numpy(),
                                        dtype=torch.long).to(device)[node_edge_idx]
        junction_idx = torch.as_tensor(edges_gdf.set_index("edge_idx").sort_index()["junction_idx"].to_numpy(),
                                       dtype=torch.long).to(device)[node_edge_idx]

    for i in range(B):
        m = Ms[i]
        static_feats[i, :m] = locals_static[i].to(device)
        dyn_slots[i, :m]    = locals_dyn[i].to(device)
        hop_idx[i, :m]      = locals_hop[i].to(device)
        route_flag[i, :m]   = locals_flag[i].to(device)
        true_tt_mask[i, :m] = locals_true[i].to(device)
        prior_eta[i, :m]    = locals_prior[i].to(device)
        link_len[i, :m] = locals_len[i].to(device)
        route_mult[i, :m] = locals_mult[i].to(device)
        for r in range(6):
            if FAST_COLLATE:
                ij = locals_masks[r][i].long()
                masks_per_rel[r][i, ij[:, 0], ij[:, 1]] = True
            else:
                masks_per_rel[r][i, :m, :m] = locals_masks[r][i].to(device)
        mseq = int(locals_seq_idx[i].numel())
        if mseq > 0:
            route_seq_idx[i, :mseq] = locals_seq_idx[i].to(device)
            seq_true_tt[i, :mseq]   = locals_seq_true[i].to(device)
            seq_link_len[i, :mseq]  = locals_seq_len[i].to(device)
            seq_mask[i, :mseq]      = True

    _batch = dict(
    static_feats=static_feats,
    dyn_slots=dyn_slots,
    hop_idx=hop_idx,
    route_flag=route_flag,
    masks_per_rel=masks_per_rel,
    trip_tt=trip_tt,
    lengths=torch.tensor(Ms, dtype=torch.long, device=device),
    link_len=link_len,
    route_mult=route_mult,
    prior_eta=prior_eta,
    trip_ids=trip_ids,
    route_seq_idx=route_seq_idx,
    seq_true_tt=seq_true_tt,
    seq_link_len=seq_link_len,
    seq_mask=seq_mask,
    node_edge_idx=node_edge_idx,
    road_type_idx=road_type_idx,
    junction_idx=junction_idx,
    )
    if FAST_COLLATE and out_device.type != "cpu":
        _batch = to_device_batch(_batch, out_device)
    return _batch


@lru_cache(maxsize=100_000)
def _cached_local_for_route(edge_seq_tuple):
    hg = hop_src2dst if hop_src2dst is not None else (rel_src2dst_list[1] if rel_src2dst_list else {})
    return induce_local_subgraph_fast(
        list(edge_seq_tuple),
        rel_src2dst_list,
        hg,
        max_nodes=CFG["max_route_nodes"]
    )


def to_device_batch(batch, device):
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        elif isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
            out[k] = [t.to(device, non_blocking=True) for t in v]
        elif isinstance(v, list) and v and isinstance(v[0], list) and isinstance(v[0][0], torch.Tensor):
            out[k] = [[t.to(device, non_blocking=True) for t in row] for row in v]
        else:
            out[k] = v
    return out


rel_src2dst_list = None
rel_edges_list_global = None
downstream_src2dst = None
dyn_store = None
hop_src2dst = None

_mu_s_t = None; _sd_s_t = None
_mu_d_t = None; _sd_d_t = None

def norm_static(X: torch.Tensor) -> torch.Tensor:
    return (X - _mu_s_t) / _sd_s_t

def norm_dyn(D: torch.Tensor) -> torch.Tensor:
    return (D - _mu_d_t) / _sd_d_t

def temporal_split(df: pd.DataFrame,
                   time_col: str,
                   gap_days: int = 7,
                   train_end: Optional[str]=None,
                   test_start: Optional[str]=None,
                   val_ratio: float=0.2):
    d = df.copy()
    d[time_col] = pd.to_datetime(d[time_col], errors="coerce")
    d = d.dropna(subset=[time_col])

    first_dep = d.groupby("TRIP_ID")[time_col].min().rename("first_dep")
    d = d.join(first_dep, on="TRIP_ID")

    min_date, max_date = d["first_dep"].min(), d["first_dep"].max()
    total_days = max((max_date - min_date).days, 1)

    train_cut = min_date + pd.Timedelta(days=int(0.6 * total_days))
    val_cut   = min_date + pd.Timedelta(days=int(0.8 * total_days))
    gap = pd.Timedelta(days=min(7, max(1, total_days // 20)))
    if val_cut - train_cut <= gap:
        gap = pd.Timedelta(hours=12) if val_cut - train_cut > pd.Timedelta(hours=12) else pd.Timedelta(hours=6)

    train_tids = set(d.loc[d["first_dep"] <= train_cut, "TRIP_ID"])
    val_tids   = set(d.loc[(d["first_dep"] > train_cut + gap) & (d["first_dep"] <= val_cut), "TRIP_ID"])
    test_tids  = set(d.loc[d["first_dep"] > val_cut, "TRIP_ID"])

    df_train = d[d["TRIP_ID"].isin(train_tids)].drop(columns=["first_dep"])
    df_val   = d[d["TRIP_ID"].isin(val_tids)].drop(columns=["first_dep"])
    df_test  = d[d["TRIP_ID"].isin(test_tids)].drop(columns=["first_dep"])
    return df_train, df_val, df_test


def _build_src2dst(edge_index: torch.LongTensor):
    d = defaultdict(list)
    s = edge_index[0].tolist()
    t = edge_index[1].tolist()
    for u, v in zip(s, t):
        d[u].append(v)
    return d

def _build_undirected_src2dst(*edge_indices: List[torch.LongTensor]):
    d = defaultdict(list)
    for ei in edge_indices:
        s, t = ei[0].tolist(), ei[1].tolist()
        for u, v in zip(s, t):
            d[u].append(v)
            d[v].append(u)
    return d

def dueta_get_core_components():
    global rel_src2dst_list, rel_edges_list_global, downstream_src2dst, dyn_store
    global _mu_s_t, _sd_s_t, _mu_d_t, _sd_d_t
    global EDGE_PRIOR_MAP

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if "edge_idx" not in edges_gdf.columns:
        edges_gdf_local = edges_gdf.reset_index(drop=True).copy()
        edges_gdf_local["edge_idx"] = edges_gdf_local.index
    else:
        edges_gdf_local = edges_gdf.copy()
    edges_gdf_local = ensure_edge_idx_and_uv(edges_gdf_local)

    edges_id_col = guess_edges_id_col(edges_gdf_local, CFG["way_id_col_edges_candidates"])
    assert edges_id_col is not None, "No id column found in edges_gdf. Add one of: way_id/id/edge_id/osmid"
    edges_gdf_local[edges_id_col] = edges_gdf_local[edges_id_col].astype(str)
    df_all_local = df_all.copy()
    df_all_local[CFG["way_id_col_df"]] = df_all_local[CFG["way_id_col_df"]].astype(str)

    edges_gdf_local, STATIC_COLS_ACTUAL, df_all_local = ensure_static_feature_columns(
        edges_gdf_local, df_all_local, CFG["static_cols"], edges_id_col, CFG["way_id_col_df"]
    )
    CFG["static_cols"] = STATIC_COLS_ACTUAL

    _spd = pd.to_numeric(edges_gdf_local["speed_limit_ms"], errors="coerce")
    _len = pd.to_numeric(edges_gdf_local["length"],          errors="coerce")
    _v   = _spd.replace(0, np.nan).fillna(PRIOR_FALLBACK_V)
    edge_prior_eta = (_len / _v).fillna(_len / PRIOR_FALLBACK_V)
    EDGE_PRIOR_MAP = dict(zip(edges_gdf_local["edge_idx"].tolist(), edge_prior_eta.values))

    rels_fo = build_first_order_relations_movement_exact(edges_gdf_local)
    edge_index_up   = rels_fo["upstream"]
    edge_index_down = rels_fo["downstream"]
    edge_index_cl   = rels_fo["conflict_left"]
    edge_index_ct   = rels_fo["conflict_through"]
    edge_index_cr   = rels_fo["conflict_right"]

    first_order_all = torch.cat([edge_index_up, edge_index_down, edge_index_cl, edge_index_ct, edge_index_cr], dim=1)
    _ho_train_tids = None
    _ho_df = df_all_local
    if HIGH_ORDER_TRAIN_ONLY:
        _tmp = df_all_local.copy()
        _tmp[CFG["time_col_df"]] = pd.to_datetime(_tmp[CFG["time_col_df"]], errors="coerce")
        _tmp = _tmp.dropna(subset=[CFG["time_col_df"], "edge_idx"])
        _ho_tr, _, _ = temporal_split(_tmp, CFG["time_col_df"])
        _ho_train_tids = set(_ho_tr["TRIP_ID"].unique())
        _ho_df = df_all_local[df_all_local["TRIP_ID"].isin(_ho_train_tids)]
        print(f"[HIGH_ORDER_TRAIN_ONLY] high-order from {len(_ho_train_tids)} training trips "
              f"({len(_ho_df)} of {len(df_all_local)} rows)")
        del _tmp, _ho_tr
    edge_index_high = build_high_order_edge_index_dueta(
        df_all=_ho_df,
        edges_gdf=edges_gdf_local,
        top_k=CFG["high_order_topk"],
        way_id_col_df=CFG["way_id_col_df"],
        way_id_col_edges=edges_id_col,
        length_col=CFG["length_col_edges"],
        slots_per_hour=CFG["slots_per_hour"],
        speed_metric_index=CFG["speed_metric_index"],
        speed_is_kmh=CFG["speed_is_kmh"],
        exclude_first_order=first_order_all
    )

    rel_edges_list_global = [edge_index_up, edge_index_down, edge_index_cl, edge_index_ct, edge_index_cr, edge_index_high]
    rel_src2dst_list = [_build_src2dst(ei) for ei in rel_edges_list_global]
    global hop_src2dst
    hop_src2dst = rel_src2dst_list[1]
    _cached_local_for_route.cache_clear()

    dyn_12x5_cols = select_dyn_12x5_cols(df_all_local)
    print("speed_is_kmh =", CFG["speed_is_kmh"], "(set True if your median_speed_* columns are in km/h)")

    df_all_local = df_all_local.copy()
    df_all_local[CFG["time_col_df"]] = pd.to_datetime(df_all_local[CFG["time_col_df"]], errors="coerce")
    df_all_local = df_all_local.dropna(subset=[CFG["time_col_df"], "edge_idx"])
    df_all_local["edge_idx"] = df_all_local["edge_idx"].astype(int)

    df_train, df_val, df_test = temporal_split(df_all_local, CFG["time_col_df"])
    print("Split sizes:", len(df_train), len(df_val), len(df_test))
    if _ho_train_tids is not None and set(df_train["TRIP_ID"].unique()) != _ho_train_tids:
        raise RuntimeError("high-order training trips differ from the training split")

    ds_train = TripDatasetDuETA(df_train, edges_gdf_local, CFG["way_id_col_df"], edges_id_col, CFG["time_col_df"])
    ds_val   = TripDatasetDuETA(df_val,   edges_gdf_local, CFG["way_id_col_df"], edges_id_col, CFG["time_col_df"])
    ds_test  = TripDatasetDuETA(df_test,  edges_gdf_local, CFG["way_id_col_df"], edges_id_col, CFG["time_col_df"])

    train_edge_ids = (
        pd.unique(df_train["edge_idx"])
        if "edge_idx" in df_train.columns
        else pd.unique(
            df_train[CFG["way_id_col_df"]].map(
                dict(zip(edges_gdf_local[edges_id_col], edges_gdf_local["edge_idx"]))
            )
        )
    )

    _static_df = (
        edges_gdf_local.set_index("edge_idx")
        .reindex(train_edge_ids)[CFG["static_cols"]]
        .astype(float)
        .copy()
    )
    _static_df["tod_sin"]    = 0.0
    _static_df["tod_cos"]    = 0.0
    _static_df["is_weekend"] = 0.0

    _mu_s = _static_df.mean()
    _sd_s = _static_df.std().replace(0, 1.0)
    _sd_s = _sd_s.fillna(1.0).where(_sd_s > 1e-6, 1.0)

    _mu_d = df_train[dyn_12x5_cols].astype(float).mean()
    _sd_d = df_train[dyn_12x5_cols].astype(float).std().replace(0, 1.0)
    _sd_d = _sd_d.fillna(1.0).where(_sd_d > 1e-6, 1.0)

    _mu_s_t = torch.tensor(_mu_s.values, dtype=torch.float, device=device)
    _sd_s_t = torch.tensor(_sd_s.values, dtype=torch.float, device=device).clamp_min(1e-12)

    _mu_d_t = torch.tensor(_mu_d.values, dtype=torch.float, device=device).view(12, 5)
    _sd_d_t = torch.tensor(_sd_d.values, dtype=torch.float, device=device).view(12, 5).clamp_min(1e-12)

    dyn_store = EdgeDynStore(df_all_local, dyn_12x5_cols, edges_gdf_local, CFG["way_id_col_df"], edges_id_col, CFG["time_col_df"])

    edges_gdf_local["road_type_idx"] = pd.factorize(edges_gdf_local["highway"].astype(str), sort=True)[0]
    edges_gdf_local["junction_idx"] = pd.factorize(edges_gdf_local["junction"].astype(str), sort=True)[0]
    _set_edge_arrays(edges_gdf_local, CFG["static_cols"])
    collate_fn = lambda b: collate_trips_dueta(
        b, rel_edges_list_global, edges_gdf_local, dyn_store,
        static_cols=CFG["static_cols"], device=device
    )

    train_loader = DataLoader(
        ds_train, batch_size=16, shuffle=True, collate_fn=collate_fn,
        num_workers=0, pin_memory=False, persistent_workers=False
    )
    val_loader = DataLoader(ds_val, batch_size=16, shuffle=False, collate_fn=collate_fn,
                            num_workers=0, pin_memory=False, persistent_workers=False)
    test_loader = DataLoader(ds_test, batch_size=16, shuffle=False, collate_fn=collate_fn,
                             num_workers=0, pin_memory=False, persistent_workers=False)

    model = DuETA(
    in_dim_static=len(CFG["static_cols"]) + 3,
    num_links=int(edges_gdf_local["edge_idx"].max()) + 1,
    num_road_types=int(edges_gdf_local["road_type_idx"].max()) + 1,
    num_junction_types=int(edges_gdf_local["junction_idx"].max()) + 1,
    hidden_dim=CFG["hidden_dim"],
    heads=CFG["heads"],
    num_layers=CFG["num_layers"],
    num_relations=6,
    hidden_dim_dyn=32,
    max_hop=512,
    use_prior=False,
    enforce_positive=True,
    use_len_seq=False,
    ).to(device)
    _mu_link_tt = float(pd.to_numeric(df_train["link_travel_time_s"], errors="coerce").dropna().mean())
    if not np.isfinite(_mu_link_tt) or _mu_link_tt <= 0:
        _mu_link_tt = 10.0

    def _inv_softplus(y: float) -> float:
        return float(np.log(np.expm1(y))) if y > 1e-6 else -10.0

    with torch.no_grad():
        b0 = _inv_softplus(_mu_link_tt)
        model.mlp[-1].bias.fill_(b0)
    print(f"[INIT] Final bias -> softplus({b0:.3f}) ≈ {_mu_link_tt:.2f}s")

    core = dict(
        device=device,
        model=model,
        loaders=dict(train=train_loader, val=val_loader, test=test_loader),
        datasets=dict(train=ds_train, val=ds_val, test=ds_test),
        cfg=CFG,
        edges_id_col=edges_id_col,
        edges_gdf=edges_gdf_local,
        df_all=df_all_local,
        dyn_cols=dyn_12x5_cols,
    )
    return core

print("[CORE] Initialized. Call dueta_get_core_components() from your strategy runner.")

def to_global_sequence_view(
    batch: dict,
    *,
    pred_seq: Optional[torch.Tensor] = None,
    pred_link: Optional[torch.Tensor] = None,
) -> dict:
    route_seq_idx = batch["route_seq_idx"]
    seq_mask      = batch["seq_mask"].bool()
    seq_true      = torch.nan_to_num(batch["seq_true_tt"])
    seq_link_len  = batch.get("seq_link_len", None)
    if seq_link_len is not None:
        seq_link_len = torch.nan_to_num(seq_link_len)
    trip_tt       = torch.nan_to_num(batch["trip_tt"])

    if pred_seq is None:
        assert pred_link is not None, "to_global_sequence_view needs either pred_seq or pred_link"
        pred_seq = torch.gather(pred_link, 1, route_seq_idx)

    return dict(
        pred_seq=pred_seq,
        seq_true=seq_true,
        seq_mask=seq_mask,
        seq_link_len=seq_link_len,
        trip_tt=trip_tt,
    )


def run_training(core):
    device = core["device"]
    model  = core["model"]
    loaders= core["loaders"]
    train_loader, val_loader, test_loader = loaders["train"], loaders["val"], loaders["test"]

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5, weight_decay=0.0)
    scheduler = None

    ckpt = CheckpointManager(dirpath=CKPT_DIRNAME, max_keep=3)
    global_step = 0

    _grad_norms = []

    def one_epoch_global(loader, epoch=0, train=True, delta_huber=DELTA_HUBER):
        model.train() if train else model.eval()

        sum_link_loss = 0.0
        sum_route_loss = 0.0
        N_routes_total = 0

        sum_abs_link = 0.0
        sum_sse_link = 0.0
        sum_mape_link = 0.0
        N_links_total = 0
        N_links_mape = 0

        sum_abs_route = 0.0
        sum_sse_route = 0.0
        sum_mape_route = 0.0
        N_routes_metric = 0
        N_routes_mape = 0

        for batch in loader:
            batch = to_device_batch(batch, device)

            static = norm_static(batch["static_feats"])
            dyn    = batch["dyn_slots"]

            fwd_kwargs = {}
            if "prior_eta" in batch:
                fwd_kwargs["prior_link_eta"] = batch["prior_eta"]

            out = model(
                static, dyn,
                batch["hop_idx"], batch["route_flag"], batch["masks_per_rel"],
                route_seq_idx=batch.get("route_seq_idx", None),
                seq_mask=batch.get("seq_mask", None),
                seq_len_m=batch.get("seq_link_len", None),
                node_edge_idx=batch["node_edge_idx"],
                road_type_idx=batch["road_type_idx"],
                junction_idx=batch["junction_idx"],
                **fwd_kwargs
            )

            if isinstance(out, (list, tuple)) and len(out) == 2 and out[0].dim() == 2:
                gv = to_global_sequence_view(batch, pred_seq=out[0])
            else:
                pred_link = out if isinstance(out, torch.Tensor) else out[0]
                gv = to_global_sequence_view(batch, pred_seq=None, pred_link=pred_link)

            pred_seq   = gv["pred_seq"]
            seq_true   = gv["seq_true"]
            seq_mask   = gv["seq_mask"]
            seq_len_m  = gv["seq_link_len"]
            trip_tt    = gv["trip_tt"]

            l_link  = link_loss_huber_macro(pred_seq, seq_true, seq_mask, delta=delta_huber)
            l_route = route_loss_mape(pred_seq, trip_tt, seq_mask)
            loss = l_link + l_route

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                _gn = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
                _grad_norms.append(float(_gn))
                optimizer.step()

            link_stats = link_metrics_micro(pred_seq, seq_true, seq_mask, seq_len_m,
                                L_min=L_MIN_METRIC, T_min=T_MIN_METRIC)
            route_stats = route_metrics_pooled(pred_seq, trip_tt, seq_mask)

            bsz = int(pred_seq.size(0))
            sum_link_loss  += float(l_link.item())  * bsz
            sum_route_loss += float(l_route.item()) * bsz
            N_routes_total += bsz

            if link_stats["n_links"] > 0:
                N_links_total += link_stats["n_links"]
                sum_abs_link  += link_stats["mae"]  * link_stats["n_links"]
                sum_sse_link  += (link_stats["rmse"] ** 2) * link_stats["n_links"]
            if link_stats["n_mape"] > 0:
                N_links_mape  += link_stats["n_mape"]
                sum_mape_link += link_stats["mape"] * link_stats["n_mape"]

            if route_stats["n_routes"] > 0:
                N_routes_metric += route_stats["n_routes"]
                sum_abs_route   += route_stats["mae"]  * route_stats["n_routes"]
                sum_sse_route   += (route_stats["rmse"] ** 2) * route_stats["n_routes"]
            if route_stats["n_mape"] > 0:
                N_routes_mape  += route_stats["n_mape"]
                sum_mape_route += route_stats["mape"] * route_stats["n_mape"]

        out = {
            "link_loss":  sum_link_loss / max(N_routes_total, 1),
            "route_loss": sum_route_loss / max(N_routes_total, 1),

            "link_mae":   (sum_abs_link  / N_links_total)                 if N_links_total    > 0 else float("nan"),
            "link_rmse":  math.sqrt(sum_sse_link  / max(N_links_total, 1)) if N_links_total    > 0 else float("nan"),
            "link_mape":  (sum_mape_link / N_links_mape)                 if N_links_mape     > 0 else float("nan"),

            "route_mae":  (sum_abs_route  / N_routes_metric)             if N_routes_metric  > 0 else float("nan"),
            "route_rmse": math.sqrt(sum_sse_route / max(N_routes_metric,1)) if N_routes_metric > 0 else float("nan"),
            "route_mape": (sum_mape_route / N_routes_mape)               if N_routes_mape    > 0 else float("nan"),
        }
        return out


    def combined_metric(va):
        return selection_metric(va["route_mape"], va["link_mape"], w_route=0.7, w_link=0.3)

    num_epochs = NUM_EPOCHS
    patience = PATIENCE
    since_best = 0
    best_comb = float("inf")
    best_state = None

    tr_last = va_last = None
    import json as _json
    import time as _time
    _hist = {"train": {}, "val": {}}
    _t_start = _time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    _grad_norms.clear()

    for epoch in range(1, num_epochs + 1):
        _t0 = _time.perf_counter()
        tr = one_epoch_global(train_loader, epoch=epoch, train=True)
        _t1 = _time.perf_counter()
        va = one_epoch_global(val_loader,   epoch=epoch, train=False)
        _g = np.asarray(_grad_norms, dtype=float); _grad_norms.clear()
        _fin = _g[np.isfinite(_g)]
        _trk = {"train_time_s": _t1 - _t0, "val_time_s": _time.perf_counter() - _t1, "n_steps": int(len(_g)),
                "grad_norm_nonfinite": int(len(_g) - len(_fin)),
                "grad_norm_mean": float(_fin.mean()) if len(_fin) else float("nan"),
                "grad_norm_p99": float(np.percentile(_fin, 99)) if len(_fin) else float("nan"),
                "grad_norm_max": float(_fin.max()) if len(_fin) else float("nan"),
                "frac_steps_clipped": 0.0,
                "peak_gpu_mb": float(torch.cuda.max_memory_allocated() / 2**20) if torch.cuda.is_available() else float("nan")}
        for _k, _v in _trk.items():
            _hist["train"].setdefault(_k, []).append(_v)
        for _k in ("link_loss", "route_loss", "link_mae", "link_rmse", "link_mape", "route_mae", "route_rmse", "route_mape"):
            _hist["train"].setdefault(_k, []).append(tr.get(_k, float("nan")))
            _hist["val"].setdefault(_k, []).append(va.get(_k, float("nan")))
        print(f"[track] epoch {epoch:02d}: train {_trk['train_time_s']:.0f}s val {_trk['val_time_s']:.0f}s | "
              f"grad norm mean {_trk['grad_norm_mean']:.3f} p99 {_trk['grad_norm_p99']:.3f} max {_trk['grad_norm_max']:.3f} "
              f"clipped {_trk['frac_steps_clipped']:.2%} non-finite {_trk['grad_norm_nonfinite']} | "
              f"peak GPU {_trk['peak_gpu_mb']:.0f} MB", flush=True)

        tr_last, va_last = tr, va
        global_step += len(train_loader)

        comb = combined_metric(va)

        improved = comb < best_comb - 1e-6
        if improved:
            best_comb = comb
            since_best = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            ckpt.save_best(model=model, optimizer=optimizer, scheduler=scheduler,
                           epoch=epoch, step=global_step, best_metric=float(best_comb),
                           extra={"val_metrics": va})
        else:
            since_best += 1

        ckpt.save_last(model=model, optimizer=optimizer, scheduler=scheduler,
                       epoch=epoch, step=global_step, best_metric=float(best_comb),
                       extra={"val_metrics": va})

        print(
            f"Epoch {epoch:02d} | "
            f"Train: Link {tr['link_loss']:.4f}/{tr['link_mae']:.3f} (MAPE={tr['link_mape']:.3f}) (RMSE={tr['link_rmse']:.3f}) "
            f"Route {tr['route_loss']:.4f}/{tr['route_mae']:.3f} (MAPE={tr['route_mape']:.3f}) (RMSE={tr['route_rmse']:.3f}) ||  "
            f"Val: Link {va['link_loss']:.4f}/{va['link_mae']:.3f} (MAPE={va['link_mape']:.3f}) (RMSE={va['link_rmse']:.3f}) "
            f"Route {va['route_loss']:.4f}/{va['route_mae']:.3f} (MAPE={va['route_mape']:.3f}) (RMSE={va['route_rmse']:.3f}) "
            f"| Combined={comb:.4f}"
        )

        if since_best >= patience:
            print("Early stopping triggered.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    te = one_epoch_global(test_loader, epoch=0, train=False)

    print("— Final Test —")
    print(f"Link:  Huber={te['link_loss']:.4f} | MAE={te['link_mae']:.4f} | RMSE={te['link_rmse']:.4f} | MAPE={te['link_mape']:.4f}")
    print(f"Route: MAPE_Loss={te['route_loss']:.4f} | MAE={te['route_mae']:.4f} | RMSE={te['route_rmse']:.4f} | MAPE={te['route_mape']:.4f}")

    _hp, _sp = os.path.join(ckpt.dir, "training_history.json"), os.path.join(ckpt.dir, "training_summary.json")
    with open(_hp, "w", encoding="utf-8") as _f:
        _json.dump(_hist, _f, indent=2, default=str)
    with open(_sp, "w", encoding="utf-8") as _f:
        _json.dump({"test": te, "best_selector": float(best_comb), "train_last": tr_last, "val_last": va_last,
                    "epochs_run": len(_hist["train"].get("train_time_s", [])),
                    "total_train_time_s": float(_time.perf_counter() - _t_start),
                    "max_peak_gpu_mb": float(max(_hist["train"].get("peak_gpu_mb", [float("nan")]))),
                    "grad_norm_max_all": float(np.nanmax(_hist["train"].get("grad_norm_max", [float("nan")]))),
                    "grad_norm_nonfinite_all": int(sum(_hist["train"].get("grad_norm_nonfinite", [0]))),
                    "clip_norm": None, "patience": patience,
                    "switches": {"NO_FUTURE_FALLBACK": NO_FUTURE_FALLBACK, "HIGH_ORDER_TRAIN_ONLY": HIGH_ORDER_TRAIN_ONLY,
                                 "DRIVEN_LEN": DRIVEN_LEN}}, _f, indent=2, default=str)
    print(f"[history] {_hp}")
    print(f"[summary] {_sp}")

    return dict(train_last=tr_last, val_last=va_last, test=te, best_selector=float(best_comb))

