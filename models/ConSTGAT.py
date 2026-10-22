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

import os, time

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
    selection_metric, L_MIN_METRIC, T_MIN_METRIC, build_seq_mask
)
from checkpoint import CheckpointManager

CKPT_DIRNAME = "ckpts_ConSTGAT"


import os, re, math, random, time
from collections import defaultdict
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch_sparse import SparseTensor
from torch_geometric.data import Data
from shapely.geometry import LineString, MultiLineString

def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
set_seed(42)

H_HIST       = 12
SLOT_BIN_MIN = 10
MASK_RATE    = 0.10
USE_A2       = False
BATCH_N_WORKERS = int(os.environ.get("CONSTGAT_WORKERS", "0"))
DYN_FAMILIES_PAPER = ["median_speed", "mean_speed", "min_speed", "max_speed", "count_vehicle"]
SPEED_HISTORY = None


def cols_if_exist(df: pd.DataFrame, names):
    return [c for c in names if c in df.columns]

def build_slot_cols(df: pd.DataFrame, family: str, num_slots: int):
    pairs = []
    for c in df.columns:
        if isinstance(c, str) and c.startswith(f"{family}_"):
            s = c.split("_")[-1]
            if s.isdigit(): pairs.append((int(s), c))
    pairs.sort(key=lambda z: z[0])
    return [c for _, c in pairs[:num_slots]]


def add_time_feats(df_all: pd.DataFrame) -> pd.DataFrame:
    d = df_all.copy()
    d['departure_time'] = pd.to_datetime(d['departure_time'], errors='coerce')
    d = d.dropna(subset=['departure_time'])
    tod = d['departure_time'].dt.hour + d['departure_time'].dt.minute/60.0
    tod_rad = 2*np.pi*(tod/24.0)
    d['tod_sin'] = np.sin(tod_rad).astype('float32')
    d['tod_cos'] = np.cos(tod_rad).astype('float32')
    dow = d['departure_time'].dt.dayofweek
    dow_rad = 2*np.pi*(dow/7.0)
    d['dow_sin'] = np.sin(dow_rad).astype('float32')
    d['dow_cos'] = np.cos(dow_rad).astype('float32')
    return d


def time_feats_for_slots(t_ref: List[pd.Timestamp], H: int, slot_min: int) -> np.ndarray:
    out = np.zeros((len(t_ref), H, 2), dtype=np.int64)
    for i, tref in enumerate(t_ref):
        for k in range(H):
            dt = tref - pd.Timedelta(minutes=(H-1-k)*slot_min)
            out[i, k, 0] = dt.dayofweek
            out[i, k, 1] = (dt.hour * 60 + dt.minute) // slot_min
    return out


def _start_end_xy(geom):
    if isinstance(geom, LineString):
        (x0,y0) = geom.coords[0]; (x1,y1) = geom.coords[-1]
        return (round(x0,6), round(y0,6)), (round(x1,6), round(y1,6))
    if isinstance(geom, MultiLineString):
        s = list(geom.geoms[0].coords)[0]
        e = list(geom.geoms[-1].coords)[-1]
        return (round(s[0],6), round(s[1],6)), (round(e[0],6), round(e[1],6))
    return None, None

def build_nodes_edges(edges_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    eg = edges_gdf.copy()
    if 'way_id' not in eg.columns and 'id' in eg.columns:
        eg = eg.rename(columns={'id':'way_id'})
    starts, ends = [], []
    for g in eg.geometry:
        s,e = _start_end_xy(g); starts.append(s); ends.append(e)
    pts = pd.Index(starts + ends)
    codes, _ = pd.factorize(pts)
    n = len(eg)
    eg['u'] = codes[:n].astype('int64')
    eg['v'] = codes[n:].astype('int64')
    eg = eg.reset_index(drop=True)
    eg['edge_idx'] = eg.index.astype('int64')
    if 'length' not in eg.columns and 'link_len_m' in eg.columns:
        eg['length'] = pd.to_numeric(eg['link_len_m'], errors='coerce')
    return eg


def build_first_order_relations(edges_gdf: gpd.GeoDataFrame):
    u2e, v2e = defaultdict(list), defaultdict(list)
    for row in edges_gdf.itertuples(index=False):
        u2e[row.u].append(row.edge_idx)
        v2e[row.v].append(row.edge_idx)
    self_src, self_dst = [], []
    down_src, down_dst = [], []
    up_src,   up_dst   = [], []
    jn_src, jn_dst = [], []
    for row in edges_gdf.itertuples(index=False):
        i, u, v = row.edge_idx, row.u, row.v
        self_src.append(i); self_dst.append(i)
        for j in u2e[v]:
            if j != i: down_src.append(i); down_dst.append(j)
        for j in v2e[u]:
            if j != i: up_src.append(i); up_dst.append(j)
        for j in u2e[u]:
            if j != i: jn_src.append(i); jn_dst.append(j)
        for j in v2e[v]:
            if j != i: jn_src.append(i); jn_dst.append(j)
    to_t = lambda a,b: torch.tensor([a,b], dtype=torch.long)
    return to_t(self_src,self_dst), to_t(down_src,down_dst), to_t(up_src,up_dst), to_t(jn_src,jn_dst)


def map_df_all(df_all: pd.DataFrame, eg: gpd.GeoDataFrame) -> pd.DataFrame:
    d  = df_all.copy()
    if 'way_id' not in d.columns:
        raise KeyError("df_all must contain 'way_id'")

    if 'way_id' not in eg.columns and 'id' in eg.columns:
        eg = eg.rename(columns={'id':'way_id'})

    eg = eg.copy()
    eg['way_id'] = eg['way_id'].astype(str)
    d['way_id']  = d['way_id'].astype(str)

    id2edge = dict(zip(eg['way_id'], eg['edge_idx']))
    d['edge_idx'] = d['way_id'].map(id2edge)

    cov = d['edge_idx'].notna().mean()
    print(f"[CONSTGAT] way_id→edge_idx coverage: {cov:.2%}  (rows={len(d)})")
    if cov == 0.0:
        raise RuntimeError("Mapping coverage is 0%. Check that way_id values/types match between df_all and edges_gdf.")

    d = d.dropna(subset=['edge_idx'])
    d['edge_idx'] = d['edge_idx'].astype(int)

    if 'link_len_m' not in d.columns:
        d['link_len_m'] = np.nan
    d['length'] = d.get('length', np.nan)
    d['link_len_m'] = d['link_len_m'].fillna(d['length']).fillna(5.0)
    d.loc[d['link_len_m'] <= 0, 'link_len_m'] = 5.0

    return d

def split_with_gap(df_all: pd.DataFrame, gap_days=7):
    first_dep = df_all.groupby('TRIP_ID')['departure_time'].min().rename('first_dep').reset_index()
    d = df_all.merge(first_dep, on='TRIP_ID', how='left')
    min_date, max_date = d['first_dep'].min(), d['first_dep'].max()
    total_days = max((max_date - min_date).days, 1)
    train_cut = min_date + pd.Timedelta(days=int(0.6*total_days))
    val_cut   = min_date + pd.Timedelta(days=int(0.8*total_days))
    gap = pd.Timedelta(days=min(7, max(1, total_days // 20)))
    if val_cut - train_cut <= gap:
        gap = pd.Timedelta(hours=12) if val_cut - train_cut > pd.Timedelta(hours=12) else pd.Timedelta(hours=6)
    df_train = d[d['first_dep'] <= train_cut].copy()
    df_val   = d[(d['first_dep'] > train_cut + gap) & (d['first_dep'] <= val_cut)].copy()
    df_test  = d[d['first_dep'] > val_cut].copy()
    for _df in (df_train, df_val, df_test):
        if 'first_dep' in _df.columns: _df.drop(columns='first_dep', inplace=True)
    return df_train, df_val, df_test


def build_background_table(df_all: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    candidate_static = [
        'length','lanes','width_m','speed_limit_ms','link_len_m',
        'has_traffic_signal'
    ]
    time_feats = ['tod_sin','tod_cos','dow_sin','dow_cos']
    keep_cols = cols_if_exist(df_all, candidate_static + time_feats + ['TRIP_ID','edge_idx'])
    if not keep_cols:
        inst = pd.DataFrame(index=pd.MultiIndex.from_frame(df_all[['TRIP_ID','edge_idx']].drop_duplicates()))
        return inst, []
    inst_raw = (df_all[keep_cols]
                .drop_duplicates(subset=['TRIP_ID','edge_idx'], keep='first')
                .set_index(['TRIP_ID','edge_idx'])
                .fillna(0.0))
    inst_cols = [c for c in inst_raw.columns]
    return inst_raw, inst_cols


def build_static_table(df_all: pd.DataFrame, eg: gpd.GeoDataFrame) -> Tuple[pd.DataFrame, List[str]]:
    candidate_static = [
        'length','lanes','width_m','speed_limit_ms',
        'has_traffic_signal'
    ]
    cols = cols_if_exist(df_all, candidate_static + ['edge_idx'])
    if not cols:
        st = pd.DataFrame(index=eg['edge_idx'], data={'dummy': 0.0})
        return st, ['dummy']
    st = (df_all[cols]
          .drop_duplicates(subset=['edge_idx'], keep='first')
          .set_index('edge_idx')
          .reindex(eg['edge_idx'])
          .fillna(0.0))
    return st, list(st.columns)

TIME_FIX = True
MAX_GAP_S = 1800
SLOT_ORDER_FIX = True


class EdgeDynHistory:
    def __init__(self, df_split: pd.DataFrame, families=None, H=12, time_col="departure_time"):
        self.H = int(H)
        self.time_col = time_col
        self.families = list(families or ["mean_speed"])

        d = df_split.copy()
        d[time_col] = pd.to_datetime(d[time_col], errors='coerce')
        d = d.dropna(subset=[time_col, "edge_idx"])

        fam_cols = {}
        valid_fams = []
        for fam in self.families:
            cols = build_slot_cols(d, fam, self.H)
            if len(cols) > 0:
                fam_cols[fam] = cols[-self.H:] if len(cols) >= self.H else cols
                if SLOT_ORDER_FIX:
                    fam_cols[fam] = list(reversed(fam_cols[fam]))
                valid_fams.append(fam)

        if not valid_fams:
            valid_fams = ["__dummy__"]
            fam_cols["__dummy__"] = []

        self.valid_families = valid_fams
        self.D = len(valid_fams)

        self.idx = {}
        for e, grp in d.groupby("edge_idx"):
            if TIME_FIX:
                grp = grp.sort_values(time_col, kind="mergesort")
            t64 = grp[time_col].astype("int64").to_numpy()

            per_fam = []
            for fam in self.valid_families:
                cols = fam_cols.get(fam, [])
                if cols:
                    mat_f = grp[cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).to_numpy(dtype="float32")
                    if mat_f.ndim == 1:
                        mat_f = mat_f.reshape(-1, 1)
                    if mat_f.shape[1] < self.H:
                        pad = np.zeros((mat_f.shape[0], self.H - mat_f.shape[1]), dtype="float32")
                        mat_f = np.concatenate([mat_f, pad], axis=1)
                    else:
                        mat_f = mat_f[:, -self.H:]
                else:
                    mat_f = np.zeros((len(grp), self.H), dtype="float32")
                per_fam.append(mat_f[:, :, None])

            mat = np.concatenate(per_fam, axis=2) if per_fam else np.zeros((len(grp), self.H, self.D), dtype="float32")
            self.idx[int(e)] = (t64, mat)

    def get(self, edge_idx: int, t_ref) -> torch.Tensor:
        if edge_idx not in self.idx:
            return torch.zeros(self.H, self.D, dtype=torch.float32)
        t64, mat = self.idx[edge_idx]
        tref = np.int64(pd.Timestamp(t_ref).value)
        pos = np.searchsorted(t64, tref, side="right")
        if TIME_FIX:
            if pos > 0 and (int(tref) - int(t64[pos - 1])) / 1e9 <= MAX_GAP_S:
                return torch.tensor(mat[pos - 1], dtype=torch.float32)
            return torch.zeros(self.H, self.D, dtype=torch.float32)
        cand = []
        if pos > 0: cand.append(pos - 1)
        if pos < len(t64): cand.append(pos)
        if not cand:
            return torch.zeros(self.H, self.D, dtype=torch.float32)
        best = min(cand, key=lambda c: abs(int(t64[c]) - int(tref)))
        return torch.tensor(mat[best], dtype=torch.float32)


class TripDataset(Dataset):
    def __init__(self, df_split, bg_table, H_hist=12):
        super().__init__()
        self.H = int(H_hist)
        self.bg_table = (bg_table if isinstance(bg_table.index, pd.MultiIndex)
                          else bg_table.set_index(['TRIP_ID','edge_idx']))

        sort_col = 'enter_ts' if 'enter_ts' in df_split.columns else 'departure_time'
        need = ['TRIP_ID','edge_idx','link_len_m','link_travel_time_s', sort_col, 'departure_time', 'departure_ts']
        df_small = df_split[cols_if_exist(df_split, need)].copy()
        for c in ['link_len_m','link_travel_time_s']:
            if c in df_small.columns: df_small[c] = pd.to_numeric(df_small[c], errors='coerce')

        self.groups = []
        for trip_id, grp in df_small.groupby('TRIP_ID', sort=False):
            grp = grp.sort_values(sort_col)
            edge_seq_np = pd.to_numeric(grp['edge_idx'], errors='coerce').dropna().astype(int).to_numpy()
            if edge_seq_np.size == 0: continue
            L = int(edge_seq_np.size)

            try:
                sub = self.bg_table.xs(trip_id, level=0, drop_level=True)
                X = sub.reindex(edge_seq_np, fill_value=0.0).to_numpy(dtype='float32', copy=False)
            except KeyError:
                X = np.zeros((L, self.bg_table.shape[1]), dtype='float32')
            x_bg = torch.tensor(X, dtype=torch.float32)

            if TIME_FIX:
                t_dep = pd.to_datetime(grp['departure_time'].iloc[0])
                t_each = [t_dep] * L
            else:
                t_each = pd.to_datetime(grp[sort_col]).tolist()

            true_tt = torch.tensor(pd.to_numeric(grp['link_travel_time_s'], errors='coerce')
                                   .fillna(0.0).to_numpy(), dtype=torch.float32)
            link_len = torch.tensor(pd.to_numeric(grp['link_len_m'], errors='coerce')
                                    .fillna(5.0).to_numpy(), dtype=torch.float32).clamp_min_(1e-3)
            trip_tt = torch.tensor(float(np.nansum(true_tt.numpy())), dtype=torch.float32)

            self.groups.append({
                'trip_id': trip_id,
                'edge_seq': torch.tensor(edge_seq_np, dtype=torch.long),
                'x_bg': x_bg,
                't_ref': t_each,
                'enter_ts': pd.to_numeric(grp['enter_ts'], errors='raise').to_numpy(float),
                'departure_ts': float(grp['departure_ts'].iloc[0]),
                'true_tt': true_tt,
                'link_len': link_len,
                'trip_tt': trip_tt
            })

    def __len__(self): return len(self.groups)
    def __getitem__(self, i): return self.groups[i]


def build_graph(eg, self_edge_index, down_edge_index, up_edge_index, jn_edge_index):
    N = int(eg['edge_idx'].max()) + 1
    graph = Data()
    graph.N = N
    graph.edge_index_0 = self_edge_index
    graph.edge_index_1 = down_edge_index
    graph.edge_index_2 = up_edge_index
    graph.edge_index_3 = jn_edge_index
    graph.adj_t = []
    for r in range(4):
        ei = getattr(graph, f'edge_index_{r}')
        adj = SparseTensor(row=ei[0], col=ei[1], sparse_sizes=(N, N))
        graph.adj_t.append(adj)
    return graph


def build_neighbor_lut(graph: Data) -> Dict[int, List[int]]:
    lut = defaultdict(set)
    for adj in graph.adj_t:
        row, col, _ = adj.coo()
        row = row.cpu().numpy(); col = col.cpu().numpy()
        for r, c in zip(row, col):
            lut[int(r)].add(int(c))
    for i in range(graph.N):
        lut[i].add(i)
    return {k: sorted(list(v)) for k, v in lut.items()}


def make_collate_fn(hist: EdgeDynHistory,
                    neighbor_lut: Dict[int, List[int]],
                    static_table: pd.DataFrame,
                    with_masking: bool,
                    edge2way: Dict[int, int]):
    assert SPEED_HISTORY is not None and SPEED_HISTORY.bin_seconds == SLOT_BIN_MIN * 60, "SPEED_HISTORY not set"
    fam_ix = [DYN_FAMILIES_PAPER.index(f) for f in hist.valid_families]
    static_cols = list(static_table.columns)
    S = np.asarray(
        static_table.reindex(range(static_table.index.max() + 1)).fillna(0.0).to_numpy(),
        dtype=np.float32
    )
    dS = S.shape[1]
    Dtr = int(hist.D)

    def _collate(batch):
        B = len(batch)
        Ls = [len(b['edge_seq']) for b in batch]
        M = max(Ls) if Ls else 1

        nb_counts = []
        for b in batch:
            for e in b['edge_seq'].cpu().tolist():
                nb_counts.append(len(neighbor_lut.get(int(e), [int(e)])))
        NBmax = max(nb_counts) if nb_counts else 1

        edge_seq = torch.full((B, M), -1, dtype=torch.long)
        x_bg     = torch.zeros((B, M, batch[0]['x_bg'].shape[1] if B > 0 else 0), dtype=torch.float32)
        true_tt  = torch.zeros((B, M), dtype=torch.float32)
        link_len = torch.zeros((B, M), dtype=torch.float32)
        trip_tt  = torch.zeros((B,), dtype=torch.float32)
        valid    = torch.zeros((B, M), dtype=torch.bool)

        nb_idx   = torch.full((B, M, NBmax), -1, dtype=torch.long)
        nb_mask  = torch.zeros((B, M, NBmax), dtype=torch.bool)

        nb_hist  = torch.zeros((B, M, NBmax, hist.H, Dtr), dtype=torch.float32)
        time_fk  = torch.zeros((B, M, hist.H, 2), dtype=torch.long)
        bg_slots = torch.zeros((B, 12, 2), dtype=torch.long)

        for b_idx, b in enumerate(batch):
            L = len(b['edge_seq'])
            e_list = b['edge_seq'].cpu().tolist()

            edge_seq[b_idx, :L] = b['edge_seq']
            x_bg[b_idx, :L]     = b['x_bg'].to(torch.float32)
            true_tt[b_idx, :L]  = b['true_tt'].to(torch.float32)
            link_len[b_idx, :L] = b['link_len'].to(torch.float32)
            trip_tt[b_idx]      = b['trip_tt'].to(torch.float32)
            valid[b_idx, :L]    = True

            t_slots = time_feats_for_slots(b['t_ref'], hist.H, SLOT_BIN_MIN)
            time_fk[b_idx, :L] = torch.from_numpy(t_slots)
            bg_slots[b_idx] = torch.from_numpy(time_feats_for_slots(
                [b['t_ref'][0] + pd.Timedelta(minutes=11 * SLOT_BIN_MIN)], 12, SLOT_BIN_MIN)[0])

            for q, e in enumerate(e_list):
                nb = neighbor_lut.get(int(e), [int(e)])
                nb = nb[:NBmax] + [-1] * max(0, NBmax - len(nb))
                nb_idx[b_idx, q, :] = torch.tensor(nb, dtype=torch.long)
                nb_valid = [n >= 0 for n in nb]
                nb_mask[b_idx, q, :] = torch.tensor(nb_valid, dtype=torch.bool)

                tref = b['t_ref'][q]
                for j, n in enumerate(nb):
                    if n < 0:
                        continue
                    end = b['departure_ts']
                    pic = SPEED_HISTORY.slots(edge2way[int(n)], end, int(b['trip_id']))[:hist.H]
                    if SLOT_ORDER_FIX:
                        pic = pic[::-1]
                    nb_hist[b_idx, q, j] = torch.from_numpy(np.ascontiguousarray(pic[:, fam_ix], dtype=np.float32))

        if with_masking and MASK_RATE > 0:
            mask = torch.rand(B, M, hist.H) < MASK_RATE
            nb_hist = nb_hist.masked_fill(((nb_idx.unsqueeze(-1) == edge_seq[:, None, None, :]).float()
                                           @ mask.float().unsqueeze(1)).gt(0).unsqueeze(-1), 0.0)

        B_, M_, NB_ = nb_idx.shape
        nb_static = torch.zeros((B_, M_, NB_, dS), dtype=torch.float32)
        flat_idx = nb_idx.view(-1).cpu().numpy()
        valid_flat = flat_idx >= 0
        if valid_flat.any():
            nb_static_np = np.zeros((flat_idx.shape[0], dS), dtype=np.float32)
            nb_static_np[valid_flat] = S[flat_idx[valid_flat]]
            nb_static = torch.from_numpy(nb_static_np.reshape(B_, M_, NB_, dS))

        trip_ids = torch.tensor([b.get("TRIP_ID", b.get("trip_id", -1)) for b in batch], dtype=torch.long)

        return {
            'edge_seq': edge_seq,
            'x_bg': x_bg,
            'true_tt': true_tt,
            'link_len': link_len,
            'trip_tt': trip_tt,
            'valid_mask': valid,
            'nb_idx': nb_idx,
            'nb_mask': nb_mask,
            'nb_hist': nb_hist,
            'time_fk': time_fk,
            'nb_static': nb_static,
            'trip_ids': trip_ids,
            'bg_slots': bg_slots,
        }
    return _collate


class ContextualRouteCNN(nn.Module):
    def __init__(self, dim: int, w: int = 2):
        super().__init__()
        k = 2*w + 1
        self.net = nn.Sequential(nn.Conv1d(dim, 32, k, padding=w), nn.ReLU())
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h.transpose(1, 2)).transpose(1, 2)


class ST3DAttention(nn.Module):
    def __init__(self, Dq: int, dS: int, dT: int, H_hist: int, d_traffic: int,
                 heads: int = 4, Dh: Optional[int] = None):
        super().__init__()
        self.H = H_hist
        self.heads = heads
        self.Dh = Dh if Dh is not None else Dq
        assert self.Dh % heads == 0, "Dh must be divisible by heads"
        self.d_k = self.Dh // heads
        self.d_traffic = int(d_traffic)

        assert Dq == self.Dh, "the query (ConSTGAT.q_fuse) must have the attention size d(H)"
        d_mst = self.d_traffic + dS + dT
        self.k_proj = nn.Sequential(nn.Linear(d_mst, self.Dh), nn.ReLU())
        self.v_proj = nn.Sequential(nn.Linear(d_mst, self.Dh), nn.ReLU())

    def forward(self, q_in: torch.Tensor, nb_hist: torch.Tensor, nb_static: torch.Tensor,
            time_fk: torch.Tensor, nb_mask: torch.Tensor) -> torch.Tensor:
        B, M, NB, H, Dtr = nb_hist.shape
        assert Dtr == self.d_traffic, "nb_hist channel dim mismatch"

        h, d = self.heads, self.d_k

        time_fk_exp   = time_fk.unsqueeze(2).expand(B, M, NB, H, time_fk.size(-1))
        nb_static_exp = nb_static.unsqueeze(3).expand(B, M, NB, H, nb_static.size(-1))

        mst = torch.cat([nb_hist, nb_static_exp, time_fk_exp], dim=-1)

        K = self.k_proj(mst).view(B, M, NB, H, h, d).permute(0, 4, 1, 2, 3, 5)
        V = self.v_proj(mst).view(B, M, NB, H, h, d).permute(0, 4, 1, 2, 3, 5)
        Q = q_in.view(B, M, q_in.size(2), h, d).permute(0, 3, 2, 1, 4)
        scores = torch.einsum('bhsmd,bhmntd->bhsmnt', Q, K) / math.sqrt(d)

        mask = nb_mask[:, None, None, :, :, None].expand_as(scores)
        scores = scores.masked_fill(~mask, float('-inf'))

        scores_flat = scores.reshape(B, h, q_in.size(2), M, NB * H)
        alpha = torch.softmax(scores_flat, dim=-1)
        alpha = torch.nan_to_num(alpha, nan=0.0)

        V_flat = V.reshape(B, h, M, NB * H, d)
        ctx = torch.einsum('bhsmk,bhmkd->bhsmd', alpha, V_flat)
        return ctx.permute(0, 3, 2, 1, 4).reshape(B, M, q_in.size(2), h * d)

class ConSTGAT(nn.Module):
    def __init__(self, in_dim_B: int, dS: int, num_edges: int, d_traffic: int,
                 H_hist: int = 12, heads: int = 4,
                 hidden: int = 64, max_route_len: int = 2048, w_ctx: int = 1,
                 use_pos_emb: bool = False, edge_type: Optional[torch.Tensor] = None):
        super().__init__()
        self.H_hist = H_hist
        self.hidden = hidden
        self.heads  = heads
        self.use_pos_emb = use_pos_emb

        self.edge_emb = nn.Embedding(num_embeddings=num_edges, embedding_dim=8)
        assert edge_type is not None and edge_type.numel() == num_edges, "pass edge_type: one road type per link"
        self.register_buffer("edge_type", edge_type.long())
        self.type_emb = nn.Embedding(num_embeddings=int(edge_type.max()) + 1, embedding_dim=8)
        self.dow_emb = nn.Embedding(num_embeddings=7, embedding_dim=8)
        self.tod_emb = nn.Embedding(num_embeddings=24 * 60 // SLOT_BIN_MIN, embedding_dim=8)
        if self.use_pos_emb:
            self.pos_emb  = nn.Embedding(max_route_len, hidden)

        self.route_ctx = ContextualRouteCNN(8 + 8 + in_dim_B - 4, w=w_ctx)

        self.q_fuse = nn.Sequential(nn.Linear(32 + 16, 32), nn.ReLU())
        self.st3d = ST3DAttention(Dq=32, dS=dS + 8 + 8, dT=16, H_hist=H_hist,
                                  d_traffic=int(d_traffic), heads=heads, Dh=32)

        self.fuse = nn.Sequential(nn.Linear(32 + 16 + 32, 64), nn.ReLU())
        self.eta_head = nn.Linear(64, 1)

    def forward(self, edge_seq: torch.Tensor, x_bg: torch.Tensor, pos_idx: torch.Tensor,
                nb_hist: torch.Tensor, nb_static: torch.Tensor, time_fk: torch.Tensor,
                nb_mask: torch.Tensor, onroute_mask: Optional[torch.Tensor] = None,
                nb_idx: Optional[torch.Tensor] = None, bg_slots: Optional[torch.Tensor] = None):
        assert nb_idx is not None, "pass nb_idx: the neighbour ids are part of X(S) (paper l.355-357)"
        assert bg_slots is not None, "pass bg_slots: x(B) of the future time slots (paper l.658-668, 713-714)"
        B, M, _ = x_bg.shape

        h_bg  = torch.cat([self.dow_emb(bg_slots[..., 0]), self.tod_emb(bg_slots[..., 1])], dim=-1)
        h_bg  = h_bg[:, None, :, :].expand(B, M, h_bg.size(1), h_bg.size(2))
        time_fk = torch.cat([self.dow_emb(time_fk[..., 0]), self.tod_emb(time_fk[..., 1])], dim=-1)

        h_id  = torch.cat([self.edge_emb(edge_seq.clamp_min(0)), self.type_emb(self.edge_type[edge_seq.clamp_min(0)]),
                           x_bg[..., :-4]], dim=-1)
        h_ctx = self.route_ctx(h_id)
        h_ctx = h_ctx[:, :, None, :].expand(B, M, bg_slots.size(1), h_ctx.size(-1))

        q_in  = self.q_fuse(torch.cat([h_ctx, h_bg], dim=-1))

        nb_static = torch.cat([nb_static, self.edge_emb(nb_idx.clamp_min(0)),
                               self.type_emb(self.edge_type[nb_idx.clamp_min(0)])], dim=-1)

        h_tc  = self.st3d(q_in=q_in, nb_hist=nb_hist, nb_static=nb_static, time_fk=time_fk, nb_mask=nb_mask)

        h     = self.fuse(torch.cat([h_ctx, h_bg, h_tc], dim=-1))
        table = F.softplus(self.eta_head(h).squeeze(-1))
        elapsed = torch.zeros(B, device=table.device, dtype=table.dtype)
        eta_link = torch.zeros(B, M, device=table.device, dtype=table.dtype)
        for i in range(M):
            k = torch.clamp(torch.div(elapsed, SLOT_BIN_MIN * 60, rounding_mode="floor").long(), max=table.size(2) - 1)
            eta_link[:, i] = table[:, i, :].gather(1, k.unsqueeze(1)).squeeze(1)
            elapsed = elapsed + eta_link[:, i].detach()

        if onroute_mask is not None:
            eta_route = (eta_link * onroute_mask.float()).sum(dim=1)
        else:
            eta_route = eta_link.sum(dim=1)

        return eta_link, eta_route


def constgat_get_core_components():
    eg = build_nodes_edges(edges_gdf)
    self_ei, down_ei, up_ei, jn_ei = build_first_order_relations(eg)
    graph = build_graph(eg, self_ei, down_ei, up_ei, jn_ei)

    d = map_df_all(df_all, eg)
    d = add_time_feats(d)

    bg_table, BG_COLS = build_background_table(d)
    assert BG_COLS[-4:] == ['tod_sin', 'tod_cos', 'dow_sin', 'dow_cos'], BG_COLS
    static_table, STATIC_COLS = build_static_table(d, eg)

    df_train, df_val, df_test = split_with_gap(d, gap_days=7)

    train_ds = TripDataset(df_train, bg_table, H_hist=H_HIST)
    val_ds   = TripDataset(df_val,   bg_table, H_hist=H_HIST)
    test_ds  = TripDataset(df_test,  bg_table, H_hist=H_HIST)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    neighbor_lut = build_neighbor_lut(graph)
    hist = EdgeDynHistory(d, families=DYN_FAMILIES_PAPER, H=H_HIST, time_col='departure_time')

    edge2way = dict(zip(eg["edge_idx"].astype(int), eg["way_id"].astype("int64")))
    collate_train = make_collate_fn(hist, neighbor_lut, static_table, with_masking=True, edge2way=edge2way)
    collate_eval  = make_collate_fn(hist, neighbor_lut, static_table, with_masking=False, edge2way=edge2way)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True,  collate_fn=collate_train, num_workers=BATCH_N_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False, collate_fn=collate_eval,  num_workers=BATCH_N_WORKERS)
    test_loader  = DataLoader(test_ds,  batch_size=16, shuffle=False, collate_fn=collate_eval,  num_workers=BATCH_N_WORKERS)

    IN_DIM_B = len(BG_COLS)
    dS = len(STATIC_COLS)
    D_TRAFFIC = int(hist.D)

    def _max_len(ds):
        return max((len(g['edge_seq']) for g in ds.groups), default=0)
    max_route_len = max(_max_len(train_ds), _max_len(val_ds), _max_len(test_ds)) + 16

    assert 'highway' in eg.columns and np.array_equal(eg['edge_idx'].to_numpy(), np.arange(len(eg))), \
        "edges need a 'highway' column and edge_idx = 0..N-1"
    edge_type = torch.tensor(pd.factorize(eg['highway'].fillna('unknown').astype(str), sort=True)[0], dtype=torch.long)
    model = ConSTGAT(
        in_dim_B=IN_DIM_B, dS=dS, num_edges=int(graph.N), d_traffic=D_TRAFFIC,
        H_hist=H_HIST, heads=1, hidden=64,
        max_route_len=int(max_route_len), w_ctx=1, use_pos_emb=False, edge_type=edge_type
    ).to(device)

    def link_huber_per_route(eta_link, true_tt, onroute_mask, delta=1.0):
        B, M = eta_link.shape
        loss_sum = 0.0
        n_routes = 0
        for b in range(B):
            vm = onroute_mask[b]
            if vm.any():
                l = huber_per_element(eta_link[b][vm], true_tt[b][vm], delta).mean()
                loss_sum += l
                n_routes += 1
        return loss_sum / max(n_routes, 1)

    core = dict(
        device=device,
        model=model,
        graph=graph,
        loaders=dict(train=train_loader, val=val_loader, test=test_loader),
        datasets=dict(train=train_ds, val=val_ds, test=test_ds),
        bg_cols=BG_COLS,
        static_cols=STATIC_COLS,
        constants=dict(H_HIST=H_HIST, L_MIN=10.0, T_MIN=1.00, EPS=1e-6),
    )
    return core

print("[CORE] Initialized. Call get_core_components() from your strategy runner.")

NUM_EPOCHS    = 100
LR            = 3e-4
WD            = 1e-4
CLIP_NORM     = 1.0
DELTA_HUBER   = 1.0
PATIENCE      = 100

def run_training(core, seed=42):
    torch.manual_seed(seed)

    device       = core["device"]
    model        = core["model"]
    train_loader = core["loaders"]["train"]
    val_loader   = core["loaders"]["val"]
    test_loader  = core["loaders"]["test"]
    consts       = core["constants"]
    L_MIN, T_MIN = consts["L_MIN"], consts["T_MIN"]

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=3e-6
    )

    ckpt = CheckpointManager(dirpath=CKPT_DIRNAME, max_keep=3)
    global_step = 0

    best_state = None
    best_combined = float('inf')
    since_best    = 0

    _grad_norms = []

    def one_epoch_global(loader, train=True, delta_huber=DELTA_HUBER):
        model.train() if train else model.eval()

        sum_link_loss = 0.0
        sum_route_loss = 0.0
        N_routes_total = 0
        sum_sse_link = 0.0
        sum_sse_route = 0.0
        sum_mae_link = 0.0
        sum_mae_route = 0.0
        sum_mape_link = 0.0; N_links_mape = 0
        sum_mape_route = 0.0; N_routes_mape = 0
        N_links_total = 0
        N_routes_metric = 0

        with torch.set_grad_enabled(train):
            for batch in loader:
                es           = batch['edge_seq'].to(device)
                onroute_mask = batch['valid_mask'].to(device)
                x_bg         = batch['x_bg'].to(device)
                true_tt      = torch.nan_to_num(batch['true_tt'].to(device))
                tripT        = torch.nan_to_num(batch['trip_tt'].to(device))
                link_len     = torch.nan_to_num(batch['link_len'].to(device)) if ('link_len' in batch) else None

                nb_idx    = batch['nb_idx'].to(device)
                nb_mask   = batch['nb_mask'].to(device)
                nb_hist   = batch['nb_hist'].to(device)
                time_fk   = batch['time_fk'].to(device)
                nb_static = batch['nb_static'].to(device)

                M = es.size(1)
                pos_idx = torch.arange(M, device=device).unsqueeze(0).expand(es.size(0), -1)

                eta_link, _ = model(
                    edge_seq=es, x_bg=x_bg, pos_idx=pos_idx,
                    nb_hist=nb_hist, nb_static=nb_static, time_fk=time_fk,
                    nb_mask=nb_mask, onroute_mask=onroute_mask, nb_idx=nb_idx,
                    bg_slots=batch['bg_slots'].to(device)
                )

                pred_seq   = eta_link
                seq_true   = true_tt
                seq_mask = build_seq_mask(batch, device)
                seq_len_m  = link_len
                trip_vec   = tripT

                pred_trip_seq = (torch.nan_to_num(pred_seq) * seq_mask.float()).sum(dim=1)

                l_link  = link_loss_huber_macro(pred_seq, seq_true, seq_mask, delta=delta_huber)
                l_route = route_loss_mape(pred_seq, trip_vec, seq_mask)
                loss = l_link + l_route

                if train:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if CLIP_NORM is not None:
                        _gn = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
                        _grad_norms.append(float(_gn))
                    optimizer.step()

                link_stats  = link_metrics_micro(pred_seq, seq_true, seq_mask, seq_len_m, L_min=L_MIN_METRIC, T_min=T_MIN_METRIC)
                route_stats = route_metrics_pooled(pred_seq, trip_vec, seq_mask)

                bsz = int(pred_seq.size(0))
                sum_link_loss  += float(l_link.item())  * bsz
                sum_route_loss += float(l_route.item()) * bsz
                N_routes_total += bsz

                n_links = int(link_stats["n_links"])
                if n_links > 0:
                    sum_mae_link  += link_stats["mae"]  * n_links
                    sum_sse_link  += (link_stats["rmse"] ** 2) * n_links
                    N_links_total += n_links
                if link_stats.get("n_mape", 0) > 0:
                    sum_mape_link += link_stats["mape"] * link_stats["n_mape"]
                    N_links_mape  += link_stats["n_mape"]

                n_routes = int(route_stats["n_routes"])
                if n_routes > 0:
                    sum_mae_route  += route_stats["mae"]  * n_routes
                    sum_sse_route  += (route_stats["rmse"] ** 2) * n_routes
                    N_routes_metric += n_routes
                if route_stats.get("n_mape", 0) > 0:
                    sum_mape_route += route_stats["mape"] * route_stats["n_mape"]
                    N_routes_mape  += route_stats["n_mape"]

        return dict(
            link_loss  = sum_link_loss  / max(N_routes_total, 1),
            route_loss = sum_route_loss / max(N_routes_total, 1),

            link_mae   = (sum_mae_link  / N_links_total)                 if N_links_total    > 0 else float("nan"),
            link_rmse  = math.sqrt(sum_sse_link  / max(N_links_total, 1)) if N_links_total    > 0 else float("nan"),
            link_mape  = (sum_mape_link / N_links_mape)                  if N_links_mape     > 0 else float("nan"),

            route_mae  = (sum_mae_route  / N_routes_metric)              if N_routes_metric  > 0 else float("nan"),
            route_rmse = math.sqrt(sum_sse_route / max(N_routes_metric,1)) if N_routes_metric > 0 else float("nan"),
            route_mape = (sum_mape_route / N_routes_mape)                if N_routes_mape    > 0 else float("nan"),
        )


    def combined_metric(stats):
        return selection_metric(stats["route_mape"], stats["link_mape"], w_route=0.7, w_link=0.3)

    tr_last = va_last = None
    _hist = {"train": {}, "val": {}}
    _t_start = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    _grad_norms.clear()

    for epoch in range(1, NUM_EPOCHS + 1):
        _t0 = time.perf_counter()
        tr = one_epoch_global(train_loader, train=True)
        _t1 = time.perf_counter()
        va = one_epoch_global(val_loader,   train=False)
        _g = np.asarray(_grad_norms, dtype=float); _grad_norms.clear()
        _fin = _g[np.isfinite(_g)]
        _trk = {"train_time_s": _t1 - _t0, "val_time_s": time.perf_counter() - _t1, "n_steps": int(len(_g)),
                "grad_norm_nonfinite": int(len(_g) - len(_fin)),
                "grad_norm_mean": float(_fin.mean()) if len(_fin) else float("nan"),
                "grad_norm_p99": float(np.percentile(_fin, 99)) if len(_fin) else float("nan"),
                "grad_norm_max": float(_fin.max()) if len(_fin) else float("nan"),
                "frac_steps_clipped": float((_fin > CLIP_NORM).mean()) if (len(_fin) and CLIP_NORM) else float("nan"),
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
            f"Train: Link {tr['link_loss']:.4f}/{tr['link_mae']:.3f} (MAPE={tr['link_mape']:.3f})  "
            f"Route {tr['route_loss']:.4f}/{tr['route_mae']:.3f} (MAPE={tr['route_mape']:.3f})  ||  "
            f"Val: Link {va['link_loss']:.4f}/{va['link_mae']:.3f} (MAPE={va['link_mape']:.3f})  "
            f"Route {va['route_loss']:.4f}/{va['route_mae']:.3f} (MAPE={va['route_mape']:.3f})  "
            f"| Combined={comb:.4f}"
        )

        if since_best >= PATIENCE:
            print("Early stopping triggered.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    te = one_epoch_global(test_loader, train=False)

    print("— Final Test —")
    print(f"Link:  Huber={te['link_loss']:.4f} | MAE={te['link_mae']:.4f} | RMSE={te['link_rmse']:.4f} | MAPE={te['link_mape']:.4f}")
    print(f"Route: MAPE_Loss={te['route_loss']:.4f} | MAE={te['route_mae']:.4f} | RMSE={te['route_rmse']:.4f} | MAPE={te['route_mape']:.4f}")

    _hp, _sp = os.path.join(ckpt.dir, "training_history.json"), os.path.join(ckpt.dir, "training_summary.json")
    with open(_hp, "w", encoding="utf-8") as _f:
        json.dump(_hist, _f, indent=2, default=str)
    with open(_sp, "w", encoding="utf-8") as _f:
        json.dump({"test": te, "best_selector": float(best_combined), "train_last": tr_last, "val_last": va_last,
                   "epochs_run": len(_hist["train"].get("train_time_s", [])),
                   "total_train_time_s": float(time.perf_counter() - _t_start),
                   "max_peak_gpu_mb": float(max(_hist["train"].get("peak_gpu_mb", [float("nan")]))),
                   "grad_norm_max_all": float(np.nanmax(_hist["train"].get("grad_norm_max", [float("nan")]))),
                   "grad_norm_nonfinite_all": int(sum(_hist["train"].get("grad_norm_nonfinite", [0]))),
                   "clip_norm": CLIP_NORM, "patience": PATIENCE}, _f, indent=2, default=str)
    print(f"[history] {_hp}")
    print(f"[summary] {_sp}")

    return dict(train_last=tr_last, val_last=va_last, test=te, best_selector=float(best_combined))

