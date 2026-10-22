import time
import math
import re
import random
import json
import os, json, hashlib
from collections import defaultdict, deque
from typing import Optional, Sequence, Tuple, Dict, List
import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiLineString
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from checkpoint import CheckpointManager
from loss_functions import (
    L_MIN_METRIC, T_MIN_METRIC,
    selection_metric,
    link_metrics_micro, route_metrics_pooled,
    link_eta_in_seconds,
    link_loss_huber_log_lenweighted,
    route_loss_huber_log,
    compute_global_losses_and_metrics_linketa as compute_global_losses_and_metrics,
)

CKPT_DIRNAME = "ckpts_LinkETA"

USE_AMP = True

HEAD_MODE = "speed"
PACE_LOSS_BETA = 0.1
ROUTE_ORDER = "enter_ts"

TRAIN_TRACK = {"grad_norms": []}


def _epoch_track(t0, t1, t2, clip):
    import math as _m
    g = np.asarray(TRAIN_TRACK["grad_norms"], dtype=float)
    TRAIN_TRACK["grad_norms"] = []
    fin = g[np.isfinite(g)]
    out = {"train_time_s": t1 - t0, "val_time_s": t2 - t1, "n_steps": int(len(g)),
           "grad_norm_nonfinite": int(len(g) - len(fin)),
           "grad_norm_mean": float(fin.mean()) if len(fin) else float("nan"),
           "grad_norm_p99": float(np.percentile(fin, 99)) if len(fin) else float("nan"),
           "grad_norm_max": float(fin.max()) if len(fin) else float("nan"),
           "frac_steps_clipped": float((fin > clip).mean()) if (len(fin) and clip) else float("nan"),
           "peak_gpu_mb": float(torch.cuda.max_memory_allocated() / 2**20) if torch.cuda.is_available() else float("nan")}
    return out
TRAIN_SEED = 42
R_MAX = 2.8
LABEL_CAP = 3.0

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
    g = edges_gdf.copy()
    if "edge_idx" not in g.columns:
        g = g.reset_index(drop=True)
        g["edge_idx"] = g.index

    id2edge = dict(zip(g[edges_id_col].tolist(), g["edge_idx"].tolist()))
    if "edge_idx" not in df_all.columns:
        df_all = df_all.copy()
        df_all["edge_idx"] = df_all[way_id_df_col].map(id2edge)

    if "length" not in g.columns:
        tmp = (
            df_all.dropna(subset=["edge_idx", "length"])
            .groupby("edge_idx")["length"]
            .first()
        )
        g["length"] = g["edge_idx"].map(tmp)
    g["length"] = pd.to_numeric(g["length"], errors="coerce").fillna(0.0)

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


def route_order_key(df: pd.DataFrame) -> pd.Series:
    if "enter_ts" not in df.columns:
        raise KeyError("route_order_key: 'enter_ts' column is required to order a route.")
    e = df["enter_ts"]
    if pd.api.types.is_numeric_dtype(e):
        return pd.to_numeric(e, errors="coerce")
    return pd.to_datetime(e, errors="coerce").astype("int64").astype("float64")


def fit_scaler(inst, df_train, INST_FEAT_COLS):
    inst_indexed = inst if isinstance(inst.index, pd.MultiIndex) \
        else inst.set_index(['TRIP_ID', 'edge_idx'])

    train_inst_idx = pd.MultiIndex.from_frame(
        df_train[['TRIP_ID', 'edge_idx']].drop_duplicates()
    )
    train_inst = inst_indexed.loc[inst_indexed.index.intersection(train_inst_idx)]

    feat_mean = train_inst[INST_FEAT_COLS].mean().astype('float32').fillna(0.0)
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
    if val_cut - train_cut <= gap:
        gap = pd.Timedelta(hours=12) if val_cut - train_cut > pd.Timedelta(hours=12) else pd.Timedelta(hours=6)

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
    g = edges_gdf.copy()
    if ("u" not in g.columns) or ("v" not in g.columns):
        raise ValueError("edges_gdf must have u,v columns before building relations.")
    if "edge_idx" not in g.columns:
        raise ValueError("edges_gdf must have edge_idx before building relations.")

    g["bearing_in"] = g["geometry"].apply(_bearing_toward_end)

    out_by_u = g.groupby("u")["edge_idx"].apply(list).to_dict()
    in_by_v  = g.groupby("v")["edge_idx"].apply(list).to_dict()

    bmap = dict(zip(g["edge_idx"], g["bearing_in"]))

    up_src, up_dst = [], []
    dn_src, dn_dst = [], []
    lt_src, lt_dst = [], []
    rt_src, rt_dst = [], []
    op_src, op_dst = [], []
    ut_src, ut_dst = [], []

    SAME_THR   = math.radians(30)
    OPP_THR    = math.radians(30)
    UTURN_THR  = math.radians(20.0)

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

            d = _ang_diff(theta_j, theta_i)

            if abs(d) <= SAME_THR:
                continue

            if abs(abs(d) - math.pi) <= OPP_THR:
                op_src.append(i); op_dst.append(j)
                continue

            if 0 < d < (math.pi - OPP_THR):
                lt_src.append(i); lt_dst.append(j)
            elif -(math.pi - OPP_THR) < d < 0:
                rt_src.append(i); rt_dst.append(j)
            else:
                if d > 0:
                    lt_src.append(i); lt_dst.append(j)
                else:
                    rt_src.append(i); rt_dst.append(j)

    for row in g.itertuples(index=False):
        i = row.edge_idx
        v_i = row.v
        theta_i = bmap[i]

        for j in out_by_u.get(v_i, []):
            if j == i:
                continue
            theta_j = bmap[j]
            d = _ang_diff(theta_i, theta_j)

            if abs(abs(d) - math.pi) <= UTURN_THR:
                ut_src.append(i)
                ut_dst.append(j)

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


CACHE_DIR = None
CACHE_SCHEMA = "2026-09-29.v1"
SPEED_HISTORY = None


def _sha(*parts) -> str:
    h = hashlib.sha256()
    for p in parts:
        if isinstance(p, (bytes, bytearray)):
            h.update(p)
        else:
            h.update(repr(p).encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()


def hash_frame(df: pd.DataFrame, cols: Sequence[str]) -> str:
    cols = [c for c in cols if c in df.columns]
    parts = [len(df), tuple(cols), tuple(str(df[c].dtype) for c in cols)]
    for c in cols:
        parts.append(pd.util.hash_pandas_object(df[c], index=False).to_numpy().tobytes())
    return _sha(*parts)


def hash_tensors(tensors) -> str:
    parts = []
    for t in tensors:
        t = t.detach().cpu().contiguous()
        parts += [tuple(t.shape), str(t.dtype), t.numpy().tobytes()]
    return _sha(*parts)


def _cache_file(kind: str, fp: str) -> Optional[str]:
    if not CACHE_DIR:
        return None
    return os.path.join(CACHE_DIR, f"{kind}_{fp[:24]}.pt")


def _atomic_torch_save(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _serialize_groups(groups):
    out = []
    for g in groups:
        gg = dict(g)
        gg["masks"] = [(int(m.shape[0]), m.bool().nonzero(as_tuple=False).to(torch.int32)) for m in g["masks"]]
        out.append(gg)
    return out


def _deserialize_groups(groups):
    out = []
    for gg in groups:
        g = dict(gg)
        dense = []
        for M, idx in gg["masks"]:
            d = torch.zeros(M, M, dtype=torch.bool)
            if idx.numel():
                ii = idx.long()
                d[ii[:, 0], ii[:, 1]] = True
            dense.append(d)
        g["masks"] = dense
        out.append(g)
    return out


def load_cached(kind: str, fp: str, check=None):
    path = _cache_file(kind, fp)
    if path is None or not os.path.isfile(path):
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema") != CACHE_SCHEMA or payload.get("fingerprint") != fp:
            print(f"[cache] {kind}: header mismatch in {path} -> rebuild")
            return None
        if check is not None and not check(payload):
            print(f"[cache] {kind}: content check failed for {path} -> rebuild")
            return None
        print(f"[cache] {kind}: loaded {path}")
        return payload
    except Exception as e:
        print(f"[cache] {kind}: could not load {path} ({e}) -> rebuild")
        return None


def save_cached(kind: str, fp: str, payload: dict):
    path = _cache_file(kind, fp)
    if path is None:
        return
    try:
        payload = dict(payload, schema=CACHE_SCHEMA, fingerprint=fp,
                       created=time.strftime("%Y-%m-%d %H:%M:%S"))
        _atomic_torch_save(payload, path)
        print(f"[cache] {kind}: saved {path}")
    except Exception as e:
        print(f"[cache] {kind}: could not save {path} ({e}); continuing without cache")


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

class TripDatasetLinkETA(Dataset):
    def __init__(
        self,
        df_all: pd.DataFrame,
        edges_gdf: pd.DataFrame,
        inst_table: pd.DataFrame,
        rel_edges_list: List[torch.Tensor],
        static_feat_cols: List[str],
        time_feat_cols: List[str],
        dyn_feat_cols: List[str],
        center_idx: int = 0,
        K_first: int = 10,
        K_conflict: int = 5,
        K_high: int = 3,
        build_seed: Optional[int] = None,
        cache_fp: Optional[str] = None,
        split_name: str = "",
    ):
        super().__init__()
        self.df_all = df_all
        self.edges_gdf = edges_gdf
        assert SPEED_HISTORY is not None and SPEED_HISTORY.bin_seconds == BIN_SECONDS, "SPEED_HISTORY not set"
        assert CFG["dyn_prefixes"] == ["median_speed", "mean_speed", "min_speed", "max_speed", "count_vehicle"]
        _idc = guess_edges_id_col(edges_gdf, CFG["way_id_col_edges_candidates"])
        self.edge2way = dict(zip(edges_gdf["edge_idx"].astype(int), edges_gdf[_idc].astype("int64")))
        self.inst_table = inst_table
        self.static_feat_cols = static_feat_cols or []
        self.time_feat_cols = time_feat_cols or []
        self.dyn_feat_cols = dyn_feat_cols or []
        self.sort_col = CFG["time_col_df"] if CFG["time_col_df"] in df_all.columns else "enter_ts"

        self.idx_count = CFG["dyn_prefixes"].index("count_vehicle") \
            if "count_vehicle" in CFG["dyn_prefixes"] else None

        if not (isinstance(inst_table.index, pd.MultiIndex) and inst_table.index.names[:2] == ['TRIP_ID', 'edge_idx']):
            inst_table = inst_table.set_index(['TRIP_ID', 'edge_idx']).sort_index()
        self.inst_table = inst_table

        cols = list(self.inst_table.columns)
        self.dyn_idx = [cols.index(c) for c in self.dyn_feat_cols if c in cols]

        tau = 600.0 / max(math.log(2.0), 1e-6)
        idx = torch.arange(H_SLICES, dtype=torch.float32)
        ages = (idx - float(center_idx)).abs() * float(BIN_SECONDS)
        self.w_rec = torch.exp(-ages / max(tau, 1.0))

        self.rel_src2dst_list = [_build_src2dst(ei) for ei in rel_edges_list]

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

        self.hop_src2dst_full = self.rel_src2dst_list[1]

        self.groups = []

        def _tid(x):
            try:
                return str(int(x))
            except (TypeError, ValueError):
                return str(x)
        expected_ids = {_tid(x) for x in pd.unique(df_all["TRIP_ID"])}

        def _check(payload):
            ids = [_tid(g["trip_id"]) for g in payload["groups"]]
            return (payload.get("split") == split_name
                    and payload.get("n_groups") == len(ids)
                    and len(ids) == len(set(ids))
                    and set(ids) <= expected_ids)

        payload = load_cached(f"groups_{split_name}", cache_fp, _check) if cache_fp else None
        if payload is not None:
            self.groups = _deserialize_groups(payload["groups"])
        else:
            _st = random.getstate()
            if build_seed is not None:
                random.seed(int(build_seed))
            try:
                self._build_groups()
            finally:
                random.setstate(_st)
            if cache_fp:
                save_cached(f"groups_{split_name}", cache_fp,
                            dict(split=split_name, n_groups=len(self.groups),
                                 groups=_serialize_groups(self.groups)))

    def _build_groups(self):
        df_small = self.df_all[["TRIP_ID", "edge_idx", "link_len_m", "link_travel_time_s",
                                "trip_travel_time_s", self.sort_col]].copy()
        df_small["_route_order"] = route_order_key(self.df_all)
        df_small["_enter_s"] = pd.to_numeric(self.df_all["enter_ts"], errors="raise")
        for c in ('link_len_m', 'link_travel_time_s', 'trip_travel_time_s'):
            if c in df_small.columns:
                df_small[c] = pd.to_numeric(df_small[c], errors='coerce')

        for trip_id, grp in df_small.groupby('TRIP_ID', sort=False):
            if ROUTE_ORDER == "legacy":
                grp = grp.sort_values(self.sort_col)
            else:
                grp = grp.sort_values("_route_order", kind="mergesort")
            edge_seq_on = pd.to_numeric(grp['edge_idx'], errors='coerce').dropna().astype(int).to_numpy()
            if edge_seq_on.size == 0:
                continue
            t_arr_on = pd.to_datetime(grp[self.sort_col].to_numpy())
            L_on = len(edge_seq_on)
            ent_on = grp["_enter_s"].to_numpy(dtype=float)
            assert len(ent_on) == L_on
            drv = set(edge_seq_on.tolist())

            link_len_np = pd.to_numeric(grp['link_len_m'], errors='coerce').fillna(5.0).to_numpy()
            true_tt_np = pd.to_numeric(grp['link_travel_time_s'], errors='coerce').fillna(0.0).to_numpy()

            t_val = float(grp['trip_travel_time_s'].iloc[0]) \
                if 'trip_travel_time_s' in grp.columns and pd.notna(grp['trip_travel_time_s'].iloc[0]) \
                else float(np.nansum(true_tt_np))
            if not (np.isfinite(t_val) and t_val > 0):
                t_val = float(np.nansum(true_tt_np))
            trip_tt = torch.tensor(t_val, dtype=torch.float32)
            trip_tt_label = torch.tensor(float(np.nansum(true_tt_np)), dtype=torch.float32)

            max_nodes = int(CFG.get("max_nodes_local", 360))
            node_edge_idx = []
            node_tref = []
            route_flag = []

            neigh_parent = []

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
                    node_tref.append(t_ref)
                    route_flag.append(False)
                    neigh_parent.append(k)

            M = len(node_edge_idx)
            node_edge_idx = np.array(node_edge_idx, dtype=int)
            route_flag = torch.tensor(route_flag, dtype=torch.bool)
            neigh_parent_tensor = torch.tensor(neigh_parent, dtype=torch.long)

            route_seq_idx = []
            seq_true_tt = []
            seq_link_len = []
            seq_mask = []

            for k in range(L_on):
                node_pos = None
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

            try:
                sub = self.inst_table.xs(trip_id, level=0, drop_level=True)
                rows = sub.reindex(node_edge_idx, fill_value=0.0).to_numpy(dtype='float32', copy=False)
            except KeyError:
                Fdim = self.inst_table.shape[1]
                rows = np.zeros((M, Fdim), dtype='float32')
            x_seq = torch.tensor(rows, dtype=torch.float32)

            cols_all = self.inst_table.columns.tolist()
            for j in range(M):
                e = int(node_edge_idx[j])
                t_ref = node_tref[j]

                if self.time_feat_cols:
                    tf = _compute_time_feature_vector(t_ref, self.time_feat_cols)
                    for col_name, val in tf.items():
                        if col_name in cols_all:
                            idx_col = cols_all.index(col_name)
                            x_seq[j, idx_col] = float(val)

                if self.dyn_feat_cols and e not in drv:
                    pic = SPEED_HISTORY.slots(self.edge2way[e], ent_on[int(neigh_parent[j])], int(trip_id))[:H_SLICES]
                    flat = pic.reshape(-1).astype("float32")
                    m = min(len(flat), len(self.dyn_idx))
                    if m > 0:
                        x_seq[j, self.dyn_idx[:m]] = torch.from_numpy(flat[:m])

            if self.idx_count is not None:
                count_vals = x_seq[:, self.dyn_idx[:H_SLICES * len(CFG["dyn_prefixes"])]].reshape(
                    M, H_SLICES, len(CFG["dyn_prefixes"]))[:, :, self.idx_count]
                valid = (count_vals > 0).float()
                w = (self.w_rec.view(1, -1) * valid)
                logcnt1h_node = torch.log1p((count_vals * w).sum(dim=1))
            else:
                logcnt1h_node = torch.zeros(M)

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

            start_edge = int(edge_seq_on[0])
            dist_full = _bfs_hops_global(self.hop_src2dst_full, start_edge, cap=512)
            hop_idx = torch.tensor([dist_full.get(int(e), 512) for e in node_edge_idx], dtype=torch.long)

            self.groups.append(dict(
                trip_id=int(trip_id),
                node_edge_idx=torch.tensor(node_edge_idx, dtype=torch.long),
                x_seq=x_seq,
                route_flag=route_flag,
                hop_idx=hop_idx,
                masks=masks,
                logcnt1h_node=logcnt1h_node,
                route_seq_idx=route_seq_idx,
                seq_true_tt=seq_true_tt,
                seq_link_len=seq_link_len,
                seq_mask=seq_mask,
                trip_tt=trip_tt,
                trip_tt_label=trip_tt_label,
                neigh_parent=neigh_parent_tensor,
            ))

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
    logcnt1h_node = torch.zeros(B, Mmax, dtype=torch.float32)

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
    trip_ids = []

    for i, g in enumerate(batch):
        M = M_list[i]
        node_edge_idx[i, :M] = g["node_edge_idx"]
        x_seq[i, :M] = g["x_seq"]
        route_flag[i, :M] = g["route_flag"]
        hop_idx[i, :M] = g["hop_idx"]
        logcnt1h_node[i, :M] = g["logcnt1h_node"]
        for r in range(len(masks_per_rel)):
            masks_per_rel[r][i, :M, :M] = g["masks"][r]

        L = L_list[i]
        route_seq_idx[i, :L] = g["route_seq_idx"]
        seq_true_tt[i, :L] = g["seq_true_tt"]
        seq_link_len[i, :L] = g["seq_link_len"]
        seq_mask[i, :L] = g["seq_mask"]
        trip_tt[i] = g["trip_tt"]
        trip_tt_label[i] = g["trip_tt_label"]
        trip_ids.append(g["trip_id"])

    return dict(
        node_edge_idx=node_edge_idx,
        x_seq=x_seq,
        route_flag=route_flag,
        hop_idx=hop_idx,
        masks_per_rel=masks_per_rel,
        logcnt1h_node=logcnt1h_node,
        route_seq_idx=route_seq_idx,
        seq_true_tt=seq_true_tt,
        seq_link_len=seq_link_len,
        seq_mask=seq_mask,
        trip_tt=trip_tt,
        trip_tt_label=trip_tt_label,
        trip_ids=trip_ids,
    )


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
        B, M, D = x.shape
        agg = torch.zeros(B, M, self.heads, self.dk, device=x.device, dtype=x.dtype)

        for r in range(self.R):
            q = self.to_q[r](x).view(B, M, self.heads, self.dk)
            k = self.to_k[r](x).view(B, M, self.heads, self.dk)
            v = self.to_v[r](x).view(B, M, self.heads, self.dk)

            scores = torch.einsum("bmhd,bnhd->bhmn", q, k) / math.sqrt(self.dk)
            mask = masks[r].unsqueeze(1)
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


class LinkETA(nn.Module):
    def __init__(self, in_dim, hidden_dim=32, heads=8, relations=8,
                 num_layers=1, dropout=0.1,
                 input_dropout=0.1):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.relations = relations

        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.inp_dropout = nn.Dropout(input_dropout)
        self.route_enc = RouteAwareEncodings(max_hop=512, dim=hidden_dim)

        self.P = len(CFG["dyn_prefixes"])
        self.H_SLICES = CFG.get("slots_per_hour", 12)

        self.dyn_encoder = nn.Sequential(
            nn.Conv1d(self.P, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.dyn_pool = nn.AdaptiveAvgPool1d(1)
        self.dyn_gate = nn.Parameter(torch.tensor(0.0))

        self.layers = nn.ModuleList([
            RelationalTransformerLayer(hidden_dim, heads, relations, dropout=dropout)
            for _ in range(num_layers)
        ])

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

        self.residual_head = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, 1, 1),
        )

        self.route_calib = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.g_range     = 0.3
        self.delta_range = 0.4

        nn.init.zeros_(self.speed_head[-1].weight)
        nn.init.zeros_(self.speed_head[-1].bias)
        nn.init.zeros_(self.count_head[-1].weight)
        nn.init.zeros_(self.count_head[-1].bias)
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)
        nn.init.zeros_(self.route_calib[-1].weight)
        nn.init.zeros_(self.route_calib[-1].bias)

    def _encode_dyn_slots(self, x_seq_nodes: torch.Tensor) -> torch.Tensor:
        B, M, _ = x_seq_nodes.shape


        if "DYN_IDX" not in globals():
            return torch.zeros(B, M, self.hidden_dim, device=x_seq_nodes.device, dtype=x_seq_nodes.dtype)

        dyn_idx = DYN_IDX
        num_dyn = dyn_idx.numel()
        expected_dyn = self.H_SLICES * self.P
        if num_dyn != expected_dyn:
            return torch.zeros(B, M, self.hidden_dim, device=x_seq_nodes.device, dtype=x_seq_nodes.dtype)

        x_dyn = x_seq_nodes[..., dyn_idx]

        BM = B * M
        x_dyn = x_dyn.reshape(BM, self.H_SLICES, self.P).permute(0, 2, 1)

        dyn_feat = self.dyn_encoder(x_dyn)
        dyn_feat = self.dyn_pool(dyn_feat).squeeze(-1)
        dyn_feat = dyn_feat.view(B, M, self.hidden_dim)

        gate = torch.sigmoid(self.dyn_gate)
        return gate * dyn_feat

    def forward(self,
                x_seq_nodes,
                hop_idx,
                route_flag_nodes,
                masks_per_rel,
                route_seq_idx,
                seq_link_len,
                vff_seq,
                seq_mask,
                r_max=3.0,
                return_parts=True):
        B, M, _ = x_seq_nodes.size()
        L = route_seq_idx.size(1)

        x_seq_nodes = torch.nan_to_num(x_seq_nodes)

        h = self.input_proj(x_seq_nodes)
        h_dyn = self._encode_dyn_slots(x_seq_nodes)
        h = h + h_dyn

        h = self.inp_dropout(h)
        h = self.route_enc(h, hop_idx, route_flag_nodes.long())

        for layer in self.layers:
            h = layer(h, masks_per_rel)

        D = h.size(-1)
        gather_idx = route_seq_idx.unsqueeze(-1).expand(-1, -1, D)
        h_seq = torch.gather(h, 1, gather_idx)

        enc_t = h_seq.transpose(1, 2)

        speed_1h_raw = self.speed_head(enc_t).squeeze(1)
        logcnt_1h = self.count_head(enc_t).squeeze(1)
        sp_hat = torch.sigmoid(speed_1h_raw) * r_max

        seq_link_len = torch.nan_to_num(seq_link_len).clamp_min(1e-3)
        vff_seq      = torch.nan_to_num(vff_seq).clamp_min(1.0)
        seq_mask_f   = seq_mask.float()


        if HEAD_MODE == "pace":
            sp_hat   = F.softplus(speed_1h_raw) + 1.0 / r_max
            eta_base = seq_link_len * sp_hat / vff_seq
        else:
            v_pred   = (sp_hat.clamp_min(1e-3) * vff_seq).clamp_min(0.5)
            eta_base = seq_link_len / v_pred

        res_raw = self.residual_head(enc_t).squeeze(1)
        res_raw = torch.nan_to_num(res_raw)
        delta   = self.delta_range * torch.tanh(res_raw)

        eta_phys = eta_base * torch.exp(delta)

        route_repr = (h_seq * seq_mask_f.unsqueeze(-1)).sum(dim=1) / \
                     seq_mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
        g_raw = self.route_calib(route_repr).squeeze(-1)
        g     = 1.0 + self.g_range * torch.tanh(g_raw)

        link_eta_seq = eta_phys * g.unsqueeze(-1)
        link_eta_seq = link_eta_seq * seq_mask_f

        trip_eta = link_eta_seq.sum(dim=1)

        if return_parts:
            return (
                link_eta_seq,
                trip_eta,
                sp_hat,
                logcnt_1h,
                g,
                delta,
                eta_phys,
            )

        return link_eta_seq, trip_eta, sp_hat, logcnt_1h, g, delta, eta_phys

DWA_T = 2.0


def vff_noise_range(epoch):
    return 0.10 if epoch < 8 else 0.04


clip_norm = 1.0

num_epochs = 100


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
    b_n_links       = int(stats.get("n_links", 0))
    b_n_links_mape  = int(stats.get("n_links_mape", 0))
    b_n_routes      = int(stats.get("n_routes", 0))
    b_n_routes_mape = int(stats.get("n_routes_mape", 0))

    def as_float(x):
        if isinstance(x, torch.Tensor):
            return float(x.item())
        return float(x)

    if b_n_routes > 0:
        agg["link_loss_sum"]  += as_float(stats["link_loss"])  * b_n_routes
        agg["route_loss_sum"] += as_float(stats["route_loss"]) * b_n_routes

    if b_n_links > 0:
        agg["link_mae_sum"] += as_float(stats["link_mae"]) * b_n_links
        agg["link_sse_sum"] += (as_float(stats["link_rmse"]) ** 2) * b_n_links

    if b_n_links_mape > 0:
        m_link = as_float(stats["link_mape"])
        if not math.isnan(m_link):
            agg["link_mape_sum"] += m_link * b_n_links_mape

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


REL_NAMES_DEFAULT = ["self", "downstream", "upstream", "left_turn", "opposite", "right_turn", "high_order", "u_turn"]


def linketa_get_core_components(edges_gdf, df_all):
    try:
        from torch.amp import GradScaler, autocast
    except ImportError:
        from torch.cuda.amp import GradScaler, autocast

    def select_dynamic_12x5_columns_full(df_all: pd.DataFrame) -> List[str]:
        order = list(range(0, 12))
        cols = [f"{p}_{s}" for s in order for p in CFG["dyn_prefixes"]]
        for c in cols:
            if c not in df_all.columns:
                df_all[c] = 0.0
        return cols

    def build_instance_features(df_all):
        static_feat_cols = list(CFG["static_cols"])
        time_feat_cols = list(CFG["time_cols"])

        for col in static_feat_cols:
            if col not in df_all.columns:
                df_all[col] = 0.0

        per_pass = ["link_len_m"] if "link_len_m" in static_feat_cols else []
        edge_cols = [c for c in static_feat_cols if c not in per_pass]
        edge_static = (df_all.groupby('edge_idx')[edge_cols].first().fillna(0.0)
                       if edge_cols else pd.DataFrame(index=df_all['edge_idx'].unique()))

        dyn_feat_cols = select_dynamic_12x5_columns_full(df_all)

        BASE_COLS = cols_if_exist(df_all, ["TRIP_ID", "edge_idx"] + time_feat_cols + dyn_feat_cols + per_pass)
        sort_cols = ['TRIP_ID', 'enter_ts'] if 'enter_ts' in df_all.columns else ['TRIP_ID', 'departure_time']

        inst_raw = (
            df_all.sort_values(sort_cols)
                  .drop_duplicates(subset=['TRIP_ID', 'edge_idx'], keep='first')
                  [BASE_COLS]
                  .set_index(['TRIP_ID', 'edge_idx'])
                  .fillna(0.0)
        )
        inst = inst_raw.join(edge_static, on='edge_idx', how='left').fillna(0.0)
        if per_pass:
            inst = inst[[c for c in inst.columns if c not in static_feat_cols] + static_feat_cols]
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
        order = list(range(0, 12))
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
        last1h = sorted(idxs[:slots_per_hour])
        if len(last1h) == 0:
            raise ValueError(
                "No dynamic slot columns found. Expected columns like "
                f"'median_speed_<k>' or {CFG['dyn_prefixes']} with numeric suffix."
            )
        flat_cols_12x5 = sum([[f"{p}_{s}" for p in SLOT_PREFIXES] for s in last1h], [])

        if length_col in edges_gdf.columns:
            edge_len = dict(zip(edges_gdf["edge_idx"], edges_gdf[length_col]))
        else:
            tmp = df_all.dropna(subset=[way_id_col_df, length_col]).copy()
            tmp["edge_idx"] = tmp[way_id_col_df].map(id2edge)
            edge_len = tmp.dropna(subset=["edge_idx"]).groupby("edge_idx")[length_col].first().to_dict()

        corr_accum: Dict[Tuple[int, int], float] = {}
        if ROUTE_ORDER == "legacy":
            grouped = df_all.sort_values(CFG["time_col_df"]).groupby("TRIP_ID", sort=False)
        else:
            _ordered = df_all.assign(_route_order=route_order_key(df_all))
            grouped = _ordered.sort_values(["TRIP_ID", "_route_order"], kind="mergesort").groupby("TRIP_ID", sort=False)

        for _, grp in grouped:
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
        if (len(_dwa_hist_speed) < 2 or
            len(_dwa_hist_link)  < 2 or
            len(_dwa_hist_route) < 2):
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
            "day_of_week",
            "tod_sin", "tod_cos", "dow_sin", "dow_cos",
        ],
        hidden_dim=32,
        heads=8,
        num_layers=1,
        high_order_topk=5,
        speed_metric_index=0,
        K_first=5,
        K_conflict=5,
        K_high=5,
        max_nodes_local=360,
    )
    globals()["CFG"] = CFG
    set_seed(TRAIN_SEED)
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
    globals()["DYN_IDX"] = DYN_IDX
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
    _ho_len_col = CFG["length_col_edges"] if CFG["length_col_edges"] in edges_vff.columns else "length"
    _ho_slot_cols = [c for c in df_train.columns if any(str(c).startswith(f"{p}_") for p in CFG["dyn_prefixes"])]
    _ho_fp = _sha(
        "high_order", CACHE_SCHEMA,
        *(["route_order=legacy"] if ROUTE_ORDER != "enter_ts" else []),
        hash_frame(df_train, ["TRIP_ID", CFG["way_id_col_df"], "enter_ts"] + _ho_slot_cols),
        hash_frame(edges_vff, [edges_id_col, "edge_idx", _ho_len_col]),
        hash_tensors([first_order_all]),
        CFG["high_order_topk"], CFG["slots_per_hour"], CFG["speed_metric_index"], CFG["speed_is_kmh"],
        tuple(CFG["dyn_prefixes"]),
        "slots=latest(B2)",
    )
    _ho = load_cached("high_order", _ho_fp) if CACHE_DIR else None
    if _ho is not None:
        edge_index_high = _ho["edge_index_high"]
    else:
        edge_index_high = build_high_order_edge_index_dueta(
            df_all=df_train,
            edges_gdf=edges_vff,
            top_k=CFG["high_order_topk"],
            way_id_col_df=CFG["way_id_col_df"],
            way_id_col_edges=edges_id_col,
            length_col=_ho_len_col,
            slots_per_hour=CFG["slots_per_hour"],
            speed_metric_index=CFG["speed_metric_index"],
            speed_is_kmh=CFG["speed_is_kmh"],
            exclude_first_order=first_order_all,
        )
        if CACHE_DIR:
            save_cached("high_order", _ho_fp, dict(edge_index_high=edge_index_high))
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
    H_SLICES, BIN_SECONDS = 12, 10 * 60
    globals()["H_SLICES"] = H_SLICES
    globals()["BIN_SECONDS"] = BIN_SECONDS
    L_MIN, T_MIN = 10.0, 1.5
    _group_cols = ["TRIP_ID", "edge_idx", "link_len_m", "link_travel_time_s",
                   "trip_travel_time_s", CFG["time_col_df"], "enter_ts"]
    _common_fp = _sha(
        "groups", CACHE_SCHEMA,
        *(["route_order=legacy"] if ROUTE_ORDER != "enter_ts" else []),
        hash_frame(inst.reset_index(), ["TRIP_ID", "edge_idx"] + list(inst.columns)),
        hash_frame(df_all_local, ["edge_idx", CFG["time_col_df"]] + list(dyn_12x5_cols)),
        hash_frame(edges_vff, ["edge_idx", "v_ff"]),
        hash_tensors(rel_edges),
        tuple(STATIC_FEAT_COLS), tuple(TIME_FEAT_COLS), tuple(DYN_FEAT_COLS),
        CFG["K_first"], CFG["K_conflict"], CFG["K_high"], CFG.get("max_nodes_local", 360),
        tuple(CFG["dyn_prefixes"]), CFG["speed_is_kmh"], H_SLICES, BIN_SECONDS,
        "dyn=own row + SpeedHistory; volume target from the node's own inputs", SPEED_HISTORY.fingerprint,
    ) if CACHE_DIR else None
    _split_seed = {"train": 42001, "val": 42002, "test": 42003}

    def _make_ds(df_split, split):
        fp = _sha(_common_fp, split, _split_seed[split], hash_frame(df_split, _group_cols)) if _common_fp else None
        return TripDatasetLinkETA(
            df_split, edges_vff, inst,
            rel_edges,
            static_feat_cols=STATIC_FEAT_COLS,
            time_feat_cols=TIME_FEAT_COLS,
            dyn_feat_cols=DYN_FEAT_COLS,
            K_first=CFG["K_first"],
            K_conflict=CFG["K_conflict"],
            K_high=CFG["K_high"],
            build_seed=_split_seed[split],
            cache_fp=fp,
            split_name=split,
        )

    train_ds = _make_ds(df_train, "train")
    val_ds = _make_ds(df_val, "val")
    test_ds = _make_ds(df_test, "test")
    print("Num trips in train_ds:", len(train_ds))
    print("Num trips in val_ds:", len(val_ds))
    print("Num trips in test_ds:", len(test_ds))
    _dwa_hist_speed = deque(maxlen=2)
    _dwa_hist_link  = deque(maxlen=2)
    _dwa_hist_route = deque(maxlen=2)
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)
    print("DWA temperature:", DWA_T)
    print("AMP enabled:", USE_AMP)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vff_vec = vff_vec.to(device)
    IN_DIM = len(INST_FEAT_COLS)
    print("IN_DIM:", IN_DIM)
    IN_DIM = inst.shape[1]
    model = LinkETA(
        in_dim=IN_DIM,
        hidden_dim=CFG["hidden_dim"],
        heads=CFG["heads"],
        relations=len(rel_edges),
        num_layers=CFG["num_layers"],
        dropout=0.3,
        input_dropout=0.2,
    ).to(device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            "speed_loss": [],
            "route_loss": [], "link_loss": [],
            "route_mape": [], "link_mape": [],
            "speed_mae_actual": [],
            "vol_mae": [],
        },
        "val": {
            "speed_loss": [],
            "route_loss": [], "link_loss": [],
            "route_mape": [], "link_mape": [],
            "speed_mae_actual": [],
            "vol_mae": [],
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


    def one_epoch(loader, epoch=0, train=True,
                  w_speed=1/3, w_link=1/3, w_route=1/3):
        model.train() if train else model.eval()

        sum_speed_loss = 0.0
        sum_link_loss = sum_route_loss = 0.0
        sum_mae_link = sum_mape_link = 0.0
        sum_sse_link = 0.0
        N_links = N_links_mape = 0

        sum_mae_route = sum_mape_route = 0.0
        sum_sse_route = 0.0
        N_routes_loss = N_routes_metrics = N_routes_mape = 0

        sum_vol_mae = 0.0
        N_vol = 0
        sum_speed_mae_actual = 0.0
        N_speed_actual = 0

        for batch in loader:
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

            lc_true_node = torch.nan_to_num(batch["logcnt1h_node"], nan=0.0)
            gather_for_seq = lambda node_vals: torch.gather(
                node_vals, 1, route_seq_idx
            )
            lc_true = gather_for_seq(lc_true_node)

            vff_seq_b = vff_vec[node_edge_idx]
            vff_seq_b = torch.gather(vff_seq_b, 1, route_seq_idx)
            vff_seq_clean = vff_seq_b.clone()
            if train:
                rng = vff_noise_range(epoch)
                if rng > 0:
                    noise = 1.0 + rng * (2.0 * torch.rand_like(vff_seq_b) - 1.0)
                    vff_seq_b = vff_seq_b * noise

            R_MAX_TARGET = R_MAX
            sp_actual = (seq_link_len / seq_true_tt.clamp_min(1e-3)) / vff_seq_b.clamp_min(1.0)
            sp_actual = sp_actual.clamp(0.0, R_MAX_TARGET)

            sp_actual_clean = (
                seq_link_len / seq_true_tt.clamp_min(1e-3)
            ) / vff_seq_clean.clamp_min(1.0)
            was_pinned = sp_actual_clean >= (LABEL_CAP - 1e-3)

            sp_w_actual = (
                seq_mask
                & (seq_true_tt  >= 1.0)
                & (seq_link_len >= 10.0)
                & (~was_pinned)
            ).float()

            if HEAD_MODE == "pace":
                head_target = (seq_true_tt * vff_seq_b.clamp_min(1.0) / seq_link_len).clamp_min(1.0 / R_MAX_TARGET)
                head_beta = PACE_LOSS_BETA
            else:
                head_target = sp_actual
                head_beta = 0.1

            x_seq_norm = (x_seq_nodes - SCALER_MEAN.view(1, 1, -1)) / SCALER_STD.view(1, 1, -1)


            Bsz = seq_true_tt.size(0)

            with torch.amp.autocast('cuda', enabled=USE_AMP):
                link_eta_seq, trip_eta, sp_pred, lc_pred, g, delta, eta_phys_seq = model(
                    x_seq_norm,
                    hop_idx,
                    route_flag_nodes,
                    masks_per_rel,
                    route_seq_idx,
                    seq_link_len,
                    vff_seq_b,
                    seq_mask,
                    r_max=R_MAX_TARGET,
                    return_parts=True,
                )

                link_eta_seq = torch.nan_to_num(link_eta_seq, nan=0.0)
                trip_eta = torch.nan_to_num(trip_eta, nan=0.0)
                sp_pred = torch.nan_to_num(sp_pred, nan=0.0)
                lc_pred = torch.nan_to_num(lc_pred, nan=0.0)

                pred_seq = link_eta_in_seconds(link_eta_seq, seq_link_len)
                seq_true = seq_true_tt.clamp_min(0.0)

                loss_link  = link_loss_huber_log_lenweighted(
                    link_eta_in_seconds(torch.nan_to_num(eta_phys_seq, nan=0.0), seq_link_len), seq_true, seq_mask,
                    link_len=seq_link_len,
                    delta=0.3,
                )
                loss_route = route_loss_huber_log(pred_seq, trip_tt, seq_mask, delta=0.3)

                denom = sp_w_actual.sum().clamp_min(1.0)
                per = F.smooth_l1_loss(sp_pred, head_target, reduction='none', beta=head_beta)
                loss_speed = (per * sp_w_actual).sum() / denom


                if seq_mask.any():
                    loss_vol = F.smooth_l1_loss(lc_pred[seq_mask], lc_true[seq_mask])
                else:
                    loss_vol = torch.zeros((), device=device)


                final_loss = (w_speed * loss_speed + w_link * loss_link + w_route * loss_route
                              + 0.10 * loss_vol)

            if train:
                optimizer.zero_grad(set_to_none=True)
                if USE_AMP:
                    scaler.scale(final_loss).backward()
                    if clip_norm is not None:
                        scaler.unscale_(optimizer)
                        _gn = nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                        TRAIN_TRACK["grad_norms"].append(float(_gn))
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    final_loss.backward()
                    if clip_norm is not None:
                        _gn = nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                        TRAIN_TRACK["grad_norms"].append(float(_gn))
                    optimizer.step()

            link_stats = link_metrics_micro(pred_seq, seq_true, seq_mask,
                                            link_len=seq_link_len,
                                            L_min=L_MIN_METRIC,
                                            T_min=T_MIN_METRIC)
            route_stats = route_metrics_pooled(pred_seq, trip_tt, seq_mask)

            sum_speed_loss += float(loss_speed.item()) * Bsz
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


            sp_actual_mask = sp_w_actual.bool()
            if sp_actual_mask.any():
                sp_abs_actual = (sp_pred[sp_actual_mask] - head_target[sp_actual_mask]).abs().mean().item()
                n_act = int(sp_actual_mask.sum().item())
                sum_speed_mae_actual += sp_abs_actual * n_act
                N_speed_actual += n_act

            if seq_mask.any():
                vol_abs = (lc_pred[seq_mask] - lc_true[seq_mask]).abs().mean().item()
                n_valid = int(seq_mask.sum().item())
                sum_vol_mae += vol_abs * n_valid
                N_vol += n_valid

        return {
            'speed_loss': sum_speed_loss / max(N_routes_loss, 1),
            'link_loss':  sum_link_loss  / max(N_routes_loss, 1),
            'route_loss': sum_route_loss / max(N_routes_loss, 1),
            'link_mae':   sum_mae_link   / max(N_links, 1),
            'link_rmse':  math.sqrt(sum_sse_link / max(N_links, 1)) if N_links > 0 else float("nan"),
            'link_mape':  (sum_mape_link / max(N_links_mape, 1)) if N_links_mape > 0 else float('nan'),
            'route_mae':  sum_mae_route / max(N_routes_metrics, 1),
            'route_rmse': math.sqrt(sum_sse_route / max(N_routes_metrics, 1)) if N_routes_metrics > 0 else float("nan"),
            'route_mape': (sum_mape_route / max(N_routes_mape, 1)) if N_routes_mape > 0 else float('nan'),
            'speed_mae_actual':  (sum_speed_mae_actual / max(N_speed_actual, 1)) if N_speed_actual else float('nan'),
            'vol_mae': (sum_vol_mae / max(N_vol, 1)) if N_vol else float('nan'),
        }


    def eval_linketa(core,
                     ckpt_path: str = None,
                     label: str = "LinkETA",
                     split: str = "test"):
        device = core["device"]
        model  = core["model"].to(device)

        if "loaders" in core and split in core["loaders"]:
            loader = core["loaders"][split]
        else:
            key = f"{split}_loader"
            if key in core:
                loader = core[key]
            else:
                raise KeyError(f"No DataLoader found for split='{split}' in core.")

        vff_vec_local = core.get("graph", {}).get("vff_vec", vff_vec).to(device)
        scaler_mean   = core.get("scalers", {}).get("mean", SCALER_MEAN).to(device)
        scaler_std    = core.get("scalers", {}).get("std", SCALER_STD).to(device)

        if ckpt_path is not None:
            pkg = torch.load(ckpt_path, map_location=device)
            state = pkg["model"] if isinstance(pkg, dict) and "model" in pkg else pkg
            model.load_state_dict(state, strict=True)

        model.eval()

        agg = _init_metric_accumulator()

        sum_vol_mae = 0.0
        N_vol = 0
        sum_speed_mae_actual = 0.0
        N_speed_actual = 0

        trip_rows = []
        link_rows = []

        for batch in loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            node_edge_idx    = batch["node_edge_idx"]
            x_seq_nodes      = batch["x_seq"]
            route_flag_nodes = batch["route_flag"].bool()
            hop_idx          = batch["hop_idx"].long()
            masks_per_rel    = [m.to(device) for m in batch["masks_per_rel"]]

            route_seq_idx = batch["route_seq_idx"].long()
            seq_true_tt   = torch.nan_to_num(batch["seq_true_tt"], nan=0.0)
            seq_link_len  = torch.nan_to_num(batch["seq_link_len"], nan=0.0)
            seq_link_len  = seq_link_len.clamp_min(1e-3)
            seq_mask      = batch["seq_mask"].bool()
            trip_tt       = torch.nan_to_num(batch["trip_tt"], nan=0.0)

            lc_true_node = torch.nan_to_num(batch["logcnt1h_node"],  nan=0.0)

            gather_for_seq = lambda node_vals: torch.gather(
                node_vals, 1, route_seq_idx
            )

            lc_true = gather_for_seq(lc_true_node)

            vff_node = vff_vec_local[node_edge_idx]
            vff_seq  = gather_for_seq(vff_node)

            x_seq_norm = (x_seq_nodes - scaler_mean.view(1, 1, -1)) / scaler_std.view(1, 1, -1)
            x_seq_norm = torch.nan_to_num(x_seq_norm)

            with torch.amp.autocast('cuda', enabled=USE_AMP):
                (
                    link_eta_seq,
                    trip_eta,
                    sp_pred,
                    lc_pred,
                    g,
                    delta,
                    eta_phys_seq,
                ) = model(
                    x_seq_norm,
                    hop_idx,
                    route_flag_nodes,
                    masks_per_rel,
                    route_seq_idx,
                    seq_link_len,
                    vff_seq,
                    seq_mask,
                    r_max=R_MAX,
                    return_parts=True,
                )

                link_eta_seq = torch.nan_to_num(link_eta_seq, nan=0.0)
                trip_eta     = torch.nan_to_num(trip_eta,     nan=0.0)
                sp_pred      = torch.nan_to_num(sp_pred,      nan=0.0)
                lc_pred      = torch.nan_to_num(lc_pred,      nan=0.0)

                pred_seq = link_eta_in_seconds(link_eta_seq, seq_link_len)
                seq_true = seq_true_tt.clamp_min(0.0)

                stats = compute_global_losses_and_metrics(
                    pred_seq=pred_seq,
                    trip_tt=trip_tt,
                    seq_true=seq_true,
                    seq_mask=seq_mask,
                    seq_link_len=seq_link_len,
                    delta_huber=0.3,
                )

            _accumulate_batch_metrics(agg, stats)


            R_MAX_TARGET = R_MAX
            sp_actual_raw = (seq_link_len / seq_true_tt.clamp_min(1e-3)) / vff_seq.clamp_min(1.0)
            sp_actual = sp_actual_raw.clamp(0.0, R_MAX_TARGET)
            was_pinned = sp_actual_raw.clamp(0.0, LABEL_CAP) >= (LABEL_CAP - 1e-3)
            sp_w_actual = (
                seq_mask
                & (seq_true_tt  >= 1.0)
                & (seq_link_len >= 10.0)
                & (~was_pinned)
            )
            if HEAD_MODE == "pace":
                head_target = (seq_true_tt * vff_seq.clamp_min(1.0) / seq_link_len).clamp_min(1.0 / R_MAX_TARGET)
            else:
                head_target = sp_actual
            if sp_w_actual.any():
                sp_abs_act = (sp_pred[sp_w_actual] - head_target[sp_w_actual]).abs().mean().item()
                n_sp_act   = int(sp_w_actual.sum().item())
                sum_speed_mae_actual += sp_abs_act * n_sp_act
                N_speed_actual       += n_sp_act

            if seq_mask.any():
                vol_abs = (lc_pred[seq_mask] - lc_true[seq_mask]).abs().mean().item()
                n_vol   = int(seq_mask.sum().item())
                sum_vol_mae += vol_abs * n_vol
                N_vol       += n_vol

            route_true_from_links = (seq_true_tt * seq_mask.float()).sum(dim=1)
            yR = torch.where(trip_tt > 0, trip_tt, route_true_from_links)
            pR = (pred_seq * seq_mask.float()).sum(dim=1)

            trip_ids     = batch["trip_ids"]
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

        core_metrics = _finalize_metric_accumulator(agg)


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
            "speed_mae_actual":  speed_mae_actual,
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
        r_max=R_MAX,
    )
    return core


def run_training(core):
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


    best_combined, best_state = float('inf'), None
    patience, since_best = 15, 0

    tr_last = va_last = None
    _t_start = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    TRAIN_TRACK["grad_norms"] = []

    for epoch in range(num_epochs):
        wS, wL, wR = _dwa_weights()

        _t0 = time.perf_counter()
        tr = one_epoch(train_loader, epoch=epoch, train=True,
                       w_speed=wS, w_link=wL, w_route=wR)
        _t1 = time.perf_counter()
        va = one_epoch(val_loader,   epoch=epoch, train=False,
                       w_speed=wS, w_link=wL, w_route=wR)
        _trk = _epoch_track(_t0, _t1, time.perf_counter(), clip_norm)
        for _k, _v in _trk.items():
            HIST["train"].setdefault(_k, []).append(_v)
        print(f"[track] epoch {epoch:02d}: train {_trk['train_time_s']:.0f}s val {_trk['val_time_s']:.0f}s | "
              f"grad norm mean {_trk['grad_norm_mean']:.3f} p99 {_trk['grad_norm_p99']:.3f} max {_trk['grad_norm_max']:.3f} "
              f"clipped {_trk['frac_steps_clipped']:.2%} non-finite {_trk['grad_norm_nonfinite']} | "
              f"peak GPU {_trk['peak_gpu_mb']:.0f} MB", flush=True)

        tr_last, va_last = tr, va

        _dwa_hist_speed.append(tr['speed_loss'])
        _dwa_hist_link .append(tr['link_loss'])
        _dwa_hist_route.append(tr['route_loss'])

        for k in ["speed_loss", "route_loss", "link_loss", "route_mape", "link_mape",
                  "speed_mae_actual", "vol_mae"]:
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
                extra={"val_metrics": va, "head_mode": HEAD_MODE, "route_order": ROUTE_ORDER,
                       "pace_loss_beta": PACE_LOSS_BETA, "r_max": R_MAX}
            )
        else:
            since_best += 1

        ckpt.save_last(
            model=model, optimizer=optimizer, scheduler=scheduler,
            epoch=epoch, step=global_step, best_metric=float(best_combined),
            extra={"val_metrics": va, "head_mode": HEAD_MODE, "route_order": ROUTE_ORDER,
                       "pace_loss_beta": PACE_LOSS_BETA, "r_max": R_MAX}
        )

        print(
            f"Epoch {epoch:02d} | "
            f"Train: Link {tr['link_loss']:.4f}/{tr['link_mae']:.3f} (MAPE={tr['link_mape']:.3f}) (RMSE={tr['link_rmse']:.3f})  "
            f"Route {tr['route_loss']:.4f}/{tr['route_mae']:.3f} (MAPE={tr['route_mape']:.3f}) (RMSE={tr['route_rmse']:.3f})  "
            f"Speed[actual]={tr['speed_mae_actual']:.4f} Vol={tr['vol_mae']:.4f}  ||  "
            f"Val: Link {va['link_loss']:.4f}/{va['link_mae']:.3f} (MAPE={va['link_mape']:.3f}) (RMSE={va['link_rmse']:.3f})  "
            f"Route {va['route_loss']:.4f}/{va['route_mae']:.3f} (MAPE={va['route_mape']:.3f}) (RMSE={va['route_rmse']:.3f})  "
            f"| DWA wS={wS:.3f} wL={wL:.3f} wR={wR:.3f}  | Combined={comb:.3f}"
        )


        if since_best >= patience:
            print("Early stopping triggered.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    wS, wL, wR = _dwa_weights()
    te = one_epoch(test_loader, train=False,
                   w_speed=wS, w_link=wL, w_route=wR)


    print("— Final Test —")
    print(f"Link:  Huber={te['link_loss']:.4f} | MAE={te['link_mae']:.4f} | RMSE={te['link_rmse']:.4f} | MAPE={te['link_mape']:.4f}")
    print(f"Route: MAPE_Loss={te['route_loss']:.4f} | MAE={te['route_mae']:.4f} | RMSE={te['route_rmse']:.4f} | MAPE={te['route_mape']:.4f}")
    print(f"Speed ratio MAE vs target: {te['speed_mae_actual']:.4f} | VolMAE={te['vol_mae']:.4f}")

    hist_path    = os.path.join(ckpt.dir, "training_history.json")
    summary_path = os.path.join(ckpt.dir, "training_summary.json")
    with open(hist_path, "w", encoding="utf-8") as _f:
        json.dump(HIST, _f, indent=2, default=str)
    with open(summary_path, "w", encoding="utf-8") as _f:
        json.dump({"test": te, "best_selector": float(best_combined),
                   "head_mode": HEAD_MODE, "route_order": ROUTE_ORDER,
                   "pace_loss_beta": PACE_LOSS_BETA,
                   "r_max": R_MAX, "label_cap": LABEL_CAP,
                   "train_last": tr_last, "val_last": va_last,
                   "epochs_run": len(HIST["train"].get("train_time_s", [])),
                   "total_train_time_s": float(time.perf_counter() - _t_start),
                   "max_peak_gpu_mb": float(max(HIST["train"].get("peak_gpu_mb", [float("nan")]))),
                   "grad_norm_max_all": float(np.nanmax(HIST["train"].get("grad_norm_max", [float("nan")]))),
                   "grad_norm_nonfinite_all": int(sum(HIST["train"].get("grad_norm_nonfinite", [0]))),
                   "clip_norm": clip_norm}, _f, indent=2, default=str)
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

