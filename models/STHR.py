import os
import sys
import json
import math
import time
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import Dataset, DataLoader
from shapely.geometry import LineString, MultiLineString

from loss_functions import (
    link_metrics_micro, route_metrics_pooled, L_MIN_METRIC, T_MIN_METRIC,
)
from checkpoint import CheckpointManager

OFFICIAL_CODE_DIR = Path(__file__).resolve().parents[1] / "baselines" / "STHR_CODE-main" / "code"

CKPT_DIRNAME = "ckpts_STHR"

NUM_EPOCHS = 100
PATIENCE = 15
BATCH_SIZE = 16
SEED = 42

edges_gdf = None
df_all = None
SPEED_HISTORY = None

README = dict(slot_size=10, pre_slot_num=12, split_scale=0.25, global_num_hidden=64, node_feature_dim=16,
              local_num_hidden=32, st_hidden_size=32, st_total_key_depth=12, st_total_value_depth=24,
              day_embedding_dim=4, time_embedding_dim=2, lr=0.001, lamb=0.5, alpha_1=0.1, alpha_2=0.1)
TIME_UNIT_S = 600.0

CFG = dict(
    way_id_col_df="way_id",
    time_col_df="departure_time",
    way_id_col_edges_candidates=["way_id", "id", "edge_id", "osmid"],
    speed_prefixes=["min_speed", "max_speed", "median_speed", "mean_speed"],
    slot_size=README["slot_size"],
    pre_slot_num=README["pre_slot_num"],
    dyn_max_gap_s=1800,
    split_scale=README["split_scale"],
    n_raw_bins=20,
    onehot_attrs=["highway", "oneway", "lanes", "bridge", "junction", "tunnel"],
)


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _import_official():
    try:
        import dgl
    except ImportError as e:
        raise RuntimeError(
            "STHR runs the official code, which needs DGL 1.0.1 + PyTorch 1.13.1. "
            "Run main.py from the 'baseline' conda env (see env/baseline_requirements.txt).") from e
    if str(OFFICIAL_CODE_DIR) not in sys.path:
        sys.path.insert(0, str(OFFICIAL_CODE_DIR))
    import Leim2
    return Leim2


def _guess_id_col(g: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in g.columns:
            return c
    return None


def _endpoints(geom):
    if isinstance(geom, LineString):
        return geom.coords[0], geom.coords[-1]
    if isinstance(geom, MultiLineString):
        return list(geom.geoms[0].coords)[0], list(geom.geoms[-1].coords)[-1]
    return (np.nan, np.nan), (np.nan, np.nan)


def _split_multi(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return [None]
    s = str(v).strip()
    if s == "" or s.lower() in ("nan", "none"):
        return [None]
    parts = [p.strip() for p in s.split(";") if p.strip()]
    return parts or [None]


def _parse_maxspeed_kmh(v) -> float:
    s = _split_multi(v)[0]
    if s is None:
        return np.nan
    num = ""
    for ch in s:
        if ch.isdigit() or ch == ".":
            num += ch
        elif num:
            break
    if not num:
        return np.nan
    x = float(num)
    if "mph" in s.lower():
        x *= 1.609344
    return x


def build_road_network(edges_local: pd.DataFrame):
    N = len(edges_local)
    ends = [_endpoints(ge) for ge in edges_local.geometry]
    s_xy = np.array([e[0][:2] for e in ends], dtype=np.float64)
    e_xy = np.array([e[1][:2] for e in ends], dtype=np.float64)

    key_s = [(round(x, 6), round(y, 6)) for x, y in s_xy]
    key_e = [(round(x, 6), round(y, 6)) for x, y in e_xy]
    node_ids = {}
    for k in key_s + key_e:
        if k not in node_ids:
            node_ids[k] = len(node_ids)
    u = np.array([node_ids[k] for k in key_s], dtype=np.int64)
    v = np.array([node_ids[k] for k in key_e], dtype=np.int64)
    n_nodes = len(node_ids)
    node_xy = np.zeros((n_nodes, 2), dtype=np.float64)
    for (x, y), i in node_ids.items():
        node_xy[i] = (x, y)

    min_x, max_x = np.nanmin(node_xy[:, 0]), np.nanmax(node_xy[:, 0])
    min_y, max_y = np.nanmin(node_xy[:, 1]), np.nanmax(node_xy[:, 1])
    nx_ = (node_xy[:, 0] - min_x) / max(max_x - min_x, 1e-12)
    ny_ = (node_xy[:, 1] - min_y) / max(max_y - min_y, 1e-12)
    in_deg = np.bincount(v, minlength=n_nodes)
    out_deg = np.bincount(u, minlength=n_nodes)
    node_pro = np.stack([nx_, ny_, in_deg, out_deg], axis=1).astype(np.float32)

    feats = [np.stack([nx_[u], ny_[u], nx_[v], ny_[v]], axis=1)]
    for attr in CFG["onehot_attrs"]:
        col = edges_local[attr] if attr in edges_local.columns else pd.Series([None] * N)
        vals = [_split_multi(x) for x in col]
        cats = [None]
        for vs in vals:
            for x in vs:
                if x not in cats:
                    cats.append(x)
        idx = {c: i for i, c in enumerate(cats)}
        oh = np.zeros((N, len(cats)), dtype=np.float64)
        for r, vs in enumerate(vals):
            for x in vs:
                oh[r, idx[x]] = 1.0
        feats.append(oh)

    if "length" in edges_local.columns:
        length_m = pd.to_numeric(edges_local["length"], errors="coerce").to_numpy(dtype=np.float64)
    else:
        length_m = np.full(N, np.nan)
    if np.isnan(length_m).any():
        metric = edges_local.to_crs(edges_local.estimate_utm_crs()).length.to_numpy()
        length_m = np.where(np.isnan(length_m), metric, length_m)

    ms = np.array([_parse_maxspeed_kmh(x) for x in edges_local.get("maxspeed", pd.Series([None] * N))])
    hw = [(_split_multi(x)[0] or "none") for x in edges_local.get("highway", pd.Series([None] * N))]
    hw_med = pd.Series(ms).groupby(pd.Series(hw)).median().to_dict()
    glob_med = float(np.nanmedian(ms)) if np.isfinite(ms).any() else 50.0
    ms = np.array([m if np.isfinite(m) else (hw_med.get(h, np.nan) if np.isfinite(hw_med.get(h, np.nan)) else glob_med)
                   for m, h in zip(ms, hw)])
    feats.append(np.stack([length_m / 1000.0, ms / 100.0], axis=1))
    link_pro = np.nan_to_num(np.concatenate(feats, axis=1)).astype(np.float32)

    out_by_u = defaultdict(list)
    for i in range(N):
        out_by_u[int(u[i])].append(i)
    turn_src, turn_dst = [], []
    for i in range(N):
        for j in out_by_u.get(int(v[i]), []):
            if j != i:
                turn_src.append(i); turn_dst.append(j)

    edge_dict = {i: link_pro[i].tolist() for i in range(N)}
    node_dict = {i: row.tolist() for i, row in enumerate(node_pro)}
    edge2node = {i: (int(u[i]), int(v[i])) for i in range(N)}
    node2edge = {(int(u[i]), int(v[i])): i for i in range(N)}
    input_dict, output_dict = {}, {}
    for i in range(N):
        input_dict.setdefault(int(v[i]), []).append(int(u[i]))
        output_dict.setdefault(int(u[i]), []).append(int(v[i]))
    turning = {(a, b): t for t, (a, b) in enumerate(zip(turn_src, turn_dst))}

    return dict(
        graph_args=dict(node2edge_dict=node2edge, edge2node_dict=edge2node, input_dict=input_dict,
                        output_dict=output_dict, node_dict=node_dict, edge_dict=edge_dict,
                        turning_dict=turning),
        turn_src=np.asarray(turn_src, dtype=np.int64),
        turn_dst=np.asarray(turn_dst, dtype=np.int64),
        n_links=N, link_feature_dim=link_pro.shape[1],
    )


class SpeedStore:
    def __init__(self, edge2way: Dict[int, int]):
        assert SPEED_HISTORY is not None and SPEED_HISTORY.bin_seconds == CFG["slot_size"] * 60, "SPEED_HISTORY not set"
        self.edge2way = edge2way
        self.fam_ix = [["median_speed", "mean_speed", "min_speed", "max_speed", "count_vehicle"].index(p)
                       for p in CFG["speed_prefixes"]]

    def get(self, edge_idx: int, t_ns: int, trip_id: int) -> np.ndarray:
        pic = SPEED_HISTORY.slots(self.edge2way[int(edge_idx)], t_ns / 1e9, int(trip_id))[:CFG["pre_slot_num"]][::-1]
        out = np.ascontiguousarray(pic[:, self.fam_ix], dtype=np.float32)
        out[pic[:, 4] == 0] = np.nan
        return out


def temporal_split(df: pd.DataFrame, time_col: str):
    d = df.copy()
    first_dep = d.groupby("TRIP_ID")[time_col].min().rename("first_dep")
    d = d.join(first_dep, on="TRIP_ID")
    min_date, max_date = d["first_dep"].min(), d["first_dep"].max()
    total_days = max((max_date - min_date).days, 1)
    train_cut = min_date + pd.Timedelta(days=int(0.6 * total_days))
    val_cut = min_date + pd.Timedelta(days=int(0.8 * total_days))
    gap = pd.Timedelta(days=min(7, max(1, total_days // 20)))
    if val_cut - train_cut <= gap:
        gap = pd.Timedelta(hours=12) if val_cut - train_cut > pd.Timedelta(hours=12) else pd.Timedelta(hours=6)
    df_train = d[d["first_dep"] <= train_cut].drop(columns=["first_dep"])
    df_val = d[(d["first_dep"] > train_cut + gap) & (d["first_dep"] <= val_cut)].drop(columns=["first_dep"])
    df_test = d[d["first_dep"] > val_cut].drop(columns=["first_dep"])
    return df_train, df_val, df_test


def compute_speed_bins(df_train: pd.DataFrame) -> List[float]:
    L = pd.to_numeric(df_train["link_len_m"], errors="coerce")
    T = pd.to_numeric(df_train["link_travel_time_s"], errors="coerce")
    ok = (T >= 1.0) & (L >= 10.0)
    spd = (L[ok] / T[ok] * 3.6).clip(0.36, 180.0).to_numpy()
    raw = np.quantile(spd, np.linspace(1.0 / CFG["n_raw_bins"], 1.0, CFG["n_raw_bins"]))
    for k in range(1, len(raw)):
        raw[k] = max(raw[k], raw[k - 1] + 1e-2)
    raw_edge_split = [round(float(x), 1) for x in raw]

    split_scale = CFG["split_scale"]
    edge_split = []
    if split_scale > 1:
        pre = 0
        for p in raw_edge_split:
            delta = (p - pre) / split_scale
            for j in range(int(split_scale) - 1):
                edge_split.append(round(pre + delta * (j + 1), 2))
            edge_split.append(p)
            pre = p
    else:
        delta_step = int(1 / split_scale)
        for i in range(0, len(raw_edge_split), delta_step):
            edge_split.append(raw_edge_split[i])
        if edge_split[-1] != raw_edge_split[-1]:
            edge_split.append(raw_edge_split[-1])
    print(f"[STHR] raw speed quantiles (km/h): {raw_edge_split}")
    print(f"[STHR] speed-bin upper edges (km/h): {edge_split}")
    return edge_split


class TripDatasetSTHR(Dataset):
    def __init__(self, df: pd.DataFrame, net: dict, store: SpeedStore,
                 spd_mu: np.ndarray, spd_sd: np.ndarray, time_col: str):
        self.store = store
        self.spd_mu = spd_mu.astype(np.float32)
        self.spd_sd = spd_sd.astype(np.float32)
        S, slot = CFG["pre_slot_num"], CFG["slot_size"]

        preds, succs = defaultdict(list), defaultdict(list)
        for tid, (a, b) in enumerate(zip(net["turn_src"], net["turn_dst"])):
            succs[int(a)].append((int(b), tid))
            preds[int(b)].append(int(a))

        d = df.copy()
        if "enter_ts" in d.columns:
            d["_ord"] = pd.to_datetime(d["enter_ts"], errors="coerce")
            d = d.sort_values(["TRIP_ID", "_ord"], kind="mergesort")
        else:
            d = d.sort_values(["TRIP_ID"], kind="mergesort")

        self.samples = []
        for trip_id, g in d.groupby("TRIP_ID", sort=False):
            route = g["edge_idx"].to_numpy(dtype=np.int64)
            if route.size == 0:
                continue
            link_tt = pd.to_numeric(g["link_travel_time_s"], errors="coerce").fillna(0.0).to_numpy(np.float32)
            link_len = g["link_len_m"].to_numpy(np.float32)
            t_val = pd.to_numeric(g["trip_travel_time_s"], errors="coerce").iloc[0] \
                if "trip_travel_time_s" in g.columns else np.nan
            if not (np.isfinite(t_val) and t_val > 0):
                t_val = float(np.nansum(link_tt))
            dep = pd.Timestamp(g[time_col].iloc[0])

            nodes, pos = [], {}
            for e in route.tolist():
                if e not in pos:
                    pos[e] = len(nodes); nodes.append(e)
            for e in route.tolist():
                for nb in preds.get(e, []) + [b for b, _ in succs.get(e, [])]:
                    if nb not in pos:
                        pos[nb] = len(nodes); nodes.append(nb)
            l_src, l_dst, l_tid = [], [], []
            for a in nodes:
                for b, tid in succs.get(a, []):
                    if b in pos:
                        l_src.append(pos[a]); l_dst.append(pos[b]); l_tid.append(tid)

            times = [dep - pd.Timedelta(minutes=slot * k) for k in range(S, 0, -1)] + [dep]
            self.samples.append(dict(
                trip_id=trip_id, route=route, link_tt=link_tt, link_len=link_len,
                trip_tt=float(t_val), dep_ns=int(dep.value),
                nodes=np.asarray(nodes, dtype=np.int64),
                l_src=np.asarray(l_src, dtype=np.int64),
                l_dst=np.asarray(l_dst, dtype=np.int64),
                l_tid=np.asarray(l_tid, dtype=np.int64),
                day_idx=np.asarray([t.dayofweek for t in times], dtype=np.int64),
                time_idx=np.asarray([(t.hour * 60 + t.minute) // slot for t in times], dtype=np.int64),
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        blocks = np.stack([self.store.get(e, s["dep_ns"], s["trip_id"]) for e in s["nodes"]], axis=0)
        s = dict(s)
        s["node_data"] = np.nan_to_num((blocks - self.spd_mu) / self.spd_sd, nan=0.0).astype(np.float32)
        return s


def make_collate(node_size: int):
    def collate(batch):
        B = len(batch)
        L = max(len(b["route"]) for b in batch)
        node_dec = torch.full((B, L), node_size, dtype=torch.long)
        src_mask = torch.zeros(B, L, dtype=torch.bool)
        y = torch.zeros(B, L)
        seq_len = torch.zeros(B, L)
        trip_tt = torch.zeros(B)
        day = torch.zeros(B, CFG["pre_slot_num"] + 1, dtype=torch.long)
        tim = torch.zeros(B, CFG["pre_slot_num"] + 1, dtype=torch.long)
        srcs, dsts, tids, local_edges, xs, node_lens = [], [], [], [], [], []
        cum = 0
        for i, b in enumerate(batch):
            n = len(b["route"])
            node_dec[i, :n] = torch.from_numpy(b["route"])
            src_mask[i, :n] = True
            y[i, :n] = torch.from_numpy(b["link_tt"])
            seq_len[i, :n] = torch.from_numpy(b["link_len"])
            trip_tt[i] = b["trip_tt"]
            day[i] = torch.from_numpy(b["day_idx"])
            tim[i] = torch.from_numpy(b["time_idx"])
            srcs.append(torch.from_numpy(b["l_src"]) + cum)
            dsts.append(torch.from_numpy(b["l_dst"]) + cum)
            tids.append(torch.from_numpy(b["l_tid"]))
            local_edges.append(torch.from_numpy(b["nodes"]))
            xs.append(torch.from_numpy(b["node_data"]))
            node_lens.append(len(b["nodes"]))
            cum += len(b["nodes"])
        max_nodes = max(node_lens)
        node_idxs = torch.zeros(B, max_nodes, dtype=torch.long)
        node_masks = torch.zeros(B, max_nodes, dtype=torch.long)
        off = 0
        for i, n in enumerate(node_lens):
            node_idxs[i, :n] = torch.arange(n) + off
            node_masks[i, :n] = 1
            off += n
        return dict(
            enc_turning=torch.cat(tids), enc_edges=torch.cat(local_edges),
            enc_src=torch.cat(srcs), enc_dst=torch.cat(dsts),
            enc_x=torch.cat(xs, dim=0), node_idxs=node_idxs, node_masks=node_masks,
            node_lens=torch.tensor(node_lens, dtype=torch.long),
            day=day, time=tim, node_dec=node_dec, src_mask=src_mask,
            seq_true_tt=y, seq_link_len=seq_len, trip_tt=trip_tt,
            trip_ids=[b["trip_id"] for b in batch],
        )
    return collate


def _to_device(x, device):
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, (tuple, list)):
        return type(x)(_to_device(v, device) for v in x)
    return x


def official_batch(b, device):
    x_enc = (b["enc_turning"], b["enc_edges"], b["node_lens"], b["enc_src"], b["enc_dst"],
             b["enc_x"], b["node_idxs"], b["node_masks"])
    x = ((b["day"], b["time"]), (x_enc, b["node_dec"]), b["src_mask"].to(torch.uint8))
    y = (b["seq_true_tt"] / TIME_UNIT_S).masked_fill(~b["src_mask"], -1.0)
    return _to_device(x, device), y.to(device)


def link_times(out, loss_fn):
    edge_delta, edge_bias, lens_, src_node_mask = out
    batch_size, edge_len, _ = edge_bias.shape
    edge_speed = torch.sum((loss_fn.speed_bins.unsqueeze(0).unsqueeze(1).repeat(batch_size, edge_len, 1)
                            - edge_delta * loss_fn.speed_delta.unsqueeze(0).unsqueeze(1).repeat(
                                batch_size, edge_len, 1)) * edge_bias, dim=-1)
    return (lens_ / edge_speed * 6.0).masked_fill(src_node_mask == 0, 0)


def sthr_get_core_components():
    O = _import_official()
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    edges_local = edges_gdf.reset_index(drop=True).copy()
    edges_local["edge_idx"] = np.arange(len(edges_local), dtype=int)
    id_col = _guess_id_col(edges_local, CFG["way_id_col_edges_candidates"])
    assert id_col is not None, "No id column found in edges_gdf."
    edges_local[id_col] = edges_local[id_col].astype(str)

    d = df_all.copy()
    d[CFG["way_id_col_df"]] = d[CFG["way_id_col_df"]].astype(str)
    d["edge_idx"] = d[CFG["way_id_col_df"]].map(dict(zip(edges_local[id_col], edges_local["edge_idx"])))
    d = d.dropna(subset=["edge_idx"])
    d["edge_idx"] = d["edge_idx"].astype(int)
    tcol = CFG["time_col_df"]
    d[tcol] = pd.to_datetime(d[tcol], errors="coerce", utc=True).dt.tz_localize(None)
    d = d.dropna(subset=[tcol])
    if "link_len_m" not in d.columns:
        d["link_len_m"] = np.nan
    d["link_len_m"] = pd.to_numeric(d["link_len_m"], errors="coerce") \
        .fillna(pd.to_numeric(d.get("length", np.nan), errors="coerce")).fillna(5.0)
    d.loc[d["link_len_m"] <= 0, "link_len_m"] = 5.0

    net = build_road_network(edges_local)
    road = O.RoadNetwork(net["graph_args"])
    g, inv_g = road.buildDglGraph()
    ttypes = np.array([road.edge_graph.edges[k][0] for k in range(len(road.edge_graph.edges))])
    print(f"[STHR] LCG: {net['n_links']} links, {len(ttypes)} turns "
          f"(back={int((ttypes == 0).sum())}, forth={int((ttypes == 1).sum())}, "
          f"right={int((ttypes == 2).sum())}, left={int((ttypes == 3).sum())}); "
          f"link feature dim={net['link_feature_dim']}")

    df_train, df_val, df_test = temporal_split(d, tcol)
    print(f"[STHR] split rows: train={len(df_train)} val={len(df_val)} test={len(df_test)}")
    speed_bins = compute_speed_bins(df_train)

    store = SpeedStore(dict(zip(edges_local["edge_idx"].astype(int), edges_local[id_col].astype("int64"))))
    tr = np.stack([store.get(e, int(pd.Timestamp(t).value), tid)
                   for e, t, tid in zip(df_train["edge_idx"], df_train[tcol], df_train["TRIP_ID"])]).astype(np.float64)
    spd_mu = np.nanmean(tr, axis=(0, 1))
    spd_sd = np.nanstd(tr, axis=(0, 1))
    spd_sd = np.where(spd_sd > 1e-6, spd_sd, 1.0)
    print(f"[STHR] speed feature mean={np.round(spd_mu, 3)} std={np.round(spd_sd, 3)}")

    ds = {k: TripDatasetSTHR(v, net, store, spd_mu, spd_sd, tcol)
          for k, v in (("train", df_train), ("val", df_val), ("test", df_test))}
    for k, v in ds.items():
        print(f"[STHR] {k}: {len(v)} trips")
    max_route = max(len(s["route"]) for v in ds.values() for s in v.samples)
    max_length = max_route + 16
    print(f"[STHR] longest route = {max_route} links -> decoder max_length = {max_length}")

    collate = make_collate(net["n_links"])
    loaders = dict(
        train=DataLoader(ds["train"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate, num_workers=0),
        val=DataLoader(ds["val"], batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate, num_workers=0),
        test=DataLoader(ds["test"], batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate, num_workers=0),
    )

    conf = yaml.safe_load(open(OFFICIAL_CODE_DIR / "config-bash.yaml", encoding="utf-8"))
    mp = conf["model_para"]
    r = README
    mp["device"] = str(device)
    mp["graph"] = net["graph_args"]
    mp["patcher_gat_param"]["num_hidden"] = r["global_num_hidden"]
    mp["patcher_gat_param"]["out_dim"] = r["global_num_hidden"] * 3
    mp["patcher_gat_param"]["node_feature_dim"] = r["node_feature_dim"]
    mp["localgraph_gat_param"]["num_hidden"] = r["local_num_hidden"]
    mp["localgraph_gat_param"]["out_dim"] = r["local_num_hidden"] * 3
    mp["localgraph_gat_param"]["node_feature_dim"] = r["node_feature_dim"]
    mp["stparam"]["lens"] = r["pre_slot_num"]
    mp["stparam"]["slot_size"] = r["slot_size"]
    mp["stparam"]["hidden_size"] = r["st_hidden_size"]
    mp["stparam"]["total_key_depth"] = r["st_total_key_depth"]
    mp["stparam"]["total_value_depth"] = r["st_total_value_depth"]
    mp["stparam"]["time_embedding_dim"] = r["time_embedding_dim"]
    mp["stparam"]["day_embedding_dim"] = r["day_embedding_dim"]
    mp["stparam"]["max_length"] = max_length
    mp["use_global_gat"] = mp["use_local_gat"] = True
    mp["use_encoder_attention"] = mp["use_decoder_attention"] = True
    mp["road_network"] = road
    mp["g"], mp["inv_g"] = g, inv_g
    mp["n_split_dim"] = len(speed_bins)
    mp["stparam"]["embedding_size_1"] = mp["localgraph_gat_param"]["out_dim"]
    mp["node_size"], mp["edge_size"] = net["n_links"], len(ttypes)

    torch.manual_seed(10)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(10)
    model = O.Leim(mp)
    model.to(device)
    loss_fn = O.NLLLoss(speed_bins, r["alpha_1"], r["alpha_2"], r["lamb"], str(device))
    mape_fn = O.TestNLLLoss(speed_bins, "mape", str(device))
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[STHR] official Leim from {OFFICIAL_CODE_DIR} | parameters {n_params:,} | "
          f"batch size {BATCH_SIZE} | device {device}")

    return dict(
        device=device, model=model, loss_fn=loss_fn, mape_fn=mape_fn, official=O,
        loaders=loaders, datasets=ds, cfg=dict(CFG), readme=dict(r),
        speed_bins=speed_bins, max_length=max_length, n_params=n_params, edges_id_col=id_col,
    )


def _run(core, loader, train=False, grad_clamp=10.0, optimizer=None, collect=False, label="STHR"):
    model, loss_fn, mape_fn, device = core["model"], core["loss_fn"], core["mape_fn"], core["device"]
    model.train() if train else model.eval()
    acc = dict(ll=0.0, rl=0.0, mape=0.0, n=0, l_abs=0.0, l_sse=0.0, n_l=0, l_ape=0.0, n_lm=0,
               r_abs=0.0, r_sse=0.0, n_r=0, r_ape=0.0, n_rm=0)
    trip_rows, link_rows = [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for b in loader:
            x, y = official_batch(b, device)
            if train:
                optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            if train:
                loss[-1].backward()
                _g = [p.grad.detach() for p in model.parameters() if p.grad is not None]
                _gn = float(torch.norm(torch.stack([x.norm(2) for x in _g]))) if _g else float("nan")
                core.setdefault("_grad_norms", []).append(
                    (_gn, bool(any(bool((x.abs() > grad_clamp).any()) for x in _g))))
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad.data.clamp_(-grad_clamp, grad_clamp)
                optimizer.step()
            bs = len(b["trip_ids"])
            with torch.no_grad():
                pred = link_times(out, loss_fn).detach() * TIME_UNIT_S
                mask = b["src_mask"].to(device)
                y0 = b["seq_true_tt"].to(device)
                seq_len = b["seq_link_len"].to(device)
                trip_tt = b["trip_tt"].to(device)
                acc["mape"] += float(mape_fn(out, y)) * bs
                ls = link_metrics_micro(pred, y0, mask, seq_len, L_min=L_MIN_METRIC, T_min=T_MIN_METRIC)
                rs = route_metrics_pooled(pred, trip_tt, mask)
            acc["ll"] += float(loss[0]) * bs
            acc["rl"] += float(loss[1]) * bs
            acc["n"] += bs
            if ls["n_links"]:
                acc["l_abs"] += ls["mae"] * ls["n_links"]; acc["l_sse"] += ls["rmse"] ** 2 * ls["n_links"]
                acc["n_l"] += ls["n_links"]
            if ls["n_mape"]:
                acc["l_ape"] += ls["mape"] * ls["n_mape"]; acc["n_lm"] += ls["n_mape"]
            if rs["n_routes"]:
                acc["r_abs"] += rs["mae"] * rs["n_routes"]; acc["r_sse"] += rs["rmse"] ** 2 * rs["n_routes"]
                acc["n_r"] += rs["n_routes"]
            if rs["n_mape"]:
                acc["r_ape"] += rs["mape"] * rs["n_mape"]; acc["n_rm"] += rs["n_mape"]
            if collect:
                _collect_rows(b, pred, y0, mask, seq_len, trip_tt, trip_rows, link_rows, label)
    nan = float("nan")
    res = dict(
        link_loss=acc["ll"] / max(acc["n"], 1), route_loss=acc["rl"] / max(acc["n"], 1),
        official_route_mape=acc["mape"] / max(acc["n"], 1),
        link_mae=acc["l_abs"] / acc["n_l"] if acc["n_l"] else nan,
        link_rmse=math.sqrt(acc["l_sse"] / acc["n_l"]) if acc["n_l"] else nan,
        link_mape=acc["l_ape"] / acc["n_lm"] if acc["n_lm"] else nan,
        route_mae=acc["r_abs"] / acc["n_r"] if acc["n_r"] else nan,
        route_rmse=math.sqrt(acc["r_sse"] / acc["n_r"]) if acc["n_r"] else nan,
        route_mape=acc["r_ape"] / acc["n_rm"] if acc["n_rm"] else nan,
    )
    return (res, trip_rows, link_rows) if collect else res


def _collect_rows(b, pred, y0, mask, seq_len, trip_tt, trip_rows, link_rows, label):
    route_true = (y0 * mask.float()).sum(1)
    yR = torch.where(trip_tt > 0, trip_tt, route_true).cpu().numpy()
    pR = (pred * mask.float()).sum(1).cpu().numpy()
    rlen = (seq_len * mask.float()).sum(1).cpu().numpy()
    tt = trip_tt.cpu().numpy()
    p, t, m = pred.cpu(), y0.cpu(), mask.cpu()
    for i, tid in enumerate(b["trip_ids"]):
        trip_rows.append((label, int(tid), float(yR[i]), float(pR[i]),
                          "trip_tt" if tt[i] > 0 else "sum_links", float(rlen[i]), 0.0))
        for l in range(p.shape[1]):
            if bool(m[i, l]):
                link_rows.append((label, int(tid), float(t[i, l]), float(p[i, l])))


def run_training(core):
    model = core["model"]
    optimizer = torch.optim.Adam(model.parameters(), lr=core["readme"]["lr"])
    grad_clamp = yaml.safe_load(open(OFFICIAL_CODE_DIR / "config-bash.yaml", encoding="utf-8")) \
        ["training_para"]["grad_clamp"]
    ckpt = CheckpointManager(dirpath=CKPT_DIRNAME, max_keep=3)

    best, best_state, since_best, step = float("inf"), None, 0, 0
    history = {"train": [], "val": [], "epoch_time_s": [], "peak_gpu_mb": []}
    tr = va = None
    t_start = time.perf_counter()
    for epoch in range(1, NUM_EPOCHS + 1):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        tr = _run(core, core["loaders"]["train"], train=True, grad_clamp=grad_clamp, optimizer=optimizer)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ep_time = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else 0.0
        _t1 = time.perf_counter()
        va = _run(core, core["loaders"]["val"])
        step += len(core["loaders"]["train"])
        _f = core.pop("_grad_norms", [])
        _g = np.asarray([x[0] for x in _f], dtype=float)
        _fin = _g[np.isfinite(_g)]
        _trk = {"train_time_s": ep_time, "val_time_s": time.perf_counter() - _t1, "n_steps": int(len(_g)),
                "grad_norm_nonfinite": int(len(_g) - len(_fin)),
                "grad_norm_mean": float(_fin.mean()) if len(_fin) else float("nan"),
                "grad_norm_p99": float(np.percentile(_fin, 99)) if len(_fin) else float("nan"),
                "grad_norm_max": float(_fin.max()) if len(_fin) else float("nan"),
                "frac_steps_clamped": float(np.mean([x[1] for x in _f])) if _f else float("nan"),
                "peak_gpu_mb": peak}
        tr.update(_trk)
        print(f"[track] epoch {epoch:02d}: train {_trk['train_time_s']:.0f}s val {_trk['val_time_s']:.0f}s | "
              f"grad norm mean {_trk['grad_norm_mean']:.3f} p99 {_trk['grad_norm_p99']:.3f} max {_trk['grad_norm_max']:.3f} "
              f"clamped {_trk['frac_steps_clamped']:.2%} non-finite {_trk['grad_norm_nonfinite']} | "
              f"peak GPU {_trk['peak_gpu_mb']:.0f} MB", flush=True)
        history["train"].append(tr); history["val"].append(va)
        history["epoch_time_s"].append(ep_time); history["peak_gpu_mb"].append(peak)

        score = va["official_route_mape"]
        if score < best:
            best, since_best = score, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            ckpt.save_best(model=model, optimizer=optimizer, scheduler=None, epoch=epoch, step=step,
                           best_metric=float(best), extra={"val_metrics": va})
        else:
            since_best += 1
        ckpt.save_last(model=model, optimizer=optimizer, scheduler=None, epoch=epoch, step=step,
                       best_metric=float(best), extra={"val_metrics": va})
        print(f"Epoch {epoch:02d} | {ep_time:7.1f}s | peak {peak:7.0f} MB | "
              f"Train loss link {tr['link_loss']:.4f} route {tr['route_loss']:.4f} | "
              f"Val route MAPE={va['route_mape']:.4f} (official {va['official_route_mape']:.2f}%) "
              f"route MAE={va['route_mae']:.2f} link MAPE={va['link_mape']:.4f}")
        if since_best >= PATIENCE:
            print("Early stopping triggered.")
            break

    total_s = time.perf_counter() - t_start
    if best_state is not None:
        model.load_state_dict(best_state)
    te = _run(core, core["loaders"]["test"])
    print("— Final Test —")
    print(f"Link:  MAE={te['link_mae']:.4f} | RMSE={te['link_rmse']:.4f} | MAPE={te['link_mape']:.4f}")
    print(f"Route: MAE={te['route_mae']:.4f} | RMSE={te['route_rmse']:.4f} | MAPE={te['route_mape']:.4f}")

    summary = dict(test=te, best_val_official_route_mape=float(best), epochs_run=len(history["train"]),
                   total_train_time_s=total_s,
                   mean_epoch_time_s=float(np.mean(history["epoch_time_s"])) if history["epoch_time_s"] else None,
                   max_peak_gpu_mb=float(max(history["peak_gpu_mb"])) if history["peak_gpu_mb"] else None,
                   grad_norm_max_all=float(np.nanmax([e["grad_norm_max"] for e in history["train"]])) if history["train"] else None,
                   grad_norm_nonfinite_all=int(sum(e["grad_norm_nonfinite"] for e in history["train"])),
                   n_params=core["n_params"], batch_size=BATCH_SIZE, speed_bins_kmh=core["speed_bins"],
                   loss_time_unit_s=TIME_UNIT_S,
                   max_length=core["max_length"], implementation="official Leim2.py (unmodified)",
                   torch=torch.__version__)
    with open(os.path.join(ckpt.dir, "training_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, default=str)
    with open(os.path.join(ckpt.dir, "training_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[history] {os.path.join(ckpt.dir, 'training_history.json')}")
    print(f"[summary] {os.path.join(ckpt.dir, 'training_summary.json')}")
    return dict(train_last=tr, val_last=va, test=te, best_selector=float(best))


@torch.no_grad()
def eval_sthr(core, ckpt_path=None, label="STHR", split="test"):
    if ckpt_path:
        pkg = torch.load(ckpt_path, map_location=core["device"])
        core["model"].load_state_dict(pkg["model"] if isinstance(pkg, dict) and "model" in pkg else pkg)
    res, trip_rows, link_rows = _run(core, core["loaders"][split], collect=True, label=label)
    metrics = dict(res)
    metrics.update(n_routes=len(trip_rows), n_links=len(link_rows))
    trip_df = pd.DataFrame(trip_rows, columns=["model", "TRIP_ID", "y_true", "y_pred",
                                               "target_source", "route_len_m", "gain_g"])
    link_df = pd.DataFrame(link_rows, columns=["model", "TRIP_ID", "true_tt_link", "pred_tt_link"])
    return metrics, trip_df, link_df

