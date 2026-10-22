import os
import json
import time
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sps
import torch
import torch.nn as nn
import torch.nn.functional as F
from shapely.geometry import MultiLineString
from shapely.ops import linemerge
from torch_geometric.nn import SAGEConv

import LinkETA as li
from loss_functions import route_metrics_pooled
from checkpoint import CheckpointManager

CKPT_DIRNAME = "ckpts_GSPINN"
CACHE_DIR = None
CITY_FOLDER = None

NUM_EPOCHS = 100
PATIENCE = 15
SEED = 42

LR = 1e-4
HIDDEN = 128
DROPOUT = 0.1
SMOOTH_L1_BETA = 50.0
LAMBDA_GRID = (0.0, 0.1, 0.3, 0.5, 1.0, 1.5)
LAMBDA_MONO = 0.5

N_HOPS = 3
STATE_S = 600
FLOW_WINDOW_S = 3600
CLASS_WINDOW_S = 600
SIM_TIME_S = 3600
SIGNAL_RADIUS_M = 0.5

STATIC_FEATURES = ["num_lanes", "length_m", "speed_limit_mps", "avg_curvature", "has_traffic_signal"]
NODE_FEATURES = STATIC_FEATURES + ["analytical_flow_veh_per_s", "demand_class"]
POS_FEATS = ["length_m", "avg_curvature"]
DF_ALL_STATIC = {"num_lanes": "lanes", "length_m": "length", "speed_limit_mps": "speed_limit_ms",
                 "has_traffic_signal": "has_traffic_signal"}


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def curvature(shape):
    if not isinstance(shape, (list, tuple)) or len(shape) < 3:
        return 0.0
    angles = []
    for i in range(1, len(shape) - 1):
        try:
            p1, p2, p3 = np.array(shape[i-1]), np.array(shape[i]), np.array(shape[i+1])
            v1, v2 = p2 - p1, p3 - p2
            dot = np.dot(v1, v2)
            denom = (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            angle = np.arccos(np.clip(dot / denom, -1, 1))
            angles.append(angle)
        except:
            continue
    return np.mean(angles) if angles else 0.0


def road_points(geom):
    parts = list(geom.geoms) if isinstance(geom, MultiLineString) else [geom]
    pts = [tuple(c) for c in parts[0].coords]
    for p in parts[1:]:
        c = [tuple(x) for x in p.coords]
        if np.hypot(c[0][0] - pts[-1][0], c[0][1] - pts[-1][1]) > 1e-6:
            m = linemerge(geom)
            return [[tuple(x) for x in s.coords] for s in (m.geoms if isinstance(m, MultiLineString) else [m])]
        pts.extend(c[1:])
    return [pts]


def road_static_table(edges_gdf, crs_metric):
    import map_matching as mm
    st = mm.build_static_features(edges_gdf)
    out = pd.DataFrame(index=pd.RangeIndex(len(edges_gdf), name="edge_idx"))
    out["num_lanes"] = pd.to_numeric(st["lanes"]).to_numpy(np.float64)
    out["length_m"] = pd.to_numeric(st["length"]).to_numpy(np.float64)
    out["speed_limit_mps"] = pd.to_numeric(st["speed_limit_ms"]).to_numpy(np.float64)
    out["avg_curvature"] = [float(np.mean([curvature(s) for s in road_points(g)]))
                            for g in edges_gdf.to_crs(crs_metric).geometry]
    if not np.isfinite(out.to_numpy()).all():
        raise SystemExit("[GSPINN] a static road input is missing (NaN)")
    return out


def signal_counts(edges_gdf, ids, crs_metric, pbf):
    fp = li._sha("gspinn_signals", li.CACHE_SCHEMA, pbf.name, pbf.stat().st_size, crs_metric, SIGNAL_RADIUS_M,
                 *ids.tolist(), *edges_gdf.geometry.to_wkb().tolist())
    got = li.load_cached("gspinn_signals", fp) if li.CACHE_DIR else None
    if got is not None:
        return got["signal_count"]
    from pyrosm import OSM
    from scipy.spatial import cKDTree
    n = OSM(str(pbf), bounding_box=edges_gdf.total_bounds.tolist()).get_data_by_custom_criteria(
        custom_filter={"highway": ["traffic_signals"]}, filter_type="keep",
        keep_nodes=True, keep_ways=False, keep_relations=False)
    if n is None or not len(n):
        raise SystemExit(f"[GSPINN] no highway=traffic_signals node of {pbf.name} in the road box: "
                         f"signal_count would be a constant input")
    pts, road = [], []
    for i, g in enumerate(edges_gdf.to_crs(crs_metric).geometry):
        for p in (q for s in road_points(g) for q in s):
            pts.append(p); road.append(i)
    road = np.asarray(road)
    nm = n.to_crs(crs_metric)
    cnt = np.zeros(len(edges_gdf), dtype=np.int64)
    for h in cKDTree(np.asarray(pts)).query_ball_point(np.c_[nm.geometry.x, nm.geometry.y], r=SIGNAL_RADIUS_M):
        for i in set(road[h].tolist()):
            cnt[i] += 1
    print(f"[GSPINN] {len(n):,} signal nodes in {pbf.name}; roads with a signal {int((cnt > 0).sum()):,} of {len(cnt):,}")
    if li.CACHE_DIR:
        li.save_cached("gspinn_signals", fp, dict(signal_count=cnt, n_signal_nodes=int(len(n))))
    return cnt


def shared_point_adjacency(edges_gdf):
    owner = defaultdict(set)
    for i, geom in enumerate(edges_gdf.geometry):
        for p in (list(geom.geoms) if isinstance(geom, MultiLineString) else [geom]):
            for x, y in p.coords:
                owner[(round(x, 6), round(y, 6))].add(i)
    r, c = [], []
    for s in owner.values():
        if len(s) > 1:
            for a in s:
                for b in s:
                    if a != b:
                        r.append(a); c.append(b)
    N = len(edges_gdf)
    A = sps.csr_matrix((np.ones(len(r), np.float32), (np.asarray(r, np.int64), np.asarray(c, np.int64))), shape=(N, N))
    A.data[:] = 1.0
    return A


def state_tables(hist, way2edge, n_roads, starts):
    W = np.empty(len(hist.T), dtype=np.int64)
    for w, ix in hist.by_way.items():
        W[ix] = w
    o = np.argsort(hist.T, kind="mergesort")
    T, W, TR, AV = hist.T[o], W[o], np.asarray(hist.TR, dtype=np.int64)[o], hist.avail_ts[o]
    uw = np.unique(W)
    E = np.array([way2edge.get(int(w), -1) for w in uw], dtype=np.int64)[np.searchsorted(uw, W)]
    rows, cols, vals = [], [], []
    level = np.zeros(len(starts), dtype=np.int64)
    for r, S in enumerate(starts.astype(np.float64)):
        a, b = np.searchsorted(T, S - FLOW_WINDOW_S, "left"), np.searchsorted(T, S, "left")
        k = (AV[a:b] < S) & (E[a:b] >= 0)
        if k.any():
            pairs = np.unique(np.stack([E[a:b][k], TR[a:b][k]], 1), axis=0)
            e, n = np.unique(pairs[:, 0], return_counts=True)
            rows.append(np.full(len(e), r)); cols.append(e); vals.append(n / FLOW_WINDOW_S)
        a = np.searchsorted(T, S - CLASS_WINDOW_S, "left")
        level[r] = len(np.unique(TR[a:b][AV[a:b] < S]))
    if not rows:
        raise SystemExit("[GSPINN] no speed record before any state start: the flow input would be zero everywhere")
    flow = sps.csr_matrix((np.concatenate(vals).astype(np.float32), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(len(starts), n_roads))
    return flow, level, int((E < 0).sum())


class GSPINNNet(nn.Module):
    def __init__(self, node_in_dim, x_mean, x_std, tt_mu, tt_sd, hidden=HIDDEN, dropout=DROPOUT):
        super().__init__()
        self.register_buffer("x_mean", x_mean)
        self.register_buffer("x_std", x_std)
        self.register_buffer("tt_mu", torch.tensor(float(tt_mu)))
        self.register_buffer("tt_sd", torch.tensor(float(tt_sd)))
        self.input_proj = nn.Linear(node_in_dim, hidden)
        self.dropout = nn.Dropout(dropout)
        self.conv1 = SAGEConv(hidden, hidden)
        self.norm1 = nn.LayerNorm(hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.conv3 = SAGEConv(hidden, hidden)
        self.norm3 = nn.LayerNorm(hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.readout = nn.Sequential(
            nn.Linear(hidden + 1, hidden // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden // 2),
            nn.Linear(hidden // 2, 1)
        )
        nn.init.xavier_uniform_(self.readout[0].weight)
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)

    def encode(self, x_raw, edge_index):
        x = (x_raw - self.x_mean) / self.x_std
        x = self.dropout(F.relu(self.input_proj(x)))
        for conv, norm in ((self.conv1, self.norm1), (self.conv2, self.norm2), (self.conv3, self.norm3)):
            x = self.dropout(F.relu(norm(conv(x, edge_index))))
        return x

    def forward(self, x_raw, edge_index, paths, mask, seq_len_norm):
        H = self.encode(x_raw, edge_index)
        E = H[paths.clamp(min=0)] * mask.unsqueeze(-1)
        lengths = mask.sum(dim=1).cpu()
        with torch.backends.cudnn.flags(enabled=False):
            packed = nn.utils.rnn.pack_padded_sequence(E, lengths, batch_first=True, enforce_sorted=False)
            _, h_n = self.gru(packed)
        trip_emb = torch.cat([h_n[-1], seq_len_norm.to(h_n.device)], dim=1)
        return self.tt_mu + self.tt_sd * self.readout(trip_emb).squeeze(1)


def monotonicity_loss_on_nodes(pred, x_sub):
    grad = torch.autograd.grad(outputs=pred, inputs=x_sub, grad_outputs=torch.ones_like(pred), create_graph=True)[0]
    return sum(F.relu(-grad[:, NODE_FEATURES.index(f)]).mean() for f in POS_FEATS) / len(POS_FEATS)


def gspinn_get_core_components(edges_gdf, df_all):
    import sampling
    li.CKPT_DIRNAME = CKPT_DIRNAME
    li.CACHE_DIR = CACHE_DIR
    lcore = li.linketa_get_core_components(edges_gdf, df_all)
    assert CITY_FOLDER is not None, "CITY_FOLDER not set (main.py sets it)"
    cfg = json.loads((Path(sampling.CONFIG_DIR) / f"{CITY_FOLDER}.json").read_text(encoding="utf-8"))
    crs = cfg["map_matching"]["crs_metric"]
    pbf = Path(sampling.RAW_ROOT) / CITY_FOLDER / cfg["area"]["pbf"]
    idc = li.guess_edges_id_col(edges_gdf, li.CFG["way_id_col_edges_candidates"])
    ids = edges_gdf[idc].astype("int64").to_numpy()
    e2w = lcore["datasets"]["train"].edge2way
    if len(e2w) != len(ids) or any(int(e2w[i]) != int(ids[i]) for i in range(len(ids))):
        raise SystemExit("[GSPINN] LinkETA's edge_idx is not the row order of the road file")

    fp_roads = li._sha("gspinn_roads", li.CACHE_SCHEMA, crs,
                       li.hash_frame(edges_gdf, [idc, "lanes", "maxspeed", "width", "highway", "length"]),
                       *edges_gdf.geometry.to_wkb().tolist())
    got = li.load_cached("gspinn_roads", fp_roads) if li.CACHE_DIR else None
    if got is not None:
        road, adj = got["road"], got["adj"]
    else:
        road = road_static_table(edges_gdf, crs)
        adj = shared_point_adjacency(edges_gdf)
        if li.CACHE_DIR:
            li.save_cached("gspinn_roads", fp_roads, dict(road=road, adj=adj))
    import map_matching as mm
    rule = mm.build_static_features(edges_gdf).reset_index(drop=True)
    road["has_traffic_signal"] = pd.to_numeric(rule["has_traffic_signal"]).to_numpy(np.float64)
    dfs = df_all.groupby("way_id")[list(DF_ALL_STATIC.values())].first()
    pos = pd.Series(np.arange(len(ids)), index=ids)
    hit = dfs.index.isin(pos.index)
    rows_e = pos.loc[dfs.index[hit]].to_numpy()
    for f, col in DF_ALL_STATIC.items():
        v = pd.to_numeric(dfs[col][hit]).to_numpy(np.float64)
        if not np.allclose(road[f].to_numpy()[rows_e], v, rtol=0, atol=1e-9):
            raise SystemExit(f"[GSPINN] df_all {col} differs from the pipeline rule on a driven road")
        road.iloc[rows_e, road.columns.get_loc(f)] = v
    print(f"[GSPINN] road features from df_all on {int(hit.sum()):,} driven roads, pipeline rule on the other "
          f"{len(ids) - int(hit.sum()):,}; has_traffic_signal non-zero on {int((road['has_traffic_signal'] != 0).sum())} roads")
    static = road[STATIC_FEATURES].to_numpy(np.float64)
    N = len(ids)

    d = df_all[["TRIP_ID", "departure_time"]].copy()
    d["departure_time"] = pd.to_datetime(d["departure_time"], errors="coerce")
    dep = {int(k): v.timestamp() for k, v in d.dropna().groupby("TRIP_ID", sort=False)["departure_time"].first().items()}
    trips = {s: [int(g["trip_id"]) for g in lcore["datasets"][s].groups] for s in ("train", "val", "test")}
    start = {t: int(dep[t] // STATE_S) * STATE_S for s in trips for t in trips[s]}
    starts = np.array(sorted(set(start.values())), dtype=np.int64)
    srow = {t: int(np.searchsorted(starts, S)) for t, S in start.items()}
    way2edge = {int(w): int(e) for e, w in e2w.items()}
    fp_states = li._sha("gspinn_states", li.CACHE_SCHEMA, li.SPEED_HISTORY.fingerprint, starts.tobytes(), ids.tobytes(),
                        FLOW_WINDOW_S, CLASS_WINDOW_S)
    got = li.load_cached("gspinn_states", fp_states) if li.CACHE_DIR else None
    if got is not None:
        flow, level, n_out = got["flow"], got["level"], got["n_out"]
    else:
        flow, level, n_out = state_tables(li.SPEED_HISTORY, way2edge, N, starts)
        if li.CACHE_DIR:
            li.save_cached("gspinn_states", fp_states, dict(flow=flow, level=level, n_out=n_out))
    routes = {int(g["trip_id"]): g["node_edge_idx"].numpy()[g["route_seq_idx"].numpy()]
              for s in ("train", "val", "test") for g in lcore["datasets"][s].groups}
    tids = sorted(routes)
    fp_khop = li._sha("gspinn_khop", li.CACHE_SCHEMA, fp_roads, N_HOPS, np.asarray(tids, np.int64).tobytes(),
                      np.asarray([len(routes[t]) for t in tids], np.int64).tobytes(),
                      np.concatenate([routes[t] for t in tids]).astype(np.int64).tobytes())
    got = li.load_cached("gspinn_khop", fp_khop) if li.CACHE_DIR else None
    if got is not None:
        khop_nodes = got["khop_nodes"]
    else:
        khop_nodes = {t: _khop(adj, np.unique(r)).astype(np.int32) for t, r in routes.items()}
        if li.CACHE_DIR:
            li.save_cached("gspinn_khop", fp_khop, dict(khop_nodes=khop_nodes))
    tr_rows = np.unique([srow[t] for t in trips["train"]])
    median = float(np.median(level[tr_rows]))
    cls = (level >= median).astype(np.float64)
    len_max = max(len(g["route_seq_idx"]) for g in lcore["datasets"]["train"].groups)

    n_tr = len(tr_rows)
    n = n_tr * N
    ftr = flow[tr_rows].astype(np.float64)
    sums = np.concatenate([n_tr * static.sum(0), [ftr.sum(), N * cls[tr_rows].sum()]])
    sq = np.concatenate([n_tr * (static ** 2).sum(0), [ftr.multiply(ftr).sum(), N * (cls[tr_rows] ** 2).sum()]])
    mean = sums / n
    std = np.sqrt(np.maximum(sq - n * mean ** 2, 0.0) / (n - 1)) + 1e-6

    print(f"[GSPINN] roads {N:,}, links {adj.nnz // 2:,} (shared point), roads without a link "
          f"{np.mean(np.diff(adj.indptr) == 0):.4f}; records on roads outside the road file {n_out:,}")
    for s in ("train", "val", "test"):
        rs = np.array([srow[t] for t in trips[s]])
        pos_n = tot = 0
        for g in lcore["datasets"][s].groups:
            es = g["node_edge_idx"].numpy()[g["route_seq_idx"].numpy()]
            fv = flow[srow[int(g["trip_id"])]].toarray().ravel()[es]
            pos_n += int((fv > 0).sum()); tot += len(es)
        print(f"[GSPINN] {s}: {len(trips[s]):,} trips in {len(np.unique(rs)):,} states; route roads with flow > 0 "
              f"{pos_n / max(tot, 1):.3f}; demand class 1 on {cls[np.unique(rs)].mean():.3f} of states")
    print(f"[GSPINN] demand class: training-state median of the {CLASS_WINDOW_S // 60}-min level {median:.1f}; "
          f"longest training route {len_max} roads")
    print("[GSPINN] input mean / std: " + ", ".join(f"{f} {m:.4g} / {v:.4g}" for f, m, v in zip(NODE_FEATURES, mean, std)))

    tt = pd.Series([float(g["trip_tt"]) for g in lcore["datasets"]["train"].groups])
    tt_mu, tt_sd = float(tt[tt > 0].mean()), float(tt[tt > 0].std() or 1.0)
    print(f"[GSPINN] readout scale: training trip time mean {tt_mu:.1f} s, std {tt_sd:.1f} s")

    device = lcore["device"]
    set_seed(SEED)
    model = GSPINNNet(len(NODE_FEATURES), torch.tensor(mean, dtype=torch.float32).view(1, -1),
                      torch.tensor(std, dtype=torch.float32).view(1, -1), tt_mu, tt_sd).to(device)
    norm = dict(x_mean=mean.tolist(), x_std=std.tolist(), class_median=median, len_max=int(len_max),
                n_train_states=int(n_tr), tt_mu=tt_mu, tt_sd=tt_sd)
    return dict(device=device, model=model, linketa_core=lcore, loaders=lcore["loaders"], static=static, adj=adj,
                flow=flow, state_starts=starts, state_level=level, state_class=cls, trip_state_row=srow,
                khop_nodes=khop_nodes, len_max=len_max, norm=norm, crs=crs, pbf=str(pbf),
                n_params=sum(p.numel() for p in model.parameters()))


def _khop(adj, seeds, k=N_HOPS):
    m = np.zeros(adj.shape[0], dtype=bool)
    m[seeds] = True
    for _ in range(k):
        m |= (adj @ m.astype(np.float32)) > 0
    return np.flatnonzero(m)


def _state_x(core, r, nodes):
    x = np.empty((len(nodes), len(NODE_FEATURES)), dtype=np.float32)
    x[:, :len(STATIC_FEATURES)] = core["static"][nodes]
    x[:, len(STATIC_FEATURES)] = core["flow"][r].toarray().ravel()[nodes]
    x[:, len(STATIC_FEATURES) + 1] = core["state_class"][r]
    return x


def _batch_graph(core, batch, whole=False):
    dev = core["device"]
    route = torch.gather(batch["node_edge_idx"], 1, batch["route_seq_idx"]).numpy()
    mask = batch["seq_mask"].bool().numpy()
    rows = np.array([core["trip_state_row"][int(t)] for t in batch["trip_ids"]])
    paths = np.zeros(route.shape, dtype=np.int64)
    xs, eis, off = [], [], 0
    for r in np.unique(rows):
        mem = np.flatnonzero(rows == r)
        nodes = (np.arange(core["adj"].shape[0]) if whole else
                 np.unique(np.concatenate([core["khop_nodes"][int(batch["trip_ids"][i])] for i in mem])))
        sub = core["adj"][nodes][:, nodes].tocoo()
        eis.append(np.stack([sub.row, sub.col]).astype(np.int64) + off)
        xs.append(_state_x(core, r, nodes))
        for i in mem:
            paths[i, mask[i]] = np.searchsorted(nodes, route[i, mask[i]]) + off
        off += len(nodes)
    paths_t, mask_t = torch.from_numpy(paths).to(dev), torch.from_numpy(mask).to(dev)
    return dict(x_raw=torch.from_numpy(np.concatenate(xs)).to(dev),
                edge_index=torch.from_numpy(np.concatenate(eis, axis=1)).to(dev),
                paths=paths_t, mask=mask_t,
                seq_len_norm=torch.tensor(mask.sum(1) / core["len_max"], dtype=torch.float32, device=dev).view(-1, 1),
                visited=torch.unique(paths_t[mask_t]))


def _with_grad_inputs(g):
    x_sub = g["x_raw"][g["visited"]].clone().detach().requires_grad_(True)
    x_mod = g["x_raw"].clone()
    x_mod[g["visited"]] = x_sub
    return x_sub, x_mod


def _route_metrics(pred, trip_tt):
    return route_metrics_pooled(pred.unsqueeze(1), trip_tt, torch.ones_like(pred, dtype=torch.bool).unsqueeze(1))


def _run_epoch(core, split, optimizer=None):
    model, train = core["model"], optimizer is not None
    model.train() if train else model.eval()
    preds, trues, loss_sum, mono_sum, n, nb = [], [], 0.0, 0.0, 0, 0
    for batch in core["loaders"][split]:
        trip_tt = torch.nan_to_num(batch["trip_tt"].to(core["device"]), nan=0.0)
        pos = trip_tt > 0
        g = _batch_graph(core, batch)
        if train:
            x_sub, x_mod = _with_grad_inputs(g)
            pred = model(x_mod, g["edge_index"], g["paths"], g["mask"], g["seq_len_norm"])
            data_loss = (F.smooth_l1_loss(pred[pos], trip_tt[pos], beta=SMOOTH_L1_BETA)
                         if pos.any() else pred.sum() * 0.0)
            mono_loss = monotonicity_loss_on_nodes(pred, x_sub)
            loss = data_loss + LAMBDA_MONO * mono_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            _g = [p.grad.detach() for p in model.parameters() if p.grad is not None]
            _gn = float(torch.norm(torch.stack([x.norm(2) for x in _g]))) if _g else float("nan")
            core.setdefault("_grad_norms", []).append(_gn)
            optimizer.step()
            mono_sum += float(mono_loss.item()); nb += 1
        else:
            with torch.no_grad():
                pred = model(g["x_raw"], g["edge_index"], g["paths"], g["mask"], g["seq_len_norm"])
                loss = (F.smooth_l1_loss(pred[pos], trip_tt[pos], beta=SMOOTH_L1_BETA)
                        if pos.any() else pred.sum() * 0.0)
        loss_sum += float(loss.item()) * int(pos.sum().item()); n += int(pos.sum().item())
        preds.append(pred.detach()); trues.append(trip_tt)
    m = _route_metrics(torch.cat(preds), torch.cat(trues))
    out = dict(loss=loss_sum / max(n, 1), route_mae=m["mae"], route_rmse=m["rmse"], route_mape=m["mape"])
    if train:
        out["mono"] = mono_sum / max(nb, 1)
    return out


def evaluate_test_monotonicity(core, split="test"):
    model = core["model"]
    model.eval()
    total, nb = 0.0, 0
    for batch in core["loaders"][split]:
        g = _batch_graph(core, batch)
        x_sub, x_mod = _with_grad_inputs(g)
        pred = model(x_mod, g["edge_index"], g["paths"], g["mask"], g["seq_len_norm"])
        total += float(monotonicity_loss_on_nodes(pred, x_sub).item()); nb += 1
    return total / nb if nb else float("nan")


def run_training(core):
    if LAMBDA_MONO is None:
        raise SystemExit(f"[GSPINN] LAMBDA_MONO not set: choose it from {LAMBDA_GRID} in the 100-trip smoke test "
                         f"(main.py --lambda-mono)")
    set_seed(SEED)
    out_dir = CKPT_DIRNAME
    os.makedirs(out_dir, exist_ok=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    model = core["model"]
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    ckpt = CheckpointManager(dirpath=out_dir, max_keep=3)
    print(f"[GSPINN] parameters {core['n_params']:,} | epochs {NUM_EPOCHS} | patience {PATIENCE} | lr {LR} | "
          f"lambda {LAMBDA_MONO} | SmoothL1 beta {SMOOTH_L1_BETA} | inputs {NODE_FEATURES} | F+ {POS_FEATS} | "
          f"device {core['device']}")
    history = {"train": [], "val": [], "epoch_time_s": [], "lr": []}
    best, best_state, since_best = float("inf"), None, 0
    best_va = {}
    t_start = time.perf_counter()
    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.perf_counter()
        tr = _run_epoch(core, "train", optimizer)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ep_time = time.perf_counter() - t0
        _t1 = time.perf_counter()
        va = _run_epoch(core, "val")
        _g = np.asarray(core.pop("_grad_norms", []), dtype=float)
        _fin = _g[np.isfinite(_g)]
        _trk = {"train_time_s": ep_time, "val_time_s": time.perf_counter() - _t1, "n_steps": int(len(_g)),
                "grad_norm_nonfinite": int(len(_g) - len(_fin)),
                "grad_norm_mean": float(_fin.mean()) if len(_fin) else float("nan"),
                "grad_norm_p99": float(np.percentile(_fin, 99)) if len(_fin) else float("nan"),
                "grad_norm_max": float(_fin.max()) if len(_fin) else float("nan"),
                "peak_gpu_mb": float(torch.cuda.max_memory_allocated() / 1024 ** 2) if torch.cuda.is_available() else float("nan")}
        tr.update(_trk)
        print(f"[track] epoch {epoch:03d}: train {_trk['train_time_s']:.0f}s val {_trk['val_time_s']:.0f}s | "
              f"grad norm mean {_trk['grad_norm_mean']:.3f} p99 {_trk['grad_norm_p99']:.3f} max {_trk['grad_norm_max']:.3f} "
              f"non-finite {_trk['grad_norm_nonfinite']} | peak GPU {_trk['peak_gpu_mb']:.0f} MB", flush=True)
        history["train"].append(tr); history["val"].append(va)
        history["epoch_time_s"].append(ep_time); history["lr"].append(optimizer.param_groups[0]["lr"])
        if va["route_mae"] < best - 1e-6:
            best, since_best, best_va = va["route_mae"], 0, va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            ckpt.save_best(model=model, optimizer=optimizer, scheduler=None, epoch=epoch, step=epoch,
                           best_metric=float(best), extra={"val_metrics": va, "model": "GSPINN", "lambda_mono": LAMBDA_MONO})
        else:
            since_best += 1
        ckpt.save_last(model=model, optimizer=optimizer, scheduler=None, epoch=epoch, step=epoch,
                       best_metric=float(best), extra={"val_metrics": va, "model": "GSPINN", "lambda_mono": LAMBDA_MONO})
        print(f"Epoch {epoch:03d} | {ep_time:6.1f}s | Train loss {tr['loss']:.4f} mono {tr['mono']:.3e} "
              f"MAPE {tr['route_mape']*100:.2f}% | Val MAE {va['route_mae']:.2f}s MAPE {va['route_mape']*100:.2f}% "
              f"(best MAE {best:.2f}s)")
        if since_best >= PATIENCE:
            print("Early stopping triggered.")
            break
    total_s = time.perf_counter() - t_start
    peak = torch.cuda.max_memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else 0.0
    if best_state is not None:
        model.load_state_dict(best_state)
    metrics, _, _ = eval_gspinn(core)
    metrics["test_mono"] = evaluate_test_monotonicity(core)
    print("— Final Test —")
    print(f"Route: MAE={metrics['route_mae']:.4f} | RMSE={metrics['route_rmse']:.4f} | MAPE={metrics['route_mape']:.4f} "
          f"|| monotonicity penalty on test {metrics['test_mono']:.3e}")
    summary = dict(test=metrics, best_val_route_mae=float(best),
                   best_val_route_mape=float(best_va.get("route_mape", float("nan"))), epochs_run=len(history["train"]),
                   total_train_time_s=total_s, mean_epoch_time_s=float(np.mean(history["epoch_time_s"])),
                   max_peak_gpu_mb=float(peak), n_params=core["n_params"],
                   batch_size=core["loaders"]["train"].batch_size, norm=core["norm"], torch=torch.__version__,
                   grad_norm_max_all=float(np.nanmax([e["grad_norm_max"] for e in history["train"]])) if history["train"] else None,
                   grad_norm_nonfinite_all=int(sum(e["grad_norm_nonfinite"] for e in history["train"])),
                   config=dict(LR=LR, HIDDEN=HIDDEN, DROPOUT=DROPOUT, SMOOTH_L1_BETA=SMOOTH_L1_BETA,
                               LAMBDA_MONO=LAMBDA_MONO, LAMBDA_GRID=LAMBDA_GRID, N_HOPS=N_HOPS, STATE_S=STATE_S,
                               FLOW_WINDOW_S=FLOW_WINDOW_S, CLASS_WINDOW_S=CLASS_WINDOW_S,
                               SIGNAL_RADIUS_M=SIGNAL_RADIUS_M, NODE_FEATURES=NODE_FEATURES, POS_FEATS=POS_FEATS,
                               NUM_EPOCHS=NUM_EPOCHS, PATIENCE=PATIENCE, SEED=SEED))
    with open(os.path.join(out_dir, "training_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, default=str)
    with open(os.path.join(out_dir, "training_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[summary] {os.path.join(out_dir, 'training_summary.json')}")
    return dict(test=metrics, best_selector=float(best))


@torch.no_grad()
def eval_gspinn(core, ckpt_path: Optional[str] = None, label: str = "GSPINN", split: str = "test"):
    model = core["model"]
    if ckpt_path:
        pkg = torch.load(ckpt_path, map_location=core["device"])
        model.load_state_dict(pkg["model"] if isinstance(pkg, dict) and "model" in pkg else pkg, strict=True)
    model.eval()
    rows, preds, trues = [], [], []
    for batch in core["loaders"][split]:
        g = _batch_graph(core, batch)
        pred = model(g["x_raw"], g["edge_index"], g["paths"], g["mask"], g["seq_len_norm"])
        trip_tt = torch.nan_to_num(batch["trip_tt"].to(core["device"]), nan=0.0)
        seq_mask = batch["seq_mask"].to(core["device"]).bool()
        seq_len = torch.nan_to_num(batch["seq_link_len"].to(core["device"]))
        y = torch.where(trip_tt > 0, trip_tt,
                        (torch.nan_to_num(batch["seq_true_tt"].to(core["device"])) * seq_mask.float()).sum(1))
        rlen = (seq_len * seq_mask.float()).sum(1)
        for i, tid in enumerate(batch["trip_ids"]):
            rows.append((label, int(tid), float(y[i]), float(pred[i]),
                         "trip_tt" if float(trip_tt[i]) > 0 else "sum_links", float(rlen[i]), 0.0))
        preds.append(pred); trues.append(trip_tt)
    m = _route_metrics(torch.cat(preds), torch.cat(trues))
    nan = float("nan")
    metrics = dict(link_loss=nan, link_mae=nan, link_rmse=nan, link_mape=nan, route_loss=m["mae"],
                   route_mae=m["mae"], route_rmse=m["rmse"], route_mape=m["mape"], n_links=0, n_routes=len(rows))
    trip_df = pd.DataFrame(rows, columns=["model", "TRIP_ID", "y_true", "y_pred", "target_source", "route_len_m", "gain_g"])
    link_df = pd.DataFrame(columns=["model", "TRIP_ID", "true_tt_link", "pred_tt_link"])
    return metrics, trip_df, link_df

