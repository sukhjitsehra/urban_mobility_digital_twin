import argparse
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main
import LinkETA as li
import ablations as ab

P_LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)


def _selected(trip_id, pos, p):
    return zlib.crc32(f"{trip_id}:{pos}".encode()) / 2 ** 32 < p


class _Blanker:

    def __init__(self, core, sink):
        dev = core["device"]
        dyn = li.DYN_IDX.to(dev)
        mean = core["scalers"]["mean"].to(dev).float()
        std = core["scalers"]["std"].to(dev).float()
        self.dyn = dyn
        self.blank_value = ((0.0 - mean[dyn]) / std[dyn])
        self.sink, self.p = sink, 0.0
        self.n_blanked = 0
        self.max_dev = 0.0
        self.blanked_rows = []

    def __call__(self, module, args):
        if self.p <= 0:
            return None
        x, route_flag = args[0].clone(), args[2].bool()
        trip_ids = self.sink["trip_ids"]
        sel = torch.zeros_like(route_flag)
        for b, tid in enumerate(trip_ids):
            L = int(route_flag[b].sum())
            for j in range(L):
                if _selected(tid, j, self.p):
                    sel[b, j] = True
                    self.blanked_rows.append((str(tid), j))
        if sel.any():
            xb = x[sel]
            xb[:, self.dyn] = self.blank_value.to(xb.dtype)
            x[sel] = xb
            self.max_dev = max(self.max_dev, float((x[sel][:, self.dyn] - self.blank_value).abs().max()))
            self.n_blanked += int(sel.sum())
        return (x,) + tuple(args[1:])


class _Batches:
    def __init__(self, loader, sink):
        self.loader, self.sink = loader, sink

    def __iter__(self):
        for batch in self.loader:
            self.sink["trip_ids"] = batch["trip_ids"]
            yield batch


def _wape(err, true):
    m = true > 0
    return float(np.abs(err[m]).sum() / max(np.abs(true[m]).sum(), 1e-12))


def build(city, model_name, edges_gdf, df_all):
    import tempfile
    folder = main.results_dir(city)
    if model_name == "full":
        ab._restore()
        ckpt = folder / main.ckpt_dirname("LinkETA", city) / "best.ckpt"
        ref = json.loads((folder / "eval_outputs" / "eval_metrics_linketa.json").read_text())["LinkETA"]["route_mape"]
        spec = None
    else:
        spec = ab.VARIANTS[model_name]
        ab._apply_before_build(spec)
        ckpt = folder / main.ckpt_dirname("LinkETA", city, suffix=f"_abl_{model_name}") / "best.ckpt"
        ref = json.loads((ckpt.parent / "training_summary.json").read_text())["test"]["route_mape"]
    try:
        li_mod = main._setup_linketa_new(city)
        li_mod.CKPT_DIRNAME = tempfile.mkdtemp(prefix="robustness_")
        core = li_mod.linketa_get_core_components(edges_gdf, df_all)
        if spec is not None:
            ab._apply_after_build(spec, core)
    finally:
        ab._restore()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    pkg = torch.load(ckpt, map_location=core["device"], weights_only=False)
    core["model"].load_state_dict(pkg["model"], strict=True)
    print(f"[robustness] {model_name}: loaded {ckpt} (epoch {pkg.get('epoch')}); saved test route MAPE {ref:.4f}")
    return core, ref


def evaluate(city, model_name, core, ref, out_csv, levels=P_LEVELS, trips_csv=None):
    sink = {}
    blank = _Blanker(core, sink)
    loaders = core["loaders"]
    original = loaders["test"]
    loaders["test"] = _Batches(original, sink)
    hook = core["model"].register_forward_pre_hook(blank)
    rows = []
    try:
        for p in levels:
            blank.p, blank.n_blanked, blank.max_dev, blank.blanked_rows = p, 0, 0.0, []
            m, trip_df, link_df = core["eval_linketa"](core, label=f"LinkETA_{model_name}")
            n_links = len(link_df)
            if p == 0:
                main_ok = abs(m["route_mape"] - ref) < 1e-3
                assert main_ok, f"{model_name}: p=0 route MAPE {m['route_mape']:.4f} != saved {ref:.4f}"
            else:
                assert abs(blank.n_blanked - p * n_links) <= 0.01 * n_links, \
                    f"{model_name}: blanked {blank.n_blanked} roads, expected about {p * n_links:.0f}"
                assert blank.max_dev < 1e-4, f"{model_name}: blanked input differs from 'no vehicle' ({blank.max_dev})"
            link_df = link_df.copy()
            link_df["TRIP_ID"] = link_df["TRIP_ID"].astype(str)
            link_df["pos"] = link_df.groupby("TRIP_ID").cumcount()
            blanked = set(blank.blanked_rows)
            is_b = np.array([(t, k) in blanked for t, k in zip(link_df.TRIP_ID, link_df.pos)])
            lt, lp = link_df.true_tt_link.to_numpy(), link_df.pred_tt_link.to_numpy()
            ok = lt >= li.T_MIN_METRIC

            def link_mape(mask):
                mm = mask & ok & (lt > 0)
                return float(np.mean(np.abs(lp[mm] - lt[mm]) / lt[mm])) if mm.any() else float("nan")
            row = dict(city=city, model=model_name, p_blanked=p,
                       route_mape=m["route_mape"], route_mae=m["route_mae"],
                       route_wape=_wape(trip_df.y_pred.to_numpy() - trip_df.y_true.to_numpy(), trip_df.y_true.to_numpy()),
                       link_mape=m["link_mape"], link_wape=_wape(lp - lt, lt),
                       link_mape_tmin_all=link_mape(np.ones_like(is_b)), link_mape_tmin_blanked=link_mape(is_b), link_wape_blanked=_wape((lp - lt)[is_b], lt[is_b]) if is_b.any() else float("nan"),
                       n_links=n_links, n_blanked=int(is_b.sum()), n_routes=len(trip_df),
                       evaluated=time.strftime("%Y-%m-%d %H:%M:%S"))
            rows.append(row)
            ab._append_csv(out_csv, row)
            if trips_csv is not None:
                t = trip_df[["TRIP_ID", "y_true", "y_pred"]].assign(city=city, model=model_name, p_blanked=p)
                t.to_csv(trips_csv, mode="a", header=not trips_csv.exists(), index=False)
            print(f"[robustness] {model_name} p={p:.2f}: route MAPE {row['route_mape']:.4f} WAPE {row['route_wape']:.4f} | "
                  f"link MAPE {row['link_mape']:.4f} | T>=1.5s link MAPE all {row['link_mape_tmin_all']:.4f} "
                  f"blanked {row['link_mape_tmin_blanked']:.4f} ({row['n_blanked']:,} links)")
    finally:
        hook.remove()
        loaders["test"] = original
    return rows


SEEN_BINS = [(0, 0, "0"), (1, 4, "1-4"), (5, 19, "5-19"), (20, None, ">=20")]


def unseen_roads(city, variants):
    import collections
    import pandas as pd
    import tempfile
    from loss_functions import L_MIN_METRIC, T_MIN_METRIC
    folder = main.results_dir(city)
    edges_gdf, df_all = main.load_dataset(city)
    ab._restore()
    li_mod = main._setup_linketa_new(city)
    li_mod.CKPT_DIRNAME = tempfile.mkdtemp(prefix="unseen_")
    core = li_mod.linketa_get_core_components(edges_gdf, df_all)

    seen = collections.Counter()
    for g in core["datasets"]["train"].groups:
        seen.update(g["node_edge_idx"].numpy()[g["route_seq_idx"].numpy()].tolist())
    recs = []
    for g in core["datasets"]["test"].groups:
        edges = g["node_edge_idx"].numpy()[g["route_seq_idx"].numpy()]
        tt = np.nan_to_num(g["seq_true_tt"].numpy())
        ln = np.nan_to_num(g["seq_link_len"].numpy())
        tid = str(int(g["trip_id"]))
        for k, (e, t, l_) in enumerate(zip(edges, tt, ln)):
            recs.append((tid, k, int(e), float(t), float(l_), seen.get(int(e), 0)))
    ref = pd.DataFrame(recs, columns=["TRIP_ID", "pos", "edge", "true_g", "len_m", "n_train"]).set_index(["TRIP_ID", "pos"])
    print(f"[unseen] {city}: {len(ref):,} test link occurrences on {ref.edge.nunique():,} roads; "
          f"training traversed {len(seen):,} roads")

    files = {"full": folder / "eval_outputs" / "eval_links_linketa.csv"}
    files.update({v: folder / "eval_outputs" / f"eval_links_abl_{v}.csv" for v in variants})
    rows = []
    occ = ref.copy()
    for name, path in files.items():
        d = pd.read_csv(path, dtype={"TRIP_ID": str})
        d["pos"] = d.groupby("TRIP_ID").cumcount()
        j = ref.join(d.set_index(["TRIP_ID", "pos"])[["true_tt_link", "pred_tt_link"]], how="inner")
        assert len(j) == len(ref) == len(d), f"{name}: rows {len(d)} / joined {len(j)} / test groups {len(ref)}"
        bad = (np.abs(j.true_tt_link - j.true_g) > 1e-3).sum()
        assert bad == 0, f"{name}: {bad} rows whose true link time differs from the test groups"
        occ[f"pred_{name}"] = j.pred_tt_link.reindex(occ.index)
        for lo, hi, lab in SEEN_BINS:
            m = (j.n_train >= lo) & (True if hi is None else (j.n_train <= hi))
            s = j[m]
            err = s.pred_tt_link - s.true_tt_link
            mm = (s.true_tt_link >= T_MIN_METRIC) & (s.len_m >= L_MIN_METRIC) & (s.true_tt_link > 0)
            row = dict(city=city, model=name, train_traversals=lab, occurrences=int(len(s)),
                       roads=int(s.edge.nunique()),
                       link_mae=float(err.abs().mean()) if len(s) else float("nan"),
                       link_wape=_wape(err.to_numpy(), s.true_tt_link.to_numpy()) if len(s) else float("nan"),
                       link_mape=float((err[mm].abs() / s.true_tt_link[mm]).mean()) if mm.any() else float("nan"),
                       mape_occurrences=int(mm.sum()))
            rows.append(row)
            print(f"[unseen] {name:10s} trained x{lab:>5s}: {row['occurrences']:8,} occ ({row['roads']:,} roads) | "
                  f"MAE {row['link_mae']:6.2f}s WAPE {row['link_wape']:.4f} MAPE {row['link_mape']:.4f}")
    occ.reset_index().rename(columns={"true_g": "true_tt_link"}).assign(city=city).to_csv(
        folder / "eval_outputs" / "robustness_unseen_roads_occ.csv", index=False)
    out = pd.DataFrame(rows)
    out.to_csv(folder / "eval_outputs" / "robustness_unseen_roads.csv", index=False)
    print(f"[unseen] saved {folder / 'eval_outputs' / 'robustness_unseen_roads.csv'}")
    return out


DEPTHS = (None, 0, 1, 2, 3, 4, 5, 6)
FAR = 99


def _depth_label(d):
    return "none" if d is None else str(d)


def _bfs(adj, seed, cap):
    dist, frontier = {seed: 0}, [seed]
    for d in range(1, cap + 1):
        nxt = []
        for u in frontier:
            for v in adj.get(u, ()):
                if v not in dist:
                    dist[v] = d
                    nxt.append(v)
        frontier = nxt
    return dist


class _AreaHider:

    def __init__(self, core, sink, dist_by_trip, n_nodes_by_trip):
        dev = core["device"]
        dyn = li.DYN_IDX.to(dev)
        mean = core["scalers"]["mean"].to(dev).float()
        std = core["scalers"]["std"].to(dev).float()
        self.dyn, self.blank_value = dyn, ((0.0 - mean[dyn]) / std[dyn])
        self.sink, self.dist, self.n_nodes = sink, dist_by_trip, n_nodes_by_trip
        self.depth, self.n_hidden_nodes, self.n_hidden_route, self.max_dev = None, 0, 0, 0.0

    def __call__(self, module, args):
        if self.depth is None:
            return None
        x, route_flag = args[0].clone(), args[2].bool().cpu()
        nei = self.sink["node_edge_idx"]
        sel = torch.zeros(x.shape[:2], dtype=torch.bool)
        for b, tid in enumerate(self.sink["trip_ids"]):
            tid = int(tid)
            dist = self.dist[tid]
            for j in range(self.n_nodes[tid]):
                dd = dist.get(int(nei[b, j]))
                if dd is not None and dd <= self.depth:
                    sel[b, j] = True
            L = int(route_flag[b].sum())
            self.n_hidden_route += int(sel[b, :L].sum())
        if sel.any():
            sel = sel.to(x.device)
            xb = x[sel]
            xb[:, self.dyn] = self.blank_value.to(xb.dtype)
            x[sel] = xb
            self.max_dev = max(self.max_dev, float((x[sel][:, self.dyn] - self.blank_value).abs().max()))
            self.n_hidden_nodes += int(sel.sum())
        return (x,) + tuple(args[1:])


def _disruption_reference(core):
    import collections
    import pandas as pd
    ds = core["datasets"]["test"]
    adj = collections.defaultdict(set)
    rel = core["graph"]["rel_edges"]
    for m in (li._build_src2dst(rel[1]), li._build_src2dst(rel[2])):
        for s, dsts in m.items():
            for t in dsts:
                adj[int(s)].add(int(t))
                adj[int(t)].add(int(s))
    if not adj:
        raise AssertionError("empty road graph: downstream/upstream relations missing")
    cap = max(d for d in DEPTHS if d is not None)
    dist_by_trip, n_nodes_by_trip, recs = {}, {}, []
    for g in ds.groups:
        tid = int(g["trip_id"])
        edges = g["node_edge_idx"].numpy()[g["route_seq_idx"].numpy()]
        seed_pos = zlib.crc32(str(tid).encode()) % len(edges)
        dist = _bfs(adj, int(edges[seed_pos]), cap)
        dist_by_trip[tid], n_nodes_by_trip[tid] = dist, int(g["node_edge_idx"].numel())
        tt = np.nan_to_num(g["seq_true_tt"].numpy())
        ln = np.nan_to_num(g["seq_link_len"].numpy())
        for k, e in enumerate(edges):
            recs.append((str(tid), k, int(e), dist.get(int(e), FAR), k == seed_pos, float(ln[k]), float(tt[k])))
    ref = pd.DataFrame(recs, columns=["TRIP_ID", "pos", "edge", "dist_from_seed", "is_seed", "len_m", "true_g"])
    return adj, dist_by_trip, n_nodes_by_trip, ref.set_index(["TRIP_ID", "pos"])


def disruption(city, model_name, core, ref_mape, out_dir, depths=DEPTHS):
    from loss_functions import L_MIN_METRIC, T_MIN_METRIC
    adj, dist_by_trip, n_nodes_by_trip, ref = _disruption_reference(core)
    seeds = ref[ref.is_seed]
    print(f"[disruption] {city} {model_name}: {len(seeds):,} trips / seeds, {len(ref):,} route links")
    sink = {}
    hider = _AreaHider(core, sink, dist_by_trip, n_nodes_by_trip)
    loaders = core["loaders"]
    original = loaders["test"]

    class _B(_Batches):
        def __iter__(self):
            for batch in self.loader:
                self.sink["trip_ids"] = batch["trip_ids"]
                self.sink["node_edge_idx"] = batch["node_edge_idx"]
                yield batch
    loaders["test"] = _B(original, sink)
    hook = core["model"].register_forward_pre_hook(hider)
    try:
        for d in depths:
            hider.depth, hider.n_hidden_nodes, hider.n_hidden_route, hider.max_dev = d, 0, 0, 0.0
            m, trip_df, link_df = core["eval_linketa"](core, label=f"LinkETA_{model_name}")
            lab = _depth_label(d)
            if d is None:
                assert abs(m["route_mape"] - ref_mape) < 1e-3, \
                    f"{model_name}: reference route MAPE {m['route_mape']:.4f} != saved {ref_mape:.4f}"
                hidden = np.zeros(len(ref), dtype=bool)
            else:
                assert hider.max_dev < 1e-4, f"{model_name}: hidden input differs from 'no vehicle' ({hider.max_dev})"
                hidden = (ref.dist_from_seed <= d).to_numpy()
                assert hider.n_hidden_route == int(hidden.sum()), \
                    f"{model_name} d={d}: hook hid {hider.n_hidden_route} route roads, BFS says {int(hidden.sum())}"
            ld = link_df.copy()
            ld["TRIP_ID"] = ld["TRIP_ID"].astype(str)
            ld["pos"] = ld.groupby("TRIP_ID").cumcount()
            j = ref.join(ld.set_index(["TRIP_ID", "pos"])[["true_tt_link", "pred_tt_link"]], how="inner")
            assert len(j) == len(ref) == len(ld), f"{model_name}: rows {len(ld)} / joined {len(j)} / test {len(ref)}"
            bad = int((np.abs(j.true_tt_link - j.true_g) > 1e-3).sum())
            assert bad == 0, f"{model_name}: {bad} rows whose true link time differs from the test groups"
            j = j.assign(hidden=hidden)

            def stats(s):
                err = (s.pred_tt_link - s.true_tt_link).to_numpy()
                mm = ((s.true_tt_link >= T_MIN_METRIC) & (s.len_m >= L_MIN_METRIC) & (s.true_tt_link > 0)).to_numpy()
                mape = float(np.mean(np.abs(err[mm]) / s.true_tt_link.to_numpy()[mm])) if mm.any() else float("nan")
                return (float(np.abs(err).mean()) if len(s) else float("nan"),
                        _wape(err, s.true_tt_link.to_numpy()) if len(s) else float("nan"), mape, int(mm.sum()))
            s_mae, s_wape, s_mape, s_n = stats(j[j.is_seed])
            h_mae, h_wape, h_mape, h_n = stats(j[j.hidden]) if j.hidden.any() else (float("nan"),) * 3 + (0,)
            row = dict(city=city, model=model_name, depth=lab,
                       route_mape=m["route_mape"],
                       route_wape=_wape(trip_df.y_pred.to_numpy() - trip_df.y_true.to_numpy(), trip_df.y_true.to_numpy()),
                       seed_link_mae=s_mae, seed_link_wape=s_wape, seed_link_mape=s_mape, seed_mape_n=s_n,
                       hidden_link_mae=h_mae, hidden_link_wape=h_wape, hidden_link_mape=h_mape, hidden_mape_n=h_n,
                       n_hidden_route_links=int(j.hidden.sum()), n_hidden_nodes=hider.n_hidden_nodes,
                       n_links=len(j), n_routes=len(trip_df), evaluated=time.strftime("%Y-%m-%d %H:%M:%S"))
            ab._append_csv(out_dir / "robustness_disruption.csv", row)
            out_links = j.reset_index()[["TRIP_ID", "pos", "edge", "dist_from_seed", "is_seed", "hidden", "len_m",
                                         "true_tt_link", "pred_tt_link"]].assign(city=city, model=model_name, depth=lab)
            p_links = out_dir / "robustness_disruption_links.csv"
            out_links.to_csv(p_links, mode="a", header=not p_links.exists(), index=False)
            p_trips = out_dir / "robustness_disruption_trips.csv"
            trip_df[["TRIP_ID", "y_true", "y_pred"]].assign(city=city, model=model_name, depth=lab).to_csv(
                p_trips, mode="a", header=not p_trips.exists(), index=False)
            print(f"[disruption] {model_name:9s} depth {lab:>4s}: route MAPE {row['route_mape']:.4f} | seed link "
                  f"MAE {s_mae:6.2f}s WAPE {s_wape:.4f} MAPE {s_mape:.4f} | hidden route links "
                  f"{row['n_hidden_route_links']:,} WAPE {h_wape:.4f} | hidden nodes {hider.n_hidden_nodes:,}")
    finally:
        hook.remove()
        loaders["test"] = original


BLOCK_SPACING, BLOCK_OFFSET = 20, 10


def block_positions(L, depth, spacing=BLOCK_SPACING, offset=BLOCK_OFFSET):
    hid = np.zeros(L, dtype=bool)
    for c in range(offset, L, spacing):
        hid[max(0, c - depth):min(L, c + depth + 1)] = True
    return hid


def hidden_nodes(group, depth, version):
    nodes = group["node_edge_idx"].numpy()
    rf = group["route_flag"].numpy().astype(bool)
    rpos = group["route_seq_idx"].numpy()
    hid_pos = block_positions(len(rpos), depth)
    hid_roads = set(nodes[rpos[hid_pos]].tolist())
    mask = np.isin(nodes, list(hid_roads)) if hid_roads else np.zeros(len(nodes), dtype=bool)
    if version == "B":
        par = group["neigh_parent"].numpy()
        mask |= (~rf) & hid_pos[np.clip(par, 0, len(rpos) - 1)]
    return mask


class _NodeHider:

    def __init__(self, core, sink, groups_by_trip):
        dev = core["device"]
        dyn = li.DYN_IDX.to(dev)
        mean = core["scalers"]["mean"].to(dev).float()
        std = core["scalers"]["std"].to(dev).float()
        self.dyn, self.blank_value = dyn, ((0.0 - mean[dyn]) / std[dyn])
        self.sink, self.groups = sink, groups_by_trip
        self.level, self.n_hidden, self.max_dev = None, 0, 0.0

    def __call__(self, module, args):
        if self.level is None:
            return None
        version, depth = self.level
        x = args[0].clone()
        sel = torch.zeros(x.shape[:2], dtype=torch.bool)
        for b, tid in enumerate(self.sink["trip_ids"]):
            m = hidden_nodes(self.groups[int(tid)], depth, version)
            sel[b, :len(m)] = torch.from_numpy(m)
        if sel.any():
            sel = sel.to(x.device)
            xb = x[sel]
            xb[:, self.dyn] = self.blank_value.to(xb.dtype)
            x[sel] = xb
            self.max_dev = max(self.max_dev, float((x[sel][:, self.dyn] - self.blank_value).abs().max()))
            self.n_hidden += int(sel.sum())
        return (x,) + tuple(args[1:])


BLOCK_LEVELS = (None,) + tuple(("A", d) for d in range(10)) + tuple(("B", d) for d in range(10))


def _level_label(level):
    return "none" if level is None else f"{level[0]}{level[1]}"


def hidden_blocks(city, model_name, core, ref_mape, out_dir, levels=BLOCK_LEVELS):
    from loss_functions import L_MIN_METRIC, T_MIN_METRIC
    import pandas as pd
    ds = core["datasets"]["test"]
    cap = int(li.CFG.get("max_nodes_local", 360))
    groups = {int(g["trip_id"]): g for g in ds.groups}
    recs = []
    for tid, g in groups.items():
        L = len(g["route_seq_idx"])
        below = int(g["node_edge_idx"].numel()) < cap
        tt = np.nan_to_num(g["seq_true_tt"].numpy())
        ln = np.nan_to_num(g["seq_link_len"].numpy())
        centres = set(range(BLOCK_OFFSET, L, BLOCK_SPACING))
        for k in range(L):
            recs.append((str(tid), k, below, k in centres, float(ln[k]), float(tt[k])))
    ref = pd.DataFrame(recs, columns=["TRIP_ID", "pos", "below_cap", "is_centre", "len_m", "true_g"])
    ref = ref.set_index(["TRIP_ID", "pos"])
    print(f"[blocks] {city} {model_name}: {len(groups):,} trips ({ref.groupby(level=0).below_cap.first().sum():,} below "
          f"the {cap}-node cap), {len(ref):,} route links")
    sink = {}
    hider = _NodeHider(core, sink, groups)
    loaders = core["loaders"]
    original = loaders["test"]

    class _B(_Batches):
        def __iter__(self):
            for batch in self.loader:
                self.sink["trip_ids"] = batch["trip_ids"]
                yield batch
    loaders["test"] = _B(original, sink)
    hook = core["model"].register_forward_pre_hook(hider)
    try:
        for level in levels:
            hider.level, hider.n_hidden, hider.max_dev = level, 0, 0.0
            m, trip_df, link_df = core["eval_linketa"](core, label=f"LinkETA_{model_name}")
            lab = _level_label(level)
            if level is None:
                assert abs(m["route_mape"] - ref_mape) < 1e-3, \
                    f"{model_name}: reference route MAPE {m['route_mape']:.4f} != saved {ref_mape:.4f}"
                hid = np.zeros(len(ref), dtype=bool)
            else:
                version, depth = level
                expected = sum(int(hidden_nodes(g, depth, version).sum()) for g in groups.values())
                assert hider.n_hidden == expected, f"{lab}: hook hid {hider.n_hidden} nodes, rules give {expected}"
                assert hider.max_dev < 1e-4, f"{lab}: hidden input differs from 'no vehicle' ({hider.max_dev})"
                hid = np.concatenate([block_positions(len(groups[int(t)]["route_seq_idx"]), depth)
                                      for t in ref.index.get_level_values(0).unique()])
            ld = link_df.copy()
            ld["TRIP_ID"] = ld["TRIP_ID"].astype(str)
            ld["pos"] = ld.groupby("TRIP_ID").cumcount()
            j = ref.assign(hidden=hid).join(ld.set_index(["TRIP_ID", "pos"])[["true_tt_link", "pred_tt_link"]],
                                            how="inner")
            assert len(j) == len(ref) == len(ld), f"{lab}: rows {len(ld)} / joined {len(j)} / test {len(ref)}"
            assert int((np.abs(j.true_tt_link - j.true_g) > 1e-3).sum()) == 0, f"{lab}: true link times differ"

            def stats(s):
                err = (s.pred_tt_link - s.true_tt_link).to_numpy()
                mm = ((s.true_tt_link >= T_MIN_METRIC) & (s.len_m >= L_MIN_METRIC) & (s.true_tt_link > 0)).to_numpy()
                return (float(np.abs(err).mean()) if len(s) else float("nan"),
                        _wape(err, s.true_tt_link.to_numpy()) if len(s) else float("nan"),
                        float(np.mean(np.abs(err[mm]) / s.true_tt_link.to_numpy()[mm])) if mm.any() else float("nan"))
            b = j[j.below_cap]
            h_mae, h_wape, h_mape = stats(b[b.hidden]) if b.hidden.any() else (float("nan"),) * 3
            c_mae, c_wape, c_mape = stats(b[b.is_centre])
            td = trip_df.assign(TRIP_ID=trip_df.TRIP_ID.astype(str))
            below_trips = set(b.index.get_level_values(0))
            hit_trips = set(b[b.is_centre].index.get_level_values(0))
            t_hit = td[td.TRIP_ID.isin(hit_trips) & (td.y_true > 0)]
            row = dict(city=city, model=model_name, level=lab,
                       version="" if level is None else level[0], depth="" if level is None else level[1],
                       route_mape_all=m["route_mape"],
                       route_mape_hit=float((np.abs(t_hit.y_pred - t_hit.y_true) / t_hit.y_true).mean()),
                       hidden_link_mae=h_mae, hidden_link_wape=h_wape, hidden_link_mape=h_mape,
                       centre_link_mae=c_mae, centre_link_wape=c_wape, centre_link_mape=c_mape,
                       n_hidden_route_links=int(b.hidden.sum()), n_centre_links=int(b.is_centre.sum()),
                       n_hidden_nodes=hider.n_hidden, n_trips_below_cap=len(below_trips), n_trips_hit=len(hit_trips),
                       evaluated=time.strftime("%Y-%m-%d %H:%M:%S"))
            ab._append_csv(out_dir / "robustness_blocks.csv", row)
            p_links = out_dir / "robustness_blocks_links.csv"
            j.reset_index()[["TRIP_ID", "pos", "below_cap", "is_centre", "hidden", "len_m", "true_tt_link",
                             "pred_tt_link"]].assign(city=city, model=model_name, level=lab).to_csv(
                p_links, mode="a", header=not p_links.exists(), index=False)
            p_trips = out_dir / "robustness_blocks_trips.csv"
            td[["TRIP_ID", "y_true", "y_pred"]].assign(city=city, model=model_name, level=lab).to_csv(
                p_trips, mode="a", header=not p_trips.exists(), index=False)
            print(f"[blocks] {model_name} {lab:>4s}: route MAPE hit trips {row['route_mape_hit']:.4f} | hidden links "
                  f"{row['n_hidden_route_links']:,} MAE {h_mae:6.2f}s | centre MAE {c_mae:6.2f}s | "
                  f"hidden nodes {hider.n_hidden:,}")
    finally:
        hook.remove()
        loaders["test"] = original


RELATION_LEVELS = ("none", "no_rel_high_order", "no_flow", "no_turn", "no_graph")


class _MaskDropper:

    def __init__(self):
        self.drop, self.left = set(), 0

    def __call__(self, module, args):
        if not self.drop:
            return None
        masks = list(args[3])
        for r in self.drop:
            masks[r] = torch.zeros_like(masks[r])
            self.left += int(masks[r].sum())
        return tuple(args[:3]) + (masks,) + tuple(args[4:])


def relations(city, core, ref_mape, out_dir):
    dropper = _MaskDropper()
    hook = core["model"].register_forward_pre_hook(dropper)
    rows, ape = [], {}
    p_trips = out_dir / "robustness_relations_trips.csv"
    try:
        for lev in RELATION_LEVELS:
            dropper.drop = set() if lev == "none" else {ab.REL_NAMES.index(n) for n in ab.VARIANTS[lev]["drop"]}
            dropper.left = 0
            m, trip_df, _ = core["eval_linketa"](core, label=f"LinkETA_{lev}")
            if lev == "none":
                assert abs(m["route_mape"] - ref_mape) < 1e-3, \
                    f"none: route MAPE {m['route_mape']:.4f} != saved {ref_mape:.4f}"
            assert dropper.left == 0, f"{lev}: {dropper.left} mask entries left in the dropped relations"
            t = trip_df[trip_df.y_true > 0].set_index("TRIP_ID")
            ape[lev] = (np.abs(t.y_pred - t.y_true) / t.y_true).sort_index()
            rows.append(dict(city=city, level=lev, route_mape=m["route_mape"], route_mae=m["route_mae"],
                             link_mape=m["link_mape"], n_routes=len(trip_df)))
            trip_df[["TRIP_ID", "y_true", "y_pred"]].assign(city=city, level=lev).to_csv(
                p_trips, mode="a", header=not p_trips.exists(), index=False)
    finally:
        hook.remove()
    base = ape["none"]
    idx = np.random.default_rng(0).integers(0, len(base), (2000, len(base)))
    b = base.to_numpy()
    for row in rows:
        a = ape[row["level"]].reindex(base.index).to_numpy()
        boot = 100 * (a[idx].mean(1) - b[idx].mean(1))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        row.update(route_mape_change_pp=float(100 * (a.mean() - b.mean())), ci_lo=float(lo), ci_hi=float(hi),
                   evaluated=time.strftime("%Y-%m-%d %H:%M:%S"))
        ab._append_csv(out_dir / "robustness_relations.csv", row)
        print(f"[relations] {city} {row['level']:>17s}: route MAPE {100 * row['route_mape']:.2f}% | link MAPE "
              f"{100 * row['link_mape']:.2f}% | change {row['route_mape_change_pp']:+.2f} pp [{lo:+.2f}, {hi:+.2f}]")


ZONE_MODELS = {"full": "linketa", "zone_out": "abl_zone_out", "zone_out_control": "abl_zone_out_control"}


def zone(city, core, out_dir):
    import pandas as pd
    from loss_functions import L_MIN_METRIC, T_MIN_METRIC
    roads = ab._zone_roads(ab.VARIANTS["zone_out"]["zone"])
    ev = core["graph"]["edges_vff"]
    idc = li.guess_edges_id_col(ev, li.CFG["way_id_col_edges_candidates"])
    in_zone = dict(zip(ev["edge_idx"].astype(int), ev[idc].astype(str).isin(roads)))
    recs = []
    for g in core["datasets"]["test"].groups:
        edges = g["node_edge_idx"].numpy()[g["route_seq_idx"].numpy()]
        tt = np.nan_to_num(g["seq_true_tt"].numpy())
        ln = np.nan_to_num(g["seq_link_len"].numpy())
        tid = str(int(g["trip_id"]))
        for k, (e, t, l_) in enumerate(zip(edges, tt, ln)):
            recs.append((tid, k, bool(in_zone.get(int(e), False)), float(t), float(l_)))
    ref = pd.DataFrame(recs, columns=["TRIP_ID", "pos", "zone", "true_g", "len_m"]).set_index(["TRIP_ID", "pos"])
    share = ref.groupby(level=0).zone.mean()
    subsets = {"through": share.index[share > 0], "mostly_inside": share.index[share >= 0.5],
               "outside": share.index[share == 0]}
    print(f"[zone] {city}: {len(roads):,} zone roads; test trips {len(share):,}: through {len(subsets['through']):,}, "
          f"mostly inside {len(subsets['mostly_inside']):,}, outside {len(subsets['outside']):,}; "
          f"zone link rows {int(ref.zone.sum()):,}")
    link, route = {}, {}
    for name, tag in ZONE_MODELS.items():
        d = pd.read_csv(out_dir / f"eval_links_{tag}.csv", dtype={"TRIP_ID": str})
        d["pos"] = d.groupby("TRIP_ID").cumcount()
        j = ref.join(d.set_index(["TRIP_ID", "pos"])[["true_tt_link", "pred_tt_link"]], how="inner")
        assert len(j) == len(ref) == len(d), f"{name}: rows {len(d)} / joined {len(j)} / test groups {len(ref)}"
        bad = int((np.abs(j.true_tt_link - j.true_g) > 1e-3).sum())
        assert bad == 0, f"{name}: {bad} rows whose true link time differs from the test groups"
        z = j[j.zone]
        mm = (z.true_tt_link >= T_MIN_METRIC) & (z.len_m >= L_MIN_METRIC) & (z.true_tt_link > 0)
        err = (z.pred_tt_link - z.true_tt_link).abs()
        link[name] = pd.DataFrame({"ape": (err / z.true_tt_link).where(mm, 0.0), "n": mm.astype(int),
                                   "ae": err, "nl": 1}).groupby(level=0).sum()
        t = pd.read_csv(out_dir / f"eval_trips_{tag}.csv", dtype={"TRIP_ID": str}).set_index("TRIP_ID")
        t = t[t.y_true > 0]
        route[name] = np.abs(t.y_pred - t.y_true) / t.y_true
    rng = np.random.default_rng(0)
    rows = []

    def add(subset, metric, ids, value_fn):
        idx = rng.integers(0, len(ids), (2000, len(ids)))
        vals = {n: value_fn(n, ids, None) for n in ZONE_MODELS}
        boots = {n: value_fn(n, ids, idx) for n in ZONE_MODELS}
        for n in ZONE_MODELS:
            row = dict(city=city, subset=subset, metric=metric, model=n, value=float(vals[n]), n_trips=len(ids))
            for other in ("full", "zone_out_control"):
                if n == other or (other == "zone_out_control" and n != "zone_out"):
                    continue
                lo, hi = np.percentile(100 * (boots[n] - boots[other]), [2.5, 97.5])
                row[f"change_vs_{other}_pp"] = float(100 * (vals[n] - vals[other]))
                row[f"ci_vs_{other}"] = f"[{lo:+.2f}, {hi:+.2f}]"
            rows.append(row)
            ab._append_csv(out_dir / "robustness_zone.csv", row)
            print(f"[zone] {subset:13s} {metric:9s} {n:16s} {100 * vals[n]:6.2f}%" +
                  "".join(f" | vs {o}: {row[f'change_vs_{o}_pp']:+.2f} pp {row[f'ci_vs_{o}']}"
                          for o in ("full", "zone_out_control") if f"change_vs_{o}_pp" in row))

    def link_mape(n, ids, idx):
        p = link[n].reindex(ids).fillna(0.0)
        a, c = p.ape.to_numpy(), p.n.to_numpy()
        return a.sum() / c.sum() if idx is None else a[idx].sum(1) / c[idx].sum(1)

    def route_mape(n, ids, idx):
        a = route[n].reindex(ids).to_numpy()
        return a.mean() if idx is None else a[idx].mean(1)
    add("zone_links", "link_mape", np.array(subsets["through"]), link_mape)
    for s in ("through", "mostly_inside", "outside"):
        add(s, "route_mape", np.array(subsets[s]), route_mape)
    return rows


def _keep_old(path):
    if path.exists():
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(path.stat().st_mtime))
        path.rename(path.with_name(f"{path.stem}_preserved_{stamp}{path.suffix}"))


def parse_args():
    ap = argparse.ArgumentParser(description="LinkETA robustness tests (missing own data; unseen roads; disruption).")
    ap.add_argument("--datasets", default="porto")
    ap.add_argument("--models", default="full,no_graph", help="'full' and/or ablations.py variant names")
    ap.add_argument("--test", default="missing_data",
                    choices=("missing_data", "unseen_roads", "disruption", "blocks", "relations", "zone"),
                    help="[2026-10-09] which test to run (default: missing_data, as before)")
    ap.add_argument("--p-grid", action="store_true",
                    help="[2026-10-09] missing_data on p = 0, 0.05, ..., 1 (curve); writes "
                         "robustness_missing_data_curve.csv and robustness_missing_data_trips.csv")
    ap.add_argument("--levels", default=None,
                    help="[2026-10-09] prototype only: comma list of p levels (missing_data) or depths "
                         "('none' or integers, disruption) instead of the defaults")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    names = [m.strip() for m in args.models.split(",") if m.strip()]
    for city in main._resolve_datasets(args.datasets):
        if args.test == "unseen_roads":
            unseen_roads(city, [n for n in names if n != "full"])
            continue
        edges_gdf, df_all = main.load_dataset(city)
        out_dir = main.results_dir(city) / "eval_outputs"
        if args.test == "relations":
            for f in ("robustness_relations.csv", "robustness_relations_trips.csv"):
                _keep_old(out_dir / f)
            core, ref = build(city, "full", edges_gdf, df_all)
            relations(city, core, ref, out_dir)
            continue
        if args.test == "zone":
            _keep_old(out_dir / "robustness_zone.csv")
            core, ref = build(city, "full", edges_gdf, df_all)
            zone(city, core, out_dir)
            continue
        if args.test == "blocks":
            levels = BLOCK_LEVELS if args.levels is None else tuple(
                None if s.strip() == "none" else (s.strip()[0], int(s.strip()[1:])) for s in args.levels.split(","))
            for f in ("robustness_blocks.csv", "robustness_blocks_links.csv", "robustness_blocks_trips.csv"):
                _keep_old(out_dir / f)
            for name in names:
                core, ref = build(city, name, edges_gdf, df_all)
                hidden_blocks(city, name, core, ref, out_dir, levels)
            continue
        if args.test == "disruption":
            depths = DEPTHS if args.levels is None else tuple(
                None if s.strip() == "none" else int(s) for s in args.levels.split(","))
            for f in ("robustness_disruption.csv", "robustness_disruption_links.csv", "robustness_disruption_trips.csv"):
                _keep_old(out_dir / f)
            for name in names:
                core, ref = build(city, name, edges_gdf, df_all)
                disruption(city, name, core, ref, out_dir, depths)
            continue
        out_csv = out_dir / "robustness_missing_data.csv"
        if args.p_grid:
            levels = tuple(round(0.05 * i, 2) for i in range(21)) if args.levels is None else tuple(
                float(s) for s in args.levels.split(","))
            out_csv, trips_csv = out_dir / "robustness_missing_data_curve.csv", out_dir / "robustness_missing_data_trips.csv"
            _keep_old(out_csv)
            _keep_old(trips_csv)
        for name in names:
            core, ref = build(city, name, edges_gdf, df_all)
            if args.p_grid:
                evaluate(city, name, core, ref, out_csv, levels=levels, trips_csv=trips_csv)
            else:
                evaluate(city, name, core, ref, out_csv)

