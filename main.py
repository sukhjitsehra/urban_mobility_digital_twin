import argparse
import contextlib
import json
import os
import runpy
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import torch

import importlib

for _sub in ("models", "evaluation", "preprocess"):
    _p = str(Path(__file__).resolve().parent / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)


class _LazyModule:
    def __init__(self, name):
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_mod", None)

    def _load(self):
        if self._mod is None:
            object.__setattr__(self, "_mod", importlib.import_module(self._name))
        return self._mod

    def __getattr__(self, attr):
        return getattr(self._load(), attr)

    def __setattr__(self, attr, value):
        setattr(self._load(), attr, value)


du = _LazyModule("DuETA")
con = _LazyModule("ConSTGAT")
sthr = _LazyModule("STHR")
tfeta = _LazyModule("TransETA")
dreta = _LazyModule("DeeprETA")
pbeta = _LazyModule("ProbETA")
gspinn = _LazyModule("GSPINN")
_evaluator = _LazyModule("evaluator")


def eval_linketa(*a, **k):  return _evaluator.eval_linketa(*a, **k)
def eval_dueta(*a, **k):    return _evaluator.eval_dueta(*a, **k)
def eval_constgat(*a, **k): return _evaluator.eval_constgat(*a, **k)
def eval_sthr(*a, **k):     return sthr.eval_sthr(*a, **k)
def eval_transeta(*a, **k): return tfeta.eval_transeta(*a, **k)
def eval_deepreta(*a, **k): return dreta.eval_deepreta(*a, **k)
def eval_probeta(*a, **k):  return pbeta.eval_probeta(*a, **k)
def eval_gspinn(*a, **k):   return gspinn.eval_gspinn(*a, **k)


BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR / "data"
RESULTS_ROOT = BASE_DIR / "results"


CANON_MARKER = "SHA256SUMS"


def _find_input_data_root() -> Path:
    env = os.environ.get("ETA_DATA_ROOT")
    if env:
        return Path(env).resolve()
    for parent in BASE_DIR.parents:
        if (parent / "data" / CANON_MARKER).is_file():
            return parent / "data"
    raise FileNotFoundError(
        f"Canonical data folder not found: no parent of {BASE_DIR} contains data/{CANON_MARKER}. "
        f"Set ETA_DATA_ROOT to the folder that holds <city>/df_all_<city>.csv.")

CITIES = ["Porto", "San Francisco", "Chengdu", "Daegu"]
EXTRA_CITIES = ["Ann Arbor", "SF Bus"]

CITY_FOLDERS = {
    "Porto":         "porto",
    "San Francisco": "sf",
    "Chengdu":       "chengdu",
    "Daegu":         "daegu",
    "Ann Arbor":     "annarbor",
    "SF Bus":        "sfbus",
}


def city_folder_name(city: str) -> str:
    return CITY_FOLDERS.get(city, city)


def ckpt_dirname(model: str, city: str, suffix: str = "") -> str:
    return f"ckpts_{model}_{city_folder_name(city)}{suffix}"


def _setup_linketa_new(city: str):
    import LinkETA as li
    folder = city_dir(city)
    folder.mkdir(parents=True, exist_ok=True)
    li.CKPT_DIRNAME = str(results_dir(city) / ckpt_dirname("LinkETA", city, suffix=CKPT_SUFFIX))
    if R_MAX_OVERRIDE is not None:
        li.R_MAX = float(R_MAX_OVERRIDE)
    li.CACHE_DIR = None if NO_BUILD_CACHE else str(folder / "cache_LinkETA")
    return li


def city_dir(city: str) -> Path:
    return DATA_ROOT / city_folder_name(city)


def results_dir(city: str) -> Path:
    return RESULTS_ROOT / city_folder_name(city)


def city_input_files(city: str):
    folder = city_dir(city)
    code = city_folder_name(city)
    _v = os.environ.get("DF_ALL_VARIANT", "").strip()
    trips = folder / (f"df_all_{code}_{_v}.csv" if _v else f"df_all_{code}.csv")
    edges = folder / f"edges_{code}.geojson"
    for p in (trips, edges):
        if not p.is_file():
            raise FileNotFoundError(f"Input file for '{city}' not found: {p}")
    return trips, edges


def _discover_edges(folder: Path) -> Path:
    for cand in [folder / "edges.geojson", *sorted(folder.glob("*edges*.geojson")), *sorted(folder.glob("*.geojson"))]:
        if cand.exists():
            return cand
    raise FileNotFoundError(f"No .geojson edges file found in {folder}")


def _discover_trips(folder: Path) -> Path:
    preferred = folder / "trips.csv"
    if preferred.exists():
        return preferred
    csvs = sorted(folder.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No .csv trips file found in {folder}")
    return csvs[0]


def load_dataset(city: str, max_trips: int = None):
    trips_path, edges_path = city_input_files(city)
    print(f"[DATA] {city}")
    print(f"       edges: {edges_path}")
    print(f"       trips: {trips_path}")
    edges_gdf = gpd.read_file(edges_path)
    df_all = pd.read_csv(trips_path)
    if not pd.api.types.is_integer_dtype(df_all["TRIP_ID"]):
        codes, uniq = pd.factorize(df_all["TRIP_ID"].astype(str), sort=True)
        map_path = trips_path.with_name(trips_path.stem + "_trip_id_map.csv")
        pd.DataFrame({"TRIP_ID": range(len(uniq)), "TRIP_ID_source": uniq}).to_csv(map_path, index=False)
        df_all["TRIP_ID"] = codes.astype("int64")
        print(f"[DATA] text TRIP_IDs -> integer codes ({len(uniq)} trips); table: {map_path}")
    if max_trips and max_trips > 0:
        n_total = df_all["TRIP_ID"].nunique()
        if max_trips < n_total:
            sampled = df_all["TRIP_ID"].drop_duplicates().sample(
                n=max_trips, random_state=42)
            df_all = df_all[df_all["TRIP_ID"].isin(sampled)].reset_index(drop=True)
            print(f"[DEBUG] sampled {max_trips} of {n_total} trips (random, seed=42)")
        else:
            print(f"[DEBUG] --max-trips={max_trips} >= total trips ({n_total}); using full set")
    sp = trips_path.with_name(f"speeds_{city_folder_name(city)}.parquet")
    if not sp.is_file():
        raise FileNotFoundError(f"Speed records for '{city}' not found: {sp} (written by run_map_matching.py)")
    import map_matching as mm
    import LinkETA as li
    hist = mm.SpeedHistory(pd.read_parquet(sp))
    li.SPEED_HISTORY = hist
    con.SPEED_HISTORY = hist
    du.SPEED_HISTORY = hist
    sthr.SPEED_HISTORY = hist
    print(f"       speeds: {sp} ({len(hist.T):,} records)")
    return edges_gdf, df_all


def check_model_inputs(city, model, core):
    S = ["median_speed", "mean_speed", "min_speed", "max_speed", "count_vehicle"]
    B, NS, N = 600, 12, 60
    close = lambda a, b: np.allclose(np.asarray(a, float), np.asarray(b, float), rtol=1e-5, atol=1e-4)
    trips_path, _ = city_input_files(city)
    df = pd.read_csv(trips_path)
    sp = pd.read_parquet(trips_path.with_name(f"speeds_{city_folder_name(city)}.parquet"))
    t = sp.timestamp.astype("datetime64[ns]").astype("int64").to_numpy() / 1e9
    avail_ts = sp.avail_ts.to_numpy(float)
    TR, V, gi = sp.trip_id.to_numpy().astype(np.int64), sp.speed_m_s.to_numpy(float), sp.groupby("way_id").indices

    def brute(w, end, tr):
        out = np.zeros((NS, 5))
        ix = gi.get(int(w))
        if ix is None:
            return out
        tw, vw, rw = t[ix], V[ix], TR[ix]
        for k in range(NS):
            m = (tw >= end - (k + 1) * B) & (tw < end - k * B) & (rw != tr) & (avail_ts[ix] < end)
            if m.any():
                out[k] = (np.median(vw[m]), vw[m].mean(), vw[m].min(), vw[m].max(), len(np.unique(rw[m])))
        return out

    d = df.sort_values(["TRIP_ID", "enter_ts"], kind="mergesort")
    own = {}
    Hm = d[[f"{s}_{k}" for k in range(NS) for s in S]].to_numpy(float).reshape(len(d), NS, 5)
    for (tr, w), h in zip(zip(d.TRIP_ID.to_numpy(), d.way_id.to_numpy()), Hm):
        own.setdefault((int(tr), int(w)), []).append(h)

    rd = rd_ok = nb = nb_ok = n_tr = 0
    if model == "LinkETA":
        ds = core["datasets"]["test"]
        di = list(ds.dyn_idx)[:NS * 5]
        for g in ds.groups[:N]:
            tr = int(g["trip_id"])
            ents = d[d.TRIP_ID == tr].enter_ts.to_numpy(float)
            ne, x, par, rf = g["node_edge_idx"].numpy(), g["x_seq"].numpy(), g["neigh_parent"].numpy(), g["route_flag"].numpy()
            drv = set(ne[rf.astype(bool)].tolist())
            for j, e in enumerate(ne):
                w, got = ds.edge2way[int(e)], x[j, di].reshape(NS, 5)
                if int(e) in drv:
                    rd += 1; rd_ok += int(close(got, own[(tr, w)][0]))
                else:
                    nb += 1; nb_ok += int(close(got, brute(w, ents[int(par[j])], tr)))
        n_tr = min(N, len(ds.groups))
    elif model == "DeeprETA":
        lc = core["linketa_core"]
        e2w, vff = lc["datasets"]["train"].edge2way, lc["graph"]["vff_vec"].detach().cpu().numpy().astype(np.float64)
        dep = df.groupby("TRIP_ID")["departure_ts"].first()
        H = []
        for g in lc["datasets"]["train"].groups:
            tr = int(g["trip_id"])
            hr = pd.Timestamp(int(dep[tr]), unit="s").hour
            es = g["node_edge_idx"].numpy()[g["route_seq_idx"].numpy()]
            for e, L, tt in zip(es, g["seq_link_len"].numpy(), g["seq_true_tt"].numpy()):
                if tt >= 1.0 and L >= 10.0:
                    H.append((tr, int(e), hr, float(min(L / tt, 40.0))))
        H = pd.DataFrame(H, columns=["trip", "e", "hr", "v"])
        by_e, Ht, Hh, Hv = H.groupby("e").indices, H.trip.to_numpy(), H.hr.to_numpy(), H.v.to_numpy()

        def hist_ref(e, hr, tr):
            ix = by_e.get(e)
            if ix is not None:
                keep = ix[Ht[ix] != tr]
                for sel in (keep[Hh[keep] == hr], keep):
                    if len(sel):
                        return Hv[sel].mean(), np.median(Hv[sel])
            return vff[e], vff[e]

        val_ok = end_ok = 0
        for split in ("test", "train"):
            rows = core["rows"][split]
            for g in lc["datasets"][split].groups[:N]:
                tr = int(g["trip_id"])
                end, hr = float(dep[tr]), pd.Timestamp(int(dep[tr]), unit="s").hour
                es = g["node_edge_idx"].numpy()[g["route_seq_idx"].numpy()]
                lens = g["seq_link_len"].numpy().astype(np.float64)
                x = np.stack([brute(e2w[int(e)], end, tr) for e in es])
                for q, e in enumerate(es):
                    rd += 1; rd_ok += int(close(dreta.li.SPEED_HISTORY.slots(e2w[int(e)], end, tr)[:NS], x[q]))
                hm = np.array([hist_ref(int(e), hr, tr) for e in es])
                seen = (x[:, :, 4] > 0) & (x[:, :, 0] > 0)
                v = np.maximum(np.where(seen.any(1), x[np.arange(len(es)), seen.argmax(1), 0], hm[:, 1]), dreta.MIN_SPEED)
                Ls, r = max(lens.sum(), 1e-6), rows.loc[tr]
                val_ok += int(close(r["y0"], (lens / v).sum()) and close(r["rt_speed"], (lens * v).sum() / Ls)
                              and close(r["hist_speed"], (lens * hm[:, 0]).sum() / Ls))
                o, de = dreta.TRIP_ENDS[tr]
                geo = dreta.geo_tokens((o[1], o[0]), (de[1], de[0]))
                end_ok += int(all(int(r[k]) == geo[k] for k in dreta.GEO_FEATS))
                n_tr += 1
        ok = rd == rd_ok and rd > 0 and val_ok == n_tr and end_ok == n_tr
        print(f"[input] {city} DeeprETA: {'PASS' if ok else 'FAIL'}; {n_tr} trips (test and train); route-link reads "
              f"{rd_ok:,}/{rd:,} = other trips' records before the departure; RE-ETA, real-time and historical speed "
              f"(the trip itself left out) {val_ok}/{n_tr}; begin/end = first/last GPS point {end_ok}/{n_tr}", flush=True)
        if not ok:
            raise SystemExit(f"[input] {city} DeeprETA: inputs differ from the paper's rule; stopped before training")
        return
    elif model == "GSPINN":
        lc, gs = core["linketa_core"], gspinn
        e2w = lc["datasets"]["train"].edge2way
        dep = df.groupby("TRIP_ID")["departure_ts"].first()
        stat = df.groupby("way_id")[["lanes", "length", "speed_limit_ms"]].first()

        def distinct(end, width, tr, w=None):
            ix = slice(None) if w is None else gi.get(int(w))
            if ix is None:
                return 0
            tw, rw, aw = t[ix], TR[ix], avail_ts[ix]
            m = (tw >= end - width) & (tw < end) & (aw < end) & (rw != tr)
            return len(np.unique(rw[m]))

        route_ok = st_ok = lvl_ok = 0
        for split in ("test", "train"):
            for g in lc["datasets"][split].groups[:N]:
                tr = int(g["trip_id"])
                r = core["trip_state_row"][tr]
                S = float(core["state_starts"][r])
                es = g["node_edge_idx"].numpy()[g["route_seq_idx"].numpy()]
                ways = [int(e2w[int(e)]) for e in es]
                route_ok += int(ways == d.loc[d.TRIP_ID == tr, "way_id"].astype("int64").tolist()
                                and S <= float(dep[tr]) < S + gs.STATE_S)
                fr = core["flow"][r].toarray().ravel()
                for e, w in zip(es, ways):
                    rd += 1; rd_ok += int(close(fr[int(e)] * gs.FLOW_WINDOW_S, distinct(S, gs.FLOW_WINDOW_S, tr, w)))
                    st_ok += int(close(core["static"][int(e), :3], stat.loc[w].to_numpy(float)))
                lv = distinct(S, gs.CLASS_WINDOW_S, tr)
                lvl_ok += int(lv == int(core["state_level"][r])
                              and float(core["state_class"][r]) == float(lv >= core["norm"]["class_median"]))
                n_tr += 1
        batch = next(iter(lc["loaders"]["test"]))
        rows_b = np.array([core["trip_state_row"][int(x)] for x in batch["trip_ids"]])
        keep = np.flatnonzero(np.isin(rows_b, np.unique(rows_b)[:2]))
        sub = {k: batch[k][torch.as_tensor(keep)] for k in ("node_edge_idx", "route_seq_idx", "seq_mask")}
        sub["trip_ids"] = [batch["trip_ids"][i] for i in keep]
        import copy
        mdl = copy.deepcopy(core["model"])
        mdl.readout[-1].reset_parameters()
        mdl.eval()
        res = []
        _tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        for whole in (False, True):
            gg = gs._batch_graph(core, sub, whole=whole)
            x_sub, x_mod = gs._with_grad_inputs(gg)
            pred = mdl(x_mod, gg["edge_index"], gg["paths"], gg["mask"], gg["seq_len_norm"])
            grad = torch.autograd.grad(pred.sum(), x_sub)[0]
            res.append((pred.detach().cpu().numpy(), grad.cpu().numpy(), int(gg["x_raw"].shape[0])))
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = _tf32
        mdl.train()
        dp, dg = float(np.abs(res[0][0] - res[1][0]).max()), float(np.abs(res[0][1] - res[1][1]).max())
        exact = dp <= 1e-4 * max(float(np.abs(res[1][0]).max()), 1.0) and dg <= 1e-4 * float(np.abs(res[1][1]).max()) + 1e-12
        ok = rd == rd_ok and rd > 0 and st_ok == rd and route_ok == n_tr and lvl_ok == n_tr and exact
        print(f"[input] {city} GSPINN: {'PASS' if ok else 'FAIL'}; {n_tr} trips (test and train); route = LinkETA's "
              f"route order, state = 10-min departure slot {route_ok}/{n_tr}; flow reads {rd_ok:,}/{rd:,} = other "
              f"vehicles in the {gs.FLOW_WINDOW_S // 60} min before the state start; lanes, length, speed limit = df_all "
              f"rows {st_ok:,}/{rd:,}; demand level and class {lvl_ok}/{n_tr}; 3-hop graph vs whole graph "
              f"({res[0][2]:,} vs {res[1][2]:,} nodes, full float32) max |diff| prediction {dp:.2e} s of max "
              f"{float(np.abs(res[1][0]).max()):.2e} s, penalty gradient {dg:.2e} of max "
              f"{float(np.abs(res[1][1]).max()):.2e}", flush=True)
        if not ok:
            raise SystemExit(f"[input] {city} GSPINN: inputs differ from the agreed rule; stopped before training")
        return
    elif model == "ConSTGAT":
        ds, ld = core["datasets"]["test"], core["loaders"]["test"]
        eg = con.build_nodes_edges(con.edges_gdf)
        e2w = dict(zip(eg.edge_idx.astype(int), eg.way_id.astype("int64")))
        slot_ok = 0
        for bi, batch in enumerate(ld):
            nb_idx, nb_hist, bg = batch["nb_idx"].numpy(), batch["nb_hist"].numpy(), batch["bg_slots"].numpy()
            for b_i, g in enumerate(ds.groups[bi * ld.batch_size:(bi + 1) * ld.batch_size]):
                es, tr, dep = g["edge_seq"].tolist(), int(g["trip_id"]), float(g["departure_ts"])
                t0 = pd.Timestamp(g["t_ref"][0])
                exp = [(d.dayofweek, (d.hour * 60 + d.minute) // con.SLOT_BIN_MIN)
                       for d in (t0 + pd.Timedelta(minutes=k * con.SLOT_BIN_MIN) for k in range(12))]
                slot_ok += int(np.array_equal(bg[b_i], np.array(exp)) and abs(t0.value / 1e9 - dep) < 1e-6)
                n_tr += 1
                for q in range(len(es)):
                    for j, n in enumerate(nb_idx[b_i, q]):
                        if n < 0:
                            continue
                        good = int(close(nb_hist[b_i, q, j], brute(e2w[int(n)], dep, tr)[::-1]))
                        if int(n) in es:
                            rd += 1; rd_ok += good
                        else:
                            nb += 1; nb_ok += good
            if n_tr >= N:
                break
        ok = rd == rd_ok and nb == nb_ok and nb > 0 and slot_ok == n_tr
        print(f"[input] {city} ConSTGAT: {'PASS' if ok else 'FAIL'}; {n_tr} test trips; roads driven "
              f"{rd_ok:,}/{rd:,} and other roads {nb_ok:,}/{nb:,} = other trips' records before the departure; future "
              f"slots start at the departure {slot_ok}/{n_tr}", flush=True)
        if not ok:
            raise SystemExit(f"[input] {city} ConSTGAT: inputs differ from the paper's rule; stopped before training")
        return
    elif model == "DuETA":
        ds, cf = core["datasets"]["test"], core["loaders"]["test"].collate_fn
        eg, idc = core["edges_gdf"], core["edges_id_col"]
        e2w = dict(zip(eg.edge_idx.astype(int), eg[idc].astype("int64")))
        for i in range(min(N, len(ds))):
            item = ds[i]
            dyn = cf([item])["dyn_slots"][0].cpu().numpy()
            es, tr, dep = item["edge_seq"].tolist(), int(item["TRIP_ID"]), pd.Timestamp(item["t_ref"]).value / 1e9
            local = du._cached_local_for_route(tuple(es))["local_nodes"].tolist()
            n_tr += 1
            for m, e in enumerate(local):
                good = int(close(dyn[m], brute(e2w[int(e)], dep, tr)[:6][::-1]))
                if int(e) in es:
                    rd += 1; rd_ok += good
                else:
                    nb += 1; nb_ok += good
        ok = rd == rd_ok and nb == nb_ok and nb > 0
        print(f"[input] {city} DuETA: {'PASS' if ok else 'FAIL'}; {n_tr} test trips; roads driven {rd_ok:,}/{rd:,} and "
              f"other roads {nb_ok:,}/{nb:,} = other trips' records in the hour before the departure", flush=True)
        if not ok:
            raise SystemExit(f"[input] {city} DuETA: inputs differ from the paper's rule; stopped before training")
        return
    elif model == "STHR":
        ds = core["datasets"]["test"]
        st = ds.store
        for s in ds.samples[:N]:
            tr, route = int(s["trip_id"]), set(s["route"].tolist())
            n_tr += 1
            for e in s["nodes"]:
                b = brute(st.edge2way[int(e)], s["dep_ns"] / 1e9, tr)[::-1]
                exp = b[:, st.fam_ix].astype(np.float32)
                exp[b[:, 4] == 0] = np.nan
                good = int(np.allclose(st.get(int(e), s["dep_ns"], tr), exp, rtol=1e-5, atol=1e-4, equal_nan=True))
                if int(e) in route:
                    rd += 1; rd_ok += good
                else:
                    nb += 1; nb_ok += good
        ok = rd == rd_ok and nb == nb_ok and nb > 0
        print(f"[input] {city} STHR: {'PASS' if ok else 'FAIL'}; {n_tr} test trips; roads driven {rd_ok:,}/{rd:,} and "
              f"other roads {nb_ok:,}/{nb:,} = other trips' records in the 2 h before the departure", flush=True)
        if not ok:
            raise SystemExit(f"[input] {city} STHR: inputs differ from the paper's rule; stopped before training")
        return
    ok = rd == rd_ok and nb == nb_ok and nb > 0
    print(f"[input] {city} {model}: {'PASS' if ok else 'FAIL'}; {n_tr} test trips; roads driven {rd_ok:,}/{rd:,} = own "
          f"row (first pass); other roads {nb_ok:,}/{nb:,} = other trips' records before the moment", flush=True)
    if not ok:
        raise SystemExit(f"[input] {city} {model}: the model reads traffic history that differs from the clean "
                         f"definitions; stopped before training")


@contextlib.contextmanager
def working_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield path
    finally:
        os.chdir(prev)


def _require_ckpt(city: str, model: str, suffix: str = "") -> Path:
    p = results_dir(city) / ckpt_dirname(model, city, suffix=suffix) / "best.ckpt"
    if not p.exists():
        raise FileNotFoundError(
            f"Missing checkpoint for {model} on {city}: {p}\n"
            f"Train {model} on {city} first (run that block before Evaluation)."
        )
    return p


PRESERVE_REASON = {
    "LinkETA": ("Trained BEFORE the 2026-09-29 route-order fix: links within each trip were in a "
                "scrambled order (sorted by departure_time, constant within a trip), so the 1-D "
                "convolution heads saw wrong neighbours and the high-order relation was built from "
                "scrambled routes. Its published evaluation also used r_max=2.8 while training used 3.0. "
                "Kept for reference only; not comparable with runs after the fix."),
    "DuETA": ("Trained BEFORE the 2026-09-29 route-order fix (links within each trip were in a "
              "scrambled order; high-order relation built from scrambled routes). Kept for reference only."),
}


def _preserve_previous(folder: Path, model: str, what: str = "run") -> Optional[Path]:
    if not folder.exists() or not any(folder.iterdir()):
        return None
    files = sorted(p for p in folder.rglob("*") if p.is_file())
    last = max((p.stat().st_mtime for p in files), default=folder.stat().st_mtime)
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(last))
    kept = folder.with_name(f"{folder.name}_preserved_{stamp}")
    n = 1
    while kept.exists():
        kept = folder.with_name(f"{folder.name}_preserved_{stamp}_{n}"); n += 1
    folder.rename(kept)
    summary = kept / "training_summary.json"
    reason = PRESERVE_REASON.get(model, 'Previous ' + what + ', moved aside before a new one was written.')
    try:
        _s = json.loads(summary.read_text(encoding="utf-8")) if summary.exists() else {}
    except Exception:
        _s = {}
    if "route_order" in _s:
        reason = (f"Previous {what} (post-fix: route_order={_s.get('route_order')}, "
                  f"r_max={_s.get('r_max', 3.0)}, head_mode={_s.get('head_mode')}), "
                  f"moved aside before a new one was written.")
    lines = [
        f"PRESERVED {what.upper()} — {model}",
        f"original location : {folder}",
        f"preserved on      : {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"last file written : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(last))}",
        f"reason            : {reason}",
        "",
        "files:",
    ] + [f"  {p.relative_to(kept)}  ({p.stat().st_size:,} bytes)" for p in sorted(kept.rglob('*')) if p.is_file()]
    if summary.exists():
        lines += ["", "training_summary.json:", summary.read_text(encoding="utf-8")]
    (kept / "PRESERVED_README.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"      previous {what} preserved at {kept}")
    return kept


def run_dueta(city, edges_gdf, df_all):
    print(f"\n=== DuETA on {city} ===")
    du.CKPT_DIRNAME = ckpt_dirname("DuETA", city, suffix=CKPT_SUFFIX)
    _preserve_previous(results_dir(city) / du.CKPT_DIRNAME, "DuETA")
    if EPOCHS_OVERRIDE is not None:
        du.NUM_EPOCHS = EPOCHS_OVERRIDE
    if PATIENCE_OVERRIDE is not None:
        du.PATIENCE = PATIENCE_OVERRIDE
    print(f"      epochs={du.NUM_EPOCHS}  patience={du.PATIENCE}")
    du.edges_gdf = edges_gdf
    du.df_all = df_all
    core = du.dueta_get_core_components()
    check_model_inputs(city, "DuETA", core)
    with working_dir(results_dir(city)):
        du.run_training(core)


def run_constgat(city, edges_gdf, df_all):
    print(f"\n=== ConSTGAT on {city} ===")
    con.CKPT_DIRNAME = ckpt_dirname("ConSTGAT", city, suffix=CKPT_SUFFIX)
    _preserve_previous(results_dir(city) / con.CKPT_DIRNAME, "ConSTGAT")
    if EPOCHS_OVERRIDE is not None:
        con.NUM_EPOCHS = EPOCHS_OVERRIDE
    if PATIENCE_OVERRIDE is not None:
        con.PATIENCE = PATIENCE_OVERRIDE
    print(f"      epochs={con.NUM_EPOCHS}  patience={con.PATIENCE}")
    con.edges_gdf = edges_gdf
    con.df_all = df_all
    core = con.constgat_get_core_components()
    check_model_inputs(city, "ConSTGAT", core)
    with working_dir(results_dir(city)):
        con.run_training(core)


def run_sthr(city, edges_gdf, df_all):
    print(f"\n=== STHR on {city} ===")
    sthr.CKPT_DIRNAME = ckpt_dirname("STHR", city)
    if EPOCHS_OVERRIDE is not None:
        sthr.NUM_EPOCHS = EPOCHS_OVERRIDE
    if PATIENCE_OVERRIDE is not None:
        sthr.PATIENCE = PATIENCE_OVERRIDE
    if BATCH_SIZE_OVERRIDE is not None:
        sthr.BATCH_SIZE = BATCH_SIZE_OVERRIDE
    print(f"      epochs={sthr.NUM_EPOCHS}  patience={sthr.PATIENCE}  batch_size={sthr.BATCH_SIZE}")
    _preserve_previous(results_dir(city) / sthr.CKPT_DIRNAME, "STHR")
    sthr.edges_gdf = edges_gdf
    sthr.df_all = df_all
    core = sthr.sthr_get_core_components()
    check_model_inputs(city, "STHR", core)
    with working_dir(results_dir(city)):
        sthr.run_training(core)


def _setup_transeta(city):
    folder = city_dir(city)
    folder.mkdir(parents=True, exist_ok=True)
    tfeta.CKPT_DIRNAME = str(results_dir(city) / ckpt_dirname("TransETA", city))
    tfeta.CACHE_DIR = None if NO_BUILD_CACHE else str(folder / "cache_LinkETA")
    if EPOCHS_OVERRIDE is not None:
        tfeta.NUM_EPOCHS = EPOCHS_OVERRIDE
    if PATIENCE_OVERRIDE is not None:
        tfeta.PATIENCE = PATIENCE_OVERRIDE
    return tfeta


def run_transeta(city, edges_gdf, df_all):
    print(f"\n=== TransETA on {city} ===")
    _preserve_previous(results_dir(city) / ckpt_dirname("TransETA", city), "TransETA")
    tf = _setup_transeta(city)
    print(f"      epochs={tf.NUM_EPOCHS}  patience={tf.PATIENCE}  ckpt_dir={tf.CKPT_DIRNAME}")
    core = tf.transeta_get_core_components(edges_gdf, df_all)
    check_model_inputs(city, "LinkETA", core["linketa_core"])
    with working_dir(results_dir(city)):
        tf.run_training(core)


def _setup_deepreta(city):
    folder = city_dir(city)
    folder.mkdir(parents=True, exist_ok=True)
    dreta.CKPT_DIRNAME = str(results_dir(city) / ckpt_dirname("DeeprETA", city, suffix=CKPT_SUFFIX))
    if OMEGA_OVERRIDE is not None:
        dreta.OMEGA = float(OMEGA_OVERRIDE)
    dreta.CACHE_DIR = None if NO_BUILD_CACHE else str(folder / "cache_LinkETA")
    if EPOCHS_OVERRIDE is not None:
        dreta.NUM_EPOCHS = EPOCHS_OVERRIDE
    if PATIENCE_OVERRIDE is not None:
        dreta.PATIENCE = PATIENCE_OVERRIDE
    trips_path, _ = city_input_files(city)
    sm = pd.read_csv(trips_path.with_name(f"sampled_{city_folder_name(city)}.csv"), usecols=["TRIP_ID", "poly"])
    if not pd.api.types.is_integer_dtype(sm["TRIP_ID"]):
        mp = pd.read_csv(trips_path.with_name(trips_path.stem + "_trip_id_map.csv"))
        sm["TRIP_ID"] = sm["TRIP_ID"].astype(str).map(dict(zip(mp.TRIP_ID_source.astype(str), mp.TRIP_ID)))
        sm = sm.dropna(subset=["TRIP_ID"])
    if sm["TRIP_ID"].duplicated().any() or sm["poly"].isna().any():
        raise SystemExit(f"[DATA] sampled_{city_folder_name(city)}.csv: TRIP_ID not unique or poly missing")
    dreta.TRIP_ENDS = {int(t): ((float(p[0][0]), float(p[0][1])), (float(p[-1][0]), float(p[-1][1])))
                       for t, p in zip(sm["TRIP_ID"], map(json.loads, sm["poly"]))}
    return dreta


def run_deepreta(city, edges_gdf, df_all):
    print(f"\n=== DeeprETA on {city} ===")
    _preserve_previous(results_dir(city) / ckpt_dirname("DeeprETA", city, suffix=CKPT_SUFFIX), f"DeeprETA{CKPT_SUFFIX}")
    dr = _setup_deepreta(city)
    print(f"      epochs={dr.NUM_EPOCHS}  patience={dr.PATIENCE}  omega={dr.OMEGA}  ckpt_dir={dr.CKPT_DIRNAME}")
    core = dr.deepreta_get_core_components(edges_gdf, df_all)
    check_model_inputs(city, "DeeprETA", core)
    with working_dir(results_dir(city)):
        dr.run_training(core)


def _setup_probeta(city):
    folder = city_dir(city)
    folder.mkdir(parents=True, exist_ok=True)
    pbeta.CKPT_DIRNAME = str(results_dir(city) / ckpt_dirname("ProbETA", city, suffix=CKPT_SUFFIX))
    pbeta.CACHE_DIR = None if NO_BUILD_CACHE else str(folder / "cache_LinkETA")
    if EPOCHS_OVERRIDE is not None:
        pbeta.NUM_EPOCHS = EPOCHS_OVERRIDE
    if PATIENCE_OVERRIDE is not None:
        pbeta.PATIENCE = PATIENCE_OVERRIDE
    return pbeta


def run_probeta(city, edges_gdf, df_all):
    print(f"\n=== ProbETA on {city} ===")
    _preserve_previous(results_dir(city) / ckpt_dirname("ProbETA", city, suffix=CKPT_SUFFIX), f"ProbETA{CKPT_SUFFIX}")
    pb = _setup_probeta(city)
    print(f"      epochs={pb.NUM_EPOCHS}  patience={pb.PATIENCE}  ckpt_dir={pb.CKPT_DIRNAME}")
    core = pb.probeta_get_core_components(edges_gdf, df_all)
    with working_dir(results_dir(city)):
        pb.run_training(core)


def _setup_gspinn(city):
    folder = city_dir(city)
    folder.mkdir(parents=True, exist_ok=True)
    gspinn.CKPT_DIRNAME = str(results_dir(city) / ckpt_dirname("GSPINN", city, suffix=CKPT_SUFFIX))
    gspinn.CACHE_DIR = None if NO_BUILD_CACHE else str(folder / "cache_LinkETA")
    gspinn.CITY_FOLDER = city_folder_name(city)
    if EPOCHS_OVERRIDE is not None:
        gspinn.NUM_EPOCHS = EPOCHS_OVERRIDE
    if PATIENCE_OVERRIDE is not None:
        gspinn.PATIENCE = PATIENCE_OVERRIDE
    if LAMBDA_MONO_OVERRIDE is not None:
        gspinn.LAMBDA_MONO = float(LAMBDA_MONO_OVERRIDE)
    return gspinn


def run_gspinn(city, edges_gdf, df_all):
    print(f"\n=== GSPINN on {city} ===")
    _preserve_previous(results_dir(city) / ckpt_dirname("GSPINN", city, suffix=CKPT_SUFFIX), f"GSPINN{CKPT_SUFFIX}")
    gs = _setup_gspinn(city)
    print(f"      epochs={gs.NUM_EPOCHS}  patience={gs.PATIENCE}  lambda={gs.LAMBDA_MONO}  ckpt_dir={gs.CKPT_DIRNAME}")
    core = gs.gspinn_get_core_components(edges_gdf, df_all)
    check_model_inputs(city, "GSPINN", core)
    with working_dir(results_dir(city)):
        gs.run_training(core)


def run_linketa(city, edges_gdf, df_all):
    print(f"\n=== LinkETA on {city} ===")
    _preserve_previous(results_dir(city) / ckpt_dirname("LinkETA", city, suffix=CKPT_SUFFIX),
                       "LinkETA" if not CKPT_SUFFIX else f"LinkETA{CKPT_SUFFIX}")
    li = _setup_linketa_new(city)
    core = li.linketa_get_core_components(edges_gdf, df_all)
    check_model_inputs(city, "LinkETA", core)
    if EPOCHS_OVERRIDE is not None:
        core["num_epochs"] = EPOCHS_OVERRIDE
    print(f"      r_max={li.R_MAX}  label_cap={li.LABEL_CAP}  epochs={core['num_epochs']}  "
          f"ckpt_dir={li.CKPT_DIRNAME}")
    with working_dir(results_dir(city)):
        li.run_training(core)


def run_evaluation(city, edges_gdf, df_all):
    print(f"\n=== Evaluation on {city}  models={EVAL_MODELS}  output_suffix='{OUTPUT_SUFFIX}' ===")

    metrics_dict = {}
    trip_dfs     = []
    link_dfs     = []

    if "DuETA" in EVAL_MODELS:
        du.edges_gdf = edges_gdf
        du.df_all = df_all
        dueta_core = du.dueta_get_core_components()
        dueta_ckpt = torch.load(_require_ckpt(city, "DuETA", suffix=CKPT_SUFFIX), map_location=dueta_core["device"])
        dueta_core["model"].load_state_dict(dueta_ckpt["model"], strict=True)
        dueta_metrics, dueta_trip_df, dueta_link_df = eval_dueta(dueta_core)
        print("DuETA:", dueta_metrics)
        metrics_dict["DuETA"] = dueta_metrics
        trip_dfs.append(dueta_trip_df)
        link_dfs.append(dueta_link_df)

    if "ConSTGAT" in EVAL_MODELS:
        con.edges_gdf = edges_gdf
        con.df_all = df_all
        constgat_core = con.constgat_get_core_components()
        constgat_ckpt = torch.load(_require_ckpt(city, "ConSTGAT", suffix=CKPT_SUFFIX), map_location=constgat_core["device"])
        constgat_core["model"].load_state_dict(constgat_ckpt["model"], strict=True)
        constgat_metrics, constgat_trip_df, constgat_link_df = eval_constgat(constgat_core)
        print("ConSTGAT:", constgat_metrics)
        metrics_dict["ConSTGAT"] = constgat_metrics
        trip_dfs.append(constgat_trip_df)
        link_dfs.append(constgat_link_df)

    if "LinkETA" in EVAL_MODELS:
        li = _setup_linketa_new(city)
        linketa_core = li.linketa_get_core_components(edges_gdf, df_all)
        ckpt_path = _require_ckpt(city, "LinkETA", suffix=CKPT_SUFFIX)
        linketa_ckpt = torch.load(ckpt_path, map_location=linketa_core["device"])
        _ck_mode = (linketa_ckpt.get("extra") or {}).get("head_mode", "speed")
        if _ck_mode != li.HEAD_MODE:
            raise RuntimeError(f"LinkETA checkpoint was trained with HEAD_MODE='{_ck_mode}' but "
                               f"LinkETA.HEAD_MODE is '{li.HEAD_MODE}'. Set it to match before evaluating.")
        _ck_rmax = float((linketa_ckpt.get("extra") or {}).get("r_max", 3.0))
        if abs(_ck_rmax - li.R_MAX) > 1e-9:
            raise RuntimeError(f"LinkETA checkpoint was trained with r_max={_ck_rmax} but LinkETA.R_MAX "
                               f"is {li.R_MAX}. Pass --r-max {_ck_rmax} to evaluate it.")
        linketa_core["model"].load_state_dict(linketa_ckpt["model"], strict=True)
        linketa_metrics, linketa_trip_df, linketa_link_df = eval_linketa(linketa_core)
        print("LinkETA:", linketa_metrics)
        metrics_dict["LinkETA"] = linketa_metrics
        trip_dfs.append(linketa_trip_df)
        link_dfs.append(linketa_link_df)

    if "STHR" in EVAL_MODELS:
        if BATCH_SIZE_OVERRIDE is not None:
            sthr.BATCH_SIZE = BATCH_SIZE_OVERRIDE
        sthr.edges_gdf = edges_gdf
        sthr.df_all = df_all
        sthr_core = sthr.sthr_get_core_components()
        sthr_ckpt = torch.load(_require_ckpt(city, "STHR"), map_location=sthr_core["device"])
        sthr_core["model"].load_state_dict(sthr_ckpt["model"], strict=True)
        sthr_metrics, sthr_trip_df, sthr_link_df = eval_sthr(sthr_core)
        print("STHR:", sthr_metrics)
        metrics_dict["STHR"] = sthr_metrics
        trip_dfs.append(sthr_trip_df)
        link_dfs.append(sthr_link_df)

    if "TransETA" in EVAL_MODELS:
        tf = _setup_transeta(city)
        tf_core = tf.transeta_get_core_components(edges_gdf, df_all)
        tf_metrics, tf_trip_df, tf_link_df = eval_transeta(tf_core, ckpt_path=str(_require_ckpt(city, "TransETA")))
        print("TransETA:", tf_metrics)
        metrics_dict["TransETA"] = tf_metrics
        trip_dfs.append(tf_trip_df)
        link_dfs.append(tf_link_df)

    if "DeeprETA" in EVAL_MODELS:
        dr = _setup_deepreta(city)
        dr_core = dr.deepreta_get_core_components(edges_gdf, df_all)
        dr_metrics, dr_trip_df, dr_link_df = eval_deepreta(
            dr_core, ckpt_path=str(_require_ckpt(city, "DeeprETA", suffix=CKPT_SUFFIX)))
        print("DeeprETA:", dr_metrics)
        metrics_dict["DeeprETA"] = dr_metrics
        trip_dfs.append(dr_trip_df)
        link_dfs.append(dr_link_df)

    if "ProbETA" in EVAL_MODELS:
        pb = _setup_probeta(city)
        pb_core = pb.probeta_get_core_components(edges_gdf, df_all)
        pb_metrics, pb_trip_df, pb_link_df = eval_probeta(
            pb_core, ckpt_path=str(_require_ckpt(city, "ProbETA", suffix=CKPT_SUFFIX)))
        print("ProbETA:", pb_metrics)
        metrics_dict["ProbETA"] = pb_metrics
        trip_dfs.append(pb_trip_df)
        link_dfs.append(pb_link_df)

    if "GSPINN" in EVAL_MODELS:
        gs = _setup_gspinn(city)
        gs_core = gs.gspinn_get_core_components(edges_gdf, df_all)
        gs_metrics, gs_trip_df, gs_link_df = eval_gspinn(
            gs_core, ckpt_path=str(_require_ckpt(city, "GSPINN", suffix=CKPT_SUFFIX)))
        print("GSPINN:", gs_metrics)
        metrics_dict["GSPINN"] = gs_metrics
        trip_dfs.append(gs_trip_df)
        link_dfs.append(gs_link_df)

    if not trip_dfs:
        print("[EVAL] no models selected; nothing to write.")
        return None, None

    df_eval = pd.concat(trip_dfs, ignore_index=True)
    df_eval["abs_err_s"]  = (df_eval["y_pred"] - df_eval["y_true"]).abs()
    df_eval["model_trip"] = df_eval["model"].astype(str) + "::" + df_eval["TRIP_ID"].astype(str)

    df_eval_link = pd.concat(link_dfs, ignore_index=True)
    df_eval_link["abs_err_s"] = (df_eval_link["pred_tt_link"] - df_eval_link["true_tt_link"]).abs()
    print(f"[EVAL] trips: {len(df_eval):,}  links: {len(df_eval_link):,}")

    out_dir = results_dir(city) / "eval_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    full_suffix  = OUTPUT_SUFFIX
    if "STHR" in EVAL_MODELS:
        if "_sthr" not in full_suffix:
            full_suffix += "_sthr"
        names = [f"eval_metrics{full_suffix}.json", f"eval_trips{full_suffix}.csv",
                 f"eval_links{full_suffix}.csv"]
        if any((out_dir / n).exists() for n in names):
            full_suffix += "_" + time.strftime("%Y%m%d_%H%M%S")
    metrics_path = out_dir / f"eval_metrics{full_suffix}.json"
    trips_path   = out_dir / f"eval_trips{full_suffix}.csv"
    links_path   = out_dir / f"eval_links{full_suffix}.csv"
    _existing = [p for p in (metrics_path, trips_path, links_path) if p.exists()]
    if _existing:
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(max(p.stat().st_mtime for p in _existing)))
        keep_dir = out_dir / f"preserved_{stamp}"
        n = 1
        while keep_dir.exists():
            keep_dir = out_dir / f"preserved_{stamp}_{n}"; n += 1
        keep_dir.mkdir(parents=True)
        for p in _existing:
            p.rename(keep_dir / p.name)
        reasons = "; ".join(f"{m}: {PRESERVE_REASON[m]}" for m in EVAL_MODELS if m in PRESERVE_REASON)
        (keep_dir / "PRESERVED_README.txt").write_text("\n".join([
            "PRESERVED EVALUATION OUTPUTS",
            f"original folder : {out_dir}",
            f"preserved on    : {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"files           : {', '.join(p.name for p in _existing)}",
            f"replaced by a new evaluation of: {', '.join(EVAL_MODELS)}",
            f"notes           : {reasons or 'previous evaluation outputs, moved aside before new ones were written.'}",
        ]), encoding="utf-8")
        print(f"[EVAL] previous outputs preserved in {keep_dir}")

    combined_metrics = {
        "city": city,
        "linketa_version": "new",
        "models_evaluated": list(EVAL_MODELS),
        **{k: {kk: (float(vv) if isinstance(vv, (int, float)) else vv)
               for kk, vv in m.items()}
           for k, m in metrics_dict.items()},
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(combined_metrics, f, indent=2, default=str)
    df_eval.to_csv(trips_path, index=False)
    df_eval_link.to_csv(links_path, index=False)
    print(f"[saved] metrics -> {metrics_path}")
    print(f"[saved] trips   -> {trips_path}  ({len(df_eval):,} rows)")
    print(f"[saved] links   -> {links_path}  ({len(df_eval_link):,} rows)")

    return df_eval, df_eval_link


BLOCKS = {
    "ConSTGAT":   run_constgat,
    "DuETA":      run_dueta,
    "LinkETA":    run_linketa,
    "STHR":       run_sthr,
    "TransETA":   run_transeta,
    "DeeprETA":   run_deepreta,
    "ProbETA":    run_probeta,
    "GSPINN":     run_gspinn,
    "Evaluation": run_evaluation,
}


TASKS = {
    "benchmark":    BASE_DIR / "evaluation" / "benchmark.py",
    "robustness":   BASE_DIR / "evaluation" / "robustness.py",
    "ablations":    BASE_DIR / "evaluation" / "ablations.py",
    "significance": BASE_DIR / "evaluation" / "significance_test.py",
    "figures":      BASE_DIR / "figures" / "figures.py",
}


def run_task(name, argv):
    script = TASKS[name]
    sys.argv = [str(script)] + list(argv)
    runpy.run_path(str(script), run_name="__main__")


def _pick_from_menu(title, options, allow_multi=False):
    print(f"\n{title}")
    for i, name in enumerate(options, 1):
        print(f"  {i}) {name}")
    if allow_multi:
        print("  a) all")
    prompt = "Select (comma-separated for multiple): " if allow_multi else "Select: "
    while True:
        raw = input(prompt).strip()
        if not raw:
            continue
        if allow_multi and raw.lower() in ("a", "all"):
            return list(options)
        try:
            idxs = [int(x) - 1 for x in raw.split(",")] if allow_multi else [int(raw) - 1]
            picked = [options[i] for i in idxs]
            return picked if allow_multi else picked[0]
        except (ValueError, IndexError):
            print("Invalid choice, try again.")


def _resolve_blocks(arg):
    if arg.lower() == "all":
        return list(BLOCKS)
    lookup = {k.lower(): k for k in BLOCKS}
    out = []
    for n in (x.strip() for x in arg.split(",") if x.strip()):
        if n.lower() not in lookup:
            raise ValueError(f"Unknown block '{n}'. Choose from: {list(BLOCKS)}")
        out.append(lookup[n.lower()])
    return out


def _resolve_datasets(arg):
    if arg.lower() == "all":
        return list(CITIES)
    lookup = {k.lower(): k for k in CITIES}
    for display, folder in CITY_FOLDERS.items():
        lookup[folder.lower()] = display
    out = []
    for raw in (x.strip() for x in arg.split(",") if x.strip()):
        if raw.lower() not in lookup:
            raise ValueError(
                f"Unknown dataset '{raw}'. Choose from display names "
                f"{CITIES} or short codes {list(CITY_FOLDERS.values())}."
            )
        out.append(lookup[raw.lower()])
    return out


EPOCHS_OVERRIDE = None
PATIENCE_OVERRIDE = None
BATCH_SIZE_OVERRIDE = None
NO_BUILD_CACHE = False
R_MAX_OVERRIDE = None
CKPT_SUFFIX = ""
OMEGA_OVERRIDE = None
LAMBDA_MONO_OVERRIDE = None
EVAL_MODELS = ["DuETA", "ConSTGAT", "LinkETA"]
OUTPUT_SUFFIX = ""
_EVAL_MODEL_CHOICES = ("DuETA", "ConSTGAT", "LinkETA", "STHR", "TransETA", "DeeprETA", "ProbETA", "GSPINN")


def parse_args():
    p = argparse.ArgumentParser(description="LinkETA experiment runner. Other tasks: python main.py "
                                            "{" + ",".join(TASKS) + "} [task arguments]")
    p.add_argument("--blocks", help=f"Comma-separated: {','.join(BLOCKS)} (or 'all')")
    p.add_argument("--datasets", "--dataset", dest="datasets",
                   help=f"Comma-separated cities (or 'all'). Choices: {', '.join(CITIES)}")
    p.add_argument("--max-trips", type=int, default=None,
                   help="Debug: sample at most N TRIP_IDs from the dataset before training "
                        "(random, seed=42). Useful for verifying the pipeline runs end-to-end "
                        "in a few minutes. Omit for full dataset.")
    p.add_argument("--epochs", type=int, default=None,
                   help="Override max number of training epochs. Currently applied to DuETA and STHR.")
    p.add_argument("--patience", type=int, default=None,
                   help="Override early-stopping patience (epochs without improvement). "
                        "Currently applied to DuETA and STHR.")
    p.add_argument("--no-build-cache", action="store_true", default=False,
                   help="[2026-09-29] Do not read/write the LinkETA build cache "
                        "(data/<city>/cache_LinkETA/); always rebuild relations and trip groups.")
    p.add_argument("--r-max", type=float, default=None,
                   help="[2026-10-05] LinkETA speed-ratio bound r_max (model output, training-target cap "
                        "and evaluation). Default: LinkETA.R_MAX = 2.8 (canonical since 2026-10-06; "
                        "use --r-max 3.0 for checkpoints trained before, which carry no r_max tag).")
    p.add_argument("--ckpt-suffix", default="",
                   help="[2026-10-05] Suffix for the LinkETA checkpoint folder, e.g. '_rmax2.8' -> "
                        "ckpts_LinkETA_<city>_rmax2.8. Keeps existing runs untouched. "
                        "[2026-10-08] Also applies to DeeprETA (ckpts_DeeprETA_<city><suffix>).")
    p.add_argument("--omega", type=float, default=None,
                   help="[2026-10-08] DeeprETA asymmetric-Huber weight omega in [0,1] (paper Eq. 14). "
                        "Default: DeeprETA.OMEGA. Used for the omega selection on Porto validation.")
    p.add_argument("--lambda-mono", type=float, default=None,
                   help="[2026-10-17] GSPINN weight lambda of the monotonicity penalty (paper Eq. 17), one of "
                        "GSPINN.LAMBDA_GRID. Default: GSPINN.LAMBDA_MONO. Used for the 100-trip smoke-test selection.")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Override the batch size. Currently applied to STHR only "
                        "(default 500, from the STHR paper/README).")
    p.add_argument("--eval-models", default=None,
                   help=f"Comma-separated subset of models to run during the Evaluation "
                        f"block. Choices: {', '.join(_EVAL_MODEL_CHOICES)} (or 'all'). "
                        f"Default: DuETA,ConSTGAT,LinkETA — preserves prior behavior. Use "
                        f"e.g. --eval-models STHR to evaluate only a single baseline "
                        f"without recomputing the others.")
    p.add_argument("--output-suffix", default="",
                   help="Optional suffix appended to evaluation output "
                        "filenames (e.g. '_sthr' produces eval_trips_sthr.csv). Useful for "
                        "running a single baseline alone without overwriting an existing "
                        "combined-models output. Default: empty.")
    return p.parse_args()


def _resolve_eval_models(arg):
    if arg is None or arg.strip() == "":
        return list(EVAL_MODELS)
    raw = arg.strip()
    if raw.lower() == "all":
        return list(_EVAL_MODEL_CHOICES)
    lookup = {m.lower(): m for m in _EVAL_MODEL_CHOICES}
    out = []
    for n in (x.strip() for x in raw.split(",") if x.strip()):
        if n.lower() not in lookup:
            raise ValueError(
                f"Unknown eval model '{n}'. Choose from {list(_EVAL_MODEL_CHOICES)} or 'all'.")
        out.append(lookup[n.lower()])
    return out


def main():
    global EPOCHS_OVERRIDE, PATIENCE_OVERRIDE, BATCH_SIZE_OVERRIDE
    global EVAL_MODELS, OUTPUT_SUFFIX, NO_BUILD_CACHE
    global R_MAX_OVERRIDE, CKPT_SUFFIX
    global OMEGA_OVERRIDE
    global LAMBDA_MONO_OVERRIDE
    args = parse_args()
    R_MAX_OVERRIDE = args.r_max
    OMEGA_OVERRIDE = args.omega
    LAMBDA_MONO_OVERRIDE = args.lambda_mono
    CKPT_SUFFIX = args.ckpt_suffix or ""
    NO_BUILD_CACHE = args.no_build_cache
    EPOCHS_OVERRIDE = args.epochs
    PATIENCE_OVERRIDE = args.patience
    BATCH_SIZE_OVERRIDE = args.batch_size
    EVAL_MODELS  = _resolve_eval_models(args.eval_models)
    OUTPUT_SUFFIX = args.output_suffix or ""
    blocks = _resolve_blocks(args.blocks) if args.blocks else _pick_from_menu(
        "Which block(s) to run?", list(BLOCKS), allow_multi=True
    )
    datasets = _resolve_datasets(args.datasets) if args.datasets else _pick_from_menu(
        "Which dataset(s)?", CITIES, allow_multi=True
    )

    print(f"\n[PLAN] Datasets: {datasets} | Blocks: {blocks}")
    print(f"[BASE] {BASE_DIR}")

    for dataset in datasets:
        print(f"\n{'='*70}\n[CITY] {dataset}  data: {city_dir(dataset)}  results: {results_dir(dataset)}\n{'='*70}")
        edges_gdf, df_all = load_dataset(dataset, max_trips=args.max_trips)
        for name in blocks:
            BLOCKS[name](dataset, edges_gdf, df_all)

    print("\n[DONE]")


if __name__ == "__main__":
    try:
        if len(sys.argv) > 1 and sys.argv[1] in TASKS:
            run_task(sys.argv[1], sys.argv[2:])
        else:
            main()
    except KeyboardInterrupt:
        print("\n[ABORTED] interrupted by user.")
        sys.exit(130)

