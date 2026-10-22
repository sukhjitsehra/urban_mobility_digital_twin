import json
import os
import pickle
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
_FOLDER = sys.argv[1] if __name__ == "__main__" and len(sys.argv) > 1 else None
CFG = (json.load(open(REPO / "preprocess" / "configs" / f"{_FOLDER}.json", encoding="utf-8"))
       if _FOLDER and (REPO / "preprocess" / "configs" / f"{_FOLDER}.json").exists() else None)
MMCFG = (CFG or {}).get("map_matching") or {}
for _env, _key in (("CRS_METRIC", "crs_metric"), ("OSRM_URL", "osrm_url"), ("LOCAL_TZ", "local_tz"),
                   ("DYN_ALIGN", "dyn_align"), ("MM_SLOTS", "slots"), ("MM_BUF_M", "buf_m"), ("MM_RADIUS_M", "radius_m")):
    if MMCFG.get(_key):
        os.environ[_env] = str(MMCFG[_key])
if CFG is not None:
    _missing = [k for k in ("crs_metric", "osrm_url") if not MMCFG.get(k)]
    if _missing:
        raise SystemExit(f"preprocess/configs/{_FOLDER}.json: map_matching {_missing} not set")

import map_matching as mm

DATA_ROOT = Path(os.environ.get("ETA_DATA_ROOT", REPO / "data"))
WORK_ROOT = Path(os.environ["MM_WORK"]) if os.environ.get("MM_WORK") else None


def paths(folder):
    d = DATA_ROOT / folder
    trips = d / (f"sampled_{folder}.csv" if CFG is not None else f"trips_{folder}.parquet")
    work = (WORK_ROOT / f"{folder}_mm") if WORK_ROOT else d / "mm_work"
    out_dir = Path(os.environ["MM_OUT_DIR"]) if os.environ.get("MM_OUT_DIR") else d
    return trips, d / f"edges_{folder}.geojson", out_dir / f"df_all_{folder}.csv", work


def load_trips(folder, max_trips):
    f = paths(folder)[0]
    t = (pd.read_csv(f, dtype={"TRIP_ID": str}) if f.suffix == ".csv" else pd.read_parquet(f)).sort_values("TRIP_ID")
    if os.environ.get("MM_TRIP_IDS"):
        keep = set(pd.read_csv(os.environ["MM_TRIP_IDS"], dtype={"TRIP_ID": str})["TRIP_ID"])
        t = t[t["TRIP_ID"].astype(str).isin(keep)]
        print(f"MM_TRIP_IDS: {len(t)} of {len(keep)} requested trips found in {f.name}", flush=True)
    if max_trips:
        t = t.head(max_trips)
    t["poly"] = t.poly.apply(lambda p: [tuple(x) for x in (json.loads(p) if isinstance(p, str) else p)])
    if "point_ts" in t.columns:
        t["point_ts"] = t.point_ts.apply(lambda p: [int(x) for x in (json.loads(p) if isinstance(p, str) else p)])
    return t


def shard(folder, i, n, max_trips=None):
    _, edges, _, work = paths(folder)
    work.mkdir(parents=True, exist_ok=True)
    t = load_trips(folder, max_trips).iloc[i::n]
    p = mm.MapMatchingPipeline(gpd.read_file(edges), crs_metric=mm.CRS_METRIC)
    t0 = time.time()
    res = p.process_trips(t, max_workers=int(MMCFG.get("workers_per_shard", 4)))
    with open(work / f"df_res_{i}.pkl", "wb") as f:
        pickle.dump(res, f)
    print(f"[shard {i}/{n}] {len(res)} trips in {time.time() - t0:.0f} s; OSRM chunks {int(res.osrm_chunks.sum())}, "
          f"failed {int(res.osrm_failed_chunks.sum())} (trips affected {int((res.osrm_failed_chunks > 0).sum())})",
          flush=True)


TOL_S = 1e-3
MAX_RATIO, MAX_END_SHARE = 1.5, 30.0


def hav(lon1, lat1, lon2, lat2):
    p = np.pi / 180
    a = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    return 2 * 6371000.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _log(rows, check, n_bad, note=""):
    rows.append(dict(check=check, trips_failing=int(n_bad), note=note))
    print(f"[check] {check}: {n_bad} trips failing" + (f" ({note})" if note else ""), flush=True)


def check_matching(res, trips, rows):
    assert res.TRIP_ID.is_unique and set(res.TRIP_ID.astype(str)) == set(trips.TRIP_ID.astype(str)), \
        "trips returned by the shards differ from the sampled trips"
    print(f"[check] every sampled trip returned exactly once: {len(res)} of {len(trips)}", flush=True)
    empty = res.matched_coords.apply(len) < 2
    _log(rows, "empty OSRM match (< 2 matched points)", empty.sum())
    failed = (res.osrm_failed_chunks > 0) if "osrm_failed_chunks" in res.columns else pd.Series(False, index=res.index)
    ex = res.loc[failed, "osrm_error"].head(3).tolist() if failed.any() else []
    _log(rows, "OSRM request failed for part of the trip", failed.sum(), f"examples: {ex}" if ex else "")
    bad = set(res.loc[empty | failed, "TRIP_ID"].astype(str))
    return res[~(empty | failed)].copy(), bad


def check_links(feat, trips, edge_ids, rows):
    bad = {}
    f = feat.sort_values(["TRIP_ID", "enter_ts"])
    tr = trips.set_index(trips.TRIP_ID.astype(str))

    def flag(name, ids, note=""):
        ids = set(map(str, ids))
        _log(rows, name, len(ids), note)
        for i in ids:
            bad.setdefault(i, name)

    flag("TRIP_ID missing", f.loc[f.TRIP_ID.isna(), "TRIP_ID"])
    flag("TRIP_ID not in the sampled trips", set(f.TRIP_ID.astype(str)) - set(tr.index))
    key = ["way_id", "enter_ts", "exit_ts", "link_travel_time_s", "link_len_m", "TIMESTAMP", "trip_travel_time_s",
           "departure_time"]
    nn = f[[c for c in key if c in f.columns]].isna().any(axis=1)
    flag("missing value in a key column", f.loc[nn, "TRIP_ID"])
    flag("link id not in the road network", f.loc[~f.way_id.isin(edge_ids), "TRIP_ID"])
    flag("link travel time <= 0", f.loc[f.link_travel_time_s <= 0, "TRIP_ID"])
    flag("link length <= 0", f.loc[f.link_len_m <= 0, "TRIP_ID"])
    flag("enter_ts >= exit_ts", f.loc[f.enter_ts >= f.exit_ts, "TRIP_ID"])
    flag("link time != exit_ts - enter_ts",
         f.loc[((f.exit_ts - f.enter_ts) - f.link_travel_time_s).abs() > TOL_S, "TRIP_ID"])
    nxt = f.groupby("TRIP_ID").enter_ts.shift(-1)
    gap = ((nxt - f.exit_ts).abs() > TOL_S) & nxt.notna()
    flag("links not contiguous (exit != next enter)", f.loc[gap, "TRIP_ID"])
    g = f.groupby("TRIP_ID")
    first, last, tsum = g.enter_ts.min(), g.exit_ts.max(), g.link_travel_time_s.sum()
    T0 = pd.to_numeric(tr.TIMESTAMP).reindex(first.index)
    TT = pd.to_numeric(tr.trip_travel_time_s).reindex(first.index)
    flag("first link entry != TIMESTAMP", first.index[(first - T0).abs() > TOL_S])
    flag("last link exit != TIMESTAMP + trip time", last.index[(last - T0 - TT).abs() > TOL_S])
    d = (tsum - TT).abs()
    flag("sum of link times != trip time", d.index[d > TOL_S],
         f"mean |diff| {d.mean():.6f} s, max {d.max():.6f} s over {len(d)} trips")
    gps = pd.Series({tid: hav(*(lambda xy: (xy[:-1, 0], xy[:-1, 1], xy[1:, 0], xy[1:, 1]))(np.asarray(p, float))).sum()
                     for tid, p in tr.poly[tr.index.isin(first.index)].items()}).reindex(first.index)
    ratio = g.link_len_m.sum() / gps.clip(lower=1.0)
    flag(f"matched route > {MAX_RATIO} x GPS path length", ratio.index[ratio > MAX_RATIO])
    end = (g.link_travel_time_s.first() + g.link_travel_time_s.last() * (g.size() > 1)) / tsum * 100
    flag(f"first + last link > {MAX_END_SHARE:.0f}% of trip time", end.index[end > MAX_END_SHARE])
    return bad


def check_stops(feat, trips, rows, stop_m=5.0, stop_s=60.0):
    g = {t: x for t, x in feat.groupby(feat.TRIP_ID.astype(str))}
    n_stops = n_bad = 0
    for r in trips.itertuples(index=False):
        x = g.get(str(r.TRIP_ID))
        if x is None or len(r.poly) < 2:
            continue
        p = np.radians(np.asarray(r.poly, float))
        if hasattr(r, "point_ts"):
            t = np.asarray(r.point_ts, float)
        else:
            t = int(r.TIMESTAMP) + float(r.trip_travel_time_s) / (len(p) - 1) * np.arange(len(p))
        a = np.sin(np.diff(p[:, 1]) / 2) ** 2 + np.cos(p[:-1, 1]) * np.cos(p[1:, 1]) * np.sin(np.diff(p[:, 0]) / 2) ** 2
        still = 2 * 6_371_000.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1))) <= stop_m
        i = 0
        while i < len(still):
            if not still[i]:
                i += 1
                continue
            j = i
            while j < len(still) and still[j]:
                j += 1
            if t[j] - t[i] >= stop_s:
                n_stops += 1
                n_bad += not ((x.enter_ts <= t[i] + 1.0) & (x.exit_ts >= t[j] - 1.0)).any()
            i = j
    _log(rows, "stop spread over more than one link (reported only)", n_bad, f"of {n_stops} stops >= {stop_s:.0f} s")


def check_leaks(feat, speeds, trips, hist_ids, res, rows):
    S = ["median_speed", "mean_speed", "min_speed", "max_speed", "count_vehicle"]
    B = mm.BIN_SECONDS
    d, sp = feat, speeds
    K = sorted(int(c.rsplit("_", 1)[1]) for c in d.columns if c.startswith("count_vehicle_"))
    Hs = mm.SpeedHistory(sp, slots=len(K))
    tid, way, ent = d.TRIP_ID.astype("int64").to_numpy(), d.way_id.to_numpy(), d.enter_ts.to_numpy(float)
    Hm = d[[f"{s}_{k}" for k in K for s in S]].to_numpy(float).reshape(len(d), len(K), len(S))
    sp_ids = set(sp.trip_id.astype(str))
    ok = True

    eq = np.array([np.allclose(Hs.slots(way[i], ent[i], tid[i]), Hm[i], rtol=1e-9, atol=1e-9) for i in range(len(d))])
    drop = {}
    for i in d.TRIP_ID[~eq].astype(str).unique():
        drop.setdefault(i, "leak: history value not rebuilt exactly from the speed records")
    _log(rows, "leak: history value not rebuilt exactly from the speed records", d.TRIP_ID[~eq].nunique(),
         f"{int(eq.sum()):,} of {len(d):,} rows rebuilt")

    t = sp.timestamp.astype("datetime64[ns]").astype("int64").to_numpy() / 1e9
    avail_ts = sp.avail_ts.to_numpy(float)
    TRa, Va, Wa = sp.trip_id.to_numpy().astype(np.int64), sp.speed_m_s.to_numpy(float), sp.way_id.to_numpy()
    gi = sp.groupby("way_id").indices
    rng = np.random.default_rng(0)
    q = rng.choice(len(sp), size=min(3000, len(sp)), replace=False)
    bad = own_in = 0
    for i in q:
        w, tr = Wa[i], int(TRa[i])
        end = t[i] + rng.uniform(0, 1200)
        ix = gi[w]
        tw, vw, rw = t[ix], Va[ix], TRa[ix]
        got = Hs.slots(w, end, tr)
        for k in range(len(K)):
            m = (tw >= end - (k + 1) * B) & (tw < end - k * B)
            own_in += int((m & (rw == tr)).any())
            m &= (rw != tr) & (avail_ts[ix] < end)
            v = vw[m]
            exp = (np.median(v), v.mean(), v.min(), v.max(), len(np.unique(rw[m]))) if m.any() else (0,) * 5
            bad += int(not np.allclose(got[k], exp, rtol=1e-9, atol=1e-9))
    ok &= bad == 0
    _log(rows, "leak: own record or a record not yet known in a history window (random reads vs a plain filter)",
         bad, f"mismatching slots in {len(q):,} reads; {own_in} windows held the trip's own record")

    dropped = ((set(trips.TRIP_ID.astype(str)) | set(hist_ids)) - set(d.TRIP_ID.astype(str))
               - (set(hist_ids) & set(res.TRIP_ID.astype(str))))
    ok &= not (dropped & sp_ids)
    _log(rows, "leak: dropped trip in the speed records", len(dropped & sp_ids), f"of {len(dropped)} dropped trips")
    for i in set(hist_ids) & set(d.TRIP_ID.astype(str)):
        drop.setdefault(i, "leak: history-only trip in df_all")
    _log(rows, "leak: history-only trip in df_all", len(set(hist_ids) & set(d.TRIP_ID.astype(str))))

    early = (ent - len(K[:12]) * B) < t.min()
    for i in d.TRIP_ID[early].astype(str).unique():
        drop.setdefault(i, "leak: link entered < 2 h after the first speed record")
    _log(rows, "leak: link entered < 2 h after the first speed record", d.TRIP_ID[early].nunique(),
         f"{int(early.sum())} rows")

    have = set(zip(d.TRIP_ID.astype("int64"), d.way_id))
    pairs = sp[sp.trip_id.astype(str).isin(set(d.TRIP_ID.astype(str)))].groupby(["trip_id", "way_id"]).size()
    orphan = pairs[[(a, w) not in have for a, w in pairs.index]]
    _log(rows, "speed records on roads without the trip's own row (reported only)", orphan.index.get_level_values(0).nunique(),
         f"{orphan.sum() / len(sp) * 100:.2f}% of records; the models leave the trip out by its ID on every road")
    return drop, ok


def run_checks(feat, trips, edge_ids, rows, pre_bad, strict, out, link_bad, speeds, hist_ids, res):
    bad = dict(link_bad)
    for i, why in check_links(feat, trips, edge_ids, rows).items():
        bad.setdefault(i, why)
    check_stops(feat, trips, rows)
    no_links = set(trips.TRIP_ID.astype(str)) - set(feat.TRIP_ID.astype(str)) - set(pre_bad) - set(link_bad)
    _log(rows, "no links after matching", len(no_links))
    for i in no_links:
        bad.setdefault(i, "no links after matching")
    for i in pre_bad:
        bad.setdefault(i, "matching")
    if bad and strict:
        raise SystemExit(f"--strict: {len(bad)} invalid trips; see the [check] lines above")
    valid = feat[~feat.TRIP_ID.astype(str).isin(bad)]
    leak_bad, leak_ok = check_leaks(valid, speeds, trips, hist_ids, res, rows)
    for i, why in leak_bad.items():
        bad.setdefault(i, why)
    valid = valid[~valid.TRIP_ID.astype(str).isin(leak_bad)]
    slots = [c for c in valid.columns if c.startswith("count_vehicle_")]
    if slots:
        print(f"[check] history coverage: slot0 filled {(valid.count_vehicle_0 > 0).mean():.4f}, "
              f"any slot {(valid[slots].sum(1) > 0).mean():.4f}", flush=True)
    rep_dir = out.parent / "mm_report"
    rep_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(rep_dir / f"{out.stem}_checks.csv", index=False)
    pd.Series(bad, name="first_failed_check", dtype=object).rename_axis("TRIP_ID").to_csv(
        rep_dir / f"{out.stem}_invalid_trips.csv")
    if not leak_ok:
        raise SystemExit("leak check failed, the history rule itself (see the [check] leak lines above): df_all not "
                         f"written; reports in {rep_dir}; the map-matching results stay in mm_work/, so after the fix "
                         "only 'run_map_matching.py <folder> features <n>' is rerun")
    print(f"[check] summary: sampled {len(trips)} -> matched {len(trips) - len(pre_bad)} -> valid "
          f"{valid.TRIP_ID.nunique()} trips, {len(valid)} link rows; invalid trips dropped: {len(bad)}", flush=True)
    return valid


def features(folder, n, max_trips=None):
    _, edges, out, work = paths(folder)
    parts = []
    for i in range(n):
        with open(work / f"df_res_{i}.pkl", "rb") as f:
            parts.append(pickle.load(f))
    res = pd.concat(parts, ignore_index=True)
    trips = load_trips(folder, max_trips)
    assert len(res) == len(trips) and res.TRIP_ID.is_unique, (len(res), len(trips))
    print(f"[features] {len(res)} trips; empty matches {(res.matched_coords.apply(len) < 2).sum()}", flush=True)
    rows = []
    res, pre_bad = check_matching(res, trips, rows)
    p = mm.MapMatchingPipeline(gpd.read_file(edges), crs_metric=mm.CRS_METRIC)
    t0 = time.time()
    links = p.build_features(res, skip_dynamic_features=True).merge(
        trips[["TRIP_ID", "TIMESTAMP", "trip_travel_time_s"]], on="TRIP_ID", how="left")
    link_bad = check_links(links, trips, set(p.edge_ids.tolist()), rows)
    res = res[~res.TRIP_ID.astype(str).isin(link_bad)]
    print(f"[features] {len(link_bad)} trips dropped before the history; {len(res)} trips build df_all and the history",
          flush=True)
    feat = p.build_features(res)
    print(f"[features] {len(feat)} link rows in {time.time() - t0:.0f} s", flush=True)
    speeds = mm.collect_raw_speeds_from_df(res, p.transformer)
    hist_ids = (set(trips.TRIP_ID[trips.history_only.astype(bool)].astype(str)) if "history_only" in trips.columns
                else set())
    if hist_ids:
        n_hist_kept = feat[feat.TRIP_ID.astype(str).isin(hist_ids)].TRIP_ID.nunique()
        feat = feat[~feat.TRIP_ID.astype(str).isin(hist_ids)]
        trips = trips[~trips.TRIP_ID.astype(str).isin(hist_ids)]
        pre_bad = set(pre_bad) - hist_ids
        link_bad = {k: v for k, v in link_bad.items() if k not in hist_ids}
        print(f"[features] history-only trips: {len(hist_ids)} sampled, {n_hist_kept} in the history, none in df_all",
              flush=True)
    if CFG is not None:
        tcols = ["TRIP_ID"] + [c for c in trips.columns if c not in feat.columns
                               and c not in ("TRIP_ID", "poly", "n_points", "trip_time_bucket", "history_only")]
        tr = trips[tcols].copy()
        if "point_ts" in tr.columns:
            tr["point_ts"] = tr.point_ts.apply(lambda p: json.dumps(p, separators=(",", ":")))
        feat = feat.merge(tr, on="TRIP_ID", how="left")
    else:
        feat = feat.merge(trips[["TRIP_ID", "trip_travel_time_s"]], on="TRIP_ID", how="left")
    feat = run_checks(feat, trips, set(p.edge_ids.tolist()), rows, pre_bad, STRICT, out, link_bad, speeds, hist_ids, res)
    out.parent.mkdir(parents=True, exist_ok=True)
    sp_out = out.with_name(f"speeds_{folder}.parquet")
    for f_old in (out, out.with_suffix(".parquet"), sp_out):
        if f_old.exists():
            stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(f_old.stat().st_mtime))
            kept = f_old.with_name(f"{f_old.stem}_preserved_{stamp}{f_old.suffix}")
            f_old.rename(kept)
            print(f"[features] existing {f_old.name} kept as {kept.name}", flush=True)
    feat.to_csv(out, index=False)
    feat.assign(TRIP_ID=feat.TRIP_ID.astype(str)).to_parquet(out.with_suffix(".parquet"), index=False)
    print(f"[features] saved {out.with_suffix('.parquet')}", flush=True)
    speeds.to_parquet(sp_out, index=False)
    print(f"[features] saved {sp_out}: {len(speeds)} speed records of {speeds.trip_id.nunique()} trips", flush=True)
    slots = [f"count_vehicle_{k}" for k in range(mm.SLOTS)]
    print(f"[features] saved {out}: {len(feat)} rows, {feat.TRIP_ID.nunique()} trips "
          f"({feat.TRIP_ID.nunique() / len(trips) * 100:.1f}% of {len(trips)}), {feat.way_id.nunique()} roads; "
          f"slot0 filled {(feat.count_vehicle_0 > 0).mean():.4f}, any slot {(feat[slots].sum(1) > 0).mean():.4f}",
          flush=True)


def run_all(folder, n, max_trips, strict):
    import subprocess
    logs = REPO / "internal_analysis" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    procs = []
    for i in range(n):
        cmd = [sys.executable, "-u", __file__, folder, "shard", str(i), str(n)] + ([str(max_trips)] if max_trips else [])
        procs.append(subprocess.Popen(cmd, stdout=open(logs / f"mm_{folder}_shard_{i}.log", "w"),
                                      stderr=subprocess.STDOUT, env={**os.environ, "OMP_NUM_THREADS": "1"}))
    codes = [p.wait() for p in procs]
    for i in range(n):
        print((logs / f"mm_{folder}_shard_{i}.log").read_text().strip().splitlines()[-1], flush=True)
    if any(codes):
        raise SystemExit(f"shard exit codes {codes}; see internal_analysis/logs/mm_{folder}_shard_<i>.log")
    print(f"[all] {n} shards done in {time.time() - t0:.0f} s", flush=True)
    global STRICT
    STRICT = strict
    features(folder, n, max_trips)
    print(f"[all] finished in {time.time() - t0:.0f} s", flush=True)


STRICT = False

if __name__ == "__main__":
    folder, step = sys.argv[1], sys.argv[2]
    if step == "shard":
        shard(folder, int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]) if len(sys.argv) > 5 else None)
    elif step == "all":
        import argparse
        ap = argparse.ArgumentParser(prog="run_map_matching.py <folder> all")
        ap.add_argument("--shards", type=int, default=12)
        ap.add_argument("--max-trips", type=int, default=None)
        ap.add_argument("--strict", action="store_true", help="stop on any invalid trip instead of dropping it")
        a = ap.parse_args(sys.argv[3:])
        run_all(folder, a.shards, a.max_trips, a.strict)
    else:
        STRICT = "--strict" in sys.argv
        args = [x for x in sys.argv[3:] if x != "--strict"]
        features(folder, int(args[0]), int(args[1]) if len(args) > 1 else None)

