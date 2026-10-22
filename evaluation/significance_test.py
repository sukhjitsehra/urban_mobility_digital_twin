import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


BASE_DIR  = Path(__file__).resolve().parents[1]
DATA_ROOT = BASE_DIR / "data"
RESULTS_ROOT = BASE_DIR / "results"
CITIES = ["Porto", "San Francisco", "Chengdu", "Daegu"]
CITY_FOLDERS = {
    "Porto":         "porto",
    "San Francisco": "sf",
    "Chengdu":       "chengdu",
    "Daegu":         "daegu",
    "Ann Arbor":     "annarbor",
    "SF Bus":        "sfbus",
}


def city_dir(city: str) -> Path:
    return DATA_ROOT / CITY_FOLDERS.get(city, city)


def results_dir(city: str) -> Path:
    return RESULTS_ROOT / CITY_FOLDERS.get(city, city)


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
                f"Unknown dataset '{raw}'. Choices: display names {CITIES} "
                f"or short codes {list(CITY_FOLDERS.values())}.")
        out.append(lookup[raw.lower()])
    return out


def _bootstrap_ci_mean_diff(diffs, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(diffs)
    means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[i] = diffs[idx].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def _cohens_dz(diffs):
    s = float(np.std(diffs, ddof=1))
    if s == 0.0:
        return float("nan")
    return float(np.mean(diffs) / s)


def _paired_tests(err_A, err_B, label_A, label_B, level, n_boot):
    err_A = np.asarray(err_A, dtype=np.float64)
    err_B = np.asarray(err_B, dtype=np.float64)
    if len(err_A) != len(err_B):
        raise ValueError(f"length mismatch: {len(err_A)} vs {len(err_B)}")
    diffs = err_A - err_B
    n = len(diffs)

    t_stat, t_p = stats.ttest_rel(err_A, err_B)
    try:
        w_stat, w_p = stats.wilcoxon(err_A, err_B, zero_method="wilcox")
    except ValueError:
        w_stat, w_p = float("nan"), 1.0

    d_z = _cohens_dz(diffs)
    ci_lo, ci_hi = _bootstrap_ci_mean_diff(diffs, n_boot=n_boot)

    return {
        "level":               level,
        "model_A":             label_A,
        "model_B":             label_B,
        "n":                   int(n),
        "mean_err_A":          float(err_A.mean()),
        "mean_err_B":          float(err_B.mean()),
        "mean_diff_A_minus_B": float(diffs.mean()),
        "ci95_lo":             ci_lo,
        "ci95_hi":             ci_hi,
        "t_stat":              float(t_stat),
        "t_pvalue":            float(t_p),
        "wilcoxon_stat":       float(w_stat),
        "wilcoxon_pvalue":     float(w_p),
        "cohens_dz":           d_z,
        "sig_at_0.05":         bool(t_p < 0.05 and w_p < 0.05),
    }


def _trip_level_pairs(eval_trips_path, models, n_boot):
    df = pd.read_csv(eval_trips_path)
    missing = [m for m in models if m not in df["model"].unique()]
    if missing:
        raise ValueError(f"missing models in {eval_trips_path}: {missing}")
    wide = df.pivot(index="TRIP_ID", columns="model", values="abs_err_s")
    wide = wide.dropna(subset=models)
    rows = []
    for A, B in combinations(models, 2):
        rows.append(_paired_tests(
            wide[A].to_numpy(), wide[B].to_numpy(), A, B, "trip", n_boot))
    return rows, len(wide)


def _link_level_pairs(eval_links_path, models, n_boot):
    df = pd.read_csv(eval_links_path)
    missing = [m for m in models if m not in df["model"].unique()]
    if missing:
        raise ValueError(f"missing models in {eval_links_path}: {missing}")
    df["link_idx"] = df.groupby(["model", "TRIP_ID"]).cumcount()
    wide = df.pivot_table(index=["TRIP_ID", "link_idx"],
                          columns="model", values="abs_err_s")
    wide = wide.dropna(subset=models)
    rows = []
    for A, B in combinations(models, 2):
        rows.append(_paired_tests(
            wide[A].to_numpy(), wide[B].to_numpy(), A, B, "link", n_boot))
    return rows, len(wide)


def _filter_by_reference(rows, reference):
    out = []
    for r in rows:
        if reference not in (r["model_A"], r["model_B"]):
            continue
        if r["model_A"] == reference:
            r2 = dict(r)
            r2["model_A"], r2["model_B"] = r["model_B"], r["model_A"]
            r2["mean_err_A"], r2["mean_err_B"] = r["mean_err_B"], r["mean_err_A"]
            r2["mean_diff_A_minus_B"] = -r["mean_diff_A_minus_B"]
            r2["ci95_lo"], r2["ci95_hi"] = -r["ci95_hi"], -r["ci95_lo"]
            r2["t_stat"]    = -r["t_stat"]
            r2["cohens_dz"] = -r["cohens_dz"]
            out.append(r2)
        else:
            out.append(r)
    return out


def run_city(city, models, reference, n_boot):
    out_dir = results_dir(city) / "eval_outputs"
    trips_csv = out_dir / "eval_trips.csv"
    links_csv = out_dir / "eval_links.csv"
    if not trips_csv.exists():
        print(f"[{city}] SKIP — {trips_csv} not found")
        return None

    print(f"\n[{city}] trip-level paired tests ...")
    trip_rows, n_trips = _trip_level_pairs(trips_csv, models, n_boot)
    print(f"[{city}] paired n_trips = {n_trips}")

    link_rows = []
    if links_csv.exists():
        print(f"[{city}] link-level paired tests ...")
        link_rows, n_links = _link_level_pairs(links_csv, models, n_boot)
        print(f"[{city}] paired n_links = {n_links}")
    else:
        print(f"[{city}] no eval_links.csv; skipping link-level tests")

    rows = trip_rows + link_rows
    if reference:
        rows = _filter_by_reference(rows, reference)

    df = pd.DataFrame(rows)
    df.insert(0, "city", city)
    out_path = out_dir / "significance.csv"
    df.to_csv(out_path, index=False)
    print(f"[{city}] saved {out_path}")
    return df


MODEL_FILES = {
    "LinkETA":  ["linketa"],
    "ConSTGAT": ["constgat"],
    "DuETA":    ["dueta"],
    "STHR":     ["sthr"],
    "TransETA": ["transeta"],
    "DeeprETA": ["deepreta"],
    "GSPINN":   ["gspinn"],
}
ROUTE_METRICS = ["route_mape", "route_mae", "route_rmse", "route_wape"]
LINK_METRICS = ["link_mae", "link_rmse", "link_wape", "link_mape"]
T_MIN_LINK, L_MIN_LINK = 1.5, 10.0


def _read_df_all(city):
    d = pd.read_csv(city_dir(city) / f"df_all_{CITY_FOLDERS.get(city, city)}.csv",
                    usecols=["TRIP_ID", "enter_ts", "departure_time", "link_len_m", "link_travel_time_s"])
    d["TRIP_ID"] = d.TRIP_ID.astype("int64")
    return d


def _departures(d, city, trip_ids):
    d = d.drop_duplicates("TRIP_ID")
    dep = pd.Series(pd.to_datetime(d.departure_time, utc=True, format="mixed").to_numpy(), index=d.TRIP_ID.to_numpy())
    out = dep.reindex(np.asarray(trip_ids, dtype="int64"))
    if out.isna().any():
        raise ValueError(f"{city}: {int(out.isna().sum())} test trips without a departure time")
    return pd.DatetimeIndex(out.to_numpy())


def _block_ids(dep, hours):
    start = (dep.hour // hours) * hours
    return np.array([f"{a} {b:02d}" for a, b in zip(dep.strftime("%Y-%m-%d"), start)])


def _attach_link_len(d, lk, city, name):
    a = d[d.TRIP_ID.isin(lk.TRIP_ID.unique())]
    a = a.assign(k=a.groupby("TRIP_ID").cumcount(),
                 g=a.groupby("TRIP_ID").enter_ts.transform(lambda x: (x != x.shift()).cumsum()))
    lk = lk.assign(k=lk.groupby("TRIP_ID").cumcount()).merge(a[["TRIP_ID", "k", "g"]], on=["TRIP_ID", "k"], how="left")
    if lk.g.isna().any():
        raise ValueError(f"{city} {name}: eval_links has more links per trip than df_all")
    for x in (a, lk):
        x["r"] = x.sort_values(["TRIP_ID", "g", x.columns[x.columns.isin(["true_tt_link", "link_travel_time_s"])][0]],
                               kind="stable").groupby(["TRIP_ID", "g"]).cumcount()
    m = lk.merge(a[["TRIP_ID", "g", "r", "link_len_m", "link_travel_time_s"]], on=["TRIP_ID", "g", "r"], how="left")
    dt = np.abs(m.true_tt_link - m.link_travel_time_s)
    if m.link_len_m.isna().any() or np.nanmax(dt) > 1e-2:
        raise ValueError(f"{city} {name}: eval_links rows do not align with df_all (max dt {np.nanmax(dt):.3g})")
    moved = int((m.k.to_numpy() != m.merge(a[["TRIP_ID", "g", "r", "k"]], on=["TRIP_ID", "g", "r"],
                                           how="left").k_y.to_numpy()).sum())
    if moved:
        print(f"[{city}] {name}: {moved} links re-matched inside equal-enter_ts groups")
    return m.drop(columns=["k", "g", "r", "link_travel_time_s"])


def _load_city(city):
    out_dir = results_dir(city) / "eval_outputs"
    models, trips, links, tags = {}, {}, {}, {}
    for name, cand in MODEL_FILES.items():
        for tag in cand:
            p = out_dir / f"eval_trips_{tag}.csv"
            if p.exists():
                t = pd.read_csv(p, usecols=["TRIP_ID", "y_true", "y_pred", "route_len_m"])
                t["TRIP_ID"] = t.TRIP_ID.astype("int64")
                trips[name], tags[name] = t.set_index("TRIP_ID"), tag
                pl = out_dir / f"eval_links_{tag}.csv"
                if pl.exists():
                    lk = pd.read_csv(pl, usecols=["TRIP_ID", "true_tt_link", "pred_tt_link"])
                    if lk.pred_tt_link.notna().any():
                        lk["TRIP_ID"] = lk.TRIP_ID.astype("int64")
                        links[name] = lk
                break
    if "LinkETA" not in trips:
        raise ValueError(f"{city}: no LinkETA predictions")
    ref = trips["LinkETA"]
    for name, t in trips.items():
        if len(t) != len(ref) or not t.index.isin(ref.index).all():
            raise ValueError(f"{city}: {name} test trips differ from LinkETA's ({len(t)} vs {len(ref)})")
        dy = np.abs(t.y_true.reindex(ref.index).to_numpy() - ref.y_true.to_numpy())
        if np.nanmax(dy) > 1e-2:
            raise ValueError(f"{city}: {name} true route times differ from LinkETA's (max {np.nanmax(dy):.3g} s)")
    return ref.index.to_numpy(), trips, links, tags


def _route_block_sums(t, ids, blocks, uniq):
    t = t.reindex(ids)
    e = (t.y_pred - t.y_true).to_numpy(dtype=float)
    y = t.y_true.to_numpy(dtype=float)
    pos = y > 0
    cols = {"n": np.ones_like(e), "abs": np.abs(e), "sq": e * e, "y": y,
            "ape": np.where(pos, np.abs(e) / np.where(pos, y, 1.0), 0.0), "npos": pos.astype(float)}
    df = pd.DataFrame(cols).assign(b=blocks).groupby("b").sum().reindex(uniq).fillna(0.0)
    return df


def _link_block_sums(lk, trip_block, uniq):
    b = trip_block.reindex(lk.TRIP_ID.to_numpy()).to_numpy()
    e = (lk.pred_tt_link - lk.true_tt_link).to_numpy(dtype=float)
    y = lk.true_tt_link.to_numpy(dtype=float)
    ok = (y >= T_MIN_LINK) & (lk.link_len_m.to_numpy(dtype=float) >= L_MIN_LINK)
    cols = {"n": np.ones_like(e), "abs": np.abs(e), "sq": e * e, "y": np.abs(y),
            "ape": np.where(ok, np.abs(e) / np.where(ok, y, 1.0), 0.0), "npos": ok.astype(float)}
    return pd.DataFrame(cols).assign(b=b).groupby("b").sum().reindex(uniq).fillna(0.0)


def _metrics_from_sums(S, W, prefix):
    with np.errstate(divide="ignore", invalid="ignore"):
        return _metrics_from_sums_(S, W, prefix)


def _metrics_from_sums_(S, W, prefix):
    A = W @ S.to_numpy()
    c = {k: A[:, i] for i, k in enumerate(S.columns)}
    return {f"{prefix}_mae": c["abs"] / c["n"], f"{prefix}_rmse": np.sqrt(c["sq"] / c["n"]),
            f"{prefix}_wape": c["abs"] / c["y"],
            f"{prefix}_mape": c["ape"] / c["npos"]}


def _holm(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def _unit_dz(t_base, t_ref, ids, lk_base=None, lk_ref=None):
    def dz(x):
        x = x[np.isfinite(x)]
        return float(np.mean(x) / np.std(x, ddof=1)) if len(x) > 1 and np.std(x, ddof=1) > 0 else np.nan
    out = {}
    a, b = t_base.reindex(ids), t_ref.reindex(ids)
    y = b.y_true.to_numpy(dtype=float)
    ea, eb = (a.y_pred - a.y_true).to_numpy(dtype=float), (b.y_pred - b.y_true).to_numpy(dtype=float)
    pos = y > 0
    out["route_mae"] = dz(np.abs(ea) - np.abs(eb))
    out["route_rmse"] = dz(ea ** 2 - eb ** 2)
    out["route_mape"] = dz((np.abs(ea) - np.abs(eb))[pos] / y[pos])
    if lk_base is not None and lk_ref is not None:
        if not (np.array_equal(lk_base.TRIP_ID.to_numpy(), lk_ref.TRIP_ID.to_numpy())
                and np.allclose(lk_base.true_tt_link.to_numpy(), lk_ref.true_tt_link.to_numpy(), atol=1e-2)):
            raise ValueError("link rows of baseline and LinkETA are not in the same order")
        yl = lk_ref.true_tt_link.to_numpy(dtype=float)
        la = np.abs((lk_base.pred_tt_link - lk_base.true_tt_link).to_numpy(dtype=float))
        lb = np.abs((lk_ref.pred_tt_link - lk_ref.true_tt_link).to_numpy(dtype=float))
        ok = (yl >= T_MIN_LINK) & (lk_ref.link_len_m.to_numpy(dtype=float) >= L_MIN_LINK)
        out["link_mae"] = dz(la - lb)
        out["link_rmse"] = dz(la ** 2 - lb ** 2)
        out["link_mape"] = dz((la - lb)[ok] / yl[ok])
    return out


def run_city_blocks(city, n_boot, seed, hours_list=(1,)):
    ids, trips, links, tags = _load_city(city)
    d = _read_df_all(city)
    dep = _departures(d, city, ids)
    links = {n: _attach_link_len(d, lk, city, n) for n, lk in links.items()}
    del d
    out_dir = results_dir(city) / "eval_outputs"
    if "LinkETA" in links:
        key = ["TRIP_ID", "true_tt_link"]
        ref_l = links["LinkETA"].assign(_r=lambda x: x.groupby(key).cumcount())
        for n in list(links):
            if n != "LinkETA":
                o = links[n].assign(_r=lambda x: x.groupby(key).cumcount())
                o = ref_l[key + ["_r"]].merge(o, on=key + ["_r"], how="left")
                if o.pred_tt_link.isna().any():
                    raise ValueError(f"{city} {n}: link rows cannot be paired with LinkETA's")
                links[n] = o.drop(columns="_r")
    dz_unit = {n: _unit_dz(trips[n], trips["LinkETA"], ids, links.get(n), links.get("LinkETA"))
               for n in trips if n != "LinkETA"}

    mets, tests = [], []
    for hours in hours_list:
        blocks = _block_ids(dep, hours)
        uniq = np.unique(blocks)
        trip_block = pd.Series(blocks, index=ids)
        rng = np.random.default_rng(seed)
        draws = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
        W = np.zeros((n_boot, len(uniq)))
        np.add.at(W, (np.repeat(np.arange(n_boot), len(uniq)), draws.ravel()), 1.0)
        W_full = np.ones((1, len(uniq)))
        W_eye = np.eye(len(uniq))
        rows, samples, blockwise = [], {}, {}
        for name, t in trips.items():
            S = _route_block_sums(t, ids, blocks, uniq)
            full = _metrics_from_sums(S, W_full, "route")
            js = out_dir / f"eval_metrics_{tags[name]}.json"
            if not js.exists():
                raise ValueError(f"{city} {name}: {js.name} missing, cannot check the predictions file")
            saved = json.loads(js.read_text())[name]
            bs = _metrics_from_sums(S, W, "route")
            per_block = _metrics_from_sums(S, W_eye, "route")
            if name in links:
                LS = _link_block_sums(links[name], trip_block, uniq)
                full.update(_metrics_from_sums(LS, W_full, "link"))
                bs.update(_metrics_from_sums(LS, W, "link"))
                per_block.update(_metrics_from_sums(LS, W_eye, "link"))
            same_len = np.allclose(t.route_len_m.reindex(ids).to_numpy(), trips["LinkETA"].route_len_m.reindex(ids).to_numpy(),
                                   rtol=1e-4, atol=1e-2)
            if not same_len and "link_mape" in full and hours == hours_list[0]:
                print(f"[{city}] {name}: own link lengths differ from df_all; link MAPE (common mask) "
                      f"{full['link_mape'][0]:.4f}, saved with own lengths {saved['link_mape']:.4f}")
            for k in [k for k in full if not k.endswith("_wape") and (same_len or k != "link_mape")]:
                if abs(full[k][0] - saved[k]) > 1e-4 * max(1.0, abs(saved[k])):
                    raise ValueError(f"{city} {name}: recomputed {k} {full[k][0]:.6f} != saved {saved[k]:.6f}")
            samples[name], blockwise[name] = bs, per_block
            for k, v in bs.items():
                rows.append(dict(city=city, block_hours=hours, model=name, source=f"eval_trips_{tags[name]}.csv",
                                 metric=k, full_sample=float(full[k][0]), boot_mean=float(np.mean(v)),
                                 boot_sd=float(np.std(v, ddof=1)),
                                 ci95_lo=float(np.percentile(v, 2.5)), ci95_hi=float(np.percentile(v, 97.5)),
                                 n_trips=len(ids), n_blocks=len(uniq), n_boot=n_boot))
        mets.append(pd.DataFrame(rows))
        ref = samples["LinkETA"]
        for name, bs in samples.items():
            if name == "LinkETA":
                continue
            for k in bs:
                if k not in ref:
                    continue
                gap = bs[k] - ref[k]
                p = min(1.0, 2 * min((np.sum(gap <= 0) + 1) / (n_boot + 1), (np.sum(gap >= 0) + 1) / (n_boot + 1)))
                sd = float(np.std(gap, ddof=1))
                p_se = float(2 * stats.norm.sf(abs(np.mean(gap)) / sd)) if sd > 0 else 0.0
                bg = blockwise[name][k] - blockwise["LinkETA"][k]
                bg = bg[np.isfinite(bg)]
                dz = float(np.mean(bg) / np.std(bg, ddof=1)) if len(bg) > 1 and np.std(bg, ddof=1) > 0 else np.nan
                tests.append(dict(city=city, block_hours=hours, baseline=name, metric=k,
                                  linketa=float(np.mean(ref[k])), baseline_value=float(np.mean(bs[k])),
                                  gap_mean=float(np.mean(gap)), gap_ci95_lo=float(np.percentile(gap, 2.5)),
                                  gap_ci95_hi=float(np.percentile(gap, 97.5)), gap_sd=sd, p_boot=p, p_se=p_se,
                                  d_z_unit=dz_unit[name].get(k, np.nan), d_z_blocks=dz,
                                  blocks_baseline_worse=float(np.mean(bg > 0)), n_blocks=len(uniq), n_boot=n_boot))
        print(f"[{city}] {hours} h blocks: {len(ids)} test trips in {len(uniq)} blocks")
    print(f"[{city}] models {list(trips)}; link-level for {list(links)}")
    return pd.concat(mets, ignore_index=True), pd.DataFrame(tests)


def main_blocks(args):
    cities = _resolve_datasets(args.datasets)
    hours_list = [int(h) for h in args.block_hours.split(",")]
    mets, tests = [], []
    for c in cities:
        m, t = run_city_blocks(c, args.n_boot, args.seed, hours_list)
        mets.append(m)
        tests.append(t)
    T = pd.concat(tests, ignore_index=True)
    M = pd.concat(mets, ignore_index=True)
    T["p_holm"] = np.nan
    T["p_holm_all"] = np.nan
    for h, g in T.groupby("block_hours"):
        for k, idx in g.groupby("metric").groups.items():
            T.loc[idx, "p_holm"] = _holm(T.loc[idx, "p_boot"].to_numpy())
        T.loc[g.index, "p_holm_all"] = _holm(g.p_se.to_numpy())
    ex0 = (T.gap_ci95_lo > 0) | (T.gap_ci95_hi < 0)
    dis = int(((T.p_se < 0.05) != ex0).sum())
    print(f"[blocks] p_se<0.05 vs 95% range excluding 0: {dis} of {len(T)} tests disagree")
    main_h = hours_list[0]
    for c in cities:
        T[(T.city == c) & (T.block_hours == main_h)].to_csv(results_dir(c) / "eval_outputs" / "bootstrap_tests.csv", index=False)
        M[(M.city == c) & (M.block_hours == main_h)].to_csv(results_dir(c) / "eval_outputs" / "bootstrap_metrics.csv", index=False)
    out = RESULTS_ROOT / "bootstrap_summary"
    out.mkdir(exist_ok=True)
    M.to_csv(out / "bootstrap_metrics_all.csv", index=False)
    T.to_csv(out / "bootstrap_tests_all.csv", index=False)
    print(f"[blocks] saved {out}/bootstrap_metrics_all.csv and bootstrap_tests_all.csv "
          f"(per-city files: {main_h} h blocks)")
    return M, T


def parse_args():
    p = argparse.ArgumentParser(
        description="Paired statistical significance tests on per-trip / per-link predictions.")
    p.add_argument("--datasets", "--cities", dest="datasets", default="all",
                   help="Comma-separated city names/codes or 'all'.")
    p.add_argument("--models", default="ConSTGAT,DuETA,LinkETA",
                   help="Comma-separated models to compare (must appear in eval_trips.csv).")
    p.add_argument("--reference", default=None,
                   help="If set, only output rows comparing other models against this "
                        "reference. Differences are oriented as (other - reference); "
                        "positive value => the reference is BETTER.")
    p.add_argument("--bootstrap-n", dest="n_boot", type=int, default=1000,
                   help="Number of bootstrap resamples for the 95%% CI. Default 1000.")
    p.add_argument("--mode", choices=["pairs", "blocks"], default="pairs",
                   help="pairs = original trip/link-level tests on eval_trips.csv; blocks = hour-block bootstrap.")
    p.add_argument("--seed", type=int, default=42, help="Bootstrap seed (blocks mode).")
    p.add_argument("--block-hours", default="1,3,6,12,24",
                   help="Block lengths in clock hours (blocks mode); the first is the main result, the others a "
                        "sensitivity check for correlation across hours (24 = day blocks).")
    return p.parse_args()


def _print_headline(merged):
    cols = ["city", "level", "model_A", "model_B", "n",
            "mean_err_A", "mean_err_B", "mean_diff_A_minus_B",
            "ci95_lo", "ci95_hi", "t_pvalue", "wilcoxon_pvalue",
            "cohens_dz", "sig_at_0.05"]
    print("\n=== headline ===")
    with pd.option_context("display.float_format", "{:.4g}".format,
                           "display.width", 220,
                           "display.max_columns", None):
        print(merged[cols].to_string(index=False))


def main():
    args = parse_args()
    if args.mode == "blocks":
        main_blocks(args)
        print("[DONE]")
        return
    cities = _resolve_datasets(args.datasets)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    print(f"cities:      {cities}")
    print(f"models:      {models}")
    print(f"reference:   {args.reference or '(none — all pairwise)'}")
    print(f"bootstrap n: {args.n_boot}")

    all_dfs = []
    for c in cities:
        df = run_city(c, models, args.reference, args.n_boot)
        if df is not None:
            all_dfs.append(df)

    if all_dfs:
        merged = pd.concat(all_dfs, ignore_index=True)
        _print_headline(merged)
    print("\n[DONE]")


if __name__ == "__main__":
    main()

