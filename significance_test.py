#!/usr/bin/env python
# coding: utf-8
"""
significance_test.py — paired statistical tests on Meta_Learning predictions.

For each city in --cities, loads the per-trip and per-link prediction CSVs
written by main.py (run_evaluation) and runs paired statistical tests for
every method pair (default MetaST / MetaTTE / MetaTraffic):

  - Paired t-test on absolute errors (parametric, scipy.stats.ttest_rel)
  - Wilcoxon signed-rank test (non-parametric, robust to heavy tails)
  - Cohen's d_z (effect size for paired samples)
  - 95%% percentile bootstrap CI on the mean of paired differences

Outputs (per city, written to checkpoints/<run>/<city>/eval_outputs/):
  - significance.csv    one row per (model_A, model_B, level)

Conventions:
  - diff = err_A - err_B  → positive diff means model_A is WORSE (more error).
  - With --reference R, every row is oriented as (other - reference) so a
    positive diff / cohens_dz means the reference is better.

Run:
    python significance_test.py --run porto_sf_source_chengdu_target
    python significance_test.py --run porto_sf_source_chengdu_target --cities chengdu
    python significance_test.py --run porto_sf_source_chengdu_target --reference MetaTraffic
    python significance_test.py --run porto_sf_source_chengdu_target --bootstrap-n 2000
"""

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Reuse path helpers from main.py (BASE_DIR + run/city dir conventions).
from main import BASE_DIR, CHECKPOINTS_ROOT, CITY_CONFIGS


# ---- core stats ----------------------------------------------------------
def _bootstrap_ci_mean_diff(diffs, n_boot=1000, seed=42):
    """Percentile bootstrap 95% CI on mean(diff)."""
    rng = np.random.default_rng(seed)
    n = len(diffs)
    means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[i] = diffs[idx].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def _cohens_dz(diffs):
    """Cohen's d_z = mean(diff) / std(diff, ddof=1)."""
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
        # All-zero differences make Wilcoxon undefined; treat as non-significant.
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


# ---- per-level pairing ---------------------------------------------------
def _trip_level_pairs(eval_trips_path, models, split, n_boot):
    df = pd.read_csv(eval_trips_path)
    if "split" in df.columns:
        df = df[df["split"] == split]
    missing = [m for m in models if m not in df["model"].unique()]
    if missing:
        raise ValueError(f"missing models in {eval_trips_path}: {missing}")
    wide = df.pivot_table(index="TRIP_ID", columns="model",
                          values="abs_err_s", aggfunc="mean")
    wide = wide.dropna(subset=models)
    rows = []
    for A, B in combinations(models, 2):
        rows.append(_paired_tests(
            wide[A].to_numpy(), wide[B].to_numpy(), A, B, "trip", n_boot))
    return rows, len(wide)


def _link_level_pairs(eval_links_path, models, split, n_boot):
    df = pd.read_csv(eval_links_path)
    if "split" in df.columns:
        df = df[df["split"] == split]
    missing = [m for m in models if m not in df["model"].unique()]
    if missing:
        raise ValueError(f"missing models in {eval_links_path}: {missing}")
    # Pair links by their position in each trip — rows are emitted in the same
    # (TRIP_ID, link_idx) order by every collector, so cumcount aligns them.
    df = df.copy()
    df["link_idx"] = df.groupby(["model", "TRIP_ID"]).cumcount()
    wide = df.pivot_table(index=["TRIP_ID", "link_idx"], columns="model",
                          values="abs_err_s", aggfunc="mean")
    wide = wide.dropna(subset=models)
    rows = []
    for A, B in combinations(models, 2):
        rows.append(_paired_tests(
            wide[A].to_numpy(), wide[B].to_numpy(), A, B, "link", n_boot))
    return rows, len(wide)


# ---- reference orientation ----------------------------------------------
def _filter_by_reference(rows, reference):
    """Keep only pairs that include `reference`; orient with reference as B
    so a positive diff / d_z means the reference is BETTER."""
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


# ---- per-city orchestration ---------------------------------------------
def run_city(run_name, city, models, reference, split, n_boot):
    out_dir   = CHECKPOINTS_ROOT / run_name / city / "eval_outputs"
    trips_csv = out_dir / "eval_trips.csv"
    links_csv = out_dir / "eval_links.csv"
    if not trips_csv.exists():
        print(f"[{city}] SKIP — {trips_csv} not found")
        return None

    print(f"\n[{city}] trip-level paired tests (split={split}) ...")
    trip_rows, n_trips = _trip_level_pairs(trips_csv, models, split, n_boot)
    print(f"[{city}] paired n_trips = {n_trips}")

    link_rows = []
    if links_csv.exists():
        print(f"[{city}] link-level paired tests (split={split}) ...")
        try:
            link_rows, n_links = _link_level_pairs(links_csv, models, split, n_boot)
            print(f"[{city}] paired n_links = {n_links}")
        except ValueError as e:
            print(f"[{city}] link-level skipped: {e}")
    else:
        print(f"[{city}] no eval_links.csv; skipping link-level tests")

    rows = trip_rows + link_rows
    if reference:
        rows = _filter_by_reference(rows, reference)

    df = pd.DataFrame(rows)
    df.insert(0, "city", city)
    df.insert(1, "run", run_name)
    df.insert(2, "split", split)
    out_path = out_dir / "significance.csv"
    df.to_csv(out_path, index=False)
    print(f"[{city}] saved {out_path}")
    return df


# ---- CLI -----------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Paired statistical significance tests on Meta_Learning "
                    "per-trip / per-link predictions.")
    p.add_argument("--run", required=False, default=None,
                   help="Run folder name under checkpoints/. Default: most "
                        "recently modified run folder.")
    p.add_argument("--cities", default=None,
                   help="Comma-separated city keys. Default: every city "
                        "subfolder in the run with eval_trips.csv.")
    p.add_argument("--models", default="MetaST,MetaTTE,MetaETA",
                   help="Comma-separated models to compare (must appear in eval_trips.csv).")
    p.add_argument("--split", default="test", choices=("train", "val", "test"),
                   help="Which split's predictions to test (default: test).")
    p.add_argument("--reference", default=None,
                   help="If set, only output rows comparing other models against this "
                        "reference. Differences are oriented (other - reference); "
                        "positive => the reference is BETTER.")
    p.add_argument("--bootstrap-n", dest="n_boot", type=int, default=1000,
                   help="Number of bootstrap resamples for the 95%% CI. Default 1000.")
    return p.parse_args()


def _autodiscover_run():
    if not CHECKPOINTS_ROOT.exists():
        raise SystemExit(f"No checkpoints/ folder at {CHECKPOINTS_ROOT}")
    runs = sorted([p for p in CHECKPOINTS_ROOT.iterdir()
                   if p.is_dir() and "_source" in p.name],   # [2026-06-07] sources-keyed runs
                  key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        raise SystemExit("No run folders found under checkpoints/.")
    return runs[0].name


def _print_headline(merged):
    cols = ["city", "level", "model_A", "model_B", "n",
            "mean_err_A", "mean_err_B", "mean_diff_A_minus_B",
            "ci95_lo", "ci95_hi", "t_pvalue", "wilcoxon_pvalue",
            "cohens_dz", "sig_at_0.05"]
    print("\n=== headline ===")
    cols_present = [c for c in cols if c in merged.columns]
    print(merged[cols_present].to_string(index=False))


def main():
    args = parse_args()
    run_name = args.run or _autodiscover_run()
    print(f"[SIG] run = {run_name}")

    run_dir = CHECKPOINTS_ROOT / run_name
    if not run_dir.is_dir():
        raise SystemExit(f"Run folder not found: {run_dir}")

    if args.cities:
        cities = [c.strip() for c in args.cities.split(",") if c.strip()]
    else:
        cities = sorted(p.name for p in run_dir.iterdir()
                        if p.is_dir() and p.name in CITY_CONFIGS
                        and (p / "eval_outputs" / "eval_trips.csv").exists())
        if not cities:
            raise SystemExit(
                f"No city subfolders with eval_trips.csv found in {run_dir}.")

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    print(f"[SIG] cities={cities}  models={models}  split={args.split}  "
          f"reference={args.reference}  bootstrap_n={args.n_boot}")

    all_dfs = []
    for c in cities:
        df = run_city(run_name, c, models, args.reference, args.split, args.n_boot)
        if df is not None:
            all_dfs.append(df)

    if all_dfs:
        merged = pd.concat(all_dfs, ignore_index=True)
        _print_headline(merged)

    print("\n[DONE]")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[ABORTED] interrupted by user.")
        sys.exit(130)
