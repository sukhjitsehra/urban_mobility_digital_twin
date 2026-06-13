#!/usr/bin/env python
# coding: utf-8
"""
figures.py — IEEE-style figures for the cross-city meta-learning experiments.

Consumes the per-city eval_outputs (written by main.py run_evaluation and
benchmark.py) and produces six figure families comparing MetaST, MetaTTE and
MetaTraffic — mirroring LinkETA/figures.py family-for-family.

Inputs (per city, under checkpoints/<run>/<city>/eval_outputs/):
    eval_trips.csv               from main.py
    eval_links.csv               from main.py
    benchmark_by_link_count.csv  from benchmark.py
    benchmark_scalability.csv    from benchmark.py --scalability   (optional)

Outputs (per city, written to checkpoints/<run>/<city>/figs/):
    1. reliability_<city>.{pdf,png}     calibration curves (with BCE in legend)
    2. mae_vs_length_<city>             grouped MAE bars + MAPE twin-axis line
    3. mape_vs_length_<city>            grouped MAPE bars
    4. mae_vs_duration_<city>           grouped MAE bars + MAPE twin-axis line
    5. ecdf_<city>                      absolute-error ECDF with P90 markers
    6. latency_trip_<city>              rolling MAPE + ms/trip-per-bin overlay
    7. scalability_<city>               total inference time vs n_links

Run:
    python figures.py                                    # all runs, all cities, split=test
    python figures.py --run porto_sf_source_chengdu_target
    python figures.py --run porto_sf_source_chengdu_target --cities chengdu --split test
"""

import argparse
from math import ceil
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import MultipleLocator


# --- Paths (script-relative) ---------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
CHECKPOINTS_ROOT = BASE_DIR / "checkpoints"


# --- Style (notebook IEEE Plots + LinkETA house style) -------------------
META_ORDER = ["MetaST", "MetaTTE", "MetaETA"]   # [2026-06-06] MetaTraffic -> MetaETA (framework label)
META_COLOR = {
    "MetaST":  "#BB5566",
    "MetaTTE": "#004488",
    "MetaETA": "#DDAA33",
}
META_MARKER = {
    "MetaST":  "^",
    "MetaTTE": "o",
    "MetaETA": "s",
}

ACCURACY_MODELS = META_ORDER  # accuracy families plot all three
LATENCY_MODELS  = META_ORDER  # latency families plot all three

ONE_COL_IN = 3.5
RATIO = 0.62

mpl.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "Times", "Cambria", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
    "lines.linewidth":  1.4,
    "axes.linewidth":   0.6,
    "figure.dpi":       150,
    "savefig.dpi":      600,
    "axes.labelsize":   8.0,
    "xtick.labelsize":  7.0,
    "ytick.labelsize":  7.0,
    "legend.fontsize":  6.5,
    "axes.titlesize":   9.0,
})

LEN_BINS   = (0, 3, 6, 9, 12, 15, 18, 1e6)
LEN_LABELS = ["0-3", "3-6", "6-9", "9-12", "12-15", "15-18", "18+"]
DUR_BINS   = (0, 10, 20, 30, 40, 50, 60, 1e9)
DUR_LABELS = ["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60+"]


# =============================================================================
# Loaders + small helpers
# =============================================================================
def _eval_dir(run_dir: Path, city: str) -> Path:
    return run_dir / city / "eval_outputs"


def _figs_dir(run_dir: Path, city: str) -> Path:
    d = run_dir / city / "figs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _read_csv_optional(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"  [warn] failed to read {path}: {e}")
        return None


def load_eval_trips(run_dir: Path, city: str, split: str) -> Optional[pd.DataFrame]:
    df = _read_csv_optional(_eval_dir(run_dir, city) / "eval_trips.csv")
    if df is None or df.empty:
        return None
    if "split" in df.columns:
        df = df[df["split"] == split].copy()
    return df if not df.empty else None


def load_eval_links(run_dir: Path, city: str, split: str) -> Optional[pd.DataFrame]:
    df = _read_csv_optional(_eval_dir(run_dir, city) / "eval_links.csv")
    if df is None or df.empty:
        return None
    if "split" in df.columns:
        df = df[df["split"] == split].copy()
    return df if not df.empty else None


def load_benchmark_by_link_count(run_dir: Path, city: str) -> Optional[pd.DataFrame]:
    return _read_csv_optional(_eval_dir(run_dir, city) / "benchmark_by_link_count.csv")


def load_benchmark_scalability(run_dir: Path, city: str) -> Optional[pd.DataFrame]:
    return _read_csv_optional(_eval_dir(run_dir, city) / "benchmark_scalability.csv")


def _safe_mape(p, y, eps=1e-6):
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    return np.abs(p - y) / np.clip(np.abs(y), eps, None)


def _methods_in(df: pd.DataFrame) -> List[str]:
    have = set(df["model"].unique())
    return [m for m in META_ORDER if m in have]


def _bin_midpoint(bin_str):
    s = str(bin_str).strip()
    if s.endswith("+"):
        try:
            return float(s.rstrip("+")) * 1.5
        except ValueError:
            return float("nan")
    if "-" in s:
        try:
            lo, hi = s.split("-")
            return (float(lo) + float(hi)) / 2
        except ValueError:
            return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _save(fig, figs_dir: Path, name: str):
    base = figs_dir / name
    fig.savefig(f"{base}.pdf", bbox_inches="tight")
    fig.savefig(f"{base}.png", bbox_inches="tight", dpi=600)
    plt.close(fig)
    print(f"  [saved] {base.relative_to(BASE_DIR)}.{{pdf,png}}")


def _draw_graph_paper_grid(ax, lim_max, minor=1.0, major=5.0):
    """Minor grid every 1 unit, major every 5 units (LinkETA reliability style)."""
    top = ceil(lim_max / minor) * minor
    ax.set_xlim(0, top); ax.set_ylim(0, top)
    ax.xaxis.set_major_locator(MultipleLocator(major))
    ax.yaxis.set_major_locator(MultipleLocator(major))
    ax.xaxis.set_minor_locator(MultipleLocator(minor))
    ax.yaxis.set_minor_locator(MultipleLocator(minor))
    ax.grid(False); ax.set_axisbelow(True)
    xs_minor = np.arange(0, top + 1e-9, minor)
    ax.vlines(xs_minor, 0, top, colors="0.90", linewidth=0.3, zorder=0)
    ax.hlines(xs_minor, 0, top, colors="0.90", linewidth=0.3, zorder=0)
    xs_major = np.arange(0, top + 1e-9, major)
    ax.vlines(xs_major, 0, top, colors="0.78", linewidth=0.5, zorder=0)
    ax.hlines(xs_major, 0, top, colors="0.78", linewidth=0.5, zorder=0)


# =============================================================================
# Family 1 — Reliability (per-city calibration with BCE legend)
# =============================================================================
def plot_reliability(run_dir: Path, city: str, split: str, n_bins: int = 8):
    df = load_eval_trips(run_dir, city, split)
    if df is None:
        return
    df = df[df["model"].isin(ACCURACY_MODELS)].copy()
    df = df[(df["y_pred"] > 0) & (df["y_true"] > 0)]
    if df.empty:
        return
    df["pred_min"] = df["y_pred"] / 60.0
    df["true_min"] = df["y_true"] / 60.0

    # Shared x-cap = smallest per-model p99 prediction (LinkETA approach)
    p99 = df.groupby("model")["pred_min"].quantile(0.99)
    p99 = p99[p99.index.isin(ACCURACY_MODELS)]
    if p99.empty:
        return
    x_cap = float(p99.min())
    df_capped = df[df["pred_min"] <= x_cap].copy()
    if df_capped.empty:
        return

    edges = np.unique(np.quantile(df_capped["pred_min"].to_numpy(),
                                  np.linspace(0, 1, n_bins + 1)))
    if edges.size < 2:
        return

    rel_rows = []
    bce_map = {}
    for m in ACCURACY_MODELS:
        d = df_capped[df_capped["model"] == m]
        if d.empty:
            continue
        d2 = d.copy()
        d2["bin"] = pd.cut(d2["pred_min"], bins=edges, include_lowest=True)
        d2["bin"] = d2["bin"].cat.remove_unused_categories()
        grp = d2.groupby("bin", observed=True).agg(
            mean_pred=("pred_min", "mean"),
            mean_true=("true_min", "mean"),
            count=("pred_min", "size"),
        ).reset_index(drop=True)
        N = grp["count"].sum()
        bce_map[m] = float((grp["count"] / N
                            * (grp["mean_true"] - grp["mean_pred"]).abs()).sum()
                           ) if N > 0 else float("nan")
        for _, row in grp.iterrows():
            rel_rows.append({"model": m,
                             "mean_pred": row["mean_pred"],
                             "mean_true": row["mean_true"]})
    if not rel_rows:
        return
    rel = pd.DataFrame(rel_rows)

    fig, ax = plt.subplots(figsize=(ONE_COL_IN, ONE_COL_IN * RATIO))
    x_lim = float(rel["mean_pred"].max()) * 1.05
    y_lim = float(rel["mean_true"].max()) * 1.05
    frame = max(x_lim, y_lim)
    _draw_graph_paper_grid(ax, frame, minor=1.0, major=5.0)
    ax.set_xlim(0, x_lim); ax.set_ylim(0, y_lim)
    diag_end = min(x_lim, y_lim)
    ax.plot([0, diag_end], [0, diag_end],
            linestyle="--", linewidth=1.0, color="0.35", zorder=1)

    for m in ACCURACY_MODELS:
        g = rel[rel["model"] == m]
        if g.empty:
            continue
        bce = bce_map.get(m, float("nan"))
        label = f"{m} (BCE={bce:.2f} min)" if np.isfinite(bce) else m
        ax.plot(g["mean_pred"], g["mean_true"],
                linestyle="-", marker=META_MARKER[m], markersize=2.5,
                color=META_COLOR[m], label=label, zorder=2)

    ax.set_xlabel("Binned mean predicted (min)")
    ax.set_ylabel("Binned mean actual (min)")
    ax.legend(frameon=False, loc="upper left", ncol=1, fontsize=6,
              borderpad=0.2, handlelength=1.4)
    _save(fig, _figs_dir(run_dir, city), f"reliability_{city}")


# =============================================================================
# Families 2 & 3 — MAE / MAPE vs route length
# =============================================================================
def _prep_trips_for_length(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["model"].isin(ACCURACY_MODELS)].copy()
    df["km"] = df["route_len_m"] / 1000.0
    df["bin"] = pd.cut(df["km"], bins=LEN_BINS, right=False,
                       include_lowest=True, labels=LEN_LABELS)
    if "abs_err_s" not in df.columns:
        df["abs_err_s"] = (df["y_pred"] - df["y_true"]).abs()
    df["ape"] = _safe_mape(df["y_pred"], df["y_true"])
    return df


def plot_mae_vs_length(run_dir: Path, city: str, split: str):
    df = load_eval_trips(run_dir, city, split)
    if df is None or "route_len_m" not in df.columns:
        return
    df = _prep_trips_for_length(df)
    methods = _methods_in(df)
    mae_w = df.groupby(["bin", "model"], observed=True)["abs_err_s"] \
              .mean().unstack().reindex(LEN_LABELS)
    mape_w = df.groupby(["bin", "model"], observed=True)["ape"] \
               .mean().unstack().reindex(LEN_LABELS)

    fig, ax = plt.subplots(figsize=(ONE_COL_IN, ONE_COL_IN * RATIO))
    x = np.arange(len(LEN_LABELS))
    bw = 0.85 / max(len(methods), 1)
    y_max = 0.0
    for j, m in enumerate(methods):
        if m not in mae_w.columns:
            continue
        y = np.nan_to_num(mae_w[m].to_numpy(), nan=0.0)
        y_max = max(y_max, float(y.max()))
        ax.bar(x + j * bw, y, width=bw, color=META_COLOR[m], edgecolor="none")

    ax.set_xticks(x + (len(methods) - 1) * bw / 2)
    ax.set_xticklabels(LEN_LABELS, rotation=0)
    ax.set_xlabel("Route length (km)")
    ax.set_ylabel("MAE (s)")
    ax.tick_params(axis="x", which="major", labelsize=7)
    ax.grid(True, axis="y", color="0.88", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(0, y_max * 1.18 if y_max > 0 else 1)

    ax2 = ax.twinx()
    y2_max = 0.0
    for m in methods:
        if m not in mape_w.columns:
            continue
        y2 = mape_w[m].to_numpy() * 100.0
        finite = y2[np.isfinite(y2)]
        if finite.size:
            y2_max = max(y2_max, float(finite.max()))
        ax2.plot(x + (len(methods) - 1) * bw / 2, y2,
                 color=META_COLOR[m], marker=META_MARKER[m], ms=3.0,
                 linewidth=1.2, label=m)
    ax2.set_ylabel("MAPE (%)")
    ax2.set_ylim(0, max(10, (int(y2_max // 10) + 2) * 10))
    ax2.legend(loc="upper left", frameon=False, fontsize=6, ncol=2,
               borderpad=0.2, handlelength=1.2)
    _save(fig, _figs_dir(run_dir, city), f"mae_vs_length_{city}")


def plot_mape_vs_length(run_dir: Path, city: str, split: str):
    df = load_eval_trips(run_dir, city, split)
    if df is None or "route_len_m" not in df.columns:
        return
    df = _prep_trips_for_length(df)
    methods = _methods_in(df)
    mape_w = df.groupby(["bin", "model"], observed=True)["ape"] \
               .mean().unstack().reindex(LEN_LABELS) * 100.0

    fig, ax = plt.subplots(figsize=(ONE_COL_IN, ONE_COL_IN * RATIO))
    x = np.arange(len(LEN_LABELS))
    bw = 0.85 / max(len(methods), 1)
    y_max = 0.0
    for j, m in enumerate(methods):
        if m not in mape_w.columns:
            continue
        y = np.nan_to_num(mape_w[m].to_numpy(), nan=0.0)
        y_max = max(y_max, float(y.max()))
        ax.bar(x + j * bw, y, width=bw, color=META_COLOR[m], edgecolor="none",
               label=m)

    ax.set_xticks(x + (len(methods) - 1) * bw / 2)
    ax.set_xticklabels(LEN_LABELS, rotation=0)
    ax.set_xlabel("Route length (km)")
    ax.set_ylabel("MAPE (%)")
    ax.tick_params(axis="x", which="major", labelsize=7)
    ax.grid(True, axis="y", color="0.88", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(0, y_max * 1.18 if y_max > 0 else 1)
    ax.legend(loc="upper left", frameon=False, fontsize=6, ncol=2,
              borderpad=0.2, handlelength=1.2)
    _save(fig, _figs_dir(run_dir, city), f"mape_vs_length_{city}")


# =============================================================================
# Family 4 — MAE / MAPE vs trip duration
# =============================================================================
def plot_mae_vs_duration(run_dir: Path, city: str, split: str):
    df = load_eval_trips(run_dir, city, split)
    if df is None:
        return
    df = df[df["model"].isin(ACCURACY_MODELS)].copy()
    df["dur_min"] = df["y_true"] / 60.0
    df["bin"]     = pd.cut(df["dur_min"], bins=DUR_BINS, right=False,
                           include_lowest=True, labels=DUR_LABELS)
    if "abs_err_s" not in df.columns:
        df["abs_err_s"] = (df["y_pred"] - df["y_true"]).abs()
    df["ape"]     = _safe_mape(df["y_pred"], df["y_true"])
    methods = _methods_in(df)

    mae_w  = df.groupby(["bin", "model"], observed=True)["abs_err_s"] \
               .mean().unstack().reindex(DUR_LABELS)
    mape_w = df.groupby(["bin", "model"], observed=True)["ape"] \
               .mean().unstack().reindex(DUR_LABELS)

    fig, ax = plt.subplots(figsize=(ONE_COL_IN, ONE_COL_IN * RATIO))
    x = np.arange(len(DUR_LABELS))
    bw = 0.85 / max(len(methods), 1)
    y_max = 0.0
    for j, m in enumerate(methods):
        if m not in mae_w.columns:
            continue
        y = np.nan_to_num(mae_w[m].to_numpy(), nan=0.0)
        y_max = max(y_max, float(y.max()))
        ax.bar(x + j * bw, y, width=bw, color=META_COLOR[m], edgecolor="none")

    ax.set_xticks(x + (len(methods) - 1) * bw / 2)
    ax.set_xticklabels(DUR_LABELS, rotation=0)
    ax.set_xlabel("Trip duration (min)")
    ax.set_ylabel("MAE (s)")
    ax.tick_params(axis="x", which="major", labelsize=7)
    ax.grid(True, axis="y", color="0.88", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(0, y_max * 1.18 if y_max > 0 else 1)

    ax2 = ax.twinx()
    y2_max = 0.0
    for m in methods:
        if m not in mape_w.columns:
            continue
        y2 = mape_w[m].to_numpy() * 100.0
        finite = y2[np.isfinite(y2)]
        if finite.size:
            y2_max = max(y2_max, float(finite.max()))
        ax2.plot(x + (len(methods) - 1) * bw / 2, y2,
                 color=META_COLOR[m], marker=META_MARKER[m], ms=3.0,
                 linewidth=1.2, label=m)
    ax2.set_ylabel("MAPE (%)")
    ax2.set_ylim(0, max(10, (int(y2_max // 10) + 2) * 10))
    ax2.legend(loc="upper left", frameon=False, fontsize=6, ncol=2,
               borderpad=0.2, handlelength=1.2)
    _save(fig, _figs_dir(run_dir, city), f"mae_vs_duration_{city}")


# =============================================================================
# Family 5 — ECDF of absolute error
# =============================================================================
def _ecdf_graph_paper(ax, x_max_min=20.0):
    ax.set_axisbelow(True)
    ax.set_xlim(0, x_max_min); ax.set_ylim(0.0, 1.0)
    ax.xaxis.set_major_locator(MultipleLocator(5.0))
    ax.xaxis.set_minor_locator(MultipleLocator(1.0))
    ax.yaxis.set_major_locator(MultipleLocator(0.10))
    ax.yaxis.set_minor_locator(MultipleLocator(0.02))
    ax.grid(False)
    cf, cm, cM = "0.92", "0.86", "0.70"
    xs_f = np.arange(0, x_max_min + 1e-9, 1.0)
    xs_m = np.arange(0, x_max_min + 1e-9, 5.0)
    xs_M = np.arange(0, x_max_min + 1e-9, 10.0)
    ax.vlines(xs_f, 0, 1, colors=cf, linewidth=0.35, zorder=0)
    ax.vlines(xs_m, 0, 1, colors=cm, linewidth=0.50, zorder=0)
    ax.vlines(xs_M, 0, 1, colors=cM, linewidth=0.70, zorder=0)
    ys_f = np.arange(0.0, 1.0 + 1e-9, 0.02)
    ys_m = np.arange(0.0, 1.0 + 1e-9, 0.10)
    ys_M = np.arange(0.0, 1.0 + 1e-9, 0.50)
    ax.hlines(ys_f, 0, x_max_min, colors=cf, linewidth=0.35, zorder=0)
    ax.hlines(ys_m, 0, x_max_min, colors=cm, linewidth=0.50, zorder=0)
    ax.hlines(ys_M, 0, x_max_min, colors=cM, linewidth=0.70, zorder=0)


def _ecdf_annotate(ax, errs_min, color, percentiles=(90,),
                   y_text_offset=-0.04, font_size=5.5, x_max=20.0):
    v = np.asarray(errs_min, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return
    qs = np.quantile(v, np.array(percentiles) / 100.0)
    x_trans = ax.get_xaxis_transform()
    for q, p in zip(qs, percentiles):
        yp = p / 100.0
        q_vis = min(q, x_max)
        ax.vlines(q_vis, 0.0, yp, colors=color, linewidth=1.0, alpha=0.9, zorder=2)
        ax.hlines(yp, 0.0, q_vis, colors=color, linewidth=0.8, alpha=0.6, zorder=2)
        if q <= x_max:
            ax.text(q, y_text_offset, f"{q:.2f}",
                    color=color, ha="center", va="top", rotation=90,
                    transform=x_trans, clip_on=False,
                    fontsize=font_size, zorder=3)


def plot_ecdf(run_dir: Path, city: str, split: str, x_max_min: float = 20.0):
    df = load_eval_trips(run_dir, city, split)
    if df is None:
        return
    df = df[df["model"].isin(ACCURACY_MODELS)].copy()
    if "abs_err_s" not in df.columns:
        df["abs_err_s"] = (df["y_pred"] - df["y_true"]).abs()
    df["abs_err_min"] = df["abs_err_s"] / 60.0

    fig, ax = plt.subplots(figsize=(ONE_COL_IN, ONE_COL_IN * RATIO))
    for m in ACCURACY_MODELS:
        d = df[df["model"] == m]["abs_err_min"]
        if d.empty:
            continue
        xs = np.sort(np.abs(d.to_numpy()))
        ys = np.arange(1, xs.size + 1, dtype=float) / xs.size
        ax.plot(xs, ys, label=m, color=META_COLOR[m], linewidth=1.6, zorder=3)
        _ecdf_annotate(ax, d.to_numpy(), META_COLOR[m],
                       percentiles=(90,), y_text_offset=-0.04, x_max=x_max_min)

    _ecdf_graph_paper(ax, x_max_min=x_max_min)
    ax.set_xlabel("Absolute error (min)")
    ax.set_ylabel("Cumulative fraction of trips")
    leg = ax.legend(loc="lower right", frameon=True, fontsize=6.5)
    leg.get_frame().set_edgecolor("0.0"); leg.get_frame().set_linewidth(0.5)
    leg.get_frame().set_facecolor("1.0"); leg.get_frame().set_alpha(0.9)
    fig.tight_layout(rect=[0, 0.16, 1, 1])
    _save(fig, _figs_dir(run_dir, city), f"ecdf_{city}")


# =============================================================================
# Family 6 — Trip latency overlay (rolling MAPE + ms/trip per link-count bin)
# =============================================================================
def plot_latency_trip(run_dir: Path, city: str, split: str):
    trips = load_eval_trips(run_dir, city, split)
    links = load_eval_links(run_dir, city, split)
    bench = load_benchmark_by_link_count(run_dir, city)
    if trips is None or links is None or bench is None:
        return

    # n_links per trip from the link-level CSV (paired by TRIP_ID).
    n_links = links.groupby(["model", "TRIP_ID"]).size().rename("n_links").reset_index()
    df = trips.merge(n_links, on=["model", "TRIP_ID"], how="inner")
    if df.empty:
        return
    df["ape"] = _safe_mape(df["y_pred"], df["y_true"])

    fig, ax = plt.subplots(figsize=(ONE_COL_IN, ONE_COL_IN * RATIO))
    ax2 = ax.twinx()
    x_max = 0.0; y_left_max = 0.0; y_right_max = 0.0

    for m in LATENCY_MODELS:
        d = df[df["model"] == m].copy()
        if d.empty:
            continue
        d = d.sort_values("n_links").reset_index(drop=True)
        rolled = d["ape"].rolling(window=200, min_periods=50).mean() * 100.0
        ax.plot(d["n_links"], rolled, color=META_COLOR[m], linewidth=1.4)
        x_max = max(x_max, float(d["n_links"].max()))
        if not rolled.isna().all():
            y_left_max = max(y_left_max, float(np.nanmax(rolled.to_numpy())))

        b = bench[bench["model"] == m].dropna(subset=["mean_ms_per_trip"])
        b = b[b["n_trips"] > 0]
        if not b.empty:
            mids = [_bin_midpoint(s) for s in b["bin"]]
            ms_vals = b["mean_ms_per_trip"].to_numpy()
            pairs = sorted(zip(mids, ms_vals))
            xs = [p[0] for p in pairs]
            ys = [p[1] for p in pairs]
            ax2.plot(xs, ys, color=META_COLOR[m], linewidth=1.2,
                     marker=META_MARKER[m], markersize=4)
            y_right_max = max(y_right_max, float(np.nanmax(ys)))

    ax.set_xlim(0, x_max * 1.02 if x_max > 0 else 100)
    ax.set_ylim(0, max(35.0, y_left_max * 1.1 if y_left_max > 0 else 35.0))
    ax2.set_ylim(0, y_right_max * 1.30 if y_right_max > 0 else 1)

    ax.set_xlabel("Route length (links)")
    ax.set_ylabel("Trip MAPE (%)")
    ax2.set_ylabel("ms / trip (binned)")
    ax.grid(True, axis="y", color="0.92", linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)
    handles = [plt.Line2D([0], [0], color=META_COLOR[m], marker=META_MARKER[m],
                          markersize=4, linewidth=1.4, label=m)
               for m in LATENCY_MODELS]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=6,
              borderpad=0.2, handlelength=1.4)
    _save(fig, _figs_dir(run_dir, city), f"latency_trip_{city}")


# =============================================================================
# Family 7 — Scalability: total inference time vs n_links
# =============================================================================
def plot_scalability(run_dir: Path, city: str):
    df = load_benchmark_scalability(run_dir, city)
    if df is None:
        return
    df = df[df["model"].isin(LATENCY_MODELS)].copy()

    fig, ax = plt.subplots(figsize=(ONE_COL_IN, ONE_COL_IN * RATIO))
    x_max = 0.0; y_max = 0.0
    for m in LATENCY_MODELS:
        d = df[df["model"] == m].sort_values("fraction")
        if d.empty:
            continue
        x = d["n_links"].to_numpy()
        y = d["total_s"].to_numpy()
        ax.plot(x, y, color=META_COLOR[m], linewidth=1.4,
                marker=META_MARKER[m], markersize=4, label=m)
        x_max = max(x_max, float(x.max())); y_max = max(y_max, float(y.max()))

    ax.set_xlim(0, x_max * 1.05 if x_max > 0 else 1)
    ax.set_ylim(0, y_max * 1.15 if y_max > 0 else 1)
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{int(v/1000)}K" if v >= 1000 else str(int(v))))
    ax.set_xlabel("Number of links")
    ax.set_ylabel("Inference time (s)")
    ax.grid(True, color="0.92", linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left", fontsize=6,
              borderpad=0.2, handlelength=1.4)
    _save(fig, _figs_dir(run_dir, city), f"scalability_{city}")


# =============================================================================
# Driver
# =============================================================================
def make_all_for_city(run_dir: Path, city: str, split: str) -> int:
    """Run every family for one city; return count of successful figures."""
    print(f"  {city}:")
    count = 0
    for fn, name in [
        (lambda: plot_reliability(run_dir, city, split),     "reliability"),
        (lambda: plot_mae_vs_length(run_dir, city, split),   "mae_vs_length"),
        (lambda: plot_mape_vs_length(run_dir, city, split),  "mape_vs_length"),
        (lambda: plot_mae_vs_duration(run_dir, city, split), "mae_vs_duration"),
        (lambda: plot_ecdf(run_dir, city, split),            "ecdf"),
        (lambda: plot_latency_trip(run_dir, city, split),    "latency_trip"),
        (lambda: plot_scalability(run_dir, city),            "scalability"),
    ]:
        try:
            fn()
            count += 1
        except Exception as e:
            print(f"    [skip] {name}: {e}")
    return count


def discover_runs() -> List[Path]:
    if not CHECKPOINTS_ROOT.exists():
        return []
    return sorted(p for p in CHECKPOINTS_ROOT.iterdir()
                  if p.is_dir() and "_source" in p.name)   # [2026-06-07] sources-keyed runs


def cities_in_run(run_dir: Path) -> List[str]:
    return sorted(p.name for p in run_dir.iterdir()
                  if p.is_dir() and (p / "eval_outputs").is_dir())


def parse_args():
    p = argparse.ArgumentParser(description="Meta_Learning evaluation figures (all families)")
    p.add_argument("--run", default=None,
                   help="Run folder name under checkpoints/. Default: every run found.")
    p.add_argument("--cities", default=None,
                   help="Comma-separated city keys. Default: every city in the run "
                        "that has an eval_outputs/ dir.")
    p.add_argument("--split", default="test", choices=("train", "val", "test"),
                   help="Which split's predictions to plot (default: test).")
    return p.parse_args()


def main():
    args = parse_args()

    if args.run:
        run_dirs = [CHECKPOINTS_ROOT / args.run]
        if not run_dirs[0].is_dir():
            raise SystemExit(f"Run folder not found: {run_dirs[0]}")
    else:
        run_dirs = discover_runs()
        if not run_dirs:
            raise SystemExit(
                f"No run folders found under {CHECKPOINTS_ROOT}. "
                f"Run the Evaluation block first (python main.py --blocks Evaluation ...).")

    requested = (
        [c.strip() for c in args.cities.split(",") if c.strip()] if args.cities else None)

    total = 0
    for run_dir in run_dirs:
        cities = requested or cities_in_run(run_dir)
        if not cities:
            print(f"[run] {run_dir.name}: no eval_outputs/ dirs found, skipping.")
            continue
        print(f"\n[run] {run_dir.name}  cities={cities}  split={args.split}")
        for city in cities:
            total += make_all_for_city(run_dir, city, args.split)

    print(f"\n[DONE] generated {total} figure(s).")


if __name__ == "__main__":
    main()
