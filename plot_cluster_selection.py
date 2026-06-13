"""
plot_cluster_selection.py — cluster-count selection figures (decoupled)
=======================================================================
Draws the elbow selection graph for the physical (p) and topology (t) K-Means
sweeps as TWO SEPARATE figures.

This module is independent of the clustering itself.  structural_clustering.py
saves the sweep results to ``cluster_selection/selection_stats.json`` while it
runs; this module only READS that file and renders the plots.  So if a figure
comes out wrong you re-run *this* file, never the (slow) clustering:

    python plot_cluster_selection.py
    python plot_cluster_selection.py --dir checkpoints/cluster_selection

Selection is by the ELBOW of the inertia (WCSS) curve.  The silhouette score is
still computed by the sweep and stored in selection_stats.json for the record,
but it is NOT plotted (it has no interior optimum for the topology features, so
it is not used as the selector).  Each figure is therefore a pure elbow plot.

Style: LinkETA house palette (experiments/LinkETA/figures.py) MINUS the golden
#DDAA33.  Inertia = #004488 (blue), chosen p/t = #BB5566 (red).  All axis labels
/ ticks / titles are BLACK and Title-Cased; the legend is framed and clear of
the curve.

Outputs (to the selection dir), two figures, each as PNG and PDF:
    cluster_selection_physical.{png,pdf}
    cluster_selection_topology.{png,pdf}
"""

import os
import json
import math
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


# --- Defaults / palette --------------------------------------------------
DEFAULT_DIR = "checkpoints/cluster_selection"
STATS_FILENAME = "selection_stats.json"

C_INERTIA = "#004488"   # LinkETA blue   (inertia / WCSS curve)
C_CHOSEN  = "#BB5566"   # LinkETA red    (chosen p/t marker + vline)
C_ELBOW   = "0.60"      # grey dotted    (raw elbow, when it differs from chosen)


def _apply_style():
    """LinkETA serif house style with everything forced to black text."""
    mpl.rcParams.update({
        "font.family":      "serif",
        "font.serif":       ["Times New Roman", "Times", "Cambria", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "pdf.fonttype":     42,
        "ps.fonttype":      42,
        "axes.linewidth":   0.7,
        "figure.dpi":       150,
        "savefig.dpi":      600,
        "text.color":       "black",
        "axes.edgecolor":   "black",
        "axes.labelcolor":  "black",
        "xtick.color":      "black",
        "ytick.color":      "black",
        "axes.labelsize":   9.5,
        "xtick.labelsize":  8.5,
        "ytick.labelsize":  8.5,
        "legend.fontsize":  8.0,
        "axes.titlesize":   11.0,
    })


# --- Persistence (shared with structural_clustering.py) ------------------

def save_selection_stats(out_dir, stats_p, stats_t, P, T):
    """Write the sweep results so the plots can be regenerated without
    re-running K-Means.  ``stats_*`` are the dicts returned by
    structural_clustering._sweep_cluster_counts (ks/inertias/silhouettes/
    k_elbow/k_silhouette).  NaN silhouettes are stored as null."""
    def _clean_sils(sils):
        return [None if (s is None or (isinstance(s, float) and math.isnan(s)))
                else float(s) for s in sils]

    def _pack(stats, chosen):
        return {
            "ks":           [int(k) for k in stats["ks"]],
            "inertias":     [float(v) for v in stats["inertias"]],
            "silhouettes":  _clean_sils(stats["silhouettes"]),
            "k_elbow":      int(stats["k_elbow"]),
            "k_silhouette": int(stats["k_silhouette"]),
            "chosen":       int(chosen),
        }

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, STATS_FILENAME)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"physical": _pack(stats_p, P),
                   "topology": _pack(stats_t, T)}, fh, indent=2)
    print(f"  Saved selection stats: {path}")
    return path


def load_selection_stats(out_dir):
    path = os.path.join(out_dir, STATS_FILENAME)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"selection stats not found at {path}. Run structural_clustering.py "
            f"first (it writes {STATS_FILENAME} during the sweep).")
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    for sub in data.values():
        sub["silhouettes"] = [float("nan") if s is None else float(s)
                              for s in sub["silhouettes"]]
    return data


# --- Plotting ------------------------------------------------------------

def _plot_one(stats, kind, out_dir):
    """Render one pure-elbow selection figure (physical or topology) and save it.

    Inertia (WCSS) only — silhouette is computed by the sweep and kept in
    selection_stats.json, but selection is by elbow so it is not drawn."""
    sym  = "p" if kind == "physical" else "t"
    name = "Physical" if kind == "physical" else "Topology"

    ks      = stats["ks"]
    inertia = stats["inertias"]
    chosen  = stats["chosen"]
    k_elbow = stats["k_elbow"]

    fig, ax = plt.subplots(figsize=(5.0, 3.4))

    # --- Inertia (WCSS) elbow curve, blue -------------------------------
    ax.plot(ks, inertia, marker="o", ms=4.0, lw=1.6, color=C_INERTIA, zorder=3)
    ax.set_xlabel(f"Number of Clusters ({sym})")
    ax.set_ylabel("Inertia (WCSS)")
    ax.grid(True, color="0.88", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(MultipleLocator(2))

    # --- Chosen k (the elbow): red dashed vline + solid red dot ---------
    chosen_inertia = inertia[ks.index(chosen)] if chosen in ks else None
    label = (f"Chosen {sym} = {chosen}  (inertia = {chosen_inertia:,.0f})"
             if chosen_inertia is not None else f"Chosen {sym} = {chosen}")
    ax.axvline(chosen, color=C_CHOSEN, lw=1.4, ls="--", zorder=2, label=label)
    if chosen_inertia is not None:
        ax.plot([chosen], [chosen_inertia], marker="o", ms=7.5,
                color=C_CHOSEN, zorder=4)
    # Raw elbow only on the off chance it disagrees with the chosen value
    if k_elbow != chosen:
        ax.axvline(k_elbow, color=C_ELBOW, lw=0.9, ls=":", zorder=1,
                   label=f"Elbow = {k_elbow}")

    # Headroom so the legend (upper-right) clears the curve
    imin, imax = min(inertia), max(inertia)
    span = (imax - imin) or 1.0
    ax.set_ylim(imin - 0.04 * span, imax + 0.16 * span)

    # --- Title + framed legend ------------------------------------------
    ax.set_title(f"{name} Clustering — Elbow Method ({sym} = {chosen})",
                 fontweight="bold")
    leg = ax.legend(loc="upper right", frameon=True, borderpad=0.4,
                    handlelength=1.8, labelcolor="black")
    leg.get_frame().set_edgecolor("0.0")
    leg.get_frame().set_linewidth(0.5)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_alpha(0.95)

    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f"cluster_selection_{kind}")
    fig.savefig(f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {stem}.png  |  {stem}.pdf")


def plot_all(stats_p, stats_t, P, T, out_dir):
    """Plot directly from in-memory sweep results (used by clustering)."""
    _apply_style()
    _plot_one({**stats_p, "chosen": int(P)}, "physical", out_dir)
    _plot_one({**stats_t, "chosen": int(T)}, "topology", out_dir)


def plot_from_dir(out_dir=DEFAULT_DIR):
    """Plot from the saved selection_stats.json (the decoupled entry point)."""
    _apply_style()
    data = load_selection_stats(out_dir)
    _plot_one(data["physical"], "physical", out_dir)
    _plot_one(data["topology"], "topology", out_dir)


def main():
    ap = argparse.ArgumentParser(
        description="Render cluster-count selection figures from saved sweep stats.")
    ap.add_argument("--dir", default=DEFAULT_DIR,
                    help=f"Directory holding {STATS_FILENAME} (default: {DEFAULT_DIR}).")
    args = ap.parse_args()
    plot_from_dir(args.dir)


if __name__ == "__main__":
    main()
