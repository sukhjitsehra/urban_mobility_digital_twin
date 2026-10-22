import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as cm

NAME = "fig04_sparse_data"
CITY, FOLDER = "Daegu", "daegu"
DATA_FILE = cm.DATA / f"{NAME}.csv"
MODELS = ("full", "no_graph")
LABEL = {"full": "LinkETA", "no_graph": "LinkETA without graph context"}
STYLE = {"full": dict(color=cm.HC["blue"], marker="o", ls="-"),
         "no_graph": dict(color=cm.HC["red"], marker="s", ls="--")}
N_BOOT, SEED = 1000, 0


def build():
    out = cm.ROOT / "results" / FOLDER / "eval_outputs"
    curve = pd.read_csv(out / "robustness_missing_data_curve.csv")
    trips = pd.read_csv(out / "robustness_missing_data_trips.csv", dtype={"TRIP_ID": str})
    old = pd.read_csv(out / "robustness_missing_data.csv")
    ps = sorted(curve.p_blanked.unique())
    rng = np.random.default_rng(SEED)
    rows, idx = [], None
    for p in ps:
        ape = {}
        for m in MODELS:
            t = trips[(trips.model == m) & np.isclose(trips.p_blanked, p)].sort_values("TRIP_ID")
            t = t[t.y_true > 0]
            ape[m] = (t.TRIP_ID.to_numpy(), (np.abs(t.y_pred - t.y_true) / t.y_true).to_numpy())
            c = curve[(curve.model == m) & np.isclose(curve.p_blanked, p)]
            cm.require(len(c) == 1, f"{m} p={p}: one curve row expected, found {len(c)}")
            cm.require(abs(ape[m][1].mean() - float(c.route_mape.iloc[0])) < 1e-4,
                       f"{m} p={p}: trips give route MAPE {ape[m][1].mean():.6f}, curve {float(c.route_mape.iloc[0]):.6f}")
            o = old[(old.model == m) & np.isclose(old.p_blanked, p)]
            if len(o):
                cm.require(abs(float(o.route_mape.iloc[-1]) - float(c.route_mape.iloc[0])) < 1e-6,
                           f"{m} p={p}: curve {float(c.route_mape.iloc[0]):.8f} != earlier run {float(o.route_mape.iloc[-1]):.8f}")
        cm.require((ape["full"][0] == ape["no_graph"][0]).all(), f"p={p}: models differ in test trips")
        a, b = ape["full"][1], ape["no_graph"][1]
        if idx is None:
            idx = rng.integers(0, len(a), size=(N_BOOT, len(a)))
        gap = 100 * (b[idx].mean(axis=1) - a[idx].mean(axis=1))
        lo, hi = np.percentile(gap, [2.5, 97.5])
        rows.append(dict(p=p, n_trips=len(a), full_mape=100 * a.mean(), no_graph_mape=100 * b.mean(),
                         gap=100 * (b.mean() - a.mean()), gap_lo=lo, gap_hi=hi))
    df = pd.DataFrame(rows)
    df.to_csv(DATA_FILE, index=False)
    return df


def draw(df):
    import matplotlib.pyplot as plt
    x = 100 * df.p
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(cm.SINGLE, 0.95 * cm.SINGLE), sharex=True,
                                 gridspec_kw=dict(height_ratios=[1.6, 1.0], hspace=0.08))
    for m in MODELS:
        ax.plot(x, df[f"{m}_mape"], label=LABEL[m], ms=2.8, lw=1.0, markevery=2, **STYLE[m])
    ax.set_ylabel("Route MAPE (%)")
    ax.legend(frameon=False, loc="upper left")
    bx.fill_between(x, df.gap_lo, df.gap_hi, color=cm.GREY, lw=0)
    bx.plot(x, df.gap, color=cm.INK, lw=1.0)
    bx.axhline(0, color=cm.INK, lw=0.5, ls=":")
    bx.set_ylabel("Difference (pp)")
    bx.set_xlabel("Route roads without their own traffic data (%)")
    for a in (ax, bx):
        a.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.17, right=0.97, top=0.97, bottom=0.13)
    cm.save(fig, NAME)


def run(from_data=False):
    cm.apply()
    if from_data:
        df = pd.read_csv(DATA_FILE)
        cm.require(((df.gap - (df.no_graph_mape - df.full_mape)).abs() < 1e-9).all(), "stored gap != difference")
    else:
        df = build()
    print(df.to_string(index=False))
    draw(df)


if __name__ == "__main__":
    run(from_data="--from-data" in sys.argv)

