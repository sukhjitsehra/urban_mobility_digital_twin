import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as cm

NAME = "fig03_disruption"
CITY, FOLDER = "Daegu", "daegu"
DATA_FILE = cm.DATA / f"{NAME}.csv"
MODELS = ("full", "no_graph")
LABEL = {"full": "LinkETA", "no_graph": "LinkETA without graph context"}
STYLE = {"full": dict(color=cm.HC["blue"], marker="o", ls="-"),
         "no_graph": dict(color=cm.HC["red"], marker="s", ls="--")}
N_BOOT, SEED = 1000, 0


def _depth_x(lab):
    return -1 if lab == "none" else int(lab)


def build():
    out = cm.ROOT / "results" / FOLDER / "eval_outputs"
    summ = pd.read_csv(out / "robustness_disruption.csv")
    links = pd.read_csv(out / "robustness_disruption_links.csv", dtype={"TRIP_ID": str, "depth": str})
    summ["depth"] = summ["depth"].astype(str)

    saved = {"full": json.loads((out / "eval_metrics_linketa.json").read_text())["LinkETA"]["route_mape"]}
    abl = pd.read_csv(out / "ablations_eval.csv")
    saved["no_graph"] = float(abl[abl.variant == "no_graph"].route_mape.iloc[-1])
    for m in MODELS:
        r = summ[(summ.model == m) & (summ.depth == "none")]
        cm.require(len(r) == 1, f"{m}: one reference row expected, found {len(r)}")
        cm.require(abs(float(r.route_mape.iloc[0]) - saved[m]) < 1e-3,
                   f"{m}: reference route MAPE {float(r.route_mape.iloc[0]):.4f} != saved {saved[m]:.4f}")

    seeds = links[links.is_seed.astype(str) == "True"].copy()
    seeds["abs_err"] = (seeds.pred_tt_link - seeds.true_tt_link).abs()
    depths = sorted(summ.depth.unique(), key=_depth_x)
    rows, err = [], {}
    for d in depths:
        per = {}
        for m in MODELS:
            s = seeds[(seeds.model == m) & (seeds.depth == d)].sort_values("TRIP_ID")
            cm.require(s.TRIP_ID.is_unique, f"{m} d={d}: more than one seed in a trip")
            r = summ[(summ.model == m) & (summ.depth == d)]
            cm.require(len(r) == 1 and abs(float(r.seed_link_mae.iloc[0]) - s.abs_err.mean()) < 1e-9,
                       f"{m} d={d}: seed MAE {s.abs_err.mean():.6f} != summary {float(r.seed_link_mae.iloc[0]):.6f}")
            per[m] = s
        a, b = per["full"], per["no_graph"]
        cm.require(a.TRIP_ID.tolist() == b.TRIP_ID.tolist() and a.pos.tolist() == b.pos.tolist(),
                   f"d={d}: models differ in trips or seed positions")
        err[d] = (a.abs_err.to_numpy(), b.abs_err.to_numpy())
        rows.append(dict(depth=d, n_seeds=len(a), full_mae=a.abs_err.mean(), no_graph_mae=b.abs_err.mean()))

    hid = links[links.hidden.astype(str) == "True"]
    prev = None
    for d in [x for x in depths if x != "none"]:
        h = {m: set(zip(hid[(hid.model == m) & (hid.depth == d)].TRIP_ID, hid[(hid.model == m) & (hid.depth == d)].pos))
             for m in MODELS}
        cm.require(h["full"] == h["no_graph"], f"d={d}: hidden roads differ between models")
        if prev is not None:
            cm.require(prev <= h["full"], f"d={d}: hidden area not nested in the next depth")
        prev = h["full"]

    rng = np.random.default_rng(SEED)
    n = len(next(iter(err.values()))[0])
    idx = rng.integers(0, n, size=(N_BOOT, n))
    for r in rows:
        fa, nb = err[r["depth"]]
        gap = nb[idx].mean(axis=1) - fa[idx].mean(axis=1)
        r["gap"] = r["no_graph_mae"] - r["full_mae"]
        r["gap_lo"], r["gap_hi"] = np.percentile(gap, [2.5, 97.5])
    df = pd.DataFrame(rows)
    df.to_csv(DATA_FILE, index=False)
    return df


def draw(df):
    import matplotlib.pyplot as plt
    x = np.array([_depth_x(d) for d in df.depth])
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(cm.SINGLE, 0.95 * cm.SINGLE), sharex=True,
                                 gridspec_kw=dict(height_ratios=[1.6, 1.0], hspace=0.08))
    for m in MODELS:
        ax.plot(x, df[f"{m}_mae"], label=LABEL[m], ms=3.5, lw=1.0, **STYLE[m])
    ax.set_ylabel("Seed-road link MAE (s)")
    ax.legend(frameon=False, loc="upper left")
    bx.fill_between(x, df.gap_lo, df.gap_hi, color=cm.GREY, lw=0)
    bx.plot(x, df.gap, color=cm.INK, marker="D", ms=3.0, lw=1.0)
    bx.axhline(0, color=cm.INK, lw=0.5, ls=":")
    bx.set_ylabel("Difference (s)")
    bx.set_xticks(x)
    bx.set_xticklabels(["none" if d == "none" else d for d in df.depth])
    bx.set_xlabel("Hidden radius around the seed road (hops)")
    for a in (ax, bx):
        a.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.17, right=0.97, top=0.97, bottom=0.13)
    cm.save(fig, NAME)


def run(from_data=False):
    cm.apply()
    if from_data:
        df = pd.read_csv(DATA_FILE, dtype={"depth": str})
        cm.require(((df.gap - (df.no_graph_mae - df.full_mae)).abs() < 1e-9).all(), "stored gap != difference")
        cm.require((df.gap_lo <= df.gap_hi).all(), "interval bounds out of order")
    else:
        df = build()
    print(df.to_string(index=False))
    draw(df)


if __name__ == "__main__":
    run(from_data="--from-data" in sys.argv)

