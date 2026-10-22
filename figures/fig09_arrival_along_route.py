import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as cm
from fig06_strata_lead import MODEL_COLOR

NAME = "fig09_arrival_along_route"
DATA_FILE = cm.DATA / f"{NAME}.csv"
CITIES = {"porto": "Porto", "sf": "San Francisco", "chengdu": "Chengdu", "daegu": "Daegu",
          "annarbor": "Ann Arbor", "sfbus": "SF bus"}
MODELS = {"LinkETA": "linketa", "ConSTGAT": "constgat", "DuETA": "dueta", "STHR": "sthr"}
POINTS = (0.5, 1.0)
WIDTH = 468 / 72.27


def build():
    rows = []
    for c, cname in CITIES.items():
        L = {}
        for m, tag in MODELS.items():
            d = pd.read_csv(cm.ROOT / "results" / c / "eval_outputs" / f"eval_links_{tag}.csv",
                            usecols=["TRIP_ID", "true_tt_link", "pred_tt_link"], dtype={"TRIP_ID": str})
            d["pos"] = d.groupby("TRIP_ID", sort=False).cumcount()
            L[m] = d
        ref = L["LinkETA"]
        n_trip = ref.groupby("TRIP_ID").size()
        for m in MODELS:
            cm.require(L[m].groupby("TRIP_ID").size().reindex(n_trip.index).equals(n_trip),
                       f"{cname} {m}: trips or links per trip differ from LinkETA")
            j = ref.merge(L[m], on=["TRIP_ID", "pos"], suffixes=("", "_m"))
            cm.require(len(j) == len(ref) and bool((np.abs(j.true_tt_link - j.true_tt_link_m) < 1e-3).all()),
                       f"{cname} {m}: true link times differ from LinkETA")
        for m, tag in MODELS.items():
            d = L[m]
            n = d.TRIP_ID.map(n_trip)
            cp = d.groupby("TRIP_ID", sort=False).pred_tt_link.cumsum()
            ct = d.groupby("TRIP_ID", sort=False).true_tt_link.cumsum()
            k = d.pos + 1
            met = json.loads((cm.ROOT / "results" / c / "eval_outputs" / f"eval_metrics_{tag}.json").read_text())
            met = met[m] if m in met else met[next(x for x in met if isinstance(met[x], dict))]
            for p in POINTS:
                sel = k == np.ceil(p * n).clip(lower=1).astype(int)
                cm.require(int(sel.sum()) == len(n_trip), f"{cname} {m}: one point per trip expected at {p}")
                e = float(np.abs(cp[sel] - ct[sel]).mean())
                if p == 1.0:
                    cm.require(abs(e - met["route_mae"]) <= 1e-4 * met["route_mae"],
                               f"{cname} {m}: destination error {e:.3f} != route MAE {met['route_mae']:.3f}")
                    cm.require(int(met["n_links"]) == len(d), f"{cname} {m}: {len(d)} links != n_links {met['n_links']}")
                rows.append(dict(city=cname, model=m, point=p, mae_s=e, n_trips=len(n_trip)))
        print(f"  {cname}: {len(n_trip):,} trips checked")
    df = pd.DataFrame(rows)
    df.to_csv(DATA_FILE, index=False)
    return df


def draw(df):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    cm.apply()
    fig, ax = plt.subplots(figsize=(WIDTH, 0.40 * WIDTH))
    fig.subplots_adjust(left=0.085, right=0.995, top=0.97, bottom=0.25)
    w = 0.2
    for i, cname in enumerate(CITIES.values()):
        for k, m in enumerate(MODELS):
            x = i + (k - 1.5) * w
            v = df[(df.city == cname) & (df.model == m)].set_index("point").mae_s
            ax.bar(x, v[1.0], width=w * 0.92, color=MODEL_COLOR[m], alpha=0.35, lw=0)
            ax.bar(x, v[0.5], width=w * 0.45, color=MODEL_COLOR[m], lw=0)
    ax.set_xticks(range(len(CITIES)))
    ax.set_xticklabels(list(CITIES.values()))
    ax.set_xlim(-0.5, len(CITIES) - 0.5)
    ax.set_ylabel("Arrival-time MAE (s per trip)")
    ax.grid(axis="y", color=cm.GREY, lw=0.4)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    hand = [Patch(color=MODEL_COLOR[m], label=m) for m in MODELS]
    hand += [Patch(color="#999999", alpha=0.35, label="at the destination"),
             Patch(color="#444444", label="after half of the route's links")]
    ax.legend(handles=hand, ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False, fontsize=7,
              handlelength=1.4, columnspacing=1.2)
    cm.save(fig, NAME)


def run(from_data=False):
    cm.apply()
    if from_data:
        df = pd.read_csv(DATA_FILE)
        cm.require(len(df) == len(CITIES) * len(MODELS) * len(POINTS), "saved data incomplete")
    else:
        df = build()
    draw(df)


if __name__ == "__main__":
    run(from_data="--from-data" in sys.argv)

