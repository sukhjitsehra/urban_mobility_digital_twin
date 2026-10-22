import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as cm

NAME = "fig07_trip_starts"
DATA_FILE = cm.DATA / f"{NAME}.csv"
CITIES = {"porto": "Porto", "sf": "San Francisco", "daegu": "Daegu", "chengdu": "Chengdu",
          "annarbor": "Ann Arbor", "sfbus": "SF bus"}
ZONE = {"porto": "Europe/Lisbon", "sf": "America/Los_Angeles", "daegu": None, "chengdu": None,
        "annarbor": "America/Detroit", "sfbus": "America/Los_Angeles"}
TRIPS = {"porto": 28416, "sf": 32333, "daegu": 37050, "chengdu": 10110, "annarbor": 19047, "sfbus": 10000}
STYLE = {"porto": ("#4477AA", "-"), "sf": ("#EE6677", (0, (5, 1.5))), "daegu": ("#CCBB44", (0, (5, 1.5, 1.5, 1.5))),
         "chengdu": ("#228833", (0, (1.5, 1.2))), "annarbor": ("#66CCEE", (0, (9, 2))),
         "sfbus": ("#AA3377", (0, (8, 1.5, 1.5, 1.5, 1.5, 1.5)))}
WIDTH = 0.48 * 468 / 72.27 * 25.4 * cm.MM


def compute():
    rows = []
    for city in CITIES:
        d = pd.read_csv(cm.ROOT / "data" / city / f"df_all_{city}.csv", usecols=["TRIP_ID", "departure_ts"])
        d = d.drop_duplicates("TRIP_ID")
        cm.require(d.departure_ts.notna().all(), f"{city}: trips without a departure time")
        cm.require(len(d) == TRIPS[city], f"{city}: {len(d)} trips, dataset table has {TRIPS[city]}")
        t = pd.to_datetime(d.departure_ts, unit="s", utc=True)
        hour = (t.dt.tz_convert(ZONE[city]) if ZONE[city] else t).dt.hour
        share = hour.value_counts(normalize=True).reindex(range(24), fill_value=0.0) * 100
        rows += [dict(city=city, hour=h, share=float(s), trips=len(d)) for h, s in share.items()]
        print(f"[{NAME}] {CITIES[city]}: {len(d)} trips, peak at {int(share.idxmax()):02d}:00 local")
    D = pd.DataFrame(rows)
    D.to_csv(DATA_FILE, index=False, float_format="%.10g")
    return D


def _recheck(D):
    for city in CITIES:
        g = D[D.city == city]
        cm.require(len(g) == 24 and abs(g.share.sum() - 100) < 1e-6, f"{city}: shares do not add up to 100")
        cm.require(int(g.trips.iloc[0]) == TRIPS[city], f"{city}: trip count")


def draw(D):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    cm.apply()
    fig, ax = plt.subplots(figsize=(WIDTH, 0.492 * WIDTH))
    fig.subplots_adjust(left=0.13, right=0.72, top=0.97, bottom=0.26)
    for city, (col, ls) in STYLE.items():
        g = D[D.city == city].sort_values("hour")
        ax.plot(g.hour, g.share, color=col, ls=ls, lw=1.2)
    ax.set_xlim(0, 23)
    ax.set_xticks(range(0, 24, 3))
    ax.set_yticks(range(0, 11, 2))
    ax.set_xlabel("Hour of the day (local time)", fontsize=7)
    ax.set_ylabel("Trip starts (%)", fontsize=7)
    ax.grid(axis="y", color=cm.GREY, lw=0.4)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    hand = [Line2D([], [], color=c, ls=ls, lw=1.2, label=CITIES[k]) for k, (c, ls) in STYLE.items()]
    fig.legend(handles=hand, loc="center right", ncol=1, fontsize=7, frameon=False, handlelength=2.4,
               bbox_to_anchor=(1.0, 0.6), labelspacing=0.3)
    cm.save(fig, NAME)


def run(from_data=False):
    D = pd.read_csv(DATA_FILE) if from_data else compute()
    _recheck(D)
    draw(D)


if __name__ == "__main__":
    run(from_data="--from-data" in sys.argv)

