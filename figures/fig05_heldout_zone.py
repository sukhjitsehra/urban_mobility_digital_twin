import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as cm

NAME = "fig05_heldout_zone"
FOLDER = "daegu"
TRAIN_C, HELD_C = "#0077BB", "#EE7733"
RUN_LOG = cm.ROOT / "internal_analysis" / "logs" / "abl_daegu_zone_out.log"


def _ablations():
    sys.path.insert(0, str(cm.ROOT / "evaluation"))
    import ablations as ab
    ab.li.CFG = {"way_id_col_df": "way_id", "way_id_col_edges_candidates": ["way_id", "id", "edge_id", "osmid"]}
    return ab


def build():
    import geopandas as gpd
    ab = _ablations()
    zone = ab.VARIANTS["zone_out"]["zone"]
    trips_path, edges_path = ab.main.city_input_files(FOLDER)
    d = pd.read_csv(trips_path, usecols=["TRIP_ID", "way_id", "departure_time"])
    d["departure_time"] = pd.to_datetime(d["departure_time"])
    d["way_id"] = d["way_id"].astype(str)
    base = ab._ORIG["temporal_split_by_trip"](d.copy())
    held = ab._make_split(zone)(d.copy())
    out_tr, out_va = set(base[0].TRIP_ID) - set(held[0].TRIP_ID), set(base[1].TRIP_ID) - set(held[1].TRIP_ID)
    removed, kept = out_tr | out_va, set(held[0].TRIP_ID)
    roads = ab._zone_roads(zone)
    log = RUN_LOG.read_text(encoding="utf-8", errors="replace")
    nums = {}
    for part in ("train", "val"):
        m = re.search(rf"\[ablation:zone\] {part}: [\d,]+ trips, ([\d,]+) with a link in the zone \(([\d,]+) roads\); "
                      rf"removed ([\d,]+) zone trips", log)
        cm.require(m is not None, f"no '{part}' line of the zone split in {RUN_LOG.name}")
        nums[part] = [int(g.replace(",", "")) for g in m.groups()]
    cm.require(nums["train"][2] == len(out_tr) and nums["val"][2] == len(out_va),
               f"held out train/val {len(out_tr)}/{len(out_va)} != run {nums['train'][2]}/{nums['val'][2]}")
    cm.require(nums["train"][1] == len(roads), f"zone roads {len(roads)} != run {nums['train'][1]}")
    through = set(d.loc[d.way_id.isin(roads), "TRIP_ID"])
    cm.require(removed <= through and not (kept & through), "held-out / kept trips do not follow the zone rule")
    cnt = {g: d[d.TRIP_ID.isin(s)].groupby("way_id").size() for g, s in (("kept", kept), ("held_out", removed))}
    for g, s in (("kept", kept), ("held_out", removed)):
        cm.require(int(cnt[g].sum()) == int(d.TRIP_ID.isin(s).sum()), f"{g}: traversals != link rows")
    e = gpd.read_file(edges_path)
    idc = ab.li.guess_edges_id_col(e, ab.li.CFG["way_id_col_edges_candidates"])
    e = e.assign(way_id=e[idc].astype(str)).to_crs(4326)
    df = e[["way_id", "geometry"]].assign(n_kept=lambda x: x.way_id.map(cnt["kept"]).fillna(0).astype(int),
                                         n_held_out=lambda x: x.way_id.map(cnt["held_out"]).fillna(0).astype(int))
    df = df[(df.n_kept > 0) | (df.n_held_out > 0)]
    cm.require(int(df.n_kept.sum()) == int(cnt["kept"].sum()) and int(df.n_held_out.sum()) == int(cnt["held_out"].sum()),
               "traversals on roads missing from the network")
    meta = dict(zone=zone, n_train=len(base[0].TRIP_ID.unique()), n_val=len(base[1].TRIP_ID.unique()),
                n_kept=len(kept), n_held_out_train=len(out_tr), n_held_out_val=len(out_va), n_zone_roads=len(roads))
    df.assign(wkt=df.geometry.apply(lambda g: g.wkt)).drop(columns="geometry").to_csv(cm.DATA / f"{NAME}.csv", index=False)
    (cm.DATA / f"{NAME}_meta.json").write_text(json.dumps(meta, indent=1, default=str))
    return df, meta


def draw(df, meta):
    import contextily as cx
    import geopandas as gpd
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    from shapely.geometry import box
    cm.apply()
    z = meta["zone"]
    s = z["size_m"]
    sq = gpd.GeoSeries([box(z["cell"][0] * s, z["cell"][1] * s, (z["cell"][0] + 1) * s, (z["cell"][1] + 1) * s)],
                       crs=z["crs"]).to_crs(4326).iloc[0]
    fig, ax = plt.subplots(figsize=(cm.SINGLE, cm.SINGLE * 0.86))
    fig.subplots_adjust(left=0.14, right=0.96, bottom=0.12, top=0.995)

    def segs(sub):
        out, wid = [], []
        for g, n in zip(sub.geometry, sub.n):
            for p in getattr(g, "geoms", [g]):
                out.append(np.asarray(p.coords)[:, :2])
                wid.append(n)
        return out, np.asarray(wid, float)
    for col, colour, lw_min, lw_max, zo, alpha in (("n_kept", TRAIN_C, 0.3, 0.7, 2, 0.7),
                                                  ("n_held_out", HELD_C, 0.3, 0.9, 3, 0.9)):
        sub = df[df[col] > 0].assign(n=lambda x: x[col])
        lines, n = segs(sub)
        lw = lw_min + (lw_max - lw_min) * np.log1p(n) / np.log1p(n.max())
        ax.add_collection(LineCollection(lines, colors=colour, linewidths=lw, alpha=alpha, zorder=zo, capstyle="round"))
    ax.plot(*sq.exterior.xy, color=cm.INK, lw=1.2, ls=(0, (4, 2)), zorder=4)
    x0, y0, x1, y1 = df.total_bounds
    pad = 0.01
    ax.set_xlim(x0 - pad, x1 + pad)
    ax.set_ylim(y0 - pad, y1 + pad)
    ax.set_aspect(1 / np.cos(np.deg2rad((y0 + y1) / 2)))
    import contextily.tile as cx_tile
    cx_tile.USER_AGENT = "LinkETA-figures/1.0 (research figure; contextily)"
    cx.add_basemap(ax, crs="EPSG:4326", source=cx.providers.OpenStreetMap.Mapnik, attribution=False, zorder=0,
                   alpha=0.55)
    ax.text(0.005, 0.005, "© OpenStreetMap contributors", transform=ax.transAxes, fontsize=7, ha="left",
            va="bottom", zorder=5, bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    held = meta["n_held_out_train"] + meta["n_held_out_val"]
    from matplotlib.ticker import MultipleLocator
    ax.xaxis.set_major_locator(MultipleLocator(0.04))
    ax.legend(handles=[
        Line2D([], [], color=TRAIN_C, lw=1.2, label=f"training trips ({meta['n_kept']:,})"),
        Line2D([], [], color=HELD_C, lw=1.6, label=f"held-out trips through the area ({held:,})"),
        Line2D([], [], color=cm.INK, lw=1.2, ls=(0, (4, 2)), label="held-out area (2 km)")],
        loc="upper left", frameon=True, framealpha=0.9, handlelength=1.6, borderpad=0.3, labelspacing=0.25)
    cm.save(fig, NAME)


def run(from_data=False):
    cm.apply()
    if from_data:
        import geopandas as gpd
        from shapely import wkt
        t = pd.read_csv(cm.DATA / f"{NAME}.csv", dtype={"way_id": str})
        df = gpd.GeoDataFrame(t.drop(columns="wkt"), geometry=t.wkt.apply(wkt.loads), crs=4326)
        meta = json.loads((cm.DATA / f"{NAME}_meta.json").read_text())
        cm.require(int((df.n_held_out > 0).sum()) > 0 and meta["n_kept"] == meta["n_train"] - meta["n_held_out_train"],
                   "saved data inconsistent with its meta")
    else:
        df, meta = build()
    draw(df, meta)


if __name__ == "__main__":
    run(from_data="--from-data" in sys.argv)

