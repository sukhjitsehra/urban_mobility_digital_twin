import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "data" / "sfbus"
TZ = "America/Los_Angeles"
MIN_POINTS, MIN_DURATION_S = 5, 60
BUS_HIGHWAYS = ["motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link", "secondary",
                "secondary_link", "tertiary", "tertiary_link", "unclassified", "residential", "living_street",
                "service", "road", "busway"]


def trips(src):
    t = pd.read_csv(src, usecols=["TRIP_ID", "vehicle_id", "route_id", "direction_id", "trip_travel_time_s",
                                  "POLYLINE", "point_ts"])
    n_src = len(t)
    t = t.sort_values("TRIP_ID", kind="mergesort").reset_index(drop=True)
    assert t.TRIP_ID.is_unique
    t = t.rename(columns={"TRIP_ID": "trip_key"})
    t["TRIP_ID"] = np.arange(1, len(t) + 1, dtype="int64")
    print(f"[sfbus] runs listed twice under two routes (same bus, points and times), second copy dropped: "
          f"{int(t.duplicated(['vehicle_id', 'POLYLINE', 'point_ts']).sum())}")
    t = t[~t.duplicated(["vehicle_id", "POLYLINE", "point_ts"], keep="first")].copy()
    lens = t.point_ts.str.count(",") + 1
    local = pd.to_datetime(pd.Series(np.concatenate(t.point_ts.apply(json.loads).to_numpy())),
                           format="%Y-%m-%dT%H:%M:%S")
    utc = local.dt.tz_localize(TZ, ambiguous="NaT", nonexistent="NaT")
    sec = (utc.astype("int64") // 10**9).where(utc.notna())
    owner = np.repeat(np.arange(len(t)), lens.to_numpy())
    bad = np.unique(owner[sec.isna().to_numpy()])
    sec = sec.to_numpy()
    starts = np.r_[0, np.cumsum(lens.to_numpy())[:-1]]
    t["point_ts"] = [list(sec[s:s + n].astype("int64")) if i not in set(bad) else None
                     for i, (s, n) in enumerate(zip(starts, lens))]
    t["poly"] = t.POLYLINE.apply(lambda p: [(round(lon, 6), round(lat, 6)) for lat, lon in json.loads(p)])
    t = t.drop(columns=["POLYLINE"])
    n_dst = len(bad)
    t = t[t.point_ts.notna()].copy()
    assert (t.poly.apply(len) == t.point_ts.apply(len)).all(), "points and timestamps differ in length"
    keep = t.poly.apply(lambda p: [lon != 0 and lat != 0 for lon, lat in p])
    n_zero = int(keep.apply(lambda k: len(k) - sum(k)).sum())
    t["poly"] = [[q for q, k in zip(p, kk) if k] for p, kk in zip(t.poly, keep)]
    t["point_ts"] = [[s for s, k in zip(p, kk) if k] for p, kk in zip(t.point_ts, keep)]
    t["n_points"] = t.point_ts.apply(len)
    t["TIMESTAMP"] = t.point_ts.apply(lambda p: int(p[0]))
    dur = t.point_ts.apply(lambda p: float(p[-1] - p[0]))
    assert np.allclose(dur, t.trip_travel_time_s), "trip_travel_time_s differs from last - first point time"
    t = t[(t.n_points >= MIN_POINTS) & (t.trip_travel_time_s >= MIN_DURATION_S)].copy()
    assert all(np.all(np.diff(p) > 0) for p in t.point_ts), "timestamps not strictly increasing"
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_parquet(OUT / "trips_sfbus.parquet", index=False)
    xy = np.array([q for p in t.poly for q in p])
    bbox = [float(xy[:, 0].min()), float(xy[:, 1].min()), float(xy[:, 0].max()), float(xy[:, 1].max())]
    (OUT / "bbox_sfbus.json").write_text(json.dumps(bbox))
    start = pd.to_datetime(t.TIMESTAMP, unit="s", utc=True).dt.tz_convert(TZ)
    print(f"[sfbus] runs {n_src:,}; dropped {n_dst} with a DST-gap/ambiguous point time; (0,0) points dropped "
          f"{n_zero}; kept {len(t):,} "
          f"({t.vehicle_id.nunique()} buses, {t.route_id.nunique()} routes)")
    print(f"[sfbus] points per run median {t.n_points.median():.0f} (max {t.n_points.max()}), "
          f"duration median {t.trip_travel_time_s.median() / 60:.1f} min; local start {start.min()} .. {start.max()}")
    print(f"[sfbus] bbox lon/lat {bbox}")


def edges(pbf):
    from pyrosm import OSM
    bbox = json.loads((OUT / "bbox_sfbus.json").read_text())
    pad_lon, pad_lat = 1000 / (111320 * np.cos(np.radians((bbox[1] + bbox[3]) / 2))), 1000 / 110540
    bb = [bbox[0] - pad_lon, bbox[1] - pad_lat, bbox[2] + pad_lon, bbox[3] + pad_lat]
    osm = OSM(pbf, bounding_box=bb)
    g = osm.get_network(network_type="all")
    n_all = len(g)
    g = g[g.highway.isin(BUS_HIGHWAYS)].copy()
    g = g[g.geometry.notna() & ~g.geometry.is_empty]
    assert g.id.is_unique, "duplicate way ids"
    g.to_file(OUT / "edges_sfbus.geojson", driver="GeoJSON")
    print(f"[sfbus] edges: {len(g):,} bus-usable ways of {n_all:,} in bbox {bb}; "
          f"busway {int((g.highway == 'busway').sum())}; total length {g["length"].sum() / 1000:.0f} km (OSM length col)")


if __name__ == "__main__":
    {"trips": trips, "edges": edges}[sys.argv[1]](sys.argv[2])

