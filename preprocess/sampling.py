import argparse
import ast
import json
import os
import re
import time
from hashlib import blake2b
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

REPO = Path(__file__).resolve().parents[1]
CONFIG_DIR = Path(__file__).resolve().parent / "configs"
RAW_ROOT = Path(os.environ.get("ETA_RAW_ROOT", REPO / "raw_data"))
DATA_ROOT = Path(os.environ.get("ETA_DATA_ROOT", REPO / "data"))

MIN_TT_S = 180.0
MAX_GAP_S, MAX_SLOW_SHARE, SLOW_MS = 180.0, 30.0, 1.0

COVERAGE_MIN = 70.0
PROXY_GAP = 15.0
MM_KEEP = {"sf": (1702 / 2000, 1821 / 2000), "porto": (1820 / 2000, 1618 / 2000)}


def preserve(path):
    path = Path(path)
    if not path.exists():
        return
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(path.stat().st_mtime))
    kept = path.with_name(f"{path.stem}_preserved_{stamp}{path.suffix}")
    n = 1
    while kept.exists():
        kept = path.with_name(f"{path.stem}_preserved_{stamp}_{n}{path.suffix}"); n += 1
    path.rename(kept)
    print(f"Existing {path.name} kept as {kept.name}")


class StepLog:
    def __init__(self):
        self.rows = []

    def __call__(self, step, n_in, n_out, note=""):
        self.rows.append(dict(step=step, trips_in=int(n_in), trips_out=int(n_out), dropped=int(n_in - n_out),
                              note=note))


def _objcol(items):
    a = np.empty(len(items), dtype=object)
    for i, v in enumerate(items):
        a[i] = v
    return a


def _parse(x):
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        try:
            return json.loads(s)
        except ValueError:
            return ast.literal_eval(s)
    return x


def normalize_poly(x):
    x = _parse(x)
    if x is None:
        return np.zeros((0, 2)), 0
    try:
        a = np.asarray(x, dtype=float)
        if a.ndim == 2 and a.shape[1] >= 2 and np.isfinite(a[:, :2]).all():
            return a[:, :2].copy(), 0
    except (TypeError, ValueError):
        pass
    out = []
    for pt in x:
        if pt is None or len(pt) < 2:
            continue
        lon, lat = float(pt[0]), float(pt[1])
        if np.isfinite(lon) and np.isfinite(lat):
            out.append((lon, lat))
    a = np.asarray(out, dtype=float).reshape(-1, 2)
    return a, len(x) - len(a)


def tripid_u64(s, salt):
    h = blake2b((salt + s).encode("utf-8"), digest_size=8)
    return np.int64(int.from_bytes(h.digest(), "big") & ((1 << 63) - 1))


def load_and_check(cfg, raw_dir, limit, log):
    inp = cfg["input"]
    path = Path(raw_dir) / inp["path"]
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
        if limit:
            df = df.iloc[:limit]
    else:
        df = pd.read_csv(path, nrows=limit)
    print(f"Loaded {path.name}: shape {df.shape}")
    print(f"Columns: {list(df.columns)}")
    log("input rows", len(df), len(df))

    def keep(mask, step, msg):
        nonlocal df
        n0 = len(df)
        df = df[np.asarray(mask, bool)].reset_index(drop=True)
        log(step, n0, len(df))
        print(f"{msg}, shape: {(len(df), n_cols)} (dropped {n0 - len(df)})")

    n_cols = df.shape[1]
    for col, val in (inp.get("row_filter") or {}).items():
        keep(df[col] == val, f"row filter {col} == {val}", f"After keeping {col} == {val}")

    res = [normalize_poly(x) for x in df[inp["poly_col"]]]
    df["_poly"] = _objcol([r[0] for r in res])
    fixed = sum(1 for r in res if r[1])
    log("polyline parsed", len(df), len(df), f"{fixed} trips had None/short/non-finite points removed")
    print(f"Polylines parsed: {len(df)} trips, {fixed} trips had None/short/non-finite points removed")
    keep(df["_poly"].map(len) > 0, "non-empty polyline", "After removing empty polylines")
    keep(df["_poly"].map(len) >= 2, "after >=2 points", "After removing short polylines")

    ts_col = inp.get("point_ts_col")
    if ts_col:
        df["_ts"] = df[ts_col].map(_parse)
        keep([isinstance(t, (list, tuple, np.ndarray)) and len(t) == len(p) for t, p in zip(df["_ts"], df["_poly"])],
             "point_ts same length as polyline", "After removing trips whose point_ts length != polyline length")
        keep([all(v is not None for v in t) for t in df["_ts"]], "point_ts none missing",
             "After removing trips with a missing point time")
        df["_ts"] = _objcol([np.asarray(t, dtype=np.int64) for t in df["_ts"]])
        keep([bool((np.diff(t) >= 0).all()) for t in df["_ts"]], "point_ts never decreasing",
             "After removing trips whose point times decrease")
        keep([int(t[0]) == int(s) for t, s in zip(df["_ts"], df[inp["start_col"]])],
             "TIMESTAMP == first point time", "After removing trips whose TIMESTAMP != first point time")
    else:
        df["_ts"] = None
        log("point_ts", len(df), len(df), "not in source")
        print("point_ts: not in source")

    if inp.get("tt_col"):
        df["_tt"] = pd.to_numeric(df[inp["tt_col"]])
        if ts_col:
            keep([int(t[-1]) - int(t[0]) == v for t, v in zip(df["_ts"], df["_tt"])],
                 "trip time == last - first point time", "After removing trips whose trip time != last - first")
    elif ts_col:
        df["_tt"] = [float(t[-1] - t[0]) for t in df["_ts"]]
    else:
        df["_tt"] = df["_poly"].map(lambda p: max(len(p) - 1, 0) * float(inp["interval_s"]))
        log("trip time", len(df), len(df), f"(points - 1) x {inp['interval_s']} s")
        print(f"trip_travel_time_s = (points - 1) x {inp['interval_s']} s")
    keep(df["_tt"] > 0, "trip time > 0", "After removing trips with trip time <= 0")
    lo, hi = inp.get("min_trip_time_s"), inp.get("max_trip_time_s")
    if lo is not None:
        keep(df["_tt"] >= lo, f"trip time >= {lo} s", f"After removing trips shorter than {lo} s")
    if hi is not None:
        keep(df["_tt"] < hi, f"trip time < {hi} s", f"After removing trips of {hi} s ({hi / 60:.0f} min) or longer")
    print(f"Mean trip time (s): {df['_tt'].mean()}")

    tid = cfg.get("trip_id", {})
    df["_id"] = df[inp["id_col"]]
    if tid.get("hash"):
        df["_id"] = df["_id"].astype(str).map(lambda s: tripid_u64(s, tid.get("salt", "TRIP_V1")))
    print(f"rows: {len(df)}")
    print(f"unique {'numeric ' if tid.get('hash') else ''}TRIP_IDs: {df['_id'].nunique()}")
    rep = df["_id"].duplicated(keep=False)
    if rep.any() and tid.get("on_duplicate", "drop") == "drop":
        n_ids = df.loc[rep, "_id"].nunique()
        keep(~rep.to_numpy(), "unique TRIP_ID", f"After removing {n_ids} repeated TRIP_IDs (all their rows)")
    else:
        log("unique TRIP_ID", len(df), len(df), f"{int(rep.sum())} rows with a repeated id kept (on_duplicate=report)"
            if rep.any() else "")
        if rep.any():
            print(f"Repeated TRIP_IDs kept (on_duplicate=report): {int(rep.sum())} rows")
    copy = pd.Series([(s, p.tobytes()) for s, p in zip(df[inp["start_col"]], df["_poly"])]).duplicated(keep=False)
    keep(~copy.to_numpy(), "no exact copy of another trip",
         f"After removing {int(copy.sum())} trips listed twice under two ids (both copies)")
    df["_start"] = pd.to_numeric(df[inp["start_col"]]).astype(np.int64)
    return df


_PI, _A, _EE = np.pi, 6378245.0, 0.00669342162296594323
_X_PI = _PI * 3000.0 / 180.0


def _out_of_china(lon, lat):
    return not (72.004 <= lon <= 137.8347 and 0.8293 <= lat <= 55.8271)


def _tlat(x, y):
    r = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * np.sqrt(abs(x))
    r += (20.0 * np.sin(6.0 * x * _PI) + 20.0 * np.sin(2.0 * x * _PI)) * 2.0 / 3.0
    r += (20.0 * np.sin(y * _PI) + 40.0 * np.sin(y / 3.0 * _PI)) * 2.0 / 3.0
    r += (160.0 * np.sin(y / 12.0 * _PI) + 320 * np.sin(y * _PI / 30.0)) * 2.0 / 3.0
    return r


def _tlon(x, y):
    r = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * np.sqrt(abs(x))
    r += (20.0 * np.sin(6.0 * x * _PI) + 20.0 * np.sin(2.0 * x * _PI)) * 2.0 / 3.0
    r += (20.0 * np.sin(x * _PI) + 40.0 * np.sin(x / 3.0 * _PI)) * 2.0 / 3.0
    r += (150.0 * np.sin(x / 12.0 * _PI) + 300.0 * np.sin(x / 30.0 * _PI)) * 2.0 / 3.0
    return r


def _wgs84_to_gcj02(lon, lat):
    if _out_of_china(lon, lat):
        return lon, lat
    dlat, dlon = _tlat(lon - 105.0, lat - 35.0), _tlon(lon - 105.0, lat - 35.0)
    rl = lat / 180.0 * _PI
    mg = 1 - _EE * np.sin(rl) ** 2
    sm = np.sqrt(mg)
    return (lon + (dlon * 180.0) / (_A / sm * np.cos(rl) * _PI),
            lat + (dlat * 180.0) / ((_A * (1 - _EE)) / (mg * sm) * _PI))


def _gcj02_to_wgs84(lon, lat, max_iter=8, tol=1e-7):
    if _out_of_china(lon, lat):
        return lon, lat
    wlon, wlat = lon, lat
    for _ in range(max_iter):
        glon, glat = _wgs84_to_gcj02(wlon, wlat)
        dlon, dlat = glon - lon, glat - lat
        wlon, wlat = wlon - dlon, wlat - dlat
        if abs(dlon) < tol and abs(dlat) < tol:
            break
    return wlon, wlat


def _bd09_to_wgs84(lon, lat):
    x, y = lon - 0.0065, lat - 0.006
    z = np.sqrt(x * x + y * y) - 0.00002 * np.sin(y * _X_PI)
    th = np.arctan2(y, x) - 0.000003 * np.cos(x * _X_PI)
    return _gcj02_to_wgs84(z * np.cos(th), z * np.sin(th))


def _bboxes(polys):
    n = np.fromiter((len(p) for p in polys), np.int64, len(polys))
    xy = np.concatenate(polys)
    st = np.r_[0, np.cumsum(n)[:-1]]
    return (np.minimum.reduceat(xy[:, 0], st), np.maximum.reduceat(xy[:, 0], st),
            np.minimum.reduceat(xy[:, 1], st), np.maximum.reduceat(xy[:, 1], st))


def _in_box(bb, box_, m):
    x0, x1, y0, y1 = bb
    minx, miny, maxx, maxy = box_
    return (x0 >= minx - m) & (x1 <= maxx + m) & (y0 >= miny - m) & (y1 <= maxy + m)


def _largest_part(g):
    from shapely.validation import make_valid
    g = make_valid(g)
    if g.geom_type in ("MultiPolygon", "GeometryCollection"):
        g = max([p for p in g.geoms if p.geom_type == "Polygon"], key=lambda p: p.area)
    if g.geom_type != "Polygon":
        raise RuntimeError(f"Expected Polygon after cleaning, got {g.geom_type}")
    return g


def _from_lines(lines):
    from shapely.ops import linemerge, polygonize_full, unary_union
    if not lines:
        raise RuntimeError("No polygon and no boundary lines to polygonize.")
    polys = polygonize_full(linemerge(unary_union(lines)))[0]
    pl = [g for g in getattr(polys, "geoms", [polys]) if g.geom_type == "Polygon"]
    if not pl:
        raise RuntimeError("Polygonization produced no polygons (boundary ways may be incomplete).")
    return _largest_part(unary_union(pl))


def admin_polygon(a, raw_dir, cache):
    if cache.exists():
        print(f"Area polygon read from {cache.name}")
        return gpd.read_file(cache).geometry.iloc[0]
    from pyrosm import OSM
    from shapely.geometry import Point
    from shapely.ops import unary_union
    t = time.time()
    pbf = str(Path(raw_dir) / a["pbf"])
    osm = OSM(pbf, bounding_box=a["load_bbox"]) if a.get("load_bbox") else OSM(pbf)
    bnd = osm.get_boundaries(**a["boundary_query"])
    print("bnd columns:", list(bnd.columns))
    rx = re.compile(a["name_regex"])
    name_mask = pd.Series(False, index=bnd.index)
    for c in [c for c in a["name_cols"] if c in bnd.columns]:
        s = bnd[c].astype(str).fillna("")
        mode = a.get("name_match", "match")
        name_mask |= (s.str.fullmatch(rx) if mode == "fullmatch" else
                      s.str.contains(rx, na=False) if mode == "contains" else s.str.match(rx))
    lvl_mask = pd.Series(True, index=bnd.index)
    if a.get("admin_levels"):
        if "boundary" in bnd.columns:
            lvl_mask &= bnd["boundary"].fillna("").str.lower().eq("administrative")
        if "admin_level" in bnd.columns:
            lvl_mask &= bnd["admin_level"].astype(str).isin(a["admin_levels"])
    cand, how = bnd[name_mask & lvl_mask], "name + admin level"
    geom = None
    if cand.empty and a.get("fallback") == "name_only":
        cand, how = bnd[name_mask], "name only (fallback)"
    elif cand.empty and a.get("fallback") == "nearest_large":
        pc = bnd[lvl_mask & bnd.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
        how = "nearest large admin polygon (fallback)"
        if pc.empty:
            geom = _from_lines(bnd[lvl_mask].geometry.tolist())
        else:
            ctr = gpd.GeoSeries([Point(*a["fallback_center"])], crs=4326).to_crs(a["metric_epsg"]).iloc[0]
            pm = pc.to_crs(a["metric_epsg"])
            pm = pm.assign(_d=pm.geometry.centroid.distance(ctr), _a=pm.geometry.area)
            near = pm[pm["_d"] <= 200_000]
            near = near if not near.empty else pm[pm["_d"] <= 400_000]
            geom = (_largest_part(unary_union(pc.geometry.tolist())) if near.empty else
                    gpd.GeoSeries([near.sort_values("_a", ascending=False).iloc[0].geometry],
                                  crs=a["metric_epsg"]).to_crs(4326).iloc[0])
    if geom is None:
        if cand.empty:
            raise RuntimeError(f"Could not find any boundary matching {a['name_regex']}")
        print(f"Matched geometry types ({how}):", cand.geometry.geom_type.value_counts().to_dict())
        if a["select"] == "union_largest_part":
            polys = cand[cand.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].geometry.tolist()
            geom = (_largest_part(unary_union(polys)) if polys else
                    _from_lines(cand[cand.geometry.geom_type.isin(["LineString", "MultiLineString",
                                                                    "LinearRing"])].geometry.tolist()))
        elif a["select"] == "largest_candidate":
            cm = cand.to_crs(a["metric_epsg"])
            geom = cm.assign(_a=cm.area).sort_values("_a", ascending=False).head(1).to_crs(4326).geometry.unary_union
        else:
            raise ValueError(a["select"])
    buf = gpd.GeoSeries([geom], crs=4326).to_crs(a["metric_epsg"]).buffer(a["buffer_m"]).to_crs(4326).iloc[0]
    if a.get("single_polygon") and buf.geom_type != "Polygon":
        buf = _largest_part(buf)
    if a.get("min_vertices") and len(buf.exterior.coords) < a["min_vertices"]:
        raise RuntimeError("Degenerate polygon (too few exterior vertices).")
    km2 = gpd.GeoSeries([buf], crs=4326).to_crs(a["metric_epsg"]).area.iloc[0] / 1e6
    print(f"Area polygon bounds (buffered {a['buffer_m']} m) [W,S,E,N]: {buf.bounds}")
    print(f"Buffered polygon area: {km2:.2f} km2 ({time.time() - t:.0f} s)")
    gpd.GeoSeries([buf], crs=4326).to_file(cache, driver="GeoJSON")
    print(f"Saved: {cache.name}")
    return gpd.read_file(cache).geometry.iloc[0]


def area_filter(df, cfg, raw_dir, report_dir, log):
    from shapely.geometry import box
    a = cfg["area"]
    poly = admin_polygon(a, raw_dir, Path(report_dir) / f"{cfg['folder']}_area_polygon.geojson")
    bounds = poly.bounds
    m_sw = a.get("swap_margin_deg", 0.001)

    polys = df["_poly"].tolist()
    bb = _bboxes(polys)
    fit = _in_box(bb, bounds, m_sw)
    fit_sw = _in_box((bb[2], bb[3], bb[0], bb[1]), bounds, m_sw)
    ymax = np.maximum(np.abs(bb[2]), np.abs(bb[3]))
    xmax = np.maximum(np.abs(bb[0]), np.abs(bb[1]))
    sw = (fit_sw & ~fit) | (~(fit_sw & ~fit) & (ymax > 90.0) & (xmax <= 90.0))
    for i in np.flatnonzero(sw):
        polys[i] = polys[i][:, ::-1].copy()
    df["_poly"] = _objcol(polys)
    log("lon/lat order", len(df), len(df), f"{int(sw.sum())} trips stored as (lat, lon), swapped")
    print(f"Coord order normalized for {int(sw.sum())} trips.")

    n0 = len(df)
    bb = _bboxes(df["_poly"].tolist())
    ok = (np.maximum(np.abs(bb[0]), np.abs(bb[1])) <= 180.0) & (np.maximum(np.abs(bb[2]), np.abs(bb[3])) <= 90.0)
    df = df[ok].reset_index(drop=True)
    log("coordinates are lon/lat degrees", n0, len(df))
    print(f"Trips whose coordinates are not lon/lat degrees: {n0 - len(df)}")

    cs = a.get("coord_sys", "WGS84").upper()
    if cs != "WGS84":
        f = {"GCJ02": _gcj02_to_wgs84, "BD09": _bd09_to_wgs84}[cs]
        df["_poly"] = _objcol([np.array([f(float(x), float(y)) for x, y in p]) for p in df["_poly"]])
        log(f"{cs} -> WGS84", len(df), len(df))
        print(f"Converted {len(df)} trips from {cs} to WGS84.")

    bb = _bboxes(df["_poly"].tolist())
    n_box = int(_in_box(bb, bounds, a.get("bbox_margin_deg", 0.001)).sum())
    log("inside area bbox (printed only)", len(df), len(df), f"{n_box} trips")
    print(f"{n_box} trips inside the {cfg['city']} bounding box")

    ex0, ey0, ex1, ey1 = gpd.read_file(REPO / "data" / cfg["folder"] / f"edges_{cfg['folder']}.geojson").total_bounds
    n0 = len(df)
    centre_src = df
    df = df[_in_box(bb, (ex0, ey0, ex1, ey1), 0.0)].reset_index(drop=True)
    log("inside the road network", n0, len(df), f"whole route in lon {ex0:.3f}..{ex1:.3f}, lat {ey0:.3f}..{ey1:.3f}")
    print(f"{len(df)} trips entirely inside the road network of {cfg['city']}")

    area_geom = poly
    c = a.get("centre_box")
    if c:
        src = centre_src if c.get("median_of", "input") == "input" else df
        xy = np.concatenate(src["_poly"].tolist())
        lon_c, lat_c = float(np.median(xy[:, 0])), float(np.median(xy[:, 1]))
        k = 111_320.0 * max(1e-12, np.cos(np.deg2rad(lat_c)))
        mm = c["half_size_km"] * 1000.0
        cb = [lon_c - mm / k, lat_c - mm / 111_320.0, lon_c + mm / k, lat_c + mm / 111_320.0]
        mg = max(c["margin_m"] / 111_320.0, c["margin_m"] / k)
        print(f"Trajectory activity center: ({lon_c}, {lat_c})")
        print(f"Center bbox: {cb}")
        n0 = len(df)
        df = df[_in_box(_bboxes(df["_poly"].tolist()), cb, mg)].reset_index(drop=True)
        log("inside centre box", n0, len(df), f"+-{c['half_size_km']} km, margin {c['margin_m']} m")
        print(f"Trips fully inside center bbox: {len(df)}")
        area_geom = poly.intersection(box(*cb))
        print(f"Clipped area bounds: {area_geom.bounds}")
    return df, area_geom


def bucket_and_sample(df, cfg, report_dir, log):
    b = cfg["buckets"]
    edges = np.arange(0, b["max_min"] + b["width_min"], b["width_min"])
    labels = [f"{edges[i]}-{edges[i + 1]} mins" for i in range(len(edges) - 1)]
    df["trip_time_bucket"] = pd.cut(df["_tt"] / 60, bins=edges, labels=labels, right=False)
    cnt = df["trip_time_bucket"].value_counts().reindex(labels).fillna(0).astype(int)
    tab = pd.DataFrame({"count": cnt, "proportion_%": cnt / max(1, cnt.sum()) * 100})
    tab.index.name = "trip_time_bucket"
    print(tab)
    tab.to_csv(Path(report_dir) / f"{cfg['folder']}_bucket_counts_in_area.csv")
    log("sampled", len(df), len(df), "every trip in the area kept")
    print(f"No sampling: all {len(df)} trips in the area kept")
    return df


def area_map(area_geom, path, zoom, tiles, attr=None):
    import folium
    from shapely.geometry import box, mapping
    minx, miny, maxx, maxy = area_geom.bounds
    m = folium.Map(location=[area_geom.centroid.y, area_geom.centroid.x], tiles=tiles, attr=attr, zoom_start=zoom)
    folium.GeoJson(mapping(area_geom), name="area",
                   style_function=lambda f: {"color": "#E53935", "weight": 2, "fill": True, "fillOpacity": 0.12}).add_to(m)
    folium.Rectangle([(miny, minx), (maxy, maxx)], color="#1E88E5", weight=2, dash_array="6,6", fill=False,
                     tooltip="bounding box").add_to(m)
    sliver = box(minx, miny, maxx, maxy).difference(area_geom)
    if not sliver.is_empty:
        folium.GeoJson(mapping(sliver), name="bbox minus area",
                       style_function=lambda f: {"color": "#777777", "weight": 1, "fill": True,
                                                 "fillOpacity": 0.10}).add_to(m)
    folium.LayerControl().add_to(m)
    m.save(str(path))
    print(f"Saved: {Path(path).name}")


def trip_heatmap(polys, area_geom, h, path):
    import folium
    from folium.plugins import HeatMap
    from pyproj import Transformer
    from shapely.geometry import mapping
    to_m = Transformer.from_crs(4326, h["metric_epsg"], always_xy=True)
    to_deg = Transformer.from_crs(h["metric_epsg"], 4326, always_xy=True)
    pts, step = [], float(h["step_m"])
    for p in polys:
        x, y = to_m.transform(p[:, 0], p[:, 1])
        cum = np.r_[0.0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
        if not np.isfinite(cum[-1]) or cum[-1] <= 0:
            continue
        d = np.minimum(np.arange(0.0, cum[-1] + step, step), cum[-1])
        lon, lat = to_deg.transform(np.interp(d, cum, x), np.interp(d, cum, y))
        pts.extend([[a_, b_] for a_, b_ in zip(lat.tolist(), lon.tolist())])
        if len(pts) > h["max_points"]:
            break
    m = folium.Map(location=[area_geom.centroid.y, area_geom.centroid.x], tiles=h.get("tiles", "OpenStreetMap"),
                   attr=h.get("tiles_attr"), zoom_start=h["zoom_start"])
    HeatMap(pts, radius=h["radius"], blur=h["blur"], max_zoom=13).add_to(m)
    folium.GeoJson(mapping(area_geom), style_function=lambda f: {"color": "#3949ab", "weight": 2, "fill": False}).add_to(m)
    m.save(str(path))
    print(f"Saved: {Path(path).name} | sampled points: {len(pts)} | step_m: {step:.0f}")


def time_of_week(starts, tz, path_png, path_csv):
    import calendar
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = pd.to_datetime(starts, unit="s", utc=True)
    t = t.dt.tz_convert(tz) if tz else t
    tab = pd.crosstab(t.dt.dayofweek, t.dt.hour).reindex(index=range(7), columns=range(24), fill_value=0)
    tab.to_csv(path_csv)
    fig, ax = plt.subplots(figsize=(16, 6))
    im = ax.imshow(tab.values, cmap="Blues", aspect="auto")
    for i in range(7):
        for j in range(24):
            ax.text(j, i, str(tab.values[i, j]), ha="center", va="center", fontsize=6)
    ax.set_xticks(range(24)); ax.set_yticks(range(7))
    ax.set_yticklabels([calendar.day_name[d][:3] for d in range(7)])
    ax.set_xlabel(f"Hour of Day ({tz or 'UTC'})"); ax.set_ylabel("Day of Week (0=Mon)")
    ax.set_title("Trips per Hour and Day of Week")
    fig.colorbar(im); fig.tight_layout(); fig.savefig(path_png, dpi=120); plt.close(fig)
    per_day = tab.sum(axis=1)
    per_day.index = [calendar.day_name[d] for d in per_day.index]
    print(per_day.rename("TRIP_ID").to_string())


def point_diagnostics(df, interval_s, path):
    R = 6_371_000.0
    d_all, dt_all = [], []
    for p, t in zip(df["_poly"], df["_ts"]):
        lon, lat = np.radians(p[:, 0]), np.radians(p[:, 1])
        a = np.sin(np.diff(lat) / 2) ** 2 + np.cos(lat[:-1]) * np.cos(lat[1:]) * np.sin(np.diff(lon) / 2) ** 2
        d_all.append(2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1))))
        dt_all.append(np.diff(t).astype(float) if t is not None else np.full(len(p) - 1, float(interval_s or np.nan)))
    d, dt = np.concatenate(d_all), np.concatenate(dt_all)
    ok = np.isfinite(dt) & (dt > 0)
    v = d[ok] / dt[ok] * 3.6
    pct = [50, 75, 90, 95, 99, 99.5, 99.9, 99.99, 100]
    rows = []
    for name, x, unit in (("dt (seconds) between consecutive points", dt[ok], "s"),
                          ("dist_m between consecutive points", d, "m"), ("Implied speed (km/h)", v, "km/h")):
        if not len(x):
            continue
        print(f"=== {name} ===")
        r = dict(quantity=name)
        for p_ in pct:
            r[f"p{p_}"] = float(np.percentile(x, p_))
            print(f"  p{p_:>6}: {r[f'p{p_}']:.1f} {unit}")
        rows.append(r)
    print("=== Counts above speed thresholds ===")
    for thr in (60, 80, 100, 120, 140, 160, 180, 200, 250, 300, 500, 1000):
        cnt = int((v > thr).sum())
        print(f"  > {thr:>5} km/h: {cnt:>12,}  ({100 * cnt / max(1, len(v)):.4f}%)")
        rows.append(dict(quantity=f"steps faster than {thr} km/h", p100=cnt))
    rows.append(dict(quantity="steps with zero gap", p100=int((dt == 0).sum())))
    rows.append(dict(quantity="steps with zero distance (repeated point)", p100=int((d == 0).sum())))
    print(f"Steps with zero gap: {int((dt == 0).sum())}; with zero distance: {int((d == 0).sum())}")
    pd.DataFrame(rows).to_csv(path, index=False)


def hav(lon1, lat1, lon2, lat2):
    p = np.pi / 180
    a = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    return 2 * 6371000.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def gps_rules(pool):
    n = pool["n_points"].to_numpy()
    tr = np.repeat(np.arange(len(n)), n)
    off = np.r_[0, np.cumsum(n)[:-1]]
    xy = np.fromstring(",".join(pool["poly"].str.replace("[", "", regex=False).str.replace("]", "", regex=False)),
                       sep=",").reshape(-1, 2)
    assert len(xy) == n.sum(), "poly and n_points disagree"
    tt = pool["trip_travel_time_s"].to_numpy(float)
    if "point_ts" in pool.columns:
        t = np.fromstring(",".join(pool["point_ts"].str.strip("[]")), sep=",")
        assert len(t) == len(xy), "point_ts and poly disagree"
    else:
        t = pool["TIMESTAMP"].to_numpy(float)[tr] + tt[tr] / np.maximum(n[tr] - 1, 1) * (np.arange(len(tr)) - off[tr])
    same = tr[1:] == tr[:-1]
    st, dt = tr[:-1][same], np.diff(t)[same]
    step = hav(xy[:-1, 0][same], xy[:-1, 1][same], xy[1:, 0][same], xy[1:, 1][same])
    gap = np.zeros(len(n))
    np.maximum.at(gap, st, dt)
    slow = np.bincount(st, weights=dt * (step / np.maximum(dt, 1e-6) < SLOW_MS), minlength=len(n))
    dur = np.maximum(t[off + n - 1] - t[off], 1e-6)
    clean = (tt >= MIN_TT_S) & (gap <= MAX_GAP_S) & (slow / dur * 100 <= MAX_SLOW_SHARE)
    return clean, xy[:, 0], xy[:, 1], t, tr


def snap_points(X, Y, edges_path, crs):
    if len(X) > 2_000_000:
        from concurrent.futures import ProcessPoolExecutor
        parts = np.array_split(np.arange(len(X)), int(np.ceil(len(X) / 1_500_000)))
        with ProcessPoolExecutor(max_workers=min(len(parts), os.cpu_count() or 1)) as ex:
            return np.concatenate(list(ex.map(snap_points, [X[p] for p in parts], [Y[p] for p in parts],
                                              [edges_path] * len(parts), [crs] * len(parts))))
    import shapely
    from pyproj import Transformer
    g = gpd.read_file(edges_path)[["id", "geometry"]]
    g = g[g.geometry.notna()].set_crs("EPSG:4326", allow_override=True).to_crs(crs)
    tree, ids = shapely.STRtree(g.geometry.values), g["id"].astype("int64").to_numpy()
    mx, my = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(X, Y)
    way = np.full(len(X), -1, np.int64)
    q = tree.query_nearest(shapely.points(mx, my), max_distance=25.0)
    way[q[0]] = ids[q[1]]
    return way


def trip_coverage(X, Y, T, TR, use, n_trips, edges_path, crs):
    m = use[TR]
    x, y, t, tid = X[m], Y[m], T[m], TR[m]
    ut, loc = np.unique(tid, return_inverse=True)
    first, last = np.r_[True, loc[1:] != loc[:-1]], np.r_[loc[1:] != loc[:-1], True]
    if np.median(np.diff(t)[~first[1:]]) > 20:
        t0, t1 = t[first], t[last]
        k = (np.floor((t1 - t0) / 15.0) + 2).astype(int)
        q = np.repeat(np.arange(len(ut)), k)
        tq = np.minimum(np.repeat(t0, k) + (np.arange(k.sum()) - np.repeat(np.r_[0, np.cumsum(k)[:-1]], k)) * 15.0,
                        np.repeat(t1, k))
        tg = t + loc * 1e7
        o = np.argsort(tg, kind="mergesort")
        x, y = np.interp(tq + q * 1e7, tg[o], x[o]), np.interp(tq + q * 1e7, tg[o], y[o])
        t, loc = tq, q
    way = snap_points(x, y, edges_path, crs)
    ok = way >= 0
    entry = np.r_[True, (way[1:] != way[:-1]) | (loc[1:] != loc[:-1])] & ok
    ts = np.round(t).astype(np.int64) - int(np.round(t.min())) + 1200
    wr = pd.factorize(way)[0].astype(np.int64)
    s1 = np.sort(wr[ok] * 2 ** 32 + ts[ok])
    k1 = wr[entry] * 2 ** 32 + ts[entry]
    n_all = np.searchsorted(s1, k1, "left") - np.searchsorted(s1, k1 - 600, "left")
    pr = pd.factorize(loc.astype(np.int64) * 10 ** 8 + wr)[0].astype(np.int64)
    s3 = np.sort(pr[ok] * 2 ** 32 + ts[ok])
    k3 = pr[entry] * 2 ** 32 + ts[entry]
    n_own = np.searchsorted(s3, k3, "left") - np.searchsorted(s3, k3 - 600, "left")
    ne, nc = np.zeros(n_trips), np.zeros(n_trips)
    ne[ut] = np.bincount(loc[entry], minlength=len(ut))
    nc[ut] = np.bincount(loc[entry], weights=(n_all > n_own).astype(float), minlength=len(ut))
    return ne, nc


def search_windows(day, mins, ne, nc, ok, sel, keep):
    from scipy.optimize import linprog
    E = list(sel["length_bands"]) + [np.inf]
    nb = len(E) - 1
    bi = np.digitize(mins, E) - 1
    ok = ok & (bi >= 0) & (bi < nb)
    nd = int(day[ok].max()) + 1
    A = np.zeros((4, nd, nb))
    for j, v in enumerate((np.ones(ok.sum()), ne[ok], nc[ok], mins[ok])):
        np.add.at(A[j], (day[ok], bi[ok]), v)
    cA = np.concatenate([np.zeros((4, 1, nb)), np.cumsum(A, 1)], 1)
    kp = np.array([keep[0] if lo_ < 20 else keep[1] for lo_ in E[:-1]])
    th = (sel.get("coverage_min", COVERAGE_MIN) - PROXY_GAP) / 100
    lo, hi = sel["mean_minutes"]
    rows = []
    for D in range(int(sel["min_days"]), nd + 1):
        best = None
        for st in range(0, nd - D + 1):
            n, e, c, mn = cA[:, st + D] - cA[:, st]
            av = np.floor(n * kp)
            nz = n > 0
            m, er, cr = (np.where(nz, v / np.maximum(n, 1), 0) for v in (mn, e, c))
            lb = np.where(np.array(E[:-1]) >= 30, np.minimum(av, sel["long_band_min_trips"]), 0)
            r = linprog(-np.ones(nb), A_ub=[m - hi, lo - m, th * er - cr,
                                            sel["short_band_min_share"] * np.ones(nb) - np.eye(nb)[0]],
                        b_ub=[0, 0, 0, 0], bounds=list(zip(lb, av)), method="highs")
            if r.status == 0 and (best is None or r.x.sum() > best[1].sum()):
                best = (st, r.x, (r.x @ cr) / max(r.x @ er, 1e-9) * 100 + PROXY_GAP, (r.x @ m) / max(r.x.sum(), 1e-9))
        if best:
            rows.append((D,) + best)
            if best[1].sum() >= sel["min_trips"]:
                break
    return rows


def run(cfg, limit=None):
    city, folder = cfg["city"], cfg["folder"]
    raw_dir = RAW_ROOT / folder
    data_dir = DATA_ROOT / folder
    report_dir = data_dir / ("sampling_report" + ("_proto" if limit else ""))
    report_dir.mkdir(parents=True, exist_ok=True)
    log, t0 = StepLog(), time.time()
    print(f"===== {city}: {raw_dir} -> {data_dir}" + (f" (prototype, first {limit} rows)" if limit else ""))
    v = cfg["visual"]
    from hashlib import sha1
    pool = report_dir / f"{folder}_area_pool.parquet"
    meta_path = report_dir / f"{folder}_area_pool_meta.json"
    meta = dict(raw=cfg["input"]["path"], raw_bytes=(raw_dir / cfg["input"]["path"]).stat().st_size,
                pbf=cfg["area"]["pbf"], pbf_bytes=(raw_dir / cfg["area"]["pbf"]).stat().st_size,
                input=cfg["input"], trip_id=cfg.get("trip_id", {}), area=cfg["area"], buckets=cfg["buckets"],
                local_tz=(cfg.get("map_matching") or {}).get("local_tz"), limit=limit,
                edges_bytes=(REPO / "data" / folder / f"edges_{folder}.geojson").stat().st_size,
                area_sha1=(sha1((REPO / "data" / folder / "sampling_report" / f"{folder}_area_polygon.geojson")
                                .read_bytes()).hexdigest()
                           if (REPO / "data" / folder / "sampling_report" / f"{folder}_area_polygon.geojson").exists()
                           else None))
    if pool.exists() and meta_path.exists() and \
            {k: x for k, x in json.loads(meta_path.read_text()).items() if k != "steps"} == json.loads(json.dumps(meta)):
        log.rows = json.loads(meta_path.read_text())["steps"]
        area_geom = admin_polygon(cfg["area"], raw_dir,
                                  REPO / "data" / folder / "sampling_report" / f"{folder}_area_polygon.geojson")
        print(f"Area pool read from cache: {pool.name} (same raw data, config and area polygon)")
    else:
        df = load_and_check(cfg, raw_dir, limit, log)
        area_df, area_geom = area_filter(df, cfg, raw_dir, REPO / "data" / folder / "sampling_report", log)
        sample = bucket_and_sample(area_df, cfg, report_dir, log)

        out = pd.DataFrame({
            "TRIP_ID": sample["_id"].astype(str).to_numpy(),
            "TIMESTAMP": sample["_start"].to_numpy(),
            "trip_travel_time_s": sample["_tt"].to_numpy(),
            "poly": [json.dumps(p.tolist(), separators=(",", ":")) for p in sample["_poly"]],
            "n_points": sample["_poly"].map(len).to_numpy(),
            "trip_time_bucket": sample["trip_time_bucket"].astype(str).to_numpy(),
        })
        if cfg["input"].get("point_ts_col"):
            out.insert(4, "point_ts", [json.dumps(t.tolist(), separators=(",", ":")) for t in sample["_ts"]])
        for c in cfg["input"].get("keep_cols", []):
            out[c] = [json.dumps(np.asarray(x).tolist()) if isinstance(x, (list, np.ndarray)) else x for x in sample[c]]
        dep = pd.to_datetime(out["TIMESTAMP"], unit="s")
        tz = (cfg.get("map_matching") or {}).get("local_tz")
        if tz:
            dep = dep.dt.tz_localize("UTC").dt.tz_convert(tz).dt.tz_localize(None)
        out["departure_time"] = dep
        out["hour_of_day"] = dep.dt.hour
        out["day_of_week"] = dep.dt.dayofweek
        out["month"] = dep.dt.month
        out["day_of_month"] = dep.dt.day
        out["week_of_year"] = dep.dt.isocalendar().week.astype(int)
        out["day_of_year"] = dep.dt.dayofyear
        out.to_parquet(pool, index=False)
        meta["area_sha1"] = sha1((REPO / "data" / folder / "sampling_report" / f"{folder}_area_polygon.geojson")
                                 .read_bytes()).hexdigest()
        meta_path.write_text(json.dumps(dict(meta, steps=log.rows), indent=1))
        print(f"Saved: {pool.name}, shape {out.shape} ({time.time() - t0:.0f} s)")
        area_map(area_geom, report_dir / f"{folder}_area_vs_bbox.html", v["zoom_start"], v.get("tiles", "OpenStreetMap"),
                 v.get("tiles_attr"))


    sel = cfg.get("selection")
    if not sel:
        raise SystemExit(f"preprocess/configs/{folder}.json: no 'selection' block (the criteria of the final trips)")
    P = pd.read_parquet(pool)
    clean, X, Y, T, TR = gps_rules(P)
    log("GPS rules (trip time, GPS gap, crawling)", len(P), int(clean.sum()))
    print(f"[{city}] GPS rules: {int(clean.sum()):,} clean trips of {len(P):,} ({time.time() - t0:.0f} s)")
    cov_path = report_dir / f"{folder}_trip_coverage.npz"
    cov_key = json.dumps(dict(meta, rules=[MIN_TT_S, MAX_GAP_S, MAX_SLOW_SHARE, SLOW_MS]), sort_keys=True, default=str)
    z = np.load(cov_path) if cov_path.exists() else None
    if z is not None and str(z["key"]) == cov_key and len(z["ne"]) == len(P):
        ne, nc = z["ne"], z["nc"]
        print(f"Trip coverage read from cache: {cov_path.name}")
    else:
        ne, nc = trip_coverage(X, Y, T, TR, clean, len(P), REPO / "data" / folder / f"edges_{folder}.geojson",
                               cfg["map_matching"]["crs_metric"])
        np.savez(cov_path, ne=ne, nc=nc, key=cov_key)
        print(f"Saved: {cov_path.name} ({time.time() - t0:.0f} s)")
    del X, Y, T, TR
    dep = pd.to_datetime(P["departure_time"])
    day = (dep.dt.normalize() - dep.dt.normalize().min()).dt.days.to_numpy()
    mins = P["trip_travel_time_s"].to_numpy(float) / 60
    keep = MM_KEEP.get(folder, tuple(np.mean(list(MM_KEEP.values()), axis=0)))
    ok = (clean & (ne > 0) & (nc / np.maximum(ne, 1) >= sel["trip_coverage_floor"])
          & (P["TIMESTAMP"].to_numpy(float) >= P["TIMESTAMP"].to_numpy(float)[clean].min() + 7200))
    rows = search_windows(day, mins, ne, nc, ok, sel, keep)
    reach = [r for r in rows if r[2].sum() >= sel["min_trips"]]
    if not reach:
        raise SystemExit(f"[{city}] no window of >= {sel['min_days']} days reaches {sel['min_trips']:,} trips (most: "
                         f"{max((r[2].sum() for r in rows), default=0):,.0f}); nothing written")
    low = reach[0]
    D, st, x, cov, mean = low
    E = list(sel["length_bands"]) + [np.inf]
    bi = np.digitize(mins, E) - 1
    kp = np.array([keep[0] if e < 20 else keep[1] for e in E[:-1]])
    d0 = dep.dt.normalize().min()
    inwin = (day >= st) & (day < st + D)
    rng = np.random.default_rng(42)
    pick_ix = np.concatenate([np.sort(rng.choice(np.flatnonzero(ok & inwin & (bi == b)),
                                                 size=min(int((ok & inwin & (bi == b)).sum()), int(round(x[b] / kp[b]))),
                                                 replace=False)) for b in range(len(E) - 1)])
    a_ts = d0 + pd.Timedelta(days=st)
    hist_m = (clean & ~np.isin(np.arange(len(P)), pick_ix) & (dep < a_ts + pd.Timedelta(days=D)).to_numpy()
              & (dep + pd.to_timedelta(P["trip_travel_time_s"].astype(float), unit="s")
                 > a_ts - pd.Timedelta(hours=2)).to_numpy())
    pick = P.iloc[pick_ix].sort_values("TIMESTAMP", kind="mergesort")
    hist = P[hist_m].sort_values("TIMESTAMP", kind="mergesort")
    log("eligible (clean, trip coverage >= floor)", int(clean.sum()), int(ok.sum()), f"floor {sel['trip_coverage_floor']}")
    log("trips to predict", int((ok & inwin).sum()), len(pick), f"{D} days from {a_ts.date()}")
    log("history-only trips", int(clean.sum()), len(hist), "every other clean trip on the road in the window or the 2 h "
        "before it")
    print(f"[{city}] chosen window {a_ts.date()} + {D} days; coverage (estimated) {cov:.1f}%")
    tab = {}
    for name, r in ((city, (D, st, x, cov, mean)),):
        a_r = d0 + pd.Timedelta(days=r[1])
        e_r = a_r + pd.Timedelta(days=r[0] - 1)
        tm = int((clean & (dep < a_r + pd.Timedelta(days=r[0])).to_numpy()
                  & (dep + pd.to_timedelta(P["trip_travel_time_s"].astype(float), unit="s")
                     > a_r - pd.Timedelta(hours=2)).to_numpy()).sum())
        if tm != len(pick) + len(hist):
            raise SystemExit(f"[{city}] trips to map-match {tm:,} != trips written {len(pick) + len(hist):,}; nothing written")
        sent = sum(round(v / k) for v, k in zip(r[2], kp))
        tab[name] = {f"1. Coverage ≥ {sel.get('coverage_min', COVERAGE_MIN):.0f}% (estimated)": f"{r[3]:.1f}%",
                     f"2. Trips kept after map matching ≥ N ({sel['min_trips']:,})":
                         f"{r[2].sum():,.0f} ({sent:,} trips to predict sent)",
                     f"3. ≥ {sel['min_days']} days": f"{r[0]} days, {a_r.day} {a_r:%b}"
                                                     + (f" {a_r:%Y}" if a_r.year != e_r.year else "")
                                                     + f" – {e_r.day} {e_r:%b %Y}",
                     "4. Trips per band " + " / ".join(f"{a}–{b}" if np.isfinite(b) else f"{a}+" for a, b in zip(E[:-1], E[1:]))
                     + " min": " / ".join(f"{v:,.0f}" for v in r[2]),
                     f"5. Mean trip time {sel['mean_minutes'][0]}–{sel['mean_minutes'][1]} min": f"{r[4]:.1f}",
                     "Trips to map-match in total": f"{tm:,} ({len(pick):,} + {len(hist):,} history)"}
    tab = pd.DataFrame(tab)
    tab.to_csv(report_dir / f"{folder}_selection.csv")
    print(tab.to_string())

    out = pd.concat([pick.assign(history_only=False), hist.assign(history_only=True)])
    dup = int(out.duplicated(["TIMESTAMP", "poly"]).sum())
    log("no trip twice under two IDs (leak check)", len(out), len(out) - dup)
    if dup:
        raise SystemExit(f"[{city}] leak check failed: {dup} trips are in the sample twice under two IDs; nothing written")
    print(f"[{city}] leak check, no trip twice under two IDs: PASS, 0 exact copies in {len(out):,} trips")
    f_out = (report_dir if limit else data_dir) / f"sampled_{folder}.csv"
    preserve(f_out)
    out.to_csv(f_out, index=False)
    print(f"Saved: {f_out.name}, shape {out.shape} ({time.time() - t0:.0f} s)")

    steps = pd.DataFrame(log.rows)
    steps.to_csv(report_dir / f"{folder}_sampling_steps.csv", index=False)
    print(steps.to_string(index=False))

    from shapely.geometry import box
    trip_heatmap(_objcol([np.asarray(json.loads(p), float) for p in pick.poly]),
                 box(*gpd.read_file(REPO / "data" / folder / f"edges_{folder}.geojson").total_bounds), v,
                 report_dir / f"{folder}_trips_heatmap.html")
    time_of_week(pick.TIMESTAMP, v.get("tz"), report_dir / f"{folder}_trips_by_hour_weekday.png",
                 report_dir / f"{folder}_trips_by_hour_weekday.csv")
    point_diagnostics(pd.DataFrame({"_poly": _objcol([np.asarray(json.loads(p), float) for p in pick.poly]),
                                    "_ts": (_objcol([np.asarray(json.loads(x), np.int64) for x in pick.point_ts])
                                            if "point_ts" in pick.columns else None)}),
                      cfg["input"].get("interval_s"), report_dir / f"{folder}_point_diagnostics.csv")

    link = data_dir / f"edges_{folder}.geojson"
    if not link.exists():
        link.symlink_to(REPO / "data" / folder / f"edges_{folder}.geojson")
    print(f"===== {city} done: {len(P):,} trips in the pool -> {len(pick):,} to predict + {len(hist):,} history-only "
          f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="raw_data/<city> trip table -> data/<city>/sampled_<city>.csv + edges")
    ap.add_argument("city", help="name of preprocess/configs/<city>.json")
    ap.add_argument("--limit", type=int, default=None, help="prototype: first N input rows (outputs go to the report folder)")
    args = ap.parse_args()
    run(json.load(open(CONFIG_DIR / f"{args.city}.json", encoding="utf-8")), args.limit)

