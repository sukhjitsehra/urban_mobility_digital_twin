import math
import ast
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from threading import local
from typing import Dict, Iterable, List, Optional, Tuple, Sequence

import numpy as np
import pandas as pd
import requests
import geopandas as gpd
from pyproj import Transformer
from shapely.geometry import LineString, Point, MultiLineString

try:
    from shapely import STRtree
except ImportError:
    from shapely.strtree import STRtree

from shapely.ops import nearest_points as _nearest_points, linemerge

OSRM_URL = os.environ.get("OSRM_URL", "http://localhost:5001")
MAX_POINTS = 100
SAMPLING_SECONDS = 15
BUF_M = float(os.environ.get("MM_BUF_M", 5.0))
RADIUS_M = float(os.environ.get("MM_RADIUS_M", 25.0))
CRS_METRIC = os.environ.get("CRS_METRIC", "EPSG:3763")
LOCAL_TZ = os.environ.get("LOCAL_TZ") or None
DYN_ALIGN = os.environ.get("DYN_ALIGN", "departure")

BIN_SECONDS = 600
SLOTS = int(os.environ.get("MM_SLOTS", 18))
MIN_DT_S = 0.1
MIN_DD_M = 0.1
MAX_VALID_SPEED_MS = 80

_thread = local()

def get_session() -> requests.Session:
    if not hasattr(_thread, "session"):
        _thread.session = requests.Session()
    return _thread.session

def _chunked(seq: List, n: int):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]

def build_spatial_index(edges_gdf: gpd.GeoDataFrame, crs_metric: str = CRS_METRIC):
    valid = edges_gdf.geometry.notnull() & ~edges_gdf.geometry.is_empty
    edges = edges_gdf.loc[valid, ["id", "geometry"]].copy()
    edges = edges.set_geometry("geometry").set_crs("EPSG:4326", allow_override=True)

    edges_proj = edges.to_crs(crs_metric)
    edge_lengths_m = edges_proj.geometry.length.to_numpy()
    edge_geoms_proj = np.array(edges_proj.geometry.to_list(), dtype=object)
    edge_ids = edges_proj["id"].to_numpy()

    tree = STRtree(edge_geoms_proj)

    transformer = Transformer.from_crs("EPSG:4326", crs_metric, always_xy=True)
    inv_transformer = Transformer.from_crs(crs_metric, "EPSG:4326", always_xy=True)

    id_to_idx = {int(eid): idx for idx, eid in enumerate(edge_ids)}
    return (
        edges_proj,
        edge_geoms_proj,
        edge_ids,
        edge_lengths_m,
        tree,
        transformer,
        inv_transformer,
        id_to_idx,
    )

_thread = local()

def get_session() -> requests.Session:
    if not hasattr(_thread, "session"):
        _thread.session = requests.Session()
    return _thread.session

def _chunked(seq: List, n: int):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]

def osrm_full_geometry(
    poly: List[Tuple[float, float]],
    timestamps: Optional[List[int]] = None,
    radiuses: Optional[List[float]] = None,
    osrm_url: str = OSRM_URL,
    max_points: int = MAX_POINTS,
    timeout: int = 15,
) -> Tuple[List[Tuple[float, float]], float, float, List[Dict]]:
    coords_geom: List[Tuple[float, float]] = []
    tot_dist_m = 0.0
    tot_dur_s = 0.0
    steps: List[Dict] = []

    if not poly or len(poly) < 2:
        return coords_geom, tot_dist_m, tot_dur_s, steps

    sess = get_session()
    _thread.osrm_chunks, _thread.osrm_failed, _thread.osrm_err = 0, 0, ""

    for seg_idx, seg in enumerate(_chunked(poly, max_points)):
        if len(seg) < 2:
            continue
        _thread.osrm_chunks += 1
        lonlats = ";".join(f"{lon},{lat}" for lon, lat in seg)

        params = {"overview": "full", "geometries": "geojson", "steps": "true"}

        if timestamps is not None:
            ts_chunk = timestamps[seg_idx * max_points : seg_idx * max_points + len(seg)]
            params["timestamps"] = ";".join(str(t) for t in ts_chunk)
        if radiuses is not None:
            rad_chunk = radiuses[seg_idx * max_points : seg_idx * max_points + len(seg)]
            params["radiuses"] = ";".join("{:.3f}".format(r if np.isfinite(r) else -1) for r in rad_chunk)

        try:
            r = sess.get(f"{osrm_url}/match/v1/driving/{lonlats}", params=params, timeout=timeout)
            r.raise_for_status()
            js = r.json()
            matchings = js.get("matchings", [])
            for m in matchings:
                tot_dist_m += float(m.get("distance", 0.0))
                tot_dur_s += float(m.get("duration", 0.0))
                for leg in m.get("legs", []):
                    for st in leg.get("steps", []):
                        coords = st["geometry"]["coordinates"]
                        coords_geom.extend(coords)
                        steps.append({
                            "coords": coords,
                            "distance": float(st.get("distance", 0.0)),
                            "duration": float(st.get("duration", 0.0)),
                        })
        except requests.RequestException as e:
            _thread.osrm_failed += 1
            _thread.osrm_err = str(getattr(getattr(e, "response", None), "text", "") or e)[:200]
            continue

    deduped: List[Tuple[float, float]] = []
    for pt in coords_geom:
        t = tuple(pt)
        if not deduped or t != deduped[-1]:
            deduped.append(t)

    return deduped, tot_dist_m, tot_dur_s, steps

from math import atan2, degrees, hypot, cos, radians

def _bearing_deg_xy(x1: float, y1: float, x2: float, y2: float) -> float:
    ang = degrees(atan2((x2 - x1), (y2 - y1)))
    return (ang + 360.0) % 360.0

def _cos_delta(b1: float, b2: float) -> float:
    d = abs(b1 - b2)
    if d > 180.0:
        d = 360.0 - d
    return cos(radians(d))

def _closest_edge_segment_bearing(edge, x, y):
    if isinstance(edge, MultiLineString):
        coords = []
        for linestring in edge.geoms:
            coords.extend(linestring.coords)
    else:
        coords = list(edge.coords)

    best_i = 0
    best_d2 = float("inf")
    for i, (ex, ey) in enumerate(coords):
        d2 = (ex - x) ** 2 + (ey - y) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best_i = i

    cand_segs = []
    if best_i > 0:
        cand_segs.append((coords[best_i - 1], coords[best_i]))
    if best_i < len(coords) - 1:
        cand_segs.append((coords[best_i], coords[best_i + 1]))
    if not cand_segs and len(coords) >= 2:
        cand_segs.append((coords[0], coords[1]))

    best_seg = cand_segs[0]
    best_d = float("inf")
    for (a, b) in cand_segs:
        ax, ay = a; bx, by = b
        vx, vy = bx - ax, by - ay
        wx, wy = x - ax, y - ay
        c1 = vx * wx + vy * wy
        if c1 <= 0:
            d = hypot(x - ax, y - ay)
        else:
            c2 = vx * vx + vy * vy
            if c2 <= c1:
                d = hypot(x - bx, y - by)
            else:
                t = c1 / c2
                px, py = ax + t * vx, ay + t * vy
                d = hypot(x - px, y - py)
        if d < best_d:
            best_d = d
            best_seg = (a, b)
    (ax, ay), (bx, by) = best_seg
    return _bearing_deg_xy(ax, ay, bx, by)

def _projected_length_on_edge(edge: LineString, x1: float, y1: float, x2: float, y2: float) -> float:
    p1 = Point(x1, y1)
    p2 = Point(x2, y2)
    try:
        d1 = edge.project(p1)
        d2 = edge.project(p2)
        return abs(float(d2 - d1))
    except Exception:
        return 0.0

def _to_metric_xy(lon: float, lat: float, transformer: Transformer) -> Tuple[float, float]:
    x, y = transformer.transform(lon, lat)
    return float(x), float(y)

def dedupe_consecutive(seq: Iterable) -> List:
    out = []
    prev = object()
    for s in seq:
        if s != prev:
            out.append(s)
            prev = s
    return out

def assign_segments(
    coords: List[Tuple[float, float]],
    tree: STRtree,
    edge_geoms_proj: np.ndarray,
    edge_ids: np.ndarray,
    transformer: Transformer,
    id_to_idx: Dict[int, int],
    buf_m: float = BUF_M,
    nearest_cutoff_m: float = 35.0,
    w_len: float = 1.0,
    w_dir: float = 0.5,
    w_dist: float = 0.2,
) -> Tuple[
    List[int], Dict[int, float], float, Dict[str, List[float]],
    List[int], List[float], List[float]
]:
    if not coords or len(coords) < 2:
        return [], {}, 0.0, {"overlap_ratio": [], "cos_delta": [], "nearest_dist": [], "used_fallback": []}, [], [], [0.0]

    way_length_contrib: Dict[int, float] = defaultdict(float)
    assigned_way_ids: List[int] = []
    total_len_m = 0.0

    diag = {"overlap_ratio": [], "cos_delta": [], "nearest_dist": [], "used_fallback": []}

    prev_way: Optional[int] = None
    prev_run_len = 0

    seg_way_ids: List[int] = []
    seg_len_list: List[float] = []
    vertex_cumlen: List[float] = [0.0]
    cum_len = 0.0

    for (lon1, lat1), (lon2, lat2) in zip(coords, coords[1:]):
        x1, y1 = _to_metric_xy(lon1, lat1, transformer)
        x2, y2 = _to_metric_xy(lon2, lat2, transformer)
        seg = LineString([(x1, y1), (x2, y2)])
        seg_len = float(seg.length)
        if seg_len == 0.0:
            seg_way_ids.append(prev_way if prev_way is not None else -1)
            seg_len_list.append(0.0)
            vertex_cumlen.append(cum_len)
            diag["overlap_ratio"].append(0.0)
            diag["cos_delta"].append(0.0)
            diag["nearest_dist"].append(0.0)
            diag["used_fallback"].append(True)
            continue

        seg_bearing = _bearing_deg_xy(x1, y1, x2, y2)
        buf = seg.buffer(buf_m)
        cand_idx = tree.query(buf, predicate="intersects")

        best_idx = -1
        best_score = -1e18
        best_proj_len = 0.0
        best_cos = 0.0
        best_near_d = float("inf")

        mx, my = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        mid_pt = Point(mx, my)

        if len(cand_idx):
            for idx in cand_idx:
                edge = edge_geoms_proj[int(idx)]
                proj_len = _projected_length_on_edge(edge, x1, y1, x2, y2)
                overlap_ratio = min(1.0, proj_len / seg_len) if seg_len > 0 else 0.0
                edge_bearing = _closest_edge_segment_bearing(edge, mx, my)
                cdir = _cos_delta(seg_bearing, edge_bearing)
                try:
                    np1, np2 = _nearest_points(edge, mid_pt)
                    near_d = np1.distance(np2)
                except Exception:
                    near_d = 0.0

                score = (w_len * overlap_ratio) + (w_dir * cdir) - (w_dist * (near_d / max(buf_m, 1e-6)))
                if prev_way is not None and int(edge_ids[int(idx)]) == prev_way:
                    stick = min(0.4, 0.05 * max(1, prev_run_len))
                    score += stick

                if score > best_score:
                    best_score = score
                    best_idx = int(idx)
                    best_proj_len = float(proj_len)
                    best_cos = float(cdir)
                    best_near_d = float(near_d)

        used_fallback = False
        if best_idx == -1 or best_proj_len == 0.0:
            try:
                nearest_idx = tree.nearest(mid_pt)
                if isinstance(nearest_idx, (int, np.integer)):
                    nearest_idx = int(nearest_idx)
                else:
                    nearest_geom = nearest_idx
                    nearest_idx = int(np.where(edge_geoms_proj == nearest_geom)[0][0])
            except Exception:
                nearest_idx = None

            if nearest_idx is not None:
                edge = edge_geoms_proj[int(nearest_idx)]
                try:
                    np1, np2 = _nearest_points(edge, mid_pt)
                    near_d = np1.distance(np2)
                except Exception:
                    near_d = float("inf")

                if near_d <= nearest_cutoff_m or prev_way is None:
                    best_idx = int(nearest_idx)
                    best_proj_len = seg_len
                    edge_bearing = _closest_edge_segment_bearing(edge, mx, my)
                    best_cos = _cos_delta(seg_bearing, edge_bearing)
                    best_near_d = float(near_d)
                    used_fallback = True
                else:
                    if prev_way is not None and prev_way in id_to_idx:
                        best_idx = int(id_to_idx[prev_way])
                        edge = edge_geoms_proj[best_idx]
                        best_proj_len = seg_len
                        edge_bearing = _closest_edge_segment_bearing(edge, mx, my)
                        best_cos = _cos_delta(seg_bearing, edge_bearing)
                        best_near_d = float("inf")
                        used_fallback = True
                    else:
                        seg_way_ids.append(-1)
                        seg_len_list.append(seg_len)
                        cum_len += seg_len
                        vertex_cumlen.append(cum_len)
                        diag["overlap_ratio"].append(0.0)
                        diag["cos_delta"].append(0.0)
                        diag["nearest_dist"].append(float("inf"))
                        diag["used_fallback"].append(True)
                        prev_way = -1
                        prev_run_len = 1 if prev_way == -1 else 0
                        continue

        way_id = int(edge_ids[best_idx])
        if prev_way == way_id:
            prev_run_len += 1
        else:
            prev_run_len = 1
        prev_way = way_id

        assigned_way_ids.append(way_id)
        way_length_contrib[way_id] += best_proj_len
        total_len_m += best_proj_len

        diag["overlap_ratio"].append(min(1.0, best_proj_len / seg_len))
        diag["cos_delta"].append(best_cos)
        diag["nearest_dist"].append(best_near_d)
        diag["used_fallback"].append(used_fallback)

        seg_way_ids.append(way_id)
        seg_len_list.append(seg_len)
        cum_len += seg_len
        vertex_cumlen.append(cum_len)

    way_sequence = dedupe_consecutive(assigned_way_ids)
    return way_sequence, way_length_contrib, total_len_m, diag, seg_way_ids, seg_len_list, vertex_cumlen

def process_trip(
    poly: List[Tuple[float, float]],
    tree: STRtree,
    edge_geoms_proj: np.ndarray,
    edge_ids: np.ndarray,
    transformer: Transformer,
    id_to_idx: Dict[int, int],
    timestamps: Optional[List[int]] = None,
    radiuses: Optional[List[float]] = None,
    buf_m: float = BUF_M,
    nearest_cutoff_m: float = 35.0,
    w_len: float = 1.0,
    w_dir: float = 0.5,
    w_dist: float = 0.2,
) -> Tuple[
    List[int], float, float, float, Dict[int, float], Dict[str, List[float]],
    List[Tuple[float, float]], List[int], List[float], List[float], List[Dict]
]:
    coords, osrm_dist_m, osrm_dur_s, osrm_steps = osrm_full_geometry(
        poly,
        timestamps=timestamps,
        radiuses=radiuses,
    )
    ways, contrib, osm_len_m, diag, seg_way_ids, seg_len_list, vertex_cumlen = assign_segments(
        coords,
        tree,
        edge_geoms_proj,
        edge_ids,
        transformer,
        id_to_idx=id_to_idx,
        buf_m=buf_m,
        nearest_cutoff_m=nearest_cutoff_m,
        w_len=w_len,
        w_dir=w_dir,
        w_dist=w_dist,
    )
    return (ways, osm_len_m, osrm_dist_m, osrm_dur_s, contrib, diag,
            coords, seg_way_ids, seg_len_list, vertex_cumlen, osrm_steps)

def run_batch(
    df: pd.DataFrame,
    tree: STRtree,
    edge_geoms_proj: np.ndarray,
    edge_ids: np.ndarray,
    transformer: Transformer,
    id_to_idx: Dict[int, int],
    use_timestamps: bool = True,
    sampling_seconds: int = SAMPLING_SECONDS,
    max_workers: int = 4,
    buf_m: float = BUF_M,
    nearest_cutoff_m: float = 35.0,
    w_len: float = 1.0,
    w_dir: float = 0.5,
    w_dist: float = 0.2,
    start_ts_col: str = "TIMESTAMP",
    point_ts_col: str = "point_ts",
) -> pd.DataFrame:
    def _make_args(row) -> Tuple:
        poly = row.poly
        timestamps = None
        radiuses = None
        if use_timestamps:
            if hasattr(row, point_ts_col) and isinstance(getattr(row, point_ts_col), (list, tuple, np.ndarray)):
                pts = list(getattr(row, point_ts_col))
                if len(pts) == len(poly) and len(pts) >= 2:
                    timestamps = [int(t) for t in pts]
            if timestamps is None:
                start_ts = int(getattr(row, start_ts_col))
                n = len(poly)
                if hasattr(row, "trip_travel_time_s") and n >= 2:
                    step = float(row.trip_travel_time_s) / (n - 1)
                    timestamps = [int(round(start_ts + k * step)) for k in range(n)]
                else:
                    timestamps = [int(start_ts + k * sampling_seconds) for k in range(n)]
            radiuses = [RADIUS_M] * len(poly)
        return (
            poly,
            tree,
            edge_geoms_proj,
            edge_ids,
            transformer,
            id_to_idx,
            timestamps,
            radiuses,
            buf_m,
            nearest_cutoff_m,
            w_len,
            w_dir,
            w_dist,
        )

    def _run(args):
        r = process_trip(*args)
        return r, (getattr(_thread, "osrm_chunks", 0), getattr(_thread, "osrm_failed", 0), getattr(_thread, "osrm_err", ""))

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        pairs = list(ex.map(_run, map(_make_args, df.itertuples(index=False))))
    results = [p[0] for p in pairs]

    df_out = df.copy()
    df_out["way_id_sequence"]   = [r[0] for r in results]
    df_out["osm_partial_len_m"] = [r[1] for r in results]
    df_out["osrm_distance_m"]   = [r[2] for r in results]
    df_out["osrm_duration_s"]   = [r[3] for r in results]
    df_out["matched_coords"]    = [r[6] for r in results]
    df_out["seg_way_ids"]       = [r[7] for r in results]
    df_out["seg_len_m"]         = [r[8] for r in results]
    df_out["vertex_cumlen_m"]   = [r[9] for r in results]
    df_out["osrm_steps"]        = [r[10] for r in results]

    df_out["signed_err_m"] = df_out["osrm_distance_m"] - df_out["osm_partial_len_m"]
    df_out["osrm_chunks"] = [p[1][0] for p in pairs]
    df_out["osrm_failed_chunks"] = [p[1][1] for p in pairs]
    df_out["osrm_error"] = [p[1][2] for p in pairs]
    return df_out

def build_edge_neighbors(edges_proj: gpd.GeoDataFrame, edge_ids: np.ndarray):
    neighbors = defaultdict(set)

    if {"u", "v"}.issubset(edges_proj.columns):
        u = edges_proj["u"].astype("int64").to_numpy()
        v = edges_proj["v"].astype("int64").to_numpy()

        at_node = defaultdict(set)
        for eid, uu, vv in zip(edge_ids, u, v):
            e = int(eid)
            at_node[int(uu)].add(e)
            at_node[int(vv)].add(e)

        for node, eids in at_node.items():
            for e in eids:
                neighbors[e].update(eids)
        return neighbors

    def endpoints_of_geom(g):
        if g.is_empty:
            return None
        if isinstance(g, LineString):
            coords = list(g.coords)
            return Point(coords[0]), Point(coords[-1])
        if isinstance(g, MultiLineString):
            try:
                m = linemerge(g)
            except Exception:
                m = g
            if isinstance(m, LineString):
                coords = list(m.coords)
                return Point(coords[0]), Point(coords[-1])
            parts = [ls for ls in g.geoms if not ls.is_empty]
            if not parts:
                return None
            comp = max(parts, key=lambda ls: ls.length)
            coords = list(comp.coords)
            return Point(coords[0]), Point(coords[-1])
        b = g.boundary
        if hasattr(b, "geoms") and len(b.geoms) >= 2:
            return b.geoms[0], b.geoms[-1]
        return None

    def snap(pt: Point):
        return (round(pt.x, 1), round(pt.y, 1))

    at_node = defaultdict(set)
    for eid, g in zip(edge_ids, edges_proj.geometry):
        ep = endpoints_of_geom(g)
        if ep is None:
            continue
        a, b = ep
        na = snap(a); nb = snap(b)
        e = int(eid)
        at_node[na].add(e)
        at_node[nb].add(e)

    for node, eids in at_node.items():
        for e in eids:
            neighbors[e].update(eids)

    return neighbors

SMOOTH_USES_TIME = os.environ.get("MM_SMOOTH_TIME", "0") == "1"


def smooth_seg_ids(seg_ids, seg_len_m, seg_time_s, neighbors,
                   min_run_len_m=10.0, min_run_time_s=3.0):
    seg_ids = list(seg_ids)
    n = len(seg_ids)
    if n == 0:
        return seg_ids

    i = 1
    while i < n-1:
        if seg_ids[i-1] == seg_ids[i+1] != seg_ids[i]:
            if seg_len_m[i] < min_run_len_m or (SMOOTH_USES_TIME and seg_time_s[i] < min_run_time_s):
                seg_ids[i] = seg_ids[i-1]
        i += 1

    i = 0
    while i < n:
        j = i + 1
        while j < n and seg_ids[j] == seg_ids[i]:
            j += 1
        run_len = sum(seg_len_m[i:j])
        run_time = sum(seg_time_s[i:j])
        prev_id = seg_ids[i-1] if i > 0 else None
        next_id = seg_ids[j] if j < n else None
        eid = seg_ids[i]
        if run_len < min_run_len_m or (SMOOTH_USES_TIME and run_time < min_run_time_s):
            connected_prev = prev_id is not None and (eid in neighbors.get(prev_id, set()))
            connected_next = next_id is not None and (eid in neighbors.get(next_id, set()))
            if not connected_prev and not connected_next:
                if prev_id is not None:
                    for k in range(i, j): seg_ids[k] = prev_id
                elif next_id is not None:
                    for k in range(i, j): seg_ids[k] = next_id
        i = j

    return seg_ids

def _vertex_times(gps_measures, gps_times, verts_m):
    run_start = np.flatnonzero(np.r_[True, np.diff(gps_measures) > 1e-6])
    run_end = np.r_[run_start[1:] - 1, len(gps_measures) - 1]
    m_u = gps_measures[run_start]
    xp = np.empty(2 * len(m_u)); fp = np.empty(2 * len(m_u))
    xp[0::2], fp[0::2] = m_u - 1e-7, gps_times[run_start]
    xp[1::2], fp[1::2] = m_u, gps_times[run_end]
    t = np.interp(verts_m, xp, fp)
    t[verts_m <= m_u[0]] = gps_times[0]
    return t


def link_times_from_matched_smooth(
    poly: List[Tuple[float, float]],
    timestamps: List[int],
    matched_coords: List[Tuple[float, float]],
    seg_way_ids: List[int],
    seg_len_m: List[float],
    vertex_cumlen_m: List[float],
    transformer: Transformer,
    neighbors: Dict[int, set],
    min_run_len_m: float = 10.0,
    min_run_time_s: float = 3.0
) -> pd.DataFrame:
    if not matched_coords or len(matched_coords) < 2 or not seg_way_ids:
        return pd.DataFrame(columns=["way_id","enter_ts","exit_ts","link_travel_time_s","link_len_m"])

    mx, my = zip(*[transformer.transform(lon, lat) for lon, lat in matched_coords])
    matched_line = LineString(list(zip(mx, my)))

    gps_times = np.asarray(timestamps, dtype=float)
    gps_measures = [matched_line.project(Point(*transformer.transform(lon, lat))) for lon, lat in poly]
    gps_measures = np.maximum.accumulate(np.array(gps_measures, dtype=float))

    verts_m = np.array(vertex_cumlen_m, dtype=float)
    lo, hi = gps_measures[0], gps_measures[-1]
    verts_m = np.clip(verts_m, lo, hi)

    t_at_vert = _vertex_times(gps_measures, gps_times, verts_m)

    seg_time_s = np.maximum(0.0, np.diff(t_at_vert))

    seg_ids_sm = smooth_seg_ids(seg_way_ids, seg_len_m, seg_time_s, neighbors,
                                min_run_len_m=min_run_len_m, min_run_time_s=min_run_time_s)

    rows = []
    i = 0
    while i < len(seg_ids_sm):
        j = i + 1
        while j < len(seg_ids_sm) and seg_ids_sm[j] == seg_ids_sm[i]:
            j += 1
        way = int(seg_ids_sm[i])
        enter_ts = float(t_at_vert[i])
        exit_ts  = float(t_at_vert[j])
        link_time = float(np.sum(seg_time_s[i:j]))
        link_len  = float(np.sum(seg_len_m[i:j]))
        rows.append({"way_id": way, "enter_ts": enter_ts, "exit_ts": exit_ts,
                     "link_travel_time_s": link_time, "link_len_m": link_len,
                     "_covered_m": float(verts_m[j] - verts_m[i])})
        i = j

    out = pd.DataFrame(rows)

    gt = float(gps_times[-1] - gps_times[0])
    if out.empty:
        return out
    s = float(out["link_travel_time_s"].sum())
    if s <= 0.0:
        out.loc[:, "link_travel_time_s"] = 0.0
        out.loc[out.index[-1], "link_travel_time_s"] = gt
        out.loc[out.index[-1], "exit_ts"] = out.loc[out.index[0], "enter_ts"] + gt
        return out.drop(columns="_covered_m")

    if abs(s - gt) > 1e-6:
        scale = gt / s
        out["link_travel_time_s"] *= scale
        delta = gt - float(out["link_travel_time_s"].sum())
        out.loc[out.index[-1], "link_travel_time_s"] += delta
        out.loc[out.index[-1], "exit_ts"] = out.loc[out.index[-2], "exit_ts"] + out.loc[out.index[-1], "link_travel_time_s"] \
                                            if len(out) > 1 else out.loc[out.index[0], "enter_ts"] + out.loc[out.index[0], "link_travel_time_s"]

    if SMOOTH_USES_TIME:
        out = out[out["link_travel_time_s"] > 1e-6].reset_index(drop=True)
    else:
        out = out[out["_covered_m"] > 1e-3].reset_index(drop=True)
    out = out.drop(columns="_covered_m")
    if not out.empty:
        cum = np.cumsum(out["link_travel_time_s"].to_numpy(dtype=float))
        out["exit_ts"] = float(gps_times[0]) + cum
        out["enter_ts"] = out["exit_ts"] - out["link_travel_time_s"]
    return out

def compute_gt_vertex_times(
    poly: List[Tuple[float, float]],
    timestamps: List[int],
    matched_coords: List[Tuple[float, float]],
    transformer: Transformer,
    vertex_cumlen_m: List[float]
) -> Tuple[np.ndarray, LineString]:
    if not matched_coords or len(matched_coords) < 2:
        return np.array([], dtype=float), LineString([])
    mx, my = zip(*[transformer.transform(lon, lat) for lon, lat in matched_coords])
    matched_line = LineString(list(zip(mx, my)))

    gps_times = np.asarray(timestamps, dtype=float)
    gps_measures = [matched_line.project(Point(*transformer.transform(lon, lat))) for lon, lat in poly]
    gps_measures = np.maximum.accumulate(np.array(gps_measures, dtype=float))

    verts_m = np.array(vertex_cumlen_m, dtype=float)
    lo, hi = gps_measures[0], gps_measures[-1]
    verts_m = np.clip(verts_m, lo, hi)

    t_at_vert = _vertex_times(gps_measures, gps_times, verts_m)
    return t_at_vert, matched_line

def explode_link_times(df_res: pd.DataFrame,
                       transformer: Transformer,
                       neighbors: Dict[int, set],
                       point_ts_col: str = "point_ts",
                       trip_id_col: str = "TRIP_ID",
                       min_run_len_m: float = 10.0,
                       min_run_time_s: float = 3.0) -> pd.DataFrame:
    rows = []
    for r in df_res.itertuples(index=False):
        poly = list(getattr(r, "poly"))
        pts = list(getattr(r, point_ts_col)) if hasattr(r, point_ts_col) else None
        if pts is None or len(pts) != len(poly) or len(pts) < 2:
            start_ts = int(getattr(r, "TIMESTAMP"))
            n = len(poly)
            if hasattr(r, "trip_travel_time_s") and n >= 2:
                step = float(getattr(r, "trip_travel_time_s")) / (n - 1)
                pts = [int(round(start_ts + k * step)) for k in range(n)]
            else:
                pts = [int(start_ts + k * SAMPLING_SECONDS) for k in range(n)]

        lt = link_times_from_matched_smooth(
            poly=poly,
            timestamps=list(pts),
            matched_coords=list(getattr(r, "matched_coords")),
            seg_way_ids=list(getattr(r, "seg_way_ids")),
            seg_len_m=list(getattr(r, "seg_len_m")),
            vertex_cumlen_m=list(getattr(r, "vertex_cumlen_m")),
            transformer=transformer,
            neighbors=neighbors,
            min_run_len_m=min_run_len_m,
            min_run_time_s=min_run_time_s
        )
        if not lt.empty and hasattr(r, trip_id_col):
            lt[trip_id_col] = getattr(r, trip_id_col)
        if not lt.empty and hasattr(r, "TIMESTAMP"):
            lt["departure_ts"] = int(getattr(r, "TIMESTAMP"))
            dep = pd.to_datetime(int(getattr(r, "TIMESTAMP")), unit='s')
            if LOCAL_TZ:
                dep = dep.tz_localize("UTC").tz_convert(LOCAL_TZ).tz_localize(None)
            lt["departure_time"] = dep
        rows.append(lt)

    per_link_times = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=[trip_id_col, "way_id", "enter_ts", "exit_ts", "link_travel_time_s", "link_len_m"]
    )
    return per_link_times

BIN_SECONDS = 600
SLOTS = int(os.environ.get("MM_SLOTS", 18))
MIN_DT_S = 0.1
MIN_DD_M = 0.1
MAX_VALID_SPEED_MS = 80

def step_speeds_from_gt_timeline_robust(
    steps: List[Dict],
    matched_line: LineString,
    vertex_cumlen: Sequence[float],
    t_at_vert: np.ndarray,
    seg_way_ids: Sequence[int],
    transformer: Transformer,
    trip_id: int,
    gps_times: Sequence[float],
    min_dt_s: float = MIN_DT_S,
    min_dd_m: float = MIN_DD_M,
    max_speed_ms: float = MAX_VALID_SPEED_MS,
) -> pd.DataFrame:
    vcm = np.asarray(vertex_cumlen, dtype=float)
    gps_times = np.asarray(gps_times, dtype=float)
    rows = []
    for st in steps:
        coords = st.get("coords") or []
        if len(coords) < 2:
            continue

        x0, y0 = transformer.transform(coords[0][0], coords[0][1])
        x1, y1 = transformer.transform(coords[-1][0], coords[-1][1])
        s0 = matched_line.project(Point(x0, y0))
        s1 = matched_line.project(Point(x1, y1))
        if not np.isfinite(s0) or not np.isfinite(s1) or s1 <= s0:
            continue

        i0 = int(np.searchsorted(vcm, s0, side="left"))
        i1 = int(np.searchsorted(vcm, s1, side="right"))
        i0 = max(0, min(i0, len(vcm) - 1))
        i1 = max(i0 + 1, min(i1, len(vcm)))

        t0 = float(t_at_vert[i0])
        t1 = float(t_at_vert[i1 - 1])
        dt = t1 - t0
        dd = float(s1 - s0)

        if dt < min_dt_s or dd < min_dd_m:
            continue
        speed = dd / dt
        if not (0.0 < speed <= max_speed_ms):
            continue
        avail_ts = float(gps_times[min(int(np.searchsorted(gps_times, t1, side="left")), len(gps_times) - 1)])

        span = {}
        for k in range(i0, i1):
            if k < len(seg_way_ids) and int(seg_way_ids[k]) >= 0:
                a = min(max(float(t_at_vert[k]), t0), t1)
                b = min(max(float(t_at_vert[min(k + 1, len(t_at_vert) - 1)]), t0), t1)
                lo, hi = span.get(int(seg_way_ids[k]), (a, b))
                span[int(seg_way_ids[k])] = (min(lo, a), max(hi, b))
        for way, (a, b) in span.items():
            rows.append((way, pd.to_datetime((a + b) * 0.5, unit="s"), speed, int(trip_id), avail_ts))

    return pd.DataFrame(rows, columns=["way_id", "timestamp", "speed_m_s", "trip_id", "avail_ts"])

def collect_raw_speeds_from_df(df_res: pd.DataFrame,
                               transformer: Transformer,
                               point_ts_col: str = "point_ts",
                               trip_id_col: str = "TRIP_ID") -> pd.DataFrame:
    parts = []
    for r in df_res.itertuples(index=False):
        poly = list(getattr(r, "poly"))
        pts = list(getattr(r, point_ts_col)) if hasattr(r, point_ts_col) else None
        if pts is None or len(pts) != len(poly) or len(pts) < 2:
            start_ts = int(getattr(r, "TIMESTAMP"))
            n = len(poly)
            if hasattr(r, "trip_travel_time_s") and n >= 2:
                step = float(getattr(r, "trip_travel_time_s")) / (n - 1)
                pts = [int(round(start_ts + k * step)) for k in range(n)]
            else:
                pts = [int(start_ts + k * SAMPLING_SECONDS) for k in range(n)]

        t_at_vert, matched_line = compute_gt_vertex_times(
            poly=poly,
            timestamps=pts,
            matched_coords=list(getattr(r, "matched_coords")),
            transformer=transformer,
            vertex_cumlen_m=list(getattr(r, "vertex_cumlen_m")),
        )
        step_df = step_speeds_from_gt_timeline_robust(
            steps=list(getattr(r, "osrm_steps")),
            matched_line=matched_line,
            vertex_cumlen=list(getattr(r, "vertex_cumlen_m")),
            t_at_vert=t_at_vert,
            seg_way_ids=list(getattr(r, "seg_way_ids")),
            transformer=transformer,
            trip_id=int(getattr(r, trip_id_col)) if hasattr(r, trip_id_col) else -1,
            gps_times=pts,
        )
        parts.append(step_df)

    if parts:
        raw = pd.concat(parts, ignore_index=True)
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=False)
    else:
        raw = pd.DataFrame(columns=["way_id", "timestamp", "speed_m_s", "trip_id", "avail_ts"])
    return raw

def build_dynamic_bins(raw_speeds: pd.DataFrame,
                       bin_seconds: int = BIN_SECONDS) -> pd.DataFrame:
    if raw_speeds.empty:
        return pd.DataFrame(columns=["way_id", "bin",
                                     "median_speed", "mean_speed",
                                     "min_speed", "max_speed", "count_vehicle"])
    df = raw_speeds.copy()
    df["bin"] = (df["timestamp"].astype("int64") // 10**9 // bin_seconds).astype("int64")
    agg = (df.groupby(["way_id", "bin"])
             .agg(median_speed=("speed_m_s", "median"),
                  mean_speed=("speed_m_s", "mean"),
                  min_speed=("speed_m_s", "min"),
                  max_speed=("speed_m_s", "max"),
                  count_vehicle=("trip_id", pd.Series.nunique))
             .reset_index())
    return agg

def attach_dynamic_with_enter_bin_excluding_self(per_link, dyn_bins, raw,
                                                 trip_id_col="TRIP_ID",
                                                 bin_seconds=BIN_SECONDS,
                                                 slots=SLOTS):
    if per_link.empty:
        return per_link.copy()

    out = per_link.copy()
    out["trip_id"] = out[trip_id_col].astype(int)
    ent = out["enter_ts"].astype(float).to_numpy()
    own = out["trip_id"].to_numpy()
    r = raw.assign(_t=raw["timestamp"].astype("int64").to_numpy() / 1e9).sort_values(["way_id", "_t"], kind="mergesort")
    T, V, TR = r["_t"].to_numpy(), r["speed_m_s"].to_numpy(dtype=float), r["trip_id"].to_numpy()
    avail_ts = r["avail_ts"].to_numpy(dtype=float)
    by_way = r.groupby("way_id").indices
    names = ["median_speed", "mean_speed", "min_speed", "max_speed", "count_vehicle"]
    res = np.zeros((len(out), slots, len(names)))
    ks = np.arange(slots)
    for way, rows_ix in out.groupby("way_id").indices.items():
        pos = by_way.get(way)
        if pos is None:
            continue
        t_w, v_w, tr_w = T[pos], V[pos], TR[pos]
        e = ent[rows_ix][:, None]
        hi = e - ks[None, :] * bin_seconds
        lo = hi - bin_seconds
        assert (hi <= e).all(), "a slot would end after the link entry"
        left, right = np.searchsorted(t_w, lo, side="left"), np.searchsorted(t_w, hi, side="left")
        for a, k in zip(*np.nonzero(right > left)):
            sel = slice(left[a, k], right[a, k])
            keep = (tr_w[sel] != own[rows_ix[a]]) & (avail_ts[pos[sel]] < ent[rows_ix[a]])
            if keep.any():
                vv = v_w[sel][keep]
                res[rows_ix[a], k] = (np.median(vv), vv.mean(), vv.min(), vv.max(), len(np.unique(tr_w[sel][keep])))
    dyn_col_dict = {f"{s}_{k}": res[:, k, j] for k in range(slots) for j, s in enumerate(names)}
    return pd.concat([out, pd.DataFrame(dyn_col_dict, index=out.index)], axis=1, copy=False)

    out = per_link.copy()
    out["_enter_bin"] = (out["enter_ts"].astype(float) // bin_seconds).astype("int64")
    out["trip_id"] = out[trip_id_col].astype(int)

    all_bins  = dyn_bins.copy()
    trip_bins = (
        raw.assign(bin=(raw["timestamp"].astype("int64") // 10**9 // bin_seconds).astype("int64"))
        .groupby(["way_id", "bin", "trip_id"])
        .agg(median_speed=("speed_m_s", "median"),
             mean_speed  =("speed_m_s", "mean"),
             min_speed   =("speed_m_s", "min"),
             max_speed   =("speed_m_s", "max"),
             count_vehicle=("trip_id", "nunique"))
        .reset_index()
    )
    raw_b = raw.assign(bin=(raw["timestamp"].astype("int64") // 10**9 // bin_seconds).astype("int64"))[
        ["way_id", "bin", "trip_id", "speed_m_s"]]

    dyn_col_dict = {}

    for k in range(slots):
        need = out[["way_id", "_enter_bin", "trip_id"]].copy()
        need["bin"] = need["_enter_bin"] - k

        tmp_all  = need.merge(all_bins,  how="left", on=["way_id", "bin"])
        tmp_self = need.merge(trip_bins, how="left", on=["way_id", "bin", "trip_id"])

        has_self = tmp_self["count_vehicle"].notna().to_numpy()
        oth = pd.DataFrame(0.0, index=range(len(need)),
                           columns=["median_speed", "mean_speed", "min_speed", "max_speed", "count_vehicle"])
        if has_self.any():
            m = (need.reset_index(drop=True).loc[has_self, ["way_id", "bin", "trip_id"]].reset_index()
                 .merge(raw_b, on=["way_id", "bin"], suffixes=("", "_r")))
            m = m[m["trip_id_r"] != m["trip_id"]]
            g = m.groupby("index")
            agg = pd.DataFrame({"median_speed": g["speed_m_s"].median(), "mean_speed": g["speed_m_s"].mean(),
                                "min_speed": g["speed_m_s"].min(), "max_speed": g["speed_m_s"].max(),
                                "count_vehicle": g["trip_id_r"].nunique()})
            oth.loc[agg.index, agg.columns] = agg.to_numpy()

        for stat in ["median_speed", "mean_speed", "min_speed", "max_speed"]:
            v_all  = tmp_all[stat].astype(float).fillna(0.0)
            v_self = tmp_self[stat].astype(float).fillna(0.0)
            vals = pd.Series(np.where(has_self, oth[stat].to_numpy(), v_all.to_numpy()))
            dyn_col_dict[f"{stat}_{k}"] = vals.to_numpy()

        c_all  = tmp_all["count_vehicle"].astype(float).fillna(0.0)
        c_self = tmp_self["count_vehicle"].astype(float).fillna(0.0)
        counts = pd.Series(np.where(has_self, oth["count_vehicle"].to_numpy(), c_all.to_numpy()))
        dyn_col_dict[f"count_vehicle_{k}"] = counts.to_numpy()

    out = pd.concat([out.drop(columns=["_enter_bin"]), pd.DataFrame(dyn_col_dict, index=out.index)],
                    axis=1, copy=False)
    return out

def attach_dynamic_before_departure(per_link, dyn_bins, bin_seconds=BIN_SECONDS, slots=SLOTS):
    if per_link.empty:
        return per_link.copy()
    out = per_link.copy()
    dep_bin = (out["departure_ts"].astype("int64") // bin_seconds).astype("int64")
    bins = dyn_bins.set_index(["way_id", "bin"])
    cols = {}
    for k in range(slots):
        key = pd.MultiIndex.from_arrays([out["way_id"].astype("int64").to_numpy(), (dep_bin - 1 - k).to_numpy()],
                                        names=["way_id", "bin"])
        got = bins.reindex(key)
        for stat in ["median_speed", "mean_speed", "min_speed", "max_speed", "count_vehicle"]:
            cols[f"{stat}_{k}"] = got[stat].astype(float).fillna(0.0).to_numpy()
    assert int(((dep_bin - 1) * bin_seconds + bin_seconds <= out["departure_ts"].astype("int64")).sum()) == len(out), \
        "a dynamic slot would include data after departure"
    return pd.concat([out, pd.DataFrame(cols, index=out.index)], axis=1, copy=False)


class SpeedHistory:
    def __init__(self, raw, bin_seconds=BIN_SECONDS, slots=SLOTS):
        assert "avail_ts" in raw.columns, "speed records without avail_ts (written before 2026-10-14): rebuild them"
        r = raw.assign(_t=raw["timestamp"].astype("datetime64[ns]").astype("int64").to_numpy() / 1e9).sort_values(
            ["way_id", "_t"], kind="mergesort")
        self.T, self.V, self.TR = r["_t"].to_numpy(), r["speed_m_s"].to_numpy(dtype=float), r["trip_id"].to_numpy()
        self.avail_ts = r["avail_ts"].to_numpy(dtype=float)
        self.by_way = {int(w): ix for w, ix in r.groupby("way_id").indices.items()}
        self.bin_seconds, self.n_slots = bin_seconds, slots
        self.fingerprint = str(pd.util.hash_pandas_object(r[["way_id", "_t", "speed_m_s", "trip_id", "avail_ts"]],
                                                          index=False).sum())
        self._memo = {}

    def slots(self, way_id, end_ts, trip_id):
        key = (int(way_id), float(end_ts), int(trip_id))
        out = self._memo.get(key)
        if out is not None:
            return out
        out = np.zeros((self.n_slots, 5))
        pos = self.by_way.get(int(way_id))
        if pos is not None:
            t_w, v_w, tr_w = self.T[pos], self.V[pos], self.TR[pos]
            hi = float(end_ts) - np.arange(self.n_slots) * self.bin_seconds
            left = np.searchsorted(t_w, hi - self.bin_seconds, side="left")
            right = np.searchsorted(t_w, hi, side="left")
            for k in np.nonzero(right > left)[0]:
                sel = slice(left[k], right[k])
                keep = (tr_w[sel] != int(trip_id)) & (self.avail_ts[pos[sel]] < float(end_ts))
                if keep.any():
                    vv = v_w[sel][keep]
                    out[k] = (np.median(vv), vv.mean(), vv.min(), vv.max(), len(np.unique(tr_w[sel][keep])))
        self._memo[key] = out
        return out


def make_final_per_link_features(
    df_res: pd.DataFrame,
    transformer: Transformer,
    neighbors: Dict[int, set],
    edges_static: Optional[pd.DataFrame] = None,
    point_ts_col: str = "point_ts",
    trip_id_col: str = "TRIP_ID",
    min_run_len_m: float = 10.0,
    min_run_time_s: float = 3.0,
    skip_dynamic_features: bool = False,
) -> pd.DataFrame:
    per_link = explode_link_times(
        df_res,
        transformer=transformer,
        neighbors=neighbors,
        point_ts_col=point_ts_col,
        trip_id_col=trip_id_col,
        min_run_len_m=min_run_len_m,
        min_run_time_s=min_run_time_s,
    )

    if skip_dynamic_features:
        final_df = per_link
    else:
        raw = collect_raw_speeds_from_df(
            df_res,
            transformer=transformer,
            point_ts_col=point_ts_col,
            trip_id_col=trip_id_col,
        )
        dyn_bins = build_dynamic_bins(raw)

        final_df = attach_dynamic_with_enter_bin_excluding_self(per_link, dyn_bins, raw,
                                                                trip_id_col=trip_id_col,
                                                                bin_seconds=BIN_SECONDS,
                                                                slots=SLOTS)

    if edges_static is not None and "way_id" in final_df.columns:
        cols = [c for c in edges_static.columns if c != "id"]
        final_df = final_df.merge(
            edges_static.rename(columns={"id": "way_id"}),
            how="left", on="way_id"
        )

    dyn_cols = [c for c in final_df.columns if any(c.startswith(s) for s in
                ["median_speed_", "mean_speed_", "min_speed_", "max_speed_", "count_vehicle_"])]
    final_df[dyn_cols] = final_df[dyn_cols].fillna(0.0)

    return final_df

def parse_lanes(s) -> int:
    if pd.isna(s) or not isinstance(s, str):
        return 1
    parts = [int(p) for p in s.split(';') if p.isdigit()]
    return max(parts) if parts else 1

def parse_speed(s) -> float:
    if pd.isna(s) or not isinstance(s, str):
        return np.nan
    tok = s.split()
    try:
        v = float(tok[0])
    except:
        return np.nan
    return v * (0.44704 if 'mph' in s else 1000/3600)

def parse_width(s) -> float:
    if pd.isna(s) or not isinstance(s, str):
        return np.nan
    vals = []
    for p in s.split(';'):
        try:
            vals.append(float(p))
        except:
            pass
    return np.mean(vals) if vals else np.nan


def build_static_features(edges_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    edges_sub = edges_gdf.copy()

    if 'lanes' in edges_sub.columns:
        edges_sub['lanes'] = edges_sub['lanes'].apply(parse_lanes)
    else:
        edges_sub['lanes'] = 1

    if 'maxspeed' in edges_sub.columns:
        edges_sub['speed_limit_ms'] = edges_sub['maxspeed'].apply(parse_speed)
        edges_sub['speed_limit_ms'].fillna(edges_sub['speed_limit_ms'].median(), inplace=True)
    else:
        edges_sub['speed_limit_ms'] = 13.9

    if 'width' in edges_sub.columns:
        edges_sub['width_m'] = edges_sub['width'].apply(parse_width)
        edges_sub['width_m'].fillna(edges_sub['width_m'].median(), inplace=True)
    else:
        edges_sub['width_m'] = 3.5

    if 'highway' in edges_sub.columns:
        hw = pd.get_dummies(edges_sub['highway'].fillna('unknown'), prefix='hw', dtype=int)
        edges_sub = pd.concat([edges_sub, hw], axis=1)

    edges_sub['has_traffic_signal'] = 0

    static_cols = ['id', 'length', 'lanes', 'width_m', 'speed_limit_ms', 'has_traffic_signal']
    hw_cols = [c for c in edges_sub.columns if c.startswith('hw_')]
    static_cols.extend(hw_cols)

    static_cols = [c for c in static_cols if c in edges_sub.columns]

    static_df = edges_sub[static_cols].copy()
    static_df = static_df.rename(columns={'id': 'way_id'})

    return static_df

class MapMatchingPipeline:
    def __init__(
        self,
        edges_gdf: gpd.GeoDataFrame,
        crs_metric: str = CRS_METRIC,
        osrm_url: str = OSRM_URL,
    ):
        self.edges_gdf = edges_gdf
        self.crs_metric = crs_metric
        self.osrm_url = osrm_url

        (self.edges_proj, self.edge_geoms_proj, self.edge_ids,
         self.edge_lengths_m, self.tree, self.transformer,
         self.inv_transformer, self.id_to_idx) = build_spatial_index(edges_gdf, crs_metric)

        self.neighbors = build_edge_neighbors(self.edges_proj, self.edge_ids)

        self.static_features = build_static_features(edges_gdf)

        print(f"MapMatchingPipeline initialized with {len(edges_gdf)} edges")

    def process_trips(
        self,
        df: pd.DataFrame,
        max_workers: int = 4,
        sampling_seconds: int = SAMPLING_SECONDS,
    ) -> pd.DataFrame:
        if 'poly' not in df.columns and 'POLYLINE' in df.columns:
            df = df.copy()
            df['poly'] = df['POLYLINE'].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )

        return run_batch(
            df,
            tree=self.tree,
            edge_geoms_proj=self.edge_geoms_proj,
            edge_ids=self.edge_ids,
            transformer=self.transformer,
            id_to_idx=self.id_to_idx,
            use_timestamps=True,
            sampling_seconds=sampling_seconds,
            max_workers=max_workers,
        )

    def build_features(
        self,
        df_res: pd.DataFrame,
        skip_dynamic_features: bool = False,
    ) -> pd.DataFrame:
        return make_final_per_link_features(
            df_res=df_res,
            transformer=self.transformer,
            neighbors=self.neighbors,
            edges_static=self.static_features,
            skip_dynamic_features=skip_dynamic_features,
        )

    def run_full_pipeline(
        self,
        df: pd.DataFrame,
        max_workers: int = 4,
        sampling_seconds: int = SAMPLING_SECONDS,
        skip_dynamic_features: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        print("Step 1/2: Map matching...")
        df_res = self.process_trips(df, max_workers, sampling_seconds)

        print("Step 2/2: Building features...")
        final_features = self.build_features(df_res, skip_dynamic_features=skip_dynamic_features)

        print(f"Done! {len(df_res)} trips → {len(final_features)} link records")
        return df_res, final_features

