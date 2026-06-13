"""
Structural Clustering — Final Version (v3)
==========================================
p (physical) clusters  (4 numeric features: length, lanes, width_m, speed_limit_ms)
t (topology) clusters  (11 features: 7 relation degrees + betweenness + pagerank + clustering_coeff + closeness)

The number of p and t clusters is chosen by the ELBOW method over a sweep
(silhouette score reported + plotted alongside); selection graphs are saved to
FIG_DIR in the LinkETA figure palette. Pin a value via P_OVERRIDE / T_OVERRIDE.
A link therefore belongs to a (p, t) cluster slot.

Outputs per city (saved to data/{city}/):
  - cluster_assignments.pt   : {p_labels, t_labels, P, T, ... + legacy k_labels/s_labels/K/S aliases}
  # OLD: static_feats_27.pt       : [N, 27] tensor of extended static features from geojson
  - static_feats_31.pt       : [N, 31] tensor = 27 geojson + 4 topology (betweenness, pagerank, clustering_coeff, closeness)
  - city_descriptor.pt       : city-level graph descriptor for city-similarity weighting

No original precomputed files are modified.
"""

import os, json, time
import numpy as np
import pandas as pd
import geopandas as gpd
import torch
import networkx as nx
from scipy.stats import entropy as _scipy_entropy
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# === Configuration ====================================================
CITIES = {
    "porto":   {"geojson": "meta_learning_data/porto/edges_porto.geojson",   "relations": "meta_learning_data/porto/relations.pt",   "out": "meta_learning_data/porto",   "crs_metric": "EPSG:32629"},
    "sf":      {"geojson": "meta_learning_data/sf/edges_sf.geojson",         "relations": "meta_learning_data/sf/relations.pt",       "out": "meta_learning_data/sf",      "crs_metric": "EPSG:26910"},
    "chengdu": {"geojson": "meta_learning_data/chengdu/edges_chengdu.geojson","relations": "meta_learning_data/chengdu/relations.pt", "out": "meta_learning_data/chengdu", "crs_metric": "EPSG:32648"},
    "daegu":   {"geojson": "meta_learning_data/daegu/edges_daegu.geojson",     "relations": "meta_learning_data/daegu/relations.pt",   "out": "meta_learning_data/daegu",   "crs_metric": "EPSG:32652"},
    "shanghai":{"geojson": "meta_learning_data/shanghai/edges_shanghai.geojson","relations": "meta_learning_data/shanghai/relations.pt", "out": "meta_learning_data/shanghai", "crs_metric": "EPSG:32651"},
}

# === Cluster-count selection (v3) =====================================
# OLD: K_PHYSICAL = 7 ; S_TOPO = 9   (hardcoded, no justification / no plots)
# NEW: p (physical) and t (topology) counts chosen by the ELBOW method over a
#      sweep; silhouette reported alongside; selection graphs saved to FIG_DIR.
#      Set P_OVERRIDE / T_OVERRIDE to an int to pin a value (bypass selection).
P_RANGE = range(2, 16)        # physical-cluster sweep (inclusive of 2..15)
T_RANGE = range(2, 16)        # topology-cluster sweep
P_OVERRIDE = None             # int -> bypass elbow selection for p
T_OVERRIDE = None             # int -> bypass elbow selection for t
CLUSTER_SELECT_BY = "elbow"   # "elbow" (default) or "silhouette"
FIG_DIR = "checkpoints/cluster_selection"          # selection figures + stats (under checkpoints/)

RELATION_NAMES = [
    'self', 'downstream', 'upstream', 'left_turn',
    'opposite', 'right_turn', 'high_order', 'u_turn',
]

# 15 highway types for one-hot encoding
HIGHWAY_ONEHOT = [
    'living_street', 'motorway', 'motorway_link', 'pedestrian',
    'primary', 'primary_link', 'residential', 'secondary',
    'secondary_link', 'service', 'tertiary', 'tertiary_link',
    'trunk', 'trunk_link', 'unclassified',
]

# The 27 feature names (for reference / schema)
FEATURE_NAMES_27 = (
    ['length', 'lanes', 'width_m', 'speed_limit_ms']              # 0-3   numeric
    + [f'hw_{t}' for t in HIGHWAY_ONEHOT]                         # 4-18  highway one-hot (15)
    + ['has_traffic_signal', 'link_len_m']                         # 19-20
    + ['oneway', 'bridge', 'tunnel', 'junction',                   # 21-26 new binary
       'has_name', 'has_service_subtype']
)

# v2: The 31 feature names = 27 geojson + 4 per-link topology centralities
# These 4 come from the topology computation (betweenness, pagerank, clustering_coeff, closeness)
# and are appended AFTER structural clustering generates them.
FEATURE_NAMES_31 = (
    FEATURE_NAMES_27                                                # 0-26  original geojson
    + ['betweenness', 'pagerank', 'clustering_coeff', 'closeness']  # 27-30 per-link topology
)

# K-clustering uses only features 0-3 (4 numeric)
K_FEAT_INDICES = [0, 1, 2, 3]

# S-clustering topology feature names
# OLD: 10 features (no closeness)
# TOPO_NAMES = [
#     'deg_downstream', 'deg_upstream', 'deg_left_turn',
#     'deg_opposite', 'deg_right_turn', 'deg_high_order', 'deg_u_turn',
#     'betweenness', 'pagerank', 'clustering_coeff',
# ]
# NEW (v2): 11 features — added closeness centrality
TOPO_NAMES = [
    'deg_downstream', 'deg_upstream', 'deg_left_turn',
    'deg_opposite', 'deg_right_turn', 'deg_high_order', 'deg_u_turn',
    'betweenness', 'pagerank', 'clustering_coeff', 'closeness',
]


# === Parsing helpers ==================================================

def parse_width(val):
    if pd.isna(val) or val in ('', 'None'):
        return np.nan
    s = str(val).strip().lower()
    if "'" in s:
        parts = s.replace('"', '').split("'")
        try:
            feet = float(parts[0])
            inches = float(parts[1]) if len(parts) > 1 and parts[1] else 0
            return feet * 0.3048 + inches * 0.0254
        except Exception:
            return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan


def parse_maxspeed(val):
    if pd.isna(val) or val in ('', 'None'):
        return np.nan
    s = str(val).strip().lower()
    if 'mph' in s:
        try:
            return float(s.replace('mph', '').strip()) * 0.44704
        except Exception:
            return np.nan
    try:
        return float(s) / 3.6
    except Exception:
        return np.nan


def parse_lanes(val):
    if pd.isna(val) or val in ('', 'None'):
        return np.nan
    try:
        return float(str(val).strip())
    except Exception:
        return np.nan


# === Extract 27 features from geojson ================================

def extract_27_features(gdf: gpd.GeoDataFrame) -> np.ndarray:
    """
    [N, 27] float32 feature matrix from geojson.
    Index mapping matches FEATURE_NAMES_27.
    """
    N = len(gdf)
    feats = np.zeros((N, 27), dtype=np.float32)

    # 0: length
    feats[:, 0] = pd.to_numeric(gdf['length'], errors='coerce').fillna(0).values
    # 1: lanes
    feats[:, 1] = gdf['lanes'].apply(parse_lanes).fillna(0).values
    # 2: width_m
    if 'width' in gdf.columns:
        feats[:, 2] = gdf['width'].apply(parse_width).fillna(0).values
    elif 'width_m' in gdf.columns:
        feats[:, 2] = pd.to_numeric(gdf['width_m'], errors='coerce').fillna(0).values
    # 3: speed_limit_ms
    if 'maxspeed' in gdf.columns:
        feats[:, 3] = gdf['maxspeed'].apply(parse_maxspeed).fillna(0).values
    # 4-18: highway one-hot (15 types)
    hw_col = gdf['highway'].fillna('').astype(str).str.lower().str.strip()
    for i, hw_type in enumerate(HIGHWAY_ONEHOT):
        feats[:, 4 + i] = (hw_col == hw_type).astype(np.float32)
    # 19: has_traffic_signal
    if 'has_traffic_signal' in gdf.columns:
        feats[:, 19] = pd.to_numeric(gdf['has_traffic_signal'], errors='coerce').fillna(0).values
    # 20: link_len_m (= length)
    feats[:, 20] = feats[:, 0]
    # 21: oneway
    if 'oneway' in gdf.columns:
        feats[:, 21] = gdf['oneway'].fillna('').astype(str).str.lower().isin(
            ['yes', 'true', '1', '-1']).astype(np.float32)
    # 22: bridge
    if 'bridge' in gdf.columns:
        feats[:, 22] = gdf['bridge'].fillna('').astype(str).str.lower().apply(
            lambda x: 1.0 if x not in ('', 'no', 'none', 'nan') else 0.0)
    # 23: tunnel
    if 'tunnel' in gdf.columns:
        feats[:, 23] = gdf['tunnel'].fillna('').astype(str).str.lower().apply(
            lambda x: 1.0 if x not in ('', 'no', 'none', 'nan') else 0.0)
    # 24: junction
    if 'junction' in gdf.columns:
        feats[:, 24] = gdf['junction'].fillna('').astype(str).str.lower().isin(
            ['roundabout', 'circular', 'mini_roundabout']).astype(np.float32)
    # 25: has_name
    if 'name' in gdf.columns:
        feats[:, 25] = gdf['name'].fillna('').astype(str).apply(
            lambda x: 1.0 if x.strip() not in ('', 'none', 'None', 'nan') else 0.0)
    # 26: has_service_subtype
    if 'service' in gdf.columns:
        feats[:, 26] = gdf['service'].fillna('').astype(str).apply(
            lambda x: 1.0 if x.strip() not in ('', 'none', 'None', 'nan') else 0.0)

    return feats


# === Topology metrics =================================================

def load_relations(rel_path: str):
    rel_raw = torch.load(rel_path, map_location="cpu", weights_only=False)
    if isinstance(rel_raw, dict) and 'rel_edges' in rel_raw:
        names = rel_raw.get('names', RELATION_NAMES)
        return {name: rel_raw['rel_edges'][name].numpy() for name in names}
    elif isinstance(rel_raw, dict):
        return {name: rel_raw[name].numpy() for name in RELATION_NAMES}
    else:
        return {RELATION_NAMES[i]: rel_raw[i].numpy() for i in range(len(rel_raw))}


def compute_topology_metrics(n_nodes: int, relations: dict) -> np.ndarray:
    """11 topology metrics: 7 relation degrees (no self) + betweenness + pagerank + clustering_coeff + closeness."""
    print(f"    Computing topology for {n_nodes} nodes...")

    # 7 relation degree features (skip 'self')
    t0 = time.time()
    rel_deg = np.zeros((n_nodes, 7), dtype=np.float32)
    for i, rn in enumerate(RELATION_NAMES[1:]):  # skip 'self'
        if rn not in relations:
            continue
        coo = relations[rn]
        if coo.shape[1] == 0:
            continue
        src = np.bincount(coo[0], minlength=n_nodes)[:n_nodes]
        dst = np.bincount(coo[1], minlength=n_nodes)[:n_nodes]
        rel_deg[:, i] = src + dst
    print(f"      Degree profile (7): {time.time()-t0:.1f}s")

    # Build undirected graph (no self-loops)
    t0 = time.time()
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    for rn, coo in relations.items():
        if rn == 'self' or coo.shape[1] == 0:
            continue
        G.add_edges_from(zip(coo[0].tolist(), coo[1].tolist()))
    print(f"      Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges ({time.time()-t0:.1f}s)")

    # Betweenness (approx)
    t0 = time.time()
    k_sample = min(500, n_nodes)
    bc = nx.betweenness_centrality(G, k=k_sample, seed=42)
    bc_arr = np.array([bc.get(i, 0.0) for i in range(n_nodes)], dtype=np.float32)
    print(f"      Betweenness (k={k_sample}): {time.time()-t0:.1f}s")

    # PageRank
    t0 = time.time()
    pr = nx.pagerank(G, alpha=0.85, max_iter=100)
    pr_arr = np.array([pr.get(i, 0.0) for i in range(n_nodes)], dtype=np.float32)
    print(f"      PageRank: {time.time()-t0:.1f}s")

    # Clustering coefficient
    t0 = time.time()
    cc = nx.clustering(G)
    cc_arr = np.array([cc.get(i, 0.0) for i in range(n_nodes)], dtype=np.float32)
    print(f"      Clustering coeff: {time.time()-t0:.1f}s")

    # v2: Closeness centrality (per-node in dual graph = per-link)
    t0 = time.time()
    cl = nx.closeness_centrality(G)
    cl_arr = np.array([cl.get(i, 0.0) for i in range(n_nodes)], dtype=np.float32)
    print(f"      Closeness centrality: {time.time()-t0:.1f}s")

    # OLD: return np.column_stack([rel_deg, bc_arr, pr_arr, cc_arr])  # [N, 10]
    return np.column_stack([rel_deg, bc_arr, pr_arr, cc_arr, cl_arr])  # [N, 11]


# === Cluster-count selection: elbow + silhouette =====================

def _elbow_knee(ks, inertias):
    """Dependency-free elbow ('kneedle'): the k whose (normalised) inertia point
    lies farthest below the chord joining the first and last sweep points."""
    ks = np.asarray(ks, dtype=float)
    ys = np.asarray(inertias, dtype=float)
    x = (ks - ks.min()) / max(ks.max() - ks.min(), 1e-12)
    y = (ys - ys.min()) / max(ys.max() - ys.min(), 1e-12)
    x0, y0, x1, y1 = x[0], y[0], x[-1], y[-1]
    denom = np.hypot(y1 - y0, x1 - x0) + 1e-12
    dist = np.abs((y1 - y0) * x - (x1 - x0) * y + x1 * y0 - y1 * x0) / denom
    return int(ks[int(np.argmax(dist))])


def _sweep_cluster_counts(X, k_range, label, sample_size=5000, seed=42):
    """Fit KMeans for every k in k_range; collect inertia (WCSS) + silhouette.
    Returns dict(ks, inertias, silhouettes, k_elbow, k_silhouette)."""
    ks, inertias, sils = [], [], []
    n = X.shape[0]
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=seed, n_init=5, max_iter=300)
        lab = km.fit_predict(X)
        try:
            ss = float(silhouette_score(X, lab, sample_size=min(sample_size, n),
                                        random_state=seed))
        except Exception:
            ss = float("nan")
        ks.append(int(k)); inertias.append(float(km.inertia_)); sils.append(ss)
        print(f"      [{label}] k={int(k):>2}  inertia={inertias[-1]:>12.1f}  silhouette={ss:.4f}")
    k_elbow = _elbow_knee(ks, inertias)
    valid = [(k, s) for k, s in zip(ks, sils) if not np.isnan(s)]
    k_sil = max(valid, key=lambda kv: kv[1])[0] if valid else k_elbow
    return dict(ks=ks, inertias=inertias, silhouettes=sils,
                k_elbow=k_elbow, k_silhouette=k_sil)


# NOTE: figure rendering now lives in plot_cluster_selection.py (decoupled).
# This script only persists the sweep results (save_selection_stats); the plots
# are drawn by reading that file, so a bad figure never costs a clustering re-run.


# === Main =============================================================

def main():
    print("=" * 70)
    print(f"  STRUCTURAL CLUSTERING — p (physical) & t (topology) via elbow method")
    print("=" * 70)

    city_feats27 = {}   # {city: [N, 27]}
    city_topo = {}      # {city: [N, 11]}
    city_n = {}

    for city, cfg in CITIES.items():
        print(f"\n{'~'*60}")
        print(f"  {city.upper()}")
        print(f"{'~'*60}")

        # Load geojson → 27 features
        t0 = time.time()
        gdf = gpd.read_file(cfg["geojson"])
        print(f"  Loaded geojson: {len(gdf)} links ({time.time()-t0:.1f}s)")

        feats27 = extract_27_features(gdf)
        city_feats27[city] = feats27
        city_n[city] = len(gdf)
        print(f"  27 features: {feats27.shape}")

        # OLD: Save static_feats_27.pt  (no longer the primary output)
        # out_path = os.path.join(cfg["out"], "static_feats_27.pt")
        # torch.save({
        #     "features": torch.tensor(feats27),
        #     "feature_names": FEATURE_NAMES_27,
        #     "n_links": len(gdf),
        # }, out_path)
        # print(f"  Saved {out_path}")
        # NOTE: static_feats_31.pt is saved later, after topology metrics are computed

        # Topology metrics
        relations = load_relations(cfg["relations"])
        topo = compute_topology_metrics(len(gdf), relations)
        city_topo[city] = topo
        print(f"  11 topology metrics: {topo.shape}")

        # v2: Save static_feats_31.pt = 27 geojson + 4 per-link topology
        # betweenness=col7, pagerank=col8, clustering_coeff=col9, closeness=col10
        feats31 = np.concatenate([feats27, topo[:, 7:11]], axis=1)  # [N, 31]
        out_path_31 = os.path.join(cfg["out"], "static_feats_31.pt")
        torch.save({
            "features": torch.tensor(feats31),
            "feature_names": FEATURE_NAMES_31,
            "n_links": len(gdf),
        }, out_path_31)
        print(f"  Saved {out_path_31}  ({feats31.shape})")

    # === Pool all cities =====================================
    all_cities = list(CITIES.keys())
    all_feats27 = np.concatenate([city_feats27[c] for c in all_cities], axis=0)
    all_topo    = np.concatenate([city_topo[c] for c in all_cities], axis=0)
    city_labels = np.concatenate([[c]*city_n[c] for c in all_cities])
    total = len(all_feats27)

    print(f"\n{'='*70}")
    print(f"  Pooled: {total} links ({' + '.join(f'{c}:{city_n[c]}' for c in all_cities)})")
    print(f"{'='*70}")

    # === p-clustering (physical: 4 numeric features) — elbow select ==
    # OLD: hardcoded K=K_PHYSICAL. NEW: elbow over P_RANGE (silhouette alongside).
    p_feats = all_feats27[:, K_FEAT_INDICES]  # [N, 4]
    scaler_P = StandardScaler()
    p_norm = scaler_P.fit_transform(p_feats)

    print(f"\n  [p] physical-cluster sweep over {list(P_RANGE)} (4 numeric features)")
    stats_p = _sweep_cluster_counts(p_norm, list(P_RANGE), "p")
    P = (int(P_OVERRIDE) if P_OVERRIDE is not None
         else (stats_p["k_silhouette"] if CLUSTER_SELECT_BY == "silhouette"
               else stats_p["k_elbow"]))
    print(f"  [p] elbow={stats_p['k_elbow']}  silhouette-max={stats_p['k_silhouette']}"
          f"  ->  P={P}" + ("  (override)" if P_OVERRIDE is not None else f"  (by {CLUSTER_SELECT_BY})"))

    # === t-clustering (topology: 11 features) — elbow select =========
    scaler_T = StandardScaler()
    t_norm = scaler_T.fit_transform(all_topo)

    print(f"\n  [t] topology-cluster sweep over {list(T_RANGE)} (11 topology features)")
    stats_t = _sweep_cluster_counts(t_norm, list(T_RANGE), "t")
    T = (int(T_OVERRIDE) if T_OVERRIDE is not None
         else (stats_t["k_silhouette"] if CLUSTER_SELECT_BY == "silhouette"
               else stats_t["k_elbow"]))
    print(f"  [t] elbow={stats_t['k_elbow']}  silhouette-max={stats_t['k_silhouette']}"
          f"  ->  T={T}" + ("  (override)" if T_OVERRIDE is not None else f"  (by {CLUSTER_SELECT_BY})"))

    # === Selection graphs (how p, t were picked) — decoupled renderer ===
    # Persist the sweep results, then render by READING them back, exercising
    # the exact path you can re-run on its own (python plot_cluster_selection.py)
    # without re-fitting K-Means.
    import plot_cluster_selection as pcs
    pcs.save_selection_stats(FIG_DIR, stats_p, stats_t, P, T)
    pcs.plot_from_dir(FIG_DIR)

    # === Final fits at the chosen p, t ===============================
    km_P = KMeans(n_clusters=P, random_state=42, n_init=10, max_iter=300)
    p_labels = km_P.fit_predict(p_norm)
    km_T = KMeans(n_clusters=T, random_state=42, n_init=10, max_iter=300)
    t_labels = km_T.fit_predict(t_norm)

    print(f"\n  Physical clusters (p={P}):")
    header = f"  {'p':<5}"
    for c in all_cities:
        header += f"  {c:>12}"
    header += f"  {'total':>10}"
    print(header)
    for p in range(P):
        row = f"  p={p:<2}"
        for c in all_cities:
            mask = (p_labels == p) & (city_labels == c)
            n = mask.sum()
            pct = 100 * n / city_n[c]
            row += f"  {n:>5}({pct:4.1f}%)"
        row += f"  {(p_labels==p).sum():>10}"
        print(row)

    print(f"\n  Topology clusters (t={T}):")
    header = f"  {'t':<5}"
    for c in all_cities:
        header += f"  {c:>12}"
    header += f"  {'total':>10}"
    print(header)
    for t in range(T):
        row = f"  t={t:<2}"
        for c in all_cities:
            mask = (t_labels == t) & (city_labels == c)
            n = mask.sum()
            pct = 100 * n / city_n[c]
            row += f"  {n:>5}({pct:4.1f}%)"
        row += f"  {(t_labels==t).sum():>10}"
        print(row)

    # === 2D overlap analysis =================================
    print(f"\n{'='*70}")
    print(f"  2D OVERLAP — (p, t) combinations")
    print(f"{'='*70}")

    offsets = {}
    offset = 0
    city_pairs = {}
    for c in all_cities:
        n = city_n[c]
        cp = p_labels[offset:offset+n]
        ct = t_labels[offset:offset+n]
        city_pairs[c] = set(zip(cp.tolist(), ct.tolist()))
        offsets[c] = (offset, offset + n)
        offset += n

    all_combos = set()
    for c in all_cities:
        all_combos |= city_pairs[c]

    print(f"\n  Unique (p,t): {len(all_combos)} / {P * T} possible")
    for c in all_cities:
        print(f"    {c}: {len(city_pairs[c])}")

    # Pairwise
    print(f"\n  Pairwise overlap:")
    for i, c1 in enumerate(all_cities):
        for c2 in all_cities[i+1:]:
            shared = city_pairs[c1] & city_pairs[c2]
            print(f"    {c1} & {c2}: {len(shared)} shared")

    all_way = set.intersection(*[city_pairs[c] for c in all_cities]) if all_cities else set()
    print(f"    ALL {len(all_cities)}: {len(all_way)} shared")

    # Link-level coverage
    print(f"\n  LINK-LEVEL TRANSFER COVERAGE:")
    for target in all_cities:
        sources = [c for c in all_cities if c != target]
        source_combos = set()
        for s in sources:
            source_combos |= city_pairs[s]
        tgt_start, tgt_end = offsets[target]
        tgt_p = p_labels[tgt_start:tgt_end]
        tgt_t = t_labels[tgt_start:tgt_end]
        covered = sum(1 for i in range(city_n[target]) if (tgt_p[i], tgt_t[i]) in source_combos)
        print(f"    {target} as target: {covered}/{city_n[target]} ({100*covered/city_n[target]:.1f}%)")

    # === Save cluster assignments ============================
    print(f"\n{'='*70}")
    print(f"  SAVING CLUSTER ASSIGNMENTS")
    print(f"{'='*70}")

    for c in all_cities:
        start, end = offsets[c]
        p_t = torch.tensor(p_labels[start:end], dtype=torch.long)
        t_t = torch.tensor(t_labels[start:end], dtype=torch.long)
        out = {
            # --- canonical (v3) p / t naming: a link belongs to a (p, t) slot ---
            "p_labels": p_t,
            "t_labels": t_t,
            "P": P,
            "T": T,
            "p_features": ["length", "lanes", "width_m", "speed_limit_ms"],
            "t_features": TOPO_NAMES,
            # --- legacy aliases (k == p physical, s == t topology): same tensors/
            #     values, kept so existing consumers (MetaTraffic.build_prototype_bank
            #     / evaluator / main) keep loading without a cross-file rename ---
            "k_labels": p_t,
            "s_labels": t_t,
            "K": P,
            "S": T,
            "k_features": ["length", "lanes", "width_m", "speed_limit_ms"],
            "s_features": TOPO_NAMES,
            "feature_names_27": FEATURE_NAMES_27,
            "feature_names_31": FEATURE_NAMES_31,
        }
        path = os.path.join(CITIES[c]["out"], "cluster_assignments.pt")
        torch.save(out, path)
        print(f"  Saved {path}: {end-start} links")

    print(f"\n  DONE.")

    # === v2: Compute & save city-level descriptors ============
    print(f"\n{'='*70}")
    print(f"  CITY-LEVEL DESCRIPTORS (for city-similarity weighting)")
    print(f"{'='*70}")
    for city, cfg in CITIES.items():
        gdf = gpd.read_file(cfg["geojson"])
        relations = load_relations(cfg["relations"])
        desc = compute_city_descriptor(len(gdf), gdf, relations, crs_metric=cfg["crs_metric"])
        desc_path = os.path.join(cfg["out"], "city_descriptor.pt")
        torch.save(desc, desc_path)
        print(f"  {city}: saved {desc_path}")
        print(f"    descriptor ({len(desc['descriptor_names'])} dims): "
              f"{dict(zip(desc['descriptor_names'], [f'{v:.4f}' for v in desc['descriptor'].tolist()]))}")


# =============================================================================
# CITY-LEVEL DESCRIPTOR: For city-similarity weighted prototype bank
# =============================================================================
CITY_DESCRIPTOR_NAMES = [
    'mean_closeness', 'std_closeness',
    'mean_betweenness', 'std_betweenness',
    'mean_clustering_coeff', 'std_clustering_coeff',
    'mean_degree', 'std_degree',
    'graph_density',
    'avg_path_length',
    'accessibility_index',
    'detour_index',
    'shannon_wiener_diversity',
]


def compute_city_descriptor(
    n_nodes: int,
    gdf: gpd.GeoDataFrame,
    relations: dict,
    crs_metric: str = "EPSG:32648",
) -> dict:
    """
    Compute a city-level descriptor vector for cross-city similarity comparison.

    Combines graph-theoretic metrics (from the Sehra et al. papers) into
    a single vector per city for cosine similarity weighting.

    Args:
        n_nodes:    Number of edges/links in the city (= nodes in the dual graph).
        gdf:        GeoDataFrame of edges (EPSG:4326).
        relations:  Dict of {relation_name: COO array}.
        crs_metric: Projected CRS for this city (metres), e.g. "EPSG:32629".

    Returns dict with 'descriptor' (tensor) and 'descriptor_names' (list).
    """
    # --- Reproject GeoDataFrame to metric CRS (metres) for distance computations ---
    gdf_m = gdf.to_crs(crs_metric)

    # --- Build unweighted graph (for closeness, betweenness, clustering, degree) ---
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    for rn, coo in relations.items():
        if rn == 'self' or coo.shape[1] == 0:
            continue
        G.add_edges_from(zip(coo[0].tolist(), coo[1].tolist()))

    # --- Build weighted graph (edge weight = physical length in metres) ---
    # Used for avg_path_length, accessibility, and detour index.
    G_w = nx.Graph()
    G_w.add_nodes_from(range(n_nodes))
    edge_lengths_m = _compute_edge_lengths_m(gdf_m)  # [N] array, metres
    for rn, coo in relations.items():
        if rn == 'self' or coo.shape[1] == 0:
            continue
        src_arr = coo[0].tolist()
        dst_arr = coo[1].tolist()
        for s, d in zip(src_arr, dst_arr):
            # Average the physical lengths of the two endpoint links as edge weight
            w = (edge_lengths_m[s] + edge_lengths_m[d]) / 2.0
            if w < 1e-6:
                w = 1.0  # fallback for zero-length edges
            if G_w.has_edge(s, d):
                G_w[s][d]['weight'] = min(G_w[s][d]['weight'], w)  # keep shortest
            else:
                G_w.add_edge(s, d, weight=w)

    print(f"    Building city descriptor (N={G.number_of_nodes()}, E={G.number_of_edges()}, "
          f"E_w={G_w.number_of_edges()}, CRS={crs_metric})...")

    t0 = time.time()

    # --- 1. Per-node metrics aggregated to city level (mean, std) ---

    # Closeness centrality (unweighted — topology)
    cl = nx.closeness_centrality(G)
    cl_vals = np.array([cl.get(i, 0.0) for i in range(n_nodes)], dtype=np.float32)
    mean_closeness = float(np.mean(cl_vals))
    std_closeness = float(np.std(cl_vals))

    # Betweenness centrality (approx, unweighted)
    k_sample = min(500, n_nodes)
    bc = nx.betweenness_centrality(G, k=k_sample, seed=42)
    bc_vals = np.array([bc.get(i, 0.0) for i in range(n_nodes)], dtype=np.float32)
    mean_betweenness = float(np.mean(bc_vals))
    std_betweenness = float(np.std(bc_vals))

    # Clustering coefficient
    cc = nx.clustering(G)
    cc_vals = np.array([cc.get(i, 0.0) for i in range(n_nodes)], dtype=np.float32)
    mean_cc = float(np.mean(cc_vals))
    std_cc = float(np.std(cc_vals))

    # Degree distribution
    deg = np.array([G.degree(i) for i in range(n_nodes)], dtype=np.float32)
    mean_degree = float(np.mean(deg))
    std_degree = float(np.std(deg))

    # --- 2. Whole-graph metrics ---

    # Graph density
    graph_density = float(nx.density(G))

    # Average path length — WEIGHTED (metres) on largest connected component
    # Sehra (2019): "evaluates the reachability, efficiency, and connectivity"
    # Using physical distance (metres) so it's comparable across cities.
    largest_cc_w = max(nx.connected_components(G_w), key=len)
    if len(largest_cc_w) > 1:
        G_lcc_w = G_w.subgraph(largest_cc_w)
        # Sample-based approximation for large graphs
        if len(largest_cc_w) > 2000:
            sample_nodes = list(largest_cc_w)
            rng = np.random.RandomState(42)
            rng.shuffle(sample_nodes)
            sample_nodes = sample_nodes[:500]
            total_path = 0.0
            n_pairs = 0
            for src in sample_nodes:
                lengths = nx.single_source_dijkstra_path_length(G_lcc_w, src, weight='weight')
                for tgt, d in lengths.items():
                    if tgt != src:
                        total_path += d
                        n_pairs += 1
            avg_path_length = float(total_path / max(n_pairs, 1))
        else:
            avg_path_length = float(nx.average_shortest_path_length(G_lcc_w, weight='weight'))
    else:
        avg_path_length = 0.0

    # --- 3. Sehra-paper metrics: Accessibility, Detour, Shannon-Wiener ---

    # Accessibility index (Sehra 2019, Eq. 5-6):
    #   D_i = sum of shortest distances from node i to all others (weighted, metres)
    #   A'_i = D_i / ((n-1) * mean(D))   — normalised per-node ranking
    #   City-level = mean(A'_i)
    # OLD (v2): weighted closeness centrality — different aggregation, not matching paper
    # cl_w = nx.closeness_centrality(G_w, distance='weight')
    # cl_w_vals = np.array([cl_w.get(i, 0.0) for i in range(n_nodes)], dtype=np.float32)
    # accessibility_index = float(np.mean(cl_w_vals))
    largest_cc_acc = max(nx.connected_components(G_w), key=len)
    G_lcc_acc = G_w.subgraph(largest_cc_acc)
    lcc_nodes = sorted(largest_cc_acc)
    n_lcc = len(lcc_nodes)
    if n_lcc > 1:
        # Sample-based for large graphs
        if n_lcc > 2000:
            rng_acc = np.random.RandomState(42)
            sample_acc = list(lcc_nodes)
            rng_acc.shuffle(sample_acc)
            sample_acc = sample_acc[:500]
        else:
            sample_acc = lcc_nodes
        Di_vals = []
        for src in sample_acc:
            dists = nx.single_source_dijkstra_path_length(G_lcc_acc, src, weight='weight')
            Di = sum(d for t, d in dists.items() if t != src)
            Di_vals.append(Di)
        Di_arr = np.array(Di_vals, dtype=np.float64)
        mean_D = float(np.mean(Di_arr))
        # A'_i = D_i / ((n_lcc - 1) * mean_D)  — Sehra Eq. 6
        if mean_D > 1e-9:
            Ai_norm = Di_arr / ((n_lcc - 1) * mean_D)
            accessibility_index = float(np.mean(Ai_norm))
        else:
            accessibility_index = 0.0
    else:
        accessibility_index = 0.0
    print(f"      Accessibility index (Sehra Eq.6): {accessibility_index:.6f}")

    # Detour index (Sehra 2019, Eq. 7):
    #   DI = d_S(i,j) / d_H(i,j)  — graph distance / Euclidean distance
    #   DI ≥ 1.0; closer to 1.0 = more efficient/straight network.
    # OLD (v1): per-edge sinuosity (wrong formula + MultiLineString bug → always 1.0)
    # OLD (v2): Euclidean / graph (inverted from paper — reciprocal)
    # detour_vals = _compute_detour_ratios(gdf)
    # detour_index = float(np.mean(detour_vals)) if len(detour_vals) > 0 else 1.0
    detour_index = _compute_network_detour_index(G_w, gdf_m, n_nodes)
    print(f"      Detour index (Sehra Eq.7, >=1.0): {detour_index:.6f}")

    # Shannon-Wiener diversity (Sehra 2019, Eq. 3):
    #   H' = -Σ P_i * ln(P_i),  where P_i = n_i / N
    #   n_i = total road LENGTH for highway type i (metres)
    #   N   = total road length for all types
    # OLD: used edge COUNT per type (wrong — paper specifies length proportions)
    # hw_counts = gdf['highway'].fillna('').astype(str).str.lower().str.strip().value_counts()
    # hw_probs = hw_counts.values.astype(np.float64)
    hw_type_col = gdf_m['highway'].fillna('').astype(str).str.lower().str.strip()
    hw_length_col = gdf_m.geometry.length  # metres (already projected)
    hw_length_by_type = pd.Series(hw_length_col.values, index=hw_type_col.values)
    hw_length_sums = hw_length_by_type.groupby(hw_length_by_type.index).sum()
    hw_length_sums = hw_length_sums[hw_length_sums.index != '']  # drop empty strings
    hw_probs = hw_length_sums.values.astype(np.float64)
    hw_probs = hw_probs[hw_probs > 0]
    hw_probs = hw_probs / hw_probs.sum()
    shannon_wiener = float(_scipy_entropy(hw_probs))  # natural log entropy

    elapsed = time.time() - t0
    print(f"    City descriptor computed in {elapsed:.1f}s")

    descriptor = torch.tensor([
        mean_closeness, std_closeness,
        mean_betweenness, std_betweenness,
        mean_cc, std_cc,
        mean_degree, std_degree,
        graph_density,
        avg_path_length,
        accessibility_index,
        detour_index,
        shannon_wiener,
    ], dtype=torch.float32)

    return {
        'descriptor': descriptor,
        'descriptor_names': CITY_DESCRIPTOR_NAMES,
        'city_n_nodes': n_nodes,
        'city_n_edges': G.number_of_edges(),
    }


# OLD: _compute_detour_ratios — per-edge sinuosity (wrong metric + MultiLineString bug)
# def _compute_detour_ratios(gdf: gpd.GeoDataFrame) -> np.ndarray:
#     """Compute per-edge detour ratio = physical_length / straight_line_distance."""
#     ...  (silently returned 1.0 for all MultiLineString geometries)


def _compute_edge_lengths_m(gdf_m: gpd.GeoDataFrame) -> np.ndarray:
    """
    Compute physical length in metres for each edge.
    Handles both LineString and MultiLineString geometries.
    gdf_m must already be in a projected (metric) CRS.
    """
    lengths = np.zeros(len(gdf_m), dtype=np.float64)
    for i, geom in enumerate(gdf_m.geometry):
        if geom is None or geom.is_empty:
            continue
        lengths[i] = geom.length  # in projected CRS = metres
    return lengths


def _get_node_coords_m(gdf_m: gpd.GeoDataFrame, n_nodes: int) -> np.ndarray:
    """
    For each node (= edge in the road graph), compute a representative
    coordinate in the projected CRS (metres).  Uses the centroid of
    each edge's geometry.

    Returns [n_nodes, 2] float64 array of (x, y) in metres.
    """
    coords = np.zeros((n_nodes, 2), dtype=np.float64)
    for i, geom in enumerate(gdf_m.geometry):
        if i >= n_nodes:
            break
        if geom is None or geom.is_empty:
            continue
        c = geom.centroid
        coords[i, 0] = c.x
        coords[i, 1] = c.y
    return coords


def _compute_network_detour_index(
    G_w: nx.Graph,
    gdf_m: gpd.GeoDataFrame,
    n_nodes: int,
    n_sample: int = 500,
    seed: int = 42,
) -> float:
    """
    Network-level detour index (Sehra 2019, Eq. 7).

    For sampled node pairs (i, j), compute:
        DI = mean( graph_dist(i,j) / Euclidean_dist(i,j) )

    DI ≥ 1.0 (by triangle inequality).
    Values close to 1.0 = efficient / direct network.
    Values >> 1.0 = circuitous routing.

    Uses:
      - G_w: weighted graph (edge weights = metres)
      - gdf_m: GeoDataFrame in projected CRS (for node coordinates)
    """
    # Get node coordinates in metres
    node_xy = _get_node_coords_m(gdf_m, n_nodes)  # [N, 2]

    # Work on the largest connected component
    largest_cc = max(nx.connected_components(G_w), key=len)
    cc_nodes = sorted(largest_cc)

    if len(cc_nodes) < 2:
        return 1.0

    rng = np.random.RandomState(seed)
    sample_nodes = list(cc_nodes)
    rng.shuffle(sample_nodes)
    sample_nodes = sample_nodes[:min(n_sample, len(sample_nodes))]

    G_lcc = G_w.subgraph(largest_cc)

    ratios = []
    for src in sample_nodes:
        # Dijkstra from src → all reachable nodes (weighted = metres)
        graph_dists = nx.single_source_dijkstra_path_length(G_lcc, src, weight='weight')
        src_xy = node_xy[src]
        for tgt, g_dist in graph_dists.items():
            if tgt == src or g_dist < 1e-6:
                continue
            # Euclidean distance in metres
            e_dist = np.linalg.norm(node_xy[tgt] - src_xy)
            if e_dist < 1e-6:
                continue
            # Sehra Eq. 7: DI = graph_distance / Euclidean  (always ≥ 1.0)
            ratios.append(g_dist / e_dist)

    if len(ratios) == 0:
        return 1.0

    detour = float(np.mean(ratios))
    print(f"      Detour index: {len(ratios):,} pairs sampled from {len(sample_nodes)} source nodes")
    return detour


if __name__ == "__main__":
    main()
