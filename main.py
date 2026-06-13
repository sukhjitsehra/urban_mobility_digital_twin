#!/usr/bin/env python
# coding: utf-8
"""
Meta_Learning experiment runner.

Cross-city meta-learning for travel-time estimation. Three meta-learning
methods sit on top of a shared LinkETA backbone:

    MetaST       FOMAML + STMemory (edge-pattern clustering, G=4)
    MetaTTE      Reptile (raw LinkETA, no memory / clustering)
    MetaTraffic  MAML (FOMAML) with a traffic-profile generator + structural
                 cluster-prototype transfer

Unlike LinkETA (where each city trains independently), meta-learning binds a
set of SOURCE cities (meta-trained jointly) to one or more TARGET cities
(transfer / fine-tune). Roles and per-city trip counts are chosen at run time;
city data paths live in CITY_CONFIGS below.

Layout (script-relative; main.py chdir's here so all relative paths resolve
identically regardless of where it is launched from):
    Meta_Learning/
        main.py
        LinkETA_Fixed.py  MetaST.py  MetaTTE.py  MetaTraffic.py
        eval.py  checkpoint.py  loss_functions.py   (flat model modules)
        structural_clustering.py                     (MetaTraffic prototype artifacts)
        meta_learning_data/<city>/                   (per-city data + precomputed:
            df_all_<city>.csv  edges_<city>.geojson  link_traffic_profile.parquet
            static_feats_31.pt  cluster_assignments.pt  city_descriptor.pt
            ckpts_LinkETA_<city>/best.ckpt           (per-city base LinkETA ckpt)
        checkpoints/<sources>_source_<targets>_target/<city>/
            ckpts_MetaST_<city>/  ckpts_MetaTTE_<city>/  ckpts_MetaTraffic_<city>/
            figs/                                    (per-city eval trip-pred CSVs)

            
Run interactively:
    python main.py
Run with flags:
    python main.py --blocks MetaST --sources porto,sf --targets chengdu
    python main.py --blocks MetaST,MetaTTE,MetaTraffic,Evaluation --sources porto,sf --targets chengdu
    python main.py --blocks all --sources porto,sf --targets chengdu --trips porto=1000,sf=1000,chengdu=500
    python main.py --blocks MetaST --sources porto,sf --targets chengdu --meta-iters 1000 --inner-steps 10

This file is a thin orchestrator: it reproduces the cell-by-cell flow of
main.ipynb (archived) but contains no model code. All model logic stays in the
unchanged flat model modules and structural_clustering.py.
"""


import argparse
import json
import math
import os
import sys
from pathlib import Path

import pandas as pd
import geopandas as gpd

# [2026-06-07] Force line-buffered stdout so logs (build progress, per-round
# [adapt] transfer metrics, etc.) appear LIVE even when stdout is piped (VS Code
# integrated terminal, log redirects). Without this Python block-buffers (~8 KB)
# a non-TTY stdout and the run looks "frozen" while it is actually computing.
try:
    sys.stdout.reconfigure(line_buffering=True)
except (AttributeError, ValueError):
    pass  # older Python / already-detached stream — harmless


# --- Paths (script-relative, location-independent) -----------------------
BASE_DIR = Path(__file__).resolve().parent
# Per-city data + precomputed artifacts (static feats, clusters, profiles,
# per-city base LinkETA checkpoints) all live under meta_learning_data/<city>/.
DATA_ROOT = BASE_DIR / "meta_learning_data"
# Meta-method checkpoints + evaluation outputs are grouped per run under
# checkpoints/<sources>_source_<targets>_target/<city>/ .
CHECKPOINTS_ROOT = BASE_DIR / "checkpoints"


def city_dir(city: str) -> Path:
    """Per-city data folder, e.g. meta_learning_data/porto/."""
    return DATA_ROOT / city


# [2026-06-07] DECOUPLED LAYOUT. theta0 (meta-trained) depends only on the SOURCES,
# so it is filed at a sources-keyed run dir (tagged with source trip-counts + seed
# so different meta-trainings don't overwrite each other). Targets are subfolders
# holding only the transfer (adapted) ckpts + eval/figs — they depend on the target.
#
#   checkpoints/<sources>_source_t<srctrips>_s<seed>/      (= the "run", target-independent)
#       ckpts_MetaETA/ ckpts_MetaST/ ckpts_MetaTTE/         theta0 (+ interval/history/summary)
#       <target>/ ckpts_<method>_transfer/  eval_outputs/  figs/
#
def _source_trips_tag(source_keys) -> str:
    """Compact tag of the SOURCE trip-counts so re-meta-training with a different
    #trips/seed gets its own dir (None -> 'all')."""
    vals = [str(CITY_CONFIGS[k].get("sample_trips") or "all") for k in source_keys]
    return "t" + "_".join(vals) + f"_s{SEED}"


def run_folder_name(source_keys, target_keys=None) -> str:
    """Sources-keyed run name, e.g. 'porto_sf_source_t2000_2000_s42'.
    Target-INDEPENDENT (target_keys ignored; kept for signature back-compat)."""
    return f"{'_'.join(source_keys)}_source_{_source_trips_tag(source_keys)}"


def run_dir(ctx) -> Path:
    return CHECKPOINTS_ROOT / run_folder_name(ctx["source_keys"])


def meta_ckpt_dir(ctx, method: str) -> str:
    """theta0 (meta-trained, SOURCES-only) dir, e.g.
    checkpoints/porto_sf_source_t.../ckpts_MetaETA/. Independent of any target,
    so one meta-training is reused to transfer to any held-out target."""
    d = run_dir(ctx) / f"ckpts_{method}"
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


def target_dir(ctx, city: str) -> Path:
    d = run_dir(ctx) / city
    d.mkdir(parents=True, exist_ok=True)
    return d


def transfer_ckpt_dir(ctx, method: str, city: str) -> str:
    """Per-target ADAPTED ckpt dir, e.g.
    checkpoints/porto_sf_source_t.../chengdu/ckpts_MetaETA_transfer/."""
    d = target_dir(ctx, city) / f"ckpts_{method}_transfer"
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


def city_figs_dir(ctx, city: str) -> str:
    d = target_dir(ctx, city) / "figs"
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


def city_eval_outputs_dir(ctx, city: str) -> str:
    """Per-target evaluator output dir (eval_metrics.json / eval_trips.csv /
    eval_links.csv + benchmark_*.csv / significance.csv)."""
    d = target_dir(ctx, city) / "eval_outputs"
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


# --- City configuration --------------------------------------------------
# Verbatim from main.ipynb. `role` and `sample_trips` are defaults; both are
# overridden at run time by --sources/--targets and --trips respectively.
CITY_CONFIGS = {
    "porto": {
        "name": "Porto",
        "role": "source",                # source city for meta-training
        "sample_trips": None,            # None = use ALL trips
        "trips_path": r"./meta_learning_data/porto/df_all_porto.csv",
        "edges_path": r"./meta_learning_data/porto/edges_porto.geojson",
        "precomputed_dir": r"./meta_learning_data/porto",
        "CHECKPOINT_PATH": r"./meta_learning_data/porto/ckpts_LinkETA_porto/best.ckpt",
        "crs_metric": "EPSG:32629",
    },
    "sf": {
        "name": "San Francisco",
        "role": "source",
        "sample_trips": None,            # None = use ALL trips
        "trips_path": r"./meta_learning_data/sf/df_all_sf.csv",
        "edges_path": r"./meta_learning_data/sf/edges_sf.geojson",
        "precomputed_dir": r"./meta_learning_data/sf",
        "CHECKPOINT_PATH": r"./meta_learning_data/sf/ckpts_LinkETA_sf/best.ckpt",
        "crs_metric": "EPSG:26910",
    },
    "chengdu": {
        "name": "Chengdu",
        "role": "target",                # target city for transfer
        "sample_trips": None,            # None = use ALL trips
        "trips_path": r"./meta_learning_data/chengdu/df_all_chengdu.csv",
        "edges_path": r"./meta_learning_data/chengdu/edges_chengdu.geojson",
        "precomputed_dir": r"./meta_learning_data/chengdu",
        "CHECKPOINT_PATH": r"./meta_learning_data/chengdu/ckpts_LinkETA_chengdu/best.ckpt",
        "crs_metric": "EPSG:32648",
    },
}

# Shared LinkETA architecture constants (verbatim from main.ipynb).
IN_DIM = 86

# CLI-overridable hyperparameters (None => use each method's notebook default).
META_ITERS_OVERRIDE = None
INNER_STEPS_OVERRIDE = None
ADAPT_STEPS_OVERRIDE = None
SEED = 42
# [2026-06-06] interval meta-checkpoint cadence (theta0_iter{it}.ckpt every N
# meta-iters; 0 disables). best.ckpt + last.ckpt are always saved regardless.
SAVE_EVERY = 5
# [2026-06-07] number of transfer/adaptation repeats at score time (meta-training
# is done once; transfer is cheap). N>1 => report mean ± std over the N reps in
# the eval table + eval_metrics.json. Default 1 = single run, behaviour unchanged.
TRANSFER_REPS = 1
# [2026-06-07] diagnostic: disable the MetaETA prototype bank at transfer (pure
# W_gen, alpha=0) to isolate the prototype-injection effect. Default off.
NO_PROTOTYPE = False


def _mi(default):
    """Resolve meta_iters: CLI override if given, else the notebook default."""
    return META_ITERS_OVERRIDE if META_ITERS_OVERRIDE is not None else default


def _is(default):
    """Resolve inner_steps: CLI override if given, else the notebook default."""
    return INNER_STEPS_OVERRIDE if INNER_STEPS_OVERRIDE is not None else default


def _as(default):
    """Resolve adapt_steps: CLI override if given, else the notebook default."""
    return ADAPT_STEPS_OVERRIDE if ADAPT_STEPS_OVERRIDE is not None else default


# =============================================================================
# Core construction (notebook: "Input Data" + "Build one core per city")
# =============================================================================
def build_cores(city_keys, cache_datasets=False):
    """Build one LinkETA core per active city, then run a one-epoch sanity check.

    Mirrors the notebook's core-building cell: load each city's trips/edges,
    sub-sample TRIP_IDs (per-city sample_trips), init LinkETA with that city's
    precomputed data + checkpoint, and assemble the core (splits, loaders,
    optimizer, scheduler).
    """
    from LinkETA import (
        init_linketa, linketa_get_core_components, one_epoch,
    )

    cores = {}
    for city_key in city_keys:
        cfg = CITY_CONFIGS[city_key]
        print(f"\n{'='*60}")
        print(f"  Building core: {cfg['name']}  ({cfg['role']})")
        print(f"{'='*60}")

        # 1) Load city data as DataFrames
        city_df = pd.read_csv(cfg["trips_path"])
        city_edges = gpd.read_file(cfg["edges_path"])

        # 2) Sub-sample trips (per-city control via sample_trips)
        sample_n = cfg.get("sample_trips")   # None = use all trips
        if sample_n is not None:
            trip_ids = city_df["TRIP_ID"].drop_duplicates()
            n = min(sample_n, len(trip_ids))
            picked = trip_ids.sample(n=n, random_state=42)
            city_df = city_df[city_df["TRIP_ID"].isin(picked)]
        print(f"  Trips: {city_df['TRIP_ID'].nunique()}, Rows (links): {len(city_df)}")

        # 3) Init LinkETA with this city's precomputed data & checkpoint
        base = init_linketa(
            precomputed_dir=cfg["precomputed_dir"],
            checkpoint_path=cfg["CHECKPOINT_PATH"],
        )

        # 4) Build core (train/val/test split, datasets, loaders, optimizer, scheduler)
        #    Pass the precomputed `base` for the fast rerun path (stateless).
        cores[city_key] = linketa_get_core_components(
            city_df, city_edges, base=base,
            cache_datasets=cache_datasets, cache_seed=SEED)

    # ---- Summary ----
    print(f"\n{'='*60}")
    source_keys = [k for k in city_keys if CITY_CONFIGS[k]["role"] == "source"]
    target_keys = [k for k in city_keys if CITY_CONFIGS[k]["role"] == "target"]
    print(f"Source cities: {source_keys}")
    print(f"Target cities: {target_keys}")
    for k, c in cores.items():
        splits = {s: len(d) for s, d in c["datasets"].items()}
        print(f"  {k} ({CITY_CONFIGS[k]['role']}): {splits}")

    # ---- Verify: run one eval epoch on each city's test set ----
    print("\n[VERIFY] one_epoch on each city's test set:")
    for city_key, core in cores.items():
        result = one_epoch(core, core["loaders"]["test"], epoch=0, train=False)
        clean = {k: f"{v:.4f}" for k, v in result.items() if not math.isnan(v)}
        print(f"  {city_key}: {clean}")

    return cores


# =============================================================================
# Block: MetaST  (notebook section "# MetaST")
# =============================================================================
def run_metast(ctx):
    cores = ctx["cores"]
    source_keys = ctx["source_keys"]
    target_keys = ctx["target_keys"]
    print(f"\n{'='*70}\n[BLOCK] MetaST  sources={source_keys} targets={target_keys}\n{'='*70}")

    import gc
    import torch
    import MetaST as MetaST
    from MetaST import (
        extract_edge_patterns_from_profile,
        run_metast_clustering_pipeline,
        get_n_edges_from_core,
        create_metast_model,
        set_globals_from_core,
        meta_train_metast,
    )
    from LinkETA import LinkETA, CFG

    # ---- Step 1: Extract 24h edge patterns from precomputed traffic profiles ----
    pattern_dfs = {}
    for city_key in (source_keys + target_keys):
        cfg = CITY_CONFIGS[city_key]
        profile_path = os.path.join(cfg["precomputed_dir"], "link_traffic_profile.parquet")
        traffic_profile = pd.read_parquet(profile_path)
        edges_gdf = cores[city_key]["graph"]["edges_vff"]
        patterns = extract_edge_patterns_from_profile(
            traffic_profile=traffic_profile,
            edges_gdf=edges_gdf,
        )
        pattern_dfs[city_key] = patterns
        print(f"  {cfg['name']}: {len(patterns)} edges with 24h patterns")
    print(f"\nPattern extraction complete for {list(pattern_dfs.keys())}")

    # ---- Step 2: Cluster edge patterns (G=4 fixed) ----
    n_edges = {}
    for city_key in (source_keys + target_keys):
        n_edges[city_key] = get_n_edges_from_core(cores[city_key])
        print(f"  {city_key}: {n_edges[city_key]} edges in graph")

    source_patterns = {CITY_CONFIGS[k]["name"]: pattern_dfs[k] for k in source_keys}
    target_patterns = {CITY_CONFIGS[k]["name"]: pattern_dfs[k] for k in target_keys}
    n_edges_named = {CITY_CONFIGS[k]["name"]: n_edges[k] for k in (source_keys + target_keys)}

    print(f"\nSource cities: {list(source_patterns.keys())}")
    print(f"Target cities: {list(target_patterns.keys())}")

    G = 4  # paper-faithful fixed G
    clustering_result = run_metast_clustering_pipeline(
        source_pattern_dfs=source_patterns,
        target_pattern_dfs=target_patterns,
        n_edges_per_city=n_edges_named,
        n_clusters=G,
    )
    print(f"\nFinal G (n_clusters): {clustering_result['n_clusters']}")

    # ---- Step 3: Inject edge_cluster_labels tensors into each core ----
    name_to_key = {CITY_CONFIGS[k]["name"]: k for k in (source_keys + target_keys)}
    for city_name, tensor in clustering_result["edge_cluster_tensors"].items():
        city_key = name_to_key[city_name]
        device = cores[city_key]["device"]
        cores[city_key]["graph"]["edge_cluster_labels"] = tensor.to(device)
        print(f"  Injected edge_cluster_labels into cores['{city_key}'] -> device={device}, shape={tuple(tensor.shape)}")
    print("\nClustering complete, labels injected into all cores.")

    # ---- Step 4: Create FRESH randomly-initialized model + STMemory (Paper Alg.1 Line 1) ----
    gc.collect()
    torch.cuda.empty_cache()

    device = cores[source_keys[0]]["device"]
    fresh_base = LinkETA(
        in_dim=IN_DIM,
        hidden_dim=CFG["hidden_dim"],       # 32
        heads=CFG["heads"],                 # 8
        relations=8,                        # all cities have 8 relation types
        num_layers=CFG["num_layers"],       # 1
        max_route_len=CFG.get("max_route_len", 1052),
        num_experts=3,
        dropout=0.3,
        input_dropout=0.2,
    ).to(device)
    print(f"Created FRESH LinkETA (random init): {sum(p.numel() for p in fresh_base.parameters())/1e6:.2f}M params")

    wrapped_model = create_metast_model(
        base_model=fresh_base,
        G=G,        # 4 clusters
        d_mem=8,    # paper d_mem
        device=device,
    )
    MetaST.model = wrapped_model
    for k in cores:
        cores[k]["model"] = wrapped_model   # shared model across all cities
    print(f"Model type: {type(wrapped_model).__name__}")
    print("theta0 and M randomly initialized (no checkpoint loaded) - paper-faithful")

    # ---- Step 5: Meta-Train on source cities ----
    set_globals_from_core(cores[source_keys[0]])
    source_cores = [cores[k] for k in source_keys]
    source_names = [CITY_CONFIGS[k]["name"] for k in source_keys]
    # [2026-06-07] meta-training is SOURCES-only; no target needed (theta0 -> meta_ckpt_dir)

    theta0_meta = meta_train_metast(
        source_cores=source_cores,
        source_names=source_names,
        meta_iters=_mi(1000),     # paper: 20k (notebook uses 1000 for pipeline validation)
        inner_steps=_is(10),      # paper: K=5
        inner_lr=1e-3,            # higher for meaningful inner adaptation with SGD
        outer_lr=1e-4,            # 10x paper for faster convergence in short runs
        lambda_mem=1e-4,          # paper: gamma
        dwa_T=2.0,                # DWA temperature (0 = disable, use fixed weights)
        seed=SEED,
        enable_metrics=True,
        log_every=10,
        patience=50,              # generous patience for 200 iters
        max_query_batches=5,      # speed vs accuracy tradeoff for outer loss
        # [2026-06-07] theta0 is SOURCES-only -> sources-keyed run dir (no target)
        ckpt_dir=meta_ckpt_dir(ctx, "MetaST"),
        save_every=SAVE_EVERY,    # [2026-06-06] interval theta0 ckpts (0=off)
    )

    # [2026-06-06] DECOUPLED: no transfer here. theta0 (+ training_history +
    # interval ckpts) saved by meta_train_metast; adaptation happens in Evaluation.
    ctx["wrapped_model"] = wrapped_model
    ctx["theta0_meta"] = theta0_meta
    return theta0_meta


# =============================================================================
# Block: MetaTTE  (notebook section "# MetaTTE")
# =============================================================================
def run_metatte(ctx):
    cores = ctx["cores"]
    source_keys = ctx["source_keys"]
    target_keys = ctx["target_keys"]
    print(f"\n{'='*70}\n[BLOCK] MetaTTE  sources={source_keys} targets={target_keys}\n{'='*70}")

    import MetaTTE as MetaTTE
    from MetaTTE import (
        set_globals_from_core as metatte_set_globals,
        meta_train_reptile,
    )
    from LinkETA import LinkETA, CFG

    # ---- Step 1: Create FRESH base LinkETA model (no STMemory) ----
    device_reptile = cores[source_keys[0]]["device"]
    reptile_base_model = LinkETA(
        in_dim=IN_DIM,
        hidden_dim=CFG["hidden_dim"],       # 32
        heads=CFG["heads"],                 # 8
        relations=8,
        num_layers=CFG["num_layers"],       # 1
        max_route_len=CFG.get("max_route_len", 1052),
        num_experts=3,
        dropout=0.3,
        input_dropout=0.2,
    ).to(device_reptile)

    MetaTTE.model = reptile_base_model
    MetaTTE.device = device_reptile
    for k in cores:
        cores[k]["model"] = reptile_base_model   # shared model
    print(f"Created FRESH LinkETA for Reptile: {sum(p.numel() for p in reptile_base_model.parameters())/1e6:.2f}M params")
    print("theta0 randomly initialized (no checkpoint, no STMemory) - MetaTTE paper-faithful")

    # ---- Step 2: Reptile Meta-Training on source cities ----
    metatte_set_globals(cores[source_keys[0]])
    source_cores = [cores[k] for k in source_keys]
    source_names = [CITY_CONFIGS[k]["name"] for k in source_keys]
    # [2026-06-07] meta-training is SOURCES-only; no target needed

    reptile_result = meta_train_reptile(
        source_cores=source_cores,
        source_names=source_names,
        meta_iters=_mi(200),       # aligned: 1000 across all methods (notebook uses 200)
        inner_steps=_is(20),       # paper: k=10
        inner_lr=1e-3,             # inner Adam LR
        beta=0.1,                  # paper: beta=0.1 (MetaTTE Section V-B2)
        w_link=0.5,                # initial weights (DWA will update dynamically)
        w_route=0.5,
        use_adam_inner=True,       # MetaTTE uses Adam
        adam_betas=(0.0, 0.999),   # Reptile paper: beta1=0 (no momentum)
        dwa_T=2.0,                 # DWA temperature (same as MetaST)
        seed=SEED,
        log_every=10,
        eval_every=10,
        patience=10,               # generous patience for 200 iters
        enable_metrics=True,
        max_query_batches=5,       # aligned: partial val eval (same as MetaST/MetaTraffic)
        # [2026-06-07] theta0 is SOURCES-only -> sources-keyed run dir (no target)
        ckpt_dir=meta_ckpt_dir(ctx, "MetaTTE"),
        save_every=SAVE_EVERY,     # [2026-06-06] interval theta0 ckpts (0=off)
    )
    theta0_reptile = reptile_result["meta_state"]
    print(f"\nReptile meta-training complete. State dict keys: {len(theta0_reptile)}")

    # [2026-06-06] DECOUPLED: no transfer here. theta0 (+ training_history +
    # interval ckpts) saved by meta_train_reptile; adaptation happens in Evaluation.
    ctx["reptile_base_model"] = reptile_base_model
    ctx["device_reptile"] = device_reptile
    ctx["theta0_reptile"] = theta0_reptile
    return theta0_reptile


# =============================================================================
# MetaTraffic per-city data preparation (notebook "MetaTraffic Step 1")
# =============================================================================
def prepare_metatraffic_data(ctx):
    """Create the MetaTraffic + LinkETA models and assemble per-city tensors.

    Shared by the MetaTraffic training block and the MetaTraffic evaluation
    block. Bootstraps structural-clustering artifacts once if missing. Caches
    the result in ctx so it is built at most once per process.
    """
    if "mt_prepared" in ctx:
        return ctx["mt_prepared"]

    cores = ctx["cores"]
    source_keys = ctx["source_keys"]
    city_keys = ctx["city_keys"]

    import torch
    from MetaTraffic import (
        MetaTrafficModel,
        extract_traffic_profile_tensor,
    )
    from LinkETA import CFG, LinkETA

    device_mt = cores[source_keys[0]]["device"]

    # --- Create fresh MetaTraffic model (31 static features: 27 geojson + 4 topology) ---
    mt_model = MetaTrafficModel(
        static_dim=31,
        time_dim=8,
        meta_dim=CFG["hidden_dim"],           # 32
        traffic_dim=60,                       # 12 slots x 5 metrics
        num_relations=8,
        gat_heads=4,
        gat_layers=2,
        gat_dropout=0.1,
        generator_k=16,                       # richer per-link params
        refine_hidden=128,                    # non-linear MLP: 60->128->128->60
    ).to(device_mt)

    # --- Create fresh LinkETA for MetaTraffic pipeline (separate from cores) ---
    linketa_mt = LinkETA(
        in_dim=86,
        hidden_dim=CFG["hidden_dim"],
        heads=CFG["heads"],
        relations=8,
        num_layers=CFG["num_layers"],
        max_route_len=CFG.get("max_route_len", 1052),
        num_experts=3,
        dropout=0.3,
        input_dropout=0.2,
    ).to(device_mt)

    print(f"MetaTraffic params: {sum(p.numel() for p in mt_model.parameters()):,}")
    print(f"LinkETA (MT) params: {sum(p.numel() for p in linketa_mt.parameters()):,}")

    # --- Prepare per-city data dictionaries ---
    MT_CACHE_DIR = str(DATA_ROOT / "metatraffic_cache")
    os.makedirs(MT_CACHE_DIR, exist_ok=True)

    # Bootstrap: generate structural-clustering artifacts once if any are missing.
    _sc_required = ("static_feats_31.pt", "cluster_assignments.pt", "city_descriptor.pt")
    _sc_missing = [
        (ck, fname)
        for ck in city_keys
        for fname in _sc_required
        if not os.path.exists(os.path.join(str(city_dir(ck)), fname))
    ]
    if _sc_missing:
        print(f"\n[bootstrap] Missing structural-clustering artifacts: {_sc_missing}")
        print("[bootstrap] Running structural_clustering.main() to generate them ...")
        import structural_clustering
        structural_clustering.main()
        print("[bootstrap] Structural clustering complete.\n")
    else:
        print("\n[bootstrap] Structural-clustering artifacts present for all cities - skipping regeneration.\n")

    city_static_feats = {}
    city_traffic_real = {}
    city_rel_coo = {}
    city_scaler_mean = {}
    city_scaler_std = {}
    city_vff = {}
    city_dyn_idx = {}
    city_time_idx = {}
    city_clusters = {}       # cluster assignments per city
    city_descriptors = {}    # city-level descriptors for similarity weighting

    for ck in city_keys:
        core = cores[ck]
        tp_path = os.path.join(MT_CACHE_DIR, f"{ck}_traffic_profile.pt")

        # Static features [N, 31] from static_feats_31.pt (structural_clustering.py v2)
        sf_31_path = os.path.join(str(city_dir(ck)), "static_feats_31.pt")
        if os.path.exists(sf_31_path):
            sf_data = torch.load(sf_31_path, map_location=device_mt, weights_only=False)
            sf = sf_data['features']    # [N, 31]
            print(f"  {ck}: loaded static_feats_31  {sf.shape}  ({sf_data['feature_names'][:5]}...)")
        else:
            raise FileNotFoundError(
                f"  {ck}: static_feats_31.pt not found at {sf_31_path}. "
                f"Run structural_clustering.py first.")

        # Cluster assignments
        cl_path = os.path.join(str(city_dir(ck)), "cluster_assignments.pt")
        if os.path.exists(cl_path):
            cl_data = torch.load(cl_path, map_location="cpu", weights_only=False)
            city_clusters[ck] = cl_data
            print(f"  {ck}: loaded clusters K={cl_data['K']}, S={cl_data['S']}")
        else:
            raise FileNotFoundError(
                f"  {ck}: cluster_assignments.pt not found at {cl_path}. "
                f"Run structural_clustering.py first.")

        # City-level descriptor (for similarity weighting)
        desc_path = os.path.join(str(city_dir(ck)), "city_descriptor.pt")
        if os.path.exists(desc_path):
            desc_data = torch.load(desc_path, map_location="cpu", weights_only=False)
            city_descriptors[ck] = desc_data['descriptor']
            print(f"  {ck}: loaded city_descriptor ({len(desc_data['descriptor_names'])} dims)")
        else:
            print(f"  {ck}: city_descriptor.pt not found - run structural_clustering.py to generate")

        # Traffic profile [N, 7, 24, 60]  (day_of_week x hour_of_day x 60 dyn features)
        if os.path.exists(tp_path):
            tp = torch.load(tp_path, map_location=device_mt, weights_only=True)
            if tp.dim() != 4 or tp.shape[1] != 7 or tp.shape[2] != 24:
                print(f"  {ck}: stale cache shape {tp.shape}, re-extracting...")
                os.remove(tp_path)
                tp = None
            else:
                print(f"  {ck}: loaded traffic_profile from cache  {tp.shape}")
        else:
            tp = None

        if tp is None:
            profile_parquet = os.path.join(
                CITY_CONFIGS[ck]["precomputed_dir"], "link_traffic_profile.parquet")
            tp_df = pd.read_parquet(profile_parquet)
            tp = extract_traffic_profile_tensor(
                tp_df, core["graph"]["edges_vff"],
                dyn_prefixes=CFG["dyn_prefixes"], n_slots=12, device=device_mt)
            torch.save(tp.cpu(), tp_path)
            print(f"  {ck}: extracted & saved traffic_profile  {tp.shape}")

        city_static_feats[ck] = sf.to(device_mt)
        city_traffic_real[ck] = tp.to(device_mt)
        rel_coo_list = core["graph"]["rel_edges"]
        city_rel_coo[ck] = [r.to(device_mt) for r in rel_coo_list]
        city_scaler_mean[ck] = core["scalers"]["mean"]
        city_scaler_std[ck] = core["scalers"]["std"]
        city_vff[ck] = core["graph"]["vff_vec"]
        city_dyn_idx[ck] = core["scalers"]["dyn_idx"]
        city_time_idx[ck] = list(range(8))  # [0..7]

    # --- Initialize sigmoid output range from SOURCE city traffic profiles ---
    source_traffic = {k: city_traffic_real[k] for k in source_keys}
    mt_model.init_output_range(source_traffic, margin=0.1)

    # --- Initialize static normalization from SOURCE city static features ---
    source_static = {k: city_static_feats[k] for k in source_keys}
    mt_model.init_static_normalization(source_static)

    print(f"\nMetaTraffic model created, per-city data prepared (31 static features)")
    for ck in city_keys:
        print(f"  {ck}: static {city_static_feats[ck].shape}, "
              f"traffic {city_traffic_real[ck].shape}, "
              f"rel_coo {len(city_rel_coo[ck])} relations, "
              f"clusters K={city_clusters[ck]['K']} S={city_clusters[ck]['S']}, "
              f"descriptor={'present' if ck in city_descriptors else 'missing'}")

    prepared = dict(
        mt_model=mt_model, linketa_mt=linketa_mt,
        city_static_feats=city_static_feats, city_traffic_real=city_traffic_real,
        city_rel_coo=city_rel_coo, city_scaler_mean=city_scaler_mean,
        city_scaler_std=city_scaler_std, city_vff=city_vff,
        city_dyn_idx=city_dyn_idx, city_time_idx=city_time_idx,
        city_clusters=city_clusters, city_descriptors=city_descriptors,
    )
    ctx["mt_prepared"] = prepared
    return prepared


# =============================================================================
# Block: MetaTraffic  (notebook section "# MetaTraffic")
# =============================================================================
def run_metatraffic(ctx):
    cores = ctx["cores"]
    source_keys = ctx["source_keys"]
    target_keys = ctx["target_keys"]
    print(f"\n{'='*70}\n[BLOCK] MetaTraffic  sources={source_keys} targets={target_keys}\n{'='*70}")

    # [2026-06-06] DECOUPLED: training block trains + saves theta0 only.
    # Transfer/adaptation to targets happens in the separate Evaluation block.
    from MetaTraffic import train_metatraffic_source

    p = prepare_metatraffic_data(ctx)
    mt_model = p["mt_model"]
    linketa_mt = p["linketa_mt"]
    city_static_feats = p["city_static_feats"]
    city_traffic_real = p["city_traffic_real"]
    city_rel_coo = p["city_rel_coo"]
    city_scaler_mean = p["city_scaler_mean"]
    city_scaler_std = p["city_scaler_std"]
    city_vff = p["city_vff"]
    city_dyn_idx = p["city_dyn_idx"]
    city_time_idx = p["city_time_idx"]
    city_clusters = p["city_clusters"]
    city_descriptors = p["city_descriptors"]

    # ---- Step 2: MAML (FOMAML) meta-training on source cities ----
    source_cores = {k: cores[k] for k in source_keys}
    # [2026-06-07] meta-training is SOURCES-only; no target needed
    history_mt = train_metatraffic_source(
        source_cores=source_cores,
        meta_traffic_model=mt_model,
        linketa_model=linketa_mt,
        city_static_feats={k: city_static_feats[k] for k in source_keys},
        city_traffic_real={k: city_traffic_real[k] for k in source_keys},
        city_rel_coo={k: city_rel_coo[k] for k in source_keys},
        city_scaler_mean={k: city_scaler_mean[k] for k in source_keys},
        city_scaler_std={k: city_scaler_std[k] for k in source_keys},
        city_vff={k: city_vff[k] for k in source_keys},
        city_dyn_idx={k: city_dyn_idx[k] for k in source_keys},
        city_time_idx={k: city_time_idx[k] for k in source_keys},
        meta_iters=_mi(1000),         # generous budget - early stopping finds convergence
        inner_steps=_is(10),          # K=10 inner SGD steps (support)
        inner_lr=1e-3,                # inner-loop SGD LR (conservative for cascaded models)
        outer_lr=2e-4,                # Adam meta-optimizer LR (v3.2)
        lambda_traffic=0.5,           # traffic prediction weight
        patience=50,                  # auto-stops when converged
        log_every=10,
        ckpt_dir=meta_ckpt_dir(ctx, "MetaETA"),   # [2026-06-07] sources-only theta0
        dwa_T=2.0,                    # DWA temperature for link/route balancing
        max_query_batches=5,          # limit query batches for speed
        clip_inner=10.0,              # gradient clipping inside inner SGD loop
        seed=SEED,
        enable_metrics=True,
        save_every=SAVE_EVERY,        # [2026-06-06] interval theta0 ckpts (0=off)
    )

    # [2026-06-06] DECOUPLED: no transfer here. theta0 (+ training_history +
    # interval ckpts) is saved by train_metatraffic_source; adaptation to each
    # target happens in the separate Evaluation/transfer block.
    ctx["history_mt"] = history_mt
    return history_mt


# =============================================================================
# Block: Evaluation  (notebook section "# New Evaluation")
# =============================================================================
def _aggregate_city_reps(per_rep):
    """[2026-06-07] Aggregate N transfer reps for ONE city. The per-trip/link
    artifacts come from the BEST-val rep (lowest val selection metric); every
    metric is reported as mean + sample-std (ddof=1) across the reps via parallel
    `<metric>_std` keys. Used when --transfer-reps > 1."""
    import math as _m
    import numpy as _np
    import pandas as _pd
    from loss_functions import selection_metric
    per_rep = [r for r in per_rep if r]
    if not per_rep:
        return {}

    def _selm(r):
        v = r.get("val", {}) or {}
        return selection_metric(v.get("route_mape"), v.get("link_mape"))

    best = min(per_rep, key=_selm)

    # ---- per-trip / per-link prediction std ACROSS reps (val/test loaders are
    # shuffle=False -> same trips/order each rep, so we can align by keys) ----
    # trip_df: primary y_pred from best rep + y_pred_std_reps over all reps (by TRIP_ID/split)
    best_trip = best.get("trip_df")
    if isinstance(best_trip, _pd.DataFrame) and not best_trip.empty:
        best_trip = best_trip.copy()
        tf = [r["trip_df"] for r in per_rep
              if isinstance(r.get("trip_df"), _pd.DataFrame) and not r["trip_df"].empty]
        if len(tf) > 1 and {"split", "TRIP_ID", "y_pred"}.issubset(best_trip.columns):
            allt = _pd.concat(tf, ignore_index=True)
            g = (allt.groupby(["split", "TRIP_ID"])["y_pred"].std(ddof=1)
                 .reset_index().rename(columns={"y_pred": "y_pred_std_reps"}))
            best_trip = best_trip.merge(g, on=["split", "TRIP_ID"], how="left")
        else:
            best_trip["y_pred_std_reps"] = float("nan")
    # link_df: primary pred_tt_link from best rep + pred_tt_link_std_reps over reps
    best_link = best.get("link_df")
    if isinstance(best_link, _pd.DataFrame) and not best_link.empty:
        best_link = best_link.copy()
        lf = [r["link_df"] for r in per_rep
              if isinstance(r.get("link_df"), _pd.DataFrame) and not r["link_df"].empty]
        if len(lf) > 1 and {"split", "TRIP_ID", "pred_tt_link"}.issubset(best_link.columns):
            def _with_idx(df):
                df = df.copy()
                df["_lidx"] = df.groupby(["split", "TRIP_ID"]).cumcount()
                return df
            best_link = _with_idx(best_link)
            allk = _pd.concat([_with_idx(f) for f in lf], ignore_index=True)
            g = (allk.groupby(["split", "TRIP_ID", "_lidx"])["pred_tt_link"].std(ddof=1)
                 .reset_index().rename(columns={"pred_tt_link": "pred_tt_link_std_reps"}))
            best_link = (best_link.merge(g, on=["split", "TRIP_ID", "_lidx"], how="left")
                         .drop(columns=["_lidx"]))
        else:
            best_link["pred_tt_link_std_reps"] = float("nan")

    out = {"trip_df": best_trip, "link_df": best_link, "n_reps": len(per_rep)}
    for split in ("val", "test"):
        dicts = [(r.get(split) or {}) for r in per_rep]
        keys = set().union(*[set(d.keys()) for d in dicts]) if dicts else set()
        agg = {}
        for k in sorted(keys):
            vals = [float(d[k]) for d in dicts
                    if isinstance(d.get(k), (int, float)) and not _m.isnan(float(d[k]))]
            if vals:
                agg[k] = float(_np.mean(vals))
                agg[k + "_std"] = float(_np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        out[split] = agg
    return out


def _run_transfer_reps(fn, city_name, base_transfer_ckpt_dir, **kwargs):
    """[2026-06-07] Run a transfer_to_target_* function TRANSFER_REPS times from the
    same (fixed) meta theta0, with a fresh seed per rep (varies the adaptation
    stochasticity), and aggregate to mean ± std. reps==1 -> single call, unchanged.
    Each rep writes its adapted ckpt to <base>_rep{k}; the best-val rep's artifacts
    are the ones surfaced downstream (figures/significance)."""
    reps = max(1, int(TRANSFER_REPS))
    if reps <= 1:
        return fn(transfer_ckpt_dir=base_transfer_ckpt_dir, **kwargs)
    import torch as _t
    import random as _r
    try:
        import numpy as _np
        _has_np = True
    except Exception:
        _has_np = False
    per_rep = []
    for rep in range(reps):
        _t.manual_seed(SEED + rep)
        _r.seed(SEED + rep)
        if _has_np:
            _np.random.seed(SEED + rep)
        print(f"    [transfer rep {rep + 1}/{reps}] seed={SEED + rep}")
        res = fn(transfer_ckpt_dir=f"{base_transfer_ckpt_dir}_rep{rep}", **kwargs)
        per_rep.append(res.get(city_name, {}))
    # Copy the BEST-val rep's adapted ckpt to the canonical transfer dir so
    # benchmark.py / re-testing (which load the no-suffix dir) use the best rep.
    from loss_functions import selection_metric as _selm_fn
    import shutil as _sh
    import os as _os

    def _selm(i):
        v = per_rep[i].get("val", {}) or {}
        return _selm_fn(v.get("route_mape"), v.get("link_mape"))
    if per_rep:
        best_idx = min(range(len(per_rep)), key=_selm)
        _src = f"{base_transfer_ckpt_dir}_rep{best_idx}"
        try:
            if _os.path.isdir(_src):
                if _os.path.isdir(base_transfer_ckpt_dir):
                    _sh.rmtree(base_transfer_ckpt_dir)
                _sh.copytree(_src, base_transfer_ckpt_dir)
                print(f"    [transfer reps] best-val rep = {best_idx}; "
                      f"adapted ckpt -> {base_transfer_ckpt_dir}")
        except Exception as _e:
            print(f"    [transfer reps] WARN: could not promote best ckpt ({_e})")
    print(f"    [transfer reps] aggregated mean±std over {reps} reps "
          f"(artifacts from best-val rep)")
    return {city_name: _aggregate_city_reps(per_rep)}


def run_evaluation(ctx):
    """[2026-06-06 DECOUPLED] Transfer + score block. Loads each method's
    meta-trained theta0, adapts to each held-out TARGET city with the correct
    log-space backbone losses + aligned budget, saves the ADAPTED checkpoint
    (for benchmark/testing), collects per-trip/per-link predictions, and writes
    the LinkETA-style outputs (eval_metrics.json / eval_trips.csv /
    eval_links.csv) under checkpoints/<run>/<city>/eval_outputs/.

    Scored on TARGET cities ONLY (LOO removed — it was not consistent across the
    three methods). Source-city numbers are in-sample diagnostics reported from
    each trainer's training_history.csv (best iter by route_mape).
    """
    cores = ctx["cores"]
    source_keys = ctx["source_keys"]
    target_keys = ctx["target_keys"]
    city_keys = ctx["city_keys"]
    print(f"\n{'='*70}\n[BLOCK] Transfer+Score  targets={target_keys}\n{'='*70}")

    from evaluator import (
        transfer_to_target_metast,
        transfer_to_target_metatte,
        transfer_to_target_metatraffic,
    )
    from LinkETA import LinkETA, CFG

    # [2026-06-07] theta0 ckpts live at the SOURCES-keyed run dir (target-independent);
    # the same theta0 is loaded and adapted for every target scored.

    # === Setup: reconstruct architectures if not in-memory =================
    # [2026-06-07] --methods filter: only set up + transfer the selected methods,
    # so e.g. MetaETA can be scored without requiring MetaST/MetaTTE theta0 ckpts.
    methods = ctx.get("eval_methods") or list(EVAL_METHODS_ALL)
    print(f"[EVAL] methods: {methods}")

    import MetaST as MetaST
    from MetaST import create_metast_model
    import MetaTTE as MetaTTE

    wrapped_model = None
    metast_ckpt = None
    if "MetaST" in methods:
        wrapped_model = ctx.get("wrapped_model")
        if wrapped_model is None:
            _dev = cores[source_keys[0]]["device"]
            _base = LinkETA(
                in_dim=IN_DIM, hidden_dim=CFG["hidden_dim"], heads=CFG["heads"],
                relations=8, num_layers=CFG["num_layers"],
                max_route_len=CFG.get("max_route_len", 1052),
                num_experts=3, dropout=0.3, input_dropout=0.2,
            ).to(_dev)
            wrapped_model = create_metast_model(_base, G=4, d_mem=8, device=_dev)
            print("Created fresh wrapped_model (weights loaded from checkpoint)")
        MetaST.model = wrapped_model
        metast_ckpt = meta_ckpt_dir(ctx, "MetaST")    # [2026-06-07] sources-only theta0

    reptile_base_model = None
    device_reptile = None
    metatte_ckpt = None
    if "MetaTTE" in methods:
        reptile_base_model = ctx.get("reptile_base_model")
        device_reptile = ctx.get("device_reptile")
        if reptile_base_model is None:
            device_reptile = cores[source_keys[0]]["device"]
            reptile_base_model = LinkETA(
                in_dim=IN_DIM, hidden_dim=CFG["hidden_dim"], heads=CFG["heads"],
                relations=8, num_layers=CFG["num_layers"],
                max_route_len=CFG.get("max_route_len", 1052),
                num_experts=3, dropout=0.3, input_dropout=0.2,
            ).to(device_reptile)
            print("Created fresh reptile_base_model (weights loaded from checkpoint)")
        MetaTTE.model = reptile_base_model
        MetaTTE.device = device_reptile
        metatte_ckpt = meta_ckpt_dir(ctx, "MetaTTE")

    # MetaTraffic data + models (cached on ctx; built once per process)
    mt_model = linketa_mt = None
    city_static_feats = city_traffic_real = city_rel_coo = None
    city_scaler_mean = city_scaler_std = city_vff = None
    city_dyn_idx = city_time_idx = city_clusters = city_descriptors = None
    metaeta_ckpt = None
    all_sources = []
    do_mt = False
    if "MetaETA" in methods:
        p = prepare_metatraffic_data(ctx)
        mt_model = p["mt_model"]
        linketa_mt = p["linketa_mt"]
        city_static_feats = p["city_static_feats"]
        city_traffic_real = p["city_traffic_real"]
        city_rel_coo = p["city_rel_coo"]
        city_scaler_mean = p["city_scaler_mean"]
        city_scaler_std = p["city_scaler_std"]
        city_vff = p["city_vff"]
        city_dyn_idx = p["city_dyn_idx"]
        city_time_idx = p["city_time_idx"]
        city_clusters = p["city_clusters"]
        city_descriptors = p["city_descriptors"]
        metaeta_ckpt = meta_ckpt_dir(ctx, "MetaETA")

        # [2026-06-06] LOO removed: targets are held out, so the prototype bank uses
        # ALL source cities. Needs >=1 usable source.
        all_sources = [k for k in source_keys
                       if k in city_clusters and k in city_static_feats and k in city_rel_coo]
        do_mt = len(all_sources) >= 1
        if not do_mt:
            print(f"[EVAL] MetaETA needs >=1 usable source city (have {all_sources}); "
                  f"MetaETA will be skipped.")

    # === Per-city loop: run all three methods, concat, save ================
    results_metast_eval   = {}
    results_metatte_eval  = {}
    results_mt_eval_loo   = {}
    summary_rows = []

    for ck in target_keys:        # [2026-06-06] TARGETS ONLY (held-out)
        cfg = CITY_CONFIGS[ck]
        city_name = cfg["name"]
        out_dir = city_eval_outputs_dir(ctx, ck)
        print(f"\n{'#'*90}\n# Transfer+score {city_name} ({ck})  ->  {out_dir}\n{'#'*90}")

        ms_city, mtte_city, mtr_city = {}, {}, {}

        # ---- MetaST (aligned budget: 500 steps / patience 10 / eval 10; SGD @ 1e-3) ----
        if "MetaST" in methods:
            res_ms = _run_transfer_reps(
                transfer_to_target_metast, city_name,
                transfer_ckpt_dir(ctx, "MetaST", ck),
                cores=cores,
                city_configs={ck: cfg},
                wrapped_model=wrapped_model,
                ckpt_dir=metast_ckpt,
                adapt_steps=_as(500),
                adapt_lr=1e-3,
                patience=10,
                eval_every=10,
            )
            results_metast_eval.update(res_ms)
            ms_city = res_ms.get(city_name, {})

        # ---- MetaTTE (aligned budget; Reptile-faithful Adam @ 1e-5) ----
        if "MetaTTE" in methods:
            res_mtte = _run_transfer_reps(
                transfer_to_target_metatte, city_name,
                transfer_ckpt_dir(ctx, "MetaTTE", ck),
                cores=cores,
                city_configs={ck: cfg},
                base_model=reptile_base_model,
                ckpt_dir=metatte_ckpt,
                adapt_steps=_as(500),
                adapt_lr=1e-5,
                patience=10,
                eval_every=10,
                use_adam=True,
                adam_betas=(0.0, 0.999),
            )
            results_metatte_eval.update(res_mtte)
            mtte_city = res_mtte.get(city_name, {})

        # ---- MetaETA (prototype bank from ALL sources; aligned budget; SGD @ 1e-3) ----
        if do_mt:
            print(f"\n[MetaETA] target={ck} ({city_name}) | sources={all_sources}")
            res_mtr = _run_transfer_reps(
                transfer_to_target_metatraffic, city_name,
                transfer_ckpt_dir(ctx, "MetaETA", ck),
                cores=cores,
                city_configs={ck: cfg},
                meta_traffic_model=mt_model,
                linketa_model=linketa_mt,
                city_static_feats=city_static_feats,
                city_traffic_real=city_traffic_real,
                city_rel_coo=city_rel_coo,
                city_scaler_mean=city_scaler_mean,
                city_scaler_std=city_scaler_std,
                city_vff=city_vff,
                city_dyn_idx=city_dyn_idx,
                city_time_idx=city_time_idx,
                ckpt_dir=metaeta_ckpt,
                # [2026-06-09] adapt_lr now matches meta-training inner_lr (1e-3).
                # The earlier 1e-3 instability came from the traffic head being
                # clipped to the SOURCE range; transfer_to_target_metatraffic now
                # re-calibrates output_scale/shift to the TARGET range
                # (recalibrate_output_range=True), so the head starts at the right
                # scale and 1e-3 is stable. If the backbone needs gentler steps,
                # set adapt_lr_backbone (per-group LR) below.
                adapt_steps=_as(60),
                adapt_lr=1e-3,
                # adapt_lr_backbone=1e-4,    # uncomment to protect the LinkETA backbone
                # [2026-06-09] adapt_lr_generator: per-group LR for the traffic
                # generator (param_gen/GAT/refine/output head/alpha gate) — the
                # transfer bottleneck. None -> = adapt_lr (uniform, matches
                # meta-training). Raise it to let predicted traffic + the alpha gate
                # specialise to the target faster; tune per experiment.
                # adapt_lr_generator=5e-3,
                recalibrate_output_range=True,
                # [2026-06-09] alpha_init_logit=0.0 -> the prototype<->W_gen gate is
                # LEARNED per target from a neutral prior (alpha=0.5) instead of the
                # unlearned, indefensible 0.88 from theta0. Set None to keep 0.88.
                alpha_init_logit=0.0,
                patience=10,
                dwa_eval_every=10,
                source_static_feats={k: city_static_feats[k] for k in all_sources},
                source_rel_coo={k: city_rel_coo[k] for k in all_sources},
                source_clusters={k: city_clusters[k] for k in all_sources},
                city_clusters=city_clusters,
                source_city_descriptors={k: city_descriptors[k]
                                         for k in all_sources if k in city_descriptors} or None,
                city_descriptors=city_descriptors,
                K=city_clusters[ck]["K"] if ck in city_clusters else 6,
                S=city_clusters[ck]["S"] if ck in city_clusters else 14,
                disable_prototype=NO_PROTOTYPE,   # [2026-06-07] --no-prototype diagnostic
            )
            mtr_city = res_mtr.get(city_name, {})
            results_mt_eval_loo[city_name] = mtr_city

        # ---- Concat per-method dfs and save LinkETA-style files ----
        per_method = []
        if "MetaST" in methods:
            per_method.append(("MetaST", ms_city))
        if "MetaTTE" in methods:
            per_method.append(("MetaTTE", mtte_city))
        if mtr_city:
            per_method.append(("MetaETA", mtr_city))

        trip_frames, link_frames, metrics_dict = [], [], {}
        for method_name, mres in per_method:
            t = mres.get("trip_df")
            l = mres.get("link_df")
            if isinstance(t, pd.DataFrame) and not t.empty:
                trip_frames.append(t)
            if isinstance(l, pd.DataFrame) and not l.empty:
                link_frames.append(l)
            metrics_dict[method_name] = {
                "val":  {k: (float(v) if isinstance(v, (int, float)) else v)
                         for k, v in mres.get("val",  {}).items()},
                "test": {k: (float(v) if isinstance(v, (int, float)) else v)
                         for k, v in mres.get("test", {}).items()},
            }

        if trip_frames:
            df_trips = pd.concat(trip_frames, ignore_index=True)
            df_trips["abs_err_s"]  = (df_trips["y_pred"] - df_trips["y_true"]).abs()
            df_trips["model_trip"] = (df_trips["model"].astype(str) + "::"
                                      + df_trips["TRIP_ID"].astype(str))
            # [2026-06-07] keep schema consistent: per-trip pred std across reps
            # (NaN when --transfer-reps==1).
            if "y_pred_std_reps" not in df_trips.columns:
                df_trips["y_pred_std_reps"] = float("nan")
        else:
            df_trips = pd.DataFrame()

        if link_frames:
            df_links = pd.concat(link_frames, ignore_index=True)
            df_links["abs_err_s"] = (df_links["pred_tt_link"] - df_links["true_tt_link"]).abs()
            if "pred_tt_link_std_reps" not in df_links.columns:
                df_links["pred_tt_link_std_reps"] = float("nan")
        else:
            df_links = pd.DataFrame()

        combined_metrics = {
            "city":            ck,
            "city_name":       city_name,
            "role":            cfg.get("role", "unknown"),
            "run":             run_folder_name(source_keys, target_keys),
            "models_evaluated": [m for m, _ in per_method],
            **metrics_dict,
        }
        json_path  = os.path.join(out_dir, "eval_metrics.json")
        trips_path = os.path.join(out_dir, "eval_trips.csv")
        links_path = os.path.join(out_dir, "eval_links.csv")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(combined_metrics, f, indent=2, default=str)
        df_trips.to_csv(trips_path, index=False)
        df_links.to_csv(links_path, index=False)
        print(f"[saved {ck}] metrics -> {json_path}")
        print(f"[saved {ck}] trips   -> {trips_path}  ({len(df_trips):,} rows)")
        print(f"[saved {ck}] links   -> {links_path}  ({len(df_links):,} rows)")

        # Aggregate one-row-per-(city,method) summary for the run-level table.
        for method_name, mres in per_method:
            t = mres.get("test", {}) or {}
            summary_rows.append({
                "city":         ck,
                "model":        method_name,
                "n_reps":       mres.get("n_reps", 1),                 # [2026-06-07]
                "link_mae":     t.get("link_mae",      float("nan")),
                "link_mae_std": t.get("link_mae_std",  float("nan")),  # [2026-06-07] std over reps
                "link_wape":    t.get("link_wape",     float("nan")),  # [2026-06-07] robust link %
                "link_wape_std":t.get("link_wape_std", float("nan")),
                "link_mdape":   t.get("link_mdape",    float("nan")),
                "link_p90":     t.get("link_p90",      float("nan")),
                "link_mape":    t.get("link_mape",     float("nan")),  # legacy (inflated by micro-links)
                "route_mae":    t.get("route_mae",     float("nan")),
                "route_mae_std":t.get("route_mae_std", float("nan")),
                "route_wape":   t.get("route_wape",    float("nan")),
                "route_mape":   t.get("route_mape",    float("nan")),
                "route_mape_std":t.get("route_mape_std",float("nan")),
                "traffic_rmse": t.get("traffic_rmse",  float("nan")),
            })

    if summary_rows:
        _summary_df = pd.DataFrame(summary_rows)
        print("\n" + "=" * 90)
        print("Evaluation summary (test split)")
        print(_summary_df.to_string(index=False))
        # [2026-06-07] Persist the run-level results table (mean + *_std over reps)
        # so std reporting lives on disk too, consistent with eval_metrics.json.
        _summary_path = os.path.join(run_dir(ctx), "results_summary.csv")
        _summary_df.to_csv(_summary_path, index=False)
        print(f"[saved] run results summary -> {_summary_path}")

    ctx["results_metast_eval"]  = results_metast_eval
    ctx["results_metatte_eval"] = results_metatte_eval
    ctx["results_mt_eval_loo"]  = results_mt_eval_loo
    return results_metast_eval, results_metatte_eval, results_mt_eval_loo


BLOCKS = {
    "MetaST":      run_metast,
    "MetaTTE":     run_metatte,
    "MetaTraffic": run_metatraffic,
    "Evaluation":  run_evaluation,
}

# [2026-06-07] Transfer methods scored inside the Evaluation block. Selectable
# via --methods so a single method (e.g. MetaETA) can be scored on its own.
EVAL_METHODS_ALL = ["MetaST", "MetaTTE", "MetaETA"]


# --- Selection logic -----------------------------------------------------
def _pick_from_menu(title, options, allow_multi=False):
    print(f"\n{title}")
    for i, name in enumerate(options, 1):
        print(f"  {i}) {name}")
    if allow_multi:
        print("  a) all")
    prompt = "Select (comma-separated for multiple): " if allow_multi else "Select: "
    while True:
        raw = input(prompt).strip()
        if not raw:
            continue
        if allow_multi and raw.lower() in ("a", "all"):
            return list(options)
        try:
            idxs = [int(x) - 1 for x in raw.split(",")] if allow_multi else [int(raw) - 1]
            picked = [options[i] for i in idxs]
            return picked if allow_multi else picked[0]
        except (ValueError, IndexError):
            print("Invalid choice, try again.")


def _resolve_blocks(arg):
    if arg.lower() == "all":
        return list(BLOCKS)
    lookup = {k.lower(): k for k in BLOCKS}
    out = []
    for n in (x.strip() for x in arg.split(",") if x.strip()):
        if n.lower() not in lookup:
            raise ValueError(f"Unknown block '{n}'. Choose from: {list(BLOCKS)}")
        out.append(lookup[n.lower()])
    return out


def _resolve_methods(arg):
    """Parse --methods into a subset of EVAL_METHODS_ALL (Evaluation block only)."""
    if not arg or arg.lower() == "all":
        return list(EVAL_METHODS_ALL)
    lookup = {m.lower(): m for m in EVAL_METHODS_ALL}
    out = []
    for n in (x.strip() for x in arg.split(",") if x.strip()):
        if n.lower() not in lookup:
            raise ValueError(f"Unknown method '{n}'. Choose from: {EVAL_METHODS_ALL}")
        out.append(lookup[n.lower()])
    return out


def _resolve_cities(arg):
    """Parse a comma-separated list of city keys against CITY_CONFIGS."""
    lookup = {k.lower(): k for k in CITY_CONFIGS}
    out = []
    for raw in (x.strip() for x in arg.split(",") if x.strip()):
        if raw.lower() not in lookup:
            raise ValueError(
                f"Unknown city '{raw}'. Choose from: {list(CITY_CONFIGS)}.")
        out.append(lookup[raw.lower()])
    return out


def _resolve_trips(arg):
    """Parse --trips into {city_key: n}. Accepts either a single int applied to
    every city, or a comma-separated 'city=n' list (e.g. 'porto=1000,chengdu=500').
    """
    overrides = {}
    raw = arg.strip()
    if not raw:
        return overrides
    if "=" not in raw:
        n = int(raw)
        return {k: n for k in CITY_CONFIGS}
    lookup = {k.lower(): k for k in CITY_CONFIGS}
    for part in (x.strip() for x in raw.split(",") if x.strip()):
        key, _, val = part.partition("=")
        key = key.strip().lower()
        if key not in lookup:
            raise ValueError(f"Unknown city '{key}' in --trips. Choose from: {list(CITY_CONFIGS)}.")
        overrides[lookup[key]] = int(val.strip())
    return overrides


def parse_args():
    p = argparse.ArgumentParser(description="Meta_Learning experiment runner")
    p.add_argument("--blocks",
                   help="Comma-separated: MetaST,MetaTTE,MetaTraffic,Evaluation (or 'all')")
    p.add_argument("--methods",
                   help="(Evaluation block only) Comma-separated subset of transfer "
                        "methods to score: MetaST,MetaTTE,MetaETA (or 'all'). "
                        "Default: all three. Lets you score one method (e.g. MetaETA) "
                        "without needing the others' meta-trained theta0 checkpoints.")
    p.add_argument("--sources",
                   help=f"Comma-separated source city keys for meta-training. "
                        f"Choices: {', '.join(CITY_CONFIGS)}")
    p.add_argument("--targets",
                   help=f"Comma-separated target city keys for transfer. "
                        f"Choices: {', '.join(CITY_CONFIGS)}")
    p.add_argument("--trips", default="",
                   help="Per-city TRIP_ID sample sizes. Either a single int "
                        "applied to all cities, or 'city=n' list "
                        "(e.g. 'porto=1000,sf=1000,chengdu=500'). Omit to use "
                        "the per-city defaults in CITY_CONFIGS.")
    p.add_argument("--meta-iters", type=int, default=None,
                   help="Override meta-training iterations for all selected methods "
                        "(default: each method's notebook value).")
    p.add_argument("--inner-steps", type=int, default=None,
                   help="Override inner-loop adaptation steps for all selected methods "
                        "(default: each method's notebook value).")
    p.add_argument("--adapt-steps", type=int, default=None,
                   help="Override target fine-tuning steps during transfer for all "
                        "selected methods (default: each method's notebook value).")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed passed to meta-training (default: 42).")
    p.add_argument("--save-every", type=int, default=5,
                   help="Interval meta-checkpoint cadence: save theta0_iter{it}.ckpt "
                        "every N meta-iters (0 disables; best/last always saved). "
                        "Default: 5.")
    p.add_argument("--cache-datasets", action="store_true",
                   help="Cache off-route-neighbour trip groups to "
                        "meta_learning_data/<city>/dataset_cache/ and reuse on "
                        "later runs (cache miss/any error => build as usual).")
    p.add_argument("--transfer-reps", type=int, default=1,
                   help="Repeat the transfer/adaptation N times (from the fixed "
                        "meta theta0, seed=SEED+rep) and report mean +/- std over "
                        "the reps in the eval table + eval_metrics.json. Per-trip "
                        "CSVs + adapted ckpt come from the best-val rep. Default 1 "
                        "(single run, unchanged).")
    p.add_argument("--no-prototype", action="store_true",
                   help="MetaETA diagnostic: skip the structural cluster prototype "
                        "bank at transfer (pure W_gen, alpha=0). Isolates whether "
                        "the prototype injection is what breaks target transfer.")
    return p.parse_args()


def main():
    global META_ITERS_OVERRIDE, INNER_STEPS_OVERRIDE, ADAPT_STEPS_OVERRIDE, SEED, SAVE_EVERY, TRANSFER_REPS, NO_PROTOTYPE

    # Run from the script dir so the notebook's relative paths (./meta_learning_data,
    # data/, ckpts_*, figs/) resolve exactly as they did in main.ipynb.
    os.chdir(BASE_DIR)

    args = parse_args()
    META_ITERS_OVERRIDE = args.meta_iters
    INNER_STEPS_OVERRIDE = args.inner_steps
    ADAPT_STEPS_OVERRIDE = args.adapt_steps
    SEED = args.seed
    SAVE_EVERY = args.save_every
    TRANSFER_REPS = args.transfer_reps
    NO_PROTOTYPE = args.no_prototype

    blocks = _resolve_blocks(args.blocks) if args.blocks else _pick_from_menu(
        "Which block(s) to run?", list(BLOCKS), allow_multi=True
    )
    sources = _resolve_cities(args.sources) if args.sources else _pick_from_menu(
        "Which SOURCE city(ies)? (meta-training)", list(CITY_CONFIGS), allow_multi=True
    )
    # [2026-06-07] meta-training is SOURCES-only (theta0 is target-independent), so
    # --targets is REQUIRED only for the Evaluation (transfer) block.
    needs_target = "Evaluation" in blocks
    if args.targets:
        targets = _resolve_cities(args.targets)
    elif needs_target:
        targets = _pick_from_menu(
            "Which TARGET city(ies)? (transfer)", list(CITY_CONFIGS), allow_multi=True)
    else:
        targets = []
    eval_methods = _resolve_methods(args.methods)
    trip_overrides = _resolve_trips(args.trips)

    if not sources:
        raise SystemExit("No source cities selected.")
    if needs_target and not targets:
        raise SystemExit("The Evaluation (transfer) block requires --targets.")

    # Apply CLI roles + trip counts onto CITY_CONFIGS (data paths stay as declared).
    for k in CITY_CONFIGS:
        if k in sources:
            CITY_CONFIGS[k]["role"] = "source"
        elif k in targets:
            CITY_CONFIGS[k]["role"] = "target"
    for k, n in trip_overrides.items():
        CITY_CONFIGS[k]["sample_trips"] = n

    # Active cities = sources + targets (sources first, preserving config order).
    source_keys = [k for k in CITY_CONFIGS if k in sources]
    target_keys = [k for k in CITY_CONFIGS if k in targets and k not in sources]
    city_keys = source_keys + target_keys

    print(f"\n[PLAN] Blocks: {blocks}")
    if "Evaluation" in blocks:
        print(f"[PLAN] Eval methods: {eval_methods}")
    print(f"[PLAN] Sources: {source_keys} | Targets: {target_keys}")
    # [2026-06-07] theta0 (sources-only) run dir — pass this as --run to
    # benchmark.py / significance_test.py / figures.py.
    print(f"[PLAN] Run dir (theta0): checkpoints/{run_folder_name(source_keys)}")
    print(f"[PLAN] Trips: " + ", ".join(
        f"{k}={CITY_CONFIGS[k]['sample_trips']}" for k in city_keys))
    if META_ITERS_OVERRIDE is not None:
        print(f"[PLAN] meta_iters override: {META_ITERS_OVERRIDE}")
    if INNER_STEPS_OVERRIDE is not None:
        print(f"[PLAN] inner_steps override: {INNER_STEPS_OVERRIDE}")
    if ADAPT_STEPS_OVERRIDE is not None:
        print(f"[PLAN] adapt_steps override: {ADAPT_STEPS_OVERRIDE}")
    print(f"[BASE] {BASE_DIR}")

    # Build cores once for all active cities, then run the selected blocks.
    cores = build_cores(city_keys, cache_datasets=args.cache_datasets)
    ctx = {
        "cores": cores,
        "source_keys": source_keys,
        "target_keys": target_keys,
        "city_keys": city_keys,
        "eval_methods": eval_methods,
    }
    for name in blocks:
        BLOCKS[name](ctx)

    print("\n[DONE]")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[ABORTED] interrupted by user.")
        sys.exit(130)
