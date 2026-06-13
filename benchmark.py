#!/usr/bin/env python
# coding: utf-8
"""
benchmark.py — pure-forward inference latency for MetaST / MetaTTE / MetaTraffic.

Mirrors LinkETA/benchmark.py and ports the latency cells from main.ipynb (the
archived `benchmark_linketa_style` + `benchmark_metatraffic_pipeline`
functions): per-batch ms (mean / p50 / p90), per-trip ms, per-link ms,
trips-per-second; optionally stratified by link-count bin; optional scalability
sweep across fractions of the test loader.

Outputs (under checkpoints/<run>/<city>/eval_outputs/):
    benchmark_summary.csv         one row per (model, city, rep)
    benchmark_by_link_count.csv   one row per (model, city, link-count bin)
    benchmark_scalability.csv     (with --scalability) one row per
                                  (model, city, fraction, rep)

Run:
    python benchmark.py --run porto_sf_source_chengdu_target
    python benchmark.py --run porto_sf_source_chengdu_target --cities chengdu --reps 3
    python benchmark.py --run porto_sf_source_chengdu_target --scalability
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Reuse the orchestrator's path helpers + core-build pipeline so we don't
# reimplement city/checkpoint plumbing here.
from main import (
    BASE_DIR, CHECKPOINTS_ROOT, CITY_CONFIGS, IN_DIM,
    build_cores, prepare_metatraffic_data,
)


def _transfer_dir(run_name, city, method):
    """[2026-06-07] Per-target ADAPTED ckpt dir under the (literal) run folder:
    checkpoints/<run>/<city>/ckpts_<method>_transfer/. Uses run_name verbatim so
    the trips/seed tag baked into the folder name is honoured (no recomputation)."""
    return str(CHECKPOINTS_ROOT / run_name / city / f"ckpts_{method}_transfer")


def _eval_out_dir(run_name, city):
    d = CHECKPOINTS_ROOT / run_name / city / "eval_outputs"
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


# --- Bin / sweep configuration -------------------------------------------
LC_BINS   = (0, 5, 10, 20, 40, 80, 1e9)
LC_LABELS = ["1-4", "5-9", "10-19", "20-39", "40-79", "80+"]
SCAL_FRACS = (0.2, 0.4, 0.6, 0.8, 1.0)


def parse_run_folder(name: str):
    """[2026-06-07] '<sources>_source_<trips/seed tag>' -> source_keys.
    Target-independent (theta0 is sources-only); benchmark targets come from
    --cities or the city subfolders present. Tolerates the legacy
    '<sources>_source_<targets>_target' by taking the head before '_source'."""
    if "_source" not in name:
        raise ValueError(
            f"Bad run folder name (expected '<sources>_source_...'): {name}")
    head = name.split("_source", 1)[0]
    return head.split("_")


def _sync(dev):
    if isinstance(dev, torch.device) and dev.type == "cuda":
        torch.cuda.synchronize(dev)


def _summary(batch_ms, n_trips, n_links, model, city, split):
    """Aggregate per-batch timings into a single row (matching the notebook's
    _summary_from_batch_times schema, plus city/split + per-link metric)."""
    arr = np.asarray(batch_ms, dtype=float)
    if arr.size == 0:
        return {
            "model": model, "city": city, "split": split,
            "n_batches": 0, "n_trips": 0, "n_links": 0,
            "total_s": float("nan"),
            "mean_ms_per_batch": float("nan"),
            "p50_ms_per_batch":  float("nan"),
            "p90_ms_per_batch":  float("nan"),
            "mean_ms_per_trip":  float("nan"),
            "mean_ms_per_link":  float("nan"),
            "trips_per_second":  float("nan"),
        }
    total_s = float(arr.sum() / 1000.0)
    return {
        "model": model, "city": city, "split": split,
        "n_batches": int(arr.size),
        "n_trips":   int(n_trips),
        "n_links":   int(n_links),
        "total_s":   total_s,
        "mean_ms_per_batch": float(arr.mean()),
        "p50_ms_per_batch":  float(np.quantile(arr, 0.50)),
        "p90_ms_per_batch":  float(np.quantile(arr, 0.90)),
        "mean_ms_per_trip":  float(arr.sum() / max(n_trips, 1)),
        "mean_ms_per_link":  float(arr.sum() / max(n_links, 1)) if n_links else float("nan"),
        "trips_per_second":  float(n_trips / max(total_s, 1e-12)),
    }


def _subset_count(loader, fraction, max_batches):
    """Number of batches to time (fraction of total, capped by max_batches)."""
    total = len(loader)
    return min(max_batches, max(1, int(round(total * fraction))))


# =============================================================================
# Timing functions (ported from main.ipynb)
# =============================================================================
def benchmark_linketa_style(core, model, model_name, city, split,
                            max_batches, warmup, fraction=1.0):
    """Time MetaST / MetaTTE forward (the LinkETA backbone path)."""
    dev = core["device"]
    loader = core["loaders"][split]

    scaler_mean = core["scalers"]["mean"].to(dev)
    scaler_std  = core["scalers"]["std"].to(dev)
    vff_vec     = core["graph"]["vff_vec"].to(dev)

    model = model.to(dev)
    model.eval()

    def _prep_then_call(batch, timed):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(dev)
        x_seq = batch["x_seq"]
        masks_per_rel = [m.to(dev) for m in batch["masks_per_rel"]]
        route_seq_idx = batch["route_seq_idx"]
        seq_link_len  = torch.nan_to_num(batch["seq_link_len"], nan=0.0).clamp_min(1e-3)
        seq_mask      = batch["seq_mask"].bool()
        node_edge_idx = batch["node_edge_idx"]

        x_seq_norm = (x_seq - scaler_mean.view(1, 1, -1)) / scaler_std.view(1, 1, -1)
        x_seq_norm = torch.nan_to_num(x_seq_norm, nan=0.0, posinf=0.0, neginf=0.0)

        vff_seq_b = vff_vec[node_edge_idx]
        vff_seq_b = torch.gather(vff_seq_b, 1, route_seq_idx)

        _sync(dev)
        t0 = time.perf_counter() if timed else None
        out = model(
            x_seq_norm, batch["hop_idx"], batch["route_flag"], masks_per_rel,
            route_seq_idx, seq_link_len, vff_seq_b, seq_mask,
            r_max=3.0, return_parts=True,
        )
        _sync(dev)
        if not timed:
            return None, None
        t1 = time.perf_counter()
        # force materialization of one output tensor (matches notebook recipe)
        if isinstance(out, (tuple, list)) and len(out) > 1:
            _ = out[1]
        return (t1 - t0) * 1000.0, seq_mask

    # Warmup
    with torch.no_grad():
        for w, batch in enumerate(loader):
            if w >= warmup:
                break
            _prep_then_call(batch, timed=False)

    # Measure
    n_to_time = _subset_count(loader, fraction, max_batches)
    batch_ms, trip_records = [], []
    n_trips_total = 0
    n_links_total = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_to_time:
                break
            ms, seq_mask = _prep_then_call(batch, timed=True)
            batch_ms.append(ms)
            n_links_per_trip = seq_mask.sum(dim=1).cpu().numpy()
            bsz = int(n_links_per_trip.shape[0])
            n_trips_total += bsz
            n_links_total += int(n_links_per_trip.sum())
            per_trip_ms = ms / max(bsz, 1)
            for nl in n_links_per_trip:
                trip_records.append({"n_links": int(nl), "per_trip_ms": float(per_trip_ms)})

    return _summary(batch_ms, n_trips_total, n_links_total, model_name, city, split), trip_records


def benchmark_metatraffic_pipeline(core, city, split, mt_model, linketa_mt,
                                   static_feats, rel_coo, scaler_mean_t,
                                   scaler_std_t, vff_city_t, dyn_idx, time_idx,
                                   max_batches, warmup, fraction=1.0,
                                   model_name="MetaETA"):
    """Time the MetaETA end-to-end pipeline (predict_traffic + LinkETA forward)."""
    dev = core["device"]
    loader = core["loaders"][split]

    mt_model.eval()
    linketa_mt.eval()
    mt_model.register_city_norm(city, static_feats)
    mt_model.set_city_norm(city)

    try:
        from MetaTraffic import USE_AMP_MT
        use_amp = bool(USE_AMP_MT)
    except Exception:
        use_amp = False

    with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
        h_struct = mt_model.encode_structure(static_feats, rel_coo)
    n_edges = int(static_feats.shape[0])

    def _prep_then_call(batch, timed):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(dev)

        node_edge_idx = batch["node_edge_idx"]
        x_seq = batch["x_seq"].clone()
        B_t, M_t, _ = x_seq.shape
        flat_edge_idx = node_edge_idx.reshape(-1).clamp(0, n_edges - 1)
        flat_time = x_seq.reshape(B_t * M_t, -1)[:, time_idx]

        _sync(dev)
        t0 = time.perf_counter() if timed else None

        with torch.amp.autocast("cuda", enabled=use_amp):
            pred_traffic_flat, _ = mt_model.predict_traffic(
                h_struct, static_feats, flat_time, flat_edge_idx)
        pred_traffic_flat = torch.nan_to_num(
            pred_traffic_flat, nan=0.0, posinf=1e4, neginf=-1e4)
        x_seq[:, :, dyn_idx] = pred_traffic_flat.reshape(B_t, M_t, -1).to(x_seq.dtype)

        masks_per_rel = [m.to(dev) for m in batch["masks_per_rel"]]
        route_seq_idx = batch["route_seq_idx"]
        seq_link_len  = torch.nan_to_num(batch["seq_link_len"], nan=0.0).clamp_min(1e-3)
        seq_mask      = batch["seq_mask"].bool()
        vff_seq_b = vff_city_t[node_edge_idx]
        vff_seq_b = torch.gather(vff_seq_b, 1, route_seq_idx)
        x_seq_norm = (x_seq - scaler_mean_t.view(1, 1, -1)) / scaler_std_t.view(1, 1, -1)

        with torch.amp.autocast("cuda", enabled=use_amp):
            out = linketa_mt(
                x_seq_norm, batch["hop_idx"], batch["route_flag"], masks_per_rel,
                route_seq_idx, seq_link_len, vff_seq_b, seq_mask,
                r_max=3.0, return_parts=True,
            )
        _sync(dev)
        if not timed:
            return None, None
        t1 = time.perf_counter()
        if isinstance(out, (tuple, list)) and len(out) > 1:
            _ = out[1]
        return (t1 - t0) * 1000.0, seq_mask

    # Warmup
    with torch.no_grad():
        for w, batch in enumerate(loader):
            if w >= warmup:
                break
            _prep_then_call(batch, timed=False)

    # Measure
    n_to_time = _subset_count(loader, fraction, max_batches)
    batch_ms, trip_records = [], []
    n_trips_total = 0
    n_links_total = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_to_time:
                break
            ms, seq_mask = _prep_then_call(batch, timed=True)
            batch_ms.append(ms)
            n_links_per_trip = seq_mask.sum(dim=1).cpu().numpy()
            bsz = int(n_links_per_trip.shape[0])
            n_trips_total += bsz
            n_links_total += int(n_links_per_trip.sum())
            per_trip_ms = ms / max(bsz, 1)
            for nl in n_links_per_trip:
                trip_records.append({"n_links": int(nl), "per_trip_ms": float(per_trip_ms)})

    return _summary(batch_ms, n_trips_total, n_links_total, model_name, city, split), trip_records


def _stratify_by_links(trip_records, model_name, city, split):
    """Per-trip records -> one row per (model, city, link-count bin)."""
    if not trip_records:
        return pd.DataFrame()
    df = pd.DataFrame(trip_records)
    df["bin"] = pd.cut(df["n_links"], bins=LC_BINS, right=False, labels=LC_LABELS)
    rows = []
    for label in LC_LABELS:
        sub = df[df["bin"] == label]
        if sub.empty:
            rows.append({"model": model_name, "city": city, "split": split, "bin": label,
                         "n_trips": 0, "mean_ms_per_trip": float("nan"),
                         "p50_ms_per_trip": float("nan"), "p90_ms_per_trip": float("nan")})
        else:
            rows.append({
                "model": model_name, "city": city, "split": split, "bin": label,
                "n_trips": int(sub.shape[0]),
                "mean_ms_per_trip": float(sub["per_trip_ms"].mean()),
                "p50_ms_per_trip":  float(sub["per_trip_ms"].quantile(0.50)),
                "p90_ms_per_trip":  float(sub["per_trip_ms"].quantile(0.90)),
            })
    return pd.DataFrame(rows)


# =============================================================================
# Checkpoint loading helpers
# =============================================================================
def _load_ckpt(ckpt_dir, map_location="cpu"):
    path = os.path.join(ckpt_dir, "best.ckpt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    return torch.load(path, map_location=map_location, weights_only=False)


def _load_metatraffic_combined(ckpt_dir, mt_model, linketa_mt, device):
    """MetaTraffic's checkpoint stores meta_traffic.* + linketa.* under a single
    'model' key — split and load each into its own module (matches evaluator)."""
    pkg = _load_ckpt(ckpt_dir, map_location=device)
    combined = pkg["model"]
    mt_state, le_state = {}, {}
    for k, v in combined.items():
        if k.startswith("meta_traffic."):
            mt_state[k[len("meta_traffic."):]] = v
        elif k.startswith("linketa."):
            le_state[k[len("linketa."):]] = v
    # proto_W / proto_b / target_*_labels are transient transfer buffers that
    # are absent from older meta-trained checkpoints — strict=False permits this.
    mt_model.load_state_dict(mt_state, strict=False)
    linketa_mt.load_state_dict(le_state, strict=True)


# =============================================================================
# CLI driver
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Meta_Learning inference benchmark")
    p.add_argument("--run", required=False, default=None,
                   help="Run folder name under checkpoints/ (e.g. "
                        "'porto_sf_source_chengdu_target'). Default: most "
                        "recent run folder found under checkpoints/.")
    p.add_argument("--cities", default=None,
                   help="Comma-separated city keys. Default: every city "
                        "subfolder in the run.")
    p.add_argument("--split", default="test", choices=("train", "val", "test"))
    p.add_argument("--max-batches", type=int, default=40,
                   help="Cap on timed batches per (model, city, fraction). Default 40.")
    p.add_argument("--warmup", type=int, default=5, help="Warmup batches. Default 5.")
    p.add_argument("--reps", type=int, default=1,
                   help="Repetitions (each writes a row with `rep` index). Default 1.")
    p.add_argument("--scalability", action="store_true",
                   help=f"Also run a scalability sweep at {SCAL_FRACS} of the loader, "
                        f"writing benchmark_scalability.csv.")
    p.add_argument("--methods", default="MetaST,MetaTTE,MetaETA",
                   help="Comma-separated subset of methods to benchmark. Default all.")
    return p.parse_args()


def _autodiscover_run():
    if not CHECKPOINTS_ROOT.exists():
        raise SystemExit(f"No checkpoints/ folder at {CHECKPOINTS_ROOT}")
    runs = sorted([p for p in CHECKPOINTS_ROOT.iterdir()
                   if p.is_dir() and "_source" in p.name],
                  key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        raise SystemExit("No run folders found under checkpoints/.")
    return runs[0].name


def main():
    os.chdir(BASE_DIR)
    args = parse_args()

    run_name = args.run or _autodiscover_run()
    print(f"[BENCH] run = {run_name}")
    source_keys = parse_run_folder(run_name)   # [2026-06-07] sources only

    # Cities to benchmark: --cities or every city-subfolder present in the run.
    if args.cities:
        cities = [c.strip() for c in args.cities.split(",") if c.strip()]
    else:
        run_dir = CHECKPOINTS_ROOT / run_name
        cities = sorted(p.name for p in run_dir.iterdir()
                        if p.is_dir() and p.name in CITY_CONFIGS)
        if not cities:
            raise SystemExit(f"No city subfolders found in {run_dir}.")

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    print(f"[BENCH] cities={cities}  methods={methods}  split={args.split}  "
          f"reps={args.reps}  max_batches={args.max_batches}  scalability={args.scalability}")

    # Build cores for all involved cities (sources are needed for MetaTraffic
    # normalization init, even if not in --cities).
    needed = list(dict.fromkeys(list(cities) + list(source_keys)))
    ordered = [c for c in CITY_CONFIGS if c in needed]
    cores = build_cores(ordered)
    ctx = {
        "cores": cores,
        "source_keys": source_keys,
        "target_keys": list(cities),
        "city_keys": ordered,
    }

    # Lazy imports of the model modules (heavy).
    from LinkETA import LinkETA, CFG
    from MetaST import create_metast_model
    import MetaST as _MetaST
    import MetaTTE as _MetaTTE

    device = cores[source_keys[0]]["device"]

    # ---- MetaST ----
    wrapped = None
    if "MetaST" in methods:
        base = LinkETA(
            in_dim=IN_DIM, hidden_dim=CFG["hidden_dim"], heads=CFG["heads"],
            relations=8, num_layers=CFG["num_layers"],
            max_route_len=CFG.get("max_route_len", 1052),
            num_experts=3, dropout=0.3, input_dropout=0.2,
        ).to(device)
        wrapped = create_metast_model(base, G=4, d_mem=8, device=device)
        # [2026-06-07] adapted (transfer) ckpts are PER-CITY -> loaded inside the
        # per-city loop below (each target has its own adapted weights).

    # ---- MetaTTE ----
    reptile = None
    if "MetaTTE" in methods:
        reptile = LinkETA(
            in_dim=IN_DIM, hidden_dim=CFG["hidden_dim"], heads=CFG["heads"],
            relations=8, num_layers=CFG["num_layers"],
            max_route_len=CFG.get("max_route_len", 1052),
            num_experts=3, dropout=0.3, input_dropout=0.2,
        ).to(device)
        _MetaTTE.device = device
        # [2026-06-07] per-city adapted ckpt loaded in the loop below.

    # ---- MetaETA (heavy prep; combined ADAPTED ckpt loaded per-city below) ----
    mt_prep = None
    if "MetaETA" in methods:
        mt_prep = prepare_metatraffic_data(ctx)

    # ---- Per-city benchmark loop ----
    summary_rows = []
    by_lc_rows   = []
    scal_rows    = []

    for ck in cities:
        out_dir = _eval_out_dir(run_name, ck)
        print(f"\n{'#'*90}\n# Benchmarking {ck}  ->  {out_dir}\n{'#'*90}")

        # [2026-06-07] load THIS city's ADAPTED (transfer) ckpts into the shells.
        if wrapped is not None:
            d = _transfer_dir(run_name, ck, "MetaST")
            wrapped.load_state_dict(_load_ckpt(d, map_location=device)["model"], strict=True)
            _MetaST.model = wrapped
            print(f"  [load] MetaST  adapted <- {d}")
        if reptile is not None:
            d = _transfer_dir(run_name, ck, "MetaTTE")
            reptile.load_state_dict(_load_ckpt(d, map_location=device)["model"], strict=True)
            _MetaTTE.model = reptile
            print(f"  [load] MetaTTE adapted <- {d}")
        if mt_prep is not None:
            d = _transfer_dir(run_name, ck, "MetaETA")
            _load_metatraffic_combined(d, mt_prep["mt_model"], mt_prep["linketa_mt"], device)
            print(f"  [load] MetaETA adapted <- {d}")

        for rep in range(args.reps):
            # ---- MetaST ----
            if wrapped is not None:
                s, recs = benchmark_linketa_style(
                    cores[ck], wrapped, "MetaST", ck, args.split,
                    args.max_batches, args.warmup, fraction=1.0)
                s["rep"] = rep
                summary_rows.append(s)
                by_lc_rows.extend(_stratify_by_links(recs, "MetaST", ck, args.split)
                                  .assign(rep=rep).to_dict("records"))
                print(f"  MetaST  rep={rep}:  {s['mean_ms_per_trip']:.3f} ms/trip  "
                      f"({s['trips_per_second']:.1f} trips/s)")

                if args.scalability:
                    for fr in SCAL_FRACS:
                        if fr >= 1.0:
                            continue
                        s_fr, _ = benchmark_linketa_style(
                            cores[ck], wrapped, "MetaST", ck, args.split,
                            args.max_batches, args.warmup, fraction=fr)
                        scal_rows.append({**s_fr, "fraction": fr, "rep": rep})

            # ---- MetaTTE ----
            if reptile is not None:
                s, recs = benchmark_linketa_style(
                    cores[ck], reptile, "MetaTTE", ck, args.split,
                    args.max_batches, args.warmup, fraction=1.0)
                s["rep"] = rep
                summary_rows.append(s)
                by_lc_rows.extend(_stratify_by_links(recs, "MetaTTE", ck, args.split)
                                  .assign(rep=rep).to_dict("records"))
                print(f"  MetaTTE rep={rep}:  {s['mean_ms_per_trip']:.3f} ms/trip  "
                      f"({s['trips_per_second']:.1f} trips/s)")

                if args.scalability:
                    for fr in SCAL_FRACS:
                        if fr >= 1.0:
                            continue
                        s_fr, _ = benchmark_linketa_style(
                            cores[ck], reptile, "MetaTTE", ck, args.split,
                            args.max_batches, args.warmup, fraction=fr)
                        scal_rows.append({**s_fr, "fraction": fr, "rep": rep})

            # ---- MetaTraffic ----
            if mt_prep is not None:
                s, recs = benchmark_metatraffic_pipeline(
                    cores[ck], ck, args.split,
                    mt_prep["mt_model"], mt_prep["linketa_mt"],
                    mt_prep["city_static_feats"][ck].to(device),
                    [r.to(device) for r in mt_prep["city_rel_coo"][ck]],
                    mt_prep["city_scaler_mean"][ck].to(device),
                    mt_prep["city_scaler_std"][ck].to(device),
                    mt_prep["city_vff"][ck].to(device),
                    mt_prep["city_dyn_idx"][ck],
                    mt_prep["city_time_idx"][ck],
                    args.max_batches, args.warmup, fraction=1.0,
                )
                s["rep"] = rep
                summary_rows.append(s)
                by_lc_rows.extend(_stratify_by_links(recs, "MetaETA", ck, args.split)
                                  .assign(rep=rep).to_dict("records"))
                print(f"  MetaETA rep={rep}:  {s['mean_ms_per_trip']:.3f} ms/trip  "
                      f"({s['trips_per_second']:.1f} trips/s)")

                if args.scalability:
                    for fr in SCAL_FRACS:
                        if fr >= 1.0:
                            continue
                        s_fr, _ = benchmark_metatraffic_pipeline(
                            cores[ck], ck, args.split,
                            mt_prep["mt_model"], mt_prep["linketa_mt"],
                            mt_prep["city_static_feats"][ck].to(device),
                            [r.to(device) for r in mt_prep["city_rel_coo"][ck]],
                            mt_prep["city_scaler_mean"][ck].to(device),
                            mt_prep["city_scaler_std"][ck].to(device),
                            mt_prep["city_vff"][ck].to(device),
                            mt_prep["city_dyn_idx"][ck],
                            mt_prep["city_time_idx"][ck],
                            args.max_batches, args.warmup, fraction=fr,
                        )
                        scal_rows.append({**s_fr, "fraction": fr, "rep": rep})

        # Save per-city CSVs (filtered to just this city's rows).
        sub_sum = [r for r in summary_rows if r["city"] == ck]
        sub_lc  = [r for r in by_lc_rows   if r["city"] == ck]
        sub_sc  = [r for r in scal_rows    if r["city"] == ck]

        sum_csv = os.path.join(out_dir, "benchmark_summary.csv")
        lc_csv  = os.path.join(out_dir, "benchmark_by_link_count.csv")
        sc_csv  = os.path.join(out_dir, "benchmark_scalability.csv")

        pd.DataFrame(sub_sum).to_csv(sum_csv, index=False)
        pd.DataFrame(sub_lc).to_csv(lc_csv, index=False)
        print(f"[saved {ck}] summary  -> {sum_csv}  ({len(sub_sum)} rows)")
        print(f"[saved {ck}] by_link_count -> {lc_csv}  ({len(sub_lc)} rows)")
        if args.scalability:
            pd.DataFrame(sub_sc).to_csv(sc_csv, index=False)
            print(f"[saved {ck}] scalability -> {sc_csv}  ({len(sub_sc)} rows)")

    print("\n[DONE] benchmark complete.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[ABORTED] interrupted by user.")
        sys.exit(130)
