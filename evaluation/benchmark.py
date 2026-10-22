import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import (
    BASE_DIR, CITIES, CITY_FOLDERS, DATA_ROOT,
    city_dir, results_dir, ckpt_dirname,
    _discover_edges, _discover_trips,
    load_dataset,
    _setup_linketa_new,
    _resolve_datasets,
)
from main import (
    du, con, sthr, tfeta,
    _setup_transeta, _setup_deepreta, _setup_gspinn, _require_ckpt,
    eval_linketa, eval_dueta, eval_constgat, eval_sthr, eval_transeta, eval_deepreta, eval_gspinn,
)


LINK_COUNT_BINS = [
    (1,    10,    "1-10"),
    (11,   25,    "11-25"),
    (26,   50,    "26-50"),
    (51,   100,   "51-100"),
    (101,  200,   "101-200"),
    (201,  500,   "201-500"),
    (501,  10**9, ">500"),
]

TRIP_TT_BINS = [
    (0,     300,   "<5min"),
    (301,   600,   "5-10min"),
    (601,   900,   "10-15min"),
    (901,   1200,  "15-20min"),
    (1201,  1800,  "20-30min"),
    (1801,  2700,  "30-45min"),
    (2701,  10**9, ">45min"),
]

SCALABILITY_FRACTIONS = [0.2, 0.4, 0.6, 0.8, 1.0]

L_MIN_M = 10.0
T_MIN_S = 1.5


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _to_device_inplace(batch, device):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)


def _make_subset_loader(loader, fraction, seed=42):
    from torch.utils.data import Subset, DataLoader
    dataset  = loader.dataset
    n_total  = len(dataset)
    n_subset = max(1, int(round(n_total * fraction)))
    rng      = np.random.default_rng(seed)
    idx      = sorted(rng.choice(n_total, size=n_subset, replace=False).tolist())
    return DataLoader(
        Subset(dataset, idx),
        batch_size=loader.batch_size,
        collate_fn=loader.collate_fn,
        shuffle=False,
        num_workers=getattr(loader, "num_workers", 0),
        pin_memory=False,
    )


def _load_ckpt_robust(model, ckpt_path, device):
    pkg = torch.load(ckpt_path, map_location=device)
    state = pkg["model"] if isinstance(pkg, dict) and "model" in pkg else pkg
    model.load_state_dict(state, strict=True)


def _hardware_info():
    info = {"torch_version": torch.__version__}
    if torch.cuda.is_available():
        info["cuda"] = True
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["cuda_version"] = torch.version.cuda
    else:
        info["cuda"] = False
    import os
    import platform
    info["python_version"] = platform.python_version()
    info["platform"] = platform.platform()
    try:
        with open("/proc/cpuinfo") as f:
            info["cpu"] = next(ln.split(":", 1)[1].strip() for ln in f if ln.startswith("model name"))
    except (OSError, StopIteration):
        info["cpu"] = platform.processor()
    info["cpu_count"] = os.cpu_count()
    info["torch_threads"] = torch.get_num_threads()
    return info


def _link_metrics(preds, trues, lens):
    if preds.size == 0:
        return dict(mae=float("nan"), rmse=float("nan"), mape=float("nan"),
                    n_links=0, n_links_mape=0)
    abs_err = np.abs(preds - trues)
    sq_err  = (preds - trues) ** 2
    mae  = float(abs_err.mean())
    rmse = float(np.sqrt(sq_err.mean()))
    mape_mask = (trues >= T_MIN_S) & (lens >= L_MIN_M) & (trues > 0)
    if mape_mask.any():
        mape = float((abs_err[mape_mask] / np.maximum(trues[mape_mask], 1e-6)).mean())
        n_mape = int(mape_mask.sum())
    else:
        mape = float("nan"); n_mape = 0
    return dict(mae=mae, rmse=rmse, mape=mape,
                n_links=int(preds.size), n_links_mape=n_mape)


def _route_metrics(preds, trues):
    if preds.size == 0:
        return dict(mae=float("nan"), rmse=float("nan"), mape=float("nan"),
                    n_routes=0, n_routes_mape=0)
    abs_err = np.abs(preds - trues)
    sq_err  = (preds - trues) ** 2
    mae  = float(abs_err.mean())
    rmse = float(np.sqrt(sq_err.mean()))
    pos  = trues > 0
    if pos.any():
        mape = float((abs_err[pos] / np.maximum(trues[pos], 1e-6)).mean())
        n_mape = int(pos.sum())
    else:
        mape = float("nan"); n_mape = 0
    return dict(mae=mae, rmse=rmse, mape=mape,
                n_routes=int(preds.size), n_routes_mape=n_mape)


def _stratify(trip_records, axis_fn, bins):
    out = {}
    for lo, hi, name in bins:
        matching = [t for t in trip_records
                    if (lo <= axis_fn(t) <= hi)]
        n_trips = len(matching)
        if n_trips == 0:
            out[name] = {"n_trips": 0, "n_links": 0}
            continue
        per_trip_lat = np.array([t["per_trip_ms"] for t in matching], dtype=np.float64)
        trip_pred    = np.array([t["trip_eta"]   for t in matching], dtype=np.float64)
        trip_true    = np.array([t["trip_true"]  for t in matching], dtype=np.float64)
        link_preds   = np.concatenate([t["link_preds"] for t in matching]) if matching else np.array([])
        link_trues   = np.concatenate([t["link_trues"] for t in matching]) if matching else np.array([])
        link_lens    = np.concatenate([t["link_lens"]  for t in matching]) if matching else np.array([])
        total_links  = int(link_preds.size)
        ms_per_link_arr = np.array(
            [t["per_trip_ms"] / max(t["route_length"], 1) for t in matching],
            dtype=np.float64,
        )
        link_m  = _link_metrics(link_preds, link_trues, link_lens)
        route_m = _route_metrics(trip_pred, trip_true)
        out[name] = {
            "n_trips":              n_trips,
            "n_links":              total_links,
            "mean_ms_per_trip":     float(per_trip_lat.mean()),
            "median_ms_per_trip":   float(np.median(per_trip_lat)),
            "p95_ms_per_trip":      float(np.percentile(per_trip_lat, 95)),
            "mean_ms_per_link":     float(ms_per_link_arr.mean()),
            "link_mae":             link_m["mae"],
            "link_rmse":            link_m["rmse"],
            "link_mape":            link_m["mape"],
            "route_mae":            route_m["mae"],
            "route_rmse":           route_m["rmse"],
            "route_mape":           route_m["mape"],
        }
    return out


def _aggregate_one_run(n_params, batch_records, trip_records, peak_gpu_mb):
    if not batch_records or not trip_records:
        return {"error": "empty run"}

    n_test_trips   = sum(b for b, _ in batch_records)
    n_test_batches = len(batch_records)
    total_ms       = sum(t for _, t in batch_records)
    total_s        = total_ms / 1000.0

    n_test_links = sum(t["route_length"] for t in trip_records)

    batch_ms = np.array([t for _, t in batch_records], dtype=np.float64)

    per_trip_lat = np.array([t["per_trip_ms"] for t in trip_records], dtype=np.float64)

    per_link_lat = np.array(
        [t["per_trip_ms"] / max(t["route_length"], 1) for t in trip_records],
        dtype=np.float64,
    )

    trip_pred = np.array([t["trip_eta"]  for t in trip_records], dtype=np.float64)
    trip_true = np.array([t["trip_true"] for t in trip_records], dtype=np.float64)
    route_m = _route_metrics(trip_pred, trip_true)

    link_preds = np.concatenate([t["link_preds"] for t in trip_records]) if trip_records else np.array([])
    link_trues = np.concatenate([t["link_trues"] for t in trip_records]) if trip_records else np.array([])
    link_lens  = np.concatenate([t["link_lens"]  for t in trip_records]) if trip_records else np.array([])
    link_m = _link_metrics(link_preds, link_trues, link_lens)

    by_link_count = _stratify(trip_records, lambda t: t["route_length"], LINK_COUNT_BINS)
    by_trip_tt = _stratify(trip_records, lambda t: max(t["trip_true"], 0.0), TRIP_TT_BINS)

    return {
        "n_params":       int(n_params),
        "n_test_trips":   int(n_test_trips),
        "n_test_batches": int(n_test_batches),
        "n_test_links":   int(n_test_links),
        "latency": {
            "total_inference_s":      float(total_s),
            "mean_ms_per_batch":      float(batch_ms.mean()),
            "std_ms_per_batch":       float(batch_ms.std()),
            "median_ms_per_batch":    float(np.median(batch_ms)),
            "first_batch_ms":         float(batch_ms[0]),
            "mean_ms_per_trip":       float(per_trip_lat.mean()),
            "median_ms_per_trip":     float(np.median(per_trip_lat)),
            "p95_ms_per_trip":        float(np.percentile(per_trip_lat, 95)),
            "mean_ms_per_link":       float(per_link_lat.mean()),
            "median_ms_per_link":     float(np.median(per_link_lat)),
            "throughput_trips_per_s": float(n_test_trips / total_s) if total_s > 0 else 0.0,
            "throughput_links_per_s": float(n_test_links / total_s) if total_s > 0 else 0.0,
            "peak_gpu_mb":            float(peak_gpu_mb),
        },
        "accuracy": {
            "route_mae":   route_m["mae"],
            "route_rmse":  route_m["rmse"],
            "route_mape":  route_m["mape"],
            "link_mae":    link_m["mae"],
            "link_rmse":   link_m["rmse"],
            "link_mape":   link_m["mape"],
            "n_links":     link_m["n_links"],
            "n_routes":    route_m["n_routes"],
        },
        "by_link_count": by_link_count,
        "by_trip_tt":    by_trip_tt,
    }


_LATENCY_KEYS = (
    "mean_ms_per_trip", "median_ms_per_trip", "p95_ms_per_trip",
    "mean_ms_per_link", "median_ms_per_link",
    "throughput_trips_per_s", "throughput_links_per_s",
    "mean_ms_per_batch", "median_ms_per_batch", "first_batch_ms",
    "peak_gpu_mb", "total_inference_s",
)


def _aggregate_runs(per_run_aggregates):
    if not per_run_aggregates:
        return {"error": "no runs"}
    base = dict(per_run_aggregates[0])
    base["n_repetitions"] = len(per_run_aggregates)

    lat_stats = {}
    for k in _LATENCY_KEYS:
        vals = [r["latency"][k] for r in per_run_aggregates if k in r.get("latency", {})]
        if not vals:
            continue
        m = float(np.mean(vals))
        s = float(np.std(vals))
        lat_stats[k] = {
            "mean":   m,
            "std":    s,
            "min":    float(np.min(vals)),
            "max":    float(np.max(vals)),
            "cv":     float(s / m) if m > 0 else float("nan"),
            "values": [float(v) for v in vals],
        }
    base["latency"] = lat_stats

    if len(per_run_aggregates) > 1:
        for axis in ("by_link_count", "by_trip_tt"):
            merged = {}
            for bin_name in per_run_aggregates[0][axis].keys():
                bin_runs = [r[axis][bin_name] for r in per_run_aggregates
                            if bin_name in r[axis] and r[axis][bin_name].get("n_trips", 0) > 0]
                if not bin_runs:
                    merged[bin_name] = per_run_aggregates[0][axis][bin_name]
                    continue
                ref = dict(per_run_aggregates[0][axis][bin_name])
                for k in ("mean_ms_per_trip", "median_ms_per_trip",
                          "p95_ms_per_trip", "mean_ms_per_link"):
                    if k in ref:
                        vals = [r[k] for r in bin_runs if k in r]
                        if vals:
                            ref[k] = float(np.mean(vals))
                            ref[f"{k}_std"] = float(np.std(vals))
                merged[bin_name] = ref
            base[axis] = merged

    return base


def _compute_pareto(results):
    points = []
    for name, r in results.items():
        if "latency" not in r or "accuracy" not in r:
            continue
        lat = r["latency"]["mean_ms_per_trip"]
        ms = lat["mean"] if isinstance(lat, dict) else float(lat)
        mape = r["accuracy"]["route_mape"]
        points.append({"model": name, "ms_per_trip": ms, "route_mape": mape})
    for p in points:
        dominated = False
        for q in points:
            if q is p:
                continue
            if (q["ms_per_trip"] <= p["ms_per_trip"]
                    and q["route_mape"] <= p["route_mape"]
                    and (q["ms_per_trip"] < p["ms_per_trip"]
                         or q["route_mape"] < p["route_mape"])):
                dominated = True
                break
        p["is_pareto"] = not dominated
    return points


def _run_scalability_sweep(model_name, pass_fn, model, base_loader, device, n_params):
    rows = []
    for frac in SCALABILITY_FRACTIONS:
        loader = _make_subset_loader(base_loader, frac)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        br, tr = pass_fn(model, loader, device)
        agg = _aggregate_one_run(n_params, br, tr, peak_gpu_mb=0.0)
        lat = agg["latency"]
        acc = agg["accuracy"]
        rows.append({
            "fraction":               frac,
            "n_trips":                agg["n_test_trips"],
            "n_links":                agg["n_test_links"],
            "total_inference_s":      lat["total_inference_s"],
            "mean_ms_per_trip":       lat["mean_ms_per_trip"],
            "median_ms_per_trip":     lat["median_ms_per_trip"],
            "throughput_trips_per_s": lat["throughput_trips_per_s"],
            "throughput_links_per_s": lat["throughput_links_per_s"],
            "route_mae":              acc["route_mae"],
            "route_rmse":             acc["route_rmse"],
            "route_mape":             acc["route_mape"],
            "link_mae":               acc["link_mae"],
            "link_rmse":              acc["link_rmse"],
            "link_mape":              acc["link_mape"],
        })
        print(f"    [{model_name} scalability] frac={frac:.0%}  "
              f"n_trips={agg['n_test_trips']}  "
              f"ms/trip={lat['mean_ms_per_trip']:.3f}  "
              f"tput={lat['throughput_trips_per_s']:.0f}/s  "
              f"mape={acc['route_mape']:.4f}")
    return rows


@torch.no_grad()
def _single_pass_dueta(model, loader, device):
    batch_records, trip_records = [], []
    for batch in loader:
        static = du.norm_static(batch["static_feats"])
        dyn    = du.norm_dyn(batch["dyn_slots"])
        hop_idx       = batch["hop_idx"]
        route_flag    = batch["route_flag"]
        masks_rel     = batch["masks_per_rel"]
        trip_tt       = torch.nan_to_num(batch["trip_tt"], nan=0.0)
        route_seq_idx = batch["route_seq_idx"]
        seq_mask      = batch["seq_mask"].bool()
        seq_true      = torch.nan_to_num(batch["seq_link_len"]*0 + batch["seq_true_tt"])
        seq_link_len  = torch.nan_to_num(batch.get("seq_link_len", torch.zeros_like(seq_true)))

        fwd_kwargs = {}
        if "prior_eta" in batch:
            fwd_kwargs["prior_link_eta"] = batch["prior_eta"]

        _sync(); t0 = time.perf_counter()
        try:
            out = model(static, dyn, hop_idx, route_flag, masks_rel,
                        route_seq_idx=route_seq_idx, seq_mask=seq_mask, **fwd_kwargs)
        except TypeError:
            out = model(static, dyn, hop_idx, route_flag, masks_rel, **fwd_kwargs)
        _sync(); elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if isinstance(out, (list, tuple)):
            if len(out) == 2 and out[0].dim() == 2:
                pred_seq, trip_eta = out
            else:
                pred_link = out[0]
                pred_seq  = torch.gather(pred_link, 1, route_seq_idx)
                trip_eta  = (pred_seq * seq_mask.float()).sum(dim=1)
        else:
            pred_link = out
            pred_seq  = torch.gather(pred_link, 1, route_seq_idx)
            trip_eta  = (pred_seq * seq_mask.float()).sum(dim=1)

        B = trip_eta.shape[0]
        per_trip_ms = elapsed_ms / max(B, 1)
        batch_records.append((B, elapsed_ms))

        pred_seq_cpu = pred_seq.detach().cpu().numpy()
        seq_true_cpu = seq_true.detach().cpu().numpy()
        seq_mask_cpu = seq_mask.detach().cpu().numpy()
        seq_len_cpu  = seq_link_len.detach().cpu().numpy()
        trip_eta_cpu = trip_eta.detach().cpu().numpy()
        route_true_from_links = (seq_true.detach().cpu().numpy() * seq_mask_cpu).sum(axis=1)
        trip_tt_cpu  = trip_tt.detach().cpu().numpy()

        for i in range(B):
            mask_i = seq_mask_cpu[i].astype(bool)
            true_i = trip_tt_cpu[i] if trip_tt_cpu[i] > 0 else float(route_true_from_links[i])
            trip_records.append({
                "route_length": int(mask_i.sum()),
                "trip_eta":     float(trip_eta_cpu[i]),
                "trip_true":    float(true_i),
                "per_trip_ms":  float(per_trip_ms),
                "link_preds":   pred_seq_cpu[i, mask_i].astype(np.float32),
                "link_trues":   seq_true_cpu[i, mask_i].astype(np.float32),
                "link_lens":    seq_len_cpu[i, mask_i].astype(np.float32),
            })

    return batch_records, trip_records


@torch.no_grad()
def _single_pass_constgat(model, loader, device):
    batch_records, trip_records = [], []
    for batch in loader:
        es           = batch['edge_seq'].to(device)
        onroute_mask = batch['valid_mask'].to(device)
        M = es.size(1)
        pos_idx = torch.arange(M, device=device).unsqueeze(0).expand(es.size(0), -1)

        x_bg     = batch['x_bg'].to(device)
        true_tt  = batch['true_tt'].to(device)
        link_len = batch['link_len'].to(device)
        tripT    = batch['trip_tt'].to(device)

        nb_mask   = batch['nb_mask'].to(device)
        nb_hist   = batch['nb_hist'].to(device)
        time_fk   = batch['time_fk'].to(device)
        nb_static = batch['nb_static'].to(device)

        _sync(); t0 = time.perf_counter()
        eta_link, eta_route = model(
            edge_seq=es, x_bg=x_bg, pos_idx=pos_idx,
            nb_hist=nb_hist, nb_static=nb_static, time_fk=time_fk,
            nb_mask=nb_mask, onroute_mask=onroute_mask,
        )
        _sync(); elapsed_ms = (time.perf_counter() - t0) * 1000.0

        seq_mask = onroute_mask.bool()
        B = eta_route.shape[0]
        per_trip_ms = elapsed_ms / max(B, 1)
        batch_records.append((B, elapsed_ms))

        eta_link_cpu = eta_link.detach().cpu().numpy()
        true_tt_cpu  = true_tt.detach().cpu().numpy()
        link_len_cpu = link_len.detach().cpu().numpy()
        seq_mask_cpu = seq_mask.detach().cpu().numpy()
        eta_route_cpu = eta_route.detach().cpu().numpy()
        tripT_cpu    = tripT.detach().cpu().numpy()
        route_true_from_links = (true_tt_cpu * seq_mask_cpu).sum(axis=1)

        for i in range(B):
            mask_i = seq_mask_cpu[i].astype(bool)
            true_i = tripT_cpu[i] if tripT_cpu[i] > 0 else float(route_true_from_links[i])
            trip_records.append({
                "route_length": int(mask_i.sum()),
                "trip_eta":     float(eta_route_cpu[i]),
                "trip_true":    float(true_i),
                "per_trip_ms":  float(per_trip_ms),
                "link_preds":   eta_link_cpu[i, mask_i].astype(np.float32),
                "link_trues":   true_tt_cpu[i, mask_i].astype(np.float32),
                "link_lens":    link_len_cpu[i, mask_i].astype(np.float32),
            })

    return batch_records, trip_records


@torch.no_grad()
def _single_pass_linketa(model, loader, device, vff_vec_local, scaler_mean, scaler_std, use_amp,
                         r_max=3.0):
    batch_records, trip_records = [], []
    for batch in loader:
        _to_device_inplace(batch, device)

        node_edge_idx    = batch["node_edge_idx"]
        x_seq_nodes      = batch["x_seq"]
        route_flag_nodes = batch["route_flag"].bool()
        hop_idx          = batch["hop_idx"].long()
        masks_per_rel    = [m.to(device) for m in batch["masks_per_rel"]]
        route_seq_idx    = batch["route_seq_idx"].long()
        seq_link_len     = torch.nan_to_num(batch["seq_link_len"], nan=0.0).clamp_min(1e-3)
        seq_mask         = batch["seq_mask"].bool()
        trip_tt          = torch.nan_to_num(batch["trip_tt"], nan=0.0)
        seq_true_tt      = torch.nan_to_num(batch["seq_true_tt"], nan=0.0)

        vff_node = vff_vec_local[node_edge_idx]
        vff_seq  = torch.gather(vff_node, 1, route_seq_idx)

        x_seq_norm = (x_seq_nodes - scaler_mean.view(1, 1, -1)) / scaler_std.view(1, 1, -1)
        x_seq_norm = torch.nan_to_num(x_seq_norm)

        _sync(); t0 = time.perf_counter()
        with torch.amp.autocast('cuda', enabled=use_amp):
            (link_eta_seq, trip_eta, sp_pred, lc_pred, g_, delta_, eta_phys_seq) = model(
                x_seq_norm, hop_idx, route_flag_nodes, masks_per_rel,
                route_seq_idx, seq_link_len, vff_seq, seq_mask,
                r_max=r_max, return_parts=True,
            )
        _sync(); elapsed_ms = (time.perf_counter() - t0) * 1000.0

        link_eta_seq = torch.nan_to_num(link_eta_seq, nan=0.0)
        trip_eta = torch.nan_to_num(trip_eta, nan=0.0)
        B = trip_eta.shape[0]
        per_trip_ms = elapsed_ms / max(B, 1)
        batch_records.append((B, elapsed_ms))

        link_eta_cpu = link_eta_seq.detach().cpu().numpy()
        seq_true_cpu = seq_true_tt.detach().cpu().numpy()
        seq_mask_cpu = seq_mask.detach().cpu().numpy()
        seq_len_cpu  = seq_link_len.detach().cpu().numpy()
        trip_eta_cpu = trip_eta.detach().cpu().numpy()
        trip_tt_cpu  = trip_tt.detach().cpu().numpy()
        route_true_from_links = (seq_true_cpu * seq_mask_cpu).sum(axis=1)

        for i in range(B):
            mask_i = seq_mask_cpu[i].astype(bool)
            true_i = trip_tt_cpu[i] if trip_tt_cpu[i] > 0 else float(route_true_from_links[i])
            trip_records.append({
                "route_length": int(mask_i.sum()),
                "trip_eta":     float(trip_eta_cpu[i]),
                "trip_true":    float(true_i),
                "per_trip_ms":  float(per_trip_ms),
                "link_preds":   link_eta_cpu[i, mask_i].astype(np.float32),
                "link_trues":   seq_true_cpu[i, mask_i].astype(np.float32),
                "link_lens":    seq_len_cpu[i, mask_i].astype(np.float32),
            })

    return batch_records, trip_records


def benchmark_dueta(core, ckpt_path, n_repetitions, run_scalability=False):
    device = core["device"]
    model  = core["model"].to(device)
    _load_ckpt_robust(model, ckpt_path, device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())

    per_run = []
    for rep in range(n_repetitions):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        br, tr = _single_pass_dueta(model, core["loaders"]["test"], device)
        peak = (torch.cuda.max_memory_allocated() / 1024**2) if torch.cuda.is_available() else 0.0
        per_run.append(_aggregate_one_run(n_params, br, tr, peak))
        print(f"    rep {rep+1}/{n_repetitions}  ms/trip={per_run[-1]['latency']['mean_ms_per_trip']:.3f}  "
              f"ms/link={per_run[-1]['latency']['mean_ms_per_link']:.4f}  gpu={peak:.0f}MB")

    scalability_rows = []
    if run_scalability:
        print(f"  [DuETA] scalability sweep ...")
        scalability_rows = _run_scalability_sweep(
            "DuETA",
            lambda m, ldr, dev: _single_pass_dueta(m, ldr, dev),
            model, core["loaders"]["test"], device, n_params,
        )
    return _aggregate_runs(per_run), scalability_rows


def benchmark_constgat(core, ckpt_path, n_repetitions, run_scalability=False):
    device = core["device"]
    model  = core["model"].to(device)
    _load_ckpt_robust(model, ckpt_path, device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())

    per_run = []
    for rep in range(n_repetitions):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        br, tr = _single_pass_constgat(model, core["loaders"]["test"], device)
        peak = (torch.cuda.max_memory_allocated() / 1024**2) if torch.cuda.is_available() else 0.0
        per_run.append(_aggregate_one_run(n_params, br, tr, peak))
        print(f"    rep {rep+1}/{n_repetitions}  ms/trip={per_run[-1]['latency']['mean_ms_per_trip']:.3f}  "
              f"ms/link={per_run[-1]['latency']['mean_ms_per_link']:.4f}  gpu={peak:.0f}MB")

    scalability_rows = []
    if run_scalability:
        print(f"  [ConSTGAT] scalability sweep ...")
        scalability_rows = _run_scalability_sweep(
            "ConSTGAT",
            lambda m, ldr, dev: _single_pass_constgat(m, ldr, dev),
            model, core["loaders"]["test"], device, n_params,
        )
    return _aggregate_runs(per_run), scalability_rows


def benchmark_linketa(core, ckpt_path, n_repetitions, run_scalability=False):
    device = core["device"]
    model  = core["model"].to(device)
    _load_ckpt_robust(model, ckpt_path, device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())

    vff_vec_local = core.get("graph", {}).get("vff_vec")
    if vff_vec_local is None:
        raise KeyError("benchmark_linketa requires core['graph']['vff_vec']")
    vff_vec_local = vff_vec_local.to(device)
    scaler_mean = core.get("scalers", {}).get("mean")
    scaler_std  = core.get("scalers", {}).get("std")
    if scaler_mean is None or scaler_std is None:
        raise KeyError("benchmark_linketa requires core['scalers']['mean'] and ['std']")
    scaler_mean = scaler_mean.to(device)
    scaler_std  = scaler_std.to(device)
    use_amp = bool(core.get("use_amp", False))

    per_run = []
    for rep in range(n_repetitions):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        br, tr = _single_pass_linketa(model, core["loaders"]["test"], device,
                                      vff_vec_local, scaler_mean, scaler_std, use_amp,
                                      r_max=core.get("r_max", 3.0))
        peak = (torch.cuda.max_memory_allocated() / 1024**2) if torch.cuda.is_available() else 0.0
        per_run.append(_aggregate_one_run(n_params, br, tr, peak))
        print(f"    rep {rep+1}/{n_repetitions}  ms/trip={per_run[-1]['latency']['mean_ms_per_trip']:.3f}  "
              f"ms/link={per_run[-1]['latency']['mean_ms_per_link']:.4f}  gpu={peak:.0f}MB")

    scalability_rows = []
    if run_scalability:
        print(f"  [LinkETA] scalability sweep ...")
        _vff, _sm, _ss, _amp = vff_vec_local, scaler_mean, scaler_std, use_amp
        _rmax = core.get("r_max", 3.0)
        scalability_rows = _run_scalability_sweep(
            "LinkETA",
            lambda m, ldr, dev: _single_pass_linketa(m, ldr, dev, _vff, _sm, _ss, _amp, r_max=_rmax),
            model, core["loaders"]["test"], device, n_params,
        )
    return _aggregate_runs(per_run), scalability_rows


def _ckpt_for(city, model_name):
    return results_dir(city) / ckpt_dirname(model_name, city) / "best.ckpt"


def benchmark_city(city, models, edges_gdf, df_all,
                   n_repetitions, run_scalability=False):
    print(f"\n{'='*72}")
    print(f"[CITY] {city}  ({results_dir(city)})  models={models}  reps={n_repetitions}"
          f"{'  scalability=on' if run_scalability else ''}")
    print(f"{'='*72}")
    results      = {}
    scalability  = {}

    if "DuETA" in models:
        ck = _ckpt_for(city, "DuETA")
        if not ck.exists():
            print(f"[DuETA]    SKIP — ckpt not found: {ck}")
        else:
            print(f"[DuETA]    building core ...")
            du.edges_gdf = edges_gdf
            du.df_all = df_all
            core = du.dueta_get_core_components()
            print(f"[DuETA]    benchmarking {ck}")
            results["DuETA"], scalability["DuETA"] = benchmark_dueta(
                core, ck, n_repetitions, run_scalability=run_scalability)
            del core
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    if "ConSTGAT" in models:
        ck = _ckpt_for(city, "ConSTGAT")
        if not ck.exists():
            print(f"[ConSTGAT] SKIP — ckpt not found: {ck}")
        else:
            print(f"[ConSTGAT] building core ...")
            con.edges_gdf = edges_gdf
            con.df_all = df_all
            core = con.constgat_get_core_components()
            print(f"[ConSTGAT] benchmarking {ck}")
            results["ConSTGAT"], scalability["ConSTGAT"] = benchmark_constgat(
                core, ck, n_repetitions, run_scalability=run_scalability)
            del core
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    if "LinkETA" in models:
        ck = _ckpt_for(city, "LinkETA")
        if not ck.exists():
            print(f"[LinkETA]  SKIP — ckpt not found: {ck}")
        else:
            print(f"[LinkETA]  building core ...")
            li = _setup_linketa_new(city)
            core = li.linketa_get_core_components(edges_gdf, df_all)
            print(f"[LinkETA]  benchmarking {ck}")
            results["LinkETA"], scalability["LinkETA"] = benchmark_linketa(
                core, ck, n_repetitions, run_scalability=run_scalability)
            del core, li
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    return results, scalability


def _lat_mean(r, key):
    v = r["latency"].get(key)
    if v is None:
        return float("nan")
    return v["mean"] if isinstance(v, dict) else float(v)


def _lat_std(r, key):
    v = r["latency"].get(key)
    if isinstance(v, dict):
        return v.get("std", 0.0)
    return 0.0


def save_results(city, results, hardware, n_repetitions,
                 scalability=None, output_suffix=""):
    out_dir = results_dir(city) / "eval_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix    = output_suffix or ""
    bench_path  = out_dir / f"benchmark{suffix}.json"
    summary_csv = out_dir / f"benchmark_summary{suffix}.csv"
    by_lc_csv   = out_dir / f"benchmark_by_link_count{suffix}.csv"
    by_tt_csv   = out_dir / f"benchmark_by_trip_tt{suffix}.csv"
    scal_csv    = out_dir / f"benchmark_scalability{suffix}.csv"

    payload = {
        "city":            city,
        "linketa_version": "new",
        "n_repetitions":   n_repetitions,
        "hardware":        hardware,
        "link_count_bins": [{"lo": lo, "hi": hi, "name": n} for lo, hi, n in LINK_COUNT_BINS],
        "trip_tt_bins":    [{"lo": lo, "hi": hi, "name": n} for lo, hi, n in TRIP_TT_BINS],
        "models":          results,
        "pareto":          _compute_pareto(results),
    }
    with open(bench_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    summary_rows = []
    for name, r in results.items():
        if "latency" not in r:
            continue
        summary_rows.append({
            "city":                       city,
            "model":                      name,
            "n_params":                   r.get("n_params"),
            "n_repetitions":              r.get("n_repetitions", 1),
            "n_test_trips":               r.get("n_test_trips"),
            "n_test_links":               r.get("n_test_links"),
            "ms_per_trip_mean":           _lat_mean(r, "mean_ms_per_trip"),
            "ms_per_trip_std":            _lat_std(r,  "mean_ms_per_trip"),
            "ms_per_trip_p95":            _lat_mean(r, "p95_ms_per_trip"),
            "ms_per_link_mean":           _lat_mean(r, "mean_ms_per_link"),
            "ms_per_link_std":            _lat_std(r,  "mean_ms_per_link"),
            "throughput_trips_per_s":     _lat_mean(r, "throughput_trips_per_s"),
            "throughput_links_per_s":     _lat_mean(r, "throughput_links_per_s"),
            "first_batch_ms":             _lat_mean(r, "first_batch_ms"),
            "peak_gpu_mb":                _lat_mean(r, "peak_gpu_mb"),
            "route_mae":                  r["accuracy"]["route_mae"],
            "route_rmse":                 r["accuracy"]["route_rmse"],
            "route_mape":                 r["accuracy"]["route_mape"],
            "link_mae":                   r["accuracy"]["link_mae"],
            "link_rmse":                  r["accuracy"]["link_rmse"],
            "link_mape":                  r["accuracy"]["link_mape"],
        })
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)

    lc_rows = []
    for name, r in results.items():
        if "by_link_count" not in r:
            continue
        for bin_name, b in r["by_link_count"].items():
            row = {"city": city, "model": name, "link_count_bin": bin_name}
            row.update(b)
            lc_rows.append(row)
    pd.DataFrame(lc_rows).to_csv(by_lc_csv, index=False)

    tt_rows = []
    for name, r in results.items():
        if "by_trip_tt" not in r:
            continue
        for bin_name, b in r["by_trip_tt"].items():
            row = {"city": city, "model": name, "trip_tt_bin": bin_name}
            row.update(b)
            tt_rows.append(row)
    pd.DataFrame(tt_rows).to_csv(by_tt_csv, index=False)

    if scalability:
        scal_rows = []
        for model_name, frac_rows in scalability.items():
            for row in frac_rows:
                scal_rows.append({"city": city, "model": model_name, **row})
        if scal_rows:
            pd.DataFrame(scal_rows).to_csv(scal_csv, index=False)
            print(f"[saved] {scal_csv}")

    print(f"\n[saved] {bench_path}")
    print(f"[saved] {summary_csv}")
    print(f"[saved] {by_lc_csv}")
    print(f"[saved] {by_tt_csv}")


_EVAL_TAG = {"LinkETA": "linketa", "ConSTGAT": "constgat", "DuETA": "dueta", "STHR": "sthr",
             "TransETA": "transeta", "DeeprETA": "deepreta", "GSPINN": "gspinn"}


class _ForwardTimer:

    def __init__(self, model, linketa_layout):
        self.calls, self.linketa = [], linketa_layout
        self._t0 = self._last = None
        self._h = [model.register_forward_pre_hook(self._pre), model.register_forward_hook(self._post)]

    def _pre(self, module, args):
        _sync()
        now = time.perf_counter()
        if self._last is not None and self.calls and np.isnan(self.calls[-1]["loop_ms"]):
            self.calls[-1]["loop_ms"] = (now - self._last) * 1000.0
        self._t0 = self._last = now

    def _post(self, module, args, out):
        _sync()
        rec = {"forward_ms": (time.perf_counter() - self._t0) * 1000.0, "loop_ms": float("nan"),
               "nodes": float("nan"), "links": float("nan")}
        if self.linketa:
            rec["nodes"] = int(args[0].shape[1])
            rec["links"] = int(args[7].sum())
        self.calls.append(rec)

    def end(self):
        _sync()
        if self._last is not None and self.calls and np.isnan(self.calls[-1]["loop_ms"]):
            self.calls[-1]["loop_ms"] = (time.perf_counter() - self._last) * 1000.0

    def remove(self):
        for h in self._h:
            h.remove()


def _eval_core(name, city, edges_gdf, df_all):
    ck = _require_ckpt(city, name)
    if name == "LinkETA":
        core, fn = _setup_linketa_new(city).linketa_get_core_components(edges_gdf, df_all), eval_linketa
    elif name == "DuETA":
        du.edges_gdf = edges_gdf
        du.df_all = df_all
        core, fn = du.dueta_get_core_components(), eval_dueta
    elif name == "ConSTGAT":
        con.edges_gdf = edges_gdf
        con.df_all = df_all
        core, fn = con.constgat_get_core_components(), eval_constgat
    elif name == "STHR":
        sthr.edges_gdf = edges_gdf
        sthr.df_all = df_all
        core, fn = sthr.sthr_get_core_components(), eval_sthr
    elif name == "TransETA":
        core, fn = _setup_transeta(city).transeta_get_core_components(edges_gdf, df_all), eval_transeta
    elif name == "DeeprETA":
        core, fn = _setup_deepreta(city).deepreta_get_core_components(edges_gdf, df_all), eval_deepreta
    elif name == "GSPINN":
        core, fn = _setup_gspinn(city).gspinn_get_core_components(edges_gdf, df_all), eval_gspinn
    else:
        raise ValueError(f"unknown model {name}; choices {list(_EVAL_TAG)}")
    pkg = torch.load(ck, map_location=core["device"])
    core["model"].load_state_dict(pkg["model"] if isinstance(pkg, dict) and "model" in pkg else pkg, strict=True)
    return core, fn, ck


def _timed_pass(core, eval_fn, linketa_layout):
    timer = _ForwardTimer(core["model"], linketa_layout)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    _sync()
    t0 = time.perf_counter()
    try:
        metrics, trip_df, link_df = eval_fn(core)
    finally:
        timer.end()
        timer.remove()
    wall_s = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else float("nan")
    return metrics, trip_df, link_df, timer.calls, wall_s, peak


def _assembly_ms(core):
    import random
    ds = core["datasets"]["test"]
    df_t, groups = ds.df_all, ds.groups
    st, nst, tst = random.getstate(), np.random.get_state(), torch.random.get_rng_state()
    out = {}
    try:
        for tid, rows in df_t.groupby("TRIP_ID", sort=False):
            ds.df_all, ds.groups = rows, []
            t0 = time.perf_counter()
            ds._build_groups()
            ms = (time.perf_counter() - t0) * 1000.0
            if len(ds.groups) != 1:
                raise RuntimeError(f"assembly of trip {tid}: {len(ds.groups)} groups, expected 1")
            out[str(int(tid))] = ms
    finally:
        ds.df_all, ds.groups = df_t, groups
        random.setstate(st)
        np.random.set_state(nst)
        torch.random.set_rng_state(tst)
    if len(out) != len(groups):
        raise RuntimeError(f"assembly: {len(out)} trips timed, the test set has {len(groups)} groups")
    return out


def _forest_ms(core):
    import os
    try:
        import deepforest
    except (ImportError, ValueError) as e:
        print(f"[TransETA] forest stage NOT timed on this machine: deep-forest does not load ({type(e).__name__}: {e})")
        return None
    m = tfeta._new_forest()
    m.load(os.path.join(tfeta.CKPT_DIRNAME, tfeta.FOREST_DIR))
    st = core["static"]["test"]
    X = st[tfeta.FOREST_FEATS].to_numpy(np.float64)
    saved = np.array([core["forest_preds"]["test"][int(t)] for t in st.index])
    dev = float(np.abs(np.asarray(m.predict(X)).reshape(-1) - saved).max())
    if dev > 1e-6:
        raise RuntimeError(f"[TransETA] loaded forest differs from its saved test predictions (max {dev:.3g} s)")
    out = {}
    for i, tid in enumerate(st.index):
        t0 = time.perf_counter()
        m.predict(X[i:i + 1])
        out[str(int(tid))] = (time.perf_counter() - t0) * 1000.0
    return out


def _append_rows(path, rows, model):
    new = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        if (old["model"] == model).any():
            old.to_csv(path.with_name(f"{path.stem}_preserved_{time.strftime('%Y%m%d_%H%M%S')}{path.suffix}"),
                       index=False)
            old = old[old["model"] != model]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False)
    print(f"[saved] {path} ({len(rows)} rows of {model})")


def benchmark_model(name, city, edges_gdf, df_all, batch_sizes, n_reps, replay, suffix="", max_trips=None):
    from torch.utils.data import DataLoader, Subset
    import gc
    out_dir = results_dir(city) / "eval_outputs"
    saved = json.loads((out_dir / f"eval_metrics_{_EVAL_TAG[name]}.json").read_text())[name]["route_mape"]
    st = pd.read_csv(out_dir / f"eval_trips_{_EVAL_TAG[name]}.csv", usecols=["TRIP_ID", "y_true", "y_pred"])
    st.index = st.TRIP_ID.astype("int64").to_numpy()
    print(f"\n[{city}] {name}: building the core and loading the final checkpoint ...")
    t_build = time.perf_counter()
    core, eval_fn, ck = _eval_core(name, city, edges_gdf, df_all)
    build_s = time.perf_counter() - t_build
    summ_p = ck.parent / "training_summary.json"
    summ = json.loads(summ_p.read_text()) if summ_p.exists() else {}
    hw = _hardware_info()
    common = dict(city=city, model=name, device=str(core["device"]),
                  gpu=hw.get("gpu_name", "") if str(core["device"]).startswith("cuda") else "",
                  cpu=hw["cpu"], torch=hw["torch_version"])
    n_params = int(sum(p.numel() for p in core["model"].parameters()))
    lk = name == "LinkETA"
    base = core["loaders"]["test"]
    ds = base.dataset if not max_trips else Subset(base.dataset, list(range(min(int(max_trips), len(base.dataset)))))

    def _vs_saved(tdf):
        s = st.reindex(tdf.TRIP_ID.astype("int64").to_numpy())
        if s.y_pred.isna().any():
            raise RuntimeError(f"{name}: {int(s.y_pred.isna().sum())} evaluated trips are not in eval_trips_{_EVAL_TAG[name]}.csv")
        pos = s.y_true.to_numpy() > 0
        mape_s = float(np.mean(np.abs(s.y_pred.to_numpy() - s.y_true.to_numpy())[pos] / s.y_true.to_numpy()[pos]))
        return mape_s, float(np.abs(tdf.y_pred.to_numpy() - s.y_pred.to_numpy()).max())

    inf_rows, trip_rows, wl_rows, ref_pred, n_links_ref = [], [], [], None, None
    try:
        for b in batch_sizes:
            core["loaders"]["test"] = DataLoader(ds, batch_size=b, shuffle=False,
                                                 collate_fn=base.collate_fn, num_workers=base.num_workers)
            passes = []
            for r in range(n_reps if b != 1 else 1):
                m, tdf, ldf, calls, wall, peak = _timed_pass(core, eval_fn, lk)
                mape_s, dpred = _vs_saved(tdf)
                if b == batch_sizes[0] and (not abs(m["route_mape"] - mape_s) < 1e-3
                                            or (not max_trips and not abs(m["route_mape"] - saved) < 1e-3)):
                    raise RuntimeError(f"{name} batch {b}: route MAPE {m['route_mape']:.5f} != saved {mape_s:.5f} "
                                       f"(these trips) / {saved:.5f} (test set)")
                passes.append((m, tdf, ldf, calls, wall, peak))
                print(f"  batch {b:3d} pass {r + 1}: {len(tdf)} trips in {wall:.2f} s | model "
                      f"{sum(c['forward_ms'] for c in calls) / len(tdf):.3f} ms/trip | route MAPE {m['route_mape']:.4f} "
                      f"(saved, same trips {mape_s:.4f}) | max |pred - saved pred| {dpred:.3f} s | peak GPU {peak:.0f} MB")
            m, tdf, ldf, calls, _, _ = passes[0]
            n_trips = len(tdf)
            n_links = len(ldf) if len(ldf) else float("nan")
            pred = pd.Series(tdf.y_pred.to_numpy(), index=tdf.TRIP_ID.astype("int64").to_numpy())
            if ref_pred is None:
                ref_pred, diff = pred, 0.0
            else:
                diff = float((pred.reindex(ref_pred.index) - ref_pred).abs().max())
                if np.isnan(diff):
                    raise RuntimeError(f"{name}: batch {b} predictions miss trips of batch {batch_sizes[0]}")
            model_ms = [sum(c["forward_ms"] for c in p[3]) / len(p[1]) for p in passes]
            loop_ms = [p[4] * 1000.0 / len(p[1]) for p in passes]
            row = dict(common, batch_size=b, n_passes=len(passes), n_trips=n_trips, n_links=n_links,
                       model_ms_per_trip=float(np.mean(model_ms)), model_ms_per_trip_sd=float(np.std(model_ms)),
                       loop_ms_per_trip=float(np.mean(loop_ms)), loop_ms_per_trip_sd=float(np.std(loop_ms)),
                       trips_per_s=float(np.mean([len(p[1]) / p[4] for p in passes])),
                       peak_gpu_mb=float(np.nanmax([p[5] for p in passes])) if torch.cuda.is_available() else float("nan"),
                       route_mape=float(m["route_mape"]), route_mape_saved=float(_vs_saved(tdf)[0]),
                       max_pred_diff_vs_saved_s=float(_vs_saved(tdf)[1]),
                       max_pred_diff_vs_first_batch_s=diff, n_params=n_params, core_build_s=build_s,
                       train_time_s_logged=summ.get("total_train_time_s"), train_epochs_logged=summ.get("epochs_run"),
                       train_peak_gpu_mb_logged=summ.get("max_peak_gpu_mb"))
            if b == 1:
                if len(calls) != n_trips:
                    raise RuntimeError(f"{name} batch 1: {len(calls)} timed calls for {n_trips} test trips")
                tids = tdf.TRIP_ID.astype("int64").astype(str).to_numpy()
                if lk:
                    per_trip = ldf.TRIP_ID.astype("int64").astype(str).value_counts()
                    bad = sum(int(c["links"]) != int(per_trip.get(t, -1)) for c, t in zip(calls, tids))
                    if bad:
                        raise RuntimeError(f"LinkETA batch 1: {bad} timed calls whose links differ from the trip's link rows")
                for c, t in zip(calls, tids):
                    trip_rows.append(dict(common, batch_size=1, trip_id=t, nodes=c["nodes"], links=c["links"],
                                          forward_ms=c["forward_ms"], loop_ms=c["loop_ms"],
                                          assembly_ms=float("nan"), forest_ms=float("nan")))
                for q in (50, 95, 99):
                    row[f"model_ms_p{q}"] = float(np.percentile([c["forward_ms"] for c in calls], q))
                    row[f"loop_ms_p{q}"] = float(np.percentile([c["loop_ms"] for c in calls], q))
            inf_rows.append(row)
            if lk and b == batch_sizes[0]:
                n_links_ref = n_links
        core["loaders"]["test"] = base

        extra = {}
        if lk and 1 in batch_sizes:
            print(f"  assembling each test trip's graph from its rows (LinkETA's _build_groups) ...")
            extra["assembly_ms"] = _assembly_ms(core)
        if name == "TransETA" and 1 in batch_sizes:
            f = _forest_ms(core)
            if f is not None:
                extra["forest_ms"] = f
        for col, mp in extra.items():
            for tr in trip_rows:
                tr[col] = mp[tr["trip_id"]]
        if trip_rows:
            e2e = np.array([tr["loop_ms"] + np.nan_to_num(tr["assembly_ms"]) + np.nan_to_num(tr["forest_ms"])
                            for tr in trip_rows])
            r1 = next(r for r in inf_rows if r["batch_size"] == 1)
            for q in (50, 95, 99):
                r1[f"end_to_end_ms_p{q}"] = float(np.percentile(e2e, q))
                for col in extra:
                    r1[f"{col.replace('_ms', '')}_ms_p{q}"] = float(np.percentile([tr[col] for tr in trip_rows], q))
            r1["end_to_end_parts"] = "+".join(["loop"] + list(extra))

        if lk and replay:
            core["loaders"]["test"] = DataLoader(ds, batch_size=batch_sizes[0], shuffle=False,
                                                 collate_fn=base.collate_fn, num_workers=base.num_workers)
            for k in [1] + list(replay):
                tot_s = fwd_ms = 0.0
                trips = links = 0
                peak = 0.0
                for _ in range(k):
                    m, tdf, ldf, calls, wall, pk = _timed_pass(core, eval_fn, lk)
                    if not abs(m["route_mape"] - _vs_saved(tdf)[0]) < 1e-3:
                        raise RuntimeError(f"LinkETA replay: route MAPE {m['route_mape']:.5f} != saved {_vs_saved(tdf)[0]:.5f}")
                    tot_s += wall
                    fwd_ms += sum(c["forward_ms"] for c in calls)
                    trips += len(tdf)
                    links += len(ldf)
                    peak = max(peak, pk) if torch.cuda.is_available() else float("nan")
                if links != k * n_links_ref:
                    raise RuntimeError(f"LinkETA replay x{k}: {links} links processed, expected {k * n_links_ref}")
                wl_rows.append(dict(common, batch_size=batch_sizes[0], replay=k, n_trips=trips, n_links=links,
                                    total_s=tot_s, model_s=fwd_ms / 1000.0, ms_per_link=tot_s * 1000.0 / links,
                                    model_ms_per_link=fwd_ms / links, links_per_s=links / tot_s, peak_gpu_mb=peak))
                print(f"  replay x{k:3d}: {links:,} links in {tot_s:.1f} s ({links / tot_s:,.0f} links/s)")
            core["loaders"]["test"] = base
    finally:
        core["loaders"]["test"] = base

    _append_rows(out_dir / f"benchmark_inference{suffix}.csv", inf_rows, name)
    if trip_rows:
        _append_rows(out_dir / f"benchmark_trips{suffix}.csv", trip_rows, name)
    if wl_rows:
        _append_rows(out_dir / f"benchmark_workload{suffix}.csv", wl_rows, name)
    hw_path = out_dir / f"benchmark_hardware{suffix}.json"
    hw_all = json.loads(hw_path.read_text()) if hw_path.exists() else {}
    hw_all[name] = dict(hw, measured=time.strftime("%Y-%m-%d %H:%M:%S"))
    hw_path.write_text(json.dumps(hw_all, indent=2))
    del core
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_args():
    p = argparse.ArgumentParser(
        description="Inference benchmark v2 for trained ETA models (uses existing ckpts).")
    p.add_argument("--datasets", "--cities", dest="datasets", default="all",
                   help=f"Comma-separated city names/codes (or 'all'). Choices: {', '.join(CITIES)}.")
    p.add_argument("--models", default="DuETA,ConSTGAT,LinkETA",
                   help="Comma-separated subset of models to benchmark. "
                        "Choices: DuETA, ConSTGAT, LinkETA.")
    p.add_argument("--max-trips", type=int, default=None,
                   help="Sub-sample at most N TRIP_IDs (random, seed=42). Useful for "
                        "quick latency smoke tests. Note: scalers are re-fit on the "
                        "subset, which can produce out-of-distribution accuracy "
                        "numbers — latency is fine but MAPE may be unreliable.")
    p.add_argument("--repetitions", type=int, default=5,
                   help="Number of full inference passes per model for cross-rep "
                        "mean/std/min/max/cv. Default: 5. Use 1 for a quick sanity run.")
    p.add_argument("--scalability", action="store_true", default=False,
                   help="After the N-rep benchmark, run a single-rep sweep at "
                        "20/40/60/80/100%% of the test set and save "
                        "benchmark_scalability.csv. Shows ms/trip stays flat "
                        "(O(n) inference complexity) as trip count grows.")
    p.add_argument("--output-suffix", default="",
                   help="Optional suffix appended to all benchmark output filenames "
                        "(e.g. '_dueta' produces benchmark_summary_dueta.csv). Useful "
                        "for running a single model (e.g. --models DuETA) "
                        "without overwriting an existing combined-models output. "
                        "Default: empty.")
    p.add_argument("--batch-sizes", default="16,1",
                   help="Comma-separated batch sizes; the first is the reference for the prediction check, 1 gives "
                        "the per-request (streaming) figures. Default 16,1.")
    p.add_argument("--replay", default="10,100",
                   help="LinkETA only: run the test set k times back to back for each k (plus k=1). Empty: none.")
    return p.parse_args()


def main():
    args = parse_args()
    cities = _resolve_datasets(args.datasets)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    n_reps = max(1, int(args.repetitions))

    hardware = _hardware_info()
    output_suffix = args.output_suffix or ""
    print(f"Hardware: {hardware}")
    print(f"Models: {models}    Repetitions per model: {n_reps}")
    print(f"Cities:  {cities}")
    print(f"Scalability sweep: {'on' if args.scalability else 'off'}")
    if output_suffix:
        print(f"Output suffix: '{output_suffix}'")

    if args.scalability:
        raise SystemExit("--scalability belongs to the old benchmark (not used since 2026-10-20)")
    batch_sizes = [int(b) for b in args.batch_sizes.split(",") if b.strip()]
    replay = [int(k) for k in args.replay.split(",") if k.strip()]
    for city in cities:
        edges_gdf, df_all = load_dataset(city)
        for name in models:
            benchmark_model(name, city, edges_gdf, df_all, batch_sizes, n_reps, replay, output_suffix,
                            max_trips=args.max_trips)

    print("\n[DONE]")


if __name__ == "__main__":
    main()

