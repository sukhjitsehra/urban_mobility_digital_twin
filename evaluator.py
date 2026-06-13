"""
eval.py — Evaluate meta-learned checkpoints on all cities (source + target).
=========================================================================

For each method (MetaST / MetaTTE / MetaTraffic), this module:
  1. Loads the best checkpoint from disk.
  2. For every city (source AND target), transfers the meta-learned θ₀
     by running a small number of adaptation steps on the city's train set.
  3. Evaluates on the city's test set and collects Link + Route metrics.
  4. Prints a consolidated results table.

Usage:
    from evaluator import evaluate_metast, evaluate_metatte, evaluate_metatraffic

All three functions share the same evaluation contract:
  - Load ckpt  →  adapt on train  →  evaluate on test  →  return dict of metrics.
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from collections import deque

from LinkETA import (
    one_epoch, LinkETA, CFG, lam_g_at,
)
from loss_functions import (
    link_metrics_micro, route_metrics_pooled,
    L_MIN_METRIC, T_MIN_METRIC,
    link_loss_huber_macro, route_loss_mape,
    # [2026-06-06] transfer/eval now adapts with the SAME log-space backbone losses
    # as base LinkETA + meta-training (no more old huber_macro/mape during adaptation).
    link_eta_in_seconds, link_loss_huber_log_lenweighted, route_loss_huber_log,
)
from checkpoint import (
    CheckpointManager,
    metrics_table_header, metrics_table_row, metrics_table_footer,
)


# =============================================================================
# [2026-06-06] Canonical LinkETA speed loss vs sp_actual (mirrors base LinkETA
# one_epoch). Identical to the helpers in MetaST.py / MetaTTE.py so the transfer
# adaptation objective matches meta-training exactly.
# =============================================================================
def _backbone_speed_loss(sp_pred, seq_link_len, seq_true_tt, vff_seq, seq_mask,
                         vff_seq_clean=None, r_max=3.0):
    if vff_seq_clean is None:
        vff_seq_clean = vff_seq
    sp_actual = (seq_link_len / seq_true_tt.clamp_min(1e-3)) / vff_seq.clamp_min(1.0)
    sp_actual = sp_actual.clamp(0.0, r_max)
    sp_actual_clean = (seq_link_len / seq_true_tt.clamp_min(1e-3)) / vff_seq_clean.clamp_min(1.0)
    was_pinned = sp_actual_clean >= (r_max - 1e-3)
    sp_w_actual = (
        seq_mask
        & (seq_true_tt >= 1.0)
        & (seq_link_len >= 10.0)
        & (~was_pinned)
    ).float()
    denom = sp_w_actual.sum().clamp_min(1.0)
    per = F.smooth_l1_loss(sp_pred, sp_actual, reduction="none", beta=0.1)
    return (per * sp_w_actual).sum() / denom

import hashlib
import numpy as np
import pandas as pd


# =====================================================================
# Helpers
# =====================================================================

def _scaler_fingerprint(city_scaler_mean: Dict[str, torch.Tensor],
                        city_scaler_std: Dict[str, torch.Tensor]) -> str:
    """Hash of scaler statistics to detect mismatch between checkpoint and current data."""
    parts = []
    for ck in sorted(city_scaler_mean.keys()):
        m = city_scaler_mean[ck].cpu().float().numpy()
        s = city_scaler_std[ck].cpu().float().numpy()
        parts.append(f"{ck}:{m[:5].tobytes().hex()}{s[:5].tobytes().hex()}")
    return hashlib.md5("|".join(parts).encode()).hexdigest()[:12]

def _print_table(title: str, results: Dict[str, dict]):
    """[2026-06-07] Print the TARGET transfer results with the FULL Link + Route
    metric suite (MAE/RMSE/MAPE/WAPE/MdAPE/P90), using the SHARED formatter so the
    target table matches the SOURCE in-sample table exactly. Non-metric entries
    (e.g. the stored trip_df/link_df frames) are skipped — they are not splits."""
    metrics_table_header(title)
    for city_name, city_metrics in results.items():
        for split, m in city_metrics.items():
            if not isinstance(m, dict):   # skip trip_df / link_df DataFrames
                continue
            metrics_table_row(city_name, split, m)
    metrics_table_footer()


def _load_checkpoint(ckpt_dir: str, map_location="cpu"):
    """Load best.ckpt from a checkpoint directory."""
    best_path = os.path.join(ckpt_dir, "best.ckpt")
    if not os.path.isfile(best_path):
        raise FileNotFoundError(
            f"No best checkpoint found at {best_path}. "
            f"Train the model first."
        )
    pkg = torch.load(best_path, map_location=map_location)
    print(f"  Loaded checkpoint: {best_path}")
    print(f"    Epoch/Iter: {pkg.get('epoch', '?')}, "
          f"Best metric: {pkg.get('best_metric', '?'):.6f}, "
          f"Method: {pkg.get('extra', {}).get('method', '?')}")
    return pkg


def _collect_linketa_preds(core, loader, model_name: str):
    """Collect per-trip and per-link predictions from an already-adapted
    LinkETA-style model (MetaST wrapped or MetaTTE base).

    Assumes the model is already in eval mode with adapted weights loaded.
    Stateless: the model + scalers + v_ff are read from `core` (the adapted /
    wrapped model must be stored at core["model"] by set_globals_from_core).

    Returns
    -------
    (trip_df, link_df) : tuple of pandas.DataFrame
        trip_df columns: model, TRIP_ID, y_true, y_pred, route_len_m,
                         target_source, gain_g
        link_df columns: model, TRIP_ID, true_tt_link, pred_tt_link
    """
    _cur_model = core["model"]
    SCALER_MEAN = core["scalers"]["mean"]
    SCALER_STD  = core["scalers"]["std"]
    vff_vec     = core["graph"]["vff_vec"]
    device = next(_cur_model.parameters()).device
    SCALER_MEAN = SCALER_MEAN.to(device)
    SCALER_STD  = SCALER_STD.to(device)
    vff_vec     = vff_vec.to(device)
    trip_rows = []
    link_rows = []
    fallback_id = 0
    for batch in loader:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        node_edge_idx = batch["node_edge_idx"]
        x_seq = batch["x_seq"]
        route_seq_idx = batch["route_seq_idx"]
        seq_link_len = torch.nan_to_num(batch["seq_link_len"], nan=0.0).clamp_min(1e-3)
        seq_mask = batch["seq_mask"].bool()
        trip_tt = torch.nan_to_num(batch["trip_tt"], nan=0.0)
        # Per-link truth (needed for link_df + target_source fallback tag).
        seq_true_tt = torch.nan_to_num(batch["seq_true_tt"], nan=0.0)

        x_seq_norm = (x_seq - SCALER_MEAN.view(1, 1, -1)) / SCALER_STD.view(1, 1, -1)
        vff_seq_b = vff_vec[node_edge_idx]
        vff_seq_b = torch.gather(vff_seq_b, 1, route_seq_idx)

        out = _cur_model(
            x_seq_norm, batch["hop_idx"], batch["route_flag"],
            [m.to(device) for m in batch["masks_per_rel"]],
            route_seq_idx, seq_link_len, vff_seq_b, seq_mask,
            r_max=3.0, return_parts=True,
        )
        # Match route_metrics_pooled: use link_eta_seq (already masked), sum to get route pred.
        link_eta_seq = torch.nan_to_num(out[0], nan=0.0)
        trip_pred = (link_eta_seq * seq_mask.float()).sum(dim=1)
        # gain_g lives at out[4] when return_parts=True; fall back to 1.0 if not exposed.
        if isinstance(out, (tuple, list)) and len(out) >= 5:
            gain_g = torch.nan_to_num(out[4], nan=1.0)
        else:
            gain_g = torch.ones_like(trip_pred)

        y_true = trip_tt
        route_len_m = (seq_link_len * seq_mask.float()).sum(dim=1)

        bsz = int(y_true.shape[0])
        trip_ids = _extract_trip_ids(batch, bsz, fallback_id)
        fallback_id += bsz

        yt = y_true.detach().cpu().numpy()
        yp = trip_pred.detach().cpu().numpy()
        rl = route_len_m.detach().cpu().numpy()
        gg = gain_g.detach().cpu().numpy()
        # target_source: "trip_tt" when the GPS-derived trip duration is present,
        # "missing" otherwise (preserves current y_true=trip_tt semantics; the
        # tag lets downstream filters drop trips without a real ground-truth).
        trip_tt_np = trip_tt.detach().cpu().numpy()
        for i in range(bsz):
            trip_rows.append({
                "model":         model_name,
                "TRIP_ID":       int(trip_ids[i]),
                "y_true":        float(yt[i]),
                "y_pred":        float(yp[i]),
                "route_len_m":   float(rl[i]),
                "target_source": "trip_tt" if float(trip_tt_np[i]) > 0 else "missing",
                "gain_g":        float(gg[i]),
            })

        # ---- link-level rows (paired across models by (TRIP_ID, link_idx)) ----
        # link_eta_seq is shaped [B, L]; only positions where seq_mask is True
        # are valid links in that trip's route.
        pred_seq_cpu = link_eta_seq.detach().cpu()
        true_seq_cpu = seq_true_tt.detach().cpu()
        mask_cpu     = seq_mask.detach().cpu()
        B0, L0 = pred_seq_cpu.shape
        for b in range(B0):
            tid = int(trip_ids[b])
            for l in range(L0):
                if not bool(mask_cpu[b, l]):
                    continue
                link_rows.append((
                    model_name, tid,
                    float(true_seq_cpu[b, l].item()),
                    float(pred_seq_cpu[b, l].item()),
                ))

    trip_df = pd.DataFrame(trip_rows)
    link_df = pd.DataFrame(
        link_rows,
        columns=["model", "TRIP_ID", "true_tt_link", "pred_tt_link"],
    )
    return trip_df, link_df


def _extract_trip_ids(batch, bsz, fallback_id):
    """Extract trip IDs from a batch, with fallback to sequential integers."""
    if "trip_ids" in batch:
        tid = batch["trip_ids"]
        if isinstance(tid, torch.Tensor):
            trip_ids = tid.detach().cpu().numpy().reshape(-1)
        elif isinstance(tid, np.ndarray):
            trip_ids = tid.reshape(-1)
        elif isinstance(tid, (list, tuple)):
            trip_ids = np.array([
                int(x.detach().cpu().item()) if isinstance(x, torch.Tensor) else int(x)
                for x in tid
            ])
        else:
            trip_ids = np.array([int(tid)])
        if trip_ids.shape[0] != bsz:
            trip_ids = np.arange(fallback_id, fallback_id + bsz)
    else:
        trip_ids = np.arange(fallback_id, fallback_id + bsz)
    return trip_ids


def _collect_metatraffic_preds(
    loader, meta_traffic_model, linketa_model,
    static_feats, rel_coo, scaler_mean, scaler_std,
    vff_city, dyn_idx, time_idx,
    model_name: str = "MetaETA",   # [2026-06-06] framework label (was "MetaTraffic")
):
    """Collect per-trip AND per-link predictions for the MetaTraffic +
    LinkETA pipeline.

    Assumes both models are in eval mode with adapted weights loaded,
    and city norm is already set.

    Returns
    -------
    (trip_df, link_df) : tuple of pandas.DataFrame  (same schema as the
        LinkETA-style collector — see _collect_linketa_preds)
    """
    from MetaTraffic import USE_AMP_MT

    device = next(meta_traffic_model.parameters()).device
    n_edges = int(static_feats.shape[0])

    with torch.amp.autocast("cuda", enabled=USE_AMP_MT):
        h_struct = meta_traffic_model.encode_structure(static_feats, rel_coo)

    trip_rows = []
    link_rows = []
    fallback_id = 0
    for batch in loader:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        node_edge_idx = batch["node_edge_idx"]
        x_seq = batch["x_seq"].clone()
        B, M, _ = x_seq.shape
        flat_edge_idx = node_edge_idx.reshape(-1).clamp(0, n_edges - 1)
        flat_time = x_seq.reshape(B * M, -1)[:, time_idx]

        with torch.amp.autocast("cuda", enabled=USE_AMP_MT):
            pred_traffic_flat, _ = meta_traffic_model.predict_traffic(
                h_struct, static_feats, flat_time, flat_edge_idx)
        pred_traffic_flat = torch.nan_to_num(
            pred_traffic_flat, nan=0.0, posinf=1e4, neginf=-1e4)
        x_seq[:, :, dyn_idx] = pred_traffic_flat.reshape(B, M, -1).to(x_seq.dtype)

        masks_per_rel = [m.to(device) for m in batch["masks_per_rel"]]
        route_seq_idx = batch["route_seq_idx"]
        seq_link_len = torch.nan_to_num(batch["seq_link_len"], nan=0.0).clamp_min(1e-3)
        seq_mask = batch["seq_mask"].bool()
        trip_tt = torch.nan_to_num(batch["trip_tt"], nan=0.0)
        seq_true_tt = torch.nan_to_num(batch["seq_true_tt"], nan=0.0)

        vff_seq_b = vff_city[node_edge_idx]
        vff_seq_b = torch.gather(vff_seq_b, 1, route_seq_idx)
        x_seq_norm = (x_seq - scaler_mean.view(1, 1, -1)) / scaler_std.view(1, 1, -1)

        with torch.amp.autocast("cuda", enabled=USE_AMP_MT):
            lk_out = linketa_model(
                x_seq_norm, batch["hop_idx"], batch["route_flag"], masks_per_rel,
                route_seq_idx, seq_link_len, vff_seq_b, seq_mask,
                r_max=3.0, return_parts=True,
            )
        link_eta_seq = torch.nan_to_num(lk_out[0], nan=0.0)
        if isinstance(lk_out, (tuple, list)) and len(lk_out) >= 5:
            gain_g = torch.nan_to_num(lk_out[4], nan=1.0)
        else:
            gain_g = None

        # Match route_metrics_pooled: sum of link preds, compare to trip_tt
        trip_pred = (link_eta_seq * seq_mask.float()).sum(dim=1)
        if gain_g is None:
            gain_g = torch.ones_like(trip_pred)
        y_true = trip_tt
        route_len_m = (seq_link_len * seq_mask.float()).sum(dim=1)

        bsz = int(y_true.shape[0])
        trip_ids = _extract_trip_ids(batch, bsz, fallback_id)
        fallback_id += bsz

        yt = y_true.detach().cpu().numpy()
        yp = trip_pred.detach().cpu().numpy()
        rl = route_len_m.detach().cpu().numpy()
        gg = gain_g.detach().cpu().numpy()
        trip_tt_np = trip_tt.detach().cpu().numpy()
        for i in range(bsz):
            trip_rows.append({
                "model":         model_name,
                "TRIP_ID":       int(trip_ids[i]),
                "y_true":        float(yt[i]),
                "y_pred":        float(yp[i]),
                "route_len_m":   float(rl[i]),
                "target_source": "trip_tt" if float(trip_tt_np[i]) > 0 else "missing",
                "gain_g":        float(gg[i]),
            })

        # ---- link-level rows ----
        pred_seq_cpu = link_eta_seq.detach().cpu()
        true_seq_cpu = seq_true_tt.detach().cpu()
        mask_cpu     = seq_mask.detach().cpu()
        B0, L0 = pred_seq_cpu.shape
        for b in range(B0):
            tid = int(trip_ids[b])
            for l in range(L0):
                if not bool(mask_cpu[b, l]):
                    continue
                link_rows.append((
                    model_name, tid,
                    float(true_seq_cpu[b, l].item()),
                    float(pred_seq_cpu[b, l].item()),
                ))

    trip_df = pd.DataFrame(trip_rows)
    link_df = pd.DataFrame(
        link_rows,
        columns=["model", "TRIP_ID", "true_tt_link", "pred_tt_link"],
    )
    return trip_df, link_df


# =====================================================================
# MetaST Evaluation
# =====================================================================

def transfer_to_target_metast(
    cores: Dict[str, dict],
    city_configs: Dict[str, dict],
    wrapped_model: nn.Module,
    ckpt_dir: str = "ckpts_MetaETA_MetaST",
    transfer_ckpt_dir: str = None,        # [2026-06-06] where to save the ADAPTED ckpt
    adapt_steps: int = 500,               # aligned budget cap
    adapt_lr: float = 1e-3,               # [2026-06-06] MetaST SGD LR aligned to MetaETA
    patience: int = 10,                   # aligned early-stopping patience
    eval_every: int = 10,                 # aligned val cadence
    w_speed: float = 1.0 / 3.0,           # 3-task {speed,link,route}
    w_link: float = 1.0 / 3.0,
    w_route: float = 1.0 / 3.0,
    collect_trip_preds: bool = False,
    trip_preds_dir: str = "figs",
) -> Dict[str, dict]:
    """
    Evaluate MetaST checkpoint on ALL cities (source + target).

    For each city:
      1. Load meta-learned θ₀ from best.ckpt
      2. Run adapt_steps gradient steps on the city's train set
      3. Evaluate on val and test sets

    Parameters
    ----------
    cores : dict
        {city_key: core_dict} for all cities.
    city_configs : dict
        CITY_CONFIGS dict with city metadata.
    wrapped_model : nn.Module
        The LinkETAWithSTMemory model (same architecture used during training).
    ckpt_dir : str
        Directory containing MetaST checkpoints.
    adapt_steps : int
        Number of inner adaptation steps per city.
    adapt_lr : float
        Learning rate for adaptation.
    collect_trip_preds : bool
        If True, save per-trip predictions to CSV after evaluating each split.
    trip_preds_dir : str
        Directory for trip prediction CSVs (default "figs").

    Returns
    -------
    dict: {city_name: {"val": metrics, "test": metrics, "trip_df": df, "link_df": df}}
        trip_df / link_df cover BOTH splits (val + test), tagged with `split`
        and `city` columns so main.py can concat across methods cleanly.
    """
    import MetaST as MetaST
    from MetaST import (
        set_globals_from_core, inner_adapt_steps,
        clone_state_cpu, load_state_cpu,
    )

    print(f"\n{'='*90}")
    print(f"  MetaST Evaluation — Loading from {ckpt_dir}")
    print(f"{'='*90}")

    # Load checkpoint
    pkg = _load_checkpoint(ckpt_dir)
    meta_state_cpu = {k: v.cpu() for k, v in pkg["model"].items()}

    # Inject model into MetaST module
    MetaST.model = wrapped_model
    device = next(wrapped_model.parameters()).device

    results = {}
    _legacy_proto_warned = False
    for city_key, cfg in city_configs.items():
        city_name = cfg["name"]
        core = cores[city_key]
        role = cfg.get("role", "unknown")

        print(f"\n  --- {city_name} ({role}) ---")

        # Load meta-learned θ₀
        load_state_cpu(wrapped_model, meta_state_cpu, device)
        set_globals_from_core(core)

        # [2026-06-06] Aligned transfer protocol: adapt up to adapt_steps with
        # early stopping (patience on val route_mape), chaining inner_adapt_steps
        # in chunks of eval_every. Same budget/patience as MetaTTE/MetaETA;
        # MetaST keeps its own SGD optimizer + adapt_lr.
        cur_state = meta_state_cpu
        best_state_es = meta_state_cpu
        best_val = float("inf")
        no_improve = 0
        _chunk = max(1, eval_every)
        _n_rounds = max(1, adapt_steps // _chunk)
        for _r in range(_n_rounds):
            cur_state, _support = inner_adapt_steps(
                cur_state, core, inner_steps=_chunk, inner_lr=adapt_lr,
                w_speed=w_speed, w_link=w_link, w_route=w_route,
            )
            load_state_cpu(wrapped_model, cur_state, device)
            wrapped_model.eval()
            with torch.no_grad():
                _vm = one_epoch(core, core["loaders"]["val"], epoch=0, train=False,
                                w_speed=w_speed, w_link=w_link, w_route=w_route)
            _vr = _vm.get("route_mape", float("inf"))
            if math.isnan(_vr):
                _vr = float("inf")
            if _vr < best_val:
                best_val = _vr
                best_state_es = cur_state
                no_improve = 0
            else:
                no_improve += 1
            # [2026-06-07] per-round adaptation progress (so improvement is visible).
            _step_done = (_r + 1) * _chunk
            print(f"    [adapt] MetaST  {city_name}  step {_step_done:>4}/{adapt_steps}  "
                  f"val route_mape={_vr:.4f}  link_mape={_vm.get('link_mape', float('nan')):.4f}  "
                  f"{'<- best' if no_improve == 0 else f'(no_improve {no_improve}/{patience})'}")
            if no_improve >= patience:
                print(f"    [adapt] MetaST  {city_name}  early stop @ step {_step_done}  "
                      f"best val route_mape={best_val:.4f}")
                break

        # Load best adapted weights and evaluate
        load_state_cpu(wrapped_model, best_state_es, device)
        wrapped_model.to(device)
        wrapped_model.eval()

        # [2026-06-06] Save ADAPTED (transfer) checkpoint for benchmark/testing.
        if transfer_ckpt_dir is not None:
            _mgr = CheckpointManager(dirpath=transfer_ckpt_dir, max_keep=3)
            _mgr.save_best(model=wrapped_model, optimizer=None, scheduler=None,
                           epoch=0, step=adapt_steps, best_metric=best_val,
                           extra={"method": "MetaST", "target": city_key})
            print(f"    [saved adapted ckpt] {transfer_ckpt_dir}")

        city_results = {}
        trip_frames, link_frames = [], []
        with torch.no_grad():
            for split_name in ("val", "test"):
                loader = core["loaders"][split_name]
                m = one_epoch(core, loader, epoch=0, train=False,
                              w_speed=w_speed, w_link=w_link, w_route=w_route)
                city_results[split_name] = m

                print(f"    {split_name}: "
                      f"Link  MAE={m['link_mae']:.4f}  RMSE={m['link_rmse']:.4f}  MAPE={m['link_mape']:.4f}  |  "
                      f"Route MAE={m['route_mae']:.4f}  RMSE={m['route_rmse']:.4f}  MAPE={m['route_mape']:.4f}  |  "
                      f"speed_mae={m.get('speed_mae', float('nan')):.4f}  vol_mae={m.get('vol_mae', float('nan')):.4f}")

                # Collect per-trip + per-link predictions for downstream saving by main.py.
                trip_df_s, link_df_s = _collect_linketa_preds(core, loader, "MetaST")
                trip_df_s["split"] = split_name
                trip_df_s["city"]  = city_key
                link_df_s["split"] = split_name
                link_df_s["city"]  = city_key
                trip_frames.append(trip_df_s)
                link_frames.append(link_df_s)

        city_results["trip_df"] = (
            pd.concat(trip_frames, ignore_index=True) if trip_frames else pd.DataFrame())
        city_results["link_df"] = (
            pd.concat(link_frames, ignore_index=True) if link_frames else pd.DataFrame())
        results[city_name] = city_results

    _print_table("MetaST — All Cities Evaluation", results)
    return results


# =====================================================================
# MetaTTE Evaluation
# =====================================================================

def transfer_to_target_metatte(
    cores: Dict[str, dict],
    city_configs: Dict[str, dict],
    base_model: nn.Module,
    ckpt_dir: str = "ckpts_MetaETA_MetaTTE",
    transfer_ckpt_dir: str = None,        # [2026-06-06] where to save the ADAPTED ckpt
    adapt_steps: int = 500,               # aligned budget cap
    adapt_lr: float = 1e-5,               # MetaTTE keeps Adam @ 1e-5 (Reptile-faithful)
    patience: int = 10,                   # aligned early-stopping patience
    eval_every: int = 10,                 # aligned val cadence
    w_speed: float = 1.0 / 3.0,           # 3-task {speed,link,route}
    w_link: float = 1.0 / 3.0,
    w_route: float = 1.0 / 3.0,
    use_adam: bool = True,
    adam_betas: Tuple[float, float] = (0.0, 0.999),
    collect_trip_preds: bool = False,
    trip_preds_dir: str = "figs",
) -> Dict[str, dict]:
    """
    Evaluate MetaTTE checkpoint on ALL cities (source + target).

    For each city:
      1. Load meta-learned θ₀ from best.ckpt
      2. Run adapt_steps gradient steps on the city's train set (Reptile inner loop)
      3. Evaluate on val and test sets

    Parameters
    ----------
    cores : dict
        {city_key: core_dict} for all cities.
    city_configs : dict
        CITY_CONFIGS dict with city metadata.
    base_model : nn.Module
        The base LinkETA model (same architecture used during training).
    ckpt_dir : str
        Directory containing MetaTTE checkpoints.
    adapt_steps : int
        Number of inner adaptation steps per city.
    adapt_lr : float
        Learning rate for adaptation.
    collect_trip_preds : bool
        If True, save per-trip predictions to CSV after evaluating each split.
    trip_preds_dir : str
        Directory for trip prediction CSVs (default "figs").

    Returns
    -------
    dict: {city_name: {"val": metrics, "test": metrics, "trip_df": df, "link_df": df}}
        trip_df / link_df cover BOTH splits, tagged with `split` + `city` so
        main.py can concat across methods.
    """
    import MetaTTE as MetaTTE
    from MetaTTE import (
        set_globals_from_core, reptile_inner_steps,
        clone_state_cpu, load_state_cpu,
    )

    print(f"\n{'='*90}")
    print(f"  MetaTTE Evaluation — Loading from {ckpt_dir}")
    print(f"{'='*90}")

    # Load checkpoint
    pkg = _load_checkpoint(ckpt_dir)
    meta_state_cpu = {k: v.cpu() for k, v in pkg["model"].items()}

    # Inject model into MetaTTE module
    MetaTTE.model = base_model
    device = next(base_model.parameters()).device
    MetaTTE.device = device

    results = {}
    _legacy_proto_warned = False
    for city_key, cfg in city_configs.items():
        city_name = cfg["name"]
        core = cores[city_key]
        role = cfg.get("role", "unknown")

        print(f"\n  --- {city_name} ({role}) ---")

        # Load meta-learned θ₀
        load_state_cpu(base_model, meta_state_cpu, device)
        set_globals_from_core(core, keep_model=True)

        # [2026-06-06] Aligned transfer protocol: adapt up to adapt_steps with early
        # stopping (patience on val route_mape), chaining reptile_inner_steps in
        # chunks of eval_every. Same budget/patience as MetaST/MetaETA; MetaTTE keeps
        # its Reptile-faithful Adam(b1=0) optimizer + adapt_lr.
        cur_state = meta_state_cpu
        best_state_es = meta_state_cpu
        best_val = float("inf")
        no_improve = 0
        _chunk = max(1, eval_every)
        _n_rounds = max(1, adapt_steps // _chunk)
        for _r in range(_n_rounds):
            load_state_cpu(base_model, cur_state, device)
            inner_result = reptile_inner_steps(
                core=core, inner_steps=_chunk, inner_lr=adapt_lr,
                w_speed=w_speed, w_link=w_link, w_route=w_route,
                use_adam=use_adam, adam_betas=adam_betas,
            )
            cur_state = inner_result["adapted_state"]
            load_state_cpu(base_model, cur_state, device)
            base_model.eval()
            with torch.no_grad():
                _vm = one_epoch(core, core["loaders"]["val"], epoch=0, train=False,
                                w_speed=w_speed, w_link=w_link, w_route=w_route)
            _vr = _vm.get("route_mape", float("inf"))
            if math.isnan(_vr):
                _vr = float("inf")
            if _vr < best_val:
                best_val = _vr
                best_state_es = cur_state
                no_improve = 0
            else:
                no_improve += 1
            # [2026-06-07] per-round adaptation progress (so improvement is visible).
            _step_done = (_r + 1) * _chunk
            print(f"    [adapt] MetaTTE {city_name}  step {_step_done:>4}/{adapt_steps}  "
                  f"val route_mape={_vr:.4f}  link_mape={_vm.get('link_mape', float('nan')):.4f}  "
                  f"{'<- best' if no_improve == 0 else f'(no_improve {no_improve}/{patience})'}")
            if no_improve >= patience:
                print(f"    [adapt] MetaTTE {city_name}  early stop @ step {_step_done}  "
                      f"best val route_mape={best_val:.4f}")
                break

        # Load best adapted weights and evaluate
        load_state_cpu(base_model, best_state_es, device)
        base_model.to(device)
        base_model.eval()

        # [2026-06-06] Save ADAPTED (transfer) checkpoint for benchmark/testing.
        if transfer_ckpt_dir is not None:
            _mgr = CheckpointManager(dirpath=transfer_ckpt_dir, max_keep=3)
            _mgr.save_best(model=base_model, optimizer=None, scheduler=None,
                           epoch=0, step=adapt_steps, best_metric=best_val,
                           extra={"method": "MetaTTE", "target": city_key})
            print(f"    [saved adapted ckpt] {transfer_ckpt_dir}")

        city_results = {}
        trip_frames, link_frames = [], []
        with torch.no_grad():
            for split_name in ("val", "test"):
                loader = core["loaders"][split_name]
                m = one_epoch(core, loader, epoch=0, train=False,
                              w_speed=w_speed, w_link=w_link, w_route=w_route)
                city_results[split_name] = m

                print(f"    {split_name}: "
                      f"Link  MAE={m['link_mae']:.4f}  RMSE={m['link_rmse']:.4f}  MAPE={m['link_mape']:.4f}  |  "
                      f"Route MAE={m['route_mae']:.4f}  RMSE={m['route_rmse']:.4f}  MAPE={m['route_mape']:.4f}  |  "
                      f"speed_mae={m.get('speed_mae', float('nan')):.4f}  vol_mae={m.get('vol_mae', float('nan')):.4f}")

                trip_df_s, link_df_s = _collect_linketa_preds(core, loader, "MetaTTE")
                trip_df_s["split"] = split_name
                trip_df_s["city"]  = city_key
                link_df_s["split"] = split_name
                link_df_s["city"]  = city_key
                trip_frames.append(trip_df_s)
                link_frames.append(link_df_s)

        city_results["trip_df"] = (
            pd.concat(trip_frames, ignore_index=True) if trip_frames else pd.DataFrame())
        city_results["link_df"] = (
            pd.concat(link_frames, ignore_index=True) if link_frames else pd.DataFrame())
        results[city_name] = city_results

    _print_table("MetaTTE — All Cities Evaluation", results)
    return results


# =====================================================================
# MetaTraffic Evaluation
# =====================================================================

def transfer_to_target_metatraffic(
    cores: Dict[str, dict],
    city_configs: Dict[str, dict],
    meta_traffic_model,       # MetaTrafficModel (same architecture)
    linketa_model: nn.Module, # LinkETA model (same architecture)
    city_static_feats: Dict[str, torch.Tensor],
    city_traffic_real: Dict[str, torch.Tensor],
    city_rel_coo: Dict[str, list],
    city_scaler_mean: Dict[str, torch.Tensor],
    city_scaler_std: Dict[str, torch.Tensor],
    city_vff: Dict[str, torch.Tensor],
    city_dyn_idx: Dict[str, torch.Tensor],
    city_time_idx: Dict[str, list],
    ckpt_dir: str = "ckpts_MetaETA",
    transfer_ckpt_dir: str = None,  # [2026-06-06] where to save the ADAPTED ckpt
    adapt_steps: int = 60,
    adapt_lr: float = 1e-3,               # [2026-06-09] match meta-training inner_lr (was 1e-5)
    adapt_lr_generator: Optional[float] = None,  # [2026-06-09] per-group LR for the MetaTraffic GENERATOR (param_gen/GAT/refine/output head/alpha); None -> = adapt_lr
    adapt_lr_backbone: Optional[float] = None,  # [2026-06-09] per-group LR for the LinkETA backbone; None -> = adapt_lr
    recalibrate_output_range: bool = True,      # [2026-06-09] re-fit traffic head output_scale/shift to TARGET city traffic
    alpha_init_logit: Optional[float] = 0.0,    # [2026-06-09] reset proto_alpha_logit at transfer so the prototype<->W_gen gate is LEARNED from a neutral prior (0.0 -> alpha=0.5); None -> keep theta0's (unlearned 0.88)
    dwa_T: float = 2.0,                   # DWA temperature (0 = disable)
    dwa_eval_every: int = 10,              # DWA recompute + early-stopping eval interval
    patience: int = 10,                    # early stopping patience (in eval intervals)
    # --- BUG 2 FIX: Structural cluster transfer (optional) ---
    source_static_feats: Optional[Dict[str, torch.Tensor]] = None,
    source_rel_coo: Optional[Dict[str, list]] = None,
    source_clusters: Optional[Dict[str, dict]] = None,
    city_clusters: Optional[Dict[str, dict]] = None,
    source_city_descriptors: Optional[Dict[str, torch.Tensor]] = None,
    city_descriptors: Optional[Dict[str, torch.Tensor]] = None,
    K: int = 7,
    S: int = 9,
    disable_prototype: bool = False,   # [2026-06-07] diagnostic: skip prototype bank (pure W_gen)
    collect_trip_preds: bool = False,
    trip_preds_dir: str = "figs",
) -> Dict[str, dict]:
    """
    Evaluate MetaTraffic checkpoint on ALL cities (source + target).

    For each city:
      1. Load meta-learned θ₀ (MetaTraffic + LinkETA) from best.ckpt
      2. Run adapt_steps gradient steps on the city's train set
      3. Evaluate on val and test sets using MetaTraffic predictions

    Parameters
    ----------
    cores : dict
        {city_key: core_dict} for all cities.
    city_configs : dict
        CITY_CONFIGS dict with city metadata.
    meta_traffic_model : MetaTrafficModel
        MetaTrafficModel (same architecture used during training).
    linketa_model : LinkETA
        LinkETA model (same architecture used during training).
    city_* : dict
        Per-city data tensors (same as used during training).
    ckpt_dir : str
        Directory containing MetaTraffic checkpoints.
    adapt_steps : int
        Number of inner adaptation steps per city.
    adapt_lr : float
        Learning rate for adaptation.
    source_static_feats : dict, optional
        Source city static feats for prototype bank building.
    source_rel_coo : dict, optional
        Source city COO tensors for prototype bank building.
    source_clusters : dict, optional
        Source city cluster assignments {city_key: {'k_labels', 's_labels'}}.
    city_clusters : dict, optional
        Per-city cluster assignments {city_key: {'k_labels', 's_labels'}}.
    source_city_descriptors : dict, optional
        {city_key: descriptor tensor} for city-similarity weighting.
    city_descriptors : dict, optional
        {city_key: descriptor tensor} for all evaluation cities.
    K, S : int
        Number of physical / topology clusters.

    Returns
    -------
    dict: {city_name: {"val": metrics, "test": metrics, "trip_df": df, "link_df": df}}
        trip_df / link_df cover BOTH splits, tagged with `split` + `city`.
    """
    from MetaTraffic import (
        metatraffic_one_epoch, _CombinedModel,
        extract_per_trip_time_slot, compute_per_trip_traffic_loss,
        USE_AMP_MT,
    )

    print(f"\n{'='*90}")
    print(f"  MetaTraffic Evaluation — Loading from {ckpt_dir}")
    print(f"{'='*90}")

    # Load checkpoint (saved via _CombinedModel wrapper)
    pkg = _load_checkpoint(ckpt_dir)

    # BUG 1 FIX: Validate scaler fingerprint (detect stale scaler files)
    saved_hash = pkg.get("extra", {}).get("scaler_hash")
    current_hash = _scaler_fingerprint(city_scaler_mean, city_scaler_std)
    if saved_hash:
        if saved_hash != current_hash:
            print(f"  WARNING: Scaler mismatch! Checkpoint trained with hash={saved_hash}, "
                  f"current={current_hash}. Results may be unreliable.")
        else:
            print(f"  Scaler fingerprint OK: {current_hash}")
    else:
        print(f"  INFO: Checkpoint has no scaler_hash in extra. "
              f"Current hash={current_hash} (cannot verify consistency)")

    combined_state = pkg["model"]

    # Split combined state back into MetaTraffic + LinkETA
    mt_state = {}
    le_state = {}
    for k, v in combined_state.items():
        if k.startswith("meta_traffic."):
            mt_state[k[len("meta_traffic."):]] = v
        elif k.startswith("linketa."):
            le_state[k[len("linketa."):]] = v
        else:
            # Fallback: try to identify by prefix
            print(f"  Warning: unexpected key '{k}' in checkpoint")

    device = next(meta_traffic_model.parameters()).device

    # Store the meta-learned θ₀ for reloading per city
    meta_mt_state = {k: v.cpu().clone() for k, v in mt_state.items()}
    meta_le_state = {k: v.cpu().clone() for k, v in le_state.items()}

    # BUG 6 FIX: Pre-register normalization for ALL cities upfront
    # (_per_city_norm is a plain dict not in state_dict — must be rebuilt)
    for ck, sf in city_static_feats.items():
        meta_traffic_model.register_city_norm(ck, sf)
    if hasattr(meta_traffic_model, "validate_city_norm_ready"):
        meta_traffic_model.validate_city_norm_ready(list(city_static_feats.keys()))

    results = {}
    _legacy_proto_warned = False
    for city_key, cfg in city_configs.items():
        city_name = cfg["name"]
        core = cores[city_key]
        role = cfg.get("role", "unknown")

        print(f"\n  --- {city_name} ({role}) ---")

        # Ensure transient prototype buffers do not make strict checkpoint loading
        # fail when evaluating legacy checkpoints.
        if hasattr(meta_traffic_model, "clear_prototypes"):
            meta_traffic_model.clear_prototypes()

        # Reload meta-learned theta_0 (fresh for each city)
        mt_incompat = meta_traffic_model.load_state_dict(
            {k: v.to(device) for k, v in meta_mt_state.items()}, strict=False)
        allowed_missing = {
            "proto_W",
            "proto_b",
            "target_k_labels",
            "target_s_labels",
        }
        bad_missing = [k for k in mt_incompat.missing_keys if k not in allowed_missing]
        if bad_missing or mt_incompat.unexpected_keys:
            raise RuntimeError(
                "MetaTraffic checkpoint/model mismatch. "
                f"missing={bad_missing}, unexpected={mt_incompat.unexpected_keys}"
            )
        if (not _legacy_proto_warned) and any(k in allowed_missing for k in mt_incompat.missing_keys):
            print("    INFO: Legacy checkpoint detected (prototype buffers absent); "
                  "loading base MetaTraffic weights only.")
            _legacy_proto_warned = True

        linketa_model.load_state_dict(
            {k: v.to(device) for k, v in meta_le_state.items()}, strict=True)

        # Per-city data
        static_feats = city_static_feats[city_key]
        traffic_real = city_traffic_real[city_key]
        rel_coo = city_rel_coo[city_key]
        scaler_mean = city_scaler_mean[city_key]
        scaler_std = city_scaler_std[city_key]
        vff_city = city_vff[city_key]
        dyn_idx = city_dyn_idx[city_key]
        time_idx = city_time_idx[city_key]
        n_edges = static_feats.shape[0]

        # Register + activate per-city static feature normalization
        # (mirrors transfer_to_target lines 2165-2166 in MetaTraffic.py)
        # Use city_key (lowercase) consistently — metatraffic_one_epoch
        # receives the same key for its city_name arg.
        meta_traffic_model.register_city_norm(city_key, static_feats)
        meta_traffic_model.set_city_norm(city_key)

        # BUG 2 FIX: Inject prototype bank if cluster data is available
        # (mirrors transfer_to_target lines 2168-2195 in MetaTraffic.py)
        # [2026-06-07] disable_prototype skips this entirely -> pure W_gen forward
        # (diagnostic: isolates the prototype-injection effect on transfer).
        if disable_prototype:
            if hasattr(meta_traffic_model, "clear_prototypes"):
                meta_traffic_model.clear_prototypes()
            print("    [no-prototype] prototype bank DISABLED (pure W_gen, alpha=0)")
        elif (source_clusters is not None and city_clusters is not None
                and city_key in city_clusters
                and source_static_feats is not None and source_rel_coo is not None):
            city_weights = None
            if (source_city_descriptors is not None
                    and city_descriptors is not None
                    and city_key in city_descriptors):
                from MetaTraffic import compute_city_similarity_weights
                city_weights = compute_city_similarity_weights(
                    source_city_descriptors, city_descriptors[city_key])
            meta_traffic_model.build_prototype_bank(
                city_static_feats=source_static_feats,
                city_rel_coo=source_rel_coo,
                city_clusters=source_clusters,
                K=K, S=S, city_weights=city_weights,
            )
            meta_traffic_model.set_target_clusters(
                k_labels=city_clusters[city_key]['k_labels'],
                s_labels=city_clusters[city_key]['s_labels'],
            )
            alpha = float(torch.sigmoid(meta_traffic_model.proto_alpha_logit).detach().cpu())
            print(f"    Prototype bank injected (alpha={alpha:.4f})")
        else:
            if hasattr(meta_traffic_model, "clear_prototypes"):
                meta_traffic_model.clear_prototypes()

        # [2026-06-09] Re-assert the TARGET city's static normalization. When the
        # prototype bank is built above, build_prototype_bank() iterates the source
        # cities and calls set_city_norm(source) internally, leaving static_mean/std
        # on the LAST source city. The adapt-loop forward (encode_structure /
        # predict_traffic) must use the TARGET's normalization from step 1.
        meta_traffic_model.set_city_norm(city_key)

        # [2026-06-09] FIX (root cause of ~100% target MAPE): re-calibrate the
        # traffic head's output range to the TARGET city. output_scale/output_shift
        # arrive from theta_0, fit on SOURCE cities only (main.py:
        # init_output_range(source_traffic)). A target whose traffic magnitude
        # differs (e.g. Chengdu speeds/counts >> porto/sf) gets clipped by the
        # source-range sigmoid head, so predicted traffic is many sigma off ->
        # wrong sp_hat -> wrong ETA. Re-fitting here lets the head emit
        # target-scale traffic (diagnostic: Chengdu speed z-RMSE 8.5 -> 1.6).
        if recalibrate_output_range:
            meta_traffic_model.init_output_range({city_key: traffic_real}, margin=0.1)

        # [2026-06-09] LEARNED prototype<->generation gate. theta_0 carries an
        # UNLEARNED proto_alpha_logit (hardcoded init 2.0 -> alpha=0.88; the
        # prototype is OFF during meta-training, so the gate never received a
        # gradient). Reset it here to a neutral prior so the gate is genuinely
        # learned per target from the data, rather than defended as a magic 0.88
        # (alpha_init_logit=0.0 -> alpha=0.5; None -> keep theta_0's value). The
        # generator LR group below gives the gate real leverage to move.
        if (alpha_init_logit is not None
                and hasattr(meta_traffic_model, "proto_alpha_logit")
                and meta_traffic_model.proto_W is not None):
            with torch.no_grad():
                meta_traffic_model.proto_alpha_logit.fill_(float(alpha_init_logit))
            print(f"    [alpha] prototype gate reset to neutral prior alpha="
                  f"{float(torch.sigmoid(meta_traffic_model.proto_alpha_logit)):.4f} "
                  f"(logit={float(alpha_init_logit):.2f}); learned during adaptation")

        # Adapt on train set (mirrors transfer_to_target lines 2214-2229)
        meta_traffic_model.train()
        linketa_model.train()
        # [2026-06-09] Three-way per-group LR. The MetaTraffic GENERATOR (param_gen /
        # GAT / refine / output head / proto_alpha_logit) is the transfer bottleneck
        # (predicted traffic must specialise to the target) and gets
        # adapt_lr_generator; the LinkETA BACKBONE gets adapt_lr_backbone. Both
        # default (None) to adapt_lr == meta-training inner_lr -> uniform unless
        # tuned. Raising adapt_lr_generator lets the predicted traffic + alpha gate
        # specialise to the target faster without forcing the backbone to that rate.
        _lr_generator = adapt_lr if adapt_lr_generator is None else adapt_lr_generator
        _lr_backbone = adapt_lr if adapt_lr_backbone is None else adapt_lr_backbone
        _mt_params = list(meta_traffic_model.parameters())
        _le_params = list(linketa_model.parameters())
        adapt_params = _mt_params + _le_params  # flat list for grad-norm clipping
        adapt_opt = torch.optim.SGD([
            {"params": _mt_params, "lr": _lr_generator},
            {"params": _le_params, "lr": _lr_backbone},
        ])

        traffic_mask = (traffic_real.abs().sum(dim=(-1, -2, -3)) > 0)
        dyn_scaler_mean = scaler_mean[dyn_idx]
        dyn_scaler_std = scaler_std[dyn_idx]

        train_loader = core["loaders"]["train"]
        train_iter = iter(train_loader)

        # [2026-06-06] DWA state — 4-task {speed,link,route,traffic}, init 1/4 each
        _dwa_hist_speed = deque(maxlen=2)
        _dwa_hist_link = deque(maxlen=2)
        _dwa_hist_route = deque(maxlen=2)
        _dwa_hist_traffic = deque(maxlen=2)
        _dwa_enabled = (dwa_T > 0)
        _w_speed = 1.0 / 4.0
        _w_link = 1.0 / 4.0
        _w_route = 1.0 / 4.0
        _w_traffic = 1.0 / 4.0
        _dwa_sum_s = 0.0
        _dwa_sum_l = 0.0
        _dwa_sum_r = 0.0
        _dwa_sum_t = 0.0
        _dwa_count = 0

        # Early stopping state (matching transfer_to_target lines 2248-2253)
        best_val_loss = float('inf')
        best_state = None
        no_improve = 0
        stopped_early = False
        # [2026-06-07] RESTORED: per-component first-eval baselines so the 4-task
        # early-stop composite is scale-free (each component ~1.0). Without this the
        # raw traffic_rmse (~3) dominated route_mape (~0.18) and selected bad-route
        # checkpoints. (Matches the original transfer_to_target normalization.)
        _es_baseline_speed = None
        _es_baseline_link = None
        _es_baseline_route = None
        _es_baseline_traffic = None

        for step in range(adapt_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            # 0) Encode structure once (matching transfer_to_target line 2284)
            with torch.amp.autocast('cuda', enabled=USE_AMP_MT):
                h_struct = meta_traffic_model.encode_structure(
                    static_feats, rel_coo)

            # 1) Per-node traffic prediction (matching transfer_to_target lines 2288-2300)
            node_edge_idx = batch["node_edge_idx"]
            x_seq = batch["x_seq"].clone()
            B_t, M_t, _ = x_seq.shape

            flat_edge_idx = node_edge_idx.reshape(-1).clamp(0, n_edges - 1)
            flat_time = x_seq.reshape(B_t * M_t, -1)[:, time_idx]

            with torch.amp.autocast('cuda', enabled=USE_AMP_MT):
                predicted_traffic_flat, _ = meta_traffic_model.predict_traffic(
                    h_struct, static_feats, flat_time, flat_edge_idx)
            predicted_traffic_flat = torch.nan_to_num(
                predicted_traffic_flat, nan=0.0, posinf=1e4, neginf=-1e4)

            # 2) Inject into x_seq
            pred_traffic_batch = predicted_traffic_flat.reshape(B_t, M_t, -1)
            x_seq[:, :, dyn_idx] = pred_traffic_batch.to(x_seq.dtype)

            # 3) LinkETA forward (matching transfer_to_target lines 2307-2336)
            masks_per_rel = [m.to(device) for m in batch["masks_per_rel"]]
            route_seq_idx = batch["route_seq_idx"]
            seq_true_tt = torch.nan_to_num(batch["seq_true_tt"], nan=0.0)
            seq_link_len = torch.nan_to_num(batch["seq_link_len"], nan=0.0).clamp_min(1e-3)
            seq_mask = batch["seq_mask"].bool()
            trip_tt = torch.nan_to_num(batch["trip_tt"], nan=0.0)
            hop_idx = batch["hop_idx"]
            route_flag = batch["route_flag"]

            sp_true_node = torch.nan_to_num(batch["sp_ratio_agg"], nan=0.0)
            lc_true_node = torch.nan_to_num(batch["logcnt1h_node"], nan=0.0)
            sp_w_node = torch.nan_to_num(batch["speed_w_node"], nan=0.0)
            _gather = lambda v: torch.gather(v, 1, route_seq_idx)
            sp_true = _gather(sp_true_node)
            lc_true = _gather(lc_true_node)
            sp_w = _gather(sp_w_node)

            vff_seq_b = vff_city[node_edge_idx]
            vff_seq_b = torch.gather(vff_seq_b, 1, route_seq_idx)

            x_seq_norm = (x_seq - scaler_mean.view(1, 1, -1)
                          ) / scaler_std.view(1, 1, -1)
            sp_mask = (sp_w > 0) & seq_mask

            link_eta_seq, trip_eta, sp_pred, lc_pred, g, delta, eta_phys = linketa_model(
                x_seq_norm, hop_idx, route_flag, masks_per_rel,
                route_seq_idx, seq_link_len, vff_seq_b, seq_mask,
                r_max=3.0, return_parts=True,
            )
            link_eta_seq = torch.nan_to_num(link_eta_seq, nan=0.0)
            sp_pred = torch.nan_to_num(sp_pred, nan=0.0)
            lc_pred = torch.nan_to_num(lc_pred, nan=0.0)

            # 4) Losses — [2026-06-06] log-space + sp_actual, identical to base LinkETA
            #    + meta-training (was old huber_macro/mape + 0.05*speed + lam_g).
            pred_seq = link_eta_in_seconds(link_eta_seq, seq_link_len)
            loss_link = link_loss_huber_log_lenweighted(
                pred_seq, seq_true_tt, seq_mask, link_len=seq_link_len, delta=0.3)
            loss_route = route_loss_huber_log(pred_seq, trip_tt, seq_mask, delta=0.3)

            loss_speed = _backbone_speed_loss(
                sp_pred, seq_link_len, seq_true_tt, vff_seq_b, seq_mask)

            if seq_mask.any():
                loss_vol = F.smooth_l1_loss(lc_pred[seq_mask], lc_true[seq_mask])
            else:
                loss_vol = torch.zeros((), device=device)

            trip_days, trip_hours = extract_per_trip_time_slot(
                batch["x_seq"], route_seq_idx, seq_mask, time_idx)
            loss_traffic = compute_per_trip_traffic_loss(
                pred_traffic_batch, node_edge_idx, traffic_real,
                trip_days, trip_hours, route_seq_idx, seq_mask,
                traffic_mask,
                dyn_scaler_mean=dyn_scaler_mean,
                dyn_scaler_std=dyn_scaler_std,
                n_edges=n_edges,
            )

            # 4-task DWA {speed,link,route,traffic} + fixed 0.10*vol; lam_g dropped.
            loss = (_w_speed * loss_speed + _w_link * loss_link + _w_route * loss_route
                    + _w_traffic * loss_traffic + 0.10 * loss_vol)

            adapt_opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(adapt_params, 10.0)
            adapt_opt.step()

            # DWA: accumulate and recompute
            _dwa_sum_s += float(loss_speed.detach())
            _dwa_sum_l += float(loss_link.detach())
            _dwa_sum_r += float(loss_route.detach())
            _dwa_sum_t += float(loss_traffic.detach())
            _dwa_count += 1

            if (step + 1) % dwa_eval_every == 0 and _dwa_count > 0:
                _dwa_hist_speed.append(_dwa_sum_s / _dwa_count)
                _dwa_hist_link.append(_dwa_sum_l / _dwa_count)
                _dwa_hist_route.append(_dwa_sum_r / _dwa_count)
                _dwa_hist_traffic.append(_dwa_sum_t / _dwa_count)
                _dwa_sum_s = 0.0
                _dwa_sum_l = 0.0
                _dwa_sum_r = 0.0
                _dwa_sum_t = 0.0
                _dwa_count = 0

                if _dwa_enabled and len(_dwa_hist_link) >= 2:
                    r_speed = float(_dwa_hist_speed[-1] / (_dwa_hist_speed[-2] + 1e-8))
                    r_link = float(_dwa_hist_link[-1] / (_dwa_hist_link[-2] + 1e-8))
                    r_route = float(_dwa_hist_route[-1] / (_dwa_hist_route[-2] + 1e-8))
                    r_traffic = float(_dwa_hist_traffic[-1] / (_dwa_hist_traffic[-2] + 1e-8))
                    # Clamp DWA ratios to prevent exp() overflow
                    _DWA_RATIO_MAX = 10.0
                    r_speed = max(min(r_speed, _DWA_RATIO_MAX), -_DWA_RATIO_MAX)
                    r_link = max(min(r_link, _DWA_RATIO_MAX), -_DWA_RATIO_MAX)
                    r_route = max(min(r_route, _DWA_RATIO_MAX), -_DWA_RATIO_MAX)
                    r_traffic = max(min(r_traffic, _DWA_RATIO_MAX), -_DWA_RATIO_MAX)
                    e_s = math.exp(r_speed / dwa_T)
                    e_l = math.exp(r_link / dwa_T)
                    e_r = math.exp(r_route / dwa_T)
                    e_t = math.exp(r_traffic / dwa_T)
                    s = e_s + e_l + e_r + e_t
                    _w_speed = e_s / s
                    _w_link = e_l / s
                    _w_route = e_r / s
                    _w_traffic = e_t / s

            # Early stopping: periodic val evaluation (matching transfer_to_target lines 2436-2496)
            if (step + 1) % dwa_eval_every == 0:
                meta_traffic_model.eval()
                linketa_model.eval()
                with torch.no_grad():
                    val_m = metatraffic_one_epoch(
                        loader=core["loaders"]["val"],
                        meta_traffic_model=meta_traffic_model,
                        linketa_model=linketa_model,
                        optimizer=adapt_opt,
                        static_feats=static_feats,
                        traffic_real=traffic_real,
                        rel_coo=rel_coo,
                        time_col_indices=time_idx,
                        dyn_col_indices=dyn_idx,
                        scaler_mean=scaler_mean,
                        scaler_std=scaler_std,
                        vff_vec_city=vff_city,
                        epoch=0,
                        train=False,
                        w_speed=_w_speed,
                        w_link=_w_link,
                        w_route=_w_route,
                        w_traffic=_w_traffic,
                        city_name=city_key,
                    )
                val_speed_loss = val_m.get('speed_loss', float('inf'))
                val_link_loss = val_m.get('link_loss', float('inf'))
                val_route_mape = val_m.get('route_mape', float('inf'))
                val_trmse = val_m.get('traffic_rmse', float('inf'))

                # Guard NaN — treat as 0 contribution
                if math.isnan(val_speed_loss):
                    val_speed_loss = 0.0
                    print(f"    Warning: speed_loss is NaN for {city_name}, excluding from early stopping")
                if math.isnan(val_link_loss):
                    val_link_loss = 0.0
                    print(f"    Warning: link_loss is NaN for {city_name}, excluding from early stopping")
                if math.isnan(val_route_mape):
                    val_route_mape = 0.0
                    print(f"    Warning: route_mape is NaN for {city_name}, excluding from early stopping")
                if math.isnan(val_trmse):
                    val_trmse = 0.0
                    print(f"    Warning: traffic_rmse is NaN for {city_name}, excluding from early stopping")

                # [2026-06-07] RESTORED per-component baseline normalization: record
                # each metric's FIRST-eval value, then divide later values by it so
                # all components are ~1.0 scale and no single metric (traffic_rmse)
                # can dominate the DWA-weighted selection criterion.
                if _es_baseline_speed is None and not math.isinf(val_speed_loss) and val_speed_loss > 0:
                    _es_baseline_speed = max(val_speed_loss, 1e-6)
                if _es_baseline_link is None and not math.isinf(val_link_loss) and val_link_loss > 0:
                    _es_baseline_link = max(val_link_loss, 1e-6)
                if _es_baseline_route is None and not math.isinf(val_route_mape) and val_route_mape > 0:
                    _es_baseline_route = max(val_route_mape, 1e-6)
                if _es_baseline_traffic is None and not math.isinf(val_trmse) and val_trmse > 0:
                    _es_baseline_traffic = max(val_trmse, 1e-6)

                norm_speed = val_speed_loss / _es_baseline_speed if _es_baseline_speed else val_speed_loss
                norm_link = val_link_loss / _es_baseline_link if _es_baseline_link else val_link_loss
                norm_route = val_route_mape / _es_baseline_route if _es_baseline_route else val_route_mape
                norm_traffic = val_trmse / _es_baseline_traffic if _es_baseline_traffic else 0.0

                val_dwa_loss = (_w_speed * norm_speed
                                + _w_link * norm_link
                                + _w_route * norm_route
                                + _w_traffic * norm_traffic)

                # BUG 4b FIX: Always save first eval as fallback
                if best_state is None:
                    best_val_loss = val_dwa_loss
                    best_state = {
                        'mt': {k: v.detach().cpu().clone()
                               for k, v in meta_traffic_model.state_dict().items()},
                        'le': {k: v.detach().cpu().clone()
                               for k, v in linketa_model.state_dict().items()},
                    }
                    no_improve = 0
                    print(f"    Saved initial best state (fallback): dwa_loss={val_dwa_loss:.6f}")
                elif val_dwa_loss < best_val_loss:
                    best_val_loss = val_dwa_loss
                    best_state = {
                        'mt': {k: v.detach().cpu().clone()
                               for k, v in meta_traffic_model.state_dict().items()},
                        'le': {k: v.detach().cpu().clone()
                               for k, v in linketa_model.state_dict().items()},
                    }
                    no_improve = 0
                else:
                    no_improve += 1
                # [2026-06-07] per-round adaptation progress (so improvement is visible).
                print(f"    [adapt] MetaETA {city_name}  step {step+1:>4}/{adapt_steps}  "
                      f"val route_mape={val_route_mape:.4f}  link_loss={val_link_loss:.4f}  "
                      f"traffic_rmse={val_trmse:.4f}  dwa_loss={val_dwa_loss:.6f}  "
                      f"{'<- best' if no_improve == 0 else f'(no_improve {no_improve}/{patience})'}")
                if no_improve >= patience:
                    print(f"    Early stopping at step {step+1} "
                          f"(no improvement for {patience} evals)")
                    stopped_early = True
                meta_traffic_model.train()
                linketa_model.train()
                if stopped_early:
                    break

        # Restore best state if available
        if best_state is not None:
            meta_traffic_model.load_state_dict(
                {k: v.to(device) for k, v in best_state['mt'].items()}, strict=True)
            linketa_model.load_state_dict(
                {k: v.to(device) for k, v in best_state['le'].items()}, strict=True)

        # [2026-06-09] Report the LEARNED prototype<->generation gate for this target
        # (defensible: data-driven per city, not a fixed constant).
        if (hasattr(meta_traffic_model, "proto_alpha_logit")
                and meta_traffic_model.proto_W is not None):
            print(f"    [alpha] learned prototype gate alpha="
                  f"{float(torch.sigmoid(meta_traffic_model.proto_alpha_logit)):.4f} "
                  f"({city_name}, optimized during adaptation)")

        # [2026-06-06] Save ADAPTED (transfer) checkpoint for benchmark/testing.
        if transfer_ckpt_dir is not None:
            _combined = _CombinedModel(meta_traffic_model, linketa_model)
            _mgr = CheckpointManager(dirpath=transfer_ckpt_dir, max_keep=3)
            _mgr.save_best(model=_combined, optimizer=None, scheduler=None,
                           epoch=0, step=adapt_steps, best_metric=best_val_loss,
                           extra={"method": "MetaETA", "target": city_key,
                                  "scaler_hash": _scaler_fingerprint(
                                      city_scaler_mean, city_scaler_std)})
            print(f"    [saved adapted ckpt] {transfer_ckpt_dir}")

        # Evaluate on val and test (using city_key for norm lookup)
        meta_traffic_model.eval()
        linketa_model.eval()

        city_results = {}
        trip_frames, link_frames = [], []
        for split_name in ("val", "test"):
            loader = core["loaders"][split_name]
            with torch.no_grad():
                m = metatraffic_one_epoch(
                    loader=loader,
                    meta_traffic_model=meta_traffic_model,
                    linketa_model=linketa_model,
                    optimizer=None,
                    static_feats=static_feats,
                    traffic_real=traffic_real,
                    rel_coo=rel_coo,
                    time_col_indices=time_idx,
                    dyn_col_indices=dyn_idx,
                    scaler_mean=scaler_mean,
                    scaler_std=scaler_std,
                    vff_vec_city=vff_city,
                    epoch=0,
                    train=False,
                    w_speed=_w_speed,
                    w_link=_w_link,
                    w_route=_w_route,
                    w_traffic=_w_traffic,
                    city_name=city_key,
                )
            city_results[split_name] = m

            print(f"    {split_name}: "
                  f"Link  MAE={m.get('link_mae', float('nan')):.4f}  RMSE={m.get('link_rmse', float('nan')):.4f}  MAPE={m.get('link_mape', float('nan')):.4f}  |  "
                  f"Route MAE={m.get('route_mae', float('nan')):.4f}  RMSE={m.get('route_rmse', float('nan')):.4f}  MAPE={m.get('route_mape', float('nan')):.4f}  |  "
                  f"speed_mae={m.get('speed_mae', float('nan')):.4f}  vol_mae={m.get('vol_mae', float('nan')):.4f}  "
                  f"traffic_rmse={m.get('traffic_rmse', float('nan')):.4f}")

            # Always collect trip + link preds; main.py orchestrates the save.
            with torch.no_grad():
                trip_df_s, link_df_s = _collect_metatraffic_preds(
                    loader, meta_traffic_model, linketa_model,
                    static_feats, rel_coo, scaler_mean, scaler_std,
                    vff_city, dyn_idx, time_idx,
                )
            trip_df_s["split"] = split_name
            trip_df_s["city"]  = city_key
            link_df_s["split"] = split_name
            link_df_s["city"]  = city_key
            trip_frames.append(trip_df_s)
            link_frames.append(link_df_s)

        city_results["trip_df"] = (
            pd.concat(trip_frames, ignore_index=True) if trip_frames else pd.DataFrame())
        city_results["link_df"] = (
            pd.concat(link_frames, ignore_index=True) if link_frames else pd.DataFrame())
        results[city_name] = city_results

    _print_table("MetaTraffic — All Cities Evaluation", results)
    return results
