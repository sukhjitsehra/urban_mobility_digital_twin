# =============================================================================
# MetaTTE: Reptile-based Meta-Learning for LinkETA
# =============================================================================
#
# Implements the Reptile meta-learning algorithm for travel time estimation,
# following the MetaTTE paper (Wang et al., IEEE TITS 2022) and the original
# Reptile paper (Nichol et al., 2018).
#
# Key design choices (from the two papers):
#
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ REPTILE ALGORITHM (Nichol et al., Algorithm 1)                            │
# │                                                                           │
# │   Initialize φ (the vector of initial parameters)                         │
# │   for iteration = 1, 2, … do                                             │
# │       Sample task τ, corresponding to loss L_τ                            │
# │       Compute φ̃ = U^k_τ(φ), denoting k steps of SGD or Adam             │
# │       Update φ ← φ + ε(φ̃ − φ)                                           │
# │   end for                                                                 │
# │                                                                           │
# │ The meta-gradient is  g_Reptile = (φ − φ̃)/α, which optimizes:           │
# │   E[g_Reptile] = (k)AvgGrad − ½k(k−1)α·AvgGradInner                    │
# │                                                                           │
# │ This maximizes within-task generalization (AvgGradInner term)             │
# │ without requiring second-order derivatives or a train/test split          │
# │ per task.                                                                 │
# └─────────────────────────────────────────────────────────────────────────────┘
#
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ MetaTTE ADAPTATION (Wang et al., Algorithm 2)                             │
# │                                                                           │
# │   Randomly initialize θ                                                   │
# │   for r in [1, η) do                                                      │
# │       Randomly select a TTE-Task T_i from p_T                            │
# │       Save current model parameters to θ₁                                │
# │       for j in [1, k] do                                                  │
# │           Sample data from D^train_Ti                                     │
# │           Forward-backward with Adam on L_Ti                              │
# │       end for                                                             │
# │       Save current model parameters to θ₂                                │
# │       θ_f = β(1 − r/η)(θ₂ − θ₁)        ← linear LR decay              │
# │       Reset model parameters to θ_f                                       │
# │   end for                                                                 │
# │                                                                           │
# │ Key: β = step size, η = max iterations, k = inner steps (10 in paper)    │
# │ No STMemory, no FOMAML query loop, no train/test split per task.          │
# │ Much simpler than MAML/FOMAML; only first-order derivatives.             │
# └─────────────────────────────────────────────────────────────────────────────┘
#
# Comparison with MetaST (our FOMAML implementation):
# ┌──────────────────────┬──────────────────────────────┬──────────────────────┐
# │ Aspect               │ MetaST (FOMAML)              │ MetaTTE (Reptile)    │
# ├──────────────────────┼──────────────────────────────┼──────────────────────┤
# │ Inner loop           │ K SGD steps on support       │ K SGD/Adam steps     │
# │ Outer update         │ ∇ query_loss(φ̃) via autograd│ ε(φ̃ − φ) direct    │
# │ Train/test split     │ Yes (support/query)          │ No                   │
# │ 2nd-order info       │ Implicit (FOMAML ≈ MAML)    │ From k>1 inner steps │
# │ Memory module        │ STMemory + clustering loss   │ None (pure LinkETA)  │
# │ LR schedule (outer)  │ Adam with fixed LR           │ Linear decay to 0    │
# │ Task weighting       │ DWA (dynamic)                │ DWA (dynamic)        │
# │ City scheduling      │ All cities every iter        │ All cities every iter│
# │ Implementation       │ functional_call + autograd   │ state_dict copy      │
# └──────────────────────┴──────────────────────────────┴──────────────────────┘
#
# =============================================================================

# =============================================================================
# IMPORTS
# =============================================================================

import time
import math
import random
import copy
import os
import json
from collections import deque
from typing import Optional, Dict, List, Tuple, Any

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import LinkETA as _linketa_mod
from LinkETA import (
    one_epoch, lam_g_at, LinkETA, CFG,
)
from loss_functions import (
    link_loss_huber_macro, route_loss_mape,
    link_metrics_micro, route_metrics_pooled,
    selection_metric, L_MIN_METRIC, T_MIN_METRIC,
    compute_global_losses_and_metrics,
    # [2026-06-06] Controlled-backbone alignment: MetaTTE now optimizes the SAME
    # LinkETA backbone losses as MetaST / MetaTraffic / base LinkETA so the
    # backbone confers no per-method advantage (fair comparison). Only the
    # meta-strategy (Reptile) differs.
    link_eta_in_seconds, link_loss_huber_log_lenweighted, route_loss_huber_log,
)
from checkpoint import CheckpointManager, write_training_history


# =============================================================================
# [2026-06-06] Canonical LinkETA speed loss vs sp_actual (mirrors base
# LinkETA one_epoch). Shared by every forward+loss site in this file so the
# speed-supervision target is identical across MetaST / MetaTTE / MetaTraffic.
#   sp_actual = link_len / (link_tt · v_ff), clamped to r_max; rows pinned at the
#   r_max floor by Stage-1 redistribution are excluded from the speed loss.
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


# =============================================================================
# MODULE-LEVEL GLOBALS (swapped per city by set_globals_from_core)
# =============================================================================

model = None
CORE = None                  # [MetaETA stateless] active core; passed to one_epoch(CORE, ...)
device = None
optimizer = None
scheduler = None
train_loader = None
val_loader = None
test_loader = None
train_ds = None
val_ds = None
test_ds = None
inst = None
INST_FEAT_COLS = None
STATIC_FEAT_COLS = None
TIME_FEAT_COLS = None
DYN_FEAT_COLS = None
edges_vff = None
rel_edges = None
vff_vec = None
DYN_STORE = None
H_SLICES = None
BIN_SECONDS = None
L_MIN = None
T_MIN = None
SCALER_MEAN = None
SCALER_STD = None
DYN_IDX = None
HIST = None
num_epochs = None
clip_norm = None
ckpt = None


# =============================================================================
# CORE GLOBALS MANAGEMENT
# =============================================================================

def set_globals_from_core(core, keep_model=True):
    """
    Swap all module-level globals to the given city's core dict.

    This mirrors MetaST.set_globals_from_core() but without any
    STMemory / clustering-specific globals.

    Parameters
    ----------
    core : dict
        City core built by ``linketa_get_core_components``.
    keep_model : bool
        If True, preserve the current shared model (for Reptile meta-training
        where one model is shared across all tasks). If False, restore the
        per-core base model.
    """
    global device, model, optimizer, scheduler
    global train_loader, val_loader, test_loader
    global train_ds, val_ds, test_ds
    global inst, INST_FEAT_COLS, STATIC_FEAT_COLS, TIME_FEAT_COLS, DYN_FEAT_COLS
    global edges_vff, rel_edges, vff_vec
    global DYN_STORE
    global H_SLICES, BIN_SECONDS, L_MIN, T_MIN
    global SCALER_MEAN, SCALER_STD, DYN_IDX
    global HIST, num_epochs, clip_norm, ckpt
    global CORE

    # [MetaETA stateless] one_epoch takes `core` explicitly (no activate_core /
    # LinkETA module globals). Keep a handle and point core["model"] at the model
    # one_epoch must run (the shared Reptile model when keep_model).
    CORE = core
    if keep_model and model is not None:
        core["model"] = model
    else:
        model = core["model"]

    device = core["device"]

    # Loaders / datasets
    train_loader = core["loaders"]["train"]
    val_loader   = core["loaders"]["val"]
    test_loader  = core["loaders"]["test"]
    train_ds     = core["datasets"]["train"]
    val_ds       = core["datasets"]["val"]
    test_ds      = core["datasets"]["test"]

    # Tables / column lists
    inst             = core["inst_table"]
    INST_FEAT_COLS   = core["inst_feat_cols"]
    STATIC_FEAT_COLS = core["static_cols"]
    TIME_FEAT_COLS   = core["time_cols"]
    DYN_FEAT_COLS    = core["dyn_cols"]

    # Graph data
    edges_vff = core["graph"]["edges_vff"]
    rel_edges = core["graph"]["rel_edges"]
    vff_vec   = core["graph"]["vff_vec"]

    # Dynamic store + config
    DYN_STORE = core["dyn_store"]

    # Constants
    H_SLICES    = core["constants"]["H_SLICES"]
    BIN_SECONDS = core["constants"]["BIN_SECONDS"]
    L_MIN       = core["constants"]["L_MIN"]
    T_MIN       = core["constants"]["T_MIN"]

    # Scalers
    SCALER_MEAN = core["scalers"]["mean"]
    SCALER_STD  = core["scalers"]["std"]
    DYN_IDX     = core["scalers"]["dyn_idx"]

    # Misc
    optimizer  = core.get("optimizer", None)
    scheduler  = core.get("scheduler", None)
    HIST       = core.get("history", None)
    num_epochs = core.get("num_epochs", None)
    clip_norm  = core.get("clip_norm", None)
    ckpt       = core.get("checkpoint_manager", None)


# =============================================================================
# STATE MANAGEMENT UTILITIES
# =============================================================================

def clone_state_cpu(m: nn.Module) -> dict:
    """Deep-copy model state dict to CPU."""
    return {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}


def load_state_cpu(m: nn.Module, state_cpu: dict, dev: torch.device):
    """Load a CPU state dict onto the model, moving tensors to device."""
    m.load_state_dict({k: v.to(dev) for k, v in state_cpu.items()}, strict=True)


def fmt_metrics(tag: str, m: dict) -> str:
    """Format metrics dictionary for readable logging."""
    def g(key, default=float("nan")):
        return m.get(key, default)
    return (
        f"{tag} | "
        f"Link: loss={g('link_loss'):.4f} mae={g('link_mae'):.4f} "
        f"rmse={g('link_rmse'):.4f} mape={g('link_mape'):.4f}  ||  "
        f"Route: loss={g('route_loss'):.4f} mae={g('route_mae'):.4f} "
        f"rmse={g('route_rmse'):.4f} mape={g('route_mape'):.4f}  ||  "
        f"Traffic: speed_mae={g('speed_mae'):.4f} vol_mae={g('vol_mae'):.4f}"
    )


# =============================================================================
# REPTILE INNER LOOP: k gradient steps on a single task
# =============================================================================

def reptile_inner_steps(
    core: dict,
    inner_steps: int = 10,
    inner_lr: float = 1e-3,
    w_speed: float = 0.5,   # [2026-06-06] DWA-managed speed weight (3-task)
    w_link: float = 0.5,
    w_route: float = 0.5,
    use_adam: bool = True,
    adam_betas: Tuple[float, float] = (0.0, 0.999),
) -> dict:
    """
    Reptile inner loop: perform k gradient steps on a single task (city).

    This corresponds to computing φ̃ = U^k_τ(φ) from the Reptile paper.

    MetaTTE paper specifics (Section IV-C, Algorithm 2):
    - Uses Adam optimizer for inner loop (with β₁=0 as recommended by
      Reptile paper Section 6.1: "momentum reduced performance")
    - k = 10 batches of training data per iteration
    - No train/test split within inner loop

    Parameters
    ----------
    core : dict
        City core dict (task τ).
    inner_steps : int
        Number of gradient steps k (MetaTTE uses k=10).
    inner_lr : float
        Step size for inner optimizer.
    w_link : float
        Weight for link-level loss (from DWA or fixed).
    w_route : float
        Weight for route-level loss (from DWA or fixed).
    use_adam : bool
        If True, use Adam (MetaTTE default). If False, use SGD.
    adam_betas : tuple
        Beta values for Adam. Default (0.0, 0.999) follows Reptile paper
        recommendation of β₁=0 (no momentum).

    Returns
    -------
    dict with keys:
        'adapted_state': state_dict after k steps (θ₂ in MetaTTE notation)
        'support_metrics': metrics on support set after adaptation
        'steps_done': actual number of steps performed
        'avg_link_loss': avg link loss across inner steps (for DWA)
        'avg_route_loss': avg route loss across inner steps (for DWA)
    """
    set_globals_from_core(core, keep_model=True)
    model.to(device)
    model.train()

    # Create inner optimizer (fresh each call — MetaTTE paper Section IV-C)
    if use_adam:
        inner_opt = torch.optim.Adam(
            model.parameters(), lr=inner_lr, betas=adam_betas
        )
    else:
        inner_opt = torch.optim.SGD(
            model.parameters(), lr=inner_lr
        )

    train_iter = iter(train_loader)
    steps_done = 0

    # DWA: track per-step speed/link/route losses for this city
    _sum_speed_loss = 0.0
    _sum_link_loss  = 0.0
    _sum_route_loss = 0.0

    for step in range(inner_steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        # Move batch to device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        # Unpack batch fields (same as MetaST/LinkETA canonical)
        node_edge_idx    = batch["node_edge_idx"]
        x_seq_nodes      = batch["x_seq"]
        route_flag_nodes = batch["route_flag"]
        hop_idx          = batch["hop_idx"]
        masks_per_rel    = [m.to(device) for m in batch["masks_per_rel"]]

        route_seq_idx = batch["route_seq_idx"]
        seq_true_tt   = torch.nan_to_num(batch["seq_true_tt"], nan=0.0)
        seq_link_len  = torch.nan_to_num(batch["seq_link_len"], nan=0.0).clamp_min(1e-3)
        seq_mask      = batch["seq_mask"].bool()
        trip_tt       = torch.nan_to_num(batch["trip_tt"], nan=0.0)

        sp_true_node = torch.nan_to_num(batch["sp_ratio_agg"], nan=0.0)
        lc_true_node = torch.nan_to_num(batch["logcnt1h_node"], nan=0.0)
        sp_w_node    = torch.nan_to_num(batch["speed_w_node"], nan=0.0)

        gather_for_seq = lambda node_vals: torch.gather(node_vals, 1, route_seq_idx)
        sp_true = gather_for_seq(sp_true_node)
        lc_true = gather_for_seq(lc_true_node)
        sp_w    = gather_for_seq(sp_w_node)

        vff_seq_b = vff_vec[node_edge_idx]
        vff_seq_b = torch.gather(vff_seq_b, 1, route_seq_idx)

        x_seq_norm = (x_seq_nodes - SCALER_MEAN.view(1, 1, -1)) / SCALER_STD.view(1, 1, -1)

        sp_mask = (sp_w > 0) & seq_mask

        # Forward through base LinkETA (no STMemory)
        # LinkETA returns 7-tuple: (link_eta, trip_eta, sp, lc, g, delta, eta_phys)
        out = model(
            x_seq_norm,
            hop_idx,
            route_flag_nodes,
            masks_per_rel,
            route_seq_idx,
            seq_link_len,
            vff_seq_b,
            seq_mask,
            r_max=3.0,
            return_parts=True,
        )
        link_eta_seq, trip_eta, sp_pred, lc_pred, g, delta, eta_phys_seq = out[:7]

        link_eta_seq = torch.nan_to_num(link_eta_seq, nan=0.0)
        trip_eta     = torch.nan_to_num(trip_eta, nan=0.0)
        sp_pred      = torch.nan_to_num(sp_pred, nan=0.0)
        lc_pred      = torch.nan_to_num(lc_pred, nan=0.0)

        # [2026-06-06] Controlled backbone losses (identical to base LinkETA):
        #   link/route in log1p space, speed vs sp_actual, no lam_g g-reg.
        pred_seq = link_eta_in_seconds(link_eta_seq, seq_link_len)
        seq_true = seq_true_tt

        loss_link  = link_loss_huber_log_lenweighted(
            pred_seq, seq_true, seq_mask, link_len=seq_link_len, delta=0.3)
        loss_route = route_loss_huber_log(pred_seq, trip_tt, seq_mask, delta=0.3)

        loss_speed = _backbone_speed_loss(
            sp_pred, seq_link_len, seq_true_tt, vff_seq_b, seq_mask)

        if seq_mask.any():
            loss_vol = F.smooth_l1_loss(lc_pred[seq_mask], lc_true[seq_mask])
        else:
            loss_vol = torch.zeros((), device=device)

        # 3-task {speed,link,route} (DWA-managed) + fixed 0.10*vol — matches base LinkETA.
        inner_loss = (
            w_speed * loss_speed +
            w_link * loss_link +
            w_route * loss_route +
            0.10 * loss_vol
        )

        inner_opt.zero_grad(set_to_none=True)
        inner_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        inner_opt.step()
        steps_done += 1

        # DWA: track individual losses (detached)
        _sum_speed_loss += float(loss_speed.detach().cpu().item())
        _sum_link_loss  += float(loss_link.detach().cpu().item())
        _sum_route_loss += float(loss_route.detach().cpu().item())

    # Evaluate on support set after adaptation (φ̃ quality check)
    model.eval()
    with torch.no_grad():
        support_metrics = one_epoch(
            CORE, train_loader, epoch=0, train=False,
            w_link=w_link, w_route=w_route,
        )
    support_metrics["inner_steps"] = steps_done
    model.train()

    adapted_state = clone_state_cpu(model)

    return {
        "adapted_state": adapted_state,
        "support_metrics": support_metrics,
        "steps_done": steps_done,
        "avg_speed_loss": _sum_speed_loss / max(steps_done, 1),
        "avg_link_loss": _sum_link_loss / max(steps_done, 1),
        "avg_route_loss": _sum_route_loss / max(steps_done, 1),
    }


# =============================================================================
# REPTILE META-TRAINING LOOP
# =============================================================================
# Paper-faithful implementation of MetaTTE Algorithm 2:
#   - ONE random task (city) per iteration (serial Reptile)
#   - DWA dynamically reweights link vs route losses
#   - Batch size is inherited from the shared DataLoaders (same as MetaST)
#
# Difference from MetaST:
#   - Outer update = ε(θ₂ − θ₁) on single task (Reptile)
#     instead of FOMAML query-set gradients averaged over all cities
#   - No train/test split per task — inner loop uses train_loader only
#   - No STMemory / clustering loss
#   - ONE random task per iteration (MetaTTE Algorithm 2 Line 3)
#     vs MetaST which processes ALL cities every iteration
# =============================================================================

def meta_train_reptile(
    source_cores: List[dict],
    source_names: List[str],
    meta_iters: int = 7000,
    inner_steps: int = 10,
    inner_lr: float = 1e-3,
    beta: float = 0.1,
    w_speed: float = 0.5,   # [2026-06-06] DWA-managed speed weight (3-task)
    w_link: float = 0.5,
    w_route: float = 0.5,
    use_adam_inner: bool = True,
    adam_betas: Tuple[float, float] = (0.0, 0.999),
    dwa_T: float = 2.0,
    seed: int = 42,
    log_every: int = 10,
    eval_every: int = 50,
    patience: int = None,
    min_delta: float = 0.0,
    enable_metrics: bool = True,
    max_query_batches: int = 5,
    ckpt_dir: str = "ckpts_MetaTTE",  # directory to save best/last checkpoints
    save_every: int = 5,              # [2026-06-06] interval theta0 ckpts (0=off)
) -> dict:
    """
    Reptile meta-training for LinkETA — paper-faithful MetaTTE Algorithm 2.

    ONE random task (city) is sampled per iteration, exactly as specified
    in MetaTTE Algorithm 2 Line 3: "Randomly select a TTE-Task Ti from pT".
    DWA reweights link/route losses (shared with MetaST for fair comparison).

    ┌────────────────────────────────────────────────────────────────────────┐
    │ MetaTTE Algorithm 2 (Wang et al., IEEE TITS 2022):                   │
    │                                                                       │
    │   Randomly initialize θ                                               │
    │   for r in [1, η) do                                                  │
    │       Randomly select a TTE-Task Ti from pT  ← ONE task per iter    │
    │       Save current model parameters to θ₁                            │
    │       for j in [1, k] do                                              │
    │           Sample batch from D^train_Ti                                │
    │           Forward-backward with Adam on L_Ti (DWA-weighted)          │
    │       end for                                                         │
    │       Save current model parameters to θ₂                            │
    │       θ ← θ₁ + ε(θ₂ − θ₁)    where ε = β(1 − r/η)                │
    │   end for                                                             │
    └────────────────────────────────────────────────────────────────────────┘

    Parameters
    ----------
    source_cores : list of dict
        Core dicts for source cities (TTE-Tasks).
    source_names : list of str
        Human-readable names for logging.
    meta_iters : int
        Maximum iterations η (MetaTTE: 7000).
    inner_steps : int
        Number of inner gradient steps k (MetaTTE: 10).
    inner_lr : float
        Learning rate for inner optimizer.
    beta : float
        Step size β for meta-update (MetaTTE: 0.1).
    w_link : float
        Initial weight for link-level loss (updated by DWA if enabled).
    w_route : float
        Initial weight for route-level loss (updated by DWA if enabled).
    use_adam_inner : bool
        Use Adam (True, MetaTTE default) or SGD (False) for inner loop.
    adam_betas : tuple
        Adam beta parameters. (0.0, 0.999) per Reptile paper recommendation.
    dwa_T : float
        DWA temperature. >0 enables DWA, 0 disables (fixed w_link/w_route).
    seed : int
        Random seed.
    log_every : int
        Print progress every N iterations.
    eval_every : int
        Run eval on val sets every N iterations.
    patience : int or None
        Early stopping patience on averaged val loss. None = no early stopping.
    min_delta : float
        Minimum improvement for early stopping.
    enable_metrics : bool
        If True, print detailed per-city metrics at log_every intervals.
    max_query_batches : int
        Max val batches per city for early-stopping eval (matches MetaST/MetaTraffic).

    Returns
    -------
    dict with keys:
        'meta_state': final meta-learned state dict (CPU)
        'history': list of logged metrics per eval point
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_tasks = len(source_cores)
    eta = meta_iters  # max iterations (MetaTTE notation)

    # ---- DWA (Dynamic Weight Averaging) — [2026-06-06] 3-task: speed, link, route ----
    _dwa_hist_speed = deque(maxlen=2)
    _dwa_hist_link  = deque(maxlen=2)
    _dwa_hist_route = deque(maxlen=2)
    _dwa_enabled = (dwa_T > 0)
    # [2026-06-06] Normalize pre-DWA weights to sum 1.0 (DWA softmax also sums to
    # 1.0) so there is no discontinuity at DWA kick-in, and all methods start DWA
    # from uniform weights (fair comparison).
    _w0 = w_speed + w_link + w_route
    w_speed, w_link, w_route = w_speed / _w0, w_link / _w0, w_route / _w0
    w_speed_init, w_link_init, w_route_init = w_speed, w_link, w_route

    print("=" * 100)
    print("MetaTTE: Reptile Meta-Training for LinkETA (Algorithm 2)")
    print("=" * 100)
    print(f"  Source cities (TTE-Tasks): {source_names}")
    print(f"  Task scheduling: ONE random task per iteration (paper Algorithm 2 Line 3)")
    print(f"  Max meta iterations (η): {eta}")
    print(f"  Inner steps (k): {inner_steps}")
    print(f"  Inner LR: {inner_lr}")
    print(f"  Inner optimizer: {'Adam(β₁=0)' if use_adam_inner else 'SGD'}")
    print(f"  Meta step size (β): {beta}")
    print(f"  Outer schedule: linear decay ε = β(1 − r/η)")
    print(f"  DWA: {'ON (T={:.1f})'.format(dwa_T) if _dwa_enabled else 'OFF (fixed w_link={:.2f}, w_route={:.2f})'.format(w_link, w_route)}")
    print(f"  Batch size: inherited from DataLoader (same as MetaST)")
    print(f"  enable_metrics={enable_metrics} | log_every={log_every} | eval_every={eval_every}")
    if patience is not None:
        print(f"  Early stopping: patience={patience}, min_delta={min_delta}, max_query_batches={max_query_batches}")
    print("=" * 100)

    # Tracking
    history = []
    _hist_rows = []   # [2026-06-06] per-source (support) metrics for training_history.csv
    best_val_loss = float("inf")
    best_state = None
    best_meta_iter = None       # [2026-06-07] iter of min avg_val_loss (selected model)
    best_summary = None         # [2026-06-07] per-city val metrics @ that iter
    bad_count = 0
    t_start = time.time()

    # ---- Checkpoint manager for saving to disk ----
    _ckpt_mgr = CheckpointManager(dirpath=ckpt_dir, max_keep=3)
    print(f"  Checkpoint dir: {ckpt_dir}")

    for r in range(1, eta + 1):
        do_log = enable_metrics and ((r % log_every == 0) or (r <= 3))

        # ---- DWA: recompute task weights from loss-change ratios (3-task) ----
        if (_dwa_enabled and len(_dwa_hist_speed) >= 2
                and len(_dwa_hist_link) >= 2 and len(_dwa_hist_route) >= 2):
            r_speed = float(_dwa_hist_speed[-1] / (_dwa_hist_speed[-2] + 1e-8))
            r_link  = float(_dwa_hist_link[-1]  / (_dwa_hist_link[-2]  + 1e-8))
            r_route = float(_dwa_hist_route[-1] / (_dwa_hist_route[-2] + 1e-8))
            e_speed = math.exp(r_speed / dwa_T)
            e_link  = math.exp(r_link  / dwa_T)
            e_route = math.exp(r_route / dwa_T)
            s = e_speed + e_link + e_route
            w_speed = e_speed / s
            w_link  = e_link  / s
            w_route = e_route / s

        # ─── Step 1: Randomly select ONE TTE-Task (Algorithm 2, Line 3) ───
        task_idx = random.randint(0, n_tasks - 1)
        core = source_cores[task_idx]
        task_name = source_names[task_idx]

        # ─── Step 2: Save current model parameters to θ₁ (Line 4) ───
        theta1 = clone_state_cpu(model)

        # ─── Step 3: k inner gradient steps on task Ti → θ₂ (Lines 5-10) ───
        inner_result = reptile_inner_steps(
            core=core,
            inner_steps=inner_steps,
            inner_lr=inner_lr,
            w_speed=w_speed,
            w_link=w_link,
            w_route=w_route,
            use_adam=use_adam_inner,
            adam_betas=adam_betas,
        )
        theta2 = inner_result["adapted_state"]

        # ─── Step 4: Meta Update (Line 11-12, Eq. 9) ───
        #   θf = β(1 − r/η)(θ₂ − θ₁)
        #   θ ← θ₁ + θf   (equivalently: θ ← θ₁ + ε(θ₂ − θ₁))
        epsilon = beta * (1.0 - r / eta)

        meta_state = {}
        for key in theta1:
            meta_state[key] = theta1[key] + epsilon * (theta2[key] - theta1[key])

        load_state_cpu(model, meta_state, device)

        # ---- DWA: update loss history from this iteration's task ----
        if _dwa_enabled:
            _dwa_hist_speed.append(inner_result["avg_speed_loss"])
            _dwa_hist_link.append(inner_result["avg_link_loss"])
            _dwa_hist_route.append(inner_result["avg_route_loss"])

        # [2026-06-06] interval theta0 checkpoint for the meta-iter sweep
        if save_every and (r % save_every == 0):
            _ckpt_mgr.save(
                os.path.join(ckpt_dir, f"theta0_iter{r}.ckpt"),
                model=model, optimizer=None, scheduler=None,
                epoch=r, step=r, best_metric=best_val_loss,
                extra={"method": "MetaTTE", "meta_iter": r, "epsilon": epsilon},
            )

        # ─── Logging ───
        if r % log_every == 0 or r == 1:
            elapsed = time.time() - t_start
            sup_m = inner_result["support_metrics"]
            # [2026-06-06] per-source (support) metrics for training_history.csv.
            # NOTE: MetaTTE samples ONE city/iter; these are support-set metrics
            # (vs val for MetaST/MetaETA) — the readily-available per-source signal.
            _hist_rows.append({
                "meta_iter": r, "city": task_name,
                "link_mae": sup_m.get("link_mae"), "link_rmse": sup_m.get("link_rmse"),
                "link_mape": sup_m.get("link_mape"),
                "route_mae": sup_m.get("route_mae"), "route_rmse": sup_m.get("route_rmse"),
                "route_mape": sup_m.get("route_mape"),
                "link_wape": sup_m.get("link_wape"), "link_mdape": sup_m.get("link_mdape"),
                "link_p90": sup_m.get("link_p90"),
                "route_wape": sup_m.get("route_wape"), "route_mdape": sup_m.get("route_mdape"),
                "route_p90": sup_m.get("route_p90"),   # [2026-06-07] full metric parity
            })
            print(
                f"[Reptile] Iter {r:5d}/{eta} | ε={epsilon:.6f} | "
                f"Task={task_name} | "
                f"DWA w_speed={w_speed:.3f} w_link={w_link:.3f} w_route={w_route:.3f} | "
                f"link_mape={sup_m.get('link_mape', float('nan')):.4f} "
                f"route_mape={sup_m.get('route_mape', float('nan')):.4f} | "
                f"{elapsed:.1f}s"
            )
            if do_log:
                print(f"  " + fmt_metrics("Support(k steps)", sup_m))

        # ─── Periodic evaluation on source city val sets (partial, composite loss) ───
        # Uses the same composite loss formula as MetaST/MetaTraffic for fair comparison:
        #   w_link * Huber + w_route * MAPE + 0.05 * speed + 0.10 * vol + λ_g * |g-1|
        # Evaluated on max_query_batches per city (not full val set).
        if r % eval_every == 0 or r == 1:
            avg_val_loss = 0.0
            model.eval()
            with torch.no_grad():
                for ci, (c, cn) in enumerate(zip(source_cores, source_names)):
                    set_globals_from_core(c, keep_model=True)
                    _vl = c["loaders"]["val"]

                    _city_loss_sum = 0.0
                    _city_n = 0
                    for _bi, _batch in enumerate(_vl):
                        if _bi >= max_query_batches:
                            break
                        for _k, _v in _batch.items():
                            if isinstance(_v, torch.Tensor):
                                _batch[_k] = _v.to(device)

                        # Forward through model
                        _node_edge_idx = _batch["node_edge_idx"]
                        _x_seq = _batch["x_seq"]
                        _route_flag = _batch["route_flag"]
                        _hop_idx = _batch["hop_idx"]
                        _masks = [m.to(device) for m in _batch["masks_per_rel"]]
                        _route_seq_idx = _batch["route_seq_idx"]
                        _seq_true_tt = torch.nan_to_num(_batch["seq_true_tt"], nan=0.0)
                        _seq_link_len = torch.nan_to_num(_batch["seq_link_len"], nan=0.0).clamp_min(1e-3)
                        _seq_mask = _batch["seq_mask"].bool()
                        _trip_tt = torch.nan_to_num(_batch["trip_tt"], nan=0.0)

                        _sp_true = torch.nan_to_num(_batch.get("sp_ratio_agg", torch.zeros_like(_seq_true_tt)), nan=0.0)
                        _lc_true = torch.nan_to_num(_batch.get("logcnt1h_node", torch.zeros_like(_seq_true_tt)), nan=0.0)
                        _sp_w = torch.nan_to_num(_batch.get("speed_w_node", torch.zeros_like(_seq_true_tt)), nan=0.0)
                        _sp_true = torch.gather(_sp_true, 1, _route_seq_idx)
                        _lc_true = torch.gather(_lc_true, 1, _route_seq_idx)
                        _sp_w = torch.gather(_sp_w, 1, _route_seq_idx)
                        _sp_mask = (_sp_w > 0) & _seq_mask

                        _vff_seq = c["graph"]["vff_vec"][_node_edge_idx]
                        _vff_seq = torch.gather(_vff_seq, 1, _route_seq_idx)

                        _scaler_mean = c["scalers"]["mean"]
                        _scaler_std = c["scalers"]["std"]
                        _x_norm = (_x_seq - _scaler_mean.view(1, 1, -1)) / _scaler_std.view(1, 1, -1)

                        _link_eta, _trip_eta, _sp_pred, _lc_pred, _g, _delta, _eta_phys = model(
                            _x_norm, _hop_idx, _route_flag, _masks,
                            _route_seq_idx, _seq_link_len, _vff_seq, _seq_mask,
                            r_max=3.0, return_parts=True,
                        )
                        _link_eta = torch.nan_to_num(_link_eta, nan=0.0)
                        _sp_pred = torch.nan_to_num(_sp_pred, nan=0.0)
                        _lc_pred = torch.nan_to_num(_lc_pred, nan=0.0)

                        # [2026-06-06] Controlled backbone composite (same as base LinkETA):
                        #   link/route log1p-Huber, speed vs sp_actual, 3-task + 0.10*vol.
                        _pred_seq = link_eta_in_seconds(_link_eta, _seq_link_len)
                        _loss_link = link_loss_huber_log_lenweighted(
                            _pred_seq, _seq_true_tt, _seq_mask, link_len=_seq_link_len, delta=0.3)
                        _loss_route = route_loss_huber_log(_pred_seq, _trip_tt, _seq_mask, delta=0.3)

                        _loss_speed = _backbone_speed_loss(
                            _sp_pred, _seq_link_len, _seq_true_tt, _vff_seq, _seq_mask)

                        if _seq_mask.any():
                            _loss_vol = F.smooth_l1_loss(_lc_pred[_seq_mask], _lc_true[_seq_mask])
                        else:
                            _loss_vol = torch.zeros((), device=device)

                        _batch_loss = (
                            w_speed * _loss_speed
                            + w_link * _loss_link
                            + w_route * _loss_route
                            + 0.10 * _loss_vol
                        )
                        _city_loss_sum += float(_batch_loss)
                        _city_n += 1

                    _city_avg = _city_loss_sum / max(_city_n, 1)
                    avg_val_loss += _city_avg
                    print(f"  [Eval] {cn}: composite_val_loss={_city_avg:.6f} ({_city_n} batches)")

            avg_val_loss /= max(n_tasks, 1)
            history.append({"iter": r, "avg_val_loss": avg_val_loss, "epsilon": epsilon,
                            "w_link": w_link, "w_route": w_route})
            model.train()

            # Early stopping
            if avg_val_loss < (best_val_loss - min_delta):
                best_val_loss = avg_val_loss
                best_state = clone_state_cpu(model)
                bad_count = 0
                # [2026-06-07] Snapshot per-city VAL metrics of the SELECTED theta_0.
                # Reptile selects on theta_0's val loss directly (no query split), so
                # evaluate the current theta_0 on each source city's full val set —
                # giving BOTH cities at the SAME (selected) iteration for the summary.
                best_meta_iter = r
                best_summary = {}
                model.eval()
                with torch.no_grad():
                    for _c, _cn in zip(source_cores, source_names):
                        set_globals_from_core(_c, keep_model=True)
                        best_summary[_cn] = one_epoch(
                            _c, _c["loaders"]["val"], epoch=0, train=False,
                            w_link=w_link, w_route=w_route,
                        )
                model.train()
                # Save best checkpoint to disk
                _ckpt_mgr.save_best(
                    model=model, optimizer=None, scheduler=None,
                    epoch=r, step=r, best_metric=best_val_loss,
                    extra={"method": "MetaTTE", "meta_iter": r, "epsilon": epsilon},
                )
            else:
                bad_count += 1

            # Save last checkpoint every eval
            _ckpt_mgr.save_last(
                model=model, optimizer=None, scheduler=None,
                epoch=r, step=r, best_metric=best_val_loss,
                extra={"method": "MetaTTE", "meta_iter": r, "epsilon": epsilon},
            )

            if patience is not None and bad_count >= patience:
                print(f"\n[Reptile] Early stopping at iter {r} | "
                      f"best_val_loss={best_val_loss:.6f}")
                if best_state is not None:
                    load_state_cpu(model, best_state, device)
                break

    # Final state
    if best_state is not None and patience is not None:
        final_state = best_state
    else:
        final_state = clone_state_cpu(model)

    elapsed = time.time() - t_start
    print("=" * 100)
    print(f"MetaTTE Reptile meta-training complete. "
          f"Total time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"Final DWA weights: w_speed={w_speed:.4f}, w_link={w_link:.4f}, w_route={w_route:.4f}")
    print(f"Best val loss: {best_val_loss:.6f}")
    print(f"Checkpoints saved to: {ckpt_dir}/")
    print("=" * 100)

    # [2026-06-06] Persist per-source training history + best-iter summary.
    write_training_history(ckpt_dir, _hist_rows, method="MetaTTE",
                           best_summary=best_summary, best_meta_iter=best_meta_iter)

    return {
        "meta_state": final_state,
        "history": history,
    }
