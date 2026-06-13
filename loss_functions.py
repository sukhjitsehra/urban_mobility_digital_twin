import pandas as pd
import ast
import requests
import numpy as np
import json
from scipy.sparse import csr_matrix, vstack
from scipy.optimize import nnls
import scipy
import os, time
from tqdm import tqdm
import networkx as nx
from scipy.sparse import csr_matrix
from shapely.geometry import LineString, MultiLineString, Point
from shapely.strtree import STRtree
from scipy.spatial import cKDTree

from folium.plugins import HeatMap
from pyrosm import OSM
from concurrent.futures import ThreadPoolExecutor
import geopandas as gpd
from collections import defaultdict
from shapely.geometry import LineString
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import math
from torch_geometric.data import Data
from torch_sparse import SparseTensor
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp import autocast, GradScaler
try:
    # torch >= 2.0 usually
    from torch.amp import GradScaler, autocast
except Exception:
    # older torch
    from torch.cuda.amp import GradScaler, autocast
from torch import amp
from torch import nn, Tensor
import torch.nn.functional as F
import torch
from contextlib import nullcontext
from collections import defaultdict
import inspect
from collections import Counter
from typing import Optional

import os, time  # if not already imported

import os, time, random
import math, bisect, re, collections
from typing import List, Dict, Tuple, Optional, Sequence
from functools import lru_cache
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from shapely.geometry import LineString, MultiLineString
from torch.cuda.amp import autocast, GradScaler


# ---------- Tiny normaliser ----------
_TINY = 1e-6
# # Stub thresholds for METRICS ONLY (never used in training losses)
L_MIN_METRIC = 10.0   # meters
T_MIN_METRIC = 1.5    # seconds


def selection_metric(route_mape, link_mape, w_route=0.7, w_link=0.3):
    vals, ws = [], []
    if route_mape is not None and not math.isnan(route_mape):
        vals.append(route_mape); ws.append(w_route)
    if link_mape is not None and not math.isnan(link_mape):
        vals.append(link_mape); ws.append(w_link)
    if not ws:
        return float('inf')
    wsum = sum(ws)
    return sum(v*w for v, w in zip(vals, ws)) / wsum


def _huber_per_element(err: torch.Tensor, delta: float) -> torch.Tensor:
    delta_t = torch.as_tensor(delta, dtype=err.dtype, device=err.device)
    abs_err = err.abs()
    return torch.where(abs_err < delta_t,
                       0.5 * err * err,
                       delta_t * (abs_err - 0.5 * delta_t))
# ========== LOSSES (sequence-aware) ==========
def link_loss_huber_macro(pred_seq: torch.Tensor,
                          seq_true: torch.Tensor,
                          seq_mask: torch.Tensor,
                          delta: float = 1.0) -> torch.Tensor:
    pred_seq = torch.nan_to_num(pred_seq)
    seq_true = torch.nan_to_num(seq_true)
    m = seq_mask.bool()
    if not m.any():
        return torch.zeros((), device=pred_seq.device)
    per = _huber_per_element(pred_seq - seq_true, delta) * m.float()
    cnt = m.sum(dim=1).clamp_min(1)
    per_route = per.sum(dim=1) / cnt
    return per_route.mean()

def route_loss_mape(pred_seq: torch.Tensor,
                    trip_tt: torch.Tensor,
                    seq_mask: torch.Tensor) -> torch.Tensor:
    pred_seq = torch.nan_to_num(pred_seq)
    trip_tt = torch.nan_to_num(trip_tt)
    seq_mask = seq_mask.bool()
    pred_route = (pred_seq * seq_mask.float()).sum(dim=1)
    pos = trip_tt > 0
    if not pos.any():
        return torch.zeros((), device=pred_seq.device)
    ape = (pred_route[pos] - trip_tt[pos]).abs() / trip_tt[pos].clamp_min(_TINY)
    return ape.mean()

# ========== METRICS (sequence-aware, with n_mape) ==========
def link_metrics_micro(pred_seq: torch.Tensor,
                       seq_true: torch.Tensor,
                       seq_mask: torch.Tensor,
                       link_len: Optional[torch.Tensor] = None,
                       L_min: float = L_MIN_METRIC,
                       T_min: float = T_MIN_METRIC) -> dict:
    """Link-level metrics. In addition to MAE/RMSE/MAPE, returns robust
    percentage-error reporting metrics (see [2026-06-07]):
      - wape : Sum|err| / Sum(true) over ALL valid on-route links (= MAE/mean(true);
               denominator-stable, NOT inflated by micro-links).
      - mdape, ape_p90 : median / 90th-percentile of per-link APE over the
               thresholded set (>= T_min s, >= L_min m), robust to short-link tails.
    Also returns accumulation building blocks (sum_abs, sum_true, ape_vals) so
    callers can pool exact WAPE/MdAPE/P90 across batches via finalize_pct_metrics().
    """
    pred_seq = torch.nan_to_num(pred_seq)
    seq_true = torch.nan_to_num(seq_true)
    on = seq_mask.bool()
    if not on.any():
        return dict(mae=float("nan"), rmse=float("nan"), mape=float("nan"),
                    wape=float("nan"), mdape=float("nan"), ape_p90=float("nan"),
                    n_links=0, n_mape=0,
                    sum_abs=0.0, sum_true=0.0, ape_vals=torch.empty(0))
    # MAE/RMSE over all on-route positions
    y_pred_all = pred_seq[on]
    y_true_all = seq_true[on]
    diff = (y_pred_all - y_true_all)
    mae  = diff.abs().mean().item()
    rmse = torch.sqrt(torch.mean(diff * diff)).item()
    n_links = int(on.sum().item())
    # WAPE building blocks over ALL valid on-route links (denominator-stable)
    sum_abs  = diff.abs().sum().item()
    sum_true = y_true_all.clamp_min(0.0).sum().item()
    wape = (sum_abs / sum_true) if sum_true > _TINY else float("nan")
    # MAPE / MdAPE / P90 over thresholded set (excludes noise-dominated micro-links)
    mape_mask = on & (seq_true >= T_min)
    if link_len is not None:
        mape_mask = mape_mask & (torch.nan_to_num(link_len) >= L_min)
    y_true_m = seq_true[mape_mask]
    y_pred_m = pred_seq[mape_mask]
    pos = y_true_m > 0
    n_mape = int(pos.sum().item())
    if pos.any():
        ape_vals = ((y_pred_m[pos] - y_true_m[pos]).abs()
                    / y_true_m[pos].clamp_min(_TINY)).detach().cpu()
        mape    = ape_vals.mean().item()
        mdape   = ape_vals.median().item()
        ape_p90 = torch.quantile(ape_vals, 0.9).item()
    else:
        ape_vals = torch.empty(0)
        mape = mdape = ape_p90 = float("nan")
    return dict(mae=mae, rmse=rmse, mape=mape,
                wape=wape, mdape=mdape, ape_p90=ape_p90,
                n_links=n_links, n_mape=n_mape,
                sum_abs=sum_abs, sum_true=sum_true, ape_vals=ape_vals)


def finalize_pct_metrics(sum_abs: float, sum_true: float, ape_list) -> dict:
    """Pool per-batch building blocks into EXACT WAPE / MdAPE / P90.
       sum_abs / sum_true : running sums (for WAPE).
       ape_list           : list of per-batch APE tensors (for the percentiles).
    """
    wape = (sum_abs / sum_true) if sum_true > _TINY else float("nan")
    cat = [a for a in (ape_list or []) if a is not None and a.numel() > 0]
    if cat:
        allape = torch.cat(cat)
        mdape   = allape.median().item()
        ape_p90 = torch.quantile(allape, 0.9).item()
    else:
        mdape = ape_p90 = float("nan")
    return dict(wape=wape, mdape=mdape, ape_p90=ape_p90)


@torch.no_grad()
def route_metrics_pooled(pred_seq: torch.Tensor,
                         trip_tt: torch.Tensor,
                         seq_mask: torch.Tensor) -> dict:
    pred_seq = torch.nan_to_num(pred_seq)
    trip_tt = torch.nan_to_num(trip_tt)
    seq_mask = seq_mask.bool()
    pred_route = (pred_seq * seq_mask.float()).sum(dim=1)
    n_routes = int(pred_route.shape[0])
    diff = (pred_route - trip_tt)
    mae  = diff.abs().mean().item()
    rmse = torch.sqrt(torch.mean(diff * diff)).item()
    pos = trip_tt > 0
    n_mape = int(pos.sum().item())
    # [2026-06-07] WAPE over all routes with positive truth (denominator-stable),
    # plus MdAPE / P90 of route APE (robust central tendency).
    sum_abs  = diff[pos].abs().sum().item() if pos.any() else 0.0
    sum_true = trip_tt[pos].clamp_min(0.0).sum().item() if pos.any() else 0.0
    wape = (sum_abs / sum_true) if sum_true > _TINY else float("nan")
    if pos.any():
        ape_vals = (diff[pos].abs() / trip_tt[pos].clamp_min(_TINY)).detach().cpu()
        mape    = ape_vals.mean().item()
        mdape   = ape_vals.median().item()
        ape_p90 = torch.quantile(ape_vals, 0.9).item()
    else:
        ape_vals = torch.empty(0)
        mape = mdape = ape_p90 = float("nan")
    return dict(mae=mae, rmse=rmse, mape=mape,
                wape=wape, mdape=mdape, ape_p90=ape_p90,
                n_routes=n_routes, n_mape=n_mape,
                sum_abs=sum_abs, sum_true=sum_true, ape_vals=ape_vals)

def compute_global_losses_and_metrics(pred_seq: torch.Tensor,
                                      trip_tt: torch.Tensor,
                                      seq_true: torch.Tensor,
                                      seq_mask: torch.Tensor,
                                      seq_link_len: Optional[torch.Tensor] = None,
                                      delta_huber: float = 1.0) -> dict:
    l_link  = link_loss_huber_macro(pred_seq, seq_true, seq_mask, delta=delta_huber)
    l_route = route_loss_mape(pred_seq, trip_tt, seq_mask)
    link_stats  = link_metrics_micro(pred_seq, seq_true, seq_mask, seq_link_len, L_min=L_MIN_METRIC, T_min=T_MIN_METRIC)
    route_stats = route_metrics_pooled(pred_seq, trip_tt, seq_mask)
    out = dict(link_loss=l_link, route_loss=l_route)
    out.update({
        "link_mae":  link_stats["mae"],
        "link_rmse": link_stats["rmse"],
        "link_mape": link_stats["mape"],
        "route_mae":  route_stats["mae"],
        "route_rmse": route_stats["rmse"],
        "route_mape": route_stats["mape"],
        "n_links": link_stats.get("n_links", 0),
        "n_links_mape": link_stats.get("n_mape", 0),
        "n_routes": route_stats.get("n_routes", 0),
        "n_routes_mape": route_stats.get("n_mape", 0),
    })
    return out

def composite_loss(link_loss: torch.Tensor,
                   route_loss: torch.Tensor,
                   loss_speed: Optional[torch.Tensor] = None,
                   loss_volume: Optional[torch.Tensor] = None,
                   w_link: float = 1.0,
                   w_route: float = 1.0,
                   w_speed: float = 0.0,
                   w_volume: float = 0.0) -> torch.Tensor:
    
    ls = w_link  * link_loss + w_route * route_loss
    if (loss_speed is not None) and (w_speed != 0.0):
        ls = ls + w_speed * loss_speed
    if (loss_volume is not None) and (w_volume != 0.0):
        ls = ls + w_volume * loss_volume
    return ls

def build_seq_mask(batch, device):
    # pick the base “on-route” flag
    if 'route_flag' in batch:
        base = batch['route_flag'].to(device).bool()
    elif 'valid_mask' in batch:
        base = batch['valid_mask'].to(device).bool()
    elif 'onroute_mask' in batch:
        base = batch['onroute_mask'].to(device).bool()
    else:
        raise KeyError("No route/on-route mask found (route_flag | valid_mask | onroute_mask).")

    # intersect with positional validity if we have indices
    if 'edge_seq' in batch:
        base = base & (batch['edge_seq'].to(device) >= 0)

    return base


# ============================================================================
# LinkETA-only training losses (log1p-Huber, length-weighted).
# DuETA / ConSTGAT use link_loss_huber_macro + route_loss_mape above.
# LinkETA replaced both with the log-space versions below for long-trip stability.
# Metrics (link_metrics_micro, route_metrics_pooled) are shared with the OLD
# block above so MAE / RMSE / MAPE are computed identically across all models.
# ============================================================================

def link_eta_in_seconds(model_link_eta: torch.Tensor,
                        seq_link_len: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.nan_to_num(model_link_eta).clamp_min(0.0)


def link_loss_huber_log_lenweighted(pred_seq: torch.Tensor,
                                    seq_true: torch.Tensor,
                                    seq_mask: torch.Tensor,
                                    link_len: Optional[torch.Tensor] = None,
                                    delta: float = 0.3) -> torch.Tensor:
    pred_seq = torch.nan_to_num(pred_seq).clamp_min(0.0)
    seq_true = torch.nan_to_num(seq_true).clamp_min(0.0)
    m = seq_mask.bool()
    if not m.any():
        return torch.zeros((), device=pred_seq.device)
    log_p = torch.log1p(pred_seq)
    log_t = torch.log1p(seq_true)
    per   = _huber_per_element(log_p - log_t, delta)
    if link_len is None:
        per_masked = per * m.float()
        cnt = m.sum(dim=1).clamp_min(1)
        per_route = per_masked.sum(dim=1) / cnt
        return per_route.mean()
    w = torch.nan_to_num(link_len).clamp_min(0.0) * m.float()
    denom = w.sum().clamp_min(1.0)
    return (per * w).sum() / denom


def route_loss_huber_log(pred_seq: torch.Tensor,
                         trip_tt: torch.Tensor,
                         seq_mask: torch.Tensor,
                         delta: float = 0.3) -> torch.Tensor:
    pred_seq = torch.nan_to_num(pred_seq).clamp_min(0.0)
    trip_tt  = torch.nan_to_num(trip_tt).clamp_min(0.0)
    seq_mask = seq_mask.bool()
    pred_route = (pred_seq * seq_mask.float()).sum(dim=1)
    pos = trip_tt > 0
    if not pos.any():
        return torch.zeros((), device=pred_seq.device)
    log_p = torch.log1p(pred_route[pos])
    log_t = torch.log1p(trip_tt[pos])
    return _huber_per_element(log_p - log_t, delta).mean()


def compute_global_losses_and_metrics_linketa(pred_seq: torch.Tensor,
                                              trip_tt: torch.Tensor,
                                              seq_true: torch.Tensor,
                                              seq_mask: torch.Tensor,
                                              seq_link_len: Optional[torch.Tensor] = None,
                                              delta_huber: float = 0.3) -> dict:
    l_link  = link_loss_huber_log_lenweighted(
        pred_seq, seq_true, seq_mask, link_len=seq_link_len, delta=delta_huber,
    )
    l_route = route_loss_huber_log(pred_seq, trip_tt, seq_mask, delta=delta_huber)
    link_stats  = link_metrics_micro(pred_seq, seq_true, seq_mask, seq_link_len,
                                     L_min=L_MIN_METRIC, T_min=T_MIN_METRIC)
    route_stats = route_metrics_pooled(pred_seq, trip_tt, seq_mask)
    out = dict(link_loss=l_link, route_loss=l_route)
    out.update({
        "link_mae":  link_stats["mae"],
        "link_rmse": link_stats["rmse"],
        "link_mape": link_stats["mape"],
        "route_mae":  route_stats["mae"],
        "route_rmse": route_stats["rmse"],
        "route_mape": route_stats["mape"],
        "n_links": link_stats.get("n_links", 0),
        "n_links_mape": link_stats.get("n_mape", 0),
        "n_routes": route_stats.get("n_routes", 0),
        "n_routes_mape": route_stats.get("n_mape", 0),
    })
    return out

