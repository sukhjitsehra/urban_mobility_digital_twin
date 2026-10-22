import pandas as pd
import ast
import requests
import numpy as np
import json
from scipy.sparse import csr_matrix, vstack
from scipy.optimize import nnls
import scipy

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import math
from torch_geometric.data import Data
from torch_sparse import SparseTensor
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp import autocast, GradScaler
try:
    from torch.amp import GradScaler, autocast
except Exception:
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

import os, time

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

import DuETA as du

from DuETA import EdgeDynStore, build_first_order_relations_movement_exact, \
                   build_high_order_edge_index_dueta, collate_trips_dueta

from loss_functions import (
    link_loss_huber_macro, route_loss_mape,
    link_metrics_micro, route_metrics_pooled,
    compute_global_losses_and_metrics,
    selection_metric, L_MIN_METRIC, T_MIN_METRIC,
)


def _init_metric_accumulator():
    return {
        "link_loss_sum": 0.0,
        "route_loss_sum": 0.0,
        "link_mae_sum": 0.0,
        "link_sse_sum": 0.0,
        "link_mape_sum": 0.0,
        "route_mae_sum": 0.0,
        "route_sse_sum": 0.0,
        "route_mape_sum": 0.0,
        "total_links": 0,
        "total_links_mape": 0,
        "total_routes": 0,
        "total_routes_mape": 0,
    }


def _accumulate_batch_metrics(agg: dict, stats: dict):
    b_n_links       = int(stats.get("n_links", 0))
    b_n_links_mape  = int(stats.get("n_links_mape", 0))
    b_n_routes      = int(stats.get("n_routes", 0))
    b_n_routes_mape = int(stats.get("n_routes_mape", 0))

    def as_float(x):
        if isinstance(x, torch.Tensor):
            return float(x.item())
        return float(x)

    if b_n_routes > 0:
        agg["link_loss_sum"]  += as_float(stats["link_loss"])  * b_n_routes
        agg["route_loss_sum"] += as_float(stats["route_loss"]) * b_n_routes

    if b_n_links > 0:
        agg["link_mae_sum"] += as_float(stats["link_mae"]) * b_n_links
        agg["link_sse_sum"] += (as_float(stats["link_rmse"]) ** 2) * b_n_links

    if b_n_links_mape > 0:
        m_link = as_float(stats["link_mape"])
        if not math.isnan(m_link):
            agg["link_mape_sum"] += m_link * b_n_links_mape

    if b_n_routes > 0:
        agg["route_mae_sum"] += as_float(stats["route_mae"]) * b_n_routes
        agg["route_sse_sum"] += (as_float(stats["route_rmse"]) ** 2) * b_n_routes

    if b_n_routes_mape > 0:
        m_route = as_float(stats["route_mape"])
        if not math.isnan(m_route):
            agg["route_mape_sum"] += m_route * b_n_routes_mape

    agg["total_links"]       += b_n_links
    agg["total_links_mape"]  += b_n_links_mape
    agg["total_routes"]      += b_n_routes
    agg["total_routes_mape"] += b_n_routes_mape


def _finalize_metric_accumulator(agg: dict) -> dict:
    n_links       = agg["total_links"]
    n_links_mape  = agg["total_links_mape"]
    n_routes      = agg["total_routes"]
    n_routes_mape = agg["total_routes_mape"]

    if n_routes > 0:
        link_loss  = agg["link_loss_sum"]  / n_routes
        route_loss = agg["route_loss_sum"] / n_routes
    else:
        link_loss = route_loss = float("nan")

    if n_links > 0:
        link_mae  = agg["link_mae_sum"] / n_links
        link_rmse = math.sqrt(agg["link_sse_sum"] / n_links)
    else:
        link_mae = link_rmse = float("nan")

    if n_links_mape > 0:
        link_mape = agg["link_mape_sum"] / n_links_mape
    else:
        link_mape = float("nan")

    if n_routes > 0:
        route_mae  = agg["route_mae_sum"] / n_routes
        route_rmse = math.sqrt(agg["route_sse_sum"] / n_routes)
    else:
        route_mae = route_rmse = float("nan")

    if n_routes_mape > 0:
        route_mape = agg["route_mape_sum"] / n_routes_mape
    else:
        route_mape = float("nan")

    return dict(
        link_loss=link_loss,
        link_mae=link_mae,
        link_rmse=link_rmse,
        link_mape=link_mape,
        route_loss=route_loss,
        route_mae=route_mae,
        route_rmse=route_rmse,
        route_mape=route_mape,
        n_links=int(n_links),
        n_routes=int(n_routes),
    )


@torch.no_grad()
def eval_linketa(core,
                 ckpt_path: str = None,
                 label: str = "LinkETA",
                 split: str = "test"):
    device = core["device"]
    model  = core["model"].to(device)

    if "loaders" in core and split in core["loaders"]:
        loader = core["loaders"][split]
    else:
        key = f"{split}_loader"
        if key in core:
            loader = core[key]
        else:
            raise KeyError(f"No DataLoader found for split='{split}' in core.")

    vff_vec_local = core.get("graph", {}).get("vff_vec")
    if vff_vec_local is None:
        raise KeyError("eval_linketa requires core['graph']['vff_vec']")
    vff_vec_local = vff_vec_local.to(device)
    scaler_mean = core.get("scalers", {}).get("mean")
    scaler_std  = core.get("scalers", {}).get("std")
    if scaler_mean is None or scaler_std is None:
        raise KeyError("eval_linketa requires core['scalers']['mean'] and ['std']")
    scaler_mean = scaler_mean.to(device)
    scaler_std  = scaler_std.to(device)
    use_amp = bool(core.get("use_amp", False))

    if ckpt_path is not None:
        pkg = torch.load(ckpt_path, map_location=device)
        state = pkg["model"] if isinstance(pkg, dict) and "model" in pkg else pkg
        model.load_state_dict(state, strict=True)

    model.eval()

    agg = _init_metric_accumulator()

    sum_vol_mae = 0.0
    N_vol = 0

    trip_rows = []
    link_rows = []

    for batch in loader:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        node_edge_idx    = batch["node_edge_idx"]
        x_seq_nodes      = batch["x_seq"]
        route_flag_nodes = batch["route_flag"].bool()
        hop_idx          = batch["hop_idx"].long()
        masks_per_rel    = [m.to(device) for m in batch["masks_per_rel"]]

        route_seq_idx = batch["route_seq_idx"].long()
        seq_true_tt   = torch.nan_to_num(batch["seq_true_tt"], nan=0.0)
        seq_link_len  = torch.nan_to_num(batch["seq_link_len"], nan=0.0)
        seq_link_len  = seq_link_len.clamp_min(1e-3)
        seq_mask      = batch["seq_mask"].bool()
        trip_tt       = torch.nan_to_num(batch["trip_tt"], nan=0.0)

        lc_true_node = torch.nan_to_num(batch["logcnt1h_node"],  nan=0.0)

        gather_for_seq = lambda node_vals: torch.gather(
            node_vals, 1, route_seq_idx
        )

        lc_true = gather_for_seq(lc_true_node)

        vff_node = vff_vec_local[node_edge_idx]
        vff_seq  = gather_for_seq(vff_node)

        x_seq_norm = (x_seq_nodes - scaler_mean.view(1, 1, -1)) / scaler_std.view(1, 1, -1)
        x_seq_norm = torch.nan_to_num(x_seq_norm)

        with torch.amp.autocast('cuda', enabled=use_amp):
            (
                link_eta_seq,
                trip_eta,
                sp_pred,
                lc_pred,
                g,
                delta,
                eta_phys_seq,
            ) = model(
                x_seq_norm,
                hop_idx,
                route_flag_nodes,
                masks_per_rel,
                route_seq_idx,
                seq_link_len,
                vff_seq,
                seq_mask,
                r_max=core.get("r_max", 3.0),
                return_parts=True,
            )

            link_eta_seq = torch.nan_to_num(link_eta_seq, nan=0.0)
            trip_eta     = torch.nan_to_num(trip_eta,     nan=0.0)
            sp_pred      = torch.nan_to_num(sp_pred,      nan=0.0)
            lc_pred      = torch.nan_to_num(lc_pred,      nan=0.0)

            pred_seq = link_eta_seq
            seq_true = seq_true_tt

            stats = compute_global_losses_and_metrics(
                pred_seq=pred_seq,
                trip_tt=trip_tt,
                seq_true=seq_true,
                seq_mask=seq_mask,
                seq_link_len=seq_link_len,
                delta_huber=1.0,
            )

        _accumulate_batch_metrics(agg, stats)

        if seq_mask.any():
            vol_abs = (lc_pred[seq_mask] - lc_true[seq_mask]).abs().mean().item()
            n_vol   = int(seq_mask.sum().item())
            sum_vol_mae += vol_abs * n_vol
            N_vol       += n_vol

        route_true_from_links = (seq_true_tt * seq_mask.float()).sum(dim=1)
        yR = torch.where(trip_tt > 0, trip_tt, route_true_from_links)
        pR = (pred_seq * seq_mask.float()).sum(dim=1)

        trip_ids     = batch["trip_ids"]
        if isinstance(trip_ids, torch.Tensor):
            trip_ids_np = trip_ids.detach().cpu().numpy()
        else:
            trip_ids_np = np.array(trip_ids)
        y_true_np    = yR.detach().cpu().numpy()
        y_pred_np    = pR.detach().cpu().numpy()
        route_len_np = (seq_link_len * seq_mask.float()).sum(dim=1).detach().cpu().numpy()
        g_np         = g.detach().cpu().numpy()
        src_tag_np   = np.where(trip_tt.detach().cpu().numpy() > 0, "trip_tt", "sum_links")

        for i in range(len(trip_ids_np)):
            trip_rows.append((
                label,
                int(trip_ids_np[i]) if np.isscalar(trip_ids_np[i]) else int(trip_ids_np[i]),
                float(y_true_np[i]),
                float(y_pred_np[i]),
                src_tag_np[i],
                float(route_len_np[i]),
                float(g_np[i]),
            ))

        pred_seq_cpu = pred_seq.detach().cpu()
        seq_true_cpu = seq_true_tt.detach().cpu()
        mask_cpu     = seq_mask.detach().cpu()

        B0, L0 = pred_seq_cpu.shape
        for b in range(B0):
            tid = int(trip_ids_np[b]) if np.isscalar(trip_ids_np[b]) else int(trip_ids_np[b])
            for l in range(L0):
                if not bool(mask_cpu[b, l]):
                    continue
                link_rows.append((
                    label,
                    tid,
                    float(seq_true_cpu[b, l].item()),
                    float(pred_seq_cpu[b, l].item()),
                ))

    core_metrics = _finalize_metric_accumulator(agg)

    vol_mae = sum_vol_mae / N_vol if N_vol > 0 else float("nan")

    metrics = dict(core_metrics)
    metrics.update({"vol_mae": vol_mae})

    trip_df = pd.DataFrame(
        trip_rows,
        columns=["model", "TRIP_ID", "y_true", "y_pred",
                 "target_source", "route_len_m", "gain_g"],
    )
    link_df = pd.DataFrame(
        link_rows,
        columns=["model", "TRIP_ID", "true_tt_link", "pred_tt_link"],
    )

    return metrics, trip_df, link_df


@torch.no_grad()
def eval_dueta(core, ckpt_path: str = None, label: str = "DuETA", split: str = "test"):
    global CFG
    CFG = dict(core["cfg"])

    device = core["device"]
    model  = core["model"].to(device)

    if "loaders" in core and split in core["loaders"]:
        loader = core["loaders"][split]
    else:
        key = f"{split}_loader"
        if key in core:
            loader = core[key]
        else:
            raise KeyError(f"No DataLoader found for split='{split}' in core.")

    if ckpt_path:
        pkg = torch.load(ckpt_path, map_location=device)
        state = pkg["model"] if isinstance(pkg, dict) and "model" in pkg else pkg
        model.load_state_dict(state, strict=True)

    model.eval()

    agg = _init_metric_accumulator()
    trip_rows = []
    link_rows = []

    for batch in loader:
        static = du.norm_static(batch["static_feats"])
        dyn    = batch["dyn_slots"]

        hop_idx       = batch["hop_idx"]
        route_flag    = batch["route_flag"]
        masks_rel     = batch["masks_per_rel"]
        trip_tt       = torch.nan_to_num(batch["trip_tt"], nan=0.0)

        route_seq_idx = batch["route_seq_idx"]
        seq_mask      = batch["seq_mask"].bool()
        seq_true      = torch.nan_to_num(batch["seq_link_len"]*0 + batch["seq_true_tt"])
        seq_len_m     = torch.nan_to_num(batch.get("seq_link_len", torch.zeros_like(seq_true)))

        fwd_kwargs = {}
        if "prior_eta" in batch:
            fwd_kwargs["prior_link_eta"] = batch["prior_eta"]

        try:
            out = model(
                static, dyn,
                hop_idx, route_flag, masks_rel,
                route_seq_idx=route_seq_idx, seq_mask=seq_mask,
                seq_len_m=seq_len_m,
                node_edge_idx=batch["node_edge_idx"],
                road_type_idx=batch["road_type_idx"],
                junction_idx=batch["junction_idx"],
                **fwd_kwargs
            )
        except TypeError:
            out = model(static, dyn, hop_idx, route_flag, masks_rel, **fwd_kwargs)

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

        stats = compute_global_losses_and_metrics(
            pred_seq=pred_seq,
            trip_tt=trip_tt,
            seq_true=seq_true,
            seq_mask=seq_mask,
            seq_link_len=seq_len_m,
            delta_huber=1.0,
        )
        _accumulate_batch_metrics(agg, stats)

        route_true_from_links = (seq_true * seq_mask.float()).sum(dim=1)
        yR = torch.where(trip_tt > 0, trip_tt, route_true_from_links)
        pR = trip_eta

        trip_ids = batch["trip_ids"]
        if isinstance(trip_ids, torch.Tensor):
            trip_ids_np = trip_ids.detach().cpu().numpy()
        else:
            trip_ids_np = np.array(trip_ids)
        y_true_np    = yR.detach().cpu().numpy()
        y_pred_np    = pR.detach().cpu().numpy()
        route_len_np = (seq_len_m * seq_mask.float()).sum(dim=1).detach().cpu().numpy()
        src_tag_np   = np.where(trip_tt.detach().cpu().numpy() > 0, "trip_tt", "sum_links")

        for i in range(len(trip_ids_np)):
            trip_rows.append((
                label,
                int(trip_ids_np[i]),
                float(y_true_np[i]),
                float(y_pred_np[i]),
                src_tag_np[i],
                float(route_len_np[i]),
                0.0,
            ))

        pred_seq_cpu = pred_seq.detach().cpu()
        seq_true_cpu = seq_true.detach().cpu()
        mask_cpu     = seq_mask.detach().cpu()
        B0, L0 = pred_seq_cpu.shape
        for b in range(B0):
            tid = int(trip_ids_np[b])
            for l in range(L0):
                if not bool(mask_cpu[b, l]):
                    continue
                link_rows.append((
                    label, tid,
                    float(seq_true_cpu[b, l].item()),
                    float(pred_seq_cpu[b, l].item()),
                ))

    metrics = _finalize_metric_accumulator(agg)
    trip_df = pd.DataFrame(
        trip_rows,
        columns=["model", "TRIP_ID", "y_true", "y_pred",
                 "target_source", "route_len_m", "gain_g"],
    )
    link_df = pd.DataFrame(
        link_rows,
        columns=["model", "TRIP_ID", "true_tt_link", "pred_tt_link"],
    )
    return metrics, trip_df, link_df


@torch.no_grad()
def eval_constgat(core, ckpt_path: Optional[str] = None, label: str = "ConSTGAT", split: str = "test"):
    device = core["device"]
    model  = core["model"].to(device)

    if "loaders" in core and split in core["loaders"]:
        loader = core["loaders"][split]
    else:
        key = f"{split}_loader"
        if key in core:
            loader = core[key]
        else:
            raise KeyError(f"No DataLoader found for split='{split}' in core.")

    if ckpt_path:
        pkg = torch.load(ckpt_path, map_location=device)
        state = pkg["model"] if isinstance(pkg, dict) and "model" in pkg else pkg
        model.load_state_dict(state, strict=True)

    model.eval()

    agg = _init_metric_accumulator()
    trip_rows = []
    link_rows = []

    for batch in loader:
        es   = batch['edge_seq'].to(device)
        onroute_mask = batch['valid_mask'].to(device)
        M = es.size(1)
        pos_idx = torch.arange(M, device=device).unsqueeze(0).expand(es.size(0), -1)

        x_bg     = batch['x_bg'].to(device)
        true_tt  = batch['true_tt'].to(device)
        link_len = batch['link_len'].to(device)
        tripT    = batch['trip_tt'].to(device)

        nb_idx    = batch['nb_idx'].to(device)
        nb_mask   = batch['nb_mask'].to(device)
        nb_hist   = batch['nb_hist'].to(device)
        time_fk   = batch['time_fk'].to(device)
        nb_static = batch['nb_static'].to(device)

        eta_link, eta_route = model(
            edge_seq=es, x_bg=x_bg, pos_idx=pos_idx,
            nb_hist=nb_hist, nb_static=nb_static, time_fk=time_fk,
            nb_mask=nb_mask, onroute_mask=onroute_mask,
            nb_idx=nb_idx, bg_slots=batch['bg_slots'].to(device)
        )

        pred_seq  = eta_link
        seq_true  = true_tt
        seq_mask  = onroute_mask.bool()
        seq_len_m = link_len
        trip_tt   = tripT

        stats = compute_global_losses_and_metrics(
            pred_seq=pred_seq,
            trip_tt=trip_tt,
            seq_true=seq_true,
            seq_mask=seq_mask,
            seq_link_len=seq_len_m,
            delta_huber=1.0,
        )
        _accumulate_batch_metrics(agg, stats)

        route_true_from_links = (seq_true * seq_mask.float()).sum(dim=1)
        yR = torch.where(trip_tt > 0, trip_tt, route_true_from_links)
        pR = eta_route

        trip_ids = batch["trip_ids"]
        if isinstance(trip_ids, torch.Tensor):
            trip_ids_np = trip_ids.detach().cpu().numpy()
        else:
            trip_ids_np = np.array(trip_ids)
        y_true_np    = yR.detach().cpu().numpy()
        y_pred_np    = pR.detach().cpu().numpy()
        route_len_np = (seq_len_m * seq_mask.float()).sum(dim=1).detach().cpu().numpy()
        src_tag_np   = np.where(trip_tt.detach().cpu().numpy() > 0, "trip_tt", "sum_links")

        for i in range(len(trip_ids_np)):
            trip_rows.append((
                label,
                int(trip_ids_np[i]),
                float(y_true_np[i]),
                float(y_pred_np[i]),
                src_tag_np[i],
                float(route_len_np[i]),
                0.0,
            ))

        pred_seq_cpu = pred_seq.detach().cpu()
        seq_true_cpu = seq_true.detach().cpu()
        mask_cpu     = seq_mask.detach().cpu()
        B0, L0 = pred_seq_cpu.shape
        for b in range(B0):
            tid = int(trip_ids_np[b])
            for l in range(L0):
                if not bool(mask_cpu[b, l]):
                    continue
                link_rows.append((
                    label, tid,
                    float(seq_true_cpu[b, l].item()),
                    float(pred_seq_cpu[b, l].item()),
                ))

    metrics = _finalize_metric_accumulator(agg)
    trip_df = pd.DataFrame(
        trip_rows,
        columns=["model", "TRIP_ID", "y_true", "y_pred",
                 "target_source", "route_len_m", "gain_g"],
    )
    link_df = pd.DataFrame(
        link_rows,
        columns=["model", "TRIP_ID", "true_tt_link", "pred_tt_link"],
    )
    return metrics, trip_df, link_df

