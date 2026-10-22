import os
import json
import math
import time
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold

import LinkETA as li
from loss_functions import route_metrics_pooled
from checkpoint import CheckpointManager

CKPT_DIRNAME = "ckpts_TransETA"
CACHE_DIR = None

NUM_EPOCHS = 100
PATIENCE = 15
SEED = 42

LR = 1e-3
LR_STEP_EPOCHS = 2
LR_GAMMA = 0.5
N_HEADS = 8
D_HEAD = 64
D_MODEL = N_HEADS * D_HEAD
D_FCN = 512
N_ENC = 5
N_DEC = 5
N_GCN = 5
EMB_MONTH, EMB_DAY, EMB_HOUR, EMB_MINUTE = 4, 10, 8, 15
DF_PARAMS = dict(n_bins=255, bin_subsample=200000, max_layers=20, n_estimators=3, n_trees=100)

GCN_DIM = 64
T2V_DIM = 16
DROPOUT = 0.1
OOF_FOLDS = 5

FOREST_PREDS_FILE = "deep_forest_preds.npz"
FOREST_DIR = "deep_forest"


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Time2Vec(nn.Module):
    def __init__(self, k: int):
        super().__init__()
        self.lin = nn.Linear(1, 1)
        self.per = nn.Linear(1, k - 1)

    def forward(self, tau):
        return torch.cat([self.lin(tau), torch.sin(self.per(tau))], dim=-1)


class GCN(nn.Module):
    def __init__(self, in_dim: int, dim: int, n_layers: int):
        super().__init__()
        self.inp = nn.Linear(in_dim, dim)
        self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_layers)])

    def forward(self, x, a_norm):
        h = self.inp(x)
        for i, lin in enumerate(self.layers):
            h = torch.bmm(a_norm, lin(h))
            if i < len(self.layers) - 1:
                h = F.relu(h)
        return h


class InputTransform(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.gcn = GCN(in_dim, GCN_DIM, N_GCN)
        self.t2v = Time2Vec(T2V_DIM)
        self.proj = nn.Linear(GCN_DIM + 2 + T2V_DIM, D_MODEL)

    def forward(self, x_nodes, a_norm, route_seq_idx, lonlat_seq, tau):
        h = self.gcn(x_nodes, a_norm)
        h = torch.gather(h, 1, route_seq_idx.unsqueeze(-1).expand(-1, -1, h.size(-1)))
        t = self.t2v(tau).unsqueeze(1).expand(-1, h.size(1), -1)
        return self.proj(torch.cat([h, lonlat_seq, t], dim=-1))


def sinusoidal_pe(L: int, d: int, device) -> torch.Tensor:
    pos = torch.arange(L, device=device, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / d))
    pe = torch.zeros(L, d, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class TransETA(nn.Module):
    def __init__(self, in_dim: int, mean_trip_tt: float, sd_trip_tt: float):
        super().__init__()
        self.enc_in = InputTransform(in_dim)
        self.dec_in = InputTransform(in_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(D_MODEL, N_HEADS, D_FCN, DROPOUT, activation="relu", batch_first=True),
            N_ENC, enable_nested_tensor=False)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(D_MODEL, N_HEADS, D_FCN, DROPOUT, activation="relu", batch_first=True),
            N_DEC)
        self.dec_fcn = nn.Sequential(nn.Linear(D_MODEL, D_FCN), nn.ReLU())
        self.emb_month = nn.Embedding(12, EMB_MONTH)
        self.emb_day = nn.Embedding(7, EMB_DAY)
        self.emb_hour = nn.Embedding(24, EMB_HOUR)
        self.emb_minute = nn.Embedding(60, EMB_MINUTE)
        self.static_fcn = nn.Sequential(
            nn.Linear(EMB_MONTH + EMB_DAY + EMB_HOUR + EMB_MINUTE + 2, D_FCN), nn.ReLU())
        self.out = nn.Sequential(nn.Linear(2 * D_FCN, D_FCN), nn.ReLU(), nn.Linear(D_FCN, 1))
        self.register_buffer("tt_mu", torch.tensor(float(mean_trip_tt)))
        self.register_buffer("tt_sd", torch.tensor(float(sd_trip_tt)))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, x_nodes, a_norm, route_seq_idx, seq_mask, lonlat_seq, tau, static_cat, static_num):
        B, L = route_seq_idx.shape
        pe = sinusoidal_pe(L, D_MODEL, x_nodes.device).unsqueeze(0)
        pad = ~seq_mask
        enc = self.encoder(self.enc_in(x_nodes, a_norm, route_seq_idx, lonlat_seq, tau) + pe,
                           src_key_padding_mask=pad)
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=x_nodes.device), diagonal=1)
        dec = self.decoder(self.dec_in(x_nodes, a_norm, route_seq_idx, lonlat_seq, tau) + pe, enc,
                           tgt_mask=causal, tgt_key_padding_mask=pad, memory_key_padding_mask=pad)
        last = seq_mask.sum(dim=1).clamp_min(1) - 1
        h_dec = self.dec_fcn(dec[torch.arange(B, device=dec.device), last])
        s = torch.cat([self.emb_month(static_cat[:, 0]), self.emb_day(static_cat[:, 1]),
                       self.emb_hour(static_cat[:, 2]), self.emb_minute(static_cat[:, 3]), static_num], dim=-1)
        h_static = self.static_fcn(s)
        z = self.out(torch.cat([h_dec, h_static], dim=-1)).squeeze(-1)
        return F.softplus(self.tt_mu + self.tt_sd * z)


def _static_table(dataset, dep: Dict[int, pd.Timestamp]) -> pd.DataFrame:
    rows = []
    for g in dataset.groups:
        tid = int(g["trip_id"])
        ts = dep[tid]
        rows.append(dict(trip_id=tid, month=ts.month - 1, dow=ts.dayofweek, hour=ts.hour, minute=ts.minute,
                         tau=(ts.hour + ts.minute / 60.0 + ts.second / 3600.0) / 24.0,
                         dist_m=float(g["seq_link_len"].sum()), trip_tt=float(g["trip_tt"])))
    return pd.DataFrame(rows).set_index("trip_id")


def transeta_get_core_components(edges_gdf, df_all):
    li.CKPT_DIRNAME = CKPT_DIRNAME
    li.CACHE_DIR = CACHE_DIR
    lcore = li.linketa_get_core_components(edges_gdf, df_all)

    d = df_all[["TRIP_ID", "departure_time"]].copy()
    d["departure_time"] = pd.to_datetime(d["departure_time"], errors="coerce")
    d = d.dropna(subset=["departure_time"]).groupby("TRIP_ID", sort=False)["departure_time"].first()
    dep = {int(k): v for k, v in d.items()}
    static = {s: _static_table(lcore["datasets"][s], dep) for s in ("train", "val", "test")}

    ev = lcore["graph"]["edges_vff"]
    mid = ev.geometry.apply(lambda gm: gm.interpolate(0.5, normalized=True))
    xy = np.stack([mid.x.to_numpy(), mid.y.to_numpy()], axis=1).astype(np.float64)
    xy = (xy - xy.mean(0)) / np.where(xy.std(0) > 0, xy.std(0), 1.0)
    n_edges = int(ev["edge_idx"].max()) + 1
    lonlat = np.zeros((n_edges, 2), dtype=np.float32)
    lonlat[ev["edge_idx"].to_numpy()] = xy

    tr = static["train"]
    norm = dict(dist_mu=float(tr.dist_m.mean()), dist_sd=float(tr.dist_m.std() or 1.0),
                tt_mu=float(tr.trip_tt.mean()), tt_sd=float(tr.trip_tt.std() or 1.0))

    device = lcore["device"]
    set_seed(SEED)
    model = TransETA(in_dim=len(lcore["inst_feat_cols"]), mean_trip_tt=norm["tt_mu"],
                     sd_trip_tt=norm["tt_sd"]).to(device)
    return dict(device=device, model=model, linketa_core=lcore, loaders=lcore["loaders"],
                scalers=lcore["scalers"], lonlat=torch.tensor(lonlat, device=device),
                static=static, norm=norm, forest_preds=None,
                n_params=sum(p.numel() for p in model.parameters()))


FOREST_FEATS = ["month", "dow", "hour", "minute", "dist_m"]


def _new_forest():
    from deepforest import CascadeForestRegressor
    return CascadeForestRegressor(**DF_PARAMS, random_state=SEED, n_jobs=-1, verbose=0)


def fit_deep_forest(core, out_dir: str) -> dict:
    t0 = time.perf_counter()
    tr = core["static"]["train"]
    X, y = tr[FOREST_FEATS].to_numpy(np.float64), tr["trip_tt"].to_numpy(np.float64)
    oof = np.zeros(len(tr))
    for k, (a, b) in enumerate(KFold(OOF_FOLDS, shuffle=True, random_state=SEED).split(X)):
        m = _new_forest()
        m.fit(X[a], y[a])
        oof[b] = np.asarray(m.predict(X[b])).reshape(-1)
        print(f"[TransETA] deep forest fold {k + 1}/{OOF_FOLDS}: OOF MAE {np.abs(oof[b] - y[b]).mean():.2f} s")
    full = _new_forest()
    full.fit(X, y)
    preds = {"train": (tr.index.to_numpy(), oof)}
    for s in ("val", "test"):
        st = core["static"][s]
        preds[s] = (st.index.to_numpy(), np.asarray(full.predict(st[FOREST_FEATS].to_numpy(np.float64))).reshape(-1))
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, FOREST_PREDS_FILE),
             **{f"{s}_ids": p[0] for s, p in preds.items()}, **{f"{s}_pred": p[1] for s, p in preds.items()})
    try:
        full.save(os.path.join(out_dir, FOREST_DIR))
    except Exception as e:
        print(f"[TransETA] deep forest save warning: {e}")
    fit_s = time.perf_counter() - t0
    print(f"[TransETA] deep forest fitted in {fit_s:.1f} s | train OOF MAE {np.abs(oof - y).mean():.2f} s | "
          f"layers {full.n_layers_}")
    return dict(forest_fit_time_s=fit_s, forest_train_oof_mae_s=float(np.abs(oof - y).mean()),
                forest_n_layers=int(full.n_layers_))


def load_forest_preds(core, out_dir: str):
    z = np.load(os.path.join(out_dir, FOREST_PREDS_FILE))
    fp = {}
    for s in ("train", "val", "test"):
        ids, pred = z[f"{s}_ids"], z[f"{s}_pred"]
        if set(ids.tolist()) != set(core["static"][s].index.tolist()):
            raise RuntimeError(f"[TransETA] {s}: trip ids in {FOREST_PREDS_FILE} differ from the current split.")
        fp[s] = dict(zip(ids.tolist(), pred.tolist()))
    core["forest_preds"] = fp


def _prepare(core, batch, split):
    dev = core["device"]
    x = batch["x_seq"].to(dev)
    x = torch.nan_to_num((x - core["scalers"]["mean"].view(1, 1, -1)) / core["scalers"]["std"].view(1, 1, -1))
    masks = batch["masks_per_rel"]
    a = (masks[1] | masks[2]).to(dev)
    a = a | a.transpose(1, 2)
    a = a.float() + torch.eye(a.size(1), device=dev).unsqueeze(0)
    dinv = a.sum(-1).clamp_min(1.0).pow(-0.5)
    a_norm = dinv.unsqueeze(-1) * a * dinv.unsqueeze(1)
    route_seq_idx = batch["route_seq_idx"].to(dev)
    seq_mask = batch["seq_mask"].to(dev).bool()
    edge_seq = torch.gather(batch["node_edge_idx"].to(dev), 1, route_seq_idx)
    lonlat_seq = core["lonlat"][edge_seq]
    st = core["static"][split].loc[[int(t) for t in batch["trip_ids"]]]
    n = core["norm"]
    fp = core["forest_preds"][split]
    forest = np.array([fp[int(t)] for t in batch["trip_ids"]], dtype=np.float64)
    static_cat = torch.tensor(st[["month", "dow", "hour", "minute"]].to_numpy(np.int64), device=dev)
    static_num = torch.tensor(np.stack([(st.dist_m.to_numpy() - n["dist_mu"]) / n["dist_sd"],
                                        (forest - n["tt_mu"]) / n["tt_sd"]], axis=1), dtype=torch.float32, device=dev)
    tau = torch.tensor(st.tau.to_numpy(), dtype=torch.float32, device=dev).unsqueeze(1)
    trip_tt = torch.nan_to_num(batch["trip_tt"].to(dev), nan=0.0)
    seq_link_len = torch.nan_to_num(batch["seq_link_len"].to(dev), nan=0.0)
    return (x, a_norm, route_seq_idx, seq_mask, lonlat_seq, tau, static_cat, static_num), trip_tt, seq_link_len, seq_mask


def _route_metrics(pred, trip_tt):
    return route_metrics_pooled(pred.unsqueeze(1), trip_tt, torch.ones_like(pred, dtype=torch.bool).unsqueeze(1))


def _run_epoch(core, split, optimizer=None):
    model, train = core["model"], optimizer is not None
    model.train() if train else model.eval()
    preds, trues, loss_sum, n = [], [], 0.0, 0
    with torch.set_grad_enabled(train):
        for batch in core["loaders"][split]:
            inputs, trip_tt, _, _ = _prepare(core, batch, split)
            pred = model(*inputs)
            pos = trip_tt > 0
            loss = (pred[pos] - trip_tt[pos]).abs().mean() if pos.any() else pred.sum() * 0.0
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                _g = [p.grad.detach() for p in model.parameters() if p.grad is not None]
                _gn = float(torch.norm(torch.stack([x.norm(2) for x in _g]))) if _g else float("nan")
                core.setdefault("_grad_norms", []).append(_gn)
                optimizer.step()
            loss_sum += float(loss.item()) * int(pos.sum().item())
            n += int(pos.sum().item())
            preds.append(pred.detach())
            trues.append(trip_tt)
    m = _route_metrics(torch.cat(preds), torch.cat(trues))
    return dict(loss_mae=loss_sum / max(n, 1), route_mae=m["mae"], route_rmse=m["rmse"], route_mape=m["mape"])


def run_training(core):
    set_seed(SEED)
    out_dir = CKPT_DIRNAME
    os.makedirs(out_dir, exist_ok=True)
    device = core["device"]
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    forest_info = fit_deep_forest(core, out_dir)
    load_forest_preds(core, out_dir)

    model = core["model"]
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_EPOCHS, gamma=LR_GAMMA)
    ckpt = CheckpointManager(dirpath=out_dir, max_keep=3)
    print(f"[TransETA] parameters {core['n_params']:,} | epochs {NUM_EPOCHS} | patience {PATIENCE} | "
          f"lr {LR} (x{LR_GAMMA} every {LR_STEP_EPOCHS} epochs) | device {device}")

    history = {"train": [], "val": [], "epoch_time_s": [], "lr": []}
    best, best_state, since_best = float("inf"), None, 0
    t_start = time.perf_counter()
    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.perf_counter()
        tr = _run_epoch(core, "train", optimizer)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ep_time = time.perf_counter() - t0
        _t1 = time.perf_counter()
        va = _run_epoch(core, "val")
        _g = np.asarray(core.pop("_grad_norms", []), dtype=float)
        _fin = _g[np.isfinite(_g)]
        _trk = {"train_time_s": ep_time, "val_time_s": time.perf_counter() - _t1, "n_steps": int(len(_g)),
                "grad_norm_nonfinite": int(len(_g) - len(_fin)),
                "grad_norm_mean": float(_fin.mean()) if len(_fin) else float("nan"),
                "grad_norm_p99": float(np.percentile(_fin, 99)) if len(_fin) else float("nan"),
                "grad_norm_max": float(_fin.max()) if len(_fin) else float("nan"),
                "peak_gpu_mb": float(torch.cuda.max_memory_allocated() / 1024 ** 2) if torch.cuda.is_available() else float("nan")}
        tr.update(_trk)
        print(f"[track] epoch {epoch:03d}: train {_trk['train_time_s']:.0f}s val {_trk['val_time_s']:.0f}s | "
              f"grad norm mean {_trk['grad_norm_mean']:.3f} p99 {_trk['grad_norm_p99']:.3f} max {_trk['grad_norm_max']:.3f} "
              f"non-finite {_trk['grad_norm_nonfinite']} | peak GPU {_trk['peak_gpu_mb']:.0f} MB", flush=True)
        history["train"].append(tr); history["val"].append(va)
        history["epoch_time_s"].append(ep_time); history["lr"].append(optimizer.param_groups[0]["lr"])
        scheduler.step()
        score = va["route_mape"]
        if score < best - 1e-6:
            best, since_best = score, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            ckpt.save_best(model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch, step=epoch,
                           best_metric=float(best), extra={"val_metrics": va, "model": "TransETA"})
        else:
            since_best += 1
        ckpt.save_last(model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch, step=epoch,
                       best_metric=float(best), extra={"val_metrics": va, "model": "TransETA"})
        print(f"Epoch {epoch:03d} | {ep_time:6.1f}s | lr {history['lr'][-1]:.2e} | "
              f"Train MAE {tr['loss_mae']:.2f}s MAPE {tr['route_mape']*100:.2f}% | "
              f"Val MAE {va['route_mae']:.2f}s MAPE {va['route_mape']*100:.2f}% (best {best*100:.2f}%)")
        if since_best >= PATIENCE:
            print("Early stopping triggered.")
            break
    total_s = time.perf_counter() - t_start
    peak = torch.cuda.max_memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else 0.0
    if best_state is not None:
        model.load_state_dict(best_state)

    metrics, _, _ = eval_transeta(core)
    print("— Final Test —")
    print(f"Route: MAE={metrics['route_mae']:.4f} | RMSE={metrics['route_rmse']:.4f} | MAPE={metrics['route_mape']:.4f}")
    summary = dict(test=metrics, best_val_route_mape=float(best), epochs_run=len(history["train"]),
                   total_train_time_s=total_s, mean_epoch_time_s=float(np.mean(history["epoch_time_s"])),
                   max_peak_gpu_mb=float(peak), n_params=core["n_params"], batch_size=core["loaders"]["train"].batch_size,
                   grad_norm_max_all=float(np.nanmax([e["grad_norm_max"] for e in history["train"]])) if history["train"] else None,
                   grad_norm_nonfinite_all=int(sum(e["grad_norm_nonfinite"] for e in history["train"])),
                   **forest_info, torch=torch.__version__,
                   config=dict(LR=LR, LR_STEP_EPOCHS=LR_STEP_EPOCHS, LR_GAMMA=LR_GAMMA, D_MODEL=D_MODEL, N_HEADS=N_HEADS,
                               D_FCN=D_FCN, N_ENC=N_ENC, N_DEC=N_DEC, N_GCN=N_GCN, GCN_DIM=GCN_DIM, T2V_DIM=T2V_DIM,
                               DROPOUT=DROPOUT, OOF_FOLDS=OOF_FOLDS, DF_PARAMS=DF_PARAMS,
                               NUM_EPOCHS=NUM_EPOCHS, PATIENCE=PATIENCE, SEED=SEED))
    with open(os.path.join(out_dir, "training_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, default=str)
    with open(os.path.join(out_dir, "training_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[summary] {os.path.join(out_dir, 'training_summary.json')}")
    return dict(test=metrics, best_selector=float(best))


@torch.no_grad()
def eval_transeta(core, ckpt_path: Optional[str] = None, label: str = "TransETA", split: str = "test"):
    model = core["model"]
    if ckpt_path:
        pkg = torch.load(ckpt_path, map_location=core["device"])
        model.load_state_dict(pkg["model"] if isinstance(pkg, dict) and "model" in pkg else pkg, strict=True)
    if core["forest_preds"] is None:
        load_forest_preds(core, os.path.dirname(ckpt_path) if ckpt_path else CKPT_DIRNAME)
    model.eval()
    rows, preds, trues = [], [], []
    for batch in core["loaders"][split]:
        inputs, trip_tt, seq_link_len, seq_mask = _prepare(core, batch, split)
        pred = model(*inputs)
        route_true_from_links = (torch.nan_to_num(batch["seq_true_tt"].to(pred.device)) * seq_mask.float()).sum(1)
        y = torch.where(trip_tt > 0, trip_tt, route_true_from_links)
        rlen = (seq_link_len * seq_mask.float()).sum(1)
        for i, tid in enumerate(batch["trip_ids"]):
            rows.append((label, int(tid), float(y[i]), float(pred[i]),
                         "trip_tt" if float(trip_tt[i]) > 0 else "sum_links", float(rlen[i]), 0.0))
        preds.append(pred); trues.append(trip_tt)
    m = _route_metrics(torch.cat(preds), torch.cat(trues))
    nan = float("nan")
    metrics = dict(link_loss=nan, link_mae=nan, link_rmse=nan, link_mape=nan, route_loss=m["mae"],
                   route_mae=m["mae"], route_rmse=m["rmse"], route_mape=m["mape"], n_links=0, n_routes=len(rows))
    trip_df = pd.DataFrame(rows, columns=["model", "TRIP_ID", "y_true", "y_pred", "target_source", "route_len_m", "gain_g"])
    link_df = pd.DataFrame(columns=["model", "TRIP_ID", "true_tt_link", "pred_tt_link"])
    return metrics, trip_df, link_df

