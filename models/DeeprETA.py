import os
import json
import math
import time
import random
from typing import Dict, Optional

from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils import murmurhash3_32

import LinkETA as li
from loss_functions import route_metrics_pooled
from checkpoint import CheckpointManager

CKPT_DIRNAME = "ckpts_DeeprETA"
CACHE_DIR = None
TRIP_ENDS = None

NUM_EPOCHS = 100
PATIENCE = 15
SEED = 42

LR = 1e-3
OMEGA = 0.7
DELTA = 1.0
EMB_DIM = 8
N_QUANTILES = 256
HASH_BINS = 4096
GEOHASH_RES = (4, 5, 6, 7)
HASH_SEEDS = (1, 2)
ATT_DIM = 4
FC_DIM = 2048
MIN_SPEED = 0.5
Y0_MODE = "realtime"
N_SLOTS = 12

CONT_FEATS = ["dist_m", "y0", "rt_speed", "hist_speed"]
GEO_FEATS = [f"geo_{kind}_r{r}_h{h}" for r in GEOHASH_RES for kind in ("o", "d", "od") for h in range(len(HASH_SEEDS))]
CAT_FEATS = [("minute_of_week", 10080), ("minute_of_day", 1440), ("day_of_week", 7)]


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


_B32 = "0123456789bcdefghjkmnpqrstuvwxyz"


def geohash(lat: float, lng: float, length: int) -> str:
    la = min(int((lat + 90.0) / 180.0 * 4294967295.0), 4294967295)
    lo = min(int((lng + 180.0) / 360.0 * 4294967295.0), 4294967295)
    v = 0
    for i in range(31, -1, -1):
        v = (v << 1) | ((lo >> i) & 1)
        v = (v << 1) | ((la >> i) & 1)
    return "".join(_B32[(v >> (64 - 5 * (k + 1))) & 31] for k in range(length))


def hash_bins(key: str):
    return [murmurhash3_32(key, seed=s, positive=True) % HASH_BINS for s in HASH_SEEDS]


def geo_tokens(o_latlng, d_latlng) -> Dict[str, int]:
    out = {}
    for r in GEOHASH_RES:
        go, gd = geohash(*o_latlng, r), geohash(*d_latlng, r)
        for kind, key in (("o", go), ("d", gd), ("od", go + "_" + gd)):
            for h, b in enumerate(hash_bins(key)):
                out[f"geo_{kind}_r{r}_h{h}"] = b
    return out


class LinearAttention(nn.Module):
    def __init__(self, emb_dim: int, att_dim: int):
        super().__init__()
        self.q = nn.Linear(emb_dim, att_dim, bias=False)
        self.k = nn.Linear(emb_dim, att_dim, bias=False)
        self.v = nn.Linear(emb_dim, att_dim, bias=False)
        self.out = nn.Linear(att_dim, emb_dim, bias=False)

    def forward(self, e):
        q = F.elu(self.q(e)) + 1.0
        k = F.elu(self.k(e)) + 1.0
        v = self.v(e)
        kv = torch.einsum("bnd,bne->bde", k, v)
        z = 1.0 / (torch.einsum("bnd,bd->bn", q, k.sum(1)) + 1e-6)
        att = torch.einsum("bnd,bde,bn->bne", q, kv, z)
        return e + self.out(att)


class DeeprETANet(nn.Module):
    def __init__(self, sd_res: float, y_max: float):
        super().__init__()
        self.cat_emb = nn.ModuleList([nn.Embedding(v, EMB_DIM) for _, v in CAT_FEATS])
        self.cont_emb = nn.ModuleList([nn.Embedding(N_QUANTILES, EMB_DIM) for _ in CONT_FEATS])
        self.geo_emb = nn.ModuleList([nn.Embedding(HASH_BINS, EMB_DIM) for _ in GEO_FEATS])
        n = len(CAT_FEATS) + len(CONT_FEATS) + len(GEO_FEATS)
        self.interaction = LinearAttention(EMB_DIM, ATT_DIM)
        self.fc = nn.Sequential(nn.Linear(n * EMB_DIM, FC_DIM), nn.ReLU(), nn.Linear(FC_DIM, 1))
        self.calib_bias = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.fc[-1].weight)
        nn.init.zeros_(self.fc[-1].bias)
        self.register_buffer("sd_res", torch.tensor(float(sd_res)))
        self.register_buffer("y_max", torch.tensor(float(y_max)))

    def forward(self, cat, cont, geo, y0):
        e = torch.stack([m(cat[:, i]) for i, m in enumerate(self.cat_emb)]
                        + [m(cont[:, i]) for i, m in enumerate(self.cont_emb)]
                        + [m(geo[:, i]) for i, m in enumerate(self.geo_emb)], dim=1)
        h = self.interaction(e)
        r_hat = self.sd_res * (self.fc(h.flatten(1)).squeeze(-1) + self.calib_bias)
        return torch.clamp(y0 + r_hat, min=0.0, max=float(self.y_max))


def asymmetric_huber(pred, y, scale, omega=OMEGA, delta=DELTA):
    e = (pred - y) / scale
    a = e.abs()
    hub = torch.where(a < delta, 0.5 * e * e, delta * (a - 0.5 * delta))
    w = torch.where(y < pred, torch.full_like(e, omega), torch.full_like(e, 1.0 - omega))
    return (w * hub).mean()


def _route_arrays(g, edge2way, end_ts):
    ridx = g["route_seq_idx"].numpy()
    edges = g["node_edge_idx"].numpy()[ridx]
    lens = g["seq_link_len"].numpy().astype(np.float64)
    x = np.stack([li.SPEED_HISTORY.slots(edge2way[int(e)], end_ts, int(g["trip_id"]))[:N_SLOTS] for e in edges])
    med = x[:, :, li.CFG["dyn_prefixes"].index("median_speed")]
    cnt = x[:, :, li.CFG["dyn_prefixes"].index("count_vehicle")]
    return edges, lens, med, cnt


def _realtime_speed(med, cnt, vff):
    ok = (cnt > 0) & (med > 0)
    first = np.where(ok.any(1), ok.argmax(1), -1)
    v = np.where(first >= 0, med[np.arange(len(med)), np.clip(first, 0, None)], vff)
    return np.maximum(v, MIN_SPEED)


def deepreta_get_core_components(edges_gdf, df_all):
    li.CKPT_DIRNAME = CKPT_DIRNAME
    li.CACHE_DIR = CACHE_DIR
    lcore = li.linketa_get_core_components(edges_gdf, df_all)
    edge2way = lcore["datasets"]["train"].edge2way
    vff = lcore["graph"]["vff_vec"].detach().cpu().numpy().astype(np.float64)

    d = df_all[["TRIP_ID", "departure_time"]].copy()
    d["departure_time"] = pd.to_datetime(d["departure_time"], errors="coerce")
    dep = {int(k): v for k, v in d.dropna().groupby("TRIP_ID", sort=False)["departure_time"].first().items()}

    assert TRIP_ENDS is not None, "TRIP_ENDS not set (main.py sets it from sampled_<city>.csv)"

    hs, hc = {}, {}
    for g in lcore["datasets"]["train"].groups:
        edges = g["node_edge_idx"].numpy()[g["route_seq_idx"].numpy()]
        lens, tts = g["seq_link_len"].numpy(), g["seq_true_tt"].numpy()
        hr = dep[int(g["trip_id"])].hour
        for e, L, t in zip(edges, lens, tts):
            if t >= 1.0 and L >= 10.0:
                v = min(L / t, 40.0)
                for key in ((int(e), hr), (int(e), -1)):
                    hs[key] = hs.get(key, 0.0) + v; hc[key] = hc.get(key, 0) + 1
    hist = {k: hs[k] / hc[k] for k in hs}
    hps = {}
    for g in lcore["datasets"]["train"].groups:
        edges = g["node_edge_idx"].numpy()[g["route_seq_idx"].numpy()]
        lens, tts = g["seq_link_len"].numpy(), g["seq_true_tt"].numpy()
        hr = dep[int(g["trip_id"])].hour
        for e, L, t in zip(edges, lens, tts):
            if t >= 1.0 and L >= 10.0:
                p = max(t / L, 1.0 / 40.0)
                for key in ((int(e), hr), (int(e), -1)):
                    hps[key] = hps.get(key, 0.0) + p
    hist_pace = {k: hps[k] / hc[k] for k in hps}
    hsl = defaultdict(list)
    for g in lcore["datasets"]["train"].groups:
        edges = g["node_edge_idx"].numpy()[g["route_seq_idx"].numpy()]
        lens, tts = g["seq_link_len"].numpy(), g["seq_true_tt"].numpy()
        hr = dep[int(g["trip_id"])].hour
        for e, L, t in zip(edges, lens, tts):
            if t >= 1.0 and L >= 10.0:
                v = min(L / t, 40.0)
                for key in ((int(e), hr), (int(e), -1)):
                    hsl[key].append(v)
    hist_med = {k: float(np.median(v)) for k, v in hsl.items()}

    def _hist_without_own(e, hr, own):
        for key in ((e, hr), (e, -1)):
            o = own.get(key, [])
            n = hc.get(key, 0) - len(o)
            if n > 0:
                if not o:
                    return hist[key], hist_pace[key], hist_med[key]
                rest = list(hsl[key])
                for v, _ in o:
                    rest.remove(v)
                return ((hs[key] - sum(v for v, _ in o)) / n, (hps[key] - sum(p for _, p in o)) / n,
                        float(np.median(rest)))
        return vff[e], 1.0 / vff[e], vff[e]

    rows = {}
    for split in ("train", "val", "test"):
        recs = []
        for g in lcore["datasets"][split].groups:
            tid = int(g["trip_id"]); ts = dep[tid]
            edges, lens, med, cnt = _route_arrays(g, edge2way, ts.timestamp())
            hr = ts.hour
            own = defaultdict(list)
            if split == "train":
                for e, L, t in zip(edges, g["seq_link_len"].numpy(), g["seq_true_tt"].numpy()):
                    if t >= 1.0 and L >= 10.0:
                        for key in ((int(e), hr), (int(e), -1)):
                            own[key].append((min(L / t, 40.0), max(t / L, 1.0 / 40.0)))
            v_h, p_h, v_m = map(np.array, zip(*[_hist_without_own(int(e), hr, own) for e in edges]))
            v_rt = _realtime_speed(med, cnt, v_m)
            y0_rt = float((lens / v_rt).sum())
            y0_hs = float((lens / np.maximum(v_h, MIN_SPEED)).sum())
            y0_hp = float((lens * p_h).sum())
            y0_hm = float((lens / np.maximum(v_m, MIN_SPEED)).sum())
            y0 = {"realtime": y0_rt, "hist_speed": y0_hs, "hist_pace": y0_hp, "hist_median": y0_hm}[Y0_MODE]
            L = max(lens.sum(), 1e-6)
            o, dd = TRIP_ENDS[tid]
            rec = dict(trip_id=tid, minute_of_week=ts.dayofweek * 1440 + ts.hour * 60 + ts.minute,
                       minute_of_day=ts.hour * 60 + ts.minute, day_of_week=ts.dayofweek,
                       dist_m=float(lens.sum()), y0=y0,
                       y0_rt=y0_rt, y0_hs=y0_hs, y0_hp=y0_hp, y0_hm=y0_hm,
                       rt_speed=float((lens * v_rt).sum() / L), hist_speed=float((lens * v_h).sum() / L),
                       trip_tt=float(g["trip_tt"]))
            rec.update(geo_tokens((o[1], o[0]), (dd[1], dd[0])))
            recs.append(rec)
        rows[split] = pd.DataFrame(recs).set_index("trip_id")

    for split, df in rows.items():
        y = df.trip_tt.to_numpy()
        ok = y > 0
        msg = []
        for col, lab in (("y0_rt", "realtime"), ("y0_hs", "hist_speed"), ("y0_hp", "hist_pace"), ("y0_hm", "hist_median")):
            e = df[col].to_numpy()[ok] - y[ok]
            msg.append(f"{lab}: MAPE {np.mean(np.abs(e) / y[ok]) * 100:.2f}% MAE {np.mean(np.abs(e)):.1f}s "
                       f"bias {np.mean(e / y[ok]) * 100:+.1f}%")
        print(f"[DeeprETA] RE-ETA {split} (n={int(ok.sum())}) | " + " | ".join(msg) + f" | model uses {Y0_MODE}")

    edges_q = {f: np.unique(np.quantile(rows["train"][f].to_numpy(), np.linspace(0, 1, N_QUANTILES + 1)[1:-1]))
               for f in CONT_FEATS}
    for split, df in rows.items():
        for f in CONT_FEATS:
            df[f + "_q"] = np.searchsorted(edges_q[f], df[f].to_numpy(), side="right")

    tr = rows["train"]
    res = tr.trip_tt - tr.y0
    norm = dict(sd_tt=float(tr.trip_tt.std()), sd_res=float(res.std()), y_max=float(tr.trip_tt.max()))
    device = lcore["device"]
    set_seed(SEED)
    model = DeeprETANet(sd_res=norm["sd_res"], y_max=norm["y_max"]).to(device)
    return dict(device=device, model=model, linketa_core=lcore, loaders=lcore["loaders"], rows=rows,
                norm=norm, n_params=sum(p.numel() for p in model.parameters()))


def _tensors(core, split, trip_ids):
    df = core["rows"][split].loc[[int(t) for t in trip_ids]]
    dev = core["device"]
    cat = torch.tensor(df[[c for c, _ in CAT_FEATS]].to_numpy(np.int64), device=dev)
    cont = torch.tensor(df[[f + "_q" for f in CONT_FEATS]].to_numpy(np.int64), device=dev)
    geo = torch.tensor(df[GEO_FEATS].to_numpy(np.int64), device=dev)
    y0 = torch.tensor(df["y0"].to_numpy(), dtype=torch.float32, device=dev)
    return cat, cont, geo, y0


def _route_metrics(pred, trip_tt):
    return route_metrics_pooled(pred.unsqueeze(1), trip_tt, torch.ones_like(pred, dtype=torch.bool).unsqueeze(1))


def _run_epoch(core, split, optimizer=None):
    model, train = core["model"], optimizer is not None
    model.train() if train else model.eval()
    preds, trues, loss_sum, n = [], [], 0.0, 0
    with torch.set_grad_enabled(train):
        for batch in core["loaders"][split]:
            cat, cont, geo, y0 = _tensors(core, split, batch["trip_ids"])
            trip_tt = torch.nan_to_num(batch["trip_tt"].to(core["device"]), nan=0.0)
            pred = model(cat, cont, geo, y0)
            pos = trip_tt > 0
            loss = (asymmetric_huber(pred[pos], trip_tt[pos], core["norm"]["sd_tt"], omega=OMEGA)
                    if pos.any() else pred.sum() * 0.0)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                _g = [p.grad.detach() for p in model.parameters() if p.grad is not None]
                _gn = float(torch.norm(torch.stack([x.norm(2) for x in _g]))) if _g else float("nan")
                core.setdefault("_grad_norms", []).append(_gn)
                optimizer.step()
            loss_sum += float(loss.item()) * int(pos.sum().item()); n += int(pos.sum().item())
            preds.append(pred.detach()); trues.append(trip_tt)
    m = _route_metrics(torch.cat(preds), torch.cat(trues))
    return dict(loss=loss_sum / max(n, 1), route_mae=m["mae"], route_rmse=m["rmse"], route_mape=m["mape"])


def run_training(core):
    set_seed(SEED)
    out_dir = CKPT_DIRNAME
    os.makedirs(out_dir, exist_ok=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    model = core["model"]
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    ckpt = CheckpointManager(dirpath=out_dir, max_keep=3)
    print(f"[DeeprETA] parameters {core['n_params']:,} | epochs {NUM_EPOCHS} | patience {PATIENCE} | "
          f"lr {LR} cosine | omega {OMEGA} delta {DELTA} | tokens "
          f"{len(CAT_FEATS) + len(CONT_FEATS) + len(GEO_FEATS)} | device {core['device']}")
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
        if va["route_mape"] < best - 1e-6:
            best, since_best = va["route_mape"], 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            ckpt.save_best(model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch, step=epoch,
                           best_metric=float(best), extra={"val_metrics": va, "model": "DeeprETA", "omega": OMEGA})
        else:
            since_best += 1
        ckpt.save_last(model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch, step=epoch,
                       best_metric=float(best), extra={"val_metrics": va, "model": "DeeprETA", "omega": OMEGA})
        print(f"Epoch {epoch:03d} | {ep_time:6.1f}s | lr {history['lr'][-1]:.2e} | Train loss {tr['loss']:.4f} "
              f"MAPE {tr['route_mape']*100:.2f}% | Val MAE {va['route_mae']:.2f}s MAPE {va['route_mape']*100:.2f}% "
              f"(best {best*100:.2f}%)")
        if since_best >= PATIENCE:
            print("Early stopping triggered.")
            break
    total_s = time.perf_counter() - t_start
    peak = torch.cuda.max_memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else 0.0
    if best_state is not None:
        model.load_state_dict(best_state)
    metrics, _, _ = eval_deepreta(core)
    print("— Final Test —")
    print(f"Route: MAE={metrics['route_mae']:.4f} | RMSE={metrics['route_rmse']:.4f} | MAPE={metrics['route_mape']:.4f} "
          f"|| RE-ETA y0 alone: MAE={metrics['re_eta_route_mae']:.4f} | MAPE={metrics['re_eta_route_mape']:.4f}")
    summary = dict(test=metrics, best_val_route_mape=float(best), epochs_run=len(history["train"]),
                   total_train_time_s=total_s, mean_epoch_time_s=float(np.mean(history["epoch_time_s"])),
                   max_peak_gpu_mb=float(peak), n_params=core["n_params"],
                   batch_size=core["loaders"]["train"].batch_size, norm=core["norm"], torch=torch.__version__,
                   grad_norm_max_all=float(np.nanmax([e["grad_norm_max"] for e in history["train"]])) if history["train"] else None,
                   grad_norm_nonfinite_all=int(sum(e["grad_norm_nonfinite"] for e in history["train"])),
                   config=dict(LR=LR, OMEGA=OMEGA, DELTA=DELTA, EMB_DIM=EMB_DIM, N_QUANTILES=N_QUANTILES,
                               HASH_BINS=HASH_BINS, GEOHASH_RES=GEOHASH_RES, HASH_SEEDS=HASH_SEEDS,
                               ATT_DIM=ATT_DIM, FC_DIM=FC_DIM, MIN_SPEED=MIN_SPEED,
                               Y0_MODE=Y0_MODE,
                               NUM_EPOCHS=NUM_EPOCHS, PATIENCE=PATIENCE, SEED=SEED))
    with open(os.path.join(out_dir, "training_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, default=str)
    with open(os.path.join(out_dir, "training_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[summary] {os.path.join(out_dir, 'training_summary.json')}")
    return dict(test=metrics, best_selector=float(best))


@torch.no_grad()
def eval_deepreta(core, ckpt_path: Optional[str] = None, label: str = "DeeprETA", split: str = "test"):
    model = core["model"]
    if ckpt_path:
        pkg = torch.load(ckpt_path, map_location=core["device"])
        model.load_state_dict(pkg["model"] if isinstance(pkg, dict) and "model" in pkg else pkg, strict=True)
    model.eval()
    rows, preds, y0s, trues = [], [], [], []
    for batch in core["loaders"][split]:
        cat, cont, geo, y0 = _tensors(core, split, batch["trip_ids"])
        pred = model(cat, cont, geo, y0)
        trip_tt = torch.nan_to_num(batch["trip_tt"].to(core["device"]), nan=0.0)
        seq_mask = batch["seq_mask"].to(core["device"]).bool()
        seq_len = torch.nan_to_num(batch["seq_link_len"].to(core["device"]))
        y = torch.where(trip_tt > 0, trip_tt,
                        (torch.nan_to_num(batch["seq_true_tt"].to(core["device"])) * seq_mask.float()).sum(1))
        rlen = (seq_len * seq_mask.float()).sum(1)
        for i, tid in enumerate(batch["trip_ids"]):
            rows.append((label, int(tid), float(y[i]), float(pred[i]),
                         "trip_tt" if float(trip_tt[i]) > 0 else "sum_links", float(rlen[i]), 0.0))
        preds.append(pred); y0s.append(y0); trues.append(trip_tt)
    m = _route_metrics(torch.cat(preds), torch.cat(trues))
    m0 = _route_metrics(torch.cat(y0s), torch.cat(trues))
    nan = float("nan")
    metrics = dict(link_loss=nan, link_mae=nan, link_rmse=nan, link_mape=nan, route_loss=m["mae"],
                   route_mae=m["mae"], route_rmse=m["rmse"], route_mape=m["mape"], n_links=0, n_routes=len(rows),
                   re_eta_route_mae=m0["mae"], re_eta_route_rmse=m0["rmse"], re_eta_route_mape=m0["mape"])
    trip_df = pd.DataFrame(rows, columns=["model", "TRIP_ID", "y_true", "y_pred", "target_source", "route_len_m", "gain_g"])
    link_df = pd.DataFrame(columns=["model", "TRIP_ID", "true_tt_link", "pred_tt_link"])
    return metrics, trip_df, link_df

