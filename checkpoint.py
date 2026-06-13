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

from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from torch import amp
from torch import nn, Tensor
import torch.nn.functional as F
import torch
from contextlib import nullcontext
from collections import defaultdict
import inspect
from collections import Counter
from typing import Optional


import os, time, random
import math, bisect, re, collections
from typing import List, Dict, Tuple, Optional, Sequence
from functools import lru_cache
import numpy as np
import pandas as pd
import torch

import warnings
# [2026-06-07] Silence the torch.load `weights_only=False` FutureWarning. Every
# load in this project reads our OWN trusted artifacts (checkpoints carrying
# optimizer/scheduler/RNG/cluster objects, precomputed tensors) for which
# weights_only=True cannot work. Opting out is the documented, safe choice; this
# filter just removes the repeated log spam.
warnings.filterwarnings("ignore", message=r".*weights_only.*", category=FutureWarning)


# ---------------------------------------------------------------------------
# [2026-06-07] Shared metric-table formatting so SOURCE (training_summary) and
# TARGET (transfer eval) logs print the SAME columns, consistently. Both link
# and route show: MAE, RMSE, MAPE, WAPE, MdAPE, P90.
# ---------------------------------------------------------------------------
_TBL_W = 150


def metrics_table_header(title: str, col2: str = "Split"):
    print(f"\n{'='*_TBL_W}")
    print(f"  {title}")
    print(f"{'='*_TBL_W}")
    print(f"  {'City':<14}{col2:<11}"
          f"{'L-MAE':>9}{'L-RMSE':>9}{'L-MAPE':>8}{'L-WAPE':>8}{'L-MdAPE':>8}{'L-P90':>8}"
          f"  |{'R-MAE':>10}{'R-RMSE':>10}{'R-MAPE':>8}{'R-WAPE':>8}{'R-MdAPE':>8}{'R-P90':>8}")
    print(f"  {'-'*(_TBL_W-2)}")


def metrics_table_row(city, col2, m: dict):
    """Print one metrics row. `m` is any dict with link_*/route_* keys; missing
    keys render as nan."""
    def g(k):
        v = m.get(k, float('nan'))
        try:
            return float(v)
        except (TypeError, ValueError):
            return float('nan')
    print(f"  {str(city):<14}{str(col2):<11}"
          f"{g('link_mae'):>9.3f}{g('link_rmse'):>9.3f}{g('link_mape'):>8.3f}"
          f"{g('link_wape'):>8.3f}{g('link_mdape'):>8.3f}{g('link_p90'):>8.3f}"
          f"  |{g('route_mae'):>10.3f}{g('route_rmse'):>10.3f}{g('route_mape'):>8.3f}"
          f"{g('route_wape'):>8.3f}{g('route_mdape'):>8.3f}{g('route_p90'):>8.3f}")


def metrics_table_footer():
    print(f"{'='*_TBL_W}")


class CheckpointManager:
    def __init__(self, dirpath="ckpts_DuETA", max_keep=3):
        self.dir = dirpath; self.max_keep = max_keep
        os.makedirs(self.dir, exist_ok=True)
        self.best_path = os.path.join(self.dir, "best.ckpt")
        self.last_path = os.path.join(self.dir, "last.ckpt")

    def _pack_rng(self):
        pkg = {"py_random": random.getstate(),
               "np_random": np.random.get_state(),
               "torch_cpu": torch.get_rng_state().cpu().numpy().tolist()}
        if torch.cuda.is_available(): pkg["torch_cuda"] = torch.cuda.get_rng_state_all()
        return pkg

    def _unpack_rng(self, pkg):
        try:
            random.setstate(pkg["py_random"]); np.random.set_state(pkg["np_random"])
            torch.set_rng_state(torch.tensor(pkg["torch_cpu"]))
            if torch.cuda.is_available() and "torch_cuda" in pkg:
                torch.cuda.set_rng_state_all(pkg["torch_cuda"])
        except Exception as _e:
            print("[ckpt] RNG restore warning:", _e)

    def save(self, tag_path, model, optimizer, scheduler, epoch, step, best_metric=None, extra=None):
        state = {
            "epoch": int(epoch), "step": int(step),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "best_metric": best_metric, "rng": self._pack_rng(),
            "timestamp": time.time(), "extra": extra or {},
        }
        tmp = tag_path + ".tmp"; torch.save(state, tmp); os.replace(tmp, tag_path)
        return tag_path

    def save_best(self, *args, **kwargs):
        p = self.save(self.best_path, *args, **kwargs)
        hist = os.path.join(self.dir, f"best_{int(time.time())}.ckpt")
        try:
            torch.save(torch.load(p, map_location="cpu"), hist)
            all_hist = sorted([f for f in os.listdir(self.dir) if f.startswith("best_") and f.endswith(".ckpt")])
            while len(all_hist) > self.max_keep:
                os.remove(os.path.join(self.dir, all_hist.pop(0)))
        except Exception as _e:
            print("[ckpt] history save warning:", _e)
        return p

    def save_last(self, *args, **kwargs):  return self.save(self.last_path, *args, **kwargs)

    def load(self, path, model, optimizer=None, scheduler=None, map_location=None):
        if not os.path.isfile(path): raise FileNotFoundError(path)
        pkg = torch.load(path, map_location=map_location or "cpu")
        model.load_state_dict(pkg["model"], strict=True)
        if optimizer is not None and pkg.get("optimizer") is not None: optimizer.load_state_dict(pkg["optimizer"])
        if scheduler is not None and pkg.get("scheduler") is not None: scheduler.load_state_dict(pkg["scheduler"])
        if "rng" in pkg: self._unpack_rng(pkg["rng"])
        start_epoch = int(pkg.get("epoch", 0)) + 1
        step = int(pkg.get("step", 0))
        best_metric = pkg.get("best_metric", float("inf"))
        return start_epoch, step, best_metric, pkg.get("extra", {}
                                                       )


_SUMMARY_METRIC_KEYS = (
    "link_mae", "link_rmse", "link_mape", "link_wape", "link_mdape", "link_p90",
    "route_mae", "route_rmse", "route_mape", "route_wape", "route_mdape", "route_p90",
)


def _safe_float(v):
    try:
        return float(v)
    except Exception:
        return float("nan")


def write_training_history(ckpt_dir, rows, method="meta",
                           best_summary=None, best_meta_iter=None):
    """[2026-06-07] Persist per-source metrics over meta-iters (CSV) + a best-iter
    summary (JSON). Two summary modes:

      * best_summary given -> SELECTED-MODEL summary: each city's metrics are the
        ones captured at the SINGLE iteration where the meta-selection loss
        (meta_loss_avg / avg_val_loss) was minimized — i.e. the deployed theta_0.
        This is the correct, consistent definition across MetaST/MetaTTE/MetaETA:
        every city is reported at the SAME iteration (the one that was selected).

      * best_summary None  -> LEGACY fallback: per-city argmin val route_mape.
        NOTE this picks a DIFFERENT iter per city and does NOT correspond to the
        deployed model; kept only for back-compat.

    rows           : list of dicts (meta_iter, city, link_*, route_*) for the CSV.
    best_summary   : {city: metrics_dict} captured at the selected iteration.
    best_meta_iter : int, the selected iteration (identical for every city).
    Writes training_history.csv + training_summary.json into ckpt_dir.
    """
    os.makedirs(ckpt_dir, exist_ok=True)

    # Always persist the full per-iter history CSV (diagnostic) when available.
    df = None
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(ckpt_dir, "training_history.csv"), index=False)

    summary = {}
    if best_summary is not None:
        # SELECTED-MODEL summary — both cities at the argmin-meta_loss iteration.
        for city, m in best_summary.items():
            summary[str(city)] = {
                "method": method,
                "best_meta_iter": int(best_meta_iter) if best_meta_iter is not None else -1,
                **{k: _safe_float(m.get(k)) for k in _SUMMARY_METRIC_KEYS},
            }
    elif df is not None and "route_mape" in df.columns and "city" in df.columns:
        # Legacy fallback (per-city argmin route_mape — NOT the deployed model).
        for city, g in df.groupby("city"):
            gv = g.dropna(subset=["route_mape"])
            if gv.empty:
                continue
            best = gv.loc[gv["route_mape"].idxmin()]
            summary[str(city)] = {
                "method": method,
                "best_meta_iter": int(best["meta_iter"]),
                **{k: _safe_float(best.get(k)) for k in _SUMMARY_METRIC_KEYS},
            }

    if not summary:
        return

    with open(os.path.join(ckpt_dir, "training_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  [training history] {os.path.join(ckpt_dir, 'training_history.csv')}"
          f"  +  training_summary.json")
    # [2026-06-07] Print the summary with the SAME table as the TARGET transfer eval.
    label = ("SELECTED-MODEL (per-city metrics @ argmin meta_loss)"
             if best_summary is not None
             else "SOURCE in-sample best (val/query, argmin route_mape)")
    metrics_table_header(f"{method} — {label}", col2="Best@iter")
    for c, s in summary.items():
        metrics_table_row(c, f"it{s['best_meta_iter']}", s)
    metrics_table_footer()
