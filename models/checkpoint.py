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

