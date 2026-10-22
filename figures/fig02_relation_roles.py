import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as cm
from fig01_relation_distinctness import ORDER, LABEL

NAME = "fig02_relation_roles"
CITY = "Porto"
DATA_FILE = cm.DATA / f"{NAME}.csv"
N_SLOTS = 12
RATIO_EDGES = [0.0, 0.25, 0.5, 0.75, 1.0, np.inf]
RATIO_LABEL = ["< 0.25", "0.25-0.5", "0.5-0.75", "0.75-1", "> 1"]


class _Batches:

    def __init__(self, loader, sink):
        self.loader, self.sink = loader, sink

    def __iter__(self):
        for batch in self.loader:
            self.sink["node_edge_idx"] = batch["node_edge_idx"]
            yield batch


class _Collector:
    def __init__(self, n_rel, cnt_idx, spd_idx, mean, std, vff, sink):
        import torch
        self.n_rel, self.cnt_idx, self.spd_idx = n_rel, cnt_idx, spd_idx
        self.mean, self.std, self.vff, self.sink = mean, std, vff, sink
        self.slots_rows, self.ratio_rows, self.norm_rows = [], [], []
        self.route_flag, self.x_in = None, None
        self.raw_count_max = 0.0
        self.raw_count_min = 0.0

    def pre(self, module, args):
        self.x_in, self.route_flag = args[0], args[2].bool()

    def __call__(self, layer, inputs, output):
        import torch
        x, masks = inputs
        B, M, D = x.shape
        cm.require(self.route_flag is not None and tuple(self.route_flag.shape) == (B, M),
                   "route flags of the batch missing")
        with torch.no_grad():
            norms = []
            for r in range(layer.R):
                q = layer.to_q[r](x).view(B, M, layer.heads, layer.dk)
                k = layer.to_k[r](x).view(B, M, layer.heads, layer.dk)
                v = layer.to_v[r](x).view(B, M, layer.heads, layer.dk)
                s = torch.einsum("bmhd,bnhd->bhmn", q, k) / math.sqrt(layer.dk)
                s = s.masked_fill(~masks[r].unsqueeze(1), float("-inf"))
                a = torch.softmax(s, dim=-1)
                a = torch.where(torch.isfinite(a), a, torch.zeros_like(a))
                norms.append(torch.einsum("bhmn,bnhd->bmhd", a, v).reshape(B, M, D).float().norm(dim=-1))
            route = self.route_flag
            raw = self.x_in.float() * self.std + self.mean
            cnt = raw[..., self.cnt_idx]
            spd = raw[..., self.spd_idx]
            self.raw_count_min = min(self.raw_count_min, float(cnt[route].min()))
            self.raw_count_max = max(self.raw_count_max, float(cnt[route].max()))
            seen = cnt > 1e-3
            n_seen = seen.sum(-1)
            first = torch.where(seen.any(-1), seen.float().argmax(-1), torch.full_like(n_seen, -1))
            latest = torch.gather(spd, -1, first.clamp(min=0).unsqueeze(-1)).squeeze(-1)
            vff = self.vff[self.sink["node_edge_idx"].to(x.device)]
            ratio = torch.where(first >= 0, latest / vff.clamp(min=1.0), torch.full_like(latest, float("nan")))
            self.slots_rows.append(n_seen[route].cpu().numpy())
            self.ratio_rows.append(ratio[route].cpu().numpy())
            self.norm_rows.append(torch.stack([n[route] for n in norms], 1).cpu().numpy())


def compute():
    import torch
    sys.path.insert(0, str(cm.ROOT))
    import main
    import LinkETA as li

    edges_gdf, df_all = main.load_dataset(CITY)
    li_mod = main._setup_linketa_new(CITY)
    li_mod.CKPT_DIRNAME = tempfile.mkdtemp(prefix="fig02_")
    core = li_mod.linketa_get_core_components(edges_gdf, df_all)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    ckpt = main.results_dir(CITY) / main.ckpt_dirname("LinkETA", CITY) / "best.ckpt"
    pkg = torch.load(ckpt, map_location=core["device"], weights_only=False)
    cm.require(abs(float(pkg["extra"]["r_max"]) - float(li_mod.R_MAX)) < 1e-9, "checkpoint r_max != R_MAX")
    core["model"].load_state_dict(pkg["model"], strict=True)
    rel_names = list(li.REL_NAMES_DEFAULT)

    cols = list(core["inst_feat_cols"])
    cnt_idx = [cols.index(f"count_vehicle_{s}") for s in range(N_SLOTS)]
    spd_idx = [cols.index(f"median_speed_{s}") for s in range(N_SLOTS)]
    dev = core["device"]
    mean = core["scalers"]["mean"].to(dev).float()
    std = core["scalers"]["std"].to(dev).float()
    vff = core["graph"]["vff_vec"].to(dev).float()

    sink = {}
    col = _Collector(len(rel_names), cnt_idx, spd_idx, mean, std, vff, sink)
    loaders = core["loaders"]
    original = loaders["test"]
    loaders["test"] = _Batches(original, sink)
    model, layer = core["model"], core["model"].layers[0]
    h1 = model.register_forward_pre_hook(col.pre)
    h2 = layer.register_forward_hook(col)
    try:
        metrics, _, _ = core["eval_linketa"](core, label="LinkETA")
    finally:
        h1.remove(); h2.remove(); loaders["test"] = original
    ref = json.loads((main.results_dir(CITY) / "eval_outputs" / "eval_metrics_linketa.json").read_text())["LinkETA"]
    cm.require(abs(metrics["route_mape"] - ref["route_mape"]) < 1e-3,
               f"route MAPE {metrics['route_mape']:.4f} != saved {ref['route_mape']:.4f}")
    slots = np.concatenate(col.slots_rows)
    ratio = np.concatenate(col.ratio_rows)
    norms = np.concatenate(col.norm_rows).astype(np.float64)
    cm.require(len(slots) == int(ref["n_links"]), f"route roads {len(slots)} != saved test links {int(ref['n_links'])}")
    cm.require(col.raw_count_min > -1e-3, f"recovered vehicle count below 0 ({col.raw_count_min})")
    reorder = [rel_names.index(n) for n in ORDER]
    norms = norms[:, reorder]

    rows = []
    def add(panel, group, mask):
        tot = norms[mask].sum(0)
        rows.append(dict(panel=panel, group=group, n_roads=int(mask.sum()),
                         **{f"norm_{n}": float(tot[i]) for i, n in enumerate(ORDER)}))
    for s in range(N_SLOTS + 1):
        add("data", str(s), slots == s)
    add("congestion", "no observed slot", np.isnan(ratio))
    for lo, hi, lab in zip(RATIO_EDGES[:-1], RATIO_EDGES[1:], RATIO_LABEL):
        add("congestion", lab, (ratio >= lo) & (ratio < hi))
    add("all", "all route roads", np.ones(len(slots), dtype=bool))
    meta = dict(city=CITY, checkpoint=str(ckpt), epoch=int(pkg["epoch"]), test_route_mape=float(metrics["route_mape"]),
                saved_route_mape=float(ref["route_mape"]), route_roads=int(len(slots)),
                raw_count_max=col.raw_count_max)
    return pd.DataFrame(rows), meta


def check(df):
    norm_cols = [f"norm_{n}" for n in ORDER]
    all_row = df[df.panel == "all"].iloc[0]
    total = float(all_row["n_roads"])
    shares = {}
    for panel in ("data", "congestion"):
        d = df[df.panel == panel].reset_index(drop=True)
        cm.require(int(d.n_roads.sum()) == int(total), f"{panel}: road counts {d.n_roads.sum()} != {total}")
        cm.require(np.allclose(d[norm_cols].sum(0).to_numpy(), all_row[norm_cols].to_numpy().astype(float),
                               rtol=1e-6), f"{panel}: group totals do not add up to the overall totals")
        s = d[norm_cols].to_numpy() / d[norm_cols].to_numpy().sum(1, keepdims=True)
        cm.require(np.allclose(s.sum(1), 1.0), f"{panel}: shares do not sum to 1")
        shares[panel] = (d, s)
    overall = all_row[norm_cols].to_numpy().astype(float)
    return shares, overall / overall.sum()


GROUP_OF = {"self": "Flow", "downstream": "Flow", "upstream": "Flow", "left_turn": "Turn-based",
            "right_turn": "Turn-based", "opposite": "Turn-based", "u_turn": "Turn-based", "high_order": "High-order"}
COLOUR = {"self": "#C6D8EA", "downstream": "#5A8DC0", "upstream": "#004488",
          "left_turn": "#FEE391", "right_turn": "#FEC44F", "opposite": "#EC7014", "u_turn": "#993404",
          "high_order": "#BB5566"}
HATCH = {"self": "", "downstream": "", "upstream": "", "left_turn": "////", "right_turn": "\\\\\\\\",
         "opposite": "xxx", "u_turn": "....", "high_order": ""}


def draw(shares, overall):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(cm.FULL, cm.FULL * 0.42), gridspec_kw=dict(width_ratios=[13, 6]))
    for ax, panel, xlabel in ((axes[0], "data", "Look-back slots with observed vehicles (of 12)"),
                              (axes[1], "congestion", "Observed speed / free-flow speed")):
        d, s = shares[panel]
        x = np.arange(len(d))
        base = np.zeros(len(d))
        for i, name in enumerate(ORDER):
            ax.bar(x, s[:, i], bottom=base, width=0.86, color=COLOUR[name], hatch=HATCH[name],
                   edgecolor="white", linewidth=0.4)
            base += s[:, i]
        for xi, n in zip(x, d.n_roads):
            ax.text(xi, 1.015, f"{n / 1000:.0f}k" if n >= 1000 else str(n), ha="center", va="bottom", fontsize=7)
        labels = list(d.group)
        if panel == "congestion":
            labels = ["none" if g == "no observed slot" else g for g in labels]
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7, rotation=0 if panel == "data" else 30)
        ax.set_xlim(-0.6, len(d) - 0.4); ax.set_ylim(0, 1.0)
        ax.set_xlabel(xlabel, fontsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("Share of the message received", fontsize=8)
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
    axes[1].set_yticklabels([])
    axes[0].text(-0.6, 1.075, "Route roads per group:", ha="left", va="bottom", fontsize=7)
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=COLOUR[n], hatch=HATCH[n], edgecolor="white", label=LABEL[n]) for n in ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=8, frameon=False, fontsize=7,
               bbox_to_anchor=(0.5, -0.02), handlelength=1.4, columnspacing=1.0)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.88, bottom=0.30, wspace=0.06)
    return fig


def run(from_data=False):
    cm.apply()
    if from_data:
        cm.require(DATA_FILE.exists(), f"{DATA_FILE} missing; run without --from-data first")
        df = pd.read_csv(DATA_FILE, dtype={"group": str})
    else:
        df, meta = compute()
        df.to_csv(DATA_FILE, index=False, float_format="%.10g")
        (cm.DATA / f"{NAME}_meta.json").write_text(json.dumps(meta, indent=2))
        print(f"  [data]  figures/data/{NAME}.csv  (route MAPE {meta['test_route_mape']:.4f} vs saved "
              f"{meta['saved_route_mape']:.4f}; {meta['route_roads']:,} route roads)")
    shares, overall = check(df)
    print("  overall share (route roads): " + ", ".join(f"{LABEL[n]} {100 * v:.1f}%" for n, v in zip(ORDER, overall)))
    cm.save(draw(shares, overall), NAME)


if __name__ == "__main__":
    run(from_data="--from-data" in sys.argv)

