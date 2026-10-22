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

NAME = "fig01_relation_distinctness"
CITY = "Chengdu"
ORDER = ["self", "downstream", "upstream", "left_turn", "right_turn", "opposite", "u_turn", "high_order"]
LABEL = {"self": "Self", "downstream": "Downstream", "upstream": "Upstream", "left_turn": "Left-turn",
         "right_turn": "Right-turn", "opposite": "Opposite", "u_turn": "U-turn", "high_order": "High-order"}
GROUPS = [("Flow", 0, 2), ("Turn-based", 3, 6), ("High-order", 7, 7)]
SCOPE = "route"


def _data_file(scope):
    return cm.DATA / (f"{NAME}.csv" if scope == "route" else f"{NAME}_allnodes.csv")


DATA_FILE = _data_file(SCOPE)


def _graph_pairs(ds, rel_names):
    pairs = {}
    for r, name in enumerate(rel_names):
        s = set()
        for src, dsts in ds.rel_src2dst_list[r].items():
            for d in dsts:
                s.add((int(src), int(d)))
        pairs[name] = s
    return pairs


class _MessageStats:

    def __init__(self, n_rel, dim, scope="all"):
        import torch
        z = lambda *s: torch.zeros(*s, dtype=torch.float64)
        self.R, self.D, self.scope = n_rel, dim, scope
        self.norm_sum = z(n_rel)
        self.n_nodes = 0
        self.pair_n = z(n_rel, n_rel)
        self.pair_sx = z(n_rel, n_rel, dim); self.pair_sy = z(n_rel, n_rel, dim)
        self.pair_xy = z(n_rel, n_rel, dim, dim)
        self.pair_xx = z(n_rel, n_rel, dim, dim); self.pair_yy = z(n_rel, n_rel, dim, dim)
        self.route_flag = None

    def __call__(self, layer, inputs, output):
        import torch
        x, masks = inputs
        B, M, D = x.shape
        cm.require(self.route_flag is not None and tuple(self.route_flag.shape) == (B, M),
                   "route flags of the batch missing (model pre-hook did not run)")
        with torch.no_grad():
            msgs = []
            for r in range(layer.R):
                q = layer.to_q[r](x).view(B, M, layer.heads, layer.dk)
                k = layer.to_k[r](x).view(B, M, layer.heads, layer.dk)
                v = layer.to_v[r](x).view(B, M, layer.heads, layer.dk)
                s = torch.einsum("bmhd,bnhd->bhmn", q, k) / math.sqrt(layer.dk)
                s = s.masked_fill(~masks[r].unsqueeze(1), float("-inf"))
                a = torch.softmax(s, dim=-1)
                a = torch.where(torch.isfinite(a), a, torch.zeros_like(a))
                msgs.append(torch.einsum("bhmn,bnhd->bmhd", a, v).reshape(B, M, D))
            connected = self.route_flag.to(x.device).clone()
            for r in range(layer.R):
                connected |= masks[r].any(-1) | masks[r].any(-2)
            if self.scope == "route":
                connected = self.route_flag.to(x.device).clone()
            X = [m[connected].double().cpu() for m in msgs]
            recv = [masks[r].any(-1)[connected].cpu() for r in range(layer.R)]
        self.n_nodes += X[0].shape[0]
        for r in range(self.R):
            self.norm_sum[r] += X[r].norm(dim=1).sum()
            for t in range(r + 1, self.R):
                both = recv[r] & recv[t]
                if both.any():
                    a, b = X[r][both], X[t][both]
                    self.pair_n[r, t] += a.shape[0]
                    self.pair_sx[r, t] += a.sum(0); self.pair_sy[r, t] += b.sum(0)
                    self.pair_xy[r, t] += a.T @ b
                    self.pair_xx[r, t] += a.T @ a; self.pair_yy[r, t] += b.T @ b


def _linear_cka(xy, xx, yy, sx, sy, n):
    import torch
    mx, my = sx / n, sy / n
    cxy = xy - n * torch.outer(mx, my)
    cxx = xx - n * torch.outer(mx, mx)
    cyy = yy - n * torch.outer(my, my)
    return float(torch.linalg.norm(cxy) ** 2 / (torch.linalg.norm(cxx) * torch.linalg.norm(cyy)))


def compute():
    import torch
    sys.path.insert(0, str(cm.ROOT))
    import main
    import LinkETA as li

    edges_gdf, df_all = main.load_dataset(CITY)
    li_mod = main._setup_linketa_new(CITY)
    li_mod.CKPT_DIRNAME = tempfile.mkdtemp(prefix="fig01_")
    core = li_mod.linketa_get_core_components(edges_gdf, df_all)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    ckpt = main.results_dir(CITY) / main.ckpt_dirname("LinkETA", CITY) / "best.ckpt"
    pkg = torch.load(ckpt, map_location=core["device"], weights_only=False)
    cm.require(abs(float(pkg["extra"]["r_max"]) - float(li_mod.R_MAX)) < 1e-9,
               f"checkpoint r_max {pkg['extra']['r_max']} != pipeline R_MAX {li_mod.R_MAX}")
    core["model"].load_state_dict(pkg["model"], strict=True)
    rel_names = list(li.REL_NAMES_DEFAULT)
    cm.require(sorted(rel_names) == sorted(ORDER), f"relation names changed: {rel_names}")

    pairs = _graph_pairs(core["datasets"]["train"], rel_names)
    cm.require(pairs["u_turn"] <= pairs["downstream"], "U-turn pairs not all downstream pairs")

    model, layer = core["model"], core["model"].layers[0]
    all_stats = {s: _MessageStats(len(rel_names), model.hidden_dim, scope=s) for s in ("all", "route")}

    def take_route_flag(module, args):
        for st in all_stats.values():
            st.route_flag = args[2].bool()
    h1 = model.register_forward_pre_hook(take_route_flag)
    hs = [layer.register_forward_hook(st) for st in all_stats.values()]
    metrics, _, _ = core["eval_linketa"](core, label="LinkETA")
    h1.remove()
    for h in hs:
        h.remove()
    ref = json.loads((main.results_dir(CITY) / "eval_outputs" / "eval_metrics_linketa.json").read_text())["LinkETA"]
    cm.require(abs(metrics["route_mape"] - ref["route_mape"]) < 1e-3,
               f"hooked evaluation route MAPE {metrics['route_mape']:.4f} != saved {ref['route_mape']:.4f}")

    idx = {n: i for i, n in enumerate(rel_names)}
    out = {}
    for scope, stats in all_stats.items():
        out[scope] = _table(stats, pairs, idx)
    cm.require(all_stats["route"].n_nodes < all_stats["all"].n_nodes, "route scope did not restrict the nodes")
    meta = dict(city=CITY, checkpoint=str(ckpt), epoch=int(pkg["epoch"]), r_max=float(pkg["extra"]["r_max"]),
                test_route_mape=float(metrics["route_mape"]), saved_route_mape=float(ref["route_mape"]),
                test_nodes=int(all_stats["all"].n_nodes), route_nodes=int(all_stats["route"].n_nodes), scope=SCOPE)
    return out, meta


def _table(stats, pairs, idx):
    share = (stats.norm_sum / stats.norm_sum.sum()).numpy()
    rows = []
    for a in ORDER:
        for b in ORDER:
            ia, ib = idx[a], idx[b]
            u = len(pairs[a] | pairs[b])
            jac = len(pairs[a] & pairs[b]) / u if u else float("nan")
            if a == b:
                cka = 1.0
            else:
                r, t = min(ia, ib), max(ia, ib)
                n = float(stats.pair_n[r, t])
                cka = (_linear_cka(stats.pair_xy[r, t], stats.pair_xx[r, t], stats.pair_yy[r, t],
                                   stats.pair_sx[r, t], stats.pair_sy[r, t], n) if n > 1 else float("nan"))
            rows.append(dict(relation_a=a, relation_b=b, jaccard=jac, cka=cka,
                             nodes_both=float(stats.pair_n[min(ia, ib), max(ia, ib)]) if a != b else float("nan"),
                             message_share_a=float(share[ia]), pairs_a=len(pairs[a])))
    return pd.DataFrame(rows)


def check(df):
    J = df.pivot(index="relation_a", columns="relation_b", values="jaccard").loc[ORDER, ORDER].to_numpy()
    C = df.pivot(index="relation_a", columns="relation_b", values="cka").loc[ORDER, ORDER].to_numpy()
    share = df.drop_duplicates("relation_a").set_index("relation_a").loc[ORDER, "message_share_a"].to_numpy()
    cm.require(np.allclose(J, J.T, equal_nan=True) and np.allclose(C, C.T, atol=1e-9, equal_nan=True),
               "matrices not symmetric")
    cm.require(np.nanmin(J) >= 0 and np.nanmax(J) <= 1 + 1e-9, "Jaccard outside [0, 1]")
    cm.require(np.nanmin(C) >= -1e-9 and np.nanmax(C) <= 1 + 1e-6, "CKA outside [0, 1]")
    cm.require(np.allclose(np.diag(J), 1.0), "Jaccard diagonal is not 1")
    cm.require(abs(share.sum() - 1) < 1e-6, f"message shares sum to {share.sum()}")
    return J, C, share


def draw(J, C, share):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle

    cmap_j = LinearSegmentedColormap.from_list("tol_green", ["#FFFFFF", "#CCDDAA", "#228833"])
    cmap_c = LinearSegmentedColormap.from_list("tol_blue", cm.SEQ_BLUE)
    n = len(ORDER)
    w = 0.48 * 468 / 72.27 * 25.4 * cm.MM
    fig = plt.figure(figsize=(w, w * 0.945))
    ax = fig.add_axes([0.175, 0.01, 0.74, 0.78])
    ax.set_xlim(0, n); ax.set_ylim(n, 0); ax.set_aspect("equal"); ax.axis("off")
    for i in range(n):
        for j in range(n):
            if i == j:
                face, txt, col, weight = cm.GREY, f"{100 * share[i]:.1f}%", cm.INK, "bold"
            elif j > i:
                v = J[i, j]
                face, txt, col, weight = cmap_j(v), f"{v:.3f}", (cm.INK if v < 0.55 else "white"), "normal"
            else:
                v = C[i, j]
                face, txt, col, weight = cmap_c(v), f"{v:.2f}", (cm.INK if v < 0.55 else "white"), "normal"
            ax.add_patch(Rectangle((j, i), 1, 1, facecolor=face, edgecolor="white", linewidth=0.8))
            ax.text(j + 0.5, i + 0.5, txt, ha="center", va="center", fontsize=7, color=col, fontweight=weight)
    for k, name in enumerate(ORDER):
        ax.text(-0.12, k + 0.5, LABEL[name], ha="right", va="center", fontsize=7)
        ax.text(k + 0.5, -0.12, LABEL[name], ha="left", va="bottom", fontsize=7, rotation=45,
                rotation_mode="anchor")
    for title, a, b in GROUPS:
        if a > 0:
            ax.plot([a, a], [0, n], color=cm.INK, linewidth=0.8)
            ax.plot([0, n], [a, a], color=cm.INK, linewidth=0.8)
    ax.add_patch(Rectangle((0, 0), n, n, fill=False, edgecolor=cm.INK, linewidth=0.8))
    ax.set_ylim(n + 0.05, 0)
    return fig


def run(from_data=False):
    cm.apply()
    if from_data:
        cm.require(DATA_FILE.exists(), f"{DATA_FILE} missing; run without --from-data first")
        df = pd.read_csv(DATA_FILE)
    else:
        out, meta = compute()
        for scope, d in out.items():
            check(d)
            d.to_csv(_data_file(scope), index=False, float_format="%.10g")
        df = out[SCOPE]
        (cm.DATA / f"{NAME}_meta.json").write_text(json.dumps(meta, indent=2))
        print(f"  [data]  {_data_file('route').name} (figure, route roads: {meta['route_nodes']:,} nodes) and "
              f"{_data_file('all').name} (all nodes: {meta['test_nodes']:,}); test route MAPE "
              f"{meta['test_route_mape']:.4f}, saved {meta['saved_route_mape']:.4f}")
    J, C, share = check(df)
    cm.save(draw(J, C, share), NAME)


if __name__ == "__main__":
    run(from_data="--from-data" in sys.argv)

