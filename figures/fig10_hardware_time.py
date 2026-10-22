import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as cm

NAME = "fig10_hardware_time"
DATA_FILE = cm.DATA / f"{NAME}.csv"
SRC = cm.ROOT / "results" / "chengdu" / "eval_outputs"
DEV = [("a10", "GPU cloud\n(A10)"), ("rtx4070", "Laptop GPU\n(RTX 4070)"), ("cpu", "CPU\n(Xeon)")]
MODELS = ["ConSTGAT", "DuETA", "STHR", "TransETA", "DeeprETA", "GSPINN", "LinkETA"]
TABLE12 = {"ConSTGAT": (0.51, 3.62), "DuETA": (1.98, 13.09), "STHR": (5.40, 6.37), "TransETA": (1.53, 3.73),
           "DeeprETA": (0.31, 0.99), "GSPINN": (0.44, 1.22), "LinkETA": (3.43, 4.58)}
WIDTH = 0.42 * 468 / 72.27


def build():
    rows = []
    for dev, _ in DEV:
        d = pd.read_csv(SRC / f"benchmark_inference_{dev}.csv")
        d = d[d.batch_size == 16] if "batch_size" in d else d
        for _, r in d.iterrows():
            cm.require(abs(r.route_mape - r.route_mape_saved) < 1e-3,
                       f"{dev} {r.model}: route MAPE {r.route_mape:.5f} != saved {r.route_mape_saved:.5f}")
            rows.append(dict(device=dev, model=r.model, loop_ms=float(r.loop_ms_per_trip),
                             model_ms=float(r.model_ms_per_trip)))
    df = pd.DataFrame(rows)
    lap = df[df.device == "rtx4070"].set_index("model")
    for m, (mod, loop) in TABLE12.items():
        cm.require(m in lap.index and round(lap.loc[m, "model_ms"], 2) == mod and round(lap.loc[m, "loop_ms"], 2) == loop,
                   f"laptop GPU {m}: {lap.loc[m].to_dict() if m in lap.index else 'missing'} != Table 12 {mod}/{loop}")
    df.to_csv(DATA_FILE, index=False)
    return df


def draw(df):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    from matplotlib.patches import Rectangle
    cm.apply()
    M = np.array([[float(df[(df.device == d) & (df.model == m)].loop_ms.iloc[0])
                   if ((df.device == d) & (df.model == m)).any() else np.nan for d, _ in DEV] for m in MODELS])
    fig, ax = plt.subplots(figsize=(WIDTH, 0.92 * WIDTH))
    fig.subplots_adjust(left=0.22, right=0.98, top=0.90, bottom=0.02)
    cmap = LinearSegmentedColormap.from_list("t", cm.SEQ_BLUE[1:])
    cmap.set_bad("#F2F2F2")
    ax.imshow(np.ma.masked_invalid(M), cmap=cmap, norm=LogNorm(0.3, 30), aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            s = "–" if np.isnan(v) else (f"{v:.1f}" if v >= 1 else f"{v:.2f}")
            ax.text(j, i, s, ha="center", va="center", fontsize=7,
                    color="white" if (not np.isnan(v) and v > 6) else cm.INK)
    i = MODELS.index("LinkETA")
    ax.add_patch(Rectangle((-0.5, i - 0.5), M.shape[1], 1, fill=False, ec=cm.INK, lw=1.2))
    ax.set_xticks(range(len(DEV)))
    ax.set_xticklabels(["Cloud GPU", "Laptop GPU", "CPU"], fontsize=7)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels(MODELS, fontsize=7)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cm.save(fig, NAME)


def run(from_data=False):
    cm.apply()
    df = pd.read_csv(DATA_FILE) if from_data else build()
    draw(df)


if __name__ == "__main__":
    run(from_data="--from-data" in sys.argv)

