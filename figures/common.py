import os
import time
from pathlib import Path

import matplotlib as mpl

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PLOTS = HERE / "plots"
DATA = HERE / "data"
PAPER_FIGURES = ROOT / "paper" / "Mubarak_Obj1" / "Elsevier MLWA" / "figures"

MM = 1 / 25.4
SINGLE = 90 * MM
ONE_HALF = 140 * MM
FULL = 190 * MM

RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 8.0,
    "axes.labelsize": 8.0,
    "axes.titlesize": 8.0,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "legend.fontsize": 7.0,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.0,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "hatch.linewidth": 0.5,
    "savefig.dpi": 1000,
}
MIN_FONT_PT = 7.0

HC = {"blue": "#004488", "red": "#BB5566", "yellow": "#DDAA33"}
SEQ_BLUE = ["#FFFFFF", "#EAF0F7", "#C6D8EA", "#93B5D8", "#5A8DC0", "#2E6AA8", "#004488"]
SEQ_YLORBR = ["#FFFFE5", "#FFF7BC", "#FEE391", "#FEC44F", "#FB9A29", "#EC7014", "#CC4C02"]
INK = "#222222"
GREY = "#DDDDDD"


def apply():
    mpl.rcParams.update(RC)
    PLOTS.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)


def require(cond, msg):
    if not cond:
        raise AssertionError(f"[check failed] {msg}")


def check_fonts(fig):
    small = [(t.get_text(), t.get_fontsize()) for t in fig.findobj(mpl.text.Text)
             if t.get_text().strip() and t.get_fontsize() < MIN_FONT_PT - 1e-9]
    require(not small, f"text below {MIN_FONT_PT} pt: {small[:5]}")


def save(fig, name):
    import matplotlib.pyplot as plt
    check_fonts(fig)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 1000})):
        final = PLOTS / f"{name}.{ext}"
        tmp = PLOTS / f"_{name}.tmp.{ext}"
        fig.savefig(tmp, **kw)
        for _ in range(20):
            try:
                os.replace(tmp, final)
                break
            except OSError:
                time.sleep(0.5)
        else:
            raise OSError(f"could not replace {final}; is it open elsewhere?")
    plt.close(fig)
    print(f"  [saved] figures/plots/{name}.pdf and .png")
    import shutil
    for ext in ("pdf", "png"):
        shutil.copy2(PLOTS / f"{name}.{ext}", PAPER_FIGURES / f"{name}.{ext}")
    print(f"  [copied] paper/Mubarak_Obj1/Elsevier MLWA/figures/{name}.pdf and .png")

