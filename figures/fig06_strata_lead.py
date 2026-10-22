import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as cm

NAME = "fig06_strata_lead"
DATA_FILE = cm.DATA / f"{NAME}.csv"
CITIES = {"porto": "Porto", "sf": "San Francisco", "chengdu": "Chengdu", "daegu": "Daegu",
          "annarbor": "Ann Arbor", "sfbus": "SF bus"}
MODELS = {"LinkETA": "linketa", "ConSTGAT": "constgat", "DuETA": "dueta", "STHR": "sthr",
          "TransETA": "transeta", "DeeprETA": "deepreta", "GSPINN": "gspinn"}
STRATA = {
    "length": dict(edges=[0, 3, 6, 9, 12, 15, 18, np.inf], labels=["0-3", "3-6", "6-9", "9-12", "12-15", "15-18", "18+"],
                   unit="km", title="Route length"),
    "duration": dict(edges=[0, 10, 20, 30, 40, 50, 60, np.inf],
                     labels=["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60+"], unit="min",
                     title="Trip duration"),
}
METRICS = {"mae": "MAE: seconds saved per trip", "mape": "MAPE: percentage points saved"}
MIN_TRIPS = 30
N_BOOT, SEED = 2000, 42
STYLE = {"porto": ("#4477AA", "o"), "sf": ("#EE6677", "s"), "chengdu": ("#228833", "^"),
         "daegu": ("#CCBB44", "D"), "annarbor": ("#66CCEE", "p"), "sfbus": ("#AA3377", "v")}
LOSS = "#BB5566"


def _city_trips(city):
    out = cm.ROOT / "results" / city / "eval_outputs"
    T = None
    for m, tag in MODELS.items():
        x = pd.read_csv(out / f"eval_trips_{tag}.csv", usecols=["TRIP_ID", "y_true", "y_pred", "route_len_m"])
        x["TRIP_ID"] = x.TRIP_ID.astype("int64")
        x = x.set_index("TRIP_ID").sort_index()
        saved = json.loads((out / f"eval_metrics_{tag}.json").read_text())[m]
        e = (x.y_pred - x.y_true).abs()
        pos = x.y_true > 0
        mae, mape = float(e.mean()), float((e[pos] / x.y_true[pos]).mean())
        cm.require(abs(mae - saved["route_mae"]) <= 1e-4 * max(1, abs(saved["route_mae"]))
                   and abs(mape - saved["route_mape"]) <= 1e-4 * max(1, abs(saved["route_mape"])),
                   f"{city} {m}: recomputed MAE/MAPE {mae:.4f}/{mape:.5f} != saved "
                   f"{saved['route_mae']:.4f}/{saved['route_mape']:.5f}")
        if T is None:
            T = pd.DataFrame({"y_true": x.y_true, "len_km": x.route_len_m / 1000})
        else:
            cm.require(x.index.equals(T.index), f"{city} {m}: test trips differ from LinkETA's")
            cm.require(np.allclose(x.y_true, T.y_true, atol=1e-2), f"{city} {m}: true trip times differ from LinkETA's")
        T[m] = x.y_pred
    d = pd.read_csv(cm.ROOT / "data" / city / f"df_all_{city}.csv", usecols=["TRIP_ID", "departure_time"])
    d = d.drop_duplicates("TRIP_ID").set_index(d.drop_duplicates("TRIP_ID").TRIP_ID.astype("int64"))
    dep = pd.to_datetime(d.departure_time.reindex(T.index), utc=True, format="mixed")
    cm.require(dep.notna().all(), f"{city}: {int(dep.isna().sum())} test trips without a departure time")
    T["block"] = dep.dt.strftime("%Y-%m-%d %H").to_numpy()
    T["dur_min"] = T.y_true / 60
    return T


def _metric(T, m, metric, idx=None):
    e = (T[m] - T.y_true).abs().to_numpy()
    if metric == "mae":
        return e
    y = T.y_true.to_numpy()
    return np.where(y > 0, e / np.where(y > 0, y, 1.0) * 100, np.nan)


def compute():
    rows = []
    for city, cname in CITIES.items():
        print(f"[{NAME}] {cname}: reading predictions and departure times ...")
        T = _city_trips(city)
        for sname, s in STRATA.items():
            col = "len_km" if sname == "length" else "dur_min"
            T["bin"] = pd.cut(T[col], s["edges"], labels=s["labels"], right=False)
            cm.require(int(T.bin.notna().sum()) == len(T), f"{city} {sname}: {int(T.bin.isna().sum())} trips outside the bins")
            rng = np.random.default_rng(SEED)
            for b in s["labels"]:
                g = T[T.bin == b]
                for metric in METRICS:
                    vals = {m: _metric(g, m, metric) for m in MODELS}
                    ok = ~np.isnan(vals["LinkETA"])
                    mean = {m: float(np.nanmean(v)) if ok.any() else np.nan for m, v in vals.items()}
                    rival = min((m for m in MODELS if m != "LinkETA"), key=lambda m: mean[m]) if ok.any() else ""
                    row = dict(city=city, stratum=sname, bin=b, metric=metric, n_trips=len(g),
                               linketa=mean["LinkETA"], rival=rival, rival_value=mean.get(rival, np.nan),
                               lead=mean.get(rival, np.nan) - mean["LinkETA"],
                               **{f"value_{m}": mean[m] for m in MODELS})
                    if len(g) >= MIN_TRIPS:
                        blocks = pd.Series(g.block.to_numpy()[ok])
                        uniq = blocks.unique()
                        bi = blocks.map({u: i for i, u in enumerate(uniq)}).to_numpy()
                        d = vals[rival][ok] - vals["LinkETA"][ok]
                        S = np.bincount(bi, weights=d, minlength=len(uniq)), np.bincount(bi, minlength=len(uniq))
                        draws = rng.integers(0, len(uniq), (N_BOOT, len(uniq)))
                        W = np.zeros((N_BOOT, len(uniq)))
                        np.add.at(W, (np.repeat(np.arange(N_BOOT), len(uniq)), draws.ravel()), 1.0)
                        boot = (W @ S[0]) / np.maximum(W @ S[1], 1)
                        lo, hi = np.percentile(boot, [2.5, 97.5])
                        row.update(ci_lo=float(lo), ci_hi=float(hi), n_blocks=len(uniq),
                                   significant=bool(lo > 0 or hi < 0), drawn=True)
                    else:
                        row.update(ci_lo=np.nan, ci_hi=np.nan, n_blocks=0, significant=False, drawn=False)
                    rows.append(row)
            cnt = T.bin.value_counts()
            cm.require(int(cnt.sum()) == len(T), f"{city} {sname}: bin counts {int(cnt.sum())} != test trips {len(T)}")
    D = pd.DataFrame(rows)
    D.to_csv(DATA_FILE, index=False, float_format="%.10g")
    print(f"[{NAME}] data written to {DATA_FILE}")
    return D


def _recheck(D):
    for _, r in D.iterrows():
        if not r.linketa == r.linketa:
            continue
        base = {m: r[f"value_{m}"] for m in MODELS if m != "LinkETA"}
        cm.require(r.rival == min(base, key=base.get), f"{r.city} {r.stratum} {r.bin} {r.metric}: rival is not the best baseline")
        cm.require(abs(r.lead - (base[r.rival] - r.linketa)) < 1e-6 * max(1.0, abs(r.linketa)),
                   f"{r.city} {r.stratum} {r.bin} {r.metric}: lead mismatch")
        cm.require(r.drawn == (r.n_trips >= MIN_TRIPS), f"{r.city} {r.stratum} {r.bin}: drawn flag")


MODEL_COLOR = {"LinkETA": "#004488", "ConSTGAT": "#EE6677", "DuETA": "#228833", "STHR": "#CCBB44",
               "TransETA": "#66CCEE", "DeeprETA": "#AA3377", "GSPINN": "#BBBBBB"}


def _pale(hexcol, f=0.55):
    c = np.array([int(hexcol[i:i + 2], 16) for i in (1, 3, 5)]) / 255
    return tuple(c + (1 - c) * f)


def _ink_on(rgb):
    r, g, b = rgb[:3]
    return "white" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.45 else cm.INK


def draw(D):
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    from matplotlib.patches import Patch
    cm.apply()
    BAR = 0.62
    for sname, s in STRATA.items():
        fig, axes = plt.subplots(2, 1, figsize=(468 / 72.27 * 25.4 * cm.MM, 112 * cm.MM))
        fig.subplots_adjust(left=0.13, right=0.875, top=0.945, bottom=0.16, hspace=0.36)
        texts = []
        for ax, (metric, mtitle), tag in zip(axes, (("mae", "MAE: LinkETA's lead over the best other model, s per trip"),
                                                     ("mape", "MAPE: LinkETA's lead over the best other model, "
                                                              "percentage points")), "ab"):
            for k, city in enumerate(CITIES):
                g = D[(D.city == city) & (D.stratum == sname) & (D.metric == metric)].set_index("bin").reindex(s["labels"])
                share = g.n_trips / g.n_trips.sum() * 100
                x0, best = 0.0, 0.0
                for b, r in g.iterrows():
                    w = share[b]
                    if r.drawn:
                        win = "LinkETA" if r.lead > 0 else r.rival
                        col = to_rgb(MODEL_COLOR[win])
                        ax.barh(-k, w, left=x0, height=BAR, color=col, edgecolor="white", lw=0.6)
                        best += w if win == "LinkETA" else 0.0
                        val = f"{r.lead:+.0f}" if metric == "mae" else f"{r.lead:+.1f}"
                        texts.append((ax, x0, w, -k, f"{b} {s['unit']}  {val}", val, _ink_on(col)))
                    x0 += w
                ax.text(101.5, -k, f"{best:.0f}%", va="center", ha="left", fontsize=7, fontweight="bold",
                        color=MODEL_COLOR["LinkETA"])
            ax.set_yticks(-np.arange(len(CITIES)))
            ax.set_yticklabels(list(CITIES.values()), fontsize=7)
            ax.tick_params(axis="y", length=0)
            ax.set_xlim(0, 100)
            ax.set_ylim(-len(CITIES) + 1 - BAR, BAR)
            ax.set_title(f"({tag}) {mtitle}", fontsize=7, loc="left")
            ax.text(101.5, BAR * 0.75, "LinkETA\nbest", fontsize=7, va="bottom", ha="left")
            for sp in ("top", "right", "left"):
                ax.spines[sp].set_visible(False)
        axes[-1].set_xlabel(f"Share of the city's test trips (%), {s['title'].lower()} bins from short (left) to long "
                            f"(right)", fontsize=7)
        fig.canvas.draw()
        rend = fig.canvas.get_renderer()
        for ax, x0, w, y, full, val, ink in texts:
            seg = ax.transData.transform((x0 + w, y))[0] - ax.transData.transform((x0, y))[0]
            for label in (full, val):
                t = ax.text(x0 + w / 2, y, label, ha="center", va="center", fontsize=7, color=ink)
                if t.get_window_extent(rend).width <= seg - 2:
                    break
                t.remove()
        hand = [Patch(facecolor=MODEL_COLOR[m], edgecolor="white", label=m) for m in MODELS]
        fig.legend(handles=hand, loc="lower center", ncol=len(MODELS), fontsize=7, frameon=False, handlelength=1.2,
                   columnspacing=0.9, handletextpad=0.35)
        cm.save(fig, f"{NAME}_{sname}")


def run(from_data=False):
    if from_data:
        D = pd.read_csv(DATA_FILE)
    else:
        D = compute()
    _recheck(D)
    s = D[D.drawn]
    print(f"[{NAME}] bins drawn: {len(s)}; LinkETA ahead in {int((s.lead > 0).sum())}, "
          f"significantly ahead in {int(((s.lead > 0) & s.significant).sum())}, "
          f"significantly behind in {int(((s.lead < 0) & s.significant).sum())}")
    draw(D)


if __name__ == "__main__":
    run(from_data="--from-data" in sys.argv)

