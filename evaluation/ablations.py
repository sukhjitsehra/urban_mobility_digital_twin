import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main
import LinkETA as li

REL_NAMES = list(li.REL_NAMES_DEFAULT)
DOWN = REL_NAMES.index("downstream")

_ORIG = {
    "TripDatasetLinkETA": li.TripDatasetLinkETA,
    "LinkETA": li.LinkETA,
    "temporal_split_by_trip": li.temporal_split_by_trip,
}


def _restore():
    for name, obj in _ORIG.items():
        setattr(li, name, obj)


def _make_dataset_cls(drop=(), k_override=None):
    drop_idx = {REL_NAMES.index(n) for n in drop}
    k_override = dict(k_override or {})
    Base = _ORIG["TripDatasetLinkETA"]

    class AblationTripDataset(Base):
        def __init__(self, df_all, edges_gdf, inst_table, rel_edges_list, *args, **kwargs):
            self._hop_full = li._build_src2dst(rel_edges_list[DOWN])
            rel = [torch.empty((2, 0), dtype=torch.long) if i in drop_idx else ei
                   for i, ei in enumerate(rel_edges_list)]
            kwargs.update(k_override)
            kwargs["cache_fp"] = None
            super().__init__(df_all, edges_gdf, inst_table, rel, *args, **kwargs)

        def _build_groups(self):
            self.hop_src2dst_full = self._hop_full
            super()._build_groups()

    return AblationTripDataset


def _zone_roads(zone):
    import geopandas as gpd
    e = gpd.read_file(main.city_input_files(zone["city"])[1])
    idc = li.guess_edges_id_col(e, li.CFG["way_id_col_edges_candidates"])
    c = e.to_crs(zone["crs"]).geometry.centroid
    inside = ((c.x // zone["size_m"]).astype(int) == zone["cell"][0]) & \
             ((c.y // zone["size_m"]).astype(int) == zone["cell"][1])
    return set(e.loc[inside.to_numpy(), idc].astype(str))


def _make_split(zone):
    base = _ORIG["temporal_split_by_trip"]

    def temporal_split_by_trip(df_all):
        parts = base(df_all)
        roads = _zone_roads(zone)
        through = set(df_all.loc[df_all[li.CFG["way_id_col_df"]].astype(str).isin(roads), "TRIP_ID"])
        rng = np.random.default_rng(42)
        out = []
        for name, part in zip(("train", "val"), parts[:2]):
            ids = np.sort(part["TRIP_ID"].unique())
            hit = np.isin(ids, list(through))
            drop = rng.choice(ids[~hit], size=int(hit.sum()), replace=False) if zone.get("control") else ids[hit]
            out.append(part[~part["TRIP_ID"].isin(drop)].copy())
            print(f"[ablation:zone] {name}: {len(ids):,} trips, {int(hit.sum()):,} with a link in the zone "
                  f"({len(roads):,} roads); removed {len(drop):,} "
                  f"{'random trips outside the zone (control)' if zone.get('control') else 'zone trips'}", flush=True)
        return out[0], out[1], parts[2]
    return temporal_split_by_trip


class LinkETANoPhysics(_ORIG["LinkETA"]):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hd = self.hidden_dim
        self.direct_eta_head = nn.Sequential(
            nn.Conv1d(hd, hd, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hd, 1, 1),
        )
        self.speed_head.register_forward_pre_hook(lambda module, args: setattr(self, "enc_t", args[0]))

    def forward(self, *args, **kwargs):
        _, _, sp_hat, logcnt_1h, g, _, _ = super().forward(*args, **kwargs)
        seq_mask = args[7] if len(args) > 7 else kwargs["seq_mask"]
        eta_direct = F.softplus(torch.nan_to_num(self.direct_eta_head(self.enc_t).squeeze(1))) + 1e-3
        link_eta_seq = eta_direct * seq_mask.float()
        return (link_eta_seq, link_eta_seq.sum(dim=1), sp_hat, logcnt_1h,
                torch.ones_like(g), torch.zeros_like(eta_direct), eta_direct)


def _zero_history_hook(module, args):
    x = args[0].clone()
    dyn = li.DYN_IDX.to(x.device)
    x[..., dyn] = 0.0
    return (x,) + tuple(args[1:])


def _variants():
    return {
        "no_rel_high_order": dict(desc="relation 'high_order' removed", drop=["high_order"]),
        "no_flow": dict(desc="upstream + downstream removed", drop=["upstream", "downstream"]),
        "no_turn": dict(desc="left/right turn, opposite, U-turn removed",
                        drop=["left_turn", "right_turn", "opposite", "u_turn"]),
        "no_graph": dict(desc="no graph context: all relations removed, no neighbours",
                         drop=list(REL_NAMES), k={"K_first": 0, "K_conflict": 0, "K_high": 0}),
        "no_g": dict(desc="route calibration off, g = 1", g_range=0.0),
        "no_delta": dict(desc="per-link residual off, delta = 0", delta_range=0.0),
        "no_history": dict(desc="dynamic traffic history inputs zeroed", history=False),
        "no_physics": dict(desc="direct neural ETA head instead of the physics baseline", physics=False),
        "zone_out": dict(desc="training and validation trips through the held-out zone removed",
                         zone=dict(city="daegu", crs="EPSG:5179", cell=(547, 881), size_m=2000.0)),
        "zone_out_control": dict(desc="as many random training and validation trips outside the zone removed",
                                 zone=dict(city="daegu", crs="EPSG:5179", cell=(547, 881), size_m=2000.0, control=True)),
    }


VARIANTS = _variants()
CACHE_DIR_OVERRIDE = None
RUN_TAG = ""


def _apply_before_build(spec):
    _restore()
    if spec.get("drop") or spec.get("k"):
        li.TripDatasetLinkETA = _make_dataset_cls(spec.get("drop", ()), spec.get("k"))
    if spec.get("physics") is False:
        li.LinkETA = LinkETANoPhysics
    if spec.get("zone"):
        li.temporal_split_by_trip = _make_split(spec["zone"])


def _apply_after_build(spec, core):
    model = core["model"]
    if "g_range" in spec:
        model.g_range = float(spec["g_range"])
    if "delta_range" in spec:
        model.delta_range = float(spec["delta_range"])
    if spec.get("history") is False:
        model.register_forward_pre_hook(_zero_history_hook)


_PROBE = {}


def _probe_hook(module, args):
    if "dyn_absmax" not in _PROBE:
        x = args[0]
        _PROBE["dyn_absmax"] = float(x[..., li.DYN_IDX.to(x.device)].abs().max())
    return None


def _check_variant(name, spec, core):
    ds = core["datasets"]["train"] if "datasets" in core else None
    if ds is not None and len(ds) > 0:
        first = ds.groups[: min(50, len(ds))]
        sums = defaultdict(int)
        for g in first:
            for r, m in enumerate(g["masks"]):
                sums[REL_NAMES[r]] += int(m.sum())
        n_nodes = sum(int(g["node_edge_idx"].numel()) for g in first)
        hop_route = sum(int(g["hop_idx"][g["route_flag"]].sum()) for g in first)
        print(f"[ablation:{name}] first 50 train trips: nodes={n_nodes} route_hop_sum={hop_route} "
              f"mask entries per relation={dict(sums)}")
    m = core["model"]
    print(f"[ablation:{name}] model={type(m).__name__} g_range={m.g_range} delta_range={m.delta_range} "
          f"pre_hooks={len(m._forward_pre_hooks)} direct_head={'direct_eta_head' in dict(m.named_children())}")
    _PROBE.clear()
    m.register_forward_pre_hook(_probe_hook)


def _append_csv(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


def _evaluate_and_save(city, name, core, folder, ckpt_dir):
    m, trip_df, link_df = main.eval_linketa(core, label=f"LinkETA_abl_{name}")
    trip_df["abs_err_s"] = (trip_df["y_pred"] - trip_df["y_true"]).abs()
    link_df["abs_err_s"] = (link_df["pred_tt_link"] - link_df["true_tt_link"]).abs()
    out = folder / "eval_outputs"
    out.mkdir(parents=True, exist_ok=True)
    trip_df.to_csv(out / f"eval_trips_abl_{name}{RUN_TAG}.csv", index=False)
    link_df.to_csv(out / f"eval_links_abl_{name}{RUN_TAG}.csv", index=False)
    row = dict(city=city, variant=name,
               route_mape=m.get("route_mape"), route_mae=m.get("route_mae"), route_rmse=m.get("route_rmse"),
               link_mape=m.get("link_mape"), link_mae=m.get("link_mae"), link_rmse=m.get("link_rmse"),
               vol_mae=m.get("vol_mae"),
               n_routes=len(trip_df), n_links=len(link_df), ckpt_dir=str(ckpt_dir),
               evaluated=time.strftime("%Y-%m-%d %H:%M:%S"))
    _append_csv(out / f"ablations_eval{RUN_TAG}.csv", row)
    print(f"[ablation:{name}] eval: route MAPE {row['route_mape']:.4f} | link MAPE {row['link_mape']:.4f} | "
          f"files eval_*_abl_{name}{RUN_TAG}.csv")
    return row


def run_variant(city, name, edges_gdf, df_all, epochs=None, max_trips=None, eval_only=False):
    spec = VARIANTS[name]
    if spec.get("zone") and main.city_folder_name(city) != spec["zone"]["city"]:
        raise SystemExit(f"{name}: the zone is defined for {spec['zone']['city']}, not {city}")
    print(f"\n=== LinkETA ablation '{name}' on {city}: {spec['desc']} ===")
    folder = main.results_dir(city)
    data_folder = main.city_dir(city)
    ckpt_dir = folder / main.ckpt_dirname("LinkETA", city, suffix=f"_abl_{name}{RUN_TAG}")
    if eval_only:
        _apply_before_build(spec)
        try:
            li_mod = main._setup_linketa_new(city)
            li_mod.CKPT_DIRNAME = str(ckpt_dir)
            core = li_mod.linketa_get_core_components(edges_gdf, df_all)
            _apply_after_build(spec, core)
            pkg = torch.load(ckpt_dir / "best.ckpt", map_location=core["device"], weights_only=False)
            core["model"].load_state_dict(pkg["model"], strict=True)
            print(f"[ablation:{name}] loaded {ckpt_dir / 'best.ckpt'} (epoch {pkg.get('epoch')})")
            return _evaluate_and_save(city, name, core, folder, ckpt_dir)
        finally:
            _restore()
    main._preserve_previous(ckpt_dir, f"LinkETA_abl_{name}")
    _apply_before_build(spec)
    try:
        li_mod = main._setup_linketa_new(city)
        li_mod.CKPT_DIRNAME = str(ckpt_dir)
        if CACHE_DIR_OVERRIDE and li_mod.CACHE_DIR:
            li_mod.CACHE_DIR = str(data_folder / CACHE_DIR_OVERRIDE)
        core = li_mod.linketa_get_core_components(edges_gdf, df_all)
        if epochs is not None:
            core["num_epochs"] = epochs
        _apply_after_build(spec, core)
        _check_variant(name, spec, core)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        with main.working_dir(folder):
            res = li_mod.run_training(core)
        train_s = time.perf_counter() - t0
        peak_mb = torch.cuda.max_memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else 0.0
        _evaluate_and_save(city, name, core, folder, ckpt_dir)
    finally:
        _restore()
    te = res["test"]
    row = dict(city=city, variant=name, description=spec["desc"],
               link_mape=te.get("link_mape"), link_mae=te.get("link_mae"), link_rmse=te.get("link_rmse"),
               route_mape=te.get("route_mape"), route_mae=te.get("route_mae"), route_rmse=te.get("route_rmse"),
               speed_mae_actual=te.get("speed_mae_actual"), vol_mae=te.get("vol_mae"),
               epochs_run=len(res["history"]["train"].get("route_loss", [])),
               train_time_s=round(train_s, 1), peak_gpu_mb=round(peak_mb, 1),
               max_trips=max_trips or "", epochs_cap=epochs or core["num_epochs"],
               r_max=li_mod.R_MAX, ckpt_dir=str(ckpt_dir),
               finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    _append_csv(folder / "eval_outputs" / f"ablations{RUN_TAG}.csv", row)
    with open(ckpt_dir / "ablation_spec.json", "w", encoding="utf-8") as f:
        json.dump(dict(variant=name, spec=spec, row=row), f, indent=2, default=str)
    print(f"[ablation:{name}] first-batch max |dynamic input| seen by the model = {_PROBE.get('dyn_absmax')}")
    print(f"[ablation:{name}] test route MAPE {row['route_mape']:.4f} | link MAPE {row['link_mape']:.4f} "
          f"| {row['train_time_s']} s | peak {row['peak_gpu_mb']} MB")
    return row


def parse_args():
    p = argparse.ArgumentParser(description="LinkETA ablations (LinkETA.py and main.py unchanged).")
    p.add_argument("--datasets", default="porto", help="Cities, as in main.py (e.g. porto or all).")
    p.add_argument("--variants", default=None, help="Comma-separated variant names, or 'all' (the component ablations).")
    p.add_argument("--list", action="store_true", help="List the variants and exit.")
    p.add_argument("--max-trips", type=int, default=None, help="Prototype: sample N trips (seed 42), as in main.py.")
    p.add_argument("--epochs", type=int, default=None, help="Prototype: cap the number of epochs.")
    p.add_argument("--no-build-cache", action="store_true",
                   help="Do not read/write the LinkETA build cache (as main.py --no-build-cache).")
    p.add_argument("--tag", default="", help="Suffix for checkpoint folders and the results CSV, e.g. _t500.")
    p.add_argument("--eval-only", action="store_true",
                   help="Do not train: evaluate each variant's saved best.ckpt (test predictions).")
    p.add_argument("--cache-dir", default=None,
                   help="Use data/<city>/<name>/ as build cache instead of cache_LinkETA (tests).")
    return p.parse_args()


def main_cli():
    args = parse_args()
    if args.list or not args.variants:
        for k, s in VARIANTS.items():
            print(f"{k:22s} {s['desc']}")
        return
    names = [n for n, s in VARIANTS.items() if not s.get("zone")] if args.variants.strip().lower() == "all" else \
        [n.strip() for n in args.variants.split(",") if n.strip()]
    bad = [n for n in names if n not in VARIANTS]
    if bad:
        raise SystemExit(f"Unknown variant(s): {bad}. Use --list.")
    main.NO_BUILD_CACHE = bool(args.no_build_cache)
    global CACHE_DIR_OVERRIDE, RUN_TAG
    CACHE_DIR_OVERRIDE = args.cache_dir
    RUN_TAG = args.tag or ""
    for city in main._resolve_datasets(args.datasets):
        edges_gdf, df_all = main.load_dataset(city, max_trips=args.max_trips)
        for name in names:
            run_variant(city, name, edges_gdf, df_all, epochs=args.epochs, max_trips=args.max_trips,
                        eval_only=args.eval_only)


if __name__ == "__main__":
    main_cli()

