"""
make_traffic_profile.py
=======================================================================
Generate the per-link historical traffic profile (`link_traffic_profile.parquet`)
that MetaTraffic consumes as ground truth, for one or more cities.

This reproduces the DuETA `Traffic_Precompute` procedure that produced the
EXISTING profiles for Porto, San Francisco and Chengdu, so the output is
byte-compatible with them:

    columns : way_id, day_of_week, hour_of_day,
              median_speed_0..11, mean_speed_0..11, min_speed_0..11,
              max_speed_0..11, count_vehicle_0..11        (3 + 60 = 63 cols)
    dtypes  : way_id=int32, day_of_week=int8, hour_of_day=int8, traffic=float32
    rows    : full grid  n_unique_ways x 7 (dow) x 24 (hour), hierarchically filled

Per-city I/O follows the Porto/SF layout under meta_learning_data/<city>/:
    input  : df_all_<city>.parquet  ->  df_all.parquet  ->  df_all_<city>.csv  (first found)
    output : meta_learning_data/<city>/link_traffic_profile.parquet

Usage:
    python make_traffic_profile.py                       # default: daegu,shanghai
    python make_traffic_profile.py --cities daegu
    python make_traffic_profile.py --cities shanghai,daegu
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Paths / constants
# -----------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(HERE, "meta_learning_data")

STATS = ["median_speed", "mean_speed", "min_speed", "max_speed", "count_vehicle"]
K = 12  # ten-minute slots per look-back window (0..11); matches existing profiles
TRAFFIC_COLS = [f"{s}_{k}" for s in STATS for k in range(K)]
PROFILE_COLS = ["way_id", "day_of_week", "hour_of_day"] + TRAFFIC_COLS

OUT_NAME = "link_traffic_profile.parquet"


def _resolve_input(city: str) -> str:
    """Pick the df_all source for a city, following Porto/SF conventions."""
    cdir = os.path.join(DATA_ROOT, city)
    candidates = [
        os.path.join(cdir, f"df_all_{city}.parquet"),
        os.path.join(cdir, "df_all.parquet"),
        os.path.join(cdir, f"df_all_{city}.csv"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(
        f"No df_all source found for '{city}'. Looked for:\n  " + "\n  ".join(candidates))


def _load_df_all(path: str) -> pd.DataFrame:
    needed = ["way_id", "day_of_week", "hour_of_day"] + TRAFFIC_COLS
    if path.endswith(".parquet"):
        # read only the columns we need (lighter)
        df = pd.read_parquet(path, columns=needed)
    else:
        df = pd.read_csv(path, usecols=needed, low_memory=False)
    return df


# -----------------------------------------------------------------------------
# DuETA k-means (verbatim algorithm used for Porto/SF/Chengdu)
# -----------------------------------------------------------------------------
def _kmeans(X, k, n_iter=80, seed=42):
    rng = np.random.default_rng(seed)
    centers = X[rng.choice(len(X), size=k, replace=False)]
    lab = np.zeros(len(X), dtype=int)
    for _ in range(n_iter):
        d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        lab = d2.argmin(axis=1)
        new_centers = np.vstack([
            X[lab == j].mean(axis=0) if np.any(lab == j) else centers[j]
            for j in range(k)
        ])
        if np.allclose(new_centers, centers, atol=1e-6):
            break
        centers = new_centers
    inertia = ((X - centers[lab]) ** 2).sum()
    return lab, centers, inertia


def build_link_traffic_profile(df_all: pd.DataFrame) -> pd.DataFrame:
    """Return the hierarchically-filled per-link traffic profile (63 cols)."""
    df = df_all.copy()
    df["way_id"] = df["way_id"].astype(np.int64)            # int64 working dtype (no overflow)
    df["day_of_week"] = df["day_of_week"].astype(int)
    df["hour_of_day"] = df["hour_of_day"].astype(int)

    missing = [c for c in TRAFFIC_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected traffic columns (examples): {missing[:10]}")

    # ---- A) regimes (k=7) over (day_of_week, hour_of_day) ----------------
    pattern = (
        df.groupby(["day_of_week", "hour_of_day"], as_index=False)
          .agg(s0=("mean_speed_0", "mean"),
               c0=("count_vehicle_0", "mean"),
               n=("way_id", "size"))
    )
    dow = pattern["day_of_week"].to_numpy()
    hr = pattern["hour_of_day"].to_numpy()
    s0 = pattern["s0"].to_numpy()
    X = np.column_stack([
        s0,
        np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24),
        np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
    ])
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)

    labels, _, _ = _kmeans(X, 7, seed=42)
    pattern["regime_id"] = labels
    order = pattern.groupby("regime_id")["s0"].mean().sort_values().index.tolist()
    remap = {rid: i for i, rid in enumerate(order)}        # 0 = slowest
    pattern["regime"] = pattern["regime_id"].map(remap).astype(int)

    df = df.merge(pattern[["day_of_week", "hour_of_day", "regime"]],
                  on=["day_of_week", "hour_of_day"], how="left")

    # ---- B) fallback tables (means) -------------------------------------
    t_hourly = df.groupby(["way_id", "day_of_week", "hour_of_day"], as_index=False)[TRAFFIC_COLS].mean()
    t_wdr    = df.groupby(["way_id", "day_of_week", "regime"], as_index=False)[TRAFFIC_COLS].mean()
    t_wr     = df.groupby(["way_id", "regime"], as_index=False)[TRAFFIC_COLS].mean()
    t_w      = df.groupby(["way_id"], as_index=False)[TRAFFIC_COLS].mean()
    t_dh     = df.groupby(["day_of_week", "hour_of_day"], as_index=False)[TRAFFIC_COLS].mean()
    global_mean = df[TRAFFIC_COLS].mean()

    # ---- C) full grid + hierarchical fill -------------------------------
    way_ids = np.sort(df["way_id"].dropna().unique())
    grid = pd.MultiIndex.from_product(
        [way_ids, range(0, 7), range(0, 24)],
        names=["way_id", "day_of_week", "hour_of_day"],
    ).to_frame(index=False)
    grid = grid.merge(pattern[["day_of_week", "hour_of_day", "regime"]],
                      on=["day_of_week", "hour_of_day"], how="left")

    profile = grid.merge(t_hourly, on=["way_id", "day_of_week", "hour_of_day"], how="left")

    def fill_from(profile_df, table, on_keys):
        tmp = profile_df[on_keys].merge(table, on=on_keys, how="left", suffixes=("", "_fb"))
        for c in TRAFFIC_COLS:
            profile_df[c] = profile_df[c].fillna(tmp[c])
        return profile_df

    profile = fill_from(profile, t_wdr, ["way_id", "day_of_week", "regime"])
    profile = fill_from(profile, t_wr,  ["way_id", "regime"])
    profile = fill_from(profile, t_w,   ["way_id"])
    profile = fill_from(profile, t_dh,  ["day_of_week", "hour_of_day"])
    for c in TRAFFIC_COLS:
        profile[c] = profile[c].fillna(global_mean[c])
    profile[TRAFFIC_COLS] = profile[TRAFFIC_COLS].fillna(0.0)

    profile = profile.drop(columns=["regime"])

    # ---- compact dtypes (match existing Porto/SF/Chengdu profiles) ------
    profile[TRAFFIC_COLS] = profile[TRAFFIC_COLS].astype(np.float32)
    profile["way_id"] = profile["way_id"].astype(np.int32)
    profile["day_of_week"] = profile["day_of_week"].astype(np.int8)
    profile["hour_of_day"] = profile["hour_of_day"].astype(np.int8)

    return profile[PROFILE_COLS]


def process_city(city: str) -> None:
    print(f"\n{'='*64}\n  {city.upper()}\n{'='*64}")
    src = _resolve_input(city)
    print(f"  input : {os.path.relpath(src, HERE)}")
    df = _load_df_all(src)
    print(f"  loaded df_all: {df.shape}")

    # Drop invalid / unmatched links (e.g. Daegu has way_id == -1)
    bad = (df["way_id"] <= 0) | df["way_id"].isna()
    if bad.any():
        print(f"  WARNING: dropping {int(bad.sum()):,} rows with invalid way_id (<=0/NaN), "
              f"affecting {df.loc[bad, 'way_id'].nunique()} id value(s)")
        df = df[~bad]

    profile = build_link_traffic_profile(df)

    # coverage check (same metric as the DuETA notebook)
    nonzero_pct = (profile[TRAFFIC_COLS].to_numpy().sum(axis=1) > 0).mean() * 100
    print(f"  profile: {profile.shape}  | unique ways: {profile['way_id'].nunique():,} "
          f"| grid = ways x 7 x 24")
    print(f"  non-zero traffic rows after hierarchical fill: {nonzero_pct:.2f}%")

    # schema self-check against an existing reference profile, if present
    ref_path = os.path.join(DATA_ROOT, "chengdu", OUT_NAME)
    if os.path.exists(ref_path):
        ref_cols = list(pd.read_parquet(ref_path, columns=None).head(0).columns)
        ok = list(profile.columns) == ref_cols
        print(f"  schema matches chengdu reference (names+order): {ok}")
        if not ok:
            raise RuntimeError("Generated schema does not match reference profile!")

    out = os.path.join(DATA_ROOT, city, OUT_NAME)
    tmp = out + ".tmp"
    profile.to_parquet(tmp, index=False, compression="zstd")
    os.replace(tmp, out)
    print(f"  SAVED -> {os.path.relpath(out, HERE)}  "
          f"({os.path.getsize(out)/1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser(description="Generate link_traffic_profile.parquet per city.")
    ap.add_argument("--cities", default="daegu,shanghai",
                    help="comma-separated city keys (folders under meta_learning_data/)")
    args = ap.parse_args()
    cities = [c.strip() for c in args.cities.split(",") if c.strip()]
    print(f"Generating traffic profiles for: {cities}")
    for city in cities:
        process_city(city)
    print("\nDONE.")


if __name__ == "__main__":
    main()
