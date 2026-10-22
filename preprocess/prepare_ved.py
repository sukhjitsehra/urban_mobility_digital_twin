import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

COLS = ["DayNum", "VehId", "Trip", "Timestamp(ms)", "Latitude[deg]", "Longitude[deg]"]
EPOCH0 = pd.Timestamp("2017-11-01", tz="UTC").value // 10**9
MIN_POINTS, MIN_DURATION_S = 5, 60


def main(src):
    files = sorted(glob.glob(str(Path(src) / "VED_*_week.csv")))
    assert len(files) == 54, f"expected 54 weekly files, found {len(files)}"
    d = pd.concat([pd.read_csv(f, usecols=COLS) for f in files], ignore_index=True)
    n_rows, n_trips = len(d), d[["VehId", "Trip"]].drop_duplicates().shape[0]
    d = d.sort_values(["VehId", "Trip", "Timestamp(ms)"], kind="mergesort")
    key = ["VehId", "Trip"]
    assert (d.groupby(key).DayNum.nunique() == 1).all(), "DayNum is not constant within a trip"
    same_trip = (d.VehId.diff() == 0) & (d.Trip.diff() == 0)
    repeat = same_trip & (d["Latitude[deg]"].diff() == 0) & (d["Longitude[deg]"].diff() == 0)
    d = d[~repeat].copy()
    d["point_ts"] = (EPOCH0 + (d.DayNum - 1.0) * 86400.0 + d["Timestamp(ms)"] / 1000.0).round().astype("int64")
    same_trip = (d.VehId.diff() == 0) & (d.Trip.diff() == 0)
    d = d[~(same_trip & (d.point_ts.diff() <= 0))]
    g = d.groupby(key, sort=False)
    trips = pd.DataFrame({
        "poly": g.apply(lambda x: list(zip(x["Longitude[deg]"].round(6), x["Latitude[deg]"].round(6))), include_groups=False),
        "point_ts": g.point_ts.apply(list),
    }).reset_index()
    trips["n_points"] = trips.point_ts.apply(len)
    trips["TIMESTAMP"] = trips.point_ts.apply(lambda p: int(p[0]))
    trips["trip_travel_time_s"] = trips.point_ts.apply(lambda p: float(p[-1] - p[0]))
    trips = trips[(trips.n_points >= MIN_POINTS) & (trips.trip_travel_time_s >= MIN_DURATION_S)].copy()
    trips["TRIP_ID"] = (trips.VehId.astype("int64") * 1_000_000 + trips.Trip.astype("int64"))
    assert trips.TRIP_ID.is_unique
    assert all(np.all(np.diff(p) > 0) for p in trips.point_ts), "timestamps not strictly increasing"
    out = Path(__file__).resolve().parents[1] / "data" / "annarbor" / "trips_annarbor.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    trips.to_parquet(out, index=False)
    start = pd.to_datetime(trips.TIMESTAMP, unit="s")
    print(f"[ved] rows {n_rows:,} -> {len(d):,} after dropping repeats; trips {n_trips:,} -> {len(trips):,} kept")
    print(f"[ved] points per trip median {trips.n_points.median():.0f}, duration median {trips.trip_travel_time_s.median() / 60:.1f} min")
    print(f"[ved] first start {start.min()} UTC, last start {start.max()} UTC; vehicles {trips.VehId.nunique()}")
    print(f"[ved] saved {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else r"E:\Projects\NonTaxi_raw\VED\extracted")

