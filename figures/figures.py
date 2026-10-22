import importlib
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

FIGURES = [
    "fig01_relation_distinctness",
    "fig02_relation_roles",
    "fig03_disruption",
    "fig04_sparse_data",
    "fig05_heldout_zone",
    "fig06_strata_lead",
    "fig07_trip_starts",
    "fig09_arrival_along_route",
    "fig10_hardware_time",
]
PER_CITY = "per_city_plots"


def main(argv):
    from_data = "--from-data" in argv
    selected = [a for a in argv if not a.startswith("--")]
    todo = [f for f in FIGURES if not selected or any(f.startswith(s) for s in selected)]
    per_city = not selected or any(PER_CITY.startswith(s) for s in selected)
    if not todo and not per_city:
        sys.exit(f"no figure matches {selected}; choose from {FIGURES + [PER_CITY]}")
    t0 = time.time()
    for name in todo:
        print(f"\n=== {name}")
        importlib.import_module(name).run(from_data=from_data)
    if per_city:
        print(f"\n=== {PER_CITY}")
        importlib.import_module(PER_CITY).main()
    print(f"\ndone: {len(todo) + int(per_city)} figure module(s) in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main(sys.argv[1:])

