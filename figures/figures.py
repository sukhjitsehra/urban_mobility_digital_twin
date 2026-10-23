import importlib
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

FIGURES = [
    "fig01_relation_distinctness",
    "fig05_heldout_zone",
    "fig06_strata_lead",
    "fig07_trip_starts",
    "fig09_arrival_along_route",
    "fig10_hardware_time",
]


def main(argv):
    from_data = "--from-data" in argv
    selected = [a for a in argv if not a.startswith("--")]
    todo = [f for f in FIGURES if not selected or any(f.startswith(s) for s in selected)]
    if not todo:
        sys.exit(f"no figure matches {selected}; choose from {FIGURES}")
    t0 = time.time()
    for name in todo:
        print(f"\n=== {name}")
        importlib.import_module(name).run(from_data=from_data)
    print(f"\ndone: {len(todo)} figure(s) in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main(sys.argv[1:])
