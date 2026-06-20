"""
run_all
=======

Convenience script that runs every exercise in sequence and regenerates
every plot under ``exercises/outputs/``. Useful for confirming the whole
repository still works end to end (e.g. after changing ``space_base.py``).

Usage
-----
    python run_all.py            # run everything
    python run_all.py ex01 ex05  # run only exercises matching these prefixes

Note: Exercises 6/7, 11/12 and 13 numerically integrate over long, fine
time grids and may each take up to a couple of minutes to run.
"""
import importlib
import sys
import time
from pathlib import Path

EXERCISES = [
    "ex01_uniform_gravity",
    "ex02_realistic_gravity",
    "ex03_drag_uniform_atmosphere",
    "ex04_isothermal_atmosphere",
    "ex05_lunar_probe_haywire",
    "ex06_07_mars_aerobraking",
    "ex08a_hohmann_transfer_mars",
    "ex08b_asteroid_2013la2_transfer",
    "ex09_martian_escape_trajectory",
    "ex10_space_station_inertia_tensor",
    "ex11_12_asteroid_spin_stability",
    "ex13_kuiper_belt_interception",
    "ex14_fast_track_to_the_moon",
]


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    filters = sys.argv[1:]
    modules = [m for m in EXERCISES if not filters or any(m.startswith(f) for f in filters)]

    for i, module_name in enumerate(modules, start=1):
        print(f"\n{'=' * 70}\n[{i}/{len(modules)}] Running exercises.{module_name}\n{'=' * 70}")
        start = time.time()
        module = importlib.import_module(f"exercises.{module_name}")
        module.main()
        print(f"(finished in {time.time() - start:.1f} s)")


if __name__ == "__main__":
    main()
