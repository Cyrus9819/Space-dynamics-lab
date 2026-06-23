"""
Exercise 10 — The Moment of Inertia Tensor of a Space Station
=================================================================

A modular space station is modelled as six point masses at fixed
positions. We compute the station's total mass, centre of mass, and full
3x3 inertia tensor:

    I = [[ Ixx, Ixy, Ixz],
         [ Ixy, Iyy, Iyz],
         [ Ixz, Iyz, Izz]]

Diagonalising ``I`` (an eigenvalue problem, since ``I`` is symmetric) gives
the *principal moments of inertia* and the corresponding *principal axes*
— the orientation about which the station would spin without wobbling.

We then repeat the calculation after increasing the mass of the solar
arrays module, to see how a single module's mass affects the overall mass
distribution, centre of mass, and rotational inertia.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import center_of_mass, inertia_tensor  # noqa: E402

# Columns: [mass (t), x (m), y (m), z (m)]
MODULE_NAMES = [
    "Astrophysics module",
    "Robotics repair plant",
    "Power plant",
    "Docking module",
    "Communications module",
    "Solar arrays",
]
MODULES_ORIGINAL = np.array([
    [3, 6, -2, 0],
    [4, 3, 1, -2],
    [7, -4, 1, 2],
    [5, -4, -2, -4],
    [4, 3, 2, -1],
    [3, 2, -1, 6],
], dtype=float)

MODULES_UPDATED = MODULES_ORIGINAL.copy()
MODULES_UPDATED[5, 0] = 5  # solar arrays' mass increased from 3 t to 5 t


def report(modules, label):
    masses, positions = modules[:, 0], modules[:, 1:]
    total_mass, com = center_of_mass(masses, positions)
    inertia = inertia_tensor(masses, positions, about=com)
    moments, axes = np.linalg.eig(inertia)

    print(f"--- {label} ---")
    print(f"Total mass        : {total_mass:.1f} t")
    print(f"Centre of mass     : {np.round(com, 4)} m")
    print("Inertia tensor (t m^2):")
    print(np.round(inertia, 2))
    print(f"Principal moments  : {np.round(moments, 2)} t m^2")
    print()
    return total_mass, com, inertia, moments, axes


def main():
    _, _, inertia_orig, moments_orig, _ = report(MODULES_ORIGINAL, "Original configuration (solar arrays = 3 t)")
    _, _, inertia_new, moments_new, _ = report(MODULES_UPDATED, "Updated configuration (solar arrays = 5 t)")

    print("--- Effect of the heavier solar arrays ---")
    print(f"Change in principal moments: {np.round(moments_new - moments_orig, 2)} t m^2")
    print("\nIncreasing the solar arrays' mass raises every principal moment of\n"
          "inertia (the station becomes more resistant to changes in spin), and\n"
          "shifts the centre of mass away from the origin. Because the solar\n"
          "arrays sit close to the original centre of mass in the x-y plane, the\n"
          "z-axis moment is affected least, while the x- and y-dominated moments\n"
          "shift more.")


if __name__ == "__main__":
    main()
