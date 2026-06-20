"""
Exercise 4 — An Isothermal Atmosphere Model
=============================================

A constant air density is a poor approximation once altitude becomes
significant. Here we use the standard isothermal-atmosphere model, where
density falls off exponentially with altitude:

    rho(h) = rho_surface * exp(-h / H)

with ``rho_surface`` the sea-level density and ``H`` the atmospheric scale
height (the altitude over which density drops by a factor of e). Both
values are already stored on ``GravBody.earth()``.

Combined with realistic (inverse-square) gravity, this gives a much more
faithful re-entry/ascent model than Exercises 1-3.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from space_base import GravBody, Probe  # noqa: E402
from utils import savefig  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

G = 6.674e-11
V0 = 850.0
N_STEPS = 100_000
T_FINAL = 50.0

DRAG_COEFFICIENT = 1.0
AREA = 0.01
MASS = 1.0

earth = GravBody.earth()


def atmosphere_density(altitude):
    """Isothermal exponential atmosphere model: rho(h) = rho0 * exp(-h/H)."""
    return earth.surface_density * np.exp(-altitude / earth.scale_height)


def projectile(_, posvel):
    z, vz = posvel
    gravity = G * earth.mass / (earth.radius + z) ** 2
    drag_force = -0.5 * DRAG_COEFFICIENT * AREA * atmosphere_density(z) * abs(vz) * vz
    return vz, -gravity + drag_force / MASS


def plot_density_profile():
    h = np.linspace(0, 100_000, 5_000)
    rho = atmosphere_density(h)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(h, rho)
    ax.set_xlabel("Altitude (m)")
    ax.set_ylabel("Density (kg/m^3)")
    ax.set_title("Isothermal atmosphere density profile")
    ax.grid(alpha=0.3)
    savefig(fig, OUTPUT_DIR / "ex04_density_profile.png")


def main():
    plot_density_profile()

    probe = Probe(projectile, T_FINAL, N_STEPS, x0=0.0, vx0=V0, event=0)
    t, posvel = probe.odesolve()
    z = posvel[:, 0]
    t_end = len(t) - 2

    print(f"Maximum height        : {max(z):.4f} m")
    print(f"Time of flight        : {t[t_end]:.4f} s")
    print(f"Altitude at touchdown  : {z[t_end]:.6f} m (sanity check)")
    print("Compared with the constant-density model of Exercise 3, the probe "
          "reaches a slightly higher apex: above ~40 km the air is thin enough "
          "that drag becomes negligible, letting the probe coast higher before "
          "gravity pulls it back.")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, z)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Height (m)")
    ax.set_title("Exercise 4 — Trajectory through an isothermal atmosphere")
    ax.grid(alpha=0.3)
    out_path = OUTPUT_DIR / "ex04_isothermal_atmosphere.png"
    savefig(fig, out_path)
    print(f"Plots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
