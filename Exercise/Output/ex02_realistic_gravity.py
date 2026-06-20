"""
Exercise 2 — Realistic (Inverse-Square) Gravity
=================================================

We repeat Exercise 1 but replace the uniform gravitational field with a
realistic inverse-square field:

    g(z) = G M_earth / (R_earth + z)^2

and compare the resulting trajectory against the uniform-gravity case.
Because gravity weakens with altitude, the realistic trajectory should
reach a (slightly) higher apex and take (slightly) longer to fall back
than the uniform-gravity approximation.

The relevant specific (per-unit-mass) energy is now:

    E = 1/2 v^2 - G M_earth / (R_earth + z)
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from space_base import GravBody, Probe  # noqa: E402
from utils import savefig  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

G = 6.674e-11        # gravitational constant, m^3 kg^-1 s^-2
GRAVITY = 9.81       # m/s^2, uniform gravity for the baseline case
V0 = 850.0           # m/s, initial vertical speed
N_STEPS = 500_000    # number of steps for the ODE solver

earth = GravBody.earth()


def projectile_uniform_gravity(_, posvel):
    z, vz = posvel
    return vz, -GRAVITY


def projectile_realistic_gravity(_, posvel):
    z, vz = posvel
    g = G * earth.mass / (earth.radius + z) ** 2
    return vz, -g


def main():
    # --- Baseline: uniform gravity -------------------------------------
    t_final_uniform = 2 * V0 / GRAVITY
    probe_uniform = Probe(projectile_uniform_gravity, t_final_uniform, N_STEPS,
                           x0=0.0, vx0=V0, event=0)
    _, posvel_uniform = probe_uniform.odesolve()
    max_height_uniform = np.max(posvel_uniform[:, 0])

    # --- Realistic, inverse-square gravity ------------------------------
    # We don't know the flight time analytically anymore, so we give the
    # solver a generous upper bound and let the ground-impact event cut it
    # short automatically.
    t_final_guess = 200.0
    probe_real = Probe(projectile_realistic_gravity, t_final_guess, N_STEPS,
                        x0=0.0, vx0=V0, event=0)
    t, posvel = probe_real.odesolve()
    z, vz = posvel[:, 0], posvel[:, 1]

    max_height_real = np.max(z)
    t_end = len(t) - 2  # penultimate index: last point is (numerically) at z=0
    t_flight_real = t[t_end]

    print(f"Max height (uniform gravity)   : {max_height_uniform:10.3f} m")
    print(f"Max height (realistic gravity) : {max_height_real:10.3f} m")
    print(f"Difference                     : {max_height_real - max_height_uniform:10.3f} m")
    print(f"Time of flight (uniform)       : {t_final_uniform:10.4f} s")
    print(f"Time of flight (realistic)     : {t_flight_real:10.4f} s")
    print(f"Altitude at penultimate step   : {z[t_end]:.4f} m (sanity check, should be near 0)")

    # --- Energy conservation check --------------------------------------
    energy0 = 0.5 * vz[0] ** 2 - G * earth.mass / (earth.radius + z[0])
    energy1 = 0.5 * vz[t_end] ** 2 - G * earth.mass / (earth.radius + z[t_end])
    error_pct = 100 * abs((energy1 - energy0) / energy0)
    print(f"Energy conservation error      : {error_pct:.3e} %")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, z)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Height (m)")
    ax.set_title("Exercise 2 — Trajectory with realistic (inverse-square) gravity")
    ax.grid(alpha=0.3)
    out_path = OUTPUT_DIR / "ex02_realistic_gravity.png"
    savefig(fig, out_path)
    print(f"Plot saved to {out_path}")


if __name__ == "__main__":
    main()
