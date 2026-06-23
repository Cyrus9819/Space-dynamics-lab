"""
Exercise 3 — Drag in a Uniform-Density Atmosphere
===================================================

We add aerodynamic drag to the uniform-gravity projectile of Exercise 1.
Drag opposes the velocity vector and grows with the square of speed:

    F_drag = -1/2 C_D rho A |V| V

where ``V`` is the velocity vector, ``rho`` is the (here constant)
atmospheric density, ``A`` is the cross-sectional area and ``C_D`` is the
drag coefficient.

With drag included, mechanical energy is no longer conserved — kinetic
energy is dissipated as heat, which is exactly why re-entry vehicles need
heat shields.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from space_base import Probe  # noqa: E402
from utils import savefig  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

GRAVITY = 9.81           # m/s^2
V0 = 850.0                # m/s, initial vertical speed
N_STEPS = 100_000

# Probe parameters (match the defaults baked into space_base.Probe).
DRAG_COEFFICIENT = 1.0    # dimensionless
AREA = 0.01                # m^2
MASS = 1.0                 # kg
AIR_DENSITY = 1.217         # kg/m^3, sea-level density (constant for this exercise)

T_FINAL = 200.0  # generous upper bound; the ground-impact event cuts this short


def projectile(_, posvel):
    z, vz = posvel
    drag_force = -0.5 * DRAG_COEFFICIENT * AREA * AIR_DENSITY * abs(vz) * vz
    return vz, -GRAVITY + drag_force / MASS


def main():
    probe = Probe(projectile, T_FINAL, N_STEPS, x0=0.0, vx0=V0, event=0)
    t, posvel = probe.odesolve()
    z, vz = posvel[:, 0], posvel[:, 1]
    t_end = len(t) - 2

    print(f"Maximum height        : {max(z):.3f} m")
    print(f"Time of flight        : {t[t_end]:.4f} s")
    print(f"Altitude at touchdown  : {z[t_end]:.4f} m (sanity check)")
    print(f"Terminal velocity      : {vz[t_end]:.3f} m/s (negative = falling)")

    energy0 = 0.5 * MASS * vz[0] ** 2 + MASS * GRAVITY * z[0]
    energy1 = 0.5 * MASS * vz[t_end] ** 2 + MASS * GRAVITY * z[t_end]
    error_pct = 100 * abs((energy1 - energy0) / energy0)
    print(f"Energy lost to drag    : {error_pct:.2f} %  (no longer conserved, as expected)")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(t, z)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Height (m)")
    ax1.set_title("Height vs time")
    ax1.grid(alpha=0.3)

    ax2.plot(t, vz, color="tab:orange")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Velocity (m/s)")
    ax2.set_title("Velocity vs time")
    ax2.axhline(vz[t_end], color="grey", linestyle="--", linewidth=1,
                label=f"terminal velocity ≈ {vz[t_end]:.1f} m/s")
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig.suptitle("Exercise 3 — Quadratic drag in a uniform atmosphere")

    out_path = OUTPUT_DIR / "ex03_drag_uniform_atmosphere.png"
    savefig(fig, out_path)
    print(f"Plot saved to {out_path}")


if __name__ == "__main__":
    main()
