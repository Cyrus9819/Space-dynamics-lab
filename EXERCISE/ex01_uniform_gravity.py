"""
Exercise 1 — Vertical Motion Under Uniform Gravity
====================================================

A projectile is launched straight up from the ground with an initial speed
of 850 m/s and falls back under a *constant* gravitational acceleration of
9.81 m/s^2. This is the simplest possible exercise of the ODE solver in
``space_base.py``: the trajectory is an exact parabola, and the total
mechanical energy

    E = 1/2 m v^2 + m g z

must be conserved (up to numerical-integration error).

Equation of motion
-------------------
    d^2z/dt^2 = -g

Analytic time of flight
------------------------
From SUVAT (v = u - g t), the projectile returns to the ground when
``0 = u t - 1/2 g t^2``, i.e. at t = 0 (launch) and:

    t_flight = 2u / g
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from space_base import Probe  # noqa: E402
from utils import savefig  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

GRAVITY = 9.81      # m/s^2, uniform gravitational acceleration
V0 = 850.0           # m/s, initial vertical speed
N_STEPS = 2_000      # number of steps for the ODE solver


def projectile(_, posvel):
    """ODE driver: d(z)/dt = vz,  d(vz)/dt = -g."""
    z, vz = posvel
    return vz, -GRAVITY


def main():
    t_flight = 2 * V0 / GRAVITY
    print(f"Analytic time of flight : {t_flight:.4f} s")

    probe = Probe(projectile, t_flight, N_STEPS, x0=0.0, vx0=V0, event=0)
    t, posvel = probe.odesolve()
    z, vz = posvel[:, 0], posvel[:, 1]

    energy0 = 0.5 * probe.mass * vz[0] ** 2 + probe.mass * GRAVITY * z[0]
    energy1 = 0.5 * probe.mass * vz[-1] ** 2 + probe.mass * GRAVITY * z[-1]
    rel_error = abs((energy1 - energy0) / energy0)

    print(f"Maximum altitude reached : {max(z):.2f} m")
    print(f"Relative energy error    : {rel_error:.3e}  (should be ~machine precision)")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, z)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Height (m)")
    ax.set_title("Exercise 1 — Vertical motion under uniform gravity")
    ax.grid(alpha=0.3)
    out_path = OUTPUT_DIR / "ex01_uniform_gravity.png"
    savefig(fig, out_path)
    print(f"Plot saved to {out_path}")


if __name__ == "__main__":
    main()
