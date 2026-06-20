"""
Exercise 8a — Hohmann Transfer Orbit (Earth to Mars)
======================================================

A Hohmann transfer is the minimum-energy two-burn manoeuvre between two
circular, coplanar orbits: an elliptical transfer orbit tangent to the
departure orbit at its periapsis and tangent to the arrival orbit at its
apoapsis.

Given the perihelion (Earth's orbital radius) and aphelion (Mars' orbital
radius) of the transfer ellipse:

    a = (r_perihelion + r_aphelion) / 2
    e = (r_aphelion - r_perihelion) / (r_aphelion + r_perihelion)

The departure speed follows from the vis-viva equation:

    V = sqrt(G M_sun (2/r - 1/a))

and the time for the (one-way) transfer is half the ellipse's orbital
period, from Kepler's third law:

    P = sqrt(4 pi^2 a^3 / (G M_sun))
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from space_base import GravBody, Probe  # noqa: E402
from utils import au_to_m, m_to_au, plot_circle, savefig  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

G = 6.674e-11
sun = GravBody.sun()


def probe_equations(_, posvel):
    x, y, vx, vy = posvel
    r = np.sqrt(x ** 2 + y ** 2)
    f = -G * sun.mass / r ** 3
    return vx, vy, f * x, f * y


def main():
    r_per = au_to_m(1.0)      # Earth's orbital radius
    r_aph = au_to_m(1.524)     # Mars' orbital radius

    a = (r_per + r_aph) / 2
    e = (r_aph - r_per) / (r_aph + r_per)
    print(f"Transfer semi-major axis : {m_to_au(a):.4f} AU")
    print(f"Transfer eccentricity    : {e:.4f}")

    v_per = np.sqrt(G * sun.mass * (2 / r_per - 1 / a))
    print(f"Departure speed          : {v_per / 1e3:.3f} km/s")

    period = np.sqrt(4 * np.pi ** 2 * a ** 3 / (G * sun.mass)) / 2
    print(f"One-way transfer time    : {period / (24 * 3600):.2f} days")

    # --- Nominal transfer: exactly meets Mars' orbit ----------------------
    probe = Probe(probe_equations, period, period / 3600,
                   x0=r_per, vx0=0.0, y0=0.0, vy0=v_per,
                   event=r_aph, eventflip=True)
    t, posvel = probe.odesolve()

    r = m_to_au(np.linalg.norm(posvel[:, :2], axis=1))
    print(f"Achieved min/max distance: {min(r):.4f} AU / {max(r):.4f} AU "
          f"(targets: 1.0 / 1.524 AU)")

    fig, ax = plt.subplots(figsize=(8, 8))
    plot_circle(ax, 1.0, color="red", label="Earth's orbit")
    plot_circle(ax, 1.524, color="red", label="Mars' orbit")
    ax.plot(m_to_au(posvel[:, 0]), m_to_au(posvel[:, 1]), color="blue", label="Transfer orbit")
    ax.set_xlabel("x (AU)")
    ax.set_ylabel("y (AU)")
    ax.set_title("Exercise 8a — Hohmann transfer orbit, Earth to Mars")
    ax.set_aspect("equal")
    ax.legend()
    savefig(fig, OUTPUT_DIR / "ex08a_hohmann_transfer.png")

    # --- Under-burn: 99.95% of the required departure speed -----------------
    probe_short = Probe(probe_equations, period, period / 3600,
                         x0=r_per, vx0=0.0, y0=0.0, vy0=v_per * 0.9995,
                         event=r_per)
    _, posvel_short = probe_short.odesolve()
    r_short = np.linalg.norm(posvel_short[:, :2], axis=1) / au_to_m(1.0)
    miss_distance_km = au_to_m(1.524 - max(r_short)) / 1e3
    print(f"\nWith only 99.95% of the required burn, the probe falls short of\n"
          f"Mars' orbit by {miss_distance_km:.0f} km — a small velocity error\n"
          f"becomes a large positional error over an interplanetary distance.")

    print(f"\nPlot saved to {OUTPUT_DIR / 'ex08a_hohmann_transfer.png'}")


if __name__ == "__main__":
    main()
