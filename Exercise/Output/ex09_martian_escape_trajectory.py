"""
Exercise 9 — The Martian: Escaping Mars Towards Earth
========================================================

A stranded astronaut needs to leave Mars from a 200 km circular parking
orbit and get on a trajectory that will eventually carry them back to
Earth via a Hohmann transfer. This requires three steps:

1. The circular parking-orbit speed at Mars:  ``V = sqrt(G M_mars / r)``.
2. The *heliocentric* speed needed at Mars' orbit to begin a Hohmann
   transfer back to Earth, converted into a speed *relative to Mars* by
   subtracting Mars' own orbital speed around the Sun.
3. The *departure burn* at periapsis of the escape hyperbola needed to
   reach that hyperbolic excess speed, found via the energy equation in
   the limit r -> infinity.

We then propagate the resulting hyperbolic escape trajectory out to 18
Mars radii to visualise it.
"""
import sys
from datetime import timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from space_base import GravBody, Probe  # noqa: E402
from utils import au_to_m, plot_circle, savefig  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

G = 6.674e-11
mars = GravBody.mars()
sun = GravBody.sun()


def probe_equations(_, posvel):
    x, y, vx, vy = posvel
    r = np.sqrt(x ** 2 + y ** 2)
    f = -G * mars.mass / r ** 3
    return vx, vy, f * x, f * y


def main():
    # --- 1. Circular parking orbit ----------------------------------------
    r = 200e3 + mars.radius
    v_init = np.sqrt(G * mars.mass / r)
    print(f"Parking-orbit speed (200 km altitude): {v_init / 1e3:.4f} km/s")

    # --- 2. Required hyperbolic excess speed, relative to Mars --------------
    r_earth, r_mars = au_to_m(1.0), au_to_m(1.524)
    a_transfer = (r_earth + r_mars) / 2
    v_hyp_abs = np.sqrt(G * sun.mass * (2 / r_mars - 1 / a_transfer))
    v_mars_orbit = np.sqrt(G * sun.mass / r_mars)
    v_hyp = v_hyp_abs - v_mars_orbit
    print(f"Heliocentric speed needed at Mars' orbit : {v_hyp_abs / 1e3:.4f} km/s")
    print(f"Mars' own orbital speed                  : {v_mars_orbit / 1e3:.4f} km/s")
    print(f"Hyperbolic excess speed (relative to Mars): {v_hyp / 1e3:.4f} km/s")

    # --- 3. Departure burn at periapsis of the escape hyperbola -------------
    v_per = np.sqrt(v_hyp ** 2 + 2 * G * mars.mass / r)
    print(f"Required periapsis (departure) speed     : {v_per / 1e3:.4f} km/s")

    a = G * mars.mass / v_hyp ** 2
    e = 1 + r / a
    beta_angle = np.arccos(1 / e)
    print(f"Escape-hyperbola semi-major axis a       : {a / 1e3:.2f} km")
    print(f"Escape-hyperbola eccentricity e          : {e:.4f}")
    print(f"Burn angle (beta)                        : {np.rad2deg(beta_angle):.2f} deg")

    # --- Propagate the escape trajectory out to 18 Mars radii ----------------
    xy0 = [-r * np.cos(beta_angle), r * np.sin(beta_angle)]
    vxy0 = [np.sin(beta_angle) * v_per, np.cos(beta_angle) * v_per]
    period_upper_bound = 18 * mars.radius / abs(v_hyp)

    probe = Probe(probe_equations, period_upper_bound, period_upper_bound,
                   x0=xy0[0], vx0=vxy0[0], y0=xy0[1], vy0=vxy0[1],
                   event=mars.radius * 18, eventflip=True)
    t, posvel = probe.odesolve()

    print(f"\nTime to reach 18 Mars radii: {timedelta(seconds=float(t[-1]))} (h:mm:ss)")

    pos_windows = sliding_window_view(posvel[:, 0:2], window_shape=[2, 2])
    segment_lengths = [
        np.sqrt((p[0, 0, 0] - p[0, 1, 0]) ** 2 + (p[0, 0, 1] - p[0, 1, 1]) ** 2)
        for p in pos_windows
    ]
    total_distance = sum(segment_lengths)
    direct_distance = np.linalg.norm(posvel[-1, 0:2])
    avg_speed = total_distance / t[-1]
    print(f"Direct distance from Mars centre at end   : {direct_distance / 1e3:.0f} km")
    print(f"Actual path length travelled              : {total_distance / 1e3:.0f} km")
    print(f"Average speed over the journey            : {avg_speed / 1e3:.3f} km/s")

    fig, ax = plt.subplots(figsize=(8, 8))
    plot_circle(ax, mars.radius / 1e3, color="red", label="Mars")
    plot_circle(ax, mars.radius * 18 / 1e3, color="red", linestyle=":", alpha=0.5, label="18 Mars radii")
    ax.plot(posvel[:, 0] / 1e3, posvel[:, 1] / 1e3, color="blue", label="Departure trajectory")
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_title("Exercise 9 — Hyperbolic departure trajectory from Mars")
    ax.set_aspect("equal")
    ax.legend()
    out_path = OUTPUT_DIR / "ex09_martian_departure.png"
    savefig(fig, out_path)
    print(f"\nPlot saved to {out_path}")


if __name__ == "__main__":
    main()
