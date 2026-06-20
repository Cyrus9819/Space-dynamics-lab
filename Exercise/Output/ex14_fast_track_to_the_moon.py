"""
Exercise 14 — A Fast Track to the Moon
=========================================

A classic Hohmann transfer to the Moon requires the least delta-v, but it
is not the fastest option. Here we analyse a different kind of transfer:
one where the post-burn point is *not* the periapsis of the resulting
orbit, characterised instead by a burn at a given altitude with the
velocity vector pointed at an angle ``psi`` away from the local horizontal.

Given the post-burn state (speed V0, altitude z0, angle psi0), the
resulting orbit's specific energy and angular momentum are:

    epsilon = 1/2 V0^2 - G M_earth / r0
    h       = r0 V0 cos(psi0)

from which the semi-major axis, eccentricity, and (via Kepler's orbit
equation and a flight-time integral) the time to reach the Moon's orbital
radius can all be derived analytically — letting us compare this fast
transfer directly against a standard Hohmann transfer from the same
departure altitude.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from space_base import GravBody, Probe  # noqa: E402
from utils import plot_circle, savefig  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

G = 6.674e-11
earth = GravBody.earth()
R_MOON = 384_400e3

Z0 = 300e3                  # initial altitude
V0 = 10.85e3                  # initial (post-burn) speed
PSI0 = np.deg2rad(6)          # angle between V and the local horizontal


def probe_equations(_, posvel):
    x, y, vx, vy = posvel
    r = np.sqrt(x ** 2 + y ** 2)
    f = -G * earth.mass / r ** 3
    return vx, vy, f * x, f * y


def flight_time(r, energy, h):
    """
    Time to travel from periapsis to a given radius `r`, for an orbit with
    specific energy `energy` and specific angular momentum `h`. See the
    module docstring for the orbital elements this is built from.
    """
    GM = G * earth.mass
    A = np.arcsin(-(GM + 2 * energy * r) / np.sqrt(GM ** 2 + 2 * energy * h ** 2)) + np.pi / 2
    B = np.sqrt(2 * energy * (h ** 2 - 2 * GM * r - 2 * energy * r ** 2)) / GM
    return (GM / (-2 * energy) ** 1.5) * (A - B)


def main():
    r0 = earth.radius + Z0

    energy0 = 0.5 * V0 ** 2 - G * earth.mass / r0
    h0 = r0 * V0 * np.cos(PSI0)
    a = -G * earth.mass / (2 * energy0)

    discriminant = r0 ** 2 - 4 * a * (r0 - a)
    e = max((-r0 + np.sqrt(discriminant)) / (2 * a),
            (-r0 - np.sqrt(discriminant)) / (2 * a))

    print(f"Specific energy    : {energy0:.2f} J/kg")
    print(f"Angular momentum    : {h0 / 1e6:.2f} x10^6 m^2/s")
    print(f"Semi-major axis     : {a / 1e3:.1f} km")
    print(f"Eccentricity        : {e:.4f}")

    travel_time = abs(flight_time(R_MOON, energy0, h0) - flight_time(r0, energy0, h0))
    print(f"Time to reach the Moon's orbit: {travel_time / (24 * 3600):.3f} days")

    xy0 = [-r0, 0.0]
    vxy0 = [-V0 * np.sin(PSI0), -V0 * np.cos(PSI0)]
    probe = Probe(probe_equations, travel_time, travel_time / 60,
                   x0=xy0[0], vx0=vxy0[0], y0=xy0[1], vy0=vxy0[1])
    t, posvel = probe.odesolve()

    # Continue propagating (free of any further burn) to show the full orbit shape.
    coast = Probe(probe_equations, travel_time * 4, travel_time * 4 / 60,
                   x0=posvel[-1, 0], vx0=posvel[-1, 2],
                   y0=posvel[-1, 1], vy0=posvel[-1, 3])
    _, posvel_after = coast.odesolve()

    r_after = np.linalg.norm(posvel_after[:, :2], axis=1)
    a_real = (np.min(r_after) + np.max(r_after)) / 2
    e_real = (np.max(r_after) - np.min(r_after)) / (np.max(r_after) + np.min(r_after))
    print(f"Simulated semi-major axis : {a_real / 1e3:.1f} km (analytic: {a / 1e3:.1f} km)")
    print(f"Simulated eccentricity    : {e_real:.4f} (analytic: {e:.4f})")

    # --- Comparison Hohmann transfer, same departure altitude -----------------
    r_per_h, r_aph_h = r0, R_MOON
    a_h = (r_per_h + r_aph_h) / 2
    journey_time_h = np.sqrt(4 * np.pi ** 2 * a_h ** 3 / (G * earth.mass)) / 2
    v0_h = np.sqrt(G * earth.mass * (2 / r_per_h - 1 / a_h))
    print(f"\nHohmann-transfer travel time: {journey_time_h / (24 * 3600):.3f} days "
          f"(vs. {travel_time / (24 * 3600):.3f} days for the fast-track burn)")

    probe_h = Probe(probe_equations, journey_time_h, journey_time_h / 60,
                     x0=-r0, vx0=0.0, y0=0.0, vy0=-v0_h)
    _, posvel_hohmann = probe_h.odesolve()

    # --- Plot: fast track vs Hohmann, with the Moon's own motion -------------
    fig, ax = plt.subplots(figsize=(10, 10))
    plot_circle(ax, earth.radius / 1e3, color="red", label="Earth")
    plot_circle(ax, R_MOON / 1e3, color="red", alpha=0.4, label="Moon's orbit")

    moon_period_days = np.sqrt(4 * np.pi ** 2 * R_MOON ** 3 / (G * earth.mass)) / (24 * 3600)
    day_angle = 2 * np.pi / moon_period_days
    days = np.arange(int(np.floor(moon_period_days)))
    ax.scatter((R_MOON / 1e3) * np.cos(days * day_angle),
                (R_MOON / 1e3) * np.sin(days * day_angle),
                color="red", s=15, label="Moon, position per day")

    ax.plot(posvel_hohmann[:, 0] / 1e3, posvel_hohmann[:, 1] / 1e3, color="darkorange",
             linestyle="--", label="Hohmann transfer")
    ax.plot(posvel[:, 0] / 1e3, posvel[:, 1] / 1e3, color="blue", label="Fast-track burn")
    ax.plot(posvel_after[:, 0] / 1e3, posvel_after[:, 1] / 1e3, color="blue",
             linestyle=":", alpha=0.5, label="Fast-track orbit (after arrival)")

    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_title("Exercise 14 — Fast track to the Moon vs. a Hohmann transfer")
    ax.set_aspect("equal")
    ax.legend(loc="upper right")
    out_path = OUTPUT_DIR / "ex14_fast_track_to_the_moon.png"
    savefig(fig, out_path)
    print(f"\nPlot saved to {out_path}")


if __name__ == "__main__":
    main()
