"""
Exercise 8b — Transfer Orbit to Asteroid 2013 LA2
====================================================

Asteroid 2013 LA2 has an eccentric orbit (e = 0.4656, a = 5.6841 AU), so
unlike the simple circular-orbit case of Exercise 8a, the "best" transfer
orbit depends on what we're optimising for:

- **Least transit time**: target the asteroid's *perihelion* — the closest
  point the asteroid ever gets to the Sun — to minimise the transfer
  ellipse's semi-major axis (and hence its period).
- **Smallest arrival burn**: target the asteroid's *aphelion* — where the
  asteroid itself is moving slowest — to minimise the velocity mismatch
  between probe and target on arrival.

These two goals trade off directly against each other: the fast transfer
needs a much bigger arrival burn, and the gentle-arrival transfer takes
roughly 3.5x longer.
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

AST_E = 0.4656
AST_A = au_to_m(5.6841)
AST_PER = AST_A * (1 - AST_E)
AST_APH = AST_A * (1 + AST_E)


def probe_equations(_, posvel):
    x, y, vx, vy = posvel
    r = np.sqrt(x ** 2 + y ** 2)
    f = -G * sun.mass / r ** 3
    return vx, vy, f * x, f * y


def transfer_orbit(r_per, r_aph):
    """Semi-major axis, eccentricity and half-period of a transfer ellipse."""
    a = (r_per + r_aph) / 2
    e = (r_aph - r_per) / (r_aph + r_per)
    period = np.sqrt(4 * np.pi ** 2 * a ** 3 / (G * sun.mass)) / 2
    return a, e, period


def plot_orbits():
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_circle(ax, 1.0, color="red", label="Earth")
    theta = np.linspace(0, 2 * np.pi, 200)
    x = m_to_au(AST_A - AST_PER) + m_to_au(AST_A) * np.cos(theta)
    y = m_to_au(AST_A) / 2 * np.sin(theta)
    ax.plot(x, y, color="green", label="Asteroid 2013 LA2")
    ax.set_xlabel("x (AU)")
    ax.set_ylabel("y (AU)")
    ax.set_title("Earth's orbit and asteroid 2013 LA2's orbit")
    ax.set_aspect("equal")
    ax.legend()
    savefig(fig, OUTPUT_DIR / "ex08b_orbits.png")


def least_transit_time():
    r_per = au_to_m(1.0)
    r_aph = AST_PER  # closest approach of the asteroid
    a, e, period = transfer_orbit(r_per, r_aph)

    ast_vel = np.sqrt(G * sun.mass * (2 / r_aph - 1 / AST_A))
    probe_vel = np.sqrt(G * sun.mass * (2 / r_aph - 1 / a))
    delta_v_arrival = abs(probe_vel + ast_vel)

    print("--- Least transit time (target the asteroid's perihelion) ---")
    print(f"Transfer a = {m_to_au(a):.4f} AU, e = {e:.4f}")
    print(f"Transit time            : {period / (24 * 3600):.1f} days")
    print(f"Arrival speed mismatch   : {delta_v_arrival / 1e3:.2f} km/s")

    v0 = np.sqrt(G * sun.mass * (2 / r_per - 1 / a))
    probe = Probe(probe_equations, period, period / 3600,
                   x0=r_per, vx0=0.0, y0=0.0, vy0=v0,
                   event=r_aph, eventflip=True)
    _, posvel = probe.odesolve()
    return posvel, period, delta_v_arrival


def smallest_arrival_burn():
    r_per = au_to_m(1.0)
    r_aph = AST_APH  # asteroid moving slowest here
    a, e, period = transfer_orbit(r_per, r_aph)

    ast_vel = np.sqrt(G * sun.mass * (2 / r_aph - 1 / AST_A))
    probe_vel = np.sqrt(G * sun.mass * (2 / r_aph - 1 / a))
    delta_v_arrival = abs(probe_vel + ast_vel)

    print("\n--- Smallest arrival burn (target the asteroid's aphelion) ---")
    print(f"Transfer a = {m_to_au(a):.4f} AU, e = {e:.4f}")
    print(f"Transit time            : {period / (24 * 3600):.1f} days "
          f"({period / (24 * 3600) / 365.25:.2f} years)")
    print(f"Arrival speed mismatch   : {delta_v_arrival / 1e3:.2f} km/s")

    v0 = np.sqrt(G * sun.mass * (2 / r_per - 1 / a))
    probe = Probe(probe_equations, period, period / 3600,
                   x0=-r_per, vx0=0.0, y0=0.0, vy0=-v0,
                   event=r_aph, eventflip=True)
    _, posvel = probe.odesolve()
    return posvel, period, delta_v_arrival


def main():
    plot_orbits()
    quickest_posvel, quickest_period, quickest_dv = least_transit_time()
    smallburn_posvel, smallburn_period, smallburn_dv = smallest_arrival_burn()

    print(f"\nTrading {quickest_period / (24*3600):.0f} days for "
          f"{smallburn_period / (24*3600):.0f} days of travel time cuts the "
          f"arrival burn from {quickest_dv/1e3:.1f} km/s down to "
          f"{smallburn_dv/1e3:.1f} km/s.")

    # --- Final comparison plot, with direction-of-travel arrows -------------
    fig, ax = plt.subplots(figsize=(9, 9))

    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), color="red", label="Earth")
    arrow_theta = np.linspace(0, 2 * np.pi, 9)[:-1]
    for th in arrow_theta:
        ax.arrow(np.cos(th), np.sin(th), -0.01 * np.sin(th), 0.01 * np.cos(th),
                  head_width=0.15, color="red")

    u = m_to_au(AST_A - AST_PER)
    a_ast, b_ast = m_to_au(AST_A), m_to_au(AST_A) / 2
    ax.plot(u + a_ast * np.cos(theta), b_ast * np.sin(theta), color="green", label="Asteroid")
    for th in arrow_theta:
        ax.arrow(u + a_ast * np.cos(th), b_ast * np.sin(th),
                  0.01 * np.sin(th), -0.01 * np.cos(th), head_width=0.15, color="green")

    ax.plot(m_to_au(quickest_posvel[:, 0]), m_to_au(quickest_posvel[:, 1]),
             color="blue", label="Quickest transfer")
    ax.plot(m_to_au(smallburn_posvel[:, 0]), m_to_au(smallburn_posvel[:, 1]),
             color="purple", label="Smallest-burn transfer")

    ax.set_xlabel("x (AU)")
    ax.set_ylabel("y (AU)")
    ax.set_title("Exercise 8b — Two transfer strategies to asteroid 2013 LA2")
    ax.set_aspect("equal")
    ax.legend()
    savefig(fig, OUTPUT_DIR / "ex08b_transfer_comparison.png")
    print(f"\nPlots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
