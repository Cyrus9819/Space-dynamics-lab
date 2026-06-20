"""
Exercise 13 — Kuiper Belt Object Interception (1994 GV9)
============================================================

Unlike the impulsive (instantaneous) burns of earlier exercises, a real
ion-thruster burns continuously over a long period, slowly losing mass as
propellant is expelled. We model this with the (continuous-thrust) rocket
equation, where thrust is delivered along the current velocity direction:

    a_thrust = (|dm/dt| * g0 * Isp / m) * v_hat

``space_base.Probe`` only supports up to 3 spatial dimensions (x, y, z
with matching velocities). To track the spacecraft's *mass* over time
without modifying ``space_base.py``, we creatively repurpose the unused
z/vz slots: z stores the current spacecraft mass, and vz stores its rate
of change (set to ``-mass_lost_rate`` while fuel remains, and 0 once dry).

We explore two strategies for reaching 1994 GV9's 43.6 AU circular orbit:

1. **Minimum fuel**: burn just enough to raise the orbit's aphelion to
   43.6 AU, then coast the rest of the way (cheap, but slow).
2. **Fastest journey**: keep burning until the spacecraft *itself* reaches
   43.6 AU (expensive, but much faster — and fast enough to leave the
   Sun's gravitational well entirely, as we'll see).

Both strategies use a simple iterative (interpolation-based) search to
find the fuel mass that hits the 43.6 AU target distance.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from space_base import GravBody, Probe  # noqa: E402
from utils import au_to_m, m_to_au, plot_circle, savefig  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

G = 6.674e-11
G0 = 9.80665          # standard gravity, used to convert Isp (s) to m/s
ISP = 3400.0           # specific impulse of the ion engine, s
MASS_LOST_RATE = 10e-6  # kg/s
DRY_MASS = 300.0        # kg
TARGET_DISTANCE_AU = 43.6

sun = GravBody.sun()


def rocket_equations(_, posvelmass):
    """
    6-element state: [x, y, mass, vx, vy, dmass/dt].
    The z/vz slots of Probe are repurposed to carry [mass, dm/dt] — see the
    module docstring for why.
    """
    x, y, mass, vx, vy, dmass_dt = posvelmass
    if mass <= DRY_MASS:
        dmass_dt_new = 0.0
    else:
        dmass_dt_new = -MASS_LOST_RATE

    r = np.sqrt(x ** 2 + y ** 2)
    f = -G * sun.mass / r ** 3
    gravity_acc = np.array([f * x, f * y])

    speed = np.linalg.norm([vx, vy])
    thrust_acc = np.array([vx, vy]) * abs(dmass_dt) * G0 * ISP / (mass * speed)

    ax, ay = gravity_acc + thrust_acc
    return vx, vy, dmass_dt_new, ax, ay, 0.0


def coast_equations(_, posvel):
    """Pure two-body (no-thrust) equations, used once the engine is dry."""
    x, y, vx, vy = posvel
    r = np.sqrt(x ** 2 + y ** 2)
    f = -G * sun.mass / r ** 3
    return vx, vy, f * x, f * y


def run_burn(fuel_mass):
    """Run the continuous-thrust burn for the given starting fuel mass,
    starting from a circular orbit at 1 AU."""
    v0 = np.sqrt(G * sun.mass / au_to_m(1.0))
    t_final = fuel_mass / MASS_LOST_RATE
    probe = Probe(rocket_equations, t_final, max(t_final / 3600, 10),
                   x0=au_to_m(1.0), vx0=0.0, y0=0.0, vy0=v0,
                   z0=DRY_MASS + fuel_mass, vz0=MASS_LOST_RATE)
    return probe.odesolve()


def reachable_distance_minimum_fuel(fuel_mass):
    """Aphelion distance (AU) reached by coasting after a minimal burn."""
    t, posvel = run_burn(fuel_mass)
    last_speed = np.linalg.norm(posvel[-1, 3:5])
    last_r = np.linalg.norm(posvel[-1, 0:2])
    a = abs(G * sun.mass / (last_speed ** 2 - 2 * G * sun.mass / last_r))
    period = np.sqrt(4 * np.pi ** 2 * a ** 3 / (G * sun.mass)) / 2

    coast = Probe(coast_equations, period, period / 3600,
                   x0=posvel[-1, 0], vx0=posvel[-1, 3],
                   y0=posvel[-1, 1], vy0=posvel[-1, 4])
    _, posvel_coast = coast.odesolve()
    return m_to_au(np.max(np.linalg.norm(posvel_coast[:, :2], axis=1))), posvel, posvel_coast


def reachable_distance_fastest(fuel_mass):
    """Maximum distance (AU) reached *during* the burn itself."""
    t, posvel = run_burn(fuel_mass)
    return m_to_au(np.max(np.linalg.norm(posvel[:, :2], axis=1))), posvel


def find_fuel_mass(reach_fn, initial_cache):
    """Iteratively interpolate to find the fuel mass that reaches the target distance."""
    cache = pd.DataFrame(initial_cache).set_index("d")
    dist = 1.0
    fuel_mass = None
    extra = None
    while abs(dist - TARGET_DISTANCE_AU) >= 0.01:
        fuel_mass = float(np.interp(TARGET_DISTANCE_AU, cache.index, cache["fuel_mass"]))
        result = reach_fn(fuel_mass)
        dist, *extra = result
        cache.loc[dist] = fuel_mass
        cache = cache.sort_index()
    return fuel_mass, extra


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # --- Illustrative single burn (1000 kg of fuel) -------------------------
    t, posvel = run_burn(1000.0)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    plot_circle(ax1, 1.0, color="red", label="Earth")
    plot_circle(ax1, TARGET_DISTANCE_AU, color="green", linestyle="--", label="1994 GV9 (43.6 AU)")
    ax1.plot(m_to_au(posvel[:, 0]), m_to_au(posvel[:, 1]), color="blue", label="Probe (1000 kg fuel)")
    ax1.set_xlabel("x (AU)"); ax1.set_ylabel("y (AU)")
    ax1.set_aspect("equal"); ax1.legend()
    ax1.set_title("Trajectory during burn")

    ax2.plot(t, posvel[:, 2])
    ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Spacecraft mass (kg)")
    ax2.set_title("Mass depletion")
    fig.suptitle("Exercise 13 — Continuous low-thrust burn (illustrative example)")
    savefig(fig, OUTPUT_DIR / "ex13_burn_illustration.png")

    # --- Strategy 1: minimum fuel --------------------------------------------
    print("--- Strategy 1: minimum fuel (burn + coast) ---")
    fuel_min, extra = find_fuel_mass(
        reachable_distance_minimum_fuel,
        {"d": [1.0, 500.0], "fuel_mass": [0.0, 500.0]},
    )
    burn_posvel, coast_posvel = extra
    print(f"Required fuel mass : {fuel_min:.1f} kg")

    r_aphelion = np.max(np.linalg.norm(coast_posvel[:, :2], axis=1))
    a_final = (au_to_m(1.0) + r_aphelion) / 2
    e_final = (r_aphelion - au_to_m(1.0)) / (r_aphelion + au_to_m(1.0))
    total_time_years = None  # computed below once we have both arrays' time axes
    print(f"Final orbit semi-major axis : {m_to_au(a_final):.2f} AU")
    print(f"Final orbit eccentricity    : {e_final:.4f}")

    fig, ax = plt.subplots(figsize=(8, 8))
    plot_circle(ax, 1.0, color="red", label="Earth")
    plot_circle(ax, TARGET_DISTANCE_AU, color="green", linestyle="--", label="1994 GV9")
    ax.plot(m_to_au(burn_posvel[:, 0]), m_to_au(burn_posvel[:, 1]), color="blue", label="Burn phase")
    ax.plot(m_to_au(coast_posvel[:, 0]), m_to_au(coast_posvel[:, 1]), color="orange",
             linestyle="--", label="Coast phase")
    ax.set_xlabel("x (AU)"); ax.set_ylabel("y (AU)")
    ax.set_aspect("equal"); ax.legend()
    ax.set_title("Exercise 13 — Strategy 1: minimum-fuel transfer")
    savefig(fig, OUTPUT_DIR / "ex13_minimum_fuel.png")

    # --- Strategy 2: fastest journey -----------------------------------------
    print("\n--- Strategy 2: fastest journey (burn all the way there) ---")
    fuel_fast, extra2 = find_fuel_mass(
        reachable_distance_fastest,
        {"d": [1.0, 500.0], "fuel_mass": [0.0, 100_000.0]},
    )
    fast_posvel, = extra2
    print(f"Required fuel mass : {fuel_fast:.1f} kg  "
          f"({fuel_fast / fuel_min:.1f}x more than Strategy 1)")

    v_final = np.linalg.norm(fast_posvel[-1, 3:5])
    dist_final = np.linalg.norm(fast_posvel[-1, 0:2])
    escape_speed = np.sqrt(2 * G * sun.mass / dist_final)
    print(f"Final speed       : {v_final / 1e3:.2f} km/s")
    print(f"Local escape speed : {escape_speed / 1e3:.2f} km/s "
          f"-> the probe is now on a hyperbolic (unbound) trajectory!")

    t_fast = fuel_fast / MASS_LOST_RATE
    print(f"Total travel time : {t_fast / (365.25 * 24 * 3600):.2f} years")

    fig, ax = plt.subplots(figsize=(8, 8))
    plot_circle(ax, 1.0, color="red", label="Earth")
    plot_circle(ax, TARGET_DISTANCE_AU, color="green", linestyle="--", label="1994 GV9")
    ax.plot(m_to_au(fast_posvel[:, 0]), m_to_au(fast_posvel[:, 1]), color="blue", label="Probe")
    ax.set_xlabel("x (AU)"); ax.set_ylabel("y (AU)")
    ax.set_aspect("equal"); ax.legend()
    ax.set_title("Exercise 13 — Strategy 2: fastest-journey transfer")
    savefig(fig, OUTPUT_DIR / "ex13_fastest_journey.png")

    print(f"\nSummary: cutting travel time from Strategy 1 to Strategy 2 multiplies\n"
          f"fuel use by ~{fuel_fast/fuel_min:.0f}x. A real mission would likely split the\n"
          f"difference, and/or use a gravity-assist flyby of Jupiter or Saturn to\n"
          f"gain free velocity.")
    print(f"\nPlots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
