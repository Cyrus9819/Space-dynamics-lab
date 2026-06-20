"""
Exercise 6 & 7 — 3D Drag, and Aerobraking at Mars
====================================================

Exercise 6 generalises the drag model of Exercises 3-4 to full 3D motion,
and Exercise 7 puts it to use: a probe arriving at Mars on a highly
elliptical orbit (periapsis 100 km above the surface, apoapsis ~48,000 km,
inclined 30 degrees to the equatorial plane) repeatedly skims the upper
atmosphere at periapsis. Each pass bleeds off a little orbital energy via
drag — "aerobraking" — gradually circularising the orbit until the probe
eventually skims too close and crashes.

Equations of motion (3D, with isothermal-atmosphere drag)
------------------------------------------------------------
    a_gravity = -(G M / r^3) * r_vec
    a_drag    = -(1/2) C_D A rho(h) |v| v_vec / m,   h = r - R_body

Note
----
The original coursework set the orbital-plane inclination using
``30 * pi / 130`` radians, which is actually ≈ 41.5°, not 30°. This has
been corrected here to ``np.deg2rad(30)`` so the inclination genuinely
matches the stated 30°; the qualitative aerobraking behaviour (and all of
the conclusions drawn from it) is unaffected.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from space_base import GravBody, Probe  # noqa: E402
from utils import plot_circle, savefig  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

G = 6.674e-11
DRAG_COEFFICIENT = 1.0
AREA = 0.01
MASS = 1.0

mars = GravBody.mars()

INCLINATION = np.deg2rad(30)  # see module docstring


def atmosphere_density(altitude):
    """Isothermal exponential atmosphere model for Mars."""
    return mars.surface_density * np.exp(-altitude / mars.scale_height)


def probe_equations(_, posvel):
    """3D equations of motion: gravity (point mass) + quadratic drag."""
    x, y, z, vx, vy, vz = posvel
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    f = -G * mars.mass / r ** 3
    gravity_acc = np.array([f * x, f * y, f * z])

    v2 = vx ** 2 + vy ** 2 + vz ** 2
    v_vec = np.array([vx, vy, vz])
    unit_v = v_vec / np.sqrt(v2)
    drag_acc = -0.5 * DRAG_COEFFICIENT * AREA * atmosphere_density(r - mars.radius) * v2 * unit_v / MASS

    ax, ay, az = gravity_acc + drag_acc
    return vx, vy, vz, ax, ay, az


def initial_conditions():
    r_p = mars.radius + 100e3
    r_a = 47_972e3
    a_initial = (r_p + r_a) / 2
    v = np.sqrt(2 * G * mars.mass * (1 / r_a - 1 / (2 * a_initial)))

    xyz0 = [r_a * np.cos(INCLINATION), 0.0, r_a * np.sin(INCLINATION)]
    vxyz0 = [0.0, v, 0.0]
    return xyz0, vxyz0, v


def run(t_final, t_num):
    xyz0, vxyz0, _ = initial_conditions()
    probe = Probe(probe_equations, t_final, t_num,
                   x0=xyz0[0], vx0=vxyz0[0],
                   y0=xyz0[1], vy0=vxyz0[1],
                   z0=xyz0[2], vz0=vxyz0[2],
                   event=mars.radius)
    return probe.odesolve()


def main():
    xyz0, vxyz0, v = initial_conditions()
    print(f"Initial periapsis-frame speed: {v:.3f} m/s")

    # --- Five-day pass: watch the orbit decay -----------------------------
    t, posvel = run(t_final=3600 * 24 * 5, t_num=3600 * 24 * 5)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(posvel[:, 0] / 1e3, posvel[:, 1] / 1e3, posvel[:, 2] / 1e3, color="red", linewidth=0.8)

    u_ang = np.linspace(0, 2 * np.pi, 60)
    v_ang = np.linspace(0, np.pi, 60)
    xs = mars.radius / 1e3 * np.outer(np.cos(u_ang), np.sin(v_ang))
    ys = mars.radius / 1e3 * np.outer(np.sin(u_ang), np.sin(v_ang))
    zs = mars.radius / 1e3 * np.outer(np.ones_like(u_ang), np.cos(v_ang))
    ax.plot_surface(xs, ys, zs, color="tab:orange", alpha=0.5)
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_zlabel("z (km)")
    ax.set_title("Exercise 6 & 7 — Aerobraking at Mars (5-day pass, 30° inclination)")
    savefig(fig, OUTPUT_DIR / "ex06_07_aerobraking_3d.png")

    # --- Eight-day pass: run until the probe crashes -----------------------
    t, posvel = run(t_final=3600 * 24 * 8, t_num=3600 * 24 * 8)
    t_end = len(t) - 2
    final_altitude = np.sqrt(np.sum(posvel[t_end, :3] ** 2)) - mars.radius
    print(f"Altitude at final logged step: {final_altitude:.2f} m (sanity check, should be small)")

    distance = np.linalg.norm(posvel[:, :3], axis=1)
    idx = argrelextrema(distance, np.greater)[0]
    apoapses = distance[idx]
    last_apoapsis = apoapses[-1]
    print(f"Apoapsis of the final complete orbit: {(last_apoapsis - mars.radius) / 1e3:.2f} km altitude")

    final_r_p = 100e3 + mars.radius
    final_r_a = last_apoapsis
    final_a = (final_r_p + final_r_a) / 2
    final_e = (final_r_a - final_r_p) / (final_r_a + final_r_p)
    print(f"Final orbit semi-major axis : {final_a / 1e3:.2f} km")
    print(f"Final orbit eccentricity    : {final_e:.4f}  (started highly eccentric, now much more circular)")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t / 3600, distance / 1e3, color="red")
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Distance from Mars centre (km)")
    ax.set_title("Exercise 6 & 7 — Orbital decay via aerobraking")
    ax.grid(alpha=0.3)
    savefig(fig, OUTPUT_DIR / "ex06_07_orbital_decay.png")
    print(f"Plots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
