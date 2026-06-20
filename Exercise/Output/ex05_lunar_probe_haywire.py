"""
Exercise 5 — Probe Goes Haywire (3D Motion Around the Moon)
==============================================================

A probe-launcher in low lunar orbit malfunctions, firing probes at the
circular-orbit speed but in a fan of different directions. The question:
is there a "safe" spot on the Moon's surface that none of the probes can
reach?

We switch to full 3D Cartesian coordinates, with gravity now expressed
vectorially:

    d^2(x,y,z)/dt^2 = -(G M / r^3) * (x, y, z),   r = |(x, y, z)|

A probe is launched from the lunar surface with the *circular orbit speed*

    V = sqrt(G M_moon / R_moon)

(i.e. ``Q = V^2 R_moon / (G M_moon) = 1``) at a range of launch angles. Each
resulting trajectory is itself a circular orbit at the surface altitude —
so, in the idealised case of a perfectly spherical Moon, every point on the
surface is reachable for the right launch angle. There is no safe spot.
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
moon = GravBody.moon()


def probe_equations(_, posvel):
    """Point-mass gravity only, no atmosphere (the Moon has none)."""
    x, y, z, vx, vy, vz = posvel
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    f = -G * moon.mass / r ** 3
    return vx, vy, vz, f * x, f * y, f * z


def main():
    # Q = V^2 R / (G M); setting Q = 1 gives exactly the circular-orbit speed.
    Q = 1
    v = np.sqrt(Q * G * moon.mass / moon.radius)
    print(f"Launch speed (circular-orbit speed): {v / 1e3:.4f} km/s")

    t_final = 3600 * 12   # 12 hours, plenty of time to complete several loops
    t_num = t_final        # 1 step per second
    xyz0 = [moon.radius, 0.0, 0.0]

    fig, ax = plt.subplots(figsize=(10, 10))
    init_angles = np.arange(-np.pi / 2, np.pi / 2, np.pi / 24)
    for angle in init_angles:
        vxyz0 = [v * np.cos(angle), v * np.sin(angle), 0.0]
        probe = Probe(probe_equations, t_final, t_num,
                       x0=xyz0[0], vx0=vxyz0[0],
                       y0=xyz0[1], vy0=vxyz0[1],
                       z0=xyz0[2], vz0=vxyz0[2],
                       event=moon.radius)
        _, posvel = probe.odesolve()
        ax.plot(posvel[:, 0] / 1e3, posvel[:, 1] / 1e3, color="red", linewidth=0.8)

    plot_circle(ax, moon.radius / 1e3, color="blue", linewidth=1.5, label="Moon")
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_title("Exercise 5 — Fan of probe trajectories around the Moon")
    ax.set_aspect("equal")
    ax.legend()
    out_path = OUTPUT_DIR / "ex05_lunar_probe_haywire.png"
    savefig(fig, out_path)
    print(f"Plot saved to {out_path}")
    print("\nConclusion: at Q = 1, the launch speed equals the local circular-orbit\n"
          "speed, so every trajectory is itself a circular orbit grazing the\n"
          "surface. With a perfectly spherical Moon, every point on the surface\n"
          "is reachable for some launch angle — there is no safe spot.")


if __name__ == "__main__":
    main()
