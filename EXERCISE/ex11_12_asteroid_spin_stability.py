"""
Exercise 11 & 12 — Spin Stability of an Irregular Asteroid
==============================================================

An irregularly shaped asteroid is modelled as 54 equal point masses laid
out on a grid (the (x, y, z) coordinates below were digitised from a
cross-sectional drawing of the asteroid at three "altitude" layers). From
this mass-lump model we compute the centre of mass and the inertia
tensor, and then study how the asteroid would tumble if set spinning about
each of its three principal axes.

This is a numerical demonstration of the **tennis-racket theorem**
(intermediate-axis theorem): torque-free rotation about the axis of
*largest* or *smallest* moment of inertia is stable, but rotation about
the *intermediate* axis is unstable — a small perturbation grows into a
large, periodic tumble.

Free-rotation (Euler's) equations
------------------------------------
    dWx/dt = -(Iz - Iy) Wy Wz / Ix
    dWy/dt = -(Ix - Iz) Wz Wx / Iy
    dWz/dt = -(Iy - Ix) Wx Wy / Iz

Since the spin kinetic energy ``E = sum_k (1/2) I_k W_k^2`` is conserved
(no external torque), every possible angular-velocity state lies on the
surface of an ellipsoid for a given E — the "spin energy ellipsoid" we
visualise below, together with sample trajectories traced out near each
principal axis.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from scipy.signal import argrelextrema

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import center_of_mass, inertia_tensor, savefig  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

# Grid spacing (metres) between adjacent mass-lump coordinates.
SCALE = 300

# (x, y, z) grid coordinates of each of the 54 equal mass lumps making up
# the asteroid model, digitised directly from the original cross-sectional
# mass model (three layers at z = -1, 0, +1).
MASSLUMPS = np.array([
    [-1, 4, 1], [0, 3, 1], [0, -5, 1], [1, 4, 1], [1, -2, 1],
    [-3, 3, 0], [-3, 1, 0], [-2, 4, 0], [-2, 2, 0], [-2, 0, 0], [-2, -4, 0],
    [-1, 5, 0], [-1, 3, 0], [-1, 1, 0], [-1, -1, 0], [-1, -3, 0], [-1, -5, 0], [-1, -7, 0],
    [0, 6, 0], [0, 4, 0], [0, 2, 0], [0, 0, 0], [0, -2, 0], [0, -4, 0], [0, -6, 0],
    [1, 5, 0], [1, 3, 0], [1, 1, 0], [1, -1, 0], [1, -3, 0], [1, -5, 0],
    [2, 4, 0], [2, 2, 0], [2, 0, 0], [2, -2, 0], [3, 1, 0],
    [-2, 2, -1], [-1, 5, -1], [-1, 3, -1], [-1, 1, -1], [-1, -5, -1],
    [0, 6, -1], [0, 4, -1], [0, 2, -1], [0, 0, -1], [0, 2, -1], [0, 4, -1], [0, 6, -1],
    [1, 5, -1], [1, 3, -1], [1, 1, -1], [1, -3, -1], [1, -5, -1], [2, 4, -1],
], dtype=float)

TOTAL_MASS = 5e13  # kg


def build_mass_model():
    """Return (masses, positions) of the asteroid's point-mass model, centred
    on its own centre of mass and expressed in metres."""
    grid_positions = MASSLUMPS * SCALE
    _, com = center_of_mass(np.ones(len(MASSLUMPS)), grid_positions)
    positions = grid_positions - com
    masses = np.full(len(MASSLUMPS), TOTAL_MASS / len(MASSLUMPS))
    return masses, positions, com


def euler_equations(_, omega, inertia):
    """Torque-free Euler's equations for rotation about principal axes."""
    Ix, Iy, Iz = inertia
    wx, wy, wz = omega
    dwx = -(Iz - Iy) * wy * wz / Ix
    dwy = -(Ix - Iz) * wz * wx / Iy
    dwz = -(Iy - Ix) * wx * wy / Iz
    return dwx, dwy, dwz


def analytic_period(I, axis_index, omega0):
    """
    Linearised (small-oscillation) period of motion near a principal axis,
    derived by treating the other two angular-velocity components as a
    harmonic oscillator (see the README for the full derivation).
    """
    i, j, k = {0: (0, 1, 2), 2: (2, 1, 0)}[axis_index]
    # Generic form valid for axis 'i' held (approximately) fixed:
    Ii, Ij, Ik = I[i], I[j], I[k]
    k_over_m = -omega0 ** 2 / (Ij * Ik) * (Ii - Ik) * (Ij - Ii)
    return 2 * np.pi / np.sqrt(k_over_m)


def main():
    masses, positions, com_grid = build_mass_model()
    print(f"Centre of mass (grid units): {np.round(com_grid / SCALE, 4)}")
    print(f"Centre of mass (metres)     : {np.round(com_grid, 2)}")

    I = inertia_tensor(masses, positions)
    moments, _ = np.linalg.eig(I)
    print(f"\nInertia tensor (kg m^2):\n{I}")
    print(f"\nPrincipal moments of inertia (kg m^2): {moments}")
    print("(Ix < Iy < Iz: spin about the X axis is the most 'compact', "
          "Z the most 'spread out'.)")

    # --- Spin energy ellipsoid + sample trajectories ------------------------
    target_energy = 3e19
    rx, ry, rz = np.sqrt(2 * target_energy / moments)

    # Six bundles of nearly-identical initial conditions, clustered near
    # each end of each principal axis (+/-X, +/-Y, +/-Z), each drawn in the
    # colour of its axis (red=X, blue=Y, green=Z).
    seed_directions = [
        (0, np.pi / 2, "red"), (np.pi / 2, np.pi / 2, "blue"), (0, 0, "green"),
        (np.pi, np.pi / 2, "red"), (3 * np.pi / 2, np.pi / 2, "blue"), (0, np.pi, "green"),
    ]
    t_final = 100
    n_traj_per_seed = 6
    t = np.linspace(0, t_final, 10_000)

    trajectories = []
    for u0, v0, color in seed_directions:
        for delta in np.linspace(0, np.pi / 24, n_traj_per_seed):
            u, v = u0 + delta / 2, v0 + delta / 2
            omega0 = np.array([rx * np.cos(u) * np.sin(v),
                                ry * np.sin(u) * np.sin(v),
                                rz * np.cos(v)])
            energy0 = 0.5 * np.sum(moments * omega0 ** 2)
            omega = odeint(euler_equations, omega0, t, args=(moments,), tfirst=True)
            energy1 = 0.5 * np.sum(moments * omega[-1] ** 2)
            accuracy = abs((energy1 - energy0) / energy0)
            trajectories.append((accuracy, omega, color))

    worst_accuracy = max(acc for acc, _, _ in trajectories)
    print(f"\nWorst energy-conservation error across all trajectories: {worst_accuracy:.4%}")

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="3d")
    u_ang = np.linspace(0, 2 * np.pi, 100)
    v_ang = np.linspace(0, np.pi, 100)
    ax.plot_surface(rx * np.outer(np.cos(u_ang), np.sin(v_ang)),
                      ry * np.outer(np.sin(u_ang), np.sin(v_ang)),
                      rz * np.outer(np.ones_like(u_ang), np.cos(v_ang)),
                      color="orange", alpha=0.35)
    for _, omega, color in trajectories:
        ax.plot(omega[:, 0], omega[:, 1], omega[:, 2], color=color, linewidth=1)
    max_radius = max(rx, ry, rz)
    for axis in "xyz":
        getattr(ax, f"set_{axis}lim")((-max_radius, max_radius))
    ax.set_xlabel("Wx")
    ax.set_ylabel("Wy")
    ax.set_zlabel("Wz")
    ax.set_title("Exercise 11 & 12 — Spin energy ellipsoid and trajectories")
    savefig(fig, OUTPUT_DIR / "ex11_12_spin_ellipsoid.png")

    print("\nThe small loops that hug the ellipsoid near the X and Z poles show\n"
          "STABLE spin: a small perturbation stays small. The large, looping\n"
          "excursions near the Y poles (the intermediate moment of inertia) show\n"
          "UNSTABLE spin — this is the tennis-racket / intermediate-axis theorem,\n"
          "and the same effect behind the 'Dzhanibekov effect' seen with tumbling\n"
          "objects in microgravity.")

    # --- Period of motion near the stable X and Z axes -----------------------
    def period_of_motion(omega_traj):
        d = np.linalg.norm(omega_traj - omega_traj[0], axis=1)
        idx = argrelextrema(d, np.less)[0]
        return t[idx[0]] if len(idx) else np.nan

    x_axis_traj = trajectories[1][1]                    # second member of the +X bundle
    z_axis_index = 2 * n_traj_per_seed + 1               # second member of the +Z bundle
    z_axis_traj = trajectories[z_axis_index][1]

    period_x_numeric = period_of_motion(x_axis_traj)
    period_z_numeric = period_of_motion(z_axis_traj)
    period_x_analytic = analytic_period(moments, 0, rx)
    period_z_analytic = analytic_period(moments, 2, rz)

    print(f"\nPeriod of motion near the X axis: numeric = {period_x_numeric:.3f} s, "
          f"analytic (linearised) = {period_x_analytic:.3f} s")
    print(f"Period of motion near the Z axis: numeric = {period_z_numeric:.3f} s, "
          f"analytic (linearised) = {period_z_analytic:.3f} s")

    print(f"\nPlot saved to {OUTPUT_DIR / 'ex11_12_spin_ellipsoid.png'}")


if __name__ == "__main__":
    main()
