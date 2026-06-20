"""
utils
=====

Small, dependency-light helpers shared across the exercise scripts, split
out so that each exercise script can stay focused on the physics rather
than repeating unit-conversion and plotting boilerplate.

Contents
--------
- Unit conversions (AU <-> metres).
- Plotting helpers (square figures, drawing circles/orbits, consistent
  figure saving).
- Rigid-body helpers (centre of mass, inertia tensor) used by the space
  station and asteroid spin-stability exercises.
"""
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

AU = 1.496e11  # metres, 1 astronomical unit


# ----------------------------------------------------------------------
# Unit conversions
# ----------------------------------------------------------------------
def au_to_m(value_au):
    """Convert a distance (or array of distances) in AU to metres."""
    return np.asarray(value_au) * AU


def m_to_au(value_m):
    """Convert a distance (or array of distances) in metres to AU."""
    return np.asarray(value_m) / AU


# ----------------------------------------------------------------------
# Plotting helpers
# ----------------------------------------------------------------------
def new_square_figure(size: float = 8.0):
    """Return a fresh square ``(fig, ax)`` pair, a common need for orbit plots."""
    fig, ax = plt.subplots(figsize=(size, size))
    return fig, ax


def plot_circle(ax, radius: float, center=(0.0, 0.0), **kwargs):
    """Draw a circle of a given radius (e.g. a planet, moon, or circular orbit)."""
    theta = np.linspace(0, 2 * np.pi, 200)
    cx, cy = center
    ax.plot(cx + radius * np.cos(theta), cy + radius * np.sin(theta), **kwargs)


def savefig(fig, path, dpi: int = 150):
    """Save a figure with consistent settings, then close it to free memory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Rigid-body helpers
# ----------------------------------------------------------------------
def center_of_mass(masses: Sequence[float], positions: Sequence[Sequence[float]]):
    """
    Compute the centre of mass of a collection of point masses.

    Parameters
    ----------
    masses
        Array of shape ``(N,)`` with each point's mass.
    positions
        Array of shape ``(N, 3)`` with each point's (x, y, z) position.

    Returns
    -------
    total_mass : float
    com : np.ndarray, shape (3,)
    """
    masses = np.asarray(masses, dtype=float)
    positions = np.asarray(positions, dtype=float)
    total_mass = masses.sum()
    com = (masses[:, None] * positions).sum(axis=0) / total_mass
    return total_mass, com


def inertia_tensor(masses: Sequence[float], positions: Sequence[Sequence[float]],
                    about: Optional[Sequence[float]] = None) -> np.ndarray:
    """
    Compute the 3x3 inertia tensor of a collection of point masses.

        I = [[ Ixx, Ixy, Ixz],
             [ Ixy, Iyy, Iyz],
             [ Ixz, Iyz, Izz]]

    where ``Ixx = sum(m * (y**2 + z**2))``, ``Ixy = -sum(m * x * y)``, etc.

    Parameters
    ----------
    masses
        Array of shape ``(N,)``.
    positions
        Array of shape ``(N, 3)``.
    about
        If given, positions are shifted by ``-about`` first (e.g. pass the
        centre of mass to get the inertia tensor about the centre of mass).
        If omitted, positions are assumed to already be expressed about the
        point of interest.
    """
    masses = np.asarray(masses, dtype=float)
    positions = np.asarray(positions, dtype=float)
    if about is not None:
        positions = positions - np.asarray(about, dtype=float)

    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
    Ixx = np.sum(masses * (y ** 2 + z ** 2))
    Iyy = np.sum(masses * (x ** 2 + z ** 2))
    Izz = np.sum(masses * (x ** 2 + y ** 2))
    Ixy = -np.sum(masses * x * y)
    Ixz = -np.sum(masses * x * z)
    Iyz = -np.sum(masses * y * z)
    return np.array([[Ixx, Ixy, Ixz],
                      [Ixy, Iyy, Iyz],
                      [Ixz, Iyz, Izz]])


def principal_axes(inertia: np.ndarray):
    """
    Diagonalise an inertia tensor.

    Returns
    -------
    moments : np.ndarray, shape (3,)
        Principal moments of inertia (eigenvalues), ascending order is NOT
        guaranteed (matches ``np.linalg.eig`` ordering).
    axes : np.ndarray, shape (3, 3)
        Columns are the corresponding principal-axis unit vectors.
    """
    moments, axes = np.linalg.eig(inertia)
    return moments, axes
