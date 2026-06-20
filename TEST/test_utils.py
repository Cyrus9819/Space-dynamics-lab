"""
Unit tests for utils.py: unit conversions and rigid-body helpers.

Run with:  pytest -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import au_to_m, center_of_mass, inertia_tensor, m_to_au  # noqa: E402


def test_au_round_trip():
    assert m_to_au(au_to_m(1.524)) == pytest.approx(1.524)


def test_center_of_mass_symmetric_pair():
    masses = [1.0, 1.0]
    positions = [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    total_mass, com = center_of_mass(masses, positions)
    assert total_mass == pytest.approx(2.0)
    assert com == pytest.approx([0.0, 0.0, 0.0])


def test_inertia_tensor_point_on_x_axis():
    """A single point mass on the x-axis only has inertia about y and z."""
    masses = [2.0]
    positions = [[3.0, 0.0, 0.0]]
    I = inertia_tensor(masses, positions)
    assert I[0, 0] == pytest.approx(0.0)        # Ixx = m(y^2+z^2) = 0
    assert I[1, 1] == pytest.approx(2 * 3.0 ** 2)  # Iyy = m(x^2+z^2)
    assert I[2, 2] == pytest.approx(2 * 3.0 ** 2)  # Izz = m(x^2+y^2)


def test_inertia_tensor_matches_known_space_station_result():
    """Cross-check against Exercise 10's validated original-configuration result."""
    modules = np.array([
        [3, 6, -2, 0],
        [4, 3, 1, -2],
        [7, -4, 1, 2],
        [5, -4, -2, -4],
        [4, 3, 2, -1],
        [3, 2, -1, 6],
    ], dtype=float)
    masses, positions = modules[:, 0], modules[:, 1:]
    _, com = center_of_mass(masses, positions)
    assert com == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)

    I = inertia_tensor(masses, positions, about=com)
    expected = np.array([[298, -6, -24],
                          [-6, 620, -20],
                          [-24, -20, 446]], dtype=float)
    assert I == pytest.approx(expected)
