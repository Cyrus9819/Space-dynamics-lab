"""
Unit tests for space_base.GravBody / space_base.Probe.

Run with:  pytest -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from space_base import GravBody, Probe  # noqa: E402


def test_gravbody_earth_values():
    earth = GravBody.earth()
    assert earth.name == "Earth"
    assert earth.mass == pytest.approx(5.9722e24)
    assert earth.radius == pytest.approx(6371e3)


def test_gravbody_classmethods_exist():
    for body in (GravBody.earth(), GravBody.moon(), GravBody.mars(), GravBody.sun()):
        assert body.mass > 0
        assert body.radius > 0


def test_probe_requires_matching_initial_conditions():
    def driver(_, posvel):
        return posvel[1], 0.0

    with pytest.raises(AttributeError):
        Probe(driver, 10.0, 100, x0=0.0)  # missing vx0


def test_probe_1d_free_fall_matches_suvat():
    """A simple uniform-gravity free fall should match the analytic SUVAT result."""
    g = 9.81

    def driver(_, posvel):
        z, vz = posvel
        return vz, -g

    v0 = 100.0
    t_flight = 2 * v0 / g
    probe = Probe(driver, t_flight, 5000, x0=0.0, vx0=v0, event=0)
    t, posvel = probe.odesolve()

    analytic_max_height = v0 ** 2 / (2 * g)
    assert np.max(posvel[:, 0]) == pytest.approx(analytic_max_height, rel=1e-3)
    # The trajectory should be cut short once it returns to z=0.
    assert posvel[-1, 0] == pytest.approx(0.0, abs=1.0)


def test_probe_energy_conservation_without_drag():
    """Without drag, total mechanical energy should be conserved to high precision."""
    g = 9.81

    def driver(_, posvel):
        z, vz = posvel
        return vz, -g

    v0 = 200.0
    t_flight = 2 * v0 / g
    probe = Probe(driver, t_flight, 5000, x0=0.0, vx0=v0, event=0)
    t, posvel = probe.odesolve()

    z, vz = posvel[:, 0], posvel[:, 1]
    energy0 = 0.5 * vz[0] ** 2 + g * z[0]
    energy1 = 0.5 * vz[-1] ** 2 + g * z[-1]
    assert energy1 == pytest.approx(energy0, rel=1e-6)
