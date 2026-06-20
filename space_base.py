"""
space_base
==========

Core building blocks shared by every exercise in this repository:

- :class:`GravBody` — a simple container for a gravitating body's physical
  properties (mass, radius, orbital separation, atmospheric model).
- :class:`Probe` — a thin, convenient wrapper around
  :func:`scipy.integrate.odeint` for propagating the state of a spacecraft
  (position + velocity, in 1D/2D/3D) through time, with optional support for
  stopping the integration early once a position-based "event" is reached
  (e.g. hitting the ground, or crossing a given radius).

This module is intentionally framework-agnostic: it knows nothing about
gravity, drag or orbital mechanics itself. All of the physics lives in the
ODE "driver" functions defined in each exercise script — ``Probe`` simply
drives whichever driver function it is given through time.
"""
import warnings
from typing import Callable, Optional, Sequence, Tuple

import numpy as np  # numerical python
from scipy.integrate import odeint  # the solver from scipy


class GravBody:
    """
    A gravitational body (star, planet or moon), described by its physical
    properties in SI units: mass, radius, separation (from its parent body,
    if any), and a simple exponential-atmosphere model (scale_height and
    surface_density).

    Only a handful of bodies are provided out of the box as convenience
    classmethods (:meth:`earth`, :meth:`moon`, :meth:`mars`, :meth:`sun`).
    For anything else, construct a ``GravBody`` directly with parameters
    sourced from a reliable reference such as the NASA Planetary Fact
    Sheets (https://nssdc.gsfc.nasa.gov/planetary/factsheet/).
    """

    def __init__(self, name: str, mass: Optional[float] = None,
                 radius: Optional[float] = None, separation: Optional[float] = None,
                 scale_height: Optional[float] = None, surface_density: Optional[float] = None):
        self.name = name.strip().capitalize()  # name of object
        self.mass = mass
        self.radius = radius
        self.separation = separation
        self.scale_height = scale_height
        self.surface_density = surface_density

    def __repr__(self) -> str:
        return (f"GravBody(name={self.name!r}, mass={self.mass!r}, "
                f"radius={self.radius!r}, separation={self.separation!r})")

    @classmethod
    def earth(cls) -> "GravBody":
        """Earth, with a simple isothermal-atmosphere model."""
        return cls('Earth', mass=5.9722e24, radius=6371e3,
                   separation=149.598e9, scale_height=8500, surface_density=1.217)

    @classmethod
    def moon(cls) -> "GravBody":
        """The Moon (airless, so no atmospheric parameters are set)."""
        return cls('Moon', mass=7.34767309e22, radius=1.7371e6,
                   separation=384.4e6)

    @classmethod
    def mars(cls) -> "GravBody":
        """Mars, with a simple isothermal-atmosphere model."""
        return cls('Mars', mass=0.64169e24, radius=3389.5e3,
                   separation=227.956e9, scale_height=11.1e3, surface_density=0.020)

    @classmethod
    def sun(cls) -> "GravBody":
        """The Sun."""
        return cls('Sun', mass=1_988_500e24, radius=695_700e3)


class Probe:
    """
    A probe/spacecraft with set values of drag-relevant properties (mass,
    cross-sectional area, drag coefficient) that propagates its own state
    (position and velocity) through time by numerically integrating a
    user-supplied ODE "driver" function.

    Handles 1D, 2D or 3D Cartesian cases transparently, based on which
    initial-condition keyword arguments are supplied.
    """

    def __init__(self, driver: Callable[[float, Sequence], Sequence], tfinal: float, tstepnum: int,
                 event: Optional[float] = None, eventflip: bool = False,
                 x0: Optional[float] = None, vx0: Optional[float] = None,
                 y0: Optional[float] = None, vy0: Optional[float] = None,
                 z0: Optional[float] = None, vz0: Optional[float] = None):
        """
        Parameters
        ----------
        driver
            The function that will drive the ODE solver. Called as
            ``driver(t, posvel)`` and must return the time-derivative of
            ``posvel`` (i.e. velocities followed by accelerations).
        tfinal
            The final time to integrate to, starting from t=0.
        tstepnum
            The number of steps in the linear time array; the more the
            better (smoother/more accurate), but the slower.
        event
            A position-magnitude threshold used to truncate the returned
            trajectory once crossed (e.g. ``event=earth.radius`` to stop
            once the probe reaches the ground).
        eventflip
            If True, the event triggers when the position magnitude rises
            *above* ``event`` rather than falls below it.
        x0, vx0
            Initial position/velocity in x (always required).
        y0, vy0
            Initial position/velocity in y (required for 2D & 3D cases).
        z0, vz0
            Initial position/velocity in z (required for 3D cases).
        """
        # Unrealistic defaults that exercises may use or override.
        self.mass = 1.
        self.area = 0.01
        self.drag_coefficient = 1.
        self.driver = driver  # the driver function for solving the differential equations
        self.tfinal = tfinal  # final time step
        self.tstepnum = int(tstepnum)  # number of steps for the solver to take
        self.event = event
        self.eventflip = eventflip

        # Build posvel0 (initial conditions array), inferring 1D/2D/3D from
        # which keyword arguments were supplied.
        if np.any([val is not None for val in (z0, vz0)]):  # 3D case
            if np.any([val is None for val in (x0, y0, z0, vx0, vy0, vz0)]):
                raise AttributeError('x0, y0, z0, vx0, vy0, vz0 must be defined')
            self.posvel0 = self.__posvel0_create__(x0, y0, z0, vx0, vy0, vz0)
        elif np.any([val is not None for val in (y0, vy0)]):  # 2D case
            if np.any([val is None for val in (x0, y0, vx0, vy0)]):
                raise AttributeError('x0, y0, vx0, vy0 must be defined')
            self.posvel0 = self.__posvel0_create__(x0, y0, vx0, vy0)
        else:  # 1D case
            if np.any([val is None for val in (x0, vx0)]):
                raise AttributeError('x0 and vx0 must be defined')
            self.posvel0 = self.__posvel0_create__(x0, vx0)
        return

    @staticmethod
    def __posvel0_create__(*conditions) -> list:
        num_vars = len(conditions)
        if num_vars % 2:
            raise ValueError('Require even number of inputs in posvel0 (2, 4, 6)')
        ind_split = num_vars // 2
        pos = list(conditions[:ind_split])
        vel = list(conditions[ind_split:])
        return pos + vel

    def odesolve(self) -> Tuple[Sequence, Sequence]:
        """
        Integrate the probe's equations of motion from t=0 to ``self.tfinal``
        and return ``(t, posvel)``.

        If ``self.event`` was set, the returned arrays are truncated at the
        first index where the position magnitude crosses the event
        threshold (in the direction set by ``self.eventflip``).
        """
        def quadsum(i):
            if len(i) == 1:
                return i[0]
            return np.sqrt(np.sum(i ** 2))
        warnings.simplefilter('ignore')
        t: np.ndarray = np.linspace(0, self.tfinal, self.tstepnum)  # linearly separated time steps
        posvel = np.array(odeint(self.driver, self.posvel0, t, tfirst=True))  # solved posvel
        if self.event is not None:
            pos: np.ndarray = posvel[:, :posvel.shape[1] // 2]
            try:
                if not self.eventflip:
                    ind: int = np.flatnonzero(np.array([quadsum(i) for i in pos]) < self.event)[0]
                else:
                    ind = np.flatnonzero(np.array([quadsum(i) for i in pos]) > self.event)[0]
                if ind == len(pos) - 1:
                    raise IndexError
            except IndexError:
                pass
            else:
                posvel = posvel[:ind + 1]
                t = t[:ind + 1]
        return t, posvel
