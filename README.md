# Space Dynamics Lab

A from-scratch numerical astrodynamics toolkit and a tour of fourteen
exercises in orbital mechanics, atmospheric drag and rigid-body dynamics —
covering everything from a ball thrown straight up, to Hohmann transfers,
aerobraking at Mars, low-thrust interplanetary cruise, and the tennis-racket
(intermediate-axis) theorem applied to a tumbling asteroid.

Every simulation is built on top of a small, self-contained ODE-integration
wrapper (`space_base.py`) rather than a heavyweight astrodynamics library —
the point of this project is to show the underlying physics and numerics,
not to hide them behind a framework.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![tests](https://github.com/yourusername/space-dynamics-lab/actions/workflows/tests.yml/badge.svg)](https://github.com/yourusername/space-dynamics-lab/actions/workflows/tests.yml)

> Replace `yourusername/space-dynamics-lab` above with your actual GitHub
> path once this repository is pushed, so the CI badge resolves correctly.

---

## Contents

- [Overview](#overview)
- [Repository structure](#repository-structure)
- [Getting started](#getting-started)
- [The core framework](#the-core-framework-space_basepy)
- [Exercise gallery](#exercise-gallery)
- [Notes on changes from the original coursework](#notes-on-changes-from-the-original-coursework)
- [Testing & continuous integration](#testing--continuous-integration)
- [License](#license)

---

## Overview

This repository grew out of a university course on space dynamics. Each
exercise increases in complexity along one of three threads:

| Thread | Exercises | Ideas covered |
|---|---|---|
| **Atmospheric flight** | 1 – 4 | Uniform vs. inverse-square gravity, quadratic drag, exponential atmosphere models |
| **Orbital mechanics** | 5, 7 – 9, 13, 14 | 3D point-mass gravity, Hohmann transfers, hyperbolic escape, continuous low-thrust trajectories |
| **Rigid-body dynamics** | 6, 10, 11, 12 | Inertia tensors, principal axes, torque-free Euler rotation, spin stability |

Rather than one monolithic notebook, every exercise is a small, focused,
independently runnable Python script under [`exercises/`](exercises/), built
on two shared modules at the repository root:

- **`space_base.py`** — the original course-provided ODE-integration
  framework (`GravBody`, `Probe`), lightly extended.
- **`utils.py`** — unit conversions, plotting helpers and rigid-body math
  (centre of mass, inertia tensor) shared by several exercises.

## Repository structure

```
space-dynamics-lab/
├── space_base.py              # core GravBody / Probe framework
├── utils.py                   # shared helpers (units, plotting, inertia tensor)
├── run_all.py                 # convenience script: runs every exercise
├── requirements.txt
├── LICENSE
├── exercises/
│   ├── ex01_uniform_gravity.py
│   ├── ex02_realistic_gravity.py
│   ├── ex03_drag_uniform_atmosphere.py
│   ├── ex04_isothermal_atmosphere.py
│   ├── ex05_lunar_probe_haywire.py
│   ├── ex06_07_mars_aerobraking.py
│   ├── ex08a_hohmann_transfer_mars.py
│   ├── ex08b_asteroid_2013la2_transfer.py
│   ├── ex09_martian_escape_trajectory.py
│   ├── ex10_space_station_inertia_tensor.py
│   ├── ex11_12_asteroid_spin_stability.py
│   ├── ex13_kuiper_belt_interception.py
│   ├── ex14_fast_track_to_the_moon.py
│   └── outputs/                # generated plots land here (see gallery below)
├── tests/
│   ├── test_space_base.py
│   └── test_utils.py
└── .github/workflows/tests.yml # CI: runs the test suite on every push
```

## Getting started

```bash
git clone https://github.com/yourusername/space-dynamics-lab.git
cd space-dynamics-lab
python -m venv .venv && source .venv/bin/activate   # optional, but recommended
pip install -r requirements.txt
```

Run a single exercise directly:

```bash
python exercises/ex01_uniform_gravity.py
```

Or run everything (regenerates every plot under `exercises/outputs/`):

```bash
python run_all.py            # all 13 scripts
python run_all.py ex08 ex14  # only scripts whose filename starts with these
```

> Exercises 6/7, 11/12 and 13 integrate over long, fine time grids (and, in
> the case of Exercise 13, run an iterative root-find that repeats the
> integration several times), so they take noticeably longer than the rest
> — typically under a minute each.

Run the test suite:

```bash
pytest -q
```

## The core framework (`space_base.py`)

Two classes do all of the heavy lifting:

- **`GravBody`** — a plain-data container for a gravitating body's mass,
  radius, orbital separation, and a simple isothermal-atmosphere model
  (surface density + scale height). Built-in bodies: `GravBody.earth()`,
  `GravBody.moon()`, `GravBody.mars()`, `GravBody.sun()`.
- **`Probe`** — wraps `scipy.integrate.odeint` to propagate a 1D, 2D or 3D
  state vector (position + velocity) through time, given a driver function
  `driver(t, posvel) -> d(posvel)/dt`. It optionally truncates the returned
  trajectory once a position-magnitude **event** is crossed (e.g. hitting
  the ground, or reaching a target orbital radius) — this is what lets
  Exercise 1's projectile stop exactly when it lands, or Exercise 7's probe
  stop exactly when it crashes into Mars.

All of the actual physics — gravity models, drag laws, rocket-thrust
equations — lives in the small `driver` functions defined inside each
exercise script, keeping `space_base.py` itself completely agnostic to what
it's simulating.

## Exercise gallery

### Exercise 1 — Vertical motion under uniform gravity
*(`exercises/ex01_uniform_gravity.py`)*

A baseline sanity check for the solver: a ball thrown straight up at
850 m/s under constant gravity traces an exact parabola, reaching
**36,824.7 m**, with mechanical energy conserved to ~10⁻¹⁶ relative error
— essentially machine precision.

![Exercise 1](exercises/outputs/ex01_uniform_gravity.png)

### Exercise 2 — Realistic (inverse-square) gravity
*(`exercises/ex02_realistic_gravity.py`)*

Swapping in `g(z) = GM / (R + z)²` instead of a constant 9.81 m/s² raises
the apex by **176.7 m** and the flight time by **1.2 s** compared to the
uniform-gravity case — gravity is measurably weaker a few hundred metres up.

![Exercise 2](exercises/outputs/ex02_realistic_gravity.png)

### Exercise 3 — Drag in a uniform atmosphere
*(`exercises/ex03_drag_uniform_atmosphere.py`)*

Adding quadratic air drag (`F = -½ C_D ρ A |v| v`) slashes the apex to just
**501.8 m** and dumps **99.8%** of the launch energy as heat by the time the
projectile lands — drag dominates uniform gravity almost completely at this
speed and density.

![Exercise 3](exercises/outputs/ex03_drag_uniform_atmosphere.png)

### Exercise 4 — An isothermal atmosphere model
*(`exercises/ex04_isothermal_atmosphere.py`)*

Replacing the constant air density with a realistic exponential profile,
`ρ(h) = ρ₀ exp(-h/H)`, lets the projectile coast a little higher
(**512.8 m**, vs. 501.8 m with constant density) because the air above
~20 km is already noticeably thinner.

![Exercise 4](exercises/outputs/ex04_density_profile.png)

### Exercise 5 — Probe goes haywire (3D motion around the Moon)
*(`exercises/ex05_lunar_probe_haywire.py`)*

A malfunctioning probe-launcher fires a fan of probes from the lunar
surface at exactly the local circular-orbit speed (**1.68 km/s**). Every
resulting trajectory is itself a circular orbit grazing the surface — so on
a perfectly spherical Moon, there is *no* safe spot; every point is
reachable for the right launch angle.

![Exercise 5](exercises/outputs/ex05_lunar_probe_haywire.png)

### Exercises 6 & 7 — 3D drag and aerobraking at Mars
*(`exercises/ex06_07_mars_aerobraking.py`)*

The drag model is generalised to 3D and applied to a probe arriving at Mars
on a highly eccentric orbit (periapsis 100 km, apoapsis ≈ 48,000 km,
inclined 30°). Each periapsis pass clips the upper atmosphere and bleeds
off energy, dragging the apoapsis down from ~48,000 km to **~792 km** after
8 days — the orbit's eccentricity drops from near-1 to **0.090**, before the
probe eventually skims too low and crashes.

![Exercises 6 & 7](exercises/outputs/ex06_07_aerobraking_3d.png)
![Exercises 6 & 7 decay](exercises/outputs/ex06_07_orbital_decay.png)

### Exercise 8a — Hohmann transfer orbit, Earth to Mars
*(`exercises/ex08a_hohmann_transfer_mars.py`)*

The textbook minimum-energy transfer: a **32.7 km/s** departure burn onto a
1.262 AU, e = 0.208 ellipse gets a probe from Earth's orbit to Mars' orbit
in **259 days**. Under-burning by just 0.05% leaves the probe over half a
million kilometres short of Mars — a stark demonstration of how sensitive
interplanetary targeting is to departure velocity.

![Exercise 8a](exercises/outputs/ex08a_hohmann_transfer.png)

### Exercise 8b — Transfer to asteroid 2013 LA2
*(`exercises/ex08b_asteroid_2013la2_transfer.py`)*

Because the target asteroid's own orbit is itself eccentric (e = 0.466),
"best" transfer depends on the goal:

| Strategy | Target point | Transit time | Arrival Δv |
|---|---|---|---|
| Least transit time | Asteroid's perihelion | 524 days | 32.7 km/s |
| Smallest arrival burn | Asteroid's aphelion | 1840 days (5.0 yr) | 12.3 km/s |

![Exercise 8b](exercises/outputs/ex08b_transfer_comparison.png)

### Exercise 9 — The Martian: escaping Mars towards Earth
*(`exercises/ex09_martian_escape_trajectory.py`)*

Starting from a 200 km circular parking orbit (**3.45 km/s**), we compute
the hyperbolic departure burn (**5.56 km/s** at periapsis, on an e = 1.59
escape hyperbola) needed to match the heliocentric speed required for a
Hohmann transfer back to Earth, then propagate the escape trajectory out to
18 Mars radii.

![Exercise 9](exercises/outputs/ex09_martian_departure.png)

### Exercise 10 — Moment of inertia tensor of a space station
*(`exercises/ex10_space_station_inertia_tensor.py`)*

A six-module space station's mass, centre of mass and full inertia tensor
are computed and diagonalised. Increasing the solar arrays' mass from 3 t
to 5 t raises every principal moment of inertia (more resistance to being
spun up or down) and shifts the centre of mass away from the origin.

```
Principal moments, original   : [293.95, 447.75, 622.30] t·m²
Principal moments, +2 t arrays: [346.85, 474.82, 694.61] t·m²
```

### Exercises 11 & 12 — Spin stability of an irregular asteroid
*(`exercises/ex11_12_asteroid_spin_stability.py`)*

A 54-point mass-lump model of an irregular asteroid is used to compute its
inertia tensor and principal moments, then to numerically integrate
torque-free Euler's equations of rotation. The result is a clean
demonstration of the **tennis-racket (intermediate-axis) theorem**: spin
near the axes of *largest* or *smallest* moment of inertia stays tightly
bounded (stable), while spin near the *intermediate* axis blows up into
large, periodic tumbles (unstable) — the same physics behind the
"Dzhanibekov effect" seen with spinning tools in microgravity.

![Exercises 11 & 12](exercises/outputs/ex11_12_spin_ellipsoid.png)

### Exercise 13 — Kuiper belt object interception (1994 GV9)
*(`exercises/ex13_kuiper_belt_interception.py`)*

A continuous-thrust ion engine (Isp = 3400 s) is modelled with a
time-varying spacecraft mass, and two strategies for reaching 1994 GV9's
43.6 AU orbit are compared via an iterative fuel-mass search:

| Strategy | Fuel required | Travel time |
|---|---|---|
| Minimum fuel (burn + coast) | 147 kg | several decades (highly eccentric coast) |
| Fastest journey (burn the whole way) | 3,400 kg (≈23×) | 10.8 years |

The fastest-journey probe ends up travelling at 75.6 km/s against a local
escape speed of just 6.4 km/s — by the time it arrives, it's no longer
gravitationally bound to the Sun at all.

![Exercise 13](exercises/outputs/ex13_fastest_journey.png)

### Exercise 14 — A fast track to the Moon
*(`exercises/ex14_fast_track_to_the_moon.py`)*

A burn at 300 km altitude, 10.85 km/s, angled 6° off the local horizontal,
reaches the Moon's orbital radius in just **3.4 days** — noticeably faster
than the **5.0 days** a same-altitude Hohmann transfer would take, at the
cost of a more eccentric (and less fuel-efficient) intermediate orbit.

![Exercise 14](exercises/outputs/ex14_fast_track_to_the_moon.png)

## Notes on changes from the original coursework

In the interest of transparency, this cleanup involved a few deliberate
changes beyond restructuring and commenting:

- **Shared helpers extracted.** Unit conversions, orbit-plotting boilerplate
  and the rigid-body inertia-tensor maths (used independently in both
  Exercise 10 and Exercises 11/12 in the original work) were consolidated
  into `utils.py` to avoid duplicated logic.
- **`GravBody` convenience constructors added.** `GravBody.moon()`,
  `GravBody.mars()` and `GravBody.sun()` were added alongside the original
  `GravBody.earth()`, so exercises no longer need to hand-type the same
  physical constants repeatedly.
- **A units bug was fixed in Exercises 6/7.** The original orbital
  inclination was set with `30 * pi / 130`, which evaluates to ≈ 41.5°, not
  the intended 30°. This has been corrected to `np.deg2rad(30)`. The
  qualitative aerobraking results (and all conclusions drawn from them) are
  unaffected.
- **Gravitational constant precision.** Scripts use `G = 6.674 × 10⁻¹¹`
  (CODATA-recommended) rather than the original `6.67 × 10⁻¹¹`. Most results
  are essentially unchanged, but a few quantities that depend on a delicate
  cancellation between kinetic and potential energy (e.g. the semi-major
  axis in Exercise 14) shift by a few percent as a result — this is an
  improvement in physical accuracy, not a bug.
- **Mass-lump asteroid model (Exercises 11/12) reproduced verbatim**,
  including a few repeated grid coordinates present in the original model,
  since changing them would no longer match the validated inertia tensor
  the rest of the analysis depends on.

## Testing & continuous integration

`tests/` contains a small `pytest` suite covering the core framework
(`GravBody`, `Probe`) and the shared rigid-body/unit-conversion helpers in
`utils.py`, including a regression check against Exercise 10's previously
validated inertia tensor. A GitHub Actions workflow
(`.github/workflows/tests.yml`) runs this suite automatically on every push
and pull request, across Python 3.10–3.12.

## License

Released under the [MIT License](LICENSE).
