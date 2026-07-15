"""
Module 4.1 — Numerical Orbit Propagator
========================================

Replaces the analytical KeplerianPropagator with a NumericalPropagator.

Architecture:
    KeplerianOrbit
        -> SpacecraftState
        -> DormandPrince853Integrator
        -> NumericalPropagator
        -> Force Model: NewtonianAttraction (central gravity only)
        -> propagate(time)
        -> PVCoordinates
        -> CSV + Plot

The initial orbit, propagation loop, output format (CSV columns),
and plots are identical to the original notebook so downstream
modules do not require modification.

Only central gravity (mu) is modeled here.  The numerical trajectory
should match the analytical Keplerian trajectory to within integrator
precision, validating the setup before J2, drag, SRP, and third-body
forces are added in modules 4.2–4.5.

Usage (Colab / local):
    1. pip install orekit-jpype
    2. Place orekit-data-main.zip in the working directory
    3. python numerical_propagator.py
"""

import sys
import os
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend so it works headless
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────
# 1.  Orekit JVM + data bootstrap
# ──────────────────────────────────────────────
import orekit_jpype as orekit
orekit.initVM()

from orekit_jpype.pyhelpers import setup_orekit_data

DATA_ZIP = "orekit-data-main.zip"
if os.path.exists(DATA_ZIP):
    setup_orekit_data(DATA_ZIP)
else:
    print(f"WARNING: {DATA_ZIP} not found in {os.getcwd()}")
    print("         Orekit will try to use the default data loader.")

# ──────────────────────────────────────────────
# 2.  Verify Orekit loaded correctly
# ──────────────────────────────────────────────
from org.orekit.time import TimeScalesFactory

utc = TimeScalesFactory.getUTC()
print("UTC:", utc)
print("SUCCESS: Orekit data loaded correctly!\n")

# ──────────────────────────────────────────────
# 3.  Imports
# ──────────────────────────────────────────────
from org.orekit.frames import FramesFactory
from org.orekit.time import AbsoluteDate
from org.orekit.utils import Constants
from org.orekit.orbits import KeplerianOrbit, PositionAngleType, OrbitType
from org.orekit.propagation import SpacecraftState
from org.orekit.propagation.numerical import NumericalPropagator
from org.hipparchus.ode.nonstiff import DormandPrince853Integrator
from org.orekit.forces.gravity import NewtonianAttraction

# ──────────────────────────────────────────────
# 4.  Initial orbit  (UNCHANGED from notebook)
# ──────────────────────────────────────────────
initial_date = AbsoluteDate(
    2026, 1, 1,
    12, 0, 0.0,
    utc
)

frame = FramesFactory.getEME2000()

orbit = KeplerianOrbit(
    7000000.0,                 # Semi-major axis (m)
    0.001,                     # Eccentricity
    math.radians(98.0),        # Inclination
    math.radians(0.0),         # Argument of Perigee
    math.radians(0.0),         # RAAN
    math.radians(0.0),         # True Anomaly
    PositionAngleType.TRUE,
    frame,
    initial_date,
    Constants.WGS84_EARTH_MU
)

# Compute orbital period from the orbit (reusable for any orbit)
orbital_period = orbit.getKeplerianPeriod()   # seconds
num_orbits     = 1                            # number of full orbits to propagate
step_size      = 60                           # output time step (s)

print("Initial orbit defined:")
print(f"  a       = {orbit.getA():.1f} m")
print(f"  e       = {orbit.getE():.6f}")
print(f"  i       = {math.degrees(orbit.getI()):.2f} deg")
print(f"  mu      = {Constants.WGS84_EARTH_MU:.4e} m^3/s^2")
print(f"  Period  = {orbital_period:.2f} s  ({orbital_period/60:.2f} min)")
print()

# ──────────────────────────────────────────────
# 5.  Numerical Propagator Setup  (NEW)
# ──────────────────────────────────────────────

# 5a. Wrap orbit in a SpacecraftState with spacecraft mass.
#     Mass is required by drag (Module 4.3) and SRP (Module 4.4).
SPACECRAFT_MASS = 1000.0   # kg
initial_state = SpacecraftState(orbit, SPACECRAFT_MASS)

# 5b. DormandPrince853 integrator
min_step  = 0.001      # minimum step size  (s)
max_step  = 1000.0     # maximum step size  (s)
init_step = 60.0       # initial step size  (s)

# Compute orbit-appropriate tolerances using Orekit's built-in method.
# This derives position & velocity tolerances from the orbit geometry,
# which is more robust than hardcoding values — especially as
# perturbations are added in later modules.
dP = 1.0               # desired position accuracy (m)
tolerances = NumericalPropagator.tolerances(dP, orbit, OrbitType.KEPLERIAN)
abs_tol = tolerances[0]  # absolute tolerances
rel_tol = tolerances[1]  # relative tolerances

integrator = DormandPrince853Integrator(
    min_step, max_step,
    abs_tol,
    rel_tol
)
integrator.setInitialStepSize(init_step)

# 5c. Build the NumericalPropagator
#     OrbitType.KEPLERIAN keeps the internal state representation
#     consistent with the KeplerianOrbit we defined above.
#     CARTESIAN would also work but is typically preferred when
#     perturbations make Keplerian elements singular (e.g. near-circular
#     equatorial orbits). For this standard LEO, Keplerian is fine.
propagator = NumericalPropagator(integrator)
propagator.setOrbitType(OrbitType.KEPLERIAN)
propagator.setInitialState(initial_state)

# 5d. Force models — structured as a list for easy extension.
#     Modules 4.2–4.5 will append J2, drag, SRP, and third-body models here.
force_models = []
force_models.append(NewtonianAttraction(Constants.WGS84_EARTH_MU))  # central gravity

for fm in force_models:
    propagator.addForceModel(fm)

print("Numerical propagator configured:")
print(f"  Integrator : DormandPrince853")
print(f"  Orbit type : KEPLERIAN")
print(f"  SC mass    : {SPACECRAFT_MASS} kg")
print(f"  Min step   : {min_step} s")
print(f"  Max step   : {max_step} s")
print(f"  Init step  : {init_step} s")
print(f"  Tolerances : computed via NumericalPropagator.tolerances(dP={dP} m)")
print(f"  Force models ({len(force_models)}):")
for fm in force_models:
    print(f"    - {fm.getClass().getSimpleName()}")
print()

# ──────────────────────────────────────────────
# 6.  Propagation loop  (uses orbital period)
# ──────────────────────────────────────────────
total_duration = int(orbital_period * num_orbits)   # seconds

times = []
x  = [];  y  = [];  z  = []
vx = [];  vy = [];  vz = []

for t in range(0, total_duration + 1, step_size):
    current_date = initial_date.shiftedBy(float(t))

    state = propagator.propagate(current_date)

    pv  = state.getPVCoordinates()
    pos = pv.getPosition()
    vel = pv.getVelocity()

    times.append(t)

    x.append(pos.getX()  / 1000)
    y.append(pos.getY()  / 1000)
    z.append(pos.getZ()  / 1000)

    vx.append(vel.getX() / 1000)
    vy.append(vel.getY() / 1000)
    vz.append(vel.getZ() / 1000)

print(f"Propagation complete: {len(times)} points over {total_duration} s ({num_orbits} orbit(s)).\n")

# ──────────────────────────────────────────────
# 7.  DataFrame  (UNCHANGED format)
# ──────────────────────────────────────────────
df = pd.DataFrame({
    "Time (s)":   times,
    "X (km)":     x,
    "Y (km)":     y,
    "Z (km)":     z,
    "VX (km/s)":  vx,
    "VY (km/s)":  vy,
    "VZ (km/s)":  vz
})

print("First 5 rows:")
print(df.head().to_string(index=True))
print()

# ──────────────────────────────────────────────
# 8.  CSV export  (UNCHANGED filename / format)
# ──────────────────────────────────────────────
CSV_FILE = "synthetic_orbit_dataset.csv"
df.to_csv(CSV_FILE, index=False)
print(f"Dataset saved to {CSV_FILE}  ({len(df)} rows)\n")

# ──────────────────────────────────────────────
# 9.  Orbit plot  (UNCHANGED from notebook)
# ──────────────────────────────────────────────
PLOT_FILE = "numerical_orbit_plot.png"

plt.figure(figsize=(8, 8))
plt.plot(df["X (km)"], df["Y (km)"])
plt.scatter(0, 0, s=250, label="Earth")
plt.xlabel("X (km)")
plt.ylabel("Y (km)")
plt.title("Numerical Propagator — Orbit (Central Gravity Only)")
plt.grid(True)
plt.axis("equal")
plt.legend()
plt.savefig(PLOT_FILE, dpi=150, bbox_inches="tight")
plt.close()
print(f"Orbit plot saved to {PLOT_FILE}")

print("\n✅ Module 4.1 complete — Numerical Propagator with central gravity.")
