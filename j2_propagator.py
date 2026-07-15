"""
Module 4.2 — J2 Perturbation (Spherical Harmonics Gravity)
===========================================================

Extends Module 4.1 by replacing NewtonianAttraction with a
HolmesFeatherstoneAttractionModel loaded from GravityFieldFactory.
This captures Earth's oblateness (J2) and optionally higher-order
terms via configurable degree/order.

Architecture:
    KeplerianOrbit
        -> SpacecraftState (with mass)
        -> DormandPrince853Integrator
        -> NumericalPropagator
        -> Force Models:
             - HolmesFeatherstoneAttractionModel (degree/order >= 2)
        -> propagate(time)
        -> PVCoordinates
        -> CSV + Plots (trajectory + comparison with central-gravity)

This module also runs a central-gravity-only propagation so the two
trajectories can be compared side by side.  The position difference
quantifies the J2 perturbation effect over one orbital period.

Usage:
    1. pip install orekit-jpype
    2. Place orekit-data-main.zip in the working directory
    3. python j2_propagator.py
"""

import sys
import os
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 1.  Orekit JVM + data bootstrap
import orekit_jpype as orekit
orekit.initVM()

from orekit_jpype.pyhelpers import setup_orekit_data

DATA_ZIP = "orekit-data-main.zip"
if os.path.exists(DATA_ZIP):
    setup_orekit_data(DATA_ZIP)
else:
    print(f"WARNING: {DATA_ZIP} not found in {os.getcwd()}")
    print("Orekit will try to use the default data loader.")


# 2.  Verify Orekit loaded correctly
from org.orekit.time import TimeScalesFactory

utc = TimeScalesFactory.getUTC()
print("UTC:", utc)
print("SUCCESS: Orekit data loaded correctly!\n")


# 3.  Imports
from org.orekit.frames import FramesFactory
from org.orekit.time import AbsoluteDate
from org.orekit.utils import Constants, IERSConventions
from org.orekit.orbits import KeplerianOrbit, PositionAngleType, OrbitType
from org.orekit.propagation import SpacecraftState
from org.orekit.propagation.numerical import NumericalPropagator
from org.hipparchus.ode.nonstiff import DormandPrince853Integrator
from org.orekit.forces.gravity import NewtonianAttraction
from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel
from org.orekit.forces.gravity.potential import GravityFieldFactory

# 4.  Initial orbit  (UNCHANGED from Module 4.1)
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

# Compute orbital period from the orbit
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

# Spacecraft mass (needed for drag/SRP in later modules)
SPACECRAFT_MASS = 1000.0   # kg



#  HELPER: build a propagator with given force models
def build_propagator(orbit, mass, force_models):
    initial_state = SpacecraftState(orbit, mass)

    # Integrator
    min_step  = 0.001
    max_step  = 1000.0
    init_step = 60.0
    dP = 1.0
    tolerances = NumericalPropagator.tolerances(dP, orbit, OrbitType.KEPLERIAN)

    integrator = DormandPrince853Integrator(
        min_step, max_step,
        tolerances[0], tolerances[1]
    )
    integrator.setInitialStepSize(init_step)

    # Propagator
    prop = NumericalPropagator(integrator)
    prop.setOrbitType(OrbitType.KEPLERIAN)
    prop.setInitialState(initial_state)

    for fm in force_models:
        prop.addForceModel(fm)

    return prop


#  HELPER: propagate and collect position/velocity
def propagate_and_collect(propagator, initial_date, total_duration, step_size):

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

    return times, x, y, z, vx, vy, vz

# 5.  Central-gravity propagation  (baseline)
print("=" * 50)
print("  Run 1: Central gravity only (baseline)")
print("=" * 50)

central_force_models = []
central_force_models.append(NewtonianAttraction(Constants.WGS84_EARTH_MU))

central_prop = build_propagator(orbit, SPACECRAFT_MASS, central_force_models)

total_duration = int(orbital_period * num_orbits)

(times_c, xc, yc, zc, vxc, vyc, vzc) = propagate_and_collect(
    central_prop, initial_date, total_duration, step_size
)
print(f"  Central-gravity propagation: {len(times_c)} points\n")


# 6.  J2-perturbed propagation
print("=" * 50)
print("  Run 2: J2 perturbation (spherical harmonics)")
print("=" * 50)

# Gravity field degree/order.
# degree=order=2 captures J2 (the dominant oblateness term).
# Increase for higher fidelity (e.g., 20x20 or 70x70).
GRAVITY_DEGREE = 2
GRAVITY_ORDER  = 0

# Load the gravity field provider from Orekit data
gravity_provider = GravityFieldFactory.getNormalizedProvider(
    GRAVITY_DEGREE, GRAVITY_ORDER
)

# Build the Holmes-Featherstone attraction model
# It needs a rotating body frame (ITRF) to compute the gravity field correctly.
body_frame = FramesFactory.getITRF(IERSConventions.IERS_2010, True)

j2_model = HolmesFeatherstoneAttractionModel(body_frame, gravity_provider)

print(f"  Gravity field : degree={GRAVITY_DEGREE}, order={GRAVITY_ORDER}")
print(f"  Body frame    : {body_frame.getName()}")
print(f"  Model         : HolmesFeatherstoneAttractionModel")
print()

# Force models for J2 run — the HolmesFeatherstone model
# inherently includes the central term (mu), so NewtonianAttraction
# is NOT needed separately.
j2_force_models = []
j2_force_models.append(j2_model)

j2_prop = build_propagator(orbit, SPACECRAFT_MASS, j2_force_models)

(times_j, xj, yj, zj, vxj, vyj, vzj) = propagate_and_collect(
    j2_prop, initial_date, total_duration, step_size
)
print(f"  J2-perturbed propagation: {len(times_j)} points\n")


# 7.  Build DataFrames
# J2-perturbed dataset  (the primary deliverable)
df_j2 = pd.DataFrame({
    "Time (s)":   times_j,
    "X (km)":     xj,
    "Y (km)":     yj,
    "Z (km)":     zj,
    "VX (km/s)":  vxj,
    "VY (km/s)":  vyj,
    "VZ (km/s)":  vzj
})

# Central-gravity dataset  (for comparison)
df_central = pd.DataFrame({
    "Time (s)":   times_c,
    "X (km)":     xc,
    "Y (km)":     yc,
    "Z (km)":     zc,
    "VX (km/s)":  vxc,
    "VY (km/s)":  vyc,
    "VZ (km/s)":  vzc
})

print("J2-perturbed — first 5 rows:")
print(df_j2.head().to_string(index=True))
print()


# 8.  Compute position difference (comparison)
dx = np.array(xj) - np.array(xc)
dy = np.array(yj) - np.array(yc)
dz = np.array(zj) - np.array(zc)
pos_diff_km = np.sqrt(dx**2 + dy**2 + dz**2)

print(f"Position difference (J2 vs central gravity):")
print(f"  Max  : {pos_diff_km.max():.6f} km  ({pos_diff_km.max()*1000:.2f} m)")
print(f"  Mean : {pos_diff_km.mean():.6f} km  ({pos_diff_km.mean()*1000:.2f} m)")
print(f"  Final: {pos_diff_km[-1]:.6f} km  ({pos_diff_km[-1]*1000:.2f} m)")
print()


# 9.  CSV export
CSV_J2      = "j2_orbit_dataset.csv"
CSV_CENTRAL = "central_gravity_orbit_dataset.csv"

df_j2.to_csv(CSV_J2, index=False)
df_central.to_csv(CSV_CENTRAL, index=False)

print(f"Saved {CSV_J2}       ({len(df_j2)} rows)")
print(f"Saved {CSV_CENTRAL}  ({len(df_central)} rows)")
print()


# 10.  Plots
# Orbit comparison (X-Y plane)
PLOT_ORBIT = "j2_orbit_comparison.png"

fig, ax = plt.subplots(figsize=(9, 9))
ax.plot(df_central["X (km)"], df_central["Y (km)"],
        label="Central Gravity", linewidth=1.2, alpha=0.7)
ax.plot(df_j2["X (km)"], df_j2["Y (km)"],
        label=f"J2 (deg={GRAVITY_DEGREE}, ord={GRAVITY_ORDER})",
        linewidth=1.2, linestyle="--")
ax.scatter(0, 0, s=250, color="dodgerblue", zorder=5, label="Earth")
ax.set_xlabel("X (km)")
ax.set_ylabel("Y (km)")
ax.set_title("Orbit Comparison — Central Gravity vs J2 Perturbation")
ax.grid(True, alpha=0.3)
ax.axis("equal")
ax.legend()
fig.savefig(PLOT_ORBIT, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Orbit comparison plot saved to {PLOT_ORBIT}")


#Position difference over time
PLOT_DIFF = "j2_position_difference.png"

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(np.array(times_j) / 60.0, pos_diff_km * 1000,
        color="crimson", linewidth=1.5)
ax.set_xlabel("Time (min)")
ax.set_ylabel("Position Difference (m)")
ax.set_title("Position Difference: J2 Perturbed vs Central Gravity")
ax.grid(True, alpha=0.3)
fig.savefig(PLOT_DIFF, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Position difference plot saved to {PLOT_DIFF}")


# 3D orbit comparison  
PLOT_3D = "j2_orbit_3d.png"

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")
ax.plot(df_central["X (km)"], df_central["Y (km)"], df_central["Z (km)"],
        label="Central Gravity", linewidth=1.0, alpha=0.6)
ax.plot(df_j2["X (km)"], df_j2["Y (km)"], df_j2["Z (km)"],
        label=f"J2 (deg={GRAVITY_DEGREE})", linewidth=1.0, linestyle="--")
ax.scatter(0, 0, 0, s=200, color="dodgerblue", label="Earth")
ax.set_xlabel("X (km)")
ax.set_ylabel("Y (km)")
ax.set_zlabel("Z (km)")
ax.set_title("3D Orbit — Central Gravity vs J2")
ax.legend()
fig.savefig(PLOT_3D, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"3D orbit plot saved to {PLOT_3D}")

print("\n Module 4.2 complete — J2 perturbation with spherical harmonics gravity.")
