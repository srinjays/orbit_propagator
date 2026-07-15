"""
Module 4.3 — Atmospheric Drag
==============================

Extends Module 4.2 by adding atmospheric drag to the J2-perturbed
NumericalPropagator.

Architecture:
    KeplerianOrbit
        -> SpacecraftState (with mass)
        -> DormandPrince853Integrator
        -> NumericalPropagator
        -> Force Models:
             - HolmesFeatherstoneAttractionModel  (J2 gravity)
             - DragForce  (NRLMSISE00 atmosphere + IsotropicDrag)
        -> propagate(time)
        -> PVCoordinates
        -> CSV + Plots (J2-only vs J2+Drag comparison + altitude decay)

New classes introduced in this module:
    OneAxisEllipsoid          — Earth shape model (WGS84 ellipsoid)
    CssiSpaceWeatherData      — F10.7 / Ap indices for NRLMSISE00
    NRLMSISE00                — empirical thermosphere density model
    IsotropicDrag             — spacecraft drag cross-section & Cd
    DragForce                 — Orekit force model wrapping atmo + drag

Usage:
    1. pip install orekit-jpype
    2. Place orekit-data-main.zip in the working directory
       (must contain SpaceWeather-All-v1.2.txt for NRLMSISE00)
    3. python drag_propagator.py
"""

import sys
import os
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 1. Orekit JVM + data bootstrap
import orekit_jpype as orekit
orekit.initVM()

from orekit_jpype.pyhelpers import setup_orekit_data

DATA_ZIP = "orekit-data-main.zip"
if os.path.exists(DATA_ZIP):
    setup_orekit_data(DATA_ZIP)
else:
    print(f"WARNING: {DATA_ZIP} not found in {os.getcwd()}")
    print("Orekit will try to use the default data loader.")


# 2. Verify Orekit loaded correctly
from org.orekit.time import TimeScalesFactory

utc = TimeScalesFactory.getUTC()
print("UTC:", utc)
print("SUCCESS: Orekit data loaded correctly!\n")


# 3. Imports
from org.orekit.frames import FramesFactory
from org.orekit.time import AbsoluteDate
from org.orekit.utils import Constants, IERSConventions
from org.orekit.orbits import KeplerianOrbit, PositionAngleType, OrbitType
from org.orekit.propagation import SpacecraftState
from org.orekit.propagation.numerical import NumericalPropagator
from org.hipparchus.ode.nonstiff import DormandPrince853Integrator

# Gravity
from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel
from org.orekit.forces.gravity.potential import GravityFieldFactory

# Drag — new in Module 4.3
# OneAxisEllipsoid:  Defines Earth as a WGS84 ellipsoid.  The atmosphere
#                    model needs this to convert satellite position to
#                    geodetic altitude, which drives density lookup.
from org.orekit.bodies import OneAxisEllipsoid

# CelestialBodyFactory: Provides the Sun ephemeris.  NRLMSISE00 needs
#                       the Sun position to compute Extreme UV heating
#                       of the upper atmosphere.
from org.orekit.bodies import CelestialBodyFactory

# CssiSpaceWeatherData:  Provides the F10.7 solar radio flux and Ap
#                         geomagnetic indices that NRLMSISE00 requires.
#                         Loaded from SpaceWeather-All-v1.2.txt inside
#                         the orekit-data zip.
from org.orekit.models.earth.atmosphere.data import CssiSpaceWeatherData

# NRLMSISE00:  Empirical thermosphere/exosphere density model.
#              Given a date, position, and space weather inputs it
#              returns the local atmospheric density (kg/m^3).
#              This is the standard model for LEO drag computation.
from org.orekit.models.earth.atmosphere import NRLMSISE00

# IsotropicDrag:  Simplest drag model — assumes the spacecraft presents
#                 the same cross-sectional area and Cd in every direction.
#                 Parameters: cross-section area (m^2), drag coefficient Cd.
from org.orekit.forces.drag import IsotropicDrag

# DragForce:  Orekit force model that combines an atmosphere model and
#             a drag-sensitive spacecraft model.  At each integration
#             step it queries the atmosphere for local density, then
#             computes the drag acceleration: a = -0.5 * rho * Cd * A/m * v^2.
from org.orekit.forces.drag import DragForce


# 4. Initial orbit (UNCHANGED from Modules 4.1 / 4.2)
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
num_orbits     = 100                            # number of full orbits to propagate
step_size      = 60                           # output time step (s)

# Earth equatorial radius — used to compute altitude from position magnitude
EARTH_RADIUS_KM = Constants.WGS84_EARTH_EQUATORIAL_RADIUS / 1000.0

print("Initial orbit defined:")
print(f"  a       = {orbit.getA():.1f} m")
print(f"  e       = {orbit.getE():.6f}")
print(f"  i       = {math.degrees(orbit.getI()):.2f} deg")
print(f"  mu      = {Constants.WGS84_EARTH_MU:.4e} m^3/s^2")
print(f"  Period  = {orbital_period:.2f} s  ({orbital_period/60:.2f} min)")
print()

# Spacecraft properties
SPACECRAFT_MASS = 1000.0   # kg — used by SpacecraftState and drag
CROSS_SECTION   = 10.0     # m^2 — projected area for drag
DRAG_CD         = 2.2      # drag coefficient (typical for LEO spacecraft)


# ══════════════════════════════════════════════
# HELPER: build a propagator with given force models
# ══════════════════════════════════════════════
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


# ══════════════════════════════════════════════
# HELPER: propagate and collect position/velocity + altitude
# ══════════════════════════════════════════════
def propagate_and_collect(propagator, initial_date, total_duration, step_size):
    """Returns (times, x, y, z, vx, vy, vz, altitudes_km)."""
    times = []
    x  = [];  y  = [];  z  = []
    vx = [];  vy = [];  vz = []
    alt = []

    for t in range(0, total_duration + 1, step_size):
        current_date = initial_date.shiftedBy(float(t))
        state = propagator.propagate(current_date)

        pv  = state.getPVCoordinates()
        pos = pv.getPosition()
        vel = pv.getVelocity()

        times.append(t)
        px = pos.getX() / 1000
        py = pos.getY() / 1000
        pz = pos.getZ() / 1000
        x.append(px);  y.append(py);  z.append(pz)
        vx.append(vel.getX() / 1000)
        vy.append(vel.getY() / 1000)
        vz.append(vel.getZ() / 1000)

        # Altitude = distance from Earth center - equatorial radius
        r_km = math.sqrt(px**2 + py**2 + pz**2)
        alt.append(r_km - EARTH_RADIUS_KM)

    return times, x, y, z, vx, vy, vz, alt


# ══════════════════════════════════════════════
# 5.  Build shared models (Earth shape, atmosphere)
# ══════════════════════════════════════════════

# 5a. Earth body (ITRF rotating frame + WGS84 ellipsoid)
#     OneAxisEllipsoid models the Earth as an oblate spheroid.
#     The ITRF frame rotates with the Earth — this is critical
#     for both the gravity field and the atmosphere model.
body_frame = FramesFactory.getITRF(IERSConventions.IERS_2010, True)

earth = OneAxisEllipsoid(
    Constants.WGS84_EARTH_EQUATORIAL_RADIUS,   # equatorial radius (m)
    Constants.WGS84_EARTH_FLATTENING,           # flattening
    body_frame                                   # body-fixed frame
)
print("Earth ellipsoid created (WGS84)")

# 5b. Sun ephemeris — NRLMSISE00 needs the Sun's position to model
#     solar EUV heating of the thermosphere.
sun = CelestialBodyFactory.getSun()
print(f"Sun ephemeris loaded: {sun}")

# 5c. Space weather data — provides F10.7 solar flux and Ap
#     geomagnetic indices.  CssiSpaceWeatherData reads from the
#     SpaceWeather-All-v1.2.txt file bundled in orekit-data.
cssi = CssiSpaceWeatherData(CssiSpaceWeatherData.DEFAULT_SUPPORTED_NAMES)
print("CSSI space weather data loaded")

# 5d. NRLMSISE00 atmosphere model
#     Combines the space weather data, Sun position, and Earth shape
#     to compute thermospheric density at any (date, position).
atmosphere = NRLMSISE00(cssi, sun, earth)
print("NRLMSISE00 atmosphere model created\n")

# 5e. J2 gravity model (reused from Module 4.2)
GRAVITY_DEGREE = 2
GRAVITY_ORDER  = 2

gravity_provider = GravityFieldFactory.getNormalizedProvider(
    GRAVITY_DEGREE, GRAVITY_ORDER
)
j2_model = HolmesFeatherstoneAttractionModel(body_frame, gravity_provider)
print(f"J2 gravity model: degree={GRAVITY_DEGREE}, order={GRAVITY_ORDER}")

# 5f. Drag model
#     IsotropicDrag assumes a constant cross-section and Cd regardless
#     of spacecraft attitude.  This is the simplest drag model and is
#     appropriate when attitude is not modeled.
spacecraft_drag = IsotropicDrag(CROSS_SECTION, DRAG_CD)

#     DragForce ties the atmosphere and the spacecraft drag model
#     together into a single Orekit force model.
drag_force = DragForce(atmosphere, spacecraft_drag)
print(f"Drag force: Cd={DRAG_CD}, A={CROSS_SECTION} m^2, atmosphere=NRLMSISE00\n")


# ══════════════════════════════════════════════
# 6.  Run 1: J2-only propagation (baseline)
# ══════════════════════════════════════════════
print("=" * 50)
print("  Run 1: J2 gravity only (baseline)")
print("=" * 50)

# HolmesFeatherstone includes the central term (mu), so
# NewtonianAttraction is NOT needed.
j2_force_models = []
j2_force_models.append(j2_model)

j2_prop = build_propagator(orbit, SPACECRAFT_MASS, j2_force_models)

total_duration = int(orbital_period * num_orbits)

(times_j, xj, yj, zj, vxj, vyj, vzj, alt_j) = propagate_and_collect(
    j2_prop, initial_date, total_duration, step_size
)
print(f"  J2-only propagation: {len(times_j)} points\n")


# ══════════════════════════════════════════════
# 7.  Run 2: J2 + Drag propagation
# ══════════════════════════════════════════════
print("=" * 50)
print("  Run 2: J2 + Atmospheric Drag")
print("=" * 50)

# Force models list — append both J2 and drag
j2_drag_force_models = []
j2_drag_force_models.append(j2_model)      # spherical harmonics gravity
j2_drag_force_models.append(drag_force)     # atmospheric drag (NRLMSISE00)

j2_drag_prop = build_propagator(orbit, SPACECRAFT_MASS, j2_drag_force_models)

print(f"  Force models ({len(j2_drag_force_models)}):")
for fm in j2_drag_force_models:
    print(f"    - {fm.getClass().getSimpleName()}")
print()

(times_d, xd, yd, zd, vxd, vyd, vzd, alt_d) = propagate_and_collect(
    j2_drag_prop, initial_date, total_duration, step_size
)
print(f"  J2+Drag propagation: {len(times_d)} points\n")


# ══════════════════════════════════════════════
# 8.  Build DataFrames
# ══════════════════════════════════════════════

# J2+Drag dataset (primary deliverable)
df_drag = pd.DataFrame({
    "Time (s)":   times_d,
    "X (km)":     xd,
    "Y (km)":     yd,
    "Z (km)":     zd,
    "VX (km/s)":  vxd,
    "VY (km/s)":  vyd,
    "VZ (km/s)":  vzd,
    "Altitude (km)": alt_d
})

# J2-only dataset (for comparison)
df_j2only = pd.DataFrame({
    "Time (s)":   times_j,
    "X (km)":     xj,
    "Y (km)":     yj,
    "Z (km)":     zj,
    "VX (km/s)":  vxj,
    "VY (km/s)":  vyj,
    "VZ (km/s)":  vzj,
    "Altitude (km)": alt_j
})

print("J2+Drag — first 5 rows:")
print(df_drag.head().to_string(index=True))
print()


# ══════════════════════════════════════════════
# 9.  Compute position difference (J2-only vs J2+Drag)
# ══════════════════════════════════════════════
dx = np.array(xd) - np.array(xj)
dy = np.array(yd) - np.array(yj)
dz = np.array(zd) - np.array(zj)
pos_diff_km = np.sqrt(dx**2 + dy**2 + dz**2)

# Altitude difference shows orbital decay
alt_diff_m = (np.array(alt_d) - np.array(alt_j)) * 1000  # in meters

print("Position difference (J2+Drag vs J2-only):")
print(f"  Max  : {pos_diff_km.max():.6f} km  ({pos_diff_km.max()*1000:.2f} m)")
print(f"  Mean : {pos_diff_km.mean():.6f} km  ({pos_diff_km.mean()*1000:.2f} m)")
print(f"  Final: {pos_diff_km[-1]:.6f} km  ({pos_diff_km[-1]*1000:.2f} m)")
print()
print("Altitude change due to drag:")
print(f"  Final alt (J2-only): {alt_j[-1]:.4f} km")
print(f"  Final alt (J2+Drag): {alt_d[-1]:.4f} km")
print(f"  Difference         : {alt_diff_m[-1]:.4f} m")
print()


# ══════════════════════════════════════════════
# 10.  CSV export
# ══════════════════════════════════════════════
CSV_DRAG   = "j2_drag_orbit_dataset.csv"
CSV_J2ONLY = "j2_only_orbit_dataset.csv"

df_drag.to_csv(CSV_DRAG, index=False)
df_j2only.to_csv(CSV_J2ONLY, index=False)

print(f"Saved {CSV_DRAG}      ({len(df_drag)} rows)")
print(f"Saved {CSV_J2ONLY}    ({len(df_j2only)} rows)")
print()


# ══════════════════════════════════════════════
# 11.  Plots
# ══════════════════════════════════════════════

# 11a. Orbit comparison (X-Y plane)
PLOT_ORBIT = "j2_drag_orbit_comparison.png"

fig, ax = plt.subplots(figsize=(9, 9))
ax.plot(df_j2only["X (km)"], df_j2only["Y (km)"],
        label="J2 Only", linewidth=1.2, alpha=0.7)
ax.plot(df_drag["X (km)"], df_drag["Y (km)"],
        label="J2 + Drag", linewidth=1.2, linestyle="--")
ax.scatter(0, 0, s=250, color="dodgerblue", zorder=5, label="Earth")
ax.set_xlabel("X (km)")
ax.set_ylabel("Y (km)")
ax.set_title("Orbit Comparison — J2 Only vs J2 + Atmospheric Drag")
ax.grid(True, alpha=0.3)
ax.axis("equal")
ax.legend()
fig.savefig(PLOT_ORBIT, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Orbit comparison plot saved to {PLOT_ORBIT}")


# 11b. Position difference over time
PLOT_DIFF = "j2_drag_position_difference.png"

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(np.array(times_d) / 60.0, pos_diff_km * 1000,
        color="crimson", linewidth=1.5)
ax.set_xlabel("Time (min)")
ax.set_ylabel("Position Difference (m)")
ax.set_title("Position Difference: J2+Drag vs J2 Only")
ax.grid(True, alpha=0.3)
fig.savefig(PLOT_DIFF, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Position difference plot saved to {PLOT_DIFF}")


# 11c. Altitude vs time (orbital decay plot)
#      This is the signature effect of atmospheric drag — the orbit
#      slowly loses energy and altitude drops over time.
PLOT_ALT = "j2_drag_altitude_decay.png"

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(np.array(times_j) / 60.0, alt_j,
        label="J2 Only", linewidth=1.2, alpha=0.7)
ax.plot(np.array(times_d) / 60.0, alt_d,
        label="J2 + Drag", linewidth=1.2, linestyle="--", color="orangered")
ax.set_xlabel("Time (min)")
ax.set_ylabel("Altitude (km)")
ax.set_title("Altitude vs Time — Orbital Decay from Atmospheric Drag")
ax.grid(True, alpha=0.3)
ax.legend()
fig.savefig(PLOT_ALT, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Altitude decay plot saved to {PLOT_ALT}")


# 11d. 3D orbit comparison
PLOT_3D = "j2_drag_orbit_3d.png"

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")
ax.plot(df_j2only["X (km)"], df_j2only["Y (km)"], df_j2only["Z (km)"],
        label="J2 Only", linewidth=1.0, alpha=0.6)
ax.plot(df_drag["X (km)"], df_drag["Y (km)"], df_drag["Z (km)"],
        label="J2 + Drag", linewidth=1.0, linestyle="--")
ax.scatter(0, 0, 0, s=200, color="dodgerblue", label="Earth")
ax.set_xlabel("X (km)")
ax.set_ylabel("Y (km)")
ax.set_zlabel("Z (km)")
ax.set_title("3D Orbit — J2 Only vs J2 + Drag")
ax.legend()
fig.savefig(PLOT_3D, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"3D orbit plot saved to {PLOT_3D}")


print("\n Module 4.3 complete — J2 + Atmospheric Drag (NRLMSISE00).")
