"""
Module 4.4 — Solar Radiation Pressure (SRP)
=============================================

Extends Module 4.3 by adding Solar Radiation Pressure to the
existing J2 + Drag NumericalPropagator.

Architecture:
    KeplerianOrbit
        -> SpacecraftState (with mass)
        -> DormandPrince853Integrator
        -> NumericalPropagator
        -> Force Models:
             - HolmesFeatherstoneAttractionModel  (J2 gravity)
             - DragForce  (NRLMSISE00 atmosphere + IsotropicDrag)
             - SolarRadiationPressure  (Sun + IsotropicRadiationSingleCoefficient)
        -> propagate(time)
        -> PVCoordinates
        -> CSV + Plots (J2+Drag vs J2+Drag+SRP comparison)

New classes introduced in this module:
    IsotropicRadiationSingleCoefficient — spacecraft SRP cross-section & Cr
    SolarRadiationPressure              — Orekit SRP force model with eclipse

Usage:
    1. pip install orekit-jpype
    2. Place orekit-data-main.zip in the working directory
    3. python srp_propagator.py
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

# Gravity (from Module 4.2)
from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel
from org.orekit.forces.gravity.potential import GravityFieldFactory

# Drag (from Module 4.3)
from org.orekit.bodies import OneAxisEllipsoid, CelestialBodyFactory
from org.orekit.models.earth.atmosphere.data import CssiSpaceWeatherData
from org.orekit.models.earth.atmosphere import NRLMSISE00
from org.orekit.forces.drag import IsotropicDrag, DragForce

# SRP — new in Module 4.4
#
# IsotropicRadiationSingleCoefficient:
#   Simplest SRP spacecraft model.  Assumes the spacecraft is a sphere
#   with a single reflectivity coefficient Cr and a fixed cross-sectional
#   area.  Cr = 1.0 means fully absorptive, Cr = 2.0 means fully
#   reflective (specular mirror).  Typical LEO spacecraft use Cr ~ 1.2–1.8.
#   Constructor: IsotropicRadiationSingleCoefficient(crossSection_m2, Cr)
from org.orekit.forces.radiation import IsotropicRadiationSingleCoefficient

# SolarRadiationPressure:
#   Orekit force model that computes the acceleration from solar photon
#   pressure.  It takes the Sun (light source), the Earth's equatorial
#   radius (to compute eclipses — the model automatically determines
#   when the spacecraft is in Earth's shadow and zeroes the SRP force),
#   and a radiation-sensitive spacecraft model.
#   Constructor: SolarRadiationPressure(sun, earthRadius_m, spacecraftModel)
from org.orekit.forces.radiation import SolarRadiationPressure


# ══════════════════════════════════════════════
# CONFIGURATION — all tunable parameters in one place
# ══════════════════════════════════════════════
CONFIG = {
    # Spacecraft
    "mass":           1000.0,    # kg
    "cross_section":  10.0,      # m^2 — used for drag and SRP
    "drag_cd":        2.2,       # drag coefficient
    "srp_cr":         1.5,       # radiation pressure coefficient

    # Gravity field
    "gravity_degree": 20,        # spherical harmonics degree
    "gravity_order":  20,        # spherical harmonics order

    # Propagation
    "num_orbits":     100,       # number of full orbits to propagate
    "step_size":      60,        # output time step (s)

    # Integrator
    "min_step":       0.001,     # s
    "max_step":       1000.0,    # s
    "init_step":      60.0,      # s
    "dP":             1.0,       # desired position accuracy for tolerances (m)
}


# 4. Initial orbit (UNCHANGED)
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

# Derived quantities
orbital_period  = orbit.getKeplerianPeriod()
EARTH_RADIUS_KM = Constants.WGS84_EARTH_EQUATORIAL_RADIUS / 1000.0

print("Initial orbit defined:")
print(f"  a       = {orbit.getA():.1f} m")
print(f"  e       = {orbit.getE():.6f}")
print(f"  i       = {math.degrees(orbit.getI()):.2f} deg")
print(f"  mu      = {Constants.WGS84_EARTH_MU:.4e} m^3/s^2")
print(f"  Period  = {orbital_period:.2f} s  ({orbital_period/60:.2f} min)")
print()

print("Configuration:")
for k, v in CONFIG.items():
    print(f"  {k:20s} = {v}")
print()

 
# HELPER: build a propagator with given force models
def build_propagator(orbit, mass, force_models):
    initial_state = SpacecraftState(orbit, mass)

    tolerances = NumericalPropagator.tolerances(
        CONFIG["dP"], orbit, OrbitType.KEPLERIAN
    )

    integrator = DormandPrince853Integrator(
        CONFIG["min_step"], CONFIG["max_step"],
        tolerances[0], tolerances[1]
    )
    integrator.setInitialStepSize(CONFIG["init_step"])

    prop = NumericalPropagator(integrator)
    prop.setOrbitType(OrbitType.KEPLERIAN)
    prop.setInitialState(initial_state)

    for fm in force_models:
        prop.addForceModel(fm)

    return prop


# HELPER: propagate and collect position/velocity + altitude
def propagate_and_collect(propagator, initial_date, total_duration, step_size):
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

        r_km = math.sqrt(px**2 + py**2 + pz**2)
        alt.append(r_km - EARTH_RADIUS_KM)

    return times, x, y, z, vx, vy, vz, alt


# ══════════════════════════════════════════════
# 5. Build shared models
# ══════════════════════════════════════════════

# 5a. Earth body (WGS84 ellipsoid in ITRF)
body_frame = FramesFactory.getITRF(IERSConventions.IERS_2010, True)

earth = OneAxisEllipsoid(
    Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
    Constants.WGS84_EARTH_FLATTENING,
    body_frame
)
print("Earth ellipsoid created (WGS84)")

# 5b. Sun ephemeris — needed by both NRLMSISE00 and SRP
sun = CelestialBodyFactory.getSun()
print(f"Sun ephemeris loaded: {sun}")

# 5c. Space weather data for NRLMSISE00
cssi = CssiSpaceWeatherData(CssiSpaceWeatherData.DEFAULT_SUPPORTED_NAMES)
print("CSSI space weather data loaded")

# 5d. NRLMSISE00 atmosphere
atmosphere = NRLMSISE00(cssi, sun, earth)
print("NRLMSISE00 atmosphere model created")

# 5e. Gravity model (high-order, from CONFIG)
gravity_provider = GravityFieldFactory.getNormalizedProvider(
    CONFIG["gravity_degree"], CONFIG["gravity_order"]
)
gravity_model = HolmesFeatherstoneAttractionModel(body_frame, gravity_provider)
print(f"Gravity model: degree={CONFIG['gravity_degree']}, order={CONFIG['gravity_order']}")

# 5f. Drag force model
# IsotropicDrag: direction-independent Cd and area.
spacecraft_drag = IsotropicDrag(CONFIG["cross_section"], CONFIG["drag_cd"])
drag_force = DragForce(atmosphere, spacecraft_drag)
print(f"Drag force: Cd={CONFIG['drag_cd']}, A={CONFIG['cross_section']} m^2")

# 5g. SRP force model — NEW in this module
#
# Step 1: IsotropicRadiationSingleCoefficient
#   Defines how the spacecraft interacts with photons.
#   cross_section: area presented to the Sun (m^2)
#   Cr: radiation pressure coefficient
#       Cr = 1.0 → fully absorptive (momentum transfer = photon momentum)
#       Cr = 2.0 → fully specular reflective (momentum transfer = 2x)
#       Cr = 1.5 → typical mixed absorption/reflection
spacecraft_srp = IsotropicRadiationSingleCoefficient(
    CONFIG["cross_section"], CONFIG["srp_cr"]
)

# Step 2: SolarRadiationPressure
#   Combines:
#     - Sun position (light source direction and distance)
#     - Earth equatorial radius (for cylindrical eclipse/shadow model —
#       when the spacecraft is behind Earth relative to the Sun, SRP
#       acceleration is automatically set to zero)
#     - Spacecraft radiation model (area + Cr)
#
#   The SRP acceleration is:  a = (P_sun / c) * Cr * A / m
#   where P_sun ≈ 4.56e-6 N/m^2 at 1 AU, and the model scales it by
#   the actual Sun-spacecraft distance.
srp_force = SolarRadiationPressure(
    sun,
    Constants.WGS84_EARTH_EQUATORIAL_RADIUS,  # for eclipse computation
    spacecraft_srp
)
print(f"SRP force: Cr={CONFIG['srp_cr']}, A={CONFIG['cross_section']} m^2, eclipse=Earth shadow\n")


# ══════════════════════════════════════════════
# 6. Run 1: J2 + Drag (baseline from Module 4.3)
# ══════════════════════════════════════════════
print("=" * 50)
print("  Run 1: J2 + Drag (baseline)")
print("=" * 50)

j2_drag_forces = []
j2_drag_forces.append(gravity_model)  # spherical harmonics gravity (includes mu)
j2_drag_forces.append(drag_force)     # atmospheric drag (NRLMSISE00)

j2_drag_prop = build_propagator(orbit, CONFIG["mass"], j2_drag_forces)

total_duration = int(orbital_period * CONFIG["num_orbits"])

print(f"  Force models ({len(j2_drag_forces)}):")
for fm in j2_drag_forces:
    print(f"    - {fm.getClass().getSimpleName()}")

(times_jd, xjd, yjd, zjd, vxjd, vyjd, vzjd, alt_jd) = propagate_and_collect(
    j2_drag_prop, initial_date, total_duration, CONFIG["step_size"]
)
print(f"  J2+Drag propagation: {len(times_jd)} points\n")


# ══════════════════════════════════════════════
# 7. Run 2: J2 + Drag + SRP
# ══════════════════════════════════════════════
print("=" * 50)
print("  Run 2: J2 + Drag + SRP")
print("=" * 50)

# Append SRP to the existing J2 + Drag force model list
j2_drag_srp_forces = []
j2_drag_srp_forces.append(gravity_model)  # high-order gravity
j2_drag_srp_forces.append(drag_force)     # atmospheric drag
j2_drag_srp_forces.append(srp_force)      # solar radiation pressure

j2_drag_srp_prop = build_propagator(orbit, CONFIG["mass"], j2_drag_srp_forces)

print(f"  Force models ({len(j2_drag_srp_forces)}):")
for fm in j2_drag_srp_forces:
    print(f"    - {fm.getClass().getSimpleName()}")

(times_s, xs, ys, zs, vxs, vys, vzs, alt_s) = propagate_and_collect(
    j2_drag_srp_prop, initial_date, total_duration, CONFIG["step_size"]
)
print(f"  J2+Drag+SRP propagation: {len(times_s)} points\n")


# ══════════════════════════════════════════════
# 8. Build DataFrames
# ══════════════════════════════════════════════

# J2+Drag+SRP dataset (primary deliverable)
df_srp = pd.DataFrame({
    "Time (s)":      times_s,
    "X (km)":        xs,
    "Y (km)":        ys,
    "Z (km)":        zs,
    "VX (km/s)":     vxs,
    "VY (km/s)":     vys,
    "VZ (km/s)":     vzs,
    "Altitude (km)": alt_s
})

# J2+Drag baseline dataset
df_jd = pd.DataFrame({
    "Time (s)":      times_jd,
    "X (km)":        xjd,
    "Y (km)":        yjd,
    "Z (km)":        zjd,
    "VX (km/s)":     vxjd,
    "VY (km/s)":     vyjd,
    "VZ (km/s)":     vzjd,
    "Altitude (km)": alt_jd
})

print("J2+Drag+SRP — first 5 rows:")
print(df_srp.head().to_string(index=True))
print()


# ══════════════════════════════════════════════
# 9. Compute position difference (J2+Drag vs J2+Drag+SRP)
# ══════════════════════════════════════════════
dx = np.array(xs) - np.array(xjd)
dy = np.array(ys) - np.array(yjd)
dz = np.array(zs) - np.array(zjd)
pos_diff_km = np.sqrt(dx**2 + dy**2 + dz**2)

alt_diff_m = (np.array(alt_s) - np.array(alt_jd)) * 1000

print("Position difference (J2+Drag+SRP vs J2+Drag):")
print(f"  Max  : {pos_diff_km.max():.6f} km  ({pos_diff_km.max()*1000:.2f} m)")
print(f"  Mean : {pos_diff_km.mean():.6f} km  ({pos_diff_km.mean()*1000:.2f} m)")
print(f"  Final: {pos_diff_km[-1]:.6f} km  ({pos_diff_km[-1]*1000:.2f} m)")
print()
print("Altitude difference due to SRP:")
print(f"  Final alt (J2+Drag)     : {alt_jd[-1]:.4f} km")
print(f"  Final alt (J2+Drag+SRP) : {alt_s[-1]:.4f} km")
print(f"  Difference              : {alt_diff_m[-1]:.4f} m")
print()


# ══════════════════════════════════════════════
# 10. CSV export
# ══════════════════════════════════════════════
CSV_SRP     = "j2_drag_srp_orbit_dataset.csv"
CSV_JD_BASE = "j2_drag_baseline_orbit_dataset.csv"

df_srp.to_csv(CSV_SRP, index=False)
df_jd.to_csv(CSV_JD_BASE, index=False)

print(f"Saved {CSV_SRP}              ({len(df_srp)} rows)")
print(f"Saved {CSV_JD_BASE}    ({len(df_jd)} rows)")
print()


# ══════════════════════════════════════════════
# 11. Plots
# ══════════════════════════════════════════════

# 11a. Orbit comparison (X-Y plane)
PLOT_ORBIT = "j2_drag_srp_orbit_comparison.png"

fig, ax = plt.subplots(figsize=(9, 9))
ax.plot(df_jd["X (km)"], df_jd["Y (km)"],
        label="J2 + Drag", linewidth=1.2, alpha=0.7)
ax.plot(df_srp["X (km)"], df_srp["Y (km)"],
        label="J2 + Drag + SRP", linewidth=1.2, linestyle="--")
ax.scatter(0, 0, s=250, color="dodgerblue", zorder=5, label="Earth")
ax.set_xlabel("X (km)")
ax.set_ylabel("Y (km)")
ax.set_title("Orbit Comparison — J2+Drag vs J2+Drag+SRP")
ax.grid(True, alpha=0.3)
ax.axis("equal")
ax.legend()
fig.savefig(PLOT_ORBIT, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Orbit comparison plot saved to {PLOT_ORBIT}")


# 11b. Position difference over time
PLOT_DIFF = "j2_drag_srp_position_difference.png"

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(np.array(times_s) / 60.0, pos_diff_km * 1000,
        color="darkorchid", linewidth=1.5)
ax.set_xlabel("Time (min)")
ax.set_ylabel("Position Difference (m)")
ax.set_title("Position Difference: J2+Drag+SRP vs J2+Drag")
ax.grid(True, alpha=0.3)
fig.savefig(PLOT_DIFF, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Position difference plot saved to {PLOT_DIFF}")


# 11c. Altitude vs time
PLOT_ALT = "j2_drag_srp_altitude.png"

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(np.array(times_jd) / 60.0, alt_jd,
        label="J2 + Drag", linewidth=1.2, alpha=0.7)
ax.plot(np.array(times_s) / 60.0, alt_s,
        label="J2 + Drag + SRP", linewidth=1.2, linestyle="--", color="darkorchid")
ax.set_xlabel("Time (min)")
ax.set_ylabel("Altitude (km)")
ax.set_title("Altitude vs Time — Effect of SRP on Orbit")
ax.grid(True, alpha=0.3)
ax.legend()
fig.savefig(PLOT_ALT, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Altitude plot saved to {PLOT_ALT}")


# 11d. 3D orbit comparison
PLOT_3D = "j2_drag_srp_orbit_3d.png"

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")
ax.plot(df_jd["X (km)"], df_jd["Y (km)"], df_jd["Z (km)"],
        label="J2 + Drag", linewidth=1.0, alpha=0.6)
ax.plot(df_srp["X (km)"], df_srp["Y (km)"], df_srp["Z (km)"],
        label="J2 + Drag + SRP", linewidth=1.0, linestyle="--")
ax.scatter(0, 0, 0, s=200, color="dodgerblue", label="Earth")
ax.set_xlabel("X (km)")
ax.set_ylabel("Y (km)")
ax.set_zlabel("Z (km)")
ax.set_title("3D Orbit — J2+Drag vs J2+Drag+SRP")
ax.legend()
fig.savefig(PLOT_3D, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"3D orbit plot saved to {PLOT_3D}")


print("\n Module 4.4 complete — J2 + Drag + Solar Radiation Pressure.")
