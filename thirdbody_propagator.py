"""
Module 4.5 — Third-Body Gravity (Sun + Moon)
==============================================

Extends Module 4.4 by adding third-body gravitational perturbations
from the Sun and Moon.  This is the "full physics" propagator that
includes all force models developed across Modules 4.1–4.5:

    - High-order spherical harmonics gravity  (HolmesFeatherstone, 20x20)
    - Atmospheric drag                        (NRLMSISE00 + IsotropicDrag)
    - Solar radiation pressure                (SRP with eclipse)
    - Third-body gravity                      (Sun + Moon)

Architecture:
    KeplerianOrbit
        -> SpacecraftState (with mass)
        -> DormandPrince853Integrator
        -> NumericalPropagator
        -> Force Models:
             - HolmesFeatherstoneAttractionModel  (gravity field)
             - DragForce                           (atmosphere)
             - SolarRadiationPressure              (photon pressure)
             - ThirdBodyAttraction (Sun)            (solar gravity)
             - ThirdBodyAttraction (Moon)           (lunar gravity)
        -> propagate(time)
        -> PVCoordinates
        -> CSV + Plots

New class introduced in this module:
    ThirdBodyAttraction — gravitational pull of a remote celestial body
                          on the spacecraft.  Uses point-mass gravity
                          from JPL ephemerides.

Usage:
    1. pip install orekit-jpype
    2. Place orekit-data-main.zip in the working directory
    3. python thirdbody_propagator.py
"""

import sys
import os
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    "num_orbits":     100,         # number of full orbits to propagate
    "step_size":      60,        # output time step (s)

    # Integrator
    "min_step":       0.001,     # s
    "max_step":       1000.0,    # s
    "init_step":      60.0,      # s
    "dP":             1.0,       # desired position accuracy for tolerances (m)
}


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

# Gravity (Module 4.2)
from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel
from org.orekit.forces.gravity.potential import GravityFieldFactory

# Drag (Module 4.3)
from org.orekit.bodies import OneAxisEllipsoid, CelestialBodyFactory
from org.orekit.models.earth.atmosphere.data import CssiSpaceWeatherData
from org.orekit.models.earth.atmosphere import NRLMSISE00
from org.orekit.forces.drag import IsotropicDrag, DragForce

# SRP (Module 4.4)
from org.orekit.forces.radiation import IsotropicRadiationSingleCoefficient
from org.orekit.forces.radiation import SolarRadiationPressure

# Third-body gravity — NEW in Module 4.5
#
# ThirdBodyAttraction:
#   Computes the gravitational acceleration on the spacecraft from a
#   remote celestial body (Sun, Moon, or any body available via
#   CelestialBodyFactory).  Uses point-mass gravity — the body is
#   treated as a point with mass mu, and its position is read from
#   JPL DE ephemerides bundled in orekit-data.
#
#   The perturbation is computed as the DIFFERENCE between:
#     (1) the third body's pull on the spacecraft, and
#     (2) the third body's pull on Earth's center
#   because the propagation is Earth-centered.  This "indirect" term
#   accounts for the fact that Earth itself is accelerated by the
#   third body, and only the tidal (differential) acceleration
#   matters for relative motion.
#
#   Constructor: ThirdBodyAttraction(celestialBody)
from org.orekit.forces.gravity import ThirdBodyAttraction


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

# 5b. Sun and Moon ephemerides
#     CelestialBodyFactory reads JPL DE ephemerides from orekit-data.
#     The Sun is needed for NRLMSISE00, SRP, and third-body gravity.
#     The Moon is needed for third-body gravity.
sun  = CelestialBodyFactory.getSun()
moon = CelestialBodyFactory.getMoon()
print(f"Sun  ephemeris loaded: {sun}")
print(f"Moon ephemeris loaded: {moon}")

# 5c. Space weather data + atmosphere
cssi = CssiSpaceWeatherData(CssiSpaceWeatherData.DEFAULT_SUPPORTED_NAMES)
atmosphere = NRLMSISE00(cssi, sun, earth)
print("NRLMSISE00 atmosphere + CSSI space weather loaded")

# 5d. Gravity field (high-order, from CONFIG)
gravity_provider = GravityFieldFactory.getNormalizedProvider(
    CONFIG["gravity_degree"], CONFIG["gravity_order"]
)
gravity_model = HolmesFeatherstoneAttractionModel(body_frame, gravity_provider)
print(f"Gravity model: degree={CONFIG['gravity_degree']}, order={CONFIG['gravity_order']}")

# 5e. Drag force
spacecraft_drag = IsotropicDrag(CONFIG["cross_section"], CONFIG["drag_cd"])
drag_force = DragForce(atmosphere, spacecraft_drag)
print(f"Drag force: Cd={CONFIG['drag_cd']}, A={CONFIG['cross_section']} m^2")

# 5f. SRP force
spacecraft_srp = IsotropicRadiationSingleCoefficient(
    CONFIG["cross_section"], CONFIG["srp_cr"]
)
srp_force = SolarRadiationPressure(
    sun,
    Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
    spacecraft_srp
)
print(f"SRP force: Cr={CONFIG['srp_cr']}, A={CONFIG['cross_section']} m^2")

# 5g. Third-body gravity — NEW
#
#     ThirdBodyAttraction(Sun):
#       The Sun's gravitational parameter is ~1.327e20 m^3/s^2.
#       At 1 AU distance, its tidal acceleration on a LEO satellite
#       is small (~5e-7 m/s^2) but accumulates over many orbits,
#       especially affecting the eccentricity and argument of perigee.
#
#     ThirdBodyAttraction(Moon):
#       The Moon's gravitational parameter is ~4.903e12 m^3/s^2.
#       At ~384,400 km, its tidal acceleration is ~1.1e-6 m/s^2 —
#       roughly twice the Sun's contribution.  It primarily perturbs
#       inclination, RAAN, and eccentricity over long time scales.
sun_gravity  = ThirdBodyAttraction(sun)
moon_gravity = ThirdBodyAttraction(moon)
print("Third-body gravity: Sun + Moon")
print()


# ══════════════════════════════════════════════
# 6. Run 1: J2 + Drag + SRP (baseline from Module 4.4)
# ══════════════════════════════════════════════
print("=" * 55)
print("  Run 1: Gravity + Drag + SRP (no third-body)")
print("=" * 55)

baseline_forces = []
baseline_forces.append(gravity_model)   # spherical harmonics (includes mu)
baseline_forces.append(drag_force)      # atmospheric drag
baseline_forces.append(srp_force)       # solar radiation pressure

baseline_prop = build_propagator(orbit, CONFIG["mass"], baseline_forces)

total_duration = int(orbital_period * CONFIG["num_orbits"])

print(f"  Force models ({len(baseline_forces)}):")
for fm in baseline_forces:
    print(f"    - {fm.getClass().getSimpleName()}")

(times_b, xb, yb, zb, vxb, vyb, vzb, alt_b) = propagate_and_collect(
    baseline_prop, initial_date, total_duration, CONFIG["step_size"]
)
print(f"  Baseline propagation: {len(times_b)} points\n")


# ══════════════════════════════════════════════
# 7. Run 2: Full physics (+ Sun + Moon third-body)
# ══════════════════════════════════════════════
print("=" * 55)
print("  Run 2: Full physics (+ Sun & Moon gravity)")
print("=" * 55)

full_forces = []
full_forces.append(gravity_model)       # spherical harmonics gravity
full_forces.append(drag_force)          # atmospheric drag
full_forces.append(srp_force)           # solar radiation pressure
full_forces.append(sun_gravity)         # Sun third-body gravity
full_forces.append(moon_gravity)        # Moon third-body gravity

full_prop = build_propagator(orbit, CONFIG["mass"], full_forces)

print(f"  Force models ({len(full_forces)}):")
for fm in full_forces:
    print(f"    - {fm.getClass().getSimpleName()}")

(times_f, xf, yf, zf, vxf, vyf, vzf, alt_f) = propagate_and_collect(
    full_prop, initial_date, total_duration, CONFIG["step_size"]
)
print(f"  Full-physics propagation: {len(times_f)} points\n")


# ══════════════════════════════════════════════
# 8. Build DataFrames
# ══════════════════════════════════════════════

# Full-physics dataset (primary deliverable)
df_full = pd.DataFrame({
    "Time (s)":      times_f,
    "X (km)":        xf,
    "Y (km)":        yf,
    "Z (km)":        zf,
    "VX (km/s)":     vxf,
    "VY (km/s)":     vyf,
    "VZ (km/s)":     vzf,
    "Altitude (km)": alt_f
})

# Baseline dataset (no third-body)
df_base = pd.DataFrame({
    "Time (s)":      times_b,
    "X (km)":        xb,
    "Y (km)":        yb,
    "Z (km)":        zb,
    "VX (km/s)":     vxb,
    "VY (km/s)":     vyb,
    "VZ (km/s)":     vzb,
    "Altitude (km)": alt_b
})

print("Full-physics — first 5 rows:")
print(df_full.head().to_string(index=True))
print()


# ══════════════════════════════════════════════
# 9. Compute position difference
# ══════════════════════════════════════════════
dx = np.array(xf) - np.array(xb)
dy = np.array(yf) - np.array(yb)
dz = np.array(zf) - np.array(zb)
pos_diff_km = np.sqrt(dx**2 + dy**2 + dz**2)

alt_diff_m = (np.array(alt_f) - np.array(alt_b)) * 1000

print("Position difference (Full physics vs baseline without third-body):")
print(f"  Max  : {pos_diff_km.max():.6f} km  ({pos_diff_km.max()*1000:.2f} m)")
print(f"  Mean : {pos_diff_km.mean():.6f} km  ({pos_diff_km.mean()*1000:.2f} m)")
print(f"  Final: {pos_diff_km[-1]:.6f} km  ({pos_diff_km[-1]*1000:.2f} m)")
print()
print("Altitude difference due to third-body gravity:")
print(f"  Final alt (baseline)     : {alt_b[-1]:.4f} km")
print(f"  Final alt (full physics) : {alt_f[-1]:.4f} km")
print(f"  Difference               : {alt_diff_m[-1]:.4f} m")
print()


# ══════════════════════════════════════════════
# 10. CSV export
# ══════════════════════════════════════════════
CSV_FULL     = "full_physics_orbit_dataset.csv"
CSV_BASELINE = "no_thirdbody_baseline_orbit_dataset.csv"

df_full.to_csv(CSV_FULL, index=False)
df_base.to_csv(CSV_BASELINE, index=False)

print(f"Saved {CSV_FULL}                ({len(df_full)} rows)")
print(f"Saved {CSV_BASELINE}   ({len(df_base)} rows)")
print()


# ══════════════════════════════════════════════
# 11. Plots
# ══════════════════════════════════════════════

# 11a. Orbit comparison (X-Y plane)
PLOT_ORBIT = "full_physics_orbit_comparison.png"

fig, ax = plt.subplots(figsize=(9, 9))
ax.plot(df_base["X (km)"], df_base["Y (km)"],
        label="Without third-body", linewidth=1.2, alpha=0.7)
ax.plot(df_full["X (km)"], df_full["Y (km)"],
        label="Full physics (+ Sun & Moon)", linewidth=1.2, linestyle="--")
ax.scatter(0, 0, s=250, color="dodgerblue", zorder=5, label="Earth")
ax.set_xlabel("X (km)")
ax.set_ylabel("Y (km)")
ax.set_title("Orbit Comparison — With vs Without Third-Body Gravity")
ax.grid(True, alpha=0.3)
ax.axis("equal")
ax.legend()
fig.savefig(PLOT_ORBIT, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Orbit comparison plot saved to {PLOT_ORBIT}")


# 11b. Position difference over time
PLOT_DIFF = "full_physics_position_difference.png"

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(np.array(times_f) / 60.0, pos_diff_km * 1000,
        color="teal", linewidth=1.5)
ax.set_xlabel("Time (min)")
ax.set_ylabel("Position Difference (m)")
ax.set_title("Position Difference: Full Physics vs No Third-Body")
ax.grid(True, alpha=0.3)
fig.savefig(PLOT_DIFF, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Position difference plot saved to {PLOT_DIFF}")


# 11c. Altitude vs time
PLOT_ALT = "full_physics_altitude.png"

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(np.array(times_b) / 60.0, alt_b,
        label="Without third-body", linewidth=1.2, alpha=0.7)
ax.plot(np.array(times_f) / 60.0, alt_f,
        label="Full physics", linewidth=1.2, linestyle="--", color="teal")
ax.set_xlabel("Time (min)")
ax.set_ylabel("Altitude (km)")
ax.set_title("Altitude vs Time — Effect of Sun & Moon Gravity")
ax.grid(True, alpha=0.3)
ax.legend()
fig.savefig(PLOT_ALT, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Altitude plot saved to {PLOT_ALT}")


# 11d. 3D orbit comparison
PLOT_3D = "full_physics_orbit_3d.png"

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")
ax.plot(df_base["X (km)"], df_base["Y (km)"], df_base["Z (km)"],
        label="Without third-body", linewidth=1.0, alpha=0.6)
ax.plot(df_full["X (km)"], df_full["Y (km)"], df_full["Z (km)"],
        label="Full physics", linewidth=1.0, linestyle="--")
ax.scatter(0, 0, 0, s=200, color="dodgerblue", label="Earth")
ax.set_xlabel("X (km)")
ax.set_ylabel("Y (km)")
ax.set_zlabel("Z (km)")
ax.set_title("3D Orbit — Full Physics vs No Third-Body")
ax.legend()
fig.savefig(PLOT_3D, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"3D orbit plot saved to {PLOT_3D}")


# ══════════════════════════════════════════════
# Summary of all force models
# ══════════════════════════════════════════════
print("\n" + "=" * 55)
print("  Full-physics force model summary")
print("=" * 55)
print(f"  1. HolmesFeatherstoneAttractionModel  (deg={CONFIG['gravity_degree']}, ord={CONFIG['gravity_order']})")
print(f"  2. DragForce                          (NRLMSISE00, Cd={CONFIG['drag_cd']}, A={CONFIG['cross_section']}m^2)")
print(f"  3. SolarRadiationPressure             (Cr={CONFIG['srp_cr']}, A={CONFIG['cross_section']}m^2)")
print(f"  4. ThirdBodyAttraction                (Sun)")
print(f"  5. ThirdBodyAttraction                (Moon)")
print(f"  Spacecraft mass: {CONFIG['mass']} kg")
print(f"  Propagation:     {CONFIG['num_orbits']} orbit(s), step={CONFIG['step_size']}s")

print("\n Module 4.5 complete — Full physics orbit propagator.")
