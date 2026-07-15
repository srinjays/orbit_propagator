"""
Module 6 — Batch Least Squares Orbit Determination
=====================================================

Estimates the satellite initial state vector from radar measurements
using an iterative Gauss-Newton Batch Least Squares algorithm.

Algorithm:
    1. Start with a perturbed initial state guess
    2. Propagate to each measurement time using Orekit NumericalPropagator
    3. Compute predicted radar observables (range, rate, az, el)
    4. Compute residuals = measured - predicted
    5. Compute Jacobian H via forward finite differences
    6. Solve normal equations: dx = (H'WH)^-1 H'W dz
    7. Update state, repeat until convergence

Input:  radar_measurements.csv, full_physics_orbit_dataset.csv
Output: bls_estimated_orbit.csv, bls_residuals.png, bls_trajectory.png
"""

import os, sys, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════
CONFIG = {
    # I/O
    "radar_csv":        "radar_measurements.csv",
    "truth_csv":        "full_physics_orbit_dataset.csv",

    # Ground station (must match radar_simulator.py)
    "station_lat_deg":   35.4267,
    "station_lon_deg":  -116.8900,
    "station_alt_m":     1000.0,

    # Measurement sigmas (for weight matrix, match radar_simulator)
    "sigma_range_km":    0.010,
    "sigma_rate_kms":    0.001,
    "sigma_az_deg":      0.01,
    "sigma_el_deg":      0.01,

    # Initial state perturbation (applied to true state for initial guess)
    "pos_pert_km":       5.0,
    "vel_pert_kms":      0.005,

    # BLS tuning
    "max_iterations":    10,
    "convergence_km":    1e-6,
    "max_measurements":  300,       # subsample if more than this

    # Finite-difference step sizes
    "fd_pos_km":         0.001,     # 1 m
    "fd_vel_kms":        1e-6,      # 1 mm/s

    # Force model (match Module 4.5)
    "mass":              1000.0,
    "cross_section":     10.0,
    "drag_cd":           2.2,
    "srp_cr":            1.5,
    "gravity_degree":    20,
    "gravity_order":     20,

    # Integrator
    "min_step":          0.001,
    "max_step":          1000.0,
    "init_step":         60.0,
    "dP":                1.0,

    # Earth
    "earth_a":           6378137.0,
    "earth_f":           1.0/298.257223563,
    "omega":             7.2921159e-5,

    "random_seed":       42,
}

# ══════════════════════════════════════════════
# 1. Orekit bootstrap
# ══════════════════════════════════════════════
import orekit_jpype as orekit
orekit.initVM()
from orekit_jpype.pyhelpers import setup_orekit_data

DATA_ZIP = "orekit-data-main.zip"
if os.path.exists(DATA_ZIP):
    setup_orekit_data(DATA_ZIP)
else:
    sys.exit(f"ERROR: {DATA_ZIP} not found")

from org.orekit.time import TimeScalesFactory, AbsoluteDate
from org.orekit.frames import FramesFactory
from org.orekit.utils import Constants, IERSConventions, PVCoordinates
from org.orekit.orbits import CartesianOrbit, OrbitType
from org.orekit.propagation import SpacecraftState
from org.orekit.propagation.numerical import NumericalPropagator
from org.hipparchus.ode.nonstiff import DormandPrince853Integrator
from org.hipparchus.geometry.euclidean.threed import Vector3D
from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel
from org.orekit.forces.gravity.potential import GravityFieldFactory
from org.orekit.bodies import OneAxisEllipsoid, CelestialBodyFactory
from org.orekit.models.earth.atmosphere.data import CssiSpaceWeatherData
from org.orekit.models.earth.atmosphere import NRLMSISE00
from org.orekit.forces.drag import IsotropicDrag, DragForce
from org.orekit.forces.radiation import (IsotropicRadiationSingleCoefficient,
                                          SolarRadiationPressure)
from org.orekit.forces.gravity import ThirdBodyAttraction

utc   = TimeScalesFactory.getUTC()
eci   = FramesFactory.getEME2000()
epoch = AbsoluteDate(2026, 1, 1, 12, 0, 0.0, utc)
mu    = Constants.WGS84_EARTH_MU

print("Orekit initialised.\n")

# ══════════════════════════════════════════════
# 2. Build shared force-model objects (created once, reused)
# ══════════════════════════════════════════════
itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
earth_body = OneAxisEllipsoid(
    Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
    Constants.WGS84_EARTH_FLATTENING, itrf)

sun  = CelestialBodyFactory.getSun()
moon = CelestialBodyFactory.getMoon()
cssi = CssiSpaceWeatherData(CssiSpaceWeatherData.DEFAULT_SUPPORTED_NAMES)
atmosphere = NRLMSISE00(cssi, sun, earth_body)

gravity_provider = GravityFieldFactory.getNormalizedProvider(
    CONFIG["gravity_degree"], CONFIG["gravity_order"])
gravity_model    = HolmesFeatherstoneAttractionModel(itrf, gravity_provider)

drag_model = DragForce(atmosphere,
    IsotropicDrag(CONFIG["cross_section"], CONFIG["drag_cd"]))

srp_model = SolarRadiationPressure(sun,
    Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
    IsotropicRadiationSingleCoefficient(CONFIG["cross_section"], CONFIG["srp_cr"]))

sun_grav  = ThirdBodyAttraction(sun)
moon_grav = ThirdBodyAttraction(moon)

FORCE_MODELS = [gravity_model, drag_model, srp_model, sun_grav, moon_grav]
print(f"Force models ready ({len(FORCE_MODELS)}).\n")

# ══════════════════════════════════════════════
# 3. Coordinate helpers (from radar_simulator)
# ══════════════════════════════════════════════

def geodetic_to_ecef(lat_d, lon_d, alt_m, a, f):
    lat, lon = math.radians(lat_d), math.radians(lon_d)
    e2 = 2*f - f**2
    N = a / math.sqrt(1 - e2*math.sin(lat)**2)
    x = (N+alt_m)*math.cos(lat)*math.cos(lon)
    y = (N+alt_m)*math.cos(lat)*math.sin(lon)
    z = (N*(1-e2)+alt_m)*math.sin(lat)
    return np.array([x, y, z]) / 1000.0

def ecef_to_eci_pos(ecef, t, w):
    c, s = math.cos(w*t), math.sin(w*t)
    return np.array([ecef[0]*c - ecef[1]*s,
                     ecef[0]*s + ecef[1]*c,
                     ecef[2]])

def station_vel_eci(ecef, t, w):
    r = ecef_to_eci_pos(ecef, t, w)
    return np.array([-w*r[1], w*r[0], 0.0])

def compute_obs(sat_p, sat_v, stn_p, stn_v, lat_r, lon_r, t, w):
    """Compute [range_km, range_rate_kms, azimuth_deg, elevation_deg]."""
    dr = sat_p - stn_p
    dv = sat_v - stn_v
    rng = np.linalg.norm(dr)
    rng_rate = np.dot(dr, dv) / rng

    # ECI → ECEF relative
    c, s = math.cos(w*t), math.sin(w*t)
    dx =  dr[0]*c + dr[1]*s
    dy = -dr[0]*s + dr[1]*c
    dz =  dr[2]

    sl, cl = math.sin(lat_r), math.cos(lat_r)
    sn, cn = math.sin(lon_r), math.cos(lon_r)
    east  = -sn*dx + cn*dy
    north = -sl*cn*dx - sl*sn*dy + cl*dz
    up    =  cl*cn*dx + cl*sn*dy + sl*dz

    el = math.degrees(math.atan2(up, math.sqrt(east**2 + north**2)))
    az = math.degrees(math.atan2(east, north)) % 360.0
    return np.array([rng, rng_rate, az, el])

def wrap_angle(a):
    """Wrap to [-180, 180]."""
    return ((a + 180.0) % 360.0) - 180.0

# Station ECEF (computed once)
STN_ECEF = geodetic_to_ecef(CONFIG["station_lat_deg"], CONFIG["station_lon_deg"],
                             CONFIG["station_alt_m"], CONFIG["earth_a"], CONFIG["earth_f"])
STN_LAT_R = math.radians(CONFIG["station_lat_deg"])
STN_LON_R = math.radians(CONFIG["station_lon_deg"])
OMEGA     = CONFIG["omega"]

# ══════════════════════════════════════════════
# 4. Propagator builder
# ══════════════════════════════════════════════

def build_propagator(state_vec_km):
    """Build NumericalPropagator from a 6-element Cartesian state [x,y,z,vx,vy,vz] in km, km/s."""
    pos = Vector3D(float(state_vec_km[0]*1000), float(state_vec_km[1]*1000),
                   float(state_vec_km[2]*1000))
    vel = Vector3D(float(state_vec_km[3]*1000), float(state_vec_km[4]*1000),
                   float(state_vec_km[5]*1000))
    pv  = PVCoordinates(pos, vel)
    orb = CartesianOrbit(pv, eci, epoch, mu)

    tol = NumericalPropagator.tolerances(CONFIG["dP"], orb, OrbitType.CARTESIAN)
    integ = DormandPrince853Integrator(CONFIG["min_step"], CONFIG["max_step"],
                                       tol[0], tol[1])
    integ.setInitialStepSize(CONFIG["init_step"])

    prop = NumericalPropagator(integ)
    prop.setOrbitType(OrbitType.CARTESIAN)
    prop.setInitialState(SpacecraftState(orb, CONFIG["mass"]))
    for fm in FORCE_MODELS:
        prop.addForceModel(fm)
    return prop

# ══════════════════════════════════════════════
# 5. Predict measurements for a given state
# ══════════════════════════════════════════════

def predict_measurements(state_vec, meas_times):
    """
    Propagate state_vec to each measurement time and compute
    predicted radar observables.
    Returns (n_meas, 4) array: [range, rate, az, el] per row.
    """
    prop = build_propagator(state_vec)
    n = len(meas_times)
    preds = np.zeros((n, 4))

    for i in range(n):
        t = meas_times[i]
        date_i = epoch.shiftedBy(float(t))
        state  = prop.propagate(date_i)
        pv     = state.getPVCoordinates()
        p = np.array([pv.getPosition().getX(), pv.getPosition().getY(),
                      pv.getPosition().getZ()]) / 1000.0
        v = np.array([pv.getVelocity().getX(), pv.getVelocity().getY(),
                      pv.getVelocity().getZ()]) / 1000.0

        stn_p = ecef_to_eci_pos(STN_ECEF, t, OMEGA)
        stn_v = station_vel_eci(STN_ECEF, t, OMEGA)
        preds[i] = compute_obs(p, v, stn_p, stn_v, STN_LAT_R, STN_LON_R, t, OMEGA)

    return preds

# ══════════════════════════════════════════════
# 6. Residuals
# ══════════════════════════════════════════════

def compute_residuals(meas, preds):
    """Compute residual vector (4*n,) with angle wrapping on azimuth."""
    n = meas.shape[0]
    res = meas - preds
    # wrap azimuth residuals (column 2)
    res[:, 2] = np.array([wrap_angle(r) for r in res[:, 2]])
    return res.flatten()   # [r1, rr1, az1, el1, r2, ...]

# ══════════════════════════════════════════════
# 7. Jacobian via forward finite differences
# ══════════════════════════════════════════════

def compute_jacobian(state_vec, meas_times, nominal_preds):
    """
    H matrix (4*n_meas, 6) via forward finite differences.
    6 perturbed propagations.
    """
    n = len(meas_times)
    H = np.zeros((4*n, 6))
    steps = np.array([CONFIG["fd_pos_km"]]*3 + [CONFIG["fd_vel_kms"]]*3)

    for j in range(6):
        x_pert = state_vec.copy()
        x_pert[j] += steps[j]
        preds_pert = predict_measurements(x_pert, meas_times)
        # column j of H = (h(x+dx) - h(x)) / dx, flattened
        diff = preds_pert - nominal_preds
        diff[:, 2] = np.array([wrap_angle(d) for d in diff[:, 2]])
        H[:, j] = diff.flatten() / steps[j]

    return H

# ══════════════════════════════════════════════
# 8. BLS solver
# ══════════════════════════════════════════════

def run_bls(initial_guess, meas_array, meas_times):
    """
    Iterative Gauss-Newton BLS.

    Args:
        initial_guess: (6,) state vector [x,y,z,vx,vy,vz] km, km/s
        meas_array:    (n,4) measured [range, rate, az, el]
        meas_times:    (n,) measurement epochs in seconds

    Returns:
        x_est:    estimated state (6,)
        history:  list of dicts per iteration
    """
    # Weight vector: 1/sigma for each observable type, tiled n times
    sigmas = np.array([CONFIG["sigma_range_km"], CONFIG["sigma_rate_kms"],
                       CONFIG["sigma_az_deg"],   CONFIG["sigma_el_deg"]])
    n = len(meas_times)
    w = np.tile(1.0 / sigmas, n)    # (4n,)

    x = initial_guess.copy()
    history = []

    for it in range(CONFIG["max_iterations"]):
        print(f"\n--- BLS Iteration {it+1} ---")

        # Predict
        preds = predict_measurements(x, meas_times)

        # Residuals
        dz = compute_residuals(meas_array, preds)
        rms_res = np.sqrt(np.mean(dz**2))
        print(f"  RMS residual (mixed units): {rms_res:.6f}")

        # Jacobian
        print(f"  Computing Jacobian (6 propagations)...")
        H = compute_jacobian(x, meas_times, preds)

        # Weighted normal equations
        Hw = H * w[:, np.newaxis]       # (4n, 6) weighted
        dzw = dz * w                    # (4n,)   weighted

        N = Hw.T @ Hw                   # (6, 6)
        b = Hw.T @ dzw                  # (6,)

        try:
            dx = np.linalg.solve(N, b)
        except np.linalg.LinAlgError:
            print("  WARNING: Singular normal matrix — stopping.")
            break

        # Update state
        x = x + dx
        norm_dx = np.linalg.norm(dx[:3])  # position correction magnitude

        print(f"  State correction: pos={norm_dx*1000:.4f} m, "
              f"vel={np.linalg.norm(dx[3:])*1000:.4f} m/s")
        print(f"  Updated state: {x}")

        history.append({
            "iteration": it+1,
            "rms_residual": rms_res,
            "pos_correction_m": norm_dx * 1000,
            "vel_correction_ms": np.linalg.norm(dx[3:]) * 1000,
            "state": x.copy(),
            "residuals": dz.copy(),
        })

        if norm_dx < CONFIG["convergence_km"]:
            print(f"\n  CONVERGED at iteration {it+1}.")
            break

    return x, history

# ══════════════════════════════════════════════
# 9. Plotting
# ══════════════════════════════════════════════

def plot_residuals(history):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    iters = [h["iteration"] for h in history]

    axes[0].semilogy(iters, [h["pos_correction_m"] for h in history],
                     "o-", color="crimson", label="Position correction")
    axes[0].set_ylabel("Position Correction (m)")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[0].set_title("BLS Convergence")

    axes[1].semilogy(iters, [h["rms_residual"] for h in history],
                     "s-", color="steelblue", label="RMS residual")
    axes[1].set_xlabel("Iteration"); axes[1].set_ylabel("RMS Residual")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("bls_residuals.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Plot saved: bls_residuals.png")


def plot_trajectory(x_est, x_true, truth_csv):
    """Propagate estimated and true states, plot comparison."""
    df_truth = pd.read_csv(truth_csv)
    # Propagate estimated state for comparison
    prop = build_propagator(x_est)
    times = df_truth["Time (s)"].values
    # Subsample for speed
    step = max(1, len(times) // 500)
    times_sub = times[::step]

    est_x, est_y, est_z = [], [], []
    for t in times_sub:
        st = prop.propagate(epoch.shiftedBy(float(t)))
        p = st.getPVCoordinates().getPosition()
        est_x.append(p.getX()/1000); est_y.append(p.getY()/1000)
        est_z.append(p.getZ()/1000)

    true_x = df_truth["X (km)"].values[::step]
    true_y = df_truth["Y (km)"].values[::step]

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.plot(true_x, true_y, linewidth=1, alpha=0.6, label="Truth", color="steelblue")
    ax.plot(est_x, est_y, linewidth=1, alpha=0.8, linestyle="--",
            label="BLS Estimate", color="orangered")
    ax.scatter(0, 0, s=200, color="dodgerblue", zorder=5, label="Earth")
    ax.set_xlabel("X (km)"); ax.set_ylabel("Y (km)")
    ax.set_title("BLS Estimated Orbit vs Truth")
    ax.axis("equal"); ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig("bls_trajectory.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Plot saved: bls_trajectory.png")

    # Position error over time
    prop2 = build_propagator(x_est)
    errs = []
    t_sub2 = times[::step]
    for i, t in enumerate(t_sub2):
        st = prop2.propagate(epoch.shiftedBy(float(t)))
        p = st.getPVCoordinates().getPosition()
        ex = p.getX()/1000 - df_truth["X (km)"].values[::step][i]
        ey = p.getY()/1000 - df_truth["Y (km)"].values[::step][i]
        ez = p.getZ()/1000 - df_truth["Z (km)"].values[::step][i]
        errs.append(math.sqrt(ex**2+ey**2+ez**2)*1000)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(np.array(t_sub2)/60, errs, color="crimson", linewidth=1)
    ax.set_xlabel("Time (min)"); ax.set_ylabel("Position Error (m)")
    ax.set_title("BLS Position Error vs Truth Over Time")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("bls_position_error.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Plot saved: bls_position_error.png")


# ══════════════════════════════════════════════
# 10. Main
# ══════════════════════════════════════════════
if __name__ == "__main__":

    print("=" * 55)
    print("  Module 6 — Batch Least Squares Orbit Determination")
    print("=" * 55)

    # --- Load radar measurements ---
    df_radar = pd.read_csv(CONFIG["radar_csv"])
    print(f"\nLoaded {len(df_radar)} radar measurements.")

    # Subsample if too many
    if len(df_radar) > CONFIG["max_measurements"]:
        step = len(df_radar) // CONFIG["max_measurements"]
        df_radar = df_radar.iloc[::step].reset_index(drop=True)
        print(f"  Subsampled to {len(df_radar)} measurements.")

    meas_times = df_radar["Time (s)"].values
    meas_array = df_radar[["Meas Range (km)", "Meas Range Rate (km/s)",
                           "Meas Azimuth (deg)", "Meas Elevation (deg)"]].values

    # --- True initial state (from truth CSV, t=0) ---
    df_truth = pd.read_csv(CONFIG["truth_csv"])
    row0 = df_truth.iloc[0]
    x_true = np.array([row0["X (km)"], row0["Y (km)"], row0["Z (km)"],
                       row0["VX (km/s)"], row0["VY (km/s)"], row0["VZ (km/s)"]])
    print(f"True initial state:      {x_true}")

    # --- Perturbed initial guess ---
    rng = np.random.default_rng(CONFIG["random_seed"])
    pert = np.concatenate([
        rng.normal(0, CONFIG["pos_pert_km"], 3),
        rng.normal(0, CONFIG["vel_pert_kms"], 3)
    ])
    x_guess = x_true + pert
    print(f"Perturbation applied:    {pert}")
    print(f"Initial guess:           {x_guess}")
    print(f"Initial pos error:       {np.linalg.norm(pert[:3])*1000:.1f} m")
    print(f"Initial vel error:       {np.linalg.norm(pert[3:])*1000:.1f} m/s")

    # --- Run BLS ---
    x_est, history = run_bls(x_guess, meas_array, meas_times)

    # --- Results ---
    pos_err = np.linalg.norm(x_est[:3] - x_true[:3]) * 1000
    vel_err = np.linalg.norm(x_est[3:] - x_true[3:]) * 1000

    print("\n" + "=" * 55)
    print("  BLS RESULTS")
    print("=" * 55)
    print(f"  True state:      {x_true}")
    print(f"  Estimated state: {x_est}")
    print(f"  Position error:  {pos_err:.4f} m")
    print(f"  Velocity error:  {vel_err:.6f} m/s")
    print(f"  Iterations:      {len(history)}")

    # --- Save estimated orbit ---
    prop_est = build_propagator(x_est)
    times_all = df_truth["Time (s)"].values
    step_out = max(1, len(times_all) // 1000)
    rows = []
    for t in times_all[::step_out]:
        st = prop_est.propagate(epoch.shiftedBy(float(t)))
        p = st.getPVCoordinates().getPosition()
        v = st.getPVCoordinates().getVelocity()
        rows.append([t, p.getX()/1000, p.getY()/1000, p.getZ()/1000,
                     v.getX()/1000, v.getY()/1000, v.getZ()/1000])
    df_est = pd.DataFrame(rows, columns=["Time (s)","X (km)","Y (km)","Z (km)",
                                          "VX (km/s)","VY (km/s)","VZ (km/s)"])
    df_est.to_csv("bls_estimated_orbit.csv", index=False)
    print(f"\nSaved bls_estimated_orbit.csv ({len(df_est)} points)")

    # --- Plots ---
    print("\nGenerating plots...")
    plot_residuals(history)
    plot_trajectory(x_est, x_true, CONFIG["truth_csv"])

    print("\n Module 6 complete — Batch Least Squares OD.")
