"""
Module 7 — Extended Kalman Filter Orbit Determination
=======================================================
Sequential state estimation from radar measurements.

EKF loop per measurement:
    PREDICT:  x_pred = propagate(x, t_k → t_{k+1})
              P_pred = F P F' + Q
    UPDATE:   y = z - h(x_pred)
              K = P_pred H' (H P_pred H' + R)^{-1}
              x = x_pred + K y
              P = (I - K H) P_pred

Input:  radar_measurements.csv, bls_estimated_orbit.csv
Output: ekf_estimated_orbit.csv, ekf_*.png
"""

import os, sys, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CONFIG = {
    "radar_csv":        "radar_measurements.csv",
    "truth_csv":        "full_physics_orbit_dataset.csv",
    "bls_csv":          "bls_estimated_orbit.csv",

    "station_lat_deg":   35.4267,
    "station_lon_deg":  -116.8900,
    "station_alt_m":     1000.0,

    # Measurement noise (R matrix diagonal)
    "sigma_range_km":    0.010,
    "sigma_rate_kms":    0.001,
    "sigma_az_deg":      0.01,
    "sigma_el_deg":      0.01,

    # Initial covariance P0 diagonal
    "p0_pos_km2":        25.0,      # (5 km)^2
    "p0_vel_kms2":       2.5e-5,    # (0.005 km/s)^2

    # Process noise Q diagonal (per second, scaled by dt)
    "q_pos_km2_s":       1e-12,     # position process noise
    "q_vel_kms2_s":      1e-10,     # velocity process noise

    # Finite difference steps
    "fd_pos_km":         0.001,
    "fd_vel_kms":        1e-6,

    # Force model (same as Module 4.5 / 6)
    "mass": 1000.0, "cross_section": 10.0,
    "drag_cd": 2.2, "srp_cr": 1.5,
    "gravity_degree": 20, "gravity_order": 20,
    "min_step": 0.001, "max_step": 1000.0,
    "init_step": 60.0, "dP": 1.0,

    "earth_a": 6378137.0,
    "earth_f": 1.0/298.257223563,
    "omega":   7.2921159e-5,

    "max_measurements": 300,
}

# ── Orekit bootstrap ──
import orekit_jpype as orekit
orekit.initVM()
from orekit_jpype.pyhelpers import setup_orekit_data
if os.path.exists("orekit-data-main.zip"):
    setup_orekit_data("orekit-data-main.zip")
else:
    sys.exit("ERROR: orekit-data-main.zip not found")

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
print("Orekit ready.\n")

# ── Shared force models ──
itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
earth_body = OneAxisEllipsoid(Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
                               Constants.WGS84_EARTH_FLATTENING, itrf)
sun  = CelestialBodyFactory.getSun()
moon = CelestialBodyFactory.getMoon()
cssi = CssiSpaceWeatherData(CssiSpaceWeatherData.DEFAULT_SUPPORTED_NAMES)
atmo = NRLMSISE00(cssi, sun, earth_body)

FORCE_MODELS = [
    HolmesFeatherstoneAttractionModel(itrf,
        GravityFieldFactory.getNormalizedProvider(CONFIG["gravity_degree"],
                                                  CONFIG["gravity_order"])),
    DragForce(atmo, IsotropicDrag(CONFIG["cross_section"], CONFIG["drag_cd"])),
    SolarRadiationPressure(sun, Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
        IsotropicRadiationSingleCoefficient(CONFIG["cross_section"], CONFIG["srp_cr"])),
    ThirdBodyAttraction(sun),
    ThirdBodyAttraction(moon),
]

# ── Station setup ──
def _geo2ecef(lat_d, lon_d, alt, a, f):
    lat, lon = math.radians(lat_d), math.radians(lon_d)
    e2 = 2*f - f**2
    N = a / math.sqrt(1 - e2*math.sin(lat)**2)
    return np.array([(N+alt)*math.cos(lat)*math.cos(lon),
                     (N+alt)*math.cos(lat)*math.sin(lon),
                     (N*(1-e2)+alt)*math.sin(lat)]) / 1000.0

STN = _geo2ecef(CONFIG["station_lat_deg"], CONFIG["station_lon_deg"],
                CONFIG["station_alt_m"], CONFIG["earth_a"], CONFIG["earth_f"])
SLT = math.radians(CONFIG["station_lat_deg"])
SLN = math.radians(CONFIG["station_lon_deg"])
W   = CONFIG["omega"]

def stn_eci(t):
    c, s = math.cos(W*t), math.sin(W*t)
    return np.array([STN[0]*c - STN[1]*s, STN[0]*s + STN[1]*c, STN[2]])

def stn_vel(t):
    r = stn_eci(t)
    return np.array([-W*r[1], W*r[0], 0.0])

def obs_fn(sp, sv, t):
    """Compute [range, rate, az, el] from satellite pos/vel (km, km/s) at time t."""
    rp, rv = stn_eci(t), stn_vel(t)
    dr, dv = sp - rp, sv - rv
    rng = np.linalg.norm(dr)
    rr  = np.dot(dr, dv) / rng
    c, s = math.cos(W*t), math.sin(W*t)
    dx, dy, dz = dr[0]*c+dr[1]*s, -dr[0]*s+dr[1]*c, dr[2]
    sl, cl = math.sin(SLT), math.cos(SLT)
    sn, cn = math.sin(SLN), math.cos(SLN)
    e = -sn*dx + cn*dy
    n = -sl*cn*dx - sl*sn*dy + cl*dz
    u =  cl*cn*dx + cl*sn*dy + sl*dz
    el = math.degrees(math.atan2(u, math.sqrt(e**2+n**2)))
    az = math.degrees(math.atan2(e, n)) % 360.0
    return np.array([rng, rr, az, el])

def wrap(a):
    return ((a+180)%360)-180

# ── Propagator builder ──
def build_prop(sv):
    """Build propagator from state vector [x,y,z,vx,vy,vz] km,km/s at epoch."""
    pos = Vector3D(*[float(v*1000) for v in sv[:3]])
    vel = Vector3D(*[float(v*1000) for v in sv[3:]])
    orb = CartesianOrbit(PVCoordinates(pos, vel), eci, epoch, mu)
    tol = NumericalPropagator.tolerances(CONFIG["dP"], orb, OrbitType.CARTESIAN)
    ig  = DormandPrince853Integrator(CONFIG["min_step"], CONFIG["max_step"],
                                      tol[0], tol[1])
    ig.setInitialStepSize(CONFIG["init_step"])
    p = NumericalPropagator(ig)
    p.setOrbitType(OrbitType.CARTESIAN)
    p.setInitialState(SpacecraftState(orb, CONFIG["mass"]))
    for fm in FORCE_MODELS:
        p.addForceModel(fm)
    return p

def propagate_state(sv, t_from, t_to):
    """Propagate state vector from t_from to t_to (seconds from epoch).
    Returns new 6-element state vector in km, km/s."""
    pos = Vector3D(*[float(v*1000) for v in sv[:3]])
    vel = Vector3D(*[float(v*1000) for v in sv[3:]])
    orb = CartesianOrbit(PVCoordinates(pos, vel), eci,
                          epoch.shiftedBy(float(t_from)), mu)
    tol = NumericalPropagator.tolerances(CONFIG["dP"], orb, OrbitType.CARTESIAN)
    ig  = DormandPrince853Integrator(CONFIG["min_step"], CONFIG["max_step"],
                                      tol[0], tol[1])
    ig.setInitialStepSize(CONFIG["init_step"])
    p = NumericalPropagator(ig)
    p.setOrbitType(OrbitType.CARTESIAN)
    p.setInitialState(SpacecraftState(orb, CONFIG["mass"]))
    for fm in FORCE_MODELS:
        p.addForceModel(fm)
    st = p.propagate(epoch.shiftedBy(float(t_to)))
    pv = st.getPVCoordinates()
    return np.array([pv.getPosition().getX()/1000, pv.getPosition().getY()/1000,
                     pv.getPosition().getZ()/1000, pv.getVelocity().getX()/1000,
                     pv.getVelocity().getY()/1000, pv.getVelocity().getZ()/1000])

# ══════════════════════════════════════════════
# EKF FUNCTIONS
# ══════════════════════════════════════════════

def compute_F(x, t_from, t_to):
    """State transition matrix F (6x6) via forward finite differences."""
    x_nom = propagate_state(x, t_from, t_to)
    F = np.zeros((6, 6))
    steps = np.array([CONFIG["fd_pos_km"]]*3 + [CONFIG["fd_vel_kms"]]*3)
    for j in range(6):
        xp = x.copy(); xp[j] += steps[j]
        x_pert = propagate_state(xp, t_from, t_to)
        F[:, j] = (x_pert - x_nom) / steps[j]
    return F, x_nom

def compute_H(x_pred, t):
    """Measurement Jacobian H (4x6) via finite differences at time t."""
    p, v = x_pred[:3], x_pred[3:]
    h_nom = obs_fn(p, v, t)
    H = np.zeros((4, 6))
    steps = np.array([CONFIG["fd_pos_km"]]*3 + [CONFIG["fd_vel_kms"]]*3)
    for j in range(6):
        xp = x_pred.copy(); xp[j] += steps[j]
        h_pert = obs_fn(xp[:3], xp[3:], t)
        diff = h_pert - h_nom
        diff[2] = wrap(diff[2])
        H[:, j] = diff / steps[j]
    return H, h_nom

def make_Q(dt):
    """Process noise covariance scaled by time interval dt."""
    q = np.diag([CONFIG["q_pos_km2_s"]]*3 + [CONFIG["q_vel_kms2_s"]]*3)
    return q * abs(dt)

R = np.diag([CONFIG["sigma_range_km"]**2, CONFIG["sigma_rate_kms"]**2,
             CONFIG["sigma_az_deg"]**2,   CONFIG["sigma_el_deg"]**2])

def ekf_predict(x, P, t_from, t_to):
    """EKF prediction step."""
    F, x_pred = compute_F(x, t_from, t_to)
    P_pred = F @ P @ F.T + make_Q(t_to - t_from)
    return x_pred, P_pred

def ekf_update(x_pred, P_pred, z_meas, t):
    """EKF update step."""
    H, h_pred = compute_H(x_pred, t)
    innov = z_meas - h_pred
    innov[2] = wrap(innov[2])  # azimuth wrap

    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)

    x_upd = x_pred + K @ innov
    P_upd = (np.eye(6) - K @ H) @ P_pred
    return x_upd, P_upd, innov

# ══════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════

def plot_ekf_results(results, df_truth):
    times_min = np.array([r["t"] for r in results]) / 60.0

    # 1. Position error
    pos_err = np.array([r["pos_err_m"] for r in results])
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(times_min, pos_err, linewidth=0.8, color="crimson")
    ax.set_xlabel("Time (min)"); ax.set_ylabel("3D Position Error (m)")
    ax.set_title("EKF Position Estimation Error"); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig("ekf_position_error.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("Plot: ekf_position_error.png")

    # 2. Covariance (1-sigma position)
    sig_pos = np.array([r["sig_pos_m"] for r in results])
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.semilogy(times_min, sig_pos, color="steelblue", linewidth=0.8)
    ax.set_xlabel("Time (min)"); ax.set_ylabel("Position 1σ (m)")
    ax.set_title("EKF Position Covariance Evolution"); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig("ekf_covariance.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("Plot: ekf_covariance.png")

    # 3. Residuals
    res = np.array([r["innov"] for r in results])
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    labels = ["Range (km)", "Range Rate (km/s)", "Azimuth (deg)", "Elevation (deg)"]
    colors = ["crimson", "darkorchid", "teal", "goldenrod"]
    for i, (ax, lab, col) in enumerate(zip(axes.flat, labels, colors)):
        ax.scatter(times_min, res[:, i], s=2, alpha=0.5, color=col)
        ax.set_ylabel(lab); ax.grid(True, alpha=0.3)
        if i >= 2: ax.set_xlabel("Time (min)")
    axes[0,0].set_title("Measurement Residuals (Innovation)")
    fig.tight_layout(); fig.savefig("ekf_residuals.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("Plot: ekf_residuals.png")

    # 4. Trajectory comparison
    est_x = [r["x"][0] for r in results]
    est_y = [r["x"][1] for r in results]
    fig, ax = plt.subplots(figsize=(9, 9))
    step = max(1, len(df_truth)//500)
    ax.plot(df_truth["X (km)"].values[::step], df_truth["Y (km)"].values[::step],
            linewidth=0.8, alpha=0.5, label="Truth", color="steelblue")
    ax.plot(est_x, est_y, linewidth=1, linestyle="--", label="EKF", color="orangered")
    ax.scatter(0, 0, s=200, color="dodgerblue", zorder=5, label="Earth")
    ax.set_xlabel("X (km)"); ax.set_ylabel("Y (km)")
    ax.set_title("EKF Estimated Orbit vs Truth")
    ax.axis("equal"); ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig("ekf_trajectory.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("Plot: ekf_trajectory.png")

# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 55)
    print("  Module 7 — Extended Kalman Filter OD")
    print("=" * 55)

    # Load data
    df_radar = pd.read_csv(CONFIG["radar_csv"])
    df_truth = pd.read_csv(CONFIG["truth_csv"])
    print(f"Radar measurements: {len(df_radar)}")

    if len(df_radar) > CONFIG["max_measurements"]:
        step = len(df_radar) // CONFIG["max_measurements"]
        df_radar = df_radar.iloc[::step].reset_index(drop=True)
        print(f"  Subsampled to {len(df_radar)}")

    meas_times = df_radar["Time (s)"].values
    meas_data  = df_radar[["Meas Range (km)", "Meas Range Rate (km/s)",
                           "Meas Azimuth (deg)", "Meas Elevation (deg)"]].values

    # Initial state from BLS output (first row)
    if os.path.exists(CONFIG["bls_csv"]):
        df_bls = pd.read_csv(CONFIG["bls_csv"])
        r0 = df_bls.iloc[0]
        x = np.array([r0["X (km)"], r0["Y (km)"], r0["Z (km)"],
                       r0["VX (km/s)"], r0["VY (km/s)"], r0["VZ (km/s)"]])
        print(f"Initial state from BLS: {x}")
    else:
        # Fallback: perturbed truth
        r0 = df_truth.iloc[0]
        x = np.array([r0["X (km)"], r0["Y (km)"], r0["Z (km)"],
                       r0["VX (km/s)"], r0["VY (km/s)"], r0["VZ (km/s)"]])
        rng = np.random.default_rng(42)
        x[:3] += rng.normal(0, 5.0, 3)
        x[3:] += rng.normal(0, 0.005, 3)
        print(f"Initial state (perturbed truth): {x}")

    # True initial state for error computation
    t0 = df_truth.iloc[0]
    x_true_0 = np.array([t0["X (km)"], t0["Y (km)"], t0["Z (km)"],
                          t0["VX (km/s)"], t0["VY (km/s)"], t0["VZ (km/s)"]])

    # Initial covariance
    P = np.diag([CONFIG["p0_pos_km2"]]*3 + [CONFIG["p0_vel_kms2"]]*3)

    # ── EKF Loop ──
    results = []
    t_prev = 0.0
    print(f"\nProcessing {len(meas_times)} measurements...\n")

    for k in range(len(meas_times)):
        t_k = meas_times[k]
        z_k = meas_data[k]

        # PREDICT: propagate from t_prev to t_k
        if t_k > t_prev:
            try:
                x_pred, P_pred = ekf_predict(x, P, t_prev, t_k)
            except Exception as e:
                print(f"  [{k}] Predict failed at t={t_k:.0f}s: {e}")
                break
        else:
            x_pred, P_pred = x.copy(), P.copy()

        # UPDATE: incorporate measurement
        try:
            x_upd, P_upd, innov = ekf_update(x_pred, P_pred, z_k, t_k)
        except Exception as e:
            print(f"  [{k}] Update failed at t={t_k:.0f}s: {e}")
            x_upd, P_upd, innov = x_pred, P_pred, np.zeros(4)

        x, P = x_upd, P_upd
        t_prev = t_k

        # True state at t_k (interpolate from truth CSV)
        idx = np.argmin(np.abs(df_truth["Time (s)"].values - t_k))
        tr = df_truth.iloc[idx]
        x_true_k = np.array([tr["X (km)"], tr["Y (km)"], tr["Z (km)"],
                              tr["VX (km/s)"], tr["VY (km/s)"], tr["VZ (km/s)"]])
        pos_err = np.linalg.norm(x[:3] - x_true_k[:3]) * 1000
        vel_err = np.linalg.norm(x[3:] - x_true_k[3:]) * 1000
        sig_pos = math.sqrt(P[0,0] + P[1,1] + P[2,2]) * 1000

        results.append({"t": t_k, "x": x.copy(), "P": P.copy(),
                         "innov": innov.copy(), "pos_err_m": pos_err,
                         "vel_err_ms": vel_err, "sig_pos_m": sig_pos})

        if k % 50 == 0 or k == len(meas_times)-1:
            print(f"  [{k:4d}/{len(meas_times)}] t={t_k/60:8.1f} min  "
                  f"pos_err={pos_err:8.2f} m  vel_err={vel_err:6.4f} m/s  "
                  f"1σ={sig_pos:8.2f} m")

    # ── Summary ──
    print("\n" + "=" * 55)
    print("  EKF RESULTS")
    print("=" * 55)
    if results:
        print(f"  Final pos error:  {results[-1]['pos_err_m']:.4f} m")
        print(f"  Final vel error:  {results[-1]['vel_err_ms']:.6f} m/s")
        print(f"  Final 1σ pos:     {results[-1]['sig_pos_m']:.4f} m")
        print(f"  Measurements:     {len(results)}")

    # ── Export ──
    # Include covariance diagonal and radar residuals for AI training (Module 8.1)
    rows = [{"Time (s)": r["t"],
             "X (km)": r["x"][0], "Y (km)": r["x"][1], "Z (km)": r["x"][2],
             "VX (km/s)": r["x"][3], "VY (km/s)": r["x"][4], "VZ (km/s)": r["x"][5],
             "P_xx": r["P"][0,0], "P_yy": r["P"][1,1], "P_zz": r["P"][2,2],
             "P_vxvx": r["P"][3,3], "P_vyvy": r["P"][4,4], "P_vzvz": r["P"][5,5],
             "Resid Range (km)": r["innov"][0],
             "Resid Rate (km/s)": r["innov"][1],
             "Resid Az (deg)": r["innov"][2],
             "Resid El (deg)": r["innov"][3],
             "Pos Error (m)": r["pos_err_m"], "Vel Error (m/s)": r["vel_err_ms"],
             "Pos 1sig (m)": r["sig_pos_m"]} for r in results]
    df_out = pd.DataFrame(rows)
    df_out.to_csv("ekf_estimated_orbit.csv", index=False)
    print(f"\nSaved ekf_estimated_orbit.csv ({len(df_out)} rows)")

    # ── Plots ──
    if results:
        print("\nGenerating plots...")
        plot_ekf_results(results, df_truth)

    print("\n Module 7 complete — Extended Kalman Filter OD.")
