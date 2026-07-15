"""
Module 5.2 — Radar Sensor Simulator
======================================

Reads the ground-truth satellite trajectory (ECI/EME2000) and
simulates radar tracking measurements from a ground station.

Measurements simulated:
    1. Range          — distance from station to satellite (km)
    2. Range rate     — Doppler-derived radial velocity (km/s)
    3. Azimuth        — horizontal angle from North (deg)
    4. Elevation      — angle above local horizon (deg)

Sensor effects:
    - Gaussian noise on range, range rate, azimuth, elevation
    - Visibility constraint (elevation > minimum cutoff)
    - Ground station position (geodetic → ECEF → ECI rotation)

Architecture:
    load_truth_trajectory()          — reused from Module 5.1
    geodetic_to_ecef()               — station position
    ecef_to_eci()                    — Earth rotation at each epoch
    compute_topocentric()            — range, az, el in local frame
    simulate_radar()                 — full pipeline
    plot_results()                   — diagnostic plots

Input:   full_physics_orbit_dataset.csv
Output:  radar_measurements.csv
         radar_range_and_rate.png
         radar_azimuth_elevation.png
         radar_errors.png

Usage:
    1. Run thirdbody_propagator.py first
    2. python radar_simulator.py
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ══════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════
CONFIG = {
    # Input / Output
    "truth_csv":          "full_physics_orbit_dataset.csv",
    "output_csv":         "radar_measurements.csv",

    # Ground station location (geodetic coordinates)
    # Default: Goldstone, California — a major DSN tracking station
    "station_lat_deg":    35.4267,      # latitude (degrees North)
    "station_lon_deg":    -116.8900,    # longitude (degrees East)
    "station_alt_m":      1000.0,       # altitude above WGS84 ellipsoid (m)
    "station_name":       "Goldstone",

    # Measurement noise (1-sigma)
    "range_noise_km":     0.010,        # 10 m range accuracy (typical S-band)
    "range_rate_noise_kms": 0.001,      # 1 m/s range-rate accuracy
    "azimuth_noise_deg":  0.01,         # 0.01 deg azimuth accuracy
    "elevation_noise_deg": 0.01,        # 0.01 deg elevation accuracy

    # Visibility constraint
    # Satellite must be above this elevation to be tracked.
    # Typical radar: 5–10 deg to avoid terrain/atmosphere effects.
    "min_elevation_deg":  5.0,

    # Earth parameters (WGS84)
    "earth_radius_eq_m":  6378137.0,        # equatorial radius
    "earth_flattening":   1.0 / 298.257223563,
    "earth_rotation_rate": 7.2921159e-5,    # rad/s

    # Random seed
    "random_seed":        123,
}


# ══════════════════════════════════════════════
# 1. Load truth trajectory (reused from Module 5.1)
# ══════════════════════════════════════════════
def load_truth_trajectory(csv_path):
    """Load propagator output CSV."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Truth trajectory not found: {csv_path}\n"
            f"Run thirdbody_propagator.py first."
        )
    df = pd.read_csv(csv_path)
    print(f"Loaded truth trajectory: {csv_path}  ({len(df)} epochs)")
    return df


# ══════════════════════════════════════════════
# 2. Coordinate transformations
#
# NOTE — Orekit upgrade path (for production quality):
#   The functions below use a simplified constant-rate Earth
#   rotation model (θ = ω·t).  This is sufficient for sensor
#   simulation but ignores:
#     - Precession / nutation of the Earth's axis
#     - Polar motion (x_p, y_p)
#     - UT1-UTC corrections (Earth orientation parameters)
#     - Tidal effects on Earth rotation
#
#   For production-quality tracking, replace these manual
#   rotations with Orekit's frame transformation system:
#
#     from org.orekit.frames import FramesFactory, TopocentricFrame
#     from org.orekit.bodies import OneAxisEllipsoid, GeodeticPoint
#
#     itrf  = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
#     eci   = FramesFactory.getEME2000()
#     earth = OneAxisEllipsoid(Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
#                              Constants.WGS84_EARTH_FLATTENING, itrf)
#
#     station_geo = GeodeticPoint(lat_rad, lon_rad, alt_m)
#     topo_frame  = TopocentricFrame(earth, station_geo, "StationName")
#
#     # At each epoch:
#     transform = topo_frame.getTransformTo(eci, date)
#     # This automatically applies full IAU precession/nutation,
#     # polar motion, and UT1-UTC corrections.
#
#   This gives sub-meter accuracy vs our ~10–100 m approximation
#   over multi-day arcs.
# ══════════════════════════════════════════════

def geodetic_to_ecef(lat_deg, lon_deg, alt_m, a, f):
    """
    Convert geodetic (latitude, longitude, altitude) to ECEF (km).

    The ground station is fixed on Earth's surface. Its ECEF
    position never changes — only its ECI position changes as
    Earth rotates.

    Uses WGS84 ellipsoid:
        N = a / sqrt(1 - e^2 * sin^2(lat))
        x = (N + h) * cos(lat) * cos(lon)
        y = (N + h) * cos(lat) * sin(lon)
        z = (N*(1-e^2) + h) * sin(lat)

    Orekit equivalent:
        earth = OneAxisEllipsoid(..., itrf)
        geo   = GeodeticPoint(lat_rad, lon_rad, alt_m)
        ecef  = earth.transform(geo)  # returns Vector3D in ITRF
    """
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    e2 = 2*f - f**2  # eccentricity squared

    N = a / math.sqrt(1 - e2 * math.sin(lat)**2)

    x = (N + alt_m) * math.cos(lat) * math.cos(lon)
    y = (N + alt_m) * math.cos(lat) * math.sin(lon)
    z = (N * (1 - e2) + alt_m) * math.sin(lat)

    # Convert m → km
    return np.array([x, y, z]) / 1000.0


def ecef_to_eci(pos_ecef_km, time_s, omega):
    """
    Rotate ECEF position to ECI (EME2000) at a given time.

    Simplified model: the Earth rotates at a constant rate ω
    about the Z axis.  At t=0 (epoch) the ECEF and ECI frames
    are assumed aligned (sufficient for sensor simulation —
    we only need relative geometry, not absolute accuracy).

    Rotation angle: θ = ω * t
    ECI_x = ECEF_x * cos(θ) - ECEF_y * sin(θ)
    ECI_y = ECEF_x * sin(θ) + ECEF_y * cos(θ)
    ECI_z = ECEF_z

    Orekit equivalent (full IAU precession/nutation/polar motion):
        itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
        eci  = FramesFactory.getEME2000()
        transform = itrf.getTransformTo(eci, epoch.shiftedBy(time_s))
        pos_eci = transform.transformPosition(Vector3D(x, y, z))

    The Orekit version accounts for:
        - IAU 2000/2006 precession-nutation model
        - Earth Orientation Parameters (EOP) from IERS Bulletins
        - Polar motion (x_p, y_p)
        - UT1-UTC corrections
    These effects accumulate to ~100s of meters over 24 hours.
    """
    theta = omega * time_s
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    x_eci = pos_ecef_km[0] * cos_t - pos_ecef_km[1] * sin_t
    y_eci = pos_ecef_km[0] * sin_t + pos_ecef_km[1] * cos_t
    z_eci = pos_ecef_km[2]

    return np.array([x_eci, y_eci, z_eci])


def station_velocity_eci(pos_ecef_km, time_s, omega):
    """
    Compute the ground station's ECI velocity.

    The station co-rotates with Earth, so its ECI velocity is:
        v = ω × r_eci
    where ω = [0, 0, ω_earth] and r_eci is the station's ECI position.

    Orekit equivalent:
        transform = itrf.getTransformTo(eci, date)
        pv = transform.transformPVCoordinates(
            PVCoordinates(station_pos, Vector3D.ZERO)
        )
        vel_eci = pv.getVelocity()
    """
    r_eci = ecef_to_eci(pos_ecef_km, time_s, omega)
    # Cross product: [0, 0, ω] × [x, y, z] = [-ω*y, ω*x, 0]
    vx = -omega * r_eci[1]
    vy =  omega * r_eci[0]
    vz = 0.0
    return np.array([vx, vy, vz])


def compute_topocentric(sat_pos_eci, sat_vel_eci, stn_pos_eci, stn_vel_eci,
                        stn_lat_rad, stn_lon_rad, time_s, omega):
    """
    Compute topocentric radar observables:
        - Range (km)
        - Range rate (km/s)
        - Azimuth (deg)   — from North, clockwise
        - Elevation (deg) — above local horizon

    The ENU (East-North-Up) frame at the station is used to
    compute azimuth and elevation.  The rotation from ECI to
    ENU involves:
        1. ECEF-relative vector (undo Earth rotation)
        2. Geodetic-to-ENU rotation matrix

    Orekit equivalent (recommended upgrade):
        topo_frame = TopocentricFrame(earth, station_geo, "Name")
        el  = topo_frame.getElevation(sat_pv.getPosition(), eci, date)
        az  = topo_frame.getAzimuth(sat_pv.getPosition(), eci, date)
        rng = topo_frame.getRange(sat_pv.getPosition(), eci, date)
        rr  = topo_frame.getRangeRate(sat_pv, eci, date)

    The Orekit TopocentricFrame handles the full ITRF↔ECI chain
    internally, including Earth orientation parameters.
    """
    # Relative position and velocity in ECI
    dr = sat_pos_eci - stn_pos_eci   # station → satellite vector
    dv = sat_vel_eci - stn_vel_eci   # relative velocity

    # Range
    rng = np.linalg.norm(dr)

    # Range rate (radial velocity = dot(dr, dv) / |dr|)
    rng_rate = np.dot(dr, dv) / rng

    # To get azimuth/elevation, transform to ENU at station
    # First rotate ECI vector back to ECEF
    theta = omega * time_s
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    dx_ecef =  dr[0] * cos_t + dr[1] * sin_t
    dy_ecef = -dr[0] * sin_t + dr[1] * cos_t
    dz_ecef =  dr[2]

    # ECEF to ENU rotation
    sin_lat = math.sin(stn_lat_rad)
    cos_lat = math.cos(stn_lat_rad)
    sin_lon = math.sin(stn_lon_rad + omega * time_s)
    cos_lon = math.cos(stn_lon_rad + omega * time_s)

    # Use the station's instantaneous geographic longitude in ECI
    # Actually, for ECEF-relative we use the fixed geodetic angles:
    sin_lon = math.sin(stn_lon_rad)
    cos_lon = math.cos(stn_lon_rad)

    # ENU = R * ECEF_relative
    east  = -sin_lon * dx_ecef + cos_lon * dy_ecef
    north = -sin_lat * cos_lon * dx_ecef - sin_lat * sin_lon * dy_ecef + cos_lat * dz_ecef
    up    =  cos_lat * cos_lon * dx_ecef + cos_lat * sin_lon * dy_ecef + sin_lat * dz_ecef

    # Elevation
    horiz = math.sqrt(east**2 + north**2)
    elevation = math.degrees(math.atan2(up, horiz))

    # Azimuth (from North, clockwise)
    azimuth = math.degrees(math.atan2(east, north)) % 360.0

    return rng, rng_rate, azimuth, elevation


# ══════════════════════════════════════════════
# 3. Full radar simulation pipeline
# ══════════════════════════════════════════════
def simulate_radar(df_truth, config):
    """
    Orchestrate the radar sensor simulation.

    Steps:
        1. Compute station ECEF position (fixed)
        2. At each epoch, rotate station to ECI
        3. Compute topocentric observables (range, rate, az, el)
        4. Apply visibility filter (elevation cutoff)
        5. Add Gaussian noise
        6. Build output DataFrame
    """
    rng = np.random.default_rng(config["random_seed"])

    # Station setup
    stn_ecef = geodetic_to_ecef(
        config["station_lat_deg"],
        config["station_lon_deg"],
        config["station_alt_m"],
        config["earth_radius_eq_m"],
        config["earth_flattening"]
    )
    stn_lat_rad = math.radians(config["station_lat_deg"])
    stn_lon_rad = math.radians(config["station_lon_deg"])
    omega = config["earth_rotation_rate"]

    print(f"\nGround station: {config['station_name']}")
    print(f"  Lat = {config['station_lat_deg']:.4f} deg")
    print(f"  Lon = {config['station_lon_deg']:.4f} deg")
    print(f"  Alt = {config['station_alt_m']:.0f} m")
    print(f"  ECEF = [{stn_ecef[0]:.3f}, {stn_ecef[1]:.3f}, {stn_ecef[2]:.3f}] km")

    times_s = df_truth["Time (s)"].values
    n_epochs = len(times_s)

    true_pos = df_truth[["X (km)", "Y (km)", "Z (km)"]].values
    true_vel = df_truth[["VX (km/s)", "VY (km/s)", "VZ (km/s)"]].values

    # Compute observables at each epoch
    ranges     = np.zeros(n_epochs)
    range_rates = np.zeros(n_epochs)
    azimuths   = np.zeros(n_epochs)
    elevations = np.zeros(n_epochs)

    for i in range(n_epochs):
        t = times_s[i]

        stn_eci = ecef_to_eci(stn_ecef, t, omega)
        stn_vel = station_velocity_eci(stn_ecef, t, omega)

        r, rr, az, el = compute_topocentric(
            true_pos[i], true_vel[i],
            stn_eci, stn_vel,
            stn_lat_rad, stn_lon_rad,
            t, omega
        )

        ranges[i]      = r
        range_rates[i] = rr
        azimuths[i]    = az
        elevations[i]  = el

    # Visibility filter — only track when satellite is above cutoff
    min_el = config["min_elevation_deg"]
    visible = elevations >= min_el

    n_visible = visible.sum()
    print(f"\nVisibility analysis:")
    print(f"  Min elevation cutoff = {min_el} deg")
    print(f"  Total epochs         = {n_epochs}")
    print(f"  Visible epochs       = {n_visible}  ({n_visible/n_epochs*100:.1f}%)")
    print(f"  Below horizon        = {n_epochs - n_visible}")

    # Add noise (only to visible epochs, but compute for all)
    range_noise      = rng.normal(0, config["range_noise_km"], n_epochs)
    range_rate_noise = rng.normal(0, config["range_rate_noise_kms"], n_epochs)
    azimuth_noise    = rng.normal(0, config["azimuth_noise_deg"], n_epochs)
    elevation_noise  = rng.normal(0, config["elevation_noise_deg"], n_epochs)

    # Measured = true + noise
    meas_range      = ranges      + range_noise
    meas_range_rate = range_rates + range_rate_noise
    meas_azimuth    = (azimuths   + azimuth_noise) % 360.0
    meas_elevation  = elevations  + elevation_noise

    # Build full DataFrame
    df_all = pd.DataFrame({
        "Time (s)":              times_s,
        "True Range (km)":       ranges,
        "True Range Rate (km/s)": range_rates,
        "True Azimuth (deg)":    azimuths,
        "True Elevation (deg)":  elevations,
        "Meas Range (km)":       meas_range,
        "Meas Range Rate (km/s)": meas_range_rate,
        "Meas Azimuth (deg)":    meas_azimuth,
        "Meas Elevation (deg)":  meas_elevation,
        "Range Error (m)":       range_noise * 1000,
        "Range Rate Error (m/s)": range_rate_noise * 1000,
        "Azimuth Error (deg)":   azimuth_noise,
        "Elevation Error (deg)": elevation_noise,
        "Visible":               visible,
    })

    # Filter to visible measurements
    df_radar = df_all[df_all["Visible"]].copy()
    df_radar = df_radar.drop(columns=["Visible"])
    df_radar = df_radar.reset_index(drop=True)

    print(f"\nRadar simulation complete:")
    print(f"  Valid measurements = {len(df_radar)}")
    if len(df_radar) > 0:
        print(f"  Range        : {df_radar['True Range (km)'].min():.1f} – "
              f"{df_radar['True Range (km)'].max():.1f} km")
        print(f"  Elevation    : {df_radar['True Elevation (deg)'].min():.2f} – "
              f"{df_radar['True Elevation (deg)'].max():.2f} deg")

    return df_radar, df_all


# ══════════════════════════════════════════════
# 4. Plotting functions
# ══════════════════════════════════════════════

def plot_range_and_rate(df_radar):
    """Range and range-rate over time for visible passes."""
    PLOT_FILE = "radar_range_and_rate.png"

    times_min = df_radar["Time (s)"].values / 60.0

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax = axes[0]
    ax.plot(times_min, df_radar["True Range (km)"],
            linewidth=0.8, alpha=0.7, label="True", color="steelblue")
    ax.scatter(times_min, df_radar["Meas Range (km)"],
               s=1, alpha=0.4, label="Measured", color="orangered")
    ax.set_ylabel("Range (km)")
    ax.set_title("Radar Range")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(times_min, df_radar["True Range Rate (km/s)"],
            linewidth=0.8, alpha=0.7, label="True", color="steelblue")
    ax.scatter(times_min, df_radar["Meas Range Rate (km/s)"],
               s=1, alpha=0.4, label="Measured", color="orangered")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Range Rate (km/s)")
    ax.set_title("Radar Range Rate (Doppler)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(PLOT_FILE, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {PLOT_FILE}")


def plot_azimuth_elevation(df_radar):
    """Azimuth and elevation over time."""
    PLOT_FILE = "radar_azimuth_elevation.png"

    times_min = df_radar["Time (s)"].values / 60.0

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax = axes[0]
    ax.scatter(times_min, df_radar["Meas Azimuth (deg)"],
               s=2, alpha=0.4, color="teal")
    ax.set_ylabel("Azimuth (deg)")
    ax.set_title("Radar Azimuth (from North)")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.scatter(times_min, df_radar["Meas Elevation (deg)"],
               s=2, alpha=0.4, color="darkorchid")
    ax.axhline(CONFIG["min_elevation_deg"], color="red", linestyle="--",
               alpha=0.5, label=f"Min el = {CONFIG['min_elevation_deg']}°")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Elevation (deg)")
    ax.set_title("Radar Elevation")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(PLOT_FILE, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {PLOT_FILE}")


def plot_radar_errors(df_radar):
    """Measurement errors over time."""
    PLOT_FILE = "radar_errors.png"

    times_min = df_radar["Time (s)"].values / 60.0

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    ax = axes[0, 0]
    ax.scatter(times_min, df_radar["Range Error (m)"],
               s=1, alpha=0.4, color="crimson")
    ax.set_ylabel("Range Error (m)")
    ax.set_title("Range Error")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.scatter(times_min, df_radar["Range Rate Error (m/s)"],
               s=1, alpha=0.4, color="darkorchid")
    ax.set_ylabel("Range Rate Error (m/s)")
    ax.set_title("Range Rate Error")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.scatter(times_min, df_radar["Azimuth Error (deg)"],
               s=1, alpha=0.4, color="teal")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Azimuth Error (deg)")
    ax.set_title("Azimuth Error")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.scatter(times_min, df_radar["Elevation Error (deg)"],
               s=1, alpha=0.4, color="goldenrod")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Elevation Error (deg)")
    ax.set_title("Elevation Error")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Radar Measurement Errors", fontsize=14)
    fig.tight_layout()
    fig.savefig(PLOT_FILE, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {PLOT_FILE}")


# ══════════════════════════════════════════════
# 5. Main execution
# ══════════════════════════════════════════════
if __name__ == "__main__":

    print("=" * 55)
    print("  Module 5.2 — Radar Sensor Simulator")
    print("=" * 55)
    print()

    print("Configuration:")
    for k, v in CONFIG.items():
        print(f"  {k:25s} = {v}")
    print()

    # Load truth
    df_truth = load_truth_trajectory(CONFIG["truth_csv"])

    # Simulate radar
    df_radar, df_all = simulate_radar(df_truth, CONFIG)

    # Preview
    if len(df_radar) > 0:
        print("\nRadar measurements — first 5 rows:")
        print(df_radar.head().to_string(index=True))
        print()

    # Save CSV
    df_radar.to_csv(CONFIG["output_csv"], index=False)
    print(f"Saved {CONFIG['output_csv']}  ({len(df_radar)} measurements)")

    # Error statistics
    if len(df_radar) > 0:
        print("\nError statistics:")
        print(f"  Range error      : σ={df_radar['Range Error (m)'].std():.2f} m")
        print(f"  Range rate error : σ={df_radar['Range Rate Error (m/s)'].std():.4f} m/s")
        print(f"  Azimuth error    : σ={df_radar['Azimuth Error (deg)'].std():.4f} deg")
        print(f"  Elevation error  : σ={df_radar['Elevation Error (deg)'].std():.4f} deg")

    # Plots
    if len(df_radar) > 0:
        print("\nGenerating plots...")
        plot_range_and_rate(df_radar)
        plot_azimuth_elevation(df_radar)
        plot_radar_errors(df_radar)

    print("\n Module 5.2 complete — Radar sensor simulation.")
