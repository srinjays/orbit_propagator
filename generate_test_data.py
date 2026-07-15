"""
Synthetic orbit data generator for testing the AI pipeline.
Generates realistic EKF + truth data when Orekit is unavailable.
"""
import numpy as np
import pandas as pd

np.random.seed(42)

# -- Orbit parameters --
n_samples = 1000
t = np.linspace(0, 5400, n_samples)  # 90-minute orbit
r = 7000       # km, ~630 km altitude
omega = 2 * np.pi / 5400  # rad/s

# -- Truth orbit (circular, inclined) --
inc = np.radians(45)  # 45-degree inclination
x_t = r * np.cos(omega * t)
y_t = r * np.sin(omega * t) * np.cos(inc)
z_t = r * np.sin(omega * t) * np.sin(inc)

vx_t = -r * omega * np.sin(omega * t)
vy_t =  r * omega * np.cos(omega * t) * np.cos(inc)
vz_t =  r * omega * np.cos(omega * t) * np.sin(inc)

# -- Save truth --
df_truth = pd.DataFrame({
    "Time (s)": t,
    "X (km)": x_t, "Y (km)": y_t, "Z (km)": z_t,
    "VX (km/s)": vx_t, "VY (km/s)": vy_t, "VZ (km/s)": vz_t,
})
df_truth.to_csv("full_physics_orbit_dataset.csv", index=False)

# -- EKF estimates (truth + realistic noise) --
g = np.random.normal
pos_noise = 0.05   # km (50m position error)
vel_noise = 1e-4   # km/s (0.1 m/s velocity error)

df_ekf = pd.DataFrame({
    "Time (s)": t,
    "X (km)":     x_t  + g(0, pos_noise, n_samples),
    "Y (km)":     y_t  + g(0, pos_noise, n_samples),
    "Z (km)":     z_t  + g(0, pos_noise, n_samples),
    "VX (km/s)":  vx_t + g(0, vel_noise, n_samples),
    "VY (km/s)":  vy_t + g(0, vel_noise, n_samples),
    "VZ (km/s)":  vz_t + g(0, vel_noise, n_samples),
    # Covariance diagonal
    "P_xx":     np.abs(g(25, 5, n_samples)),
    "P_yy":     np.abs(g(25, 5, n_samples)),
    "P_zz":     np.abs(g(25, 5, n_samples)),
    "P_vxvx":   np.abs(g(2.5e-5, 5e-6, n_samples)),
    "P_vyvy":   np.abs(g(2.5e-5, 5e-6, n_samples)),
    "P_vzvz":   np.abs(g(2.5e-5, 5e-6, n_samples)),
    # Radar residuals
    "Resid Range (km)":   g(0, 0.01, n_samples),
    "Resid Rate (km/s)":  g(0, 0.001, n_samples),
    "Resid Az (deg)":     g(0, 0.01, n_samples),
    "Resid El (deg)":     g(0, 0.01, n_samples),
    # Error columns
    "Pos Error (m)":  np.random.uniform(10, 100, n_samples),
    "Vel Error (m/s)": np.random.uniform(0.01, 0.1, n_samples),
    "Pos 1sig (m)":   np.random.uniform(50, 200, n_samples),
})
df_ekf.to_csv("ekf_estimated_orbit.csv", index=False)

print(f"Generated {n_samples} samples:")
print(f"  full_physics_orbit_dataset.csv  ({len(df_truth)} rows, {len(df_truth.columns)} cols)")
print(f"  ekf_estimated_orbit.csv         ({len(df_ekf)} rows, {len(df_ekf.columns)} cols)")
print(f"  Orbit: {r} km radius, 45 deg inclination, 90 min period")
print(f"  Noise: pos={pos_noise*1000:.0f} m, vel={vel_noise*1000:.1f} m/s")
