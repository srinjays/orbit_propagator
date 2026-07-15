"""
Module 8.5 -- Evaluation & Orbit Correction
=============================================
Loads the trained model, predicts correction vectors, applies them
to the EKF orbit, and compares against truth.

Input:   orbit_correction_model.pt, ekf_estimated_orbit.csv,
         full_physics_orbit_dataset.csv, normalisation_stats.csv
Output:  corrected_orbit.csv, eval_error_comparison.png,
         eval_orbit_comparison.png, eval_error_timeseries.png

Usage:
    python evaluate_model.py
"""

import os, sys
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train_model import load_trained_model

# ======================================================
CONFIG = {
    "model_path":     "orbit_correction_model.pt",
    "ekf_csv":        "ekf_estimated_orbit.csv",
    "truth_csv":      "full_physics_orbit_dataset.csv",
    "norm_stats_csv": "normalisation_stats.csv",
    "output_csv":     "corrected_orbit.csv",
    "time_tol_s":     1.0,
    "device":         "cuda" if torch.cuda.is_available() else "cpu",
}

# ======================================================
# 1. DATA LOADING & FEATURE PREPARATION
# ======================================================

def load_and_match(config):
    """Load EKF, truth, match by timestamp, return aligned DataFrames."""
    df_ekf   = pd.read_csv(config["ekf_csv"])
    df_truth = pd.read_csv(config["truth_csv"])
    print(f"  EKF   : {len(df_ekf)} rows")
    print(f"  Truth : {len(df_truth)} rows")

    t_ekf   = df_ekf["Time (s)"].values
    t_truth = df_truth["Time (s)"].values
    sort_idx = np.argsort(t_truth)
    t_sorted = t_truth[sort_idx]

    ekf_idxs, truth_idxs = [], []
    for i, t in enumerate(t_ekf):
        j = np.searchsorted(t_sorted, t)
        best_j, best_dt = None, float("inf")
        for cj in [j-1, j]:
            if 0 <= cj < len(t_sorted):
                dt = abs(t_sorted[cj] - t)
                if dt < best_dt:
                    best_j, best_dt = cj, dt
        if best_dt <= config["time_tol_s"]:
            ekf_idxs.append(i)
            truth_idxs.append(sort_idx[best_j])

    df_ekf_m   = df_ekf.iloc[ekf_idxs].reset_index(drop=True)
    df_truth_m = df_truth.iloc[truth_idxs].reset_index(drop=True)
    print(f"  Matched: {len(df_ekf_m)} records")
    return df_ekf_m, df_truth_m


def prepare_features(df_ekf, config):
    """Build normalised feature tensor from EKF data."""
    stats = pd.read_csv(config["norm_stats_csv"])
    feat_names = stats["Feature"].tolist()
    means = stats["Mean"].values.astype(np.float32)
    stds  = stats["Std"].values.astype(np.float32)

    missing = [f for f in feat_names if f not in df_ekf.columns]
    if missing:
        sys.exit(f"ERROR: EKF CSV missing feature columns: {missing}")

    X = df_ekf[feat_names].values.astype(np.float32)
    X_norm = (X - means) / stds
    return torch.tensor(X_norm, dtype=torch.float32), feat_names


# ======================================================
# 2. PREDICTION & CORRECTION
# ======================================================

@torch.no_grad()
def predict_corrections(model, X_tensor, device):
    """Run model inference, return corrections as numpy."""
    model.eval()
    X = X_tensor.to(device)
    preds = model(X).cpu().numpy()
    return preds


def apply_corrections(df_ekf, corrections):
    """Add predicted corrections to EKF state -> corrected orbit."""
    pos = ["X (km)", "Y (km)", "Z (km)"]
    vel = ["VX (km/s)", "VY (km/s)", "VZ (km/s)"]
    state_cols = pos + vel

    corrected = df_ekf[state_cols].values.copy()
    corrected += corrections  # corrections shape: (N, 6)

    df_corr = df_ekf[["Time (s)"]].copy()
    for i, col in enumerate(state_cols):
        df_corr[col] = corrected[:, i]
    return df_corr


# ======================================================
# 3. METRICS
# ======================================================

def compute_errors(df_ekf, df_corrected, df_truth):
    """Compute per-sample 3D position and velocity errors."""
    pos = ["X (km)", "Y (km)", "Z (km)"]
    vel = ["VX (km/s)", "VY (km/s)", "VZ (km/s)"]

    ekf_pos  = df_ekf[pos].values
    ekf_vel  = df_ekf[vel].values
    corr_pos = df_corrected[pos].values
    corr_vel = df_corrected[vel].values
    true_pos = df_truth[pos].values
    true_vel = df_truth[vel].values

    # 3D errors
    ekf_pos_err  = np.linalg.norm(ekf_pos - true_pos, axis=1) * 1000   # m
    ekf_vel_err  = np.linalg.norm(ekf_vel - true_vel, axis=1) * 1000   # m/s
    corr_pos_err = np.linalg.norm(corr_pos - true_pos, axis=1) * 1000  # m
    corr_vel_err = np.linalg.norm(corr_vel - true_vel, axis=1) * 1000  # m/s

    # Per-component errors (km, km/s)
    ekf_comp  = np.hstack([ekf_pos - true_pos, ekf_vel - true_vel])
    corr_comp = np.hstack([corr_pos - true_pos, corr_vel - true_vel])

    return {
        "ekf_pos_err": ekf_pos_err, "ekf_vel_err": ekf_vel_err,
        "corr_pos_err": corr_pos_err, "corr_vel_err": corr_vel_err,
        "ekf_comp": ekf_comp, "corr_comp": corr_comp,
    }


def print_metrics(errors, times):
    """Print RMSE, MAE, Max Error comparison table."""
    labels = [("3D Position (m)", "ekf_pos_err", "corr_pos_err"),
              ("3D Velocity (m/s)", "ekf_vel_err", "corr_vel_err")]

    comp_names = ["dx (km)", "dy (km)", "dz (km)",
                  "dvx (km/s)", "dvy (km/s)", "dvz (km/s)"]

    sep = "-" * 72
    print(f"\n  {'':30s}  {'RMSE':>12s}  {'MAE':>12s}  {'Max Error':>12s}")
    print(f"  {sep}")

    for label, ek, ck in labels:
        e = errors[ek]
        c = errors[ck]
        e_rmse = np.sqrt(np.mean(e**2))
        c_rmse = np.sqrt(np.mean(c**2))
        imp = (1 - c_rmse / e_rmse) * 100 if e_rmse > 0 else 0

        print(f"  EKF       {label:<20s}  {e_rmse:12.4f}  {np.mean(e):12.4f}  {np.max(e):12.4f}")
        print(f"  AI-Corr   {label:<20s}  {c_rmse:12.4f}  {np.mean(c):12.4f}  {np.max(c):12.4f}")
        print(f"  Improvement: {imp:+.1f}%")
        print(f"  {sep}")

    # Per-component
    print(f"\n  Per-component errors:")
    print(f"  {'Component':<16s}  {'EKF RMSE':>12s}  {'AI RMSE':>12s}  {'Improvement':>12s}")
    print(f"  {sep}")
    for i, name in enumerate(comp_names):
        e_rmse = np.sqrt(np.mean(errors["ekf_comp"][:, i]**2))
        c_rmse = np.sqrt(np.mean(errors["corr_comp"][:, i]**2))
        imp = (1 - c_rmse / e_rmse) * 100 if e_rmse > 0 else 0
        unit = "m" if i < 3 else "m/s"
        e_s = e_rmse * 1000 if i < 3 else e_rmse * 1000
        c_s = c_rmse * 1000 if i < 3 else c_rmse * 1000
        print(f"  {name:<16s}  {e_s:12.4f}  {c_s:12.4f}  {imp:+11.1f}%")
    print(f"  {sep}")


# ======================================================
# 4. PLOTS
# ======================================================

def plot_error_comparison(errors, times):
    """Bar chart comparing EKF vs AI-corrected RMSE."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, label, ek, ck, unit in [
        (axes[0], "3D Position", "ekf_pos_err", "corr_pos_err", "m"),
        (axes[1], "3D Velocity", "ekf_vel_err", "corr_vel_err", "m/s"),
    ]:
        e_rmse = np.sqrt(np.mean(errors[ek]**2))
        c_rmse = np.sqrt(np.mean(errors[ck]**2))
        e_mae  = np.mean(errors[ek])
        c_mae  = np.mean(errors[ck])
        e_max  = np.max(errors[ek])
        c_max  = np.max(errors[ck])

        x = np.arange(3)
        w = 0.35
        bars1 = ax.bar(x - w/2, [e_rmse, e_mae, e_max], w, label="EKF", color="orangered", alpha=0.8)
        bars2 = ax.bar(x + w/2, [c_rmse, c_mae, c_max], w, label="AI-Corrected", color="steelblue", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(["RMSE", "MAE", "Max"])
        ax.set_ylabel(f"Error ({unit})")
        ax.set_title(f"{label} Error")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2, axis="y")

        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7)

    fig.suptitle("EKF vs AI-Corrected Error Comparison", fontsize=13)
    fig.tight_layout()
    fig.savefig("eval_error_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Plot: eval_error_comparison.png")


def plot_error_timeseries(errors, times):
    """Time-series of position and velocity errors before/after AI."""
    t_min = times / 60.0
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    ax = axes[0]
    ax.plot(t_min, errors["ekf_pos_err"], linewidth=0.8, color="orangered",
            alpha=0.7, label="EKF")
    ax.plot(t_min, errors["corr_pos_err"], linewidth=0.8, color="steelblue",
            alpha=0.7, label="AI-Corrected")
    ax.set_ylabel("3D Position Error (m)")
    ax.set_title("Position Error: Before vs After AI Correction")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(t_min, errors["ekf_vel_err"], linewidth=0.8, color="orangered",
            alpha=0.7, label="EKF")
    ax.plot(t_min, errors["corr_vel_err"], linewidth=0.8, color="steelblue",
            alpha=0.7, label="AI-Corrected")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("3D Velocity Error (m/s)")
    ax.set_title("Velocity Error: Before vs After AI Correction")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("eval_error_timeseries.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Plot: eval_error_timeseries.png")


def plot_orbit_comparison(df_ekf, df_corrected, df_truth):
    """XY orbit plot: Truth vs EKF vs AI-Corrected."""
    fig, ax = plt.subplots(figsize=(9, 9))

    ax.plot(df_truth["X (km)"], df_truth["Y (km)"],
            linewidth=1.0, color="green", alpha=0.6, label="Truth")
    ax.plot(df_ekf["X (km)"], df_ekf["Y (km)"],
            linewidth=0.8, linestyle="--", color="orangered", alpha=0.6, label="EKF")
    ax.plot(df_corrected["X (km)"], df_corrected["Y (km)"],
            linewidth=0.8, linestyle="-.", color="steelblue", alpha=0.8, label="AI-Corrected")

    ax.scatter(0, 0, s=200, color="dodgerblue", zorder=5, label="Earth")
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_title("Orbit Comparison: Truth vs EKF vs AI-Corrected")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig("eval_orbit_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Plot: eval_orbit_comparison.png")


# ======================================================
# 5. CSV EXPORT
# ======================================================

def export_corrected(df_corrected, path):
    """Save corrected orbit to CSV."""
    df_corrected.to_csv(path, index=False)
    print(f"  Saved: {path}  ({len(df_corrected)} rows)")


# ======================================================
# MAIN
# ======================================================

if __name__ == "__main__":

    print("=" * 60)
    print("  Module 8.5 -- Evaluation & Orbit Correction")
    print("=" * 60)

    # 1. Load model
    print("\nStep 1 -- Loading trained model")
    model, info = load_trained_model(CONFIG["model_path"], CONFIG["device"])

    # 2. Load and match data
    print("\nStep 2 -- Loading and matching data")
    df_ekf, df_truth = load_and_match(CONFIG)

    # 3. Prepare features
    print("\nStep 3 -- Preparing normalised features")
    X_tensor, feat_names = prepare_features(df_ekf, CONFIG)
    print(f"  Features: {X_tensor.shape}  ({len(feat_names)} dims)")

    # 4. Predict corrections
    print("\nStep 4 -- Predicting correction vectors")
    corrections = predict_corrections(model, X_tensor, CONFIG["device"])
    print(f"  Corrections: {corrections.shape}")
    print(f"  Mean |correction|: pos={np.mean(np.abs(corrections[:,:3]))*1000:.2f} m, "
          f"vel={np.mean(np.abs(corrections[:,3:]))*1000:.4f} m/s")

    # 5. Apply corrections
    print("\nStep 5 -- Applying corrections to EKF orbit")
    df_corrected = apply_corrections(df_ekf, corrections)

    # 6. Compute errors
    print("\nStep 6 -- Computing error metrics")
    times = df_ekf["Time (s)"].values
    errors = compute_errors(df_ekf, df_corrected, df_truth)
    print_metrics(errors, times)

    # 7. Plots
    print("\nStep 7 -- Generating plots")
    plot_error_comparison(errors, times)
    plot_error_timeseries(errors, times)
    plot_orbit_comparison(df_ekf, df_corrected, df_truth)

    # 8. Export
    print("\nStep 8 -- Exporting corrected orbit")
    export_corrected(df_corrected, CONFIG["output_csv"])

    # Summary
    e_pos = np.sqrt(np.mean(errors["ekf_pos_err"]**2))
    c_pos = np.sqrt(np.mean(errors["corr_pos_err"]**2))
    imp = (1 - c_pos / e_pos) * 100 if e_pos > 0 else 0

    print("\n" + "=" * 60)
    print("  EVALUATION COMPLETE")
    print("=" * 60)
    print(f"  EKF position RMSE        : {e_pos:.4f} m")
    print(f"  AI-corrected RMSE        : {c_pos:.4f} m")
    print(f"  Improvement              : {imp:+.1f}%")
    print(f"  Corrected orbit saved    : {CONFIG['output_csv']}")
    print("=" * 60)

    print("\n[OK] Module 8.5 complete -- Evaluation & Orbit Correction.")
