"""
Module 8.1 — AI Training Dataset Generator
=============================================
Produces supervised learning datasets for an AI orbit-correction model.

Pipeline
--------
1. Load EKF estimated orbit   → ekf_estimated_orbit.csv
2. Load truth trajectory      → full_physics_orbit_dataset.csv
3. Match records by timestamp (nearest-neighbour within tolerance)
4. Build feature matrix and target labels
5. Normalise features (z-score standardisation)
6. Split into Train / Validation / Test
7. Export CSVs + normalisation statistics

Feature Architecture (16 features — recommended)
-------------------------------------------------
  EKF Position       :  x, y, z               (3)
  EKF Velocity       :  vx, vy, vz            (3)
  Radar Residuals    :  range, rate, az, el    (4)
  Covariance Diag.   :  P_xx … P_vzvz         (6)
                                         TOTAL  16

If the EKF CSV was generated before the Module 7 upgrade
(no covariance / residual columns), the generator falls back
to the basic 6-feature mode automatically.

Target Labels (6)
-----------------
  Δx  = True_x  − EKF_x        (km)
  Δy  = True_y  − EKF_y        (km)
  Δz  = True_z  − EKF_z        (km)
  Δvx = True_vx − EKF_vx       (km/s)
  Δvy = True_vy − EKF_vy       (km/s)
  Δvz = True_vz − EKF_vz       (km/s)

Input:   ekf_estimated_orbit.csv, full_physics_orbit_dataset.csv
Output:  train_dataset.csv, validation_dataset.csv, test_dataset.csv,
         normalisation_stats.csv, dataset_summary.txt,
         dataset_feature_distributions.png, dataset_target_distributions.png

Usage:
    python ai_dataset_generator.py

Dependencies:
    numpy, pandas, matplotlib, scikit-learn (optional — manual split used)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════
CONFIG = {
    # ── Input files ──
    "ekf_csv":   "ekf_estimated_orbit.csv",
    "truth_csv": "full_physics_orbit_dataset.csv",

    # ── Output files ──
    "train_csv":       "train_dataset.csv",
    "val_csv":         "validation_dataset.csv",
    "test_csv":        "test_dataset.csv",
    "norm_stats_csv":  "normalisation_stats.csv",
    "summary_txt":     "dataset_summary.txt",

    # ── Timestamp matching ──
    # Maximum allowable time gap (seconds) between an EKF record and
    # its nearest truth record.  Pairs outside this tolerance are
    # discarded — they indicate interpolation would be unreliable.
    "time_match_tol_s": 1.0,

    # ── Train / Val / Test split ratios ──
    # These are applied *chronologically* (no shuffling) to preserve
    # temporal structure — essential for time-series orbit data.
    "train_ratio": 0.70,
    "val_ratio":   0.15,
    "test_ratio":  0.15,

    # ── Random seed (for any stochastic operations) ──
    "random_seed": 42,
}


# ══════════════════════════════════════════════
# 1. DATA LOADING
# ══════════════════════════════════════════════

def load_csv(path, label):
    """Load a CSV file with basic validation."""
    if not os.path.exists(path):
        sys.exit(f"ERROR: {label} not found -> {path}\n"
                 f"       Run the upstream module first.")
    df = pd.read_csv(path)
    print(f"  Loaded {label}: {path}  ({len(df)} rows, {len(df.columns)} cols)")
    return df


def load_ekf_data(config):
    """Load EKF estimated orbit and detect available feature columns."""
    df = load_csv(config["ekf_csv"], "EKF estimates")

    # Required columns (always present)
    required = ["Time (s)", "X (km)", "Y (km)", "Z (km)",
                 "VX (km/s)", "VY (km/s)", "VZ (km/s)"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: EKF CSV missing required columns: {missing}")

    # Optional enhanced columns (from upgraded Module 7)
    cov_cols   = ["P_xx", "P_yy", "P_zz", "P_vxvx", "P_vyvy", "P_vzvz"]
    resid_cols = ["Resid Range (km)", "Resid Rate (km/s)",
                  "Resid Az (deg)", "Resid El (deg)"]

    has_cov   = all(c in df.columns for c in cov_cols)
    has_resid = all(c in df.columns for c in resid_cols)

    if has_cov and has_resid:
        print("  [OK] Enhanced 16-feature mode (pos + vel + residuals + covariance)")
        feature_mode = "enhanced"
    elif has_resid:
        print("  [OK] 10-feature mode (pos + vel + residuals, no covariance)")
        feature_mode = "resid_only"
    elif has_cov:
        print("  [OK] 12-feature mode (pos + vel + covariance, no residuals)")
        feature_mode = "cov_only"
    else:
        print("  [!!] Basic 6-feature mode (pos + vel only)")
        print("    Re-run Module 7 with the upgraded exporter for 16 features.")
        feature_mode = "basic"

    return df, feature_mode, cov_cols, resid_cols


def load_truth_data(config):
    """Load ground-truth orbit trajectory."""
    df = load_csv(config["truth_csv"], "Truth trajectory")
    required = ["Time (s)", "X (km)", "Y (km)", "Z (km)",
                 "VX (km/s)", "VY (km/s)", "VZ (km/s)"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: Truth CSV missing columns: {missing}")
    return df


# ══════════════════════════════════════════════
# 2. TIMESTAMP MATCHING
# ══════════════════════════════════════════════

def match_by_timestamp(df_ekf, df_truth, tol_s):
    """
    For each EKF record, find the nearest truth record by timestamp.

    Uses numpy searchsorted for O(N log M) performance rather than
    brute-force O(N·M).  Pairs whose time gap exceeds `tol_s` are
    discarded.

    Returns
    -------
    matched : list of (ekf_idx, truth_idx, dt) tuples
    """
    t_ekf   = df_ekf["Time (s)"].values
    t_truth = df_truth["Time (s)"].values

    # Ensure truth times are sorted (they should be, but be safe)
    sort_idx = np.argsort(t_truth)
    t_sorted = t_truth[sort_idx]

    matched = []
    for i, t in enumerate(t_ekf):
        # Binary search for nearest neighbour
        j = np.searchsorted(t_sorted, t)

        # Check j and j-1 for the closest match
        candidates = []
        if j < len(t_sorted):
            candidates.append(j)
        if j > 0:
            candidates.append(j - 1)

        best_j   = candidates[0]
        best_dt  = abs(t_sorted[best_j] - t)
        for cj in candidates[1:]:
            dt = abs(t_sorted[cj] - t)
            if dt < best_dt:
                best_j, best_dt = cj, dt

        if best_dt <= tol_s:
            truth_original_idx = sort_idx[best_j]
            matched.append((i, truth_original_idx, best_dt))

    return matched


# ══════════════════════════════════════════════
# 3. FEATURE & TARGET CONSTRUCTION
# ══════════════════════════════════════════════

def build_dataset(df_ekf, df_truth, matched, feature_mode, cov_cols, resid_cols):
    """
    Construct the feature matrix X and target matrix Y from matched records.

    Features (depending on mode):
        Basic (6):     EKF position + velocity
        Enhanced (16): EKF pos + vel + radar residuals + covariance diagonal

    Targets (always 6):
        Δposition = Truth − EKF   (km)
        Δvelocity = Truth − EKF   (km/s)
    """
    pos_cols = ["X (km)", "Y (km)", "Z (km)"]
    vel_cols = ["VX (km/s)", "VY (km/s)", "VZ (km/s)"]

    # Determine which feature columns to use
    feature_cols = pos_cols + vel_cols                         # always: 6
    if feature_mode in ("enhanced", "resid_only"):
        feature_cols += resid_cols                             # + 4 = 10
    if feature_mode in ("enhanced", "cov_only"):
        feature_cols += cov_cols                               # + 6 = 12 or 16

    target_names = ["dx (km)", "dy (km)", "dz (km)",
                    "dvx (km/s)", "dvy (km/s)", "dvz (km/s)"]

    n = len(matched)
    n_feat = len(feature_cols)

    X = np.zeros((n, n_feat))
    Y = np.zeros((n, 6))
    times = np.zeros(n)

    for k, (ei, ti, dt) in enumerate(matched):
        ekf_row   = df_ekf.iloc[ei]
        truth_row = df_truth.iloc[ti]

        # Features: pull from EKF row
        X[k] = [ekf_row[c] for c in feature_cols]

        # Targets: Truth − EKF (the correction the AI should learn)
        for j, (pc, vc) in enumerate(zip(pos_cols, vel_cols)):
            if j < 3:
                Y[k, j] = truth_row[pos_cols[j]] - ekf_row[pos_cols[j]]
            else:
                break
        for j in range(3):
            Y[k, j]   = truth_row[pos_cols[j]] - ekf_row[pos_cols[j]]
            Y[k, 3+j] = truth_row[vel_cols[j]] - ekf_row[vel_cols[j]]

        times[k] = ekf_row["Time (s)"]

    return X, Y, times, feature_cols, target_names


# ══════════════════════════════════════════════
# 4. NORMALISATION
# ══════════════════════════════════════════════

def normalise_features(X, feature_cols):
    """
    Z-score normalisation: x_norm = (x - μ) / σ

    Computed on the FULL dataset before splitting so that statistics
    are consistent across train/val/test.  In production, you would
    compute on train only and apply to val/test — noted in the summary.

    Returns
    -------
    X_norm    : normalised feature matrix
    stats_df  : DataFrame with mean and std per feature
    """
    means = X.mean(axis=0)
    stds  = X.std(axis=0)

    # Guard against zero-std columns (constant features)
    stds[stds == 0] = 1.0

    X_norm = (X - means) / stds

    stats_df = pd.DataFrame({
        "Feature":  feature_cols,
        "Mean":     means,
        "Std":      stds,
    })

    return X_norm, stats_df


# ══════════════════════════════════════════════
# 5. CHRONOLOGICAL SPLIT
# ══════════════════════════════════════════════

def chronological_split(X, Y, times, train_r, val_r, test_r):
    """
    Split data chronologically (no shuffling).

    For orbit estimation, temporal ordering matters:
    - The AI should generalise to future states, not randomly
      scattered states it may have already seen nearby in time.
    - This prevents data leakage from temporal correlation.

    Returns
    -------
    dict with keys "train", "val", "test", each containing
    (X, Y, times) arrays.
    """
    n = len(X)
    n_train = int(n * train_r)
    n_val   = int(n * (train_r + val_r))
    # test = remainder

    splits = {
        "train": (X[:n_train],      Y[:n_train],      times[:n_train]),
        "val":   (X[n_train:n_val],  Y[n_train:n_val],  times[n_train:n_val]),
        "test":  (X[n_val:],         Y[n_val:],         times[n_val:]),
    }

    return splits


# ══════════════════════════════════════════════
# 6. CSV EXPORT
# ══════════════════════════════════════════════

def save_split(X, Y, times, feature_cols, target_names, path, label):
    """Save one split (train/val/test) to CSV."""
    cols = ["Time (s)"] + feature_cols + target_names
    data = np.column_stack([times.reshape(-1, 1), X, Y])
    df = pd.DataFrame(data, columns=cols)
    df.to_csv(path, index=False)
    print(f"  Saved {label:12s} -> {path}  ({len(df)} samples, "
          f"{len(feature_cols)} features, {len(target_names)} targets)")
    return df


# ══════════════════════════════════════════════
# 7. DIAGNOSTIC PLOTS
# ══════════════════════════════════════════════

def plot_feature_distributions(X, feature_cols, title_suffix=""):
    """Histogram grid of normalised feature distributions."""
    n_feat = X.shape[1]
    n_cols_plot = min(4, n_feat)
    n_rows_plot = int(np.ceil(n_feat / n_cols_plot))

    fig, axes = plt.subplots(n_rows_plot, n_cols_plot,
                              figsize=(4 * n_cols_plot, 3 * n_rows_plot))
    axes = np.atleast_2d(axes).flatten()

    for i in range(n_feat):
        ax = axes[i]
        ax.hist(X[:, i], bins=50, color="steelblue", alpha=0.7, edgecolor="white")
        ax.set_title(feature_cols[i], fontsize=9)
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.2)

    for i in range(n_feat, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(f"Feature Distributions (Normalised){title_suffix}", fontsize=13)
    fig.tight_layout()
    fig.savefig("dataset_feature_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Plot: dataset_feature_distributions.png")


def plot_target_distributions(Y, target_names):
    """Histogram grid of target (correction) distributions."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))

    colors = ["crimson", "darkorange", "teal", "darkorchid", "steelblue", "goldenrod"]
    for i, (ax, name, col) in enumerate(zip(axes.flatten(), target_names, colors)):
        ax.hist(Y[:, i], bins=50, color=col, alpha=0.7, edgecolor="white")
        ax.set_title(name, fontsize=10)
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.2)

        # Annotate stats
        mu, sig = Y[:, i].mean(), Y[:, i].std()
        ax.axvline(mu, color="black", linestyle="--", alpha=0.5)
        ax.text(0.97, 0.95, f"μ={mu:.2e}\nσ={sig:.2e}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8, family="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    fig.suptitle("Target Correction Distributions", fontsize=13)
    fig.tight_layout()
    fig.savefig("dataset_target_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Plot: dataset_target_distributions.png")


def plot_corrections_over_time(times, Y, target_names, splits):
    """Time-series of corrections with train/val/test boundaries marked."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    colors = ["crimson", "darkorange", "teal", "darkorchid", "steelblue", "goldenrod"]

    t_min = times / 60.0

    for i, (ax, name, col) in enumerate(zip(axes.flatten(), target_names, colors)):
        ax.scatter(t_min, Y[:, i], s=1, alpha=0.5, color=col)
        ax.set_ylabel(name, fontsize=9)
        ax.grid(True, alpha=0.2)

        # Mark split boundaries
        n_train = len(splits["train"][0])
        n_val   = len(splits["val"][0])
        if n_train < len(times):
            ax.axvline(t_min[n_train], color="green", linestyle="--",
                       alpha=0.6, label="Train|Val")
        if n_train + n_val < len(times):
            ax.axvline(t_min[n_train + n_val], color="red", linestyle="--",
                       alpha=0.6, label="Val|Test")

    axes[-1, 0].set_xlabel("Time (min)")
    axes[-1, 1].set_xlabel("Time (min)")
    axes[0, 0].legend(fontsize=8, loc="upper right")

    fig.suptitle("Target Corrections Over Time (with Split Boundaries)", fontsize=13)
    fig.tight_layout()
    fig.savefig("dataset_corrections_timeseries.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Plot: dataset_corrections_timeseries.png")


# ══════════════════════════════════════════════
# 8. SUMMARY REPORT
# ══════════════════════════════════════════════

def write_summary(config, feature_mode, feature_cols, target_names,
                   n_total, splits, stats_df, matched_stats):
    """Write a human-readable summary of the generated dataset."""
    path = config["summary_txt"]

    lines = []
    lines.append("=" * 65)
    lines.append("  Module 8.1 -- AI Training Dataset Summary")
    lines.append("=" * 65)
    lines.append("")
    lines.append(f"  Feature mode      : {feature_mode}")
    lines.append(f"  Number of features: {len(feature_cols)}")
    lines.append(f"  Number of targets : {len(target_names)}")
    lines.append(f"  Total samples     : {n_total}")
    lines.append("")

    lines.append("  Features:")
    for i, f in enumerate(feature_cols):
        lines.append(f"    [{i:2d}] {f}")
    lines.append("")

    lines.append("  Targets:")
    for i, t in enumerate(target_names):
        lines.append(f"    [{i:2d}] {t}")
    lines.append("")

    lines.append("  Split (chronological):")
    for name, (X, Y, t) in splits.items():
        pct = len(X) / n_total * 100 if n_total > 0 else 0
        lines.append(f"    {name:12s}: {len(X):6d} samples  ({pct:5.1f}%)"
                     f"  t=[{t[0]/60:.1f}, {t[-1]/60:.1f}] min"
                     if len(t) > 0 else f"    {name:12s}: 0 samples")
    lines.append("")

    lines.append("  Timestamp matching:")
    lines.append(f"    Tolerance        : {config['time_match_tol_s']:.2f} s")
    lines.append(f"    Matched pairs    : {matched_stats['n_matched']}")
    lines.append(f"    Discarded (gap)  : {matched_stats['n_discarded']}")
    lines.append(f"    Mean dt          : {matched_stats['mean_dt']:.4f} s")
    lines.append(f"    Max  dt          : {matched_stats['max_dt']:.4f} s")
    lines.append("")

    lines.append("  Normalisation statistics (z-score):")
    lines.append(f"    Saved to: {config['norm_stats_csv']}")
    lines.append("")
    lines.append("  NOTE: Normalisation was computed on the full dataset for")
    lines.append("        consistency.  For strict ML practice, re-compute on")
    lines.append("        the training split only and apply to val/test.")
    lines.append("")

    lines.append("  Output files:")
    lines.append(f"    {config['train_csv']}")
    lines.append(f"    {config['val_csv']}")
    lines.append(f"    {config['test_csv']}")
    lines.append(f"    {config['norm_stats_csv']}")
    lines.append(f"    {config['summary_txt']}")
    lines.append(f"    dataset_feature_distributions.png")
    lines.append(f"    dataset_target_distributions.png")
    lines.append(f"    dataset_corrections_timeseries.png")
    lines.append("")
    lines.append("=" * 65)

    text = "\n".join(lines)

    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\n  Summary: {path}")

    return text


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════

if __name__ == "__main__":

    print("=" * 65)
    print("  Module 8.1 -- AI Training Dataset Generator")
    print("=" * 65)
    print()

    # ── 1. Load data ──
    print("Step 1 -- Loading input data")
    df_ekf, feature_mode, cov_cols, resid_cols = load_ekf_data(CONFIG)
    df_truth = load_truth_data(CONFIG)
    print()

    # ── 2. Match timestamps ──
    print("Step 2 -- Matching EKF records to truth by timestamp")
    matched = match_by_timestamp(df_ekf, df_truth, CONFIG["time_match_tol_s"])
    n_ekf = len(df_ekf)
    n_matched = len(matched)
    n_discarded = n_ekf - n_matched

    if n_matched == 0:
        sys.exit("ERROR: No records matched within tolerance. "
                 f"Check that both CSVs share a common 'Time (s)' range.\n"
                 f"  EKF time range : [{df_ekf['Time (s)'].min():.1f}, "
                 f"{df_ekf['Time (s)'].max():.1f}] s\n"
                 f"  Truth time range: [{df_truth['Time (s)'].min():.1f}, "
                 f"{df_truth['Time (s)'].max():.1f}] s")

    dts = np.array([m[2] for m in matched])
    matched_stats = {
        "n_matched":   n_matched,
        "n_discarded": n_discarded,
        "mean_dt":     dts.mean(),
        "max_dt":      dts.max(),
    }
    print(f"  Matched {n_matched}/{n_ekf} EKF records  "
          f"(discarded {n_discarded}, mean dt={dts.mean():.4f} s, "
          f"max dt={dts.max():.4f} s)")
    print()

    # ── 3. Build features and targets ──
    print("Step 3 -- Constructing feature matrix and target labels")
    X, Y, times, feature_cols, target_names = build_dataset(
        df_ekf, df_truth, matched, feature_mode, cov_cols, resid_cols
    )
    print(f"  Feature matrix X: {X.shape}  (samples x features)")
    print(f"  Target  matrix Y: {Y.shape}  (samples x targets)")
    print()

    # ── 4. Normalise ──
    print("Step 4 -- Normalising features (z-score)")
    X_norm, stats_df = normalise_features(X, feature_cols)
    stats_df.to_csv(CONFIG["norm_stats_csv"], index=False)
    print(f"  Saved normalisation statistics -> {CONFIG['norm_stats_csv']}")
    print("  Feature statistics:")
    for _, row in stats_df.iterrows():  # noqa
        print(f"    {row['Feature']:28s}  mean={row['Mean']:+12.6e}  std={row['Std']:12.6e}")
    print()

    # ── 5. Chronological split ──
    print("Step 5 -- Chronological train/val/test split")
    splits = chronological_split(
        X_norm, Y, times,
        CONFIG["train_ratio"], CONFIG["val_ratio"], CONFIG["test_ratio"]
    )
    for name, (Xs, Ys, ts) in splits.items():
        if len(ts) > 0:
            print(f"  {name:12s}: {len(Xs):6d} samples  "
                  f"t=[{ts[0]/60:.1f}, {ts[-1]/60:.1f}] min")
        else:
            print(f"  {name:12s}: 0 samples")
    print()

    # ── 6. Export CSVs ──
    print("Step 6 -- Saving dataset splits")
    save_split(*splits["train"], feature_cols, target_names,
               CONFIG["train_csv"], "Train")
    save_split(*splits["val"], feature_cols, target_names,
               CONFIG["val_csv"], "Validation")
    save_split(*splits["test"], feature_cols, target_names,
               CONFIG["test_csv"], "Test")
    print()

    # ── 7. Diagnostic plots ──
    print("Step 7 -- Generating diagnostic plots")
    plot_feature_distributions(X_norm, feature_cols)
    plot_target_distributions(Y, target_names)
    plot_corrections_over_time(times, Y, target_names, splits)
    print()

    # ── 8. Summary ──
    summary = write_summary(
        CONFIG, feature_mode, feature_cols, target_names,
        n_matched, splits, stats_df, matched_stats
    )
    print(summary)

    print("\n[OK] Module 8.1 complete -- AI Training Dataset Generator.")
