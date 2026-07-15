"""
Module 8.2 -- PyTorch Dataset & DataLoader
=============================================
Wraps the CSV datasets produced by Module 8.1 into PyTorch-native
Dataset and DataLoader objects ready for neural network training.

Architecture
------------
    OrbitCorrectionDataset(Dataset)
        - Reads a single CSV split (train / val / test)
        - Auto-detects feature vs. target columns from header
        - Stores data as float32 tensors on the chosen device
        - Provides __getitem__ and __len__ for standard PyTorch use

    build_dataloaders(config) -> dict
        - Loads all three splits
        - Returns {"train": DataLoader, "val": DataLoader, "test": DataLoader}

CSV Format (from Module 8.1)
-----------------------------
    Column 0        : Time (s)       -- metadata, not used as feature
    Columns 1..N    : input features (normalised)
    Columns N+1..N+6: target labels  (dx, dy, dz, dvx, dvy, dvz)

Input:   train_dataset.csv, validation_dataset.csv, test_dataset.csv
Output:  (in-memory tensors -- no files written)

Usage:
    python pytorch_dataset.py                     # standalone verification
    from pytorch_dataset import build_dataloaders  # import in training script

Dependencies:
    torch, pandas, numpy
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


# ======================================================
# CONFIGURATION
# ======================================================
CONFIG = {
    # -- Input CSV files (from Module 8.1) --
    "train_csv":  "train_dataset.csv",
    "val_csv":    "validation_dataset.csv",
    "test_csv":   "test_dataset.csv",

    # -- DataLoader settings --
    "batch_size":       32,
    "shuffle_train":    True,     # shuffle training batches each epoch
    "shuffle_val":      False,    # keep val/test ordered for diagnostics
    "shuffle_test":     False,
    "num_workers":      0,        # 0 = main process (safest on Windows)
    "pin_memory":       False,    # set True if using CUDA
    "drop_last_train":  False,    # drop incomplete final batch in training

    # -- Target column names (must match Module 8.1 output) --
    "target_cols": ["dx (km)", "dy (km)", "dz (km)",
                    "dvx (km/s)", "dvy (km/s)", "dvz (km/s)"],

    # -- Metadata columns to exclude from features --
    "meta_cols": ["Time (s)"],

    # -- Device --
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


# ======================================================
# 1. DATASET CLASS
# ======================================================

class OrbitCorrectionDataset(Dataset):
    """
    PyTorch Dataset for orbit correction supervised learning.

    Each sample is a (features, targets) pair of float32 tensors:
        features : shape (N_feat,)  -- normalised EKF state + residuals + covariance
        targets  : shape (6,)      -- correction vector (Truth - EKF)

    The dataset auto-detects the number of features from the CSV header
    by subtracting the known target and metadata columns.

    Parameters
    ----------
    csv_path    : str   -- path to a split CSV from Module 8.1
    target_cols : list  -- column names of the 6 target labels
    meta_cols   : list  -- column names to exclude (e.g. "Time (s)")
    device      : str   -- "cpu" or "cuda"
    """

    def __init__(self, csv_path, target_cols, meta_cols, device="cpu"):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Dataset CSV not found: {csv_path}\n"
                f"Run Module 8.1 (ai_dataset_generator.py) first."
            )

        df = pd.read_csv(csv_path)
        self.csv_path = csv_path
        self.n_samples = len(df)

        # -- Identify feature columns --
        all_cols = list(df.columns)
        excluded = set(target_cols) | set(meta_cols)
        self.feature_cols = [c for c in all_cols if c not in excluded]
        self.target_cols  = target_cols
        self.n_features   = len(self.feature_cols)
        self.n_targets    = len(self.target_cols)

        # -- Validate target columns exist --
        missing_targets = [c for c in target_cols if c not in all_cols]
        if missing_targets:
            raise ValueError(
                f"Target columns not found in {csv_path}: {missing_targets}\n"
                f"Available columns: {all_cols}"
            )

        # -- Extract numpy arrays and convert to tensors --
        X_np = df[self.feature_cols].values.astype(np.float32)
        Y_np = df[self.target_cols].values.astype(np.float32)

        self.features = torch.tensor(X_np, dtype=torch.float32, device=device)
        self.targets  = torch.tensor(Y_np, dtype=torch.float32, device=device)

        # -- Also store timestamps for diagnostics (not used in training) --
        if "Time (s)" in all_cols:
            self.times = torch.tensor(
                df["Time (s)"].values.astype(np.float32),
                dtype=torch.float32, device=device
            )
        else:
            self.times = None

    def __len__(self):
        """Return number of samples."""
        return self.n_samples

    def __getitem__(self, idx):
        """
        Return a single (features, targets) pair.

        Parameters
        ----------
        idx : int or slice

        Returns
        -------
        features : tensor, shape (N_feat,)
        targets  : tensor, shape (6,)
        """
        return self.features[idx], self.targets[idx]

    def get_feature_names(self):
        """Return ordered list of feature column names."""
        return self.feature_cols

    def get_target_names(self):
        """Return ordered list of target column names."""
        return self.target_cols

    def summary(self):
        """Return a formatted summary string."""
        lines = [
            f"OrbitCorrectionDataset",
            f"  Source    : {self.csv_path}",
            f"  Samples   : {self.n_samples}",
            f"  Features  : {self.n_features}  {self.feature_cols}",
            f"  Targets   : {self.n_targets}  {self.target_cols}",
            f"  X shape   : {tuple(self.features.shape)}",
            f"  Y shape   : {tuple(self.targets.shape)}",
            f"  dtype     : {self.features.dtype}",
            f"  device    : {self.features.device}",
        ]
        return "\n".join(lines)


# ======================================================
# 2. DATALOADER FACTORY
# ======================================================

def build_dataloaders(config=None):
    """
    Build DataLoaders for all three splits (train, val, test).

    Parameters
    ----------
    config : dict, optional
        Override default CONFIG.  Keys used:
            train_csv, val_csv, test_csv,
            batch_size, shuffle_train, shuffle_val, shuffle_test,
            num_workers, pin_memory, drop_last_train,
            target_cols, meta_cols, device

    Returns
    -------
    loaders  : dict  {"train": DataLoader, "val": DataLoader, "test": DataLoader}
    datasets : dict  {"train": Dataset, "val": Dataset, "test": Dataset}
    """
    if config is None:
        config = CONFIG

    device      = config["device"]
    target_cols = config["target_cols"]
    meta_cols   = config["meta_cols"]
    batch_size  = config["batch_size"]

    # -- Load datasets --
    ds_train = OrbitCorrectionDataset(config["train_csv"], target_cols, meta_cols, device)
    ds_val   = OrbitCorrectionDataset(config["val_csv"],   target_cols, meta_cols, device)
    ds_test  = OrbitCorrectionDataset(config["test_csv"],  target_cols, meta_cols, device)

    # -- Wrap in DataLoaders --
    # NOTE: when data is already on GPU (device="cuda"), pin_memory should
    # be False and num_workers should be 0 to avoid unnecessary copies.
    use_pin    = config["pin_memory"] and device == "cpu"
    use_workers = config["num_workers"] if device == "cpu" else 0

    dl_train = DataLoader(
        ds_train,
        batch_size  = batch_size,
        shuffle     = config["shuffle_train"],
        num_workers = use_workers,
        pin_memory  = use_pin,
        drop_last   = config["drop_last_train"],
    )

    dl_val = DataLoader(
        ds_val,
        batch_size  = batch_size,
        shuffle     = config["shuffle_val"],
        num_workers = use_workers,
        pin_memory  = use_pin,
        drop_last   = False,
    )

    dl_test = DataLoader(
        ds_test,
        batch_size  = batch_size,
        shuffle     = config["shuffle_test"],
        num_workers = use_workers,
        pin_memory  = use_pin,
        drop_last   = False,
    )

    datasets = {"train": ds_train, "val": ds_val, "test": ds_test}
    loaders  = {"train": dl_train, "val": dl_val, "test": dl_test}

    return loaders, datasets


# ======================================================
# 3. SHAPE VERIFICATION
# ======================================================

def verify_shapes(loaders, datasets, config):
    """
    Exhaustive shape and type verification of all tensors.

    Checks:
        1. Dataset tensor shapes match (N, n_feat) and (N, 6)
        2. Batch shapes from DataLoader match (batch_size, n_feat) and (batch_size, 6)
        3. All tensors are float32
        4. No NaN or Inf values
        5. Feature/target counts are consistent across splits
    """
    batch_size = config["batch_size"]
    errors = []
    warnings = []

    print("\n  Shape verification:")
    print("  " + "-" * 55)

    # -- Check each split --
    for name in ["train", "val", "test"]:
        ds = datasets[name]
        dl = loaders[name]

        # Dataset-level checks
        n_feat = ds.n_features
        n_targ = ds.n_targets
        n_samp = len(ds)

        print(f"\n  [{name.upper()}]")
        print(f"    Samples     : {n_samp}")
        print(f"    Features    : {n_feat}")
        print(f"    Targets     : {n_targ}")
        print(f"    X tensor    : {tuple(ds.features.shape)}  {ds.features.dtype}")
        print(f"    Y tensor    : {tuple(ds.targets.shape)}  {ds.targets.dtype}")

        # Shape consistency
        if ds.features.shape != (n_samp, n_feat):
            errors.append(f"{name}: X shape {ds.features.shape} != expected ({n_samp}, {n_feat})")
        if ds.targets.shape != (n_samp, n_targ):
            errors.append(f"{name}: Y shape {ds.targets.shape} != expected ({n_samp}, {n_targ})")

        # dtype check
        if ds.features.dtype != torch.float32:
            errors.append(f"{name}: X dtype is {ds.features.dtype}, expected float32")
        if ds.targets.dtype != torch.float32:
            errors.append(f"{name}: Y dtype is {ds.targets.dtype}, expected float32")

        # NaN / Inf check
        x_nan = torch.isnan(ds.features).sum().item()
        y_nan = torch.isnan(ds.targets).sum().item()
        x_inf = torch.isinf(ds.features).sum().item()
        y_inf = torch.isinf(ds.targets).sum().item()

        if x_nan > 0:
            errors.append(f"{name}: X contains {x_nan} NaN values")
        if y_nan > 0:
            errors.append(f"{name}: Y contains {y_nan} NaN values")
        if x_inf > 0:
            warnings.append(f"{name}: X contains {x_inf} Inf values")
        if y_inf > 0:
            warnings.append(f"{name}: Y contains {y_inf} Inf values")

        print(f"    NaN check   : X={x_nan} Y={y_nan}  {'PASS' if x_nan + y_nan == 0 else 'FAIL'}")
        print(f"    Inf check   : X={x_inf} Y={y_inf}  {'PASS' if x_inf + y_inf == 0 else 'WARN' if x_inf + y_inf > 0 else 'PASS'}")

        # DataLoader batch check (first batch)
        if n_samp > 0:
            batch_x, batch_y = next(iter(dl))
            expected_bs = min(batch_size, n_samp)
            print(f"    Batch X     : {tuple(batch_x.shape)}  (expected ({expected_bs}, {n_feat}))")
            print(f"    Batch Y     : {tuple(batch_y.shape)}  (expected ({expected_bs}, {n_targ}))")

            if batch_x.shape[1] != n_feat:
                errors.append(f"{name}: batch X dim1 = {batch_x.shape[1]} != {n_feat}")
            if batch_y.shape[1] != n_targ:
                errors.append(f"{name}: batch Y dim1 = {batch_y.shape[1]} != {n_targ}")
            if batch_x.shape[0] > batch_size:
                errors.append(f"{name}: batch X dim0 = {batch_x.shape[0]} > batch_size {batch_size}")

    # -- Cross-split consistency --
    n_feats = [datasets[s].n_features for s in ["train", "val", "test"]]
    if len(set(n_feats)) > 1:
        errors.append(f"Feature count mismatch across splits: {n_feats}")

    n_targs = [datasets[s].n_targets for s in ["train", "val", "test"]]
    if len(set(n_targs)) > 1:
        errors.append(f"Target count mismatch across splits: {n_targs}")

    # -- Report --
    print(f"\n  {'=' * 55}")
    if errors:
        print(f"  VERIFICATION FAILED -- {len(errors)} error(s):")
        for e in errors:
            print(f"    [ERROR] {e}")
    else:
        print(f"  VERIFICATION PASSED")

    if warnings:
        print(f"  {len(warnings)} warning(s):")
        for w in warnings:
            print(f"    [WARN]  {w}")

    print(f"  {'=' * 55}")

    return len(errors) == 0


# ======================================================
# 4. STATISTICS SUMMARY
# ======================================================

def print_data_statistics(datasets):
    """Print per-feature and per-target statistics for each split."""
    for name in ["train", "val", "test"]:
        ds = datasets[name]
        print(f"\n  [{name.upper()}] Feature statistics (normalised):")
        for i, col in enumerate(ds.feature_cols):
            vals = ds.features[:, i]
            print(f"    {col:28s}  "
                  f"min={vals.min().item():+10.4f}  "
                  f"max={vals.max().item():+10.4f}  "
                  f"mean={vals.mean().item():+10.4f}  "
                  f"std={vals.std().item():8.4f}")

        print(f"\n  [{name.upper()}] Target statistics:")
        for i, col in enumerate(ds.target_cols):
            vals = ds.targets[:, i]
            print(f"    {col:28s}  "
                  f"min={vals.min().item():+12.6e}  "
                  f"max={vals.max().item():+12.6e}  "
                  f"mean={vals.mean().item():+12.6e}  "
                  f"std={vals.std().item():12.6e}")


# ======================================================
# 5. ITERATION DEMO
# ======================================================

def demo_iteration(loaders, n_batches=3):
    """Walk through a few training batches to demonstrate DataLoader usage."""
    print(f"\n  Training iteration demo (first {n_batches} batches):")
    print("  " + "-" * 55)

    dl_train = loaders["train"]
    for i, (batch_x, batch_y) in enumerate(dl_train):
        if i >= n_batches:
            break
        print(f"    Batch {i}: X={tuple(batch_x.shape)}  Y={tuple(batch_y.shape)}  "
              f"X[0,:3]={batch_x[0, :3].tolist()}  Y[0,:3]={batch_y[0, :3].tolist()}")

    total_batches = len(dl_train)
    print(f"\n    Total batches per epoch: {total_batches}")
    print(f"    Total samples: {len(dl_train.dataset)}")


# ======================================================
# MAIN -- Standalone verification
# ======================================================

if __name__ == "__main__":

    print("=" * 60)
    print("  Module 8.2 -- PyTorch Dataset & DataLoader")
    print("=" * 60)
    print()

    print(f"  PyTorch version : {torch.__version__}")
    print(f"  CUDA available  : {torch.cuda.is_available()}")
    print(f"  Device          : {CONFIG['device']}")
    print(f"  Batch size      : {CONFIG['batch_size']}")
    print()

    # -- Build loaders --
    print("Step 1 -- Loading datasets and building DataLoaders")
    loaders, datasets = build_dataloaders(CONFIG)

    for name in ["train", "val", "test"]:
        print(f"\n  {datasets[name].summary()}")

    # -- Verify shapes --
    print("\nStep 2 -- Verifying tensor shapes and data integrity")
    passed = verify_shapes(loaders, datasets, CONFIG)

    # -- Statistics --
    print("\nStep 3 -- Data statistics")
    print_data_statistics(datasets)

    # -- Demo iteration --
    print("\nStep 4 -- DataLoader iteration demo")
    demo_iteration(loaders, n_batches=3)

    # -- Final summary --
    ds_train = datasets["train"]
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Input dimensionality  : {ds_train.n_features}")
    print(f"  Output dimensionality : {ds_train.n_targets}")
    print(f"  Training samples      : {len(datasets['train'])}")
    print(f"  Validation samples    : {len(datasets['val'])}")
    print(f"  Test samples          : {len(datasets['test'])}")
    print(f"  Batch size            : {CONFIG['batch_size']}")
    print(f"  Batches per epoch     : {len(loaders['train'])}")
    print(f"  Verification          : {'PASSED' if passed else 'FAILED'}")
    print("=" * 60)

    print("\n[OK] Module 8.2 complete -- PyTorch Dataset & DataLoader.")
