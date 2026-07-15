"""
Module 8.4 -- Training Pipeline
==================================
Complete PyTorch training loop for the orbit correction neural network.

Pipeline
--------
    1. Load datasets and build DataLoaders     (Module 8.2)
    2. Build model                             (Module 8.3)
    3. Train with Adam + MSE loss
    4. ReduceLROnPlateau scheduler
    5. Early stopping on validation loss
    6. Save best model checkpoint
    7. Export training history CSV + loss plot

Input:   train_dataset.csv, validation_dataset.csv, test_dataset.csv
Output:  orbit_correction_model.pt   -- best model weights
         training_history.csv        -- epoch-level metrics
         training_loss.png           -- train vs val loss curve

Usage:
    python train_model.py

Dependencies:
    torch, pandas, matplotlib,
    pytorch_dataset (Module 8.2), orbit_correction_net (Module 8.3)
"""

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pytorch_dataset import build_dataloaders, CONFIG as DS_CONFIG
from orbit_correction_net import OrbitCorrectionNet, build_model


# ======================================================
# CONFIGURATION
# ======================================================
CONFIG = {
    # -- Training --
    "epochs":             500,
    "learning_rate":      1e-3,
    "weight_decay":       1e-5,     # L2 regularisation

    # -- Scheduler (ReduceLROnPlateau) --
    "scheduler_factor":   0.5,      # multiply LR by this on plateau
    "scheduler_patience": 15,       # epochs to wait before reducing
    "scheduler_min_lr":   1e-7,     # lower bound on LR

    # -- Early stopping --
    "early_stop_patience": 40,      # epochs without improvement before stop
    "early_stop_min_delta": 1e-8,   # minimum change to qualify as improvement

    # -- Checkpointing --
    "model_path":         "orbit_correction_model.pt",
    "history_csv":        "training_history.csv",
    "loss_plot":          "training_loss.png",

    # -- Device --
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


# ======================================================
# 1. EARLY STOPPING
# ======================================================

class EarlyStopping:
    """
    Stop training when validation loss stops improving.

    Tracks the best validation loss and counts consecutive epochs
    without improvement.  When patience is exhausted, signals stop.

    Parameters
    ----------
    patience  : int    -- epochs to wait after last improvement
    min_delta : float  -- minimum change to qualify as improvement
    """

    def __init__(self, patience=40, min_delta=1e-8):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


# ======================================================
# 2. TRAINING LOOP
# ======================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Run one training epoch.

    Returns
    -------
    avg_loss : float -- mean MSE loss over all batches
    """
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        # Forward
        y_pred = model(batch_x)
        loss   = criterion(y_pred, batch_y)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """
    Evaluate model on a dataset (validation or test).

    Returns
    -------
    avg_loss : float -- mean MSE loss
    """
    model.eval()
    total_loss = 0.0
    n_batches  = 0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        y_pred = model(batch_x)
        loss   = criterion(y_pred, batch_y)

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


# ======================================================
# 3. FULL TRAINING PIPELINE
# ======================================================

def train(model, loaders, config):
    """
    Complete training pipeline with scheduler, early stopping,
    and best-model checkpointing.

    Parameters
    ----------
    model   : OrbitCorrectionNet
    loaders : dict with "train", "val", "test" DataLoaders
    config  : dict (training hyperparameters)

    Returns
    -------
    history : list of dicts (one per epoch)
    """
    device    = config["device"]
    model     = model.to(device)
    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config["scheduler_factor"],
        patience=config["scheduler_patience"],
        min_lr=config["scheduler_min_lr"],
    )

    early_stop = EarlyStopping(
        patience=config["early_stop_patience"],
        min_delta=config["early_stop_min_delta"],
    )

    # -- Training state --
    history     = []
    best_val    = float("inf")
    best_epoch  = 0
    start_time  = time.time()

    print(f"\n  Training configuration:")
    print(f"    Epochs          : {config['epochs']}")
    print(f"    Learning rate   : {config['learning_rate']}")
    print(f"    Weight decay    : {config['weight_decay']}")
    print(f"    Scheduler       : ReduceLROnPlateau (factor={config['scheduler_factor']}, "
          f"patience={config['scheduler_patience']})")
    print(f"    Early stopping  : patience={config['early_stop_patience']}")
    print(f"    Device          : {device}")
    print(f"    Parameters      : {model.count_parameters():,d}")
    print()
    print(f"  {'Epoch':>6s}  {'Train Loss':>12s}  {'Val Loss':>12s}  "
          f"{'LR':>10s}  {'Best':>5s}  {'Time':>6s}")
    print(f"  {'-'*60}")

    for epoch in range(1, config["epochs"] + 1):
        epoch_start = time.time()

        # Train
        train_loss = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device
        )

        # Validate
        val_loss = evaluate(model, loaders["val"], criterion, device)

        # Scheduler step
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # Track best model
        is_best = val_loss < best_val
        if is_best:
            best_val   = val_loss
            best_epoch = epoch
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss":    val_loss,
                "train_loss":  train_loss,
                "input_dim":   model.input_dim,
                "hidden_dims": model.hidden_dims,
                "output_dim":  model.output_dim,
            }, config["model_path"])

        # Record history
        elapsed = time.time() - epoch_start
        history.append({
            "epoch":      epoch,
            "train_loss": train_loss,
            "val_loss":   val_loss,
            "lr":         current_lr,
            "is_best":    is_best,
            "time_s":     elapsed,
        })

        # Print progress
        if epoch <= 5 or epoch % 10 == 0 or is_best or epoch == config["epochs"]:
            marker = " *" if is_best else ""
            print(f"  {epoch:6d}  {train_loss:12.6e}  {val_loss:12.6e}  "
                  f"{current_lr:10.2e}  {marker:>5s}  {elapsed:5.2f}s")

        # Early stopping check
        if early_stop(val_loss):
            print(f"\n  Early stopping at epoch {epoch} "
                  f"(no improvement for {config['early_stop_patience']} epochs)")
            break

    total_time = time.time() - start_time

    print(f"\n  {'='*60}")
    print(f"  Training complete")
    print(f"    Total time     : {total_time:.1f}s")
    print(f"    Best epoch     : {best_epoch}")
    print(f"    Best val loss  : {best_val:.6e}")
    print(f"    Final LR       : {current_lr:.2e}")
    print(f"    Model saved    : {config['model_path']}")
    print(f"  {'='*60}")

    return history


# ======================================================
# 4. HISTORY EXPORT
# ======================================================

def save_history(history, path):
    """Save training history to CSV."""
    df = pd.DataFrame(history)
    df.to_csv(path, index=False)
    print(f"  Saved training history -> {path}  ({len(df)} epochs)")
    return df


# ======================================================
# 5. LOSS PLOT
# ======================================================

def plot_loss(history, path):
    """Plot train and validation loss curves."""
    epochs     = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss   = [h["val_loss"] for h in history]

    # Find best epoch
    best_idx   = np.argmin(val_loss)
    best_epoch = epochs[best_idx]
    best_val   = val_loss[best_idx]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # -- Linear scale --
    ax = axes[0]
    ax.plot(epochs, train_loss, linewidth=1.2, label="Train", color="steelblue")
    ax.plot(epochs, val_loss,   linewidth=1.2, label="Validation", color="orangered")
    ax.axvline(best_epoch, color="green", linestyle="--", alpha=0.5,
               label=f"Best (epoch {best_epoch})")
    ax.scatter([best_epoch], [best_val], color="green", s=60, zorder=5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Training Loss (Linear)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # -- Log scale --
    ax = axes[1]
    ax.semilogy(epochs, train_loss, linewidth=1.2, label="Train", color="steelblue")
    ax.semilogy(epochs, val_loss,   linewidth=1.2, label="Validation", color="orangered")
    ax.axvline(best_epoch, color="green", linestyle="--", alpha=0.5,
               label=f"Best (epoch {best_epoch})")
    ax.scatter([best_epoch], [best_val], color="green", s=60, zorder=5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (log)")
    ax.set_title("Training Loss (Log Scale)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # -- LR annotation --
    lr_changes = []
    for i in range(1, len(history)):
        if history[i]["lr"] != history[i-1]["lr"]:
            lr_changes.append((history[i]["epoch"], history[i]["lr"]))

    for ep, lr in lr_changes:
        for ax in axes:
            ax.axvline(ep, color="purple", linestyle=":", alpha=0.3)

    fig.suptitle("Orbit Correction Network -- Training Progress", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved loss plot -> {path}")


# ======================================================
# 6. TEST EVALUATION
# ======================================================

@torch.no_grad()
def test_evaluation(model, loader, device):
    """
    Detailed evaluation on the test set.

    Computes per-target MAE, RMSE, and max error.
    """
    model.eval()
    all_pred = []
    all_true = []

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        y_pred = model(batch_x)
        all_pred.append(y_pred.cpu())
        all_true.append(batch_y)

    preds = torch.cat(all_pred, dim=0)
    trues = torch.cat(all_true, dim=0)

    errors = preds - trues  # (N, 6)

    target_names = ["dx (km)", "dy (km)", "dz (km)",
                    "dvx (km/s)", "dvy (km/s)", "dvz (km/s)"]

    print(f"\n  Test set evaluation ({trues.shape[0]} samples):")
    print(f"  {'-'*65}")
    print(f"  {'Target':<16s}  {'MAE':>12s}  {'RMSE':>12s}  {'Max Error':>12s}")
    print(f"  {'-'*65}")

    for i, name in enumerate(target_names):
        err_i = errors[:, i]
        mae  = err_i.abs().mean().item()
        rmse = err_i.pow(2).mean().sqrt().item()
        maxe = err_i.abs().max().item()
        print(f"  {name:<16s}  {mae:12.6e}  {rmse:12.6e}  {maxe:12.6e}")

    # Overall position and velocity RMSE
    pos_err = errors[:, :3].pow(2).sum(dim=1).sqrt()  # 3D pos error per sample
    vel_err = errors[:, 3:].pow(2).sum(dim=1).sqrt()  # 3D vel error per sample

    print(f"  {'-'*65}")
    print(f"  {'3D Position':<16s}  {pos_err.mean().item():12.6e}  "
          f"{pos_err.pow(2).mean().sqrt().item():12.6e}  "
          f"{pos_err.max().item():12.6e}")
    print(f"  {'3D Velocity':<16s}  {vel_err.mean().item():12.6e}  "
          f"{vel_err.pow(2).mean().sqrt().item():12.6e}  "
          f"{vel_err.max().item():12.6e}")
    print(f"  {'-'*65}")


# ======================================================
# 7. LOAD TRAINED MODEL
# ======================================================

def load_trained_model(path, device="cpu"):
    """
    Load a trained model from a checkpoint file.

    Parameters
    ----------
    path   : str  -- path to .pt file
    device : str  -- "cpu" or "cuda"

    Returns
    -------
    model : OrbitCorrectionNet (in eval mode)
    info  : dict with training metadata
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model = OrbitCorrectionNet(
        input_dim   = checkpoint["input_dim"],
        hidden_dims = checkpoint["hidden_dims"],
        output_dim  = checkpoint["output_dim"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    info = {
        "epoch":      checkpoint["epoch"],
        "val_loss":   checkpoint["val_loss"],
        "train_loss": checkpoint["train_loss"],
    }

    print(f"  Loaded model from {path}")
    print(f"    Architecture : {checkpoint['input_dim']} -> "
          f"{' -> '.join(map(str, checkpoint['hidden_dims']))} -> "
          f"{checkpoint['output_dim']}")
    print(f"    Best epoch   : {info['epoch']}")
    print(f"    Val loss     : {info['val_loss']:.6e}")

    return model, info


# ======================================================
# MAIN
# ======================================================

if __name__ == "__main__":

    print("=" * 60)
    print("  Module 8.4 -- Training Pipeline")
    print("=" * 60)
    print()
    print(f"  PyTorch : {torch.__version__}")
    print(f"  CUDA    : {torch.cuda.is_available()}")
    print(f"  Device  : {CONFIG['device']}")

    # -- 1. Load data --
    print("\nStep 1 -- Loading datasets")
    loaders, datasets = build_dataloaders()

    for name in ["train", "val", "test"]:
        ds = datasets[name]
        print(f"  {name:12s}: {len(ds):6d} samples, {ds.n_features} features")

    # -- 2. Build model --
    print("\nStep 2 -- Building model")
    n_features = datasets["train"].n_features
    model = build_model(input_dim=n_features, device=CONFIG["device"])
    total_params = model.summary()

    # -- 3. Train --
    print("\nStep 3 -- Training")
    history = train(model, loaders, CONFIG)

    # -- 4. Save history --
    print("\nStep 4 -- Saving training history")
    save_history(history, CONFIG["history_csv"])

    # -- 5. Plot --
    print("\nStep 5 -- Plotting loss curves")
    plot_loss(history, CONFIG["loss_plot"])

    # -- 6. Reload best model and evaluate on test set --
    print("\nStep 6 -- Test set evaluation (best model)")
    best_model, info = load_trained_model(CONFIG["model_path"], CONFIG["device"])
    test_evaluation(best_model, loaders["test"], CONFIG["device"])

    # -- Summary --
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Model saved      : {CONFIG['model_path']}")
    print(f"  History saved    : {CONFIG['history_csv']}")
    print(f"  Loss plot saved  : {CONFIG['loss_plot']}")
    print(f"  Best epoch       : {info['epoch']}")
    print(f"  Best val loss    : {info['val_loss']:.6e}")
    print(f"  Total parameters : {total_params:,d}")
    print("=" * 60)

    print("\n[OK] Module 8.4 complete -- Training Pipeline.")
