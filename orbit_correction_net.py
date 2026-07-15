"""
Module 8.3 -- Orbit Correction Neural Network
================================================
Feedforward neural network that learns the correction vector
between EKF-estimated and true satellite states.

Architecture
------------
    Input  -> Linear(N_in, 128) -> ReLU
           -> Linear(128, 64)   -> ReLU
           -> Linear(64, 32)    -> ReLU
           -> Linear(32, 6)     -> Output

    N_in = 16 (enhanced) or 6 (basic), auto-detected from dataset.

    The network predicts:
        [dx, dy, dz, dvx, dvy, dvz]
    i.e. the correction to add to the EKF state to recover the
    true orbit state.

Initialisation
--------------
    Xavier (Glorot) uniform for all weight matrices.
    Zero bias initialisation.

Input:   train_dataset.csv, validation_dataset.csv, test_dataset.csv
Output:  (in-memory model -- no files written by this module)

Usage:
    python orbit_correction_net.py                    # standalone demo
    from orbit_correction_net import OrbitCorrectionNet  # import

Dependencies:
    torch, pytorch_dataset (Module 8.2)
"""

import torch
import torch.nn as nn


# ======================================================
# CONFIGURATION
# ======================================================
CONFIG = {
    # Architecture (hidden layer sizes)
    "hidden_layers": [128, 64, 32],

    # Output dimension (always 6: dx, dy, dz, dvx, dvy, dvz)
    "output_dim": 6,

    # Input dimension: set to None to auto-detect from dataset,
    # or set explicitly (6 for basic, 16 for enhanced)
    "input_dim": None,
}


# ======================================================
# 1. NETWORK DEFINITION
# ======================================================

class OrbitCorrectionNet(nn.Module):
    """
    Feedforward neural network for satellite orbit correction.

    Learns the mapping:
        EKF state (+ optional residuals & covariance) -> correction vector

    The correction vector is added to the EKF estimate to produce
    a refined state closer to the true orbit.

    Parameters
    ----------
    input_dim   : int   -- number of input features (6 or 16)
    hidden_dims : list  -- sizes of hidden layers [128, 64, 32]
    output_dim  : int   -- number of outputs (always 6)
    """

    def __init__(self, input_dim, hidden_dims=None, output_dim=6):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        self.input_dim   = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim  = output_dim

        # -- Build sequential layer stack --
        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim

        # Output layer (no activation -- regression output)
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

        # -- Apply Xavier initialisation --
        self._init_weights()

    def _init_weights(self):
        """
        Xavier (Glorot) uniform initialisation for all linear layers.

        Xavier init sets weights from U(-a, a) where:
            a = gain * sqrt(6 / (fan_in + fan_out))

        This keeps the variance of activations stable across layers,
        which is critical for training convergence — especially in
        regression tasks with small target values (our corrections
        are in the range of ~0.01–0.1 km).
        """
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : tensor, shape (batch_size, input_dim)
            Normalised input features.

        Returns
        -------
        correction : tensor, shape (batch_size, 6)
            Predicted correction vector [dx, dy, dz, dvx, dvy, dvz].
        """
        return self.network(x)

    def count_parameters(self):
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self):
        """
        Print a detailed model summary including layer shapes,
        parameter counts, and initialisation info.
        """
        sep = "-" * 65
        print()
        print("=" * 65)
        print("  OrbitCorrectionNet -- Model Summary")
        print("=" * 65)
        print(f"  Input dim    : {self.input_dim}")
        print(f"  Hidden dims  : {self.hidden_dims}")
        print(f"  Output dim   : {self.output_dim}")
        print(f"  Activation   : ReLU")
        print(f"  Init method  : Xavier uniform")
        print()
        print(f"  {'Layer':<30s} {'Shape':<20s} {'Params':>10s}")
        print(f"  {sep}")

        total = 0
        for name, param in self.named_parameters():
            shape_str = str(list(param.shape))
            n = param.numel()
            total += n
            print(f"  {name:<30s} {shape_str:<20s} {n:>10,d}")

        print(f"  {sep}")
        print(f"  {'Total trainable parameters':<50s} {total:>10,d}")
        print("=" * 65)

        # Architecture diagram
        print()
        print("  Architecture:")
        dims = [self.input_dim] + self.hidden_dims + [self.output_dim]
        for i in range(len(dims) - 1):
            act = "ReLU" if i < len(dims) - 2 else "(none)"
            print(f"    Linear({dims[i]:>4d} -> {dims[i+1]:>4d})  +  {act}")
        print()

        return total


# ======================================================
# 2. MODEL FACTORY
# ======================================================

def build_model(input_dim=None, config=None, device="cpu"):
    """
    Build an OrbitCorrectionNet with auto-detected or explicit input dim.

    Parameters
    ----------
    input_dim : int, optional
        Number of input features. If None, attempts to detect from
        the dataset via Module 8.2.
    config : dict, optional
        Override CONFIG.
    device : str
        "cpu" or "cuda"

    Returns
    -------
    model : OrbitCorrectionNet on the specified device
    """
    if config is None:
        config = CONFIG

    # -- Determine input dimension --
    if input_dim is not None:
        n_in = input_dim
    elif config["input_dim"] is not None:
        n_in = config["input_dim"]
    else:
        # Auto-detect from dataset
        try:
            from pytorch_dataset import build_dataloaders
            _, datasets = build_dataloaders()
            n_in = datasets["train"].n_features
            print(f"  Auto-detected input_dim = {n_in} from training dataset")
        except Exception as e:
            print(f"  Could not auto-detect input_dim: {e}")
            print(f"  Falling back to input_dim = 6 (basic mode)")
            n_in = 6

    model = OrbitCorrectionNet(
        input_dim   = n_in,
        hidden_dims = config["hidden_layers"],
        output_dim  = config["output_dim"],
    ).to(device)

    return model


# ======================================================
# 3. VERIFICATION
# ======================================================

def verify_model(model, device="cpu"):
    """
    Run a forward pass with random data to verify shapes and gradients.
    """
    print("\n  Forward pass verification:")
    print("  " + "-" * 55)

    batch_size = 8
    x = torch.randn(batch_size, model.input_dim, device=device)
    y_pred = model(x)

    print(f"    Input  : {tuple(x.shape)}")
    print(f"    Output : {tuple(y_pred.shape)}")
    print(f"    dtype  : {y_pred.dtype}")
    print(f"    device : {y_pred.device}")

    # Check output shape
    expected = (batch_size, model.output_dim)
    passed = True
    if y_pred.shape != expected:
        print(f"    [FAIL] Output shape {y_pred.shape} != expected {expected}")
        passed = False
    else:
        print(f"    [PASS] Output shape correct")

    # Check gradients flow
    loss = y_pred.sum()
    loss.backward()

    grad_ok = True
    for name, param in model.named_parameters():
        if param.grad is None:
            print(f"    [FAIL] No gradient for {name}")
            grad_ok = False
            passed = False

    if grad_ok:
        print(f"    [PASS] Gradients flow through all {model.count_parameters()} parameters")

    # Check weight initialisation statistics
    print(f"\n  Weight initialisation statistics:")
    for name, param in model.named_parameters():
        if "weight" in name:
            w = param.data
            print(f"    {name:<30s}  "
                  f"mean={w.mean().item():+8.5f}  "
                  f"std={w.std().item():8.5f}  "
                  f"range=[{w.min().item():+.4f}, {w.max().item():+.4f}]")

    model.zero_grad()
    return passed


# ======================================================
# MAIN -- Standalone demo
# ======================================================

if __name__ == "__main__":

    print("=" * 60)
    print("  Module 8.3 -- Orbit Correction Neural Network")
    print("=" * 60)
    print()
    print(f"  PyTorch : {torch.__version__}")
    print(f"  CUDA    : {torch.cuda.is_available()}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -- Build model --
    print("\nStep 1 -- Building model")
    model = build_model(device=device)

    # -- Print summary --
    print("\nStep 2 -- Model summary")
    total_params = model.summary()

    # -- Verify --
    print("Step 3 -- Verification")
    passed = verify_model(model, device)

    # -- Test with dataset if available --
    print("\nStep 4 -- Dataset integration test")
    try:
        from pytorch_dataset import build_dataloaders
        loaders, datasets = build_dataloaders()

        ds = datasets["train"]
        print(f"  Dataset features : {ds.n_features}")
        print(f"  Model input_dim  : {model.input_dim}")

        if ds.n_features != model.input_dim:
            print(f"  [WARN] Dimension mismatch -- rebuild model with input_dim={ds.n_features}")
        else:
            # Run one real batch
            batch_x, batch_y = next(iter(loaders["train"]))
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            y_pred = model(batch_x)

            print(f"  Batch input  : {tuple(batch_x.shape)}")
            print(f"  Batch output : {tuple(y_pred.shape)}")
            print(f"  Batch target : {tuple(batch_y.shape)}")

            # Quick loss check
            criterion = nn.MSELoss()
            loss = criterion(y_pred, batch_y)
            print(f"  Initial loss : {loss.item():.6e}")
            print(f"  [PASS] Dataset integration OK")

    except FileNotFoundError:
        print("  Dataset CSVs not found -- skipping integration test.")
        print("  Run Modules 8.1 and 8.2 first to generate data.")
    except Exception as e:
        print(f"  Integration test error: {e}")

    # -- Final --
    print("\n" + "=" * 60)
    print("  RESULT")
    print("=" * 60)
    print(f"  Architecture     : {model.input_dim} -> {' -> '.join(map(str, model.hidden_dims))} -> {model.output_dim}")
    print(f"  Parameters       : {total_params:,d}")
    print(f"  Verification     : {'PASSED' if passed else 'FAILED'}")
    print("=" * 60)

    print("\n[OK] Module 8.3 complete -- Orbit Correction Neural Network.")
