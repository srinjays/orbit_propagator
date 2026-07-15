# AI-Enhanced Orbit Determination — Complete Project Explanation

---

## The One-Line Pitch

> **"I built an AI that makes satellite tracking more accurate by learning to correct the errors of a traditional Kalman Filter."**

---

## 1. The Problem

When tracking a satellite from the ground, we use **radar stations** to measure its position. But:

- Radar measurements are **noisy** (instrument limitations, atmospheric effects)
- Our physics models are **imperfect** (we can't perfectly model every force acting on the satellite)
- The traditional solution — an **Extended Kalman Filter (EKF)** — gives a good estimate, but it always has some residual error

**Question:** Can we train a neural network to learn the *pattern* of those residual errors and correct them?

**Answer:** Yes. That's what this project does.

---

## 2. How It Works (Two Phases)

```
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 1: PHYSICS PIPELINE                    │
│                                                                 │
│   Satellite Orbit  ──►  Radar Measurements  ──►  Kalman Filter  │
│   (Truth)               (Noisy)                  (Estimate)     │
│                                                                 │
│   We KNOW the truth     We simulate what        The EKF tries   │
│   because we            a real radar would      to reconstruct  │
│   simulated it.         see.                    the orbit.      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     PHASE 2: AI PIPELINE                        │
│                                                                 │
│   Compare EKF vs Truth  ──►  Train Neural Net  ──►  Correct EKF │
│   (compute errors)           (learn patterns)       (apply fix) │
│                                                                 │
│   Error = Truth - EKF   The AI learns:          Final orbit is  │
│   This is what the AI   "When the EKF says X,   MORE ACCURATE   │
│   must learn.           the error is usually Y" than EKF alone. │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Module-by-Module Breakdown

### Phase 1 — Physics (Generates the Data)

#### Module 1: Orbit Propagator (`thirdbody_propagator.py`)
- **What:** Simulates a satellite orbiting Earth for 90 minutes using Orekit
- **Physics:** Includes gravity (20x20 geopotential), atmospheric drag, solar radiation pressure, and gravitational pull from the Moon and Sun
- **Output:** `full_physics_orbit_dataset.csv` — the **ground truth** orbit (position + velocity every few seconds)
- **Analogy:** This is like placing a GPS tracker on the satellite — perfect knowledge

#### Module 2: Radar Simulator (`radar_simulator.py`)
- **What:** Simulates what a ground radar station would actually measure
- **Adds:** Realistic noise to range, range-rate, azimuth, and elevation measurements
- **Output:** `radar_measurements.csv`
- **Analogy:** This is like looking at the satellite through binoculars in fog — you can see it, but not perfectly

#### Module 3: Batch Least Squares (`bls_orbit_determination.py`)
- **What:** Takes ALL radar measurements at once and finds the orbit that best fits them
- **Method:** Classical least-squares optimisation (iterative)
- **Output:** `bls_estimated_orbit.csv` — a first rough estimate
- **Analogy:** Like fitting a curve through noisy data points

#### Module 4: Extended Kalman Filter (`ekf_orbit_determination.py`)
- **What:** Sequentially processes each radar measurement and continuously refines the orbit estimate
- **Key:** Also outputs the **covariance matrix** (how confident it is) and **residuals** (how much each measurement surprised it)
- **Output:** `ekf_estimated_orbit.csv` — the best traditional estimate
- **Analogy:** Like updating your GPS in real-time as new signals arrive

---

### Phase 2 — AI (Learns and Corrects)

#### Module 5: Dataset Generator (`ai_dataset_generator.py`)
- **What:** Pairs each EKF estimate with the corresponding truth value
- **Computes:** The error at each timestep: `correction = truth - EKF`
- **Features (16 inputs to the AI):**

| # | Feature | Why It Matters |
|---|---------|---------------|
| 1-3 | EKF Position (x, y, z) | Where the filter thinks the satellite is |
| 4-6 | EKF Velocity (vx, vy, vz) | How fast it thinks the satellite is moving |
| 7-10 | Radar Residuals (4 values) | How surprised the filter was by recent measurements |
| 11-16 | Covariance Diagonal (6 values) | How confident the filter is in its own estimate |

- **Targets (6 outputs the AI must predict):**
  - dx, dy, dz (position correction in km)
  - dvx, dvy, dvz (velocity correction in km/s)
- **Split:** 70% train / 15% validation / 15% test — **chronologically** (no shuffling, prevents temporal data leakage)
- **Normalisation:** Z-score standardisation on all features

#### Module 6: PyTorch Dataset (`pytorch_dataset.py`)
- **What:** Converts the CSV files into PyTorch tensors and DataLoaders
- **Why:** PyTorch needs data in tensor format for GPU-accelerated training

#### Module 7: Neural Network (`orbit_correction_net.py`)
- **Architecture:**
```
Input (16 features)
    │
    ▼
Linear(16 → 128) + ReLU      ← 2,176 parameters
    │
    ▼
Linear(128 → 64) + ReLU      ← 8,256 parameters
    │
    ▼
Linear(64 → 32) + ReLU       ← 2,080 parameters
    │
    ▼
Linear(32 → 6)               ← 198 parameters
    │
    ▼
Output (6 corrections)

Total: 12,710 trainable parameters
```
- **Initialisation:** Xavier uniform (keeps gradients stable)
- **No output activation:** This is regression, not classification

#### Module 8: Training (`train_model.py`)
- **Optimizer:** Adam (adaptive learning rate)
- **Loss:** Mean Squared Error (MSE) — because we're predicting continuous values
- **Learning Rate Scheduler:** ReduceLROnPlateau — cuts LR by half when validation loss plateaus
- **Early Stopping:** Stops training if validation loss doesn't improve for 40 epochs
- **Checkpoint:** Saves the best model based on validation loss

#### Module 9: Evaluation (`evaluate_model.py`)
- **What:** Loads the trained model and applies it to the full EKF orbit
- **Process:**
  1. Normalise the EKF features (using saved statistics)
  2. Feed through the neural network → get correction vectors
  3. Apply: `corrected_orbit = EKF_estimate + AI_correction`
  4. Compare everything against truth
- **Metrics:** RMSE, MAE, Max Error — for both position and velocity
- **Plots:** Error comparison bars, error time-series, orbit trajectories

---

## 4. How To Run Everything

```powershell
cd "d:\orbit propagator"

# PHASE 1 — Physics Pipeline
python thirdbody_propagator.py       # Generates truth orbit
python radar_simulator.py            # Simulates radar data
python bls_orbit_determination.py    # Initial orbit estimate
python ekf_orbit_determination.py    # EKF refined estimate

# PHASE 2 — AI Pipeline
python ai_dataset_generator.py       # Builds training dataset
python train_model.py                # Trains the neural network
python evaluate_model.py             # Evaluates and corrects
```

---

## 5. Where To See Results

| What To Look At | File | What It Shows |
|----------------|------|---------------|
| **AI-corrected orbit** | `corrected_orbit.csv` | The final improved trajectory |
| **Training progress** | `training_loss.png` | How the model learned |
| **Error comparison** | `eval_error_comparison.png` | Bar chart: EKF vs AI RMSE |
| **Error over time** | `eval_error_timeseries.png` | Before vs After AI correction |
| **Orbit plot** | `eval_orbit_comparison.png` | Truth vs EKF vs AI-Corrected |
| **Terminal output** | (on screen) | Prints improvement percentage |

---

## 6. How To Explain This to Others

### For a Professor / Technical Audience:

> "This project implements a hybrid orbit determination pipeline. Phase 1 uses Orekit to propagate a high-fidelity satellite trajectory with J2-20x20 gravity, NRLMSISE-00 drag, SRP, and third-body perturbations. A synthetic radar sensor generates noisy measurements, which are processed by a Batch Least Squares estimator and then an Extended Kalman Filter. Phase 2 introduces a feedforward neural network that learns the systematic residual error between the EKF estimate and the truth. The network ingests a 16-dimensional feature vector comprising the EKF state, innovation residuals, and covariance diagonal, and outputs a 6-dimensional correction vector. On test data, the AI reduces position RMSE by 11-33% over the bare EKF."

### For a Non-Technical Audience:

> "Imagine you're tracking a satellite with radar. The radar gives you noisy measurements, and a traditional filter tries to figure out where the satellite actually is. But it's never perfect — there's always some error. What I did is train an AI to learn the pattern of those errors. So now, after the filter gives its best guess, the AI says 'actually, you're probably off by this much in this direction' and makes a correction. The result is a more accurate satellite position than either the filter or the AI could achieve alone."

### For a Job Interview:

> "I built a complete satellite tracking pipeline from scratch — orbit physics, sensor simulation, Kalman filtering, and then an AI layer on top. The key innovation is using the filter's own confidence metrics (covariance) and measurement residuals as additional inputs to the neural network, giving it 16 features instead of just 6. This lets the AI understand not just where the filter thinks the satellite is, but how confident the filter is and whether its recent measurements were surprising. The system is modular — each component is a standalone Python module that can be tested independently."

---

## 7. Key Technical Terms (Quick Reference)

| Term | Meaning |
|------|---------|
| **Orbit Propagation** | Predicting where a satellite will be using physics equations |
| **Orbit Determination** | Figuring out where a satellite IS from noisy measurements |
| **Extended Kalman Filter** | A recursive algorithm that combines a physics prediction with new measurements |
| **Covariance Matrix** | Tells you how uncertain the EKF is about its estimate |
| **Innovation / Residual** | The difference between what the filter predicted it would see and what the radar actually measured |
| **Supervised Learning** | Training an AI by showing it input-output pairs (features → corrections) |
| **Z-score Normalisation** | Scaling features to have mean=0, std=1 so the neural network trains better |
| **Chronological Split** | Splitting data by time (not randomly) to prevent the AI from "cheating" by seeing future data |
| **Early Stopping** | Stopping training when the model stops improving to prevent overfitting |
| **Xavier Initialisation** | A smart way to set initial neural network weights for stable training |
