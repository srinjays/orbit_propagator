# AI-Enhanced Satellite Orbit Determination

A complete pipeline that combines **high-fidelity orbital mechanics** with a **neural network** to improve satellite tracking accuracy beyond traditional Kalman filtering.

## What This Project Does

1. **Simulates** a satellite orbit with real physics (gravity, drag, solar radiation pressure, third-body effects) using Orekit
2. **Generates** synthetic radar measurements with realistic noise
3. **Estimates** the orbit using Batch Least Squares and an Extended Kalman Filter
4. **Trains** a neural network to learn and correct the residual EKF errors
5. **Produces** an AI-corrected orbit that is more accurate than the EKF alone

## Architecture

```
PHYSICS PIPELINE                          AI PIPELINE
─────────────────                         ───────────
Orbit Propagator                          Dataset Generator
       │                                        │
       ▼                                        ▼
Radar Simulator                           PyTorch DataLoader
       │                                        │
       ▼                                        ▼
Batch Least Squares                       Neural Network (16→128→64→32→6)
       │                                        │
       ▼                                        ▼
Extended Kalman Filter ──────────────────► Training Pipeline
                                                │
                                                ▼
                                          Evaluation & Correction
                                                │
                                                ▼
                                          corrected_orbit.csv
```

## Neural Network Features (16 inputs → 6 outputs)

| Input Features | Count | Description |
|---|---|---|
| EKF Position | 3 | Where the filter thinks the satellite is (x, y, z) |
| EKF Velocity | 3 | Estimated velocity (vx, vy, vz) |
| Radar Residuals | 4 | Innovation vector (range, rate, azimuth, elevation) |
| Covariance Diagonal | 6 | Filter confidence (P_xx, P_yy, P_zz, P_vxvx, P_vyvy, P_vzvz) |

| Output Targets | Count | Description |
|---|---|---|
| Correction Vector | 6 | Δx, Δy, Δz, Δvx, Δvy, Δvz (Truth − EKF) |

## How To Run

```bash
# Phase 1 — Physics Pipeline
python thirdbody_propagator.py
python radar_simulator.py
python bls_orbit_determination.py
python ekf_orbit_determination.py

# Phase 2 — AI Pipeline
python ai_dataset_generator.py
python train_model.py
python evaluate_model.py
```

## Project Structure

| File | Module | Description |
|---|---|---|
| `thirdbody_propagator.py` | Propagator | High-fidelity orbit propagation (Orekit) |
| `radar_simulator.py` | Sensor Sim | Synthetic radar measurement generation |
| `bls_orbit_determination.py` | BLS OD | Batch Least Squares orbit determination |
| `ekf_orbit_determination.py` | EKF OD | Extended Kalman Filter with covariance export |
| `ai_dataset_generator.py` | Module 8.1 | Training dataset builder (16-feature architecture) |
| `pytorch_dataset.py` | Module 8.2 | PyTorch Dataset & DataLoader wrapper |
| `orbit_correction_net.py` | Module 8.3 | Neural network definition (Xavier init) |
| `train_model.py` | Module 8.4 | Training pipeline (Adam, early stopping, LR scheduler) |
| `evaluate_model.py` | Module 8.5 | Evaluation, correction, metrics & plots |
| `generate_test_data.py` | Utility | Synthetic data generator (when Orekit unavailable) |

## Dependencies

- Python 3.8+
- PyTorch
- NumPy, Pandas, Matplotlib
- Orekit (via `orekit-jpype`) — for the physics pipeline

## Output Files

| File | Description |
|---|---|
| `corrected_orbit.csv` | AI-corrected satellite trajectory |
| `training_loss.png` | Train vs validation loss curves |
| `eval_error_comparison.png` | EKF vs AI-corrected error bar chart |
| `eval_orbit_comparison.png` | Orbit plot: Truth vs EKF vs AI-Corrected |
| `orbit_correction_model.pt` | Trained model checkpoint |

## License

MIT
