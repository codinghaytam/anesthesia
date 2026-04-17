# Anesthesia RL Repository - Agent Guide

## Project Overview
Reinforcement learning for automated anesthesia control using BIS (Bispectral Index) monitoring with PK/PD (Schnider model) simulations. 4 algorithms: Q-Learning, DP-Policy, DP-Value, DQN.

## Dependencies
```
numpy, matplotlib, torch, manim
```
Install with: `pip install -r requirements.txt`

## Running Code
- **Training**: Execute cells in the appropriate notebook (`DQN.ipynb`, `Q learning.ipynb`, `Dp-value-bis-deltabis.ipynb`, `Dp-policy-bis-deltabis.ipynb`)
- **Evaluation**: Runs automatically after training; raises RuntimeError if trained model not found
- **Notebooks**: Primary entry points, not scripts

## Key Utilities (`utils/`)
- `eval_metrics.py`: Metrics (MDPE, MDAPE, Wobble, TimeInTarget), `EvaluatorBase` class
- `eval_runner.py`: Population evaluation helpers (`run_saved_dp_evaluation`, `run_saved_q_evaluation`, `run_quick_*_evaluation`)
- Visualization utilities for RL trajectories

## Data & Artifacts
- Patient data: `data/Patients Data.csv` (required for evaluation)
- Trained models: `artifacts/*.pth` (DQN), `artifacts/*.npz` (DP/Q)
- Metrics output: `metrics/`

## Model Hyperparameters (Shared)
- BIS_TARGET = 50.0
- STEPS_PER_EPISODE = 120
- GAMMA = 0.69
- EPSILON = 0.1
- RANDOM_SEED = 42

## Evaluation
- Standard eval: 500 patients, 4 durations [300, 600, 1200, 3600] seconds
- Quick eval: 50 patients
- Metrics: MDPE, MDAPE, Wobble, TimeInTarget (±5 of target)

## Age Groups
25-29, 30-45, 46-60, 60-80, 80+
