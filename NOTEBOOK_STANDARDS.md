# Anesthesia RL Notebooks - Uniformity Standards

## General Structure
All notebooks must follow this structure order:
1. **Imports**
2. **CONFIGURATION SECTION** (all control variables at top)
3. **Helper Functions** (utility functions with docstrings)
4. **Training/Computation Section** (algorithm-specific)
5. **Visualization** (plots from training if applicable)
6. **Evaluation Section** (population-based evaluation)

---

## Configuration Variables (Top of Script)

### Training Parameters (Algorithm-Specific)
Each algorithm has its own hyperparameters, with standardized naming:

**Q-Learning (Tabular):**
```python
# --- TRAINING CONFIGURATION ---
EPISODES = 5000              # Number of training episodes
STEPS_PER_EPISODE = 3600     # Steps per training episode (5s timesteps)
ALPHA = 0.2                  # Learning rate
GAMMA = 0.69                 # Discount factor
EPSILON = 0.05               # Exploration rate
```

**Dynamic Programming (Value Iteration):**
```python
# --- TRAINING CONFIGURATION ---
NUM_ITERATIONS = 10000       # Number of Value Iteration iterations
GAMMA = 0.69                 # Discount factor
```

**DQN (Neural Network):**
```python
# --- TRAINING CONFIGURATION ---
EPISODES = 5000              # Number of training episodes
STEPS_PER_EPISODE = 3600     # Steps per training episode
GAMMA = 0.69                 # Discount factor
LEARNING_RATE = 0.001        # Neural network learning rate
```

### PK/PD Constants (Shared Across All)
```python
# --- PK/PD PARAMETERS (Schnider Model) ---
V1, V2, V3 = 4.27, 18.9, 238.0
k10, k12, k21, k13, k31 = 0.38, 0.30, 0.20, 0.19, 0.0035
ke0 = 0.17
BIS_0, BIS_MAX, EC50, HILL = 95.0, 75.0, 3.5, 2.5
BIS_TARGET = 50.0
```

### Fuzzification Parameters
```python
# --- FUZZIFICATION ---
BINS_PER_FEAT = 10
NUM_STATES = 1000  # or appropriate value per algorithm
```

### Evaluation Configuration (Mandatory - Same for All)
```python
# --- EVALUATION CONFIGURATION ---
EVAL_SAMPLE_SIZE = 500           # Number of patients to sample
EVAL_EPISODE_LENGTHS = [300, 600, 1200, 3600]  # Episode durations in seconds (VARIABLE)
EVAL_TIME_STEP = 5/60            # Time step in minutes (5 seconds)
RANDOM_SEED = 42

AGE_GROUPS = {
    '25-29': (25, 29),
    '30-45': (30, 45),
    '46-60': (46, 60),
    '60-80': (60, 80),
    '80+': (80, 120)
}
```

### Paths
```python
# --- PATHS ---
ARTIFACTS_DIR = Path("artifacts")
METRICS_DIR = Path("metrics")
DATA_PATH = Path("data/Patients Data.csv")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
```

---

## Training Section

- **For Q-Learning**: Train on many episodes, save Q-table
- **For DP**: Compute transition matrix and run value iteration, save policy
- **For DQN**: Train neural network with experience replay
- Save trained model to `ARTIFACTS_DIR`

---

## Evaluation Section (Standardized)

### Requirements:
1. **Load** patient database
2. **Sample** `EVAL_SAMPLE_SIZE` patients randomly
3. **For each patient**:
   - Generate personalized PK/PD parameters using Schnider model
   - Run simulation at EACH episode length in `EVAL_EPISODE_LENGTHS`
   - Calculate metrics for each episode length
4. **Metrics to collect** (per patient, per episode length):
   - MDPE (Median Prediction Error)
   - MDAPE (Median Absolute Prediction Error)
   - Wobble (stability measure)
   - Time in Target (% time within ±5 of BIS target)
5. **Output**:
   - One CSV per episode length: `metrics/algorithm_name_length_XXXs.csv`
   - Rows: patients
   - Columns: PatientID, Age, AgeGroup, MDPE, MDAPE, Wobble, TimeInTarget
   - Summary CSV: `metrics/algorithm_name_summary.csv` with per-age-group statistics

---

## Evaluation Strategy

### One Row Per Patient
- Columns for each episode length (e.g., `MDPE_300s`, `MDAPE_300s`, etc.)
- Additional columns: `PatientID`, `Age`, `AgeGroup`

### Metrics Calculation
For each simulated episode:
- **PE (Prediction Error)** = (BIS - BIS_TARGET) / BIS_TARGET × 100
- **MDPE** = median(PE)
- **MDAPE** = median(|PE|)
- **Wobble** = median(|PE - median(PE)|)
- **Time in Target** = % of timesteps where |BIS - BIS_TARGET| ≤ 5

### Age Group Analysis
Group results by `AGE_GROUPS` and calculate:
- Mean MDPE, MDAPE, Wobble, TimeInTarget per age group
- Standard deviation
- Count of patients per age group

---

## Directory Structure

```
anesthesia/
├── Dp.ipynb
├── Dp-policy.ipynb
├── Dp-policy-deltabis.ipynb
├── Dp-policy-bis-deltabis.ipynb
├── Dp-value-bis-deltabis.ipynb
├── Dp-value-derive-bis.ipynb
├── DQN.ipynb
├── Q learning.ipynb
├── shnider-model.ipynb
├── artifacts/               # Trained models
│   ├── dp_agent.npz
│   ├── q_agent.npz
│   └── ...
├── metrics/                 # NEW: Evaluation results
│   ├── dp_agent_300s.csv
│   ├── dp_agent_600s.csv
│   ├── dp_agent_summary.csv
│   ├── q_agent_300s.csv
│   └── ...
├── data/
│   └── Patients Data.csv
└── utils/
    ├── __init__.py
    ├── eval_runner.py
    └── rl_visualization.py
```

---

## Naming Conventions

- **Training artifacts**: `{algorithm}_{variant}_agent.npz` or `.pth`
  - Examples: `dp_agent.npz`, `q_deltabis_agent.npz`, `dqn_network.pth`

- **Evaluation CSVs**: 
  - Per-length: `{algorithm_name}_{length}s.csv`
  - Summary: `{algorithm_name}_summary.csv`
  - Examples: `dp_agent_300s.csv`, `q_deltabis_600s.csv`, `dqn_3600s.csv`

---

## Checklist for Each Notebook

- [ ] All variables at top of script (training + evaluation)
- [ ] Clear section headers (# --- SECTION NAME ---)
- [ ] Helper functions documented with docstrings
- [ ] Training section produces artifact file
- [ ] Evaluation section produces CSV files in `metrics/` directory
- [ ] Age group analysis included in summary
- [ ] Code follows uniform structure (imports → config → functions → training → eval)

