# Tuning experiments folder

This folder contains all hyperparameter tuning scripts and their results.

## Tuning scripts

### 1.`full_tune.py`
Hyperparameter tuning (with multiple combinations) for Semi-gradient n-step SARSA, REINFORCE with baseline, and One-step Actor-Critic algorithms. 
- **Output**: 
  - `tuning_results.json` - Best hyperparameters and top 10 configs

**Instructions to run**:
```bash
cd tuning_experiments
python3 full_tune.py
```

### 2. `full_tune_pi2_tiles.py`
Hyperparameter tuning for PI²-CMA-ES with tile coding.
- **Output**:
  - `pi2_tiles_full_tuning_<timestamp>.json` - All configurations and results

**Instructions to run**:
```bash
cd tuning_experiments
python3 full_tune_pi2_tiles.py
```

