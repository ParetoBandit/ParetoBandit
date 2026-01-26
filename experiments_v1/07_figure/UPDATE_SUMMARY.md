# Figure 6 Update Summary

## Overview
Updated `plot_adaptive_effeciency.py` to use the canonical dataset paths from `config_legacy.py` and ensured all required models (GPT-4-Turbo, GPT-5.1, Mixtral) are available in the dataset.

## Changes Made

### 1. Updated Configuration (`src/bandit_gpt/config_legacy.py`)
- Added new constants for all-models datasets:
  ```python
  DEV_DATA_PATH_ALL_MODELS = OFFLINE_DATASET_DIR / "dev_rewards_complete_all_models.jsonl.gz"
  HOLDOUT_DATA_PATH_ALL_MODELS = OFFLINE_DATASET_DIR / "holdout_rewards_complete_all_models.jsonl.gz"
  ```

### 2. Merged GPT-4-Turbo Data into All-Models Dataset
- **Problem**: The `dev_rewards_complete_all_models.jsonl.gz` dataset contained 42 models including GPT-5.1, but was missing GPT-4-turbo.
- **Solution**: Extracted GPT-4-turbo entries from the 2-models dataset and merged them into the all-models datasets.
- **Results**:
  - **Dev dataset**: Added 1,121 GPT-4-turbo entries (total: 48,203 entries, 43 models)
  - **Holdout dataset**: Added 750 GPT-4-turbo entries (total: 32,252 entries, 43 models)
- **Backups created**: 
  - `dev_rewards_complete_all_models.jsonl.gz.backup`
  - `holdout_rewards_complete_all_models.jsonl.gz.backup`

### 3. Updated Plot Script (`plot_adaptive_effeciency.py`)

#### Data Loading
- Changed from hardcoded paths to use `DEV_DATA_PATH_ALL_MODELS` from config
- Fixed data loading to use `reward_logit` field (the actual field in the dataset)
- Added proper logging to verify models are available

#### Model Configuration
- **OLD_MODELS**: `["mistralai/mixtral-8x7b-instruct", "openai/gpt-4-turbo"]`
- **NEIGHBOR_MODEL**: `"openai/gpt-4-turbo"` (semantic teacher for GPT-5.1)
- **NEW_MODEL**: `"openai/gpt-5.1"` (real model from dataset, not simulated)

#### Documentation
- Updated docstring to reflect correct model lineup
- Updated comments to accurately describe the semantic transfer from GPT-4-Turbo to GPT-5.1

## Dataset Structure

### All-Models Dataset Now Contains (43 models):
```
✓ mistralai/mixtral-8x7b-instruct (1,121 entries)
✓ openai/gpt-4-turbo (1,121 entries)  ← ADDED
✓ openai/gpt-5.1 (1,121 entries)
  openai/gpt-4o (1,121 entries)
  openai/gpt-5 (1,121 entries)
  ... (38 more models)
```

### Data Fields
Each entry contains:
- `model_id`: Model identifier
- `prompt`: User prompt
- `response`: Model response
- `reward_logit`: Reward signal (ranges ~-5 to +5) ← Used as reward
- `raw_score`: Binary score (0 or 1)
- `judge_details`: Multi-judge evaluation details
- `ok`: Status flag
- `teacher_used`: Whether teacher model was used
- `ts`: Timestamp

## Experiment Design

### Scenario
1. **Phase 1 (t=0 to t=299)**: Train router on Mixtral + GPT-4-Turbo
2. **Phase 2 (t=300)**: Release GPT-5.1 (superior model)
3. **Compare**:
   - **Cold Start**: Initialize GPT-5.1 with identity matrix (no prior knowledge)
   - **Semantic Transfer**: Initialize GPT-5.1 by inheriting θ from GPT-4-Turbo

### Expected Outcome
- **Cold Start**: Performance dip when GPT-5.1 is released, gradual recovery
- **Semantic Transfer**: Maintains high performance immediately (zero-shot readiness)

## Verification

Run the following to verify the dataset:
```bash
cd /Users/annette/repostitories/banditGPT
gunzip -c src/bandit_gpt/data/offline_dataset/dev_rewards_complete_all_models.jsonl.gz | \
  python3 -c "import sys, json; models = set(); \
  [models.add(json.loads(line)['model_id']) for line in sys.stdin]; \
  print('Total models:', len(models)); \
  print('Has GPT-4-turbo:', 'openai/gpt-4-turbo' in models); \
  print('Has GPT-5.1:', 'openai/gpt-5.1' in models)"
```

Expected output:
```
Total models: 43
Has GPT-4-turbo: True
Has GPT-5.1: True
```

## Next Steps

1. Run the experiment: `python3 experiments_v1/06_figure/plot_adaptive_effeciency.py`
2. Check generated figure: `experiments_v1/06_figure/results/figure6_adaptive_efficiency.png`
3. Verify the semantic transfer line maintains high reward while cold start dips

## Notes

- The script now uses **real GPT-5.1 rewards** from the dataset, not simulated data
- The semantic transfer mechanism transfers learned parameters (θ) from GPT-4-Turbo to GPT-5.1
- All model names and configurations are now consistent with the actual dataset content

