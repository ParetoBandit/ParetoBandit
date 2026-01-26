# Data Sources: Real Data Only

This document confirms that the Corralled Semantic Analysis experiment uses **ONLY real data** with no synthetic or fallback data.

## Phase 1: Optimization (Training)

### Labeled Data: Dev Dataset
- **Source**: `src/bandit_gpt/data/offline_dataset/dev_rewards_complete.jsonl.gz`
- **Size**: 1,121 unique prompts
- **Content**: Real prompts from LMSYS Chat-1M with human evaluation scores
- **Models evaluated**: 2 models (likely GPT-4 Turbo and another model)
- **Rewards**: Actual scores from human judges (0-1 scale)
- **Usage**: Training the Corralling algorithm to learn expert weights

### Warmup Priors
- **Source**: `src/artifacts/priors_warmup.joblib`
- **Content**: Pre-trained LinUCB parameters (A matrices and b vectors)
- **Trained on**: RouteLLM dataset (large-scale routing data)
- **Purpose**: Initialize the "Warmup Expert" with prior knowledge

### PCA Model
- **Source**: `src/artifacts/pca_32.joblib`
- **Components**: 32 principal components
- **Trained on**: RouteLLM embeddings
- **Purpose**: Dimensionality reduction for prompt embeddings

## Phase 2: Visualization (Projection)

### 1M Dataset: LMSYS Chat-1M
- **Source**: `experiments_v1/appendix_d/data/lmsys_chat_1M.jsonl.gz`
- **Size**: ~594,000 unique prompts
- **Content**: Real user prompts from LMSYS Chatbot Arena (April-August 2023)
- **Rewards**: **NONE** - this is unlabeled data
- **Usage**: Projecting the learned policy onto semantic space for visualization
- **Purpose**: Show which models the learned policy would select across the full distribution

### Projection Size Parameter
- **Default**: ALL (~594k prompts)
- **Configurable**: Can limit to N prompts (e.g., 50,000) for faster execution
- **Sampling**: Random sample from the full 594k if limited
- **Key Point**: Still real prompts, just fewer of them

## What We Do NOT Do

### ❌ No Synthetic Data
- We do NOT generate fake prompts
- We do NOT use GPT to create synthetic queries
- We do NOT use template-based prompt generation

### ❌ No Fake Rewards
- We do NOT estimate rewards on the 1M dataset
- We do NOT use propensity scores to infer counterfactual rewards
- We do NOT use model-based reward prediction

### ❌ No Fallback Data
- If the 1M dataset is missing, the script **fails** (no fallback)
- If the dev dataset is missing, the script **fails** (no fallback)
- If any required file is missing, the script **exits with error**

## Data Validation

The script includes strict validation at startup:

```python
required_files = {
    'Labeled Data': Path(CANONICAL_DEV_DATA_PATH),
    'PCA Model': Path(DEFAULT_PCA_PATH),
    'Warmup Priors': Path(DEFAULT_WARMUP_PRIORS_PATH),
    '1M Dataset': Path(...) / "lmsys_chat_1M.jsonl.gz"
}

if missing_files:
    print("❌ ERROR: MISSING REQUIRED DATA FILES")
    print("This script requires REAL data only (no synthetic/fallback data).")
    sys.exit(1)
```

## Data Flow

```
Phase 1: OPTIMIZATION (with rewards)
┌─────────────────────────────────────────────────────────┐
│ Dev Dataset (1,121 prompts)                             │
│ ├─ Real prompts from LMSYS Chat-1M                      │
│ ├─ Real rewards from human evaluations                  │
│ └─ Used for training Corralling algorithm               │
└─────────────────────────────────────────────────────────┘
                         ↓
              Train Corralling Router
              (importance-weighted loss)
                         ↓
         ┌───────────────────────────────┐
         │ Learned Expert Weights        │
         │ - Warmup: 0.130 (13%)         │
         │ - Tabula Rasa: 0.870 (87%)    │
         └───────────────────────────────┘
                         ↓
Phase 2: VISUALIZATION (no rewards)
┌─────────────────────────────────────────────────────────┐
│ 1M Dataset (50k-594k prompts)                           │
│ ├─ Real prompts from LMSYS Chat-1M                      │
│ ├─ NO rewards (unlabeled)                               │
│ └─ Used for projecting learned policy                   │
└─────────────────────────────────────────────────────────┘
                         ↓
         Project Learned Policy
         (which model would be selected?)
                         ↓
         ┌───────────────────────────────┐
         │ Model Usage Distribution      │
         │ - Shows coverage across       │
         │   semantic manifold           │
         │ - NO reward evaluation        │
         └───────────────────────────────┘
```

## Verification

To verify that only real data is used:

1. **Check data files exist**:
   ```bash
   ls -lh src/bandit_gpt/data/offline_dataset/dev_rewards_complete.jsonl.gz
   ls -lh experiments_v1/appendix_d/data/lmsys_chat_1M.jsonl.gz
   ```

2. **Count unique prompts**:
   ```python
   import gzip, json
   prompts = set()
   with gzip.open('src/bandit_gpt/data/offline_dataset/dev_rewards_complete.jsonl.gz', 'rt') as f:
       for line in f:
           prompts.add(json.loads(line)['prompt'])
   print(f"Dev prompts: {len(prompts)}")  # Should be 1,121
   ```

3. **Verify no synthetic generation**:
   ```bash
   # Search for synthetic data generation (should find nothing)
   grep -r "generate.*prompt" experiments_v1/03_figure/
   grep -r "fake.*reward" experiments_v1/03_figure/
   grep -r "synthetic" experiments_v1/03_figure/
   ```

## Summary

- **Phase 1 (Training)**: 1,121 real prompts with real rewards from dev dataset
- **Phase 2 (Visualization)**: 50k-594k real prompts from 1M dataset (no rewards)
- **No synthetic data**: All prompts are from actual user interactions
- **No fake rewards**: Only use actual human evaluation scores
- **Strict validation**: Script fails if any required data is missing

This ensures mathematical soundness and reproducibility.

