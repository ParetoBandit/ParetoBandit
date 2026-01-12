# Artifacts Directory

This directory contains **static, versioned ML artifacts** used by BanditGPT for reproducibility and KDD validation.

## Contents

### `pca_23.joblib` (38 KB)
- **Purpose**: Pre-trained PCA model for dimensionality reduction
- **Input**: 384-dimensional sentence embeddings (MiniLM-L6-v2)
- **Output**: 23-dimensional compressed embeddings
- **Variance Captured**: ~60%
- **Training Data**: 1,000 synthetic prompts covering 5 archetypes (math, coding, reasoning, creative, chat)
- **Usage**: Loaded by `BanditRouter` during initialization for efficient feature extraction
- **Generation**: `python create_pca_23.py`

### `priors_warmup.joblib` (59 KB)
- **Purpose**: Pre-computed warmup priors from synthetic IRT simulation
- **Training Data**: 20,000 mixed prompts (35% RouteLLM hard, 30% domain-specific, 20% simple, 15% router traps)
- **Contents**: LinUCB A and b matrices for all models in registry
- **Plasticity Factor**: 0.1 (treated as 2,000 effective samples to maintain adaptability)
- **Usage**: Loaded by `BanditRouter.create(priors="warmup")` for warm-start initialization
- **Generation**: `python scripts/generate_warmup.py`

### `priors_meta_pca.npz` (23 KB)
- **Purpose**: Legacy PCA metadata (archived)
- **Status**: Deprecated in favor of `pca_23.joblib`

## Directory Structure Philosophy

```
banditGPT/
├── artifacts/              # ✅ Static ML artifacts (versioned, KDD-ready)
│   ├── pca_23.joblib       # Primary PCA model (committed to git)
│   └── priors_warmup.joblib # Warmup initialization (committed to git)
│
├── data/                   # ❌ Runtime data (gitignored, grows over time)
│   ├── router_context.db   # 3 GB SQLite (production state)
│   └── lmsys_*.jsonl       # Evaluation datasets (large, gitignored)
│
└── src/bandit_gpt/assets/  # 📦 Package distribution fallback
    ├── pca_23.joblib       # Bundled for pip install
    └── priors_warmup.joblib # Bundled for pip install
```

## KDD Rationale

**Reproducibility**: Academic reviewers can use exact same artifacts to validate results

**Separation of Concerns**:
- `artifacts/` = **science** (static, versioned, committed to git)
- `data/` = **engineering** (dynamic, local, gitignored)

**Version Control**: All artifacts in this directory are tracked in git for exact reproducibility

## Regenerating Artifacts

If artifacts need to be regenerated (e.g., after encoder upgrade or model registry changes):

```bash
# PCA model (23 dimensions)
python create_pca_23.py

# Warmup priors (20,000 samples, ~5 minutes)
python scripts/generate_warmup.py

# Custom warmup with different model registry
python scripts/generate_warmup.py --models path/to/models.json --samples 50000
```

The router has **self-healing** capabilities: if `pca_23.joblib` is missing or has dimension mismatches, it will auto-train a new PCA model via JIT calibration.
