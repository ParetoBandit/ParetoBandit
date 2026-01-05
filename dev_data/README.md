# Development Data

This directory contains source data used to train static assets.

## Contents

### `golden_prompts.jsonl`

**Purpose**: Source data for training the complexity vector.

**Usage**: These prompts were averaged to create `src/bandit_gpt/assets/complexity_vector.npz`, which is the "Gold Standard" definition of hardness (H⃗) used by the router.

**Note for Developers**: 
- This file is **excluded from pip packages** (see `MANIFEST.in`)
- End users don't need this - they only need the resulting `complexity_vector.npz`
- Keep this in the repo to allow re-training if the definition of "Hardness" evolves

## Regenerating Assets

If you need to re-train the complexity vector:

```python
# See experiments/new_bandit/generate_complexity_vector.py
python experiments/new_bandit/generate_complexity_vector.py \
    --input dev_data/golden_prompts.jsonl \
    --output src/bandit_gpt/assets/complexity_vector.npz
```

## Why Separate from Source Code?

Following best practices:
- **Source code** (`src/`) = What ships to users
- **Development data** (`dev_data/`) = What developers use to build assets
- **Experiments** (`experiments/`) = Research and validation

This keeps the pip package lean while maintaining reproducibility.
