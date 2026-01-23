# Calibration Code Reorganization Summary

## Overview

The calibration code has been reorganized to follow Python software engineering best practices, separating reusable library code, CLI tools, research artifacts, and raw data.

---

## New Structure

### 1. Library Code (`src/bandit_gpt/`)

**Location:** `src/bandit_gpt/calibration.py`

**Purpose:** Reusable calibration logic that can be imported by other projects.

**Classes:**
- `CalibratedRouter` - Production-ready LinUCB router with domain adaptation
- `SimpleLinUCBRouter` - Lightweight router for experiments

**Functions:**
- `apply_gamma_scaling()` - Apply covariance inflation to priors
- `embed_prompt()` - Embed prompts with PCA for LinUCB

**Usage:**
```python
from bandit_gpt.calibration import CalibratedRouter
import joblib
from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer("all-MiniLM-L6-v2")
pca_model = joblib.load("artifacts/pca_23.joblib")

# Load and use router
router = CalibratedRouter.load("my_router.joblib", encoder, pca_model)
model = router.select_model("My query")
```

---

### 2. CLI Tools (`scripts/calibration/`)

**Location:** `scripts/calibration/`

**Purpose:** User-facing command-line tools for calibration workflows.

**Tools:**
- `find_gamma.py` - Find optimal gamma calibration factor
- `calibrate_router.py` - Calibrate router for production

**Usage:**
```bash
cd scripts/calibration/

# Find optimal gamma
python3 find_gamma.py \
  --calibration-data ../../data/routellm/data/canonical_dev_calibration.jsonl

# Calibrate router
python3 calibrate_router.py \
  --calibration-data ../../data/routellm/data/canonical_dev_calibration.jsonl \
  --gamma 0.010 \
  --output my_router.joblib
```

---

### 3. Research Artifacts (`experiments_v1/calibration/`)

**Location:** `experiments_v1/calibration/`

**Purpose:** KDD paper documentation, analysis scripts, and experimental results.

**Documentation (*.md, *.tex):**
- `FINAL_RESULTS_SUMMARY.md` - Executive summary
- `KDD_NARRATIVE.md` - Complete paper narrative
- `RESULTS_SECTION.tex` - LaTeX for results section
- `ADAPTABILITY_PREMIUM.md` - Cost-quality analysis
- ... (8 more docs)

**Analysis Scripts (*.py):**
- `evaluate_calibrated_router.py` - Holdout evaluation
- `evaluate_bandit_convergence.py` - Convergence metrics
- `compare_calibration_convergence.py` - Compare scenarios
- ... (7 more scripts)

**Results (results/):**
- `calibration_results/` - Gamma analysis plots
- `evaluation_results/` - Holdout evaluation
- `bandit_convergence/` - Convergence metrics
- `artifacts/` - Calibrated router checkpoints

---

### 4. Raw Data (`data/routellm/`)

**Location:** `data/routellm/data/`

**Purpose:** Raw calibration datasets (JSONL files).

**Files:**
- `canonical_dev_calibration.jsonl` - 1,121 dev prompts for calibration
- `canonical_holdout_evaluation.jsonl` - 750 holdout prompts for evaluation
- `dev_rewards_complete.jsonl.gz` - Complete dev rewards (all models)
- `holdout_rewards_complete.jsonl.gz` - Complete holdout rewards

---

## File Migration

| File | Old Location | New Location | Reason |
|------|-------------|--------------|--------|
| `CalibratedRouter` class | `data/routellm/calibration/calibrate_router.py` | `src/bandit_gpt/calibration.py` | Reusable library code |
| `find_gamma.py` (CLI) | `experiments_v1/calibration/` | `scripts/calibration/` | User-facing tool |
| `calibrate_router.py` (CLI) | `experiments_v1/calibration/` | `scripts/calibration/` | User-facing tool |
| `*.md`, `*.tex` | `data/routellm/calibration/` | `experiments_v1/calibration/` | Research artifacts |
| `*.png`, `*.json` | `data/routellm/calibration/results/` | `experiments_v1/calibration/results/` | Experimental outputs |
| Analysis scripts | `data/routellm/calibration/` | `experiments_v1/calibration/` | Research code |

---

## Benefits

### 1. **Separation of Concerns**
- **`src/`** - Reusable library code (importable)
- **`scripts/`** - User-facing CLI tools
- **`experiments_v1/`** - Research artifacts and paper documentation
- **`data/`** - Raw data only (no code or docs)

### 2. **Discoverability**
- Library code: `import bandit_gpt.calibration`
- CLI tools: `scripts/calibration/`
- Research: `experiments_v1/calibration/`

### 3. **Maintainability**
- Library changes don't affect CLI tools (thin wrappers)
- Research artifacts stay with the code that generated them
- Clear dependency structure

### 4. **Packaging**
- Library code can be installed via pip: `pip install banditgpt`
- CLI tools can be entry points: `banditgpt-calibrate`
- Research code stays in repo (not distributed)

---

## Path Updates

### CLI Tools (scripts/calibration/)

**Old paths (from `data/routellm/calibration/`):**
```python
"../../../data/routellm/artifacts/priors.joblib"
"../../../artifacts/pca_23.joblib"
```

**New paths (from `scripts/calibration/`):**
```python
"data/routellm/artifacts/priors.joblib"
"artifacts/pca_23.joblib"
```

### Research Scripts (experiments_v1/calibration/)

**Old paths:**
```python
"../data/canonical_dev_calibration.jsonl"
```

**New paths:**
```python
"../../data/routellm/data/canonical_dev_calibration.jsonl"
```

---

## Import Structure

### Before (Monolithic Script)

```python
# Everything in one file
def apply_gamma_scaling(...):
    ...

class CalibratedRouter:
    ...

def main():
    # CLI logic
    ...

if __name__ == "__main__":
    main()
```

### After (Library + CLI)

**Library (`src/bandit_gpt/calibration.py`):**
```python
def apply_gamma_scaling(...):
    ...

class CalibratedRouter:
    ...
```

**CLI (`scripts/calibration/calibrate_router.py`):**
```python
from bandit_gpt.calibration import CalibratedRouter, apply_gamma_scaling

def main():
    # Thin CLI wrapper
    router = CalibratedRouter(...)
    router.save(...)

if __name__ == "__main__":
    main()
```

---

## Testing

To verify the reorganization works:

```bash
# Test library import
python3 -c "from bandit_gpt.calibration import CalibratedRouter; print('✓ Library import works')"

# Test find_gamma CLI
cd scripts/calibration/
python3 find_gamma.py --help

# Test calibrate_router CLI
python3 calibrate_router.py --help
```

---

## Documentation

| Topic | Location |
|-------|----------|
| **Library API** | `src/bandit_gpt/calibration.py` (docstrings) |
| **CLI Tools** | `scripts/calibration/README.md` |
| **Complete Workflow** | `experiments_v1/calibration/README.md` |
| **Research Results** | `experiments_v1/calibration/FINAL_RESULTS_SUMMARY.md` |
| **KDD Paper** | `experiments_v1/calibration/KDD_NARRATIVE.md` |

---

## Next Steps

### For Users

1. **Install library:**
   ```bash
   pip install -e .  # Development install
   ```

2. **Use CLI tools:**
   ```bash
   cd scripts/calibration/
   python3 find_gamma.py --calibration-data <your_data.jsonl>
   python3 calibrate_router.py --gamma 0.010 --output <router.joblib>
   ```

3. **Import in code:**
   ```python
   from bandit_gpt.calibration import CalibratedRouter
   router = CalibratedRouter.load(...)
   ```

### For Researchers

1. **Paper artifacts:** `experiments_v1/calibration/`
2. **Run experiments:** Use analysis scripts in `experiments_v1/calibration/`
3. **Generate figures:** Results saved to `experiments_v1/calibration/results/`

### For Developers

1. **Extend library:** Edit `src/bandit_gpt/calibration.py`
2. **Add CLI features:** Edit `scripts/calibration/*.py`
3. **Run tests:** (To be added)

---

**Reorganization Date:** January 23, 2026  
**Reason:** Follow Python SWE best practices (library vs scripts vs experiments)  
**Status:** ✅ Complete

