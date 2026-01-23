# Calibration Folder Reorganization Summary

## Overview

This folder has been reorganized to follow Python software engineering best practices, separating concerns between:
- **Research artifacts** (documentation, LaTeX, plots)
- **Executable scripts** (analysis tools)
- **Results** (experimental outputs)
- **Raw data** (remains in `data/routellm/`)

## New Structure

```
experiments_v1/calibration/
├── README.md                           # User guide for calibration workflow
├── DOCUMENTATION_INDEX.md              # Complete documentation index
├── requirements.txt                    # Python dependencies
│
├── Documentation (Paper Artifacts)
│   ├── FINAL_RESULTS_SUMMARY.md        # Executive summary
│   ├── KDD_NARRATIVE.md                # Complete paper narrative
│   ├── RESULTS_AT_A_GLANCE.md          # Quick reference
│   ├── RESULTS_SECTION.tex             # LaTeX for results section
│   ├── ADAPTABILITY_PREMIUM.md         # Cost-quality analysis
│   ├── ADAPTABILITY_PREMIUM_TABLE.tex  # LaTeX tables
│   ├── MODEL_TRANSFER_INSIGHT.md       # Cross-model transfer
│   ├── MODEL_TRANSFER_SECTION.tex      # LaTeX for transfer section
│   ├── GOLDSTANDARD_METRICS_EXPLAINED.md # Convergence metrics
│   └── CONVERGENCE_EXPLAINED.md        # Why entropy fails
│
├── Scripts (Analysis Tools)
│   ├── find_gamma.py                   # Find optimal gamma factor
│   ├── calibrate_router.py             # Calibrate router
│   ├── evaluate_calibrated_router.py   # Holdout evaluation
│   ├── evaluate_bandit_convergence.py  # Convergence metrics
│   ├── evaluate_convergence_metrics.py # Additional metrics
│   ├── evaluate_with_entropy.py        # Entropy analysis
│   ├── compare_calibration_convergence.py # Compare scenarios
│   ├── prepare_canonical_dev.py        # Prepare dev data
│   ├── prepare_canonical_holdout.py    # Prepare holdout data
│   └── prepare_dev_data.py             # Dev data preparation
│
└── results/                            # Experimental outputs
    ├── artifacts/
    │   └── canonical_router_calibrated.joblib
    ├── bandit_convergence/
    │   ├── bandit_convergence_goldstandard.png
    │   └── convergence_metrics.json
    ├── calibration_convergence_comparison/
    │   ├── calibration_convergence_comparison.png
    │   └── comparison_metrics.json
    ├── calibration_results/
    │   ├── gamma_analysis.png
    │   └── gamma_results.json
    ├── entropy_analysis/
    │   ├── convergence_metrics.json
    │   └── entropy_convergence_analysis.png
    └── evaluation_results/
        ├── evaluation_comparison.png
        └── evaluation_results.json
```

## What Changed

### Before (Old Structure)
```
data/routellm/calibration/
├── *.md, *.tex              # Mixed with data
├── *.py                     # Mixed with data
├── *.png, *.json            # Mixed with data
└── artifacts/               # Mixed with data
```

### After (New Structure)
```
experiments_v1/calibration/  # Research artifacts
├── Documentation/           # Paper narrative & LaTeX
├── Scripts/                 # Analysis tools
└── results/                 # Experimental outputs

data/routellm/               # Raw data only
├── data/                    # JSONL datasets
└── artifacts/               # Pre-trained models
```

## Why This Change?

Following SWE best practices for research code:

1. **Separation of Concerns**
   - `data/` contains only raw/processed data files (`.jsonl`, `.gz`)
   - `experiments_v1/` contains research artifacts and analysis code
   - `src/` contains reusable library code (to be added)

2. **Discoverability**
   - Scripts are now in a well-known location (`experiments_v1/calibration/`)
   - Documentation is co-located with the code that generated it
   - Results are organized by experiment type

3. **Reproducibility**
   - All artifacts needed to reproduce the KDD paper are in one place
   - Clear separation between training data and experimental results
   - Version control friendly (experimental results are git-tracked)

4. **Maintainability**
   - Easy to find documentation for a specific experiment
   - Scripts reference data using relative paths
   - Results stay with the code that generated them

## Migration Notes

### Path References

Scripts in this folder now use paths relative to `experiments_v1/calibration/`:

- **Data files**: `../../data/routellm/data/canonical_dev_calibration.jsonl`
- **Artifacts**: `../../data/routellm/artifacts/priors_warmup_routellm_pca24.joblib`
- **PCA models**: `../../artifacts/pca_23_routellm.joblib`
- **Results output**: `results/` (local to this folder)

### Running Scripts

From the `experiments_v1/calibration/` directory:

```bash
# Find optimal gamma
python3 find_gamma.py \
  --calibration-data ../../data/routellm/data/canonical_dev_calibration.jsonl \
  --output results/

# Calibrate router
python3 calibrate_router.py \
  --calibration-data ../../data/routellm/data/canonical_dev_calibration.jsonl \
  --gamma 0.010 \
  --output results/my_router.joblib

# Evaluate
python3 evaluate_calibrated_router.py \
  --router results/my_router.joblib \
  --holdout-data ../../data/routellm/data/canonical_holdout_evaluation.jsonl
```

## References

- **Main README**: `../../README.md` (project overview)
- **Data README**: `../../data/routellm/README.md` (data provenance)
- **Calibration Guide**: `README.md` (this folder, user-facing guide)
- **Documentation Index**: `DOCUMENTATION_INDEX.md` (complete guide to artifacts)

---

**Reorganization Date**: January 23, 2026  
**Reason**: Follow Python SWE best practices for research projects
**Impact**: File paths in scripts updated, documentation co-located with code

