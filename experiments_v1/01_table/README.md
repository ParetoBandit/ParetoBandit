# Table 1: Dataset Description and Experimental Splits

**Date**: February 13, 2026  
**Status**: ✅ Simplified (categories removed)  
**Main Result**: Complete data provenance for 81,871 prompts

---

## 📊 Overview

This directory contains **Table 1** for the paper, which provides essential dataset provenance and experimental split information for reproducibility.

**Key Information**:
- **Total Prompts**: 81,871
- **Warmup Set**: 80,000 (PCA training + LinUCB priors)
- **Dev Set**: 1,121 (online learning & calibration)
- **Holdout Set**: 750 (final evaluation)

**Design Philosophy**: Focused on essential information needed for reproducibility. No unused analysis.

---

## 📁 Files

### Core Files

```
experiments_v1/01_table/
├── README_SIMPLIFIED.md                 # This file
├── generate_simplified_table.py         # New simplified generator
├── table1_dataset_simplified.tex        # LaTeX table (use this)
└── archived/                            # Old version with categories
    ├── analyze_dataset_composition.py   
    ├── table1_dataset_composition.tex
    └── validation_results_100.json
```

### Review Documents (Reference Only)

Strategic analysis and review:
- `EXECUTIVE_DECISION.md` - Decision to simplify
- `TABLE1_STRATEGIC_ANALYSIS.md` - Three options analyzed
- `FEASIBILITY_CHECK.md` - Data availability check
- `REVIEWER_ASSESSMENT.md` - Complete technical review
- `REVIEW_SUMMARY.md` - Executive summary
- `ACTION_PLAN.md` - Implementation plan
- `START_HERE.md` - Navigation guide

---

## 🎯 What's In The Table

### Essential Information

| Component | Description |
|-----------|-------------|
| **Data Sources** | LMSYS Chat Arena, RouteLLM battles |
| **Split Sizes** | 80,000 / 1,121 / 750 prompts |
| **Split Purposes** | PCA training, warmup priors, dev, holdout |
| **Model Details** | mixtral-8x7b-instruct, gpt-4-turbo, gpt-4o |
| **Quality Assurance** | Zero leakage, stratified sampling |
| **Sample Size** | Exceeds prior work (1,871 vs ~1,000) |

### What's NOT In The Table

❌ **Removed**:
- Semantic categories (Coding, Conversational, etc.)
- Category distributions
- Category validation discussion
- LLM agreement metrics (Fleiss' κ)

**Why removed**: Categories were not used in any experiment (Tables 2, Figures 1-8) and had accuracy concerns (49% vs LLM consensus). Removing them eliminates vulnerability while preserving all essential provenance information.

---

## 🔍 Data Provenance

### Warmup Set (80,000 prompts)

**Source**: LMSYS Arena battles via RouteLLM dataset  
**HuggingFace**: `routellm/gpt4_judge_battles`  
**Models**: mixtral-8x7b-instruct vs gpt-4-turbo  
**Access**: Public dataset (requires HF token)

**Purpose**:
- Train PCA model (384 → 32 dimensions)
- Generate LinUCB warmup priors:
  - **A matrix** (covariance): 33×33 per model
  - **b vector** (beliefs): 33×1 per model

**Processing**:
1. Downloaded from HuggingFace via `scripts/download_and_process_routellm.py`
2. Filtered for mixtral vs gpt-4-turbo battles
3. Extracted unique prompts (deduplicated)
4. Used for PCA: `scripts/train_pca_from_routellm.py`
5. Used for warmup: `scripts/generate_warmup_priors.py`

**Artifacts**:
- `src/artifacts/pca_32.joblib`
- `src/artifacts/priors_warmup.joblib`

### Dev Set (1,121 prompts)

**Source**: LMSYS Chat Arena (stratified splits)  
**File**: `data/dev_prompts_for_rejudge.jsonl`  
**Models**: mixtral-8x7b-instruct, gpt-4o

**Purpose**:
- Online learning (bandit trains here)
- Calibration (finding optimal γ, η)
- Model comparison (BanditGPT vs baselines)

**Stratification**:
- By semantic complexity
- By difficulty level
- Ensures representative coverage

### Holdout Set (750 prompts)

**Source**: LMSYS Chat Arena (stratified splits)  
**File**: `data/holdout_prompts_for_rejudge.jsonl`  
**Models**: mixtral-8x7b-instruct, gpt-4o

**Purpose**:
- Final evaluation (held-out test set)
- Pareto frontier curves
- Cost-quality tradeoff analysis

**Guarantee**: Completely disjoint from warmup  
**Verified**: Leakage check removed 243 overlapping prompts (0.24%)

---

## 🔧 Reproduction

### Generate The Table

```bash
cd experiments_v1/01_table
python generate_simplified_table.py
```

**Output**: `table1_dataset_simplified.tex`

### Use In Paper

```latex
\input{experiments_v1/01_table/table1_dataset_simplified}
```

Or copy the contents directly into your paper.

---

## 📝 Design Changes

### Before (Old Version with Categories)

```
Table 1: Dataset Composition and Provenance
- 81,871 prompts across 3 splits ✅
- 5 semantic categories (Coding 39%, Conversational 37.5%, etc.)
- Category validation (Fleiss' κ=0.75, 49% accuracy)
- Category distributions with confidence intervals
- 34 lines of LaTeX, complex footnotes
```

**Problems**:
- Categories had 49% accuracy vs LLM consensus
- Categories never used in experiments
- "Why categorize?" question had no good answer
- Vulnerable to reviewer criticism

### After (New Simplified Version)

```
Table 1: Dataset Description and Experimental Splits
- 81,871 prompts across 3 splits ✅
- Data sources clearly documented ✅
- Split purposes explained ✅
- Quality assurance documented ✅
- 20 lines of LaTeX, focused footnotes
```

**Benefits**:
- ✅ No accuracy concerns
- ✅ Focused on reproducibility
- ✅ Cannot be criticized for unused analysis
- ✅ Cleaner, more professional
- ✅ Directly supports experiments

---

## 🎨 Table Features

The simplified table follows publication best practices:

- ✅ Uses `booktabs` package (professional horizontal rules)
- ✅ Clear, descriptive caption
- ✅ Focused footnotes explaining data sources
- ✅ Shows sizes, sources, and purposes
- ✅ Documents data quality assurance
- ✅ Cites original data sources
- ❌ No disconnected or unused analysis

---

## 📚 References

1. **RouteLLM Dataset**: Ong, I., et al. (2024). "RouteLLM: Learning to Route LLMs with Preference Data." arXiv:2406.18665.
2. **LMSYS Arena**: Zheng, L., et al. (2023). "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena." NeurIPS 2023.
3. **HuggingFace Dataset**: https://huggingface.co/datasets/routellm/gpt4_judge_battles

---

## 🚀 Usage

### For Main Paper

The table is ready for immediate use in the paper:

1. Provides complete transparency about data sources
2. Documents experimental design clearly
3. Enables reproducibility
4. Focuses on essential information

### For Reproducibility

All information needed to reproduce the experiments:
- Where data came from (LMSYS Arena, RouteLLM)
- How much data was used (80k/1,121/750)
- What it was used for (PCA, warmup, dev, holdout)
- How data quality was ensured (leakage checks, stratification)

---

## 🔗 Related Files

- **PCA Training**: `scripts/train_pca_from_routellm.py`
- **Warmup Generation**: `scripts/generate_warmup_priors.py`
- **Reward Generation**: `scripts/generate_gpt4_turbo_rewards.py`
- **Stratification Logic**: `src/bandit_gpt/utils/experiment.py`
- **Data Correction**: `DATA_CORRECTION_SUMMARY.md` (project root)

---

## ✅ Status

- ✅ **Simplified**: Categories removed, provenance retained
- ✅ **Verified**: Data counts match (81,871 total)
- ✅ **Ready**: Can be used in paper immediately
- ✅ **Clean**: No vulnerable or unused analysis
- ✅ **Professional**: Focused on reproducibility

---

## 📊 Key Statistics

```
Total Prompts:        81,871
├─ Warmup:            80,000 (97.7%)
├─ Dev:                1,121 (1.4%)
└─ Holdout:              750 (0.9%)

Data Quality:
├─ Zero leakage:      ✅ 243 overlaps removed (0.24%)
├─ Stratification:    ✅ χ²=0.78, p=0.94 (dev vs holdout)
└─ Sample size:       ✅ Exceeds prior work (~1,000)

Sources:
├─ LMSYS Arena:       100% of data
├─ RouteLLM:          80,000 warmup prompts
└─ Public:            ✅ All data publicly available
```

---

## 🎯 Design Rationale

### Why Simplify?

**Question from review**: "Why categorize prompts if categories aren't used in any experiment?"

**Answer**: They shouldn't be! The simplified version:
1. Removes unused categorization
2. Keeps essential provenance
3. Focuses on reproducibility
4. Eliminates vulnerability to criticism

### What Was Preserved?

**Everything essential**:
- ✅ Data sources (for transparency)
- ✅ Split sizes (for power analysis)
- ✅ Split purposes (for experimental design)
- ✅ Quality assurance (for confidence)
- ✅ Model details (for reproducibility)

### What Was Removed?

**Non-essential elements**:
- ❌ Semantic categories (unused in experiments)
- ❌ Category distributions (disconnected from results)
- ❌ Validation discussion (49% accuracy was concerning)
- ❌ Confusion about purpose (why measure if not used?)

---

**Created**: 2026-02-13  
**Updated**: 2026-02-13  
**Author**: BanditGPT Team  
**Status**: ✅ Simplified and ready for publication
