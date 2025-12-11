# Intent Classification Scripts Cleanup Summary

## Scripts Removed (Old/Exploratory)

### From `/scripts/intent_classification/` (4 files removed)

**All OLD - pre-KDD paper approach:**

1. ❌ **`build_intent_dataset.py`** (16 KB)
   - Old dataset building script
   - Superseded by new data collection methodology

2. ❌ **`collect_real_intent_data.py`** (24 KB)
   - Old data collection approach
   - Superseded by root-level data collection

3. ❌ **`train_and_evaluate.py`** (6 KB)
   - Old training script
   - Superseded by `train_intent_classifier.py`

4. ❌ **`train_xgboost_classifier.py`** (8 KB)
   - Old XGBoost training
   - Superseded by unified training scripts

### From `/KDD/intent_classification/` (1 file removed)

5. ❌ **`evaluate_decorrelation_rigorously.py`** (13 KB)
   - Exploratory script used during IPW/INLP investigation
   - Not referenced in final paper
   - Findings incorporated into other evaluation scripts

**Total removed:** 5 scripts, ~67 KB

---

## Scripts Retained (Essential for KDD Paper)

### Root Level Training Scripts (2 files)

✅ **`train_intent_classifier.py`**
- Trains baseline model (94.5% accuracy)
- Essential for baseline results in Section 4.1

✅ **`train_intent_classifier_decorrelated.py`**
- Trains robust model with orthogonal projection (88.1% accuracy)
- Essential for robust model results in Section 4.8

### KDD Analysis Scripts (7 files)

✅ **`generate_figures.py`**
- Generates Figures 1-5 for the paper
- Confusion matrix, per-class performance, CV folds, data distribution

✅ **`test_length_artifact.py`**
- Tests baseline model on long prompts
- Discovers 100% failure rate (critical finding)
- Results shown in Section 4.6

✅ **`test_decorrelated_on_length_artifact.py`**
- Tests robust model on same long prompts
- Shows 75% bias reduction (25% failure rate)
- Results shown in Section 4.8

✅ **`evaluate_stratified_performance.py`**
- Evaluates model stability across prompt length buckets
- Generates stratified performance analysis
- Results in Table 8 (stratified accuracy)

✅ **`visualize_stratified_performance.py`**
- Creates publication-quality figures for stratified analysis
- Generates training distribution heatmap
- Generates stability metrics charts

✅ **`test_wild_prompts.py`**
- Tests generalization on unstructured real-world prompts
- Validates semantic learning vs. keyword matching
- Supports distribution shift defense (Section 4.5)

✅ **`test_shortcut_learning.py`**
- Tests if model uses keyword shortcuts
- Ensures semantic understanding
- Supports shortcut learning defense (Section 4.5)

---

## Folder Structure After Cleanup

```
/scripts/intent_classification/
├── README.md (documentation - kept)
├── CAE_COMPOSITE_SCORE_LOG.md (reference - kept)
└── TAXONOMY_CHANGE_LOG.md (reference - kept)

/KDD/intent_classification/
├── INTENT_CLASSIFICATION_SECTION.md (main paper)
├── generate_figures.py ✅
├── test_length_artifact.py ✅
├── test_decorrelated_on_length_artifact.py ✅
├── evaluate_stratified_performance.py ✅
├── visualize_stratified_performance.py ✅
├── test_wild_prompts.py ✅
├── test_shortcut_learning.py ✅
├── [JSON result files]
├── [PNG figures]
└── [Markdown documentation]

/root/
├── train_intent_classifier.py ✅
└── train_intent_classifier_decorrelated.py ✅

/llm_jury/intent/
├── __init__.py
├── classifier.py
├── training.py
├── length_debiasing.py ✅ (NEW - unified debiasing class)
└── README.md ✅ (NEW - debiasing documentation)

/examples/
└── train_with_debiasing.py ✅ (NEW - usage example)
```

---

## Rationale for Each Retained Script

| Script | Purpose | Paper Section | Removable? |
|--------|---------|---------------|------------|
| `train_intent_classifier.py` | Baseline training | 4.1 | ❌ Core |
| `train_intent_classifier_decorrelated.py` | Robust training | 4.8 | ❌ Core |
| `generate_figures.py` | Paper figures 1-5 | 4.1-4.2 | ❌ Core |
| `test_length_artifact.py` | Baseline artifact test | 4.6 | ❌ Critical finding |
| `test_decorrelated_on_length_artifact.py` | Robust artifact test | 4.8 | ❌ Critical finding |
| `evaluate_stratified_performance.py` | Stratified analysis | 4.9 | ❌ Bias proof |
| `visualize_stratified_performance.py` | Stratified figures | 4.9 | ❌ Bias proof |
| `test_wild_prompts.py` | Generalization test | 4.5 | ✅ Optional* |
| `test_shortcut_learning.py` | Shortcut test | 4.5 | ✅ Optional* |

*Optional scripts support generalization claims but are not core to the length bias story

---

## Summary

**Removed:** 5 old/exploratory scripts that are not referenced in the final KDD paper

**Retained:** 9 essential scripts that directly support paper claims:
- 2 training scripts (baseline + robust)
- 2 artifact testing scripts (baseline + robust)
- 2 stratified analysis scripts (evaluation + visualization)
- 2 generalization tests (wild prompts + shortcut learning)
- 1 figure generation script (figures 1-5)

**Result:** Clean, focused codebase with only production-ready and paper-essential scripts.

All retained scripts are:
✅ Referenced in the paper
✅ Reproduce key findings
✅ Well-documented
✅ Production-ready
