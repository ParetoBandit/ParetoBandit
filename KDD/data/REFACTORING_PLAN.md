# Refactoring & Testing Plan

## Phase 1: Create Unit Tests ✅

### Core Scripts to Test

1. **`opencompass_name_mappings.py`** (Critical)
   - Test all name mappings resolve correctly
   - Test missing names return None or original
   - Test edge cases

2. **`train_final_xgboost_models.py`** (Critical)
   - Test data loading functions
   - Test feature preparation
   - Test model training (small dataset)
   - Test model saving/loading

3. **`build_instance_level_training_data.py`** (Critical)
   - Test each intent's data loading
   - Test join operations
   - Test grading logic
   - Test NVIDIA feature extraction

4. **`validate_all_4_intents.py`** (Important)
   - Test validation workflow
   - Test correlation calculations
   - Test statistical significance

### Test Structure
```
KDD/data/tests/
├── __init__.py
├── test_opencompass_mappings.py
├── test_model_training.py
├── test_data_collection.py
├── test_validation.py
└── fixtures/
    ├── sample_predictions.json
    ├── sample_prompts.json
    └── sample_cache.json
```

---

## Phase 2: Run Baseline Tests ✅

**Goal**: Establish that current code works before refactoring

```bash
cd /Users/annette/repostitories/llm_jury/KDD/data
python -m pytest tests/ -v
```

Expected: All tests pass

---

## Phase 3: Refactor Directory Structure ✅

### Step 1: Create New Directories
```bash
mkdir -p core_scripts
mkdir -p validation
mkdir -p documentation/{methodology,validation,reviewer_responses,investigations}
mkdir -p archive/{development_scripts,old_models,intermediate_data,old_documentation,intent_specific_development}
```

### Step 2: Move Core Scripts
```bash
mv build_instance_level_training_data.py core_scripts/
mv train_final_xgboost_models.py core_scripts/
mv opencompass_name_mappings.py core_scripts/
mv fetch_all_aa_benchmarks.py core_scripts/
```

### Step 3: Move Validation Scripts
```bash
mv validate_all_4_intents.py validation/
mv validate_rag_with_mmlu_pro.py validation/
```

### Step 4: Move Documentation
```bash
# Methodology
mv MODEL_SELECTION_RATIONALE.md documentation/methodology/
mv RAG_METHODOLOGY_IMPROVEMENT.md documentation/methodology/
mv ZERO_SHOT_TRANSFER_VALIDATION.md documentation/methodology/
mv ZERO_SHOT_VALIDATION_EXPLAINED.md documentation/methodology/
mv INSTANCE_LEVEL_TRAINING_README.md documentation/methodology/

# Validation
mv FINAL_VALIDATION_COMPLETE.md documentation/validation/

# Reviewer Responses
mv CRITIQUE_RESPONSE_RAG_IMPUTATION.md documentation/reviewer_responses/
mv MINOR_NOTES_RESPONSES.md documentation/reviewer_responses/
mv PAPER_UPDATE_CHECKLIST.md documentation/reviewer_responses/
mv ALL_CRITIQUES_RESOLVED.md documentation/reviewer_responses/

# Investigations
mv RGB_BENCHMARK_ANALYSIS.md documentation/investigations/
mv NATURAL_QUESTIONS_INVESTIGATION.md documentation/investigations/
```

### Step 5: Archive Development Files
```bash
# Development scripts
mv quick_train_and_validate*.py archive/development_scripts/
mv diagnose_transfer_issue.py archive/development_scripts/
mv validate_with_existing_data.py archive/development_scripts/
mv validate_proprietary_transfer.py archive/development_scripts/
mv validate_coding_with_coding_index.py archive/development_scripts/
mv anchor_based_imputation.py archive/development_scripts/
mv analyze_imputation_results.py archive/development_scripts/
mv train_xgboost_comparison.py archive/development_scripts/
mv train_xgboost_tuned.py archive/development_scripts/

# Old models
mv intent_predictors_with_nvidia/ archive/old_models/
mv xgboost_models/ archive/old_models/
mv validation_results/ archive/old_models/

# Intermediate data
mv anchor_based_imputation/ archive/intermediate_data/
mv livecodebench_prompts.csv archive/intermediate_data/
mv all_aa_benchmarks.json archive/intermediate_data/
mv proprietary_labels/ archive/intermediate_data/

# Old documentation
mv DATA_COLLECTION_AND_EXTRAPOLATION_STRATEGY.md archive/old_documentation/
mv FINAL_FEATURE_CONFIGURATION.md archive/old_documentation/
mv INTENT_DATA_SUMMARY.md archive/old_documentation/
mv PAPER_SNIPPETS.md archive/old_documentation/
mv QUICK_REFERENCE.md archive/old_documentation/
mv DETERMINISTIC_BENCHMARK_RATIONALE.md archive/old_documentation/
mv FINAL_VALIDATION_SUMMARY.md archive/old_documentation/
mv SESSION_SUMMARY.md archive/old_documentation/
mv TRANSFER_VALIDATION_FINDINGS.md archive/old_documentation/
mv VALIDATION_STATUS_AND_NEXT_STEPS.md archive/old_documentation/
mv REVIEWER_FEEDBACK_RESPONSE.md archive/old_documentation/
mv IMPROVED_VALIDATION_SUMMARY.md archive/old_documentation/

# Intent-specific development
mv agentic/ archive/intent_specific_development/
mv reasoning/ archive/intent_specific_development/
mv coding/ archive/intent_specific_development/
mv rag/ archive/intent_specific_development/
mv summarization/ archive/intent_specific_development/
mv livebench/ archive/intent_specific_development/
mv misc/ archive/intent_specific_development/
```

### Step 6: Delete Obsolete Files
```bash
rm collect_all_benchmarks.py
rm collect_proprietary_labels.py
rm evaluate_opencompass_predictions.py
rm load_livecodebench_directly.py
rm map_opencompass_to_cache.py
rm run_complete_pipeline.sh
```

---

## Phase 4: Update Import Paths ✅

### Files That May Need Updates

1. **Test files** → Update to find core_scripts/
   ```python
   # Before
   from opencompass_name_mappings import OPENCOMPASS_TO_CACHE
   
   # After
   from core_scripts.opencompass_name_mappings import OPENCOMPASS_TO_CACHE
   ```

2. **Validation scripts** → Update to find core_scripts/
   ```python
   # Before
   from opencompass_name_mappings import OPENCOMPASS_TO_CACHE
   
   # After  
   import sys
   sys.path.insert(0, '../core_scripts')
   from opencompass_name_mappings import OPENCOMPASS_TO_CACHE
   ```

3. **Check if any scripts import each other**
   - train_final_xgboost_models.py uses opencompass_name_mappings.py
   - validate_* scripts use opencompass_name_mappings.py

---

## Phase 5: Run Post-Refactor Tests ✅

```bash
cd /Users/annette/repostitories/llm_jury/KDD/data
python -m pytest tests/ -v
```

**Expected**: All tests still pass (no regressions)

---

## Phase 6: Integration Test ✅

### Test 1: Data Collection (Smoke Test)
```bash
cd core_scripts
python build_instance_level_training_data.py --test-mode
```

### Test 2: Model Training (Smoke Test)
```bash
cd core_scripts
python train_final_xgboost_models.py --test-mode
```

### Test 3: Validation (Smoke Test)
```bash
cd validation
python validate_all_4_intents.py --test-mode
```

---

## Phase 7: Create README.md ✅

Create master README explaining new structure.

---

## Safety Measures

### Backup Before Refactoring
```bash
cd /Users/annette/repostitories/llm_jury
git add -A
git commit -m "Checkpoint before refactoring"
git branch backup-pre-refactor
```

### Rollback Plan
If anything breaks:
```bash
git reset --hard HEAD~1
git checkout backup-pre-refactor
```

---

## Success Criteria

✅ All unit tests pass (before refactoring)  
✅ Directory structure cleaned up  
✅ All unit tests still pass (after refactoring)  
✅ Core scripts run without errors  
✅ Production models still load correctly  
✅ Documentation updated with new paths  

---

## Timeline

- **Phase 1** (Unit Tests): 30-45 min
- **Phase 2** (Baseline): 5 min
- **Phase 3** (Refactor): 10 min
- **Phase 4** (Update Imports): 10 min
- **Phase 5** (Re-test): 5 min
- **Phase 6** (Integration): 10 min
- **Phase 7** (README): 10 min

**Total**: ~90 minutes

---

**Status**: Ready to begin Phase 1
