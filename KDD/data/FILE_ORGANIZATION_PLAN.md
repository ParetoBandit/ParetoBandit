# KDD/data File Organization Plan

## Files to KEEP (Production & Documentation)

### Core Scripts ✅
- `build_instance_level_training_data.py` - Main data collection
- `train_final_xgboost_models.py` - Final model training
- `opencompass_name_mappings.py` - Model name resolution
- `fetch_all_aa_benchmarks.py` - Benchmark score fetching

### Final Validation Scripts ✅
- `validate_all_4_intents.py` - Complete validation
- `validate_rag_with_mmlu_pro.py` - RAG-specific validation

### Production Models ✅
- `production_models/` - All 4 trained models + metadata

### Training Data ✅
- `instance_level_training_data/` - 113K training examples

### Final Documentation ✅
- `FINAL_SYSTEM_STATUS.md` - **NEW** Complete system summary
- `FINAL_VALIDATION_COMPLETE.md` - Validation results
- `MODEL_SELECTION_RATIONALE.md` - XGBoost justification
- `RAG_METHODOLOGY_IMPROVEMENT.md` - MMLU-Pro approach
- `CRITIQUE_RESPONSE_RAG_IMPUTATION.md` - Reviewer response
- `PAPER_UPDATE_CHECKLIST.md` - Paper updates needed
- `MINOR_NOTES_RESPONSES.md` - Minor reviewer feedback
- `ZERO_SHOT_TRANSFER_VALIDATION.md` - Transfer explanation
- `ZERO_SHOT_VALIDATION_EXPLAINED.md` - Detailed methodology
- `RGB_BENCHMARK_ANALYSIS.md` - **NEW** RGB investigation
- `NATURAL_QUESTIONS_INVESTIGATION.md` - **NEW** NQ investigation
- `INSTANCE_LEVEL_TRAINING_README.md` - Data documentation

---

## Files to ARCHIVE (Move to archive/)

### Intermediate Development Scripts
- `quick_train_and_validate.py` - Early version
- `quick_train_and_validate_v2.py` - Intermediate version
- `quick_train_and_validate_v3.py` - Pre-final version
- `diagnose_transfer_issue.py` - Debugging script
- `validate_with_existing_data.py` - Early validation
- `validate_proprietary_transfer.py` - Superseded
- `validate_all_intents.py` - Old version (kept v4)
- `validate_coding_with_coding_index.py` - Test (didn't use)

### Superseded Approaches
- `anchor_based_imputation.py` - Old imputation approach
- `anchor_based_imputation/` - Old imputation results
- `analyze_imputation_results.py` - Old analysis
- `train_xgboost_comparison.py` - Model comparison (done)
- `train_xgboost_tuned.py` - Tuning experiments (done)

### Old Model Storage
- `intent_predictors_with_nvidia/` - Superseded by production_models/
- `xgboost_models/` - Superseded by production_models/
- `validation_results/` - Superseded by production_models/

### Intermediate Data
- `livecodebench_prompts.csv` - Extracted during development
- `all_aa_benchmarks.json` - One-time fetch result
- `proprietary_labels/` - Intermediate validation data

### Old Documentation
- `DATA_COLLECTION_AND_EXTRAPOLATION_STRATEGY.md` - Superseded by FINAL_SYSTEM_STATUS
- `FINAL_FEATURE_CONFIGURATION.md` - Superseded by FINAL_SYSTEM_STATUS
- `INTENT_DATA_SUMMARY.md` - Superseded by FINAL_SYSTEM_STATUS
- `PAPER_SNIPPETS.md` - Superseded by structured docs
- `QUICK_REFERENCE.md` - Superseded by FINAL_SYSTEM_STATUS
- `DETERMINISTIC_BENCHMARK_RATIONALE.md` - Integrated into other docs
- `FINAL_VALIDATION_SUMMARY.md` - Superseded by FINAL_VALIDATION_COMPLETE
- `SESSION_SUMMARY.md` - Development notes
- `TRANSFER_VALIDATION_FINDINGS.md` - Integrated into FINAL_VALIDATION_COMPLETE
- `VALIDATION_STATUS_AND_NEXT_STEPS.md` - Development notes
- `REVIEWER_FEEDBACK_RESPONSE.md` - Superseded by specific response docs

---

## Files to DELETE (Obsolete)

### Unused Utility Scripts
- `collect_all_benchmarks.py` - Superseded by fetch_all_aa_benchmarks.py
- `collect_proprietary_labels.py` - One-time use, complete
- `evaluate_opencompass_predictions.py` - Development helper
- `load_livecodebench_directly.py` - Development helper
- `map_opencompass_to_cache.py` - Superseded by opencompass_name_mappings.py
- `run_complete_pipeline.sh` - Development script

---

## Subdirectories to ARCHIVE

### Intent-Specific Development Folders
- `agentic/` - Early development (4 intents used, not 5)
- `reasoning/` - Early development (integrated into main pipeline)
- `coding/` - Early development (integrated into main pipeline)
- `rag/` - Early development (integrated into main pipeline)
- `summarization/` - Early development (integrated into main pipeline)
- `livebench/` - Early development

### Miscellaneous
- `misc/` - Various development notes and logs

---

## Proposed New Structure

```
KDD/data/
├── README.md (NEW - explain structure)
├── FINAL_SYSTEM_STATUS.md ⭐ START HERE
│
├── core_scripts/
│   ├── build_instance_level_training_data.py
│   ├── train_final_xgboost_models.py
│   ├── opencompass_name_mappings.py
│   └── fetch_all_aa_benchmarks.py
│
├── validation/
│   ├── validate_all_4_intents.py
│   └── validate_rag_with_mmlu_pro.py
│
├── production_models/
│   ├── reasoning_xgboost_model.joblib
│   ├── coding_xgboost_model.joblib
│   ├── summarization_xgboost_model.joblib
│   ├── rag_xgboost_model.joblib
│   ├── [model_cards...]
│   └── README.md
│
├── instance_level_training_data/
│   ├── instance_level_training_data.csv
│   ├── instance_level_training_data.json
│   └── training_data_summary.txt
│
├── documentation/
│   ├── methodology/
│   │   ├── MODEL_SELECTION_RATIONALE.md
│   │   ├── RAG_METHODOLOGY_IMPROVEMENT.md
│   │   ├── ZERO_SHOT_TRANSFER_VALIDATION.md
│   │   ├── ZERO_SHOT_VALIDATION_EXPLAINED.md
│   │   └── INSTANCE_LEVEL_TRAINING_README.md
│   │
│   ├── validation/
│   │   └── FINAL_VALIDATION_COMPLETE.md
│   │
│   ├── reviewer_responses/
│   │   ├── CRITIQUE_RESPONSE_RAG_IMPUTATION.md
│   │   ├── MINOR_NOTES_RESPONSES.md
│   │   └── PAPER_UPDATE_CHECKLIST.md
│   │
│   └── investigations/
│       ├── RGB_BENCHMARK_ANALYSIS.md
│       └── NATURAL_QUESTIONS_INVESTIGATION.md
│
└── archive/ (NEW - historical reference)
    ├── development_scripts/
    ├── old_models/
    ├── intermediate_data/
    ├── old_documentation/
    └── intent_specific_development/
```

---

## Action Plan

1. **Create new structure** (core_scripts/, validation/, documentation/)
2. **Move active files** to new locations
3. **Create archive/** directory
4. **Move development files** to archive/
5. **Delete obsolete files**
6. **Create README.md** explaining structure
7. **Update import paths** if needed (minimal - mostly standalone scripts)

---

## Benefits

✅ **Clarity**: Clear separation of production vs development  
✅ **Documentation**: Grouped by purpose  
✅ **Maintenance**: Easy to find current vs historical  
✅ **Onboarding**: New team members can navigate easily  
✅ **Paper Writing**: All docs in one place  

---

**Status**: Ready to execute
**Estimated Time**: 15-20 minutes
**Risk**: Low (moving files, not deleting production code)
