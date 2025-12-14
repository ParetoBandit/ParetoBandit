# Old Code Cleanup Plan: Remove etl/ and intent/

**Date**: December 13, 2024  
**Reason**: These represent the old approach, superseded by KDD/data production models

---

## 🎯 Why Remove These?

### llm_jury/etl/
**Old approach**: ETL pipeline for collecting benchmark scores from APIs
- Collected data from Artificial Analysis, OpenRouter, Vectara
- Manual data merging and pipeline management
- Ad-hoc benchmark collection

**New approach (KDD/data)**:
- ✅ 113K instance-level examples from OpenCompass
- ✅ NVIDIA prompt complexity features
- ✅ Systematic data collection with validation
- ✅ Now in `llm_jury/prediction/`

### llm_jury/intent/
**Old approach**: Embedding-based intent classification with length debiasing
- Used sentence embeddings (all-MiniLM-L6-v2)
- Required length debiasing (orthogonal projection)
- Trained on small labeled dataset
- 5 classes: reasoning, coding, factual_qa, agentic, general

**New approach (KDD/data)**:
- ✅ NVIDIA prompt features (6 features)
- ✅ Model capability proxies (1 feature per intent)
- ✅ 113K training examples from real benchmarks
- ✅ 4 production models with proven transfer
- ✅ Now in `llm_jury/prediction/`

**Note**: The actual XGBoost classifier code is in `llm_jury/routing/xgboost_intent_classifier.py`, which we keep!

---

## 📦 What Gets Removed

### llm_jury/etl/ (10 files)
```
llm_jury/etl/
├── __init__.py
├── artificial_analysis_client.py
├── coding_benchmarks_client.py
├── complete_gpt35_data.py
├── data_merger.py
├── hallucination_leaderboard_client.py
├── llm_matcher.py
├── openai_direct_client.py
├── openrouter_ttft_client.py
├── pipeline.py
└── README_OPENAI_ETL.md
```

### llm_jury/intent/ (4 files)
```
llm_jury/intent/
├── __init__.py
├── classifier.py
├── length_debiasing.py
├── README.md
└── training.py
```

---

## 🔍 Impact Analysis

### Files That Import from etl/
1. **llm_jury/__init__.py**:
   ```python
   from llm_jury.etl import ETLPipeline, ArtificialAnalysisClient, DataMerger
   ```
   **Action**: Remove these imports

2. **scripts/data_collection/fetch_scores_llm.py**:
   ```python
   from llm_jury.etl.llm_matcher import LLMModelMatcher
   ```
   **Action**: Move to archive or update to use new approach

3. **data/README.md**: References ETL pipeline
   **Action**: Update documentation

### Files That Import from intent/
- **llm_jury/intent/__init__.py**: Self-reference only
- **Action**: None needed (will be deleted)

### Files We're NOT Removing
- ✅ **llm_jury/routing/xgboost_intent_classifier.py** (keep - different classifier)
- ✅ **tests/test_xgboost_intent_classifier.py** (keep - tests routing classifier)
- ✅ **llm_jury/routing/intent_classifier.py** (keep - enum definitions)

---

## 🔧 Cleanup Steps

### Step 1: Update Main __init__.py (2 min)

**File**: `llm_jury/__init__.py`

**Remove**:
```python
from llm_jury.etl import ETLPipeline, ArtificialAnalysisClient, DataMerger
```

**Keep**: Everything else (routing, ranking, optimization imports)

### Step 2: Move/Archive ETL Script (2 min)

**File**: `scripts/data_collection/fetch_scores_llm.py`

**Option A**: Archive it
```bash
mkdir -p archive/old_scripts/
git mv scripts/data_collection/fetch_scores_llm.py archive/old_scripts/
```

**Option B**: Delete it (if not used)
```bash
git rm scripts/data_collection/fetch_scores_llm.py
```

### Step 3: Remove etl/ and intent/ (1 min)

```bash
cd /Users/annette/repostitories/llm_jury
git rm -r llm_jury/etl/
git rm -r llm_jury/intent/
```

### Step 4: Update Documentation (5 min)

**Files to update**:
1. **data/README.md**: Remove ETL pipeline references
2. **Create DEPRECATED.md**: Document what was removed and why

### Step 5: Test Everything (5 min)

```bash
# Test imports
python -c "from llm_jury.prediction import load_model; print('✓')"

# Test routing still works
python -c "from llm_jury.routing import XGBoostIntentClassifier; print('✓')"

# Run tests
cd tests/
python -m pytest test_xgboost_intent_classifier.py -v
```

---

## ✅ Checklist

- [ ] **Step 1**: Remove etl imports from `llm_jury/__init__.py`
- [ ] **Step 2**: Archive or remove `fetch_scores_llm.py`
- [ ] **Step 3**: Remove `llm_jury/etl/` directory
- [ ] **Step 4**: Remove `llm_jury/intent/` directory
- [ ] **Step 5**: Update `data/README.md`
- [ ] **Step 6**: Create `DEPRECATED.md` documentation
- [ ] **Step 7**: Test imports work
- [ ] **Step 8**: Run relevant tests
- [ ] **Step 9**: Commit changes

---

## 📝 Commit Message Template

```
Remove deprecated etl/ and intent/ modules

These modules represented the old approach and are superseded by the
KDD/data production system now in llm_jury/prediction/.

Removed:
- llm_jury/etl/ (10 files)
  - ETL pipeline for benchmark data collection
  - Replaced by: KDD/data systematic data collection
  
- llm_jury/intent/ (4 files)
  - Embedding-based intent classification with length debiasing
  - Replaced by: llm_jury/prediction/ with NVIDIA features + XGBoost

Kept:
- llm_jury/routing/xgboost_intent_classifier.py (different system)
- llm_jury/prediction/ (new production system)

Rationale:
- Old approach: Embeddings + small labeled dataset + debiasing
- New approach: 113K examples + NVIDIA features + proven transfer
- Cleans up codebase, removes confusion about which system to use

Testing:
- All prediction module imports working
- Routing module tests passing
- No broken dependencies
```

---

## 🎯 Expected Outcome

### Before
```
llm_jury/
├── etl/           # ❌ Old ETL pipeline
├── intent/        # ❌ Old embedding-based classification
├── prediction/    # ✅ New production system
└── routing/       # ✅ XGBoost classifier (keep)
```

### After
```
llm_jury/
├── prediction/    # ✅ Production system (KDD/data models)
└── routing/       # ✅ Routing logic (uses prediction/)
```

**Result**: Clean, focused codebase with clear single approach

---

## ⚠️ Safety Notes

1. **Git preserves history**: Even after removal, old code is in git history
2. **Can restore if needed**: `git checkout <commit> -- llm_jury/etl/`
3. **Test before committing**: Make sure imports work
4. **Document the change**: Create DEPRECATED.md for reference

---

## 🚀 Benefits

1. **Clarity**: One clear approach for intent prediction
2. **Maintenance**: Less code to maintain
3. **Performance**: KDD models are proven (23/23 tests, validated transfer)
4. **Simplicity**: New developers don't see conflicting approaches

---

**Time Estimate**: 15 minutes total

**Status**: Ready to execute  
**Risk**: Low (old code preserved in git history)
