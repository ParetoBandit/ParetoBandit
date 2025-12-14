# Code Migration Plan: KDD/data → llm_jury Library

**Date**: December 13, 2024  
**Goal**: Move production code from KDD/data into llm_jury library

---

## 🎯 Vision

### Before (Current State)
```
KDD/data/
├── core_scripts/
│   ├── opencompass_name_mappings.py    # Production code
│   ├── train_final_xgboost_models.py   # Production code
│   └── build_instance_level_training_data.py  # Production code
├── production_models/                   # Production models
└── instance_level_training_data/        # Training data

llm_jury/
└── prediction/                          # Wrappers that import from KDD/data
```

### After (Desired State)
```
llm_jury/
├── prediction/
│   ├── models.py                # Model name mappings (from opencompass_name_mappings.py)
│   ├── trainer.py               # Training code (from train_final_xgboost_models.py)
│   ├── data_collector.py        # Data collection (from build_instance_level_training_data.py)
│   ├── model_loader.py          # Model loading (already created)
│   └── feature_extractor.py     # Feature extraction
├── models/                      # Production models (moved from KDD/data/)
└── data/
    └── training_data/           # Training data (moved from KDD/data/)

KDD/data/
├── FINAL_SYSTEM_STATUS.md       # Paper documentation
├── documentation/               # Methodology, validation docs
├── validation/                  # Validation scripts (paper-specific)
├── tests/                       # Tests (can reference llm_jury)
└── archive/                     # Historical development
```

---

## 📦 Migration Map

### Files to Move

| From KDD/data/ | To llm_jury/ | New Name | Priority |
|----------------|--------------|----------|----------|
| `core_scripts/opencompass_name_mappings.py` | `prediction/` | `models.py` | 🔴 HIGH |
| `core_scripts/train_final_xgboost_models.py` | `prediction/` | `trainer.py` | 🟡 MEDIUM |
| `core_scripts/build_instance_level_training_data.py` | `prediction/` | `data_collector.py` | 🟡 MEDIUM |
| `production_models/*.joblib` | `models/production/` | (same) | 🔴 HIGH |
| `production_models/*.json` | `models/production/` | (same) | 🔴 HIGH |
| `instance_level_training_data/*.csv` | `data/training/` | (same) | 🟢 LOW |

### Files to Keep in KDD/data/

| File | Reason |
|------|--------|
| `validation/validate_*.py` | Paper-specific validation, references proprietary models |
| `documentation/` | Paper methodology documentation |
| `FINAL_SYSTEM_STATUS.md` | Paper summary |
| `tests/` | Can reference llm_jury library |
| `archive/` | Historical development |

---

## 🔧 Implementation Steps

### Phase 1: Move Model Mappings (15 min) 🔴

**Action**: Move `opencompass_name_mappings.py` → `llm_jury/prediction/models.py`

```bash
# Copy with git to preserve history
cd /Users/annette/repostitories/llm_jury
git mv KDD/data/core_scripts/opencompass_name_mappings.py llm_jury/prediction/models.py
```

**Update**: `llm_jury/prediction/name_resolver.py`
```python
# Before
from opencompass_name_mappings import OPENCOMPASS_TO_CACHE

# After
from .models import OPENCOMPASS_TO_CACHE
```

---

### Phase 2: Move Production Models (10 min) 🔴

**Action**: Move model files to library

```bash
# Create models directory in library
mkdir -p llm_jury/models/production

# Move production models
git mv KDD/data/production_models/*.joblib llm_jury/models/production/
git mv KDD/data/production_models/*.json llm_jury/models/production/
git mv KDD/data/production_models/README.md llm_jury/models/production/
```

**Update**: `llm_jury/prediction/model_loader.py`
```python
def get_models_dir() -> Path:
    """Get path to production models directory."""
    # Before: ../../KDD/data/production_models/
    # After: ../models/production/
    current_dir = Path(__file__).parent
    models_dir = current_dir.parent / 'models' / 'production'
    return models_dir
```

---

### Phase 3: Move Training Code (30 min) 🟡

**Action**: Refactor `train_final_xgboost_models.py` → `llm_jury/prediction/trainer.py`

**Key Changes**:
1. Convert script to library module with functions
2. Remove `if __name__ == '__main__'` execution code
3. Create clean API:
   ```python
   def train_intent_model(
       intent: str,
       data_path: str,
       output_dir: str,
       **xgb_params
   ) -> Tuple[xgb.XGBClassifier, dict]:
       """Train a production model for an intent."""
   ```

4. Keep the exact same logic/features
5. Add comprehensive docstrings

**Create CLI wrapper** in `llm_jury/cli.py`:
```python
@click.command()
@click.option('--intent', required=True)
@click.option('--data', required=True)
@click.option('--output', default='models/production')
def train(intent, data, output):
    """Train a production XGBoost model."""
    from llm_jury.prediction.trainer import train_intent_model
    model, metrics = train_intent_model(intent, data, output)
    click.echo(f"✓ Trained {intent}: Test AUC = {metrics['test_auc']:.3f}")
```

---

### Phase 4: Move Data Collection Code (30 min) 🟡

**Action**: Refactor `build_instance_level_training_data.py` → `llm_jury/prediction/data_collector.py`

**Key Changes**:
1. Convert to library module
2. Create function-based API:
   ```python
   def collect_intent_data(
       intent: str,
       benchmarks: List[str],
       output_path: str,
       nvidia_api_key: Optional[str] = None
   ) -> pd.DataFrame:
       """Collect training data for an intent."""
   ```

3. Keep exact same data processing logic
4. Add CLI wrapper

---

### Phase 5: Move Training Data (10 min) 🟢

**Action**: Move training data (optional - can keep symlink)

```bash
# Option A: Full move
mkdir -p llm_jury/data/training
git mv KDD/data/instance_level_training_data/* llm_jury/data/training/

# Option B: Symlink (lighter)
ln -s ../../../llm_jury/data/training KDD/data/instance_level_training_data
```

---

### Phase 6: Update All Imports (20 min) 🔴

**Files to Update**:

1. **KDD/data/validation/validate_all_4_intents.py**:
   ```python
   # Before
   sys.path.insert(0, '../core_scripts')
   from opencompass_name_mappings import OPENCOMPASS_TO_CACHE
   
   # After
   from llm_jury.prediction.models import OPENCOMPASS_TO_CACHE
   from llm_jury.prediction.model_loader import load_model
   ```

2. **KDD/data/validation/validate_rag_with_mmlu_pro.py**:
   ```python
   # Before
   sys.path.insert(0, '../core_scripts')
   
   # After
   from llm_jury.prediction.models import OPENCOMPASS_TO_CACHE
   ```

3. **KDD/data/tests/test_*.py**:
   ```python
   # Before
   sys.path.insert(0, '../core_scripts')
   
   # After
   from llm_jury.prediction.models import OPENCOMPASS_TO_CACHE
   from llm_jury.prediction.model_loader import load_model
   ```

4. **llm_jury/prediction/name_resolver.py**:
   ```python
   # Before
   sys.path.insert(0, str(_kdd_data_path))
   from opencompass_name_mappings import OPENCOMPASS_TO_CACHE
   
   # After
   from .models import OPENCOMPASS_TO_CACHE
   ```

---

### Phase 7: Update Documentation (15 min) 🟡

**Update Files**:
1. `llm_jury/README.md` - Point to new structure
2. `KDD/data/README.md` - Reference library code
3. `KDD/data/FINAL_SYSTEM_STATUS.md` - Update paths

---

### Phase 8: Run All Tests (10 min) 🔴

```bash
# Test library
cd llm_jury
python -m pytest tests/ -v

# Test KDD/data (should still work with new imports)
cd KDD/data
python -m pytest tests/ -v

# Test prediction module
cd llm_jury
python -m llm_jury.prediction.model_loader
```

---

## 📊 Benefits of Migration

### Before
- ❌ Code duplication (wrappers + original)
- ❌ Complex import paths
- ❌ KDD/data mixing research and production
- ❌ Hard to use library without KDD/data

### After
- ✅ Single source of truth in llm_jury
- ✅ Clean imports (`from llm_jury.prediction import ...`)
- ✅ KDD/data = pure research/documentation
- ✅ Library can be pip installed independently

---

## 🎯 Directory Structure After Migration

```
llm_jury/
├── llm_jury/
│   ├── prediction/
│   │   ├── __init__.py
│   │   ├── models.py              # ← Moved from KDD/data (name mappings)
│   │   ├── model_loader.py        # ✓ Already created
│   │   ├── name_resolver.py       # ✓ Already created
│   │   ├── trainer.py             # ← Moved from KDD/data (refactored)
│   │   ├── data_collector.py      # ← Moved from KDD/data (refactored)
│   │   ├── feature_extractor.py   # ← New (NVIDIA features)
│   │   ├── intent_predictor.py    # ← New (main API)
│   │   └── validator.py           # ← New (wraps validation)
│   │
│   ├── models/
│   │   └── production/
│   │       ├── reasoning_xgboost_model.joblib
│   │       ├── coding_xgboost_model.joblib
│   │       ├── summarization_xgboost_model.joblib
│   │       ├── rag_xgboost_model.joblib
│   │       ├── *_model_card.json (4 files)
│   │       └── README.md
│   │
│   ├── data/
│   │   └── training/
│   │       ├── instance_level_training_data.csv
│   │       └── instance_level_training_data.json
│   │
│   ├── routing/
│   ├── ranking/
│   ├── optimization/
│   └── ... (other modules)
│
├── KDD/
│   └── data/
│       ├── FINAL_SYSTEM_STATUS.md    # Paper documentation
│       ├── README.md                  # Paper data README
│       ├── documentation/             # Methodology docs
│       ├── validation/                # Validation scripts (uses library)
│       ├── tests/                     # Tests (uses library)
│       └── archive/                   # Historical development
│
└── README.md                          # Updated to reflect new structure
```

---

## 🚧 Potential Issues & Solutions

### Issue 1: Import Errors After Move
**Solution**: Run comprehensive grep to find all imports:
```bash
cd /Users/annette/repostitories/llm_jury
grep -r "from opencompass_name_mappings" .
grep -r "sys.path.insert.*KDD" .
```

### Issue 2: Tests Fail
**Solution**: Update test fixtures and paths systematically

### Issue 3: Git History Lost
**Solution**: Use `git mv` instead of `mv` to preserve history

### Issue 4: Large Training Data Files
**Solution**: Use symlinks or keep in KDD/data with reference

---

## 📋 Execution Checklist

- [ ] **Phase 1**: Move model mappings (HIGH PRIORITY)
  - [ ] `git mv` opencompass_name_mappings.py
  - [ ] Update name_resolver.py imports
  - [ ] Test imports work

- [ ] **Phase 2**: Move production models (HIGH PRIORITY)
  - [ ] Create `llm_jury/models/production/`
  - [ ] `git mv` all .joblib and .json files
  - [ ] Update model_loader.py path
  - [ ] Test model loading works

- [ ] **Phase 3**: Move training code (MEDIUM PRIORITY)
  - [ ] Refactor train_final_xgboost_models.py
  - [ ] Create trainer.py with clean API
  - [ ] Add CLI wrapper
  - [ ] Test training still works

- [ ] **Phase 4**: Move data collection (MEDIUM PRIORITY)
  - [ ] Refactor build_instance_level_training_data.py
  - [ ] Create data_collector.py
  - [ ] Add CLI wrapper

- [ ] **Phase 5**: Move training data (LOW PRIORITY)
  - [ ] Decide: move or symlink
  - [ ] Update paths if moved

- [ ] **Phase 6**: Update all imports (HIGH PRIORITY)
  - [ ] Update KDD/data/validation/
  - [ ] Update KDD/data/tests/
  - [ ] Update llm_jury/prediction/
  - [ ] Grep for any remaining old imports

- [ ] **Phase 7**: Update documentation (MEDIUM PRIORITY)
  - [ ] Update llm_jury/README.md
  - [ ] Update KDD/data/README.md
  - [ ] Update FINAL_SYSTEM_STATUS.md

- [ ] **Phase 8**: Test everything (HIGH PRIORITY)
  - [ ] Run llm_jury tests
  - [ ] Run KDD/data tests (23/23)
  - [ ] Test model loading
  - [ ] Test name resolution
  - [ ] Integration test

---

## ⏱️ Time Estimate

| Phase | Priority | Time | Dependencies |
|-------|----------|------|--------------|
| Phase 1 | 🔴 HIGH | 15 min | None |
| Phase 2 | 🔴 HIGH | 10 min | Phase 1 |
| Phase 6 | 🔴 HIGH | 20 min | Phase 1, 2 |
| Phase 8 | 🔴 HIGH | 10 min | All phases |
| Phase 3 | 🟡 MEDIUM | 30 min | Phase 1, 2 |
| Phase 4 | 🟡 MEDIUM | 30 min | Phase 1, 2 |
| Phase 7 | 🟡 MEDIUM | 15 min | Phase 1, 2 |
| Phase 5 | 🟢 LOW | 10 min | None |

**Total**: ~2-3 hours for complete migration

**Critical Path** (Phases 1, 2, 6, 8): ~55 minutes

---

## ✅ Success Criteria

1. ✅ All production code in `llm_jury/`
2. ✅ No code in `KDD/data/core_scripts/` (empty or removed)
3. ✅ All imports working (`from llm_jury.prediction import ...`)
4. ✅ All tests passing (23/23 in KDD/data, all in llm_jury)
5. ✅ Models loadable from new location
6. ✅ Documentation updated
7. ✅ Git history preserved
8. ✅ KDD/data becomes pure research/documentation

---

**Status**: Ready to execute  
**Recommendation**: Start with Phases 1, 2, 6, 8 (critical path, ~1 hour)
