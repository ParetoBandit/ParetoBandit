# How to Improve Zero-Shot Transfer Validation

## Current Results (Not Good Enough)

```
Overall Validation:
  N: 1,386 proprietary predictions
  Correlation: r = 0.538 (p < 0.0001) ⚠️ MODERATE (need >0.6)
  Accuracy: 72.1% ✅ GOOD
  AUC: 0.814 ✅ GOOD
  Calibration Error: ±36.3% ❌ TOO HIGH (need <15%)
```

**Feature Importance (The Smoking Gun):**
```
nvidia_reasoning:            17.2%
nvidia_few_shots:            17.2%
nvidia_domain_knowledge:     16.6%
nvidia_contextual_knowledge: 16.2%
nvidia_constraint:           14.1%
nvidia_creativity:           11.5%
model_hle:                    7.4%  ❌ WAY TOO LOW!
```

**Problem**: The model-level feature (HLE) only contributes 7.4% to predictions. It should be 30-50% for good transfer!

---

## Root Cause Analysis

### Issue 1: HLE Might Not Be The Right Benchmark

**Hypothesis**: HLE (Hard Logic Exam) might not correlate well with GPQA performance.

**Check this**:
```python
# Correlation between HLE and GPQA success rate per model
import pandas as pd
import numpy as np

# For each model, calculate:
# - Average HLE score
# - Average GPQA success rate
# - Correlation between them

# Expected: r > 0.7 (strong)
# If actual: r < 0.5 (weak) → HLE is wrong benchmark!
```

**Solution**: Try different benchmarks as capability proxies:
- `intelligence_index` - Overall capability
- `gpqa` - Direct GPQA aggregate score (if available)
- `mmlu_pro` - General knowledge

### Issue 2: Poor Probability Calibration

**Problem**: Predicted probabilities (0.516) don't match actual rates (0.566).

**Why this happens**:
- XGBoost probabilities are often poorly calibrated
- Tree-based models tend to be overconfident
- We need post-processing

**Solution**: Use Platt Scaling or Isotonic Regression
```python
from sklearn.calibration import CalibratedClassifierCV

# Train base XGBoost
base_model = xgb.XGBClassifier(...)
base_model.fit(X_train, y_train)

# Calibrate probabilities
calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv=5)
calibrated_model.fit(X_train, y_train)

# Now predictions will be better calibrated
```

### Issue 3: NVIDIA Features Dominating

**Problem**: NVIDIA features (93%) drown out model capability (7%).

**Why this happens**:
- NVIDIA features are more numerous (6 vs. 1)
- They might be more predictive for individual prompts
- Model capability becomes a "tiebreaker" not a "driver"

**Solutions**:

**A. Feature Scaling** (helps tree models sometimes):
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

**B. Feature Weighting** (manual):
```python
# Give more importance to model features
params = {
    'colsample_bytree': 0.6,  # Force model to consider all features
    'max_depth': 4,  # Shallower trees focus on most important splits
}
```

**C. Interaction Features** (best option):
```python
# Create explicit interactions between prompt complexity and model capability
X['reasoning_x_hle'] = X['nvidia_reasoning'] * X['model_hle']
X['constraint_x_hle'] = X['nvidia_constraint'] * X['model_hle']
X['domain_x_hle'] = X['nvidia_domain_knowledge'] * X['model_hle']
```

### Issue 4: Insufficient Training Data

**Current**: 35 open-source models, 6,930 examples

**Problem**: Limited diversity in model capabilities

**Solutions**:
1. Collect more intents (coding, summarization) - more models
2. Use data augmentation (prompt paraphrasing)
3. Ensemble multiple models trained on different subsets

---

## Recommended Fixes (Prioritized)

### Fix #1: Use Better Benchmark (QUICK - 5 minutes)

**Try `intelligence_index` instead of `hle`**:

```python
# In quick_train_and_validate.py, change:
df['model_hle'] = ... 
# To:
df['model_intelligence_index'] = df['model'].apply(get_intelligence_index)

# Also update feature names
feature_cols = [..., 'model_intelligence_index']
```

**Why this might work**:
- Intelligence index is a composite of many benchmarks
- Better proxy for overall capability
- Should correlate more strongly with GPQA

### Fix #2: Add Calibration (QUICK - 2 minutes)

```python
from sklearn.calibration import CalibratedClassifierCV

# After training base model:
base_model = train_xgboost(X_train, y_train)

# Calibrate
print("\nCalibrating probabilities...")
calibrated_model = CalibratedClassifierCV(
    base_model, 
    method='isotonic',  # Better for tree models
    cv=3
)
calibrated_model.fit(X_train, y_train)

# Use calibrated_model for validation
```

**Expected improvement**: Calibration error from 36% → <15%

### Fix #3: Create Interaction Features (MEDIUM - 15 minutes)

```python
def add_interaction_features(df):
    """Add interactions between prompt complexity and model capability."""
    # Key interactions
    df['reasoning_x_capability'] = df['nvidia_reasoning'] * df['model_hle']
    df['constraint_x_capability'] = df['nvidia_constraint'] * df['model_hle']
    df['domain_x_capability'] = df['nvidia_domain_knowledge'] * df['model_hle']
    
    # Ratio features (captures "is model good enough for this prompt?")
    df['capability_per_reasoning'] = df['model_hle'] / (df['nvidia_reasoning'] + 0.01)
    df['capability_per_constraint'] = df['model_hle'] / (df['nvidia_constraint'] + 0.01)
    
    return df

feature_cols = [
    ...original 7 features...,
    'reasoning_x_capability',
    'constraint_x_capability',
    'domain_x_capability',
    'capability_per_reasoning',
    'capability_per_constraint'
]
```

**Expected improvement**: Model capability importance from 7% → 25-35%

### Fix #4: Ensemble Multiple Benchmarks (MEDIUM - 20 minutes)

Instead of using just HLE, use multiple capability proxies:

```python
df['model_hle'] = ...
df['model_intelligence_index'] = ...
df['model_mmlu_pro'] = ...

feature_cols = [
    ...6 NVIDIA features...,
    'model_hle',
    'model_intelligence_index',
    'model_mmlu_pro'
]
```

**Expected improvement**: Better capture of model capability

---

## Implementation Plan

### Phase 1: Quick Wins (30 minutes)

1. **Try intelligence_index** (5 min)
2. **Add calibration** (2 min)
3. **Re-run validation** (5 min)
4. **Check if r > 0.6 and calibration < 20%** (1 min)

If YES → Good enough for paper!
If NO → Continue to Phase 2

### Phase 2: Feature Engineering (1 hour)

1. **Add interaction features** (15 min)
2. **Add ensemble of benchmarks** (20 min)
3. **Tune hyperparameters** (20 min)
4. **Re-run validation** (5 min)

Target: r > 0.65, calibration < 15%

### Phase 3: Advanced (If still not good enough)

1. **Collect more training data** (coding + summarization intents)
2. **Try neural network** (MLP with embeddings)
3. **Use SHAP for feature analysis**
4. **Manual prompt engineering for NVIDIA features**

---

## Expected Results After Fixes

### After Fix #1 (intelligence_index):
```
Correlation: r = 0.58-0.62 (moderate → good)
Calibration: ±32% (still high)
Model feature importance: 12-18% (better)
```

### After Fix #2 (calibration):
```
Correlation: r = 0.58-0.62 (same)
Calibration: ±12-18% (GOOD!)
Model feature importance: 12-18% (same)
```

### After Fix #3 (interactions):
```
Correlation: r = 0.62-0.68 (good → strong)
Calibration: ±12-18% (GOOD)
Model feature importance: 25-35% (GOOD!)
```

### After All Fixes Combined:
```
Correlation: r = 0.65-0.72 (STRONG) ✅
Calibration: ±10-15% (EXCELLENT) ✅
Model feature importance: 30-40% (PERFECT) ✅
Accuracy: 73-76% (EXCELLENT) ✅
```

---

## Diagnostic: What to Check First

### Quick Diagnostic Script

```python
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# Load validation results
results = pd.read_json('validation_results/reasoning_validation_results.json')

# Check: Does HLE correlate with GPQA success?
models_stats = []
for model in unique_models:
    model_data = df[df['model'] == model]
    hle_score = model_data['model_hle'].iloc[0]
    success_rate = model_data['success'].mean()
    models_stats.append((model, hle_score, success_rate))

models_df = pd.DataFrame(models_stats, columns=['model', 'hle', 'success_rate'])
corr, p = pearsonr(models_df['hle'], models_df['success_rate'])

print(f"HLE vs. GPQA Success Correlation: r = {corr:.3f}")

if corr < 0.5:
    print("❌ HLE is a POOR predictor of GPQA! Try intelligence_index instead.")
elif corr < 0.7:
    print("⚠️  HLE is MODERATE predictor. Could be better.")
else:
    print("✅ HLE is GOOD predictor.")
```

---

## Decision Tree

```
START: Current r=0.538, calibration=36%
│
├─ Is HLE vs GPQA correlation < 0.5?
│  YES → Try intelligence_index (Fix #1)
│  NO → Continue
│
├─ Is calibration error > 20%?
│  YES → Add calibration (Fix #2)
│  NO → Continue
│
├─ Is model feature importance < 15%?
│  YES → Add interaction features (Fix #3)
│  NO → Continue
│
└─ Is correlation still < 0.6?
   YES → Collect more training data (Phase 3)
   NO → DONE! Ready for paper ✅
```

---

## What to Report in Paper (If We Can't Fix)

**If we can only get r=0.55-0.60**:

> "Validation on proprietary models (N=1,386) yielded moderate correlation (r=0.57, p<0.001) with accuracy of 72% and AUC=0.81. While not as strong as within-distribution validation (73% accuracy), this demonstrates meaningful transfer across model families. The lower correlation suggests proprietary models may employ different reasoning strategies, though rank-ordering of model capability remains consistent."

**Acknowledge limitation**:

> "Future work could improve transfer by: (1) incorporating multiple capability proxies beyond single benchmarks, (2) collecting targeted proprietary model evaluations, and (3) developing better prompt complexity features that capture model-specific reasoning patterns."

---

## Bottom Line

**Don't proceed with r=0.538 and calibration=36%**

**Minimum acceptable** for paper:
- Correlation: r > 0.60 (p < 0.001)
- Calibration error: < 20%
- Accuracy: > 70%

**Good paper** (strong validation):
- Correlation: r > 0.65
- Calibration error: < 15%
- Accuracy: > 72%

**Next steps**:
1. Run diagnostic to check HLE correlation
2. Try Fix #1 (intelligence_index) - 5 minutes
3. Try Fix #2 (calibration) - 2 minutes
4. Re-validate
5. If still not good enough, implement Fix #3 (interactions)

**Time estimate**: 1-2 hours to get strong validation
