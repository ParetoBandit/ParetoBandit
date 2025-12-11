# Intent Classification Module

This module provides tools for classifying user prompts into intent categories for intelligent model routing.

## Quick Start

```python
from llm_jury.intent.length_debiasing import LengthDebiaser
from sentence_transformers import SentenceTransformer
import xgboost as xgb

# 1. Get embeddings
embedder = SentenceTransformer('all-MiniLM-L6-v2')
X = embedder.encode(prompts)
lengths = [len(p) for p in prompts]

# 2. Remove length bias (RECOMMENDED)
debiaser = LengthDebiaser(method='orthogonal_projection')
X_clean, info = debiaser.fit_transform(X, lengths)

# 3. Train classifier
model = xgb.XGBClassifier()
model.fit(X_clean, labels)
```

## Length Debiasing Methods

### Orthogonal Projection (RECOMMENDED) ✅

**Best trade-off**: 75% artifact reduction, only 6.4% accuracy cost

```python
debiaser = LengthDebiaser(method='orthogonal_projection')
X_clean, info = debiaser.fit_transform(X, lengths)
```

**Results:**
- Accuracy: 88.1% (vs 94.5% baseline)
- Length artifact: 25% failure (vs 100% baseline)
- Correlation: -0.10 → 0.00

**Use when:** You want production-ready debiasing with proven results.

---

### Iterative Null-space Projection (INLP) ⚠️

**Over-corrects**: Removes too much information

```python
debiaser = LengthDebiaser(method='inlp', max_iterations=30)
X_clean, info = debiaser.fit_transform(X, lengths)
```

**Results:**
- Accuracy: 80.6% (vs 94.5% baseline) ❌ Too low
- Length artifact: 75% failure ❌ Worse than single projection
- 10 iterations needed

**Use when:** Research only - not recommended for production.

---

### Inverse Probability Weighting (IPW) ❌

**No effect**: Can't fix what isn't in the data

```python
debiaser = LengthDebiaser(method='ipw')
X, info = debiaser.fit_transform(X, lengths, y)
weights = info['weights']

# Use weights in training
model.fit(X, y, sample_weight=weights)
```

**Results:**
- Accuracy: 94.8%
- Length artifact: 100% failure ❌ No improvement
- Returns weights, not transformed embeddings

**Use when:** You want to reweight samples but expect no debiasing effect.

---

### No Debiasing (Baseline) ⚠️

**High accuracy but biased**

```python
debiaser = LengthDebiaser(method='none')
X_clean, info = debiaser.fit_transform(X, lengths)
```

**Results:**
- Accuracy: 94.5% ✅
- Length artifact: 100% failure ❌ Critical bias

**Use when:** Benchmarking or you don't care about length bias.

---

## Comparison Summary

| Method | Accuracy | Length Artifact | Verdict |
|--------|----------|----------------|---------|
| **Orthogonal Projection** | **88.1%** | **25% failure** | ✅ **BEST** |
| INLP | 80.6% | 75% failure | ❌ Over-corrects |
| IPW | 94.8% | 100% failure | ❌ No effect |
| Baseline | 94.5% | 100% failure | ⚠️ Biased |

---

## Advanced Usage

### Compare All Methods

```python
from llm_jury.intent.length_debiasing import compare_methods

results = compare_methods(X, lengths, y, verbose=True)
```

### Save/Load Debiaser

```python
import pickle

# Save
with open('debiaser.pkl', 'wb') as f:
    pickle.dump(debiaser, f)

# Load and apply to new data
with open('debiaser.pkl', 'rb') as f:
    debiaser = pickle.load(f)

X_new_clean = debiaser.transform(X_new, lengths_new)
```

### Custom Parameters

```python
# INLP with custom thresholds
debiaser = LengthDebiaser(
    method='inlp',
    max_iterations=20,
    r2_threshold=0.01,
    corr_threshold=0.01
)
```

---

## Why Orthogonal Projection Wins

1. **Fairness**: Removes systematic bias without destroying semantics
2. **Simplicity**: Single projection, no hyperparameters
3. **Speed**: ~10ms overhead per batch
4. **Proven**: Tested on 2,458 real prompts, fixes 75% of failures
5. **Reproducible**: Deterministic, no random initialization

**Trade-off accepted**: 6.4% accuracy cost for unbiased predictions

---

## Citation

If you use this module, please cite:

```bibtex
@inproceedings{intent-classification-kdd2025,
  title={Intent Classification with Length Debiasing for LLM Routing},
  author={...},
  booktitle={KDD},
  year={2025}
}
```

---

## References

- Orthogonal Projection: Ridge regression-based decorrelation
- INLP: Ravfogel et al. (2020) "Null It Out: Guarding Protected Attributes"
- IPW: Inverse probability weighting for sample rebalancing

---

## Support

For issues or questions:
1. Check `examples/train_with_debiasing.py` for working examples
2. See `KDD/intent_classification/COMPREHENSIVE_SOLUTION_COMPARISON.md` for detailed comparison
3. Open an issue on GitHub
