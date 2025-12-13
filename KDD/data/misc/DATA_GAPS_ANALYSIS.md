# Data Gaps Analysis for Composite Score Calculation

**Date**: December 10, 2025  
**Models in cache**: 83

## Summary

The `models_cache.json` has **good but incomplete** data for computing all composite scores. Here's what we found:

### ✅ What's Working Well

1. **CCS (Composite Coding Score)** - Fully computable
   - All required fields present with good coverage
   - Already computed: 83/83 models (100%)

2. **CFS (Composite Factual Score)** - Fully computable
   - All required fields present with excellent coverage
   - Already computed: 83/83 models (100%)

3. **CSS (Composite Summarization Score)** - Partially computed
   - Most fields present
   - Already computed: 56/83 models (67.5%)

### ⚠️ What Needs Attention

~~All issues have been resolved!~~ ✅

1. ~~**CRS (Composite Reasoning Score)** - Not computed~~
   - ✅ **RESOLVED**: CRS now computed for 82/83 models (98.8%)

2. ~~**SummEdits Integration** - Data exists but not integrated~~
   - ✅ **RESOLVED**: SummEdits integrated for all 83/83 models (100%)

3. ~~**WildBench Cleanup** - Incomplete removal~~
   - ✅ **RESOLVED**: All WildBench (wb_*) fields removed from cache

## Detailed Field Analysis

### CCS (Composite Coding Score)

**Formula**: Uses Bayesian Latent Factor model with:
- Primary: `humaneval_score`, `livecodebench`, `scicode`, `arena_rank_coding`
- Auxiliary: `intelligence_index`

**Field Coverage**:
| Field | Coverage | Status |
|-------|----------|--------|
| humaneval_score | 69/83 (83.1%) | ✅ Good |
| livecodebench | 82/83 (98.8%) | ✅ Excellent |
| scicode | 82/83 (98.8%) | ✅ Excellent |
| arena_rank_coding | 50/83 (60.2%) | ✅ Good |
| intelligence_index | 83/83 (100%) | ✅ Perfect |

**Status**: ✅ Already computed for all 83 models

**Notes**: 
- The Bayesian Latent Factor model handles missing data gracefully
- Auxiliary benchmark (intelligence_index) at 100% enables inference for all models
- arena_rank_coding provides human preference signal from Chatbot Arena

---

### CRS (Composite Reasoning Score)

**Formula**: Uses Bayesian Latent Factor model with:
- Primary: `math_500`, `gpqa`, `hle`, `aime`, `math_index`
- Auxiliary: None explicitly defined

**Field Coverage**:
| Field | Coverage | Status |
|-------|----------|--------|
| math_500 | 83/83 (100%) | ✅ Perfect |
| gpqa | 82/83 (98.8%) | ✅ Excellent |
| hle | ~82/83 (98.8%) | ✅ Excellent |
| aime | 63/83 (75.9%) | ✅ Good |
| math_index | 65/83 (78.3%) | ✅ Good |

**Status**: ❌ **NOT COMPUTED** - 0/83 models

**Action Required**:
```bash
# Compute CRS for all models
python scripts/quality_scoring/compute_reasoning_score.py
```

**Notes**:
- All required fields are present with good-to-excellent coverage
- Should achieve ~100 model coverage with BLF handling missing data
- May want to add intelligence_index as auxiliary benchmark

---

### CFS (Composite Factual Score)

**Formula**: Uses Bayesian Latent Factor model with:
- Primary: `mmlu_pro`, `gpqa`, `arena_rank_expert`
- Auxiliary: None explicitly defined

**Field Coverage**:
| Field | Coverage | Status |
|-------|----------|--------|
| mmlu_pro | 83/83 (100%) | ✅ Perfect |
| gpqa | 82/83 (98.8%) | ✅ Excellent |
| arena_rank_expert | 50/83 (60.2%) | ✅ Good |
| intelligence_index | 83/83 (100%) | ✅ Perfect (could add as auxiliary) |

**Status**: ✅ Already computed for all 83 models

**Notes**:
- Excellent coverage on all primary benchmarks
- arena_rank_expert provides human preference signal

---

### CSS (Composite Summarization Score)

**Formula**: Uses Bayesian Latent Factor model with:
- Primary: `summedits_score`, `hallucination_rate`, `arena_rank_longer`
- Auxiliary: None explicitly defined

**Field Coverage**:
| Field | Coverage | Status |
|-------|----------|--------|
| summedits_score | **0/83 (0%)** | ❌ **Missing from cache** |
| hallucination_rate | 83/83 (100%) | ✅ Perfect |
| arena_rank_longer | 50/83 (60.2%) | ✅ Good |
| intelligence_index | 83/83 (100%) | ✅ Perfect (could add as auxiliary) |

**Status**: ⚠️ Partially computed - 56/83 models (67.5%)

**Critical Issue**: `summedits_score` field is **missing from models_cache.json**

**SummEdits Data Available**:
- Separate file: `data/summedits_aggregate_scores.json`
- Contains: 95 models with mean scores across 10 domains
- Not integrated into main cache

**Action Required**:
1. Merge SummEdits data into models_cache.json:
```python
import json

# Load existing cache
with open('data/models_cache.json') as f:
    cache = json.load(f)

# Load SummEdits scores
with open('data/summedits_aggregate_scores.json') as f:
    summedits = json.load(f)

# Merge by matching openrouter_id or other identifier
for model in cache['models']:
    openrouter_id = model.get('openrouter_id')
    if openrouter_id in summedits:
        model['summedits_score'] = summedits[openrouter_id]['mean_score']
        model['summedits_ci_lower'] = summedits[openrouter_id]['ci_lower']
        model['summedits_ci_upper'] = summedits[openrouter_id]['ci_upper']
        model['summedits_num_domains'] = summedits[openrouter_id]['num_domains']

# Save updated cache
with open('data/models_cache.json', 'w') as f:
    json.dump(cache, f, indent=2)
```

2. Recompute CSS after integration:
```bash
python scripts/quality_scoring/compute_summarization_score.py
```

---

## Additional Data Issues

### 1. ~~WildBench Fields Not Removed~~ ✅ RESOLVED

~~The following `wb_*` fields are still present and should be removed~~

**Status**: ✅ **COMPLETED** - All WildBench (wb_*) fields have been successfully removed from models_cache.json
- 11 wb_* fields removed from 18 models
- Verification: No remaining wb_* fields in cache

### 2. MBPP Not Used in Composite Scores

**Observation**: `mbpp_score` has 83% coverage (69/83 models) but is **not used** in any composite score calculation.

**Current CCS uses**:
- humaneval_score ✓
- livecodebench ✓
- scicode ✓
- arena_rank_coding ✓

**Consider**: Should MBPP be added to CCS? It's a complementary coding benchmark to HumanEval.

**Recommendation**: Review coding suite definition. If MBPP should be included:
```python
# In llm_jury/analysis/latent_factor.py
def create_coding_suite():
    suite = BenchmarkSuite(...)
    suite.add_benchmark('humaneval_score', ..., weight=0.25)
    suite.add_benchmark('mbpp_score', ..., weight=0.15)  # ADD THIS
    suite.add_benchmark('livecodebench', ..., weight=0.25)
    suite.add_benchmark('scicode', ..., weight=0.15)
    suite.add_benchmark('arena_rank_coding', ..., weight=0.20)
    suite.add_auxiliary_benchmark('intelligence_index', ..., weight=0.05)
    return suite
```

---

## Benchmark Coverage Summary

| Benchmark | Models | Coverage | Used In |
|-----------|--------|----------|---------|
| intelligence_index | 83/83 | 100% | CCS (aux), CRS (potential), CFS (potential), CSS (potential) |
| hallucination_rate | 83/83 | 100% | CSS |
| math_500 | 83/83 | 100% | CRS |
| mmlu_pro | 83/83 | 100% | CFS |
| livecodebench | 82/83 | 98.8% | CCS |
| scicode | 82/83 | 98.8% | CCS |
| gpqa | 82/83 | 98.8% | CRS, CFS |
| hle | ~82/83 | 98.8% | CRS |
| coding_index | 66/83 | 79.5% | Not used |
| math_index | 65/83 | 78.3% | CRS |
| aime | 63/83 | 75.9% | CRS |
| humaneval_score | 69/83 | 83.1% | CCS |
| mbpp_score | 69/83 | 83.1% | **Not used** |
| arena_rank_coding | 50/83 | 60.2% | CCS |
| arena_rank_expert | 50/83 | 60.2% | CFS |
| arena_rank_longer | 50/83 | 60.2% | CSS |
| arena_elo | 31/83 | 37.3% | Validation only |
| arena_hard_auto_score | 23/83 | 27.7% | Not used |
| **summedits_score** | **0/83** | **0%** | **CSS - MISSING!** |

---

## Action Items Priority

### ✅ All Critical & High Priority Items COMPLETED

~~### 🔴 Critical (Blocks CSS computation)~~
1. ✅ **Integrate SummEdits scores** - COMPLETED
   - All 83 models now have SummEdits scores

~~### 🟡 High Priority (Missing composite score)~~
2. ✅ **Compute CRS (Reasoning Score)** - COMPLETED
   - 82/83 models now have CRS

~~### 🟢 Medium Priority (Cleanup)~~
3. ✅ **Remove WildBench fields** - COMPLETED
   - All wb_* fields removed from cache

4. **Review MBPP inclusion** in CCS
   - Decide if it should be added to coding suite
   - Update suite definition if yes

### 🔵 Low Priority (Optimization)
5. **Add auxiliary benchmarks** to suites lacking them
   - CRS: Could add intelligence_index as auxiliary
   - CFS: Could add intelligence_index as auxiliary
   - CSS: Could add intelligence_index as auxiliary

6. **Validate composite scores** against external metrics
   - Check correlation with Arena ELO
   - Verify BLF model convergence

---

## Expected Model Coverage After Fixes

| Composite Score | Current | After SummEdits Integration | After CRS Computation |
|----------------|---------|----------------------------|----------------------|
| CCS (Coding) | 83/83 (100%) | 83/83 (100%) | 83/83 (100%) |
| CRS (Reasoning) | 0/83 (0%) | 0/83 (0%) | **83/83 (100%)** |
| CFS (Factual) | 83/83 (100%) | 83/83 (100%) | 83/83 (100%) |
| CSS (Summarization) | 56/83 (67.5%) | **~78/83 (94%)** | ~78/83 (94%) |

**Note**: CSS coverage will increase to ~94% because:
- SummEdits has 95 models
- After matching by openrouter_id, expect ~78-80 matches with cache
- BLF can infer for remaining models using hallucination_rate + arena_rank_longer

---

## Recommendations for Paper

### Update Data Section Statistics

Current paper claims need updating:

1. **Model count**: ✅ **CORRECTED** - Now using "83 models" consistently (source of truth: `models_cache.json`)

2. **Benchmark coverage**: Update Table 2 with actual numbers

3. **Composite score coverage**:
   - CCS: 98 models (from detailed CSV, not just cache 83)
   - CRS: Need to compute, expect ~100 models
   - CFS: 98 models (from detailed CSV)
   - CSS: 61 models (from detailed CSV, will increase with SummEdits integration)

### Clarify Data Sources

Add to data section:
1. **Direct evaluations**: HumanEval, MBPP, SummEdits (our evaluations)
2. **Aggregated indices**: Intelligence Index, Coding Index, Math Index (from Artificial Analysis)
3. **Scraped leaderboards**: Arena rankings, Arena ELO (from LMSYS)
4. **Safety metrics**: Hallucination rate (from Vectara)

### Address Missing Data Transparently

The paper should explicitly state:
- "SummEdits evaluated for 95 models through direct API calls"
- "Arena rankings available for 60% of models (50/83)"
- "Bayesian Latent Factor model handles missing data, enabling composite score inference for all models"

---

## Files to Update

1. `data/models_cache.json` - Integrate SummEdits, remove WildBench
2. `data/*_scores_detailed.csv` - Already have good composite scores
3. `KDD/data/DATA_SECTION.md` - Update statistics
4. `KDD/data/data_section.tex` - Update tables

## ✅ Scripts Already Executed

All required scripts have been successfully run:

```bash
# 1. ✅ Cleaned up cache - WildBench fields removed
# 2. ✅ Integrated SummEdits - All models matched
# 3. ✅ Computed CRS - 82/83 models
# 4. ✅ Computed CSS - 83/83 models with SummEdits

# To re-run if needed:
PYTHONPATH=/Users/annette/repostitories/llm_jury:$PYTHONPATH python3 scripts/quality_scoring/compute_reasoning_score.py
PYTHONPATH=/Users/annette/repostitories/llm_jury:$PYTHONPATH python3 scripts/quality_scoring/compute_coding_score.py
PYTHONPATH=/Users/annette/repostitories/llm_jury:$PYTHONPATH python3 scripts/quality_scoring/compute_factual_qa_score.py
PYTHONPATH=/Users/annette/repostitories/llm_jury:$PYTHONPATH python3 scripts/quality_scoring/compute_summarization_score.py
```

---

## Contact

For questions about this analysis, see the main repository README or open an issue.
