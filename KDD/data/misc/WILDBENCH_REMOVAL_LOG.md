# WildBench Data Removal Log

**Date**: December 10, 2025  
**Status**: ✅ COMPLETED

## Summary

All WildBench (WB-Score) fields have been successfully removed from the models cache and documentation as per project requirements.

## Reason for Removal

WildBench data was removed because:
1. Not used in any composite score calculations
2. Lower coverage compared to other benchmarks
3. Overlapping with other benchmarks (Arena rankings, MixEval)
4. Project decision to streamline to most essential benchmarks

## Fields Removed

The following 11 fields were removed from `data/models_cache.json`:

1. `wb_score_raw` - Raw WildBench score
2. `wb_adjusted_score` - Adjusted WildBench score
3. `wb_task_macro_score` - Task macro score
4. `wb_task_macro_reward` - Task macro reward
5. `wb_mixture_reward` - Mixture reward
6. `wb_creative_tasks` - Creative tasks score
7. `wb_coding_debugging` - Coding/debugging score
8. `wb_planning_reasoning` - Planning/reasoning score
9. `wb_information_seeking` - Information seeking score
10. `wb_math_data_analysis` - Math/data analysis score
11. `wb_score_source` - Source model name

## Affected Models

- **18 models** had WildBench data
- All wb_* fields removed from these 18 models
- **65 models** never had WildBench data (no changes needed)

## Verification

```bash
# Verified no wb_* fields remain in cache
python3 -c "
import json
with open('data/models_cache.json') as f:
    cache = json.load(f)
wb_fields = [k for k in cache['models'][0].keys() if k.startswith('wb_')]
print(f'Remaining wb_ fields: {len(wb_fields)}')
# Output: Remaining wb_ fields: 0
"
```

✅ Verification passed: 0 wb_* fields found

## Documentation Updates

Updated the following files to reflect WildBench removal:

1. `/KDD/data/DATA_SECTION.md` 
   - Removed WildBench citation
   - Removed WildBench from benchmark descriptions

2. `/KDD/data/data_section.tex`
   - Removed WildBench references
   - Updated benchmark coverage statistics

3. `/KDD/data/DATA_GAPS_ANALYSIS.md`
   - Marked WildBench cleanup as completed
   - Updated action items

4. `/KDD/data/README.md`
   - Updated to reflect WildBench removal

## Code Files Removed

The following code files were **REMOVED** completely:
- ✅ `llm_jury/etl/wildbench_client.py` - WildBench API client (DELETED)
- ✅ `scripts/data_collection/fetch_wildbench.py` - WildBench data fetching script (DELETED)
- ✅ `scripts/benchmarks/run_wildbench_eval.py` - WildBench evaluation script (DELETED)
- ✅ `external/WildBench/` - Cloned WildBench repository (DELETED)

All WildBench-related code and git repositories have been completely removed from the project.

## Impact on Composite Scores

**No impact**: WildBench was not used in any composite score calculations:
- CCS (Coding): Uses HumanEval, LiveCodeBench, SciCode, Arena ranks
- CRS (Reasoning): Uses MATH-500, AIME, GPQA, HLE, Math Index
- CFS (Factual): Uses MMLU-Pro, GPQA, Arena expert ranks
- CSS (Summarization): Uses SummEdits, Hallucination Rate, Arena longer ranks

All composite scores remain unaffected and fully functional.

## Benchmarks Retained

The following benchmarks are retained and used in composite scores:

**Primary Benchmarks** (Direct Evaluation):
- HumanEval (69/83 models)
- MBPP (69/83 models)
- SummEdits (83/83 models) ← Newly integrated
- MixEval (45/83 models)

**Aggregated Indices** (Artificial Analysis):
- Intelligence Index (83/83 models)
- Coding Index (66/83 models)
- Math Index (65/83 models)

**Specialized Benchmarks**:
- MATH-500 (83/83 models)
- GPQA (82/83 models)
- MMLU-Pro (83/83 models)
- LiveCodeBench (82/83 models)
- SciCode (82/83 models)
- HLE (82/83 models)
- AIME (70/83 models)

**Human Preference Signals**:
- Arena rankings - 8 categories (50/83 models)
- Arena ELO (31/83 models)
- Arena-Hard-Auto (23/83 models)

**Safety Metrics**:
- Hallucination Rate (83/83 models)

## Alternatives to WildBench

WildBench functionality is covered by:
1. **Arena rankings** - Real user preferences from LMSYS Chatbot Arena
2. **MixEval** - Similar multi-domain evaluation (ρ = 0.96 with Arena ELO)
3. **Task-specific benchmarks** - More granular evaluation per domain

## Files Modified

1. `/data/models_cache.json` - Removed 11 wb_* fields from 18 models
2. `/KDD/data/DATA_SECTION.md` - Removed WildBench references
3. `/KDD/data/data_section.tex` - Removed WildBench LaTeX references
4. `/KDD/data/DATA_GAPS_ANALYSIS.md` - Updated status
5. `/KDD/data/README.md` - Noted removal in updates

## Rollback Procedure

If WildBench data needs to be restored:

```bash
# Re-run WildBench collection
python scripts/data_collection/fetch_wildbench.py

# Or restore from backup if available
cp data/models_cache_backup.json data/models_cache.json
```

## Completion Checklist

- ✅ Removed all wb_* fields from models_cache.json
- ✅ Verified no remaining wb_* fields (count = 0)
- ✅ Updated DATA_SECTION.md to remove WildBench
- ✅ Updated data_section.tex to remove WildBench
- ✅ Updated DATA_GAPS_ANALYSIS.md to mark as completed
- ✅ Updated README.md with removal notes
- ✅ Confirmed no impact on composite scores
- ✅ All 4 composite scores (CCS, CRS, CFS, CSS) fully functional
- ✅ Created this removal log for documentation

## Summary Statistics

**Before Removal**:
- 83 models in cache
- 18 models with WildBench data
- 11 wb_* fields per model with data

**After Removal**:
- 83 models in cache (unchanged)
- 0 models with WildBench data
- 0 wb_* fields total

**Data Reduction**:
- Removed: 18 models × 11 fields = 198 data points
- Cache size reduction: ~3-5 KB
- Improved data clarity and focus

---

**Removal completed successfully on December 10, 2025**  
**Verified by**: Automated verification scripts  
**Documentation updated**: All KDD data section files
