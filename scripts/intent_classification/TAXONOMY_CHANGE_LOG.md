# Intent Classification Taxonomy Change Log

**Date**: December 10, 2025  
**Change**: Aligned intent classes with available composite scores

## Previous Taxonomy (DEPRECATED)

1. REASONING (15%)
2. CODING (15%)
3. FACTUAL_QA (15%)
4. **AGENTIC_EXECUTION (15%)** ❌ Removed
5. GENERAL (40%)

## New Taxonomy (CURRENT)

1. **CODING (20%)** → Aligned with **CCS** (Composite Coding Score)
2. **REASONING (20%)** → Aligned with **CRS** (Composite Reasoning Score)
3. **FACTUAL_QA (20%)** → Aligned with **CFS** (Composite Factual Score)
4. **SUMMARIZATION (20%)** → Aligned with **CSS** (Composite Summarization Score) ✅ NEW
5. **GENERAL (20%)** → Catch-all (no composite score needed)

## Rationale

### Why Remove AGENTIC_EXECUTION?

- **No composite score available**: We don't have a composite score for agentic/tool-use tasks
- **Low production frequency**: Tool-calling is less common than summarization in real LLM usage
- **Difficult to benchmark**: No established benchmarks for agentic execution in our dataset

### Why Add SUMMARIZATION?

- **Composite score exists**: CSS (Composite Summarization Score) is already computed for 61 models
- **Strong benchmarks**: SummEdits, Arena rankings, CNN/DailyMail, XSum
- **High production relevance**: Summarization is a core LLM use case
- **Better score coverage**: 61 models have CSS vs. 0 models with agentic scores

### Distribution Changes

**Before**: GENERAL dominated at 40% (because it was catch-all + creative writing)  
**After**: Balanced 20% each (GENERAL is still catch-all, but focused)

This creates a more balanced dataset where:
- Each specialized class gets 20% (matches model count: 61-100 models per composite score)
- GENERAL remains flexible for edge cases
- Distribution reflects available quality signals for routing decisions

## Impact on Data Collection

### New Data Sources

**SUMMARIZATION (NEW)**:
- Primary: CNN/DailyMail, XSum
- Format: "Summarize this article: [text]..."
- Prompts: Natural user requests for summarization

**Removed**:
- ❌ Glaive Function Calling v2 (AGENTIC_EXECUTION)
- ❌ ToolBench datasets

### Updated Scripts

**Modified**:
- `collect_real_intent_data.py`:
  - Replaced `load_glaive_agentic()` → `load_summarization_data()`
  - Reordered classes: CODING → REASONING → FACTUAL_QA → SUMMARIZATION → GENERAL
- `README.md`:
  - Updated intent class descriptions
  - Added composite score mappings
  - Updated distribution to 20% each

## Composite Score Alignment

| Intent Class | Composite Score | Benchmarks | Models |
|--------------|----------------|------------|---------|
| CODING | CCS | HumanEval, MBPP, LiveCodeBench, SciCode | 98 |
| REASONING | CRS | MATH-500, AIME, GPQA | 100 |
| FACTUAL_QA | CFS | MMLU-Pro, GPQA | 98 |
| SUMMARIZATION | CSS | SummEdits, Arena rankings | 61 |
| GENERAL | None | Arena overall ranking (fallback) | N/A |

## Router Implications

With this taxonomy, the router can now:

1. **Classify intent** → CODING, REASONING, FACTUAL_QA, SUMMARIZATION, or GENERAL
2. **Select model** based on corresponding composite score:
   - CODING → Use CCS to rank models
   - REASONING → Use CRS to rank models
   - FACTUAL_QA → Use CFS to rank models
   - SUMMARIZATION → Use CSS to rank models
   - GENERAL → Use arena_rank_overall or quality_index

This creates a clean mapping from intent → quality signal → model selection.

## Migration Path

### For Existing Labeled Data

If you have labeled data with AGENTIC_EXECUTION:
1. **Option 1**: Relabel as GENERAL (if catch-all appropriate)
2. **Option 2**: Relabel as CODING (if tool-calling involves code)
3. **Option 3**: Drop samples (if small dataset)

### For Future Work

If AGENTIC_EXECUTION becomes important:
1. Create composite score for agentic tasks (e.g., ToolBench, Glaive benchmarks)
2. Add back as 6th class (reduce GENERAL to 10-15%)
3. Update router to handle 6 classes

## Data Quality Impact

✅ **Improved**:
- All intents now have quality signals (composite scores)
- Router can make informed decisions for all specialized classes
- Balanced distribution (20% each) improves classifier training

✅ **Maintained**:
- Still 100% real data (no synthetic generation)
- All data from established HuggingFace datasets
- GENERAL remains flexible catch-all

---

**Status**: ✅ Taxonomy updated and ready for data collection  
**Next Steps**: Run `collect_real_intent_data.py` with new taxonomy
