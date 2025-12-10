# Intent Classification - Ground Truth Data Collection

This folder contains scripts for collecting labeled intent classification data where **dataset source = ground truth label**.

## Methodology

Unlike traditional approaches that require teacher labeling (GPT-4/Claude) or manual annotation:

1. ✅ **Sample from domain-specific benchmarks** where prompts are definitively one intent
2. ✅ **Dataset source defines the label** (no ambiguity, no annotation needed)
3. ✅ **Compare classification methods** to see which best predicts ground-truth labels

## Collection Script

**`collect_real_intent_data.py`**
- Collects real prompts from HuggingFace datasets
- Each prompt labeled by its source dataset
- Output: `data/real_intent_prompts_labeled.json`

```bash
python scripts/intent_classification/collect_real_intent_data.py \
    --samples-per-class 2000 \
    --output data/real_intent_prompts_labeled.json
```

## Intent Classes (6 Total)

Each intent maps to a composite score for model routing:

| Intent | Distribution | Composite Score | Sources | Ground Truth |
|--------|--------------|-----------------|---------|--------------|
| **CODING** | ~17% | CCS | MBPP, HumanEval | ✅ Prompts are coding tasks |
| **REASONING** | ~17% | CRS | GSM8k, MATH | ✅ Prompts are math/reasoning |
| **FACTUAL_QA** | ~17% | CFS | Natural Questions, TriviaQA | ✅ Prompts are factual questions |
| **SUMMARIZATION** | ~17% | CSS | CNN/DailyMail, XSum | ✅ Prompts are summarization requests |
| **AGENTIC_EXECUTION** | ~17% | CAE | Glaive Function Calling v2 | ✅ Prompts are tool-use tasks |
| **GENERAL** | ~17% | Arena rankings | LMSYS Chat-1M (filtered) | ✅ Prompts are general conversation |

## Data Quality Guarantees

- ✅ **Real human prompts** from established datasets
- ✅ **No synthetic generation** (all sources are real)
- ✅ **No ambiguous labels** (dataset source = definitive intent)
- ✅ **Balanced distribution** across all 6 classes
- ✅ **No teacher labeling bias** (no LLM annotation needed)

## Next Steps

1. **Collect labeled data**: Run `collect_real_intent_data.py`
2. **Split data**: Create train/val/test splits
3. **Train classifiers**: Compare multiple approaches:
   - Embedding + XGBoost (fast, lightweight)
   - Fine-tuned transformer (BERT, RoBERTa)
   - Few-shot LLM (GPT-4, Claude)
4. **Evaluate**: See which method best predicts ground-truth labels

## Why This Approach?

Traditional intent classification requires:
- ❌ Expensive LLM labeling (GPT-4 API calls)
- ❌ Manual annotation (time-consuming, subjective)
- ❌ Label ambiguity (inter-annotator disagreement)

Our approach:
- ✅ **Free labeling** (dataset source = label)
- ✅ **No annotation needed** (immediate ground truth)
- ✅ **Zero ambiguity** (MBPP prompts ARE coding, GSM8k prompts ARE reasoning)

---

**Status**: Updated December 10, 2025 - Ground truth labeling methodology
