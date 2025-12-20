# Benchmark Data Audit - Final Report

## Executive Summary

**Finding:** The Artificial Analysis (AA) API data is the **correct and preferred** source for fair model comparison.

**Reason:** AA provides standardized, independent evaluation using the same methodology for all models.

---

## Why AA Data is More Trustworthy

### 1. Standardized Methodology (Same Ruler for All)

| Aspect | Official Tech Reports | Artificial Analysis |
|--------|----------------------|---------------------|
| **Prompt Format** | Model-specific (optimized) | Standardized (same for all) |
| **Evaluation** | Best-of-N (cherry-picked) | Pass@1 (single run) |
| **Grader** | Custom per-company | Uniform for all models |
| **Bias** | Self-reported marketing | Independent auditor |

### 2. The IFBench vs IFEval Trap (CRITICAL!)

**These are DIFFERENT benchmarks:**

| Benchmark | Source | # Prompts | GPT-4o Score |
|-----------|--------|-----------|--------------|
| **IFEval** | Google/DeepMind | ~500 | ~85% |
| **IFBench** | Artificial Analysis | 58 (adversarial) | ~34% |

⚠️ **The ~34% IFBench score is CORRECT.** It's a much harder test with adversarial constraints.

### 3. Independent Validation > Self-Reporting

> "Official Report: We investigated ourselves and found we are the best." (Low credibility)
>
> "Artificial Analysis: We tested both models with the exact same ruler." (High credibility)

---

## Benchmark Data (From AA API)

| Model | MATH-500 | Code | IFBench | Source |
|-------|----------|------|---------|--------|
| **DeepSeek V3** | **94.2%** | 91.6% | 41.0% | AA API |
| GPT-4o | 75.9% | 90.2% | 34.3% | AA API |
| Nova-Lite | 76.5% | 58.0% | 34.1% | AA API |

### Key Finding ✅
**DeepSeek V3 beats GPT-4o on ALL benchmarks in independent evaluation:**
- MATH-500: 94.2% vs 75.9% (**+24%**)
- Code: 91.6% vs 90.2% (+1.4%)
- IFBench: 41.0% vs 34.3% (+6.7%)

---

## Paper Footnote (Recommended)

Add this footnote to your methodology section:

> "Benchmarks sourced from the Artificial Analysis API (Independent Evaluation, 
> artificialanalysis.ai) to ensure methodological consistency across models. 
> Note that 'IFBench' is a distinct, adversarial instruction-following benchmark 
> (58 complex constraints) and is not comparable to Google's 'IFEval' (~500 prompts)."

---

## Files Using AA Data

1. ✅ `calibrate_multi_domain.py` - Uses `REAL_CAPABILITIES` from AA
2. ✅ `train_priors.py` - Uses `MATH_CAPABILITIES` from AA
3. ✅ `run_rq2_poisoned.py` - Uses AA MATH-500 scores (0.759, 0.942)
4. ✅ `models_cache.json` - Contains full AA API data for 81 models

---

## Source Citation

```bibtex
@misc{artificialanalysis2025,
  title={LLM Benchmark Data},
  author={{Artificial Analysis}},
  year={2025},
  url={https://artificialanalysis.ai},
  note={Independent evaluation using standardized methodology}
}
```
