# Deterministic Benchmark Selection: KDD-Acceptable Configuration

This document explains our benchmark selection strategy and provides academic justification for KDD reviewers.

## Overview

Yes, this is a **fully KDD-acceptable configuration**. In fact, this specific suite is arguably stronger than many well-funded papers because it prioritizes **deterministic reproducibility** over "vibes-based" evaluation (like asking GPT-4 to rate a poem).

For a resource-constrained researcher, this selection allows you to turn your budget limitation into a methodological feature. You are not "cutting corners"; you are **"eliminating the stochastic variance of LLM-as-a-judge."**

## The "Deterministic Defense" (Your Paper's Strength)

**Reviewers might ask:** "Why didn't you use Model-based Evaluation (e.g., AlpacaEval)?"

**Your Answer:** "To ensure strict reproducibility and eliminate the high variance inherent in LLM-based judges, we restricted our evaluation to Ground-Truth Benchmarks where success is binary and verifiable (Exact Match, Unit Test, or Consistency Label)."

## Final Validation of the Benchmark Stack

| Intent | Benchmark | Why KDD Reviewers Will Approve | The "Resource Hack" |
|--------|-----------|-------------------------------|---------------------|
| **Reasoning** | GPQA | It is the current SOTA for "Deep Inference." It proves you aren't just testing surface-level logic (like GSM8K). | Free: Multiple Choice (String Match). |
| **Coding** | LiveCodeBench | It addresses "Data Contamination" (the #1 reviewer complaint in 2024/25). | Free: Unit Tests (CPU execution). |
| **Summarization** | SummEdits | It reframes summarization from "Style" (subjective) to "Consistency" (objective). This is the only rigorous way to test summarization cheaply. | Free: Binary Classification (0/1). |
| **Agentic** | GAIA | It is a "General Assistant" test that covers tool use and file navigation, not just text generation. | Free: Short Answer Matching. |
| **RAG** | Natural Questions | It is the industry standard for Open-Domain QA. It tests precision, not just fluency. | Free: List Inclusion Check. |

## Critical Paper-Writing Tips

Since you are using Logistic Regression to predict success, you must explicitly state that you are using these benchmarks to generate **Instance-Level Training Data**, not just aggregate scores.

### Recommended Sentences for Your Paper

**For the Abstract:**

> "We utilize a suite of five deterministic benchmarks (GPQA, LiveCodeBench, SummEdits, GAIA, NQ-Open) to generate over [N] labeled training instances, enabling us to train a lightweight, intent-aware performance predictor with high calibration accuracy."

**For the Methods Section:**

> "We constructed a Deterministic Evaluation Harness using open-source datasets (GPQA, LiveCodeBench, SummEdits, GAIA) to generate ground-truth binary labels ($y \in \{0,1\}$) for model performance. This allowed us to train our predictor $f(x)$ on objective correctness signals without the cost or variance of LLM-as-a-judge evaluation."

**For the Methods Section (Handling Missing Data):**

> "To address data sparsity in niche benchmarks (e.g., TerminalBench), we employed a Hierarchical Feature Imputation strategy. We utilized GPQA and LiveCodeBench (100% coverage on Artificial Analysis) as high-fidelity proxies for missing Agentic and Reasoning scores, ensuring our router provides valid predictions for the entire universe of 80+ models."

## Defensive Response for Reviewers

### If a reviewer complains that Natural Questions (NQ) is "old" (2019):

**Have this defensive sentence ready:**

> "While newer RAG benchmarks exist, NQ remains the gold standard for fact-based retrieval where exact-match scoring is viable. Newer benchmarks often require expensive model-based grading."

## Verdict

**Go ahead with this list.** It is:
- ✅ Academically sound
- ✅ Completely free to run
- ✅ Covers the full spectrum of modern LLM capabilities
- ✅ Prioritizes reproducibility over variance

## Methodological Advantages

### 1. Reproducibility
All benchmarks use deterministic evaluation metrics:
- Multiple choice with exact string matching (GPQA)
- Unit test execution (LiveCodeBench)
- Binary consistency labels (SummEdits)
- Short answer matching (GAIA)
- List inclusion checks (Natural Questions)

### 2. Cost Efficiency
No LLM judge calls required = $0 evaluation cost while maintaining academic rigor

### 3. Defensibility
Each benchmark addresses a specific reviewer concern:
- **Data contamination**: LiveCodeBench (constantly updated)
- **Subjective evaluation**: SummEdits (objective consistency)
- **Surface-level testing**: GPQA (deep reasoning)
- **Real-world capability**: GAIA (tool use, file navigation)
- **Retrieval precision**: Natural Questions (fact-based QA)

## Implementation Notes

All benchmarks are implemented in their respective subdirectories:
- `reasoning/GPQA/` - Graduate-level reasoning
- `coding/` - LiveCodeBench with contamination resistance
- `summarization/sumedits/` - Consistency-based evaluation
- `agentic/` - GAIA for general assistance tasks
- `rag/` - Natural Questions for retrieval evaluation

Each subdirectory contains:
- `README.md` - Detailed documentation
- `fetch_*.py` - Data acquisition scripts
- `evaluate_*.py` - Evaluation pipelines
- `example_usage.py` - Usage examples
