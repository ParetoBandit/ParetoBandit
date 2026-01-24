# Table 1: Dataset Composition and Provenance

This directory contains the analysis and LaTeX table for **Table 1** in the KDD paper, which documents the complete data provenance and composition of the BanditGPT evaluation dataset.

## 📊 Overview

The table breaks down **81,871 prompts** across three splits:
- **Warmup Set (80,000)**: Used for PCA training and LinUCB warmup priors
- **Dev Set (1,121)**: Used for online learning and calibration
- **Holdout Set (750)**: Held-out test set for final evaluation

## 🎯 Purpose

This table serves multiple critical functions for KDD reviewers:

1. **Data Transparency**: Shows exactly where our data comes from
2. **Semantic Coverage**: Demonstrates breadth across 5 semantic categories
3. **Reproducibility**: Provides exact dataset sources and sizes
4. **Quality Assurance**: Documents disjoint splits (no data leakage)

## 📁 Files

```
experiments_v1/01_table/
├── README.md                           # This file
├── analyze_dataset_composition.py      # Analysis script
└── table_dataset_composition.tex       # LaTeX table output
```

## 🔍 Data Provenance

### 1. Warmup Set (80,000 prompts)

**Source**: LMSYS Arena battles via RouteLLM dataset  
**HuggingFace**: `routellm/gpt4_judge_battles`  
**Access**: Public dataset (requires HF token)

**Purpose**:
- Train PCA model (384 → 32 dimensions)
- Generate LinUCB warmup priors:
  - **A matrix** (covariance): 33×33 per model - captures feature correlations and uncertainty
  - **b vector** (beliefs): 33×1 per model - encodes reward expectations for different contexts

**Models**:
- `mistralai/mixtral-8x7b-instruct` (weak, $0.54/M tokens)
- `openai/gpt-4-turbo` (strong, $20/M tokens)

**Processing**:
1. Downloaded from HuggingFace using `scripts/download_and_process_routellm.py`
2. Filtered for mixtral vs gpt-4-turbo battles
3. Extracted unique prompts (deduplicated)
4. Used for PCA training: `scripts/train_pca_from_routellm.py`
5. Used for warmup priors: `scripts/generate_warmup_priors.py`

**Artifacts**:
- `src/artifacts/pca_32.joblib` (PCA model, 32 components)
- `src/artifacts/priors_warmup.joblib` (LinUCB priors, 33 dims with bias)

### 2. Dev Set (1,121 prompts)

**Source**: KDD rigorous splits (stratified by difficulty)  
**File**: `data/dev_prompts_for_rejudge.jsonl`

**Purpose**:
- Online learning (bandit trains here)
- Calibration (finding optimal gamma)
- Model comparison (BanditGPT vs baselines)

**Models**:
- `mistralai/mixtral-8x7b-instruct`
- `openai/gpt-4o` (note: gpt-4o, not gpt-4-turbo)

**Stratification**:
Prompts are stratified by:
- **Category**: STEM, CODE, GENERAL
- **Complexity**: Low, Med, High
- **Difficulty**: Easy, Hard, Contentious

**Processing**:
1. Generated rewards: `scripts/generate_gpt4_turbo_rewards.py`
2. Judged using GPT-4o pairwise comparison (RouteLLM methodology)
3. Split using stratified sampling: `src/bandit_gpt/utils/experiment.py`

### 3. Holdout Set (750 prompts)

**Source**: KDD rigorous splits (stratified by difficulty)  
**File**: `data/holdout_prompts_for_rejudge.jsonl`

**Purpose**:
- Final evaluation (held-out test set)
- Pareto frontier curves
- Cost-quality tradeoff analysis

**Models**: Same as Dev Set

**Guarantee**: Completely disjoint from warmup and dev sets (verified)

## 📈 Semantic Categories

Prompts are classified into 5 semantic categories using keyword-based heuristics:

| Category | Description | Examples |
|----------|-------------|----------|
| **Coding** (39.0%) | Programming, debugging, code review | "Write a Python function to...", "Debug this code..." |
| **Conversational** (37.5%) | General chat, advice, simple queries | "Tell me about...", "What's the difference between..." |
| **Creative** (10.0%) | Writing, storytelling, poetry | "Write a story about...", "Compose a poem..." |
| **Knowledge** (9.5%) | Factual questions, explanations | "What is...", "Explain the history of..." |
| **Math/Logic** (3.9%) | Mathematics, reasoning, proofs | "Solve the integral...", "Prove that..." |

### Category Distribution

```
Coding:          31,964 prompts (39.0%)
Conversational:  30,731 prompts (37.5%)
Creative:         8,184 prompts (10.0%)
Knowledge:        7,801 prompts (9.5%)
Math/Logic:       3,188 prompts (3.9%)
```

## 🔧 Reproduction

To regenerate the table:

```bash
cd experiments_v1/01_table
python analyze_dataset_composition.py
```

**Output**:
- Console: Analysis summary and statistics
- File: `table_dataset_composition.tex` (LaTeX table)

## 📝 LaTeX Integration

To include the table in your paper:

```latex
\input{experiments_v1/01_table/table_dataset_composition.tex}
```

Or copy the contents directly into your paper.

## 🎨 Table Features

The table follows KDD formatting guidelines:

- ✅ Uses `booktabs` package (professional horizontal rules)
- ✅ Includes descriptive caption
- ✅ Has detailed table notes explaining data sources
- ✅ Shows both absolute counts and percentages
- ✅ Documents semantic category breakdown
- ✅ Cites original data sources

## 🔍 Quality Assurance

### Data Leakage Prevention

**Verified**: Warmup set is completely disjoint from evaluation sets

```python
# Check performed in scripts/generate_warmup_priors.py
def check_data_leakage(train_prompts: set, eval_file: Path):
    # Raises error if any overlap detected
    # Result: 0 overlapping prompts ✅
```

### Stratification Verification

Dev and holdout sets are stratified by:
1. **Semantic category** (CODE, STEM, GENERAL)
2. **Complexity** (Low, Med, High)
3. **Difficulty** (Easy, Hard, Contentious)

This ensures representative coverage across task types and difficulty levels.

## 📚 References

1. **RouteLLM Dataset**: Ong, I., et al. (2024). "RouteLLM: Learning to Route LLMs with Preference Data." arXiv:2406.18665.
2. **LMSYS Arena**: Zheng, L., et al. (2023). "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena." NeurIPS 2023.
3. **HuggingFace Dataset**: https://huggingface.co/datasets/routellm/gpt4_judge_battles

## 🚀 Next Steps

After creating Table 1, you should:

1. **Verify category accuracy**: Manually inspect samples from each category
2. **Update paper text**: Reference this table when describing evaluation setup
3. **Cross-reference**: Ensure consistency with experimental results tables
4. **Reviewer response**: Use this table to address data provenance questions

## 📊 Statistics Summary

```
Total Prompts:        81,871
├─ Warmup:            80,000 (97.7%)
├─ Dev:                1,121 (1.4%)
└─ Holdout:              750 (0.9%)

Semantic Breakdown:
├─ Coding:            31,964 (39.0%)
├─ Conversational:    30,731 (37.5%)
├─ Creative:           8,184 (10.0%)
├─ Knowledge:          7,801 (9.5%)
└─ Math/Logic:         3,188 (3.9%)

Average Prompt Length: ~450 characters
Median Prompt Length:  ~147 characters
```

## 🔗 Related Files

- **PCA Training**: `scripts/train_pca_from_routellm.py`
- **Warmup Generation**: `scripts/generate_warmup_priors.py`
- **Reward Generation**: `scripts/generate_gpt4_turbo_rewards.py`
- **Stratification Logic**: `src/bandit_gpt/utils/experiment.py`
- **Data Correction**: `DATA_CORRECTION_SUMMARY.md` (project root)

---

**Created**: 2026-01-24  
**Author**: BanditGPT Team  
**Status**: ✅ Ready for KDD submission

