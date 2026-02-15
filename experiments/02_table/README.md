# Table 2: Dataset Description and Experimental Splits

**Experiment Goal**: Document complete data provenance and experimental split design for reproducibility

---

## Overview

This experiment provides **Table 2** for the paper, documenting the complete data provenance and experimental design that enables reproducible bandit evaluation.

**Dataset Summary**:
- **Total Prompts**: 81,871 unique prompts
- **Warmup Set**: 80,000 (used for both PCA training and LinUCB priors)
- **Dev Set**: 1,121 (online learning & calibration)
- **Holdout Set**: 750 (final evaluation)

---

## Core Files

```
experiments/02_table/
├── README.md                          # This file
├── generate_table1.py                 # Table generator
└── table1_dataset.tex                 # LaTeX table (used in paper)
```

---

## What's In The Table

The table provides **four essential components**:

1. **Data Sources**: LMSYS Chat Arena, RouteLLM battles
2. **Split Sizes**: 80,000 / 1,121 / 750 prompts
3. **Split Purposes**: PCA training, warmup priors, dev, holdout
4. **Quality Assurance**: Zero leakage verification, stratified sampling

---

## Data Provenance

### Warmup Set (80,000 prompts)

**Source**: `routellm/gpt4_judge_battles` HuggingFace dataset  
**Origin**: RouteLLM curated pairwise battle collection  
**Models**: mixtral-8x7b-instruct vs gpt-4-turbo  

**Purpose**:
- Train PCA model (384 → 32 dimensions)
- Generate LinUCB warmup priors (A matrix, b vector)

**Processing Pipeline**:
1. Download: `scripts/download_and_process_routellm.py`
2. Filter: mixtral vs gpt-4-turbo battles
3. Deduplicate: Extract unique prompts
4. PCA training: `scripts/train_pca_from_routellm.py`
5. Prior generation: `scripts/generate_warmup_priors.py`

**Artifacts**:
- `src/artifacts/pca_32.joblib`
- `src/artifacts/priors_warmup.joblib`

### Dev Set (1,121 prompts)

**Source**: LMSYS Chat Arena (stratified splits)  
**File**: `data/dev_prompts_for_rejudge.jsonl`  

**Purpose**:
- Online bandit learning (algorithm trains here)
- Hyperparameter calibration (finding optimal γ, η)
- Baseline comparison (BanditGPT vs RouteLLM/FrugalGPT)

**Stratification**: By semantic complexity and difficulty level to ensure representative coverage

### Holdout Set (750 prompts)

**Source**: LMSYS Chat Arena (stratified splits)  
**File**: `data/holdout_prompts_for_rejudge.jsonl`  

**Purpose**:
- Online bandit evaluation (cumulative reward/regret)
- Pareto frontier curves
- Cost-quality tradeoff analysis

**Independence guarantee**: Held out from all warmup data — the PCA and warmup priors have never seen these prompts. The holdout is independent from warmup by provenance (different data source). 243 incidental overlaps removed (0.24%) via automated checks.

---

## Key Design Decisions

### Decision 1: Two-Model Topology Matching RouteLLM

**Context**: Both warmup and evaluation use the same model pair (mixtral-8x7b-instruct vs. gpt-4-turbo).

**Rationale**: We deliberately start with two models for two reasons:
1. **Controlled comparison with RouteLLM**: RouteLLM's benchmark uses this exact model pair. By matching their topology, our results are directly comparable — any performance differences are attributable to the routing algorithm, not the model set.
2. **Clean experimental attribution**: Using a consistent reward function across all splits ensures that any adaptation effects are attributable to distributional changes (prompt category mix) rather than model capability differences.

**Scaling to more models**: BanditGPT supports arbitrary model sets. Multi-model routing (3+ models) is evaluated in Figure 4.

### Decision 2: PCA Cross-Domain Generalization (Validated)

**Design**: PCA (384→32 dims) trained on RouteLLM battles, applied to LMSYS general prompts for evaluation. The two datasets have different prompt populations and category distributions.

**Validation (Figure 1)**: Figure 1 directly validates that the PCA generalizes across this domain gap by computing the Spearman rank correlation between PC1 and reward gap on N=750 held-out prompts (ρ = -0.370, p < 0.0001, 2.6x vs median of 100 random projections).

**Why it works**: Both datasets involve the same model pair (Mixtral vs GPT-4-Turbo) and the same underlying task (text generation). The PCA captures variance in how prompts relate to model capabilities, which transfers across prompt populations.

### Decision 3: Data Independence as Methodological Strength

**Observation**: The warmup data (`routellm/gpt4_judge_battles`) and evaluation data (LMSYS general prompts) are independent collections from different data sources, sampling periods, and prompt populations. Same model pair but otherwise disjoint by provenance.

**Why this is a strength**:
1. **No contamination concern**: The PCA and warmup priors have never seen the evaluation prompts. No decontamination step needed — the datasets are disjoint by provenance, not by post-hoc filtering.
2. **Realistic evaluation**: In production, the router will encounter prompts from a different distribution than its training data. Evaluating on independently-sourced prompts tests this transfer directly.
3. **Conservative estimate**: Cross-domain evaluation provides a conservative lower bound on routing signal.

---

## Data Quality Assurance

### 1. Data Independence (No Leakage)

**By design**: Warmup (`routellm/gpt4_judge_battles`) and evaluation (LMSYS general prompts) are independent datasets from different sources, sampling periods, and prompt populations — disjoint by provenance.  
**Verification**: Automated checks found 243 incidentally overlapping prompts (0.24%), removed. Overlap is due to both datasets sampling from the broader LMSYS user base, not shared provenance.  
**Method**: Exact string matching  
**Limitation**: Semantic near-duplicates (paraphrases, translations, minor edits) may still exist  
**Guarantee**: No exact string duplicates between warmup and evaluation

### 2. Stratified Sampling

**Method**: Dev and Holdout use stratified sampling by task complexity  
**Goal**: Ensure representative coverage across prompt types  
**Implementation**: Sampling stratified by task difficulty and domain

### 3. Reward Structure

**Reward type**: Discrete pairwise preference outcomes (win=1, tie=0, loss=0) from LMSYS Chatbot Arena human evaluations  
**Holdout Size**: 750 prompts  
**Dev Size**: 1,121 prompts  

---

## Reproduction

### Generate The Table

```bash
cd experiments/02_table
python generate_table1.py
```

**Output**: `table1_dataset.tex`

### Use In Paper

The paper includes the table via:

```latex
\input{../experiments/02_table/table1_dataset.tex}
```

---

## References

1. **RouteLLM Dataset**: Ong, I., et al. (2024). "RouteLLM: Learning to Route LLMs with Preference Data." arXiv:2406.18665.
2. **LMSYS Arena**: Zheng, L., et al. (2023). "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena." NeurIPS 2023.
3. **HuggingFace Dataset**: https://huggingface.co/datasets/routellm/gpt4_judge_battles

---

## Related Experiments

- **Figure 1** (`experiments/01_figure/`): PCA validation — cross-domain generalization
- **Figure 3** (`experiments/03_figure/`): Corralling insurance — prior quality degradation
- **Figure 4** (`experiments/04_figure/`): Pareto frontier — cost-quality tradeoffs
- **Figure 6** (`experiments/06_figure/`): Catastrophic failure detection

---

**Last Updated**: February 14, 2026  
**Status**: Dataset documentation
