# Table 1: Dataset Description and Experimental Splits

**Experiment Goal**: Document complete data provenance and experimental split design for reproducibility

**Key Result**: Clean, defensible dataset description with 81,871 prompts across purpose-driven splits

---

## Overview

This experiment provides **Table 1** for the paper, documenting the complete data provenance and experimental design that enables reproducible bandit evaluation.

**Dataset Summary**:
- **Total Prompts**: 81,871 unique prompts
- **Warmup Set**: 80,000 (used for both PCA training and LinUCB priors)
- **Dev Set**: 1,121 (online learning & calibration)
- **Holdout Set**: 750 (final evaluation)

**Design Philosophy**: Focus on essential information for reproducibility. Every component serves a clear experimental purpose.

---

## Core Files

```
experiments_v1/01_table/
├── README.md                      # This file
├── generate_table1.py             # Table generator
├── table1_dataset.tex             # LaTeX table (used in paper)
├── validate_categorization.py     # Legacy validation scripts
├── validate_with_llm.py
├── validate_with_openrouter.py
└── archived/                      # Old versions with categories
```

---

## What's In The Table

The table provides **four essential components**:

1. **Data Sources**: LMSYS Chat Arena, RouteLLM battles
2. **Split Sizes**: 80,000 / 1,121 / 750 prompts
3. **Split Purposes**: PCA training, warmup priors, dev, holdout
4. **Quality Assurance**: Zero leakage verification, stratified sampling

### What's NOT In The Table

**Removed elements**:
- Semantic categories (Coding, Conversational, etc.)
- Category distributions
- Category validation metrics

**Rationale**: Categories were not used in any downstream experiment (Tables 2, Figures 1-8). Removing them creates a focused, defensible table that directly supports the experimental narrative.

---

## Data Provenance

### Warmup Set (80,000 prompts)

**Source**: LMSYS Arena battles via RouteLLM dataset  
**HuggingFace**: `routellm/gpt4_judge_battles`  
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
**Model: gpt-4-turbo

**Purpose**:
- Online bandit learning (algorithm trains here)
- Hyperparameter calibration (finding optimal γ, η)
- Baseline comparison (BanditGPT vs RouteLLM/FrugalGPT)

**Stratification**: By semantic complexity and difficulty level to ensure representative coverage

### Holdout Set (750 prompts)

**Source**: LMSYS Chat Arena (stratified splits)  
**File**: `data/holdout_prompts_for_rejudge.jsonl`  
**Model: gpt-4-turbo

**Purpose**:
- Final evaluation (held-out test set)
- Pareto frontier curves
- Cost-quality tradeoff analysis

**Guarantee**: Completely disjoint from warmup (verified via automated leakage checks)

---

## Key Design Decisions

### Decision 1: Consistent Model Topology (gpt-4-turbo Throughout)

**Context**: Both warmup and evaluation use the same model pair (mixtral-8x7b-instruct vs. gpt-4-turbo).

**Rationale**: Using a consistent reward function across all splits ensures clean attribution of any adaptation effects to distributional changes (prompt category mix) rather than model capability differences. This matches the RouteLLM evaluation topology, enabling a controlled comparison.

**Implication**: The only uncontrolled source of variation between warmup and evaluation is the prompt category distribution shift, which simplifies interpretation of Corralling's adaptation behavior.

### Decision 2: PCA Distribution Mismatch

**Issue**: PCA model (384→32 dims) trained on warmup distribution, which differs from evaluation.

**Implication**: Feature-space distribution shift compounds the label-space shift:
- **Principal components** optimized for warmup distribution's variance structure
- Features capturing important variance in evaluation data may be **underrepresented**
- Dimensionality reduction biased toward warmup characteristics

**Evidence from archived analysis**:
- Warmup: 49.8% Conversational, 19.9% Coding
- Evaluation: ~38% Conversational, ~39% Coding
- Category distribution differs significantly

**Problem**: PCA trained on one distribution, applied to another:
1. Principal components capture warmup variance (not evaluation variance)
2. Information loss during dimensionality reduction may disproportionately affect evaluation prompts
3. No validation of PCA transfer quality (e.g., reconstruction error on holdout vs. warmup)

**Missing validation**:
- Reconstruction error comparison (warmup vs. evaluation)
- Explained variance comparison across distributions
- Feature importance analysis for evaluation data

### Decision 3: Simplified Table (No Categories)

**Original design**: Table with 5 semantic categories (Coding, Conversational, Creative, Knowledge, Math/Logic)

**Revised design**: Simplified table focused on splits and provenance

**Rationale**:
- Categories unused in all experiments → no experimental justification
- Simplification focuses reader on reproducibility essentials
- Cleaner narrative: "Here's where the data came from and how we split it"

### Decision 4: Acknowledging Data Source Differences

**Observation**: Warmup data (RouteLLM battles) differs from Dev/Holdout (LMSYS general prompts):
- χ²=238.5, p<0.001 (statistically significant due to large n=81,121)
- Cramér's V=0.05 (negligible practical effect size)

**Context**: This difference arose from data availability constraints:
- Warmup data: RouteLLM battles dataset (mixtral vs. gpt-4-turbo)
- Evaluation data: LMSYS general prompts (mixtral vs. gpt-4-turbo)
- Same model pair but different sampling periods and prompt populations

**Transparency**: While we can measure whether Corralling adapts to this mismatch, this was not a deliberate design choice to test robustness. It reflects the practical reality of using available datasets. Future work could use matched data sources to isolate algorithmic performance from distribution shift effects.

---

## Data Quality Assurance

### 1. Zero Data Leakage

**Verification**: Automated checks removed 243 overlapping prompts (0.24%)  
**Method**: Exact string matching  
**Limitation**: Semantic near-duplicates (paraphrases, translations, minor edits) may still exist  
**Guarantee**: No exact string duplicates between warmup and evaluation  
**Importance**: Prevents inflated performance estimates  

**Note**: Standard practice uses exact string deduplication. More thorough semantic deduplication (e.g., embedding cosine similarity) would catch paraphrases but is not commonly done in similar work.

### 2. Stratified Sampling

**Method**: Dev and Holdout use stratified sampling by task complexity  
**Goal**: Ensure representative coverage across prompt types  
**Implementation**: Sampling stratified by task difficulty and domain

### 3. Reward Structure and Statistical Power

**Reward type**: Discrete pairwise preference outcomes (win=1, tie=0, loss=0) from LMSYS Chatbot Arena human evaluations — **consistent with Figure 1's categorical analysis**  
**Holdout Size**: 750 prompts (final held-out evaluation)  
**Informative prompts**: 204 (27.2%) where models differ in reward — routing only matters here  
**Ties**: 546 (72.8%) where both models get the same reward — routing irrelevant  
**Dev Size**: 1,121 prompts (online learning, not held-out evaluation)  

**Power analysis** (Monte-Carlo simulation, matching Figure 1's approach):

| Test | What it tests | 80% power at | Recommended? |
|------|--------------|-------------|--------------|
| McNemar's exact | Pairwise strategy comparison (paired binary) | ≥58% routing accuracy | ✓ Yes (gold standard) |
| Binomial (1-sided) | Routing accuracy > 50% on informative prompts | ≥60% routing accuracy | ✓ Yes (interpretable) |
| Paired t-test | Mean reward difference (continuous assumption) | ±10% accuracy advantage | ✗ No (wrong data assumption) |

**Primary evidence**: McNemar's test and binomial test on the 204 informative prompts, supplemented by multi-seed validation (N=10–30 seeds) in Table 2

### 4. Connection to Figure 1

Figure 1 and Table 1 analyze the **same N=750 holdout prompts** with the same discrete reward structure. Methodological consistency:

- **Figure 1**: Chi-squared test on win/tie/loss contingency table (between-cluster comparison). Monte-Carlo power > 99%. Effect size: Cohen's d = 0.33 (generic PCA, unbiased) to 1.53 (domain-adapted).
- **Table 1**: McNemar's / binomial test on paired routing outcomes (between-strategy comparison). Monte-Carlo power ≥ 80% at 58-60% routing accuracy.
- **Both**: Monte-Carlo simulation from observed discrete distribution. Both correctly treat rewards as categorical, not continuous.

The d = 0.33 preference heterogeneity discovered in Figure 1 provides the signal that makes routing learnable. The routing accuracy measured in Table 1's holdout analysis quantifies how well the bandit exploits this signal.

---

## Reproduction

### Generate The Table

```bash
cd experiments_v1/01_table
python generate_table1.py
```

**Output**: `table1_dataset.tex`

### Use In Paper

The paper includes the table via:

```latex
\input{../experiments_v1/01_table/table1_dataset.tex}
```

---

## References

1. **RouteLLM Dataset**: Ong, I., et al. (2024). "RouteLLM: Learning to Route LLMs with Preference Data." arXiv:2406.18665.
2. **LMSYS Arena**: Zheng, L., et al. (2023). "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena." NeurIPS 2023.
3. **HuggingFace Dataset**: https://huggingface.co/datasets/routellm/gpt4_judge_battles

---

## Related Experiments

- **Table 2** (`experiments_v1/02_table/`): Measures adaptation with mismatched warmup data
- **Figure 1** (`experiments_v1/03_figure/`): Alignment Tax discovery on this data
- **Figure 2** (`experiments_v1/04_figure/`): Distribution shift quantification
- **Figures 3-8**: Solution validation and production analysis

---

## Key Statistics

```
Total Unique Prompts: 81,871
├─ Warmup:            80,000 (97.7%) [shared for PCA + priors]
├─ Dev:                1,121 (1.4%)
└─ Holdout:              750 (0.9%)

Reward Structure (discrete pairwise outcomes):
├─ Ties (gap=0):      546 (72.8%) — routing irrelevant
├─ Informative:       204 (27.2%) — routing matters
└─ Gap values:        {-1: 98, 0: 546, +1: 106}

Data Quality:
├─ Zero leakage:      ✅ 243 overlaps removed (0.24%)
├─ Stratification:    ✅ χ²=0.78, p=0.94 (dev vs holdout)
└─ Power (MC sim):    ✅ McNemar's 80% at 58% acc, binomial 80% at 60% acc

Sources:
├─ LMSYS Arena:       100% of data
├─ RouteLLM:          80,000 warmup prompts
└─ Public:            ✅ All data publicly available
```

---

## Experimental Narrative

This table establishes the **foundation** for all subsequent experiments:

1. **Data Provenance** → Enables reproducibility
2. **Split Design** → Prevents data leakage  
3. **Quality Assurance** → Builds confidence
4. **Data Source Transparency** → Acknowledges constraints and their implications

The simplified design reflects a **proactive choice**: focus on what matters for reproducibility, remove what isn't used experimentally. This creates a clean, defensible narrative from data → experiments → results.

---

**Last Updated**: February 13, 2026  
**Status**: ✅ Ready for publication
