# Quality Scoring Module

This folder contains all scripts and documentation for computing intent-specific quality scores for LLMs.

## Overview

We use **Bayesian Latent Factor (BLF) models** to compute composite quality scores that aggregate multiple benchmarks into a single latent "quality" dimension per intent. This approach handles missing data gracefully and provides uncertainty estimates.

## Quality Scores by Intent

| Intent | Score | Field | Method | Coverage |
|--------|-------|-------|--------|----------|
| **Coding** | CCS | `ccs_100` | BLF (4 benchmarks + 1 auxiliary) | 100% |
| **Reasoning** | CRS | `reasoning_score` | BLF (5 benchmarks) | 100% |
| **Factual QA** | CFS | `cfs_100` | BLF (3 benchmarks) | 100% |
| **Summarization** | CSS | `css_100` | BLF (3 benchmarks) | 62% |
| **Creative** | — | `arena_rank_creative` | Direct (LMArena ranking) | 56% |
| **General** | — | `general_quality` | Calibrated proxy (Theil-Sen) | 100% |

## Scripts

### Composite Score Computation

| Script | Description | Output Field |
|--------|-------------|--------------|
| `compute_coding_score.py` | Compute CCS from HumanEval, LiveCodeBench, SciCode, Arena Coding | `ccs_100` |
| `compute_reasoning_score.py` | Compute CRS from MATH-500, GPQA, HLE, AIME, Math Index | `reasoning_score` |
| `compute_factual_qa_score.py` | Compute CFS from MMLU-Pro, GPQA, Arena Expert | `cfs_100` |
| `compute_summarization_score.py` | Compute CSS from SummEdits, Hallucination Rate, Arena Longer | `css_100` |
| `compute_general_quality.py` | Calibrated proxy: Intelligence Index → Arena scale | `general_quality` |

### Data & Validation

| Script | Description |
|--------|-------------|
| `update_arena_rankings.py` | Fetch and update LMArena leaderboard rankings |
| `validate_blf_scores.py` | Validate connectivity, anchors, and sparsity for all BLF scores |

## Usage

### Compute All Scores

```bash
# Update Arena rankings first
python quality_scoring/scripts/update_arena_rankings.py

# Compute composite scores
python quality_scoring/scripts/compute_reasoning_score.py
python quality_scoring/scripts/compute_coding_score.py
python quality_scoring/scripts/compute_factual_qa_score.py
python quality_scoring/scripts/compute_summarization_score.py
python quality_scoring/scripts/compute_general_quality.py

# Validate all scores
python quality_scoring/scripts/validate_blf_scores.py
```

### Validate Scores

```bash
# Full validation report
python quality_scoring/scripts/validate_blf_scores.py

# Quick summary only
python quality_scoring/scripts/validate_blf_scores.py --quiet

# Validate specific score
python quality_scoring/scripts/validate_blf_scores.py --score CCS
```

## Documentation

| Document | Description |
|----------|-------------|
| [COMPOSITE_CODING_SCORE.md](docs/COMPOSITE_CODING_SCORE.md) | CCS methodology and benchmarks |
| [COMPOSITE_REASONING_SCORE.md](docs/COMPOSITE_REASONING_SCORE.md) | CRS methodology and benchmarks |
| [LATENT_FACTOR_MODULE.md](docs/LATENT_FACTOR_MODULE.md) | Bayesian Latent Factor model details |
| [QUALITY_SCORING.md](docs/QUALITY_SCORING.md) | Overall quality scoring approach |
| [TRUST_SCORE_METRICS.md](docs/TRUST_SCORE_METRICS.md) | Trust and reliability metrics |

## Key Methods

### Bayesian Latent Factor Model

We model each benchmark score as:

```
z_{i,b} ~ Normal(α_b + λ_b × θ_i, σ_b)
```

Where:
- `θ_i` is the latent quality factor for model i
- `α_b` is the benchmark-specific intercept
- `λ_b` is the benchmark loading (how much it reflects true quality)
- `σ_b` is the benchmark-specific noise

### Covariance-Based Imputation

For models missing primary benchmarks, we use auxiliary benchmarks with high correlation to "borrow strength":

> "We utilize a hierarchical prior with covariance between latent factors to borrow statistical strength for models with missing modalities."

For CCS, we include `intelligence_index` (r=0.96 with coding benchmarks) as an auxiliary benchmark.

### Calibrated Proxy Scoring (General Intent)

For the general intent, we use Theil-Sen robust regression to map Intelligence Index to the Arena scale:

```
Arena_Score = α × Intelligence_Index + β
```

- Training: 55 models with both Arena Rank and Intelligence Index
- R² = 0.48, p < 0.001
- 95% CI for slope: [0.82, 1.56] (excludes 0 → significant)

## Output

All scores are stored in `data/models_cache.json` with fields:
- `{score}_100`: 0-100 scaled score
- `{score}`: Raw z-score (for BLF scores)
- `{score}_sd`: Standard deviation / uncertainty
- `{score}_method`: Method used (bayesian/weighted_zscore)
