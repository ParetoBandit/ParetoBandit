# Figure 1: Alignment Tax Discovery & Scale Validation

This folder generates Figure 1 for the paper, which visualizes the forensic discovery of the Alignment Tax in LMSYS data. It includes both the holdout analysis and the full 1M dataset validation.

## Overview

This experiment demonstrates that task difficulty in LLM routing is not merely about reasoning complexity, but also about **model alignment failures**. We discover an "Alignment Tax" where flagship models (GPT-4-Turbo) actually perform worse than mid-tier models (Mixtral) on strict constraint tasks due to RLHF optimization.

### Two-Part Analysis

1. **Holdout Analysis** (N=1,871): Initial discovery on LMSYS dev/holdout data
2. **1M Scale Validation** (N=594,199): Confirms semantic structure holds at production scale

## Files

### Analysis Scripts

#### Holdout Analysis
- `plot_lmsys_holdout_pca.py` - Main script for Figure 1 (holdout data)
  - Projects 1,871 LMSYS prompts onto PCA space
  - Validates clusters against ground-truth reward gaps
  - Outputs visualization showing Alignment Tax discovery

- `check_cluster_stats.py` - Statistical validation of cluster quality

#### 1M Scale Validation
- `plot_lmsys_1M_pca.py` - Extends analysis to full 1M dataset
  - Validates that bimodal structure holds at scale
  - Demonstrates spectral invariance (PC1 variance stable)
  - Shows production distribution is even more skewed to easy tasks

- `download_1M_dataset.py` - Downloads full 1M dataset from HuggingFace
  - Requires `HUGGINGFACE_API_KEY` in `.env`
  - Processes battles into model evaluations format
  - Saves to `data/battles_1M.jsonl.gz`

### LaTeX Files
- `figure_1_caption.tex` - Figure caption for the paper
- `results_explanation.tex` - Detailed explanation of holdout findings
- `figure_1M_analysis.tex` - Scale validation analysis and interpretation

### Results
- `results/figure1_lmsys_holdout_pca.png` - Holdout analysis (300 DPI)
- `results/figure1_lmsys_holdout_pca_hires.png` - High-res version (600 DPI)
- `results/figure1_lmsys_1M_pca.png` - 1M validation (300 DPI)
- `results/figure1_lmsys_1M_pca_hires.png` - High-res version (600 DPI)

## Key Findings

### Holdout Analysis (N=1,871)

The figure demonstrates:
- **Low PC1 (82.4%)**: Natural Language Zone where GPT-4-Turbo wins (+0.133)
- **High PC1 (17.6%)**: Strictness Zone where Mixtral wins (-0.682)
- **85% of High PC1**: Dominated by strict completion templates
- **Alignment Tax**: RLHF optimization makes frontier models fail at strict constraints

### 1M Scale Validation (N=594,199)

**Spectral Invariance Confirmed:**
- PC1 variance: 3.10% → 3.101% (stable across 317× scale increase)
- Decision boundary: PC1 = 0.3 (unchanged)
- Semantic structure preserved at production scale

**Distribution Shift Discovered:**
- Low PC1 cluster grows: 82.4% → **94.1%** (production is easier)
- High PC1 cluster shrinks: 17.6% → **5.9%** (alignment tax is rare but valuable)

**Implications:**
1. **Holdout was a conservative stress test** - Production traffic is even more skewed toward routine tasks
2. **Cost savings are understated** - 94.1% vs 82.4% weak-model routing in production
3. **Alignment tax is rare but valuable** - 5.9% of prompts where Mixtral beats GPT-4

## Usage

### Step 1: Run Holdout Analysis

```bash
cd /Users/annette/repostitories/banditGPT
python3 experiments_v1/01_figure/plot_lmsys_holdout_pca.py
```

**Requires:**
- Pre-trained PCA model from RouteLLM data
- LMSYS dev/holdout data with reward evaluations
- Sentence transformer model for embeddings

### Step 2: Download 1M Dataset (Optional)

```bash
cd /Users/annette/repostitories/banditGPT
python3 experiments_v1/01_figure/download_1M_dataset.py
```

**This will:**
- Download the full RouteLLM dataset from HuggingFace
- Process battles into model evaluations format
- Save to `data/battles_1M.jsonl.gz`
- Requires `HUGGINGFACE_API_KEY` in `.env`

### Step 3: Run 1M Scale Validation (Optional)

```bash
cd /Users/annette/repostitories/banditGPT
python3 experiments_v1/01_figure/plot_lmsys_1M_pca.py
```

**This will:**
- Load the 1M dataset
- Compute reward gaps (GPT-4-Turbo - Mixtral)
- Embed prompts using sentence-transformers
- Project to 2D using pre-trained PCA model
- Generate visualizations showing bimodal structure

## Understanding the Alignment Tax

### What is it?

The **Alignment Tax** is a phenomenon where RLHF-optimized flagship models perform worse on strict constraint tasks because they're trained to be "helpful chat assistants" rather than raw text completion engines.

### Why does it happen?

1. **Conversational preambles**: GPT-4 adds "Sure, here is..." which violates strict formatting
2. **Safety over-correction**: Refuses tasks that look template-like
3. **Helpfulness alignment**: Tries to "improve" prompts instead of following them exactly

### Why does it matter?

- **Not about difficulty**: These tasks aren't "easy" - they're structurally misaligned
- **Economic opportunity**: Routing these to cheaper models improves both cost AND quality
- **Production-critical**: At scale (1M dataset), this is a 5.9% revenue opportunity

## Comparison: Holdout vs 1M Dataset

| Metric | Holdout (N=1,871) | 1M Dataset (N=594,199) |
|--------|-------------------|------------------------|
| PC1 Variance | 3.10% | 3.101% ✓ |
| PC2 Variance | 2.29% | 2.294% ✓ |
| Decision Boundary | PC1 = 0.3 | PC1 = 0.3 ✓ |
| Low PC1 (Natural) | 82.4% | **94.1%** ↑ |
| High PC1 (Strict) | 17.6% | **5.9%** ↓ |

**Interpretation:**
- ✓ = Spectral invariance confirmed (structure is stable)
- ↑/↓ = Production distribution shift (more routine tasks in real deployment)

## Conservative Stress Test Framing

The holdout analysis represents a **conservative stress test**:

1. **Regret reduction is understated by 14%** - Production has fewer hard prompts
2. **Cost savings are understated** - 94.1% vs 82.4% potential weak-model routing
3. **Warmup Prior's economic waste is worse** - Over-routes 94% of traffic, not 82%

This makes our paper results **more impressive** when generalized to production scale.

## Dependencies

- `sentence-transformers` - For prompt embeddings
- `datasets` - For HuggingFace dataset loading
- `scikit-learn` - For PCA (via joblib)
- `matplotlib` - For visualizations
- `scipy` - For KDE density estimation

## Connection to Paper

This experiment provides:

1. **Figure 1**: Visual proof of Alignment Tax discovery
2. **Section 3**: Motivation for adaptive routing over static heuristics
3. **Appendix**: Scale validation and conservative stress test analysis

## Notes

- PCA model is pre-trained on RouteLLM data (32 components, 35.14% variance)
- Holdout visualization uses all 1,871 prompts
- 1M visualization downsamples to 10k points for clarity (full analysis uses all data)
- Both analyses use the same PCA projection for consistency
