# Figure 3 Integration Guide

## Overview

This document explains how Figure 3 (Optimal Gamma Analysis) integrates with:
1. Existing calibration tools (`scripts/calibration/`)
2. Other experimental figures (`01_figure/`, `02_figure/`)
3. The main calibration workflow (`experiments_v1/calibration/`)

## Relationship to Existing Work

### Scripts vs Experiments

The project has **two implementations** of gamma finding:

| Location | Purpose | Audience | Output |
|----------|---------|----------|--------|
| `scripts/calibration/find_gamma.py` | **CLI tool** for practitioners | Users calibrating routers | Console + simple plot |
| `experiments_v1/03_figure/find_optimal_gamma.py` | **Research artifact** for papers | Researchers + reviewers | Publication-quality figure + LaTeX |

### Key Differences

**CLI Tool** (`scripts/calibration/find_gamma.py`):
- Simpler, focused on practical use
- 4-panel figure
- Basic recommendations
- Faster execution (fewer gamma values by default)

**Research Tool** (`experiments_v1/03_figure/find_optimal_gamma.py`):
- Comprehensive analysis
- 8-panel publication-quality figure
- Multiple recommendation criteria
- LaTeX integration (caption + results section)
- Detailed README and usage docs
- Convergence rate analysis

### When to Use Which

**Use the CLI tool** (`scripts/calibration/`) when:
- You're calibrating a router for production
- You need a quick gamma recommendation
- You don't need publication-quality outputs

**Use the research tool** (`experiments_v1/03_figure/`) when:
- You're writing a paper
- You need comprehensive analysis
- You want to understand the tradeoffs deeply
- You need LaTeX-ready outputs

## Integration with Other Figures

### Figure 1: Semantic Task Specialization

**Location**: `experiments_v1/01_figure/`

**Relationship**: 
- Figure 1 shows **why** routing is hard (ambiguous frontier)
- Figure 3 shows **how** to calibrate for optimal adaptation

**Paper flow**:
```
Figure 1 → "Hard tasks cluster semantically"
         ↓
Figure 3 → "Here's how to calibrate for your domain"
         ↓
Figure 2 → "Convergence happens during calibration"
```

### Figure 2: Calibration Convergence

**Location**: `experiments_v1/02_figure/`

**Relationship**:
- Figure 2 shows **when** convergence happens (calibration vs holdout)
- Figure 3 shows **which gamma** enables optimal convergence

**Complementary insights**:
- Figure 2: "γ-scaling alone doesn't change policy, calibration data does"
- Figure 3: "But you need the right γ to enable calibration data to have influence"

### Calibration Experiments

**Location**: `experiments_v1/calibration/`

**Relationship**:
- Calibration experiments use the **recommended gamma** from Figure 3
- Figure 3 provides the **methodology** for choosing gamma
- Calibration results validate the **effectiveness** of the chosen gamma

## Workflow Integration

### Complete Research Workflow

```bash
# 1. Visualize the problem space
cd experiments_v1/01_figure
python plot_pca_reward_gap.py
# → Figure 1: Shows semantic structure

# 2. Find optimal gamma
cd ../03_figure
python find_optimal_gamma.py --output results/
# → Figure 3: Recommends γ = 0.010

# 3. Run calibration experiments
cd ../calibration
python calibrate_router.py --gamma 0.010
python evaluate_calibrated_router.py
# → Full calibration results

# 4. Analyze convergence timing
cd ../02_figure
python compare_calibration_convergence.py
# → Figure 2: Proves convergence during calibration
```

### Complete Production Workflow

```bash
# 1. Find optimal gamma (research tool)
cd experiments_v1/03_figure
python find_optimal_gamma.py --output results/
# Review results, note recommended gamma

# 2. Calibrate router (CLI tool)
cd ../../scripts/calibration
python calibrate_router.py \
  --calibration-data ../../src/bandit_gpt/data/offline_dataset/dev_rewards_complete.jsonl.gz \
  --gamma 0.010 \
  --output production_router.joblib

# 3. Deploy
# Use production_router.joblib in your application
```

## Data Flow

### Shared Resources

All scripts use the same underlying resources:

```
src/artifacts/
├── pca_23.joblib                    # Shared PCA model
└── priors_warmup.joblib             # Shared warmup priors

src/bandit_gpt/data/offline_dataset/
├── dev_rewards_complete.jsonl.gz    # Calibration data
└── holdout_rewards_complete.jsonl.gz # Evaluation data
```

### Data Format Compatibility

The research tool (`03_figure/`) accepts **both** formats:

1. **Aggregated format** (used by CLI tools):
```jsonl
{"prompt": "...", "rewards": {"model_a": 0.85, "model_b": 0.95}}
```

2. **Oracle format** (used by evaluation scripts):
```jsonl
{"prompt": "...", "model_id": "model_a", "raw_score": 0.85, "ok": true}
{"prompt": "...", "model_id": "model_b", "raw_score": 0.95, "ok": true}
```

The script auto-detects and converts as needed.

## Paper Integration

### LaTeX Structure

```latex
\documentclass{article}
\usepackage{graphicx}

\begin{document}

\section{Introduction}
% Motivation for routing

\section{Problem Formulation}
% Reference Figure 1 to show semantic structure
\input{experiments_v1/01_figure/figure_1_caption.tex}

\section{Methodology}
% Explain domain adaptation approach
\input{experiments_v1/03_figure/results/gamma_results.tex}
\input{experiments_v1/03_figure/results/figure_caption.tex}

\section{Experimental Results}
% Show calibration effectiveness
\input{experiments_v1/02_figure/results/figure_caption.tex}
\input{experiments_v1/calibration/RESULTS_SECTION.tex}

\section{Analysis}
% Deep dive into results

\section{Related Work}
% Compare to prior work

\section{Conclusion}
% Summary and future work

\end{document}
```

### Figure Ordering in Paper

**Recommended order**:

1. **Figure 1** (Semantic Specialization): Motivates the problem
2. **Figure 3** (Optimal Gamma): Explains methodology
3. **Figure 2** (Convergence Analysis): Validates approach
4. **Additional figures**: Performance comparisons, ablations, etc.

**Alternative order** (if methodology comes before results):

1. **Figure 1** (Semantic Specialization): Problem motivation
2. **Figure 2** (Convergence Analysis): Methodology validation
3. **Figure 3** (Optimal Gamma): Hyperparameter selection
4. **Additional figures**: Performance results

## Code Reuse

### Shared Functions

Both implementations use the same core functions from `src/bandit_gpt/calibration.py`:

```python
from bandit_gpt.calibration import (
    SimpleLinUCBRouter,      # Bandit router
    apply_gamma_scaling,     # Covariance inflation
    embed_prompt             # Prompt embedding
)
```

### Extending the Analysis

If you want to add new metrics or visualizations:

1. **For production use**: Modify `scripts/calibration/find_gamma.py`
2. **For research**: Modify `experiments_v1/03_figure/find_optimal_gamma.py`
3. **For both**: Add to `src/bandit_gpt/calibration.py` and import

Example:

```python
# Add to src/bandit_gpt/calibration.py
def compute_adaptation_speed(metrics: List[Dict]) -> float:
    """Compute how quickly the policy adapts."""
    # Your implementation
    pass

# Use in both scripts
from bandit_gpt.calibration import compute_adaptation_speed
```

## Configuration Management

### Default Paths

Both implementations use `src/bandit_gpt/config_legacy.py` for defaults:

```python
from bandit_gpt.config_legacy import (
    DEFAULT_SENTENCE_TRANSFORMER,    # "sentence-transformers/all-MiniLM-L6-v2"
    DEFAULT_WARMUP_PRIORS_PATH,      # "src/artifacts/priors_warmup.joblib"
    DEFAULT_PCA_PATH,                # "src/artifacts/pca_23.joblib"
    CANONICAL_DEV_DATA_PATH          # "src/.../dev_rewards_complete.jsonl.gz"
)
```

### Overriding Defaults

All scripts accept command-line arguments to override defaults:

```bash
python find_optimal_gamma.py \
  --warmup-priors /path/to/custom_priors.joblib \
  --pca /path/to/custom_pca.joblib \
  --calibration-data /path/to/custom_data.jsonl
```

## Testing

### Verifying Integration

Test that all components work together:

```bash
# 1. Test CLI tool
cd scripts/calibration
python find_gamma.py \
  --calibration-data ../../src/bandit_gpt/data/offline_dataset/dev_rewards_complete.jsonl.gz \
  --gamma-values 1.0 0.01 \
  --output test_cli/

# 2. Test research tool
cd ../../experiments_v1/03_figure
python find_optimal_gamma.py \
  --gamma-values 1.0 0.01 \
  --output test_research/

# 3. Verify outputs match (numerically)
python -c "
import json
cli = json.load(open('../../scripts/calibration/test_cli/gamma_results.json'))
research = json.load(open('test_research/gamma_results.json'))
print('CLI gamma:', cli['recommended_gamma'])
print('Research gamma:', research['recommended_gamma'])
assert abs(cli['recommended_gamma'] - research['recommended_gamma']) < 0.001
print('✓ Outputs match!')
"

# 4. Clean up
rm -rf ../../scripts/calibration/test_cli test_research
```

## Maintenance

### When to Update

Update both implementations when:

1. **Core algorithm changes**: Update `src/bandit_gpt/calibration.py` first, then both scripts
2. **New metrics**: Add to library, then optionally to scripts
3. **Bug fixes**: Fix in library if possible, otherwise in both scripts
4. **Default paths change**: Update `config_legacy.py`

### Version Compatibility

| Component | Version | Compatibility |
|-----------|---------|---------------|
| Python | 3.10+ | Required |
| sentence-transformers | 2.0+ | Required |
| numpy | 1.20+ | Required |
| matplotlib | 3.0+ | Required for plotting |
| joblib | 1.0+ | Required |

### Deprecation Plan

The CLI tool (`scripts/calibration/find_gamma.py`) is the **stable API** for users.

The research tool (`experiments_v1/03_figure/find_optimal_gamma.py`) is for **paper reproducibility** and may not be updated after publication.

## FAQ

### Q: Why have two implementations?

**A**: Separation of concerns:
- CLI tool: Stable, user-facing, practical
- Research tool: Comprehensive, publication-ready, may evolve

### Q: Which one should I modify for my research?

**A**: 
- For new experiments: Copy and modify `03_figure/find_optimal_gamma.py`
- For production features: Modify `scripts/calibration/find_gamma.py`
- For core functionality: Modify `src/bandit_gpt/calibration.py`

### Q: Can I use the research tool in production?

**A**: Yes, but the CLI tool is more appropriate:
- Faster (fewer gamma values by default)
- Simpler output
- More stable API

### Q: How do I cite this work?

**A**: Reference Figure 3 in your paper:

> We determine the optimal gamma using systematic evaluation across multiple values (Figure 3). Our analysis shows that γ=0.010 provides balanced influence between warmup priors and calibration data, achieving a Calibration/Prior ratio of 1.401.

## Support

For questions about integration:

1. **General calibration**: See `experiments_v1/calibration/README.md`
2. **CLI tools**: See `scripts/calibration/README.md`
3. **This figure**: See `USAGE.md` in this directory
4. **Code issues**: Check `src/bandit_gpt/calibration.py` docstrings

---

**Last Updated**: January 23, 2026  
**Maintainer**: BanditGPT Research Team  
**Related Docs**: `USAGE.md`, `README.md`, `experiments_v1/calibration/README.md`

