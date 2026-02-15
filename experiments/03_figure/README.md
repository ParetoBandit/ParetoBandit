# Figure 3: Corralling as Insurance — Prior Quality Degradation Sweep

**Figure Type:** 2-panel empirical analysis (Strategy Crossover + Adaptive Weights)  
**Last Updated:** February 14, 2026

---

## Overview

This directory answers one question for the banditGPT library:

> **When should a practitioner use each of the three routing strategies (warmup-only, corralling, tabula rasa)?**

The core experiment is a **prior quality degradation sweep** that interpolates priors from correct (α=0) through uninformative (α=0.5) to adversarial (α=1.0), testing all three strategies at each level. The result is a 2-panel figure showing the strategy crossover point and the meta-learner's adaptive weight response.

**Key Finding:** Corralling is an *insurance mechanism* — it trades peak performance for bounded worst-case regret. It never Pareto-dominates the baselines, but its regret range (40–67) is 62% narrower than warmup-only's (27–99), making it the right default when prior quality is unknown.

---

## Running

```bash
# Core experiment (produces the main 2-panel figure)
python run_all_experiments.py

# With more seeds for tighter confidence intervals
python run_all_experiments.py --seeds 50

# Skip figure generation (data only)
python run_all_experiments.py --no-plots
```

Additional experiments (2a, 2bc, 3, 5, iw) are available via `--experiments` for development use but are not required for the paper.

---

## Output

- `results/figure3_prior_degradation.pdf` — The main figure for the paper
- `results/prior_degradation/prior_degradation_statistics.json` — Full per-seed results

---

## Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Learning rate η | √(ln 2 / 750) ≈ 0.030 | Theoretically optimal for K=2, T=750 |
| Mixing floor γ | 0.05 | Prevents expert death; negligible impact on regret |
| Prior scaling | 0.05 | Reduces prior confidence to 5% of offline strength |
| Initial warmup weight w₀ | 0.7 | Moderate prior trust (user-configurable) |
| Exploration α | 2.0 (constant) | Preserves change-detection in non-stationary settings |

---

## LaTeX Integration

```latex
% Main figure (2-panel: crossover + adaptive weights)
\input{experiments/03_figure/latex_figure3_combined_caption}

% Results section: "When Does Adaptive Safety Pay Off?"
\input{experiments/03_figure/latex_section_results_meta_learning_cost}

% Strategy selection table
\input{experiments/03_figure/latex_table_strategy_guide}

% Practical deployment recommendations
\input{experiments/03_figure/latex_section_5.3_practical_recommendations}
```

---

## Implementation

**Router classes:** `src/bandit_gpt/router.py`
- `CorrallingRouter` — Meta-learner with Exp4 importance-weighted updates
- `CostAwareLinUCBRouter` — Warmup expert with prior initialization
- `CostAwareTabulaRasaRouter` — Tabula rasa expert (uninformative priors)

**Critical:** The `selection_token` returned by `select_model()` must be passed to `update()`. Omitting it disables meta-learning entirely (weights freeze).
