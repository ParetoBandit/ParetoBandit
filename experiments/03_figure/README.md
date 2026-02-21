# Figure 3: Prior Degradation Sweep

**Figure Type:** Single-panel empirical analysis (Regret vs Prior Corruption)
**Last Updated:** February 21, 2026

---

## Overview

This directory answers one question for the banditGPT library:

> **When should a practitioner use Corralling (the default) vs tabula rasa?**

The core experiment is a **prior quality degradation sweep** that interpolates priors from correct (α=0) through uninformative (α=0.5) to adversarial (α=1.0), testing three strategies at each level. The result reveals a sharp crossover at α≈0.55, separating two regimes.

**Key Finding:** Corralling dominates both alternatives for good-to-moderate priors (α ≤ 0.5), achieving 30–40% lower regret. Under adversarial priors (α ≥ 0.6), Corralling's variance explodes and tabula rasa becomes safest. The breakeven threshold is 57%: Corralling has better expected regret if the practitioner believes there is at least a 57% chance that priors are useful.

---

## Running

```bash
# Core experiment (produces the main figure)
python run_all_experiments.py

# With more seeds for tighter confidence intervals
python run_all_experiments.py --seeds 100

# Skip figure generation (data only)
python run_all_experiments.py --no-plots

# Run specific sub-experiments
python run_all_experiments.py --experiments 2a,2bc,3,prior,5,iw
```

---

## Output

- `results/figure3_prior_degradation.pdf` — The main figure for the paper
- `results/figure3_prior_degradation.png` — PNG version
- `results/prior_degradation/prior_degradation_statistics.json` — Full per-seed results

---

## Results Summary (N=100 seeds)

| Corruption α | Corralling | Warmup-Only | Tabula Rasa |
|:---:|---:|---:|---:|
| 0.0 | **37.3 ± 4.3** | 52.0 ± 3.4 | 52.0 ± 4.6 |
| 0.3 | **33.0 ± 4.2** | 57.2 ± 3.9 | 52.0 ± 4.6 |
| 0.5 | **38.4 ± 4.8** | 58.7 ± 3.8 | 52.0 ± 4.6 |
| 0.6 | 64.1 ± 19.3 | 59.4 ± 4.1 | **52.0 ± 4.6** |
| 1.0 | 75.3 ± 22.7 | 66.4 ± 3.9 | **52.0 ± 4.6** |

Warmup-only is never the best strategy at any corruption level.

---

## Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Learning rate η | √(ln 2 / 750) ≈ 0.030 | Theoretically optimal for K=2, T=750 |
| Mixing floor γ | 0.05 | Prevents expert death; negligible impact on regret |
| Prior scaling | 0.05 | Reduces prior confidence to 5% of offline strength |
| Initial warmup weight w₀ | 0.7 | Moderate prior trust (user-configurable) |
| Warmup α | 2.0 (constant) | Preserves change-detection; validated by sensitivity sweep |
| Tabula rasa α | 1.0 → 0.01 (decaying) | Converges to exploitation |
| Cost penalty | 0.0 | Quality-only experiment |

---

## Architecture

All banditGPT conditions use the production `BanditRouter` via `create_experiment_router()`:

- `BanditRouter.route()` for model selection
- `BanditRouter.process_feedback()` for reward updates
- Hybrid LinUCB policy (default) with family-based parameter sharing
- Corralling meta-learner with heterogeneous alpha schedules

Baseline strategies (warmup-only, tabula rasa) use the individual expert classes directly for isolation.

---

## LaTeX Integration

```latex
% The prior degradation figure and analysis are now integrated
% directly into the main paper (results.tex, Section 5.2.1).
% The standalone tex files below are supplementary references.

% Figure caption (single-panel)
\input{experiments/03_figure/latex_figure3_combined_caption}

% Strategy selection table
\input{experiments/03_figure/latex_table_strategy_guide}

% Practical deployment recommendations
\input{experiments/03_figure/latex_section_5.3_practical_recommendations}
```
