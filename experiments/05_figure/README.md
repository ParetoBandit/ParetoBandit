# Figure 5: K-Scaling — Hybrid vs. Disjoint LinUCB

**Sample efficiency of family parameter sharing across portfolio sizes**

This directory contains the K-scaling experiment comparing Hybrid LinUCB
(data-driven family assignments via tetrachoric correlation) against
Disjoint LinUCB (each model independent) at K=5 and K=10.

For the prior strength and exploration coefficient ablation study
(2D alpha x n_eff grid), see
[Appendix H](../appendix/H_alpha_neff_ablation/).

---

## Connection to Previous Experiments

- **Figure 3:** Validated Corralling meta-learner and prior degradation robustness
- **Figure 4:** Established cost–quality trade-off advantage over RouteLLM on 2 models

**Critical Question:** Figures 3–4 used only 2 models. Does the Hybrid
architecture provide value as the portfolio grows?

This experiment answers that question via:
1. **Controlled A/B comparison** — Same prompts, same seeds, only policy differs
2. **Scaling analysis** — K=5, K=10 models with constant dataset size
3. **Data-driven families** — Tetrachoric correlation within providers (threshold r_tet >= 0.6)
4. **Production router** — Full BanditRouter with Corralling (warmup + tabula rasa experts)

---

## Directory Structure

```
05_figure/
├── run_k_scaling_experiment.py        # Main K-scaling experiment
├── figure_5_caption.tex               # LaTeX figure float with caption
├── section_k_scaling_results.tex      # LaTeX results & discussion section
├── README.md                          # This file
└── results/
    ├── k_scaling_results.json         # K-scaling numerical results
    └── k_scaling_figure.png           # 1×2 panel figure (K=5, K=10)
```

---

## Quick Start

```bash
cd experiments/05_figure/

# Main K-scaling experiment (~2 min)
python run_k_scaling_experiment.py
```

---

## Configuration

Both experiments use the **full production `BanditRouter`** via
`create_experiment_router()`:

- **Router**: `BanditRouter.create()` with Corralling (warmup + tabula rasa experts)
- **Warmup priors**: `priors_warmup_43model.joblib` (loaded via `warmup_path`)
- **Alpha**: 0.5 (constant for warmup expert, 0.25→0.01 decaying for tabula rasa)
- **prior_n_effective**: 10.0 (main experiment), swept in ablation
- **Corralling**: learning_rate=0.1, gamma=0.05
- **Features**: 32 PCA dimensions + 1 bias = 33-dimensional context
- **Cost penalty**: 0.0 (quality-only evaluation)
- **Seeds**: 20 paired trials (42–61), global RNG seeded per trial
- **Data**: Three-way split (prior-train / online-learn / holdout) from shared utilities
- **Family map**: Computed on the **training** set only (no holdout leakage)

### Model Portfolios

Both portfolios include multi-member providers so shared families can form
at every K, controlling family density across the comparison:

| K  | Models |
|----|--------|
| 5  | gpt-4-turbo, gpt-4.1, llama-3.1-8b, llama-4-maverick, claude-sonnet-4 |
| 10 | K=5 + claude-haiku-4.5, gemma-3-27b, gemini-2.5-flash, mixtral-8x7b, deepseek-chat-v3 |

---

## Ablation Studies

The prior strength and exploration coefficient ablation studies have been
moved to [Appendix H](../appendix/H_alpha_neff_ablation/).  Key finding:
the global optimum is **(alpha=0.25, n_eff=1000, Disjoint)** at both K=5
(0.907) and K=10 (0.904), outperforming the default by ~3 pp.

---

## Related Experiments

| Scenario | Experiment |
|----------|-----------|
| 2-model Pareto frontier | [Figure 3](../03_figure/) |
| Distribution shift | [Figure 6](../06_figure/) |
| Prior degradation sweep | [Appendix E](../appendix/E_prior_degradation/) |
| Hyperparameter sensitivity | [Appendix C](../03_figure/) |

---

**Last Updated**: March 2026
