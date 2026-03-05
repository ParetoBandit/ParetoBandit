# Figure 5: Corralling Enables Recovery from Catastrophic LLM Failure

**Demonstrates that the Corralling meta-learner provides automatic failover when a production LLM degrades catastrophically.**

---

## Connection to Previous Figures

- **Figure 3:** BanditGPT vs supervised baselines (stationary environment)
- **Figure 4:** Value of warmup priors (cold start vs warm start)
- **Figure 5 (this):** Non-stationary robustness — what happens when a model fails?

**Key Question:** Can Corralling automatically detect and recover from catastrophic model failure without manual intervention?

---

## Scenario

Three-phase simulation on the K=10 portfolio (T=750 steps, 20 seeds):

| Phase | Steps | Description |
|-------|-------|-------------|
| Healthy | 0–249 | All models at normal quality (base rewards from real holdout) |
| Failure | 250–499 | GPT-4.1 (best static model) drops to R=0.10 |
| Recovery | 500–749 | GPT-4.1 returns to normal |

Context-dependent reward bonuses (orthogonal per model) ensure post-failure routing is a genuinely contextual decision.

---

## Methods Compared

| Method | Description |
|--------|-------------|
| **BanditGPT** | Corralling + warmup priors (full system) |
| **Warmup-only** | No Corralling, just warmup expert — shows priors alone can't adapt |
| **Tabula rasa** | No priors, no Corralling — learns from scratch |
| **Static** | Always route to GPT-4.1 — catastrophic on failure |
| **EMA tracker** | ε-greedy with exponential moving averages |
| **Oracle** | Per-step clairvoyant upper bound |

---

## Directory Structure

```
05_figure/
├── run_catastrophic_failure.py    # Main experiment
├── plot_results.py                # Generate Figure 5 (3-panel)
├── figure_5_caption.tex           # LaTeX figure caption
├── section_k_scaling_results.tex  # LaTeX results discussion
├── README.md                      # This file
└── results/
    ├── catastrophic_failure_results.json
    └── figure5_catastrophic_failure.png
```

---

## Quick Start

```bash
cd experiments/05_figure/

# Run the experiment (~30s)
python run_catastrophic_failure.py

# Generate the figure
python plot_results.py
```

---

## Configuration

- **Router**: Production `BanditRouter` via `create_experiment_router()`
- **Warmup priors**: K=10 portfolio-specific (`priors_warmup_k10_6comp.joblib`)
- **Hyperparameters**: From Appendix H (dev-val grid search)
- **Features**: 6 PCA dimensions + 1 bias = 7-dim context
- **Cost penalty**: 0.0 (quality-only)
- **Seeds**: 20 (offset 42–61)

---

## Related Experiments

| Scenario | Experiment |
|----------|-----------|
| Stationary evaluation | [Figure 3](../03_figure/) |
| Value of warmup priors | [Figure 4](../04_figure/) |
| Prior degradation sweep | [Appendix E](../appendix/E_prior_degradation/) |
| Detailed catastrophic failure (old) | [Appendix E](../appendix/E_catastrophic_failure_experiment/) |
| Hyperparameter tuning | [Appendix H](../appendix/H_alpha_neff_ablation/) |

---

**Last Updated**: March 2026
