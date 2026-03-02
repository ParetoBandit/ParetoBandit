# Figure 4: K=10 Multi-Model Pareto Frontier

**BanditGPT routing across a 10-model portfolio with four cost tiers**

This directory contains the K=10 multi-model evaluation, separated from the K=2 BanditGPT vs RouteLLM comparison (see [`03_figure/`](../03_figure/)).

---

## Connection to Previous Experiments

- **Figure 1:** Established semantic structure and model preference heterogeneity
- **Figure 2:** Architecture diagram
- **Figure 3:** K=2 BanditGPT vs RouteLLM (see [`03_figure/`](../03_figure/))
- **Appendix E:** Validated Corralling prior degradation sweep

**Critical Question:** Does BanditGPT's architecture (Corralling + warmup priors + family sharing) scale to larger model portfolios where RouteLLM cannot operate?

---

## Evaluation Protocol

**Train-then-freeze** with ground-truth rewards:

1. BanditGPT trains on the dev-train split (80% of the online-learn pool, ~426 prompts)
2. Router is frozen after training (greedy exploitation, alpha=0)
3. Dev-val (20%, ~107 prompts) is used exclusively for hyperparameter selection
4. Holdout set (n=750) is reserved for final evaluation
5. Lambda swept over 33 values; all results averaged over 20 seeds

---

## Directory Structure

```
04_figure/
├── run_multimodel_pareto.py     # K=10 experiment (imports shared utils from 03_figure)
├── plot_results.py              # Generate Figure 4 from results JSON
├── figure4_caption.tex          # LaTeX caption for Figure 4
├── results_discussion.tex       # LaTeX results discussion (K=10 section)
├── results_discussion.md        # Markdown results discussion (K=10 section)
├── README.md                    # This file
└── results/                     # Output directory
    ├── multimodel_pareto_results.json  # Produced by run_multimodel_pareto.py
    └── figure4_k10.png                 # Produced by plot_results.py
```

---

## Quick Start

```bash
cd experiments/04_figure/

# Run the K=10 experiment
python run_multimodel_pareto.py

# Generate the figure (can also use legacy results from 03_figure)
python plot_results.py
```

---

## Experiment Details

### Dataset
- **K=10 Train**: online-learn pool from three-way split (`dev_rewards_complete_all_models.jsonl.gz`)
- **K=10 Holdout**: full holdout (`holdout_rewards_complete_all_models.jsonl.gz`)
- **Reward signal**: `extract_reward()` = mean(vote x confidence) across multi-judge panel
- **Dev train/val split**: 80/20 deterministic split (seed=7)

### K=10 Model Pool
| Tier | Models |
|------|--------|
| Cheap | Llama-3.1-8B, Mixtral-8x7B, Gemma-3-27B |
| Mid | Claude-Haiku-4.5, DeepSeek-V3, Gemini-2.5-Flash, Llama-4-Maverick |
| Expensive | Claude-Sonnet-4, GPT-4-Turbo, GPT-4.1 |

### BanditGPT Configuration
- **Architecture**: Corralling with 2 experts (Warmup + Tabula Rasa)
- **Policy**: Hybrid LinUCB (family-based parameter sharing)
- **Warmup Priors**: `priors_warmup_43model.joblib` (43-model file; only K=10 models loaded)
- **Alpha**: 0.5 (exploration coefficient)
- **Corralling**: eta=0.1, gamma=0.05
- **Prior n_effective**: 10.0
- **Trials**: 20 seeds per configuration

### Baselines
- **Oracle**: Per-prompt best model (reward upper bound)
- **Best static**: Always route to empirically best single model (GPT-4.1)
- **Best-static + noise**: Best model with prob 1-epsilon, random otherwise
- **UCB1**: Non-contextual online bandit (train-then-freeze)
- **Random**: Uniform random routing
- **Tabula rasa**: BanditGPT without priors or Corralling (plain LinUCB)

### Statistical Reporting
- **Primary test**: Paired bootstrap CI for dev-selected Pareto AUC difference (1,000 resamples)
- 95% confidence intervals via t-distribution across 20 seeds

---

## Shared Code

This experiment imports shared evaluation functions from [`03_figure/run_prequential.py`](../03_figure/run_prequential.py):
- Data loading (`load_rewards_from_file`, `build_model_registry`, `embed_dataset`)
- Baseline evaluation (`oracle_route`, `static_route`, `random_route`, `best_static_noisy_route`, `ucb1_online_route`)
- Bandit training/evaluation (`run_pareto_sweep`, `evaluate_frozen`, `train_bandit`)
- Pareto analysis (`pareto_auc`, `dev_selected_pareto_auc`, `bootstrap_pareto_auc_difference`)

---

## Related Experiments

| Scenario | Experiment |
|----------|-----------|
| K=2 BanditGPT vs RouteLLM | [Figure 3](../03_figure/) |
| Corralling prior degradation sweep | [Appendix E](../appendix/E_prior_degradation/) |

---

**Last Updated**: March 2026
