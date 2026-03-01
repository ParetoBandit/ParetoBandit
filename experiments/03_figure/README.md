# Figure 3 & 4: BanditGPT vs RouteLLM

**Cost-quality trade-off analysis with ground-truth multi-judge rewards**

This directory contains the primary competitive evaluation for the banditGPT paper: a fair comparison against RouteLLM using canonical dev/holdout datasets.

---

## Connection to Previous Experiments

- **Figure 1:** Established semantic structure and model preference heterogeneity
- **Figure 2:** Architecture diagram
- **Appendix E:** Validated Corralling prior degradation sweep

**Critical Question:** Does banditGPT's online adaptive routing deliver practical value compared to RouteLLM's pre-trained static routing?

> **Note:** This directory is named `03_figure/` for ordering, but produces **Figures 3 and 4** in the paper.

---

## Evaluation Protocol

**Train-then-freeze** with ground-truth rewards:

1. BanditGPT trains on dev set with oracle rewards from `extract_reward()` (mean of vote x confidence across multi-judge panel)
2. Router is frozen after training
3. Both BanditGPT and RouteLLM are evaluated on the same holdout set
4. RouteLLM threshold is tuned on a val subset of dev (85/15 split) using the same cost-penalised objective as BanditGPT

This eliminates the circular BT-proxy evaluation from earlier drafts: rewards are ground-truth multi-judge scores, not proxy predictions.

---

## Directory Structure

```
03_figure/
├── run_prequential.py               # Main experiment: K=2 (vs RouteLLM) + K=10 Pareto
├── generate_pareto_frontier.py       # Pareto frontier sweep (sparse lambda, 20 trials)
├── generate_figure4.py               # Dense lambda sweep + learning curve (50 trials)
├── run_learning_curves.py            # Learning curve analysis
├── run_statistical_tests.py          # Statistical validation
├── run_baseline_ablations.py         # Baseline ablation experiments
├── run_cold_start_ablation.py        # Cold-start ablation
├── run_hyperparameter_sensitivity.py # Hyperparameter sensitivity sweep
├── run_pca_neff_ablation.py          # PCA/n_eff calibration ablation
├── run_model_onboarding.py           # Model onboarding experiment
├── check_calibration.py              # Calibration diagnostic (internal use only)
├── README.md                         # This file
└── results/                          # Output directory
```

---

## Quick Start

```bash
cd experiments/03_figure/

# Run the main experiment (K=2 + K=10)
python run_prequential.py
```

---

## Experiment Details

### Dataset
- **K=2 Dev**: ~1,121 prompts (Mixtral + GPT-4-Turbo, from `dev_rewards_2models.jsonl.gz`)
- **K=2 Holdout**: ~750 prompts (same source, `holdout_rewards_2models.jsonl.gz`)
- **K=10 Train**: online-learn pool from three-way split (`dev_rewards_complete_all_models.jsonl.gz`)
- **K=10 Holdout**: full holdout (`holdout_rewards_complete_all_models.jsonl.gz`)
- **Reward signal**: `extract_reward()` = mean(vote x confidence) across multi-judge panel
- **Split**: Canonical stratified dev/holdout (no temporal leakage)

### K=2 Model Pool (BanditGPT vs RouteLLM)
- **Mixtral 8x7B Instruct**: cheap tier
- **GPT-4-Turbo**: expensive tier
- RouteLLM's MF router was trained on data including these models (in-distribution comparison)

### K=10 Model Pool (Pareto Frontier)
- Llama-3.1-8B, Mixtral-8x7B, Gemma-3-27B, Claude-Haiku-4.5, DeepSeek-V3
- Gemini-2.5-Flash, Llama-4-Maverick, Claude-Sonnet-4, GPT-4-Turbo, GPT-4.1

### banditGPT Configuration
- **Router**: `BanditRouter.create()` via `create_experiment_router()`
- **Architecture**: Corralling with 2 experts (Warmup + Tabula Rasa)
- **Policy**: Hybrid LinUCB (family-based parameter sharing)
- **Warmup Priors**: K=2 from `priors_warmup.joblib`, K=10 from `priors_warmup_43model.joblib`
- **Alpha**: 2.0 (exploration coefficient)
- **Corralling**: eta=0.1, gamma=0.05
- **Prior n_effective**: 10.0
- **Trials**: 5 seeds per configuration
- **API**: `router.route()` / `router.process_feedback()`

### RouteLLM Configuration (K=2 only)
- **Router**: Matrix Factorization (MF, pre-trained on 100k LMSYS pairs)
- **Reference**: Ong et al. (2024), RouteLLM: Learning to Route LLMs with Preference Data
- **Thresholds**: 14 values, tuned on val split using aligned cost-quality objective
- **In-distribution**: MF model trained on data including Mixtral and GPT-4-Turbo

### Statistical Reporting
- **Primary**: Across-seeds t-test (df = 4, honest but low power)
- **Secondary**: Paired bootstrap (over-powered, flagged as exploratory)
- 95% confidence intervals via t-distribution

---

## Reproducing Results

```bash
python run_prequential.py
```

Output: `results/prequential_results.json`

---

## Related Experiments

| Scenario | Experiment |
|----------|-----------|
| Multi-model scaling (K=5, K=10) | [04_figure](../04_figure/) |
| Corralling prior degradation sweep | [Appendix E](../appendix/E_prior_degradation/) |
| Catastrophic model failure | [Figure 6](../appendix/E_catastrophic_failure_experiment/) |

---

**Last Updated**: March 2026
