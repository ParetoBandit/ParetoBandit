# Figure 3: BanditGPT vs RouteLLM (K=2)

**Cost-quality trade-off analysis with ground-truth multi-judge rewards**

This directory contains the K=2 competitive evaluation for the banditGPT paper: a fair comparison against RouteLLM using canonical dev/holdout datasets.

> **Note:** The K=10 multi-model Pareto frontier (Figure 4) has been moved to [`04_figure/`](../04_figure/).

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
4. RouteLLM threshold is tuned on full dev set using the same cost-penalised objective as BanditGPT (symmetric data access)
5. Point comparisons use isocost matching (same deployment budget), not lambda matching (which is invalid across architectures)
6. **Fairness Design (Zero-Shot vs Adaptation):** RouteLLM's 100k pre-training pairs give it an absolute advantage in zero-shot routing (dev-selected holdout quality 0.783 vs BanditGPT cold-start 0.780). To isolate algorithmic advantage without confounding data volumes, we explicitly test *adaptation speed*, showing BanditGPT consistently surpasses RouteLLM's dev-selected quality after observing just ~25 dev prompts.

This eliminates the circular BT-proxy evaluation from earlier drafts: rewards are ground-truth multi-judge scores, not proxy predictions.

---

## Directory Structure

```
03_figure/
├── run_prequential.py               # Main experiment: K=2 (vs RouteLLM) + K=10 Pareto
├── run_alpha_ablation.py            # Alpha exploration sensitivity ablation
├── plot_results.py                  # Generate Figure 3 from results JSON
├── figure3_caption.tex              # LaTeX caption for Figure 3
├── figure_alpha_ablation_caption.tex # LaTeX caption for alpha ablation
├── results_discussion.tex           # LaTeX results discussion
├── results_discussion.md            # Markdown results discussion
├── README.md                        # This file
└── results/                         # Output directory
```

> Figure 4 (K=10 Pareto frontier) lives in [`04_figure/`](../04_figure/).

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
- RouteLLM's MF router was pre-trained on supervised preference data including these models (same model pair, temporal distribution shift)

### Cost Calibration (for fair router comparison)
- **External price table is shared across routers**: when comparing routing policies, each underlying model call must be charged the same cost regardless of router (otherwise the comparison is confounded).
- **Mixtral pricing**: we set Mixtral's input/output prices to **$0.24 / 1M tokens** (same for input and output), matching the assumption used in RouteLLM's cost analysis (Appendix D, *Cost Calculation*).
- **GPT-4 pricing**: we use **$10 / 1M input** and **$30 / 1M output**, also matching RouteLLM's appendix (for `gpt-4-1106` / GPT-4-Turbo pricing).

### K=10 Model Pool (Pareto Frontier)
- Llama-3.1-8B, Mixtral-8x7B, Gemma-3-27B, Claude-Haiku-4.5, DeepSeek-V3
- Gemini-2.5-Flash, Llama-4-Maverick, Claude-Sonnet-4, GPT-4-Turbo, GPT-4.1

### banditGPT Configuration
- **Router**: `BanditRouter.create()` via `create_experiment_router()`
- **Architecture**: Corralling with 2 experts (Warmup + Tabula Rasa)
- **Policy**: Disjoint LinUCB (independent per-arm ridge regression)
- **Warmup Priors**: K=2 from `priors_warmup.joblib`, K=10 from `priors_warmup_43model.joblib` (43-model file; only the K=10 models' priors are loaded)
- **Alpha**: 1.0 for K=2, 0.1 for K=10 (selected via Appendix H 3D grid ablation; loaded automatically from `best_hparams_k2.json` / `best_hparams_k10.json`)
- **Corralling**: eta=0.1, gamma=0.05
- **Prior n_effective**: 5000.0 (selected via Appendix H ablation)
- **Trials**: 20 seeds per configuration
- **Frozen evaluation**: alpha=0 (pure exploitation) during holdout eval
- **API**: `router.route()` / `router.process_feedback()`

### RouteLLM Configuration (K=2 only)
- **Router**: Matrix Factorization (MF, pre-trained on ~100k supervised preference pairs)
- **Reference**: Ong et al. (2024), RouteLLM: Learning to Route LLMs with Preference Data
- **Thresholds**: 101 values (dense sweep), tuned on full dev set using aligned cost-quality objective
- **Temporal shift**: MF model pre-trained on preference data including Mixtral and GPT-4-Turbo; evaluation data from same platform but different time period

### Baselines
- **Best-static + noise**: Routes to the empirical best model (computed from full training set means) with prob 1-epsilon, uniformly random otherwise. This is *not* an online epsilon-greedy bandit.
- **UCB1 (non-contextual)**: Standard multi-armed bandit (no prompt features). Trains on dev set with online UCB1, then freezes to greedy. Ablates the value of contextual features — if UCB1 matches BanditGPT, prompt embeddings add no value. Included for both K=2 and K=10.
- **Random**: Uniform random routing across all models.
- **Oracle**: Per-prompt best model (reward upper bound).

### Dev Train/Val Split
- The dev set is split into **train** (80%) and **val** (20%) with a fixed seed.
- BanditGPT trains on dev-train; RouteLLM tunes tau on dev-train. Dev metrics for Pareto frontier selection come from dev-val (symmetric, no train-set evaluation asymmetry).

### Statistical Reporting
- **Primary hypothesis test**: Paired bootstrap CI for the dev-selected Pareto AUC difference (1,000 resamples). The Pareto hull is built from (dev_val_cost, dev_val_reward) — no holdout data enters hyperparameter selection. Dev-optimal indices are fixed before bootstrapping; only holdout rewards are resampled (single-level bootstrap, conditioned on dev selection).
- **Post-hoc point comparisons**: Per-seed paired t-tests (df = n_holdout - 1) at four budget levels (10th, 25th, 50th, 75th percentile of shared cost range), restricted to dev-optimal hyperparameters, with Holm-Bonferroni correction. Median per-seed p-value is descriptive (not a formal rejection threshold); formal rejection uses the Holm-corrected ensemble p-value.
- **Interpolated isocost**: Reads reward off each method's Pareto hull at exact target costs via linear interpolation, eliminating sensitivity to sweep density differences (101 RouteLLM thresholds vs 24 BanditGPT lambda values). Per-prompt paired t-tests use the nearest on-hull sweep point.
- **Stability**: Across-seeds t-test (df = 19); measures algorithmic stability.
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
| K=10 multi-model Pareto frontier | [Figure 4](../04_figure/) |
| Corralling prior degradation sweep | [Appendix E](../appendix/E_prior_degradation/) |
| Catastrophic model failure | [Figure 6](../appendix/E_catastrophic_failure_experiment/) |

---

**Last Updated**: March 2026
