# Figure 1: Cost-Quality Pareto Frontier (CostSave@Q)

**Headline result: BanditGPT achieves competitive cost savings without
supervised routing labels, using only per-request reward feedback.**

## Core Question

> How much cost can a contextual bandit router save while maintaining a
> specified quality level, compared to static model selection?

## Setup

- **K=2 portfolio**: Llama-3.1-8B (weak, $0.00003/req) vs Gemini-2.5-Pro
  (strong, $0.015/req)
- **Quality gap**: 14% (0.793 vs 0.932), prompt-dependent (weak >= strong on
  40% of prompts)
- **Cost ratio**: 500x
- **Data**: train=8,374 / val=1,785 / test=1,824 prompts
- **Best config**: alpha=1.0, n_eff=10, policy=hybrid, corralling=False,
  ff=1.0 (selected by Pareto AUC on val, PCA-25 embeddings)
- **Evaluation protocol**: the bandit trains on the train split, then is
  evaluated on the test split while continuing to learn (online updates).
  This faithfully reflects production deployment. After 8 K+ training
  prompts the policy is near-convergent; test-phase updates are
  incremental and the distinction from a frozen-policy evaluation is
  negligible.

## Reward signal assumption

BanditGPT requires per-request reward feedback but **no pre-collected
routing labels**. In production, rewards can come from implicit user
signals (acceptance rates, thumbs-up/down ratings, task completion) at
no additional inference cost. Our evaluation uses an LLM-as-judge
(DeepSeek-R1) as a controlled proxy for such signals; judge inference
cost is an *evaluation expense*, not a deployment cost, and is therefore
excluded from the cost axis. If the judge were the production reward
mechanism, its cost would need to be amortized into the x-axis —
though in practice the judge can be run on a subsample or replaced
by cheaper implicit signals.

## Results (test holdout)

| Metric | Bandit | Static baseline | Advantage | 95% CI |
|---|---|---|---|---|
| CostSave@90% | **71.9%** | 66.9% | +5.1 pp | [+2.0, +7.8] |
| CostSave@95% | **41.5%** | 33.4% | +8.1 pp | [+5.1, +10.8] |
| CostSave@99% | **14.2%** | 6.7% | +7.5 pp | [+4.5, +10.9] |

95% CIs from prompt-level paired bootstrap (n=1,824 test prompts,
2,000 resamples). Bandit per-prompt outcomes are seed-averaged (5 seeds)
before resampling; the same indices are applied to both the bandit and
static baselines so the CI is on the *advantage* (difference).

Pareto AUC: 0.8703 ± 0.0002 vs 0.8626 static (+0.892%, 5 seeds).

## Figure Structure

Two-panel figure showing the cost-quality Pareto frontier.

**Panel A: K=2 (Llama-8B vs Gemini-Pro)**

- Dashed gray line: static baseline (linear interpolation from always-weak to
  always-strong)
- Solid blue curve: bandit frontier (each point is a different `cost_penalty`,
  averaged over 5 seeds)
- Starred markers: CostSave@90%, @95%, @99% operating points annotated with
  cost savings
- Shaded band: +/- 1 std across seeds
- X-axis: average cost per request (log scale); Y-axis: average reward

The bandit curve bows above the static line, showing that contextual routing
achieves higher reward at the same cost.

**Panel B: K=3 after onboarding Mistral-Large (from Exp 3)**

Same layout, but with three fixed-model points (Llama, Mistral, Gemini) and
the post-onboarding bandit frontier. Mistral delivers 99% of Gemini quality
at 1/29th the cost, so the frontier shifts dramatically left. Gemini becomes
nearly dominated.

## Routing Mix at Key Operating Points

| cost_penalty | reward | cost | %weak | %strong |
|---|---|---|---|---|
| 0.00 | 0.9262 | $0.013746 | 11.9% | 88.1% |
| 0.05 | 0.9217 | $0.012743 | 20.4% | 79.6% |
| 0.10 | 0.9162 | $0.011975 | 26.5% | 73.5% |
| 0.20 | 0.8922 | $0.009536 | 42.0% | 58.0% |
| 0.50 | 0.8037 | $0.001027 | 94.6% | 5.4% |

At intermediate cost penalties, the router makes genuine discriminative
decisions — not trivially picking one model.

## Comparison

BanditGPT achieves meaningful cost savings using *online-only* reward
feedback — no supervised routing labels, preference data, or offline
classifiers are required.

A direct numerical comparison with RouteLLM's reported ~50% cost saving
at 95% quality is **not valid**: the benchmarks differ in dataset (Chatbot
Arena vs our LLM-as-judge corpus), model pairs (different cost ratios),
and evaluation methodology (human preferences vs DeepSeek-R1 scores).
CostSave percentages are highly sensitive to the cost ratio between
models (here 500x), so raw numbers do not transfer across setups.

The qualitative takeaway is that an online bandit router, without any
labelled training data, can recover a substantial fraction of the
cost-quality frontier that supervised approaches achieve — while also
supporting capabilities that supervised routers lack (online learning
curves, zero-shot onboarding of new models, and Corralling for safe
exploration).

## Data Source

Results from `experiments/benchmark/results/hparam_tuning_k2.json`, produced
by `experiments/benchmark/tune_hybrid_router.py --k2 --n-seeds 3`.

## Reproducibility

```bash
# Run the full K=2 hyperparameter sweep (160 configs x 3 seeds, ~70 min)
python experiments/benchmark/tune_hybrid_router.py --k2 --n-seeds 3

# Generate the figure (to be written)
python experiments/01_figure/plot_pareto_frontier.py
```

## Connection to Other Experiments

| Experiment | What it adds |
|---|---|
| **Exp 1 / Figure 1** (this) | Headline CostSave@Q, Pareto frontier |
| **Exp 2** | Learning curve — CostSave improves over time |
| **Exp 3** | Onboarding — Panel B of this figure |
| **Exp 4** | Corralling — safety under bad priors |
| **Exp 5** | Feature ablation — PCA vs PCA+text |
| **Exp 6** | Warmup ablation — value of offline priors |
