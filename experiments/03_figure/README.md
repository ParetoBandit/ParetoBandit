# Figure 3: Pareto Frontier & Learning Curve

**Cost–quality trade-off analysis: banditGPT-Hybrid vs. RouteLLM-MF**

This directory contains the scripts and data for the primary competitive evaluation (Figure 3) of the banditGPT paper.

---

## Connection to Previous Experiments

- **Figure 1:** Established semantic structure and model preference heterogeneity
- **Figure 2:** Architecture diagram
- **Appendix E:** Validated Corralling prior degradation sweep

**Critical Question:** Figures 1–3 validated our technical approach, but **does this deliver practical value in production?**

This experiment answers that question via:
1. **Pareto frontier analysis** — Quantifies cost–quality trade-offs vs. baselines
2. **Two-regime comparison** — Identifies where pre-training vs. online learning wins
3. **Learning curve** — Measures how quickly online adaptation surpasses pre-training
4. **Production readiness** — Real data (N=1,871 total: 1,121 dev + 750 holdout)

---

## Directory Structure

```
03_figure/
├── generate_pareto_frontier.py   # Main Pareto experiment (Figure 3 data)
├── generate_figure3.py           # Two-panel figure (Pareto + Learning Curve)
├── run_learning_curves.py        # Learning curve analysis (panel B)
├── run_statistical_tests.py      # Statistical validation
├── check_calibration.py          # Calibration diagnostic (internal use only)
├── README.md                     # This file
└── results/                      # Output directory
```

---

## Quick Start

```bash
cd experiments/03_figure/

# Step 1: Run the Pareto frontier sweep (~2.5 hours)
python generate_pareto_frontier.py

# Step 2: Generate the two-panel figure (~2 min, requires Step 1 results)
python generate_figure3.py
```

---

## Key Findings

### Two Regimes, No Simple Winner

The Pareto frontier reveals two distinct cost regimes with different winners:

| Cost Regime | Approx. Range | Winner | Why |
|-------------|--------------|--------|-----|
| **Low budget** | $0.0003–$0.005 | RouteLLM | Pre-trained MF model has fine-grained prompt-level discrimination from 100k pairs; selects the right few prompts for GPT-4 |
| **High budget** | $0.005+ | banditGPT | Online learning discovers the correct aggregate model ordering; adapts to task-specific preferences that OOD pre-training misses |
| **Crossover** | ~$0.005/req | — | Below this, pre-trained routing is competitive; above it, adaptive routing dominates |

**Why RouteLLM wins at low budgets:** At tight budgets, both routers mostly use Mixtral and send only a few prompts to GPT-4. RouteLLM's 100k-pair MF model has learned which *specific prompt types* benefit from GPT-4 — this fine-grained discrimination beats banditGPT's coarser 1,121-sample online learner when picking a handful of prompts for the expensive model.

**Why banditGPT wins at high budgets:** As more prompts are routed, *aggregate model calibration* matters more than per-prompt selection. banditGPT discovers that Mixtral > GPT-4 on this distribution overall, while RouteLLM's OOD preferences over-provision GPT-4 — paying 43× more for 1.3% lower quality at maximum cost.

### RouteLLM Non-Monotonicity

A striking finding: RouteLLM's quality **declines** at high cost. Dense threshold coverage (26 points, including 16 in the [0.0, 0.15] range) shows:
- Threshold 0.00–0.07 (cost $0.011–$0.013): Reward 0.809–0.812, *worse than static Mixtral*
- Threshold 0.08–0.10: Quality recovers as Mixtral routing increases
- Threshold 0.12–0.14: Peak quality (0.883) at ~$0.005–0.007
- Threshold 0.15+: Quality declines as Mixtral usage increases too much

This non-monotonicity is a predictable consequence of model preference heterogeneity: GPT-4 is worse than Mixtral on ~14% of prompts (formatting-heavy, structured tasks). Static routers cannot distinguish these prompts and over-provision the expensive model.

### Online Adaptation Value (Learning Curve)

Panel (b) of Figure 3 shows banditGPT's holdout quality as a function of online learning steps:

| Step | Reward (±95% CI) | vs. RouteLLM Peak (0.883) |
|------|-------------------|---------------------------|
| 0 | 0.839 ± 0.004 | −4.4% (priors only) |
| 50 | 0.882 ± 0.015 | −0.1% (approaching) |
| 200 | 0.890 ± 0.006 | **+0.7% (surpasses)** |
| 400 | 0.902 ± 0.004 | +1.9% (reliably above) |
| 1,121 | 0.914 ± 0.003 | +3.1% (final) |

**~200 in-distribution prompts surpass 100k OOD pre-trained pairs.**

### What This Means for Practitioners

1. **Don't assume "expensive = better."** On this dataset, GPT-4-Turbo (43× costlier) is *worse* than Mixtral on average. Lowering RouteLLM's threshold to spend more money actually *degrades* quality past the optimal point. This is common whenever a smaller model is fine-tuned or specialized for your domain.

2. **The cost crossover is deployment-specific.** On our data, RouteLLM wins below ~$0.005/request (it selects the right few prompts for GPT-4 from 100k pre-trained pairs), and banditGPT wins above that (it learns the correct overall model ranking). In your deployment: if you already know which model is best, a static router suffices. If you don't — or if the answer might change — banditGPT discovers it within ~200 prompts.

3. **The burn-in cost is bounded.** During the first ~200 prompts, banditGPT routes using warmup priors (quality 0.839 — above static Mixtral, below RouteLLM's peak). At typical production throughput (>100 req/hour), this completes within 2 hours. After the crossover, the quality gain is permanent and self-maintaining — no retraining, no label collection, no manual threshold tuning.

### Quantitative Summary

| Method | Peak Quality (±95% CI) | Cost at Peak | Gap Closure |
|--------|----------------------|-------------|-------------|
| **banditGPT-Hybrid** | **0.914 ± 0.006** | $0.0099 | **70.0%** |
| RouteLLM-MF | 0.883 | $0.0069 | 46.2% |
| Oracle | 0.953 | $0.0020 | 100% |
| Static Mixtral | 0.823 | $0.0003 | 0% (baseline) |
| Static GPT-4 | 0.812 | $0.0130 | −8.5% (regression) |

Gap Closure = (R_router − R_Mixtral) / (R_Oracle − R_Mixtral) × 100%

---

## Experiment Details

### Dataset
- **Total**: 1,871 prompts (real LMSYS Chatbot Arena traffic)
- **Development Set**: 1,121 prompts (online learning)
- **Holdout Set**: 750 prompts (frozen evaluation)
- **Split**: Chronological (no data leakage)

### Model Pool
- **Mixtral 8x7B Instruct**: $0.000294/request (cheaper, better on average)
- **GPT-4-Turbo**: $0.013000/request (expensive, worse on average on this dataset)
- **Cost Ratio**: 44.2×

### banditGPT Configuration (Production Defaults)
- **Router**: `BanditRouter.create()` via `create_experiment_router()`
- **Architecture**: Corralling with 2 experts (Warmup + Tabula Rasa)
- **Policy**: Hybrid LinUCB (family-based parameter sharing when applicable)
- **Warmup Expert**: constant α = 2.0 (sustained exploration)
- **Tabula Rasa Expert**: decaying α = 1.0 → 0.01 (converging exploitation)
- **Corralling Learning Rate**: η = 0.1 (conservative, production default)
- **Prior**: 80k RouteLLM battles, trace-normalized
- **Cost Penalties**: 10 λ values (Pareto sweep), 15 λ values (dense Figure 3 sweep)
- **Trials**: 20 independent runs per λ (seeds 42–61)
- **API**: `router.route()` / `router.process_feedback()`

### RouteLLM Configuration
- **Router**: Matrix Factorization (MF, pre-trained on 100k LMSYS pairs)
- **Reference**: Ong et al. (2024), RouteLLM: Learning to Route LLMs with Preference Data
- **Thresholds**: 26 values, with dense coverage in [0.0, 0.15] (high-cost region)
- **Processing**: Sequential (rate-limit compliant)

### Evaluation Protocol
- Normalization computed from training set only
- Frozen evaluation on holdout (no updates during evaluation)
- Identical holdout set for all methods
- 95% confidence intervals via t₁₉ distribution

---

## Scripts Overview

### `generate_pareto_frontier.py`
Main experiment script — generates complete Pareto frontier with all baselines.

**Outputs:**
- `results/pareto_results.json` — Complete data (Oracle, static, RouteLLM, banditGPT)

**Runtime:** ~2.5 hours (26 RouteLLM thresholds + 10 banditGPT λ × 20 trials)

### `generate_figure3.py`
Two-panel figure generation — denser λ sweep + learning curve.

**Outputs:**
- `results/figure3.png` — Two-panel figure (Pareto + Learning Curve)
- `results/figure3_results.json` — Detailed results with statistics

**Runtime:** ~2 minutes (uses existing RouteLLM data from Pareto results)

### `check_calibration.py`
Development diagnostic — verifies router prediction calibration. Intentionally uses `CostAwareLinUCBRouter` directly (not `BanditRouter`) because it requires access to internal `A`/`b` matrices for diagnostic purposes.

---

## Reproducing Results

```bash
# Full reproduction
python generate_pareto_frontier.py  # ~2.5 hours
python generate_figure3.py          # ~2 minutes (after step 1)

# Quick verification (with existing pareto_results.json)
python generate_figure3.py          # Re-generates figure from cached data
```

---

## Related Experiments

| Scenario | Experiment |
|----------|-----------|
| Corralling prior degradation sweep | [Appendix E](../appendix/E_prior_degradation/) |
| Catastrophic model failure | [Figure 6](../appendix/E_catastrophic_failure_experiment/) |

---

**Last Updated**: February 2026
