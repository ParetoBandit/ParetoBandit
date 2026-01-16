# Experiment 08: The Arbitrage Frontier

## Overview

This experiment demonstrates the **"Rational Luxury"** paradigm through a visualization of the router's decision boundary - the **Arbitrage Frontier**. This is the key visualization for proving to KDD reviewers that your router makes mathematically consistent, economically rational trade-offs between cost and quality.

## The Visualization

### What It Shows

**The Rational Indifference Curve** (Decision Boundary Plot):
- **Each dot** = One test prompt evaluated by the router
- **X-Axis (ΔQ)** = Quality Gain: How much better GPT-5.1 is predicted to be compared to GPT-OSS-120B
  - Values near 0: Simple prompts (both models perform similarly)
  - High values: Complex reasoning prompts (GPT-5.1 significantly outperforms)
- **Y-Axis (ΔC)** = Cost Premium: How much extra it costs to use GPT-5.1 ($/1M tokens)
- **The Dashed Line** = The Indifference Curve: The theoretical boundary defined by λ (Slope = 1/λ)

### Color Coding
- 🔴 **Red dots**: Prompts routed to GPT-5.1 (expensive, high-quality model)
- 🔵 **Blue dots**: Prompts routed to GPT-OSS-120B (cheap, efficient model)

### Model Comparison

This visualization uses a **binary model comparison** for mathematical purity:

| Model | Quality | Cost ($/1M) | Purpose |
|-------|---------|-------------|---------|
| **GPT-5.1** | 97.9% | $1.25 | Expensive, highest quality |
| **GPT-OSS-120B** | 94.7% | $0.02 | Cheap, efficient baseline |

Both models are from the Pareto frontier (see `PARETO_ANALYSIS.md` for full analysis of 42 models).

## Why KDD Reviewers Will Love This

1. **Theoretical Grounding**: Shows your router adheres to **Rational Choice Theory** from microeconomics
2. **Production Algorithm**: Uses **LinUCB** (contextual bandit with exploration), not a simplified greedy approach
3. **Interpretability**: Reveals the "Value of Information" - the slope tells you exactly how much quality improvement justifies a given cost increase
4. **Mathematical Purity**: Binary comparison ensures the 2D indifference curve is valid (no "third-body" interference)
5. **Falsifiability**: The decision boundary is predictable from the weights, not fit to data post-hoc

## Running the Experiment

```bash
cd experiments/08_arbitrage_frontier
python plot_rational_boundary.py
```

### Requirements
- Binary router configuration: `models_binary.json` (GPT-5.1 + GPT-OSS-120B)
- Binary warmup priors: `priors_warmup_binary.joblib`
- Pre-trained PCA artifact: `pca_23.joblib`
- Feature extraction pipeline (SentenceTransformer)

### Outputs
- `kdd_rational_boundary.png` - Standard resolution (300 DPI)
- `kdd_rational_boundary_hires.png` - Publication quality (600 DPI)

## Interpretation for the Paper

> **Figure 3** illustrates the decision boundary of the 'Auto' profile using the production LinUCB algorithm. The dashed line represents the router's economic indifference curve with slope = 1/λ = 50, where λ = 0.02 is the cost-quality trade-off parameter. Each dot represents a prompt, colored by the router's actual routing decision. Red dots (routed to GPT-5.1) cluster in regions where the quality gain justifies the 100× cost premium. Blue dots (routed to GPT-OSS-120B) represent prompts where the marginal quality gain is insufficient to justify the cost. Minor deviations from the theoretical boundary result from the exploration bonus (α × uncertainty) inherent in LinUCB, demonstrating the algorithm's adaptive learning behavior.

## Mathematical Foundation

The router's decision rule is based on **LinUCB** (Linear Upper Confidence Bound):

```
UCB Score = mean_quality + α × uncertainty - λ × cost

Route to Expensive Model if:
UCB_expensive > UCB_cheap

At the indifference curve (when scores are equal):
mean_quality_exp - λ × C_exp = mean_quality_chp - λ × C_chp

Simplifying:
ΔQ = λ × ΔC

Rearranging:
ΔC = (1/λ) × ΔQ
```

The slope of the indifference curve is therefore: **Slope = 1/λ**

For the "auto" profile (λ = 0.02), this means:
- Slope = 1/0.02 = **50**
- You're willing to pay up to **$50** for each unit of quality improvement (on 0-1 scale)
- Or equivalently: **$0.50 per percentage point** of accuracy gain

## Key Results

From the current experiment (35 test prompts):

| Metric | Value |
|--------|-------|
| **Total Prompts** | 35 |
| **Routed to GPT-5.1** | 6 (17.1%) |
| **Routed to GPT-OSS-120B** | 29 (82.9%) |
| **Mean Quality Gain (ΔQ)** | 0.10 (10 percentage points) |
| **Cost Premium (ΔC)** | $6.06 per 1M tokens |
| **Indifference Slope** | 50.0 |

**Key Insight**: The router is cost-conscious, routing 83% of prompts to the cheap model while reserving the expensive model for high-value tasks.

## Algorithm Details

### Production UCB (What This Plots)
- **Score**: `mean_quality + α×uncertainty - λ×cost`
- **Exploration**: Includes uncertainty bonus to encourage learning
- **Realistic**: Shows actual production behavior
- **Results**: 6/35 prompts to expensive model (17.1%)

### Why Binary Universe?
The visualization restricts the router to only 2 models to ensure:
1. **Mathematical validity**: The indifference curve is a true 2D linear boundary
2. **No mislabeling**: With 3+ models, "middle model" picks would be incorrectly labeled
3. **Clear interpretation**: Every decision is a pure binary trade-off

In production, the router has access to the full Pareto portfolio (5 models), but for this visualization we isolate the extreme endpoints.

## Files in This Directory

| File | Purpose |
|------|---------|
| `plot_rational_boundary.py` | Main script to generate the visualization |
| `kdd_rational_boundary.png` | Standard resolution output (300 DPI) |
| `kdd_rational_boundary_hires.png` | High resolution output (600 DPI) |
| `PARETO_ANALYSIS.md` | Full analysis of 42 models and Pareto frontier |
| `pareto_frontier.png` | Visualization of all 42 models on quality-cost space |
| `pareto_frontier_data.json` | Raw data for Pareto frontier analysis |
| `README.md` | This file |

## Technical Details

### Warmup Priors
The router is initialized with warmup priors trained on:
- **1,121 real dev prompts** (100% model coverage, 10× weight)
- **10,000 synthetic prompts** (hard prompts, domain-specific, traps)
- **22,242 total updates** (53% real, 47% synthetic)
- **Perfect benchmark coverage**: HLE, GPQA, LiveCode for both models

Command used:
```bash
python scripts/generate_warmup.py \
  --models src/bandit_gpt/config/models_binary.json \
  --samples 10000 \
  --output artifacts/priors_warmup_binary.joblib \
  --use-real-data \
  --real-data-weight 10.0
```

### Feature Extraction
- **Encoder**: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
- **PCA**: Reduces to 23 dimensions
- **Context**: Semantic embedding of each prompt

## Extensions

Potential follow-up analyses:
- Compare indifference curves across different profiles (speed, balanced, quality)
- Show how the frontier shifts as model prices change
- Overlay actual oracle performance data to validate quality predictions
- Demonstrate cost savings vs. baseline strategies (random, always-expensive, always-cheap)
- Extend to 3D visualization with three models (requires complex boundary representation)

## Citation

If you use this visualization approach, please cite the banditGPT framework:

```bibtex
@inproceedings{banditgpt2026,
  title={Rational Luxury: Context-Aware LLM Routing via Bayesian Bandits},
  author={...},
  booktitle={Proceedings of the 32nd ACM SIGKDD Conference},
  year={2026}
}
```

---

**Last Updated**: January 15, 2026  
**Status**: ✅ Ready for KDD submission
