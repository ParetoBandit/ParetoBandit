# Figure 6: Multi-Model Pareto Frontier (K=5, K=10)

**Experiment Goal**: Demonstrate banditGPT's multi-model routing capabilities at K>>2, exercising the full architectural stack (Dynamic Pareto Filtering, Hybrid LinUCB family sharing, Corralling meta-learning) with real warmup priors for all models.

---

## Overview

This experiment produces the primary K>>2 evaluation — a cost-quality Pareto frontier at K=5 and K=10 — using the production router with 43-model warmup priors generated from a held-out prior-training set (see `scripts/generate_multimodel_warmup_priors.py`).

### Portfolios

**K=5** (5 providers, quality range 0.745–0.983):
| Model | Provider | Holdout Reward | Approx. Cost/req |
|-------|----------|---------------|-----------------|
| Llama-3.1-8B | Meta | 0.745 | $0.00003 |
| Mixtral-8x7B | Mistral | 0.823 | $0.00029 |
| Gemini-2.5-Flash | Google | 0.953 | $0.00026 |
| Claude-Sonnet-4 | Anthropic | 0.975 | $0.00630 |
| GPT-4.1 | OpenAI | 0.983 | $0.00520 |

**K=10**: K=5 + Llama-4-Maverick, Gemma-3-27B, Claude-Haiku-4.5, GPT-4-Turbo, DeepSeek-V3

### Protocol
- **Prior training**: 355 prompts × 43 models → warmup priors (generated separately)
- **Online learning**: 533 prompts (three-way split, disjoint from prior-training and holdout)
- **Evaluation**: 750 holdout prompts (same prompts as K=2 experiments)
- **Trials**: 20 seeds × 11 λ values per condition
- **Baselines**: Oracle, best-static, random, ε-greedy (ε=0.1), tabula rasa ablation

## Reproduction

```bash
# Step 1: Generate warmup priors (if not already done)
python scripts/generate_multimodel_warmup_priors.py --pca src/artifacts/pca_32.joblib

# Step 2: Run the experiment
python experiments/06_figure/run_multimodel_pareto.py
# Output: experiments/06_figure/results/multimodel_pareto_results.json
```

---

**Last Updated**: February 22, 2026
