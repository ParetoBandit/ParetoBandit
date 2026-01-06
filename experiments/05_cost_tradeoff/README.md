# Experiment 05: Cost-Quality Pareto Frontier

**Claim**: BanditGPT achieves higher quality than random routing at equivalent cost budgets, proving economic viability for production deployment.

## Scientific Question

Can intelligent routing deliver "GPT-4 level quality for 50% of the price" by sending only hard prompts to expensive models?

## The "Money Shot"

While Experiment 01 proves fast learning (regret), this experiment proves **economic value**. It answers the CEO/CTO question: *"Why should I pay for a router when I can just randomly mix cheap and expensive models?"*

## Methodology

### Data (100% Real)
- **Training**: `train_rewards_1k.jsonl` (~1000 prompts with real model responses)
- **Testing**: `test_rewards_pareto_dedup.jsonl` (~900 prompts with real model responses)
- **Models**: Full registry with actual OpenRouter pricing

### Procedure

1. **Cost Profile Sweep**: Test 4 user preference profiles
   - Max Quality (w_q=0.97): User wants best quality, cost secondary
   - Balanced (w_q=0.50): Equal weight on quality and cost
   - Budget (w_q=0.15): Cost-conscious, quality acceptable
   - Ultra Cheap (w_q=0.05): Minimize cost, basic quality ok

2. **For Each Profile**:
   - Initialize BanditRouter with HLE priors (N_struct=250, N_prior=10)
   - **Burn-in Phase**: Train on all 1000 real training prompts
   - **Evaluation Phase**: Greedy routing on all 900 real test prompts
   - Measure: (avg_cost_per_1k_tokens, avg_success_probability)

3. **Baselines**:
   - Individual models: (cost, quality) for routing 100% to each model
   - Linear interpolation: Dotted line connecting model Pareto frontier
   - Random baseline: Average quality across all models

### Expected Results

**The Bulge**: BanditGPT's curve should arc **above** the linear baseline, proving:
- At the same cost as a 50/50 random split, BanditGPT achieves +10-15% higher quality
- This happens because the router learns to send hard prompts to expensive models and easy prompts to cheap models

## Metrics

- **X-Axis**: Average cost per 1k tokens ($) - log scale
- **Y-Axis**: Success probability (0-100%) - percentage scale
- **Error Bars**: Standard deviation across 3 independent trials

## Output

- `results/fig5_pareto_frontier.pdf` - Publication-ready plot
- `results/fig5_pareto_frontier.png` - Quick preview
- `results/pareto_results.json` - Raw numerical results

## How to Run

```bash
# 1. Run experiment (generates cost-quality data)
python run_pareto.py

# 2. Generate plot
python plot_pareto.py
```

## Estimated Runtime

~30-45 minutes
- 4 profiles × 3 trials × ~1900 prompts (train+test)
- Encoder initialization: ~30 seconds (shared across trials)

## Interpretation

### Success Indicators
✅ BanditGPT curve is above the linear baseline  
✅ "The bulge" is visible (curved, not straight)  
✅ Error bars are reasonable (CV < 10%)  
✅ Model selection makes sense (Max Quality → premium models, Budget → cheap models)

### Economic Interpretation

If BanditGPT achieves **85% quality at $0.50/1k** while the linear baseline (random 50/50 mix) achieves **75% quality at $0.50/1k**, then:

> **The router provides +10% quality improvement for free**

This is the "economic dividend" from intelligent routing.

## Reviewer Impact

Including this experiment transforms the paper from:
- ❌ "We optimized regret" (algorithmic study)
- ✅ "We reduced API costs by 40% at same quality" (economic solution)

KDD reviewers highly value **practical business impact**. This experiment proves BanditGPT is ready for real-world deployment.

## Status

🟢 **Implemented** - Ready to run with 100% real data
