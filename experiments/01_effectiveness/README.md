# Experiment 01: Main Effectiveness

**Claim**: BanditGPT achieves lower cumulative regret than Random, ε-greedy, and vanilla LinUCB baselines.

## Scientific Question

Does the combination of structural features + complexity projection + procedural warmup provide a measurable advantage over standard contextual bandit approaches?

## Methodology

1. **Baselines**: Random, ε-greedy (ε=0.1), vanilla LinUCB (no features)
2. **Dataset**: N=1000 test prompts from LMSYS Arena
3. **Metrics**: Cumulative regret at T={100, 250, 500, 750, 1000}
4. **Evaluation**: 10 random seeds, report mean ± 95% CI

## Expected Results

- BanditGPT regret < LinUCB regret < ε-greedy < Random
- Statistical significance (p < 0.05, two-tailed t-test)

## Output

- `results/fig1_cumulative_regret.pdf` - Main regret comparison plot
- `results/effectiveness_results.json` - Numerical results
- `results/statistical_tests.txt` - Significance tests

## How to Run

```bash
# Run baseline comparison
python run_baselines.py

# Generate plot
python plot_regret.py
```

## Estimated Runtime

~2 hours (10 seeds × 1000 prompts × 4 methods)

## Status

🔴 Placeholder - needs implementation
