# Figure 7: LinTS Baseline Comparison

Compares banditGPT (LinUCB + Corralling) against Linear Thompson Sampling (LinTS) at K=5 and K=10.

## Protocol

- **Data split**: Three-way (355 prior training / 533 online / ~750 holdout)
- **Cost sweep**: λ ∈ {0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0}
- **Trials**: 20 per λ value
- **Baselines**: Oracle, Best Static, ε-greedy, Random

## Reproduction

```bash
python experiments/07_figure/run_lints_comparison.py
python experiments/07_figure/generate_figure8.py
```

## Output

- `results/lints_comparison_results.json`
- `results/figure8_lints_comparison.{pdf,png}`
