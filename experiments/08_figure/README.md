# Figure 8: Cumulative Regret Curves

Tracks per-step cumulative regret during online learning for banditGPT, LinTS, ε-greedy, and Random at K=5 and K=10.

## Protocol

- **λ = 0** (no cost penalty — isolates exploration/exploitation tradeoff)
- **Trials**: 20 per method
- **Regret**: r_t = oracle_reward_t − selected_reward_t

## Reproduction

```bash
python experiments/08_figure/run_cumulative_regret.py
python experiments/08_figure/generate_figure8.py
```

## Output

- `results/cumulative_regret_results.json`
- `results/figure9_cumulative_regret.{pdf,png}`
