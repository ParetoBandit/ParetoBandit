# Experiment 08: Regret Convergence (Cold Start Defense)

## Scientific Claim

**BanditGPT's priors and procedural warmup solve the Cold Start Problem that makes standard bandits unusable in production.**

## The Problem

Standard contextual bandits suffer from the "Cold Start Problem":
- **No Prior Knowledge**: With identity covariance matrices and zero weight vectors, the bandit must explore extensively before it can exploit.
- **Thrashing**: Early decisions are essentially random, leading to high initial regret.
- **Production Risk**: This thrashing period makes standard bandits unsuitable for production, where every request matters.

## Our Solution

BanditGPT uses **HLE (Human-Level Evaluation) Priors** to bootstrap the bandit with thousands of synthetic training observations:
- **Prior N=100**: Acts as if 100 observations per model have already been collected.
- **Calibrated Beliefs**: Priors are derived from real HLE benchmark data, not random guesses.
- **Immediate Competence**: The router makes intelligent decisions from request #1.

## Experimental Design

### Algorithms Compared

| Algorithm | Prior Strength | Expected Behavior |
|-----------|---------------|-------------------|
| **Cold Start LinUCB** | N=0 | Steep slope (thrashing) |
| **ε-Greedy (ε=0.1)** | N/A | Linear slope (constant 10% exploration) |
| **BanditGPT** | N=100 | Flat slope (starts competent) |

### Methodology

1. **Data**: Real training prompts from `train_rewards_hle_models.jsonl` (~976 prompts)
2. **Protocol**: Sequential online replay with oracle reward lookup
3. **Metric**: Cumulative Regret at each request `t`
4. **Trials**: 5 runs with different shuffles for variance estimation

## Running the Experiment

```bash
# Navigate to experiment directory
cd experiments/08_regret_convergence

# Run data collection (5 trials × 3 algorithms)
python run_convergence.py

# Generate visualization
python plot_convergence.py
```

## Expected Output

### `results/fig8_regret_convergence.pdf`

Line chart showing Cumulative Regret vs. Request Number (t):

```
Cumulative Regret
    │
    │    ╱ Cold Start LinUCB (steep)
    │   ╱ 
    │  ╱   ╱ ε-Greedy (linear)
    │ ╱   ╱
    │╱   ╱
    │   ╱    _________ BanditGPT (flat)
    │__╱____/
    └─────────────────────────────→ Request (t)
```

## Key Takeaways

1. **Cold Start LinUCB**: Exhibits steep initial regret due to random exploration.
2. **ε-Greedy**: Linear slope because it never stops exploring (10% random).
3. **BanditGPT**: Near-flat slope from the start—priors provide immediate competence.

**Bottom Line**: _"We solve the Cold Start problem that makes standard bandits unusable in production."_

## Files

| File | Description |
|------|-------------|
| `run_convergence.py` | Main experiment script |
| `plot_convergence.py` | Visualization generator |
| `results/convergence_results.json` | Raw experiment data |
| `results/fig8_regret_convergence.pdf` | Main figure (KDD quality) |
| `results/fig8_early_convergence.pdf` | Zoomed view of first 200 requests |

## References

- See `experiments/06_sensitivity_analysis/` for prior strength validation
- See `experiments/01_effectiveness/` for baseline methodology
