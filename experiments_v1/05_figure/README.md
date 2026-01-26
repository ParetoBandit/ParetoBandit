# Figure 5: Corralling Algorithm - Exponential Weight Evolution

## Overview

This experiment visualizes the **Corralling Algorithm** (Agarwal et al., 2017) as implemented in `BanditRouter`. The key insight is **adaptive decommissioning of misspecified priors**: when warmup priors have high confidence but wrong beliefs, the algorithm exponentially downweights them in favor of learning from scratch.

## The Algorithm: Exponential Weights for Expert Corralling

The Corralling algorithm maintains a distribution over multiple "expert" policies and adaptively shifts weight based on observed losses:

```
p_{i,t+1} = p_{i,t} · exp(-η · ℓ_{i,t}) / Z_t
```

Where:
- **p_{i,t}**: Probability of selecting expert i at step t
- **η**: Learning rate (controls adaptation speed)
- **ℓ_{i,t}**: Importance-weighted loss estimate for expert i
- **Z_t**: Normalization constant (sum of all weights)

### Two Experts in BanditGPT

1. **Warmup Expert**: Initialized with 80k RouteLLM battle priors
   - High confidence (large A matrices)
   - Potentially wrong (domain mismatch risk)

2. **Tabula Rasa Expert**: Learns from scratch
   - Low confidence initially (identity A matrices)
   - Adapts quickly to new data

## Key Innovation: Importance-Weighted Loss Estimation

The implementation uses an unbiased loss estimator to avoid "phantom penalties":

```python
# Only the chosen expert gets penalized
p_chosen = self.weights[self.last_expert_idx]
losses[self.last_expert_idx] = observed_loss / max(p_chosen, 1e-6)

# Non-chosen experts get 0 loss (no counterfactual)
# This ensures unbiased learning
```

This prevents artificial volatility where experts are penalized for decisions they didn't make.

## Expected Behavior: "Decisive Decommissioning"

When warmup priors are **misspecified** (e.g., "expensive models are always better"), you should observe:

1. **Phase 1 (t=0-50)**: Uniform exploration
   - Both experts selected ~50% of the time
   - Warmup starts confident but makes mistakes

2. **Phase 2 (t=50-200)**: Evidence accumulation
   - Tabula Rasa discovers cheap model (e.g., Mixtral) performs well
   - Warmup insists on expensive models
   - Loss gap widens

3. **Phase 3 (t=200+)**: Exponential decommissioning
   - Sharp drop in warmup weight (exp(-η·Δℓ) decay)
   - System stabilizes on Tabula Rasa policy
   - **Final weight**: Warmup <20%, Tabula Rasa >80%

## Usage

### Basic Execution

```bash
cd experiments_v1/05_figure
python plot_corralling_weights.py
```

### Requirements

- Real LMSYS data in `data/dev_prompts_for_rejudge.jsonl` and `data/dev_rewards_gpt4turbo_rejudged.jsonl`
- Production router from `src/bandit_gpt/router.py`
- Warmup priors (automatically loaded from `artifacts/priors_warmup.joblib`)

### Output

- `results/figure5_corralling_weights.pdf`: Publication-quality plot
- `results/figure5_corralling_weights.png`: Web-friendly version

## Mathematical Foundation

### Why Exponential Weights?

The exponential weight update provides several guarantees:

1. **Logarithmic Regret Bound**: 
   ```
   Regret ≤ (ln K) / η + η·T / 8
   ```
   Where K=2 experts, T=total steps

2. **Adaptive Mixing**: Automatically interpolates between experts without manual tuning

3. **Safety Against Negative Transfer**: Even if warmup is harmful, performance converges to the better expert

### Learning Rate Trade-off

- **η=0.1**: Slow adaptation, smooth curves, conservative
- **η=1.0**: Aggressive decommissioning, sharp transitions (used in experiments)
- **η=5.0**: Extremely aggressive, may overreact to noise

## Experimental Parameters

```python
models = [
    "openai/gpt-4-turbo",
    "anthropic/claude-3-opus-20240229", 
    "mistralai/mixtral-8x7b-instruct"
]

learning_rate = 1.0  # Aggressive decommissioning
n_samples = 500      # Routing decisions to simulate
cost_penalty = 0.0   # Quality-only (isolates prediction error)
```

### Design Choice: Quality-Only Mode (cost_penalty=0.0)

**Why Zero Cost Penalty?**

Setting `cost_penalty=0.0` for both experts is a deliberate design choice that isolates the **pure prior misalignment** phenomenon:

1. **What it isolates**: Decommissioning driven purely by wrong quality predictions
   - If warmup believes "GPT-4 > Mixtral" but true data shows "Mixtral > GPT-4"
   - Prior accumulates loss from prediction errors, not cost miscalibration
   - Cleanly demonstrates the "Prior Misalignment" safety mechanism

2. **What it eliminates**: Confounding cost-quality trade-offs
   - Without cost penalty, both experts optimize pure quality
   - Comparison is fair: same objective, different initialization
   - Result shows whether warmup prior's quality beliefs are correct

3. **When this happens naturally**: Quality inversion in the dataset
   - If your LMSYS data shows Mixtral outperforms GPT-4 on reward
   - But warmup prior was trained on data where expensive models dominate
   - Decommissioning happens automatically from quality mismatch alone

**Alternative Experiment**: Non-Zero Cost Penalty

To demonstrate cost sensitivity misalignment, try:

```python
# Scenario: Warmup has CORRECT quality beliefs but WRONG cost sensitivity
warmup_expert = CostAwareLinUCBRouter(..., cost_penalty=0.0)   # Cost-blind
tabula_rasa = CostAwareTabulaRasaRouter(..., cost_penalty=0.5) # Cost-aware
```

Expected result: If both predict quality equally well, but TR discovers cost savings, it will be preferred even with correct warmup quality predictions.

## Interpreting Results

### Success Metrics

✅ **Decisive Decommissioning**: Warmup weight <20% by t=500
- Indicates prior mismatch was correctly detected
- System adapted to true reward distribution

⚖️ **Balanced Mixing**: Both weights 30-70% by t=500
- Indicates both experts contribute useful information
- Warmup priors partially correct

✅ **Warmup Dominance**: Warmup weight >80% by t=500
- Indicates priors were well-calibrated
- Cold-start penalty avoided

### Debug Checklist

If you don't observe decommissioning:

1. **Check η**: Too low (<0.5) → adaptation too slow
2. **Check data quality**: Are rewards noisy or all similar?
3. **Check prior strength**: Are A matrices too large (overconfident)?
4. **Check model diversity**: Do models have different cost/quality trade-offs?

## Connection to Paper

This experiment generates **Figure 5** in the paper, demonstrating:

> "When warmup priors encode domain-specific beliefs (e.g., 'expensive=better'), 
> the Corralling algorithm provides robustness by adaptively downweighting 
> misspecified experts once sufficient evidence accumulates."

The visualization supports the claim that **hybrid bandits with Corralling provide worst-case guarantees** against negative transfer while retaining the benefits of warmup when priors are correct.

## References

1. Agarwal et al. (2017). "Corralling a Band of Bandit Algorithms." COLT 2017.
2. Agarwal & Zhang (2022). "Corralling a Larger Band of Bandits." UAI 2022.
3. Your codebase: `src/bandit_gpt/router.py` (CorrallingRouter class, line 3376+)

## Files

- `plot_corralling_weights.py`: Main experiment script
- `README.md`: This documentation
- `results/`: Output directory for figures (created automatically)

## Advanced Usage

### Modify Learning Rate

```python
# Conservative (smooth decommissioning)
results = run_corralling_experiment(..., learning_rate=0.1)

# Aggressive (sharp decommissioning)  
results = run_corralling_experiment(..., learning_rate=5.0)
```

### Use Different Data Split

```python
# Use holdout set instead of dev
prompts, rewards = load_lmsys_data(split="holdout")
```

### Compare Multiple η Values

```python
for eta in [0.1, 0.5, 1.0, 2.0, 5.0]:
    results = run_corralling_experiment(..., learning_rate=eta)
    plot_weight_evolution(results, output_dir / f"eta_{eta}")
```

## Troubleshooting

### "Data not found" Error

Ensure you have:
```
data/dev_prompts_for_rejudge.jsonl
data/dev_rewards_gpt4turbo_rejudged.jsonl
```

If missing, the script will attempt to fall back to `holdout_*` files.

### "Model not in registry" Error

Check `src/bandit_gpt/config/models.json` contains the models specified in the script. Update the `models` list to match available models in your registry.

### "Dimension mismatch" Error

This occurs if PCA fallback changes dimensionality. Solution:
1. Ensure `artifacts/priors_warmup.joblib` matches current PCA configuration
2. Or delete saved state to trigger fresh initialization

## Next Steps

After generating Figure 5, you can:

1. **Ablation Study**: Test different η values to show adaptation speed trade-off
2. **Domain Transfer**: Test with intentionally wrong priors (e.g., swap model costs)
3. **Multi-Expert**: Extend to 3+ experts (e.g., add "pessimistic prior")
4. **Theory Validation**: Verify regret bound holds empirically

