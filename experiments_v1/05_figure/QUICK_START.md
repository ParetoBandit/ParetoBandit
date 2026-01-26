# Quick Start: Figure 5 - Corralling Weights

## TL;DR

```bash
cd experiments_v1/05_figure
python plot_corralling_weights.py
```

**Output**: `results/figure5_corralling_weights.pdf`

**Expected Runtime**: ~30 seconds (500 routing decisions)

## What This Shows

A **real-time visualization** of how the Corralling algorithm adaptively chooses between two expert policies:

1. **Warmup Expert** (Red line): Starts with 80k battle priors, high confidence
2. **Tabula Rasa Expert** (Green line): Learns from scratch, no priors

**Key Insight**: When priors are wrong, the algorithm "decommissions" the warmup expert by exponentially downweighting it.

## The Math (One Equation)

```
Weight_{t+1} = Weight_t · exp(-η · Loss_t)
```

- **η=1.0**: Learning rate (how fast to adapt)
- **Loss**: Observed mistakes by each expert
- **Result**: Bad experts lose weight exponentially fast

## What to Expect

### Good Prior Match

```
Initial:  Warmup=50%, Tabula=50%
t=100:    Warmup=60%, Tabula=40%  ← Prior validated
t=500:    Warmup=80%, Tabula=20%  ← Warmup dominates
```

**Interpretation**: Prior was correct, avoided cold-start penalty

### Bad Prior (Domain Mismatch)

```
Initial:  Warmup=50%, Tabula=50%
t=100:    Warmup=40%, Tabula=60%  ← Evidence accumulates
t=200:    Warmup=20%, Tabula=80%  ← Exponential drop
t=500:    Warmup=5%,  Tabula=95%  ← "Decisive decommissioning"
```

**Interpretation**: Prior was wrong, safely decommissioned

## Key Parameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `learning_rate` | 1.0 | 0.1-5.0 | Speed of decommissioning |
| `n_samples` | 500 | 100-1000 | How long to run experiment |
| `cost_penalty` | 0.0 | 0.0-1.0 | Cost sensitivity (0=quality-only) |
| `models` | 3 models | 2+ | Routing complexity |

### Important: Quality-Only Mode

By default, `cost_penalty=0.0` for both experts. This means:

✅ **What you're testing**: Whether warmup prior's **quality predictions** are correct
- Decommissioning = Prior has wrong beliefs about which model is better
- Example: Prior thinks "GPT-4 > Mixtral" but data shows "Mixtral > GPT-4"

❌ **What you're NOT testing**: Cost-quality trade-offs
- Both experts ignore cost, only optimize quality
- This isolates the "prior misalignment" variable cleanly

**Why this matters**: If you see decommissioning, it's because the warmup prior was trained on a different distribution (domain shift), not because it has wrong cost preferences.

## Troubleshooting

### "Data not found"

**Solution**: Ensure you have:
```
data/dev_prompts_for_rejudge.jsonl
data/dev_rewards_gpt4turbo_rejudged.jsonl
```

### "Model not in registry"

**Solution**: Edit `models` list in script to match your `config/models.json`

### "No decommissioning observed"

**Reason**: Both experts are equivalent (good thing!)

**Check**: Are losses diverging? Look at subplot 2 (cumulative loss)

## Understanding the Plot

### Subplot 1: Weight Evolution (Top)

- **Y-axis**: Expert probability (0-100%)
- **X-axis**: Routing steps (time)
- **Red line**: Warmup prior weight
- **Green line**: Tabula rasa weight
- **Dotted line**: 50% baseline (uniform)

**Key Feature**: Look for sharp drops (exponential decommissioning)

### Subplot 2: Cumulative Loss (Bottom)

- **Y-axis**: Total mistakes accumulated
- **X-axis**: Routing steps
- **Red line**: Warmup expert loss
- **Green line**: Tabula rasa loss

**Key Feature**: Diverging lines → different strategies → decommissioning

## Quick Experiments

### Test Different Learning Rates

```python
# In plot_corralling_weights.py, line ~400
learning_rate = 1.0  # Change this value

# Try:
learning_rate = 0.1  # Slow, smooth
learning_rate = 5.0  # Fast, aggressive
```

### Use Different Models

```python
# Line ~395
models = [
    "openai/gpt-4-turbo",
    "anthropic/claude-3-opus-20240229",
    "mistralai/mixtral-8x7b-instruct"
]

# Try your own:
models = ["your/model-1", "your/model-2"]
```

### Change Sample Size

```python
# Line ~397
n_samples = 500  # Change to 100, 200, 1000, etc.
```

## Reading the Terminal Output

```
Step 100/500 | Weights: Warmup=0.458, TR=0.542 | Selected: mixtral-8x7b
              └─ Expert probabilities at t=100     └─ Which model was chosen
```

**Look for**:
- Warmup weight dropping below 0.3 → Decommissioning in progress
- Warmup weight above 0.7 → Prior validated
- Both around 0.5 → Balanced contribution

## Next Steps

1. **Run the script**: `python plot_corralling_weights.py`
2. **Check the plot**: `results/figure5_corralling_weights.pdf`
3. **Read the summary**: Final weights tell the story
4. **Try variations**: Modify η, models, or sample size

## One-Minute Explanation

> "We have two experts: one with strong priors (80k battles) and one learning from scratch. When the prior is wrong, the algorithm automatically detects this by tracking losses and exponentially reduces the prior's influence. This gives us safety against negative transfer while keeping the benefits when priors are correct."

## Files You'll Create

```
experiments_v1/05_figure/results/
├── figure5_corralling_weights.pdf  ← Main result
└── figure5_corralling_weights.png  ← For slides/web
```

## Success Criteria

✅ **Plot generated**: Check `results/` folder
✅ **Weight evolution**: See two lines (red/green)
✅ **Decommissioning**: Red line drops sharply (if prior is wrong)
✅ **Terminal summary**: Final weights printed

## Common Questions

**Q**: Why two experts instead of one?

**A**: Safety. If warmup prior is wrong, we need a backup that learns from scratch.

**Q**: What if neither expert dominates?

**A**: Great! Both contribute useful information. The algorithm found optimal mixing.

**Q**: Does this slow down inference?

**A**: Negligible. ~0.2ms overhead vs 100ms LLM latency.

**Q**: Can I use more than 2 experts?

**A**: Yes! Modify `CorrallingRouter` initialization. The math generalizes to K experts.

## For Paper Writers

Use this plot to demonstrate:

1. **Adaptive robustness**: Algorithm detects and corrects prior mismatch
2. **Exponential decay**: Sharp drop shows exp(-η·ℓ) dynamics
3. **Automatic calibration**: No manual tuning needed (η=1.0 works)
4. **Worst-case safety**: Converges to better expert even when prior fails

## Need Help?

- **Code**: See `src/bandit_gpt/router.py` (CorrallingRouter class)
- **Math**: See `CORRALLING_SUMMARY.md` (detailed algorithm)
- **Usage**: See `README.md` (full documentation)

