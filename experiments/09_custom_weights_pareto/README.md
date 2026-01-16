# Experiment 09: Custom Weights Pareto Demonstration

## Overview

This experiment demonstrates how custom quality/cost/latency weights affect model selection in BanditGPT and visualizes the resulting cost-quality tradeoff through a Pareto curve.

## What This Shows

1. **Cost Saver Profile** (`w_q=1.0, w_c=10.0, w_l=0.0`)
   - Heavily penalizes expensive models
   - Selects the best low-cost options
   - Shows that you can maintain reasonable quality while minimizing cost

2. **High Quality Profile** (`w_q=10.0, w_c=0.0, w_l=0.0`)
   - Maximizes quality independent of cost
   - Selects premium models (GPT-5, Claude Opus, etc.)
   - Shows the maximum quality ceiling available in the portfolio

3. **Balanced Profile** (`w_q=5.0, w_c=5.0, w_l=0.0`)
   - Balanced tradeoff between quality and cost
   - Demonstrates the "sweet spot" for most use cases

## Custom Weight Mechanics

From `router.py` (lines 2533-2631), the utility calculation is:

```python
# Base utility (deterministic tradeoff)
base_utility = (w_q * norm_quality) + (w_c * (1.0 - norm_cost)) + (w_l * (1.0 - norm_lat))

# Exploration bonus (scaled by quality weight)
exploration_bonus = alpha * alpha_scale * w_q * std

# Total utility
total_utility = base_utility + exploration_bonus + probation_bonus
```

Where:
- `w_q`: Quality weight (how much you value quality)
- `w_c`: Cost penalty weight (1.0 = base currency unit)
- `w_l`: Latency penalty weight
- `norm_quality`: Predicted quality in [0, 1] (from LinUCB)
- `norm_cost`: Normalized cost penalty in [0, 1] (from market anchors)
- `norm_lat`: Normalized latency penalty in [0, 1] (from market anchors)

## Running the Experiment

### Step 1: Run the experiment (1 trial for testing)

```bash
cd experiments/09_custom_weights_pareto
python run_custom_weights.py
```

This will:
- Load the model registry and test data
- Initialize BanditRouter with warmup priors
- Route 100 test prompts with each weight profile
- Track model selections, costs, and quality scores
- Save results to `results/custom_weights_results.json`

### Step 2: Visualize results

```bash
python plot_custom_weights.py
```

This will generate:
- `results/pareto_curve.png`: Cost vs quality scatter plot with Pareto frontier
- `results/selection_distribution.png`: Bar charts showing which models each profile selected

## Expected Results

### Cost Saver Profile
- **Expected Cost**: ~$0.50 - $1.00 per 1M tokens
- **Expected Quality**: ~90-95%
- **Top Models**: grok-3-mini, gpt-oss-120b (cheap but capable)

### High Quality Profile
- **Expected Cost**: ~$4.00 - $8.00 per 1M tokens
- **Expected Quality**: ~96-98%
- **Top Models**: GPT-5.1, Claude Opus 4.5 (premium flagship models)

### Balanced Profile
- **Expected Cost**: ~$2.00 - $3.00 per 1M tokens
- **Expected Quality**: ~93-96%
- **Top Models**: Mix of mid-tier and flagship models depending on prompt

## Scaling to Multiple Trials

Once you verify the script works with 1 trial, you can scale up:

```python
# In run_custom_weights.py, modify the main() function:

def main():
    # ... existing setup code ...
    
    # Change from 1 to 10 or more trials
    test_samples = load_test_data(n_samples=500)  # More samples
    
    # Run multiple trials for variance estimation
    n_trials = 10
    all_trial_results = []
    
    for trial in range(n_trials):
        router = BanditRouter.create(...)  # Fresh router each trial
        
        for profile_key, profile_config in profiles.items():
            result = run_profile_experiment(...)
            all_trial_results.append(result)
    
    # Aggregate results across trials
    # ... compute mean, std, confidence intervals ...
```

## Files

- `run_custom_weights.py`: Main experiment script
- `plot_custom_weights.py`: Visualization script
- `results/custom_weights_results.json`: Experiment results (generated)
- `results/pareto_curve.png`: Pareto frontier visualization (generated)
- `results/selection_distribution.png`: Model selection frequencies (generated)

## Interpretation

The Pareto curve shows the **efficiency frontier** of your model portfolio:

1. **Points on the frontier** = Pareto-optimal models (no other model has both lower cost AND higher quality)
2. **Points below the frontier** = Dominated models (there exists a cheaper model with equal quality, or equal-cost model with higher quality)
3. **Router profiles** = How different weight configurations navigate this frontier

**Key Insight**: By adjusting `w_q` and `w_c`, you can slide along the Pareto frontier to match your business constraints (e.g., "I need 95% quality for under $2/M tokens").

