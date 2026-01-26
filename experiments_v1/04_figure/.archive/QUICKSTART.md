# Figure 4: Pareto Frontier - Quick Start

## Run the Experiment

### Option 1: Combined Dev + Holdout (Default, N=1,871)
```bash
cd experiments_v1/04_figure
python generate_pareto_frontier.py
```
**Use this for**: Smoother Pareto curves with reduced variance

### Option 2: Holdout Only (Standard Practice, N=750)
```bash
cd experiments_v1/04_figure
python generate_pareto_frontier.py --holdout-only
```
**Use this for**: Strict held-out test set evaluation (standard ML practice)

## What It Does

Generates a Pareto frontier plot showing cost-quality trade-offs for different routing strategies:

1. **Oracle**: Always picks the best model (upper bound)
2. **Static baselines**: Always use Mixtral or GPT-4-turbo
3. **RouteLLM-Static**: Threshold-based routing
4. **Warmup-Only**: Prior-based routing
5. **banditGPT Hybrid**: Adaptive online learning (η=1.0)

## Data Used (REAL ONLY)

### Default Mode (Combined)
- **N=1,871 prompts** (dev + holdout combined)
- **Purpose**: Smoother curves, reduced variance
- **2 models**: Mixtral-8x7B, GPT-4-turbo
- **Real rewards**: From actual evaluations
- **Real costs**: From models.json

### Holdout-Only Mode
- **N=750 prompts** (holdout set only)
- **Purpose**: Strict held-out evaluation (standard practice)
- **Same models and real data**

## Output

```
results/
├── figure4_pareto_frontier.png          # Main plot
├── figure4_pareto_frontier_hires.png    # High-res version
└── pareto_results.json                  # Raw data
```

## Key Finding

**Mixtral outperforms GPT-4-turbo on this dataset!**
- Mixtral: 0.8156 quality @ $0.000294
- GPT-4: 0.8049 quality @ $0.013000
- Oracle: 0.9503 quality @ $0.002005 (mostly routes to Mixtral)

This demonstrates the value of adaptive routing—the "best" model varies by task.

## Troubleshooting

### Missing data files?
```bash
# Check that these exist:
ls src/bandit_gpt/data/offline_dataset/dev_rewards_2models.jsonl.gz
ls src/bandit_gpt/data/offline_dataset/holdout_rewards_2models.jsonl.gz
```

### Missing artifacts?
```bash
# Check that these exist:
ls src/artifacts/pca_32.joblib
ls src/artifacts/priors_warmup.joblib
```

### Import errors?
```bash
# Make sure you're in project root:
cd /path/to/banditGPT
python experiments_v1/04_figure/generate_pareto_frontier.py
```

## Customization

### Change learning rate
Edit line in script:
```python
learning_rate=1.0  # Try 0.1 for more conservative
```

### Change number of trials
Edit line in script:
```python
for trial in range(10):  # Try 20 for smoother curve
```

### Change threshold range
Edit line in script:
```python
thresholds = np.linspace(-0.1, 0.5, 20)  # Adjust range/density
```

## Understanding the Plot

- **X-axis**: Cost per request ($)
- **Y-axis**: Quality (reward, 0-1 scale)
- **Upper-left is better**: High quality, low cost
- **Pareto frontier**: The curve showing best achievable trade-offs
- **Production standard line**: Typical quality target (0.80)

## Next Steps

1. ✅ Generate the plot
2. 📊 Review `pareto_results.json` for exact numbers
3. 📝 Use for paper Figure 4
4. 🔬 Compare with Figure 3 (Corralling learning dynamics)
5. 📈 Reference in Table 2 (performance gap analysis)

