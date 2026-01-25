# Quick Start Guide: Figure 3 - Corralled Semantic Analysis

## Prerequisites

1. **Data**: Ensure you have the labeled data:
   - LMSYS Holdout: `src/bandit_gpt/data/offline_dataset/dev_rewards_complete.jsonl.gz`
   - (Optional) 1M Dataset: `experiments_v1/appendix_d/data/lmsys_chat_1M.jsonl.gz`

2. **Models**: Ensure you have the trained models:
   - PCA model: `src/artifacts/pca_model_routellm.joblib`
   - Warmup priors: `src/artifacts/warmup_priors_routellm.joblib`

3. **Dependencies**: Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Experiment

### Option 1: Basic Run (Recommended)

Train on LMSYS Holdout (N=1,871) and project onto 1M space:

```bash
cd /Users/annette/repostitories/banditGPT

python experiments_v1/03_figure/corralled_semantic_analysis.py \
    --learning-rate 1.0 \
    --gamma 0.05 \
    --train-size 1871
```

**Expected Runtime**: ~5-10 minutes (depending on whether 1M data is available)

### Option 2: Quick Test (No 1M Projection)

If you don't have the 1M dataset, the script will automatically skip the projection phase and use only the training data for visualization:

```bash
python experiments_v1/03_figure/corralled_semantic_analysis.py \
    --learning-rate 1.0 \
    --train-size 1871
```

**Expected Runtime**: ~2-3 minutes

### Option 3: Custom Learning Rate

Test different learning rates to see how adaptation speed changes:

```bash
# Slower adaptation
python experiments_v1/03_figure/corralled_semantic_analysis.py \
    --learning-rate 0.5 \
    --output results/eta_0.5

# Faster adaptation
python experiments_v1/03_figure/corralled_semantic_analysis.py \
    --learning-rate 2.0 \
    --output results/eta_2.0
```

### Option 4: Limited Projection (Faster)

If you have the 1M dataset but want faster results, limit the projection size:

```bash
python experiments_v1/03_figure/corralled_semantic_analysis.py \
    --learning-rate 1.0 \
    --projection-size 10000
```

**Expected Runtime**: ~3-4 minutes

## Output Files

The script creates the following files in `experiments_v1/03_figure/results/`:

1. **`figure3_corralling_semantic_analysis.png`**: Main figure for the paper
   - Left panel: Semantic space with cluster structure
   - Right panel: Expert weight evolution

2. **`figure3_corralling_semantic_analysis_hires.png`**: High-resolution version (600 DPI)

3. **`training_metrics.png`**: Training curves
   - Left panel: Cumulative regret
   - Right panel: Average reward

4. **`results.json`**: Numerical results
   ```json
   {
     "learning_rate": 1.0,
     "gamma": 0.05,
     "train_size": 1871,
     "cumulative_regret": 245.32,
     "avg_reward": 0.8456,
     "final_expert_weights": [0.247, 0.753],
     "model_usage": {
       "mixtral-8x7b-instruct-v0.1": 456,
       "gpt-3.5-turbo-0125": 234,
       ...
     }
   }
   ```

## Interpreting Results

### Expert Weights

The final expert weights tell you which expert won:

- **Tabula Rasa > Warmup**: Algorithm unlearned the warmup bias ✅
  - Example: [0.247, 0.753] → Tabula Rasa won 3.05× more weight
  - Interpretation: Warmup priors were harmful (negative transfer)

- **Warmup > Tabula Rasa**: Warmup priors were helpful
  - Example: [0.753, 0.247] → Warmup won 3.05× more weight
  - Interpretation: Warmup priors accelerated learning

### Semantic Clusters

The left panel shows two clusters:

- **Easy Cluster** (PC1 < 0.3, blue): ~94.1% of prompts
  - High semantic density (visible in KDE contours)
  - Can be served by cheaper models (Mixtral, GPT-3.5)

- **Hard Cluster** (PC1 ≥ 0.3, red): ~5.9% of prompts
  - Lower density, more diverse
  - Requires more capable models (GPT-4, Claude-3)

### Model Usage

The projected policy shows which models are selected across the semantic space:

- **High Mixtral usage** (e.g., 48%): Algorithm discovered Easy cluster
- **Balanced usage**: Algorithm uses different models for different regions
- **High flagship usage** (e.g., GPT-4 > 50%): Warmup bias persists

## Troubleshooting

### Error: "Data file not found"

**Problem**: Missing labeled data or 1M dataset

**Solution**:
1. Check if `src/bandit_gpt/data/offline_dataset/dev_rewards_complete.jsonl.gz` exists
2. If missing, run data generation script:
   ```bash
   python scripts/generate_gpt4_turbo_rewards.py
   ```
3. For 1M dataset, run:
   ```bash
   python experiments_v1/appendix_d/download_1M_dataset.py
   ```

### Error: "PCA file not found"

**Problem**: Missing PCA model

**Solution**:
1. Check if `src/artifacts/pca_model_routellm.joblib` exists
2. If missing, train PCA:
   ```bash
   python scripts/train_pca_from_routellm.py
   ```

### Error: "Warmup priors not found"

**Problem**: Missing warmup priors

**Solution**:
1. Check if `src/artifacts/warmup_priors_routellm.joblib` exists
2. If missing, generate priors:
   ```bash
   python scripts/generate_warmup_priors.py
   ```

### Slow Performance

**Problem**: Script takes too long

**Solutions**:
1. Reduce training size: `--train-size 500`
2. Limit projection: `--projection-size 10000`
3. Skip 1M projection (script auto-detects if file missing)

### Memory Issues

**Problem**: Out of memory error

**Solutions**:
1. Reduce batch size (edit script: `batch_size=32` instead of `batch_size=64`)
2. Limit projection size: `--projection-size 50000`
3. Use smaller training set: `--train-size 1000`

## Next Steps

After running the experiment:

1. **Review outputs**:
   ```bash
   open experiments_v1/03_figure/results/figure3_corralling_semantic_analysis.png
   open experiments_v1/03_figure/results/training_metrics.png
   cat experiments_v1/03_figure/results/results.json
   ```

2. **Compare learning rates**:
   ```bash
   # Run with different learning rates
   python experiments_v1/03_figure/corralled_semantic_analysis.py --learning-rate 0.5 --output results/eta_0.5
   python experiments_v1/03_figure/corralled_semantic_analysis.py --learning-rate 1.0 --output results/eta_1.0
   python experiments_v1/03_figure/corralled_semantic_analysis.py --learning-rate 2.0 --output results/eta_2.0
   
   # Compare results
   python -c "
   import json
   for eta in [0.5, 1.0, 2.0]:
       with open(f'experiments_v1/03_figure/results/eta_{eta}/results.json') as f:
           r = json.load(f)
       print(f'η={eta}: Regret={r[\"cumulative_regret\"]:.2f}, Weights={r[\"final_expert_weights\"]}')
   "
   ```

3. **Integrate into paper**:
   - Copy `figure3_corralling_semantic_analysis.png` to paper figures directory
   - Include LaTeX caption from `figure3_caption.tex`
   - Reference in results section

4. **Run ablation studies** (optional):
   - Vary gamma: `--gamma 0.01`, `--gamma 0.1`
   - Vary training size: `--train-size 500`, `--train-size 5000`
   - Compare with baseline (no Corralling)

## Expected Results

Based on our implementation, you should see:

1. **Expert Weights**: Tabula Rasa wins (~75% final weight)
   - Demonstrates successful unlearning of warmup bias

2. **Cumulative Regret**: ~200-300 (on N=1,871)
   - Competitive with oracle (which has regret = 0)

3. **Average Reward**: ~0.84-0.86
   - High reward indicates good model selection

4. **Model Usage**: Mixtral dominates (~40-50%)
   - Shows algorithm discovered Easy cluster

5. **Cluster Distribution**: Easy cluster = ~94%
   - Validates semantic structure hypothesis

## Questions?

- See `README.md` for detailed documentation
- See `IMPLEMENTATION_SUMMARY.md` for mathematical details
- See `figure3_caption.tex` for LaTeX integration
- Check `src/bandit_gpt/router.py` for CorrallingRouter implementation

