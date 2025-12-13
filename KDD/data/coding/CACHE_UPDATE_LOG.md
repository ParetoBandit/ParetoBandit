# LiveCodeBench Models Cache Update Log

## Update Date: 2025-12-13

### Summary

Successfully updated `data/models_cache.json` with LiveCodeBench scores from the coding benchmark dataset.

### Statistics

- **Total models in cache**: 82
- **Models with LiveCodeBench scores**: 81 (98.8% coverage)
- **Models without LiveCodeBench scores**: 1 (1.2%)
- **Models updated in this run**: 1 (GPT-3.5 Turbo)

### Missing Scores

Only 1 model lacks a LiveCodeBench score:
- **GPT-3.5 Turbo** (`gpt-35-turbo`) - score is `null` in source data

### Score Distribution

| Statistic | LiveCodeBench Score |
|-----------|---------------------|
| Min       | 0.019 (1.9%)       |
| 25th      | 0.232 (23.2%)      |
| Median    | 0.334 (33.4%)      |
| 75th      | 0.636 (63.6%)      |
| Max       | 0.917 (91.7%)      |

### Top 10 Performers

Based on LiveCodeBench scores in the cache:

1. **Gemini 3 Pro Preview** - 0.917 (91.7%)
2. **gpt-oss-120B** - 0.878 (87.8%)
3. **Claude Opus 4.5 (Reasoning)** - 0.871 (87.1%)
4. **GPT-5.1** - 0.868 (86.8%)
5. **o4-mini** - 0.859 (85.9%)
6. **Kimi K2 Thinking** - 0.853 (85.3%)
7. **Grok 4** - 0.819 (81.9%)
8. **o3** - 0.808 (80.8%)
9. **Gemini 2.5 Pro** - 0.801 (80.1%)
10. **DeepSeek V3.1 Terminus (Reasoning)** - 0.798 (79.8%)

### Update Process

The update was performed using the script:
```bash
python KDD/data/coding/update_models_cache_with_livecodebench.py
```

**Source**: `KDD/data/coding/livecodebench_scores.json` (83 models with scores)
**Target**: `data/models_cache.json` (82 models in cache)

### Data Integrity

✅ **All scores are authentic from Artificial Analysis API**
✅ **No imputation or estimation performed**
✅ **Models without scores have `null` values**
✅ **Update script is auditable and reproducible**

### Next Steps

The LiveCodeBench scores in the models cache can now be used for:
1. **Composite Coding Score (CCS)** calculation
2. **Model routing decisions** for coding tasks
3. **Benchmark validation** and analysis
4. **Training data generation** for intent classification

### Files Updated

- `data/models_cache.json` - Main models cache (added/updated LiveCodeBench scores)

### Script Usage

To re-run the update in the future:

```bash
# From the repository root
cd KDD/data/coding
python update_models_cache_with_livecodebench.py

# Or make it executable and run directly
chmod +x update_models_cache_with_livecodebench.py
./update_models_cache_with_livecodebench.py
```

The script will:
1. Load LiveCodeBench scores from `livecodebench_scores.json`
2. Load the main models cache from `data/models_cache.json`
3. Match models by slug
4. Update missing or changed scores
5. Save the updated cache
6. Report statistics

### Notes

- The script only updates models that are missing LiveCodeBench scores or have outdated scores
- Models already having the correct score are skipped (reported as "already had correct score")
- The script is idempotent - safe to run multiple times
- All updates are logged to stdout for auditing
