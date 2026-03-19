# Reward Data — Provenance and Reproduction

## Dataset Summary

| Split | Prompts | File |
|-------|---------|------|
| Train | 8,374 | `../train.jsonl` |
| Val | 1,785 | `../val.jsonl` |
| Test | 1,824 | `../test.jsonl` |
| **Total** | **11,983** | |

## K=3 Arms

- `meta-llama/llama-3.1-8b-instruct` (budget)
- `mistralai/mistral-large-2512` (mid-cost)
- `google/gemini-2.5-pro` (premium)

## Reproduction Pipeline

### Step 1: Collect rewards from public benchmarks

```bash
python data_collection/rewards/provenance/build_router_pareto_dataset.py
```

- Sources ~12K diverse prompts from 9 public HuggingFace benchmarks:
  TruthfulQA, GSM8K, MBPP, ARC-Challenge, OpenBookQA, WinoGrande,
  MMLU, HellaSwag, BIG-Bench Hard (32 subtasks total)
- For each prompt x arm, generates a response via OpenRouter
- Judges each response with DeepSeek-R1 using the v3 continuous rubric
  (logic x constraint x utility)
- Produces `data_collection/pareto_dataset/pareto_rewards.jsonl` and
  `pareto_classified.jsonl`
- Requires `OPENROUTER_API_KEY` environment variable

### Step 2: Stratified split

```bash
python data_collection/rewards/provenance/build_splits.py
```

- Reads `pareto_classified.jsonl` (11,983 prompts after quality filtering)
- Stratified split by source benchmark (70/15/15), seed=42
- Produces `train.jsonl`, `val.jsonl`, `test.jsonl`
- Also generates warmup priors from the train split

## Schema

Each JSONL line contains:

```json
{
  "prompt": "...",
  "source": "gsm8k",
  "difficulty": "pareto_interesting",
  "best_arm": "mistralai/mistral-large-2512",
  "reward_spread": 0.15,
  "arms": {
    "meta-llama/llama-3.1-8b-instruct": {"reward": 0.72, "cost": 0.00005},
    "mistralai/mistral-large-2512": {"reward": 0.91, "cost": 0.0004},
    "google/gemini-2.5-pro": {"reward": 0.87, "cost": 0.002}
  }
}
```

## Notes

- The `train.jsonl`, `val.jsonl`, `test.jsonl` files in the parent
  directory are identical to the copies in `experiments/benchmark/`.
- The split manifest (`split_manifest.json`) was generated alongside the
  splits and is archived in `../archive/`.
