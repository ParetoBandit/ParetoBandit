# Data Collection

All reward data, source prompts, and the scripts that produced them.

## Structure

```
data_collection/
├── rewards/                          # Canonical reward datasets
│   ├── dev_rewards_complete_all_models.jsonl.gz   # Dev set (44 models, 2,854 prompts)
│   └── holdout_rewards_complete_all_models.jsonl.gz  # Holdout set (44 models, 1,500 prompts)
├── prompts/                          # Source prompts
│   └── lmarena_battles_en.jsonl      # LMSYS Chat Arena battles (~48K unique prompts)
├── scripts/                          # Data-generating scripts
│   ├── rejudge_cot.py                # Multi-judge CoT reward generator (OpenRouter API)
│   ├── download_and_process_routellm.py  # Downloads RouteLLM battles from HuggingFace
│   ├── sample_new_prompts.py         # Samples prompts with deduplication
│   └── legacy/                       # Older data preparation scripts
├── cache/                            # API response caches
└── config/
    └── models_k10.json               # K=10 model portfolio definition
```

## Reward Signal

Each reward entry is scored by a multi-judge CoT panel (GPT-4o, Claude 3.5 Sonnet,
Llama 405b, Gemini 2.5 Pro). The canonical reward is `mean(vote * confidence)` across
judges, producing a continuous value in [0, 1]. See `src/bandit_gpt/rewards.py` for
the extraction logic.

## Generating New Rewards

```bash
python data_collection/scripts/rejudge_cot.py \
  --mode custom \
  --prompts-file data_collection/prompts/my_prompts.jsonl \
  --models-file data_collection/config/models_k10.json \
  --output-file data_collection/rewards/new_rewards.jsonl \
  --workers 32
```

The script flushes each result to disk immediately and supports resume on restart.

## Config Integration

All paths are centralized in `src/bandit_gpt/config/__init__.py`:

- `OFFLINE_DATASET_DIR` -> `data_collection/rewards/`
- `PROMPTS_DIR` -> `data_collection/prompts/`
- `CACHE_DIR` -> `data_collection/cache/`
