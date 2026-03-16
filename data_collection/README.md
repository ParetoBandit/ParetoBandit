# Data Collection

All reward data, source prompts, and the scripts that produced them.

## Datasets

### Primary: 12K Pareto Benchmark (K=3)

- **Location**: `pareto_dataset/`
- **Builder script**: `scripts/build_router_pareto_dataset.py`
- **Prompts**: 12,000 from 8 public HuggingFace benchmarks
- **Sources**: MMLU (2,655), GSM8K (2,403), OpenBookQA (1,153), HellaSwag (1,134),
  ARC-Challenge (1,060), TruthfulQA (748), WinoGrande (589), MBPP (323),
  BIG-Bench Hard (935 across 27 sub-tasks)
- **Selection**: Quality-filtered (length, ASCII ratio) → semantically deduplicated
  (cosine threshold 0.85) → random subsample to 12K. **No bias toward routing
  difficulty**; Pareto-interesting classification is applied post-hoc.
- **Models**: Llama-3.1-8B, Mistral-Large-2512, Gemini-2.5-Pro
- **Judge**: DeepSeek-R1 (single judge), v3 continuous rubric (Reasoning 40%,
  Instruction Following 30%, Communication 30%)
- **Splits** (in `rewards/`):
  - `train.jsonl` — 8,374 prompts (70%)
  - `val.jsonl` — 1,785 prompts (15%)
  - `test.jsonl` — 1,824 prompts (15%)
  - Stratified by benchmark source. Built by `experiments/benchmark/build_splits.py`.

### Archive: 4K Friction-Stratified (K=4, 3-Judge Panel)

- **Location**: `rewards/archive/k4_canonical/`
- **Builder script**: `scripts/build_diverse_prompt_set.py`
- **Prompts**: 4,009 from real user conversations + reasoning benchmarks
- **Sources**: LMSYS Chatbot Arena, WildChat (lmsys-chat-1m), BIG-Bench Hard
- **Selection**: Quality-filtered → semantically deduplicated → **friction-stratified
  KMeans centroid selection**. Friction = inter-model response disagreement measured
  via two small Ollama models (Llama-3.2-3B, Gemma-2B). This deliberately
  over-represents prompts where models disagree — a different selection philosophy
  from the 12K set.
- **Models**: Llama-3.1-8B, Mistral-Large-2512, Gemini-2.5-Pro, Gemini-2.5-Flash
- **Judge**: 3-judge panel (DeepSeek-R1, GPT-4.1-mini, Claude-3.5-Haiku)
- **Use**: Judge robustness analysis only (Appendix). Not part of the main
  evaluation to avoid conflating prompt selection strategies.

### Why Two Different Prompt Selection Approaches

The 4K friction-stratified set was our initial approach: enrich for "hard routing"
prompts by measuring inter-model disagreement upfront. While this produces a dataset
with more routing signal per-prompt, it introduces selection bias — the prompt
distribution is intentionally skewed toward cases where routing matters.

For the 12K pareto set, we switched to unbiased sampling from public benchmarks to
ensure results reflect a natural task distribution. The Pareto-interesting
classification is applied post-hoc (after reward collection), so it cannot
influence the prompt selection. This is the standard the main paper results use.

The 4K set remains valuable for an orthogonal purpose: validating that our
single-judge scoring (DeepSeek-R1) produces consistent routing decisions compared
to a 3-judge panel.

## Structure

```
data_collection/
├── pareto_dataset/                   # Primary 12K benchmark
│   ├── pareto_prompts.jsonl          # 12K prompt texts with sources
│   ├── pareto_rewards.jsonl          # Full-information rewards (3 models × 12K)
│   ├── pareto_classified.jsonl       # Post-hoc difficulty classification
│   └── pareto_balanced.jsonl         # Balanced subset (for exploratory analysis)
├── rewards/                          # Canonical splits for experiments
│   ├── train.jsonl                   # Train split (8,374 prompts)
│   ├── val.jsonl                     # Validation split (1,785 prompts)
│   ├── test.jsonl                    # Test split (1,824 prompts)
│   ├── calibration/                  # Judge calibration data (200 prompts)
│   └── archive/                      # Historical datasets
│       ├── k4_canonical/             # 4K 3-judge dataset (see above)
│       ├── k4_pipeline/              # Earlier K=4 experiments
│       ├── k5_pipeline/              # Earlier K=5 experiments
│       ├── legacy_k10/               # Original K=10 reward data
│       └── v2_binary_rubric/         # Binary rubric experiments
├── prompts/
│   ├── lmarena_battles_en.jsonl      # LMSYS Chat Arena raw data (~48K prompts)
│   └── archive/                      # Historical prompt sets
├── embeddings/                       # PCA projection data
├── warmup_priors/                    # Bandit warmup priors (from train split)
├── scripts/                          # Data-generating scripts
│   ├── build_router_pareto_dataset.py    # Builds the 12K pareto dataset
│   ├── build_diverse_prompt_set.py       # Builds the 4K friction-stratified set
│   ├── rejudge_cot.py                    # Multi-judge CoT reward generator
│   ├── merge_and_split_rewards.py        # Merges reward files and creates splits
│   ├── calibrate_cheap_judges.py         # Judge calibration experiments
│   ├── score_prompt_difficulty.py        # Prompt difficulty scoring
│   └── legacy/                           # Older scripts
├── cache/                            # API response caches
└── config/
    ├── models_k3.json                # K=3 model portfolio
    └── models_k10.json               # K=10 model portfolio (historical)
```

## Reward Signal (v3 Continuous Rubric)

Each response is scored on three axes:

| Factor | Weight | Range |
|---|---|---|
| Reasoning Quality | 40% | 0.0–1.0 |
| Instruction Following | 30% | 0.0–1.0 |
| Communication Quality | 30% | 0.0–1.0 |

The composite reward is the weighted sum, producing a continuous value in [0, 1].
For the primary dataset, scoring uses DeepSeek-R1 as a single judge (highest
discriminative signal among tested judges). See `src/pareto_bandit/rewards.py` for
the extraction logic.

## Generating New Rewards

```bash
# Single judge (DeepSeek-R1) — used for main results:
python data_collection/scripts/build_router_pareto_dataset.py

# Multi-judge panel — for robustness checks:
python data_collection/scripts/rejudge_cot.py \
  --mode custom \
  --prompts-file <prompts.jsonl> \
  --models-file data_collection/config/models_k3.json \
  --output-file <output.jsonl> \
  --workers 32
```

## Config Integration

All paths are centralized in `src/pareto_bandit/config/__init__.py`.
