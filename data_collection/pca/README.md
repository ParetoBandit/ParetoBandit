# PCA Artifacts — Provenance and Reproduction

## Artifacts

| File | Description |
|------|-------------|
| `pca_32.joblib` | Master PCA (384D -> 32D), fitted on ~46K disjoint LMSYS Arena prompts |
| `pca_25.joblib` | Strict truncation of PCA-32 (first 25 components), used by all v2 experiments |

PCA-25 is mathematically identical to PCA-32 with components 26-32 discarded.
The v2 experiments and the shipped pip package both use PCA-25 as the default.

The runtime copy of `pca_25.joblib` lives at `src/pareto_bandit/data/artifacts/pca_25.joblib`
and is loaded by `FeatureService` via `DEFAULT_PCA_PATH`.

## Reproduction Pipeline

### Step 1: Download LMSYS Arena data

```bash
python data_collection/pca/download_and_process_lmarena.py
```

- Downloads `lmarena-ai/arena-human-preference-140k` from HuggingFace
- Filters to English, `eval_order=1` (independent evaluations only)
- Deduplicates by normalized prompt text
- Outputs `data_collection/prompts/lmarena_battles_en.jsonl` (~51K battles)

### Step 2: Embed and fit PCA

```bash
python data_collection/pca/train_pca_from_routellm.py --n-components 25 32
```

- Loads `lmarena_battles_en.jsonl`
- Excludes experimental prompts (dev/holdout splits) for strict disjointness (~46K remain)
- Embeds with `all-MiniLM-L6-v2` (384D)
- Fits PCA and saves artifacts to `src/pareto_bandit/data/artifacts/`

## Encoder

All PCA artifacts are fitted on embeddings from `all-MiniLM-L6-v2`
(the default `DEFAULT_SENTENCE_TRANSFORMER` in `pareto_bandit.config`).
Using a different encoder requires retraining PCA via
`pareto_bandit.calibration.train_pca()`.
