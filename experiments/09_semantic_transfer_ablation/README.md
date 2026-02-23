# Semantic Transfer Ablation (Experiment 09)

## Purpose

Quantifies the value of **per-model semantic transfer** when bootstrapping a "new" model in the bandit router. This experiment directly addresses the claim in Section `sec:transfer_null` of the paper: whether transferring θ from a semantically similar neighbor provides statistically significant improvement over tabula rasa initialization.

## Design (Leave-One-Out)

For each model in the K=5 and K=10 portfolios:

1. **Target**: One model is treated as the simulated "newcomer" (no prior).
2. **Base models**: The remaining K−1 models receive warmup priors from the canonical prior files.
3. **Condition A (semantic transfer)**: Add target via `register_model()` — production path that bootstraps θ from the nearest neighbor.
4. **Condition B (tabula rasa)**: Add target with identity init (A=λI, b=0) — no transfer.
5. Both conditions use the same BanditRouter (Corralling + Hybrid LinUCB), same data, same seeds.

**Treatment variable**: Initialization of the target model only.

## Data

- **Warmup priors**: `MULTIMODEL_WARMUP_PRIORS_PATH` (K=5 and K=10 portfolios).
- **Prompts & rewards**: Real LMSYS Arena data from the 43-model evaluation dataset.
- **Splits**: `THREE_WAY_SPLITS_PATH` (online-learn pool + holdout).

## Prerequisites

1. Warmup priors must exist:
   ```bash
   python scripts/generate_multimodel_warmup_priors.py
   ```
2. Three-way split and data files (see `experiments/utils/multimodel.py`).

## Run

```bash
cd experiments/09_semantic_transfer_ablation
python run_semantic_transfer_experiment.py
```

## Output

- `results/semantic_transfer_ablation.json` — Full results (per-target, per-trial).
- `results/semantic_transfer_ablation_summary.txt` — Human-readable summary with Δ and p-values.

## Statistics

- Paired t-test across N_TRIALS seeds (default 20).
- Reports: mean holdout reward, 95% CI, Δ (transfer − tabula rasa), p-value.
- Significance: ** p < 0.01, * p < 0.05.
