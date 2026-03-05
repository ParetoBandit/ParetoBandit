# Figure 6 — Semantic Transfer Evaluation (K=10)

## Motivation

Appendix D found a null result for model-to-model theta-transfer.
Since then, the router, PCA components (15-comp), and encoder
(BAAI/bge-m3) have been updated.  This experiment re-evaluates whether
those changes produce meaningful semantic transfer when bootstrapping
a new model from its nearest neighbor.

## Design (Leave-One-Out)

For each model in the K=10 portfolio:

1. **Target**: One model is treated as a simulated newcomer (no prior).
2. **Base models**: The remaining K-1 models receive warmup priors from
   the canonical K=10 prior file.
3. **Condition A (semantic transfer)**: Add target via `register_model()`
   with neighbor selected by within-provider tetrachoric correlation.
4. **Condition B (tabula rasa)**: Add target with identity init (no
   transfer).
5. Both conditions use the same BanditRouter (Corralling + Hybrid
   LinUCB), same data, same seeds, same shuffled training order.

## Data Separation

| Component | Source | Role |
|-----------|--------|------|
| Prior-train pool | `THREE_WAY_SPLITS_PATH` | Excluded from online learning |
| Dev-train (80%) | `DEV_DATA_PATH_ALL_MODELS` minus prior-train | Online routing stream |
| Dev-val (20%) | Same split | Reserved (hparams from Appendix H) |
| Holdout | `HOLDOUT_DATA_PATH_ALL_MODELS` | Frozen evaluation only |
| Warmup priors | `K10_WARMUP_PRIORS_PATH` | K-1 base model initialization |

Hyperparameters are loaded from Appendix H (`best_hparams_k10.json`).

## Prerequisites

1. Warmup priors:
   ```bash
   python scripts/extract_warmup_from_multimodel.py
   ```
2. Appendix H hyperparameters:
   ```bash
   python experiments/appendix/H_alpha_neff_ablation/run_3d_grid_ablation.py
   ```
3. Embedding cache (optional, speeds up runs):
   ```bash
   python scripts/precompute_embeddings.py
   ```

## Run

```bash
python experiments/06_figure/run_semantic_transfer.py
```

## Plot

```bash
python experiments/06_figure/plot_results.py
```

## Output

- `results/semantic_transfer_results.json` — Full per-target, per-trial results
- `results/semantic_transfer_summary.txt` — Human-readable summary table
- `results/figure6_semantic_transfer.png` — Two-panel figure
- `results/figure6_semantic_transfer.pdf` — Publication-quality PDF

## Statistics

- Per-target: paired t-test (df = N_SEEDS - 1), Cohen's d, 95% CI
- Across targets: Holm-Bonferroni correction
- Aggregate: portfolio-level paired test over all (target x seed) pairs
