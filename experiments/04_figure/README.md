# Figure 4: Adaptive Drift Detection Under Distribution Shift

**When warmup priors are trained on benchmark data and deployed on
shifted real-world traffic, the router's predictions become miscalibrated.
BanditGPT's built-in covariate shift detector monitors embedding
statistics and automatically triggers a tabula rasa reset—no human
intervention required.**

## Key Finding

A controlled 2σ synthetic distribution shift (embedding perturbation +
Llama reward boost) causes warmup-only priors to over-route to Gemini.
The adaptive detector fires within ~50 steps of the change-point and
triggers a prior reset, achieving **8.9% lower regret** than warmup-only
and **2.5% lower** than an oracle forgetting factor.

---

## Setup

- **K=2 portfolio**: Llama-3.1-8B (cheap, norm_cost≈0.00) + Gemini-2.5-Pro
  (frontier, norm_cost≈0.67)
- **Prior source**: Pareto benchmark train split (8,374 prompts)
- **Phase 1 (steps 0–500)**: In-distribution Pareto prompts
- **Phase 2 (steps 501–3,306)**: K4 real-world prompts with 2σ synthetic
  shift (8 leading PCA components shifted, +0.08 Llama reward boost)
- **Cost penalty**: λ=0.2 (cost-adjusted oracle routes 60% to Llama)
- **Hyperparameters**: α=0.5 (re-tuned at n_eff=5000), disjoint LinUCB,
  no Corralling

## Conditions

| Condition | Priors | Adaptation | Drift Detection |
|-----------|--------|------------|-----------------|
| **Warmup-only** | n_eff=5000 | None | Disabled |
| **Oracle ff=0.999** | n_eff=5000 | Forgetting factor 0.999 | Disabled |
| **Adaptive (Reset)** | n_eff=5000 | Tabula rasa reset on detection | χ² threshold 2σ |
| **Tabula Rasa** | None | N/A | N/A |

### Drift Detection Parameters

- `drift_threshold=2.0`: EMA chi-squared must exceed baseline + 2σ
- `drift_burn_in_steps=50`: Steps to establish baseline statistics
- `drift_ema_alpha=0.05`: EMA smoothing (half-life ≈ 14 steps)
- `drift_confirmation_window=20`: Consecutive threshold exceedances
- `drift_adaptation=tabula_rasa_reset`: Discard A/b matrices entirely

---

## Protocol

1. K=2 warmup priors pre-trained on Pareto **train** split.
2. Phase 1: 500 in-distribution prompts (warm-up, all conditions identical).
3. Phase 2: ~2,800 shifted K4 prompts with synthetic perturbation.
4. Frozen holdout evaluation every 50 steps (read-only `select_arm()`
   to avoid contaminating drift detector state).
5. Drift detector state (EMA chi², baseline, threshold) snapshotted
   at each checkpoint.
6. All results averaged over 20 seeds with 95% t-CIs.

---

## Results Summary

| Condition | Final Regret | vs. Warmup-only |
|-----------|-------------|-----------------|
| Warmup-only | 226.6 ± 8.0 | — |
| Oracle ff=0.999 | 211.9 ± 5.1 | −6.5% |
| **Adaptive (Reset)** | **206.5 ± 4.6** | **−8.9%** |
| Tabula Rasa | 223.8 ± 8.4 | −1.2% |

Reset fires at step ~550 (within ~50 steps of the Phase 2 boundary).

---

## Figure Panels

- **(a) Online Regret** — Cumulative cost-adjusted regret. Adaptive
  achieves the lowest final regret by resetting stale priors.
- **(b) Drift Detection Signal** — EMA chi-squared score for the
  Adaptive condition, with baseline and 2σ threshold. Reset marker
  shows detection point.
- **(c) Cheap-Arm Discovery** — Llama routing fraction over time.
  Tabula Rasa starts at 100% Llama (cold-start: cost penalty alone
  determines routing without learned quality signals), then learns
  the correct ranking within ~100 steps. After the shift, Adaptive
  discovers Llama at the same rate as Oracle ff.

---

## Directory Structure

```
04_figure/
├── run_distribution_shift.py      # Main experiment (4-condition comparison)
├── calibrate_drift_threshold.py   # Regret-grounded threshold calibration
├── tune_alpha_high_neff.py        # α re-tuning at production n_eff
├── plot_results.py                # Generate Figure 4 (3-panel)
├── figure4_caption.tex            # LaTeX figure caption
├── results_discussion.tex         # LaTeX results discussion section
├── README.md                      # This file
└── results/
    ├── distribution_shift_results.json
    ├── figure4_adaptive_drift.png
    └── figure4_adaptive_drift.pdf
```

---

## Quick Start

```bash
cd experiments/04_figure/

# Run the 4-condition experiment (~10-15 min)
python run_distribution_shift.py

# Generate figures
python plot_results.py
```

---

## Connection to Other Figures

- **Figure 1**: Pareto frontier (static tradeoff, same-distribution)
- **Figure 3**: Warmup ablation (K=2, regret reduction from priors)
- **Appendix I**: PCA component ablation for embedding features
- **DriftDetector unit tests**: `tests/test_drift_detector.py`

---

**Last Updated**: March 2026
