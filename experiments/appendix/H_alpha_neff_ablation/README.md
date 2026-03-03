# Appendix H: Prior Strength, Exploration, and Forgetting Ablation

**3D alpha x n_eff x gamma grid for the Corralling router**

This appendix section investigates how the prior strength (`prior_n_effective`),
exploration coefficient (`alpha`), and forgetting factor (`gamma`) interact in
the production Corralling router with whitened PCA warmup priors.

---

## Directory Structure

```
H_alpha_neff_ablation/
├── README.md                          # This file
├── run_3d_grid_ablation.py            # 3D alpha x n_eff x gamma grid
├── section_alpha_neff_ablation.tex    # LaTeX results & discussion
├── figure_alpha_neff_caption.tex      # Figure float for heatmap
└── results/
    └── alpha_neff_gamma_grid_results.json
```

---

## Quick Start

```bash
cd experiments/appendix/H_alpha_neff_ablation/

# 3D alpha x n_eff x gamma ablation
python run_3d_grid_ablation.py
```

The script uses K=2 (Mixtral + GPT-4-Turbo) and K=10 portfolios with
the whitened PCA warmup priors. It trains on dev-train, selects the best
configuration on dev-val, and reports the corresponding holdout reward.

---

## Key Finding (with whitened PCA)

| Portfolio | Optimal (alpha, n_eff, gamma) | Dev-Val | Holdout |
|-----------|-------------------------------|---------|---------|
| K=2       | (1.0, 5000, 1.0)             | 0.8024  | 0.8152  |
| K=10      | (0.1, 5000, 1.0)             | 0.9226  | 0.8701  |

Both portfolios favour strong priors (`n_eff=5000`) and stationary learning
(`gamma=1.0`). K=2 benefits from aggressive exploration (`alpha=1.0`), while
K=10 performs best with low exploration (`alpha=0.1`).

---

## Related Experiments

| Scenario | Location |
|----------|----------|
| K-scaling main experiment | [Figure 5](../../05_figure/) |
| Alpha ablation (n_eff=10) | [Figure 3](../../03_figure/) |
| PCA/neff calibration | [Appendix C.8](../C_extended_results/C8_pca_neff_calibration.tex) |
| Hyperparameter sensitivity | [Appendix C.5](../C_extended_results/C5_hyperparameter_sensitivity.tex) |
