# Figure 4: The Value of Warmup Priors

**Ablation showing that warmup priors are a key architectural ingredient of BanditGPT**

This experiment isolates the contribution of warmup priors by comparing:
- **BanditGPT** (Corralling + warmup priors): the full system
- **Tabula Rasa** (single LinUCB, no priors, no Corralling): ablation

Both are evaluated on the K=10 portfolio under identical conditions (same data, splits, seeds). Supervised baselines (KNN, SVM, MLP) provide reference anchors.

---

## Connection to Other Figures

- **Figure 1:** Established semantic structure and model preference heterogeneity
- **Figure 2:** Architecture diagram
- **Figure 3:** BanditGPT vs LLMRouter supervised baselines (K=2 and K=10)
- **Appendix H:** Per-expert hyperparameter tuning (consumed here)

**Key Question:** How much do warmup priors contribute to BanditGPT's sample efficiency and Pareto frontier quality?

---

## Evaluation Protocol

**Train-then-freeze** with ground-truth rewards:

1. BanditGPT trains on the dev-train split (80% of the online-learn pool)
2. Router is frozen after training (greedy exploitation)
3. Dev-val (20%) used for early stopping and Pareto-frontier model selection
4. Holdout set reserved for final evaluation
5. Lambda swept over 33 values; all results averaged over 20 seeds

Hyperparameters loaded from Appendix H (per-expert dev-val tuning).

---

## Directory Structure

```
04_figure/
├── run_warmup_ablation.py      # Main experiment (imports shared utils from 03_figure)
├── plot_results.py             # Generate Figure 4 (two-panel: Pareto + learning curve)
├── figure4_caption.tex         # LaTeX caption
├── results_discussion.tex      # LaTeX results discussion
├── README.md                   # This file
└── results/
    ├── warmup_ablation_results.json   # Produced by run_warmup_ablation.py
    └── figure4_warmup_ablation.png    # Produced by plot_results.py
```

---

## Quick Start

```bash
cd experiments/04_figure/

# Run the warmup ablation experiment
python run_warmup_ablation.py

# Generate the two-panel figure
python plot_results.py
```

---

## Experiment Details

### Outputs

**(a) Pareto Frontier** — BanditGPT (warmup) vs Tabula Rasa cost--quality frontiers, with supervised baselines as reference points and bootstrap CI for Pareto AUC difference.

**(b) Learning Curve** — Holdout reward vs online training steps, demonstrating:
- Warmup priors deliver supervised-baseline-quality routing from step 0
- Tabula rasa starts from scratch and needs many interactions to converge
- The "step-0 advantage" quantifies the value of offline-to-online transfer

### Sample Efficiency Metrics

- **Steps-to-match**: number of online steps to reach each supervised baseline's quality
- **Speedup**: ratio of tabula rasa steps to warmup steps for matching a threshold
- **AUC advantage**: area under the learning curve difference

### Shared Code

Imports shared evaluation functions from [`03_figure/run_prequential.py`](../03_figure/run_prequential.py):
- Data loading, embedding, and splitting
- Baseline evaluation (oracle, static, random, UCB1)
- Bandit training/evaluation (Pareto sweep, learning curve)
- Pareto analysis (bootstrap CI, AUC)

Supervised baselines from [`utils/supervised_baselines.py`](../utils/supervised_baselines.py).

---

## Related Experiments

| Scenario | Experiment |
|----------|-----------|
| BanditGPT vs supervised baselines | [Figure 3](../03_figure/) |
| Hyperparameter tuning (consumed here) | [Appendix H](../appendix/H_alpha_neff_ablation/) |

---

**Last Updated**: March 2026
