# Appendix: Cold-Start vs Warmup Prior Regret

Demonstrates the value of warmup priors by comparing cumulative
regret of ParetoBandit (warmup) vs Tabula Rasa (cold start) on the
K=3 portfolio under stationary conditions.

## Setup

- **Arms**: Llama-3.1-8B, Mistral-Large-2512, Gemini-2.5-Pro
- **Data**: val.jsonl (1,785 prompts), 20 seeds
- **Hyperparameters (warmup)**: alpha=0.10, n_eff=1000, disjoint LinUCB, gamma=0.995
- **Hyperparameters (tabula rasa)**: alpha=0.10, n_eff=1, disjoint LinUCB, gamma=0.995
- **Conditions**: ParetoBandit (warmup), Tabula Rasa, Random

## Run

```bash
python experiments/appendix/warmup_ablation/run_warmup_ablation.py
python experiments/appendix/warmup_ablation/generate_figure.py
python experiments/appendix/warmup_ablation/generate_uncertainty_figure.py
```

## Key Results

| Condition | Regret ± SE | Regret@200 | Mean Reward | Oracle Agr. |
|-----------|-------------|------------|-------------|-------------|
| ParetoBandit (warmup) | **72.0 ± 0.7** | **5.9 ± 0.2** | **0.923** | **0.642** |
| Tabula Rasa | 79.4 ± 1.0 | 10.3 ± 0.9 | 0.919 | 0.632 |
| Random | 145.3 ± 1.2 | 16.3 ± 0.4 | 0.882 | 0.528 |

Both bandit conditions use their Experiment 04 optimal hyperparameters.
Warmup priors reduce total regret by 9% and early regret by 43%.
