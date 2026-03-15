# Appendix: Cold-Start vs Warmup Prior Regret

Demonstrates the value of warmup priors by comparing cumulative
regret of BanditGPT (warmup) vs Tabula Rasa (cold start) on the
K=3 portfolio under stationary conditions.

## Setup

- **Arms**: Llama-3.1-8B, Mistral-Large-2512, Gemini-2.5-Pro
- **Data**: val.jsonl (1,785 prompts), 20 seeds
- **Hyperparameters**: alpha=1.0, n_eff=5000, disjoint LinUCB, gamma=1.0
- **Conditions**: BanditGPT (warmup), Tabula Rasa, Random

## Run

```bash
python experiments/appendix/warmup_ablation/run_warmup_ablation.py
python experiments/appendix/warmup_ablation/generate_figure.py
```

## Key Results

| Condition | Regret ± SE | Regret@200 | Mean Reward | Oracle Agr. |
|-----------|-------------|------------|-------------|-------------|
| BanditGPT (warmup) | **58.0 ± 0.4** | **5.6 ± 0.2** | **0.930** | **0.662** |
| Tabula Rasa | 90.1 ± 0.6 | 14.1 ± 0.4 | 0.913 | 0.644 |
| Random | 145.3 ± 1.2 | 16.3 ± 0.4 | 0.881 | 0.528 |

Warmup priors reduce total regret by 36% and early regret by 60%.
