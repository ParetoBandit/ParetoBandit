# Figure 12: Efficiency of Learning - Time-to-Value Analysis

## Fast Adaptation Narrative

This plot demonstrates BanditGPT's rapid learning capability compared to static baselines.

### Efficiency of Learning
We evaluate the sample efficiency of BanditGPT against static baselines. Starting from a cold state (with benchmark priors), BanditGPT rapidly learns the optimal routing policy.

### Key Findings

**Crossover Point**: BanditGPT surpasses the performance of strong static baselines (RouteLLM, FrugalGPT) after just **100 requests**.

**Convergence**: By **1,000 requests**, BanditGPT saturates at a quality of **~0.96**, maintaining a consistent performance gap over the baselines (**~0.927**) for the remainder of the deployment.

**Stability**: The curve remains stable in the final phase (1000-2000 requests), demonstrating robust policy convergence.

## Methodology

- **Data**: 2,000 queries from RouteLLM battle dataset (shuffled to remove ordering artifacts)
- **Smoothing**: Exponential Moving Average (EMA) with span=300
- **Runs**: 5 runs per router for statistical stability
- **Reward**: Based on hallucination rates (1.0 - hallucination/100)
- **BanditGPT Config**: 
  - Priors: `benchmark` (HLE-based initialization)
  - Exploration: `balanced` (α=1.0)
  - Forgetting Factor: 1.0

## Baselines

- **BanditGPT (Ours)**: Online learning with LinUCB + benchmark priors
- **BaRP Oracle (Target)**: Always routes to strongest model (theoretical maximum)
- **RouteLLM (Static)**: Pre-trained Matrix Factorization router
- **FrugalGPT (Cascade)**: Cascade-based routing with learned scorer
