# Figure 7: Stability-Regret Frontier

![Figure 7](file:///Users/annette/repostitories/llm_jury/final_release/kdd_paper/figure_7/stability_frontier.png)

## Caption

**Figure 7: Stability-Regret Frontier.** A comparison of routing stability (System Churn) versus performance (Cumulative Regret) in a live production simulation.

- **Standard LinUCB (Gray X)**: Represents a reactive online learner (γ=0.9, no priors). It suffers from high regret (~12.6) due to the "Cold Start" problem and extreme churn (98.7%) as it reacts to noise.

- **BanditGPT (Green Curve)**: Represents our Prior-Informed architecture across a sweep of inertia settings (γ).

**The Win**: The annotated "Sweet Spot" (γ ≈ 0.98-0.99) achieves near-zero regret (0.4-0.7) and near-zero churn (0.3-2.3%), proving that combining Priors with high Inertia eliminates the "Curiosity Tax" inherent in standard bandit algorithms.

## Key Insights

### The "Cost of Learning" (Regret Gap)
The Gray X (Standard LinUCB) sits at a Cumulative Regret of ~12.6, while BanditGPT's optimal configuration (Green dot at bottom-left) achieves ~0.4. This **30x improvement** visualizes the benefit of our Priors—BanditGPT starts near the optimal solution, whereas the baseline must pay the "Curiosity Tax" to discover it through exploration.

### The "Stability Gap" (Churn)
Standard LinUCB exhibits **98.7% churn** (thrashes on nearly every request), while BanditGPT's stable configuration achieves **0.3% churn**. This proves our Inertia claim: proper forgetting factor tuning creates a stable system that doesn't overreact to noise.

### Sensitivity Analysis (Green Pareto Curve)
By sweeping γ from 1.0 to 0.80, we demonstrate that:
- **γ=1.00**: Most stable (0.3% churn, 0.40 regret) - optimal for production
- **γ=0.98-0.99**: Sweet spot balancing stability and adaptability
- **γ≤0.90**: Approaches Standard LinUCB behavior (high churn, high regret)

This highlights that the contribution isn't just "use a bandit" - it's identifying the precise **"Sweet Spot"** where Priors + Inertia create a Pareto-optimal solution.

## Methodology

### Dataset
- **Source**: BanditGPT test set
- **Size**: 300 prompts (sampled for computational efficiency)
- **Models**: 50 models from production registry

### Baselines

1. **Standard LinUCB (γ=0.9, no priors)**
   - Represents "reactive" online learning
   - Cold start: no prior knowledge
   - Aggressive forgetting to chase new signals
   - Result: 98.7% churn, 12.57 regret

2. **BanditGPT (γ sweep, with HLE priors)**
   - Prior-informed initialization
   - Inertia control via forgetting factor
   - Result: Pareto frontier dominates baseline

### Metrics

- **System Churn**: % of requests where model selection changed from previous request
- **Cumulative Regret**: Sum of (oracle_reward - actual_reward) across all requests
- **Oracle**: Best possible model (min hallucination rate)

## Scientific Validity

### Fair Comparison
The Standard LinUCB baseline uses:
- Same model registry as BanditGPT
- Same reward function (1 - hallucination_rate/100)
- Active learning (updates after each request)
- Realistic forgetting factor (γ=0.9) for adaptivity

This is the **correct baseline** for a stability analysis - it shows what happens when you optimize for adaptation without stability constraints.

### No "Static Oracle" Fallacy
Unlike Figure 9 (which correctly uses Oracle Proxies for safety analysis), Figure 7 compares against an **online learner** to show the stability-adaptability tradeoff. A static oracle would show (0% churn, 0 regret) which defeats the purpose of the analysis.

## Comparison to Prior Work

### Standard Bandits (e.g., LinUCB, Thompson Sampling)
- **Optimization**: Maximize immediate reward
- **Stability**: Not considered - chase every signal
- **Result**: High churn (98.7%), moderate regret (12.6)

### BanditGPT
- **Optimization**: Maximize reward + minimize churn
- **Stability**: Controlled via forgetting factor (γ)
- **Result**: Low churn (0.3%), low regret (0.4) - Pareto dominance

## Takeaways

1. **Priors Eliminate Curiosity Tax**: BanditGPT with HLE priors achieves 30x lower regret than cold-start baseline

2. **Inertia Prevents Thrashing**: γ=0.98-0.99 reduces churn from 98.7% to 0.3-2.3%

3. **Sweet Spot Exists**: Not all γ values are equal - the curve shows clear Pareto frontier

4. **Production-Ready**: The optimal configuration (γ=0.99) balances stability and adaptability for real-world deployment

This figure demonstrates that BanditGPT's contribution is not just using contextual bandits, but **systematically identifying the parameter regime** where online learning becomes stable enough for production use.
