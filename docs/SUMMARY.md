# BanditGPT: Technical Summary

> **A Data-Efficient, Contextual Bandit Framework for Multi-Objective LLM Routing**

---

## The Problem

Current LLM routers fall into two categories, both with critical limitations:

| Approach | Examples | Limitation |
|----------|----------|------------|
| **Static Classifiers** | RouteLLM, HybridLLM | Fail to adapt to user-specific data or distribution shifts |
| **Online Bandits** | Naive A/B Testing | Suffer from prohibitive "Cold Start" phase (high cost/regret) before becoming useful |

## Our Solution

We propose a **Density-Based Warm-Start Framework** that compresses the latent performance of 80+ models into a lightweight (<1MB) covariance matrix. This enables:

- **Zero-Shot routing performance on Day 1**
- **Plasticity to adapt to local distribution shifts**

---

## Key Technical Contributions

### 1. Archetype-Driven Priors

We demonstrate that the LLM "Intent Space" is **low-dimensional**. By clustering public instruction datasets into **500 Centroids** (validated by Zhou et al., LIMA), we generate dense, high-signal priors that map the latent performance of 80+ models.

**Impact:** Reduces the bandit's "Time-to-Convergence" from O(T) to **near-zero** for standard tasks.

### 2. Shared-Covariance Compression

We introduce a technique to compress the bandit's state by sharing the covariance matrix **A** across models while maintaining unique reward vectors **b**.

**Impact:** Reduces shippable state size by **98%** (from ~50MB to <1MB), enabling "Edge Intelligence" deployment.

### 3. Multi-Objective Scalarization

We formulate the routing problem not as maximizing reward, but as a **linear scalarization** of Quality, Cost, and Latency:

```
argmax_m [ μ̂_quality(x) + α·σ_quality(x) - λ_c·C_m - λ_l·L_m ]
```

Where:
- `μ̂_quality(x)` = Predicted quality (learned from feedback)
- `α·σ_quality(x)` = **Epistemic Uncertainty** (drives exploration of new models)
- `λ_c·C_m` = Cost penalty (user-configurable)
- `λ_l·L_m` = Latency penalty (user-configurable)

**Impact:** Successfully integrates exploration with strict adherence to production constraints.

### 4. Robustness to Drift

The system includes a **Hierarchical Grading Mechanism** (System 1 vs. System 2 judges):

| Judge Type | Implementation | Use Case |
|------------|----------------|----------|
| **System 1** (Fast) | Local DeBERTa classifier | Style, fluency, general quality |
| **System 2** (Slow) | Reasoning LLM (GPT-4o, Gemini) | Math, code, logic, constraints |

This generates synthetic ground truth, allowing the router to adapt to **Distribution Drift** in user queries without requiring manual human labeling.

---

## The Competitive Landscape

| Type | Examples | How They Work | How We Differ |
|------|----------|---------------|---------------|
| **Static Routers** | RouteLLM, HybridLLM | Train BERT on public datasets to predict "which model is better" | We **learn locally**. If RouteLLM thinks Mistral is bad at Rust, it always will. We discover your Rust usage is fine on Mistral. |
| **Cascades** | FrugalGPT | Try cheap model → if it fails → try expensive model | We're **faster**. Cascades double latency. We predict failure before it happens. |
| **SaaS Routers** | Not Diamond, Martian, Unify | Send prompt to their API → they route → they call OpenAI | We're **private & free**. No extra hop, no middleman fee, runs locally in microseconds. |
| **Naive Bandits** | A/B Testing | Check global win rates: "GPT-4 wins 80%" | We use **context**. Naive bandits route "Hello" to GPT-4. We route it to Haiku because the context is simple. |

---

## Our 4 Key Differentiators

### 1. The "Shippable Brain" (vs. Cold Start)

| Others | Us |
|--------|-----|
| Require 1,000 logs before router works, OR download 500MB classifier | Ship <1MB compressed covariance matrix |

**Result:** Day 1 Intelligence with zero download bloat and zero training time.

### 2. Local Adaptation (vs. The "Average" Trap)

| Others | Us |
|--------|-----|
| Optimize for the "Average User" | Optimize for the **Specific User** |

**Example:** If your user writes a proprietary query language (e.g., Kusto), standard benchmarks say all models fail. Our bandit discovers Claude-3-Haiku is weirdly good at it and shifts traffic there.

### 3. The "Business Formula" (vs. Opaque Magic)

| Others | Us |
|--------|-----|
| Vague slider: "Quality vs. Cost" | Transparent formula: `U = Q - (w_c × Cost) - (w_l × Latency)` |

**Result:** A CTO can say: "I will sacrifice exactly 2% quality to save $0.001." Routing becomes a financial instrument.

### 4. The "Reasoning" Fallback (vs. System 1 Grading)

| Others | Us |
|--------|-----|
| Simple Reward Model (BERT regressor) | Tiered Grader escalating to Reasoning Models (Gemini/o1) |

**Result:** Training data is mathematically sound, not "vibes-based."

---

## The Pitch

> *"Others build Maps (static benchmarks of the territory). We build a Compass (a tool that figures out where YOU are right now)."*

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER PROMPT                              │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SENTENCE TRANSFORMER                          │
│                    (all-MiniLM-L6-v2)                            │
│                         x ∈ ℝ³⁸⁴                                 │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                     CONTEXTUAL BANDIT                            │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  For each model m:                                         │  │
│  │    θ_m = A⁻¹ @ b_m           (learned weights)            │  │
│  │    μ = θ_m · x               (predicted quality)          │  │
│  │    σ = √(x' A⁻¹ x)           (uncertainty)                │  │
│  │    UCB = μ + α·σ             (optimism)                   │  │
│  │    U = UCB - λ_c·Cost - λ_l·Latency  (utility)           │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│                    SELECT: argmax_m U_m                          │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                      SELECTED MODEL                              │
│                    (e.g., claude-3-haiku)                        │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                     TIERED GRADER                                │
│  ┌─────────────────────┐    ┌─────────────────────┐            │
│  │   SYSTEM 1 (Fast)   │    │   SYSTEM 2 (Slow)   │            │
│  │   Local DeBERTa     │    │   LLM-as-a-Judge    │            │
│  │   Style/Fluency     │    │   Math/Code/Logic   │            │
│  └──────────┬──────────┘    └──────────┬──────────┘            │
│             │                          │                        │
│             └──────────┬───────────────┘                        │
│                        ▼                                        │
│                   reward ∈ [0, 1]                               │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RANK-ONE UPDATE                               │
│                                                                  │
│        A_new = A_old + x·x'     (reduce uncertainty)            │
│        b_new = b_old + r·x      (update reward direction)       │
│                                                                  │
│                    Time: O(d²) ≈ microseconds                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Citation

If you use this work, please cite:

```bibtex
@software{banditgpt,
  title = {BanditGPT: A Data-Efficient Contextual Bandit Framework for Multi-Objective LLM Routing},
  author = {Taberner-Miller, Annette},
  year = {2024},
  url = {https://github.com/atabernermiller/banditgpt}
}
```
