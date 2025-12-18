# LLM Jury

**A Local-First, Adaptive Router for Intelligent LLM Model Selection**

> *"Others build Maps (static benchmarks of the territory). We build a Compass (a tool that figures out where YOU are right now)."*

---

## The Problem

Current LLM routers are either:

- **(a) Static Classifiers** (e.g., RouteLLM) — which fail to adapt to user-specific data, or
- **(b) Online Bandits** — which suffer from a prohibitive "Cold Start" phase (high cost/regret) before they become useful.

## Our Solution

We propose a **Density-Based Warm-Start Framework** that compresses the latent performance of 80+ models into a lightweight **(<1MB)** covariance matrix. This enables:

- ✅ **Zero-Shot routing performance on Day 1**
- ✅ **Plasticity to adapt to local distribution shifts**

---

## Why LLM Jury?

Most existing solutions fall into two traps: they are either **Static Classifiers** (they don't learn from your specific traffic) or **SaaS APIs** (they own the intelligence, not you).

LLM Jury fills the gap by being **lightweight**, **offline**, and **self-improving**.

### The Competitive Landscape

| Type | Examples | How They Work | How We Differ |
|------|----------|---------------|---------------|
| **Static Routers** | RouteLLM, HybridLLM | Train a BERT classifier on public datasets to predict "which model is better" | We **learn locally**. If RouteLLM thinks Mistral is bad at Rust, it always will. Our router discovers *your* Rust usage is fine on Mistral. |
| **Cascades** | FrugalGPT (Stanford) | Try cheap model → if it fails → try expensive model | We're **faster**. Cascades double latency (wait for failure). We predict failure *before* it happens. |
| **SaaS Routers** | Not Diamond, Martian, Unify | Send prompt to their API → they route → they call OpenAI | We're **private & free**. No extra hop, no middleman fee. Runs locally in microseconds. You own the weights. |
| **Naive Bandits** | A/B Testing Tools | Check global win rates: "GPT-4 wins 80%" | We use **context**. Naive bandits route "Hello" to GPT-4 because it's a "winner." We route it to Haiku because the context is simple. |

### Our 4 Key Differentiators

| # | Differentiator | Others | Us |
|---|----------------|--------|-----|
| 1 | **Shippable Brain** | Require 1,000 logs before working, OR download 500MB classifier | Ship <1MB priors. Day 1 intelligence. |
| 2 | **Local Adaptation** | Optimize for "Average User" | Optimize for *your* user. Discover Claude is good at your proprietary query language. |
| 3 | **Business Formula** | Vague "Quality vs. Cost" slider | Transparent: `U = Q - (w_c × Cost) - (w_l × Latency)` |
| 4 | **Reasoning Fallback** | Simple BERT reward model | Tiered Grader escalates to reasoning LLMs for math/code/logic |

---

## Features

- **Smart Routing**: Learns which models excel at which types of prompts
- **Real-Time Learning**: Microsecond updates via rank-one matrix operations (no retraining)
- **Cost-Aware**: Balances quality against cost and latency using configurable profiles
- **Tiered Grading**: Soft grader (local) + hard verifier (LLM-as-Judge) for accuracy
- **Warm Start**: Ships with pre-computed priors from 500 archetype prompts

- **Smart Routing**: Learns which models excel at which types of prompts
- **Real-Time Learning**: Microsecond updates via rank-one matrix operations (no retraining)
- **Cost-Aware**: Balances quality against cost and latency using configurable profiles
- **Tiered Grading**: Soft grader (local) + hard verifier (LLM-as-Judge) for accuracy
- **Warm Start**: Ships with pre-computed priors from 500 archetype prompts

## Quick Start

```python
from llm_jury.async_bandit import BanditRouter

# Create router with automatic prior loading
router = BanditRouter.create(model_registry, priors="merged")

# Route a prompt (uses learned priors + uncertainty exploration)
model_id, log = router.route(
    "Write a Python function to parse JSON",
    profile="balanced",      # Cost/quality trade-off
    exploration="safe",      # Risk appetite
)

# After getting feedback, update the bandit
router.bandit.update(model_id, log.context_vector, reward=0.95)
```

## How It Works

### The Casino Analogy

Imagine 80 slot machines (models) in a casino. You have limited budget (prompts) and want to find the best payout (quality). The **bandit** balances:

- **Exploitation**: Keep using the machine that paid well yesterday
- **Exploration**: Try unknown machines that might pay even better

### The Math: Prompt → Model Selection

```
1. EMBED:  Prompt → x (384-dim vector via Sentence Transformer)
2. LOOKUP: θ = A⁻¹ @ b (model's learned weight vector)
3. SCORE:  quality = θ·x + α·√(x'A⁻¹x)  (mean + exploration bonus)
4. DECIDE: utility = quality - λ_cost·Cost - λ_latency·Latency
```

The model with highest **utility** wins.

### Real-Time Learning (Rank-One Update)

When feedback arrives, we don't retrain a neural network. Instead:

```python
A_new = A_old + x·x'    # "I've seen this prompt type" (uncertainty ↓)
b_new = b_old + r·x     # "This worked well/poorly" (push θ toward/away)
```

This takes **microseconds**, not hours.

## Configuration

### Optimization Profiles

Control cost/quality trade-offs with named presets:

| Profile | λ_cost | λ_latency | Use Case |
|---------|--------|-----------|----------|
| `quality_first` | 0.1 | 0.05 | Best quality, ignore cost |
| `balanced` | 10.0 | 0.10 | Reasonable trade-off |
| `cost_saver` | 50.0 | 0.20 | Aggressive cost cutting |
| `low_latency` | 1.0 | 0.50 | Speed over cost |

```python
router.route(prompt, profile="cost_saver")
```

### Exploration Rate

Control risk appetite (how often to try unproven models):

| Setting | Alpha | Use Case |
|---------|-------|----------|
| `static` | 0.0 | Zero risk (fintech/production) |
| `safe` | 0.1 | **Default** - minimal exploration |
| `balanced` | 0.5 | Standard bandit behavior |
| `aggressive` | 2.0 | Day-1 calibration / shadow mode |

```python
router.route(prompt, exploration="aggressive")  # Day 1: learn fast
router.route(prompt, exploration="static")      # Production: zero risk
```

## CLI

```bash
# Get recommendations for a prompt
python -m llm_jury.async_bandit.cli recommend \
    --prompt "Explain quantum computing" \
    --profile balanced \
    --exploration safe \
    --top-k 5
```

## Architecture

See [docs/architecture.md](docs/architecture.md) for detailed explanation of:
- How routing works (prompt → prediction)
- Why it's called a "bandit" (exploration vs exploitation)
- The learning loop (rank-one updates)
- Optimization profiles and exploration rates

## Prior Management

The router uses two prior locations:

| Location | Path | Purpose |
|----------|------|---------|
| **Bundled** | `<package>/data/priors/shippable_priors.npz` | Library defaults (read-only) |
| **User** | `~/.llm_jury/priors/user_priors.npz` | Your learned updates |

Add new models dynamically:

```python
router.add_model("openai/gpt-5", clone_from="openai/gpt-4o")
```

## Installation

```bash
pip install llm-jury
```

Or from source:

```bash
git clone https://github.com/atabernermiller/llm_jury.git
cd llm_jury
pip install -e .
```

## Requirements

- Python 3.10+
- sentence-transformers
- transformers
- numpy
- torch

## License

MIT
