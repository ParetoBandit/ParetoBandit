# BanditGPT

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Tests](https://img.shields.io/badge/tests-127%20passed-brightgreen.svg)](#testing)

**A Local-First, Adaptive Router for Intelligent LLM Model Selection**

> *"Others build Maps (static benchmarks). We build a Compass (learns where YOU are)."*

📄 **Paper**: *Density-Based Warm-Start for Adaptive LLM Routing* (KDD 2025) — See [`kdd_paper/`](kdd_paper/README.md)

---

## The Problem

Current LLM routers are either:

- **(a) Static Classifiers** (e.g., RouteLLM) — fail to adapt to your specific data
- **(b) Online Bandits** — suffer from prohibitive "Cold Start" (high cost/regret before becoming useful)

## Our Solution

We propose a **Density-Based Warm-Start Framework** that compresses the latent performance of 80+ models into a lightweight **(<1MB)** covariance matrix:

| Metric | Cold Start | Warm Start (Ours) |
|--------|------------|-------------------|
| Day-1 Regret | High | **63.6% lower** |
| Adaptation | Slow | Immediate |
| File Size | N/A | <1 MB |

**Key Results:**
- ✅ **Zero-Shot routing performance on Day 1**
- ✅ **Plasticity to adapt to local distribution shifts**
- ✅ **97% cost reduction** vs always using GPT-4o

---

## Why BanditGPT?

Most existing solutions fall into two traps: they are either **Static Classifiers** (they don't learn from your specific traffic) or **SaaS APIs** (they own the intelligence, not you).

BanditGPT fills the gap by being **lightweight**, **offline**, and **self-improving**.

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
- **Warm Start**: Ships with expert-distilled priors for 62% regret reduction on Day 1

## Quick Start

```python
from banditgpt.core import BanditRouter

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
python -m banditgpt.core.cli recommend \
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
| **Bundled** | `<package>/data/priors/expert_priors.npz` | Expert-distilled defaults (read-only) |
| **User** | `~/.banditgpt/priors/user_priors.npz` | Your learned updates |

Add new models dynamically:

```python
router.add_model("openai/gpt-5", clone_from="openai/gpt-4o")
```

### Why Expert Priors Work

The bundled priors are generated via **Expert Distillation**, not random exploration. This is why the warm-start achieves 62% regret reduction:

| Prior Type | What It Encodes | Effect of Confidence Boost |
|------------|-----------------|---------------------------|
| **Uniform (Old)** | "Everything is average" | Boosting makes bandit *stubborn* — ignores good options |
| **Expert (New)** | "Model A wins for code prompts" | Boosting makes bandit *confident* — exploits correct answer |

**The Math**: In Bayesian terms, you are asserting that your Prior Belief is highly informative. This is valid *only if the prior is actually good*. Since we use Expert Distillation (oracle picks the optimal model 80% of the time during offline training), the prior encodes "the right answer" — so boosting it is the mathematically correct action.

The library applies a default `prior_strength=50.0` (λ_boost), which tells the bandit: *"Trust these expert priors as if they came from 50× more observations."*

### Priors integrity and migration notes

- Bundled priors are checksummed via `banditgpt/data/priors/manifest.json`. The library validates packaged priors on load; if a file is missing or corrupted, reinstall or restore it from git.
- Post-install sanity check: `python -m banditgpt.core.cli verify-priors` (or `banditgpt verify-priors` if exposed via entrypoint).
- To restore from git if needed: `git show <ref>:banditgpt/data/priors/shippable_priors.npz > shippable_priors.npz` (same for `expert_priors.npz`).

### Getting priors from git (fallback)

Priors ship inside the wheel. If you ever need to re-fetch them directly from git (e.g., corruption or custom build), you can pull them from the repo:

```bash
# From a clone or specific commit
git show <ref>:banditgpt/data/priors/shippable_priors.npz > shippable_priors.npz
git show <ref>:banditgpt/data/priors/expert_priors.npz > expert_priors.npz

# Or via GitHub raw (replace org/repo/ref as needed)
curl -L https://raw.githubusercontent.com/<org>/<repo>/<ref>/banditgpt/data/priors/shippable_priors.npz -o shippable_priors.npz
curl -L https://raw.githubusercontent.com/<org>/<repo>/<ref>/banditgpt/data/priors/expert_priors.npz -o expert_priors.npz
```

## Installation

```bash
pip install banditgpt
```

Or from source:

```bash
git clone https://github.com/atabernermiller/banditgpt.git
cd banditgpt
pip install -e .
```

### Optional Dependencies

```bash
# Full functionality (LLM-as-judge grading via OpenRouter)
pip install banditgpt[full]

# Experiments (reproduce KDD paper figures)
pip install banditgpt[experiments]

# Development
pip install banditgpt[dev]
```

## Requirements

**Core** (installed automatically):
- Python 3.10+
- numpy, torch, pandas
- sentence-transformers
- transformers

**Optional**:
- `openai` — for LLM-as-judge grading (OpenRouterTeacherVerifier)
- `python-dotenv` — for loading API keys from `.env` files
- `matplotlib` — for experiment visualizations

---

## Troubleshooting

- Missing priors / checksum failure: run `banditgpt verify-priors` (or `python -m banditgpt.core.cli verify-priors`). If corruption is detected, reinstall or restore from git (`git show <ref>:banditgpt/data/priors/expert_priors.npz > expert_priors.npz`).
- `sentence-transformers` / `transformers` missing: install core extras if you see import errors on `BanditRouter` (`pip install sentence-transformers transformers`).
- OpenRouter key required: grading via `OpenRouterTeacherVerifier` needs `OPENROUTER_API_KEY` set; without it, teacher verification is skipped and a clear error is raised.
- Enable debug logs: set `PYTHONLOGGING=DEBUG` (or configure the `logging` module) to see priors validation and router init resolution details when diagnosing installs.

---

## Testing

```bash
# Run all tests (127 tests, ~2 min)
python -m pytest tests/ -v

# Run integration tests only
python -m pytest tests/test_integration.py -v
```

| Test Suite | Tests | Coverage |
|------------|-------|----------|
| Integration (end-to-end) | 30 | Router workflow, feedback, persistence |
| Feedback Loop | 39 | Reward processing, bandit updates |
| Prior Management | 26 | Load/save priors, dynamic models |
| Optimization Profiles | 32 | Cost/quality trade-offs |

See [`tests/README.md`](tests/README.md) for details.

---

## Project Structure

```
banditgpt/
├── core/                    # Main router implementation
│   ├── bandit_router.py     # BanditRouter, LinUCB policies
│   ├── judge.py             # PriorManager, Judge abstraction
│   └── tiered_grader.py     # Quality grading (soft + hard)
├── data/
│   └── priors/              # Bundled expert priors
└── __init__.py

experiments/                  # KDD paper experiments
├── run_rq1.py               # Warm-start advantage
├── run_rq2.py               # Specialist discovery
└── run_rq3.py               # Cost-quality Pareto

kdd_paper/                    # Camera-ready figures & tables
├── figures/                 # PDF/PNG plots
├── tables/                  # Markdown tables with LaTeX
└── README.md                # Paper artifact guide

tests/                        # 127 unit & integration tests
```

---

## KDD Paper (2025)

This repository accompanies our KDD 2025 paper: **"Density-Based Warm-Start for Adaptive LLM Routing"**.

### Key Findings

| Research Question | Key Result |
|-------------------|------------|
| **RQ1: Warm-Start** | 63.6% regret reduction vs cold-start |
| **RQ2: Plasticity** | Router discovers specialists (Nova ‖θ‖=3.66 > GPT-4o ‖θ‖=1.66) |
| **RQ3: Efficiency** | 97% cost reduction, +33.8% quality vs GPT-4o |

### Reproduce Results

```bash
python experiments/run_rq1.py  # Figure 1: Regret curve
python experiments/run_rq2.py  # Figure 3: Specialist landscape
python experiments/run_rq3.py  # Figure 4: Pareto frontier
```

See [`kdd_paper/README.md`](kdd_paper/README.md) for complete artifact guide.

---

## Acknowledgments

The bundled `expert_priors.npz` was generated using these open-source datasets:

| Dataset | License | Usage |
|---------|---------|-------|
| [LMSYS Chatbot Arena](https://huggingface.co/datasets/lmsys/chatbot_arena_conversations) | CC-BY-4.0 | 497 archetype prompts (K-means clustering) |
| [LMSYS Arena Preferences](https://huggingface.co/datasets/lmsys/lmsys-arena-human-preference-55k) | CC-BY-4.0 | Quality model training (preference labels) |
| [NVIDIA HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2) | CC-BY-4.0 | Quality model training (multi-dim annotations) |

Model responses (81 models × 497 prompts) were generated via [OpenRouter](https://openrouter.ai/).

See [ACKNOWLEDGMENTS.md](ACKNOWLEDGMENTS.md) for full citations and data pipeline details.

---

## License

Apache License 2.0 — See [LICENSE](LICENSE) for details.

**Why Apache 2.0?** This is the industry standard for ML infrastructure (TensorFlow, PyTorch, Hugging Face). It includes an explicit **patent grant**, giving enterprise users safety to embed this router in commercial products.
