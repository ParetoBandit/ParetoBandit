# BanditGPT

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Tests](https://img.shields.io/badge/tests-135%20passed-brightgreen.svg)](#testing)

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
- **Probabilistic Mixture Model**: Continuous difficulty scoring eliminates utility cliffs from hard boolean gates
- **Production-Ready Concurrency**: Snapshot-Swap pattern prevents routing stalls during updates

### High-QPS Deployments: Lock Contention Fix

BanditGPT implements a **Snapshot-Swap pattern** in the LinUCB update logic to prevent lock contention in high-QPS environments:

**The Problem**: Matrix inversions (O(d³), ~50ms) previously held the thread lock, blocking all routing calls. With 10 concurrent stale updates, this created a 500ms routing stall.

**The Solution**: Three-phase update process:
1. **Snapshot** (~0.1ms): Brief lock to copy state
2. **Compute** (~50ms): Heavy math **without** lock
3. **Swap** (~0.1ms): Atomic commit of results

**Impact**:
- Lock hold time: **~50ms → ~0.2ms** (250× improvement)
- Routing proceeds in parallel during expensive O(d³) inversions
- P99 latency remains flat even during "Thundering Herd" update spikes

```python
# The router automatically handles concurrent updates efficiently
# No configuration needed - it just works!
router.route(prompt)  # Never blocks on updates
```

See [tests/test_lock_contention.py](tests/test_lock_contention.py) for concurrency tests.

### Adaptive Learning: Zombie Priors Fix

BanditGPT also addresses the **"Zombie Priors"** problem where overly stiff initial beliefs prevent the router from adapting to model drift (e.g., timeouts, quality degradation, API changes).

**The Problem**: Artificially inflating prior strength (e.g., N_structure=250) makes the router "deaf" to new evidence. With high prior weight, the router needs ~300 requests to overcome initial beliefs, failing to react to late-stage model drift.

**The Solution**: Use natural weighting in procedural warmup:
```python
# In warmup: use router.bandit.update(..., weight=1.0)
# Not: direct matrix manipulation with inflated weights
```

**Impact**:
- **100 synthetic samples** → natural magnitude ~100 in A matrix
- **1 real observation** → adds 1.0 to matrix
- **Ratio 1:100**: Stable (won't flap on 1-2 errors) + Plastic (reacts to 5-10 errors)
- Router now adapts to drift while maintaining stability

**Validated Results**: Pareto experiments show 98.8% quality maintained while enabling natural adaptation.

### Self-Healing PCA: JIT Calibration

BanditGPT prevents production outages from missing or mismatched PCA artifacts through **Just-In-Time (JIT) Calibration**:

**The Problem**: Static binary artifacts (.joblib) create fragile dependencies:
- **Missing artifact** → Crash loop
- **Dimension mismatch** (encoder upgrade) → Silent drift or crash
- **No auto-recovery** → Manual intervention required

**The Solution**: Automatic PCA validation and training on startup:

```python
# Router automatically handles PCA artifacts
router = BanditRouter(model_registry, pca_path="data/pca_32.joblib")
# If missing or mismatched: auto-trains in ~2s using synthetic data
# If valid: loads from disk
```

**How It Works**:
1. **Validate**: Check PCA dimensions match current encoder
2. **JIT Train**: If invalid, generate 1000 synthetic prompts → train PCA (~2s)
3. **Persist**: Save for next startup (cache-aside pattern)
4. **Verify**: Log explained variance, warn if < 60%

**Benefits**:
- **Zero Downtime**: Starts successfully even without artifacts
- **Version Safety**: Handles encoder upgrades automatically
- **Observability**: Logs variance capture for monitoring

See [tests/test_self_healing_pca.py](tests/test_self_healing_pca.py) for validation tests.

### Durable Context Store: Long-Delayed Feedback

BanditGPT fixes the **"Feedback Horizon Fallacy"** where long-delayed feedback (days/weeks) is lost after router restarts:

**The Problem**: Human feedback (RLHF) often arrives hours/days after routing. If router restarts between `route()` and `process_feedback()`, in-memory context is lost and feedback is dropped.

**The Solution**: SQLite-backed context persistence (production default):

```python
# Router automatically persists contexts to SQLite
router.route(prompt)  # Context saved to disk immediately

# Days later, after multiple restarts...
router.process_feedback(request_id, reward)  # Still works!
```

**How It Works**:
1. **Immediate Persistence**: Context saved to SQLite after every `route()` call (~0.1ms)
2. **Fallback on Feedback**: If not in memory, retrieve from disk (~0.05ms)
3. **WAL Mode**: Concurrent reads/writes (50k+ inserts/sec)
4. **Auto-Cleanup**: TTL-based pruning (default: 7 days)

**Benefits**:
- **Zero Data Loss**: Feedback works weeks later, even after restarts
- **High Performance**: WAL mode prevents routing stalls
- **Bounded Storage**: Automatic TTL pruning keeps DB small
- **Monitoring**: `stats()` method for observability

See [tests/test_durable_context_store.py](tests/test_durable_context_store.py) for validation tests.

### Tiered Safety: Async Toxicity Auditing

BanditGPT uses **Optimistic Safety with Asynchronous Audit** to prevent toxicity scanning from destroying P99 latency:

**The Problem**: Heavy ML-based toxicity scanners (e.g., llm-guard) add 100-300ms to every request in the synchronous hot path. For fast models like Claude Haiku (300ms response time), this doubles latency and violates SLAs.

**The Solution**: Two-tier defense system:

```python
# Tier 1: Fast heuristic in hot path (<1ms)
toxicity_score = router._fast_toxicity_heuristic(prompt)  # Regex-based

# Tier 2: Heavy ML scanner in background (async)
router.audit_queue.put((request_id, prompt, model))  # Non-blocking
```

**How It Works**:

**Tier 1 (Synchronous Hot Path)**:
- Fast regex-based pattern matching (<1ms)
- Catches obvious violations (violence, hate, explicit, security threats)
- Provides toxicity score feature for LinUCB bandit
- Blocks egregious content immediately

**Tier 2 (Async Background)**:
- Heavy ML scanner (llm-guard) runs in worker thread
- Compliance logging and detailed analysis
- Retroactive bandit correction (negative rewards for violations)
- User reputation tracking

**Performance Impact** (Validated with 100 real prompts):
- **Fast Heuristic**: P99 = 0.005ms (20,000-60,000x faster than ML scanner!)
- **Router Latency**: Mean 29.4ms (vs 130-330ms with synchronous ML scanner)
- **Zero Hot Path Blocking**: Audit happens post-flight

**Benefits**:
- **Latency-Aware**: Removes 100-300ms tax from every request
- **Statistically Sound**: Bandit gets toxicity feature via fast proxy
- **Governance**: Maintains safety via async audit + retroactive correction
- **Production-Ready**: Queue-based backpressure handling

See [test_tiered_safety.py](test_tiered_safety.py) for real data validation.

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

### Hybrid Router (Bandit-Guided Cascade)

For maximum accuracy with automatic fallback:

```python
from banditgpt import HybridRouter

# Create hybrid router with cascade_rate (λ)
hybrid = HybridRouter.create(
    model_registry=registry,
    fallback_model="openai/gpt-4o",
    cascade_rate=0.5,  # Verify ~50% of predictions
)

# Route with automatic cascade
result = hybrid.route_with_cascade(
    prompt="Write SQL to get all active users",
    generate_fn=lambda m, p: call_llm(m, p),
    verify_fn=lambda r: validate_sql(r),  # Optional verification
)

print(f"Model: {result['model_used']}, Mode: {result['mode']}")
# Mode: "single_shot" (fast) or "cascade" (accurate)
```

**Why Hybrid?** FrugalGPT's cascade is limited to 2-3 models (O(N) latency). Our Hybrid uses O(1) bandit selection over 80+ models, then cascades only when uncertain.

| Feature | FrugalGPT | HybridRouter |
|---------|-----------|--------------|
| Model Pool | 2-3 models | **80+ models** |
| Selection | Hardcoded | Context-aware |
| Latency | O(N) | **O(1)** |
| Adaptation | None | Online learning |

---

## The 3-Stage Routing Funnel

BanditGPT implements a **Constraint-Aware Architecture** where routing happens in three distinct phases:

```
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 1: HARD FILTERING (SLA Compliance)                       │
│  ─────────────────────────────────────────                      │
│  Filter out models that violate business constraints:           │
│    • min_quality=70  → Remove models with benchmark < 70%       │
│    • max_cost=1.00   → Remove models costing > $1.00/1k         │
│    • max_latency=2.0 → Remove models slower than 2 seconds      │
│                                                                 │
│  Result: Candidate pool shrinks from 80+ to ~10-20 "legal" models│
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 2: BANDIT SELECTION (Expertise)                          │
│  ─────────────────────────────────────                          │
│  Pick the best model from the filtered pool using learned prior │
│    • Contextual: "This prompt is about Python → pick CodeLlama" │
│    • O(1) lookup via pre-computed covariance matrix             │
│                                                                 │
│  Result: Single best model selected                             │
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 3: CASCADE DECISION (Lambda Tuning)                      │
│  ─────────────────────────────────────────                      │
│  Apply cascade_rate (λ) to decide if verification is needed     │
│    • λ=0.0: Never cascade (Standard Mode)                       │
│    • λ=0.5: Cascade ~50% of predictions                         │
│    • λ=1.0: Always cascade (Max Accuracy)                       │
│                                                                 │
│  Result: Execute single_shot OR cascade with fallback           │
└─────────────────────────────────────────────────────────────────┘
```

### The Business Knobs

| Knob | What It Does | Mechanism | Value |
|------|--------------|-----------|-------|
| **min_quality** | Sets a "Competence Floor" | Masks models with benchmark scores below threshold (e.g., `math_500 < 50%`) | Prevents "cheap but dumb" routing |
| **max_cost** | Sets a "Budget Cap" | Masks models costing more than limit (e.g., `> $1.00/1k`) | Strict FinOps control |
| **max_latency** | Sets a "Speed Limit" | Masks models slower than SLA (e.g., `> 2.0s`) | Critical for real-time apps |
| **cascade_rate** (λ) | Quality/Cost Slider | Controls % of predictions that get verified | Trades cost for reliability |

```python
# Example: Budget-constrained routing with quality floor
model, log, mode = router.route(
    "Solve this calculus integral",
    min_quality=70,    # Only models with benchmark ≥ 70%
    max_cost=1.00,     # Only models costing ≤ $1.00/1k
    cascade_rate=0.3,  # Verify 30% of predictions
)
```

### Unified Architecture: Standard Mode = λ=0

> *"Rather than maintaining separate codepaths, BanditGPT implements a unified routing logic where Standard Mode is simply the special case of λ=0. This ensures that critical safety features—such as hard budget constraints (`max_cost`) and benchmark quality floors (`min_quality`)—are universally applied to all queries, regardless of the verification strategy selected."*

| Mode | cascade_rate | Phase 1 (Filters) | Phase 2 (Bandit) | Phase 3 (Cascade) | Speed |
|------|-------------|-------------------|------------------|-------------------|-------|
| **Standard** | 0.0 | ✅ Applied | ✅ Applied | ❌ Skipped | O(1) |
| **Hybrid** | 0.3 | ✅ Applied | ✅ Applied | ✅ ~30% verified | O(1) + verify |
| **Max Accuracy** | 1.0 | ✅ Applied | ✅ Applied | ✅ Always | O(1) + verify |

```python
# Standard Mode: Pure O(1) with constraints
model, log, mode = router.route(
    prompt,
    min_quality=70,
    max_cost=1.00,
    cascade_rate=0.0,  # Standard Mode (default)
)
# mode = "single_shot" (always)

# Hybrid Mode: Same constraints + verification
model, log, mode = router.route(
    prompt,
    min_quality=70,
    max_cost=1.00,
    cascade_rate=0.5,  # Verify ~50%
)
# mode = "single_shot" or "cascade"
```

**Why This Matters**: You aren't just getting a "better bandit" — you're getting an **Enterprise-Ready Routing System** that respects real-world engineering constraints (budget, latency, quality floors) while still enabling adaptive, context-aware model selection.

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

### Two-Knob Prior Scaling

BanditGPT implements a **Two-Knob Framework** for independent control of prior strength:

| Knob | Parameter | What It Controls | Default | Effect |
|------|-----------|------------------|---------|--------|
| **Knob 1: Structural Stiffness** | `prior_structure_n_effective` | Covariance matrix strength (how confident we are in feature correlations) | `None` (infinite) | Higher = less exploration |
| **Knob 2: Belief Strength** | `prior_n_effective` | Mean vector strength (how confident we are in model quality scores) | `20.0` | Higher = stronger priors |

```python
# Default: Infinite structural stiffness + moderate belief strength
router = BanditRouter.create(
    priors="benchmark",
    prior_structure_n_effective=None,  # Infinite stiffness (unscaled covariance)
    prior_n_effective=20.0,            # Moderate belief strength
)

# Custom: Reduce structural stiffness while keeping belief strength
router = BanditRouter.create(
    priors="benchmark",
    prior_structure_n_effective=1000.0,  # Moderate structural stiffness
    prior_n_effective=20.0,              # Moderate belief strength
)

# Ablation: Zero beliefs, full structure (test covariance manifold only)
router = BanditRouter.create(
    priors="benchmark",
    prior_structure_n_effective=None,  # Full structural stiffness
    prior_n_effective=0.0,             # Zero belief priors
)
```

**The Math:**
- `init_scale = prior_structure_n_effective / N_offline` (scales covariance matrix `A`)
- `belief_scale = prior_n_effective / N_offline` (scales mean vectors `b`)
- With `N_offline ≈ 21,719` samples in the prior
- `prior_structure_n_effective=None` → `init_scale=1.0` (infinite stiffness, full covariance strength)

**When to Use:**
- **Production (Default)**: Use infinite stiffness for maximum zero-shot performance
- **Ablation Studies**: Vary knobs independently to isolate structural vs. belief contributions
- **Custom Domains**: Reduce structural stiffness if your domain differs significantly from training data

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
- Determinism: routing top-k selection is stable (ties broken by original order) to reduce run-to-run drift in benchmarks; regret/latency may still vary slightly due to timing noise.
- Security/IO: priors are loaded with `allow_pickle=False` and use fixed-width arrays to avoid unsafe object loading.

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
# Run all tests (135 tests, ~2 min)
python -m pytest tests/ -v

# Run integration tests only
python -m pytest tests/test_integration.py -v

# Run lock contention tests
python -m pytest tests/test_lock_contention.py -v
```

| Test Suite | Tests | Coverage |
|------------|-------|----------|
| Integration (end-to-end) | 30 | Router workflow, feedback, persistence |
| Feedback Loop | 39 | Reward processing, bandit updates |
| Prior Management | 26 | Load/save priors, dynamic models |
| Optimization Profiles | 32 | Cost/quality trade-offs |
| **Lock Contention** | **8** | **Snapshot-Swap concurrency, thread safety** |

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
