# BanditGPT

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Tests](https://img.shields.io/badge/tests-135%20passed-brightgreen.svg)](#testing)

**An adaptive, local-first router that learns which LLM works best for *your* prompts.**

---

## Overview

BanditGPT is an open-source contextual bandit framework for LLM routing. Instead of relying on static rules or pre-trained classifiers to choose between language models, it learns from your actual traffic which model performs best for each type of prompt — then adapts continuously as your usage patterns change.

**The core insight**: expensive models are not always better. On held-out LMSYS Arena prompts, roughly 14% of prompts are *actively worse* when routed to GPT-4-Turbo instead of Mixtral. A static router that equates "hard" with "needs GPT-4" will systematically over-provision the expensive model on these prompts — paying 43× more for 1.3% lower quality. BanditGPT discovers this preference structure online and routes accordingly.

**Key results** (on 750 held-out prompts, 20 independent trials):

| Metric | Value |
|--------|-------|
| Peak quality | 0.914 ± 0.003 (vs. RouteLLM: 0.883) |
| Gap closure to oracle | 70.0% (vs. RouteLLM: 46.2%) |
| Prompts to surpass RouteLLM | ~200, label-free (at moderate-to-high budgets) |
| Warm-start prior size | < 1 MB |
| Supports | Arbitrary model portfolios (K ≥ 2) |

📄 **Paper**: *Density-Based Warm-Start for Adaptive LLM Routing* — See [`paper/`](paper/README.md)

---

## Quick Start

```bash
pip install banditgpt
```

```python
from bandit_gpt import BanditRouter

# Create router with pre-trained priors (< 1 MB)
router = BanditRouter.create(model_registry, priors="warmup")

# Route a prompt — the router selects the best model for this context
model_id, log = router.route("Write a Python function to parse JSON")

# After observing quality, update the router (microsecond operation)
router.process_feedback(log.request_id, reward=0.95)
```

No labels, no retraining, no external API calls. The router runs locally and learns from its own routing outcomes.

---

## When Should You Use BanditGPT?

| Scenario | Recommended Approach | Why |
|----------|---------------------|-----|
| Need routing today, no deployment data | **RouteLLM** | Pre-trained classifier works out of the box |
| Prompt distribution shifts over time | **BanditGPT** | Adapts continuously; RouteLLM is frozen at training time |
| No labeled routing data for your domain | **BanditGPT** | Requires zero labels — learns from routing outcomes |
| Multiple models (K ≥ 3) at different price tiers | **BanditGPT** | Contextual bandit handles multi-model portfolios natively |
| Maximum quality is the priority | **BanditGPT** | After ~400 prompts, surpasses what 100k supervised pairs achieve |
| Simple cost-cutting on well-understood traffic | **RouteLLM** | Supervised pre-training is competitive at moderate budgets |

The 400-prompt crossover means a new deployment can surpass a pre-trained router within minutes of production traffic, with no human annotation.

---

## How It Works

### The Intuition

Imagine 80 slot machines in a casino, each with unknown and different payout rates. You have a limited budget and want to maximize your total payout. The **bandit algorithm** balances two goals:

- **Exploitation**: keep pulling the machine that paid well so far
- **Exploration**: try unknown machines that might pay even better

BanditGPT treats each LLM as a slot machine. But unlike a regular bandit, it uses **context** — the content of your prompt — to predict which model will perform best *for this specific request*. A coding prompt gets routed differently than a creative writing prompt, even if both could technically go to any model.

### The 3-Stage Routing Funnel

Every routing decision passes through three stages:

```
Stage 1: HARD FILTERING
├── Remove models violating constraints (budget cap, latency SLA, quality floor)
├── 80+ models → ~10-20 candidates
│
Stage 2: CONTEXTUAL BANDIT SELECTION
├── For each candidate, score: quality estimate + exploration bonus - cost penalty
├── Select the model with highest composite score
│
Stage 3: CASCADE DECISION (optional)
├── cascade_rate λ controls verification frequency
├── λ=0: pure single-shot (default)
└── λ>0: verify a fraction of predictions via fallback model
```

**Stage 1** enforces business constraints. **Stage 2** makes the intelligent routing decision. **Stage 3** optionally adds a verification layer for high-stakes deployments.

**Business knobs** control each stage:

| Knob | What It Controls | Example |
|------|-----------------|---------|
| `min_quality` | Quality floor — masks models below benchmark threshold | `min_quality=70` blocks models scoring < 70% |
| `max_cost` | Budget cap — masks models exceeding cost limit | `max_cost=1.00` enforces ≤ $1/1k tokens |
| `max_latency` | Speed limit — masks models slower than SLA | `max_latency=2.0` requires < 2s response |
| `cascade_rate` | Verification frequency (λ) | `cascade_rate=0.3` verifies 30% of predictions |

```python
model, log, mode = router.route(
    "Solve this calculus integral",
    min_quality=70,
    max_cost=1.00,
    cascade_rate=0.3,
)
```

### The Math

For readers interested in the algorithmic details:

```
1. EMBED:  prompt → x  (384-dim via Sentence Transformer)
2. REDUCE: x → x̃      (32-dim via domain-adapted PCA)
3. SCORE:  û = θ·x̃ + α√(x̃ᵀA⁻¹x̃) − λ·cost
              ↑ quality    ↑ exploration    ↑ cost penalty
              estimate     bonus
4. SELECT: pick model with highest û
```

Each model maintains two sufficient statistics:
- **A** (precision matrix): encodes "what prompt types I've seen" — uncertainty decreases as A grows
- **b** (reward-weighted features): encodes "what worked well" — quality estimates improve with data

When feedback arrives, both update via rank-one operations:

```
A ← A + x·xᵀ    (uncertainty shrinks)
b ← b + r·x      (reward signal accumulates)
```

This takes microseconds. No retraining, no gradient descent, no GPU required.

### Warm-Start Priors

Cold-starting a bandit requires hundreds of interactions before it routes well. BanditGPT ships with **warm-start priors** — a < 1 MB covariance matrix distilled from 80,000 RouteLLM battle outcomes — that encode which model tends to win for which prompt types. The router starts intelligent on Day 1 and adapts to your specific traffic from there.

### Safety via Corralling

What if the warm-start priors are wrong for your deployment? BanditGPT hedges via **Corralling**, a meta-learning algorithm that maintains two experts:

- A **Warmup Expert** initialized with offline priors
- A **Tabula Rasa Expert** that learns from scratch

The meta-learner tracks which expert is performing better and shifts weight accordingly. This provides insurance: when priors are accurate, the system exploits them with modest overhead. When priors are misleading, Corralling detects the mismatch and shifts to the cold-start expert, bounding worst-case regret.

---

## Configuration

### Optimization Profiles

| Profile | Description |
|---------|-------------|
| `auto` | Intelligent routing balancing quality and cost (default) |
| `custom` | Full control via weight dictionary |

```python
router.route(prompt, profile="auto")
router.route(prompt, profile={"w_q": 10.0, "w_c": 1.0, "w_l": 0.5})
```

### Exploration Rate

Controls how often the router tries unproven models versus exploiting known winners:

| Setting | Alpha | Use Case |
|---------|-------|----------|
| `static` | 0.0 | Zero exploration — production/fintech |
| `safe` | 0.05 | Minimal exploration (default) |
| `balanced` | 0.5 | Standard bandit behavior |
| `aggressive` | 1.0 | Day-1 calibration or shadow mode |

```python
router = BanditRouter.create(exploration="safe")
```

### Prior Initialization Modes

BanditGPT supports three initialization strategies, each with different trade-offs:

| Mode | What It Provides | File Required | Best For |
|------|-----------------|---------------|----------|
| `none` | No prior knowledge (A=λI, b=0) | None | Research baselines, tabula rasa |
| `hle` | Benchmark-guided bias, no feature correlations | None | Production (lightweight, fast) |
| `warmup` | Dense covariance with learned feature correlations | 0.85 MB | Maximum Day-0 quality |

**Decision guide**:
- Start with `hle` for most production deployments — it's lightweight and requires no files.
- Use `warmup` if you want the router to understand feature correlations from Day 1 (e.g., "models good at code tend to be good at math").
- Use `none` only for research baselines or when you want unbiased exploration.

```python
# Recommended for production
router = BanditRouter.create(registry, priors="hle")

# Maximum Day-0 quality
router = BanditRouter.create(registry, priors="warmup")
```

#### Tuning Prior Strength

Two parameters control how much the router trusts its priors:

| Parameter | Controls | Default | Effect |
|-----------|----------|---------|--------|
| `prior_structure_n_effective` | Covariance matrix confidence | `None` (full) | Higher → less exploration, more prior trust |
| `prior_n_effective` | Mean vector confidence | `20.0` | Higher → stronger belief in initial quality scores |

These let you independently tune "how confident are we in feature correlations?" versus "how confident are we in model quality rankings?" In most deployments, the defaults work well.

---

## Custom Encoders

BanditGPT ships with artifacts (PCA projection and warmup priors) trained on the default encoder (`sentence-transformers/all-MiniLM-L6-v2`). If you want to use a different sentence transformer, you must generate matching artifacts first. The library will raise a clear error if you try to use a custom encoder without them.

### Step 1: Train a PCA projection

```python
from bandit_gpt import train_pca

prompts = [...]  # Your representative corpus (200+ recommended)

pca = train_pca(
    prompts,
    encoder_model="your-org/your-encoder",
    n_components=32,
    output_path="my_pca.joblib",
)
```

### Step 2: Generate warmup priors

```python
from bandit_gpt import generate_warmup_priors

# Each entry: {"prompt": str, "rewards": {"model_id": float, ...}}
rewards_data = [...]

priors = generate_warmup_priors(
    rewards_data,
    encoder_model="your-org/your-encoder",
    pca="my_pca.joblib",
    output_path="my_priors.joblib",
)
```

### Step 3: Create the router

```python
from bandit_gpt import BanditRouter, FeatureService

fs = FeatureService(
    encoder_model="your-org/your-encoder",
    pca_path="my_pca.joblib",
)

router = BanditRouter.create(
    context_model="your-org/your-encoder",
    feature_service=fs,
    warmup_path="my_priors.joblib",
)
```

If you don't have labelled data for warmup priors, you can skip them entirely with `priors="none"` and let the router learn from scratch.

---

## Production Features

BanditGPT includes several mechanisms for production reliability:

| Feature | Problem It Solves | Mechanism |
|---------|------------------|-----------|
| **Snapshot-Swap Updates** | Matrix inversions (O(d³), ~50ms) blocked all routing during updates | Three-phase: snapshot state → compute without lock → atomic swap. Lock time: ~50ms → ~0.2ms |
| **Durable Context Store** | Feedback arriving hours/days after routing was lost on restart | SQLite-backed persistence (WAL mode). Contexts survive restarts; 7-day TTL auto-cleanup |
| **Self-Healing PCA** | Missing or mismatched PCA artifacts caused crashes | JIT validation and retraining on startup (~2s). Zero-downtime recovery from artifact corruption |
| **Tiered Safety** | ML toxicity scanners add 100-300ms per request | Fast regex heuristic in hot path (<1ms) + async ML audit in background. 20,000-60,000× faster |
| **Adaptive Priors** | Overly stiff warm-start priors ignored new evidence ("zombie mode") | Natural weighting (1:100 prior-to-real ratio) preserves stability while maintaining plasticity |

### Database Storage

BanditGPT uses SQLite for context persistence, created lazily (only when you first call `process_feedback`):

| Environment | Location | When Created |
|-------------|----------|-------------|
| Library install | `~/.bandit_gpt/router_context.db` | On first feedback |
| Development | `<repo>/data/router_context.db` | On first feedback |
| Custom | User-specified path | On first feedback |

```python
# Routing-only workflow — no database files created
router = BanditRouter.create()
for prompt in prompts:
    model = router.route(prompt)

# Feedback workflow — database created on first use
router.process_feedback(request_id, reward=0.95)
```

You can customize TTL, location, and inspect statistics:

```python
from bandit_gpt.storage import SqliteContextStore

store = SqliteContextStore(
    db_path="/var/app/bandit_router.db",
    ttl_seconds=86400 * 30,  # 30-day retention
)
router = BanditRouter.create(context_store=store)

stats = router.context_store.stats()
print(f"Contexts: {stats['total_contexts']}, Size: {stats['db_size_mb']} MB")
```

---

## Hybrid Router (Bandit-Guided Cascade)

For deployments requiring verification of routing decisions:

```python
from bandit_gpt import HybridRouter

hybrid = HybridRouter.create(
    model_registry=registry,
    fallback_model="openai/gpt-4o",
    cascade_rate=0.5,
)

result = hybrid.route_with_cascade(
    prompt="Write SQL to get all active users",
    generate_fn=lambda m, p: call_llm(m, p),
    verify_fn=lambda r: validate_sql(r),
)
# result['mode']: "single_shot" (fast) or "cascade" (verified)
```

Unlike FrugalGPT's cascade (which tries models sequentially, O(N) latency), the Hybrid Router uses O(1) bandit selection over the full portfolio and cascades only when uncertain.

---

## Limitations

We report these limitations honestly to help practitioners make informed decisions:

1. **Two-model evaluation scope.** The primary Pareto comparison uses Mixtral and GPT-4-Turbo (to match RouteLLM's evaluation). With only two models, the exploration bonus is symmetric and contributes minimally. Larger portfolios (K ≥ 3) should benefit more from contextual exploration, but quantitative results will differ.

2. **Corralling has overhead.** The meta-learner trades peak performance for safety. When priors are accurate, a single-expert warmup strategy outperforms Corralling. The insurance is justified only when prior quality is uncertain.

3. **Semantic transfer does not work.** We investigated bootstrapping new models from semantically similar existing models and found no statistically significant improvement over cold-start initialization (p > 0.07 across all configurations). Cold-start model integration remains an open problem.

4. **Linear reward assumption.** LinUCB assumes reward is linear in the feature vector. While the PCA features capture meaningful signal (ρ = -0.370, p < 0.0001), highly nonlinear preference structures may require different architectures.

5. **Stationary experts.** Each expert bandit assumes stationary rewards. Non-stationarity is handled only at the meta-level via Corralling. Gradual reward drift within a single expert requires meta-level reweighting, not expert-level adaptation.

6. **Prior quality dependency.** The warm-start advantage depends on similarity between offline training distribution and deployment traffic. For domains very different from LMSYS Arena conversations, the warmup prior's value diminishes (though the router will still learn from online data).

---

## Installation

```bash
pip install banditgpt
```

On first use, BanditGPT downloads the sentence transformer model weights (~80 MB) from Hugging Face. To pre-download them (recommended for Docker images and CI pipelines):

```bash
banditgpt --download-models
```

From source:

```bash
git clone https://github.com/atabernermiller/banditgpt.git
cd banditgpt
pip install -e .
```

Optional extras:

```bash
pip install banditgpt[full]          # LLM-as-judge grading via OpenRouter
pip install banditgpt[experiments]   # Reproduce paper figures
pip install banditgpt[dev]           # Development tools
```

### Requirements

**Core** (installed automatically): Python 3.10+, numpy, torch, pandas, sentence-transformers, transformers

**Optional**: `openai` (LLM-as-judge grading), `python-dotenv` (API key management), `matplotlib` (experiment visualization)

### Troubleshooting

| Issue | Solution |
|-------|----------|
| Missing or corrupted priors | Run `banditgpt verify-priors` or reinstall |
| Missing sentence-transformers | `pip install sentence-transformers transformers` |
| OpenRouter grading fails | Set `OPENROUTER_API_KEY` environment variable |
| Debug logging | Set `PYTHONLOGGING=DEBUG` for init/prior resolution details |

---

## Testing

```bash
python -m pytest tests/ -v          # All tests (~2 min)
python -m pytest tests/test_lock_contention.py -v   # Concurrency tests
```

See [`tests/README.md`](tests/README.md) for details on the 135-test suite covering router workflow, feedback loops, prior management, optimization profiles, and concurrency.

---

## The Paper

This repository accompanies *"Density-Based Warm-Start for Adaptive LLM Routing"* (2025).

### Key Findings

| Research Question | Result |
|-------------------|--------|
| Does prompt structure predict model preference? | Yes — ρ = −0.370, p < 0.0001, exceeding all 100 random projections |
| Does online learning surpass static routing? | Yes at moderate-to-high budgets — 70.0% gap closure to oracle (vs. 46.2% for RouteLLM) after ~200 label-free prompts. At low budgets, RouteLLM's pre-trained discrimination is competitive. |
| Does Corralling provide safety under prior mismatch? | Yes — bounded worst-case regret (32% lower than warmup-only), 95% catastrophic failure detection at K=5 |
| Does semantic transfer help cold-start new models? | No — null result across all configurations (p > 0.07). Reported transparently. |

### Reproducing Experiments

Each experiment directory maps 1:1 to a figure or table in the paper:

| Paper Object | Directory | Script |
|-------------|-----------|--------|
| Figure 1: Model Preference Heterogeneity | [`experiments/01_figure/`](experiments/01_figure/) | `plot_figure1.py` |
| Table 2: Dataset Description | [`experiments/02_table/`](experiments/02_table/) | `generate_table1.py` |
| Figure 3: Pareto Frontier | [`experiments/03_figure/`](experiments/03_figure/) | `generate_pareto_frontier.py` |
| Figure 4: Multi-Model Pareto | [`experiments/04_figure/`](experiments/04_figure/) | `run_multimodel_pareto.py` |
| Figure 9: Catastrophic Failure | [`experiments/appendix/E_catastrophic_failure_experiment/`](experiments/appendix/E_catastrophic_failure_experiment/) | `generate_figure9_5model.py` |

See [`paper/README.md`](paper/README.md) for the complete artifact guide and [`experiments/README.md`](experiments/README.md) for reproduction instructions.

---

## Prior Management

### Bundled vs. User Priors

| Location | Path | Purpose |
|----------|------|---------|
| Bundled | `<package>/data/priors/expert_priors.npz` | Expert-distilled defaults (read-only) |
| User | `~/.banditgpt/priors/user_priors.npz` | Your learned updates |

### Adding New Models

```python
router.add_model("openai/gpt-5", clone_from="openai/gpt-4o")
```

### Integrity and Recovery

Bundled priors are checksummed via `banditgpt/data/priors/manifest.json`. To verify:

```bash
python -m banditgpt.core.cli verify-priors
```

To restore from git if needed:

```bash
git show <ref>:banditgpt/data/priors/expert_priors.npz > expert_priors.npz
```

Priors are loaded with `allow_pickle=False` and use fixed-width arrays for security.

---

## Project Structure

```
banditgpt/
├── src/bandit_gpt/          # Core library
│   ├── router.py            # BanditRouter, LinUCB, Corralling (~4700 lines)
│   ├── storage.py           # SQLite context persistence
│   ├── feature_service.py   # Prompt embedding + PCA
│   ├── calibration.py       # train_pca(), generate_warmup_priors() for custom encoders
│   ├── baselines.py         # Baseline comparison implementations
│   └── utils/               # Warmup, heuristics
├── paper/                   # LaTeX source and figures
├── experiments/             # Reproducible experiments (1:1 with paper figures)
├── tests/                   # 440+ tests across 40+ files
├── scripts/                 # Data processing and prior generation
└── data/                    # Experimental datasets
```

---

## Technical Deep Dive

This section provides additional technical depth for researchers and advanced practitioners.

### Feature Linearization

BanditGPT uses a "Binary + Log" transformation (φ(x) = [𝟙(x>0), ln(1+x)]) rather than binning or piecewise linearization, for three reasons:

1. **Sample efficiency.** Binning would expand the feature space from d=14 to d≈140. Since LinUCB regret scales as O(d√T), this would extend the cold-start period by an order of magnitude.

2. **Zero-inflated features.** Prompt telemetry features like `code_density` and `latex_count` are highly sparse. The binary indicator models the qualitative mode switch (chat vs. coding) while the log term captures magnitude.

3. **Weber-Fechner alignment.** LLM processing difficulty follows logarithmic scaling — the difference between 200 and 100 tokens matters more than between 10,200 and 10,100. The log term encodes this with a single parameter.

### Snapshot-Swap Concurrency

The router uses a three-phase update pattern to prevent lock contention in high-QPS deployments:

1. **Snapshot** (~0.1ms): Brief lock to copy state
2. **Compute** (~50ms): Matrix inversion without holding the lock
3. **Swap** (~0.1ms): Atomic commit of computed results

This reduces lock hold time from ~50ms to ~0.2ms (250× improvement), allowing routing to proceed in parallel during expensive O(d³) inversions.

See [`tests/test_lock_contention.py`](tests/test_lock_contention.py) for concurrency validation.

### Prior Initialization: Mathematical Details

**HLE mode** injects benchmark scores as a bias term while keeping the covariance matrix as pure identity (A = λI). This avoids the rank deficiency problem: with d≈53 dimensions, estimating a full covariance matrix requires ~530 samples for stability. Small-sample covariances hallucinate spurious correlations.

**Warmup mode** uses a dense covariance matrix from 20,000 simulated interactions (740,000 Bayesian updates). The covariance encodes learned correlations (code ↔ math, reasoning ↔ complexity). Scaling via `prior_n_effective` controls plasticity:

```
θ_scaled = (A × s)⁻¹(b × s) = A⁻¹b = θ_raw   (preferences preserved)
```

Without scaling (raw N=20,000), each real observation contributes only 0.005% — the router becomes "deaf" to new data. With N=1,000 scaling, the ratio is 0.1% — stable yet adaptive.

### Corralling: Formal Properties

The meta-learner uses importance-weighted loss estimates for unbiased expert evaluation:

```
ℓ̂(t,e) = (1 - r_t) / p_t(e)    if e was selected
         = 0                      otherwise
```

Expert weights update via exponential descent: w(t+1,e) ∝ exp(−η Σ ℓ̂(s,e)).

A mixing floor γ=0.05 prevents expert death: p(i,t) = (1−γ)·w(i,t)/Σw + γ/K. This guarantees every expert retains at least 2.5% selection probability, enabling recovery if the optimal expert changes.

In production, the implementation updates only the selected expert (O(1) instead of O(K)), trading theoretical regret guarantees for strict latency SLAs (< 20ms overhead).

---

## Acknowledgments

The bundled `expert_priors.npz` was generated using these open-source datasets:

| Dataset | License | Usage |
|---------|---------|-------|
| [LMSYS Chatbot Arena](https://huggingface.co/datasets/lmsys/chatbot_arena_conversations) | CC-BY-4.0 | 497 archetype prompts |
| [LMSYS Arena Preferences](https://huggingface.co/datasets/lmsys/lmsys-arena-human-preference-55k) | CC-BY-4.0 | Quality model training |
| [NVIDIA HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2) | CC-BY-4.0 | Quality model training |

Model responses (81 models × 497 prompts) were generated via [OpenRouter](https://openrouter.ai/).

---

## License

Apache License 2.0 — See [LICENSE](LICENSE) for details.
