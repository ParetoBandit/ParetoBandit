# ParetoBandit

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Tests](https://img.shields.io/badge/tests-135%20passed-brightgreen.svg)](#testing)

**An adaptive, local-first router that learns which LLM works best for *your* prompts.**

---

## Overview

ParetoBandit is an open-source contextual bandit framework for LLM routing. Instead of relying on static rules or pre-trained classifiers to choose between language models, it learns from your actual traffic which model performs best for each type of prompt — then adapts continuously as your usage patterns change.

**The core insight**: expensive models are not always better. On held-out LMSYS Arena prompts, roughly 14% of prompts are *actively worse* when routed to GPT-4-Turbo instead of Mixtral. A static router that equates "hard" with "needs GPT-4" will systematically over-provision the expensive model on these prompts — paying 43× more for 1.3% lower quality. ParetoBandit discovers this preference structure online and routes accordingly.

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
pip install paretobandit
```

```python
from pareto_bandit import BanditRouter

# Create router — learns from scratch, no external data needed
router = BanditRouter.create(model_registry)

# Route a prompt — the router selects the best model for this context
model_id, log = router.route("Write a Python function to parse JSON")

# After observing quality, update the router (microsecond operation)
router.process_feedback(log.request_id, reward=0.95)
```

No labels, no retraining, no external API calls. The router runs locally and learns from its own routing outcomes.

### Bring Your Own Provider

ParetoBandit is provider-agnostic. For a single provider, pass any adapter directly:

```python
from pareto_bandit import BanditRouter, OpenAIClient

router = BanditRouter.create(model_registry)
client = OpenAIClient(api_key="sk-...")

model_id, response, log = router.route_and_call("Write a Python function to parse JSON", client)
router.process_feedback(log.request_id, reward=0.95)
```

For **mixed-provider portfolios**, use `MultiProviderClient` — it maps each model's provider prefix to the right client automatically:

```python
from pareto_bandit import (
    BanditRouter, MultiProviderClient,
    OpenAIClient, AnthropicClient, OllamaClient,
)

router = BanditRouter.create(model_registry)
client = MultiProviderClient({
    "openai":     OpenAIClient(api_key="sk-..."),
    "anthropic":  AnthropicClient(api_key="sk-ant-..."),
    "meta-llama": OllamaClient(),
})

model_id, response, log = router.route_and_call("Parse this JSON", client)
```

| Provider | Adapter | Install |
|----------|---------|---------|
| OpenRouter | `OpenRouterClient` | `pip install paretobandit[openrouter]` |
| OpenAI | `OpenAIClient` | `pip install paretobandit[openai]` |
| Anthropic | `AnthropicClient` | `pip install paretobandit[anthropic]` |
| Google Gemini | `GeminiClient` | `pip install paretobandit[gemini]` |
| Ollama (local) | `OllamaClient` | `pip install paretobandit[ollama]` |
| DeepSeek, Grok, Together, etc. | `OpenAIClient(base_url=...)` | `pip install paretobandit[openai]` |

See the [API Reference](docs/API_REFERENCE.md#providers) for the full multi-provider workflow.

---

## When Should You Use ParetoBandit?

| Scenario | Recommended Approach | Why |
|----------|---------------------|-----|
| Need routing today, no deployment data | **RouteLLM** | Pre-trained classifier works out of the box |
| Prompt distribution shifts over time | **ParetoBandit** | Adapts continuously; RouteLLM is frozen at training time |
| No labeled routing data for your domain | **ParetoBandit** | Requires zero labels — learns from routing outcomes |
| Multiple models (K ≥ 3) at different price tiers | **ParetoBandit** | Contextual bandit handles multi-model portfolios natively |
| Maximum quality is the priority | **ParetoBandit** | After ~400 prompts, surpasses what 100k supervised pairs achieve |
| Simple cost-cutting on well-understood traffic | **RouteLLM** | Supervised pre-training is competitive at moderate budgets |

The 400-prompt crossover means a new deployment can surpass a pre-trained router within minutes of production traffic, with no human annotation.

---

## Production Use Cases

ParetoBandit is designed for any system that routes heterogeneous prompts across multiple LLMs under cost, latency, or quality constraints. Below are concrete deployment archetypes where the library provides the most value.

### Primary use cases

| Use Case | Why ParetoBandit Fits | Key Features Used |
|----------|-------------------|-------------------|
| **Customer support platforms** — Route millions of tickets where "reset my password" and "your API returns 500 on nested JSON" require very different models | Traffic mix shifts after product launches and seasonal spikes; hand-written heuristics break. Online learning adapts automatically, and the 65/35 easy-to-hard ratio means >60% of traffic routes to the cheapest model. | `max_cost`, `cost_penalty`, geometric forgetting |
| **LLM API gateways** — Proxy services (LiteLLM, Portkey, Helicone-class) where each customer's traffic is different | One-size-fits-all routing leaves money on the table. Instantiate a router per customer, each learning its own optimal policy. New models are added to the fleet via `register_model()` without per-customer reconfiguration. | `MultiProviderClient`, `register_model()`, per-tenant instances |
| **Coding assistant backends** — IDE plugins where autocomplete needs <200ms but complex refactoring needs frontier reasoning | The boundary between "easy" and "hard" is fuzzy and prompt-dependent. Hard latency constraints guarantee the SLA; the bandit learns which queries genuinely need a frontier model. New models drop monthly and integrate without re-tuning. | `max_latency`, `register_model()`, geometric forgetting |
| **RAG-based enterprise search** — Internal chatbots where simple factual lookups and complex multi-hop reasoning coexist | Over-routing everything to GPT-4 wastes budget; under-routing complex queries produces hallucinations. The prompt embedding captures complexity, and cost penalty steers simple queries to cheap models. | `cost_penalty`, exploration presets |

### Secondary use cases

| Use Case | Why ParetoBandit Fits | Caveat |
|----------|-------------------|--------|
| **Content generation platforms** — Short social media copy vs. long-form brand-voice content require different quality tiers | The router learns from editorial feedback what "good enough" means per content type. | Needs sufficient volume (~100+ prompts/day) for fast convergence. |
| **Multi-tenant AI-as-a-service** — Vertical SaaS wrapping LLMs for SMB customers with diverse traffic patterns | Per-tenant router instances learn independently; fleet-wide model changes propagate via `register_model()`. | Each tenant needs enough traffic for per-tenant learning to converge. |
| **Agentic workflows** — Different steps in an agent pipeline (planning, code gen, formatting) have wildly different difficulty | Route per-step with the step's prompt as context; the bandit discovers which steps genuinely need a frontier model. | Not yet experimentally validated (flagged as future work in the paper). |

### When ParetoBandit is *not* the right tool

| Situation | Better Alternative |
|-----------|-------------------|
| Fewer than ~50 prompts/day — insufficient data for online learning | Static rules or RouteLLM's pre-trained classifier |
| Single model — no routing decision to make | Direct API call |
| Regulated domain where routing decisions must be fully deterministic and auditable | Rule-based routing with human-defined policies |
| All prompts are near-identical (e.g., same template, different entity) — no contextual signal | Random load balancing or round-robin |

### Worked examples

Each primary use case has a runnable example with a synthetic oracle, full learning loop, baseline comparisons, and 4-panel visualisation:

| Scenario | Script | Features Demonstrated |
|----------|--------|----------------------|
| Cost-constrained SaaS startup | [`examples/scenario_cost_constrained_startup.py`](examples/scenario_cost_constrained_startup.py) | Hard budget ceiling, aggressive cost penalty, per-category cost analysis |
| Latency-sensitive IDE plugin | [`examples/scenario_latency_sensitive_app.py`](examples/scenario_latency_sensitive_app.py) | TTFT constraint, geometric forgetting under distribution shift |
| Quality-critical enterprise + model onboarding | [`examples/scenario_quality_critical_enterprise.py`](examples/scenario_quality_critical_enterprise.py) | Quality-first routing, hot `register_model()`, newcomer adoption ramp |
| General tutorial (5-model portfolio) | [`examples/hands_on_tutorial.py`](examples/hands_on_tutorial.py) | Full walkthrough: exploration, cost penalty, priors, model onboarding |

All examples run locally in under a minute with no API keys.

---

## How It Works

### The Intuition

Imagine 80 slot machines in a casino, each with unknown and different payout rates. You have a limited budget and want to maximize your total payout. The **bandit algorithm** balances two goals:

- **Exploitation**: keep pulling the machine that paid well so far
- **Exploration**: try unknown machines that might pay even better

ParetoBandit treats each LLM as a slot machine. But unlike a regular bandit, it uses **context** — the content of your prompt — to predict which model will perform best *for this specific request*. A coding prompt gets routed differently than a creative writing prompt, even if both could technically go to any model.

### The 2-Stage Routing Pipeline

Every routing decision passes through two stages:

```
Stage 1: HARD FILTERING
├── Remove models violating constraints (budget cap, latency SLA, quality floor)
├── Full portfolio → surviving candidates
│
Stage 2: CONTEXTUAL BANDIT SELECTION
├── For each candidate, score: quality estimate + exploration bonus - cost penalty
└── Select the model with highest composite score
```

**Stage 1** enforces business constraints. **Stage 2** makes the intelligent routing decision.

**Business knobs** control each stage:

| Knob | What It Controls | Example |
|------|-----------------|---------|
| `min_quality` | Quality floor — masks models below benchmark threshold | `min_quality=70` blocks models scoring < 70% |
| `max_cost` | Budget cap — masks models exceeding cost limit | `max_cost=1.00` enforces ≤ $1/1k tokens |
| `max_latency` | Speed limit — masks models slower than SLA | `max_latency=2.0` requires < 2s response |

```python
model, log = router.route(
    "Solve this calculus integral",
    min_quality=70,
    max_cost=1.00,
)
```

### The Math

For readers interested in the algorithmic details:

```
1. EMBED:  prompt → x  (via Sentence Transformer, custom encoder, or pre-computed vector)
2. REDUCE: x → x̃      (optional PCA compression, e.g. 384D → 32D)
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

Cold-starting a bandit requires hundreds of interactions before it routes well. ParetoBandit ships with **warm-start priors** — a < 1 MB covariance matrix distilled from 80,000 RouteLLM battle outcomes — that encode which model tends to win for which prompt types. The router starts intelligent on Day 1 and adapts to your specific traffic from there.

### Robustness to Prior Mismatch

What if the warm-start priors are wrong for your deployment? ParetoBandit hedges via two mechanisms:

- **Tunable prior strength** (`n_eff`): Controls how many pseudo-observations the offline priors contribute. A conservative `n_eff` lets online evidence override potentially stale priors within a few hundred requests.
- **Geometric forgetting** (`gamma`): Exponentially discounts stale observations, giving recent evidence a half-life of ~200 steps. If a model degrades or the prior was inaccurate, the router adapts within hundreds of requests rather than remaining anchored by thousands of stale data points.

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

ParetoBandit supports two initialization strategies, each with different trade-offs:

| Mode | What It Provides | File Required | Best For |
|------|-----------------|---------------|----------|
| `none` | No prior knowledge (A=λI, b=0) | None | Research baselines, tabula rasa |
| `warmup` | Dense covariance with learned feature correlations | 0.85 MB | Production & maximum Day-0 quality |

**Decision guide**:
- The default (`"none"`) starts with standard LinUCB cold-start — typically converges in 20–50 requests.
- If you have historical reward data, generate custom priors with `generate_warmup_priors()` for faster convergence.

```python
# Default: cold start (no external data needed)
router = BanditRouter.create(registry)

# With custom priors from your own data
router = BanditRouter.create(registry, priors="path/to/my_priors.joblib")
```

#### Tuning Prior Strength

Two parameters control how much the router trusts its priors:

| Parameter | Controls | Default | Effect |
|-----------|----------|---------|--------|
| `prior_structure_n_effective` | Covariance matrix confidence | `None` (full) | Higher → less exploration, more prior trust |
| `prior_n_effective` | Mean vector confidence | `20.0` | Higher → stronger belief in initial quality scores |

These let you independently tune "how confident are we in feature correlations?" versus "how confident are we in model quality rankings?" In most deployments, the defaults work well.

---

## PCA Projection

ParetoBandit compresses prompt embeddings from 1024 dimensions down to 32 via PCA before feeding them to the bandit. A pre-trained PCA artifact ships inside the wheel so the router works immediately after `pip install` — no extra downloads, no JIT retraining on first request.

### What ships and how it was trained

The bundled `pca_32.joblib` (~133 KB) was trained on **80,000 RouteLLM battle prompts** using the default sentence encoder (`BAAI/bge-m3`). This dataset is independent of ParetoBandit's dev/holdout evaluation splits, so there is no data contamination. The 32 components capture **32.7%** of the embedding variance, which is sufficient for the routing signal (see the paper's PCA ablation in `experiments/03_figure/run_pca_neff_ablation.py`).

### When the default PCA is enough

For most deployments the bundled PCA works out of the box. It was trained on a broad mix of English prompts (coding, math, reasoning, creative writing, general chat) from real human-LLM conversations, so it covers the principal axes of variation in typical production traffic.

### When to train your own

You should replace the bundled PCA when:

- **You use a different sentence encoder.** The PCA must match the encoder's embedding space. Passing a custom `encoder_model` without a matching `pca_path` raises a `ValueError`.
- **Your domain is far from general English chat.** If your traffic is dominated by a narrow domain (e.g., Japanese legal contracts, biomedical literature), the bundled projection may filter out critical semantic variance. Training a domain-specific PCA on a representative sample of your prompts will capture the axes that matter most for your routing decisions.
- **You want more (or fewer) components.** 32 is the default, but you can increase it for richer signal (at the cost of slower O(d^2) bandit updates) or decrease it for faster updates with slightly coarser embeddings.

### Training a custom PCA

```python
from pareto_bandit import train_pca

# Collect 200+ representative prompts from your actual traffic
prompts = [...]

pca = train_pca(
    prompts,
    encoder_model="BAAI/bge-m3",  # or your custom encoder
    n_components=32,
    output_path="my_pca.joblib",
)
print(f"Explained variance: {sum(pca.explained_variance_ratio_):.1%}")
```

Then pass the custom artifact when creating the router:

```python
from pareto_bandit import BanditRouter, FeatureService

fs = FeatureService(
    encoder_model="BAAI/bge-m3",
    pca_path="my_pca.joblib",
)

router = BanditRouter.create(feature_service=fs)
```

### Using a custom encoder end-to-end

If you want to swap out the sentence transformer entirely, you need matching PCA and (optionally) warmup priors:

**Step 1: Train PCA on your encoder**

```python
pca = train_pca(
    prompts,
    encoder_model="your-org/your-encoder",
    n_components=32,
    output_path="my_pca.joblib",
)
```

**Step 2: Generate warmup priors (optional)**

```python
from pareto_bandit import generate_warmup_priors

# Each entry: {"prompt": str, "rewards": {"model_id": float, ...}}
rewards_data = [...]

priors = generate_warmup_priors(
    rewards_data,
    encoder_model="your-org/your-encoder",
    pca="my_pca.joblib",
    output_path="my_priors.joblib",
)
```

**Step 3: Create the router**

```python
from pareto_bandit import BanditRouter, FeatureService

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

If you don't have labelled data for warmup priors, skip them with `priors="none"` and let the router learn from scratch.

### Self-healing fallback

If the PCA artifact is missing at runtime (e.g., deleted or moved), the `FeatureService` can JIT-train a replacement from synthetic prompts. This keeps the router available but produces a CRITICAL log warning — the synthetic distribution may not match your production traffic. In strict production deployments, set `allow_jit_training=False` to crash-fast instead of falling back silently.

---

## Bring Your Own Embeddings

ParetoBandit does **not** require the default sentence-transformer pipeline. You can supply your own embedding function — OpenAI embeddings, Cohere, a local ONNX model, or any other source — and the library handles the rest (PCA, bias term, LinUCB math). This also means you can install just `pip install paretobandit` (no PyTorch, no Hugging Face).

There are three embedding paths, from simplest to most flexible:

| Path | When to use | Requires `sentence-transformers`? |
|------|-------------|-----------------------------------|
| **Default** — `FeatureService()` | Quick start, general-purpose traffic | Yes (`pip install paretobandit[embeddings]`) |
| **Custom encoder** — `FeatureService(custom_encoder=fn)` | You want to use OpenAI, Cohere, ONNX, etc. while the library handles PCA and bias | No |
| **Pre-computed vectors** — `FeatureService.for_precomputed(dim)` | You manage the full embedding pipeline externally and pass numpy arrays directly | No |

### Path 1: Custom encoder callable

Pass any function that maps `str → np.ndarray` (a 1-D float vector). You must also specify the vector dimensionality via `embedding_dim`.

```python
import numpy as np
from pareto_bandit import BanditRouter, FeatureService

# Example: OpenAI embeddings
from openai import OpenAI
openai_client = OpenAI()

def openai_embed(prompt: str) -> np.ndarray:
    resp = openai_client.embeddings.create(
        model="text-embedding-3-small", input=prompt
    )
    return np.array(resp.data[0].embedding)

fs = FeatureService(
    custom_encoder=openai_embed,
    embedding_dim=1536,  # must match what your encoder produces
)

router = BanditRouter.create(model_registry=registry, feature_service=fs, priors="none")
model_id, log = router.route("Solve x^2 + 2x + 1 = 0")
```

**What you need to ensure:**

1. **`embedding_dim` must match.** The integer you pass must exactly equal the length of the vector your callable returns. A mismatch will raise at runtime.
2. **Return a 1-D numpy array.** Shape `(dim,)`, not `(1, dim)`. The library validates this and raises a clear error if the shape is wrong.
3. **Consistency.** Your encoder must produce vectors in the same space across the lifetime of the router. Switching encoder models mid-session invalidates learned bandit parameters.
4. **No PCA is applied by default.** When you pass a `custom_encoder` without a `pca_path`, the library uses your raw embeddings directly (+ a bias term). This means the feature dimension equals `embedding_dim + 1`. For high-dimensional embeddings (e.g., 1536 or 3072), the LinUCB covariance matrices will be larger — this is fine for correctness but uses more memory and takes O(d²) per update instead of O(32²).
5. **Optional PCA for dimensionality reduction.** If you want to compress high-dimensional custom embeddings, train a PCA on your encoder's output and pass both:

```python
# Train PCA from your custom embeddings (one-time)
from sklearn.decomposition import PCA
import joblib

embeddings = np.array([openai_embed(p) for p in representative_prompts])
pca = PCA(n_components=32).fit(embeddings)
joblib.dump(pca, "my_openai_pca.joblib")

# Use custom encoder + PCA
fs = FeatureService(
    custom_encoder=openai_embed,
    embedding_dim=1536,
    pca_path="my_openai_pca.joblib",
)
```

### Path 2: Pre-computed vectors

If you manage the full embedding pipeline externally, pass numpy arrays directly to `route()`:

```python
from pareto_bandit import BanditRouter, FeatureService
import numpy as np

dim = 65  # 64 features + 1 bias
fs = FeatureService.for_precomputed(dimension=dim)

router = BanditRouter.create(model_registry=registry, feature_service=fs, priors="none")

# At routing time, pass your own vector (last element must be 1.0 for the bias term)
vector = np.random.randn(dim)
vector[-1] = 1.0
model_id, log = router.route(vector)
```

**What you need to ensure:**

1. **The last element must be 1.0** (the bias term). The LinUCB intercept depends on this.
2. **All vectors must have the same dimension** over the router's lifetime.
3. **Passing a string prompt will raise** — there is no encoder loaded.

### Checklist: custom embedding mode

Before deploying with custom embeddings, verify:

- [ ] Your encoder callable returns a 1-D `np.ndarray` of consistent length
- [ ] `embedding_dim` matches the actual output length of your encoder
- [ ] If using PCA, the PCA artifact was trained on embeddings from the **same** encoder
- [ ] Vectors are numerically stable (no NaN, no Inf, no extreme values)
- [ ] You are not switching encoder models between router `save_state` / `load_state` — learned parameters are tied to the embedding space
- [ ] For pre-computed vectors, the last element is `1.0` (bias term)

---

## Production Features

ParetoBandit includes several mechanisms for production reliability:

| Feature | Problem It Solves | Mechanism |
|---------|------------------|-----------|
| **Snapshot-Swap Updates** | Matrix inversions (O(d³), ~50ms) blocked all routing during updates | Three-phase: snapshot state → compute without lock → atomic swap. Lock time: ~50ms → ~0.2ms |
| **Durable Context Store** | Feedback arriving hours/days after routing was lost on restart | SQLite-backed persistence (WAL mode). Contexts survive restarts; 7-day TTL auto-cleanup |
| **Self-Healing PCA** | Missing or mismatched PCA artifacts caused crashes | Pre-trained PCA ships in the wheel; JIT validation and fallback retraining on startup (~2s) if artifact is absent or corrupted |
| **Tiered Safety** | ML toxicity scanners add 100-300ms per request | Fast regex heuristic in hot path (<1ms) + async ML audit in background. 20,000-60,000× faster |
| **Adaptive Priors** | Overly stiff warm-start priors ignored new evidence ("zombie mode") | Natural weighting (1:100 prior-to-real ratio) preserves stability while maintaining plasticity |

### Database Storage

ParetoBandit uses SQLite for context persistence, created lazily (only when you first call `process_feedback`):

| Environment | Location | When Created |
|-------------|----------|-------------|
| Library install | `~/.pareto_bandit/router_context.db` | On first feedback |
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
from pareto_bandit.storage import SqliteContextStore

store = SqliteContextStore(
    db_path="/var/app/bandit_router.db",
    ttl_seconds=86400 * 30,  # 30-day retention
)
router = BanditRouter.create(context_store=store)

stats = router.context_store.stats()
print(f"Contexts: {stats['total_contexts']}, Size: {stats['db_size_mb']} MB")
```

---

## Limitations

We report these limitations honestly to help practitioners make informed decisions:

1. **Two-model evaluation scope.** The primary Pareto comparison uses Mixtral and GPT-4-Turbo (to match RouteLLM's evaluation). With only two models, the exploration bonus is symmetric and contributes minimally. Larger portfolios (K ≥ 3) should benefit more from contextual exploration, but quantitative results will differ.

2. **Semantic transfer does not work.** We investigated bootstrapping new models from semantically similar existing models and found no statistically significant improvement over cold-start initialization (p > 0.07 across all configurations). Cold-start model integration remains an open problem.

3. **Linear reward assumption.** LinUCB assumes reward is linear in the feature vector. While the PCA features capture meaningful signal (ρ = -0.370, p < 0.0001), highly nonlinear preference structures may require different architectures.

4. **Warm-start prior dependency.** At tight budgets, expensive arms receive very few online observations, so quality estimates rely heavily on the warmup prior. If the prior is poorly calibrated for a newly added model, a tight budget could delay adoption. The tunable prior strength (`n_eff`) mitigates this but does not eliminate it.

6. **Prior quality dependency.** The warm-start advantage depends on similarity between offline training distribution and deployment traffic. For domains very different from LMSYS Arena conversations, the warmup prior's value diminishes (though the router will still learn from online data).

---

## Installation

```bash
pip install paretobandit                # Core library (lightweight: numpy, pandas, scikit-learn)
pip install paretobandit[embeddings]    # + default sentence-transformer embedding pipeline
```

The **core** install is lightweight — no PyTorch, no Hugging Face downloads.  It's all you need if you [bring your own embeddings](#bring-your-own-embeddings).

The **`[embeddings]`** extra adds `torch`, `sentence-transformers`, and `transformers`, giving you the default embedding pipeline that works out of the box. On first use this downloads the sentence transformer model weights (~80 MB) from Hugging Face. To pre-download them (recommended for Docker images and CI pipelines):

```bash
paretobandit --download-models
```

From source:

```bash
git clone https://github.com/atabernermiller/paretobandit.git
cd paretobandit
pip install -e ".[embeddings]"
```

Optional extras:

```bash
pip install paretobandit[embeddings]    # Default sentence-transformer embedding pipeline
pip install paretobandit[openrouter]    # OpenRouter adapter
pip install paretobandit[openai]        # Direct OpenAI (also works for DeepSeek, Grok, etc.)
pip install paretobandit[anthropic]     # Anthropic adapter
pip install paretobandit[gemini]        # Google Gemini adapter
pip install paretobandit[ollama]        # Local Ollama adapter
pip install paretobandit[full]          # All providers + embeddings + utilities
pip install paretobandit[experiments]   # Reproduce paper figures
pip install paretobandit[dev]           # Development tools
```

### Requirements

**Core** (installed automatically): Python 3.10+, numpy, pandas, joblib, scikit-learn

**Embeddings** (optional, via `[embeddings]`): torch, sentence-transformers, transformers

**Providers** (optional): `openai`, `anthropic`, `google-genai`, `ollama`

**Utilities** (optional): `python-dotenv` (API key management), `matplotlib` (experiment visualization)

### Troubleshooting

| Issue | Solution |
|-------|----------|
| Missing or corrupted priors | Run `paretobandit verify-priors` or reinstall |
| `ImportError: sentence-transformers` | `pip install paretobandit[embeddings]`, or use a custom encoder / pre-computed vectors |
| Provider auth fails | Set the appropriate env var (`OPENROUTER_API_KEY`, `OPENAI_API_KEY`, etc.) |
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
| Does geometric forgetting enable non-stationary adaptation? | Yes — autonomous re-routing under reward drift and cost drift, with cross-seed variance 3× lower than best non-stationary baseline |
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
| User | `~/.paretobandit/priors/user_priors.npz` | Your learned updates |

### Adding New Models

```python
router.add_model("openai/gpt-5", clone_from="openai/gpt-4o")
```

### Integrity and Recovery

Bundled priors are checksummed via `paretobandit/data/priors/manifest.json`. To verify:

```bash
python -m paretobandit.core.cli verify-priors
```

To restore from git if needed:

```bash
git show <ref>:paretobandit/data/priors/expert_priors.npz > expert_priors.npz
```

Priors are loaded with `allow_pickle=False` and use fixed-width arrays for security.

---

## Documentation

- **[API Reference](docs/API_REFERENCE.md)** — Complete reference for all public classes, methods, parameters, return types, and exceptions.
- **[Contributing Guide](CONTRIBUTING.md)** — Development setup, code style, and pull request workflow.
- **[Changelog](CHANGELOG.md)** — Version history and release notes.

---

## Project Structure

```
paretobandit/
├── src/pareto_bandit/          # Core library
│   ├── router.py            # BanditRouter, LinUCB, BudgetPacer
│   ├── providers/           # LLMClient protocol + provider adapters
│   ├── storage.py           # SQLite context persistence
│   ├── feature_service.py   # Prompt embedding + PCA
│   ├── calibration.py       # train_pca(), generate_warmup_priors() for custom encoders
│   ├── baselines.py         # Baseline comparison implementations
│   └── utils/               # Warmup, heuristics
├── docs/                    # API reference and documentation
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

ParetoBandit uses a "Binary + Log" transformation (φ(x) = [𝟙(x>0), ln(1+x)]) rather than binning or piecewise linearization, for three reasons:

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

**Warmup mode** uses a dense covariance matrix from 20,000 simulated interactions (740,000 Bayesian updates). The covariance encodes learned correlations (code ↔ math, reasoning ↔ complexity). Scaling via `prior_n_effective` controls plasticity:

```
θ_scaled = (A × s)⁻¹(b × s) = A⁻¹b = θ_raw   (preferences preserved)
```

Without scaling (raw N=20,000), each real observation contributes only 0.005% — the router becomes "deaf" to new data. With N=1,000 scaling, the ratio is 0.1% — stable yet adaptive.

### Geometric Forgetting: Formal Properties

At each step, before incorporating the new observation, the sufficient statistics for the selected arm are exponentially discounted:

```
A_a ← γ · A_a + x · xᵀ
b_a ← γ · b_a + r · x
```

The forgetting factor γ ∈ (0, 1] gives observations an effective half-life of ln(2) / (1 − γ) steps. With the default γ = 0.995, this creates a memory horizon of ~200 steps: recent evidence dominates, allowing the router to track reward drift, while historical observations (including warmup priors) are geometrically discounted rather than erased.

Crucially, forgetting interacts with the cached inverse: discounting A by γ is equivalent to scaling A⁻¹ by 1/γ, a scalar division that keeps the per-update cost at O(d²) without a full matrix rebuild.

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
