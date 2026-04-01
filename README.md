<p align="center">
  <img src="https://raw.githubusercontent.com/ParetoBandit/ParetoBandit/main/blog/figures/paretobandit_icon_transparent.png" alt="ParetoBandit" width="200">
</p>

# ParetoBandit: Budget-Paced Adaptive Routing for Non-Stationary LLM Serving

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-mkdocs-blue.svg)](https://ParetoBandit.github.io/ParetoBandit/)

**ParetoBandit** is an open-source router for multi-model LLM serving.
If you're calling different LLMs and care about cost *and* quality,
ParetoBandit keeps you under a dollar budget while learning which model
to call for which prompt — and adapts on the fly when prices change,
models degrade, or you add a new one.

Under the hood it's a cost-aware contextual bandit with an online budget
pacer, geometric forgetting for non-stationarity, and a hot-swap model
registry. Routing decisions take ~microseconds on CPU.

> **Paper:** *ParetoBandit: Budget-Paced Adaptive Routing for Non-Stationary LLM Serving*
> **Author:** Annette Taberner-Miller

---

## Key Features

- **Online budget control.** A primal–dual budget pacer enforces a per-request cost ceiling over an open-ended stream with closed-loop control — no offline penalty tuning required.
- **Non-stationarity resilience.** Geometric forgetting on sufficient statistics enables rapid adaptation to price cuts, quality regressions, and distribution shifts, bootstrapped from optional offline priors.
- **Runtime model onboarding.** A hot-swap registry lets operators add or remove models at runtime; the bandit's exploration bonus discovers each newcomer's niche from live traffic alone.
- **Sub-millisecond routing.** The routing decision takes ~μs on CPU; end-to-end latency (including embedding) is <1% of typical LLM inference time.

---

## Installation

ParetoBandit needs prompt embeddings to route by content. The default pipeline uses
[all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2),
a lightweight sentence-transformer (~90 MB download, ~175 MB on disk).
Install with PyTorch and sentence-transformers included:

```bash
pip install paretobandit[embeddings]
```

The model downloads automatically on first use. To pre-download (useful for Docker/CI):

```bash
paretobandit --download-models
```

**Other install options:**

```bash
pip install paretobandit[demo]        # embeddings + matplotlib for interactive demo
pip install paretobandit              # core only (for custom encoders or precomputed vectors)
```

If you already have an embedding pipeline (e.g., OpenAI embeddings, a fine-tuned encoder,
or precomputed vectors from an upstream service), install core-only and bring your own —
see [Feature Engineering](#feature-engineering) below.

For development (from source):

```bash
git clone https://github.com/ParetoBandit/ParetoBandit.git
cd ParetoBandit
pip install -e ".[dev]"
```

---

## Quick Start

```bash
pip install paretobandit[embeddings]
```

```python
from pareto_bandit import BanditRouter

router = BanditRouter.create(
    model_registry={
        "gpt-4o":         {"input_cost_per_m": 2.50, "output_cost_per_m": 10.00},
        "claude-3-haiku": {"input_cost_per_m": 0.25, "output_cost_per_m": 1.25},
        "llama-3-70b":    {"input_cost_per_m": 0.50, "output_cost_per_m": 0.50},
    },
    priors="none",
)

model, log = router.route("Explain quantum computing", max_cost=0.005)
print(f"→ {model}  (${log.cost_usd:.4f})")
```

Pass your models and their per-million-token costs. The router learns which to call
for each prompt from live traffic — no offline training or labelled data needed.
For zero-config defaults, `BanditRouter.create()` works out of the box.

### Feeding back rewards

After observing response quality, feed back a reward so the bandit can learn:

```python
router.process_feedback(log.request_id, reward=0.85)
```

### Adding models at runtime

New models are explored automatically and adopted if they earn their niche:

```python
router.register_model(
    "gemini-2.0-flash",
    speed="fast",
    input_cost_per_m=0.10,
    output_cost_per_m=0.40,
)
```

See the [API Reference](docs/API_REFERENCE.md) for the full cost specification options
(`blended_cost_per_m`, speed profiles, latency, and more).

**CLI usage:**

```bash
paretobandit "Summarize this document" --max-cost 0.005
paretobandit --download-models   # pre-download embedding model for Docker/CI
```

For CLI usage, custom embeddings, and the budget-pacer internals, keep reading below.

---

## Feature Engineering

The router needs a numeric representation of each prompt to learn which model handles which
kind of request. ParetoBandit supports three embedding paths, from turnkey to fully custom:

### 1. Default pipeline (requires `embeddings` extra)

Uses [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
(~90 MB download) with a shipped 25-component PCA projection, compressing 384-dim embeddings
to a 26-dim feature vector (25 PCA + 1 bias). The PCA was trained on ~46K prompts from the
[LMSYS Chatbot Arena](https://huggingface.co/datasets/lmsys/chatbot_arena_conversations)
dataset and ships inside the package. No configuration needed.

```python
router = BanditRouter.create()  # downloads model on first use, loads PCA automatically
```

### 2. Custom encoder

Bring any encoder function — no `sentence-transformers` dependency required. Raw embeddings are used directly (+ bias term); optionally pair with your own PCA artifact.

```python
from pareto_bandit import BanditRouter
from pareto_bandit.feature_service import FeatureService

# Without PCA (raw embeddings)
fs = FeatureService(custom_encoder=my_encode_fn, embedding_dim=768)

# With your own PCA
fs = FeatureService(custom_encoder=my_encode_fn, embedding_dim=768, pca_path="my_pca.joblib")

router = BanditRouter.create(feature_service=fs, priors="none")
```

### 3. Precomputed feature vectors

If you already have embeddings (e.g., from an upstream service), skip encoding entirely:

```python
import numpy as np
from pareto_bandit import BanditRouter
from pareto_bandit.feature_service import FeatureService

fs = FeatureService.for_precomputed(dimension=25)
router = BanditRouter.create(feature_service=fs, priors="none")

# Pass numpy arrays instead of strings
features = np.random.randn(25)
model, log = router.route(features, max_cost=0.01)
```

### Training your own PCA

The shipped PCA (`pca_25.joblib`) was trained on general-purpose LMSYS Arena prompts.
You may want to train your own PCA if:

- You are using a **different encoder** (the shipped PCA only matches `all-MiniLM-L6-v2`).
- Your prompts are **domain-specific** (e.g., medical, legal, code-only) and a PCA
  trained on your domain may capture more relevant variance.

```python
from pareto_bandit import train_pca

pca = train_pca(
    prompts=my_prompt_corpus,           # list[str], >=100 recommended
    encoder_model="your-model-name",    # or "all-MiniLM-L6-v2" for domain-specific PCA
    n_components=25,
    output_path="my_pca.joblib",
)

router = BanditRouter.create(
    context_model="your-model-name",
    pca_path="my_pca.joblib",
)
```

---

## API Overview

Full API documentation: **[API Reference](docs/API_REFERENCE.md)**

| Class / Function | Purpose |
|---|---|
| `BanditRouter.create()` | Factory for a fully initialized router (default or custom models) |
| `BanditRouter.route()` | Route a prompt to the best model under cost/latency constraints |
| `BanditRouter.process_feedback()` | Feed back a reward signal (supports delayed feedback) |
| `BanditRouter.register_model()` | Hot-add a model at runtime |
| `BanditRouter.exploit()` | Context manager for greedy evaluation (no exploration) |
| `FeatureService` | Embedding + PCA pipeline (default, custom encoder, or precomputed) |
| `FeatureService.for_precomputed()` | Lightweight service for pre-embedded vectors |
| `BudgetPacer` | Online primal-dual budget controller (hard/soft/adaptive modes) |
| `RouterConfig` | Hyperparameter dataclass (reward range, cost anchors, etc.) |
| `train_pca()` | Train a custom PCA artifact for a non-default encoder |
| `generate_warmup_priors()` | Build offline warmup priors from labelled data |
| `SqliteContextStore` | Production context store with TTL (for delayed RLHF feedback) |

---

## Architecture

```
src/pareto_bandit/
├── router.py            # BanditRouter — main entry point, arm selection, update loop
├── policy.py            # DisjointLinUCB, prior calibration
├── budget_pacer.py      # Online primal–dual budget pacer (hard/soft/adaptive modes)
├── feature_service.py   # SentenceTransformer embedding + PCA compression
├── calibration.py       # train_pca(), generate_warmup_priors()
├── storage.py           # SqliteContextStore (delayed feedback), EphemeralContextStore
├── costs.py             # Cost model and heuristics
├── rewards.py           # Reward normalization and aggregation
├── config/              # Model registry, default hyperparameters, packaged artifacts
└── utils/               # Validation, warmup, synthetic data generation
```

### Design Principles

| Principle | Mechanism |
|---|---|
| **Budget enforcement** | Primal–dual ascent on per-request cost ceiling; no horizon assumption |
| **Non-stationarity** | Geometric forgetting on A⁻¹ and b sufficient statistics |
| **Cold-start mitigation** | Optional warm-start priors from offline data (80K RouteLLM battles) |
| **Lock-minimal concurrency** | Snapshot-swap during O(d³) matrix inversions (250× lock-time reduction) |
| **Self-healing** | Missing PCA/prior artifacts trigger JIT recovery, not crashes |

---

## Reproducing Paper Experiments

All experiments map 1:1 to figures and tables in the paper. Results are deterministic given fixed seeds.

### Full Reproduction

```bash
python experiments/reproduce.py
```

This runs all experiments in dependency order, then regenerates LaTeX macros and publication figures.

### Selective Execution

```bash
# List available experiments
python experiments/reproduce.py --list

# Run a single experiment
python experiments/reproduce.py --only 01_stationary_budget_pacing

# Regenerate LaTeX + figures only (skip expensive simulations)
python experiments/reproduce.py --skip-run
```

### Experiment Overview

| Key | Section | Description |
|---|---|---|
| `hparam_optimization` | Appendix | Hyperparameter sweep with Pareto knee-point selection |
| `cost_heuristic_validation` | Appendix | Cost heuristic validation |
| `01_stationary_budget_pacing` | §4.1 | Stationary budget pacing across 7 budget ceilings |
| `02_budget_plus_drift` | §4.2 | Budget pacing under cost drift (10× price cut) |
| `03_catastrophic_failure` | §4.3 | Catastrophic quality regression detection and rerouting |
| `04_model_onboarding` | §4.4 | Runtime model onboarding (K=3 → K=4) |
| `warmup_ablation` | Appendix | Warmup priors vs. cold-start ablation |
| `prior_mismatch` | Appendix | Prior mismatch sensitivity analysis |
| `judge_robustness` | Appendix | Cross-judge regret comparison |
| `recovery_limit` | Appendix | Recovery limit under degradation |
| `latency_benchmark` | Appendix | Routing and end-to-end latency microbenchmark |

Each experiment directory contains:
- `run_*.py` — simulation script producing result JSONs
- `generate_latex.py` — reads results, emits `_autogen.tex` macros consumed by the paper
- `generate_figure.py` — reads results, produces PNG/PDF figures
- `results/` — output artifacts (JSON, figures, autogen LaTeX)

---

## Testing

```bash
# Full test suite
python -m pytest tests/ -v

# Skip slow tests
python -m pytest tests/ -v -m "not slow"

# With coverage
python -m pytest tests/ --cov=pareto_bandit --cov-report=term-missing

# Experiment regression tests
python -m pytest experiments/tests/ -v
```

---

## Project Structure

```
paretobandit/
├── src/pareto_bandit/       # Core Python package
├── experiments/             # Paper experiment suite
│   ├── reproduce.py         # Master orchestrator
│   ├── 01_–_04_*/           # Main experiments (§4)
│   ├── appendix/            # Appendix experiments
│   ├── utils/               # Shared simulation and LaTeX utilities
│   └── tests/               # Experiment regression tests
├── tests/                   # Unit and integration tests (135+)
├── paper/                   # LaTeX source for the MLSys paper
├── data_collection/         # Raw reward data and PCA training scripts
├── docs/                    # API reference
├── pyproject.toml           # Build config (Hatch), dependencies, tool settings
├── CONTRIBUTING.md          # Development guide
└── CHANGELOG.md             # Version history
```

---

## Requirements

- **Python** ≥ 3.10
- **Core:** numpy, joblib, scikit-learn, tqdm
- **Embeddings** (optional): torch, sentence-transformers, transformers
- **Experiments:** matplotlib, scipy, python-dotenv

Full dependency specifications are in [`pyproject.toml`](pyproject.toml).
A pinned lockfile for exact reproduction of paper results is available in [`requirements-lock.txt`](requirements-lock.txt).

---

## Citing ParetoBandit

If you use ParetoBandit in your research or product, please cite:

```bibtex
@software{taberner-miller2026paretobandit,
  author       = {Taberner-Miller, Annette},
  title        = {{ParetoBandit}: Budget-Paced Adaptive Routing for Non-Stationary {LLM} Serving},
  year         = {2026},
  url          = {https://github.com/ParetoBandit/ParetoBandit},
}
```

---

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for development setup,
coding standards, and the pull request workflow. By participating you agree to abide by the
[Code of Conduct](CODE_OF_CONDUCT.md).

---

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
