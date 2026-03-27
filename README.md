# ParetoBandit: Budget-Paced Adaptive Routing for Non-Stationary LLM Serving

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-mkdocs-blue.svg)](https://ParetoBandit.github.io/ParetoBandit/)

**ParetoBandit** is an open-source, cost-aware contextual bandit router for LLM serving.
It enforces dollar-denominated per-request budgets, adapts online to price and quality shifts,
and onboards new models at runtime — all with sub-millisecond routing latency on CPU.

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

```bash
pip install paretobandit
```

With optional sentence-transformer embeddings:

```bash
pip install paretobandit[embeddings]
```

For development (from source):

```bash
git clone https://github.com/ParetoBandit/ParetoBandit.git
cd ParetoBandit
pip install -e ".[dev]"
```

---

## Quick Start

```python
from pareto_bandit import BanditRouter

# Create a router with default settings (cold start, safe exploration)
router = BanditRouter.create()

# Route a prompt — returns (selected_model, routing_log)
model, log = router.route("Explain the transformer architecture", max_cost=0.01)
print(f"Model: {model}, Cost: ${log.cost_usd:.6f}")

# After observing quality, feed back a reward to update the bandit
router.update(log.context_id, reward=0.85)
```

**CLI usage:**

```bash
# Route a prompt
paretobandit "Summarize this document" --max-cost 0.005

# Download embedding model for offline/Docker use
paretobandit --download-models
```

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

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for development setup,
coding standards, and the pull request workflow. By participating you agree to abide by the
[Code of Conduct](CODE_OF_CONDUCT.md).

---

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
