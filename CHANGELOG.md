# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-03-29

### Added
- **`MissingCostError`**: `register_model()` and `BanditRouter.create()` now raise a clear exception with usage examples when a model has no cost information, instead of silently defaulting to $10/M tokens
- **"Bring Your Own Models" docs**: README, API Reference, and blog post now include examples for custom model registries with per-token costs
- **PCA customization docs**: All documentation (README, API Reference, blog, demo CLI, notebook) explains the default LMSYS-trained PCA and how to train/use your own
- **Embedding pipeline docs**: Default encoder (`all-MiniLM-L6-v2`, ~90 MB), custom encoder, and precomputed-vector paths are clearly documented everywhere
- **Citing ParetoBandit**: README includes a BibTeX `@software` entry for attribution
- **`scripts/run_all_docker_tests.sh`**: Single command to run all 12 Docker validation targets (3 install variants × 3 Python versions + README, API Reference, and demo Dockerfiles)

### Fixed
- **CI workflow**: Fixed invalid YAML indentation in `.github/workflows/ci.yml` that caused "No jobs were run"
- **CI lint/type errors**: Resolved 1,587 ruff lint errors and 102 mypy type errors
- **CI embeddings**: CI now installs `[embeddings]` extra so sentence-transformers tests can run
- **Integration tests**: Tests that don't exercise the embedding pipeline now use `FeatureService.for_precomputed()` or `skipif` guards so they pass on core-only installs
- **PyPI README rendering**: Fixed icon URL and in-repo links for proper display on pypi.org

## [0.1.0] - 2025-02-25

### Added
- **BanditRouter**: Adaptive LLM router using Hybrid LinUCB with contextual features
- **Warm-start priors**: Dense covariance matrix distilled from 80,000 RouteLLM battle outcomes (< 1 MB)
- **FeatureService**: SentenceTransformer embedding with PCA compression (1024D → 33D with default `BAAI/bge-m3`)
- **Self-healing PCA**: JIT calibration recovers from missing or mismatched artifacts
- **SqliteContextStore**: Disk-persisted context storage for delayed feedback (RLHF) with 7-day TTL
- **Snapshot-swap concurrency**: Lock-free routing during O(d³) matrix inversions (250× lock-time reduction)
- **Calibration API**: `train_pca()` and `generate_warmup_priors()` for custom encoder support
- **CLI**: `paretobandit` command with routing, model download, and prior verification
- **Progressive model registration**: Three-tier knowledge system (archetypes, T-shirt sizing, agnostic)
- **Feature contribution analysis**: `explain_decision()` for mathematical transparency
- **Optimization profiles**: `auto` and `custom` with configurable quality/cost/latency weights
- **Hard constraint filtering**: `max_cost`, `max_latency`, `quality_floor` enforcement
- **CheckpointManager**: Atomic-write state persistence with registry change detection
- **135+ tests** covering router workflow, feedback loops, concurrency, and numerical stability
- **Paper reproduction**: Full experiment suite mapping 1:1 to paper figures and tables
