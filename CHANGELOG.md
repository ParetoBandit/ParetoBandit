# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-02-25

### Added
- **BanditRouter**: Adaptive LLM router using Hybrid LinUCB with contextual features
- **Corralling meta-learner**: Hedges between warmup and tabula rasa experts for robustness under prior mismatch
- **Warm-start priors**: Dense covariance matrix distilled from 80,000 RouteLLM battle outcomes (< 1 MB)
- **FeatureService**: SentenceTransformer embedding with PCA compression (1024D → 33D with default `BAAI/bge-m3`)
- **Self-healing PCA**: JIT calibration recovers from missing or mismatched artifacts
- **SqliteContextStore**: Disk-persisted context storage for delayed feedback (RLHF) with 7-day TTL
- **Snapshot-swap concurrency**: Lock-free routing during O(d³) matrix inversions (250× lock-time reduction)
- **Calibration API**: `train_pca()` and `generate_warmup_priors()` for custom encoder support
- **CLI**: `banditgpt` command with routing, model download, and prior verification
- **Progressive model registration**: Three-tier knowledge system (archetypes, T-shirt sizing, agnostic)
- **Feature contribution analysis**: `explain_decision()` for mathematical transparency
- **Optimization profiles**: `auto` and `custom` with configurable quality/cost/latency weights
- **Hard constraint filtering**: `max_cost`, `max_latency`, `quality_floor` enforcement
- **CheckpointManager**: Atomic-write state persistence with registry change detection
- **135+ tests** covering router workflow, feedback loops, concurrency, and numerical stability
- **Paper reproduction**: Full experiment suite mapping 1:1 to paper figures and tables
