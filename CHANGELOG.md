# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2025-01-XX

### Added
- Initial release of BanditGPT
- LinUCB contextual bandit router for LLM model selection
- Multi-objective optimization (quality, cost, latency)
- Expert-distilled priors for 62% day-1 regret reduction
- Tiered grading system (soft + hard verifier)
- Optimization profiles: `quality_first`, `balanced`, `cost_saver`, `low_latency`
- Exploration rate controls: `static`, `safe`, `balanced`, `aggressive`
- CLI for model recommendations
- Prior management with bundled and user priors
- Support for 80+ LLM models

### Changed
- Renamed project from `llm_jury` to `banditgpt`

[Unreleased]: https://github.com/atabernermiller/banditgpt/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/atabernermiller/banditgpt/releases/tag/v0.1.0
