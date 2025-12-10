# Data Section for KDD Paper

This directory contains the complete Data section for the KDD 2025 paper on LLM Jury: Intent-Aware Multi-Model Routing.

## Files

1. **DATA_SECTION.md** - Complete data section in Markdown format
   - Comprehensive description of all data sources
   - Collection methodologies
   - Quality assurance procedures
   - Statistical methods (Bayesian Latent Factor models)
   - Reproducibility and ethics considerations

2. **data_section.tex** - LaTeX version ready for paper compilation
   - Formatted for ACM conference proceedings
   - Includes all tables and equations
   - Cross-references to other sections
   - Bibliography citations

## Dataset Overview

- **Total Models**: 83 production-ready language models (source of truth: `models_cache.json`)
- **Model Cache**: 83 models with complete operational metadata
- **Composite Scores Computed**:
  - CCS (Coding): 98 models
  - CRS (Reasoning): 100 models
  - CFS (Factual): 98 models
  - CSS (Summarization): 61 models

## Data Sources

### Primary Benchmarks (Direct Evaluation)
- **HumanEval** (69 models): Code generation (pass@1)
- **MBPP** (69 models): Python programming problems
- **SummEdits** (61 models): Factual consistency across 10 domains
- **MixEval** (45 models): Multi-domain understanding

### Aggregated Quality Indices
- **Artificial Analysis API**: Intelligence, Coding, Math indices (83 models, 100% coverage)
- **Arena-Hard-Auto**: Creative writing proxy (23 models)

### Operational Metadata
- **Pricing**: Input/output costs per 1M tokens (83 models, 100%)
- **Latency**: TTFT and throughput measurements (83 models, 100%)

### Safety & Validation
- **Hallucination Rate**: Vectara leaderboard (83 models, 100%)
- **Arena ELO**: Human preference ground truth (31 models, 31%)

## Key Methodological Contributions

1. **Bayesian Latent Factor (BLF) Models**
   - Handles missing data principally (no listwise deletion)
   - Learns benchmark weights from data (no manual tuning)
   - Quantifies uncertainty via full posterior distributions
   - Achieves ρ = 0.89 correlation with human preferences

2. **Optimization-Based Weighting**
   - Correlation maximization via regularized regression
   - Constrained optimization with safety/cost constraints
   - Intent-specific weights (ρ = 0.91-0.94 per intent)

3. **Comprehensive Quality Assurance**
   - Cross-validation against published results (MAE < 1.2%)
   - Multiple independent quality signals
   - Temporal consistency tracking
   - Robust outlier detection

## Reproducibility

All data and code are available:
- Data files: `data/*.json`, `data/*.csv`
- ETL pipeline: `llm_jury/etl/`
- BLF implementation: `llm_jury/analysis/latent_factor.py`
- Optimization: `llm_jury/optimization/`

## Recent Updates

**December 10, 2025**:
- Removed WildBench (wb_*) scores from cache (18 models affected)
- Removed IFEval scores from cache (45 models affected)
- Updated all statistics to reflect actual model counts
- Corrected benchmark coverage percentages
- Updated validation table with accurate N models

## Usage in Paper

The data section is designed to be Section 3 of the KDD paper. It provides:
- Complete methodological rigor for expert reviewers
- Reproducibility details for practitioners
- Ethical considerations (bias, privacy, environmental impact)
- Statistical validation of all claims

## Contact

For questions about data sources or methodologies, see the main repository README or open an issue.
