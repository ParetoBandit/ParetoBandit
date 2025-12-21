# Quick Table Index

## 🎯 Which Script Generates Which Table?

### Main Paper Tables

- **Table 1: Benchmark Trap** → `run_rq1.py`
- **Table 2: SOTA Comparison** → `run_sota_comparison.py`
- **Table 3: Domain Breakdown** → `run_sota_comparison.py`
- **Table 4: System Impact** → `run_rq3.py`
- **Table 5: ROI Leaderboard** → `run_rq3.py`
- **Table 6: Latency Breakdown** → `benchmark_latency.py`
- **Table 7: Scaling Comparison** → Manual (in method.tex)

### Use Case Tables (Manual)

- **Use Case Summary** → `use_cases_CONCISE.tex` (manual)
- **Accessibility Comparison** → `use_cases_CONCISE.tex` (manual)

### Appendix Tables

- **Full ROI Leaderboard (81 models)** → `run_rq3.py`
- **Detailed Latency Benchmarks** → `benchmark_latency.py`
- **Hyperparameter Sensitivity** → `run_rq1.py` or `calibrate_multi_domain.py`

## 🚀 Quick Commands

### Regenerate Everything
```bash
./regenerate_all_tables.sh
```

### Individual Tables

```bash
# Table 1: Benchmark Trap
cd /Users/annette/repostitories/llm_jury
python experiments/run_rq1.py

# Tables 2 & 3: SOTA Comparison + Domain Breakdown
cd kdd_paper/scripts
python run_sota_comparison.py

# Tables 4 & 5: System Impact + ROI Leaderboard
cd /Users/annette/repostitories/llm_jury
python experiments/run_rq3.py

# Table 6: Latency Breakdown
cd kdd_paper/scripts
python benchmark_latency.py
```

## 📂 File Organization

```
table_scripts/
├── README.md                      # Comprehensive documentation
├── TABLE_INDEX.md                 # This file (quick reference)
├── regenerate_all_tables.sh       # One-command regeneration
│
├── Main Experiments (Generate Core Tables):
│   ├── run_rq1.py                 # RQ1: Benchmark trap
│   ├── run_rq2.py                 # RQ2: Plasticity metrics
│   └── run_rq3.py                 # RQ3: System impact + ROI
│
├── Specialized Benchmarks:
│   ├── benchmark_latency.py       # Latency breakdown table
│   └── run_sota_comparison.py     # SOTA + domain tables
│
└── Validation/Calibration:
    ├── validate_benchmarks.py     # Benchmark score validation
    └── calibrate_multi_domain.py  # Domain-specific calibration
```

## 📊 Output Data Files

| Data File | Location | Source Script | Tables |
|-----------|----------|---------------|--------|
| `latency_benchmark.json` | `kdd_paper/data/` | `benchmark_latency.py` | Table 6 |
| `rq1_metrics.json` | `kdd_paper/data/` | `run_rq1.py` | Table 1 |
| `rq3_cost_quality.json` | `kdd_paper/data/` | `run_rq3.py` | Tables 4, 5 |
| `comparison_table.json` | `results/sota_comparison/` | `run_sota_comparison.py` | Table 2 |
| `domain_breakdown.json` | `results/sota_comparison/` | `run_sota_comparison.py` | Table 3 |

## ⏱️ Estimated Runtimes

| Script | Runtime | Tables Generated |
|--------|---------|------------------|
| `run_rq1.py` | 15-30 min | Table 1 + hyperparameters |
| `run_rq3.py` | 15-30 min | Tables 4, 5 + appendix |
| `run_sota_comparison.py` | 30-60 min | Tables 2, 3 |
| `benchmark_latency.py` | 5-10 min | Table 6 + appendix |
| `run_rq2.py` | 20-40 min | Plasticity metrics |

**Total for all essential tables:** ~1.5-2.5 hours

## 📋 Table Reference Guide

### Table 1: The Benchmark Trap
**Shows:** Public benchmarks hurt initialization; expert distillation helps  
**Columns:** Initialization method, Regret, vs. Cold Start  
**Key Result:** Expert-distilled priors achieve +68% improvement

### Table 2: SOTA Comparison
**Shows:** BanditGPT vs. FrugalGPT vs. RouteLLM  
**Columns:** System, Cost/1k, Accuracy, Model Pool, Setup Time  
**Key Result:** Standard mode is cost leader; Hybrid matches cascade accuracy

### Table 3: Domain Breakdown
**Shows:** Performance across task types  
**Columns:** Domain, BanditGPT, FrugalGPT, Delta  
**Key Result:** Hybrid beats FrugalGPT on Instructions (+3pp)

### Table 4: System Impact
**Shows:** Cost reduction while maintaining quality  
**Columns:** Metric, GPT-4o Only, BanditGPT, Improvement  
**Key Result:** 84% cost reduction, -0.7pp accuracy (acceptable)

### Table 5: ROI Leaderboard
**Shows:** Top 15 specialists by value  
**Columns:** Rank, Model, Cost/1M, ‖θ‖, ROI Factor  
**Key Result:** Nova-Micro achieves 93.3× ROI vs. GPT-4o

### Table 6: Latency Breakdown
**Shows:** Router overhead analysis  
**Columns:** Component, P50, P95, P99, % of Total  
**Key Result:** 8.9ms P99 overhead = 1.1% of inference time

## 📝 Quick Checklist for Paper Submission

- [ ] Run `./regenerate_all_tables.sh` to ensure all data is current
- [ ] Verify JSON files exist in `kdd_paper/data/` and `results/`
- [ ] Check table formatting in LaTeX matches data files
- [ ] Ensure all numbers cited in paper match generated data
- [ ] Cross-reference with `kdd_paper/tables/*.md` files
- [ ] Compile paper and verify tables render correctly

## 🔗 Related Folders

- Plotting scripts: `/Users/annette/repostitories/llm_jury/kdd_paper/plotting_scripts/`
- Data files: `/Users/annette/repostitories/llm_jury/kdd_paper/data/`
- Markdown tables: `/Users/annette/repostitories/llm_jury/kdd_paper/tables/`
- Paper LaTeX: `/Users/annette/repostitories/llm_jury/kdd_paper/paper_submitted/`

---

**Quick Start:** Run `./regenerate_all_tables.sh` from this directory to regenerate all table data.

