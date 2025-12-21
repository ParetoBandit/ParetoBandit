# Table Generation Scripts for KDD Paper

This folder contains all scripts used to generate the tables and numerical results in the paper **"Democratizing LLM Access: Adaptive Routing with Shippable Priors"**.

## 📊 Table-to-Script Mapping

### Main Paper Tables

| Table | Script | Description | Output |
|-------|--------|-------------|--------|
| **Table 1: Benchmark Trap** | `run_rq1.py` | Shows benchmark initialization hurts vs. expert distillation | Regret metrics by initialization type |
| **Table 2: SOTA Comparison** | `run_sota_comparison.py` | Compares BanditGPT vs. FrugalGPT vs. RouteLLM | Cost, accuracy, model pool size |
| **Table 3: Domain Breakdown** | `run_sota_comparison.py` | Performance across Math, Reasoning, Instructions | Domain-specific accuracy metrics |
| **Table 4: System Impact** | `run_rq3.py` | Cost reduction and quality retention | Cost/1k, accuracy, relative savings |
| **Table 5: ROI Leaderboard** | `run_rq3.py` | Top 15 models ranked by ROI factor | Model name, cost, ‖θ‖, ROI factor |
| **Table 6: Latency Breakdown** | `benchmark_latency.py` | Router overhead analysis | P50/P95/P99 latencies by component |
| **Table 7: Scaling Comparison** | Manual (method.tex) | Fixed cascade vs. dynamic routing | Qualitative comparison |

### Use Case Tables

| Table | Location | Description | Source |
|-------|----------|-------------|--------|
| **Use Case Summary** | `use_cases_CONCISE.tex` | Impact across 4 user types | Manual composition |
| **Accessibility Comparison** | `use_cases_CONCISE.tex` | Requirements: FrugalGPT vs. RouteLLM vs. Ours | Manual composition |

### Appendix Tables

| Table | Script | Description | Output |
|-------|--------|-------------|--------|
| **Full ROI Leaderboard** | `run_rq3.py` | All 81 models ranked | Complete ROI rankings |
| **Detailed Latency Benchmarks** | `benchmark_latency.py` | Full latency breakdown | 1,000-run statistics |
| **Hyperparameter Sensitivity** | `run_rq1.py` or `calibrate_multi_domain.py` | Effect of λ_boost and α | Regret reduction by parameter |

## 📂 Script Categories

### 1. Main Experiment Scripts (Generate Core Tables)

#### `run_rq1.py`
**Research Question 1: Warm-Start Effectiveness**
- **Tables Generated:**
  - Table 1: Benchmark Trap (initialization comparison)
  - Regret reduction percentages
  - Appendix: Hyperparameter sensitivity
- **Output Directory:** `results/rq1/`
- **Key Metrics:** Cumulative regret, regret reduction %, initialization methods
- **Runtime:** ~15-30 minutes

**Tables Generated:**
```
Initialization       | Regret | vs. Cold Start
---------------------|--------|---------------
Cold Start (Random)  | 63.7   | ---
Benchmark-Init       | 83.5   | -31% (worse)
Expert-Distilled     | 20.4   | +68%
```

**Usage:**
```bash
cd /Users/annette/repostitories/llm_jury
python experiments/run_rq1.py

# Results saved to: results/rq1/metrics.json
# Look for: initialization_comparison, regret_reduction
```

#### `run_rq2.py`
**Research Question 2: Plasticity Analysis**
- **Tables Generated:**
  - Recovery time statistics
  - Poisoned prior recovery metrics
- **Output Directory:** `results/rq2/`
- **Key Metrics:** Belief update dynamics, steps to recovery
- **Runtime:** ~20-40 minutes

**Usage:**
```bash
cd /Users/annette/repostitories/llm_jury
python experiments/run_rq2.py

# Results saved to: results/rq2/plasticity_metrics.json
```

#### `run_rq3.py`
**Research Question 3: Cost-Quality Trade-offs**
- **Tables Generated:**
  - Table 4: System Impact (cost reduction, accuracy retention)
  - Table 5: ROI Leaderboard (top 15 specialists)
  - Appendix: Full ROI Leaderboard (all 81 models)
- **Output Directory:** `results/rq3/` and `kdd_paper/data/rq3_cost_quality.json`
- **Key Metrics:** Cost per 1k queries, accuracy %, ROI factor
- **Runtime:** ~15-30 minutes

**Tables Generated:**

**System Impact:**
```
Metric                    | GPT-4o Only | BanditGPT | Improvement
--------------------------|-------------|-----------|------------
Cost per 1k Queries       | $4.38       | $0.70     | 84% reduction
System Accuracy           | 96.5%       | 95.8%     | -0.7pp
Math Accuracy             | 94.2%       | 95.1%     | +0.9pp
```

**ROI Leaderboard (Top 5):**
```
Rank | Model          | Cost/1M  | ‖θ‖  | ROI Factor
-----|----------------|----------|------|------------
1    | nova-micro-v1  | $0.061   | 2.16 | 93.3x ★
2    | nova-lite-v1   | $0.105   | 3.66 | 91.6x ★
3    | llama-3.2-1b   | $0.053   | 1.48 | 73.6x ★
4    | llama-3.2-3b   | $0.060   | 1.64 | 71.8x ★
5    | deepseek-r1    | $0.068   | 1.30 | 50.5x ★
```

**Usage:**
```bash
cd /Users/annette/repostitories/llm_jury
python experiments/run_rq3.py

# Results saved to:
# - results/rq3/system_impact.json
# - results/rq3/roi_leaderboard.json
# - kdd_paper/data/rq3_cost_quality.json
```

### 2. Specialized Benchmarking Scripts

#### `benchmark_latency.py`
**Generates:** Table 6 (Latency Breakdown) + Appendix (Detailed Latency)
- **Components Measured:**
  - Embedding generation (sentence-transformers)
  - UCB selection (81 models, 384 dimensions)
  - Total router overhead
- **Statistics:** P50, P95, P99 over 1,000 runs
- **Output Directory:** `kdd_paper/data/latency_benchmark.json`
- **Runtime:** ~5-10 minutes

**Table Generated:**
```
Component           | P50    | P95    | P99    | % of Total
--------------------|--------|--------|--------|------------
Embedding           | 3.2ms  | 4.1ms  | 5.7ms  | 63.8%
UCB Selection       | 2.1ms  | 3.5ms  | 3.2ms  | 35.1%
Total Router        | 5.3ms  | 7.6ms  | 8.9ms  | 100%
Model Inference     | 482ms  | 731ms  | 845ms  | ---
Router Overhead     | 1.1%   | 1.0%   | 1.1%   | ---
```

**Usage:**
```bash
cd /Users/annette/repostitories/llm_jury/kdd_paper/scripts
python benchmark_latency.py

# Results saved to: kdd_paper/data/latency_benchmark.json
```

#### `run_sota_comparison.py`
**Generates:** Table 2 (SOTA Comparison) + Table 3 (Domain Breakdown)
- **Systems Compared:**
  - BanditGPT Standard Mode
  - BanditGPT Hybrid Mode
  - FrugalGPT
  - RouteLLM
  - GPT-4o Only
- **Metrics:** Cost/1k, accuracy, model pool size, setup time
- **Output Directory:** `results/sota_comparison/`
- **Runtime:** ~30-60 minutes

**Tables Generated:**

**SOTA Comparison:**
```
System              | Cost/1k | Accuracy | Models | Setup
--------------------|---------|----------|--------|-------
BanditGPT Standard  | $0.70   | 95.8%    | 81     | 0 min
BanditGPT Hybrid    | $1.20   | 98.1%    | 81     | 0 min
FrugalGPT          | $1.78   | 97.8%    | 5      | 2-3 days
RouteLLM           | $2.20   | 96.4%    | 2      | 1-2 days
GPT-4o Only        | $4.38   | 96.5%    | 1      | 0 min
```

**Domain Breakdown:**
```
Domain        | BanditGPT | FrugalGPT | Δ
--------------|-----------|-----------|-----
Math          | 95.1%     | 94.2%     | +0.9pp
Reasoning     | 96.4%     | 96.1%     | +0.3pp
Instructions  | 98.0%     | 95.0%     | +3.0pp ★
Overall       | 96.5%     | 95.1%     | +1.4pp
```

**Usage:**
```bash
cd /Users/annette/repostitories/llm_jury/kdd_paper/scripts
python run_sota_comparison.py

# Results saved to: results/sota_comparison/
# - comparison_table.json
# - domain_breakdown.json
```

### 3. Validation and Calibration Scripts

#### `validate_benchmarks.py`
**Purpose:** Validates benchmark scores against published results
- Ensures MMLU/GSM8K/HumanEval scores match public leaderboards
- Generates discrepancy reports
- **Output Directory:** `results/validation/`
- **Runtime:** ~10-20 minutes

**Usage:**
```bash
cd /Users/annette/repostitories/llm_jury/kdd_paper/scripts
python validate_benchmarks.py

# Results: validation/benchmark_validation.json
```

#### `calibrate_multi_domain.py`
**Purpose:** Generates domain-specific calibration data
- Trains separate priors for Math, Reasoning, Instructions
- Tests cross-domain generalization
- May contribute to hyperparameter sensitivity table
- **Output Directory:** `results/multi_domain/`
- **Runtime:** ~30-60 minutes

**Usage:**
```bash
cd /Users/annette/repostitories/llm_jury/kdd_paper/scripts
python calibrate_multi_domain.py

# Results: results/multi_domain/calibration_metrics.json
```

## 📁 Data Files

Generated data files are stored in:

### `/Users/annette/repostitories/llm_jury/kdd_paper/data/`
- **`latency_benchmark.json`** - Latency statistics (1,000 runs)
- **`rq1_metrics.json`** - RQ1 regret and initialization data
- **`rq3_cost_quality.json`** - RQ3 system impact and ROI data

### `/Users/annette/repostitories/llm_jury/results/`
- **`rq1/`** - RQ1 experimental results
- **`rq2/`** - RQ2 plasticity analysis
- **`rq3/`** - RQ3 cost-quality trade-offs
- **`sota_comparison/`** - Baseline comparisons

### `/Users/annette/repostitories/llm_jury/kdd_paper/tables/`
- **`table1_roi_leaderboard.md`** - Markdown version of ROI table
- **`table2_system_impact.md`** - Markdown version of system impact
- **`table3_latency_overhead.md`** - Markdown version of latency
- **`table4_qualitative_routing.md`** - Markdown version of comparisons

## 🚀 Quick Start: Regenerate All Tables

To regenerate all table data from scratch:

```bash
#!/bin/bash
# Navigate to project root
cd /Users/annette/repostitories/llm_jury

echo "Generating RQ1 tables (Benchmark Trap, Hyperparameters)..."
python experiments/run_rq1.py

echo "Generating RQ2 tables (Plasticity)..."
python experiments/run_rq2.py

echo "Generating RQ3 tables (System Impact, ROI Leaderboard)..."
python experiments/run_rq3.py

echo "Generating SOTA comparison tables..."
cd kdd_paper/scripts
python run_sota_comparison.py

echo "Benchmarking latency..."
python benchmark_latency.py

echo "Validating benchmark scores..."
python validate_benchmarks.py

echo "✅ All tables generated!"
echo "Check results/ and kdd_paper/data/ directories"
```

Save this as `regenerate_all_tables.sh` and run:
```bash
chmod +x regenerate_all_tables.sh
./regenerate_all_tables.sh
```

## 📝 Table Formatting

### From JSON to LaTeX

Most scripts output JSON files. To convert to LaTeX:

```python
import json
import pandas as pd

# Load data
with open('results/rq3/roi_leaderboard.json') as f:
    data = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(data)

# Generate LaTeX table
latex = df.to_latex(
    index=False,
    column_format='llccc',
    escape=False,
    float_format='%.2f'
)

print(latex)
```

### Manual Tables

Some tables are manually composed in LaTeX:
- **Use Case Summary** - Qualitative impact assessment
- **Accessibility Comparison** - Operational requirements
- **Scaling Comparison** - Architectural trade-offs

These are directly in the `.tex` files and don't have corresponding scripts.

## ⏱️ Estimated Runtimes

| Script | Runtime | Priority | Tables |
|--------|---------|----------|--------|
| `run_rq1.py` | 15-30 min | ⭐⭐⭐ Essential | Table 1 |
| `run_rq2.py` | 20-40 min | ⭐⭐ Important | Plasticity metrics |
| `run_rq3.py` | 15-30 min | ⭐⭐⭐ Essential | Tables 4, 5 |
| `run_sota_comparison.py` | 30-60 min | ⭐⭐⭐ Essential | Tables 2, 3 |
| `benchmark_latency.py` | 5-10 min | ⭐⭐⭐ Essential | Table 6 |
| `validate_benchmarks.py` | 10-20 min | ⭐ Supplementary | Validation reports |
| `calibrate_multi_domain.py` | 30-60 min | ⭐ Supplementary | Hyperparameter table |

**Total for all essential tables:** ~2-3 hours

## 🔍 Understanding Output Files

### `latency_benchmark.json`
```json
{
  "embedding": {"p50": 3.2, "p95": 4.1, "p99": 5.7},
  "ucb_selection": {"p50": 2.1, "p95": 3.5, "p99": 3.2},
  "total_router": {"p50": 5.3, "p95": 7.6, "p99": 8.9},
  "overhead_pct": 1.1
}
```

### `rq3_cost_quality.json`
```json
{
  "system_impact": {
    "gpt4o_only": {"cost": 4.38, "accuracy": 96.5},
    "banditgpt": {"cost": 0.70, "accuracy": 95.8},
    "improvement": {"cost_reduction_pct": 84.0, "accuracy_delta": -0.7}
  },
  "roi_leaderboard": [
    {"rank": 1, "model": "nova-micro-v1", "cost": 0.061, "theta_norm": 2.16, "roi": 93.3},
    ...
  ]
}
```

## 🐛 Troubleshooting

**Problem:** Missing data files  
**Solution:** Run the corresponding experiment script first (e.g., `run_rq3.py` before checking `rq3_cost_quality.json`)

**Problem:** JSON parsing errors  
**Solution:** Ensure scripts completed successfully (check for error messages in output)

**Problem:** Numbers don't match paper  
**Solution:** Some values in the paper are rounded or averaged across multiple runs. Check the exact configuration in the paper.

## 📧 Related Files

- Plotting scripts: `/Users/annette/repostitories/llm_jury/kdd_paper/plotting_scripts/`
- Paper source: `/Users/annette/repostitories/llm_jury/kdd_paper/paper_submitted/`
- Main README: `/Users/annette/repostitories/llm_jury/README.md`

---

**Quick Start:** Run `./regenerate_all_tables.sh` from this directory to regenerate all table data.

**Last Updated:** December 2025  
**Paper Version:** Concise (8-page main content)  
**Location:** `/Users/annette/repostitories/llm_jury/kdd_paper/table_scripts/`

