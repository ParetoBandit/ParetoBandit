# Plotting Scripts for KDD Paper

This folder contains all scripts used to generate the figures in the paper **"Democratizing LLM Access: Adaptive Routing with Shippable Priors"**.

## 📊 Figure-to-Script Mapping

### Main Paper Figures

| Figure | Script | Description | Output Files |
|--------|--------|-------------|--------------|
| **Figure 1: Regret Curves** | `run_rq1.py` | Cumulative regret over time comparing warm-start vs cold-start | `figure1_regret_curve.pdf/.png` |
| **Figure 2: Belief Recovery** | `run_rq2.py` | Plasticity under concept drift (poisoned priors recovery) | `figure2_belief_recovery.pdf/.png` |
| **Figure 3: Specialist Landscape** | `run_rq1.py` or generate manually | Cost-quality scatter plot of 81 models | `figure3_specialist_landscape.pdf/.png` |
| **Figure 4: Pareto Frontier** | `run_needle_in_haystack.py` or `run_pareto_experiment.py` | Cost-quality trade-off curves | `figure4_pareto_frontier.pdf/.png` |

### Appendix Figures

| Figure | Script | Description | Output Files |
|--------|--------|-------------|--------------|
| **Figure 7: SLA Tunability** | `plot_sla_figures.py` | Budget and quality constraint demonstrations | `figure7_sla_tunability.pdf/.png` |
| **Figure 8: FinOps Constraints** | `plot_sla_figures.py` | Multi-objective constraint handling | `figure8_finops_constraints.pdf/.png` |
| **Confident Failure Trap** | `create_trap_diagram.py` | Benchmark initialization pathology diagram | `figure_confident_failure.pdf/.png` |

### Architecture Diagrams (TikZ)

The following diagrams are generated using LaTeX/TikZ (source in `paper_submitted/figures/`):
- **architecture_diagram.pdf** - System architecture overview
- **distillation_diagram.pdf** - Prior distillation process
- **decision_tree_diagram.pdf** - Hybrid mode decision logic

## 📂 Script Categories

### 1. Main Experiment Scripts (Generate Core Results)

#### `run_rq1.py` 
**Research Question 1: Warm-Start Effectiveness**
- Generates: Figure 1 (regret curves), Figure 3 (specialist landscape)
- Output directory: `results/rq1/`
- Key metrics: Cumulative regret, regret reduction percentages
- Runtime: ~15-30 minutes

**Usage:**
```bash
cd /Users/annette/repostitories/llm_jury
python experiments/run_rq1.py
```

#### `run_rq2.py`
**Research Question 2: Plasticity Under Normal Conditions**
- Generates: Figure 2 (belief recovery curves)
- Output directory: `results/rq2/`
- Key metrics: Belief update dynamics, adaptation speed
- Runtime: ~20-40 minutes

**Usage:**
```bash
cd /Users/annette/repostitories/llm_jury
python experiments/run_rq2.py
```

#### `run_rq2_poisoned.py`
**Research Question 2: Plasticity Under Adversarial Conditions**
- Generates: Poisoned prior recovery plots
- Output directory: `results/rq2/`
- Tests robustness to deliberately corrupted priors
- Runtime: ~20-40 minutes

#### `run_rq3.py`
**Research Question 3: Cost-Quality Trade-offs**
- Generates: Cost reduction and quality metrics
- Output directory: `results/rq3/`
- Key metrics: Cost per 1k queries, accuracy retention
- Runtime: ~15-30 minutes

**Usage:**
```bash
cd /Users/annette/repostitories/llm_jury
python experiments/run_rq3.py
```

### 2. Specialized Plot Generation Scripts

#### `plot_sla_figures.py`
**Generates:** Figures 7 & 8 (SLA and FinOps constraint demonstrations)
- Figure 7: Budget constraint tuning (`max_budget`)
- Figure 8: Quality + latency + budget multi-objective optimization
- Output directory: `kdd_paper/figures/`
- Input: Reads from `data/` or `results/` directories

**Usage:**
```bash
cd /Users/annette/repostitories/llm_jury/kdd_paper/scripts
python plot_sla_figures.py
```

#### `plot_statistical_comparison.py`
**Generates:** Statistical comparison plots (supplementary)
- Box plots for cost/latency distributions
- Confidence intervals for accuracy comparisons
- Router behavior analysis
- Output directory: `results/figures/`

**Usage:**
```bash
cd /Users/annette/repostitories/llm_jury/kdd_paper/scripts
python plot_statistical_comparison.py
```

#### `create_trap_diagram.py`
**Generates:** Benchmark initialization trap diagram
- Illustrates the "confident failure" problem
- Shows why MMLU/GSM8K scores don't predict routing quality
- Output: `figure_confident_failure.pdf/.png`

**Usage:**
```bash
cd /Users/annette/repostitories/llm_jury/kdd_paper/scripts
python create_trap_diagram.py
```

### 3. Alternative Analysis Scripts

#### `run_pareto_experiment.py`
**Generates:** Alternative Pareto frontier analysis
- Similar to Figure 4 but with different trade-off exploration
- Output directory: `results/pareto/`

#### `run_needle_in_haystack.py`
**Generates:** Needle-in-haystack stress test + Pareto curves
- Tests routing with rare high-quality specialists
- Generates Figure 4 (Pareto frontier)
- Output directory: `results/needle_in_haystack/`

#### `run_rq1_benchmark_trap.py`
**Generates:** Benchmark trap analysis (referenced in RQ1)
- Compares benchmark-initialized vs warm-start priors
- Output directory: `results/rq1_benchmark_trap/`

## 🎨 Figure Style Guidelines

All plotting scripts follow these conventions:
- **DPI:** 150 for PNG (screen), 300 for publication quality
- **Format:** Both PDF (vector) and PNG (raster) generated
- **Font:** Arial or system default sans-serif
- **Color palette:** Colorblind-friendly (matplotlib default tab10)
- **Size:** Optimized for two-column ACM format

## 🔧 Dependencies

All scripts require:
```bash
pip install -r /Users/annette/repostitories/llm_jury/requirements.txt
```

Key dependencies:
- `matplotlib` >= 3.5.0
- `numpy` >= 1.21.0
- `pandas` >= 1.3.0
- `seaborn` >= 0.11.0 (for statistical plots)

## 📁 Output Directories

Figures are saved to multiple locations:
1. **Source results:** `/Users/annette/repostitories/llm_jury/results/<experiment>/`
2. **Figure collection:** `/Users/annette/repostitories/llm_jury/kdd_paper/figures/`
3. **Paper directory:** `/Users/annette/repostitories/llm_jury/kdd_paper/paper_submitted/figures/`

## 🚀 Quick Start: Regenerate All Paper Figures

To regenerate all figures from scratch:

```bash
#!/bin/bash
# Navigate to project root
cd /Users/annette/repostitories/llm_jury

# Run main experiments (generates Figures 1-4)
echo "Running RQ1 (Figure 1, 3)..."
python experiments/run_rq1.py

echo "Running RQ2 (Figure 2)..."
python experiments/run_rq2.py

echo "Running RQ3 (cost-quality metrics)..."
python experiments/run_rq3.py

echo "Running Pareto experiment (Figure 4)..."
python kdd_paper/scripts/run_needle_in_haystack.py

# Generate supplementary figures
cd kdd_paper/scripts

echo "Generating SLA figures (Figures 7, 8)..."
python plot_sla_figures.py

echo "Creating trap diagram..."
python create_trap_diagram.py

echo "Generating statistical comparisons..."
python plot_statistical_comparison.py

echo "✅ All figures generated!"
echo "Check results/ and kdd_paper/figures/ directories"
```

Save this as `regenerate_all_figures.sh` and run:
```bash
chmod +x regenerate_all_figures.sh
./regenerate_all_figures.sh
```

## 📝 Notes

### Figure Customization
- Most scripts have configurable parameters at the top (DPI, colors, sizes)
- Edit the `FIGURE_CONFIG` or similar dictionaries in each script
- Rerun the script to regenerate with new settings

### Data Sources
- Experiment scripts generate their own data by running the routing system
- Some plotting scripts read pre-computed results from `results/` directories
- If data is missing, rerun the corresponding experiment script first

### Reproducibility
- Random seeds are set in each experiment script for reproducibility
- Results may vary slightly due to API call variability (GPT-4o judgments)
- Expect <2% variation in numerical results across runs

## 🐛 Troubleshooting

**Problem:** Script can't find data files
**Solution:** Run the corresponding experiment script first (e.g., `run_rq1.py` before plotting)

**Problem:** Import errors
**Solution:** Ensure you're running from the correct directory and have installed all dependencies

**Problem:** Figures look different from paper
**Solution:** Check DPI settings and ensure you're using the correct color scheme

## 📧 Contact

For questions about figure generation:
- Check the paper supplementary materials
- Refer to the main README in `/Users/annette/repostitories/llm_jury/`
- Examine the docstrings in each script

---

**Last Updated:** December 2025  
**Paper Version:** Concise (8-page main content)  
**Location:** `/Users/annette/repostitories/llm_jury/kdd_paper/plotting_scripts/`

