# Quick Script Index

## 🎯 Which Script Generates Which Figure?

### Main Paper Figures

- **Figure 1 (Regret Curves)** → `run_rq1.py`
- **Figure 2 (Belief Recovery)** → `run_rq2.py`  
- **Figure 3 (Specialist Landscape)** → `run_rq1.py`
- **Figure 4 (Pareto Frontier)** → `run_needle_in_haystack.py`

### Appendix Figures

- **Figure 7 (SLA Tunability)** → `plot_sla_figures.py`
- **Figure 8 (FinOps Constraints)** → `plot_sla_figures.py`
- **Confident Failure Diagram** → `create_trap_diagram.py`

### Architecture Diagrams (TikZ in LaTeX)

- `architecture_diagram.pdf` → Compiled from `paper_submitted/figures/architecture_diagram.tex`
- `distillation_diagram.pdf` → Compiled from `paper_submitted/figures/distillation_diagram.tex`
- `decision_tree_diagram.pdf` → Compiled from `paper_submitted/figures/decision_tree_diagram.tex`

## 🚀 Quick Commands

### Regenerate Everything
```bash
./regenerate_all_figures.sh
```

### Individual Figures

```bash
# Figure 1 & 3: RQ1 warm-start analysis
cd /Users/annette/repostitories/llm_jury
python experiments/run_rq1.py

# Figure 2: RQ2 plasticity analysis
python experiments/run_rq2.py

# Figure 4: Pareto frontier
python kdd_paper/scripts/run_needle_in_haystack.py

# Figures 7 & 8: SLA demonstrations
cd kdd_paper/scripts
python plot_sla_figures.py

# Trap diagram
python create_trap_diagram.py

# Statistical comparisons (supplementary)
python plot_statistical_comparison.py
```

## 📂 File Organization

```
plotting_scripts/
├── README.md                      # Comprehensive documentation
├── SCRIPT_INDEX.md               # This file (quick reference)
├── regenerate_all_figures.sh     # One-command regeneration
│
├── Main Experiments:
│   ├── run_rq1.py                # RQ1: Warm-start effectiveness
│   ├── run_rq2.py                # RQ2: Plasticity (normal)
│   ├── run_rq2_poisoned.py       # RQ2: Plasticity (adversarial)
│   └── run_rq3.py                # RQ3: Cost-quality trade-offs
│
├── Specialized Plots:
│   ├── plot_sla_figures.py       # SLA constraint demonstrations
│   ├── plot_statistical_comparison.py  # Statistical analysis plots
│   └── create_trap_diagram.py    # Benchmark trap visualization
│
└── Alternative Analyses:
    ├── run_pareto_experiment.py  # Alternative Pareto analysis
    ├── run_needle_in_haystack.py # Needle test + Pareto
    └── run_rq1_benchmark_trap.py # Benchmark trap analysis
```

## ⏱️ Estimated Runtimes

| Script | Runtime | Priority |
|--------|---------|----------|
| `run_rq1.py` | 15-30 min | ⭐⭐⭐ Essential |
| `run_rq2.py` | 20-40 min | ⭐⭐⭐ Essential |
| `run_rq3.py` | 15-30 min | ⭐⭐⭐ Essential |
| `run_needle_in_haystack.py` | 30-60 min | ⭐⭐⭐ Essential |
| `plot_sla_figures.py` | 2-5 min | ⭐⭐ Important |
| `create_trap_diagram.py` | 1-2 min | ⭐⭐ Important |
| `plot_statistical_comparison.py` | 3-5 min | ⭐ Supplementary |
| `run_rq2_poisoned.py` | 20-40 min | ⭐ Supplementary |

**Total for all essential figures:** ~2-3 hours

## 📋 Checklist for Paper Submission

- [ ] Run `./regenerate_all_figures.sh` to ensure all figures are current
- [ ] Verify all PDFs are in `kdd_paper/paper_submitted/figures/`
- [ ] Check figure quality (300 DPI for final submission)
- [ ] Compile LaTeX diagrams: `cd paper_submitted/figures && ./compile_diagrams.sh`
- [ ] Recompile paper: `cd paper_submitted/concise_version && ./compile.sh`
- [ ] Visual inspection: Open `main_CONCISE.pdf` and verify all figures render correctly

## 🔗 Related Files

- Main README: `/Users/annette/repostitories/llm_jury/README.md`
- Paper README: `/Users/annette/repostitories/llm_jury/kdd_paper/README.md`
- Data sources: `/Users/annette/repostitories/llm_jury/results/`
- LaTeX sources: `/Users/annette/repostitories/llm_jury/kdd_paper/paper_submitted/`

---

**Quick Start:** Run `./regenerate_all_figures.sh` from this directory to regenerate all figures.

