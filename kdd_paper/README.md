# KDD 2025: Density-Based Warm-Start for Adaptive LLM Routing

This folder contains all camera-ready figures, tables, and scripts for the KDD paper.

---

## Figures

| Figure | File | Description |
|--------|------|-------------|
| **Figure 1** | `figures/figure1_regret_curve.pdf` | RQ1: Warm-Start vs Cold-Start regret comparison (63.6% reduction) |
| **Figure 2** | `figures/figure2_belief_recovery.png` | RQ2: Simulation of belief recovery under poisoned priors |
| **Figure 3** | `figures/figure3_specialist_landscape.pdf` | RQ2: Learned specialist expertise (‖θ‖ norm + similarity heatmap) |
| **Figure 4** | `figures/figure4_pareto_frontier.pdf` | RQ3: Cost-Quality Pareto frontier (log-scale) |

### Figure Captions

**Figure 1**: *Cumulative regret comparison between cold-start (no priors) and warm-start (expert-distilled priors with λ_boost=50) agents over 2,000 routing decisions. The warm-start agent achieves 63.6% lower cumulative regret.*

**Figure 2**: *Simulation of Belief Recovery Under Poisoned Priors. Controlled experiment isolating the Plasticity-Stability Dilemma. The router initially clings to the poisoned prior (GPT-4o), experiences a performance dip, then discovers and switches to the true specialist (Nova-Lite).*

**Figure 3**: *Learned Expertise Landscape. Left: Amazon Nova-Lite develops a significantly larger weight norm (‖θ‖) than GPT-4o, indicating high confidence in specific latent regions. Right: Cosine similarity heatmap confirms Nova-Lite's weights are orthogonal to GPT-4o, proving distinct specialist niches.*

**Figure 4**: *The Cost-Quality Pareto Frontier. The router identifies specialist models like Nova-Lite offering maximal learned expertise (‖θ‖ ≈ 3.7) at minimal cost (<$0.10/1M tokens), achieving 100x cost reduction vs generalist baselines.*

---

## Tables

| Table | File | Description |
|-------|------|-------------|
| **Table 1** | `tables/table1_roi_leaderboard.md` | ROI Leaderboard: Expertise per dollar (model-centric) |
| **Table 2** | `tables/table2_system_impact.md` | System Impact: Static vs Adaptive Router (router-centric) |
| **Table 3** | `tables/table3_latency_overhead.md` | Computational Overhead: Router latency analysis |
| **Table 4** | `tables/table4_qualitative_routing.md` | Qualitative Analysis: Specialist wins vs fallbacks |

### Key Results

**Table 1 (ROI Leaderboard)**:
- Nova-Micro: **93.3x** better ROI than GPT-4o
- Nova-Lite: **91.6x** better ROI than GPT-4o

**Table 2 (System Impact)**:
- Adaptive Router: **+33.8% higher quality** than GPT-4o at 67.7% lower cost

**Table 3 (Latency)**:
- Router overhead: **8.94 ms** (P99) = **1.1%** of total request time

**Table 4 (Qualitative)**:
- 28 clusters: Specialist wins (Nova > GPT-4o)
- 35 clusters: Teacher fallback (GPT-4o essential)
- 226 clusters: Equal quality → 97.6% cost savings

---

## Scripts

| Script | Description |
|--------|-------------|
| `scripts/run_rq1.py` | RQ1: Warm-start vs cold-start regret experiment |
| `scripts/run_rq2.py` | RQ2: Specialist discovery (‖θ‖ analysis) |
| `scripts/run_rq2_poisoned.py` | RQ2: Belief recovery simulation |
| `scripts/run_rq3.py` | RQ3: Cost-quality Pareto frontier |
| `scripts/generate_expert_priors.py` | Expert priors generation |
| `scripts/benchmark_latency.py` | Router latency benchmarking |

### Reproduce All Results

```bash
# From repository root
python -m llm_jury.experiment.run_rq1
python -m llm_jury.experiment.run_rq2
python -m llm_jury.experiment.run_rq2_poisoned
python -m llm_jury.experiment.run_rq3
python -m llm_jury.experiment.benchmark_latency
```

---

## Key Claims & Evidence

### RQ1: Shippable Brain Advantage
> *"Expert-distilled priors reduce cumulative regret by 63.6% compared to cold-start."*

**Evidence**: `figures/figure1_regret_curve.pdf`, `data/rq1_metrics.json`

### RQ2: Plasticity & Specialist Discovery
> *"The router successfully discovers specialist models (Nova-Lite ‖θ‖=3.66) that outperform the generalist teacher (GPT-4o ‖θ‖=1.66) in specific latent regions."*

**Evidence**: `figures/figure3_specialist_landscape.pdf`

> *"The router can unlearn 'confidently wrong' priors within ~50 interactions."*

**Evidence**: `figures/figure2_belief_recovery.png`

### RQ3: Cost-Quality Efficiency
> *"The adaptive router achieves +33.8% higher quality than GPT-4o while reducing cost by 67.7%."*

**Evidence**: `tables/table2_system_impact.md`, `figures/figure4_pareto_frontier.pdf`

> *"Router inference adds only 8.94ms (P99), representing 1.1% of total request latency."*

**Evidence**: `tables/table3_latency_overhead.md`, `data/latency_benchmark.json`

---

## Baseline Justification

**Why GPT-4o as Baseline**:

1. **Teacher-Student Consistency**: Priors are distilled from GPT-4o oracle; the router (student) must be compared to its teacher.

2. **Empirical SOTA**: GPT-4o achieves highest reward (0.602) in our corpus—the hardest opponent.

3. **Reference Stability**: Known latency/cost profile vs newer beta models with variable characteristics.

> *"We employ GPT-4o as the primary static baseline. In our evaluation corpus, it achieves the highest average reward (0.602), effectively serving as the empirical 'ceiling' for performance."*

---

## LaTeX Snippets

All tables include camera-ready LaTeX in their respective `.md` files. Key snippets:

- **Table 3 (Latency)**: See `tables/table3_latency_overhead.md`
- **Table 4 (Qualitative)**: See `tables/table4_qualitative_routing.md`

---

## Data Files

| File | Description |
|------|-------------|
| `data/rq1_metrics.json` | RQ1 regret numbers |
| `data/rq3_cost_quality.json` | Model costs and ‖θ‖ values |
| `data/latency_benchmark.json` | Router latency P50/P95/P99 |

---

## Citation

```bibtex
@inproceedings{llmjury2025,
  title={Density-Based Warm-Start for Adaptive LLM Routing},
  author={...},
  booktitle={Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year={2025}
}
```
