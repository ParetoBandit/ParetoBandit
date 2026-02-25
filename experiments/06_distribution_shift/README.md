# Experiment 10 — Distribution Shift Analysis

## Motivation

The paper claims that "since the preference structure is counter-intuitive
and shifts with deployment distributions, it cannot be reliably specified
offline."  This experiment provides the empirical evidence.

## Setup

Uses the **K=2 topology** (Mixtral-8x7B vs GPT-4-Turbo) — the only regime
where the warmup priors were built from a **different source** than the
deployment data:

| Component | Source | N |
|-----------|--------|---|
| Warmup priors | RouteLLM battle prompts | 8,000 |
| Online learning (dev) | LMSYS Arena | 766 |
| Evaluation (holdout) | LMSYS Arena | 750 |

This matches Section 4 / Figure 4 of the paper and the
`warmup_prior_construction` section.

## Analyses

### 1. Feature Distribution Shift
- **PSI = 0.763** [0.688, 0.860] — "Substantial" shift
- **KS D = 0.310**, p < 10⁻¹²⁹
- Visible secondary mode in LMSYS deployment prompts on PC1

### 2. Prior Miscalibration
- **Mixtral-8x7B**: −75% error (RouteLLM priors severely underestimate
  deployment reward)
- **GPT-4-Turbo**: −1.1% error (approximately calibrated)

### 3. Cost–Quality Pareto (λ sweep, 20 trials)

| Router | Best Quality | Cost at Best Quality | Notes |
|--------|-------------|---------------------|-------|
| Oracle | 0.953 | $0.00195 | Per-prompt optimal |
| Static frozen prior | 0.823 | $0.00029 | Degenerate: collapses to all-Mixtral at high λ |
| Hybrid banditGPT | 0.905 | $0.00861 | Smooth per-prompt Pareto frontier |

**Key finding**: The miscalibrated RouteLLM priors collapse the static
router into a binary switch — it cannot route per-prompt.  Online
adaptation recovers a smooth Pareto frontier, achieving +8.2 pp higher
quality than the best static frozen-prior point.

## Running

```bash
python3 experiments/06_distribution_shift/run_distribution_shift.py
```

## Output

- `results/distribution_shift_results.json` — full numerical results
- `results/figure_distribution_shift.png` — two-panel figure
