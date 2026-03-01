#!/usr/bin/env bash
# Copies fresh figures from experiment outputs into paper/figures/.
# Run from the paper/ directory before compiling.
set -euo pipefail
cd "$(dirname "$0")"

EXP="../experiments"
DEST="figures"

# Each line: DEST_NAME SOURCE_PATH
MAPPING=(
  # Main paper figures (Figures 1-8)
  "figure1_k10_contextual.png         $EXP/01_figure/results/figure1_k10_contextual.png"
  "figure2_architecture.png           $EXP/02_figure/results/figure2_architecture.png"
  "figure2_architecture.pdf           $EXP/02_figure/results/figure2_architecture.pdf"
  "figure3_pareto.png                 $EXP/03_figure/results/figure4.png"
  "figure4_multimodel_pareto.png      $EXP/04_figure/results/figure6_multimodel_pareto.png"
  "figure5_k_scaling.png              $EXP/05_figure/results/k_scaling_figure.png"
  "figure6_distribution_shift.png     $EXP/06_distribution_shift/results/figure_distribution_shift.png"
  "figure7_lints_comparison.pdf       $EXP/07_figure/results/figure7_lints_comparison.pdf"
  "figure8_cumulative_regret.pdf      $EXP/08_figure/results/figure8_cumulative_regret.pdf"

  # Appendix figures (Figures 9-12)
  "figure9_5model.png                 $EXP/appendix/E_catastrophic_failure_experiment/results/figure9_5model.png"
  "figure9_5model.pdf                 $EXP/appendix/E_catastrophic_failure_experiment/results/figure9_5model.pdf"
  "model_onboarding.png               $EXP/03_figure/results/model_onboarding.png"
  "figure11_prior_degradation.pdf     $EXP/appendix/E_prior_degradation/results/figure3_prior_degradation.pdf"
  "figure12_constraint_impact.pdf     $EXP/appendix/F_constraint_impact/results/figure_constraint_impact.pdf"
  "figure12_constraint_impact.png     $EXP/appendix/F_constraint_impact/results/figure_constraint_impact.png"
)

updated=0
skipped=0
missing=0

for entry in "${MAPPING[@]}"; do
  dest_name=$(echo "$entry" | awk '{print $1}')
  src=$(echo "$entry" | awk '{print $2}')
  dst="$DEST/$dest_name"

  if [ ! -e "$src" ]; then
    echo "SKIP (source missing): $src"
    ((missing++))
    continue
  fi

  if [ -e "$dst" ] && [ ! "$src" -nt "$dst" ]; then
    ((skipped++))
    continue
  fi

  cp "$src" "$dst"
  echo "  COPIED: $dest_name"
  ((updated++))
done

echo ""
echo "Done: $updated copied, $skipped up-to-date, $missing missing sources."
