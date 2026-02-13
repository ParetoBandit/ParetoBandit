#!/bin/bash
# Statistical Validation Pipeline for Table 2
# This script addresses the statistical rigor concerns from the KDD review

set -e  # Exit on error

echo "========================================================================"
echo "TABLE 2 STATISTICAL VALIDATION PIPELINE"
echo "========================================================================"
echo ""
echo "This script will:"
echo "  1. Run η=0.1 with 30 random seeds"
echo "  2. Run η=1.0 with 30 random seeds"
echo "  3. Compute confidence intervals and effect sizes"
echo "  4. Perform statistical significance tests"
echo "  5. Generate updated visualizations"
echo ""
echo "Expected runtime: ~20-30 minutes on a modern laptop"
echo "========================================================================"
echo ""

# Configuration
NUM_SEEDS=30
GAMMA=0.05

# Create output directories
mkdir -p data/eta_0.1_holdout_multiseed
mkdir -p data/eta_1.0_holdout_multiseed
mkdir -p data/statistical_comparison

echo "Step 1/3: Running η=0.1 with ${NUM_SEEDS} seeds..."
echo "------------------------------------------------------------------------"
python experiments_v1/02_table/run_holdout_evaluation_multiseed.py \
    --learning-rate 0.1 \
    --gamma ${GAMMA} \
    --num-seeds ${NUM_SEEDS} \
    --output data/eta_0.1_holdout_multiseed

echo ""
echo "Step 2/3: Running η=1.0 with ${NUM_SEEDS} seeds..."
echo "------------------------------------------------------------------------"
python experiments_v1/02_table/run_holdout_evaluation_multiseed.py \
    --learning-rate 1.0 \
    --gamma ${GAMMA} \
    --num-seeds ${NUM_SEEDS} \
    --output data/eta_1.0_holdout_multiseed

echo ""
echo "Step 3/3: Comparing learning rates with statistical tests..."
echo "------------------------------------------------------------------------"
python experiments_v1/02_table/compare_learning_rates.py \
    --eta-01-results data/eta_0.1_holdout_multiseed/results_multiseed.json \
    --eta-10-results data/eta_1.0_holdout_multiseed/results_multiseed.json \
    --output data/statistical_comparison/comparison_results.json

echo ""
echo "========================================================================"
echo "✅ STATISTICAL VALIDATION COMPLETE!"
echo "========================================================================"
echo ""
echo "Generated files:"
echo "  - data/eta_0.1_holdout_multiseed/results_multiseed.json"
echo "  - data/eta_0.1_holdout_multiseed/results_per_seed.json"
echo "  - data/eta_0.1_holdout_multiseed/multiseed_comparison.png"
echo "  - data/eta_1.0_holdout_multiseed/results_multiseed.json"
echo "  - data/eta_1.0_holdout_multiseed/results_per_seed.json"
echo "  - data/eta_1.0_holdout_multiseed/multiseed_comparison.png"
echo "  - data/statistical_comparison/comparison_results.json"
echo ""
echo "Next steps:"
echo "  1. Review comparison_results.json for significance tests"
echo "  2. Update table_02_merged.tex with mean ± CI values"
echo "  3. Add statistical test results to paper text"
echo ""
echo "Key metrics to report in paper:"
echo "  - Mean ± 95% CI for cumulative regret"
echo "  - p-values from t-test and Mann-Whitney U test"
echo "  - Cohen's d effect sizes"
echo "  - Bonferroni-corrected significance"
echo "========================================================================"
