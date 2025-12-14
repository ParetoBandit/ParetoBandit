#!/bin/bash
# Complete Training Pipeline for Intent-Specific Logistic Regression
# 
# This script runs the entire pipeline:
# 1. Analyze collinearity
# 2. Perform anchor-based imputation
# 3. Build instance-level training data
# 4. Train logistic regression models with stratified 5-fold CV

set -e  # Exit on error

echo "================================================================================"
echo "COMPLETE TRAINING PIPELINE"
echo "================================================================================"
echo ""
echo "This pipeline will:"
echo "  1. Analyze benchmark collinearity"
echo "  2. Perform anchor-based imputation for missing scores"
echo "  3. Build instance-level training data (prompts + labels + NVIDIA features)"
echo "  4. Train logistic regression models with stratified 5-fold CV"
echo ""
echo "Estimated time: 30-60 minutes (depending on dataset downloads)"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."

cd /Users/annette/repostitories/llm_jury

# Step 1: Analyze Collinearity
echo ""
echo "================================================================================"
echo "STEP 1: ANALYZING BENCHMARK COLLINEARITY"
echo "================================================================================"
echo ""

python3 KDD/data/analyze_benchmark_collinearity.py

echo ""
read -p "Review collinearity results above. Press Enter to continue..."

# Step 2: Anchor-Based Imputation
echo ""
echo "================================================================================"
echo "STEP 2: PERFORMING ANCHOR-BASED IMPUTATION"
echo "================================================================================"
echo ""

python3 KDD/data/anchor_based_imputation.py

echo ""
read -p "Review imputation results. Press Enter to continue..."

# Step 3: Analyze Imputation Significance
echo ""
echo "================================================================================"
echo "STEP 3: ANALYZING IMPUTATION STATISTICAL SIGNIFICANCE"
echo "================================================================================"
echo ""

python3 KDD/data/analyze_imputation_results.py

echo ""
read -p "Review statistical significance. All coefficients should be p < 0.05. Press Enter to continue..."

# Step 4: Build Instance-Level Training Data
echo ""
echo "================================================================================"
echo "STEP 4: BUILDING INSTANCE-LEVEL TRAINING DATA"
echo "================================================================================"
echo ""
echo "This step may take 20-40 minutes as it:"
echo "  - Downloads benchmark datasets from HuggingFace"
echo "  - Downloads evaluation results from GitHub"
echo "  - Computes NVIDIA complexity features for each prompt"
echo ""

python3 KDD/data/build_instance_level_training_data.py

echo ""
if [ ! -f "KDD/data/instance_level_training_data/instance_level_training_data.csv" ]; then
    echo "⚠️  ERROR: Instance-level training data not found!"
    echo "   Please check the output above for errors."
    exit 1
fi

echo "✓ Instance-level training data created successfully"
read -p "Press Enter to continue to model training..."

# Step 5: Train Logistic Regression Models
echo ""
echo "================================================================================"
echo "STEP 5: TRAINING LOGISTIC REGRESSION MODELS"
echo "================================================================================"
echo ""
echo "Training 5 intent-specific models with:"
echo "  - Stratified 80/20 train/test split"
echo "  - Stratified 5-fold cross-validation"
echo "  - Automatic VIF-based collinearity removal (VIF > 10)"
echo ""

python3 KDD/data/train_logistic_regression_with_nvidia.py

echo ""
echo "================================================================================"
echo "✓ PIPELINE COMPLETE!"
echo "================================================================================"
echo ""
echo "Results saved to:"
echo "  - intent_predictors_with_nvidia/*.joblib  (trained models)"
echo "  - intent_predictors_with_nvidia/*_test_data.csv  (test sets)"
echo "  - intent_predictors_with_nvidia/training_summary.json  (metrics)"
echo ""
echo "Review the output above for:"
echo "  ✓ All VIF < 10 (no collinearity)"
echo "  ✓ Stratification quality Δ < 0.02"
echo "  ✓ Test accuracy > 80%"
echo "  ✓ CV std < 0.05"
echo ""
echo "Next steps:"
echo "  1. Review training_summary.json for detailed metrics"
echo "  2. Check test set predictions for errors"
echo "  3. Use models for inference or paper writing"
echo ""
