#!/bin/bash
#
# Systematic Hyperparameter Search using the proven-working run_rq1_pca.py
#
# Vary: PCA dimensions, training epochs, expert rate, prior strength
#

set -e

echo "======================================================================"
echo "Systematic Hyperparameter Search for 20%+ Regret Reduction"
echo "======================================================================"
echo

# Test different PCA dimensions (already have 32, 48, 64)
PCA_DIMS=(32 48)

# Test different training configurations by regenerating priors
EPOCHS=(10 15 20)
EXPERT_RATES=(0.75 0.85)

# Test different prior strengths at evaluation time
PRIOR_STRENGTHS=(5 8 10 15)

RESULTS_FILE="hyperparam_search_results.txt"
> "$RESULTS_FILE"  # Clear file

test_count=0
total_tests=$((${#PCA_DIMS[@]} * ${#EPOCHS[@]} * ${#EXPERT_RATES[@]} * ${#PRIOR_STRENGTHS[@]}))

for dim in "${PCA_DIMS[@]}"; do
  for epochs in "${EPOCHS[@]}"; do
    for expert in "${EXPERT_RATES[@]}"; do
      # Generate priors with these training params
      test_count=$((test_count + 1))
      echo "[${test_count}/$total_tests] Generating priors: d=$dim, epochs=$epochs, expert=$expert"
      
      python experiments/generate_priors_with_params.py \
        --dim "$dim" \
        --epochs "$epochs" \
        --expert-rate "$expert" \
        --output "expert_priors_d${dim}_e${epochs}_exp${expert}.npz"
      
      # Test with different prior strengths
      for strength in "${PRIOR_STRENGTHS[@]}"; do
        test_name="d${dim}_e${epochs}_exp$(echo $expert | sed 's/0\.//')_s${strength}"
        echo "  Testing λ=$strength..."
        
        # Run evaluation (quiet mode)
        python experiments/eval_with_priors.py \
          --priors "expert_priors_d${dim}_e${epochs}_exp${expert}.npz" \
          --dim "$dim" \
          --strength "$strength" \
          --output "results/search/$test_name" \
          > "results/search/${test_name}.log" 2>&1
        
        # Extract result
        reduction=$(grep "Regret Reduction:" "results/search/${test_name}.log" | tail -1 | awk '{print $3}')
        echo "$test_name: $reduction" | tee -a "$RESULTS_FILE"
      done
    done
  done
done

echo
echo "======================================================================"
echo "SEARCH COMPLETE - Results:"
echo "======================================================================"
cat "$RESULTS_FILE" | sort -t: -k2 -nr | head -10
echo "======================================================================"
echo "Full results in: $RESULTS_FILE"

