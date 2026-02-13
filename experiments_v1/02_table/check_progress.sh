#!/bin/bash
# Quick progress checker for the validation pipeline

echo "========================================================================"
echo "VALIDATION PROGRESS CHECKER"
echo "========================================================================"
echo ""

# Check if validation is running
if pgrep -f "run_statistical_validation.sh" > /dev/null; then
    echo "✅ Validation is RUNNING"
else
    echo "⚠️  Validation is NOT running"
fi

echo ""
echo "------------------------------------------------------------------------"
echo "Output Files:"
echo "------------------------------------------------------------------------"

# Check eta_0.1 results
if [ -f "data/eta_0.1_holdout_multiseed/results_multiseed.json" ]; then
    echo "✅ η=0.1 results: COMPLETE"
    seeds=$(jq -r '.["Hybrid (Corralling)"].num_seeds' data/eta_0.1_holdout_multiseed/results_multiseed.json 2>/dev/null || echo "unknown")
    echo "   Seeds completed: $seeds"
else
    echo "⏳ η=0.1 results: IN PROGRESS or NOT STARTED"
fi

# Check eta_1.0 results
if [ -f "data/eta_1.0_holdout_multiseed/results_multiseed.json" ]; then
    echo "✅ η=1.0 results: COMPLETE"
    seeds=$(jq -r '.["Hybrid (Corralling)"].num_seeds' data/eta_1.0_holdout_multiseed/results_multiseed.json 2>/dev/null || echo "unknown")
    echo "   Seeds completed: $seeds"
else
    echo "⏳ η=1.0 results: IN PROGRESS or NOT STARTED"
fi

# Check comparison results
if [ -f "data/statistical_comparison/comparison_results.json" ]; then
    echo "✅ Comparison: COMPLETE"
else
    echo "⏳ Comparison: PENDING"
fi

echo ""
echo "------------------------------------------------------------------------"
echo "Latest Log Output (last 30 lines):"
echo "------------------------------------------------------------------------"

if [ -f "validation_full.log" ]; then
    tail -30 validation_full.log
else
    echo "Log file not found"
fi

echo ""
echo "========================================================================"
echo "To watch live: tail -f validation_full.log"
echo "========================================================================"
