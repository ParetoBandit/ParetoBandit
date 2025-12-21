#!/bin/bash
# ============================================================================
# Regenerate All Figures for KDD Paper
# ============================================================================
#
# This script regenerates all figures used in the paper:
# "Democratizing LLM Access: Adaptive Routing with Shippable Priors"
#
# Usage: ./regenerate_all_figures.sh
#
# Runtime: ~2-3 hours (depending on system)
# Output: results/ and kdd_paper/figures/ directories
#
# ============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "============================================================================"
echo "  Regenerating All Figures for KDD Paper"
echo "============================================================================"
echo ""

# Navigate to project root
PROJECT_ROOT="/Users/annette/repostitories/llm_jury"
cd "$PROJECT_ROOT"

# Check if virtual environment exists
if [ ! -d "venv" ] && [ ! -d ".venv" ]; then
    echo -e "${YELLOW}Warning: No virtual environment detected${NC}"
    echo "Consider creating one: python -m venv venv && source venv/bin/activate"
    echo ""
fi

# ============================================================================
# PART 1: Main Experiment Results (Figures 1-4)
# ============================================================================

echo -e "${GREEN}[1/7] Running RQ1: Warm-Start Effectiveness${NC}"
echo "  Generates: Figure 1 (regret curves), Figure 3 (specialist landscape)"
echo "  Expected runtime: ~15-30 minutes"
echo ""
python experiments/run_rq1.py
echo -e "${GREEN}✓ RQ1 complete${NC}"
echo ""

echo -e "${GREEN}[2/7] Running RQ2: Plasticity Analysis${NC}"
echo "  Generates: Figure 2 (belief recovery)"
echo "  Expected runtime: ~20-40 minutes"
echo ""
python experiments/run_rq2.py
echo -e "${GREEN}✓ RQ2 complete${NC}"
echo ""

echo -e "${GREEN}[3/7] Running RQ3: Cost-Quality Trade-offs${NC}"
echo "  Generates: Cost reduction metrics and accuracy data"
echo "  Expected runtime: ~15-30 minutes"
echo ""
python experiments/run_rq3.py
echo -e "${GREEN}✓ RQ3 complete${NC}"
echo ""

echo -e "${GREEN}[4/7] Running Pareto Frontier Experiment${NC}"
echo "  Generates: Figure 4 (Pareto frontier)"
echo "  Expected runtime: ~30-60 minutes"
echo ""
python kdd_paper/scripts/run_needle_in_haystack.py
echo -e "${GREEN}✓ Pareto experiment complete${NC}"
echo ""

# ============================================================================
# PART 2: Supplementary Figures and Diagrams
# ============================================================================

cd kdd_paper/scripts

echo -e "${GREEN}[5/7] Generating SLA Constraint Figures${NC}"
echo "  Generates: Figure 7 (SLA tunability), Figure 8 (FinOps constraints)"
echo "  Expected runtime: ~2-5 minutes"
echo ""
python plot_sla_figures.py
echo -e "${GREEN}✓ SLA figures complete${NC}"
echo ""

echo -e "${GREEN}[6/7] Creating Benchmark Trap Diagram${NC}"
echo "  Generates: figure_confident_failure.pdf"
echo "  Expected runtime: ~1-2 minutes"
echo ""
python create_trap_diagram.py
echo -e "${GREEN}✓ Trap diagram complete${NC}"
echo ""

echo -e "${GREEN}[7/7] Generating Statistical Comparison Plots${NC}"
echo "  Generates: Box plots, confidence intervals, router analysis"
echo "  Expected runtime: ~3-5 minutes"
echo ""
python plot_statistical_comparison.py
echo -e "${GREEN}✓ Statistical plots complete${NC}"
echo ""

# ============================================================================
# Summary
# ============================================================================

echo "============================================================================"
echo -e "${GREEN}  ✅ All Figures Generated Successfully!${NC}"
echo "============================================================================"
echo ""
echo "📂 Output Locations:"
echo "  - Main results:    $PROJECT_ROOT/results/"
echo "  - Figure files:    $PROJECT_ROOT/kdd_paper/figures/"
echo "  - Paper figures:   $PROJECT_ROOT/kdd_paper/paper_submitted/figures/"
echo ""
echo "📊 Generated Figures:"
echo "  ✓ Figure 1: Regret curves (warm-start vs cold-start)"
echo "  ✓ Figure 2: Belief recovery under concept drift"
echo "  ✓ Figure 3: Specialist landscape (81 models)"
echo "  ✓ Figure 4: Pareto frontier (cost-quality trade-offs)"
echo "  ✓ Figure 7: SLA tunability demonstration"
echo "  ✓ Figure 8: FinOps multi-objective constraints"
echo "  ✓ Supplementary: Trap diagram, statistical comparisons"
echo ""
echo "📝 Next Steps:"
echo "  1. Review figures in kdd_paper/figures/"
echo "  2. Check numerical results in results/*/"
echo "  3. Compile paper: cd paper_submitted/concise_version && ./compile.sh"
echo ""
echo "============================================================================"

