#!/bin/bash
# ============================================================================
# Regenerate All Tables for KDD Paper
# ============================================================================
#
# This script regenerates all table data used in the paper:
# "Democratizing LLM Access: Adaptive Routing with Shippable Priors"
#
# Usage: ./regenerate_all_tables.sh
#
# Runtime: ~1.5-2.5 hours (depending on system)
# Output: results/ and kdd_paper/data/ directories
#
# ============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "============================================================================"
echo "  Regenerating All Tables for KDD Paper"
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
# PART 1: Main Experiment Tables (Core Paper Results)
# ============================================================================

echo -e "${GREEN}[1/6] Generating Table 1: Benchmark Trap${NC}"
echo "  RQ1: Warm-start vs. cold-start vs. benchmark initialization"
echo "  Expected runtime: ~15-30 minutes"
echo ""
python experiments/run_rq1.py
echo -e "${GREEN}✓ Table 1 data generated${NC}"
echo "  Output: results/rq1/, kdd_paper/data/rq1_metrics.json"
echo ""

echo -e "${GREEN}[2/6] Generating Table 2: Plasticity Metrics${NC}"
echo "  RQ2: Belief recovery under concept drift"
echo "  Expected runtime: ~20-40 minutes"
echo ""
python experiments/run_rq2.py
echo -e "${GREEN}✓ Plasticity data generated${NC}"
echo "  Output: results/rq2/"
echo ""

echo -e "${GREEN}[3/6] Generating Tables 4 & 5: System Impact + ROI Leaderboard${NC}"
echo "  RQ3: Cost-quality trade-offs and specialist rankings"
echo "  Expected runtime: ~15-30 minutes"
echo ""
python experiments/run_rq3.py
echo -e "${GREEN}✓ Tables 4 & 5 data generated${NC}"
echo "  Output: results/rq3/, kdd_paper/data/rq3_cost_quality.json"
echo ""

# ============================================================================
# PART 2: Specialized Benchmark Tables
# ============================================================================

cd kdd_paper/scripts

echo -e "${GREEN}[4/6] Generating Tables 2 & 3: SOTA Comparison + Domain Breakdown${NC}"
echo "  Comparing BanditGPT vs. FrugalGPT vs. RouteLLM"
echo "  Expected runtime: ~30-60 minutes"
echo ""
python run_sota_comparison.py
echo -e "${GREEN}✓ Tables 2 & 3 data generated${NC}"
echo "  Output: results/sota_comparison/"
echo ""

echo -e "${GREEN}[5/6] Generating Table 6: Latency Breakdown${NC}"
echo "  Router overhead analysis (1,000 runs)"
echo "  Expected runtime: ~5-10 minutes"
echo ""
python benchmark_latency.py
echo -e "${GREEN}✓ Table 6 data generated${NC}"
echo "  Output: kdd_paper/data/latency_benchmark.json"
echo ""

# ============================================================================
# PART 3: Validation (Optional but Recommended)
# ============================================================================

echo -e "${GREEN}[6/6] Validating Benchmark Scores${NC}"
echo "  Ensuring benchmark scores match published results"
echo "  Expected runtime: ~10-20 minutes"
echo ""
python validate_benchmarks.py
echo -e "${GREEN}✓ Validation complete${NC}"
echo "  Output: results/validation/"
echo ""

# ============================================================================
# Summary
# ============================================================================

cd "$PROJECT_ROOT"

echo "============================================================================"
echo -e "${GREEN}  ✅ All Tables Generated Successfully!${NC}"
echo "============================================================================"
echo ""
echo "📂 Output Locations:"
echo "  - JSON data files: $PROJECT_ROOT/kdd_paper/data/"
echo "    • latency_benchmark.json"
echo "    • rq1_metrics.json"
echo "    • rq3_cost_quality.json"
echo ""
echo "  - Experiment results: $PROJECT_ROOT/results/"
echo "    • rq1/ (Table 1 data)"
echo "    • rq2/ (Plasticity metrics)"
echo "    • rq3/ (Tables 4 & 5 data)"
echo "    • sota_comparison/ (Tables 2 & 3 data)"
echo "    • validation/ (Benchmark validation)"
echo ""
echo "  - Markdown tables: $PROJECT_ROOT/kdd_paper/tables/"
echo ""
echo "📊 Generated Tables:"
echo "  ✓ Table 1: Benchmark Trap (initialization comparison)"
echo "  ✓ Table 2: SOTA Comparison (BanditGPT vs. baselines)"
echo "  ✓ Table 3: Domain Breakdown (Math, Reasoning, Instructions)"
echo "  ✓ Table 4: System Impact (cost reduction + quality)"
echo "  ✓ Table 5: ROI Leaderboard (top 15 specialists)"
echo "  ✓ Table 6: Latency Breakdown (router overhead)"
echo "  ✓ Appendix: Full ROI (all 81 models), detailed latency"
echo ""
echo "📝 Next Steps:"
echo "  1. Review JSON data files in kdd_paper/data/"
echo "  2. Check numerical results in results/*/"
echo "  3. Update LaTeX tables if needed in paper_submitted/"
echo "  4. Compile paper: cd paper_submitted/concise_version && ./compile.sh"
echo "  5. Verify all numbers match in the PDF"
echo ""
echo "💡 Manual Tables (no script needed):"
echo "  - Use Case Summary (qualitative analysis)"
echo "  - Accessibility Comparison (operational requirements)"
echo "  - Scaling Comparison (architectural trade-offs)"
echo ""
echo "============================================================================"

