#!/bin/bash

# ==============================================================================
# BanditGPT KDD Paper Restructuring Integration Script
# ==============================================================================
#
# This script integrates the democratization-first restructuring into your
# KDD paper. It backs up originals and applies the revised sections.
#
# Usage:
#   cd /Users/annette/repostitories/llm_jury/kdd_paper
#   chmod +x integrate_restructuring.sh
#   ./integrate_restructuring.sh [option]
#
# Options:
#   full      - Apply all changes (abstract, intro, use_cases, conclusion)
#   minimal   - Apply only intro and conclusion (minimal disruption)
#   test      - Show what would change without modifying files
#
# ==============================================================================

set -e  # Exit on error

# Configuration
PAPER_DIR="paper_submitted"
BACKUP_DIR="paper_submitted/backup_$(date +%Y%m%d_%H%M%S)"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Parse arguments
MODE=${1:-test}

# ==============================================================================
# Functions
# ==============================================================================

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

create_backup() {
    log_info "Creating backup directory: $BACKUP_DIR"
    mkdir -p "$BACKUP_DIR"
    
    # Backup files we'll modify
    for file in main.tex introduction.tex conclusion.tex abstract.tex; do
        if [ -f "$PAPER_DIR/$file" ]; then
            cp "$PAPER_DIR/$file" "$BACKUP_DIR/"
            log_info "Backed up: $file"
        fi
    done
}

show_changes() {
    log_info "Preview of changes:"
    echo ""
    echo "Files to be modified:"
    echo "  1. main.tex - Update abstract, add use_cases section"
    echo "  2. introduction.tex - Replace with democratization-first version"
    echo "  3. conclusion.tex - Replace with impact-first version"
    echo ""
    echo "New files to be created:"
    echo "  - use_cases.tex (if full mode)"
    echo ""
    echo "Backup location:"
    echo "  $BACKUP_DIR"
}

apply_introduction() {
    log_info "Applying revised introduction..."
    if [ -f "$PAPER_DIR/introduction_REVISED.tex" ]; then
        cp "$PAPER_DIR/introduction_REVISED.tex" "$PAPER_DIR/introduction.tex"
        log_info "✓ Introduction updated"
    else
        log_error "introduction_REVISED.tex not found!"
        return 1
    fi
}

apply_conclusion() {
    log_info "Applying revised conclusion..."
    if [ -f "$PAPER_DIR/conclusion_REVISED.tex" ]; then
        cp "$PAPER_DIR/conclusion_REVISED.tex" "$PAPER_DIR/conclusion.tex"
        log_info "✓ Conclusion updated"
    else
        log_error "conclusion_REVISED.tex not found!"
        return 1
    fi
}

apply_use_cases() {
    log_info "Adding use_cases section..."
    if [ -f "$PAPER_DIR/use_cases.tex" ]; then
        log_info "✓ use_cases.tex already exists"
        log_warn "Remember to add \\input{use_cases} to main.tex after \\input{introduction}"
    else
        log_error "use_cases.tex not found!"
        return 1
    fi
}

update_main_tex() {
    log_info "Updating main.tex structure..."
    
    # Check if use_cases is already included
    if grep -q "\\input{use_cases}" "$PAPER_DIR/main.tex"; then
        log_info "✓ main.tex already includes use_cases"
    else
        log_warn "You need to manually add \\input{use_cases} after \\input{introduction} in main.tex"
        echo ""
        echo "Add this line to main.tex after line 121:"
        echo "  \\input{use_cases}        % NEW: Democratization use cases"
    fi
}

compile_paper() {
    log_info "Compiling paper to verify LaTeX..."
    cd "$PAPER_DIR"
    
    # Run LaTeX compilation
    pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        log_error "LaTeX compilation failed! Check logs."
        return 1
    fi
    
    # Run BibTeX
    bibtex main > /dev/null 2>&1
    
    # Run LaTeX again (twice for references)
    pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1
    pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1
    
    # Check page count
    PAGE_COUNT=$(pdfinfo main.pdf 2>/dev/null | grep "Pages:" | awk '{print $2}')
    
    if [ -n "$PAGE_COUNT" ]; then
        log_info "✓ Compilation successful"
        log_info "Total page count: $PAGE_COUNT"
        
        if [ "$PAGE_COUNT" -gt 8 ]; then
            log_warn "Paper exceeds 8 content pages! You may need to compress sections."
            log_warn "See RESTRUCTURING_GUIDE.md Section: Page Budget Management"
        fi
    fi
    
    cd ..
}

# ==============================================================================
# Main execution
# ==============================================================================

echo "=========================================="
echo "BanditGPT Paper Restructuring"
echo "=========================================="
echo ""

case "$MODE" in
    test)
        log_info "Running in TEST mode (no files modified)"
        show_changes
        echo ""
        log_info "To apply changes, run:"
        echo "  ./integrate_restructuring.sh full    # Apply all changes"
        echo "  ./integrate_restructuring.sh minimal # Apply intro + conclusion only"
        ;;
        
    minimal)
        log_info "Running in MINIMAL mode (intro + conclusion only)"
        create_backup
        apply_introduction
        apply_conclusion
        log_info "Compiling paper..."
        compile_paper
        echo ""
        log_info "✓ Minimal restructuring complete!"
        log_info "Next steps:"
        echo "  1. Review main.pdf"
        echo "  2. If satisfied, consider adding use_cases section (run with 'full' mode)"
        echo "  3. Original files backed up to: $BACKUP_DIR"
        ;;
        
    full)
        log_info "Running in FULL mode (all changes)"
        create_backup
        apply_introduction
        apply_conclusion
        apply_use_cases
        update_main_tex
        
        log_warn "MANUAL STEP REQUIRED:"
        echo "  Edit $PAPER_DIR/main.tex and add this line after \\input{introduction}:"
        echo "  \\input{use_cases}        % Section 2: Democratization use cases"
        echo ""
        
        read -p "Press Enter after you've edited main.tex (or Ctrl+C to abort)..."
        
        log_info "Compiling paper..."
        compile_paper
        echo ""
        log_info "✓ Full restructuring complete!"
        log_info "Next steps:"
        echo "  1. Review main.pdf"
        echo "  2. Check page count (should be ≤8)"
        echo "  3. Add framing sentences to Method/Evaluation (see FRAMING_ADDITIONS.md)"
        echo "  4. Original files backed up to: $BACKUP_DIR"
        ;;
        
    *)
        log_error "Unknown option: $MODE"
        echo "Usage: $0 [test|minimal|full]"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "For detailed guidance, see:"
echo "  - RESTRUCTURING_GUIDE.md"
echo "  - FRAMING_ADDITIONS.md"
echo "  - BEFORE_AFTER_COMPARISON.md"
echo "=========================================="

