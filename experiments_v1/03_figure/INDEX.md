# Figure 3: Corralled Architecture - Complete Documentation Index

## Quick Navigation

| Document | Purpose | Audience |
|----------|---------|----------|
| [README.md](#readmemd) | High-level overview | Everyone |
| [ARCHITECTURE_NOTES.md](#architecture_notesmd) | Deep dive into design | Researchers, Engineers |
| [IMPLEMENTATION_GUIDE.md](#implementation_guidemd) | How to use the code | Engineers, Practitioners |
| [THEORY_VS_IMPLEMENTATION.md](#theory_vs_implementationmd) | Algorithm comparison | Researchers, Advanced users |
| [DIAGRAM_SPECIFICATION.md](#diagram_specificationmd) | Visual design guide | Designers, Paper authors |
| [figure_3_caption.tex](#figure_2_captiontex) | LaTeX caption | Paper authors |

---

## README.md

**Purpose**: High-level introduction to the corralled architecture

**Key Sections**:
- Overview: What is corralling and why it matters
- Key Components: Coordinator, Experts, Communication Protocol
- Key Results: Performance metrics and benefits
- Files: Quick reference to all documentation

**Best For**:
- First-time readers
- Getting oriented
- Understanding the big picture

**Read Time**: 5 minutes

---

## ARCHITECTURE_NOTES.md

**Purpose**: Comprehensive architectural design documentation

**Key Sections**:
1. **Motivation**: Why corralling solves the cold-start-vs-adaptability tradeoff
2. **Hierarchical Design**: Coordinator-expert pattern rationale
3. **Expert Specifications**: Warmup vs Tabula Rasa initialization
4. **Communication Protocol**: Three-phase cycle (Recommendation → Selection → Feedback)
5. **Theoretical Properties**: Regret bounds, learning rate analysis
6. **Implementation Details**: Code snippets, pseudocode, complexity analysis
7. **Evaluation Strategy**: How to test and validate
8. **Related Work**: Connections to bandit literature

**Best For**:
- Understanding design decisions
- Implementing your own version
- Research and publication
- Deep technical understanding

**Read Time**: 20-30 minutes

**Highlights**:
- ✅ Actual code snippets from router.py
- ✅ Pseudocode with exact update rules
- ✅ Theory vs implementation comparison table
- ✅ Computational overhead analysis
- ✅ Algorithm box for paper inclusion

---

## IMPLEMENTATION_GUIDE.md

**Purpose**: Practical guide for using CorrallingRouter in production

**Key Sections**:
1. **Quick Start**: Copy-paste code to get running
2. **Configuration**: Learning rate selection, expert setup
3. **Monitoring and Diagnostics**: Check trust weights, track performance
4. **Advanced Usage**: Custom experts, multiple experts, delayed feedback
5. **Troubleshooting**: Common problems and solutions
6. **Performance Optimization**: Reduce latency and memory
7. **Testing and Validation**: Unit tests and integration tests
8. **Production Deployment Checklist**

**Best For**:
- Engineers deploying to production
- Debugging issues
- Optimizing performance
- Understanding configuration options

**Read Time**: 15-20 minutes

**Highlights**:
- ✅ Working code examples
- ✅ Configuration guidelines with specific values
- ✅ Troubleshooting flowcharts
- ✅ Performance tuning tips
- ✅ Production checklist

---

## THEORY_VS_IMPLEMENTATION.md

**Purpose**: Detailed comparison of theoretical algorithm vs production implementation

**Key Sections**:
1. **Core Algorithm Comparison**: Overview table
2. **Initialization**: Theory vs code
3. **Selection Phase**: O(K) vs O(1) complexity
4. **Loss Estimation**: Counterfactual vs observed-only
5. **Weight Update**: Full vs simplified
6. **Regret Guarantees**: Formal vs empirical
7. **Computational Complexity**: Detailed breakdown
8. **When to Use Which Version**: Decision guide
9. **Migration Path**: How to upgrade if needed
10. **Empirical Comparison**: Benchmark results

**Best For**:
- Researchers comparing algorithms
- Understanding tradeoffs
- Deciding which version to use
- Publication and peer review

**Read Time**: 25-35 minutes

**Highlights**:
- ✅ Side-by-side code vs theory
- ✅ Complexity analysis (O(K) vs O(1))
- ✅ Regret guarantee comparison
- ✅ When to use each version
- ✅ Benchmark results on real data

---

## DIAGRAM_SPECIFICATION.md

**Purpose**: Detailed specifications for creating Figure 3 visual diagram

**Key Sections**:
1. **Visual Design Overview**: Style, format, colors
2. **Layout Structure**: ASCII art preview
3. **Component Specifications**: Exact dimensions and contents
4. **Arrow Specifications**: Types, styles, labels
5. **Mathematical Notation**: Symbols and formatting
6. **Example Values**: Realistic numbers from experiments
7. **TikZ Code Template**: LaTeX code to start with
8. **Alternative Designs**: Sequence diagram option
9. **Figure Caption Integration**: How diagram supports caption
10. **Review Checklist**: Quality assurance

**Best For**:
- Creating the figure for paper
- Designers implementing the visual
- Ensuring consistency with text
- Publication preparation

**Read Time**: 15 minutes (reference document)

**Highlights**:
- ✅ Complete TikZ template
- ✅ Color scheme (color-blind friendly)
- ✅ Exact dimensions for publication
- ✅ ASCII preview of layout
- ✅ Example values from real experiments

---

## figure_3_caption.tex

**Purpose**: LaTeX caption for inclusion in paper

**Key Content**:
- Hierarchical architecture description
- Three-layer breakdown (Coordinator, Expert, Communication)
- Three key guarantees (Robustness, Fast Convergence, Provable Regret)
- Information flow explanation (dashed = selection, solid = recommendation, bold = feedback)

**Best For**:
- Paper writing
- LaTeX inclusion
- Ensuring caption matches figure

**Usage**:
```latex
\input{experiments_v1/02_figure/figure_3_caption.tex}
```

---

## Document Relationships

```
INDEX.md (you are here)
    │
    ├─→ README.md
    │   └─→ Quick orientation
    │
    ├─→ ARCHITECTURE_NOTES.md
    │   ├─→ Design rationale
    │   ├─→ Implementation details
    │   └─→ Theory & practice
    │
    ├─→ IMPLEMENTATION_GUIDE.md
    │   ├─→ Quick start code
    │   ├─→ Configuration
    │   ├─→ Troubleshooting
    │   └─→ Production deployment
    │
    ├─→ THEORY_VS_IMPLEMENTATION.md
    │   ├─→ Algorithm comparison
    │   ├─→ Complexity analysis
    │   ├─→ Regret guarantees
    │   └─→ Benchmark results
    │
    ├─→ DIAGRAM_SPECIFICATION.md
    │   ├─→ Visual design
    │   ├─→ TikZ template
    │   └─→ Figure creation
    │
    └─→ figure_3_caption.tex
        └─→ LaTeX caption
```

---

## Reading Paths

### Path 1: Quick Understanding (15 minutes)
1. README.md - Overview
2. DIAGRAM_SPECIFICATION.md - Look at ASCII layout
3. Done! You have the basic idea.

### Path 2: Implementation (30 minutes)
1. README.md - Overview
2. IMPLEMENTATION_GUIDE.md - Quick start section
3. Run the code examples
4. IMPLEMENTATION_GUIDE.md - Monitoring section
5. Done! You can use it in production.

### Path 3: Deep Understanding (60 minutes)
1. README.md - Overview
2. ARCHITECTURE_NOTES.md - Full read
3. THEORY_VS_IMPLEMENTATION.md - Algorithm comparison
4. src/bandit_gpt/router.py - Read the actual code
5. Done! You understand the design deeply.

### Path 4: Research/Publication (90 minutes)
1. README.md - Overview
2. ARCHITECTURE_NOTES.md - Theory section
3. THEORY_VS_IMPLEMENTATION.md - Full read
4. experiments_v1/05_corralling/ - Review experiments
5. DIAGRAM_SPECIFICATION.md - Plan figure
6. figure_3_caption.tex - Review caption
7. Done! Ready to write paper section.

---

## Code References

### Primary Implementation
- **File**: `src/bandit_gpt/router.py`
- **Lines**: 3349-3484
- **Class**: `CorrallingRouter`
- **Key Methods**:
  - `__init__()` - Initialization (lines 3394-3415)
  - `select_model()` - Selection phase (lines 3417-3432)
  - `update()` - Feedback phase (lines 3434-3478)
  - `get_expert_weights()` - Diagnostics (lines 3479-3484)

### Related Experiments
- **Folder**: `experiments_v1/05_corralling/`
- **Key Files**:
  - `test_hybrid_corralling.py` - Unit tests
  - `results/CORRALLING_BREAKTHROUGH.tex` - Experimental results
  - `README.md` - Experiment documentation

---

## Paper Integration

### Where Figure 3 Should Appear

**Section 3: Methodology**
- Subsection 3.2: Hierarchical Coordinator-Expert Architecture
- After: Problem formulation (Section 3.1)
- Before: Specific expert implementations (Section 3.3)

**Purpose in Paper**:
1. Introduce the coordinator-expert abstraction
2. Show information flows between layers
3. Ground subsequent algorithmic details
4. Provide visual reference for proofs/analysis

### Supporting Text

The figure should be accompanied by:
1. **Above**: Motivation for why hierarchical design is needed
2. **Below**: Algorithm box with pseudocode (from ARCHITECTURE_NOTES.md)
3. **Next Section**: Detailed expert specifications (Warmup, Tabula Rasa)

### Cross-References

Figure 3 should reference:
- **Figure 1**: Semantic structure that informs Warmup initialization
- **Figure 3**: Convergence comparison (Coordinator vs individual experts)
- **Table 1**: Regret bounds for different algorithms

---

## FAQ

### Q1: Which document should I read first?
**A**: Start with README.md for overview, then choose based on your goal:
- Implementing: → IMPLEMENTATION_GUIDE.md
- Understanding: → ARCHITECTURE_NOTES.md
- Researching: → THEORY_VS_IMPLEMENTATION.md

### Q2: Where is the actual code?
**A**: `src/bandit_gpt/router.py`, lines 3349-3484

### Q3: How do I create Figure 3?
**A**: Follow DIAGRAM_SPECIFICATION.md, use the TikZ template provided

### Q4: What's the difference between theoretical and implemented corralling?
**A**: See THEORY_VS_IMPLEMENTATION.md, Section "Core Algorithm Comparison"

### Q5: How do I deploy this in production?
**A**: IMPLEMENTATION_GUIDE.md, Section "Production Deployment Checklist"

### Q6: What are the performance characteristics?
**A**: 
- Latency: +0.5ms overhead (~0.5% of LLM inference)
- Memory: 2× vs single router
- Regret: Within 5-10% of best expert

### Q7: Do I need the full theoretical version?
**A**: Probably not, unless you need:
- Provable worst-case guarantees
- Adversarial robustness
- Formal publication

See THEORY_VS_IMPLEMENTATION.md, Section "When to Use Which Version"

---

## Change Log

### 2026-01-24
- ✅ Created complete documentation suite
- ✅ Added implementation guide with code examples
- ✅ Added theory vs implementation comparison
- ✅ Added diagram specification with TikZ template
- ✅ Enhanced architecture notes with pseudocode
- ✅ Created this index

### Future Enhancements
- [ ] Add TikZ diagram (architecture_diagram.tex)
- [ ] Add tutorial notebook (corralling_tutorial.ipynb)
- [ ] Add video walkthrough
- [ ] Add interactive visualization (D3.js)
- [ ] Add performance benchmarks (benchmark_results.json)

---

## Contact

For questions about:
- **Implementation**: See IMPLEMENTATION_GUIDE.md troubleshooting
- **Theory**: See THEORY_VS_IMPLEMENTATION.md
- **Paper**: See figure_3_caption.tex and DIAGRAM_SPECIFICATION.md

---

## Related Documentation

### Other Figures
- **Figure 1**: `experiments_v1/01_figure/` - Semantic task specialization
- **Figure 3**: `experiments_v1/03_figure/` - Feature transfer results
- **Figure 4**: `experiments_v1/04_figure/` - Cold start ablation

### Related Experiments
- **05_corralling/**: Corralling experiments and results
- **calibration/**: Calibration and convergence analysis
- **latent_semantic_transfer/**: Transfer learning experiments

### Core Code
- **router.py**: Main routing logic
- **storage.py**: Context persistence
- **utils.py**: Helper functions
- **config_legacy.py**: Configuration

---

## Document Statistics

| Document | Lines | Words | Read Time |
|----------|-------|-------|-----------|
| INDEX.md | 444 | ~2800 | 10 min |
| README.md | 100 | ~800 | 5 min |
| ARCHITECTURE_NOTES.md | 444 | ~3200 | 25 min |
| IMPLEMENTATION_GUIDE.md | 450 | ~3500 | 20 min |
| THEORY_VS_IMPLEMENTATION.md | 450 | ~3600 | 30 min |
| DIAGRAM_SPECIFICATION.md | 400 | ~2500 | 15 min (reference) |
| figure_3_caption.tex | 33 | ~300 | 2 min |
| **Total** | **2321** | **~16700** | **~107 min** |

---

## Terminology Reference

| Term | Definition | Used In |
|------|------------|---------|
| **Coordinator** | Meta-controller that manages expert selection | All docs |
| **Expert** | Individual routing strategy (Warmup or Tabula Rasa) | All docs |
| **Trust Distribution** | Probability π over experts | ARCHITECTURE_NOTES, IMPLEMENTATION_GUIDE |
| **Cumulative Loss** | Running sum of observed losses per expert | ARCHITECTURE_NOTES, IMPLEMENTATION_GUIDE |
| **Learning Rate** | η parameter controlling adaptation speed | All docs |
| **Importance Weighting** | Adjusting loss by inverse selection probability | THEORY_VS_IMPLEMENTATION |
| **Regret** | Difference from optimal performance | ARCHITECTURE_NOTES, THEORY_VS_IMPLEMENTATION |
| **Warmup Expert** | Expert initialized with prior data | All docs |
| **Tabula Rasa Expert** | Expert starting from scratch | All docs |

---

*Last Updated: 2026-01-24*  
*Total Documentation: 7 files, ~17K words*  
*Code Reference: router.py:3349-3484*

