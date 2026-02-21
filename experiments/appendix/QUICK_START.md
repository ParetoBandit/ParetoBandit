# Appendix Quick Start Guide

**For**: Paper authors and readers  
**Purpose**: Quickly navigate and use the appendix

---

## 🎯 What's Where

| Need... | Go to... |
|---------|----------|
| **Proofs and theory** | [Appendix A](A_mathematical_foundations/) - Regret bounds, safety, prior transfer (incl. n_eff), ablation table |
| **Data details** | [Appendix B](B_dataset_details/) - Validation methodology, feature pipeline |
| **Extra experiments** | [Appendix C](C_extended_results/) - Catastrophic failure, K=5 portfolio |
| **How to implement** | [Appendix D](D_implementation_details/) - Config, experimental setup |
| **Limitations & future** | [Appendix E](E_limitations_and_future_work/) - Limitations, bandit router positioning (PILOT/BaRP) |

---

## 📖 How to Read

### For Quick Review (15 minutes)
1. Read main [README.md](README.md) - Overview
2. Scan section READMEs for key results
3. Look at figures in `*/figures/` directories

### For Detailed Review (1-2 hours)
1. Compile `APPENDIX_MASTER.tex` for full PDF
2. Read each section in order (A → E)
3. Check cross-references with main paper

### For Reproduction (Implementation)
1. Read [Appendix D](D_implementation_details/) - Implementation details
2. Check [Appendix A.3](A_mathematical_foundations/) - Prior transfer & n_eff guidance
3. Review [Appendix A.2](A_mathematical_foundations/) - Ablation validation (45 experiments)

---

## 🔨 How to Compile

### Standalone Appendix PDF
```bash
cd experiments/appendix
pdflatex APPENDIX_MASTER.tex
pdflatex APPENDIX_MASTER.tex  # Second pass for references
```

### Integrate with Main Paper
Add to your main paper `.tex` file:
```latex
% At the end of main content
\appendix
\input{experiments/appendix/APPENDIX_MASTER.tex}
```

---

## 📊 Key Statistics

- **5 Sections** (A-E)
- **15 LaTeX Files**
- **8 README Files**
- **45+ Experiments** documented
- **4+ Figures** organized
- **~20-25 Pages** (estimated)

---

## 🎨 Visual Structure

```
appendix/
│
├── 📄 APPENDIX_MASTER.tex          ← Compile this
├── 📘 README.md                    ← Read this first
│
├── A_mathematical_foundations/     ← Theory, proofs, prior transfer (incl. n_eff), ablation table
├── B_dataset_details/              ← Data & validation
├── C_extended_results/             ← Appendix C: Extra experiments
├── D_implementation_details/       ← Appendix D: Config & setup
└── E_limitations_and_future_work/  ← Appendix E: Limitations & future
```

---

## 🔍 What Goes in Appendix vs Main Paper?

### Main Paper ✅
- Core contributions and key results
- Essential figures (1-8)
- Critical tables (1-2)
- Main experimental results
- → Keep concise, ~9 pages

### Appendix ✅
- Detailed proofs & prior transfer theory (A)
- Extended data analysis (B)
- Supplementary experiments (C)
- Implementation details (D)
- Limitations & discussion (E)
- → Can be longer, ~20-25 pages

---

## 💡 Quick Tips

### Finding Content
- Use section READMEs for quick navigation
- Check `APPENDIX_CONTENT_MAP.md` for detailed mapping
- Figures organized by section in `*/figures/`

### Adding Content
1. Pick section (A-E) using content categories
2. Create file: `X#_descriptive_name.tex`
3. Update section README
4. Add `\input{}` to APPENDIX_MASTER.tex

### Citation
Reference appendix sections in main paper:
```latex
See Appendix~\ref{appendix:warmup_transfer} 
for prior transfer analysis and n_eff guidance.
```

---

## 📚 Documentation Hierarchy

```
QUICK_START.md (you are here)
    ↓
README.md (comprehensive overview)
    ↓
APPENDIX_CONTENT_MAP.md (detailed mapping)
    ↓
Section READMEs (A-E specifics)
```

---

## ✅ Pre-Submission Checklist

- [ ] Compile APPENDIX_MASTER.tex successfully
- [ ] All figures referenced and present
- [ ] Cross-references work with main paper
- [ ] Section numbers consistent
- [ ] Page numbers continuous
- [ ] References formatted correctly
- [ ] Tables use booktabs
- [ ] Fonts are Times/serif, 10pt
- [ ] Margins are 1" all sides

---

## 🚀 Most Common Tasks

### View Full Appendix
```bash
cd appendix && pdflatex APPENDIX_MASTER.tex && open APPENDIX_MASTER.pdf
```

### Find Specific Content
```bash
grep -r "semantic transfer" appendix/*.tex
```

### List All Figures
```bash
find appendix -name "*.png" -o -name "*.pdf" | grep figures
```

### Check File Count
```bash
find appendix -name "*.tex" | wc -l
```

---

## 📞 Need Help?

1. **Overview**: Read [README.md](README.md)
2. **Detailed map**: Check [APPENDIX_CONTENT_MAP.md](APPENDIX_CONTENT_MAP.md)
3. **Planning doc**: See `../APPENDIX_ORGANIZATION_PLAN.md`
4. **Section help**: Read section-specific README

---

**Status**: ✅ Ready for review and testing  
**Next**: Compile and integrate with main paper
