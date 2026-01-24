# Experiment 5 File Manifest

## Directory Structure

```
experiments_v1/05_corralling/
├── README.md                       # Main experiment documentation
├── EXPERIMENT_SUMMARY.md           # Executive summary and key results
├── FILES.md                        # This file (manifest)
├── test_hybrid_corralling.py       # Main evaluation script (Python)
│
└── results/
    ├── corralling_results.tex      # KDD-compliant LaTeX section
    ├── CORRALLING_SUCCESS.md       # Detailed analysis of bug fix
    ├── results.json                # Numerical results (JSON)
    ├── hybrid_comparison.png       # Performance plots (PNG, 281 KB)
    └── expert_weights_evolution.png # Weight adaptation plot (PNG, 217 KB)
```

---

## File Descriptions

### Documentation

**README.md** (7.2 KB)
- Quick start guide
- Main results summary
- Implementation details with code examples
- Parameter sensitivity analysis
- Reproducibility instructions
- Citation information

**EXPERIMENT_SUMMARY.md** (10.8 KB)
- Executive summary
- What we built (code + evaluation)
- Critical bug & fix explanation
- Experimental validation details
- Paper contributions
- Usage examples for production
- Computational cost analysis
- Future work recommendations

**FILES.md** (This file)
- Complete file manifest
- File descriptions and sizes
- Quick reference guide

### Code

**test_hybrid_corralling.py** (12.3 KB)
- Main evaluation script
- Compares 3 strategies: Warmup, Tabula Rasa, Hybrid (Corralling)
- Loads data from `src/bandit_gpt/data/offline_dataset/`
- Tracks expert weights over time
- Generates plots and saves results
- ~400 lines of Python code

**Dependencies:**
```python
numpy
matplotlib
tqdm
sentence-transformers
joblib
json
gzip
```

**Usage:**
```bash
python test_hybrid_corralling.py \
    --gamma 0.05 \
    --learning-rate 0.1 \
    --sample-size 1121 \
    --output results/
```

### Results

**results/results.json** (681 bytes)
```json
{
  "Warmup": {
    "cumulative_regret": 126.0,
    "avg_reward": 0.8359,
    "model_usage": {...}
  },
  "Tabula Rasa": {
    "cumulative_regret": 43.0,
    "avg_reward": 0.9099,
    "model_usage": {...}
  },
  "Hybrid (Corralling)": {
    "cumulative_regret": 88.0,
    "avg_reward": 0.8698,
    "model_usage": {...}
  }
}
```

**results/hybrid_comparison.png** (281 KB, 3600×1500 px, 300 DPI)
- Left panel: Cumulative regret over time
- Right panel: Average reward over time
- Three lines: Warmup (orange), Tabula Rasa (green), Hybrid (blue)
- Publication-ready quality

**results/expert_weights_evolution.png** (217 KB, 3000×1800 px, 300 DPI)
- Shows how Corralling adapted weights over time
- X-axis: Samples (0-1,121)
- Y-axis: Expert weight (0-1)
- Two lines: Warmup expert (orange), Tabula Rasa expert (green)
- Annotation box showing final weights
- Publication-ready quality

**results/corralling_results.tex** (9.4 KB)
- KDD-compliant LaTeX section
- Ready for inclusion in paper
- Includes:
  - Motivation and experimental setup
  - Results table (Table~\ref{tab:corralling-results})
  - Two figures with captions
  - Implementation details
  - Honest reporting of bug and fix
  - Discussion and future work
  - Key takeaways in boxed environment
  - Reproducibility instructions

**results/CORRALLING_SUCCESS.md** (9.3 KB)
- Detailed analysis of the importance weighting bug
- Before/After comparison
- Code examples showing bug and fix
- Performance analysis
- Implications for paper
- Recommendations for practitioners

---

## Core Implementation (Not in This Directory)

The CorrallingRouter class is implemented in the main library:

**src/bandit_gpt/router.py**
- Lines ~3365-3445 (80 lines)
- Class: `CorrallingRouter`
- Methods:
  - `__init__()` - Initialize with uniform weights
  - `select_model()` - Sample expert and get model selection
  - `update()` - Importance-weighted loss update
  - `get_expert_weights()` - Get current weights for monitoring

---

## Data Dependencies

This experiment requires:

1. **Evaluation Data:**
   - `src/bandit_gpt/data/offline_dataset/dev_rewards_complete.jsonl.gz`
   - 1,121 prompts with rewards for Mixtral and GPT-4-Turbo
   - ~2.1 MB compressed

2. **Warmup Priors:**
   - `src/artifacts/priors_warmup.joblib`
   - Pre-trained A/b matrices from 80k RouteLLM battles
   - ~450 KB

3. **PCA Model:**
   - `src/artifacts/pca_model.joblib`
   - Dimensionality reduction (384D → 32D)
   - ~180 KB

4. **Sentence Transformer:**
   - Model: `sentence-transformers/all-MiniLM-L6-v2`
   - Downloaded automatically on first run
   - ~90 MB

---

## Key Metrics

### Performance Results

| Strategy | Cumul. Regret | Avg Reward | GPT-4T % |
|----------|---------------|------------|----------|
| Tabula Rasa | 43.0 | 0.910 | 68.1% |
| Hybrid | 88.0 | 0.870 | 67.9% |
| Warmup | 126.0 | 0.836 | 84.6% |

### Improvement vs Warmup

- **Hybrid:** -30.2% regret, +4.0% reward
- **Tabula Rasa:** -65.9% regret, +8.9% reward

### Expert Weights (Final)

- Warmup: 0.23 (23%)
- Tabula Rasa: 0.77 (77%)

---

## Reproducibility

### Deterministic Results

All experiments use `seed=42` for:
- Data sampling
- Expert selection
- Random initialization

Running the same command twice will produce **identical results**.

### Expected Runtime

- **Hardware:** M1 MacBook Pro (2021)
- **Time:** ~30 seconds for 1,121 samples
- **CPU:** ~25% utilization (single-threaded evaluation)
- **Memory:** ~500 MB peak

---

## For Paper Submission

### Files to Include

1. **Main text:** `results/corralling_results.tex`
   - Drop into paper as Section 5 or 6
   - Compile with paper to verify formatting

2. **Figures:**
   - `results/hybrid_comparison.png` → Figure X
   - `results/expert_weights_evolution.png` → Figure Y (optional, could be appendix)

3. **Supplementary:**
   - `test_hybrid_corralling.py` → Reproducibility supplement
   - `results/results.json` → Raw data supplement

### LaTeX Compilation

```bash
# Copy tex file to paper directory
cp results/corralling_results.tex ../paper/sections/

# Update figure paths in tex file if needed
sed -i 's|experiments_v1/05_corralling/results/|figures/|g' \
    ../paper/sections/corralling_results.tex

# Copy figures
cp results/*.png ../paper/figures/

# Compile paper
cd ../paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

---

## Contact & Support

For questions about this experiment:
1. Open a GitHub issue
2. Contact the BanditGPT team
3. Review the comprehensive documentation in README.md

---

## Version History

- **v1.0 (2026-01-24):** Initial release
  - Implemented CorrallingRouter
  - Fixed importance weighting bug
  - Achieved 30% regret reduction
  - Generated KDD-compliant LaTeX

---

## License

Same as main BanditGPT repository.

---

*Last updated: 2026-01-24*  
*Experiment status: ✅ Complete*  
*Paper ready: ✅ Yes*

