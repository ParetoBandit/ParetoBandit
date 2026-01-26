# Files in 03_figure

This folder contains the implementation of Figure 3: Corralled Bandit with Semantic Projection.

## Core Implementation

### `corralled_semantic_analysis.py`
**Main implementation script** - Runs the complete Corralling experiment with two phases:
1. **Phase 1 (Optimization)**: Train on labeled data (N=1,871 or N=80k) using importance-weighted loss
2. **Phase 2 (Visualization)**: Project learned policy onto 1M semantic space

**Key Features**:
- Mathematically correct importance weighting: $\hat{\ell}_{t,e} = \frac{\mathbb{1}_{e=e^*}(1 - r_t)}{\rho_{t,e}}$
- Strict separation between optimization (with rewards) and visualization (without rewards)
- No fake numbers - only uses actual rewards from labeled data
- Generates Figure 3 and training metrics

**Usage**:
```bash
python experiments_v1/03_figure/corralled_semantic_analysis.py \
    --learning-rate 1.0 \
    --gamma 0.05 \
    --train-size 1871
```

**Output**:
- `results/figure3_corralling_semantic_analysis.png` - Main figure
- `results/training_metrics.png` - Training curves
- `results/results.json` - Numerical results

---

## Testing

### `test_corralling.py`
**Quick test script** - Verifies the Corralling implementation works correctly.

Runs on a small sample (N=100) to check:
- Importance-weighted loss estimation
- Expert weight updates
- Algorithm convergence
- Implementation correctness

**Usage**:
```bash
python experiments_v1/03_figure/test_corralling.py
```

**Expected Output**:
```
✅ All checks passed! Implementation is correct.
```

---

## Documentation

### `README.md`
**Comprehensive documentation** covering:
- Algorithm overview and motivation
- Mathematical framework (importance weighting, exponential weights)
- Implementation details (Phase 1 & 2)
- Key results and insights
- Paper strategy (Main Results, Figure 1, Figure 3)
- Usage instructions and examples
- Output file descriptions

**For**: Understanding the complete implementation

---

### `IMPLEMENTATION_SUMMARY.md`
**Executive summary** with focus on:
- Core problem (warmup bias)
- Corralling solution
- Mathematical correctness (importance weighting, no counterfactuals)
- Implementation details (code snippets)
- Why this matters (safety guarantee, practical impact)
- Paper strategy and talking points

**For**: Quick overview and key insights

---

### `QUICKSTART.md`
**Practical guide** for running experiments:
- Prerequisites (data, models, dependencies)
- Running options (basic, quick test, custom parameters)
- Output files and interpretation
- Troubleshooting common issues
- Next steps and ablation studies

**For**: Getting started quickly

---

### `FILES.md` (this file)
**Index of all files** in the folder with descriptions and relationships.

**For**: Navigation and understanding folder structure

---

## Paper Integration

### `figure3_caption.tex`
**LaTeX for paper** including:
- Figure caption for Figure 3
- Mathematical framework (Section 4.3: Corralling Algorithm)
- Semantic projection methodology (Section 4.4)
- References (Agarwal et al., 2017)

**For**: Direct integration into paper

**Sections**:
1. Figure caption with detailed explanation
2. Mathematical framework:
   - Problem setup
   - Importance-weighted loss estimation (Eq. 1)
   - Exponential weights update (Eq. 2)
   - Safety guarantee (Eq. 3)
3. Semantic projection methodology
4. References

---

## Results (Generated)

### `results/`
**Output directory** created by running `corralled_semantic_analysis.py`.

**Files**:
- `figure3_corralling_semantic_analysis.png` - Main figure (300 DPI)
- `figure3_corralling_semantic_analysis_hires.png` - High-res version (600 DPI)
- `training_metrics.png` - Training curves (regret & reward)
- `results.json` - Numerical results (JSON format)

**Structure of results.json**:
```json
{
  "learning_rate": 1.0,
  "gamma": 0.05,
  "train_size": 1871,
  "cumulative_regret": 245.32,
  "avg_reward": 0.8456,
  "final_expert_weights": [0.247, 0.753],
  "model_usage": {
    "mixtral-8x7b-instruct-v0.1": 456,
    "gpt-3.5-turbo-0125": 234,
    ...
  }
}
```

---

## File Relationships

```
03_figure/
│
├── corralled_semantic_analysis.py  ← Main implementation
│   ├── Uses: CorrallingRouter (from src/bandit_gpt/router.py)
│   ├── Loads: Labeled data, PCA model, warmup priors
│   └── Generates: results/ folder
│
├── test_corralling.py  ← Quick test
│   ├── Uses: Same components as main script
│   └── Validates: Implementation correctness
│
├── README.md  ← Comprehensive docs
│   ├── Explains: Algorithm, implementation, usage
│   └── References: All other files
│
├── IMPLEMENTATION_SUMMARY.md  ← Executive summary
│   ├── Focuses on: Key insights and strategy
│   └── Links to: README.md for details
│
├── QUICKSTART.md  ← Practical guide
│   ├── Provides: Step-by-step instructions
│   └── Troubleshoots: Common issues
│
├── figure3_caption.tex  ← Paper integration
│   ├── Contains: LaTeX for figure and math
│   └── References: Results from main script
│
├── FILES.md (this file)  ← Index
│   └── Describes: All files and relationships
│
└── results/  ← Generated outputs
    ├── figure3_corralling_semantic_analysis.png
    ├── figure3_corralling_semantic_analysis_hires.png
    ├── training_metrics.png
    └── results.json
```

---

## Workflow

### 1. Understanding the Implementation

**Start with**: `IMPLEMENTATION_SUMMARY.md`
- Get the big picture
- Understand the problem and solution
- See key insights

**Then read**: `README.md`
- Detailed algorithm explanation
- Mathematical framework
- Implementation details

**Finally check**: `figure3_caption.tex`
- See how it fits in the paper
- Review mathematical notation

---

### 2. Running the Experiment

**Start with**: `QUICKSTART.md`
- Check prerequisites
- Choose running option
- Follow step-by-step instructions

**Test first**: `test_corralling.py`
```bash
python experiments_v1/03_figure/test_corralling.py
```

**Run experiment**: `corralled_semantic_analysis.py`
```bash
python experiments_v1/03_figure/corralled_semantic_analysis.py
```

**Review outputs**: `results/` folder
- Check figures visually
- Read `results.json` for numbers

---

### 3. Integrating into Paper

**Use**: `figure3_caption.tex`
- Copy figure caption
- Include mathematical framework
- Add references

**Reference**: Results from `results.json`
- Cumulative regret
- Final expert weights
- Model usage statistics

**Cite**: Agarwal et al. (2017) for Corralling algorithm

---

## Key Concepts

### Importance-Weighted Loss

**Formula**: $\hat{\ell}_{t,e} = \frac{\mathbb{1}_{e=e^*}(1 - r_t)}{\rho_{t,e}}$

**Meaning**: 
- Only the chosen expert gets penalized
- Weight by inverse selection probability for unbiased estimation
- Non-chosen experts get 0 loss (no counterfactuals)

**Why**: Creates unbiased estimator that allows algorithm to detect which expert is better

---

### Exponential Weights

**Formula**: $w_{t+1,e} = \frac{\exp(-\eta \cdot L_{t,e})}{\sum_{e'} \exp(-\eta \cdot L_{t,e'})}$

**Meaning**:
- Experts with lower cumulative loss get higher weight
- Learning rate $\eta$ controls adaptation speed
- Weights always sum to 1 and are non-negative

**Why**: Provably optimal way to combine experts with regret bound $O(\sqrt{T \log E})$

---

### Safety Guarantee

**Statement**: $\mathbb{E}[\text{Regret}] \leq \min_{e} \mathbb{E}[\text{Regret}_e] + O(\sqrt{T \log E})$

**Meaning**:
- Corralling performs nearly as well as the best expert
- Overhead is only $O(\sqrt{T})$, which is negligible
- Protects against negative transfer from bad warmup priors

**Why**: Provides theoretical guarantee that algorithm will adapt to better expert

---

### Semantic Projection

**Concept**: Project learned policy onto 1M semantic space without evaluating rewards

**Why**:
- Shows coverage across semantic manifold
- Demonstrates cluster structure at scale
- Validates generalization from 1,871 to 1M
- No fake numbers - just visualization

**Result**: Easy cluster (94.1%) is exploitable, which algorithm discovers automatically

---

## Dependencies

### External Dependencies
- `numpy` - Numerical operations
- `matplotlib` - Plotting
- `scipy` - KDE for density estimation
- `sentence-transformers` - Embedding prompts
- `scikit-learn` - PCA (via joblib)
- `tqdm` - Progress bars

### Internal Dependencies
- `src/bandit_gpt/router.py` - CorrallingRouter class
- `src/bandit_gpt/calibration.py` - SimpleLinUCBRouter, embed_prompt
- `src/bandit_gpt/config_legacy.py` - Paths and constants

### Data Dependencies
- `src/bandit_gpt/data/offline_dataset/dev_rewards_complete.jsonl.gz` - Labeled data
- `src/artifacts/pca_model_routellm.joblib` - PCA model
- `src/artifacts/warmup_priors_routellm.joblib` - Warmup priors
- `experiments_v1/appendix_d/data/lmsys_chat_1M.jsonl.gz` - 1M dataset (optional)

---

## Common Tasks

### Run basic experiment
```bash
python experiments_v1/03_figure/corralled_semantic_analysis.py
```

### Test implementation
```bash
python experiments_v1/03_figure/test_corralling.py
```

### Compare learning rates
```bash
for eta in 0.5 1.0 2.0; do
    python experiments_v1/03_figure/corralled_semantic_analysis.py \
        --learning-rate $eta \
        --output results/eta_$eta
done
```

### View results
```bash
open experiments_v1/03_figure/results/figure3_corralling_semantic_analysis.png
cat experiments_v1/03_figure/results/results.json | python -m json.tool
```

### Clean outputs
```bash
rm -rf experiments_v1/03_figure/results/
```

---

## Questions?

- **Algorithm questions**: See `README.md` Section "The Corralling Algorithm"
- **Implementation questions**: See `IMPLEMENTATION_SUMMARY.md` Section "Implementation Details"
- **Usage questions**: See `QUICKSTART.md`
- **Paper integration**: See `figure3_caption.tex`
- **Debugging**: See `QUICKSTART.md` Section "Troubleshooting"
- **Code reference**: Check `src/bandit_gpt/router.py` (CorrallingRouter class)

---

## Version History

- **v1.0** (2026-01-25): Initial implementation
  - Corralling algorithm with importance weighting
  - Semantic projection onto 1M space
  - Comprehensive documentation
  - Test suite

