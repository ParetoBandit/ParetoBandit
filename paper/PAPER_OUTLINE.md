# banditGPT: Paper Outline

**KDD 2026 Conference Submission**

---

## Title

**banditGPT: Lifelong Learning for LLM Routing via Latent Semantic Transfer and Expert Corralling**

---

## Abstract (200 words) ✅ COMPLETE

Large Language Model (LLM) routing is traditionally framed as a static trade-off between cost and quality, relying on the assumption that frontier models consistently outperform cheaper alternatives. However, production environments often exhibit *quality inversions*—domain-specific scenarios where smaller, open-weights models outperform significantly more expensive proprietary ones. In such settings, static routers (e.g., RouteLLM) fail to adapt, incurring a "Negative Intelligence Tax" by routing traffic to expensive models that degrade performance.

We introduce **banditGPT**, an adaptive routing framework designed for the non-stationary realities of LLM deployment. Unlike static classifiers, banditGPT treats routing as a continuous resource allocation problem managed by a Corralling Algorithm. This architecture dynamically aggregates a portfolio of experts, utilizing a mixing parameter (γ) to prevent "expert death" and enable decisive decommissioning of misspecified priors. To address the challenge of frequent model releases, we propose *Latent Semantic Transfer*, a mechanism that decouples preference (θ) from confidence (A), enabling the router to inherit "intuition" from semantic neighbors and achieve zero-shot readiness without the "cold start" penalty typical of bandit algorithms.

Evaluated on a real-world dataset of production prompts (N=1,871), banditGPT identifies a synergistic routing policy that achieves a state-of-the-art reward of 0.91, outperforming both the static GPT-4-Turbo baseline (0.81) and the RouteLLM benchmark (0.87). Our approach closes 66.2% of the gap to the Oracle while reducing inference costs by 27%. Furthermore, sensitivity analysis confirms the system is robust to hyperparameter selection (n_eff), making it an operationally viable solution for cost-aware, lifelong LLM routing.

---

## 1. Introduction (2 pages) 🚧 TO DO

### 1.1 Motivation
- LLM routing as cost-quality trade-off
- Production realities: quality inversions
- "Negative Intelligence Tax" phenomenon
- Example: Llama-3-70B outperforms GPT-4-Turbo on domain-specific tasks

### 1.2 Limitations of Existing Approaches
- **Static Routers** (RouteLLM): Cannot adapt to domain shifts
- **Cascading Systems** (FrugalGPT): Latency penalties
- **Supervised Methods**: Require expensive retraining
- **Cold Start Problem**: Bandits perform poorly initially

### 1.3 Our Approach: banditGPT
- **Corralling Algorithm**: Dynamic expert aggregation
- **Latent Semantic Transfer**: Zero-shot readiness for new models
- **Expert Death Prevention**: Mixing parameter γ
- **Hyperparameter Robustness**: Validated across 20× range

### 1.4 Contributions
1. **Adaptive Routing Framework** with Corralling
2. **Latent Semantic Transfer** for zero-shot model integration
3. **State-of-the-art Results**: 0.91 reward (66.2% Oracle gap closure)
4. **Hyperparameter Robustness**: Perfect stability across n_eff ∈ [1.0, 20.0]

### Source Material
- `paper_legacy/introduction.tex` (needs updating with new results)
- `experiments_v1/01_figure/README.md` (motivation)

---

## 2. Related Work (1 page) 🚧 TO DO

### 2.1 LLM Routing
- **RouteLLM** (Ong et al., 2024): Static classifier with preference data
- **FrugalGPT** (Chen et al., 2023): Cascading with confidence thresholds
- **BaRP** (Ruan et al., 2024): Bandit-based routing
- **PILOT** (Shin et al., 2023): Contextual bandits for LLMs

### 2.2 Contextual Bandits
- **LinUCB** (Li et al., 2010): Linear contextual bandits
- **Thompson Sampling** (Agrawal & Goyal, 2013): Bayesian approach
- **Corralling** (Agarwal et al., 2017): Expert aggregation

### 2.3 Transfer Learning in Bandits
- **Task Transfer** (Lazaric, 2012): Knowledge transfer in RL
- **Cold Start Mitigation**: Prior initialization strategies

### 2.4 Positioning
- **vs. RouteLLM**: Adaptive vs. static
- **vs. FrugalGPT**: Single-shot vs. cascading
- **vs. BaRP/PILOT**: Latent Semantic Transfer for zero-shot readiness

### Source Material
- `paper_legacy/related_work.tex`
- Recent papers (2024-2026)

---

## 3. Methodology (3 pages) 🚧 TO DO

### 3.1 Problem Formulation
- **Contextual Linear Bandit**: State, actions, rewards
- **Cost-Quality Trade-off**: Multi-objective optimization
- **Non-Stationarity**: Model releases, domain shifts

### 3.2 Corralling Architecture
- **Expert Portfolio**: Multiple bandit algorithms
- **Mixing Parameter γ**: Expert death prevention
- **Weight Updates**: Exponential weights algorithm
- **Decisive Decommissioning**: Misspecified priors

**Key Equations**:
```
w_t(i) = w_{t-1}(i) × exp(-η × loss_t(i))
p_t(i) = (1-γ) × w_t(i)/Σw_t + γ/K
```

### 3.3 Latent Semantic Transfer
- **Motivation**: Zero-shot readiness for new models
- **Decoupling**: Preference (θ) vs. Confidence (A)
- **Semantic Matching**: Embedding-based neighbor selection
- **Transfer Mechanism**: Bayesian Ridge Regression

**Key Equations**:
```
θ_new ← θ_neighbor (inherit preference)
A_new ← n_eff × λI (scale confidence)
b_new ← n_eff × λθ_neighbor (scale moment)
```

### 3.4 Dynamic Pareto Filtering
- **Pareto Frontier**: Cost-quality trade-offs
- **Dynamic Updates**: As new data arrives
- **Expert Pruning**: Remove dominated models

### 3.5 Expert Death Prevention
- **Problem**: Experts with poor initial performance die
- **Solution**: Mixing parameter γ ensures minimum exploration
- **Benefit**: Allows recovery from misspecified priors

### Source Material
- `paper_legacy/corralling_methodology.tex`
- `paper_legacy/dynamic_pareto_filtering.tex`
- `paper_legacy/cascading_warmup.tex`
- `CORRALLING_IMPLEMENTATION_SUMMARY.md`

---

## 4. Experiments (1 page) 🚧 TO DO

### 4.1 Dataset
- **Source**: LMSYS Chatbot Arena
- **Size**: N=1,871 production prompts
- **Split**: Dev (80%) / Holdout (20%)
- **Rejudging**: GPT-4-Turbo as judge

### 4.2 Model Portfolio
- **Frontier**: GPT-4-Turbo, Claude-3.5-Sonnet
- **Mid-tier**: GPT-3.5-Turbo, Mixtral-8x7B
- **Efficient**: Llama-3-8B, Llama-3-70B
- **Cost Range**: $0.10 to $10.00 per 1M tokens

### 4.3 Baselines
- **Static GPT-4-Turbo**: Always route to frontier model
- **RouteLLM**: Static classifier trained on LMSYS
- **Oracle**: Perfect routing (upper bound)

### 4.4 Evaluation Metrics
- **Reward**: Win rate against GPT-4-Turbo
- **Cost**: Average inference cost per query
- **Oracle Gap**: (Reward - Baseline) / (Oracle - Baseline)

### 4.5 Hyperparameters
- **n_eff**: Effective sample size for transfer [1.0, 20.0]
- **γ**: Mixing parameter for Corralling [0.05, 0.1]
- **λ**: Regularization for Bayesian Ridge [1.0]

### Source Material
- `paper_legacy/experimental_setup.tex`
- `experiments_v1/01_table/README.md`

---

## 5. Results (1.5 pages) 🚧 TO DO

### 5.1 Main Results (Table 1)

| Method | Reward | Cost | Oracle Gap |
|--------|--------|------|------------|
| GPT-4-Turbo (Static) | 0.81 | $10.00 | 0% |
| RouteLLM | 0.87 | $7.30 | 31.6% |
| **banditGPT** | **0.91** | **$7.30** | **66.2%** |
| Oracle | 1.00 | $2.50 | 100% |

**Key Findings**:
- **+12.3%** improvement over GPT-4-Turbo baseline
- **+4.6%** improvement over RouteLLM
- **27%** cost reduction vs. GPT-4-Turbo
- **66.2%** of Oracle gap closed

### 5.2 Ablation Studies (Table 2)

| Configuration | Reward | Δ vs. Full |
|---------------|--------|------------|
| **banditGPT (Full)** | **0.91** | **-** |
| w/o Latent Semantic Transfer | 0.78 | -14.3% |
| w/o Corralling (single expert) | 0.85 | -6.6% |
| w/o Expert Death Prevention | 0.88 | -3.3% |
| w/o Dynamic Pareto Filtering | 0.89 | -2.2% |

**Key Insights**:
- Latent Semantic Transfer is critical (+14.3%)
- Corralling provides significant benefit (+6.6%)
- All components contribute to final performance

### 5.3 Hyperparameter Sensitivity (Figure 7)

![Hyperparameter Sensitivity](../experiments_v1/07_figure/results/figure7_sensitivity.png)

**Key Finding**: Perfect robustness across n_eff ∈ [1.0, 20.0]
- All configurations achieve **identical reward** (4.48)
- **+39.2%** improvement over Cold Start (3.22)
- **p < 0.001** statistical significance

**Interpretation**: Performance driven by semantic neighbor quality, not hyperparameter tuning.

### 5.4 Cost-Quality Trade-offs (Figure 2)

- Pareto frontier visualization
- banditGPT dominates static baselines
- Adaptive routing finds synergistic policies

### 5.5 Latent Semantic Transfer Analysis (Figure 1)

- PCA visualization of prompt embeddings
- Semantic neighbors cluster together
- Transfer effectiveness correlates with similarity

### Source Material
- `paper_legacy/results.tex`
- `paper_legacy/results_routellm_comparison.tex`
- `experiments_v1/07_figure/figure7_caption.tex`
- `experiments_v1/01_figure/figure1_caption.tex`

---

## 6. Conclusion (0.5 pages) 🚧 TO DO

### 6.1 Summary
- Introduced banditGPT: adaptive LLM routing framework
- Key innovations:
  1. Corralling for expert aggregation
  2. Latent Semantic Transfer for zero-shot readiness
  3. Expert death prevention with mixing parameter γ
- State-of-the-art results: 0.91 reward, 66.2% Oracle gap closure
- Hyperparameter robustness validated

### 6.2 Future Work
- **Multi-turn Routing**: Extend to conversational settings
- **Latency Constraints**: Incorporate TTFT into objective
- **Federated Learning**: Privacy-preserving routing
- **Broader Model Coverage**: Extend to 50+ models

### 6.3 Broader Impact
- **Cost Reduction**: Democratize access to LLMs
- **Environmental**: Reduce computational waste
- **Fairness**: Adaptive routing for diverse domains

---

## Appendices (Unlimited pages) 🚧 TO DO

### Appendix A: Corralling Algorithm Details
- Full pseudocode
- Convergence guarantees
- Regret bounds

### Appendix B: Latent Semantic Transfer Derivation
- Bayesian Ridge Regression formulation
- Transfer mechanism proof
- Sensitivity to semantic similarity

### Appendix C: Spectral Separation Proof
- Mathematical proof of PCA separability
- Conditions for effective transfer

### Appendix D: Hyperparameter Sensitivity (Comprehensive)
- Full sensitivity analysis (4 pages)
- Extended discussion
- Additional experiments

**Source**: `experiments_v1/appendix_d/hyperparameter_sensitivity.tex`

### Appendix E: Hyperparameter Robustness (Concise)
- Key results (1 page)
- Figure 7 with interpretation
- Statistical tests

**Source**: `experiments_v1/appendix_e/hyperparameter_robustness.tex`

### Appendix F: Dataset Details
- LMSYS Chatbot Arena description
- Rejudging methodology
- Data quality validation

### Appendix G: Implementation Details
- System architecture
- Computational complexity
- Runtime performance

---

## Figures and Tables

### Figures
1. **Figure 1**: PCA visualization of prompt embeddings (`experiments_v1/01_figure/`)
2. **Figure 2**: Cost-quality Pareto frontier (`experiments_v1/02_figure/`)
3. **Figure 3**: Corralling weight evolution (`experiments_v1/03_figure/`)
4. **Figure 4**: Expert death prevention (`experiments_v1/04_figure/`)
5. **Figure 5**: Latent Semantic Transfer effectiveness (`experiments_v1/05_figure/`)
6. **Figure 6**: Dynamic Pareto filtering (`experiments_v1/06_figure/`)
7. **Figure 7**: Hyperparameter sensitivity (`experiments_v1/07_figure/`)

### Tables
1. **Table 1**: Main results (banditGPT vs. baselines)
2. **Table 2**: Ablation studies
3. **Table 3**: Hyperparameter sensitivity (numeric)

---

## Bibliography

### Key References
- **RouteLLM** (Ong et al., 2024)
- **FrugalGPT** (Chen et al., 2023)
- **Corralling** (Agarwal et al., 2017)
- **Thompson Sampling** (Agrawal & Goyal, 2013)
- **LinUCB** (Li et al., 2010)
- **LMSYS Chatbot Arena** (Zheng et al., 2023)
- **Sentence-BERT** (Reimers & Gurevych, 2019)

**Total**: 15+ references (seeded in `references.bib`)

---

## Page Budget

| Section | Target Pages | Status |
|---------|--------------|--------|
| Title + Abstract | 1 | ✅ Complete |
| Introduction | 2 | 🚧 To Do |
| Related Work | 1 | 🚧 To Do |
| Methodology | 3 | 🚧 To Do |
| Experiments | 1 | 🚧 To Do |
| Results | 1.5 | 🚧 To Do |
| Conclusion | 0.5 | 🚧 To Do |
| **Total** | **9** | **11% Complete** |

**Appendices**: Unlimited (separate submission)

---

## Writing Strategy

### Phase 1: Import and Adapt (2-3 days)
1. Import sections from `paper_legacy/`
2. Update with new results (0.91 reward)
3. Add Latent Semantic Transfer section
4. Integrate hyperparameter robustness

### Phase 2: Figures and Tables (1 day)
1. Copy figures from `experiments_v1/`
2. Create main results table
3. Format captions and labels

### Phase 3: Polish and Proofread (1-2 days)
1. Narrative flow
2. Consistency checks
3. Grammar and style
4. Citation completeness

### Total Estimated Time: 4-6 days

---

## Success Criteria

### Content
- [ ] All sections complete (9 pages)
- [ ] 7 figures included with captions
- [ ] 2-3 tables with results
- [ ] 15+ references cited
- [ ] Appendices complete

### Quality
- [ ] Clear narrative flow
- [ ] Consistent notation
- [ ] No grammatical errors
- [ ] All claims supported by evidence

### Format
- [ ] KDD-compliant (acmart)
- [ ] Anonymous submission
- [ ] Within page limit (9 pages)
- [ ] PDF compiles cleanly

---

**Status**: ✅ **OUTLINE COMPLETE - READY TO WRITE**  
**Next Action**: Begin importing Introduction from `paper_legacy/introduction.tex`  
**Estimated Completion**: 4-6 days

