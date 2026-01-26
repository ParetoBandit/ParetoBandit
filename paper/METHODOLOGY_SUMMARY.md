# Methodology Section - Summary

**File**: `paper/sections/methodology.tex`  
**Status**: ✅ Complete and compiled  
**Date**: January 25, 2026

---

## 📝 Section Structure

### 2. Methodology
- **2.1 Problem Formulation** - Formal mathematical definition
- **2.2 The banditGPT Architecture** - Three-layer system
  - Layer 1: Dynamic Pareto Filtering
  - Layer 2: Expert Corralling for Robustness
  - Layer 3: Cost-Aware LinUCB
- **2.3 Latent Semantic Transfer** - Zero-shot admission mechanism

---

## 🎯 Key Components

### 2.1 Problem Formulation

**Formal Definition**:
- Context vector: $x_t \in \mathbb{R}^d$ (prompt embedding)
- Model registry: $\mathcal{M}_t$ (dynamic, time-varying)
- Policy: $\pi$ selects arm $a_t \in \mathcal{M}_t$
- Observations: reward $r_t \in [0,1]$, cost $c_t \in \mathbb{R}^+$

**Objective (Lagrangian Relaxation)**:
```
Maximize Σ(E[r_t(a_t)|x_t] - λ·c_t(a_t))
```

**Trade-off Parameter λ**:
- High λ → identify "easy" prompts for cheap models
- Low λ → prioritize quality regardless of cost

---

### 2.2.1 Layer 1: Dynamic Pareto Filtering

**Purpose**: Enforce efficiency by pruning dominated actions

**Mechanism**:
- Estimate expected quality: $\hat{r}_m(x_t)$ for all models
- Construct local Pareto frontier: $\mathcal{P}_{x_t} \subseteq \mathcal{M}_t$
- Exclude model $m$ if ∃ model $m'$ where:
  ```
  r̂_m'(x_t) ≥ r̂_m(x_t)  AND  c_m' < c_m
  ```

**Benefit**: Never explore strictly dominated actions → accelerates convergence

---

### 2.2.2 Layer 2: Expert Corralling for Robustness

**Purpose**: Handle non-stationary quality shifts

**Architecture**:
- Portfolio of expert policies:
  - "Warmup Expert" (static priors)
  - "Tabula Rasa Expert" (high plasticity)
- Meta-learning strategy manages expert weights

**Mixing Parameter (Critical Innovation)**:
```
p_i,t = (1-γ)·w_i,t/Σw_j,t + γ/K
```

**Parameters**:
- γ = 0.05 (mixing parameter)
- Imposes uniform exploration floor
- Prevents "Expert Death"

**Why This Matters**:
- Temporarily poor expert can recover
- Asymptotic recovery if optimal expert changes
- Safety valve for non-stationarity

---

### 2.2.3 Layer 3: Cost-Aware LinUCB

**Model**: Bayesian Ridge Regression (LinUCB)

**Reward Approximation**:
```
E[r|x] = x^T θ*_m
```

**UCB Policy**:
```
a_t = argmax_{a∈P_{x_t}} (x_t^T θ̂_a + α√(x_t^T A_a^{-1} x_t) - λc_a)
```

**Components**:
- **Exploitation**: $x_t^T \hat{\theta}_a$ (expected reward)
- **Exploration**: $\alpha\sqrt{x_t^T A_a^{-1} x_t}$ (uncertainty bonus)
- **Cost penalty**: $-\lambda c_a$ (Lagrangian term)

**Amortized Regularization Floor**:
- Track effective decay of precision matrix $A_a$
- Proactively reinject regularization: $\lambda_{ridge}I$
- Prevent singular matrix inversions
- Avoid $O(d^3)$ operations at every step

**Operational Reality**: Numerical stability for long-running production systems

---

### 2.3 Latent Semantic Transfer (Zero-Shot Admission)

**Problem**: Cold Start when new model is released

**Solution**: Transfer learning from semantic neighbors

**Mechanism**:
1. New model registered: $m_{new}$
2. Find nearest semantic neighbor: $m_{near}$
3. Transfer preference, reset confidence:
   ```
   θ_new ← θ_near
   A_new ← n_eff·I
   ```

**Key Insight**: Decouple expectation from certainty
- **Mean (θ)**: "This model behaves like GPT-4" (hypothesis)
- **Variance (A)**: High uncertainty (controlled by n_eff)

**Behavior**:
- Aggressively exploit where hypothesis holds
- Rapidly correct where it diverges
- Zero-shot readiness without random exploration

---

## 📊 Mathematical Rigor (KDD Requirements)

### Formal Definitions ✅
- [x] Context space: $\mathbb{R}^d$
- [x] Action space: $\mathcal{M}_t$
- [x] Reward/cost observations
- [x] Lagrangian objective function

### Algorithms ✅
- [x] Dynamic Pareto Filtering (Equation 2)
- [x] Expert Corralling with mixing (Equation 3)
- [x] Cost-Aware LinUCB (Equation 4)
- [x] Latent Semantic Transfer (Equation 5)

### Safety Valves ✅
- [x] Mixing parameter γ (expert death prevention)
- [x] Amortized regularization (numerical stability)
- [x] Pareto filtering (efficiency enforcement)

---

## 🔧 Operational Reality (KDD Applied Track)

### Production Considerations

1. **Non-Stationarity**
   - Problem: "Good" models become "bad" due to drift
   - Solution: Expert Corralling with mixing
   - Guarantee: Asymptotic recovery

2. **Numerical Stability**
   - Problem: Singular matrix inversions in long runs
   - Solution: Amortized regularization floor
   - Benefit: Avoid $O(d^3)$ overhead

3. **Cold Start**
   - Problem: New models require random exploration
   - Solution: Latent Semantic Transfer
   - Benefit: Zero-shot readiness

4. **Efficiency**
   - Problem: Exploring dominated actions wastes budget
   - Solution: Dynamic Pareto Filtering
   - Benefit: Accelerated convergence

---

## 📚 Citations Included

### Primary References
1. **Corralling Algorithm** - `\cite{agarwal2017corralling}`
2. **LinUCB** - `\cite{li2010contextual}`

All citations properly defined in `references.bib`.

---

## 🎓 KDD Compliance

### Formal Rigor ✅
- [x] Mathematical problem definition (Lagrangian)
- [x] Formal notation (context, actions, rewards)
- [x] Equations for all algorithms
- [x] Complexity analysis (O(d³) mentioned)

### Operational Reality ✅
- [x] Safety valves described (Corralling, Regularization, Pareto)
- [x] Production challenges addressed (non-stationarity, stability, cold start)
- [x] Not just toy simulation (real operational concerns)

### Applied Data Science Track Requirements ✅
- [x] Real-world problem (LLM routing in production)
- [x] Practical constraints (budget, latency, stability)
- [x] Operational mechanisms (mixing, regularization, filtering)
- [x] Deployment considerations (zero-shot admission)

---

## 📈 Section Metrics

### Content
- **Subsections**: 3 (Problem, Architecture, Transfer)
- **Equations**: 5 (numbered)
- **Layers**: 3 (Pareto, Corralling, LinUCB)
- **Citations**: 2 (Corralling, LinUCB)

### LaTeX Features
- `\subsubsection{}` for three layers
- `\begin{equation}...\end{equation}` for formal math
- Custom commands: `\thetavec`, `\Amat`, `\neff`
- Proper mathematical notation throughout

---

## 🔗 Integration with Paper

### Flow
1. **Abstract**: Introduces Corralling and Semantic Transfer
2. **Introduction**: Motivates safety and agility challenges
3. **Methodology**: ✅ Formally defines the solution
4. **Experiments**: (Next) Validates the approach
5. **Results**: (Next) Demonstrates effectiveness

### Cross-References
- Introduction mentions "Figure 5" (Expert death prevention)
- Introduction mentions "Figure 6" (Semantic transfer)
- Methodology provides the formal mechanisms

---

## 📝 Writing Quality

### Strengths
- ✅ Clear hierarchical structure (3 layers)
- ✅ Formal mathematical definitions
- ✅ Operational justifications for each component
- ✅ Balance of theory and practice
- ✅ Proper citations

### KDD Reviewer Appeal
- **Theory reviewers**: Formal Lagrangian, UCB equations, complexity
- **Applied reviewers**: Safety valves, production concerns, stability
- **Both**: Clear motivation for each design choice

---

## 📊 Compilation Status

### PDF Output
- **Size**: 547KB (was 462KB with intro only)
- **Pages**: 3 (title + abstract + intro + methodology)
- **Status**: ✅ Compiles cleanly
- **Citations**: ✅ Processed with bibtex

### File Structure
```
paper/
├── main.tex                    # ✅ Updated (includes methodology)
├── sections/
│   ├── introduction.tex        # ✅ Complete
│   └── methodology.tex         # ✅ NEW - Complete
├── main.pdf                    # ✅ Updated (547KB, 3 pages)
└── references.bib              # ✅ Has all citations
```

---

## 🎯 Key Innovations Formalized

### 1. Three-Layer Architecture
- **Layer 1**: Dynamic Pareto Filtering (efficiency)
- **Layer 2**: Expert Corralling (robustness)
- **Layer 3**: Cost-Aware LinUCB (optimization)

### 2. Safety Mechanisms
- **Mixing parameter γ**: Prevents expert death
- **Regularization floor**: Ensures numerical stability
- **Pareto filtering**: Enforces efficiency

### 3. Latent Semantic Transfer
- **Decoupling**: θ (mean) vs A (variance)
- **Zero-shot**: No cold start penalty
- **Adaptive**: Rapid correction where hypothesis fails

---

## 🎉 Summary

**The methodology section is complete and KDD-compliant!**

### What Works
✅ Formal mathematical problem definition  
✅ Three-layer architecture clearly explained  
✅ Safety valves for production deployment  
✅ Latent Semantic Transfer formalized  
✅ Balance of rigor and operational reality  

### Impact
- Satisfies KDD formal rigor requirements
- Addresses Applied Data Science track concerns
- Provides complete algorithmic specification
- Demonstrates production readiness

### Next Steps
1. Write Experiments section (dataset, baselines, metrics)
2. Write Results section (validate 0.91 reward claim)
3. Add figures (Pareto frontier, Corralling weights, Transfer effectiveness)

---

**Status**: ✅ **METHODOLOGY COMPLETE**  
**File**: `paper/sections/methodology.tex`  
**PDF**: `paper/main.pdf` (547KB, 3 pages)  
**Next Action**: Write Experiments section

