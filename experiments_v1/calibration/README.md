# Router Calibration Guide

This guide helps you calibrate a contextual bandit router for your specific domain using pre-trained warmup priors.

> **📦 Code Structure:**  
> - **CLI Tools**: `scripts/calibration/` (find_gamma.py, calibrate_router.py)  
> - **Library**: `src/bandit_gpt/calibration.py` (CalibratedRouter class)  
> - **Research Artifacts**: This folder (experiments_v1/calibration/)

---

## 📋 Overview

**Problem:** The pre-trained warmup priors (80,000 samples from RouteLLM) may not match your domain's optimal routing policy.

**Solution:** **Domain Calibration** - Use 100-200 labeled samples from YOUR domain to adapt the router.

**Key Innovation:** **Covariance Inflation (γ-scaling)** - We apply a one-time domain alignment transformation to weaken the warmup priors, allowing calibration samples to meaningfully update the router's beliefs and discover domain-specific structure.

---

## 🚀 Quick Start

### Step 1: Prepare Your Calibration Data

Create a JSONL file with prompts and rewards from your domain:

```jsonl
{"prompt": "How do I optimize React rendering?", "rewards": {"mistralai/mixtral-8x7b-instruct": 0.85, "openai/gpt-4-turbo": 0.95}}
{"prompt": "Explain async/await in Python", "rewards": {"mistralai/mixtral-8x7b-instruct": 0.90, "openai/gpt-4-turbo": 0.92}}
{"prompt": "Write a hello world in Rust", "rewards": {"mistralai/mixtral-8x7b-instruct": 0.95, "openai/gpt-4-turbo": 0.93}}
```

**Requirements:**
- 100-200 prompts (minimum)
- Prompts representative of your domain
- Ground-truth rewards for both models (0.0-1.0 scale)

**How to get rewards?**

Option 1: Use our CoT judge panel (recommended):
```bash
python3 ../rejudge_cot.py --mode pareto --limit 200
```

Option 2: Use your own quality labels:
- Binary: 0 = fail, 1 = pass
- Or continuous: 0.0-1.0 quality score

### Step 2: Find Your Optimal Gamma

The gamma factor controls how much influence your calibration data has:

```bash
# From project root
python3 scripts/calibration/find_gamma.py \
  --calibration-data my_calibration_data.jsonl \
  --output results/ \
  --target-usage 20.0  # Optional: if you know oracle usage %
```

**Output:**
- `results/gamma_analysis.png` - Visualization of gamma effects
- `results/gamma_results.json` - Numerical results
- Recommended gamma value printed to console

**Example output:**
```
RESULTS: Gamma Factor Comparison
================================================================================

   Gamma      Eff. N Calib/Prior  Final Strong%     Delta
--------------------------------------------------------------------------------
   1.000      80,000        0.002          78.5%     +0.0%
   0.100       8,000        0.019          65.3%    -13.2%
   0.010         800        0.186          42.1%    -36.4%
   0.005         400        0.373          28.7%    -49.8%
   0.002         160        0.931          21.2%    -57.3%  ← Best!
   0.001          80        1.863          18.9%    -59.6%

💡 RECOMMENDATION: Use gamma = 0.002 for your domain
```

**Understanding the results:**
- **Eff. N**: Effective warmup sample size after scaling
- **Calib/Prior**: Ratio of calibration influence to prior influence
  - < 0.1: Calibration has minimal impact (inertia)
  - ≈ 1.0: Balanced influence (optimal)
  - > 2.0: Calibration dominates (may overfit)
- **Final Strong%**: Strong model usage after calibration
- **Delta**: Change from baseline (gamma=1.0)

### Step 3: Calibrate Your Router

Use the recommended gamma to create your calibrated router:

```bash
# From project root
python3 scripts/calibration/calibrate_router.py \
  --calibration-data my_calibration_data.jsonl \
  --gamma 0.002 \
  --output my_calibrated_router.joblib
```

**Output:**
- Calibrated router saved to `my_calibrated_router.joblib`
- Calibration statistics printed to console

### Step 4: Use Your Calibrated Router

```python
import joblib
from sentence_transformers import SentenceTransformer
from bandit_gpt.calibration import CalibratedRouter

# Load resources
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
pca_model = joblib.load("artifacts/pca_23_routellm.joblib")

# Load your calibrated router
router = CalibratedRouter.load(
    "my_calibrated_router.joblib",
    encoder,
    pca_model
)

# Route queries
user_prompt = "How do I deploy a FastAPI app?"
selected_model = router.select_model(user_prompt)
print(f"Route to: {selected_model}")

# Call the selected LLM
response = call_llm_api(selected_model, user_prompt)
```

---

## 📊 Mathematical Foundation: Domain Alignment via Covariance Inflation

### The Challenge: Bayesian Inertia

A critical challenge in cross-domain transfer for contextual bandits is **Bayesian Inertia**—the phenomenon where strong priors from source domain training overwhelm small calibration sets from the target domain.

In LinUCB, after observing N warmup samples, the precision matrix **A ∈ ℝ^(d×d)** encodes high confidence:

```
Confidence interval: σ²_m(x) ∝ x^T A_m^(-1) x
```

When **A** is large (from 80,000 warmup samples), **σ²** becomes small, creating **mathematically rigid beliefs**. A small calibration set (e.g., 150 samples) cannot meaningfully update these beliefs, leading to maladaptation.

**Empirical Evidence:** Without intervention, our router exhibits **0% adaptation** during calibration—strong model usage remains stuck at warmup bias despite exposure to real-world samples, resulting in **80.4% cost overrun** compared to the optimal policy.

### The Solution: One-Time Covariance Inflation

We introduce a **one-time Domain Alignment phase** that occurs during deployment from source (synthetic) to target (real-world) domain. By applying covariance inflation:

```
A_adapted = A_warmup × γ,  where γ ∈ (0, 1]
```

we reduce the **effective sample size**:

```
N_eff = N_warmup × γ = 80,000 × γ
```

This increases uncertainty, allowing calibration samples to meaningfully update the router's beliefs.

### Domain-Aware Transfer: What Changes, What Doesn't

Critically, covariance inflation is **not** retraining from scratch:

- ✅ **Preserved**: Embedding function φ(x), PCA projection, linguistic features learned from 80K samples (e.g., "coding prompts usually need stronger models")
- 🔄 **Recalibrated**: Quantitative confidence about task frequency (e.g., % of prompts requiring the strong model)

This enables the router to **discover** domain-specific structure (e.g., bimodal "easy or impossible" distributions) with minimal data.

### The Critical Ratio

Adaptation succeeds when the calibration set has sufficient influence:

```
Calibration Power = N_calibration / N_eff = N_calibration / (N_warmup × γ)
```

Our experiments show that when this ratio **≈ 1** (balanced influence), the router autonomously adapts. Results from our current calibration:

| γ     | N_eff | Calib/Prior | Adaptation | Interpretation                          |
|-------|-------|-------------|------------|-----------------------------------------|
| 1.0   | 80,000| 0.014       | 0%         | ❌ Bayesian Inertia: Calibration ignored |
| 0.1   | 8,000 | 0.140       | -15%       | ⚠️ Partial: Calibration has weak impact  |
| **0.010** | **800** | **1.401** | **-24%** | ✅ **Optimal: Calibration dominates** |
| 0.005 | 400   | 2.803       | -23%       | ⚠️ Over-inflation: May lose warmup value |
| 0.001 | 80    | 14.012      | -23%       | ⚠️ Extreme: Warmup nearly discarded      |

With **γ=0.010**, we achieve:
- **Calibration/Prior ratio = 1.401**: Each calibration sample has ~1.4× the influence of a warmup sample
- **Maximum adaptation**: -24.1% change in routing behavior
- **Sample efficiency**: Full domain adaptation with only 1,121 samples (0.19% of warmup size)

### When is Recalibration Necessary?

Domain alignment is a **discrete phase**, not a continuous requirement. It is triggered only during:

1. **Distributional Shift**: Moving from smooth (synthetic) to bimodal (real-world) difficulty distributions
2. **Model Updates**: New model versions (e.g., GPT-4 → GPT-5) or pricing changes shift the Pareto frontier
3. **Domain Migration**: Deploying to a new problem space (e.g., code generation → medical Q&A)

Recalibration is **not needed** for normal data growth, user diversity, or temporal variation—standard online learning handles these automatically.

### Deployment Workflow

1. **Offline Warmup** (one-time): Train on 80K synthetic prompts → generate priors
2. **Domain Alignment** (one-time, 100-200 samples): Apply covariance inflation, collect calibration data, discover target structure
3. **Production** (ongoing): Route prompts using adapted policy, continue standard LinUCB updates

**Efficiency:** We achieve **74% of the optimal efficiency gain** using only **0.19% of the source data** (1,121 vs 80,000 samples), demonstrating practical domain-aware transfer.

---

## 🎯 Choosing the Right Gamma

### General Guidelines

| Your Situation | Recommended Gamma | Rationale |
|----------------|-------------------|-----------|
| **Small calibration set (50-150)** | 0.001 - 0.002 | Need high influence per sample |
| **Medium calibration set (150-500)** | 0.002 - 0.010 | Balanced influence |
| **Large calibration set (500-1500)** | 0.005 - 0.020 | Lower influence per sample OK |
| **Very large set (1500+)** | **0.010** (recommended) | Optimal for canonical dev (1.1K samples) |
| **Known oracle policy** | Run find_gamma.py | Match target usage % |
| **Unknown oracle** | 0.010 (default) | Works for most domains |

### Our Empirical Results (1,121 Calibration Samples)

Based on our analysis with the canonical dev set (Mixtral vs GPT-4o):

- ✅ **γ = 0.010**: Optimal (Calib/Prior = 1.401, maximum adaptation)
- ⚠️ **γ = 0.002-0.005**: Good (high adaptation, but less balanced)
- ⚠️ **γ = 0.001**: Over-inflation (calibration dominates too much)
- ❌ **γ = 0.1**: Under-inflation (minimal adaptation)
- ❌ **γ = 1.0**: No adaptation (Bayesian inertia)

### Warning Signs

**Gamma too high (e.g., 0.1-1.0):**
- ❌ Calibration has minimal effect (Calib/Prior < 0.2)
- ❌ Router usage stuck at warmup bias
- ❌ High cost overrun vs optimal policy
- 💡 Solution: Decrease gamma by 10×

**Gamma too low (e.g., 0.0001-0.0005):**
- ❌ Router nearly ignores warmup knowledge (Calib/Prior > 10)
- ❌ High variance, unstable routing
- ❌ May lose valuable linguistic features from warmup
- 💡 Solution: Increase gamma by 5-10×

**Optimal gamma (Sweet Spot):**
- ✅ Calibration/Prior ratio ≈ 0.5-2.0 (balanced influence)
- ✅ Router adapts significantly (Δ > 20% from baseline)
- ✅ Stable convergence within 200 samples
- ✅ Quality maintained or improved

---

## 📁 File Formats

### Calibration Data (Input)

```jsonl
{
  "prompt": "User query text",
  "rewards": {
    "mistralai/mixtral-8x7b-instruct": 0.85,
    "openai/gpt-4-turbo": 0.95
  }
}
```

**Fields:**
- `prompt` (string): User query
- `rewards` (dict): Model ID → reward (0.0-1.0)
  - Must include both models
  - Higher = better quality

### Calibrated Router (Output)

```python
{
  'A': {model_id: np.ndarray(24, 24)},    # LinUCB A matrices
  'b': {model_id: np.ndarray(24,)},       # LinUCB b vectors
  'models': ['mixtral-8x7b', 'gpt-4-turbo'],
  'context_dim': 24,                      # 23 PCA + 1 bias
  'alpha': 1.0,                           # Exploration
  'lambda_cost': 0.0,                     # Cost penalty
  'metadata': {
    'n_prompts_warmup': 80000,
    'gamma': 0.002,
    'n_calibration_samples': 150
  }
}
```

---

## 🔬 Advanced Usage

### Custom Alpha (Exploration)

Control exploration vs exploitation:

```bash
python3 scripts/calibration/calibrate_router.py \
  --calibration-data my_data.jsonl \
  --gamma 0.002 \
  --alpha 1.5 \  # Higher = more exploration
  --output router.joblib
```

**Recommendations:**
- `alpha=0.5`: Low exploration (confident routing)
- `alpha=1.0`: Balanced (default)
- `alpha=2.0`: High exploration (discover edge cases)

### Cost-Aware Routing

Penalize expensive model usage:

```bash
python3 scripts/calibration/calibrate_router.py \
  --calibration-data my_data.jsonl \
  --gamma 0.002 \
  --lambda-cost 0.1 \  # Cost penalty for GPT-4
  --output router.joblib
```

**Effect:**
- Lambda=0.0: Quality-first (default)
- Lambda=0.1: Slight cost preference
- Lambda=0.5: Strong cost preference

### Testing Multiple Gammas

Batch test multiple gamma values:

```bash
python3 scripts/calibration/find_gamma.py \
  --calibration-data my_data.jsonl \
  --gamma-values 0.01 0.005 0.002 0.001 0.0005 \
  --output results/
```

### Online Calibration

Continue learning after deployment:

```python
from bandit_gpt.calibration import CalibratedRouter

# Load calibrated router
router = CalibratedRouter.load("router.joblib", encoder, pca_model)

# During inference
user_prompt = "..."
selected_model = router.select_model(user_prompt)
response = call_llm(selected_model, user_prompt)

# Get feedback (e.g., user rating)
reward = get_user_feedback(response)  # 0.0-1.0

# Update router (online learning)
router.update(user_prompt, reward)

# Periodically save
router.save("router.joblib")
```

---

## 🐛 Troubleshooting

### Problem: "Invalid format" error

**Solution:** Ensure your JSONL has both `prompt` and `rewards` fields:

```python
# Correct
{"prompt": "...", "rewards": {"model_a": 0.8, "model_b": 0.9}}

# Wrong
{"text": "...", "scores": [0.8, 0.9]}
```

### Problem: Router usage doesn't change

**Symptoms:**
- Gamma=1.0 and Gamma=0.002 give same results
- Router stuck at warmup bias (e.g., 80% GPT-4)

**Solutions:**
1. Check calibration data quality:
   - Are rewards diverse? (Not all 1.0 or all 0.0)
   - Do they reflect your domain?
2. Ensure enough samples (100+ minimum)
3. Verify PCA model matches warmup
4. Try lower gamma (0.001 or 0.0005)

### Problem: Poor quality after calibration

**Symptoms:**
- Average reward drops significantly
- Router makes illogical choices

**Solutions:**
1. Increase gamma (less aggressive adaptation)
2. Check for label noise in calibration data
3. Ensure calibration samples are representative
4. Use smaller alpha (less exploration)

### Problem: High variance in routing

**Symptoms:**
- Similar prompts routed to different models
- Unstable behavior

**Solutions:**
1. Increase gamma (preserve more warmup knowledge)
2. Collect more calibration samples
3. Decrease alpha (less exploration)
4. Check for duplicate/contradictory labels

---

## 📚 References

### Related Scripts

- `../../scripts/calibration/find_gamma.py` - Find optimal gamma calibration factor
- `../../scripts/calibration/calibrate_router.py` - Calibrate router for production
- `../rejudge_cot.py` - CoT judge panel for reward generation
- `../../src/bandit_gpt/calibration.py` - Library: CalibratedRouter class

### Research Background

- **LinUCB**: Li et al. (2010) - "A contextual-bandit approach to personalized news article recommendation"
- **RouteLLM**: Ong et al. (2024) - "Learning to Route LLMs with Preference Data"
- **Bayesian Inertia**: Our term for warmup priors overwhelming calibration data

### Support

For issues or questions:
1. Check troubleshooting section above
2. Review example notebooks (coming soon)
3. Open an issue on GitHub

---

## 📊 Example Results

### Baseline (No Calibration, γ=1.0)

```
GPT-4 Usage: 78.5%
Quality: 0.9623
Cost: $high
Conclusion: Over-reliance on expensive model
```

### After Calibration (γ=0.002)

```
GPT-4 Usage: 21.2%
Quality: 0.9618
Cost: $low
Conclusion: 73% cost reduction, minimal quality loss
```

### Optimal Oracle (Ground Truth)

```
GPT-4 Usage: 19.3%
Quality: 0.9622
Cost: $lowest
Conclusion: Our target (achieved via calibration!)
```

---

## ✅ Checklist

Before deploying your calibrated router:

- [ ] Collected 100-200 calibration samples from target domain
- [ ] Generated ground-truth rewards using CoT judges or labels
- [ ] Ran `find_gamma.py` to find optimal gamma
- [ ] Reviewed gamma analysis plots
- [ ] Calibrated router with recommended gamma
- [ ] Tested router on held-out examples
- [ ] Verified quality maintained (vs baseline)
- [ ] Verified cost reduced (if applicable)
- [ ] Saved calibrated router for production
- [ ] Set up monitoring for online learning (optional)

---

## 📁 File Structure

```
scripts/calibration/                    # User-facing CLI tools
├── find_gamma.py                       # Find optimal gamma
├── calibrate_router.py                 # Calibrate router
└── README.md                          # Quick start guide

src/bandit_gpt/                        # Reusable library code
└── calibration.py                     # CalibratedRouter class, helpers

experiments_v1/calibration/            # Research artifacts (this folder)
├── README.md                          # Complete workflow guide (this file)
├── FINAL_RESULTS_SUMMARY.md           # Executive summary
├── KDD_NARRATIVE.md                   # Complete paper narrative
├── *.tex                              # LaTeX paper sections
├── *.py                               # Analysis scripts
└── results/                           # Experimental outputs
```

**Version**: 2.0  
**Last Updated**: January 23, 2026  
**Authors**: banditGPT Research Team  
**Note**: CLI tools moved to `scripts/calibration/`, library code to `src/bandit_gpt/calibration.py`
