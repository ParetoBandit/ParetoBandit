# Latent Semantic Transfer: V1 Progressive Learning

## Overview

This experiment validates the **Latent Semantic Transfer** approach for V1 of the BanditGPT router, replacing hardcoded heuristics with principled semantic similarity-based knowledge transfer.

## Motivation

**Problem with Hardcoded Heuristics:**
- Archetype mappings (e.g., "coding" → specific model) are brittle
- Fixed transfer weights don't adapt to model similarity
- Requires manual updates for each new model type
- Not theoretically grounded for a research paper

**Latent Semantic Transfer Solution:**
- Use semantic similarity between model "DNA" to find neighbors
- Dynamic prior strength (`n_effective`) based on confidence in match
- Automatic, no manual rules required
- Theoretically sound: Progressive Learning principle

## Key Components

### 1. Model DNA (`_get_model_dna`)

Creates a semantic representation by combining:
- Model ID (normalized): `"deepseek-coder-v2"` → `"deepseek coder v2"`
- Capabilities: `["coding", "math"]`
- Speed profile: `"fast"`, `"balanced"`, `"slow"`

Example:
```python
dna = _get_model_dna("deepseek-coder-v2", ["coding"], "slow")
# Returns: "deepseek coder v2 coding slow"
```

### 2. Semantic Neighbor Finding (`_find_semantic_neighbor`)

Finds the most similar existing model using:
- Sentence embeddings (via `SentenceTransformer`)
- Cosine similarity between DNA strings
- Caching for efficiency

Returns: `(neighbor_id, similarity_score)`

### 3. Dynamic Prior Strength

Automatically adjusts `n_effective` based on similarity:

| Similarity | n_effective | Transfer Strength | Use Case |
|------------|-------------|-------------------|----------|
| > 0.8 | 10.0 | Strong | GPT-4 → GPT-4-Turbo |
| 0.6 - 0.8 | 5.0 | Moderate | GPT-4 → Claude-3 |
| < 0.6 | 1.0 | Weak | GPT-4 → Llama-3-8B |

**Mathematical Interpretation:**
- `n_effective` simulates N pseudo-observations worth of confidence
- High `n_effective` → faster exploitation of neighbor's knowledge
- Low `n_effective` → more exploration, less trust in transfer

## Experimental Design

### Test Scenarios

#### 1. High Similarity (n_eff=10.0)
- GPT-4 → GPT-4-Turbo
- Claude-3-Opus → Claude-3.5-Sonnet
- **Expected:** Strong transfer, low warmup regret

#### 2. Medium Similarity (n_eff=5.0)
- GPT-4 → Gemini-Pro
- Claude-3 → GPT-3.5-Turbo
- **Expected:** Balanced transfer and exploration

#### 3. Low Similarity (n_eff=1.0)
- GPT-4 → Llama-3-8B
- Claude-3 → DeepSeek-Coder
- **Expected:** Weak transfer, high exploration

### Metrics

1. **Semantic Similarity**: Cosine similarity of DNA embeddings
2. **Initial θ Norm**: `||θ_0||` measures transferred preference strength
3. **Initial Confidence**: `max(eigenvalues(A_0))` verifies exploration potential
4. **Warmup Reward**: Average reward over first 50 samples
5. **Cumulative Regret**: Total regret vs oracle baseline

### Expected Results

**Hypothesis 1: Similarity Predicts Performance**
- Correlation between similarity and warmup reward > 0.5
- **Validates:** Semantic transfer is meaningful

**Hypothesis 2: Dynamic n_effective Reduces Regret**
- Higher similarity → lower cumulative regret
- **Validates:** Adaptive prior strength works

**Hypothesis 3: Exploration is Preserved**
- All new models have `max(λ(A)) ≈ init_lambda`
- **Validates:** No "Confident Transfer Trap"

## Running the Experiment

```bash
# From repository root
cd experiments_v1/latent_semantic_transfer

# Run validation
python validate_semantic_transfer.py
```

## Output

### Console Output
- Test 1: Semantic neighbor finding for various models
- Test 2: Transfer quality analysis (θ norm, A eigenvalues)
- Test 3: Warmup efficiency simulation
- Test 4: Summary statistics and correlations

### Visualizations
- `results/semantic_transfer_analysis.png`:
  - Similarity vs Warmup Reward
  - Similarity vs Regret
  - n_effective distribution
  - θ norm vs Performance

### Data Files
- `results/semantic_transfer_results.json`:
  - Summary statistics by transfer strength
  - Detailed metrics for each model
  - Correlation coefficients

## Theoretical Contribution

This experiment provides the **theoretical meat** for a KDD paper:

### V1 Algorithm: Latent Semantic Transfer

**Input:** New model `m_new` with metadata `(capabilities, speed)`

**Step 1:** Construct semantic DNA
```
DNA(m) = normalize(model_id) ⊕ capabilities ⊕ speed
```

**Step 2:** Find semantic neighbor
```
m_neighbor = argmax_{m' ∈ registry} cos_sim(embed(DNA(m)), embed(DNA(m')))
```

**Step 3:** Dynamic prior strength
```
n_eff = { 10.0  if sim > 0.8  (strong)
        {  5.0  if sim > 0.6  (moderate)
        {  1.0  otherwise     (weak)
```

**Step 4:** Transfer preferences, reset confidence
```
θ_neighbor = A_inv[m_neighbor] @ b[m_neighbor]
A_new = λ·I                              # Fresh uncertainty
b_new = λ·θ_neighbor·n_eff               # Scaled preferences
```

**Output:** `(A_new, b_new)` for model `m_new`

### Key Properties

1. **Automatic:** No manual rules required
2. **Adaptive:** Prior strength scales with confidence
3. **Exploration-Preserving:** Fresh A ensures wide confidence intervals
4. **Theoretically Grounded:** Progressive learning via semantic manifolds

## Integration with Paper

### Section: Progressive Learning (V1)

> "Instead of hardcoded heuristics, we propose Latent Semantic Transfer:
> a principled approach to knowledge transfer based on semantic similarity
> in the model metadata space. By embedding model 'DNA' (ID, capabilities, 
> speed) and finding the nearest neighbor, we achieve automatic warmup
> with adaptive prior strength."

### Experimental Results Section

Include:
- Figure: Similarity vs Performance (validates approach)
- Table: Summary statistics by transfer strength
- Ablation: Compare to cold start baseline

### Contributions

1. **Novel Algorithm:** First work to use semantic similarity for bandit arm initialization
2. **Dynamic Priors:** Adaptive `n_effective` based on confidence
3. **Empirical Validation:** Show correlation between similarity and warmup efficiency

## Next Steps

After validating this experiment:

1. **Run on Real Data:** Use production traffic logs
2. **Compare to Baselines:**
   - Cold start (n_eff=0)
   - Fixed heuristics (old archetype maps)
   - Uniform transfer (n_eff=5.0 for all)
3. **Tune Thresholds:** Optimize similarity cutoffs (0.8, 0.6)
4. **Scale Test:** 100+ models to validate efficiency

## References

- Li et al. (2010): LinUCB for contextual bandits
- Devlin et al. (2019): BERT for semantic embeddings
- Your Paper: Progressive Learning via Latent Semantic Transfer

