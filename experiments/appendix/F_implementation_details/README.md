# Appendix F: Implementation Details

## Overview
Comprehensive implementation details, configuration parameters, computational requirements, and practical deployment guidelines.

## Contents

### F.1: Configuration Details
**File**: `F1_configuration_details.tex`  
**Source**: `03_figure/latex_appendix_config.tex`

**Content**:
- System architecture configuration
- Hyperparameter settings and rationale
- Default values and ranges
- Configuration file format
- Environment variables

**Key Parameters**:
- `prior_n_effective`: Effective prior sample size (default: 5.0)
- `eta`: Meta-algorithm learning rate (default: 1.0)
- `gamma`: Exploration floor / mixing parameter (default: 0.05-0.10)
- `alpha`: UCB exploration bonus (default: adaptive)
- `warmup_samples`: Number of samples for warmup phase

---

### F.2: Experimental Setup
**File**: `F2_experimental_setup.tex`  
**Source**: `08_figure/experiments_setup_compact.tex`

**Content**:
- Hardware specifications
- Software dependencies
- Dataset preparation procedures
- Evaluation protocols
- Reproducibility guidelines

**Computational Requirements**:
- CPU: Standard multi-core processor (no GPU required)
- RAM: 8-16 GB for typical datasets
- Storage: ~5 GB for full LMSYS dataset
- Runtime: 2-3 minutes per experiment (typical)

---

### F.3: Strategy Selection Guide
**File**: `F3_strategy_selection_guide.tex`  
**Source**: `03_figure/latex_table_strategy_guide.tex`

**Content**:
- Decision tree for strategy selection
- When to use Corralling vs. offline optimization
- Cost profile selection guidelines
- Production deployment checklist

**Strategy Selection Table**:

| Scenario | Recommended Strategy | Key Parameters |
|----------|---------------------|----------------|
| Stable models, offline evaluation | Offline A/B testing | N/A |
| Catastrophic failure risk | Corralling (η=1.0, γ=0.10) | Fast failover |
| New model release | Semantic Transfer | n_eff ∈ [2, 10] |
| Cost-sensitive deployment | Hybrid Router | Cost profiles |
| Exploration-heavy tasks | Higher γ (0.10) | More exploration |
| Exploitation-focused | Lower γ (0.05) | Less exploration |

---

### F.4: Hyperparameter Selection Guide
**Content**:
- Practical guidelines for hyperparameter selection
- Tuning recommendations
- Diagnostic procedures
- Common pitfalls and solutions

**Quick Reference**:

```python
# Conservative defaults (recommended for most users)
config = {
    'prior_n_effective': 5.0,      # Balanced prior strength
    'eta': 1.0,                    # Standard learning rate
    'gamma': 0.05,                 # Moderate exploration floor
    'alpha': 'auto',               # Adaptive UCB bonus
}

# Aggressive adaptation (for rapidly changing environments)
config_aggressive = {
    'prior_n_effective': 2.0,      # Weaker priors, faster adaptation
    'eta': 5.0,                    # High learning rate
    'gamma': 0.10,                 # Higher exploration floor
}

# Conservative (when semantic similarity uncertain)
config_conservative = {
    'prior_n_effective': 1.0,      # Minimal prior influence
    'eta': 0.5,                    # Lower learning rate
    'gamma': 0.10,                 # Higher exploration for safety
}
```

---

## Installation and Setup

### Dependencies
```bash
pip install numpy scipy scikit-learn pandas matplotlib seaborn
pip install transformers sentence-transformers
pip install banditgpt  # Main package
```

### Quick Start
```python
from bandit_gpt import Router, SemanticTransfer

# Initialize router with defaults
router = Router(
    models=['mixtral', 'gpt-4-turbo', 'gpt-4o'],
    prior_n_effective=5.0,
    eta=1.0,
    gamma=0.05
)

# Add semantic transfer for new model
transfer = SemanticTransfer(
    source_model='gpt-4-turbo',
    target_model='gpt-4o',
    n_effective=5.0
)
router.add_model_with_transfer('gpt-4o', transfer)

# Route prompts
for prompt in prompts:
    model = router.select(prompt)
    response = model.generate(prompt)
    reward = evaluate(response)
    router.update(prompt, model, reward)
```

---

## Production Deployment Checklist

### Pre-Deployment
- [ ] Validate hyperparameters on dev set
- [ ] Run ablation studies to confirm configuration
- [ ] Test semantic transfer if adding new models
- [ ] Verify cost constraints and budget limits
- [ ] Set up monitoring and logging
- [ ] Configure failover mechanisms

### Deployment
- [ ] Start with conservative exploration (γ=0.10)
- [ ] Monitor weight evolution for first 100-500 samples
- [ ] Check for unexpected routing patterns
- [ ] Validate cost-quality trade-offs
- [ ] Set up alerts for catastrophic failures

### Post-Deployment
- [ ] Analyze usage distribution across models
- [ ] Measure cumulative regret vs. baseline
- [ ] Track cost savings and quality metrics
- [ ] Tune parameters based on production data
- [ ] Document any domain-specific adjustments

---

## Troubleshooting

### Common Issues

**Issue**: Router converges slowly to optimal policy
- **Solution**: Increase learning rate (η) or decrease prior strength (n_eff)
- **Check**: Distribution shift between warmup and deployment data

**Issue**: Unstable weights, frequent oscillations
- **Solution**: Increase exploration floor (γ) or decrease learning rate (η)
- **Check**: Reward variance, consider ensemble approaches

**Issue**: New model not being selected despite good performance
- **Solution**: Check semantic transfer initialization, may need lower n_eff
- **Check**: Verify reward feedback is being received correctly

**Issue**: Excessive exploration, high regret
- **Solution**: Decrease exploration floor (γ) or increase prior strength (n_eff)
- **Check**: Quality of warmup priors, consider updating

---

## Performance Optimization

### For Large-Scale Deployment
- Use batch updates for efficiency
- Cache embeddings for repeated prompts
- Implement async reward feedback
- Consider distributed routing for high QPS

### For Resource-Constrained Environments
- Reduce embedding dimensions (384 → 128)
- Use approximate nearest neighbor search
- Cache semantic transfer computations
- Implement lazy evaluation

---

## Related Sections
- **Appendix C**: Hyperparameter sensitivity analysis validates these choices
- **Appendix D**: Ablation studies justify configuration decisions
- **Appendix E**: Extended results demonstrate real-world performance
- **Appendix G**: Practical recommendations complement implementation details

---

## Files
```
F_implementation_details/
├── README.md                          (this file)
├── F1_configuration_details.tex      (config parameters)
├── F2_experimental_setup.tex         (setup procedures)
├── F3_strategy_selection_guide.tex   (decision guide)
├── F4_hyperparameter_guide.tex       (to be created)
└── figures/
    ├── (system architecture diagrams)
    └── (configuration flowcharts)
```
