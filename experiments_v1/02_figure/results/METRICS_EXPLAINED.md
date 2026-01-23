# Metrics Explained: What Do These Numbers Mean?

This guide provides intuitive interpretations of the key metrics in the calibration convergence analysis.

## Effective Sample Size (Eff. N)

**What it measures**: The router's confidence level, expressed as "equivalent number of observations."

**How to interpret**:
- **High Eff. N (e.g., 426)**: "I'm very confident because I've seen 426 examples. Don't try to change my mind without showing me 500+ contradictory cases."
- **Low Eff. N (e.g., 4)**: "I'm uncertain because I've only seen 4 examples. Show me 5-10 new cases and I'll happily update my beliefs."
- **Moderate Eff. N (e.g., 16)**: "I have reasonable confidence. I'll stick to my strategy but remain open to strong evidence."

**Practical implications**:
- Eff. N = 426 → Impossible to adapt with small calibration sets
- Eff. N = 4 → Can adapt with just 10-20 samples
- Eff. N = 16 → Production-ready confidence level

**Why it matters**: This metric determines how many calibration samples you need. Without γ-scaling (Eff. N = 426), you'd need 1,000+ samples to change the policy. With γ-scaling (Eff. N = 4), you need only 100-200 samples.

---

## Strong Model Usage (%)

**What it measures**: Percentage of queries routed to the expensive, high-capability model.

**How to interpret**:
- **100%**: Always use GPT-4. Maximum quality, maximum cost.
- **50%**: Use GPT-4 for half the queries. Balanced approach.
- **0.3%**: Use GPT-4 for only 2-3 out of 1,000 queries. Minimum cost, quality trade-off.

**Practical implications**:

| Usage | Cost per 1M tokens | Cost for 750 queries | Use Case |
|-------|-------------------|---------------------|----------|
| 100% | $10 (GPT-4) | $7.50 | Quality-critical applications |
| 50% | $5.25 (mixed) | $3.94 | Balanced production |
| 0.3% | $0.53 (mostly Mixtral) | $0.40 | Cost-sensitive applications |

**Why it matters**: This directly translates to your API bill. The 100% → 0.3% shift represents a **95% cost reduction** ($7.50 → $0.40 per 750 queries).

---

## Quality Score

**What it measures**: Win rate when comparing model outputs (0.0 = always wrong, 1.0 = always right).

**How to interpret**:
- **0.971**: The strong model wins 97.1% of comparisons. Near-optimal quality.
- **0.823**: The router's mixed strategy wins 82.3% of comparisons. Good but not perfect.
- **Difference (0.148)**: 14.8% quality reduction—the cost of using the weak model more often.

**Practical implications**:
- **0.971**: Suitable for mission-critical applications (medical advice, legal analysis)
- **0.823**: Acceptable for most production use cases (customer support, content generation)
- **Trade-off**: Is 14.8% quality loss worth 95% cost savings? Often yes!

**Why it matters**: This quantifies the quality-cost trade-off. The router discovered that for this domain, the weak model performs "well enough" for 99.7% of queries.

---

## Calibration/Prior Ratio

**What it measures**: Relative influence of new calibration data vs. old warmup priors.

**How to interpret**:
- **Ratio < 1.0**: Priors dominate. New data has little impact.
- **Ratio = 1.0**: Equal influence. Tie between old and new beliefs.
- **Ratio = 2.78**: Calibration data gets 2.78 votes for every 1 vote from priors.
- **Ratio > 5.0**: Calibration dominates. Priors almost ignored.

**Practical implications**:

| Ratio | Outcome | Interpretation |
|-------|---------|----------------|
| 0.5 | No convergence | Priors too strong, policy stuck |
| 1.5 | Partial convergence | Modest policy shift |
| 2.78 | Full convergence | Complete policy reversal |
| 10.0 | Over-adaptation | May discard useful prior knowledge |

**Why it matters**: This explains *why* convergence happened. With a ratio of 2.78, calibration data had enough influence to override the warmup priors and reverse the routing strategy.

---

## Percentage Point (pp) Change

**What it measures**: Absolute change in percentages (not relative change).

**How to interpret**:
- **0.3 pp change**: 100% → 99.7% (minimal shift)
- **99.4 pp change**: 99.7% → 0.3% (dramatic shift)

**Why not use relative change?**: 
- Relative: "99.7% reduction" sounds huge but could mean 100% → 0.3%
- Absolute: "99.4 pp drop" clearly shows the magnitude

**Practical implications**: The 99.4 pp drop in Panel (a) is the "cliff effect"—visual proof that calibration, not γ-scaling, drives convergence.

---

## Real-World Example

Let's say you're building a customer support chatbot:

### Before Calibration (Warmup Only)
- **Eff. N**: 426 (very confident in GPT-4-Turbo strategy)
- **Strong Usage**: 100% (every query → GPT-4)
- **Quality**: 0.971 (excellent responses)
- **Cost**: $7.50 per 750 queries
- **Problem**: Too expensive for production scale

### After γ-Scaling
- **Eff. N**: 4 (now uncertain, ready to learn)
- **Strong Usage**: 99.7% (policy unchanged—no new data yet)
- **Quality**: 0.971 (same as before)
- **Cost**: $7.50 (same as before)
- **Status**: Door opened for adaptation, but not stepped through

### After Calibration (1,121 samples)
- **Eff. N**: 16 (moderate confidence in new strategy)
- **Strong Usage**: 0.3% (only 2-3 queries per 1,000 → GPT-4o)
- **Quality**: 0.823 (good enough for customer support)
- **Cost**: $0.40 per 750 queries (95% savings!)
- **Discovery**: "For our customer support queries, Mixtral-8x7B handles 99.7% of cases adequately. Save GPT-4o for the truly complex 0.3%."

### Business Impact
- **Annual queries**: 100M
- **Cost before**: $1M/year
- **Cost after**: $53K/year
- **Savings**: $947K/year
- **Quality trade-off**: 14.8% reduction (acceptable for this use case)

---

## Key Insight

The numbers tell a story:
1. **Eff. N 426 → 4**: "I'm willing to change my mind now"
2. **Calib/Prior 2.78**: "New evidence convinced me"
3. **Usage 100% → 0.3%**: "I reversed my strategy"
4. **Quality 0.971 → 0.823**: "I accepted a trade-off"
5. **Cost $7.50 → $0.40**: "I saved 95% in costs"

This is Bayesian recalibration in action: weakening rigid priors, incorporating new evidence, and discovering domain-specific optimal policies.

