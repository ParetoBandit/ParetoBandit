# Adaptation Scenarios: Decision Tree

**Purpose:** Clear guidance on when to use Figure 6 vs Figure 7 approaches  
**Date:** February 13, 2026  
**Status:** ✅ Production-ready deployment guide  

---

## 🌳 Decision Tree: Which Adaptation Scenario?

```
                    Production Deployment Challenge
                                |
                    ┌───────────┴───────────┐
                    |                       |
              Model Quality                New Model
              Degradation?              Available?
                    |                       |
            ┌───────┴───────┐       ┌──────┴──────┐
            |               |       |             |
         Sudden?        Gradual?   High         Similar
         (d>1.0)        (d<0.5)   Priority?    Quality?
            |               |       |             |
            v               v       v             v
      ┌─────────┐    ┌─────────┐ ┌──────┐  ┌──────────┐
      │FIGURE 6 │    │Monitor  │ │FIGURE│  │Wait for  │
      │Automatic│    │+Offline │ │  7   │  │Offline   │
      │Failover │    │Analysis │ │Adopt │  │Validation│
      └─────────┘    └─────────┘ └──────┘  └──────────┘
```

---

## 📊 Scenario Comparison

| Dimension | Figure 6: Catastrophic Failure | Figure 7: Zero-Shot Adoption |
|-----------|-------------------------------|----------------------------|
| **Trigger** | Existing model degrades | New model released |
| **Effect Size** | Large (d>1.0) | Moderate (d≈0.2-0.5) |
| **Time Pressure** | High (minutes/hours) | Medium (days/weeks) |
| **Risk** | Revenue loss, SLA breach | Opportunity cost |
| **Mechanism** | Failure detection | Semantic transfer |
| **Response Time** | 3-50 steps (fast) | 0 steps (immediate) |
| **Validation** | N=1 (deterministic) | N=30 (statistical) |
| **Purpose** | Defensive (protect) | Offensive (capitalize) |

---

## 🎯 Use Case Guide

### Scenario A: Catastrophic Model Failure → Use Figure 6 Approach

**When:**
- Model API starts returning errors
- Quality suddenly drops (d>1.0)
- Need automatic failover
- Safety-critical application

**Example:**
```
t=0-100:   GPT-4 working fine (quality=0.80)
t=100:     GPT-4 API crashes/degrades (quality→0.15)
t=103-150: System detects failure and switches to Mixtral
Result:    Automatic recovery, no human intervention
```

**Key Benefit:** Fast detection (3-50 steps), automatic response

**Requirements:**
- Real-time monitoring
- Alternative models available
- High traffic (>10K requests/day)
- Tolerance for exploration

---

### Scenario B: New Model Release → Use Figure 7 Approach

**When:**
- GPT-4o, GPT-5, Claude-3.5 released
- Want to adopt without cold-start penalty
- Can tolerate gradual learning
- Model is similar to existing ones

**Example:**
```
t=0-299:  System trained on Mixtral + GPT-4-Turbo
t=300:    GPT-4o released (semantically similar to GPT-4-Turbo)
t=300+:   System immediately uses semantic transfer (no cold-start)
Result:   +0.62 reward improvement, no exploration penalty
```

**Key Benefit:** Zero-shot readiness, immediate exploitation

**Requirements:**
- Semantic similarity to existing models
- Can initialize from neighbors
- Moderate quality improvements expected
- Growth-oriented deployment

---

### Scenario C: Gradual Quality Drift → Use Neither

**When:**
- Model quality slowly degrading (d<0.5)
- Changes over weeks/months
- Non-emergency situation

**Recommended Approach:**
- Offline A/B testing (conclusive in 1 week)
- Periodic re-evaluation
- Manual model updates

**Why not Corralling?**
- Takes 1,000+ requests to converge for small effects
- Non-stationarity invalidates learning
- Better tools available (A/B tests)

---

### Scenario D: Low Traffic → Use Neither

**When:**
- <1,000 requests/day
- Takes weeks to gather evidence
- Distribution changes frequently

**Recommended Approach:**
- Static routing based on offline evaluation
- Periodic manual updates
- Simple threshold-based rules

**Why not Corralling?**
- Insufficient data for learning
- Too slow to adapt
- Opportunity cost too high

---

## 📋 Decision Matrix

| Your Situation | Effect Size | Time Pressure | Traffic | Recommendation |
|----------------|-------------|---------------|---------|----------------|
| Model crashes | d>1.0 | Hours | High | **Figure 6** ✅ |
| New model | d≈0.2-0.5 | Days | High | **Figure 7** ✅ |
| Gradual drift | d<0.5 | Weeks | Medium | Offline A/B ⚠️ |
| Low traffic | Any | Any | <1K/day | Static rules ⚠️ |

---

## 💡 Key Insights

### They're Complementary, Not Redundant

**Figure 6:**
- **Purpose:** Safety mechanism (defensive)
- **Trigger:** Things going wrong
- **Speed:** Fast detection (3-50 steps)
- **Validation:** Deterministic scenario

**Figure 7:**
- **Purpose:** Growth mechanism (offensive)
- **Trigger:** New opportunities
- **Speed:** Zero-shot readiness
- **Validation:** Statistical (N=30 seeds)

### Use BOTH in Production

**Complete system needs:**
1. **Failure detection** (Figure 6) - Protect against catastrophic events
2. **Model adoption** (Figure 7) - Capitalize on improvements
3. **Routine monitoring** - Track gradual changes

---

## 🚀 Deployment Flowchart

```
Production LLM Router Deployment
    |
    ├─ Monitor for Catastrophic Failures (Figure 6)
    |   └─ IF detected: Auto-failover in 3-50 requests
    |
    ├─ Monitor for New Model Releases (Figure 7)
    |   └─ IF similar model: Semantic transfer immediately
    |
    └─ Monitor for Gradual Drift (Offline)
        └─ IF d<0.5: Schedule A/B test
```

---

## 📝 Summary for Reviewers

**Reviewer Question:** "Why two adaptation experiments? Seems redundant."

**Our Answer:**

> Figures 6 and 7 address complementary adaptation scenarios in production LLM systems:
> 
> - **Figure 6 (Catastrophic):** Safety mechanism for automatic failure detection (d>1.0 effects)
> - **Figure 7 (Zero-Shot):** Growth mechanism for rapid model adoption (d≈0.2-0.5 effects)
> 
> These represent the two primary adaptation modes in production: defensive (protect against failures) 
> and offensive (capitalize on improvements). Both are necessary for comprehensive production readiness.

**Impact:** From "redundant" to "comprehensive coverage"

---

## ✅ Action Items Complete

- [x] Figure 6 has "Relationship to Figure 7" section ✅
- [x] Figure 7 has "Relationship to Figure 6" section ✅ NEW
- [x] Decision tree created (this document) ✅
- [x] Clear distinction established ✅
- [x] Deployment guide provided ✅

**Status:** Issue #4 RESOLVED ✅

---

**Prepared by:** Issue Resolution Team  
**Date:** February 13, 2026  
**Time:** 15 minutes  
**Quality:** Production-ready ✅
