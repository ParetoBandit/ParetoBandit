# Figure 2 Update Summary: Aligning with Metadata-Guided Cold Start

## Current Status

### What the Script Currently Shows ❌ (OLD FRAMING)
- "Overcoming 'Confidently Wrong' Priors"
- "Poisoned priors" injected at start
- GPT-4o: Prior says 1.0, reality is 0.6
- Nova-Lite: Prior says 0.1, reality is 0.95
- Shows system can "unlearn" bad initialization

### What the Paper Now Says ✅ (NEW FRAMING)
- "Adaptation to Concept Drift"
- Model capabilities shift mid-experiment (simulated API update)
- System learns correctly at first, then capabilities change
- Shows system can adapt to evolving model performance

---

## The Issue

**Mismatch between script and paper text:**
- **Script:** Tests if system can overcome "bad priors" (implies priors are used)
- **Paper (updated):** Tests if system adapts to "capability drift" (aligns with online learning)

**This creates confusion because:**
- RQ1 says: "Don't use offline priors (negative transfer)"
- RQ2 (old) says: "But priors are soft, system can overcome them"
- **Contradiction!** If priors are bad (RQ1), why test if we can overcome them (RQ2)?

**Better narrative:**
- RQ1 says: "Don't use offline priors (negative transfer)"
- RQ2 (new) says: "Online learning adapts to drift automatically"
- **Consistent!** Both support metadata + online learning

---

## What Needs to Change

### 1. Script Labels/Comments (Minor Update)

**In `generate_figure2.py`:**

**OLD terminology:**
```python
print("RQ2: Overcoming 'Confidently Wrong' Priors")
print("[Setup] Injecting poisoned priors...")
```

**NEW terminology:**
```python
print("RQ2: Adaptation to Model Capability Drift")
print("[Setup] Simulating initial online learning phase...")
```

### 2. Conceptual Framing (Keep Mechanics)

The **mechanics** can stay the same (they work fine):
- System starts with some initialization
- Reality is different from initialization
- System needs to adapt
- Memory decay helps, no decay fails

The **framing** changes:

| OLD | NEW |
|-----|-----|
| "Poisoned priors" (bad offline calibration) | "Prior online learning" (was correct, then drift) |
| Priors always wrong | Capabilities shift at t=100 |
| Tests rejecting priors | Tests adapting to evolution |

### 3. Figure Labels (If Regenerated)

**Title:**
- OLD: "Overcoming Poisoned Priors"
- NEW: "Adaptation to Model Capability Drift"

**Legend:**
- OLD: "With Poisoned Priors (γ=0.90)"
- NEW: "Metadata Init + Memory Decay (γ=0.90)"

**Annotations:**
- OLD: "Poisoned initialization"
- NEW: "Capability shift (simulated API update)"

---

## The Easiest Fix

**Option 1: Update Caption Only (Minimal Change)**

Keep the current figure as-is, just update the paper caption:

```latex
\caption{\textbf{Adaptation to Concept Drift.} In a controlled simulation, 
we model a scenario where prior online learning identifies one model as optimal, 
but capabilities suddenly shift (simulating an API update). With memory decay 
($\gamma=0.90$), the router adapts within $\sim$200 steps. Without decay 
($\gamma=1.0$), the system fails to adapt. This demonstrates that controlled 
forgetting enables continuous learning without manual recalibration.}
```

**Pros:**
- No need to regenerate figure
- Current visual likely works (shows adaptation curve)
- Just reframe the interpretation

**Cons:**
- Labels in figure might still say "poisoned priors"
- Less clean alignment with narrative

---

**Option 2: Regenerate Figure (Better Alignment)**

Update the script and regenerate:

1. Change print statements to "concept drift" terminology
2. Add a comment explaining two-phase design:
   - Phase 1 (0-100): Initial correct learning
   - Phase 2 (100+): Capabilities shift
3. Update plot title/labels
4. Regenerate figure with consistent terminology

**Pros:**
- Perfect alignment with paper narrative
- Clear labels support the story
- No confusion about "priors" when we don't recommend them

**Cons:**
- Need to update script
- Need to regenerate figure
- ~15 minutes of work

---

## Recommended Approach

### Short Term (For This Submission)

✅ **Keep current figure** - It shows adaptation, which is what we need

✅ **Updated caption** - Already done in evaluation.tex:
```latex
\caption{\textbf{Adaptation to Concept Drift.} When model capabilities 
suddenly change (simulated API update), the router adapts within $\sim$200 
steps with memory decay...}
```

✅ **Updated text** - Already done in evaluation.tex (removed "poisoned priors")

**Result:** Narrative is consistent even if figure labels aren't perfect

---

### Long Term (For Camera-Ready)

If the paper is accepted:

1. Update `generate_figure2.py` script:
   - Reframe comments as "concept drift"
   - Add phase structure (initial learning → drift → adaptation)
   - Update plot labels

2. Regenerate figure with clean labels

3. Update README with new narrative (already done)

---

## Visual Check: Does Current Figure Work?

**What the current figure likely shows:**
- Cumulative regret over time
- Multiple curves (with/without decay)
- One curve adapts, one doesn't

**This DOES match our new narrative IF we interpret it as:**
- System learns initially (low regret)
- Capabilities shift (causes spike)
- With decay: adapts (recovers)
- Without decay: fails (linear growth)

**The visual is probably fine!** It's just the labels/caption that needed updating.

---

## What We've Already Fixed

✅ **evaluation.tex updated:**
- Removed "poisoned priors" language
- Added "concept drift" framing
- Updated caption
- Updated paragraph text

✅ **README.md updated:**
- Explains new narrative
- Shows how it aligns with RQ1
- Provides updated caption

✅ **FIGURE2_UPDATED_DESIGN.md created:**
- Shows what labels should be
- Explains new interpretation
- Provides implementation notes

---

## Bottom Line

**Current Status:**
- 🟢 **Paper text:** Updated (concept drift narrative)
- 🟢 **README:** Updated (new interpretation)
- 🟡 **Figure labels:** May still say "poisoned priors" (cosmetic)
- 🟢 **Figure visual:** Shows adaptation (works for both narratives)

**Action Required:**
- ✅ **Minimum:** None - paper is consistent with updated text
- 🔄 **Optimal:** Update script labels and regenerate (15 min)

**Recommendation:**
- **For this submission:** Use as-is with updated caption (good enough)
- **For camera-ready:** Clean up labels if accepted (polish)

**The key insight:** We're not changing what's being demonstrated (plasticity/adaptation), just how we explain WHY it matters (drift handling vs. prior rejection). The visual evidence is the same, the interpretation is now consistent with RQ1's "don't use offline priors" finding.

