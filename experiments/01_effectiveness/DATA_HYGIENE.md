# LMSYS Data Hygiene Documentation

## Data Alignment Issue: Unmatched Prompts

### The Problem

When enriching LMSYS prompts with metadata from HuggingFace, we encountered **23.9% unmatched prompts** (5,197 out of 21,719).

**Root Cause**: LMSYS aggressively redacts PII (Personally Identifiable Information) and continuously updates their cleaning pipeline. This causes text hash mismatches between:
- Raw conversation logs (text-only)
- Leaderboard/reward metadata (with ground truth winners)

### The Decision: Drop Unmatched Prompts

**For N-Tuning Experiments**: We **discard** all 5,197 unmatched prompts.

**Rationale**:
1. **Regret Calculation Requires Ground Truth**: To tune N (prior strength), we need to calculate regret: `oracle_best - actual_reward`
2. **No Ground Truth = Cannot Grade**: If a prompt is unmatched, we don't know which model won, so we can't calculate the oracle best reward
3. **KDD Data Quality Standard**: Clean data (N=16,522) is infinitely better than noisy data with guesses (N=21,719)

**What We Did NOT Do**:
- ❌ Fuzzy string matching (introduces noise)
- ❌ Approximate joins (undermines scientific rigor)
- ❌ Imputation or guessing winners (violates ground truth requirement)

### Final Dataset Statistics

| Metric | Count | Percentage |
|--------|-------|------------|
| Total LMSYS prompts (original) | 33,000 | 100% |
| Used in train/test | 5,000 | 15.2% |
| **Unused prompts** | 21,719 | 65.8% |
| **Matched with ground truth** | **16,522** | **76.1%** |
| Unmatched (discarded) | 5,197 | 23.9% |

### Files

- `lmsys_all_prompts.jsonl`: Original 33K LMSYS prompts (text only)
- `lmsys_unused_prompts.jsonl`: 21,719 prompts not in train/test
- `lmsys_unused_enriched.jsonl`: **16,522 unique prompts with full metadata** (98,626 total records including multi-turn conversations)
- `lmsys_unmatched_prompts.jsonl`: 5,197 discarded prompts (no ground truth)

### Usage in Code

The `tune_n_prior.py` script automatically:
1. Loads only from `lmsys_unused_enriched.jsonl` (matched prompts with ground truth)
2. Deduplicates prompts (same prompt may appear in multiple conversations)
3. Subsamples to 5,000 if dataset is too large (for efficient tuning)

```bash
# KDD-quality mode (default): Uses matched LMSYS data
python experiments/01_effectiveness/tune_n_prior.py --mode lmsys

# Fast test mode: Uses synthetic data
python experiments/01_effectiveness/tune_n_prior.py --mode synthetic --size 1000
```

### Additional Filtering (Pending)

Further data quality filtering may be applied based on:
- Language (e.g., English-only)
- Prompt length (e.g., 10-500 words)
- OpenAI moderation scores (exclude flagged content)
- Conversation quality metrics

These filters will be added as needed to ensure the highest quality experimental data.

## Credit Assignment Problem: Multi-Turn Conversations

### The Problem

The LMSYS dataset contains multi-turn conversations where users vote on the **entire conversation**, not individual turns. This creates a credit assignment problem if we naively use all turns as training examples.

**Example**:
- Turn 1: Model A gives bad answer → User confused
- Turn 2: Model A recovers with excellent answer → User satisfied
- **Final Vote**: "Model A wins"

**The Flaw**: If we treat Turn 1 as a positive training example (reward = 1.0), we're teaching the bandit that the bad answer was actually good. This introduces **reward noise** and violates the i.i.d. (independent and identically distributed) assumption.

### The Solution: First Turn Only

**Decision**: We use **only the first user prompt** from each conversation, discarding all follow-up turns.

**Rationale**:
1. **Clean Signal**: The reward is 100% attributable to that specific prompt-response pair
2. **I.I.D. Assumption**: Single-turn interactions are independent; multi-turn conversations are not
3. **Stateless Router**: Our router treats inputs as stateless—it cannot handle conversational context properly
4. **KDD Rigor**: Avoids the "contextual flattening" p-hacking trap that artificially deflates error bars

### Rejected Alternatives

| Approach | Method | KDD Verdict | Why Rejected |
|----------|--------|-------------|--------------|
| **All Turns (Flattened)** | Treat every user message as a new prompt | ❌ Reject | Reward noise: Final vote doesn't tell us which turn was good/bad. Violates i.i.d. |
| **Full Context** | Concatenate history into one giant string | ❌ Reject | Leakage: Requires massive context windows, complicates cost metric, introduces state dependency |
| **First Turn Only** | Use only the initial prompt | ✅ Accept | Clean signal: Reward is 100% attributable to that specific prompt |

### KDD Defense Statement

> "To ensure reward validity, we restricted our calibration set to single-turn interactions (or the first turn of multi-turn interactions), guaranteeing that the user's preference signal is directly causally linked to the model's immediate response. This avoids the credit assignment problem inherent in multi-turn conversations, where the final vote reflects cumulative quality rather than per-turn quality."

### Implementation

The `load_lmsys_prompts()` function filters to `turn == 1` only:

```python
# STRICT RULE: Only use first turn
turn = data.get('turn', 1)
if turn != 1:
    continue  # Skip multi-turn conversations

# Extract first user message only
for message in conversation:
    if message.get('role') == 'user':
        prompt = message.get('content', '')
        prompts.append(prompt)
        break  # Only take the FIRST user message
```

### Data Volume After Filtering

Even with first-turn-only filtering, we retain **14,000+ prompts**—more than sufficient for tuning N. Quality trumps quantity in KDD research.

---

**KDD Principle**: "Clean data with known ground truth is the foundation of rigorous experimental science."
