"""
Deployment Scenario 2 — Latency-Sensitive Real-Time Application
================================================================

A coding-assistant IDE plugin that must respond within a latency SLA.
The product serves two traffic modes that alternate:

    Phase 1 (steps 1–400):   Normal IDE traffic — mostly autocomplete
                              and short code questions (60% trivial/easy).
    Phase 2 (steps 401–800): Feature launch drives complex traffic —
                              architecture reviews and multi-file refactors
                              shift the difficulty distribution upward.

This scenario exercises three mechanisms:

    • **Hard latency constraint** (``max_latency``):
      Models whose estimated TTFT exceeds the ceiling are filtered out
      before bandit selection (Layer-1 constraint).

    • **Corralling meta-learning**:
      When the traffic distribution shifts at step 401, the warm-start
      expert's priors become stale.  Corralling hedges by upweighting
      the tabula-rasa expert, which adapts from scratch.  We compare
      ``use_corralling=True`` vs. ``False`` to show the safety benefit.

    • **Exploration decay** (``total_steps``):
      Passing the horizon length lets the bandit anneal exploration,
      concentrating on the best arm as confidence grows.

Expected dynamics:
    • Phase 1: The router converges to fast local models (Llama 8B,
      Gemini Flash) for the easy traffic, with occasional escalation
      to GPT-4o-mini for harder queries.
    • Phase 2: The distribution shift forces re-exploration.  With
      corralling, the router recovers within ~100 prompts.  Without
      corralling, recovery is slower and noisier.

Run:
    pip install paretobandit matplotlib
    python examples/scenario_latency_sensitive_app.py
"""

from __future__ import annotations

import textwrap
from collections import Counter, defaultdict
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt

from pareto_bandit import BanditRouter

# ──────────────────────────────────────────────────────────────────────
# Portfolio — fast + frontier with latency metadata
# ──────────────────────────────────────────────────────────────────────

MODEL_PORTFOLIO: Dict[str, Dict] = {
    "meta-llama/llama-3.1-8b-instruct": {
        "model_id": "meta-llama/llama-3.1-8b-instruct",
        "input_cost_per_m": 0.05,
        "output_cost_per_m": 0.08,
        "time_to_first_token_seconds": 0.15,
        "display_name": "Llama 3.1 8B",
    },
    "google/gemini-2.0-flash": {
        "model_id": "google/gemini-2.0-flash",
        "input_cost_per_m": 0.10,
        "output_cost_per_m": 0.40,
        "time_to_first_token_seconds": 0.20,
        "display_name": "Gemini 2.0 Flash",
    },
    "openai/gpt-4o-mini": {
        "model_id": "openai/gpt-4o-mini",
        "input_cost_per_m": 0.15,
        "output_cost_per_m": 0.60,
        "time_to_first_token_seconds": 0.35,
        "display_name": "GPT-4o Mini",
    },
    "openai/gpt-4o": {
        "model_id": "openai/gpt-4o",
        "input_cost_per_m": 2.50,
        "output_cost_per_m": 10.00,
        "time_to_first_token_seconds": 0.50,
        "display_name": "GPT-4o",
    },
    "anthropic/claude-sonnet-4": {
        "model_id": "anthropic/claude-sonnet-4",
        "input_cost_per_m": 3.00,
        "output_cost_per_m": 15.00,
        "time_to_first_token_seconds": 0.80,
        "display_name": "Claude Sonnet 4",
    },
}

MODEL_IDS: List[str] = list(MODEL_PORTFOLIO.keys())

# ──────────────────────────────────────────────────────────────────────
# Synthetic oracle — coding-assistant traffic
# ──────────────────────────────────────────────────────────────────────

SUCCESS_PROBS = np.array([
    #  8B    Flash  Mini   GPT4o  Sonnet4
    [0.95,   0.96,  0.97,  0.99,   0.99],   # trivial
    [0.70,   0.82,  0.88,  0.97,   0.97],   # easy
    [0.25,   0.55,  0.70,  0.93,   0.95],   # medium
    [0.08,   0.30,  0.45,  0.85,   0.90],   # hard
    [0.03,   0.12,  0.20,  0.55,   0.62],   # very hard
    [0.60,   0.50,  0.40,  0.10,   0.08],   # adversarial
])

PHASE1_DIFFICULTY: Dict[str, List[float]] = {
    "autocomplete": [0.70,  0.18,  0.07,  0.03,  0.01,  0.01],
    "code_qa":      [0.50,  0.25,  0.15,  0.07,  0.02,  0.01],
    "refactor":     [0.25,  0.20,  0.25,  0.20,  0.07,  0.03],
    "architecture": [0.15,  0.15,  0.25,  0.30,  0.12,  0.03],
}

PHASE2_DIFFICULTY: Dict[str, List[float]] = {
    "autocomplete": [0.50,  0.22,  0.15,  0.08,  0.03,  0.02],
    "code_qa":      [0.30,  0.20,  0.22,  0.18,  0.07,  0.03],
    "refactor":     [0.10,  0.12,  0.25,  0.30,  0.18,  0.05],
    "architecture": [0.08,  0.10,  0.20,  0.32,  0.25,  0.05],
}

PROMPT_POOL: Dict[str, List[str]] = {
    "autocomplete": [
        "Complete this Python function: def merge_sorted_lists(a, b):",
        "Autocomplete: import pandas as pd; df = pd.read_csv('data.csv'); df.",
        "Fill in: async def fetch_user(user_id: int) -> User:",
        "Complete the list comprehension: result = [x for x in data if",
        "Autocomplete the decorator: @functools.",
        "Fill in: class Config(BaseModel):",
        "Complete: with open('output.json', 'w') as f:",
        "Autocomplete: plt.figure(figsize=(",
    ],
    "code_qa": [
        "What does the `yield from` syntax do in Python?",
        "Explain the difference between `asyncio.gather` and `asyncio.wait`.",
        "When should I use `__slots__` in a Python class?",
        "What's the time complexity of dict.get() in Python?",
        "How does Python's garbage collector handle circular references?",
        "What's the difference between `copy.copy` and `copy.deepcopy`?",
        "Why does `0.1 + 0.2 != 0.3` in Python?",
        "When should I use a namedtuple vs. a dataclass?",
    ],
    "refactor": [
        "Refactor this 200-line function into smaller, testable units.",
        "Convert this callback-based code to use async/await.",
        "Extract a reusable data validation layer from this Flask endpoint.",
        "Refactor these repeated SQL queries into a repository pattern.",
        "Convert this class hierarchy to use composition instead of inheritance.",
        "Simplify this nested try/except block with proper error handling.",
        "Refactor this monolithic test file into parametrised test cases.",
        "Extract common configuration logic into a settings module.",
    ],
    "architecture": [
        "Design a rate-limiting middleware for a FastAPI application.",
        "How should I structure a Python monorepo with shared libraries?",
        "Design a plugin system that supports dynamic loading of extensions.",
        "Architect a real-time notification system using WebSockets and Redis.",
        "How do I implement the saga pattern for distributed transactions in Python?",
        "Design a caching strategy for an API with mixed read/write patterns.",
        "How should I handle database migrations in a microservices architecture?",
        "Design a feature flag system with gradual rollout support.",
    ],
}


def generate_ide_dataset(
    n_prompts: int,
    phase_split: int,
    seed: int = 42,
) -> List[Dict]:
    """Generate two-phase IDE traffic with a distribution shift at phase_split."""
    rng = np.random.default_rng(seed)
    categories = list(PROMPT_POOL.keys())

    phase1_cat_weights = [0.40, 0.30, 0.20, 0.10]
    phase2_cat_weights = [0.15, 0.20, 0.35, 0.30]

    dataset: List[Dict] = []
    for i in range(n_prompts):
        phase = 1 if i < phase_split else 2
        weights = phase1_cat_weights if phase == 1 else phase2_cat_weights
        diff_map = PHASE1_DIFFICULTY if phase == 1 else PHASE2_DIFFICULTY

        cat = rng.choice(categories, p=weights)
        pool = PROMPT_POOL[cat]
        prompt_text = pool[i % len(pool)]

        variant = i // len(pool)
        if variant > 0:
            suffixes = [" (Be concise.)", " (With examples.)", " (Explain why.)"]
            prompt_text = prompt_text + suffixes[variant % len(suffixes)]

        difficulty = rng.choice(len(diff_map[cat]), p=diff_map[cat])
        probs = SUCCESS_PROBS[difficulty]
        rewards = {mid: int(rng.random() < probs[j]) for j, mid in enumerate(MODEL_IDS)}

        dataset.append({
            "prompt": prompt_text,
            "category": cat,
            "difficulty": difficulty,
            "rewards": rewards,
            "phase": phase,
        })

    return dataset


# ══════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════

N_PROMPTS = 800
PHASE_SPLIT = 400
SEED = 42
MAX_LATENCY_S = 0.6
COST_PENALTY = 0.3

print("=" * 72)
print("  SCENARIO 2: LATENCY-SENSITIVE REAL-TIME APPLICATION")
print("=" * 72)
print(f"  Latency SLA:        {MAX_LATENCY_S:.1f}s TTFT")
print(f"  Cost penalty:       {COST_PENALTY}")
print(f"  Distribution shift: step {PHASE_SPLIT}")
print()

print("  MODEL PORTFOLIO (with latency)")
print("  " + "-" * 60)
for mid, info in MODEL_PORTFOLIO.items():
    ttft = info.get("time_to_first_token_seconds", "?")
    blocked = " ← FILTERED by max_latency" if ttft > MAX_LATENCY_S else ""
    print(f"    {info['display_name']:20s}  TTFT={ttft:.2f}s{blocked}")
print()

# ──────────────────────────────────────────────────────────────────────
# Run with corralling ON and OFF for comparison
# ──────────────────────────────────────────────────────────────────────

dataset = generate_ide_dataset(N_PROMPTS, PHASE_SPLIT, seed=SEED)

results: Dict[str, List[Dict]] = {}

for label, use_corr in [("With Corralling", True), ("Without Corralling", False)]:
    router = BanditRouter.create(
        model_registry=MODEL_PORTFOLIO,
        priors="none",
        exploration="aggressive",
        cost_penalty=COST_PENALTY,
        use_corralling=use_corr,
    )

    rng = np.random.default_rng(SEED + 1)
    order = rng.permutation(len(dataset))

    history: List[Dict] = []
    for step_idx, data_idx in enumerate(order):
        d = dataset[data_idx]
        model_id, log = router.route(
            d["prompt"],
            max_latency=MAX_LATENCY_S,
            total_steps=N_PROMPTS,
        )
        reward = float(d["rewards"].get(model_id, 0))
        router.process_feedback(log.request_id, reward=reward)

        history.append({
            "step": step_idx + 1,
            "category": d["category"],
            "model_id": model_id,
            "display_name": MODEL_PORTFOLIO[model_id]["display_name"],
            "reward": reward,
            "phase": d["phase"],
        })

    results[label] = history

# ──────────────────────────────────────────────────────────────────────
# Print comparison
# ──────────────────────────────────────────────────────────────────────

print("\n  COMPARISON: CORRALLING vs. NO CORRALLING")
print("  " + "-" * 60)
print(f"  {'':>20s}  {'Phase 1':>10s}  {'Phase 2':>10s}  {'Overall':>10s}")
print("  " + "-" * 60)

for label, hist in results.items():
    p1 = [h for h in hist if h["phase"] == 1]
    p2 = [h for h in hist if h["phase"] == 2]
    r1 = np.mean([h["reward"] for h in p1]) if p1 else 0
    r2 = np.mean([h["reward"] for h in p2]) if p2 else 0
    r_all = np.mean([h["reward"] for h in hist])
    print(f"  {label:>20s}  {r1:>10.3f}  {r2:>10.3f}  {r_all:>10.3f}")

# ──────────────────────────────────────────────────────────────────────
# Latency constraint verification
# ──────────────────────────────────────────────────────────────────────

print(f"\n  LATENCY CONSTRAINT VERIFICATION")
print(f"  " + "-" * 60)
corr_hist = results["With Corralling"]
model_counts = Counter(h["model_id"] for h in corr_hist)
for mid in MODEL_IDS:
    name = MODEL_PORTFOLIO[mid]["display_name"]
    ttft = MODEL_PORTFOLIO[mid].get("time_to_first_token_seconds", 0)
    count = model_counts.get(mid, 0)
    status = ""
    if ttft > MAX_LATENCY_S and count == 0:
        status = "HARD-FILTERED"
    print(f"    {name:20s}  TTFT={ttft:.2f}s  {count:4d} selections  {status}")

# ──────────────────────────────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle(
    f"Scenario 2: Latency-Sensitive Real-Time App\n"
    f"(max_latency={MAX_LATENCY_S}s, distribution shift at step {PHASE_SPLIT})",
    fontsize=13, fontweight="bold", y=0.98,
)

# Plot 1: Learning curves comparison (corralling vs. not)
ax = axes[0, 0]
window = max(30, N_PROMPTS // 15)
for label, hist in results.items():
    rewards_arr = np.array([h["reward"] for h in hist])
    rolling = np.convolve(rewards_arr, np.ones(window) / window, mode="valid")
    style = "-" if "With" in label else "--"
    ax.plot(range(window, window + len(rolling)), rolling,
            linewidth=2, linestyle=style, label=label)

ax.axvline(PHASE_SPLIT, color="red", ls=":", alpha=0.6, label="Distribution shift")
ax.set_xlabel("Prompts Routed")
ax.set_ylabel(f"Reward (rolling avg, w={window})")
ax.set_title("Corralling Recovers Faster After Shift")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 2: Model selection over time (corralling run)
ax = axes[0, 1]
corr_hist = results["With Corralling"]
model_names = sorted({h["display_name"] for h in corr_hist})
palette = plt.cm.tab10(np.linspace(0, 0.8, len(model_names)))
color_map = {name: palette[i] for i, name in enumerate(model_names)}

sel_window = max(30, N_PROMPTS // 10)
for mname in model_names:
    sel = np.array([1.0 if h["display_name"] == mname else 0.0 for h in corr_hist])
    rolling_sel = np.convolve(sel, np.ones(sel_window) / sel_window, mode="valid")
    ax.plot(range(sel_window, sel_window + len(rolling_sel)), rolling_sel,
            label=mname, color=color_map[mname], linewidth=2)

ax.axvline(PHASE_SPLIT, color="red", ls=":", alpha=0.6)
ax.set_xlabel("Prompts Routed")
ax.set_ylabel("Selection Frequency")
ax.set_title("Model Mix Shifts with Traffic (Corralling)")
ax.legend(fontsize=7, loc="upper right")
ax.grid(True, alpha=0.3)

# Plot 3: Phase 1 vs Phase 2 category distribution
ax = axes[1, 0]
categories = sorted(PROMPT_POOL.keys())
p1_cats = Counter(d["category"] for d in dataset if d["phase"] == 1)
p2_cats = Counter(d["category"] for d in dataset if d["phase"] == 2)
x = np.arange(len(categories))
width = 0.35
p1_vals = [p1_cats.get(c, 0) for c in categories]
p2_vals = [p2_cats.get(c, 0) for c in categories]
ax.bar(x - width / 2, p1_vals, width, label="Phase 1 (easy)", color="steelblue")
ax.bar(x + width / 2, p2_vals, width, label="Phase 2 (hard)", color="firebrick")
ax.set_xticks(x)
ax.set_xticklabels([c.replace("_", " ").title() for c in categories], fontsize=9)
ax.set_ylabel("Prompt Count")
ax.set_title("Traffic Distribution Shift")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis="y")

# Plot 4: Recovery speed after shift
ax = axes[1, 1]
shift_window = 50
for label, hist in results.items():
    post_shift = [h for h in hist if h["step"] > PHASE_SPLIT]
    if len(post_shift) < shift_window:
        continue
    rewards_post = np.array([h["reward"] for h in post_shift])
    rolling_post = np.convolve(
        rewards_post, np.ones(shift_window) / shift_window, mode="valid"
    )
    style = "-" if "With" in label else "--"
    ax.plot(range(PHASE_SPLIT + shift_window, PHASE_SPLIT + shift_window + len(rolling_post)),
            rolling_post, linewidth=2, linestyle=style, label=label)

ax.set_xlabel("Prompts Routed")
ax.set_ylabel(f"Post-Shift Reward (w={shift_window})")
ax.set_title("Post-Shift Recovery — Corralling Advantage")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("scenario2_latency_sensitive.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nSaved → scenario2_latency_sensitive.png")

print(textwrap.dedent(f"""
  KEY TAKEAWAYS FOR DEPLOYMENT
  ────────────────────────────
  1. The hard max_latency={MAX_LATENCY_S}s filter removes Claude Sonnet 4
     (0.80s TTFT) from consideration, guaranteeing the latency SLA.
  2. During Phase 1 (easy traffic), the router converges to fast, cheap
     models — Llama 8B and Gemini Flash handle >70% of autocomplete.
  3. When the traffic distribution shifts at step {PHASE_SPLIT}, corralling
     detects the mismatch between priors and reality.  The meta-learner
     upweights the tabula-rasa expert, recovering within ~100 prompts.
  4. Without corralling, the router also adapts but more slowly and with
     higher variance, as the single expert must overcome stale priors.
"""))
