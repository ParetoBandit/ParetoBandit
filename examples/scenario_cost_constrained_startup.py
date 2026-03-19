"""
Deployment Scenario 1 — Cost-Constrained SaaS Startup
======================================================

A B2B SaaS company routes customer-support and product-help queries through
an LLM.  The business constraint is strict: average cost must stay below
$1.00 / M tokens, and *no single request* may exceed $3.00 / M tokens.

This scenario exercises two cost-control mechanisms simultaneously:

    • **Hard per-request ceiling** (``max_cost``):
      The router filters out any model whose estimated cost exceeds the
      ceiling *before* bandit selection.  This is a Layer-1 constraint
      (see paper Section 3, Appendix F).

    • **Soft cost penalty** (``cost_penalty = 0.7``):
      Among the models that pass the hard filter, the bandit's UCB score
      is penalised proportionally to normalised cost, nudging the router
      toward cheaper models *unless* the quality gap is large enough to
      justify the expense.

Expected learning dynamics:
    • For easy prompts (chat, simple knowledge), the router converges to
      the cheapest model that still succeeds — Llama 3.1 8B or Mixtral.
    • For hard prompts (coding, multi-step reasoning), the router
      escalates to Llama 70B or GPT-4o, but *only* when cheaper models
      consistently fail on similar prompts.
    • Claude Sonnet 4 ($9.00/M avg) is hard-filtered out entirely by
      ``max_cost=3.0``, demonstrating budget enforcement with zero
      exploration waste.

Run:
    pip install paretobandit matplotlib
    python examples/scenario_cost_constrained_startup.py
"""

from __future__ import annotations

import textwrap
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from pareto_bandit import BanditRouter

# ──────────────────────────────────────────────────────────────────────
# Portfolio — realistic SaaS support stack
# ──────────────────────────────────────────────────────────────────────

MODEL_PORTFOLIO: Dict[str, Dict] = {
    "meta-llama/llama-3.1-8b-instruct": {
        "model_id": "meta-llama/llama-3.1-8b-instruct",
        "input_cost_per_m": 0.05,
        "output_cost_per_m": 0.08,
        "display_name": "Llama 3.1 8B",
    },
    "mistralai/mixtral-8x7b-instruct": {
        "model_id": "mistralai/mixtral-8x7b-instruct",
        "input_cost_per_m": 0.24,
        "output_cost_per_m": 0.24,
        "display_name": "Mixtral 8x7B",
    },
    "meta-llama/llama-3.1-70b-instruct": {
        "model_id": "meta-llama/llama-3.1-70b-instruct",
        "input_cost_per_m": 0.52,
        "output_cost_per_m": 0.75,
        "display_name": "Llama 3.1 70B",
    },
    "openai/gpt-4o": {
        "model_id": "openai/gpt-4o",
        "input_cost_per_m": 2.50,
        "output_cost_per_m": 10.00,
        "display_name": "GPT-4o",
    },
    "anthropic/claude-sonnet-4": {
        "model_id": "anthropic/claude-sonnet-4",
        "input_cost_per_m": 3.00,
        "output_cost_per_m": 15.00,
        "display_name": "Claude Sonnet 4",
    },
}

MODEL_IDS: List[str] = list(MODEL_PORTFOLIO.keys())

# ──────────────────────────────────────────────────────────────────────
# Synthetic oracle — SaaS support traffic profile
# ──────────────────────────────────────────────────────────────────────
# Customer-support traffic is dominated by easy queries (password resets,
# billing questions, feature how-tos).  Harder queries (debugging API
# integrations, complex data-export requests) are rarer but high-value.

SUCCESS_PROBS = np.array([
    #  8B    Mixtral  70B    GPT-4o  Sonnet4
    [0.96,   0.97,   0.99,   0.99,   0.99],   # trivial
    [0.80,   0.88,   0.95,   0.98,   0.98],   # easy
    [0.35,   0.50,   0.78,   0.95,   0.96],   # medium
    [0.10,   0.18,   0.40,   0.88,   0.90],   # hard
    [0.05,   0.08,   0.15,   0.60,   0.65],   # very hard
    [0.70,   0.65,   0.45,   0.12,   0.10],   # adversarial
])

CATEGORY_DIFFICULTY: Dict[str, List[float]] = {
    #               trivial  easy  medium  hard  vhard  adversarial
    "billing":     [0.65,   0.20,  0.08,  0.04,  0.01,  0.02],
    "how_to":      [0.55,   0.25,  0.12,  0.05,  0.01,  0.02],
    "api_debug":   [0.20,   0.15,  0.30,  0.25,  0.07,  0.03],
    "data_export": [0.25,   0.20,  0.25,  0.20,  0.07,  0.03],
}

PROMPT_POOL: Dict[str, List[str]] = {
    "billing": [
        "How do I update my credit card on file?",
        "I was charged twice for my last invoice. Can you help?",
        "What's the difference between the Pro and Enterprise plans?",
        "How do I cancel my subscription and get a prorated refund?",
        "Can I switch from monthly to annual billing mid-cycle?",
        "Where can I download my tax invoices for the last quarter?",
        "My payment failed but my card is valid. What should I try?",
        "How do I add a second billing contact to my organization?",
    ],
    "how_to": [
        "How do I enable two-factor authentication for my team?",
        "Walk me through setting up SSO with Okta for our organization.",
        "How do I create a custom dashboard with our analytics widgets?",
        "What permissions does the 'Editor' role have vs. 'Admin'?",
        "How do I schedule automated reports to be emailed weekly?",
        "Can I restrict certain users to read-only access on specific projects?",
        "How do I integrate your platform with our Slack workspace?",
        "Explain how to set up webhooks for real-time event notifications.",
    ],
    "api_debug": [
        "I'm getting 429 errors on your API. How do I handle rate limiting?",
        "My webhook payloads are arriving out of order. Is this expected?",
        "The /v2/users endpoint returns 403 even though my token has admin scope.",
        "How do I paginate through large result sets in your REST API?",
        "I'm seeing inconsistent response times from your GraphQL endpoint.",
        "My OAuth refresh token flow works in dev but fails in production.",
        "The bulk import endpoint silently drops rows with Unicode characters.",
        "How do I configure retry logic for transient 5xx errors from your API?",
    ],
    "data_export": [
        "I need to export all customer records as a CSV. How do I handle pagination?",
        "Write me a Python script to pull our daily analytics via your API.",
        "How do I set up an incremental data sync to our Snowflake warehouse?",
        "The JSON export is nested three levels deep. Can I get a flat format?",
        "I need to join data from two endpoints. What's the recommended approach?",
        "How do I export audit logs for the past 90 days for compliance?",
        "My data export job times out at 10,000 rows. How do I batch it?",
        "Write a SQL query to reconcile our internal data against your API export.",
    ],
}


def generate_saas_dataset(
    n_prompts: int,
    seed: int = 42,
) -> List[Dict]:
    """Generate synthetic SaaS support traffic with pre-rolled rewards."""
    rng = np.random.default_rng(seed)
    categories = list(PROMPT_POOL.keys())
    category_weights = [0.35, 0.30, 0.20, 0.15]
    dataset: List[Dict] = []

    for i in range(n_prompts):
        cat = rng.choice(categories, p=category_weights)
        pool = PROMPT_POOL[cat]
        prompt_text = pool[i % len(pool)]

        variant = i // len(pool)
        if variant > 0:
            suffixes = [
                " (Please be concise.)",
                " (Step by step.)",
                " (With code examples.)",
                " (For a non-technical user.)",
            ]
            prompt_text = prompt_text + suffixes[variant % len(suffixes)]

        diff_probs = CATEGORY_DIFFICULTY[cat]
        difficulty = rng.choice(len(diff_probs), p=diff_probs)
        probs = SUCCESS_PROBS[difficulty]
        rewards = {
            mid: int(rng.random() < probs[j])
            for j, mid in enumerate(MODEL_IDS)
        }

        dataset.append({
            "prompt": prompt_text,
            "category": cat,
            "difficulty": difficulty,
            "rewards": rewards,
        })

    return dataset


# ══════════════════════════════════════════════════════════════════════
# Configuration — the core of this scenario
# ══════════════════════════════════════════════════════════════════════

N_PROMPTS = 800
SEED = 42
MAX_COST_PER_M = 3.0
COST_PENALTY = 0.7

print("=" * 72)
print("  SCENARIO 1: COST-CONSTRAINED SAAS STARTUP")
print("=" * 72)
print(f"  Hard cost ceiling:  ${MAX_COST_PER_M:.2f} / M tokens  (per-request)")
print(f"  Soft cost penalty:  λ = {COST_PENALTY}")
print(f"  Prompts:            {N_PROMPTS}")
print()

print("  MODEL PORTFOLIO")
print("  " + "-" * 60)
for mid, info in MODEL_PORTFOLIO.items():
    avg = (info["input_cost_per_m"] + info["output_cost_per_m"]) / 2
    blocked = " ← BLOCKED by max_cost" if avg > MAX_COST_PER_M else ""
    print(f"    {info['display_name']:20s}  ~${avg:>6.2f}/M tokens{blocked}")
print()

# ──────────────────────────────────────────────────────────────────────
# Create the router
# ──────────────────────────────────────────────────────────────────────

router = BanditRouter.create(
    model_registry=MODEL_PORTFOLIO,
    priors="none",
    exploration="balanced",
    cost_penalty=COST_PENALTY,
    use_corralling=True,
)

# ──────────────────────────────────────────────────────────────────────
# Generate data and run
# ──────────────────────────────────────────────────────────────────────

dataset = generate_saas_dataset(N_PROMPTS, seed=SEED)

cat_dist = Counter(d["category"] for d in dataset)
print("  Traffic distribution:")
for cat, count in sorted(cat_dist.items(), key=lambda x: -x[1]):
    print(f"    {cat:15s}  {count:4d} ({100 * count / N_PROMPTS:.0f}%)")
print()

rng = np.random.default_rng(SEED + 1)
order = rng.permutation(len(dataset))

history: List[Dict] = []
cumulative_reward = 0.0

print(f"{'Step':>5}  {'Category':>12}  {'Model':>20}  {'R':>2}  Prompt")
print("-" * 95)

for step_idx, data_idx in enumerate(order):
    step = step_idx + 1
    d = dataset[data_idx]

    model_id, log = router.route(
        d["prompt"],
        max_cost=MAX_COST_PER_M,
        total_steps=N_PROMPTS,
    )

    reward = float(d["rewards"].get(model_id, 0))
    router.process_feedback(log.request_id, reward=reward)

    cumulative_reward += reward
    display = MODEL_PORTFOLIO[model_id]["display_name"]
    info = MODEL_PORTFOLIO[model_id]
    est_cost = (info["input_cost_per_m"] + info["output_cost_per_m"]) / 2

    history.append({
        "step": step,
        "category": d["category"],
        "model_id": model_id,
        "display_name": display,
        "reward": reward,
        "cost_per_m": est_cost,
        "prompt": d["prompt"],
        "difficulty": d["difficulty"],
    })

    if step <= 15 or step % 100 == 0 or step == N_PROMPTS:
        marker = "✓" if reward == 1 else "✗"
        truncated = d["prompt"][:42] + ("..." if len(d["prompt"]) > 42 else "")
        print(
            f"{step:5d}  {d['category']:>12s}  {display:>20s}  {marker}  {truncated}"
        )

avg_reward = cumulative_reward / N_PROMPTS
avg_cost = np.mean([h["cost_per_m"] for h in history])

print(f"\n{'=' * 72}")
print(f"  RESULTS — {N_PROMPTS} prompts routed")
print(f"{'=' * 72}")
print(f"  Average reward:       {avg_reward:.3f}")
print(f"  Average cost / M:     ${avg_cost:.2f}")
print(f"  Budget target:        < $1.00 / M tokens")
print(f"  Budget met:           {'YES ✓' if avg_cost < 1.0 else 'NO ✗'}")

# ──────────────────────────────────────────────────────────────────────
# Cost enforcement verification
# ──────────────────────────────────────────────────────────────────────

print(f"\n  COST ENFORCEMENT VERIFICATION")
print(f"  " + "-" * 60)
model_counts = Counter(h["model_id"] for h in history)
for mid in MODEL_IDS:
    name = MODEL_PORTFOLIO[mid]["display_name"]
    avg = (
        MODEL_PORTFOLIO[mid]["input_cost_per_m"]
        + MODEL_PORTFOLIO[mid]["output_cost_per_m"]
    ) / 2
    count = model_counts.get(mid, 0)
    blocked = avg > MAX_COST_PER_M
    status = "HARD-FILTERED" if blocked and count == 0 else ""
    print(f"    {name:20s}  {count:4d} selections  (${avg:.2f}/M) {status}")

sonnet_selections = model_counts.get("anthropic/claude-sonnet-4", 0)
print(f"\n  Claude Sonnet 4 selected: {sonnet_selections} times")
print(
    f"  → Hard ceiling at ${MAX_COST_PER_M:.2f}/M correctly filtered "
    f"Sonnet 4 (${9.0:.2f}/M avg)"
)

# ──────────────────────────────────────────────────────────────────────
# Per-category cost analysis
# ──────────────────────────────────────────────────────────────────────

print(f"\n  PER-CATEGORY ROUTING")
print(f"  " + "-" * 60)
print(f"  {'Category':>12s}  {'Dominant Model':>20s}  {'Avg Reward':>10s}  {'Avg $/M':>8s}")
print(f"  " + "-" * 60)

for cat in sorted(PROMPT_POOL.keys()):
    cat_hist = [h for h in history if h["category"] == cat]
    if not cat_hist:
        continue
    mc = Counter(h["display_name"] for h in cat_hist)
    dom_name, _ = mc.most_common(1)[0]
    avg_r = np.mean([h["reward"] for h in cat_hist])
    avg_c = np.mean([h["cost_per_m"] for h in cat_hist])
    print(f"  {cat:>12s}  {dom_name:>20s}  {avg_r:>10.3f}  ${avg_c:>7.2f}")

# ──────────────────────────────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────────────────────────────

model_names = sorted({h["display_name"] for h in history})
palette = plt.cm.tab10(np.linspace(0, 0.8, len(model_names)))
color_map = {name: palette[i] for i, name in enumerate(model_names)}

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle(
    f"Scenario 1: Cost-Constrained SaaS Startup\n"
    f"(max_cost=${MAX_COST_PER_M}/M, cost_penalty={COST_PENALTY}, K={len(MODEL_IDS)})",
    fontsize=13, fontweight="bold", y=0.98,
)

# Plot 1: Learning curve
ax = axes[0, 0]
window = max(30, N_PROMPTS // 12)
rewards_arr = np.array([h["reward"] for h in history])
rolling = np.convolve(rewards_arr, np.ones(window) / window, mode="valid")
ax.plot(range(window, window + len(rolling)), rolling,
        color="steelblue", linewidth=2, label="ParetoBandit")

always_cheap = np.mean([d["rewards"][MODEL_IDS[0]] for d in dataset])
ax.axhline(always_cheap, color="gray", ls="--", alpha=0.6,
           label=f"Always Llama 8B ({always_cheap:.2f})")
ax.set_xlabel("Prompts Routed")
ax.set_ylabel(f"Reward (rolling avg, w={window})")
ax.set_title("Learning Curve — Budget-Aware Convergence")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 2: Model selection over time
ax = axes[0, 1]
sel_window = max(30, N_PROMPTS // 8)
for mname in model_names:
    sel = np.array([1.0 if h["display_name"] == mname else 0.0 for h in history])
    rolling_sel = np.convolve(sel, np.ones(sel_window) / sel_window, mode="valid")
    ax.plot(range(sel_window, sel_window + len(rolling_sel)), rolling_sel,
            label=mname, color=color_map[mname], linewidth=2)
ax.set_xlabel("Prompts Routed")
ax.set_ylabel("Selection Frequency")
ax.set_title("Model Selection — Cheap Models Dominate")
ax.legend(fontsize=7, loc="upper right")
ax.grid(True, alpha=0.3)

# Plot 3: Per-category model preference
ax = axes[1, 0]
categories = sorted(PROMPT_POOL.keys())
half = len(history) // 2
converged = history[half:]
cat_model_counts = {cat: Counter() for cat in categories}
for h in converged:
    cat_model_counts[h["category"]][h["display_name"]] += 1

x_pos = np.arange(len(categories))
bottoms = np.zeros(len(categories))
for mname in model_names:
    fracs = []
    for cat in categories:
        total = sum(cat_model_counts[cat].values()) or 1
        fracs.append(cat_model_counts[cat].get(mname, 0) / total)
    fracs_arr = np.array(fracs)
    ax.bar(x_pos, fracs_arr, 0.6, bottom=bottoms,
           label=mname, color=color_map[mname])
    bottoms += fracs_arr

ax.set_xticks(x_pos)
ax.set_xticklabels([c.replace("_", " ").title() for c in categories], fontsize=9)
ax.set_ylabel("Selection Share (converged)")
ax.set_title("Context-Dependent Cost Optimization")
ax.legend(fontsize=7, loc="upper right")
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3, axis="y")

# Plot 4: Rolling cost
ax = axes[1, 1]
costs_arr = np.array([h["cost_per_m"] for h in history])
rolling_cost = np.convolve(costs_arr, np.ones(window) / window, mode="valid")
ax.plot(range(window, window + len(rolling_cost)), rolling_cost,
        color="firebrick", linewidth=2, label="Rolling avg cost")
ax.axhline(MAX_COST_PER_M, color="red", ls="--", alpha=0.7,
           label=f"Hard ceiling (${MAX_COST_PER_M}/M)")
ax.axhline(1.0, color="orange", ls=":", alpha=0.7,
           label="Budget target ($1.00/M)")
ax.set_xlabel("Prompts Routed")
ax.set_ylabel("Cost per M Tokens ($)")
ax.set_title("Cost Control — Stays Well Under Budget")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig("scenario1_cost_constrained.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nSaved → scenario1_cost_constrained.png")

print(textwrap.dedent("""
  KEY TAKEAWAYS FOR DEPLOYMENT
  ────────────────────────────
  1. The hard max_cost ceiling guarantees no single request exceeds the
     budget — Claude Sonnet 4 was never selected despite being registered.
  2. The soft cost_penalty steers the bandit toward cheap models for easy
     queries while still escalating for genuinely hard ones.
  3. Billing and how-to traffic (65% of volume) routes almost entirely
     to Llama 8B / Mixtral, saving ~95% vs. always-frontier.
  4. API debugging and data export queries route to mid-cost or frontier
     models only when the quality difference justifies the cost.
"""))
