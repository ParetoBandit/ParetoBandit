"""
BanditGPT Hands-On Tutorial
============================

This script walks you through BanditGPT's adaptive LLM routing using a
synthetic reward oracle — no API keys, no cost, runs in under a minute.

The synthetic oracle reproduces the statistical properties observed in the
paper's LMSYS Arena evaluation dataset (1,121 prompts × 43 models):

    • Binary rewards (0/1) — matching the judge-calibrated scoring
    • ~52% of prompts where ALL models succeed (router should pick cheapest)
    • ~17–24% where cheap models fail but expensive ones succeed
    • ~1–2% where expensive models FAIL but cheap ones succeed
      (the paper's key insight: "expensive is not always better")
    • Category-dependent difficulty so the router learns contextual
      preferences (coding/reasoning are harder than general chat)

By the end you will have seen:

    1. How to define a 5-model portfolio with real cost data
    2. How the contextual bandit learns which model is best *per prompt type*
    3. How exploration, cost penalty, and priors shape routing decisions
    4. Concrete routing decisions that demonstrate cost savings
    5. Comparison against static baselines (always-cheap, always-expensive)

Run:
    pip install banditgpt matplotlib
    python examples/hands_on_tutorial.py

Each section maps 1:1 to a Jupyter notebook cell.
"""

# ──────────────────────────────────────────────────────────────────────
# Section 0 — Imports
# ──────────────────────────────────────────────────────────────────────

import textwrap
from collections import defaultdict, Counter

import numpy as np
import matplotlib.pyplot as plt

from bandit_gpt import BanditRouter

# ──────────────────────────────────────────────────────────────────────
# Section 1 — Define the 5-Model Portfolio
# ──────────────────────────────────────────────────────────────────────
#
# Each model has a cost (from real API pricing) and an unknown quality
# that the router must learn online.  Costs span two orders of magnitude
# — the router's job is to discover when cheap models are "good enough"
# and when it's worth paying for an expensive one.
#
# These 5 models match the paper's analysis of a realistic multi-tier
# portfolio: two budget models, one mid-range, and two frontier models.

MODEL_PORTFOLIO = {
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

MODEL_IDS = list(MODEL_PORTFOLIO.keys())
MODEL_NAMES = [MODEL_PORTFOLIO[m]["display_name"] for m in MODEL_IDS]

print("=" * 70)
print("  MODEL PORTFOLIO (K=5)")
print("=" * 70)
for mid, info in MODEL_PORTFOLIO.items():
    avg = (info["input_cost_per_m"] + info["output_cost_per_m"]) / 2
    print(f"  {info['display_name']:20s}  ~${avg:>6.2f}/M tokens   ({mid})")
print()


# ──────────────────────────────────────────────────────────────────────
# Section 2 — Synthetic Prompt + Reward Oracle
# ──────────────────────────────────────────────────────────────────────
#
# We generate prompts with real text (so the router's sentence-transformer
# embeddings produce meaningful features) and pre-roll binary rewards for
# each prompt × model pair.
#
# Reward generation mirrors the paper's observed distributions:
#
#   Difficulty   Share   Cheap models   Mid model   Frontier models
#   ─────────────────────────────────────────────────────────────────
#   Trivial      ~50%    succeed        succeed     succeed
#   Easy         ~15%    mostly fail    succeed     succeed
#   Medium       ~15%    fail           ~50/50      succeed
#   Hard         ~13%    fail           fail        mostly succeed
#   Very hard    ~3%     fail           fail        ~50/50
#   Adversarial  ~2%     succeed        mixed       fail
#
# The adversarial slice encodes the paper's key finding: ~14% of prompts
# are *actively worse* when routed to GPT-4-Turbo instead of Mixtral.
# A static "expensive = better" rule systematically fails here.

# Per-model success probabilities at each difficulty level.
# Rows: difficulty levels (trivial → adversarial).
# Columns: models ordered cheap → expensive.
SUCCESS_PROBS = np.array([
    #  8B    Mixtral  70B    GPT-4o  Sonnet4
    [0.98,   0.98,   0.99,   0.99,   0.99],   # trivial
    [0.30,   0.45,   0.92,   0.97,   0.97],   # easy
    [0.15,   0.25,   0.55,   0.95,   0.96],   # medium
    [0.08,   0.12,   0.20,   0.88,   0.90],   # hard
    [0.03,   0.05,   0.10,   0.55,   0.60],   # very hard
    [0.75,   0.70,   0.50,   0.15,   0.12],   # adversarial
])

# Category-specific difficulty distributions.
# These create the contextual signal: coding and reasoning have more
# hard prompts than chat or knowledge, so the router should learn to
# use expensive models selectively for those categories.
CATEGORY_DIFFICULTY = {
    #               trivial  easy  medium  hard  vhard  adversarial
    "coding":      [0.35,   0.18,  0.22,  0.17,  0.05,  0.03],
    "reasoning":   [0.30,   0.18,  0.22,  0.20,  0.07,  0.03],
    "knowledge":   [0.55,   0.18,  0.13,  0.09,  0.03,  0.02],
    "chat":        [0.70,   0.12,  0.08,  0.06,  0.02,  0.02],
}

# Real prompt texts — the router embeds these, so distinct categories
# must be semantically distinguishable.
PROMPT_POOL = {
    "coding": [
        "Write a Python function that finds the longest palindromic substring in a given string. Include type hints.",
        "Implement a thread-safe LRU cache in Python with O(1) get and put operations.",
        "Write a SQL query using window functions to find the top 3 customers by revenue in each region.",
        "Debug this recursive Fibonacci function and add memoization for efficiency.",
        "Write a bash one-liner to find all Python files modified in the last 24 hours and count their total lines.",
        "Implement a binary search tree with insert, delete, and in-order traversal in Python.",
        "Write a REST API endpoint in FastAPI that handles pagination and filtering for a product catalog.",
        "Implement the merge sort algorithm and explain its time complexity.",
        "Write a Python decorator that retries a function up to N times with exponential backoff.",
        "Create a generator function that yields prime numbers using the Sieve of Eratosthenes.",
        "Write a concurrent web scraper using asyncio that respects rate limits.",
        "Implement a trie data structure for autocomplete suggestions.",
    ],
    "reasoning": [
        "A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have left? Show your reasoning.",
        "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
        "Three people check into a $30 hotel room. The manager returns $5 via the bellboy who keeps $2. Where's the missing dollar?",
        "You have 8 identical balls and a balance scale. One ball is heavier. What is the minimum number of weighings needed?",
        "Explain the Monty Hall problem and why switching doors gives you a 2/3 chance of winning.",
        "A bat and a ball cost $1.10 together. The bat costs $1.00 more than the ball. How much does the ball cost?",
        "If you have two ropes that each take exactly one hour to burn, how can you measure 45 minutes?",
        "There are 100 lockers in a row, all closed. 100 students walk by: student 1 opens every locker, student 2 toggles every 2nd locker, etc. Which lockers are open at the end?",
        "A lily pad doubles in size every day. If it takes 48 days to cover the whole lake, how long to cover half?",
        "You're in a room with three light switches. One controls a bulb in another room. How do you determine which switch controls the bulb with only one trip?",
        "In a race, you overtake the person in second place. What position are you now in?",
        "How many times do the hands of a clock overlap in a 12-hour period?",
    ],
    "knowledge": [
        "What are the key differences between TCP and UDP? When would you choose each?",
        "Explain Type I and Type II errors in statistics with a real-world example where each is more costly.",
        "What is the CAP theorem? Describe a real system that sacrifices consistency for availability.",
        "Summarize the main arguments for and against Universal Basic Income.",
        "Explain how mRNA vaccines work to someone with a high school biology background.",
        "What is the difference between supervised, unsupervised, and reinforcement learning?",
        "Explain the Byzantine Generals Problem and why it matters for distributed systems.",
        "What causes inflation and what tools does a central bank have to control it?",
        "Describe the differences between REST and GraphQL APIs with trade-offs for each.",
        "How does HTTPS work? Explain the TLS handshake in simple terms.",
        "What is the difference between a compiler and an interpreter?",
        "Explain the concept of database normalization and when you might denormalize.",
    ],
    "chat": [
        "What's a good recipe for a quick weeknight pasta dinner?",
        "Help me write a thank-you email to a colleague who helped with a project.",
        "What are some fun activities to do in Tokyo for a first-time visitor?",
        "Recommend 5 science fiction books for someone who loved Dune.",
        "I'm feeling stressed about an upcoming presentation. Any tips to calm my nerves?",
        "Explain the rules of cricket to an American sports fan.",
        "What's the difference between latte, cappuccino, and flat white?",
        "Help me plan a surprise birthday party for a 10-year-old who loves dinosaurs.",
        "Write a short professional bio for a software engineer's LinkedIn profile.",
        "What are some good stretches to do after sitting at a desk all day?",
        "Suggest a name for a golden retriever puppy — something playful and unique.",
        "What should I consider when buying my first mechanical keyboard?",
    ],
}


def generate_synthetic_dataset(n_prompts: int, seed: int = 42):
    """Generate prompts with pre-rolled binary rewards for all 5 models.

    Returns a list of dicts, each with: prompt, category, rewards (dict).
    The reward distribution matches the paper's LMSYS Arena statistics.
    """
    rng = np.random.default_rng(seed)
    categories = list(PROMPT_POOL.keys())
    dataset = []

    for i in range(n_prompts):
        cat = categories[i % len(categories)]
        pool = PROMPT_POOL[cat]
        prompt_text = pool[i % len(pool)]

        # To create distinct embeddings for repeated prompts, append a
        # short suffix.  The router's sentence-transformer will produce
        # a slightly different vector each time, simulating unique prompts
        # within the same semantic neighbourhood.
        variant = i // len(pool)
        if variant > 0:
            suffixes = [
                f" (Explain in detail.)",
                f" (Be concise.)",
                f" (Think step by step.)",
                f" (Provide examples.)",
                f" (Compare approaches.)",
                f" (For a beginner.)",
                f" (For an expert.)",
                f" (With code examples.)",
            ]
            prompt_text = prompt_text + suffixes[variant % len(suffixes)]

        diff_probs = CATEGORY_DIFFICULTY[cat]
        difficulty = rng.choice(len(diff_probs), p=diff_probs)

        probs = SUCCESS_PROBS[difficulty]
        rewards = {mid: int(rng.random() < probs[j]) for j, mid in enumerate(MODEL_IDS)}

        dataset.append({
            "prompt": prompt_text,
            "category": cat,
            "difficulty": difficulty,
            "rewards": rewards,
        })

    return dataset


# ═══════════════════════════════════════════════════════════════════════
# ► TRY IT YOURSELF — Adjust these parameters and re-run.
# ═══════════════════════════════════════════════════════════════════════

N_PROMPTS    = 1000        # Total prompts (more = clearer learning signal)
SEED         = 42          # Random seed for reproducibility
EXPLORATION  = "aggressive"  # Try: "static", "safe", "balanced", "aggressive"
COST_PENALTY = 0.3         # Try: 0.0 (quality-only), 0.3 (default), 0.8 (cost-aggressive)
PRIORS       = "none"      # Try: "none" (cold start), "warmup" (offline priors — K=2 only)
USE_CORRALLING = True      # Try: True (meta-learning hedge), False (single expert)

print("Generating synthetic dataset...")
dataset = generate_synthetic_dataset(N_PROMPTS, seed=SEED)

# Verify the synthetic data matches paper distributions
total_all_succeed = sum(1 for d in dataset if all(d["rewards"].values()))
total_all_fail = sum(1 for d in dataset if not any(d["rewards"].values()))
cat_counts = Counter(d["category"] for d in dataset)
cheap, expensive = MODEL_IDS[1], MODEL_IDS[3]  # Mixtral vs GPT-4o
both_ok = sum(1 for d in dataset if d["rewards"][cheap] == 1 and d["rewards"][expensive] == 1)
cheap_only = sum(1 for d in dataset if d["rewards"][cheap] == 1 and d["rewards"][expensive] == 0)
exp_only = sum(1 for d in dataset if d["rewards"][cheap] == 0 and d["rewards"][expensive] == 1)

print(f"  {N_PROMPTS} prompts across {len(cat_counts)} categories")
print(f"  All 5 models succeed:       {total_all_succeed:>4d} ({100*total_all_succeed/N_PROMPTS:.0f}%)")
print(f"  Mixtral OK, GPT-4o OK:      {both_ok:>4d} ({100*both_ok/N_PROMPTS:.0f}%)")
print(f"  Mixtral FAIL, GPT-4o OK:    {exp_only:>4d} ({100*exp_only/N_PROMPTS:.0f}%)")
print(f"  Mixtral OK, GPT-4o FAIL:    {cheap_only:>4d} ({100*cheap_only/N_PROMPTS:.0f}%)  ← expensive ≠ always better")
per_model_success = {MODEL_PORTFOLIO[m]["display_name"]: np.mean([d["rewards"][m] for d in dataset]) for m in MODEL_IDS}
print(f"\n  Per-model success rates (should increase with cost):")
for name, rate in per_model_success.items():
    print(f"    {name:20s}  {rate:.1%}")
print()


# ──────────────────────────────────────────────────────────────────────
# Section 3 — Create the Router
# ──────────────────────────────────────────────────────────────────────
#
# The router uses a sentence-transformer to embed each prompt into a
# 384-dim vector, reduces it to 32 dims via PCA, and uses LinUCB with
# Corralling to select the best model.  This takes ~1s to initialise
# (model loading) and microseconds per routing decision.

print("=" * 70)
print("  CREATING ROUTER")
print("=" * 70)
print(f"  Exploration:    {EXPLORATION}")
print(f"  Cost penalty:   {COST_PENALTY}")
print(f"  Priors:         {PRIORS}")
print(f"  Corralling:     {USE_CORRALLING}")
print()

router = BanditRouter.create(
    model_registry=MODEL_PORTFOLIO,
    priors=PRIORS,
    exploration=EXPLORATION,
    cost_penalty=COST_PENALTY,
    use_corralling=USE_CORRALLING,
)


# ──────────────────────────────────────────────────────────────────────
# Section 4 — Run the Learning Loop (Offline Replay)
# ──────────────────────────────────────────────────────────────────────
#
# For each prompt:
#   1. The router selects a model based on the prompt embedding
#   2. We look up the binary reward from the synthetic oracle
#   3. We feed the reward back — the router updates in microseconds
#
# Over time the router discovers:
#   - Which models succeed on which prompt types
#   - When it's safe to use a cheap model vs. when to pay for quality
#   - Category-specific routing preferences

rng = np.random.default_rng(SEED + 1)
order = rng.permutation(len(dataset))

history = []
cumulative_reward = 0.0
cumulative_cost = 0.0

print(f"Running {N_PROMPTS} prompts through the router...\n")
print(f"{'Step':>5}  {'Category':>11}  {'Selected Model':>20}  {'Reward':>6}  {'Oracle Best':>20}  Prompt")
print("-" * 110)

for step_idx, data_idx in enumerate(order):
    step = step_idx + 1
    d = dataset[data_idx]

    model_id, log = router.route(d["prompt"], total_steps=N_PROMPTS)

    reward = float(d["rewards"].get(model_id, 0))
    router.process_feedback(log.request_id, reward=reward)

    est_cost = log.cost_usd
    cumulative_reward += reward
    cumulative_cost += est_cost

    # Identify the cheapest model that succeeds on this prompt
    oracle_best = None
    for mid in MODEL_IDS:
        if d["rewards"][mid] == 1:
            oracle_best = mid
            break
    if oracle_best is None:
        oracle_best = MODEL_IDS[0]  # all fail — cheapest is least wasteful

    display = MODEL_PORTFOLIO[model_id]["display_name"]
    oracle_display = MODEL_PORTFOLIO[oracle_best]["display_name"]
    history.append({
        "step": step,
        "category": d["category"],
        "model_id": model_id,
        "display_name": display,
        "reward": reward,
        "cost_usd": est_cost,
        "cumulative_reward": cumulative_reward,
        "cumulative_cost": cumulative_cost,
        "avg_reward": cumulative_reward / step,
        "oracle_best": oracle_best,
        "oracle_best_name": oracle_display,
        "prompt": d["prompt"],
        "all_rewards": d["rewards"],
        "difficulty": d["difficulty"],
    })

    if step <= 20 or step % 50 == 0 or step == N_PROMPTS:
        truncated = d["prompt"][:40] + ("..." if len(d["prompt"]) > 40 else "")
        marker = "✓" if reward == 1 else "✗"
        print(f"{step:5d}  {d['category']:>11s}  {display:>20s}  {marker} {reward:.0f}    {oracle_display:>20s}  {truncated}")

print(f"\n{'=' * 70}")
print(f"  DONE — {N_PROMPTS} prompts routed")
print(f"  Average reward: {cumulative_reward / N_PROMPTS:.3f}")
print(f"  Total est. cost: ${cumulative_cost:.4f}")
print(f"{'=' * 70}")


# ──────────────────────────────────────────────────────────────────────
# Section 5 — Baseline Comparisons
# ──────────────────────────────────────────────────────────────────────
#
# To appreciate what the router achieves, compare it against naive
# static strategies:
#   - Always-cheapest: always picks Llama 3.1 8B
#   - Always-expensive: always picks Claude Sonnet 4
#   - Random: picks uniformly at random
#   - Oracle: always picks the cheapest model that succeeds

print("\n" + "=" * 70)
print("  BASELINE COMPARISONS")
print("=" * 70)

baselines = {}
for label, strategy in [
    ("Always cheapest (Llama 8B)", lambda _: MODEL_IDS[0]),
    ("Always mid (Llama 70B)", lambda _: MODEL_IDS[2]),
    ("Always expensive (Sonnet 4)", lambda _: MODEL_IDS[-1]),
]:
    rewards = [d["rewards"].get(strategy(d), 0) for d in dataset]
    costs_per_m = MODEL_PORTFOLIO[strategy(dataset[0])]
    avg_cost = (costs_per_m["input_cost_per_m"] + costs_per_m["output_cost_per_m"]) / 2
    baselines[label] = {"reward": np.mean(rewards), "cost": avg_cost}

# Oracle: cheapest model that succeeds
oracle_rewards, oracle_costs = [], []
for d in dataset:
    best = None
    for mid in MODEL_IDS:
        if d["rewards"][mid] == 1:
            best = mid
            break
    if best:
        oracle_rewards.append(1.0)
        info = MODEL_PORTFOLIO[best]
        oracle_costs.append((info["input_cost_per_m"] + info["output_cost_per_m"]) / 2)
    else:
        oracle_rewards.append(0.0)
        info = MODEL_PORTFOLIO[MODEL_IDS[0]]
        oracle_costs.append((info["input_cost_per_m"] + info["output_cost_per_m"]) / 2)
baselines["Oracle (cheapest success)"] = {"reward": np.mean(oracle_rewards), "cost": np.mean(oracle_costs)}

# Router's actual performance
router_avg_reward = cumulative_reward / N_PROMPTS
router_info = {mid: MODEL_PORTFOLIO[mid] for mid in MODEL_IDS}
router_model_counts = Counter(h["model_id"] for h in history)
router_avg_cost = sum(
    router_model_counts[mid] * (router_info[mid]["input_cost_per_m"] + router_info[mid]["output_cost_per_m"]) / 2
    for mid in MODEL_IDS
) / N_PROMPTS
baselines["BanditGPT Router"] = {"reward": router_avg_reward, "cost": router_avg_cost}

print(f"\n{'Strategy':>30s}  {'Avg Reward':>10s}  {'Avg Cost/M':>10s}  {'Quality':>8s}")
print("-" * 65)
oracle_reward = baselines["Oracle (cheapest success)"]["reward"]
for label, vals in baselines.items():
    gap = vals["reward"] / oracle_reward if oracle_reward > 0 else 0
    marker = " ←" if label == "BanditGPT Router" else ""
    print(f"{label:>30s}  {vals['reward']:>10.3f}  ${vals['cost']:>9.2f}  {gap:>7.1%}{marker}")


# ──────────────────────────────────────────────────────────────────────
# Section 6 — Visualise Learning Dynamics
# ──────────────────────────────────────────────────────────────────────

model_names = sorted({h["display_name"] for h in history})
palette = plt.cm.tab10(np.linspace(0, 0.8, len(model_names)))
color_map = {name: palette[i] for i, name in enumerate(model_names)}

fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle(
    f"BanditGPT Learning Dynamics  (K={len(MODEL_IDS)}, N={N_PROMPTS}, "
    f"exploration={EXPLORATION}, λ_cost={COST_PENALTY})",
    fontsize=14, fontweight="bold", y=0.98,
)

# ── Plot 1: Learning curve (rolling avg reward) ─────────────────────
ax = axes[0, 0]
window = max(40, N_PROMPTS // 12)
rewards_arr = np.array([h["reward"] for h in history])
rolling_reward = np.convolve(rewards_arr, np.ones(window) / window, mode="valid")
x_vals = np.arange(window, window + len(rolling_reward))
ax.plot(x_vals, rolling_reward, color="steelblue", linewidth=2.2,
        label="BanditGPT (rolling avg)")

cheap_baseline = baselines["Always cheapest (Llama 8B)"]["reward"]
exp_baseline = baselines["Always expensive (Sonnet 4)"]["reward"]
oracle_baseline = baselines["Oracle (cheapest success)"]["reward"]
ax.axhline(cheap_baseline, color="gray", linestyle="--", alpha=0.6,
           label=f"Always Llama 8B ({cheap_baseline:.2f})")
ax.axhline(exp_baseline, color="salmon", linestyle="--", alpha=0.6,
           label=f"Always Sonnet 4 ({exp_baseline:.2f})")
ax.axhline(oracle_baseline, color="gold", linestyle="--", alpha=0.6,
           label=f"Oracle ({oracle_baseline:.2f})")

ax.axvspan(0, min(N_PROMPTS // 4, len(history)),
           alpha=0.06, color="blue", label="Exploration phase")
ax.set_xlabel("Prompts Routed")
ax.set_ylabel(f"Reward (rolling avg, w={window})")
ax.set_title("Learning Curve — Router Converges Toward Oracle")
ax.set_ylim(0.45, 1.02)
ax.legend(fontsize=7, loc="lower right")
ax.grid(True, alpha=0.3)

# ── Plot 2: Model selection frequency over time ─────────────────────
ax = axes[0, 1]
sel_window = max(40, N_PROMPTS // 8)
for mname in model_names:
    selections = np.array([1.0 if h["display_name"] == mname else 0.0 for h in history])
    rolling = np.convolve(selections, np.ones(sel_window) / sel_window, mode="valid")
    ax.plot(range(sel_window, sel_window + len(rolling)), rolling,
            label=mname, color=color_map[mname], linewidth=2)

ax.axhline(1.0 / len(MODEL_IDS), color="black", linestyle=":",
           alpha=0.3, label=f"Uniform (1/{len(MODEL_IDS)})")
ax.axvspan(0, min(N_PROMPTS // 4, len(history)),
           alpha=0.06, color="blue")
ax.set_xlabel("Prompts Routed")
ax.set_ylabel("Selection Frequency (rolling)")
ax.set_title(f"Exploration → Exploitation (window={sel_window})")
ax.legend(fontsize=7, loc="upper right")
ax.set_ylim(-0.05, 0.75)
ax.grid(True, alpha=0.3)

# ── Plot 3: Per-category model preference (stacked %) ────────────────
ax = axes[1, 0]
categories = sorted(PROMPT_POOL.keys())

# Only count from the second half of the run (after exploration) for
# a clearer picture of the router's learned preferences.
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
    fracs = np.array(fracs)
    ax.bar(x_pos, fracs, 0.6, bottom=bottoms,
           label=mname, color=color_map[mname])
    bottoms += fracs

ax.set_xticks(x_pos)
ax.set_xticklabels([c.title() for c in categories], fontsize=10)
ax.set_ylabel("Selection Share (converged half)")
ax.set_title("Context-Dependent Routing — Different Tasks → Different Models")
ax.legend(fontsize=7, loc="upper right")
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3, axis="y")

# ── Plot 4: Per-model reward vs cost ────────────────────────────────
ax = axes[1, 1]
scatter_pts = []
for mname in model_names:
    mhist = [h for h in history if h["display_name"] == mname]
    if not mhist:
        continue
    mid = mhist[0]["model_id"]
    info = MODEL_PORTFOLIO[mid]
    avg_cost = (info["input_cost_per_m"] + info["output_cost_per_m"]) / 2
    avg_reward = np.mean([h["reward"] for h in mhist])
    count = len(mhist)
    size = np.clip(count / N_PROMPTS * 600, 60, 400)
    ax.scatter(avg_cost, avg_reward, s=size,
               color=color_map[mname], edgecolors="white", linewidth=1, zorder=3)
    ax.annotate(f"{mname}\n({count}×, r={avg_reward:.2f})",
                (avg_cost, avg_reward), fontsize=7,
                textcoords="offset points", xytext=(10, -5))
    scatter_pts.append((avg_cost, avg_reward))

if scatter_pts:
    pts = sorted(scatter_pts)
    ax.plot([p[0] for p in pts], [p[1] for p in pts],
            color="gray", linestyle="-", alpha=0.25, linewidth=1, zorder=1)

ax.set_xlabel("Avg Cost per M Tokens ($)")
ax.set_ylabel("Avg Reward When Selected")
ax.set_title("Cost–Quality: Expensive ≠ Always Better")
ax.set_xscale("log")
ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("bandit_learning_dynamics.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nSaved → bandit_learning_dynamics.png")


# ──────────────────────────────────────────────────────────────────────
# Section 7 — Inspect Routing Decisions (Trust Building)
# ──────────────────────────────────────────────────────────────────────
#
# Without live API calls, we build trust by showing *what the router
# decided and why it was right (or wrong)*.  For each category, we pick
# an example where the router saved money and one where it upgraded to
# a frontier model — then show the oracle reward for every model.

DIFF_LABELS = ["trivial", "easy", "medium", "hard", "very hard", "adversarial"]

print("\n" + "=" * 70)
print("  ROUTING DECISION EXAMPLES")
print("  (Oracle rewards show what each model would have scored)")
print("=" * 70)

for cat in categories:
    cat_hist = [h for h in history if h["category"] == cat]
    if not cat_hist:
        continue

    # Find a cost-saving decision: router picked a cheap model that succeeded
    savings = [h for h in cat_hist if h["reward"] == 1 and h["model_id"] in MODEL_IDS[:3]]
    # Find an upgrade decision: router picked expensive and it was needed
    upgrades = [h for h in cat_hist if h["reward"] == 1 and h["model_id"] in MODEL_IDS[3:]]

    for label, examples in [("COST SAVING", savings), ("SMART UPGRADE", upgrades)]:
        if not examples:
            continue
        h = examples[len(examples) // 2]  # pick a middle example

        print(f"\n  {'─' * 64}")
        print(f"  {cat.upper()} — {label}")
        print(f"  Prompt:     {h['prompt'][:75]}{'...' if len(h['prompt']) > 75 else ''}")
        print(f"  Difficulty: {DIFF_LABELS[h['difficulty']]}")
        print(f"  Router picked: {h['display_name']} → reward = {h['reward']:.0f}")
        print(f"  Oracle best:   {h['oracle_best_name']}")
        print(f"  All model rewards:")
        for mid in MODEL_IDS:
            name = MODEL_PORTFOLIO[mid]["display_name"]
            r = h["all_rewards"][mid]
            marker = " ← selected" if mid == h["model_id"] else ""
            print(f"    {name:20s}  {'✓' if r else '✗'}{marker}")


# ──────────────────────────────────────────────────────────────────────
# Section 8 — Summary Statistics
# ──────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("  ROUTING SUMMARY")
print("=" * 70)

model_stats = defaultdict(lambda: {"count": 0, "total_reward": 0.0})
for h in history:
    s = model_stats[h["display_name"]]
    s["count"] += 1
    s["total_reward"] += h["reward"]

print(f"\n{'Model':>20s}  {'Selected':>8s}  {'Share':>6s}  {'Avg Reward':>10s}  {'True Quality':>12s}")
print("-" * 65)
for mname in model_names:
    s = model_stats[mname]
    avg_r = s["total_reward"] / max(s["count"], 1)
    mid = [m for m in MODEL_IDS if MODEL_PORTFOLIO[m]["display_name"] == mname][0]
    true_q = per_model_success[mname]
    print(f"{mname:>20s}  {s['count']:>8d}  {s['count']/N_PROMPTS:>5.1%}  {avg_r:>10.3f}  {true_q:>11.1%}")

# Per-category
print(f"\n{'Category':>11s}  {'Dominant Model':>20s}  {'Share':>6s}  {'Avg Reward':>10s}")
print("-" * 55)
for cat in categories:
    cat_hist = [h for h in history if h["category"] == cat]
    if not cat_hist:
        continue
    mc = Counter(h["display_name"] for h in cat_hist)
    dom_name, dom_count = mc.most_common(1)[0]
    avg_r = sum(h["reward"] for h in cat_hist) / len(cat_hist)
    print(f"{cat:>11s}  {dom_name:>20s}  {dom_count/len(cat_hist):>5.1%}  {avg_r:>10.3f}")


# ──────────────────────────────────────────────────────────────────────
# Section 9 — Adding a New Model to a Running Router
# ──────────────────────────────────────────────────────────────────────
#
# In production, models change constantly — a new GPT drops, a cheaper
# Llama variant appears, or you fine-tune a specialist.  BanditGPT
# handles this without retraining: register_model() adds the newcomer
# to a *running* router and bootstraps it from existing knowledge.
#
# Under the hood, two mechanisms accelerate onboarding:
#
#   1. Latent Semantic Transfer — The router matches the new model to
#      its closest existing neighbor (via embedding similarity) and
#      copies learned preferences (θ) while resetting confidence (A).
#      This gives the newcomer an informed starting point without
#      inheriting stale certainty.
#
#   2. Hybrid Family Sharing — If the newcomer belongs to the same
#      model family (e.g. openai/gpt-4o and openai/gpt-4o-mini share
#      the "openai/gpt" family), it immediately benefits from the
#      family-level reward model (β_F) learned from all family members.
#      See Appendix C.9 in the paper for the controlled experiment.
#
# We'll simulate this: after the 5-model router has learned for 1000
# steps, we register "google/gemini-2.0-flash" — a new budget model
# with strong coding skills — and watch the router adapt.

print("\n" + "=" * 70)
print("  SECTION 9: ADDING A NEW MODEL TO A RUNNING ROUTER")
print("=" * 70)

# --- Snapshot performance BEFORE adding the new model ---
pre_add_reward = cumulative_reward / N_PROMPTS
pre_add_models = list(router.bandit.models)
print(f"\n  Router state before onboarding:")
print(f"    Active models:  {len(pre_add_models)} ({', '.join(MODEL_PORTFOLIO[m]['display_name'] for m in pre_add_models)})")
print(f"    Avg reward:     {pre_add_reward:.3f}")
print(f"    Prompts seen:   {N_PROMPTS}")

# --- Register the newcomer ---
#
# Three tiers of knowledge you can provide:
#
#   Tier A — "I know what it's good at":
#       router.register_model("google/gemini-2.0-flash",
#                             speed="fast", capabilities=["coding", "reasoning"])
#
#   Tier B — "I only know it's cheap/fast":
#       router.register_model("google/gemini-2.0-flash", speed="fast")
#
#   Tier C — "I know nothing":
#       router.register_model("google/gemini-2.0-flash")
#
# We'll use Tier A — the most informative — so the router can start
# routing to Gemini immediately rather than needing a long exploration.

NEW_MODEL_ID = "google/gemini-2.0-flash"
NEW_MODEL_DISPLAY = "Gemini 2.0 Flash"
NEW_MODEL_COST = {"input_cost_per_m": 0.10, "output_cost_per_m": 0.40}

print(f"\n  Registering: {NEW_MODEL_ID}")
print(f"    Cost: ~${(NEW_MODEL_COST['input_cost_per_m'] + NEW_MODEL_COST['output_cost_per_m'])/2:.2f}/M tokens")
print(f"    Speed tier: fast")
print(f"    Capabilities: coding, reasoning")

router.register_model(
    NEW_MODEL_ID,
    speed="fast",
    capabilities=["coding", "reasoning"],
    cost_usd=(NEW_MODEL_COST["input_cost_per_m"] + NEW_MODEL_COST["output_cost_per_m"]) / 2,
)

# Update our bookkeeping so the rest of the script can track it.
MODEL_PORTFOLIO[NEW_MODEL_ID] = {
    "model_id": NEW_MODEL_ID,
    "display_name": NEW_MODEL_DISPLAY,
    **NEW_MODEL_COST,
}

post_add_models = list(router.bandit.models)
print(f"\n  Router state after onboarding:")
print(f"    Active models:  {len(post_add_models)} (K={len(post_add_models)})")
for mid in post_add_models:
    tag = " ← NEW" if mid == NEW_MODEL_ID else ""
    name = MODEL_PORTFOLIO.get(mid, {}).get("display_name", mid)
    print(f"      {name}{tag}")

# --- Generate post-onboarding traffic ---
#
# Simulate 300 more prompts.  Gemini Flash is a strong budget model:
# it excels at coding/reasoning (where it matches GPT-4o at 1/25th the
# cost) but is weaker on nuanced chat and knowledge tasks.

NEWCOMER_SUCCESS_PROBS = np.array([
    #  trivial  easy  medium  hard  vhard  adversarial
    0.97,     0.90,   0.80,  0.70,  0.40,  0.60,
])

N_ONBOARDING = 300
onboarding_dataset = generate_synthetic_dataset(N_ONBOARDING, seed=SEED + 100)

# Add rewards for the newcomer to each datapoint
onboarding_rng = np.random.default_rng(SEED + 200)
for d in onboarding_dataset:
    prob = NEWCOMER_SUCCESS_PROBS[d["difficulty"]]
    d["rewards"][NEW_MODEL_ID] = int(onboarding_rng.random() < prob)

onboarding_history = []
onboarding_cum_reward = 0.0

print(f"\n  Routing {N_ONBOARDING} post-onboarding prompts...\n")
print(f"  {'Step':>5}  {'Category':>11}  {'Selected Model':>20}  {'R':>2}  Prompt")
print("  " + "-" * 75)

onboarding_order = np.random.default_rng(SEED + 300).permutation(len(onboarding_dataset))

for step_idx, data_idx in enumerate(onboarding_order):
    step = step_idx + 1
    d = onboarding_dataset[data_idx]

    model_id, log = router.route(d["prompt"], total_steps=N_PROMPTS + N_ONBOARDING)
    reward = float(d["rewards"].get(model_id, 0))
    router.process_feedback(log.request_id, reward=reward)

    onboarding_cum_reward += reward
    display = MODEL_PORTFOLIO.get(model_id, {}).get("display_name", model_id)

    onboarding_history.append({
        "step": N_PROMPTS + step,
        "category": d["category"],
        "model_id": model_id,
        "display_name": display,
        "reward": reward,
    })

    if step <= 10 or step % 50 == 0 or step == N_ONBOARDING:
        marker = "✓" if reward == 1 else "✗"
        truncated = d["prompt"][:40] + ("..." if len(d["prompt"]) > 40 else "")
        print(f"  {step:5d}  {d['category']:>11s}  {display:>20s}  {marker}  {truncated}")

# --- Onboarding results ---
post_reward = onboarding_cum_reward / N_ONBOARDING
newcomer_selections = sum(1 for h in onboarding_history if h["model_id"] == NEW_MODEL_ID)
newcomer_hist = [h for h in onboarding_history if h["model_id"] == NEW_MODEL_ID]
newcomer_reward = (
    np.mean([h["reward"] for h in newcomer_hist]) if newcomer_hist else 0
)

print(f"\n  {'─' * 64}")
print(f"  ONBOARDING RESULTS ({N_ONBOARDING} prompts)")
print(f"  {'─' * 64}")
print(f"  Avg reward (before onboarding):  {pre_add_reward:.3f}")
print(f"  Avg reward (after onboarding):   {post_reward:.3f}")
print(f"\n  {NEW_MODEL_DISPLAY} usage:")
print(f"    Selected: {newcomer_selections}/{N_ONBOARDING} ({100*newcomer_selections/N_ONBOARDING:.1f}%)")
if newcomer_hist:
    print(f"    Avg reward when selected: {newcomer_reward:.3f}")
print(f"\n  All models (post-onboarding):")
post_counts = Counter(h["model_id"] for h in onboarding_history)
for mid in post_add_models:
    name = MODEL_PORTFOLIO.get(mid, {}).get("display_name", mid)
    count = post_counts.get(mid, 0)
    tag = " ← NEW" if mid == NEW_MODEL_ID else ""
    print(f"    {name:20s}  {count:4d} ({100*count/N_ONBOARDING:5.1f}%){tag}")

# --- Per-category newcomer adoption ---
print(f"\n  Per-category newcomer adoption:")
print(f"  {'Category':>11s}  {NEW_MODEL_DISPLAY + ' share':>20s}  {'Avg reward':>10s}")
print(f"  {'-'*48}")
for cat in sorted(PROMPT_POOL.keys()):
    cat_h = [h for h in onboarding_history if h["category"] == cat]
    if not cat_h:
        continue
    new_count = sum(1 for h in cat_h if h["model_id"] == NEW_MODEL_ID)
    avg_r = np.mean([h["reward"] for h in cat_h])
    print(f"  {cat:>11s}  {new_count}/{len(cat_h):>3d} ({100*new_count/len(cat_h):5.1f}%)  {avg_r:>10.3f}")

print(textwrap.dedent(f"""
  Key takeaway: The router started selecting {NEW_MODEL_DISPLAY}
  immediately — no retraining needed.  Semantic transfer bootstrapped
  it from the most similar existing model, and family sharing provides
  continuous knowledge transfer as the newcomer accumulates observations.

  In a real deployment, this is a one-liner:
      router.register_model("{NEW_MODEL_ID}", speed="fast",
                            capabilities=["coding", "reasoning"])
"""))


# ──────────────────────────────────────────────────────────────────────
# Section 10 — Suggested Experiments
# ──────────────────────────────────────────────────────────────────────

print("=" * 70)
print("  EXPERIMENTS TO TRY")
print("=" * 70)
print("""
  Change the parameters in Section 2 and re-run to observe different
  routing behaviour.  Each run takes under a minute with no API cost.

  A) Cold start vs. warmup priors (for the 2-model case):
     PRIORS = "none"  vs.  PRIORS = "warmup"
     → "warmup" loads dense priors from offline battles, but those
       priors only cover 2 models (Mixtral + GPT-4-turbo).  For custom
       K>2 portfolios, use "none" and let the bandit learn from scratch.
       Compare learning speed: how many prompts to surpass always-expensive?

  B) Cost sensitivity:
     COST_PENALTY = 0.0  vs.  COST_PENALTY = 0.8
     → Watch the selection frequency shift toward cheap models.
       Does average reward drop? By how much?

  C) Exploration rate:
     EXPLORATION = "static"  vs.  EXPLORATION = "aggressive"
     → Static exploits immediately (good if priors are right).
       Aggressive tries everything first (good if priors are wrong).

  D) Corralling safety net:
     USE_CORRALLING = True  vs.  USE_CORRALLING = False
     → Corralling hedges between warm-start and cold-start experts.
       It shines when priors don't match your traffic distribution.

  E) Portfolio size:
     Add or remove models from MODEL_PORTFOLIO and update SUCCESS_PROBS.
     → The bandit handles any K ≥ 2.  Larger portfolios give more room
       to optimise but require more exploration.

  F) Difficulty distribution:
     Edit CATEGORY_DIFFICULTY to make coding even harder or chat even
     easier, then watch how the per-category routing preferences shift.

  G) Model onboarding tiers (Section 9):
     Try different knowledge tiers when registering the newcomer:
       Tier A: capabilities=["coding", "reasoning"], speed="fast"
       Tier B: speed="fast" (no capabilities)
       Tier C: just the model ID (no hints at all)
     → How many prompts until the newcomer reaches the same selection
       share?  Tier A should converge fastest.
""")
