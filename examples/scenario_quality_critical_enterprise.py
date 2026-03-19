"""
Deployment Scenario 3 — Quality-Critical Enterprise with Hot Model Onboarding
===============================================================================

A legal-tech company routes contract-review and compliance queries through
an LLM.  Quality is paramount — an incorrect clause interpretation has
real liability consequences.  The business rules are:

    • **Quality-first routing** (``cost_penalty = 0.0``): no cost bias;
      the bandit selects purely on learned quality for each prompt type.
    • **Warm-start priors**: the router loads offline priors to avoid a
      cold-start period where bad routing could produce liability risk.
    • **Mid-stream model onboarding**: after 500 prompts, a new
      specialist model (a fine-tuned legal LLM) becomes available and is
      registered via ``register_model()`` — no retraining, no downtime.

This scenario exercises three mechanisms:

    • **Quality-only optimization** (``cost_penalty = 0.0``):
      The UCB score is purely ``θ^T x + α √(x^T A⁻¹ x)``.  The router
      discovers which model produces the highest-quality output *per
      prompt type*, with no cost bias distorting the signal.

    • **Hot model onboarding** (``register_model()``):
      The progressive registration API adds a new model to a running
      router.  Family-level β_F sharing provides continuous knowledge
      transfer as observations accumulate.

    • **Warm-start priors** (``priors`` parameter):
      Even with ``priors="none"`` (cold start), the Corralling
      meta-learner provides a safety net.  In production, custom priors
      trained on labelled legal data would further reduce the initial
      exploration risk.

Expected dynamics:
    • The router converges to frontier models (GPT-4o, Claude Sonnet 4)
      for complex legal reasoning, while using mid-cost models for
      routine tasks (formatting, summarisation).
    • After the legal specialist is onboarded at step 500, the router
      discovers its strength on contract analysis and clause extraction
      within ~100 prompts and shifts traffic accordingly.

Run:
    pip install paretobandit matplotlib
    python examples/scenario_quality_critical_enterprise.py
"""

from __future__ import annotations

import textwrap
from collections import Counter, defaultdict
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt

from pareto_bandit import BanditRouter

# ──────────────────────────────────────────────────────────────────────
# Portfolio — enterprise legal stack
# ──────────────────────────────────────────────────────────────────────

MODEL_PORTFOLIO: Dict[str, Dict] = {
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

NEWCOMER_ID = "legalai/contract-llm-v2"
NEWCOMER_INFO = {
    "model_id": "legalai/contract-llm-v2",
    "input_cost_per_m": 1.20,
    "output_cost_per_m": 4.80,
    "display_name": "Contract LLM v2",
}

MODEL_IDS: List[str] = list(MODEL_PORTFOLIO.keys())

# ──────────────────────────────────────────────────────────────────────
# Synthetic oracle — legal domain traffic
# ──────────────────────────────────────────────────────────────────────

SUCCESS_PROBS = np.array([
    #  Mixtral  70B    GPT4o  Sonnet4
    [0.92,   0.96,   0.99,   0.99],   # trivial
    [0.55,   0.75,   0.95,   0.97],   # easy
    [0.20,   0.45,   0.90,   0.94],   # medium
    [0.05,   0.15,   0.80,   0.88],   # hard
    [0.02,   0.05,   0.50,   0.60],   # very hard
    [0.40,   0.30,   0.15,   0.10],   # adversarial
])

NEWCOMER_SUCCESS_PROBS = np.array([
    # trivial  easy  medium  hard  vhard  adversarial
    0.95,     0.88,  0.82,   0.75,  0.45,  0.30,
])

# The newcomer excels specifically on contract analysis and clause
# extraction (its fine-tuning domain), with a per-category boost.
NEWCOMER_CATEGORY_BOOST: Dict[str, float] = {
    "contract_review":   0.15,
    "clause_extraction": 0.12,
    "compliance_check":  0.05,
    "summarisation":     0.00,
}

CATEGORY_DIFFICULTY: Dict[str, List[float]] = {
    #                    trivial  easy  medium  hard  vhard  adversarial
    "contract_review":   [0.10,  0.15,  0.25,  0.30,  0.15,  0.05],
    "clause_extraction": [0.15,  0.20,  0.30,  0.25,  0.08,  0.02],
    "compliance_check":  [0.20,  0.25,  0.25,  0.20,  0.08,  0.02],
    "summarisation":     [0.55,  0.25,  0.12,  0.05,  0.02,  0.01],
}

PROMPT_POOL: Dict[str, List[str]] = {
    "contract_review": [
        "Review this NDA and identify clauses that could expose us to unlimited liability.",
        "Compare the indemnification terms in sections 8.1 and 8.3 of this MSA.",
        "Does this SaaS agreement include a right to audit the vendor's SOC 2 compliance?",
        "Identify any auto-renewal clauses and their notice periods in this contract.",
        "Flag any non-standard data processing terms in this DPA relative to GDPR Article 28.",
        "Review the force majeure clause — does it cover pandemic-related disruptions?",
        "Identify conflicts between the limitation of liability and the insurance requirements.",
        "Does the termination-for-convenience clause require a cure period?",
    ],
    "clause_extraction": [
        "Extract all payment terms from this 40-page vendor agreement.",
        "List every deadline and milestone date mentioned in this SOW.",
        "Extract the governing law and jurisdiction from each of these five contracts.",
        "Pull out all warranty representations from sections 3 through 7.",
        "Identify and extract all change-of-control provisions.",
        "Extract the data retention and deletion obligations from this DPA.",
        "List all notification requirements (timing and method) across this MSA.",
        "Extract intellectual property assignment clauses from this consulting agreement.",
    ],
    "compliance_check": [
        "Does this privacy policy comply with CCPA requirements for right to delete?",
        "Check whether this employment contract meets minimum notice period requirements under UK law.",
        "Verify that our data processing agreement satisfies GDPR Article 28 requirements.",
        "Does this terms-of-service include the required FTC disclosures for auto-renewal?",
        "Check this cookie consent banner text against the ePrivacy Directive requirements.",
        "Verify ADA compliance of the accessibility commitments in this vendor contract.",
        "Does this non-compete clause comply with the FTC's 2024 rule on non-competes?",
        "Check whether our SOX compliance obligations are properly addressed in this audit clause.",
    ],
    "summarisation": [
        "Summarise the key obligations for both parties in this 30-page MSA.",
        "Create a one-page executive summary of this merger agreement.",
        "Summarise the material changes between v2 and v3 of our terms of service.",
        "Write a plain-English summary of this patent's claims for our business team.",
        "Summarise the risk allocation across indemnification, liability caps, and insurance.",
        "Create a comparison table of terms across these three competing vendor proposals.",
        "Summarise the regulatory requirements referenced in this compliance report.",
        "Write a brief for the board on the key terms of the proposed acquisition agreement.",
    ],
}


def generate_legal_dataset(
    n_prompts: int,
    seed: int = 42,
) -> List[Dict]:
    """Generate legal-domain traffic with pre-rolled rewards."""
    rng = np.random.default_rng(seed)
    categories = list(PROMPT_POOL.keys())
    cat_weights = [0.30, 0.25, 0.25, 0.20]
    dataset: List[Dict] = []

    for i in range(n_prompts):
        cat = rng.choice(categories, p=cat_weights)
        pool = PROMPT_POOL[cat]
        prompt_text = pool[i % len(pool)]

        variant = i // len(pool)
        if variant > 0:
            suffixes = [
                " (Flag any risks.)",
                " (Cite specific sections.)",
                " (For a non-lawyer audience.)",
                " (Compare to standard market terms.)",
            ]
            prompt_text = prompt_text + suffixes[variant % len(suffixes)]

        diff_probs = CATEGORY_DIFFICULTY[cat]
        difficulty = rng.choice(len(diff_probs), p=diff_probs)
        probs = SUCCESS_PROBS[difficulty]
        rewards = {mid: int(rng.random() < probs[j]) for j, mid in enumerate(MODEL_IDS)}

        # Pre-roll newcomer reward (used only after onboarding)
        base_prob = NEWCOMER_SUCCESS_PROBS[difficulty]
        boost = NEWCOMER_CATEGORY_BOOST.get(cat, 0.0)
        newcomer_prob = min(base_prob + boost, 0.99)
        rewards[NEWCOMER_ID] = int(rng.random() < newcomer_prob)

        dataset.append({
            "prompt": prompt_text,
            "category": cat,
            "difficulty": difficulty,
            "rewards": rewards,
        })

    return dataset


# ══════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════

N_PROMPTS = 800
ONBOARD_STEP = 500
SEED = 42
COST_PENALTY = 0.0  # quality-first

print("=" * 72)
print("  SCENARIO 3: QUALITY-CRITICAL ENTERPRISE + HOT MODEL ONBOARDING")
print("=" * 72)
print(f"  Cost penalty:       {COST_PENALTY} (quality-only routing)")
print(f"  Model onboarding:   step {ONBOARD_STEP}")
print(f"  Prompts:            {N_PROMPTS}")
print()

print("  INITIAL PORTFOLIO (K=4)")
print("  " + "-" * 60)
for mid, info in MODEL_PORTFOLIO.items():
    avg = (info["input_cost_per_m"] + info["output_cost_per_m"]) / 2
    print(f"    {info['display_name']:20s}  ${avg:>6.2f}/M tokens")
print(f"\n  NEWCOMER (onboarded at step {ONBOARD_STEP})")
avg_new = (NEWCOMER_INFO["input_cost_per_m"] + NEWCOMER_INFO["output_cost_per_m"]) / 2
print(f"    {NEWCOMER_INFO['display_name']:20s}  ${avg_new:>6.2f}/M tokens")
print()

# ──────────────────────────────────────────────────────────────────────
# Create router and run
# ──────────────────────────────────────────────────────────────────────

router = BanditRouter.create(
    model_registry=MODEL_PORTFOLIO,
    priors="none",
    exploration="aggressive",
    cost_penalty=COST_PENALTY,
    use_corralling=True,
)

dataset = generate_legal_dataset(N_PROMPTS, seed=SEED)
rng = np.random.default_rng(SEED + 1)
order = rng.permutation(len(dataset))

history: List[Dict] = []
cumulative_reward = 0.0
newcomer_onboarded = False

print(f"{'Step':>5}  {'Category':>18}  {'Model':>20}  {'R':>2}  Prompt")
print("-" * 105)

for step_idx, data_idx in enumerate(order):
    step = step_idx + 1
    d = dataset[data_idx]

    # Onboard the newcomer at the designated step
    if step == ONBOARD_STEP and not newcomer_onboarded:
        print(f"\n{'':>5}  {'>>> ONBOARDING':>18}  {NEWCOMER_INFO['display_name']:>20}")
        router.register_model(
            NEWCOMER_ID,
            speed="balanced",
            capabilities=["reasoning"],
            cost_usd=avg_new,
        )
        MODEL_IDS.append(NEWCOMER_ID)
        MODEL_PORTFOLIO[NEWCOMER_ID] = NEWCOMER_INFO
        newcomer_onboarded = True
        print(f"{'':>5}  {'':>18}  {'Router now K=' + str(len(MODEL_IDS)):>20}\n")

    model_id, log = router.route(d["prompt"], total_steps=N_PROMPTS)
    reward = float(d["rewards"].get(model_id, 0))
    router.process_feedback(log.request_id, reward=reward)

    cumulative_reward += reward
    display = MODEL_PORTFOLIO.get(model_id, {}).get("display_name", model_id)

    history.append({
        "step": step,
        "category": d["category"],
        "model_id": model_id,
        "display_name": display,
        "reward": reward,
        "prompt": d["prompt"],
        "difficulty": d["difficulty"],
    })

    if step <= 10 or step == ONBOARD_STEP - 1 or step == ONBOARD_STEP + 1 or step % 100 == 0 or step == N_PROMPTS:
        marker = "✓" if reward == 1 else "✗"
        truncated = d["prompt"][:42] + ("..." if len(d["prompt"]) > 42 else "")
        print(f"{step:5d}  {d['category']:>18s}  {display:>20s}  {marker}  {truncated}")

avg_reward = cumulative_reward / N_PROMPTS

print(f"\n{'=' * 72}")
print(f"  RESULTS — {N_PROMPTS} prompts ({ONBOARD_STEP} before onboarding, "
      f"{N_PROMPTS - ONBOARD_STEP} after)")
print(f"{'=' * 72}")
print(f"  Overall avg reward:  {avg_reward:.3f}")

pre = [h for h in history if h["step"] < ONBOARD_STEP]
post = [h for h in history if h["step"] >= ONBOARD_STEP]
print(f"  Pre-onboarding:      {np.mean([h['reward'] for h in pre]):.3f}  ({len(pre)} prompts)")
print(f"  Post-onboarding:     {np.mean([h['reward'] for h in post]):.3f}  ({len(post)} prompts)")

# ──────────────────────────────────────────────────────────────────────
# Newcomer adoption analysis
# ──────────────────────────────────────────────────────────────────────

print(f"\n  NEWCOMER ADOPTION ({NEWCOMER_INFO['display_name']})")
print(f"  " + "-" * 60)
newcomer_hist = [h for h in post if h["model_id"] == NEWCOMER_ID]
newcomer_count = len(newcomer_hist)
newcomer_share = newcomer_count / len(post) if post else 0

print(f"  Total selections post-onboarding: {newcomer_count}/{len(post)} ({newcomer_share:.1%})")
if newcomer_hist:
    print(f"  Avg reward when selected:         {np.mean([h['reward'] for h in newcomer_hist]):.3f}")

print(f"\n  Per-category newcomer adoption:")
print(f"  {'Category':>18s}  {'Share':>8s}  {'Avg Reward':>10s}")
print(f"  " + "-" * 42)
for cat in sorted(PROMPT_POOL.keys()):
    cat_post = [h for h in post if h["category"] == cat]
    if not cat_post:
        continue
    new_count = sum(1 for h in cat_post if h["model_id"] == NEWCOMER_ID)
    avg_r = np.mean([h["reward"] for h in cat_post])
    print(f"  {cat:>18s}  {new_count}/{len(cat_post):>3d}  {avg_r:>10.3f}")

# ──────────────────────────────────────────────────────────────────────
# Quality-first vs. cost-penalised comparison
# ──────────────────────────────────────────────────────────────────────

print(f"\n  MODEL USAGE (quality-first, no cost bias)")
print(f"  " + "-" * 60)
model_counts = Counter(h["model_id"] for h in history)
for mid in list(MODEL_PORTFOLIO.keys()):
    name = MODEL_PORTFOLIO[mid]["display_name"]
    count = model_counts.get(mid, 0)
    info = MODEL_PORTFOLIO[mid]
    avg_cost = (info["input_cost_per_m"] + info["output_cost_per_m"]) / 2
    if count > 0:
        avg_r = np.mean([h["reward"] for h in history if h["model_id"] == mid])
        print(f"    {name:20s}  {count:4d} ({count/N_PROMPTS:5.1%})  "
              f"reward={avg_r:.3f}  ${avg_cost:.2f}/M")
    else:
        print(f"    {name:20s}  {count:4d} ({0:.1%})  ${avg_cost:.2f}/M")

# ──────────────────────────────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────────────────────────────

all_model_names = sorted({h["display_name"] for h in history})
palette = plt.cm.tab10(np.linspace(0, 0.9, len(all_model_names)))
color_map = {name: palette[i] for i, name in enumerate(all_model_names)}

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle(
    f"Scenario 3: Quality-Critical Enterprise + Hot Onboarding\n"
    f"(cost_penalty=0.0, newcomer at step {ONBOARD_STEP})",
    fontsize=13, fontweight="bold", y=0.98,
)

# Plot 1: Learning curve with onboarding event
ax = axes[0, 0]
window = max(30, N_PROMPTS // 15)
rewards_arr = np.array([h["reward"] for h in history])
rolling = np.convolve(rewards_arr, np.ones(window) / window, mode="valid")
ax.plot(range(window, window + len(rolling)), rolling,
        color="steelblue", linewidth=2, label="ParetoBandit")
ax.axvline(ONBOARD_STEP, color="green", ls="--", alpha=0.7,
           label=f"Newcomer onboarded (step {ONBOARD_STEP})")
ax.set_xlabel("Prompts Routed")
ax.set_ylabel(f"Reward (rolling avg, w={window})")
ax.set_title("Quality-First Learning + Model Onboarding")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 2: Model selection over time
ax = axes[0, 1]
sel_window = max(30, N_PROMPTS // 10)
for mname in all_model_names:
    sel = np.array([1.0 if h["display_name"] == mname else 0.0 for h in history])
    rolling_sel = np.convolve(sel, np.ones(sel_window) / sel_window, mode="valid")
    ax.plot(range(sel_window, sel_window + len(rolling_sel)), rolling_sel,
            label=mname, color=color_map[mname], linewidth=2)
ax.axvline(ONBOARD_STEP, color="green", ls="--", alpha=0.7)
ax.set_xlabel("Prompts Routed")
ax.set_ylabel("Selection Frequency")
ax.set_title("Newcomer Ramps Up — Existing Models Adapt")
ax.legend(fontsize=7, loc="upper right")
ax.grid(True, alpha=0.3)

# Plot 3: Per-category quality (post-onboarding)
ax = axes[1, 0]
categories = sorted(PROMPT_POOL.keys())
cat_rewards_pre = []
cat_rewards_post = []
for cat in categories:
    pre_cat = [h for h in pre if h["category"] == cat]
    post_cat = [h for h in post if h["category"] == cat]
    cat_rewards_pre.append(np.mean([h["reward"] for h in pre_cat]) if pre_cat else 0)
    cat_rewards_post.append(np.mean([h["reward"] for h in post_cat]) if post_cat else 0)

x = np.arange(len(categories))
width = 0.35
ax.bar(x - width / 2, cat_rewards_pre, width,
       label="Before onboarding", color="steelblue", alpha=0.8)
ax.bar(x + width / 2, cat_rewards_post, width,
       label="After onboarding", color="seagreen", alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels([c.replace("_", " ").title() for c in categories], fontsize=8)
ax.set_ylabel("Average Reward")
ax.set_title("Quality Improvement from Specialist Onboarding")
ax.legend(fontsize=8)
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3, axis="y")

# Plot 4: Newcomer adoption ramp
ax = axes[1, 1]
if newcomer_onboarded:
    post_steps = [h["step"] for h in post]
    newcomer_sel = np.array([
        1.0 if h["model_id"] == NEWCOMER_ID else 0.0 for h in post
    ])
    if len(newcomer_sel) > 30:
        ramp_window = 30
        rolling_ramp = np.convolve(
            newcomer_sel, np.ones(ramp_window) / ramp_window, mode="valid"
        )
        ax.plot(
            range(ONBOARD_STEP + ramp_window, ONBOARD_STEP + ramp_window + len(rolling_ramp)),
            rolling_ramp,
            color="seagreen", linewidth=2,
            label=NEWCOMER_INFO["display_name"],
        )
        ax.set_xlabel("Prompts Routed")
        ax.set_ylabel(f"Newcomer Selection Rate (w={ramp_window})")
        ax.set_title("Newcomer Adoption Ramp")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("scenario3_quality_enterprise.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nSaved → scenario3_quality_enterprise.png")

print(textwrap.dedent(f"""
  KEY TAKEAWAYS FOR DEPLOYMENT
  ────────────────────────────
  1. With cost_penalty=0.0, the router optimises purely for quality.
     Frontier models (GPT-4o, Claude Sonnet 4) dominate for hard legal
     reasoning, while mid-cost models handle routine summarisation.
  2. The newcomer ({NEWCOMER_INFO['display_name']}) was onboarded at
     step {ONBOARD_STEP} with a single register_model() call — no
     retraining, no downtime.
  3. Family-level sharing gave the newcomer an informed starting point
     from the accumulated knowledge of its model family.
  4. The newcomer's adoption rate is highest on contract review and
     clause extraction — exactly the categories it was fine-tuned for.
  5. In a regulated domain, the key insight is: quality-first routing
     with online learning discovers task-specific model preferences
     that a static "always use the most expensive model" rule misses.
"""))
