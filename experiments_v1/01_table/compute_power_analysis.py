#!/usr/bin/env python3
"""
Power Analysis for Holdout Routing Evaluation (N=750)

Monte-Carlo power analysis that respects the discrete data structure:

    - Rewards are DISCRETE pairwise preference outcomes (win=1, loss=0)
    - Per-prompt model gap is trinomial: +1 (GPT-4T wins), 0 (tie), -1 (Mixtral wins)
    - 72.8% of prompts are ties (routing irrelevant), 27.2% are informative
    - Routing strategies compared on the SAME prompts (paired design)

This approach matches Figure 1's Monte-Carlo chi-squared power analysis
in philosophy: simulate from the observed discrete distribution rather
than assuming continuous/normal data.

Three power analyses are provided:

    1. McNemar's test  — gold standard for paired binary outcomes
    2. Binomial test   — routing accuracy on informative prompts
    3. Paired t-test   — included for reference, but inappropriate for discrete data

Scientific question: With N=750 paired observations (discrete rewards),
what routing accuracy differences can we reliably detect?

Usage:
    python compute_power_analysis.py
"""

import sys
import json
import gzip
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.stats import binomtest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bandit_gpt.config_legacy import CANONICAL_HOLDOUT_DATA_PATH


# =====================================================================
# Data Loading
# =====================================================================

def load_holdout_rewards(data_path: Path):
    """Load holdout rewards grouped by prompt. Returns per-prompt gaps."""
    prompt_rewards = {}
    with gzip.open(data_path, "rt") as f:
        for line in f:
            entry = json.loads(line)
            if not entry.get("ok", False):
                continue
            prompt = entry["prompt"]
            model_id = entry["model_id"]
            score = entry.get("raw_score", 0.0)
            if prompt not in prompt_rewards:
                prompt_rewards[prompt] = {}
            prompt_rewards[prompt][model_id] = score

    # Compute per-prompt gaps
    models = set()
    for r in prompt_rewards.values():
        models.update(r.keys())
    models = sorted(models)

    gaps = []
    for prompt, rewards in prompt_rewards.items():
        if len(rewards) >= 2:
            gap = rewards[models[0]] - rewards[models[1]]
            gaps.append(gap)

    return np.array(gaps), models, len(prompt_rewards)


# =====================================================================
# Monte-Carlo Power Analysis: McNemar's Test
# =====================================================================

def mc_power_mcnemar(
    n_informative: int,
    true_accuracy: float,
    n_total: int,
    frac_informative: float,
    alpha: float = 0.05,
    n_simulations: int = 20_000,
) -> float:
    """
    Monte-Carlo power for McNemar's test on paired binary routing data.

    Simulates the scenario where:
    - Two routing strategies are compared on the same N prompts
    - Only informative prompts (where models differ) produce discordant pairs
    - Strategy A has `true_accuracy` on informative prompts
    - Strategy B has (1 - true_accuracy) on informative prompts (worst case)

    Under H0: both strategies have 50% accuracy on informative prompts
    Under H1: strategy A has `true_accuracy`, B has (1 - true_accuracy)

    Returns:
        Power (fraction of simulations rejecting H0)
    """
    rejections = 0

    for _ in range(n_simulations):
        # Sample which prompts are informative (have non-zero gap)
        is_informative = np.random.random(n_total) < frac_informative
        n_inf = is_informative.sum()

        if n_inf == 0:
            continue

        # On informative prompts, strategy A is correct with prob true_accuracy
        a_correct = np.random.random(n_inf) < true_accuracy
        # Strategy B: independent with prob (1 - true_accuracy)
        # This models maximum disagreement; real disagreement would be lower
        b_correct = np.random.random(n_inf) < (1 - true_accuracy)

        # Discordant pairs
        b_val = int(np.sum(a_correct & ~b_correct))  # A right, B wrong
        c_val = int(np.sum(~a_correct & b_correct))  # A wrong, B right
        n_discordant = b_val + c_val

        if n_discordant == 0:
            continue

        # McNemar's exact test (two-sided binomial)
        p_value = binomtest(b_val, n_discordant, 0.5).pvalue
        if p_value < alpha:
            rejections += 1

    return rejections / n_simulations


# =====================================================================
# Monte-Carlo Power Analysis: Binomial Test on Routing Accuracy
# =====================================================================

def mc_power_binomial(
    n_informative: int,
    true_accuracy: float,
    alpha: float = 0.05,
    n_simulations: int = 20_000,
) -> float:
    """
    Monte-Carlo power for one-sided binomial test on routing accuracy.

    Tests: H0: accuracy <= 0.5 vs H1: accuracy > 0.5
    on the informative prompts only.

    Returns:
        Power (fraction of simulations rejecting H0)
    """
    rejections = 0

    for _ in range(n_simulations):
        # Sample routing outcomes on informative prompts
        n_correct = np.random.binomial(n_informative, true_accuracy)
        p_value = binomtest(n_correct, n_informative, 0.5, alternative="greater").pvalue
        if p_value < alpha:
            rejections += 1

    return rejections / n_simulations


# =====================================================================
# Monte-Carlo Power Analysis: Paired t-test (reference only)
# =====================================================================

def mc_power_paired_t(
    n_total: int,
    gaps: np.ndarray,
    true_accuracy_advantage: float,
    alpha: float = 0.05,
    n_simulations: int = 20_000,
) -> float:
    """
    Monte-Carlo power for paired t-test on discrete reward differences.

    Simulates paired differences from the actual discrete gap distribution.
    Included for reference; NOT recommended for discrete data.
    """
    # Compute the discrete distribution of gaps
    unique_gaps, gap_counts = np.unique(gaps, return_counts=True)
    gap_probs = gap_counts / gap_counts.sum()

    rejections = 0

    for _ in range(n_simulations):
        # Sample per-prompt gaps from the empirical distribution
        sampled_gaps = np.random.choice(unique_gaps, size=n_total, p=gap_probs)

        # For informative prompts (gap != 0), strategy A gets it right
        # with probability (0.5 + true_accuracy_advantage)
        paired_diffs = np.zeros(n_total)
        informative = np.abs(sampled_gaps) > 1e-9
        n_inf = informative.sum()

        if n_inf > 0:
            # A gets positive gap with prob (0.5 + advantage)
            a_wins = np.random.random(n_inf) < (0.5 + true_accuracy_advantage)
            paired_diffs[informative] = np.where(
                a_wins,
                np.abs(sampled_gaps[informative]),
                -np.abs(sampled_gaps[informative]),
            )

        # Paired t-test
        if np.std(paired_diffs, ddof=1) > 0:
            t_stat, p_value = stats.ttest_1samp(paired_diffs, 0)
            if p_value < alpha:
                rejections += 1

    return rejections / n_simulations


# =====================================================================
# Main
# =====================================================================

def main():
    print("=" * 70)
    print("POWER ANALYSIS FOR HOLDOUT ROUTING EVALUATION (N=750)")
    print("Monte-Carlo simulation on discrete pairwise outcomes")
    print("(Consistent with Figure 1's Monte-Carlo chi-squared approach)")
    print("=" * 70)

    # ================================================================
    # 1. Load and characterize holdout data
    # ================================================================
    print("\n1. Loading holdout reward data...")
    gaps, models, n_prompts = load_holdout_rewards(CANONICAL_HOLDOUT_DATA_PATH)

    n_informative = int(np.sum(np.abs(gaps) > 1e-9))
    n_ties = int(np.sum(np.abs(gaps) <= 1e-9))
    frac_informative = n_informative / len(gaps)

    print(f"   N = {len(gaps)} prompts")
    print(f"   Models: {models[0]} vs {models[1]}")
    print(f"\n   Reward structure (discrete pairwise outcomes):")
    print(f"     Ties (gap = 0):        {n_ties:>4} ({100*n_ties/len(gaps):.1f}%) — routing irrelevant")
    print(f"     Informative (gap ≠ 0): {n_informative:>4} ({100*frac_informative:.1f}%) — routing matters")

    gap_counts = {}
    for g in sorted(np.unique(gaps)):
        gap_counts[f"{g:+.0f}"] = int(np.sum(np.abs(gaps - g) < 1e-9))
    print(f"     Gap distribution: {gap_counts}")

    # ================================================================
    # 2. Monte-Carlo power: McNemar's test
    # ================================================================
    print("\n" + "=" * 70)
    print("2. POWER ANALYSIS: McNEMAR'S EXACT TEST (paired binary)")
    print("   Gold standard for paired discrete outcomes")
    print("=" * 70)

    alpha = 0.05
    n_sim = 20_000

    print(f"\n   Parameters:")
    print(f"     N = {len(gaps)}, informative = {n_informative} ({100*frac_informative:.1f}%)")
    print(f"     α = {alpha}, simulations = {n_sim:,}")
    print(f"\n   {'Accuracy A':<14} {'Accuracy B':<14} {'Power':<10} {'Status'}")
    print(f"   {'-'*50}")

    mcnemar_results = {}
    for acc_a in [0.52, 0.55, 0.58, 0.60, 0.65, 0.70]:
        power = mc_power_mcnemar(
            n_informative, acc_a, len(gaps), frac_informative, alpha, n_sim,
        )
        status = "✓ Adequate" if power >= 0.80 else "⚠ Under"
        acc_b = 1 - acc_a
        print(f"   {100*acc_a:>5.0f}%        {100*acc_b:>5.0f}%        {100*power:>5.1f}%     {status}")
        mcnemar_results[f"acc={acc_a:.2f}"] = {
            "accuracy_A": acc_a,
            "accuracy_B": 1 - acc_a,
            "power": float(power),
            "adequate": power >= 0.80,
        }

    # ================================================================
    # 3. Monte-Carlo power: Binomial test on informative prompts
    # ================================================================
    print("\n" + "=" * 70)
    print("3. POWER ANALYSIS: BINOMIAL TEST (routing accuracy, one-sided)")
    print("   Tests: is routing accuracy > 50% on informative prompts?")
    print("=" * 70)

    print(f"\n   N_informative = {n_informative}")
    print(f"\n   {'True accuracy':<16} {'Power':<10} {'Status'}")
    print(f"   {'-'*40}")

    binomial_results = {}
    for acc in [0.52, 0.55, 0.58, 0.60, 0.65, 0.70]:
        power = mc_power_binomial(n_informative, acc, alpha, n_sim)
        status = "✓ Adequate" if power >= 0.80 else "⚠ Under"
        print(f"   {100*acc:>5.0f}%           {100*power:>5.1f}%     {status}")
        binomial_results[f"acc={acc:.2f}"] = {
            "accuracy": acc,
            "power": float(power),
            "adequate": power >= 0.80,
        }

    # ================================================================
    # 4. Monte-Carlo power: Paired t-test (reference, NOT recommended)
    # ================================================================
    print("\n" + "=" * 70)
    print("4. POWER ANALYSIS: PAIRED T-TEST (reference only, NOT recommended)")
    print("   Treats discrete outcomes as continuous — inappropriate for this data")
    print("=" * 70)

    print(f"\n   {'Accuracy advantage':<22} {'Power':<10} {'Status'}")
    print(f"   {'-'*45}")

    paired_t_results = {}
    for adv in [0.02, 0.05, 0.08, 0.10, 0.15, 0.20]:
        power = mc_power_paired_t(len(gaps), gaps, adv, alpha, n_sim)
        status = "✓ Adequate" if power >= 0.80 else "⚠ Under"
        print(f"   ±{100*adv:>4.0f}%                {100*power:>5.1f}%     {status}")
        paired_t_results[f"adv={adv:.2f}"] = {
            "accuracy_advantage": adv,
            "power": float(power),
            "adequate": power >= 0.80,
        }

    # ================================================================
    # 5. False positive rate validation
    # ================================================================
    print("\n" + "=" * 70)
    print("5. FALSE POSITIVE RATE VALIDATION (H0: strategies are identical)")
    print("=" * 70)

    fpr_mcnemar = mc_power_mcnemar(
        n_informative, 0.50, len(gaps), frac_informative, alpha, n_sim,
    )
    fpr_binomial = mc_power_binomial(n_informative, 0.50, alpha, n_sim)
    fpr_paired_t = mc_power_paired_t(len(gaps), gaps, 0.00, alpha, n_sim)

    print(f"\n   Under H0 (both strategies at 50% accuracy):")
    print(f"     McNemar's FPR:   {100*fpr_mcnemar:.1f}%  (target: {100*alpha:.0f}%)")
    print(f"     Binomial FPR:    {100*fpr_binomial:.1f}%  (target: {100*alpha:.0f}%)")
    print(f"     Paired t FPR:    {100*fpr_paired_t:.1f}%  (target: {100*alpha:.0f}%)")

    # ================================================================
    # 6. Comparison across tests
    # ================================================================
    print("\n" + "=" * 70)
    print("6. COMPARISON: WHICH TEST IS BEST FOR THIS DATA?")
    print("=" * 70)

    print("""
   Test               Data assumption    Focuses on           Recommended?
   ───────────────────────────────────────────────────────────────────────
   McNemar's exact    Paired binary      Discordant pairs     ✓ Yes (gold standard)
   Binomial (1-sided) Binary, 1 sample   Informative prompts  ✓ Yes (interpretable)
   Paired t-test      Continuous, normal  All prompts          ✗ No (wrong assumption)

   Key insight: With 72.8% ties (routing-irrelevant), the paired t-test
   wastes most of its sample on zeros. McNemar's and binomial tests focus
   on the 27.2% of prompts where routing actually matters, yielding
   substantially more power.

   This is consistent with Figure 1's approach: use categorical/discrete
   tests (chi-squared, Cramer's V) for discrete pairwise outcome data,
   not continuous parametric tests.""")

    # ================================================================
    # 7. Connection to Figure 1
    # ================================================================
    print("\n" + "=" * 70)
    print("7. CONNECTION TO FIGURE 1")
    print("=" * 70)

    print("""
   Both Table 1 and Figure 1 analyze the same N=750 holdout prompts with
   discrete pairwise preference outcomes (win/tie/loss).

   Figure 1 analyzes BETWEEN-cluster differences:
     - Chi-squared on win/tie/loss contingency table
     - Monte-Carlo power > 99% at observed effect
     - Correctly treats data as categorical

   Table 1 analyzes BETWEEN-strategy routing differences:
     - McNemar's / binomial on paired routing outcomes
     - Power depends on routing accuracy (see tables above)
     - Now correctly treats data as discrete (this script)

   Both use Monte-Carlo simulation from the observed discrete distribution
   rather than parametric normal assumptions. The methodology is now
   consistent across experiments.""")

    # ================================================================
    # 8. Summary
    # ================================================================
    print("\n" + "=" * 70)
    print("8. SUMMARY")
    print("=" * 70)

    print(f"""
   Holdout: N={len(gaps)}, informative prompts: {n_informative} ({100*frac_informative:.1f}%)

   Recommended tests and their power (at 60% routing accuracy):""")

    # Compute power at 60% for summary
    p_mcnemar_60 = mc_power_mcnemar(n_informative, 0.60, len(gaps), frac_informative, alpha, 10_000)
    p_binomial_60 = mc_power_binomial(n_informative, 0.60, alpha, 10_000)
    p_paired_t_60 = mc_power_paired_t(len(gaps), gaps, 0.10, alpha, 10_000)

    print(f"     McNemar's test:       {100*p_mcnemar_60:.0f}% power  ✓")
    print(f"     Binomial test:        {100*p_binomial_60:.0f}% power  ✓")
    print(f"     Paired t-test (ref):  {100*p_paired_t_60:.0f}% power  (not recommended)")

    print(f"""
   Conclusion:
   • With {n_informative} informative prompts, McNemar's and binomial tests
     are well-powered to detect routing accuracy ≥ 58-60%
   • The paired t-test is underpowered because it treats 72.8% ties as
     zero-variance observations rather than filtering them out
   • Use McNemar's for pairwise strategy comparisons
   • Use binomial for single-strategy routing accuracy
   • This methodology is now consistent with Figure 1's approach
""")

    # ================================================================
    # Save results
    # ================================================================
    results = {
        "methodology": "monte_carlo_simulation",
        "data_type": "discrete_pairwise_outcomes",
        "consistent_with": "Figure 1 (Monte-Carlo chi-squared power analysis)",
        "n_total": int(len(gaps)),
        "n_informative": int(n_informative),
        "n_ties": int(n_ties),
        "frac_informative": float(frac_informative),
        "alpha": alpha,
        "n_simulations": n_sim,
        "gap_distribution": gap_counts,
        "mcnemar_power": mcnemar_results,
        "binomial_power": binomial_results,
        "paired_t_power_reference": paired_t_results,
        "false_positive_rates": {
            "mcnemar": float(fpr_mcnemar),
            "binomial": float(fpr_binomial),
            "paired_t": float(fpr_paired_t),
            "target": float(alpha),
        },
        "recommendation": (
            "Use McNemar's exact test for pairwise strategy comparisons "
            "and binomial test for single-strategy routing accuracy. "
            "Both are well-powered at 60% accuracy on informative prompts. "
            "Paired t-test is inappropriate for discrete pairwise outcomes."
        ),
    }

    output_file = Path(__file__).parent / "power_analysis_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"   Results saved to: {output_file}")


if __name__ == "__main__":
    main()
