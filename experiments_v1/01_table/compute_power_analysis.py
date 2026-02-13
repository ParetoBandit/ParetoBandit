#!/usr/bin/env python3
"""
Power Analysis for Holdout Sample Size (N=750)

Computes the minimum detectable effect size for the holdout evaluation
using the CORRECT statistical test: a paired t-test.

Design rationale:
    All routing strategies (BanditGPT, RouteLLM, Warmup, Tabula Rasa) are
    evaluated on the SAME 750 holdout prompts. Each strategy selects a model
    for each prompt, and receives the oracle reward for that (prompt, model)
    pair. The comparison is therefore paired (within-subject), not independent.

    The correct test is a paired t-test (or Wilcoxon signed-rank), which
    removes prompt-level variance and uses σ_d (the standard deviation of
    per-prompt paired differences) rather than σ (marginal reward variance).

    σ_d is estimated empirically from the per-prompt reward gap between the
    two available models (Mixtral vs GPT-4-Turbo). This gap represents the
    maximum possible paired difference for any prompt — when two strategies
    disagree, their reward difference equals this gap. When they agree, the
    paired difference is zero.

Scientific question: With N=750 paired observations, what mean reward
differences can we reliably detect?
"""

import sys
import json
import gzip
import numpy as np
from pathlib import Path
from scipy.stats import t as t_dist, norm
from typing import Dict, List, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bandit_gpt.config_legacy import CANONICAL_HOLDOUT_DATA_PATH


def load_holdout_rewards(data_path: Path) -> Dict[str, Dict[str, float]]:
    """
    Load holdout reward data and group by prompt.

    Returns:
        Dict mapping prompt -> {model_id: reward_score}
    """
    prompt_rewards: Dict[str, Dict[str, float]] = {}

    with gzip.open(data_path, 'rt') as f:
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

    return prompt_rewards


def compute_reward_gap_distribution(
    prompt_rewards: Dict[str, Dict[str, float]]
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute the per-prompt reward gap between the two models.

    For each prompt, gap_i = reward(model_A, prompt_i) - reward(model_B, prompt_i).
    This is the maximum possible paired difference when two routing strategies
    disagree on prompt i.

    Returns:
        (gaps, models): Array of signed gaps and list of model names
    """
    # Identify the two models
    all_models = set()
    for rewards in prompt_rewards.values():
        all_models.update(rewards.keys())
    models = sorted(all_models)

    if len(models) != 2:
        print(f"  Warning: Expected 2 models, found {len(models)}: {models}")
        print(f"  Using first two: {models[:2]}")
        models = models[:2]

    # Compute per-prompt gaps (only for prompts with both models)
    gaps = []
    for prompt, rewards in prompt_rewards.items():
        if models[0] in rewards and models[1] in rewards:
            gap = rewards[models[0]] - rewards[models[1]]
            gaps.append(gap)

    return np.array(gaps), models


def compute_paired_power_analysis(
    n: int,
    sigma_d: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """
    Compute minimum detectable effect size for a PAIRED t-test.

    For paired design:
        - Test statistic: t = d_bar / (σ_d / √n)
        - Degrees of freedom: df = n - 1
        - Standard error: SE = σ_d / √n

    Parameters:
        n:       Number of paired observations (prompts)
        sigma_d: Standard deviation of paired differences
        alpha:   Significance level (two-tailed)
        power:   Desired power (1 - β)

    Returns:
        delta: Minimum detectable mean difference (in raw reward units)
    """
    df = n - 1

    # Critical value for two-tailed test
    t_crit = t_dist.ppf(1 - alpha / 2, df)

    # Value from power requirement
    t_beta = t_dist.ppf(power, df)

    # Non-centrality parameter: ncp = t_crit + t_beta
    ncp = t_crit + t_beta

    # For paired test: ncp = delta / (sigma_d / sqrt(n))
    # => delta = ncp * sigma_d / sqrt(n)
    delta = ncp * sigma_d / np.sqrt(n)

    return delta


def compute_power_for_effect(
    n: int,
    delta: float,
    sigma_d: float,
    alpha: float = 0.05,
) -> float:
    """
    Compute statistical power for a given effect size under paired t-test.

    Parameters:
        n:       Number of paired observations
        delta:   True mean difference to detect
        sigma_d: Standard deviation of paired differences
        alpha:   Significance level (two-tailed)

    Returns:
        power: Probability of rejecting H0 when H1 is true
    """
    df = n - 1
    t_crit = t_dist.ppf(1 - alpha / 2, df)

    # Non-centrality parameter
    ncp = delta / (sigma_d / np.sqrt(n))

    # Power using non-central t distribution (normal approximation for large df)
    # For df >= 30 the normal approximation is excellent
    if df >= 30:
        power = 1 - norm.cdf(t_crit - ncp) + norm.cdf(-t_crit - ncp)
    else:
        from scipy.stats import nct
        power = 1 - nct.cdf(t_crit, df, ncp) + nct.cdf(-t_crit, df, ncp)

    return power


def interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d effect size (conventional thresholds)."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def main():
    print("=" * 70)
    print("POWER ANALYSIS FOR HOLDOUT EVALUATION (N=750)")
    print("Correct analysis: Paired t-test with data-driven σ_d")
    print("=" * 70)

    # ================================================================
    # Step 1: Load actual holdout reward data
    # ================================================================
    print("\n1. Loading holdout reward data...")
    print(f"   Path: {CANONICAL_HOLDOUT_DATA_PATH}")

    prompt_rewards = load_holdout_rewards(CANONICAL_HOLDOUT_DATA_PATH)
    n_prompts = len(prompt_rewards)
    print(f"   Loaded: {n_prompts} prompts with rewards")

    # ================================================================
    # Step 2: Compute per-prompt reward gap distribution
    # ================================================================
    print("\n2. Computing per-prompt reward gaps...")
    gaps, models = compute_reward_gap_distribution(prompt_rewards)
    n_paired = len(gaps)

    print(f"   Models: {models[0]} vs {models[1]}")
    print(f"   Prompts with both models: {n_paired}")

    # Reward gap statistics
    print(f"\n   Per-prompt reward gap statistics:")
    print(f"     Mean gap:    {np.mean(gaps):.4f}")
    print(f"     Std gap:     {np.std(gaps, ddof=1):.4f}")
    print(f"     Median gap:  {np.median(gaps):.4f}")
    print(f"     Min gap:     {np.min(gaps):.4f}")
    print(f"     Max gap:     {np.max(gaps):.4f}")

    # Absolute gaps (magnitude of disagreement)
    abs_gaps = np.abs(gaps)
    print(f"\n   Absolute gap statistics (|gap|):")
    print(f"     Mean |gap|:  {np.mean(abs_gaps):.4f}")
    print(f"     Std |gap|:   {np.std(abs_gaps, ddof=1):.4f}")
    print(f"     % zero gap:  {100 * np.mean(abs_gaps < 0.001):.1f}%")
    print(f"     % gap > 0.1: {100 * np.mean(abs_gaps > 0.1):.1f}%")
    print(f"     % gap > 0.5: {100 * np.mean(abs_gaps > 0.5):.1f}%")

    # Per-model marginal reward statistics
    model_rewards = {m: [] for m in models}
    for rewards in prompt_rewards.values():
        for m in models:
            if m in rewards:
                model_rewards[m].append(rewards[m])

    print(f"\n   Per-model reward statistics:")
    for m in models:
        r = np.array(model_rewards[m])
        print(f"     {m}: mean={np.mean(r):.4f}, std={np.std(r, ddof=1):.4f}")

    sigma_marginal = np.mean([np.std(np.array(v), ddof=1) for v in model_rewards.values()])
    print(f"\n   Average marginal σ (per-model): {sigma_marginal:.4f}")

    # ================================================================
    # Step 3: Estimate σ_d for paired differences
    # ================================================================
    print("\n3. Estimating σ_d (paired difference standard deviation)...")

    # The paired difference d_i = reward_A(prompt_i) - reward_B(prompt_i)
    # can only be non-zero when the two strategies select DIFFERENT models.
    #
    # When they disagree, |d_i| = |gap_i| (the model reward gap for that prompt).
    # When they agree, d_i = 0.
    #
    # σ_d depends on the disagreement rate. We provide estimates for
    # several scenarios.

    sigma_gap = np.std(gaps, ddof=1)  # SD of the full gap distribution

    print(f"\n   σ_gap (full per-prompt gap distribution): {sigma_gap:.4f}")
    print(f"   (This is σ_d when strategies ALWAYS disagree — upper bound)")

    # Scenario analysis: what if strategies disagree on p% of prompts?
    print(f"\n   Scenario analysis (σ_d by disagreement rate):")
    print(f"   {'Disagree %':<15} {'σ_d':<10} {'Description'}")
    print(f"   {'-'*55}")

    scenarios = {
        1.00: "Always disagree (upper bound)",
        0.50: "Disagree on half of prompts",
        0.30: "Moderate disagreement (typical)",
        0.20: "Low disagreement",
        0.10: "Rare disagreement",
    }

    # For a mixture of 0s and gaps:
    # If d_i = gap_i with probability p, else d_i = 0:
    # E[d_i] = p * E[gap_i]
    # Var(d_i) = p * E[gap_i^2] - (p * E[gap_i])^2
    #          = p * E[gap_i^2] - p^2 * E[gap_i]^2
    mean_gap = np.mean(gaps)
    mean_gap_sq = np.mean(gaps ** 2)

    sigma_d_scenarios = {}
    for p, desc in scenarios.items():
        var_d = p * mean_gap_sq - (p * mean_gap) ** 2
        sigma_d = np.sqrt(max(var_d, 0))
        sigma_d_scenarios[p] = sigma_d
        print(f"   {p*100:>6.0f}%        {sigma_d:.4f}     {desc}")

    # Use the full gap σ as the conservative estimate (upper bound on σ_d)
    # This is conservative because it assumes strategies always disagree
    sigma_d = sigma_gap
    print(f"\n   → Using conservative σ_d = {sigma_d:.4f} (full gap distribution)")
    print(f"     This assumes strategies always disagree (worst case for power).")
    print(f"     Actual power is likely HIGHER because strategies often agree.")

    # ================================================================
    # Step 4: Paired t-test power analysis
    # ================================================================
    print("\n" + "=" * 70)
    print("4. PAIRED T-TEST POWER ANALYSIS")
    print("=" * 70)

    n = n_paired
    alpha = 0.05
    target_power = 0.80

    print(f"\n   Parameters:")
    print(f"     N (paired observations):   {n}")
    print(f"     σ_d (paired diff std):      {sigma_d:.4f}")
    print(f"     α (significance level):     {alpha}")
    print(f"     Target power (1-β):         {target_power}")
    print(f"     Degrees of freedom:         {n - 1}")

    # Minimum detectable effect (MDE)
    mde = compute_paired_power_analysis(n, sigma_d, alpha, target_power)
    cohens_d_mde = mde / sigma_d

    print(f"\n   Minimum Detectable Effect (MDE):")
    print(f"     δ (raw reward units): {mde:.4f}")
    print(f"     Cohen's d:            {cohens_d_mde:.4f} ({interpret_cohens_d(cohens_d_mde)})")

    # ================================================================
    # Step 5: Compare to observed effects
    # ================================================================
    print("\n" + "=" * 70)
    print("5. COMPARISON TO OBSERVED EFFECTS")
    print("=" * 70)

    # From Table 2: BanditGPT avg_reward=0.912, RouteLLM=0.883
    observed_effect = 0.912 - 0.883

    print(f"\n   Observed effect (BanditGPT vs RouteLLM): δ = {observed_effect:.3f}")
    print(f"   Minimum detectable effect (MDE):         δ = {mde:.4f}")

    observed_power = compute_power_for_effect(n, observed_effect, sigma_d, alpha)

    if observed_effect >= mde:
        print(f"\n   ✓ ADEQUATELY POWERED")
        print(f"     Observed effect ({observed_effect:.3f}) exceeds MDE ({mde:.4f})")
        print(f"     Power for observed effect: {observed_power*100:.1f}%")
    else:
        print(f"\n   ⚠ UNDERPOWERED for observed effect")
        print(f"     Observed effect ({observed_effect:.3f}) is below MDE ({mde:.4f})")
        print(f"     Power for observed effect: {observed_power*100:.1f}%")

    # ================================================================
    # Step 6: Power curve
    # ================================================================
    print("\n" + "=" * 70)
    print("6. POWER CURVE (paired t-test)")
    print("=" * 70)

    print(f"\n   {'Effect δ':<12} {'Cohen d':<12} {'Power (%)':<12} {'Status'}")
    print(f"   {'-'*55}")

    test_deltas = [0.005, 0.01, 0.015, 0.02, 0.025, 0.029, 0.03, 0.035, 0.04, 0.05]
    for td in test_deltas:
        td_d = td / sigma_d
        td_power = compute_power_for_effect(n, td, sigma_d, alpha)
        marker = " ◄ observed" if abs(td - observed_effect) < 0.001 else ""
        status = "✓ Well-powered" if td_power >= 0.80 else "⚠ Underpowered"
        print(f"   {td:<12.4f} {td_d:<12.4f} {td_power*100:<12.1f} {status}{marker}")

    # ================================================================
    # Step 7: Comparison to old (incorrect) analysis
    # ================================================================
    print("\n" + "=" * 70)
    print("7. COMPARISON: OLD (two-sample) vs CORRECT (paired) ANALYSIS")
    print("=" * 70)

    # Old analysis used two-sample t-test with guessed sigma=0.3
    old_sigma = 0.3
    old_df = 2 * n - 2
    old_t_crit = t_dist.ppf(1 - alpha / 2, old_df)
    old_t_beta = t_dist.ppf(target_power, old_df)
    old_ncp = old_t_crit + old_t_beta
    old_mde = old_ncp * old_sigma * np.sqrt(2 / n)

    print(f"\n   {'Metric':<35} {'Old (wrong)':<20} {'Corrected':<20}")
    print(f"   {'-'*75}")
    print(f"   {'Test type':<35} {'Two-sample t':<20} {'Paired t':<20}")
    print(f"   {'σ used':<35} {old_sigma:<20.4f} {sigma_d:<20.4f}")
    print(f"   {'σ source':<35} {'Guessed':<20} {'Empirical (data)':<20}")
    print(f"   {'Degrees of freedom':<35} {old_df:<20} {n-1:<20}")
    print(f"   {'Standard error formula':<35} {'σ√(2/n)':<20} {'σ_d/√n':<20}")
    print(f"   {'MDE (δ)':<35} {old_mde:<20.4f} {mde:<20.4f}")
    print(f"   {'MDE Cohen d':<35} {old_mde/old_sigma:<20.4f} {cohens_d_mde:<20.4f}")

    old_power_obs = compute_power_for_effect(n, observed_effect, old_sigma * np.sqrt(2), alpha)
    print(f"   {'Power at observed δ=0.029':<35} {old_power_obs*100:<19.1f}% {observed_power*100:<19.1f}%")

    old_adequate = observed_effect >= old_mde
    new_adequate = observed_effect >= mde
    print(f"   {'Adequately powered?':<35} {'Yes' if old_adequate else 'No':<20} {'Yes' if new_adequate else 'No':<20}")

    # ================================================================
    # Step 8: Summary
    # ================================================================
    print("\n" + "=" * 70)
    print("8. SUMMARY")
    print("=" * 70)

    print(f"\n   With N={n} holdout prompts (paired design):")
    print(f"\n   1. Empirical σ_d = {sigma_d:.4f} (from per-prompt reward gaps)")
    print(f"   2. MDE = {mde:.4f} (80% power, α=0.05, paired t-test)")
    print(f"   3. Observed effect δ = {observed_effect:.3f}")
    print(f"   4. Power at observed effect: {observed_power*100:.1f}%")

    if new_adequate:
        print(f"\n   Conclusion: The holdout evaluation IS adequately powered")
        print(f"   to detect the observed effect size using the correct paired test.")
    else:
        print(f"\n   Conclusion: The holdout evaluation is marginally powered")
        print(f"   for the observed effect ({observed_power*100:.1f}% power).")
        print(f"   However, the paired test is substantially more powerful")
        print(f"   than the previously-used two-sample test.")

    print()

    # ================================================================
    # Save results
    # ================================================================
    results = {
        "test_type": "paired_t_test",
        "n": int(n),
        "alpha": float(alpha),
        "target_power": float(target_power),
        "degrees_of_freedom": int(n - 1),
        "sigma_d": float(sigma_d),
        "sigma_d_source": "empirical_per_prompt_reward_gap",
        "min_detectable_effect": float(mde),
        "cohens_d_at_mde": float(cohens_d_mde),
        "observed_effect": float(observed_effect),
        "power_at_observed_effect": float(observed_power),
        "adequately_powered": bool(new_adequate),
        "reward_gap_stats": {
            "mean_gap": float(np.mean(gaps)),
            "std_gap": float(sigma_gap),
            "median_gap": float(np.median(gaps)),
            "mean_abs_gap": float(np.mean(abs_gaps)),
            "pct_zero_gap": float(np.mean(abs_gaps < 0.001)),
        },
        "comparison_to_old_analysis": {
            "old_test_type": "two_sample_t_test",
            "old_sigma": float(old_sigma),
            "old_sigma_source": "guessed",
            "old_mde": float(old_mde),
            "old_adequately_powered": bool(old_adequate),
            "improvement_reason": "Paired test removes prompt-level variance; "
                                  "empirical σ_d replaces guessed σ",
        },
    }

    output_file = Path(__file__).parent / "power_analysis_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"   Results saved to: {output_file}")
    print()


if __name__ == "__main__":
    main()
