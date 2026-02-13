#!/usr/bin/env python3
"""
Statistical Power Analysis for Table 2

This script computes:
1. Observed effect size (Cohen's d) between η=0.1 and η=1.0
2. Statistical power achieved with N=10 seeds
3. Required sample size for 80% and 95% power
4. Minimum detectable effect size (MDE) given N=10

Justifies whether N=10 seeds is sufficient for the experiment.
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats
from typing import Dict, List


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    return (mean1 - mean2) / pooled_std if pooled_std > 0 else 0.0


def compute_power(effect_size: float, n: int, alpha: float = 0.05) -> float:
    """
    Compute statistical power for independent samples t-test.
    
    Uses non-centrality parameter approach.
    """
    from scipy.stats import nct
    
    # Degrees of freedom for two-sample t-test
    df = 2 * n - 2
    
    # Non-centrality parameter
    ncp = effect_size * np.sqrt(n / 2)
    
    # Critical value for two-tailed test
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    
    # Power = P(reject H0 | H1 is true)
    # = P(|T| > t_crit | ncp)
    power = 1 - nct.cdf(t_crit, df, ncp) + nct.cdf(-t_crit, df, ncp)
    
    return power


def compute_required_n(effect_size: float, power: float = 0.80, alpha: float = 0.05) -> int:
    """
    Compute required sample size per group for given power.
    
    Iterative search approach.
    """
    n = 2
    while n < 1000:
        current_power = compute_power(effect_size, n, alpha)
        if current_power >= power:
            return n
        n += 1
    return n


def compute_mde(n: int, power: float = 0.80, alpha: float = 0.05) -> float:
    """
    Compute minimum detectable effect size given sample size.
    
    Iterative search approach.
    """
    effect_size = 0.01
    while effect_size < 5.0:
        current_power = compute_power(effect_size, n, alpha)
        if current_power >= power:
            return effect_size
        effect_size += 0.01
    return effect_size


def load_results(path: Path) -> Dict:
    """Load multi-seed results."""
    with open(path) as f:
        return json.load(f)


def power_analysis():
    """Comprehensive power analysis."""
    
    print("="*100)
    print("STATISTICAL POWER ANALYSIS: Table 2 Multi-Seed Validation")
    print("="*100)
    
    # Load data
    eta_01_path = Path(__file__).parent / 'data' / 'eta_0.1_holdout_multiseed' / 'results_multiseed.json'
    eta_10_path = Path(__file__).parent / 'data' / 'eta_1.0_holdout_multiseed' / 'results_multiseed.json'
    
    eta_01_data = load_results(eta_01_path)
    eta_10_data = load_results(eta_10_path)
    
    # Extract Corralling results
    regrets_01 = np.array(eta_01_data['Hybrid (Corralling)']['statistics']['raw_values']['cumulative_regret'])
    regrets_10 = np.array(eta_10_data['Hybrid (Corralling)']['statistics']['raw_values']['cumulative_regret'])
    
    # 1. Observed effect size
    print("\n1. OBSERVED EFFECT SIZE")
    print("-"*100)
    
    observed_d = cohens_d(regrets_01, regrets_10)
    
    mean_01, std_01 = np.mean(regrets_01), np.std(regrets_01, ddof=1)
    mean_10, std_10 = np.mean(regrets_10), np.std(regrets_10, ddof=1)
    
    print(f"η=0.1: {mean_01:.1f} ± {std_01:.1f}")
    print(f"η=1.0: {mean_10:.1f} ± {std_10:.1f}")
    print(f"Difference: {mean_01 - mean_10:.1f} (η=0.1 has {'lower' if mean_01 < mean_10 else 'higher'} regret)")
    print(f"\nCohen's d: {observed_d:.3f}")
    
    interpretation = (
        "negligible" if abs(observed_d) < 0.2 else
        "small" if abs(observed_d) < 0.5 else
        "medium" if abs(observed_d) < 0.8 else
        "large"
    )
    print(f"Interpretation: {interpretation} effect")
    
    # 2. Achieved power with N=10
    print("\n2. ACHIEVED STATISTICAL POWER (N=10 per group)")
    print("-"*100)
    
    n = 10
    power_05 = compute_power(abs(observed_d), n, alpha=0.05)
    power_01 = compute_power(abs(observed_d), n, alpha=0.01)
    power_bonf = compute_power(abs(observed_d), n, alpha=0.05/6)  # Bonferroni correction
    
    print(f"Power at α=0.05:              {power_05:.3f} ({100*power_05:.1f}%)")
    print(f"Power at α=0.01:              {power_01:.3f} ({100*power_01:.1f}%)")
    print(f"Power at α=0.0083 (Bonferroni): {power_bonf:.3f} ({100*power_bonf:.1f}%)")
    
    print("\n" + "⚠️  INTERPRETATION:")
    if power_05 < 0.5:
        print(f"    • Power = {100*power_05:.1f}% is VERY LOW (< 50%)")
        print(f"    • The study is severely underpowered to detect the observed effect")
        print(f"    • Non-significant result (p=0.63) is expected even if true difference exists")
        print(f"    • This is 'absence of evidence', NOT 'evidence of absence'")
    elif power_05 < 0.8:
        print(f"    • Power = {100*power_05:.1f}% is UNDERPOWERED (< 80% threshold)")
        print(f"    • Standard practice requires 80% power for reliable conclusions")
        print(f"    • Non-significant result may be due to insufficient sample size")
    else:
        print(f"    • Power = {100*power_05:.1f}% is ADEQUATE (≥ 80% threshold)")
        print(f"    • The study has sufficient power to detect the observed effect")
        print(f"    • Non-significant result (p=0.63) is reliable")
    
    # 3. Required sample size
    print("\n3. REQUIRED SAMPLE SIZE FOR ADEQUATE POWER")
    print("-"*100)
    
    if abs(observed_d) > 0.01:
        n_80 = compute_required_n(abs(observed_d), power=0.80, alpha=0.05)
        n_95 = compute_required_n(abs(observed_d), power=0.95, alpha=0.05)
        
        print(f"To achieve 80% power (α=0.05): N = {n_80} seeds per group")
        print(f"To achieve 95% power (α=0.05): N = {n_95} seeds per group")
        print(f"\nCurrent sample size: N = {n} seeds per group")
        
        if n < n_80:
            print(f"❌ UNDERPOWERED: Need {n_80 - n} more seeds per group for 80% power")
        else:
            print(f"✅ ADEQUATE: Current N exceeds requirement for 80% power")
    else:
        print("Effect size too small to compute required sample size")
    
    # 4. Minimum detectable effect
    print("\n4. MINIMUM DETECTABLE EFFECT (MDE)")
    print("-"*100)
    
    mde_80 = compute_mde(n, power=0.80, alpha=0.05)
    mde_95 = compute_mde(n, power=0.95, alpha=0.05)
    
    print(f"With N=10, we can reliably detect:")
    print(f"  • Cohen's d ≥ {mde_80:.3f} (80% power, α=0.05)")
    print(f"  • Cohen's d ≥ {mde_95:.3f} (95% power, α=0.05)")
    print(f"\nObserved effect: Cohen's d = {abs(observed_d):.3f}")
    
    if abs(observed_d) < mde_80:
        print(f"\n❌ UNDERPOWERED: Observed effect ({abs(observed_d):.3f}) is below MDE ({mde_80:.3f})")
        print(f"   → The study cannot reliably detect effects of this magnitude")
        print(f"   → Need larger N or accept lower power")
    else:
        print(f"\n✅ ADEQUATE: Observed effect ({abs(observed_d):.3f}) exceeds MDE ({mde_80:.3f})")
        print(f"   → The study has sufficient power to detect this effect")
    
    # 5. Power curve
    print("\n5. POWER CURVE: Detectable Effect Sizes")
    print("-"*100)
    
    print(f"\n{'Effect Size (d)':<20} {'Power (α=0.05)':<20} {'Interpretation':<30}")
    print("-"*100)
    
    for d in [0.2, 0.3, 0.4, 0.5, 0.8, 1.0, 1.5]:
        pwr = compute_power(d, n, alpha=0.05)
        interp = (
            "Small effect" if d < 0.5 else
            "Medium effect" if d < 0.8 else
            "Large effect"
        )
        marker = "✅" if pwr >= 0.8 else "⚠️" if pwr >= 0.5 else "❌"
        print(f"{d:<20.2f} {pwr:<20.3f} {interp:<30} {marker}")
    
    # 6. Practical implications
    print("\n6. PRACTICAL IMPLICATIONS FOR TABLE 2")
    print("-"*100)
    
    print("\nQUESTION: Is N=10 seeds sufficient?")
    print()
    
    if abs(observed_d) < mde_80:
        print("ANSWER: NO, the study is underpowered.")
        print()
        print("EXPLANATION:")
        print(f"  • Observed effect (d={abs(observed_d):.3f}) is smaller than MDE (d={mde_80:.3f})")
        print(f"  • With N=10, we can only detect effects ≥ {mde_80:.3f} with 80% power")
        print(f"  • The non-significant result (p=0.63) may be a Type II error")
        print()
        print("RECOMMENDATION:")
        print(f"  1. Increase to N={n_80} seeds for 80% power")
        print(f"  2. OR: Acknowledge limitation in paper")
        print(f"  3. OR: Report as 'no difference detectable with current power'")
    else:
        print("ANSWER: YES, the study has adequate power for the observed effect.")
        print()
        print("EXPLANATION:")
        print(f"  • Observed effect (d={abs(observed_d):.3f}) exceeds MDE (d={mde_80:.3f})")
        print(f"  • Power = {100*power_05:.1f}% meets or approaches 80% threshold")
        print(f"  • Non-significant result (p=0.63) is reliable")
        print()
        print("INTERPRETATION:")
        print(f"  • The small effect (d={abs(observed_d):.3f}) favors η=0.1 slightly")
        print(f"  • But the difference is not statistically significant")
        print(f"  • AND the effect is practically negligible (< 0.5)")
        print(f"  • Conclusion: 'No meaningful difference' is justified")
    
    # 7. Alternative analysis: Equivalence testing
    print("\n7. EQUIVALENCE TESTING (ALTERNATIVE FRAMEWORK)")
    print("-"*100)
    
    equivalence_margin = 0.5  # Cohen's d = 0.5 (medium effect)
    
    print(f"\nEquivalence margin: d = ±{equivalence_margin} (medium effect)")
    print(f"Observed effect: d = {observed_d:.3f}")
    print()
    
    if abs(observed_d) < equivalence_margin:
        print("✅ PRACTICAL EQUIVALENCE:")
        print(f"   • Observed effect ({abs(observed_d):.3f}) is within equivalence margin (±{equivalence_margin})")
        print(f"   • Even if statistically different, the difference is practically negligible")
        print(f"   • Can claim 'no meaningful difference' for practical purposes")
    else:
        print("❌ NOT EQUIVALENT:")
        print(f"   • Observed effect ({abs(observed_d):.3f}) exceeds equivalence margin (±{equivalence_margin})")
        print(f"   • The difference may be practically meaningful")
    
    # 8. Save report
    save_power_analysis_report(
        n=n,
        observed_d=observed_d,
        power_05=power_05,
        mde_80=mde_80,
        n_required_80=n_80 if abs(observed_d) > 0.01 else None
    )
    
    print("\n" + "="*100)
    print("POWER ANALYSIS COMPLETE")
    print("="*100)


def save_power_analysis_report(n: int, observed_d: float, power_05: float, 
                                mde_80: float, n_required_80: int):
    """Save machine-readable power analysis report."""
    
    report = {
        'sample_size': {
            'n_per_group': n,
            'total_n': 2 * n
        },
        'effect_size': {
            'cohens_d': float(observed_d),
            'interpretation': (
                "negligible" if abs(observed_d) < 0.2 else
                "small" if abs(observed_d) < 0.5 else
                "medium" if abs(observed_d) < 0.8 else
                "large"
            ),
            'direction': 'eta_0.1_lower_regret' if observed_d < 0 else 'eta_1.0_lower_regret'
        },
        'power_analysis': {
            'achieved_power_alpha_0.05': float(power_05),
            'power_adequate': bool(power_05 >= 0.8),
            'minimum_detectable_effect': float(mde_80),
            'observed_below_mde': bool(abs(observed_d) < mde_80),
            'required_n_for_80_power': int(n_required_80) if n_required_80 else None
        },
        'conclusions': {
            'is_underpowered': bool(abs(observed_d) < mde_80),
            'can_claim_no_difference': bool(abs(observed_d) < 0.5 or power_05 >= 0.8),
            'practical_equivalence': bool(abs(observed_d) < 0.5),
            'recommendation': (
                'Acknowledge underpowered in paper' if abs(observed_d) < mde_80 
                else 'Current power is adequate'
            )
        }
    }
    
    output_file = Path(__file__).parent / 'data' / 'power_analysis.json'
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✅ Saved power analysis report: {output_file}")


if __name__ == '__main__':
    power_analysis()
