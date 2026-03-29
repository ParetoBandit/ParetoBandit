"""
Test Pareto lambda values to ensure reasonable utility ranges.
"""



def test_pareto_lambda_utility_ranges():
    """
    Verify Pareto lambda values produce reasonable utility ranges.

    With quality ∈ [0,1] and cost_penalty ∈ [0,1], utilities should:
    - Be positive for good models (quality > cost impact)
    - Not have extreme negative values
    - Reflect intended tradeoffs
    """
    lambdas = {
        'cost_saver': 1.0,
        'smart_shopper': 0.5,
        'rational_luxury': 0.05
    }

    print("\nPareto Utility Range Analysis:")
    print("="*60)

    for profile, lam in lambdas.items():
        # Worst case: quality=0, cost_penalty=1
        min_util = 0 - (lam * 1.0)

        # Best case: quality=1, cost_penalty=0
        max_util = 1 - (lam * 0.0)

        print(f"{profile:20s}: λ={lam:5.2f} → [{min_util:6.2f}, {max_util:6.2f}]")

        # Sanity checks
        assert min_util >= -2.0, f"{profile} min utility too negative: {min_util}"
        assert max_util == 1.0, f"{profile} max utility should be 1.0"
        assert min_util < max_util, f"{profile} range is inverted"

    print("\n✅ All Pareto lambda ranges are reasonable")


def test_pareto_lambda_example_scenarios():
    """
    Verify Pareto utilities for realistic scenarios.
    """
    lambdas = {
        'cost_saver': 1.0,
        'smart_shopper': 0.5,
        'rational_luxury': 0.05
    }

    scenarios = [
        ("Good + Cheap", 0.9, 0.2),      # High quality, low cost
        ("Good + Expensive", 0.9, 0.8),  # High quality, high cost
        ("Poor + Cheap", 0.3, 0.2),      # Low quality, low cost
        ("Average", 0.5, 0.5),           # Mid quality, mid cost
    ]

    print("\nPareto Utility Examples:")
    print("="*60)

    for scenario_name, quality, cost_penalty in scenarios:
        print(f"\n{scenario_name} (q={quality}, c={cost_penalty}):")
        for profile, lam in lambdas.items():
            utility = quality - (lam * cost_penalty)
            print(f"  {profile:20s}: {utility:6.2f}")

            # Good models should have positive utility in most profiles
            if quality >= 0.7:
                if profile == 'cost_saver' and cost_penalty > 0.8:
                    pass  # cost_saver may reject expensive models
                else:
                    assert utility > 0, \
                        f"{profile} should give positive utility to good model, got {utility}"

    print("\n✅ Pareto scenarios behave as expected")


def test_pareto_lambda_tradeoff_ratios():
    """
    Verify lambda values create intended quality-cost tradeoff ratios.
    """
    # For equal quality and cost deltas, utility change should match lambda
    quality_delta = 0.1
    cost_delta = 0.1

    lambdas = {
        'cost_saver': 1.0,        # Equal: Δq = Δc → Δu = 0
        'smart_shopper': 0.5,     # Quality-biased: Δq > 0.5*Δc
        'rational_luxury': 0.05   # Quality-focused: Δq >> 0.05*Δc
    }

    print("\nPareto Tradeoff Analysis (Δq=0.1, Δc=0.1):")
    print("="*60)

    for profile, lam in lambdas.items():
        # Utility change from quality improvement
        util_from_quality = quality_delta

        # Utility change from cost increase (negative)
        util_from_cost = -(lam * cost_delta)

        # Net utility change
        net_change = util_from_quality + util_from_cost

        print(f"{profile:20s}: Δu = {net_change:+.3f} (quality {util_from_quality:+.1f}, cost {util_from_cost:+.2f})")

        # Verify tradeoff characteristics
        if profile == 'cost_saver':
            assert abs(net_change) < 0.01, "cost_saver should be 50/50 balanced"
        elif profile == 'smart_shopper':
            assert net_change > 0, "smart_shopper should favor quality"
        elif profile == 'rational_luxury':
            assert net_change > 0.09, "rational_luxury should strongly favor quality"

    print("\n✅ Pareto tradeoffs match intended ratios")


if __name__ == "__main__":
    test_pareto_lambda_utility_ranges()
    test_pareto_lambda_example_scenarios()
    test_pareto_lambda_tradeoff_ratios()

    print("\n" + "="*60)
    print("✅ All Pareto lambda tests passed!")
    print("="*60)
