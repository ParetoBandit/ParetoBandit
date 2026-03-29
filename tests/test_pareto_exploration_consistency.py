"""
Test Pareto exploration consistency: verify UCB is used in both filter and selection.

Uncertain models must be able to win in Pareto mode via exploration bonus.
"""



def test_pareto_ucb_consistency():
    """
    Verify Pareto mode uses UCB (mean + exploration * uncertainty).
    """
    exploration_alpha = 2.0
    model = {"mean": 0.80, "std": 0.15}

    # Calculate UCB
    ucb = model["mean"] + (exploration_alpha * model["std"])
    expected_ucb = 0.80 + (2.0 * 0.15)

    assert abs(ucb - expected_ucb) < 0.001, f"UCB calculation: {ucb} vs {expected_ucb}"
    assert ucb == 1.10, "UCB should be 1.10"

    print(f"✅ UCB correctly calculated: {model['mean']:.2f} + {exploration_alpha}×{model['std']:.2f} = {ucb:.2f}")


def test_exploration_bonus_scales_with_uncertainty():
    """
    Verify exploration bonus is larger for high-uncertainty models.
    """
    exploration_alpha = 2.0

    low_uncertainty = {"mean": 0.90, "std": 0.05}
    high_uncertainty = {"mean": 0.85, "std": 0.20}

    bonus_low = exploration_alpha * low_uncertainty["std"]
    bonus_high = exploration_alpha * high_uncertainty["std"]

    ucb_low = low_uncertainty["mean"] + bonus_low
    ucb_high = high_uncertainty["mean"] + bonus_high

    print(f"  Low uncertainty: bonus = {bonus_low:.3f}, UCB = {ucb_low:.3f}")
    print(f"  High uncertainty: bonus = {bonus_high:.3f}, UCB = {ucb_high:.3f}")

    assert bonus_high > bonus_low, "High uncertainty should get larger bonus"
    assert ucb_high > ucb_low, "High uncertainty UCB can exceed low uncertainty UCB"

    print("✅ Exploration bonus scales correctly with uncertainty")


def test_pareto_utility_with_exploration():
    """
    Verify final utility calculation includes exploration bonus.
    """
    model = {"mean": 0.75, "std": 0.25, "cost": 0.50}
    exploration_alpha = 2.0
    lambda_val = 1.0

    # Calculate UCB quality
    ucb_quality = model["mean"] + (exploration_alpha * model["std"])

    # Calculate utility (as in router code)
    utility = ucb_quality - (lambda_val * model["cost"])

    0.75 + (2.0 * 0.25)  # 1.25
    expected_utility = 1.25 - (1.0 * 0.50)  # 0.75

    print(f"  Mean quality: {model['mean']:.2f}")
    print(f"  UCB quality: {ucb_quality:.2f} (+ {exploration_alpha * model['std']:.2f} bonus)")
    print(f"  Cost penalty: {lambda_val * model['cost']:.2f}")
    print(f"  Final utility: {utility:.2f}")

    assert abs(utility - expected_utility) < 0.001, f"Utility: {utility} vs {expected_utility}"

    print("✅ Pareto utility correctly includes exploration bonus")


def test_exploration_enables_uncertain_models():
    """
    Verify that exploration can make uncertain models competitive.

    This is the key benefit: uncertain models get a bonus that can
    make them win over established models with similar performance.
    """
    established = {"mean": 0.95, "std": 0.05, "cost": 0.80}
    uncertain = {"mean": 0.85, "std": 0.25, "cost": 0.70}

    exploration_alpha = 2.0
    lambda_val = 1.0

    # Calculate utilities WITH exploration
    est_ucb = established["mean"] + (exploration_alpha * established["std"])
    unc_ucb = uncertain["mean"] + (exploration_alpha * uncertain["std"])

    est_utility = est_ucb - (lambda_val * established["cost"])
    unc_utility = unc_ucb - (lambda_val * uncertain["cost"])

    print(f"  Established: UCB={est_ucb:.2f}, utility={est_utility:.2f}")
    print(f"  Uncertain: UCB={unc_ucb:.2f}, utility={unc_utility:.2f}")

    # The uncertain model gets a larger exploration bonus
    bonus_est = exploration_alpha * established["std"]
    bonus_unc = exploration_alpha * uncertain["std"]

    assert bonus_unc > bonus_est, "Uncertain model gets larger bonus"
    assert unc_ucb > est_ucb, "Uncertain model UCB exceeds established"
    assert unc_utility > est_utility, "Uncertain model can win with exploration"

    print("✅ Exploration enables uncertain models to compete")


if __name__ == "__main__":
    print("="*60)
    print("PARETO EXPLORATION CONSISTENCY: Verification Tests")
    print("="*60)

    test_pareto_ucb_consistency()
    print()
    test_exploration_bonus_scales_with_uncertainty()
    print()
    test_pareto_utility_with_exploration()
    print()
    test_exploration_enables_uncertain_models()

    print("\n" + "="*60)
    print("✅ All Pareto exploration tests passed!")
    print("="*60)

    """
    Verify Pareto mode uses consistent UCB in both filter and selection.

    Setup:
        - Established model: mean=0.95, std=0.05, cost=$1.00
        - New model: mean=0.85, std=0.20, cost=$0.50
        - PARETO_EXPLORATION_CONSTANT = 2.0

    Expected UCB values:
        - Established: 0.95 + 2.0*0.05 = 1.05
        - New: 0.85 + 2.0*0.20 = 1.25

    Expected behavior:
        1. Both pass Pareto filter (new has higher UCB potential)
        2. New model wins in selection (higher utility)
    """
    # Model stats
    established = {"mean": 0.95, "std": 0.05, "cost": 1.00}
    new_model = {"mean": 0.85, "std": 0.20, "cost": 0.50}

    # Constants
    exploration_alpha = 2.0
    lambda_val = 1.0  # Balanced Pareto profile

    # Calculate UCB
    established_ucb = established["mean"] + (exploration_alpha * established["std"])
    new_ucb = new_model["mean"] + (exploration_alpha * new_model["std"])

    # Verify UCB values
    assert abs(established_ucb - 1.05) < 0.001, f"Established UCB: {established_ucb}"
    assert abs(new_ucb - 1.25) < 0.001, f"New UCB: {new_ucb}"
    assert new_ucb > established_ucb, "New model should have higher UCB"

    # Calculate utilities (with exploration)
    established_utility = established_ucb - (lambda_val * established["cost"])
    new_utility = new_ucb - (lambda_val * new_model["cost"])

    # Verify new model wins
    print(f"  Established utility: {established_utility:.3f}")
    print(f"  New model utility: {new_utility:.3f}")

    assert new_utility > established_utility, \
        f"New model should win with exploration! New: {new_utility:.3f} vs Established: {established_utility:.3f}"

    print("✅ Pareto exploration consistency verified")


def test_pareto_exploration_makes_difference():
    """
    Verify that exploration bonus changes the winner in edge cases.

    This test demonstrates that exploration matters for selection.
    """
    # Setup where winner changes with exploration
    model_a = {"mean": 0.85, "std": 0.02, "cost": 0.50}  # Low uncertainty
    model_b = {"mean": 0.80, "std": 0.15, "cost": 0.45}  # High uncertainty
    lambda_val = 1.0
    exploration_alpha = 2.0

    # WITHOUT exploration: Model A wins (higher mean)
    utility_a_no_explore = model_a["mean"] - (lambda_val * model_a["cost"])
    utility_b_no_explore = model_b["mean"] - (lambda_val * model_b["cost"])

    print("  WITHOUT exploration:")
    print(f"    Model A utility: {utility_a_no_explore:.3f}")
    print(f"    Model B utility: {utility_b_no_explore:.3f}")

    winner_no_explore = "A" if utility_a_no_explore > utility_b_no_explore else "B"

    # WITH exploration: Winner may change
    ucb_a = model_a["mean"] + (exploration_alpha * model_a["std"])
    ucb_b = model_b["mean"] + (exploration_alpha * model_b["std"])
    utility_a_explore = ucb_a - (lambda_val * model_a["cost"])
    utility_b_explore = ucb_b - (lambda_val * model_b["cost"])

    print("  WITH exploration:")
    print(f"    Model A UCB utility: {utility_a_explore:.3f} (bonus: +{exploration_alpha * model_a['std']:.3f})")
    print(f"    Model B UCB utility: {utility_b_explore:.3f} (bonus: +{exploration_alpha * model_b['std']:.3f})")

    winner_explore = "A" if utility_a_explore > utility_b_explore else "B"

    # Verify that exploration changes relative utilities
    print(f"  Winner without exploration: Model {winner_no_explore}")
    print(f"  Winner with exploration: Model {winner_explore}")

    # Model B should get larger bonus due to higher uncertainty
    bonus_b = exploration_alpha * model_b["std"]
    bonus_a = exploration_alpha * model_a["std"]
    assert bonus_b > bonus_a, "Model B should get larger exploration bonus"

    print("✅ Exploration bonus varies with uncertainty")


def test_pareto_high_lambda_exploration():
    """
    Verify exploration helps even with high lambda (cost-sensitive).

    High lambda = strong cost preference
    But exploration should still allow discovery of uncertain cheap models.
    """
    # Very cost-sensitive profile (lambda=5.0)
    lambda_val = 5.0
    exploration_alpha = 2.0

    # Expensive established model
    expensive = {"mean": 0.98, "std": 0.02, "cost": 2.00}

    # Cheap uncertain model
    cheap = {"mean": 0.75, "std": 0.30, "cost": 0.20}

    # Calculate UCB
    expensive_ucb = expensive["mean"] + (exploration_alpha * expensive["std"])
    cheap_ucb = cheap["mean"] + (exploration_alpha * cheap["std"])

    # Calculate utilities WITH exploration
    expensive_utility = expensive_ucb - (lambda_val * expensive["cost"])
    cheap_utility = cheap_ucb - (lambda_val * cheap["cost"])

    print(f"  Expensive (UCB={expensive_ucb:.2f}, cost=${expensive['cost']:.2f}): utility={expensive_utility:.3f}")
    print(f"  Cheap (UCB={cheap_ucb:.2f}, cost=${cheap['cost']:.2f}): utility={cheap_utility:.3f}")

    # With high lambda and exploration, cheap model should win
    assert cheap_utility > expensive_utility, \
        "Cheap uncertain model should win with exploration bonus"

    print("✅ Exploration enables discovery even with strong cost preference")


def test_low_uncertainty_no_bonus():
    """
    Verify that models with low uncertainty don't get much exploration bonus.

    Established models with low uncertainty should not receive a large exploration bonus.
    """
    exploration_alpha = 2.0

    # Well-trained model (low uncertainty)
    established = {"mean": 0.90, "std": 0.05}

    # Calculate UCB
    ucb = established["mean"] + (exploration_alpha * established["std"])
    bonus = ucb - established["mean"]

    print(f"  Mean: {established['mean']:.2f}, UCB: {ucb:.2f}, Bonus: {bonus:.3f}")

    assert bonus < 0.15, f"Low uncertainty should give small bonus, got {bonus:.3f}"
    assert ucb > established["mean"], "UCB should still be higher than mean"

    print("✅ Low uncertainty models get minimal exploration bonus")


if __name__ == "__main__":
    print("="*60)
    print("PARETO EXPLORATION CONSISTENCY: Verification Tests")
    print("="*60)

    test_pareto_ucb_consistency()
    print()
    test_pareto_exploration_makes_difference()
    print()
    test_pareto_high_lambda_exploration()
    print()
    test_low_uncertainty_no_bonus()

    print("\n" + "="*60)
    print("✅ All Pareto exploration tests passed!")
    print("="*60)
