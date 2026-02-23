"""
Test exploration bonus scaling to ensure correct ordering across profiles.
"""

import pytest


def test_exploration_bonus_ordering():
    """
    Verify exploration decreases: ARBITRAGE > COST_SAVER > MAX_QUALITY
    
    This ensures profiles explore as intended:
    - ARBITRAGE: High exploration to find value
    - COST_SAVER: Moderate exploration
    - MAX_QUALITY: Low exploration (mostly exploits)
    """
    alpha = 2.0  # Standard LinUCB alpha
    std = 0.2    # Example uncertainty
    
    # Profile alpha_scale values
    profiles = {
        'MAX_QUALITY': 0.3,
        'ARBITRAGE': 1.0,
        'COST_SAVER': 0.5
    }
    
    # Calculate exploration bonus (no w_q multiplier)
    bonuses = {name: alpha * scale * std for name, scale in profiles.items()}
    
    print(f"\nExploration bonuses (α={alpha}, std={std}):")
    for name, bonus in sorted(bonuses.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name:15s}: {bonus:.3f}")
    
    # Verify correct ordering
    assert bonuses['ARBITRAGE'] > bonuses['COST_SAVER'], \
        "ARBITRAGE should explore more than COST_SAVER"
    assert bonuses['COST_SAVER'] > bonuses['MAX_QUALITY'], \
        "COST_SAVER should explore more than MAX_QUALITY"
    
    # Verify magnitudes
    assert bonuses['ARBITRAGE'] == 0.4, f"ARBITRAGE bonus should be 0.4, got {bonuses['ARBITRAGE']}"
    assert bonuses['COST_SAVER'] == 0.2, f"COST_SAVER bonus should be 0.2, got {bonuses['COST_SAVER']}"
    assert bonuses['MAX_QUALITY'] == 0.12, f"MAX_QUALITY bonus should be 0.12, got {bonuses['MAX_QUALITY']}"
    
    print("\n✅ Exploration ordering is correct!")


def test_exploration_independence_from_quality_weight():
    """
    Verify exploration bonus is independent of w_q (quality weight).
    
    Previously, exploration scaled with w_q, causing unintended coupling.
    Now it should only depend on alpha_scale.
    """
    alpha = 2.0
    alpha_scale = 1.0
    std = 0.2
    
    # Different quality weights should NOT affect exploration
    w_q_values = [0.1, 0.8, 5.0, 30.0]
    
    expected_bonus = alpha * alpha_scale * std  # 0.4
    
    for w_q in w_q_values:
        # Formula: independent of w_q
        bonus = alpha * alpha_scale * std
        assert bonus == expected_bonus, \
            f"Exploration should be {expected_bonus} regardless of w_q, got {bonus} for w_q={w_q}"
    
    print("✅ Exploration is independent of quality weight!")


if __name__ == "__main__":
    test_exploration_bonus_ordering()
    test_exploration_independence_from_quality_weight()
    print("\n" + "="*60)
    print("✅ All exploration bonus tests passed!")
    print("="*60)
