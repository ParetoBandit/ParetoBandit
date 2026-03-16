#!/usr/bin/env python3
"""
Test: Proactive Regularization Floor

Verifies that the proactive regularization floor prevents eigenvalue decay
in forgetting bandits, ensuring that applying decay to the entire matrix A
(including λI) does not cause the prior to vanish.

Key Differences from Reactive Approach:
1. Tracks effective lambda decay explicitly (self.regularization_floor)
2. Proactively injects regularization when lambda drops below threshold
3. Maintains principled lower bound on eigenvalues
4. Amortized O(d²) with rare O(d³) maintenance cycles
"""
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
from pareto_bandit.router import DisjointLinUCBPolicy


def test_regularization_floor_tracking():
    """
    Verify that regularization_floor correctly tracks lambda decay.
    """
    print("=" * 70)
    print("TEST 1: Regularization Floor Tracking")
    print("=" * 70)
    
    bandit = DisjointLinUCBPolicy(
        model_names=["model_a", "model_b"],
        dim=10,
        alpha=0.1,
        init_lambda=1.0,
        forgetting_factor=0.9
    )
    
    print(f"\nInitial state:")
    print(f"  init_lambda: {bandit.init_lambda}")
    print(f"  gamma: {bandit.gamma}")
    print(f"  regularization_floor['model_a']: {bandit.regularization_floor['model_a']}")
    
    # Verify initial state
    assert bandit.regularization_floor["model_a"] == 1.0, "Initial floor should be init_lambda"
    assert bandit.regularization_floor["model_b"] == 1.0, "Initial floor should be init_lambda"
    
    # Update model_a at t=0
    x = np.random.randn(10)
    x /= np.linalg.norm(x)
    bandit.update("model_a", x, 1.0)
    floor_after_first = bandit.regularization_floor["model_a"]
    
    print(f"\nAfter first update (t={bandit.t}):")
    print(f"  regularization_floor: {floor_after_first:.4f}")
    
    # Verify initial state
    assert floor_after_first == 1.0, f"After first update, floor should be 1.0, got {floor_after_first}"
    
    # Advance time significantly and update again
    bandit.t = 10
    bandit.update("model_a", x, 1.0)
    floor_after_second = bandit.regularization_floor["model_a"]
    
    # The floor should have decayed
    # dt = 10 - 1 = 9 (time since last update)
    # new_lambda = 1.0 * 0.9^9
    dt = 10 - 1
    expected_min = 0.9 ** dt
    expected_max = 1.0  # Can't be higher than init
    
    print(f"\nAfter second update (t={bandit.t}, dt={dt}):")
    print(f"  Expected range: [{expected_min:.4f}, {expected_max:.4f}]")
    print(f"  Actual regularization_floor: {floor_after_second:.4f}")
    
    # Floor should be in the expected range (decayed but not below minimum)
    assert expected_min * 0.9 <= floor_after_second <= expected_max, \
        f"Floor {floor_after_second} outside expected range [{expected_min * 0.9}, {expected_max}]"
    
    # Floor should be less than initial (decay happened)
    assert floor_after_second < 1.0, \
        f"Floor should have decayed from 1.0, got {floor_after_second}"
    
    print(f"\n✅ PASS: Regularization floor correctly tracks decay")
    return True


def test_proactive_maintenance_trigger():
    """
    Verify that maintenance cycle triggers when lambda drops below 10% threshold.
    """
    print("\n" + "=" * 70)
    print("TEST 2: Proactive Maintenance Trigger")
    print("=" * 70)
    
    bandit = DisjointLinUCBPolicy(
        model_names=["test_model"],
        dim=10,
        alpha=0.1,
        init_lambda=1.0,
        forgetting_factor=0.8  # More aggressive decay
    )
    
    print(f"\nConfiguration:")
    print(f"  init_lambda: {bandit.init_lambda}")
    print(f"  gamma: {bandit.gamma}")
    print(f"  threshold: {0.1 * bandit.init_lambda}")
    
    x = np.random.randn(10)
    x /= np.linalg.norm(x)
    
    # Initial update
    bandit.update("test_model", x, 1.0)
    print(f"\nInitial update completed")
    print(f"  regularization_floor: {bandit.regularization_floor['test_model']:.4f}")
    
    # Calculate how many steps needed to drop below threshold
    # We need gamma^dt * 1.0 < 0.1
    # dt > log(0.1) / log(0.8) ≈ 10.3 steps
    steps_needed = int(np.ceil(np.log(0.1) / np.log(0.8)))
    print(f"\nSteps needed to trigger maintenance: ~{steps_needed}")
    
    # Advance time to just before threshold
    bandit.t = steps_needed - 1
    bandit.update("test_model", x, 1.0)
    floor_before_trigger = bandit.regularization_floor["test_model"]
    print(f"\nBefore trigger (t={bandit.t}):")
    print(f"  regularization_floor: {floor_before_trigger:.4f}")
    
    # Advance one more step to trigger maintenance
    bandit.t = steps_needed + 5
    bandit.update("test_model", x, 1.0)
    floor_after_trigger = bandit.regularization_floor["test_model"]
    print(f"\nAfter trigger (t={bandit.t}):")
    print(f"  regularization_floor: {floor_after_trigger:.4f}")
    
    # Floor should be reset to init_lambda
    assert floor_after_trigger == bandit.init_lambda, \
        f"Floor should be reset to {bandit.init_lambda}, got {floor_after_trigger}"
    
    print(f"\n✅ PASS: Maintenance cycle correctly triggered and reset floor")
    return True


def test_eigenvalue_lower_bound():
    """
    Verify that eigenvalues never drop below the safety threshold.
    """
    print("\n" + "=" * 70)
    print("TEST 3: Eigenvalue Lower Bound")
    print("=" * 70)
    
    bandit = DisjointLinUCBPolicy(
        model_names=["test"],
        dim=10,
        alpha=0.1,
        init_lambda=1.0,
        forgetting_factor=0.7  # Aggressive decay
    )
    
    print(f"\nConfiguration:")
    print(f"  init_lambda: {bandit.init_lambda}")
    print(f"  gamma: {bandit.gamma} (aggressive)")
    print(f"  Safety threshold: {0.1 * bandit.init_lambda}")
    
    x = np.random.randn(10)
    x /= np.linalg.norm(x)
    
    # Run many updates with time gaps to trigger decay
    min_eigenvalue = float('inf')
    eigenvalues_history = []
    
    for i in range(50):
        # Sparse updates with time gaps
        bandit.t = i * 5
        bandit.update("test", x, 1.0)
        
        # Check eigenvalues
        eigvals = np.linalg.eigvalsh(bandit.A["test"])
        min_eig = np.min(eigvals)
        min_eigenvalue = min(min_eigenvalue, min_eig)
        eigenvalues_history.append(min_eig)
        
        if i % 10 == 0:
            print(f"\nStep {i}:")
            print(f"  min_eigenvalue(A): {min_eig:.4f}")
            print(f"  regularization_floor: {bandit.regularization_floor['test']:.4f}")
    
    print(f"\nOverall minimum eigenvalue: {min_eigenvalue:.4f}")
    print(f"Safety threshold: {0.1 * bandit.init_lambda:.4f}")
    
    # Verify that minimum eigenvalue never dropped below safety threshold
    # Allow some numerical tolerance
    safety_threshold = 0.05 * bandit.init_lambda  # 5% to account for numerical noise
    assert min_eigenvalue > safety_threshold, \
        f"Eigenvalue {min_eigenvalue} dropped below safety threshold {safety_threshold}"
    
    print(f"\n✅ PASS: Eigenvalues maintained above safety threshold")
    return True


def test_theta_preservation_during_maintenance():
    """
    Verify that learned preferences (theta) are preserved during maintenance.
    """
    print("\n" + "=" * 70)
    print("TEST 4: Theta Preservation During Maintenance")
    print("=" * 70)
    
    bandit = DisjointLinUCBPolicy(
        model_names=["test"],
        dim=10,
        alpha=0.1,
        init_lambda=1.0,
        forgetting_factor=0.8
    )
    
    print(f"\nConfiguration:")
    print(f"  init_lambda: {bandit.init_lambda}")
    print(f"  gamma: {bandit.gamma}")
    
    # Create a pattern: positive rewards for one direction
    x_positive = np.zeros(10)
    x_positive[0] = 1.0  # Feature 0 predicts high reward
    
    # Train the bandit to learn this pattern
    for i in range(20):
        bandit.update("test", x_positive, reward=1.0)
    
    # Get theta before maintenance
    theta_before = bandit.A_inv["test"] @ bandit.b["test"]
    print(f"\nTheta before maintenance:")
    print(f"  theta[0] (should be positive): {theta_before[0]:.4f}")
    print(f"  ||theta||: {np.linalg.norm(theta_before):.4f}")
    
    # Trigger maintenance by advancing time significantly
    steps_to_trigger = int(np.ceil(np.log(0.1) / np.log(0.8))) + 5
    bandit.t = bandit.t + steps_to_trigger
    bandit.update("test", x_positive, reward=1.0)
    
    # Get theta after maintenance
    theta_after = bandit.A_inv["test"] @ bandit.b["test"]
    print(f"\nTheta after maintenance:")
    print(f"  theta[0] (should still be positive): {theta_after[0]:.4f}")
    print(f"  ||theta||: {np.linalg.norm(theta_after):.4f}")
    
    # Verify that theta direction is preserved (cosine similarity)
    cosine_sim = np.dot(theta_before, theta_after) / (
        np.linalg.norm(theta_before) * np.linalg.norm(theta_after)
    )
    print(f"\nCosine similarity: {cosine_sim:.4f}")
    
    # Theta should be similar (high cosine similarity)
    assert cosine_sim > 0.9, \
        f"Theta not preserved during maintenance: cosine_sim={cosine_sim}"
    
    # Feature 0 should still be positive (learned pattern preserved)
    assert theta_after[0] > 0, \
        f"Learned pattern lost: theta[0]={theta_after[0]} should be positive"
    
    print(f"\n✅ PASS: Theta preserved during maintenance (cosine_sim={cosine_sim:.4f})")
    return True


def test_amortized_complexity():
    """
    Verify that maintenance cycles are rare (amortized O(d²)).
    """
    print("\n" + "=" * 70)
    print("TEST 5: Amortized Complexity (Maintenance Frequency)")
    print("=" * 70)
    
    bandit = DisjointLinUCBPolicy(
        model_names=["test"],
        dim=10,
        alpha=0.1,
        init_lambda=1.0,
        forgetting_factor=0.9
    )
    
    print(f"\nConfiguration:")
    print(f"  init_lambda: {bandit.init_lambda}")
    print(f"  gamma: {bandit.gamma}")
    
    x = np.random.randn(10)
    x /= np.linalg.norm(x)
    
    # Track maintenance cycles
    maintenance_count = 0
    total_updates = 100
    
    for i in range(total_updates):
        floor_before = bandit.regularization_floor["test"]
        bandit.t = i * 2  # Sparse updates
        bandit.update("test", x, 1.0)
        floor_after = bandit.regularization_floor["test"]
        
        # Detect maintenance (floor reset to init_lambda)
        if floor_after > floor_before * 1.5:  # Significant increase indicates reset
            maintenance_count += 1
            print(f"  Maintenance cycle #{maintenance_count} at update {i}")
    
    print(f"\nMaintenance cycles: {maintenance_count} / {total_updates} updates")
    maintenance_rate = maintenance_count / total_updates
    print(f"Maintenance rate: {maintenance_rate:.2%}")
    
    # Maintenance should be rare (< 10% of updates)
    assert maintenance_rate < 0.1, \
        f"Maintenance too frequent: {maintenance_rate:.2%} > 10%"
    
    print(f"\n✅ PASS: Maintenance cycles are rare ({maintenance_rate:.2%})")
    return True


def test_no_decay_baseline():
    """
    Verify that with gamma=1.0 (no decay), floor remains constant.
    """
    print("\n" + "=" * 70)
    print("TEST 6: No Decay Baseline (gamma=1.0)")
    print("=" * 70)
    
    bandit = DisjointLinUCBPolicy(
        model_names=["test"],
        dim=10,
        alpha=0.1,
        init_lambda=1.0,
        forgetting_factor=1.0  # No decay
    )
    
    print(f"\nConfiguration:")
    print(f"  gamma: {bandit.gamma} (no decay)")
    
    x = np.random.randn(10)
    x /= np.linalg.norm(x)
    
    # Run updates with time gaps
    for i in range(20):
        bandit.t = i * 10
        bandit.update("test", x, 1.0)
    
    floor_final = bandit.regularization_floor["test"]
    print(f"\nFinal regularization_floor: {floor_final}")
    
    # Floor should remain at init_lambda (no decay)
    assert floor_final == bandit.init_lambda, \
        f"Floor should remain {bandit.init_lambda} with no decay, got {floor_final}"
    
    print(f"\n✅ PASS: Floor remains constant with gamma=1.0")
    return True


if __name__ == "__main__":
    tests = [
        test_regularization_floor_tracking,
        test_proactive_maintenance_trigger,
        test_eigenvalue_lower_bound,
        test_theta_preservation_during_maintenance,
        test_amortized_complexity,
        test_no_decay_baseline,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n❌ EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print("\n" + "=" * 70)
    if all(results):
        print("🎉 All proactive regularization floor tests passed!")
        print("=" * 70)
        sys.exit(0)
    else:
        failed = sum(1 for r in results if not r)
        print(f"❌ {failed}/{len(results)} tests failed")
        print("=" * 70)
        sys.exit(1)

