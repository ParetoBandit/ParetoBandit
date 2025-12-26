#!/usr/bin/env python3
"""
Test that cluster_boost_weight parameter is configurable
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from bandit import BanditRouter

def test_configurable_boost():
    print("Testing configurable cluster_boost_weight parameter\n")
    
    # Test 1: Default value
    router1 = BanditRouter.create(priors="benchmark")
    print(f"✓ Default boost weight: {router1.cluster_boost_weight}")
    assert router1.cluster_boost_weight == 0.1, "Default should be 0.1"
    
    # Test 2: Custom value - conservative
    router2 = BanditRouter.create(priors="benchmark", cluster_boost_weight=0.05)
    print(f"✓ Conservative boost weight: {router2.cluster_boost_weight}")
    assert router2.cluster_boost_weight == 0.05
    
    # Test 3: Custom value - aggressive
    router3 = BanditRouter.create(priors="benchmark", cluster_boost_weight=0.3)
    print(f"✓ Aggressive boost weight: {router3.cluster_boost_weight}")
    assert router3.cluster_boost_weight == 0.3
    
    # Test 4: Disabled
    router4 = BanditRouter.create(priors="benchmark", cluster_boost_weight=0.0)
    print(f"✓ Disabled boost weight: {router4.cluster_boost_weight}")
    assert router4.cluster_boost_weight == 0.0
    
    print("\n✅ All parameter tests passed!")
    
    # Show example impact
    print("\n📊 Impact of different boost weights on z=1.0:")
    for weight in [0.0, 0.05, 0.1, 0.2, 0.5]:
        base = 0.85
        boosted = base * (1 + 1.0 * weight)
        print(f"  weight={weight:.2f}: {base:.3f} → {boosted:.3f} ({(boosted-base)/base*100:+.1f}%)")

if __name__ == "__main__":
    test_configurable_boost()
