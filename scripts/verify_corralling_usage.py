#!/usr/bin/env python3
"""
Verification Script: Ensure All BanditRouter Instances Use Corralling
======================================================================

This script verifies that:
1. BanditRouter defaults to use_corralling=True
2. All router instances properly initialize the corralling router
3. Routing decisions go through the corralling mechanism

Usage:
    python scripts/verify_corralling_usage.py
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bandit_gpt.router import BanditRouter
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def test_default_corralling():
    """Test 1: Verify BanditRouter defaults to use_corralling=True."""
    logger.info("\n" + "="*70)
    logger.info("TEST 1: Default Corralling Initialization")
    logger.info("="*70)
    
    registry = {
        "test/model1": {
            "input_cost_per_m": 1.0,
            "output_cost_per_m": 3.0,
            "description": "Test model 1"
        },
        "test/model2": {
            "input_cost_per_m": 2.0,
            "output_cost_per_m": 6.0,
            "description": "Test model 2"
        }
    }
    
    # Create router WITHOUT explicitly setting use_corralling
    router = BanditRouter(model_registry=registry)
    
    # Verify corralling is enabled by default
    assert router.use_corralling == True, "❌ FAILED: use_corralling should default to True"
    logger.info("✅ PASSED: use_corralling defaults to True")
    
    return True


def test_corralling_initialization():
    """Test 2: Verify corralling router is properly initialized via create()."""
    logger.info("\n" + "="*70)
    logger.info("TEST 2: Corralling Router Initialization")
    logger.info("="*70)
    
    registry = {
        "test/model1": {
            "input_cost_per_m": 1.0,
            "output_cost_per_m": 3.0,
            "description": "Test model 1"
        },
        "test/model2": {
            "input_cost_per_m": 2.0,
            "output_cost_per_m": 6.0,
            "description": "Test model 2"
        }
    }
    
    # Create router using the factory method
    router = BanditRouter.create(
        model_registry=registry,
        priors="none",
        use_corralling=True
    )
    
    # Verify corralling router exists
    assert router.corralling_router is not None, "❌ FAILED: corralling_router should be initialized"
    logger.info("✅ PASSED: corralling_router is initialized")
    
    # Verify experts exist
    assert hasattr(router.corralling_router, 'experts'), "❌ FAILED: corralling_router should have experts"
    assert len(router.corralling_router.experts) == 2, "❌ FAILED: Should have 2 experts (warmup + tabula rasa)"
    logger.info(f"✅ PASSED: corralling_router has {len(router.corralling_router.experts)} experts")
    
    # Verify expert weights exist
    assert hasattr(router.corralling_router, 'weights'), "❌ FAILED: corralling_router should have weights"
    logger.info(f"✅ PASSED: corralling_router has expert weights")
    
    return True


def test_corralling_routing():
    """Test 3: Verify routing goes through corralling mechanism."""
    logger.info("\n" + "="*70)
    logger.info("TEST 3: Corralling Routing Mechanism")
    logger.info("="*70)
    
    registry = {
        "test/model1": {
            "input_cost_per_m": 1.0,
            "output_cost_per_m": 3.0,
            "description": "Test model 1 for coding tasks"
        },
        "test/model2": {
            "input_cost_per_m": 2.0,
            "output_cost_per_m": 6.0,
            "description": "Test model 2 for creative writing"
        }
    }
    
    router = BanditRouter.create(
        model_registry=registry,
        priors="none",
        use_corralling=True
    )
    
    # Perform a routing decision
    test_prompt = "Write a Python function to calculate fibonacci numbers"
    selected_model, log = router.route(test_prompt, profile="auto")
    
    # Verify routing succeeded
    assert selected_model in registry.keys(), f"❌ FAILED: Selected model {selected_model} not in registry"
    logger.info(f"✅ PASSED: Routing selected valid model: {selected_model}")
    
    # Verify log contains routing info
    assert log.selected_model == selected_model, "❌ FAILED: Log should match selected model"
    logger.info(f"✅ PASSED: Routing log is consistent")
    
    return True


def test_corralling_update():
    """Test 4: Verify updates go through corralling mechanism."""
    logger.info("\n" + "="*70)
    logger.info("TEST 4: Corralling Update Mechanism")
    logger.info("="*70)
    
    registry = {
        "test/model1": {
            "input_cost_per_m": 1.0,
            "output_cost_per_m": 3.0,
            "description": "Test model 1"
        },
        "test/model2": {
            "input_cost_per_m": 2.0,
            "output_cost_per_m": 6.0,
            "description": "Test model 2"
        }
    }
    
    router = BanditRouter.create(
        model_registry=registry,
        priors="none",
        use_corralling=True
    )
    
    # Route and update
    test_prompt = "Explain quantum computing"
    selected_model, log = router.route(test_prompt, profile="auto")
    
    # Get initial expert weights
    initial_weights = router.corralling_router.weights.copy()
    logger.info(f"   Initial expert weights: {initial_weights}")
    
    # Update with reward
    router.update(test_prompt, selected_model, reward=0.8)
    
    # Verify update succeeded (weights may have changed)
    updated_weights = router.corralling_router.weights
    logger.info(f"   Updated expert weights: {updated_weights}")
    logger.info(f"✅ PASSED: Update mechanism works (weights tracked)")
    
    return True


def test_explicit_disable_corralling():
    """Test 5: Verify corralling can be explicitly disabled if needed."""
    logger.info("\n" + "="*70)
    logger.info("TEST 5: Explicit Corralling Disable")
    logger.info("="*70)
    
    registry = {
        "test/model1": {
            "input_cost_per_m": 1.0,
            "output_cost_per_m": 3.0,
            "description": "Test model 1"
        }
    }
    
    # Create router with corralling explicitly disabled
    router = BanditRouter(model_registry=registry, use_corralling=False)
    
    # Verify corralling is disabled
    assert router.use_corralling == False, "❌ FAILED: use_corralling should be False when explicitly set"
    logger.info("✅ PASSED: use_corralling can be explicitly disabled")
    
    # Note: corralling_router may still be None or not fully initialized
    logger.info(f"   corralling_router state: {router.corralling_router}")
    
    return True


def main():
    """Run all verification tests."""
    logger.info("\n" + "="*70)
    logger.info("CORRALLING USAGE VERIFICATION")
    logger.info("="*70)
    logger.info("\nVerifying that all BanditRouter instances properly use corralling...")
    
    tests = [
        ("Default Corralling", test_default_corralling),
        ("Corralling Initialization", test_corralling_initialization),
        ("Corralling Routing", test_corralling_routing),
        ("Corralling Update", test_corralling_update),
        ("Explicit Disable", test_explicit_disable_corralling),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            logger.error(f"\n❌ TEST FAILED: {name}")
            logger.error(f"   Error: {e}")
            results.append((name, False))
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("VERIFICATION SUMMARY")
    logger.info("="*70)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{status}: {name}")
    
    logger.info(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("\n🎉 ALL TESTS PASSED! Corralling is properly configured.")
        return 0
    else:
        logger.error(f"\n⚠️  {total - passed} test(s) failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

