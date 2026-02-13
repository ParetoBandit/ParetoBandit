#!/usr/bin/env python3
"""
Unit tests for semantic transfer functionality.

Tests the extend_priors_with_semantic_transfer function that enables
cold-start initialization for new models by transferring knowledge from
semantically similar models.

Usage:
    python experiments_v1/04_figure/test_semantic_transfer.py
"""

import sys
from pathlib import Path

# Add project root and src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import numpy as np
import joblib
from bandit_gpt.config_legacy import DEFAULT_WARMUP_PRIORS_PATH


def extend_priors_with_semantic_transfer(
    warmup_priors: dict,
    new_models: list,
    transfer_mapping: dict,
    gamma: float = 0.05
) -> dict:
    """
    Extend warmup priors with new models via semantic transfer.
    
    Args:
        warmup_priors: Dictionary with 'A', 'b', 'models', 'context_dim'
        new_models: List of model IDs to add
        transfer_mapping: Dict mapping new_model -> source_model for transfer
        gamma: Scaling factor for transferred priors (default: 0.05)
    
    Returns:
        Extended priors dictionary with new models added
    """
    extended_priors = {
        'A': warmup_priors['A'].copy(),
        'b': warmup_priors['b'].copy(),
        'models': warmup_priors['models'].copy(),
        'context_dim': warmup_priors['context_dim']
    }
    
    for new_model in new_models:
        if new_model in extended_priors['models']:
            continue
        
        source_model = transfer_mapping.get(new_model)
        if not source_model or source_model not in warmup_priors['models']:
            raise ValueError(
                f"Cannot transfer priors for {new_model}: "
                f"source {source_model} not found in warmup priors"
            )
        
        # Transfer with gamma scaling
        extended_priors['A'][new_model] = gamma * warmup_priors['A'][source_model].copy()
        extended_priors['b'][new_model] = gamma * warmup_priors['b'][source_model].copy()
        extended_priors['models'].append(new_model)
        print(f"   ✅ Transferred priors: {source_model} → {new_model} (γ={gamma})")
    
    return extended_priors


def test_semantic_transfer_basic():
    """Test basic semantic transfer functionality."""
    print("\n" + "="*80)
    print("TEST 1: Basic Semantic Transfer")
    print("="*80)
    
    # Create mock warmup priors
    context_dim = 10
    mock_priors = {
        'A': {
            'model_a': np.eye(context_dim) * 2.0,
            'model_b': np.eye(context_dim) * 3.0,
        },
        'b': {
            'model_a': np.ones(context_dim) * 0.5,
            'model_b': np.ones(context_dim) * 0.7,
        },
        'models': ['model_a', 'model_b'],
        'context_dim': context_dim
    }
    
    # Transfer from model_a to model_c
    new_models = ['model_c']
    transfer_mapping = {'model_c': 'model_a'}
    gamma = 0.05
    
    print("\n📦 Input:")
    print(f"   Original models: {mock_priors['models']}")
    print(f"   New models: {new_models}")
    print(f"   Transfer mapping: {transfer_mapping}")
    print(f"   Gamma: {gamma}")
    
    extended = extend_priors_with_semantic_transfer(
        mock_priors, new_models, transfer_mapping, gamma
    )
    
    print("\n📊 Output:")
    print(f"   Extended models: {extended['models']}")
    
    # Verify
    checks_passed = 0
    checks_total = 0
    
    # Check 1: New model added
    checks_total += 1
    if 'model_c' in extended['models']:
        print("   ✅ New model added to model list")
        checks_passed += 1
    else:
        print("   ❌ New model not in model list")
    
    # Check 2: A matrix transferred correctly
    checks_total += 1
    expected_A = gamma * mock_priors['A']['model_a']
    if np.allclose(extended['A']['model_c'], expected_A):
        print("   ✅ A matrix transferred with correct scaling")
        checks_passed += 1
    else:
        print("   ❌ A matrix not transferred correctly")
    
    # Check 3: b vector transferred correctly
    checks_total += 1
    expected_b = gamma * mock_priors['b']['model_a']
    if np.allclose(extended['b']['model_c'], expected_b):
        print("   ✅ b vector transferred with correct scaling")
        checks_passed += 1
    else:
        print("   ❌ b vector not transferred correctly")
    
    # Check 4: Original models unchanged
    checks_total += 1
    if (np.allclose(extended['A']['model_a'], mock_priors['A']['model_a']) and
        np.allclose(extended['b']['model_a'], mock_priors['b']['model_a'])):
        print("   ✅ Original models unchanged")
        checks_passed += 1
    else:
        print("   ❌ Original models were modified")
    
    # Check 5: Context dim preserved
    checks_total += 1
    if extended['context_dim'] == context_dim:
        print("   ✅ Context dimension preserved")
        checks_passed += 1
    else:
        print("   ❌ Context dimension changed")
    
    print(f"\n   Checks passed: {checks_passed}/{checks_total}")
    return checks_passed == checks_total


def test_semantic_transfer_multiple_models():
    """Test transferring multiple models at once."""
    print("\n" + "="*80)
    print("TEST 2: Multiple Model Transfer")
    print("="*80)
    
    context_dim = 8
    mock_priors = {
        'A': {
            'model_a': np.eye(context_dim) * 2.0,
            'model_b': np.eye(context_dim) * 3.0,
        },
        'b': {
            'model_a': np.ones(context_dim) * 0.5,
            'model_b': np.ones(context_dim) * 0.7,
        },
        'models': ['model_a', 'model_b'],
        'context_dim': context_dim
    }
    
    # Transfer multiple models
    new_models = ['model_c', 'model_d']
    transfer_mapping = {
        'model_c': 'model_a',
        'model_d': 'model_b'
    }
    
    print("\n📦 Input:")
    print(f"   Original models: {mock_priors['models']}")
    print(f"   New models: {new_models}")
    
    extended = extend_priors_with_semantic_transfer(
        mock_priors, new_models, transfer_mapping, gamma=0.10
    )
    
    checks_passed = 0
    checks_total = 0
    
    # Check: Both models added
    checks_total += 1
    if 'model_c' in extended['models'] and 'model_d' in extended['models']:
        print("   ✅ Both models added")
        checks_passed += 1
    else:
        print("   ❌ Not all models added")
    
    # Check: Correct number of models
    checks_total += 1
    if len(extended['models']) == 4:
        print("   ✅ Correct number of models (4)")
        checks_passed += 1
    else:
        print(f"   ❌ Wrong number of models ({len(extended['models'])})")
    
    print(f"\n   Checks passed: {checks_passed}/{checks_total}")
    return checks_passed == checks_total


def test_semantic_transfer_error_handling():
    """Test error handling for invalid inputs."""
    print("\n" + "="*80)
    print("TEST 3: Error Handling")
    print("="*80)
    
    context_dim = 5
    mock_priors = {
        'A': {'model_a': np.eye(context_dim)},
        'b': {'model_a': np.ones(context_dim)},
        'models': ['model_a'],
        'context_dim': context_dim
    }
    
    checks_passed = 0
    checks_total = 0
    
    # Test 1: Missing source model
    checks_total += 1
    try:
        extend_priors_with_semantic_transfer(
            mock_priors,
            ['model_new'],
            {'model_new': 'model_nonexistent'},
            gamma=0.05
        )
        print("   ❌ Should have raised ValueError for missing source")
    except ValueError as e:
        print(f"   ✅ Correctly raised ValueError: {str(e)[:50]}...")
        checks_passed += 1
    
    # Test 2: Empty transfer mapping
    checks_total += 1
    try:
        extend_priors_with_semantic_transfer(
            mock_priors,
            ['model_new'],
            {},  # Empty mapping
            gamma=0.05
        )
        print("   ❌ Should have raised ValueError for empty mapping")
    except ValueError:
        print("   ✅ Correctly raised ValueError for empty mapping")
        checks_passed += 1
    
    print(f"\n   Checks passed: {checks_passed}/{checks_total}")
    return checks_passed == checks_total


def test_semantic_transfer_with_real_priors():
    """Test semantic transfer with actual warmup priors."""
    print("\n" + "="*80)
    print("TEST 4: Real Priors Integration")
    print("="*80)
    
    try:
        # Load actual warmup priors
        warmup_priors = joblib.load(DEFAULT_WARMUP_PRIORS_PATH)
        
        print("\n📦 Loaded warmup priors:")
        print(f"   Models: {warmup_priors['models']}")
        print(f"   Context dim: {warmup_priors['context_dim']}")
        
        # Simulate GPT-4o transfer from GPT-4-Turbo
        new_models = ['openai/gpt-4o']
        transfer_mapping = {'openai/gpt-4o': 'openai/gpt-4-turbo'}
        
        # Check if source model exists
        if 'openai/gpt-4-turbo' not in warmup_priors['models']:
            print("   ⚠️  GPT-4-Turbo not in warmup priors, skipping real test")
            return True
        
        extended = extend_priors_with_semantic_transfer(
            warmup_priors, new_models, transfer_mapping, gamma=0.05
        )
        
        checks_passed = 0
        checks_total = 0
        
        # Check: GPT-4o added
        checks_total += 1
        if 'openai/gpt-4o' in extended['models']:
            print("   ✅ GPT-4o added to model list")
            checks_passed += 1
        else:
            print("   ❌ GPT-4o not added")
        
        # Check: Matrix dimensions correct
        checks_total += 1
        expected_shape = warmup_priors['A']['openai/gpt-4-turbo'].shape
        actual_shape = extended['A']['openai/gpt-4o'].shape
        if actual_shape == expected_shape:
            print(f"   ✅ Matrix dimensions correct: {actual_shape}")
            checks_passed += 1
        else:
            print(f"   ❌ Wrong dimensions: {actual_shape} vs {expected_shape}")
        
        # Check: Priors are scaled down
        checks_total += 1
        source_norm = np.linalg.norm(warmup_priors['A']['openai/gpt-4-turbo'])
        transfer_norm = np.linalg.norm(extended['A']['openai/gpt-4o'])
        if transfer_norm < source_norm:
            print(f"   ✅ Priors scaled down (ratio: {transfer_norm/source_norm:.3f})")
            checks_passed += 1
        else:
            print("   ❌ Priors not scaled down")
        
        print(f"\n   Checks passed: {checks_passed}/{checks_total}")
        return checks_passed == checks_total
        
    except FileNotFoundError:
        print("   ⚠️  Warmup priors not found, skipping real test")
        return True


def test_semantic_transfer_idempotency():
    """Test that running transfer twice doesn't duplicate models."""
    print("\n" + "="*80)
    print("TEST 5: Idempotency")
    print("="*80)
    
    context_dim = 5
    mock_priors = {
        'A': {'model_a': np.eye(context_dim)},
        'b': {'model_a': np.ones(context_dim)},
        'models': ['model_a'],
        'context_dim': context_dim
    }
    
    new_models = ['model_b']
    transfer_mapping = {'model_b': 'model_a'}
    
    # First transfer
    extended1 = extend_priors_with_semantic_transfer(
        mock_priors, new_models, transfer_mapping, gamma=0.05
    )
    
    print("\n   First transfer:")
    print(f"   Models: {extended1['models']}")
    
    # Second transfer (should be idempotent)
    extended2 = extend_priors_with_semantic_transfer(
        extended1, new_models, transfer_mapping, gamma=0.05
    )
    
    print("\n   Second transfer:")
    print(f"   Models: {extended2['models']}")
    
    checks_passed = 0
    checks_total = 0
    
    # Check: No duplicate models
    checks_total += 1
    if len(extended2['models']) == len(set(extended2['models'])):
        print("   ✅ No duplicate models")
        checks_passed += 1
    else:
        print("   ❌ Duplicate models found")
    
    # Check: Still only 2 models
    checks_total += 1
    if len(extended2['models']) == 2:
        print("   ✅ Correct number of models (2)")
        checks_passed += 1
    else:
        print(f"   ❌ Wrong number of models ({len(extended2['models'])})")
    
    print(f"\n   Checks passed: {checks_passed}/{checks_total}")
    return checks_passed == checks_total


def run_all_tests():
    """Run all semantic transfer tests."""
    print("="*80)
    print("SEMANTIC TRANSFER UNIT TESTS")
    print("="*80)
    
    results = []
    
    results.append(("Basic Transfer", test_semantic_transfer_basic()))
    results.append(("Multiple Models", test_semantic_transfer_multiple_models()))
    results.append(("Error Handling", test_semantic_transfer_error_handling()))
    results.append(("Real Priors", test_semantic_transfer_with_real_priors()))
    results.append(("Idempotency", test_semantic_transfer_idempotency()))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} - {name}")
    
    print(f"\n   Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✅ All tests passed!")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
