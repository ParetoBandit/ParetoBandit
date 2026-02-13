#!/usr/bin/env python3
"""
Comprehensive test suite for Figure 4 experiments.

Runs all unit tests to validate:
1. Corralling implementation
2. Semantic transfer functionality
3. Optimized hyperparameter configuration
4. Analysis scripts

Usage:
    python experiments_v1/04_figure/run_all_tests.py
"""

import sys
import subprocess
from pathlib import Path


def run_test(test_script, description):
    """Run a test script and return success status."""
    print("\n" + "="*80)
    print(f"Running: {description}")
    print("="*80)
    
    result = subprocess.run(
        [sys.executable, test_script],
        capture_output=False,
        text=True
    )
    
    success = result.returncode == 0
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"\n{status}: {description}\n")
    
    return success


def main():
    """Run all tests."""
    print("="*80)
    print("FIGURE 4 COMPREHENSIVE TEST SUITE")
    print("="*80)
    print("\nThis suite validates all components of the Figure 4 experiment:")
    print("  1. Core Corralling implementation")
    print("  2. Semantic transfer for new models")
    print("  3. Optimized hyperparameter configuration")
    
    test_dir = Path(__file__).parent
    
    tests = [
        (test_dir / "test_corralling.py", "Core Corralling Implementation"),
        (test_dir / "test_semantic_transfer.py", "Semantic Transfer Functionality"),
        (test_dir / "test_optimized_config.py", "Optimized Configuration"),
    ]
    
    results = []
    for test_script, description in tests:
        if not test_script.exists():
            print(f"\n⚠️  Test not found: {test_script}")
            results.append((description, False))
            continue
        
        success = run_test(test_script, description)
        results.append((description, success))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUITE SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for description, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status} - {description}")
    
    print(f"\n  Total: {passed}/{total} test suites passed")
    
    if passed == total:
        print("\n✅ All test suites passed! Implementation is correct.")
        print("\nThe following have been validated:")
        print("  • Corralling algorithm with importance-weighted loss")
        print("  • Expert weight updates and adaptation")
        print("  • Semantic transfer for cold-start models")
        print("  • Optimized hyperparameters (η=5.0, γ=0.10)")
        print("  • Multi-model routing (3 models)")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test suite(s) failed.")
        print("Review the output above for details.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
