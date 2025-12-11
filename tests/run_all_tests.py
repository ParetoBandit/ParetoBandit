#!/usr/bin/env python3
"""
Run all unit tests for llm_jury intent classification module.

Usage:
    python tests/run_all_tests.py
    python tests/run_all_tests.py --verbose
"""

import sys
import unittest
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import test modules
from tests import test_length_debiasing, test_intent_classifier


def run_all_tests(verbose=True):
    """
    Run all test suites.
    
    Returns:
        bool: True if all tests passed
    """
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test modules
    suite.addTests(loader.loadTestsFromModule(test_length_debiasing))
    suite.addTests(loader.loadTestsFromModule(test_intent_classifier))
    
    # Run with appropriate verbosity
    verbosity = 2 if verbose else 1
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*70)
    
    if result.wasSuccessful():
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    verbose = '--verbose' in sys.argv or '-v' in sys.argv
    success = run_all_tests(verbose=verbose)
    sys.exit(0 if success else 1)
