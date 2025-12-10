"""
Quick test script for intent classifier.

Tests basic functionality and prints example classifications.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_jury.routing.intent_classifier import IntentClassifier, IntentCategory


def main():
    print("="*60)
    print("Intent Classifier Quick Test")
    print("="*60)
    
    # Initialize classifier
    print("\nInitializing classifier...")
    classifier = IntentClassifier()
    print("✓ Classifier initialized")
    
    # Test cases
    test_cases = [
        ("Solve for x: 2x + 5 = 13", IntentCategory.REASONING),
        ("Write a Python function to reverse a string", IntentCategory.CODING),
        ("What is the capital of France?", IntentCategory.FACTUAL_QA),
        ("Plan a 7-day trip to Japan including flights and hotels", IntentCategory.AGENTIC_EXECUTION),
        ("Hello, how are you?", IntentCategory.GENERAL),
    ]
    
    print("\n" + "="*60)
    print("Test Classifications")
    print("="*60)
    
    correct = 0
    total = len(test_cases)
    
    for prompt, expected in test_cases:
        result = classifier.classify(prompt)
        is_correct = result.category == expected
        correct += is_correct
        
        status = "✓" if is_correct else "✗"
        print(f"\n{status} Prompt: {prompt[:60]}")
        print(f"  Expected:  {expected.value}")
        print(f"  Predicted: {result.category.value} (confidence: {result.confidence:.2f})")
        if result.signals:
            print(f"  Signals:   {', '.join(result.signals[:2])}")
    
    print("\n" + "="*60)
    print(f"Accuracy: {correct}/{total} ({correct/total*100:.1f}%)")
    print("="*60)
    
    # Test batch classification
    print("\n" + "="*60)
    print("Batch Classification Test")
    print("="*60)
    
    prompts = [
        "Calculate the derivative of x^2",
        "Debug this JavaScript error",
        "Explain quantum mechanics",
        "Automate this workflow with multiple steps",
        "Tell me a joke",
    ]
    
    results = classifier.classify_batch(prompts)
    
    for prompt, result in zip(prompts, results):
        print(f"\n• {prompt}")
        print(f"  → {result.category.value} ({result.confidence:.2f})")
    
    print("\n" + "="*60)
    print("All tests completed! ✓")
    print("="*60)


if __name__ == '__main__':
    main()

