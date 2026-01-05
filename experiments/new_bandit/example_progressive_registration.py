#!/usr/bin/env python3
"""
Progressive Registration API - Usage Example

Demonstrates how to register models with varying levels of knowledge.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from banditgpt.router import BanditRouter

def main():
    print("\n" + "="*70)
    print("PROGRESSIVE REGISTRATION API - USAGE EXAMPLE")
    print("="*70)
    
    # Create a router with no pre-registered models
    router = BanditRouter.create(
        model_registry={},
        priors="none",
        context_encoder=None
    )
    
    print("\n--- Tier A: Archetypes (I know the model's intent) ---\n")
    
    # Register a coding specialist
    router.register_model(
        "deepseek-coder-v2",
        capabilities=["coding"],
        speed="slow",
        cost_usd=2.0,
        latency_s=4.5
    )
    
    # Register a math specialist
    router.register_model(
        "qwen-math-72b",
        capabilities=["math", "reasoning"],
        speed="slow",
        cost_usd=3.0,
        latency_s=5.0
    )
    
    print("\n--- Tier B: T-Shirt Sizing (I know cost/speed, not HLE) ---\n")
    
    # Fast, cheap model
    router.register_model(
        "llama-3-8b-instruct",
        speed="fast",
        capabilities=["general"],
        cost_usd=0.08,
        latency_s=0.4
    )
    
    # Medium model
    router.register_model(
        "llama-3-70b-instruct",
        speed="balanced",
        capabilities=["general", "reasoning"],
        cost_usd=0.6,
        latency_s=1.2
    )
    
    print("\n--- Tier C: Agnostic (I have no information) ---\n")
    
    # Mystery model - let the bandit figure it out
    router.register_model(
        "mysterious-model-x",
        speed="balanced"
    )
    
    print("\n--- Power User: Explicit Weights ---\n")
    
    # Expert user with benchmark data
    router.register_model(
        "custom-finetuned-model",
        speed="fast",
        cost_usd=0.15,
        latency_s=0.6,
        initial_weights={
            "anchor_coding": 4.0,      # Excellent at coding (benchmarked)
            "anchor_math": 2.5,        # Good at math
            "anchor_creative": -1.0,   # Weak at creative writing
            "complexity_score": 1.5    # Handles moderate complexity
        }
    )
    
    print("\n" + "="*70)
    print("SUMMARY: Registered Models")
    print("="*70)
    print(f"\nTotal models registered: {len(router.bandit.models)}")
    print("\nModels:")
    for i, model_id in enumerate(router.bandit.models, 1):
        meta = router.registry.get(model_id, {})
        cost = meta.get("cost_per_1m_tokens", 0)
        latency = meta.get("median_latency_s", 0)
        caps = meta.get("capabilities", [])
        speed = meta.get("speed_profile", "unknown")
        
        print(f"\n{i}. {model_id}")
        print(f"   Cost: ${cost:.2f}/1M | Latency: {latency:.2f}s | Speed: {speed}")
        if caps:
            print(f"   Capabilities: {', '.join(caps)}")
    
    print("\n" + "="*70)
    print("✅ All models successfully registered!")
    print("="*70 + "\n")
    
    # Optional: Show how routing would work
    print("\n--- Test Routing (Optional) ---\n")
    coding_prompt = "Write a Python function to find the longest palindromic substring."
    
    try:
        result = router.route(coding_prompt)
        print(f"Coding prompt routed to: {result['model']}")
        print(f"Predicted utility: {result['predicted_utility']:.3f}")
    except Exception as e:
        print(f"(Routing skipped - encoder not fully initialized: {e})")

if __name__ == "__main__":
    main()
