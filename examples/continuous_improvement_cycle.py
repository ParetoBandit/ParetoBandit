#!/usr/bin/env python3
"""
Example: Basic Router Usage - Production BanditRouter

Demonstrates the production-grade BanditRouter with:
- HLE-based priors for optimal cold-start
- Multi-objective routing (quality/cost/latency)
- Clean API: route() + process_feedback()
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from bandit_gpt import BanditRouter


def demo_basic_routing():
    """Demonstrate basic routing with the production router."""
    print("=" * 70)
    print("PRODUCTION BANDITROUTER DEMO")
    print("=" * 70)
    
    # Create router with HLE priors (default)
    print("\n📊 Initializing router with HLE priors...")
    router = BanditRouter.create(priors="hle")
    
    print(f"✅ Router initialized with {len(router.registry)} models")
    print(f"   Using 24D features (23 PCA + 1 bias)")
    
    # Demo routing scenarios
    scenarios = [
        ("Write a Python function to sort a list", "coding task"),
        ("Solve the integral: ∫ x² dx from 0 to 1", "math/LaTeX task"),
        ("Write a short story about a robot", "creative task"),
        ("What is 2+2?", "simple query"),
    ]
    
    print("\n🎯 Routing examples:\n")
    
    for prompt, description in scenarios:
        model, log = router.route(prompt, profile="balanced")
        print(f"  {description:25s} → {model:30s}")
        
        # Simulate feedback (in production, this comes from actual user interaction)
        router.process_feedback(log.request_id, reward=0.8)
    
    print(f"\n✅ Demo complete!")
    print(f"   Total requests: {router.bandit.t}")
    print(f"\n💡 Router learned from feedback and will improve over time")


if __name__ == "__main__":
    demo_basic_routing()
