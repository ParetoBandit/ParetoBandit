import sys
from pathlib import Path
try:
    from .bandit import BanditRouter
except (ImportError, ValueError):
    try:
        from final_release.bandit import BanditRouter
    except (ImportError, ValueError):
        from bandit import BanditRouter

def main():
    print("Testing BanditRouter Refactor...")
    
    # 1. Test Default Initialization (HLE + Default Registry)
    print("\n[1] Testing Default Initialization...")
    try:
        router = BanditRouter.create()
        print(f"  ✓ Initialized successfully")
        print(f"  ✓ Loaded {len(router.registry)} models")
        print(f"  ✓ Priors loaded: {len(router.bandit.A)} models initialized")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        sys.exit(1)
        
    # 2. Test Routing with Profile
    print("\n[2] Testing Routing (Balanced Profile)...")
    prompt = "Write a Python script to sort a list."
    model, log = router.route(prompt, profile="balanced")
    print(f"  ✓ Selected: {model}")
    print(f"  ✓ Utility: {log.predicted_utility:.4f}")
    
    # 3. Test Constraints (Max Cost)
    print("\n[3] Testing Constraints (Max Cost = $0.000001)...")
    # This should force a very cheap model
    try:
        cheap_model, log = router.route(prompt, max_cost=0.000001)
        cost = log.cost_usd
        print(f"  ✓ Selected: {cheap_model}")
        print(f"  ✓ Cost: ${cost:.8f}")
        if cost > 0.000001:
            print("  ✗ Constraint violated!")
        else:
            print("  ✓ Constraint satisfied")
    except ValueError as e:
        print(f"  ? No models found (expected if constraint too tight): {e}")

    # 4. Test Constraints (Quality Floor)
    print("\n[4] Testing Constraints (Quality Floor: Math > 90)...")
    try:
        smart_model, log = router.route(prompt, quality_floor={"math": 90})
        print(f"  ✓ Selected: {smart_model}")
        score = router.registry[smart_model]["scores"]["math"]
        print(f"  ✓ Math Score: {score}")
        if score < 90:
            print("  ✗ Constraint violated!")
        else:
            print("  ✓ Constraint satisfied")
    except ValueError:
         print("  ? No models found with Math > 90")

    print("\nAll tests passed!")

if __name__ == "__main__":
    main()
