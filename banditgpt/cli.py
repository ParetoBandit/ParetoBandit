import sys
import argparse
from pathlib import Path
from .bandit import BanditRouter

def main():
    parser = argparse.ArgumentParser(description="BanditGPT: Adaptive LLM Router CLI")
    parser.add_argument("--version", action="store_true", help="Show version")
    parser.add_argument("prompt", nargs="?", help="Prompt to route")
    parser.add_argument("--max-cost", type=float, help="Maximum cost constraint")
    parser.add_argument("--profile", default="best_value", help="Optimization profile")
    
    args = parser.parse_args()
    
    if args.version:
        print("BanditGPT v0.1.0")
        return
        
    if not args.prompt:
        parser.print_help()
        return
        
    try:
        router = BanditRouter.create()
        model, log = router.route(args.prompt, profile=args.profile, max_cost=args.max_cost)
        print(f"Selected Model: {model}")
        print(f"Predicted Utility: {log.predicted_utility:.4f}")
        print(f"Estimated Cost: ${log.cost_usd:.8f}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
