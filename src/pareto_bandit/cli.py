import sys
import argparse
from importlib.metadata import version as _pkg_version, PackageNotFoundError
from pathlib import Path
from .router import BanditRouter


def _get_version() -> str:
    try:
        return _pkg_version("paretobandit")
    except PackageNotFoundError:
        return "0.1.0"


def main():
    parser = argparse.ArgumentParser(description="ParetoBandit: Adaptive LLM Router CLI")
    parser.add_argument("--version", action="store_true", help="Show version")
    parser.add_argument("prompt", nargs="?", help="Prompt to route")
    parser.add_argument("--max-cost", type=float, help="Maximum cost constraint")
    parser.add_argument("--profile", default="best_value", help="Optimization profile")
    parser.add_argument("--download-models", action="store_true", help="Download required models (for Docker/CI pre-warming)")
    
    args = parser.parse_args()
    
    if args.version:
        print(f"ParetoBandit v{_get_version()}")
        return

    if args.download_models:
        print("Initializing ParetoBandit models...")
        try:
            from .feature_service import FeatureService
            print("Downloading sentence transformer model (this may take a while)...")
            fs = FeatureService()
            # Trigger lazy load
            _ = fs.encoder
            print("✓ Model downloaded and ready.")
        except Exception as e:
            print(f"Error downloading models: {e}")
            sys.exit(1)
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
