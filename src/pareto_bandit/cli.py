"""Command-line interface for ParetoBandit.

Provides a thin CLI wrapper around :class:`~pareto_bandit.router.BanditRouter`
for quick routing decisions, version inspection, and model pre-downloading.

Usage::

    paretobandit "Explain the transformer architecture" --max-cost 0.01
    paretobandit --download-models
    paretobandit --version
"""

import argparse
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from .router import BanditRouter


def _get_version() -> str:
    """Return the installed package version, falling back to ``'0.1.0'``."""
    try:
        return _pkg_version("paretobandit")
    except PackageNotFoundError:
        return "0.1.0"


def main() -> None:
    """Entry point for the ``paretobandit`` console script.

    Parses CLI arguments and dispatches to version display, model
    downloading, or single-prompt routing.
    """
    parser = argparse.ArgumentParser(description="ParetoBandit: Adaptive LLM Router CLI")
    parser.add_argument("--version", action="store_true", help="Show version")
    parser.add_argument("prompt", nargs="?", help="Prompt to route")
    parser.add_argument("--max-cost", type=float, help="Maximum cost constraint")
    parser.add_argument(
        "--cost-penalty",
        type=float,
        default=0.3,
        metavar="LAMBDA",
        help="Static cost-quality trade-off weight λ_c (default: 0.3). "
             "0.0 = quality-only routing, 0.3 = moderate cost awareness, "
             "0.5+ = aggressive cost preference.",
    )
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
        router = BanditRouter.create(cost_penalty=args.cost_penalty)
        model, log = router.route(args.prompt, max_cost=args.max_cost)
        print(f"Selected Model: {model}")
        print(f"Predicted Utility: {log.predicted_utility:.4f}")
        print(f"Estimated Cost: ${log.cost_usd:.8f}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
