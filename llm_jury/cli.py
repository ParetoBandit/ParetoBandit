"""Command-line interface for LLM Jury."""

import argparse
import logging
import sys
from pathlib import Path

from llm_jury import __version__
from llm_jury.config import get_config
from llm_jury.etl.pipeline import ETLPipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def cmd_config(args):
    """Handle config commands."""
    config = get_config()

    if args.config_command == "show":
        print("📋 LLM Jury Configuration")
        print("=" * 50)
        print(f"Config directory: {config.config_dir}")
        print(f"Data directory: {config.data_dir}")
        print(f"Cache file: {config.cache_file}")
        print(f"HELM enabled: {config.helm_enabled}")
        print(f"Auto update: {config.auto_update}")
        print()
        print("API Keys:")
        api_key = config.openrouter_api_key
        if api_key:
            print(f"  OpenRouter: {api_key[:15]}...***")
        else:
            print("  OpenRouter: Not set ❌")

    elif args.config_command == "set":
        if args.key == "openrouter_api_key":
            config.openrouter_api_key = args.value
            print(f"✅ Set OpenRouter API key")
        elif args.key == "data_dir":
            config.data_dir = args.value
            print(f"✅ Set data directory: {args.value}")
        elif args.key == "helm_enabled":
            config.helm_enabled = args.value.lower() in ["true", "1", "yes"]
            print(f"✅ Set HELM enabled: {config.helm_enabled}")
        else:
            config.set(args.key, args.value)
            print(f"✅ Set {args.key} = {args.value}")

    elif args.config_command == "validate":
        is_valid, errors = config.validate()
        if is_valid:
            print("✅ Configuration is valid")
        else:
            print("❌ Configuration has errors:")
            for error in errors:
                print(f"  - {error}")
            sys.exit(1)


def cmd_update(args):
    """Handle update command."""
    logger.info("Updating model cache from OpenRouter")

    try:
        pipeline = ETLPipeline()
        output_file = pipeline.update_cache(incremental=not args.full)

        print()
        print("✅ Cache updated successfully!")
        print(f"   Output: {output_file}")

    except Exception as e:
        logger.error(f"Update failed: {e}")
        sys.exit(1)


def cmd_evaluate(args):
    """Handle evaluate command (now uses web scraping instead of HELM)."""
    logger.info(f"Fetching benchmarks from Artificial Analysis")

    try:
        pipeline = ETLPipeline()

        # Run with benchmark scraping
        output_file = pipeline.run(
            scrape_benchmarks=True,
            model_filter=args.models.split(",") if args.models else None,
        )

        print()
        print("✅ Benchmarks fetched successfully!")
        print(f"   Output: {output_file}")
        print()
        print("📊 View results:")
        print(f"   cat {output_file} | jq '.[] | select(.has_benchmarks == true)'")

    except Exception as e:
        logger.error(f"Benchmark fetch failed: {e}")
        sys.exit(1)


def cmd_fetch(args):
    """Handle fetch command."""
    logger.info("Fetching models from OpenRouter (no benchmarks)")

    try:
        pipeline = ETLPipeline()
        output_file = pipeline.run(scrape_benchmarks=False)

        print()
        print("✅ Models fetched successfully!")
        print(f"   Output: {output_file}")

    except Exception as e:
        logger.error(f"Fetch failed: {e}")
        sys.exit(1)


def cmd_init(args):
    """Handle init command."""
    print("🚀 Initializing LLM Jury")
    print()

    config = get_config()

    # Check if API key is set
    if not config.openrouter_api_key:
        print("⚠️  OpenRouter API key not set")
        api_key = input("Enter your OpenRouter API key: ").strip()
        if api_key:
            config.openrouter_api_key = api_key
            print("✅ API key saved")
        else:
            print("❌ No API key provided. You can set it later with:")
            print("   llm-jury config set openrouter_api_key YOUR_KEY")
            sys.exit(1)

    # Validate config
    is_valid, errors = config.validate()
    if not is_valid:
        print("❌ Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)

    # Fetch initial data
    print()
    print("📥 Fetching initial model data from OpenRouter...")
    try:
        pipeline = ETLPipeline()
        output_file = pipeline.run(evaluate_with_helm=False)

        print()
        print("✅ LLM Jury initialized successfully!")
        print(f"   Cache: {output_file}")
        print()
        print("📚 Next steps:")
        print("   1. Update cache: llm-jury update")
        print("   2. Evaluate models: llm-jury evaluate --models anthropic/claude-3.5-sonnet")
        print("   3. Use in Python: from llm_jury import get_recommendations")

    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="LLM Jury - Intelligent LLM Model Routing and Recommendation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"llm-jury {__version__}")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Init command
    init_parser = subparsers.add_parser(
        "init",
        help="Initialize LLM Jury (first-time setup)"
    )

    # Config command
    config_parser = subparsers.add_parser(
        "config",
        help="Manage configuration"
    )
    config_subparsers = config_parser.add_subparsers(dest="config_command")

    config_show_parser = config_subparsers.add_parser("show", help="Show configuration")

    config_set_parser = config_subparsers.add_parser("set", help="Set configuration value")
    config_set_parser.add_argument("key", help="Configuration key")
    config_set_parser.add_argument("value", help="Configuration value")

    config_validate_parser = config_subparsers.add_parser("validate", help="Validate configuration")

    # Update command
    update_parser = subparsers.add_parser(
        "update",
        help="Update model cache from OpenRouter"
    )
    update_parser.add_argument(
        "--full",
        action="store_true",
        help="Full rebuild instead of incremental update"
    )

    # Fetch command
    fetch_parser = subparsers.add_parser(
        "fetch",
        help="Fetch models from OpenRouter (without HELM evaluation)"
    )

    # Evaluate command (now uses scraping)
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Fetch benchmark data from Artificial Analysis"
    )
    evaluate_parser.add_argument(
        "--models",
        help="Comma-separated list of model IDs to filter (optional)"
    )

    # Parse args
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Route to command handler
    try:
        if args.command == "init":
            cmd_init(args)
        elif args.command == "config":
            cmd_config(args)
        elif args.command == "update":
            cmd_update(args)
        elif args.command == "fetch":
            cmd_fetch(args)
        elif args.command == "evaluate":
            cmd_evaluate(args)
    except KeyboardInterrupt:
        print("\n\n❌ Interrupted by user")
        sys.exit(130)


if __name__ == "__main__":
    main()

