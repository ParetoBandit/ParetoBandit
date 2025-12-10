#!/usr/bin/env python3
"""Fetch benchmark scores using LLM-powered model name matching.

NOTE: Most benchmark data collection methods have been removed (December 2025).
This script is kept for reference but most functions return 0 (removed).

Removed benchmarks:
- IFEval: Not used in composite scores
- WildBench: Redundant with Arena rankings and MixEval
- Arena-Hard-Auto: Only manual Arena rankings are used

For Arena data, use: scripts/quality_scoring/update_arena_rankings.py (manual curation)
For benchmarks, use domain-specific scripts (HumanEval, MBPP, MixEval, etc.)

Usage:
    # This script is deprecated - most functions return 0
    python scripts/fetch_scores_llm.py --all
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
import sys
from typing import List

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_jury.etl.llm_matcher import LLMModelMatcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def fetch_ifeval_with_llm(cache_models: List, matcher: LLMModelMatcher, dry_run: bool = False) -> int:
    """IFEval support removed (December 2025).
    
    IFEval was removed from the cache as it is not used in any composite scores
    (CCS, CRS, CFS, CSS). Use domain-specific benchmarks instead.
    """
    logger.info("\n=== IFEval (REMOVED) ===")
    logger.info("IFEval support has been removed from this project.")
    logger.info("Use domain-specific benchmarks (HumanEval, MBPP, MixEval, etc.) instead.")
    return 0


def fetch_wildbench_with_llm(cache_models: list, matcher: LLMModelMatcher, dry_run: bool = False) -> int:
    """WildBench support removed (December 2025).
    
    WildBench was removed from the project as it was not used in composite scores
    and was redundant with Arena rankings and MixEval benchmarks.
    """
    logger.info("\n=== WildBench (REMOVED) ===")
    logger.info("WildBench support has been removed from this project.")
    logger.info("Use Arena rankings or MixEval for multi-domain evaluation instead.")
    return 0


def fetch_arena_hard_with_llm(cache_models: list, matcher: LLMModelMatcher, dry_run: bool = False) -> int:
    """Arena-Hard-Auto support removed (December 2025).
    
    Arena-Hard-Auto was removed as arena_hard_auto_score is not used in any 
    composite scores (CCS, CRS, CFS, CSS). Only arena_rank_* fields from manual 
    LMArena curation are used.
    """
    logger.info("\n=== Arena-Hard-Auto (REMOVED) ===")
    logger.info("Arena-Hard-Auto client has been removed from this project.")
    logger.info("Use manual Arena ranking curation (update_arena_rankings.py) instead.")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Fetch benchmark scores using LLM-powered matching"
    )
    parser.add_argument(
        "--cache-file",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "models_cache.json",
        help="Path to models_cache.json"
    )
    parser.add_argument("--ifeval", action="store_true", help="Fetch IFEval scores")
    parser.add_argument("--wildbench", action="store_true", help="Fetch WildBench scores")
    parser.add_argument("--arena-hard", action="store_true", help="Fetch Arena-Hard-Auto scores")
    parser.add_argument("--all", action="store_true", help="Fetch all benchmarks")
    parser.add_argument("--dry-run", action="store_true", help="Show matches without saving")
    parser.add_argument("--model", default="openai/gpt-4o-mini", help="LLM model for matching")
    
    args = parser.parse_args()
    
    if not any([args.ifeval, args.wildbench, args.arena_hard, args.all]):
        parser.print_help()
        print("\nError: Specify at least one benchmark (--ifeval, --wildbench, --arena-hard, or --all)")
        sys.exit(1)
    
    # Load cache
    logger.info(f"Loading cache from {args.cache_file}")
    with open(args.cache_file) as f:
        cache = json.load(f)
    
    models = cache.get("models", [])
    logger.info(f"Loaded {len(models)} models from cache")
    
    # Initialize matcher
    try:
        matcher = LLMModelMatcher(model=args.model)
        logger.info(f"Using {args.model} for matching")
    except ValueError as e:
        logger.error(f"Error: {e}")
        logger.error("Set OPENROUTER_API_KEY environment variable")
        sys.exit(1)
    
    total_matches = 0
    
    # Fetch selected benchmarks
    if args.ifeval or args.all:
        total_matches += fetch_ifeval_with_llm(models, matcher, args.dry_run)
    
    if args.wildbench or args.all:
        total_matches += fetch_wildbench_with_llm(models, matcher, args.dry_run)
    
    if args.arena_hard or args.all:
        total_matches += fetch_arena_hard_with_llm(models, matcher, args.dry_run)
    
    # Save if not dry run
    if not args.dry_run and total_matches > 0:
        # Update metadata
        if "metadata" not in cache:
            cache["metadata"] = {}
        cache["metadata"]["llm_matching"] = {
            "method": "LLM-powered semantic matching",
            "model": args.model,
            "last_run": datetime.now().isoformat(),
        }
        
        cache["models"] = models
        
        # Backup and save
        backup_path = args.cache_file.with_suffix(".json.bak")
        with open(backup_path, "w") as f:
            json.dump(cache, f, indent=2)
        
        with open(args.cache_file, "w") as f:
            json.dump(cache, f, indent=2)
        
        logger.info(f"\n✅ Saved {total_matches} total matches to {args.cache_file}")
    elif args.dry_run:
        logger.info(f"\n[DRY RUN] Would save {total_matches} total matches")


if __name__ == "__main__":
    main()

