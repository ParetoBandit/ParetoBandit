#!/usr/bin/env python3
"""
Rejudge GPT-4-Turbo using the same multi-judge CoT system as Mixtral and GPT-4o.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from rejudge_cot import CoTRewardGenerator
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rejudge GPT-4-Turbo with multi-judge CoT")
    parser.add_argument("--prompts", type=str, required=True, help="Path to prompts JSONL file")
    parser.add_argument("--models", type=str, required=True, help="Path to models JSON file")
    parser.add_argument("--output", type=str, required=True, help="Path to output JSONL file")
    parser.add_argument("--cache", type=str, required=True, help="Path to cache JSONL file")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of prompts")
    parser.add_argument("--workers", type=int, default=32, help="Number of parallel workers")
    args = parser.parse_args()
    
    gen = CoTRewardGenerator(max_workers=args.workers)
    
    gen.run(
        prompts_file=Path(args.prompts),
        models_file=Path(args.models),
        output_file=Path(args.output),
        cache_file=Path(args.cache),
        is_lmsys=False,
        limit=args.limit
    )
    
    print("\n✅ Rejudging complete!")
    print(f"   Output: {args.output}")

