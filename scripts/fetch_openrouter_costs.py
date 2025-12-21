#!/usr/bin/env python3
"""
Fetch missing input/output token costs from OpenRouter API.

This script:
1. Loads the models_cache.json
2. Identifies models missing input_cost_per_m or output_cost_per_m
3. Fetches pricing from OpenRouter's models API
4. Updates the cache with the missing cost data

Usage:
    python scripts/fetch_openrouter_costs.py
    
    # Dry run (show what would be updated)
    python scripts/fetch_openrouter_costs.py --dry-run
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


OPENROUTER_MODELS_API = "https://openrouter.ai/api/v1/models"


def fetch_openrouter_models() -> List[Dict]:
    """Fetch all models from OpenRouter API."""
    try:
        response = requests.get(OPENROUTER_MODELS_API, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data.get("data", [])
    except Exception as e:
        print(f"ERROR: Failed to fetch models from OpenRouter: {e}")
        return []


def load_models_cache(cache_path: Path) -> Dict:
    """Load models_cache.json."""
    with open(cache_path) as f:
        return json.load(f)


def save_models_cache(cache_path: Path, data: Dict) -> None:
    """Save models_cache.json with backup."""
    # Create backup
    backup_path = cache_path.with_suffix('.json.bak')
    if cache_path.exists():
        import shutil
        shutil.copy(cache_path, backup_path)
        print(f"Backed up cache to {backup_path}")
    
    # Save updated cache
    with open(cache_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Updated cache: {cache_path}")


def main():
    parser = argparse.ArgumentParser(description="Fetch missing cost data from OpenRouter")
    parser.add_argument(
        "--cache",
        type=str,
        default=str(PROJECT_ROOT / "banditgpt" / "data" / "models_cache.json"),
        help="Path to models_cache.json",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be updated without saving",
    )
    args = parser.parse_args()
    
    cache_path = Path(args.cache)
    if not cache_path.exists():
        print(f"ERROR: Cache file not found: {cache_path}")
        return 1
    
    print("=" * 80)
    print("Fetching Missing Cost Data from OpenRouter")
    print("=" * 80)
    print()
    
    # Load cache
    data = load_models_cache(cache_path)
    models = data.get("models", [])
    
    print(f"Loaded {len(models)} models from cache")
    
    # Find models missing cost data
    models_missing_costs = []
    for model in models:
        openrouter_id = model.get("openrouter_id")
        if not openrouter_id:
            continue
        
        input_cost = model.get("input_cost_per_m")
        output_cost = model.get("output_cost_per_m")
        
        # Check if either is missing or zero
        missing_input = not input_cost or float(input_cost) <= 0
        missing_output = not output_cost or float(output_cost) <= 0
        
        if missing_input or missing_output:
            models_missing_costs.append({
                "model": model,
                "missing_input": missing_input,
                "missing_output": missing_output,
            })
    
    print(f"Found {len(models_missing_costs)} models with missing cost data")
    print()
    
    if not models_missing_costs:
        print("No models need cost updates!")
        return 0
    
    # Fetch OpenRouter models
    print("Fetching model data from OpenRouter API...")
    openrouter_models = fetch_openrouter_models()
    
    if not openrouter_models:
        print("ERROR: Could not fetch models from OpenRouter")
        return 1
    
    print(f"Fetched {len(openrouter_models)} models from OpenRouter")
    print()
    
    # Build lookup by model ID
    openrouter_lookup = {}
    for or_model in openrouter_models:
        model_id = or_model.get("id")
        if model_id:
            openrouter_lookup[model_id] = or_model
    
    # Update models
    updated_count = 0
    not_found_count = 0
    
    for item in models_missing_costs:
        model = item["model"]
        openrouter_id = model.get("openrouter_id")
        display_name = model.get("display_name", model.get("name", openrouter_id))
        
        print(f"[{updated_count + not_found_count + 1}/{len(models_missing_costs)}] {display_name}")
        print(f"  Model ID: {openrouter_id}")
        
        # Look up in OpenRouter data
        or_model = openrouter_lookup.get(openrouter_id)
        
        if not or_model:
            print(f"  ✗ Not found in OpenRouter API")
            not_found_count += 1
            print()
            continue
        
        # Extract pricing
        pricing = or_model.get("pricing", {})
        
        # OpenRouter returns prices in dollars per token, we store per 1M tokens
        input_cost_per_token = pricing.get("prompt")  # $/token
        output_cost_per_token = pricing.get("completion")  # $/token
        
        if input_cost_per_token is None or output_cost_per_token is None:
            print(f"  ✗ No pricing data in OpenRouter API")
            not_found_count += 1
            print()
            continue
        
        # Convert to $/1M tokens
        input_cost_per_m = float(input_cost_per_token) * 1_000_000
        output_cost_per_m = float(output_cost_per_token) * 1_000_000
        
        # Update model
        old_input = model.get("input_cost_per_m", 0)
        old_output = model.get("output_cost_per_m", 0)
        
        if item["missing_input"]:
            model["input_cost_per_m"] = input_cost_per_m
            print(f"  ✓ Input: ${old_input:.4f} → ${input_cost_per_m:.4f} per 1M tokens")
        else:
            print(f"  - Input: ${old_input:.4f} per 1M tokens (already set)")
        
        if item["missing_output"]:
            model["output_cost_per_m"] = output_cost_per_m
            print(f"  ✓ Output: ${old_output:.4f} → ${output_cost_per_m:.4f} per 1M tokens")
        else:
            print(f"  - Output: ${old_output:.4f} per 1M tokens (already set)")
        
        # Also update legacy fields if present
        if "price_1m_input" in model and item["missing_input"]:
            model["price_1m_input"] = input_cost_per_m
        if "price_1m_output" in model and item["missing_output"]:
            model["price_1m_output"] = output_cost_per_m
        
        # Recalculate blended cost (3:1 ratio)
        blended = (3 * input_cost_per_m + output_cost_per_m) / 4
        model["price_1m_blended"] = blended
        model["price_1m_blended_3_to_1"] = blended
        
        updated_count += 1
        print()
    
    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total models checked: {len(models_missing_costs)}")
    print(f"Successfully updated: {updated_count}")
    print(f"Not found in OpenRouter: {not_found_count}")
    print()
    
    if args.dry_run:
        print("DRY RUN: No changes saved")
        return 0
    
    if updated_count > 0:
        save_models_cache(cache_path, data)
        print()
        print("✓ Cache updated successfully!")
    else:
        print("No updates to save.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

