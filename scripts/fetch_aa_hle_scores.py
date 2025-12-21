#!/usr/bin/env python3
"""
Fetch HLE (Humanity's Last Exam) benchmark scores from Artificial Analysis API.

This script:
1. Loads the models_cache.json
2. Fetches HLE scores from Artificial Analysis API
3. Updates the cache with HLE benchmark data

Usage:
    export AA_API_KEY=your_key_here  # Set in .env
    python scripts/fetch_aa_hle_scores.py
    
    # Dry run (show what would be updated)
    python scripts/fetch_aa_hle_scores.py --dry-run
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import requests
from dotenv import load_dotenv

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")


AA_API_BASE = "https://api.artificialanalysis.ai"


def fetch_aa_models(api_key: str) -> List[Dict]:
    """Fetch all models from Artificial Analysis API."""
    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        # Try the leaderboard endpoint which includes benchmarks
        response = requests.get(f"{AA_API_BASE}/leaderboard", headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        # The response might be a list or dict with models
        if isinstance(data, list):
            return data
        return data.get("models", data.get("data", []))
    except Exception as e:
        print(f"ERROR: Failed to fetch models from Artificial Analysis: {e}")
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


def normalize_model_name(name: str) -> str:
    """Normalize model name for matching."""
    return name.lower().replace("-", "").replace("_", "").replace(" ", "")


def find_aa_model_by_name(cache_model: Dict, aa_models: List[Dict]) -> Optional[Dict]:
    """
    Find matching AA model by name matching.
    
    Tries multiple strategies:
    1. Exact openrouter_id match
    2. Display name match
    3. Normalized name match
    """
    openrouter_id = cache_model.get("openrouter_id", "")
    display_name = cache_model.get("display_name", "")
    name = cache_model.get("name", "")
    
    # Strategy 1: Try exact openrouter_id match
    for aa_model in aa_models:
        aa_id = aa_model.get("model_id", "")
        if openrouter_id and aa_id and openrouter_id.lower() == aa_id.lower():
            return aa_model
    
    # Strategy 2: Try display name match
    for aa_model in aa_models:
        aa_name = aa_model.get("model_name", "")
        if display_name and aa_name and display_name.lower() == aa_name.lower():
            return aa_model
    
    # Strategy 3: Normalized name match
    cache_normalized = normalize_model_name(display_name or name)
    for aa_model in aa_models:
        aa_name = aa_model.get("model_name", "")
        aa_normalized = normalize_model_name(aa_name)
        if cache_normalized and aa_normalized and cache_normalized == aa_normalized:
            return aa_model
    
    return None


def main():
    parser = argparse.ArgumentParser(description="Fetch HLE scores from Artificial Analysis")
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
    
    # Check API key (try both names)
    api_key = os.environ.get("AA_API_KEY") or os.environ.get("ARTIFICIAL_ANALYSIS_API_KEY")
    if not api_key:
        print("ERROR: AA_API_KEY or ARTIFICIAL_ANALYSIS_API_KEY not set!")
        print("Set it in .env file or export AA_API_KEY=your_key")
        return 1
    
    cache_path = Path(args.cache)
    if not cache_path.exists():
        print(f"ERROR: Cache file not found: {cache_path}")
        return 1
    
    print("=" * 80)
    print("Fetching HLE Scores from Artificial Analysis")
    print("=" * 80)
    print()
    
    # Load cache
    data = load_models_cache(cache_path)
    models = data.get("models", [])
    
    print(f"Loaded {len(models)} models from cache")
    print()
    
    # Fetch AA models
    print("Fetching model data from Artificial Analysis API...")
    aa_models = fetch_aa_models(api_key)
    
    if not aa_models:
        print("ERROR: Could not fetch models from Artificial Analysis")
        return 1
    
    print(f"Fetched {len(aa_models)} models from Artificial Analysis")
    print()
    
    # Update models with HLE scores
    updated_count = 0
    not_found_count = 0
    already_has_count = 0
    
    for i, model in enumerate(models):
        openrouter_id = model.get("openrouter_id")
        display_name = model.get("display_name", model.get("name", openrouter_id))
        
        # Check if already has HLE
        if model.get("hle") is not None:
            already_has_count += 1
            continue
        
        print(f"[{i+1}/{len(models)}] {display_name}")
        print(f"  Model ID: {openrouter_id}")
        
        # Find matching AA model
        aa_model = find_aa_model_by_name(model, aa_models)
        
        if not aa_model:
            print(f"  ✗ Not found in Artificial Analysis")
            not_found_count += 1
            print()
            continue
        
        # Extract HLE score
        quality = aa_model.get("quality", {})
        hle_score = quality.get("hle")
        
        if hle_score is None:
            print(f"  ✗ No HLE score available")
            not_found_count += 1
            print()
            continue
        
        # Update model
        old_hle = model.get("hle", "N/A")
        model["hle"] = float(hle_score)
        
        print(f"  ✓ HLE: {old_hle} → {hle_score}")
        updated_count += 1
        print()
    
    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total models: {len(models)}")
    print(f"Already had HLE: {already_has_count}")
    print(f"Successfully updated: {updated_count}")
    print(f"Not found / No HLE data: {not_found_count}")
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

