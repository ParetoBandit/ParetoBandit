#!/usr/bin/env python3
"""
Update models_cache.json with LiveCodeBench scores.

This script reads LiveCodeBench scores from livecodebench_scores.json and
updates the main models_cache.json file with these scores.

Usage:
    python update_models_cache_with_livecodebench.py
"""

import json
from pathlib import Path
from typing import Dict, List


def load_json(filepath: Path) -> dict:
    """Load JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_json(filepath: Path, data: dict, indent: int = 2):
    """Save JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent)


def update_models_cache_with_livecodebench():
    """Update models_cache.json with LiveCodeBench scores."""
    
    # Paths
    script_dir = Path(__file__).parent
    livecodebench_scores_path = script_dir / "livecodebench_scores.json"
    models_cache_path = script_dir.parent.parent.parent / "data" / "models_cache.json"
    
    print(f"Loading LiveCodeBench scores from: {livecodebench_scores_path}")
    livecodebench_data = load_json(livecodebench_scores_path)
    
    print(f"Loading models cache from: {models_cache_path}")
    models_cache = load_json(models_cache_path)
    
    # Create lookup dictionary by slug
    lcb_scores = {}
    for model in livecodebench_data.get("models", []):
        slug = model.get("slug")
        score = model.get("livecodebench")
        if slug:
            lcb_scores[slug] = score
    
    print(f"\nLiveCodeBench scores available for {len(lcb_scores)} models")
    
    # Update models cache
    updated_count = 0
    missing_count = 0
    already_had_score = 0
    
    models = models_cache.get("models", [])
    
    for model in models:
        slug = model.get("slug")
        
        if not slug:
            continue
        
        # Check if model already has livecodebench score in cache
        current_lcb = model.get("livecodebench")
        
        if slug in lcb_scores:
            new_score = lcb_scores[slug]
            
            # Update if missing or different
            if current_lcb is None:
                model["livecodebench"] = new_score
                updated_count += 1
                print(f"✓ Updated {model.get('name', slug)}: {new_score}")
            elif current_lcb != new_score:
                old_score = current_lcb
                model["livecodebench"] = new_score
                updated_count += 1
                print(f"✓ Updated {model.get('name', slug)}: {old_score} -> {new_score}")
            else:
                already_had_score += 1
        else:
            if current_lcb is None:
                missing_count += 1
                print(f"⚠ Missing LiveCodeBench score for: {model.get('name', slug)}")
    
    # Save updated cache
    print(f"\nSaving updated models cache to: {models_cache_path}")
    save_json(models_cache_path, models_cache)
    
    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Total models in cache:           {len(models)}")
    print(f"Models updated:                  {updated_count}")
    print(f"Models already had correct score: {already_had_score}")
    print(f"Models missing LiveCodeBench:    {missing_count}")
    print(f"\n✓ Models cache updated successfully!")
    
    return {
        "total_models": len(models),
        "updated": updated_count,
        "already_correct": already_had_score,
        "missing": missing_count
    }


if __name__ == "__main__":
    try:
        results = update_models_cache_with_livecodebench()
        exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
