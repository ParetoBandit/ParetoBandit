#!/usr/bin/env python3
"""
Fetch IFBench (Instruction Following Benchmark) scores from Artificial Analysis API.

IFBench evaluates models' ability to follow complex instructions accurately.
This benchmark is useful for assessing instruction-following capabilities.

Usage:
    export ARTIFICIAL_ANALYSIS_API_KEY="your_api_key"
    python fetch_ifbench_scores.py

Requirements:
    - requests
    - Artificial Analysis API key (https://artificialanalysis.ai)
"""

import os
import sys
import json
import requests
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path


def fetch_ifbench_scores(api_key: str) -> List[Dict]:
    """
    Fetch IFBench scores from Artificial Analysis API.
    
    Args:
        api_key: Artificial Analysis API key
        
    Returns:
        List of model dictionaries with IFBench scores
        
    Raises:
        requests.exceptions.RequestException: If API request fails
        ValueError: If API key is invalid
    """
    API_ENDPOINT = "https://artificialanalysis.ai/api/v2/data/llms/models"
    
    headers = {
        "x-api-key": api_key,
        "Accept": "application/json"
    }
    
    print("Fetching model data from Artificial Analysis API...")
    
    try:
        response = requests.get(API_ENDPOINT, headers=headers, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if data.get("status") != 200:
            raise ValueError(f"API returned non-200 status: {data.get('status')}")
        
        models = data.get("data", [])
        print(f"✓ Fetched {len(models)} models from API")
        
        # Extract relevant fields including IFBench
        extracted_models = []
        for model in models:
            evaluations = model.get("evaluations", {})
            model_creator = model.get("model_creator", {})
            
            extracted_models.append({
                "name": model.get("name"),
                "slug": model.get("slug"),
                "creator_name": model_creator.get("name"),
                "creator_slug": model_creator.get("slug"),
                "ifbench": evaluations.get("ifbench"),
                "source": "artificial_analysis_api",
                "aa_id": model.get("id"),
                "fetch_date": datetime.now().isoformat()
            })
        
        return extracted_models
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            raise ValueError("Invalid or missing Artificial Analysis API key")
        elif e.response.status_code == 429:
            raise ValueError("Rate limit exceeded (1000 requests/day). Try again later.")
        else:
            raise
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Failed to fetch from API: {e}")


def analyze_coverage(models: List[Dict]) -> Dict:
    """Analyze IFBench coverage statistics."""
    total = len(models)
    with_ifbench = sum(1 for m in models if m.get("ifbench") is not None)
    without_ifbench = total - with_ifbench
    
    ifbench_scores = [m["ifbench"] for m in models if m.get("ifbench") is not None]
    
    stats = {
        "total_models": total,
        "models_with_ifbench": with_ifbench,
        "models_without_ifbench": without_ifbench,
        "coverage_percentage": round(with_ifbench / total * 100, 1),
    }
    
    if ifbench_scores:
        ifbench_scores.sort()
        stats.update({
            "min_score": min(ifbench_scores),
            "max_score": max(ifbench_scores),
            "median_score": ifbench_scores[len(ifbench_scores) // 2],
            "mean_score": sum(ifbench_scores) / len(ifbench_scores),
        })
    
    return stats


def update_models_cache(models: List[Dict]) -> Dict:
    """
    Update models_cache.json with IFBench scores.
    
    Returns:
        Statistics about the update
    """
    # Find models_cache.json
    cache_path = Path(__file__).parent.parent.parent.parent / "data" / "models_cache.json"
    
    if not cache_path.exists():
        print(f"⚠️  Warning: models_cache.json not found at {cache_path}")
        return {
            "updated": 0,
            "already_correct": 0,
            "missing": 0
        }
    
    print(f"\nUpdating models cache at: {cache_path}")
    
    # Load cache
    with open(cache_path, 'r') as f:
        cache_data = json.load(f)
    
    cache_models = cache_data.get("models", [])
    
    # Create lookup by slug
    ifbench_lookup = {m["slug"]: m.get("ifbench") for m in models}
    
    # Update cache
    updated = 0
    already_correct = 0
    missing = 0
    
    for model in cache_models:
        slug = model.get("slug")
        if slug in ifbench_lookup:
            new_score = ifbench_lookup[slug]
            current_score = model.get("ifbench")
            
            if current_score != new_score:
                model["ifbench"] = new_score
                updated += 1
                print(f"  ✓ Updated {model.get('name')}: {current_score} → {new_score}")
            else:
                already_correct += 1
        else:
            if model.get("ifbench") is None:
                missing += 1
    
    # Save cache
    with open(cache_path, 'w') as f:
        json.dump(cache_data, f, indent=2)
    
    return {
        "updated": updated,
        "already_correct": already_correct,
        "missing": missing,
        "total": len(cache_models)
    }


def main():
    """Main execution function."""
    # Get API key from environment
    api_key = os.getenv("ARTIFICIAL_ANALYSIS_API_KEY")
    
    if not api_key:
        print("❌ Error: ARTIFICIAL_ANALYSIS_API_KEY not set")
        print("\nPlease set your API key:")
        print("  export ARTIFICIAL_ANALYSIS_API_KEY='your_api_key'")
        print("\nGet an API key at: https://artificialanalysis.ai")
        sys.exit(1)
    
    try:
        # Fetch data
        models = fetch_ifbench_scores(api_key)
        
        # Analyze coverage
        stats = analyze_coverage(models)
        
        print("\n" + "="*70)
        print("IFBench Coverage Statistics")
        print("="*70)
        print(f"Total Models:           {stats['total_models']}")
        print(f"Models with IFBench:    {stats['models_with_ifbench']} ({stats['coverage_percentage']}%)")
        print(f"Models without IFBench: {stats['models_without_ifbench']}")
        
        if stats.get('min_score'):
            print(f"\nScore Range:")
            print(f"  Minimum:  {stats['min_score']:.3f} ({stats['min_score']*100:.1f}%)")
            print(f"  Mean:     {stats['mean_score']:.3f} ({stats['mean_score']*100:.1f}%)")
            print(f"  Median:   {stats['median_score']:.3f} ({stats['median_score']*100:.1f}%)")
            print(f"  Maximum:  {stats['max_score']:.3f} ({stats['max_score']*100:.1f}%)")
        
        # Show models without IFBench
        models_without = [m for m in models if m.get("ifbench") is None]
        if models_without:
            print(f"\nModels without IFBench scores ({len(models_without)}):")
            for m in models_without[:10]:
                print(f"  - {m['name']} ({m['creator_name']})")
            if len(models_without) > 10:
                print(f"  ... and {len(models_without) - 10} more")
        
        # Save to file
        output_file = Path(__file__).parent / "ifbench_scores.json"
        output_data = {
            "metadata": {
                "benchmark": "IFBench (Instruction Following Benchmark)",
                "fetch_date": datetime.now().isoformat(),
                "source": "artificial_analysis_api",
                "total_models": stats["total_models"],
                "models_with_ifbench": stats["models_with_ifbench"],
                "coverage_percentage": stats["coverage_percentage"]
            },
            "models": models
        }
        
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n✓ Saved {len(models)} models to {output_file}")
        
        # Show top performers
        models_with_scores = [m for m in models if m.get("ifbench") is not None]
        if models_with_scores:
            models_with_scores.sort(key=lambda x: x["ifbench"], reverse=True)
            
            print(f"\n" + "="*70)
            print(f"Top 10 IFBench Performers")
            print("="*70)
            print(f"{'Rank':<6} {'Score':<10} {'Model':<35} {'Creator':<20}")
            print("-" * 70)
            for i, m in enumerate(models_with_scores[:10], 1):
                score_pct = f"{m['ifbench']*100:.1f}%"
                print(f"{i:<6} {score_pct:<10} {m['name'][:35]:<35} {m['creator_name'][:20]:<20}")
        
        # Update models cache
        print("\n" + "="*70)
        print("Updating Models Cache")
        print("="*70)
        cache_stats = update_models_cache(models)
        
        if cache_stats["updated"] > 0 or cache_stats["already_correct"] > 0:
            print(f"\nCache Update Summary:")
            print(f"  Total models in cache: {cache_stats['total']}")
            print(f"  Updated:               {cache_stats['updated']}")
            print(f"  Already correct:       {cache_stats['already_correct']}")
            print(f"  Missing IFBench:       {cache_stats['missing']}")
            print(f"\n✓ Models cache updated successfully!")
        
    except ValueError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
