#!/usr/bin/env python3
"""
Fetch all available benchmark scores from Artificial Analysis API.

This script fetches comprehensive benchmark data including:
- Intelligence Index (AA proprietary)
- Coding Index (AA proprietary)
- Math Index (AA proprietary)
- AIME 2025
- LCR (Logic & Reasoning)
- Updates for IFBench, TAU2, TerminalBench, AIME

Usage:
    export ARTIFICIAL_ANALYSIS_API_KEY="your_api_key"
    python fetch_all_aa_benchmarks.py

Requirements:
    - requests
    - Artificial Analysis API key
"""

import os
import sys
import json
import requests
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path


def fetch_all_benchmarks(api_key: str) -> List[Dict]:
    """
    Fetch all benchmark scores from Artificial Analysis API.
    
    Args:
        api_key: Artificial Analysis API key
        
    Returns:
        List of model dictionaries with all benchmark scores
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
        
        # Extract all relevant benchmark fields
        extracted_models = []
        for model in models:
            evaluations = model.get("evaluations", {})
            model_creator = model.get("model_creator", {})
            
            extracted_models.append({
                "name": model.get("name"),
                "slug": model.get("slug"),
                "creator_name": model_creator.get("name"),
                "creator_slug": model_creator.get("slug"),
                
                # Existing benchmarks (ensure they're included)
                "gpqa": evaluations.get("gpqa"),
                "mmlu_pro": evaluations.get("mmlu_pro"),
                "hle": evaluations.get("hle"),
                "livecodebench": evaluations.get("livecodebench"),
                "scicode": evaluations.get("scicode"),
                "math_500": evaluations.get("math_500"),
                "aime": evaluations.get("aime"),
                "ifbench": evaluations.get("ifbench"),
                "tau2": evaluations.get("tau2"),
                "terminalbench_hard": evaluations.get("terminalbench_hard"),
                
                # NEW: AA Proprietary Indices
                "intelligence_index": evaluations.get("artificial_analysis_intelligence_index"),
                "coding_index": evaluations.get("artificial_analysis_coding_index"),
                "math_index": evaluations.get("artificial_analysis_math_index"),
                
                # NEW: AIME 2025
                "aime_25": evaluations.get("aime_25"),
                
                # NEW: LCR (Logic & Reasoning)
                "lcr": evaluations.get("lcr"),
                
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
    """Analyze benchmark coverage statistics."""
    total = len(models)
    
    benchmarks = {
        "gpqa": "GPQA (Reasoning)",
        "mmlu_pro": "MMLU-Pro (Knowledge)",
        "hle": "HLE (Hard Logic)",
        "livecodebench": "LiveCodeBench (Coding)",
        "scicode": "SciCode (Scientific Coding)",
        "math_500": "MATH-500",
        "aime": "AIME (2024)",
        "aime_25": "AIME 2025 (NEW)",
        "ifbench": "IFBench (Instruction Following)",
        "tau2": "TAU2",
        "terminalbench_hard": "TerminalBench Hard",
        "lcr": "LCR - Logic & Reasoning (NEW)",
        "intelligence_index": "Intelligence Index (AA) (NEW)",
        "coding_index": "Coding Index (AA) (NEW)",
        "math_index": "Math Index (AA) (NEW)",
    }
    
    coverage = {}
    for key, name in benchmarks.items():
        count = sum(1 for m in models if m.get(key) is not None)
        coverage[key] = {
            "name": name,
            "count": count,
            "percentage": round(count / total * 100, 1)
        }
    
    return coverage


def update_models_cache(models: List[Dict]) -> Dict:
    """
    Update models_cache.json with all benchmark scores.
    
    Returns:
        Statistics about the update
    """
    # Find models_cache.json
    cache_path = Path(__file__).parent.parent.parent / "data" / "models_cache.json"
    
    if not cache_path.exists():
        print(f"⚠️  Warning: models_cache.json not found at {cache_path}")
        return {"total_updates": 0, "updates_by_field": {}, "total_models": 0}
    
    print(f"\nUpdating models cache at: {cache_path}")
    
    # Load cache
    with open(cache_path, 'r') as f:
        cache_data = json.load(f)
    
    cache_models = cache_data.get("models", [])
    
    # Create lookup by slug
    aa_lookup = {m["slug"]: m for m in models}
    
    # Track updates
    updates_by_field = {}
    total_updates = 0
    
    # Fields to update
    benchmark_fields = [
        "gpqa", "mmlu_pro", "hle", "livecodebench", "scicode", 
        "math_500", "aime", "aime_25", "ifbench", "tau2", 
        "terminalbench_hard", "lcr", "intelligence_index", 
        "coding_index", "math_index"
    ]
    
    for field in benchmark_fields:
        updates_by_field[field] = {"added": 0, "updated": 0}
    
    # Update cache
    for model in cache_models:
        slug = model.get("slug")
        if slug in aa_lookup:
            aa_model = aa_lookup[slug]
            
            for field in benchmark_fields:
                new_value = aa_model.get(field)
                current_value = model.get(field)
                
                if new_value is not None and current_value != new_value:
                    if current_value is None:
                        updates_by_field[field]["added"] += 1
                        print(f"  ✓ Added {field} for {model.get('name')}: {new_value}")
                    else:
                        updates_by_field[field]["updated"] += 1
                        print(f"  ✓ Updated {field} for {model.get('name')}: {current_value} → {new_value}")
                    
                    model[field] = new_value
                    total_updates += 1
    
    # Save cache
    with open(cache_path, 'w') as f:
        json.dump(cache_data, f, indent=2)
    
    return {
        "total_updates": total_updates,
        "updates_by_field": updates_by_field,
        "total_models": len(cache_models)
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
        models = fetch_all_benchmarks(api_key)
        
        # Analyze coverage
        coverage = analyze_coverage(models)
        
        print("\n" + "="*70)
        print("Benchmark Coverage in AA API")
        print("="*70)
        
        # Group by new vs existing
        new_benchmarks = ["aime_25", "lcr", "intelligence_index", "coding_index", "math_index"]
        
        print("\n🆕 NEW Benchmarks:")
        for key in new_benchmarks:
            stats = coverage[key]
            print(f"  {stats['name']:<40} {stats['count']:>4}/{len(models)} ({stats['percentage']:>5.1f}%)")
        
        print("\n📊 Existing Benchmarks (for update):")
        for key, stats in coverage.items():
            if key not in new_benchmarks:
                print(f"  {stats['name']:<40} {stats['count']:>4}/{len(models)} ({stats['percentage']:>5.1f}%)")
        
        # Save to file
        output_file = Path(__file__).parent / "all_aa_benchmarks.json"
        output_data = {
            "metadata": {
                "fetch_date": datetime.now().isoformat(),
                "source": "artificial_analysis_api",
                "total_models": len(models),
                "benchmarks_included": list(coverage.keys())
            },
            "coverage": coverage,
            "models": models
        }
        
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n✓ Saved {len(models)} models to {output_file}")
        
        # Update models cache
        print("\n" + "="*70)
        print("Updating Models Cache")
        print("="*70)
        
        cache_stats = update_models_cache(models)
        
        print(f"\n" + "="*70)
        print("Update Summary")
        print("="*70)
        print(f"Total updates applied: {cache_stats['total_updates']}")
        print(f"Models in cache: {cache_stats['total_models']}")
        
        print(f"\nUpdates by benchmark:")
        for field, stats in cache_stats["updates_by_field"].items():
            total = stats["added"] + stats["updated"]
            if total > 0:
                print(f"  {field:<25} {stats['added']:>3} added, {stats['updated']:>3} updated")
        
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
