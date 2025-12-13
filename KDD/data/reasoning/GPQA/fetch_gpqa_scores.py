#!/usr/bin/env python3
"""
Standalone script to fetch GPQA scores from Artificial Analysis API.

This script fetches benchmark data including GPQA (Graduate-Level Google-Proof Q&A)
scores from the Artificial Analysis API and saves them to a JSON file.

Usage:
    export ARTIFICIAL_ANALYSIS_API_KEY="your_api_key"
    python fetch_gpqa_scores.py

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


def fetch_gpqa_scores(api_key: str) -> List[Dict]:
    """
    Fetch GPQA scores from Artificial Analysis API.
    
    Args:
        api_key: Artificial Analysis API key
        
    Returns:
        List of model dictionaries with GPQA scores
        
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
        
        # Extract relevant fields
        extracted_models = []
        for model in models:
            evaluations = model.get("evaluations", {})
            model_creator = model.get("model_creator", {})
            
            extracted_models.append({
                "name": model.get("name"),
                "slug": model.get("slug"),
                "creator_name": model_creator.get("name"),
                "creator_slug": model_creator.get("slug"),
                "gpqa": evaluations.get("gpqa"),
                "mmlu_pro": evaluations.get("mmlu_pro"),
                "math_500": evaluations.get("math_500"),
                "hle": evaluations.get("hle"),
                "aime": evaluations.get("aime"),
                "intelligence_index": evaluations.get("artificial_analysis_intelligence_index"),
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
    """Analyze GPQA coverage statistics."""
    total = len(models)
    with_gpqa = sum(1 for m in models if m.get("gpqa") is not None)
    without_gpqa = total - with_gpqa
    
    gpqa_scores = [m["gpqa"] for m in models if m.get("gpqa") is not None]
    
    stats = {
        "total_models": total,
        "models_with_gpqa": with_gpqa,
        "models_without_gpqa": without_gpqa,
        "coverage_percentage": round(with_gpqa / total * 100, 1),
    }
    
    if gpqa_scores:
        gpqa_scores.sort()
        stats.update({
            "min_score": min(gpqa_scores),
            "max_score": max(gpqa_scores),
            "median_score": gpqa_scores[len(gpqa_scores) // 2],
        })
    
    return stats


def main():
    """Main execution function."""
    # Get API key
    api_key = os.getenv("ARTIFICIAL_ANALYSIS_API_KEY")
    
    if not api_key:
        print("❌ Error: ARTIFICIAL_ANALYSIS_API_KEY not set")
        print("\nPlease set your API key:")
        print("  export ARTIFICIAL_ANALYSIS_API_KEY='your_api_key'")
        print("\nGet an API key at: https://artificialanalysis.ai")
        sys.exit(1)
    
    try:
        # Fetch data
        models = fetch_gpqa_scores(api_key)
        
        # Analyze coverage
        stats = analyze_coverage(models)
        
        print("\n" + "="*60)
        print("GPQA Coverage Statistics")
        print("="*60)
        print(f"Total Models:           {stats['total_models']}")
        print(f"Models with GPQA:       {stats['models_with_gpqa']} ({stats['coverage_percentage']}%)")
        print(f"Models without GPQA:    {stats['models_without_gpqa']}")
        
        if stats.get('min_score'):
            print(f"\nScore Range:")
            print(f"  Minimum:  {stats['min_score']:.3f} ({stats['min_score']*100:.1f}%)")
            print(f"  Median:   {stats['median_score']:.3f} ({stats['median_score']*100:.1f}%)")
            print(f"  Maximum:  {stats['max_score']:.3f} ({stats['max_score']*100:.1f}%)")
        
        # Show models without GPQA
        models_without = [m for m in models if m.get("gpqa") is None]
        if models_without:
            print(f"\nModels without GPQA scores:")
            for m in models_without:
                print(f"  - {m['name']} ({m['creator_name']})")
        
        # Save to file
        output_file = "gpqa_scores.json"
        output_data = {
            "metadata": {
                "fetch_date": datetime.now().isoformat(),
                "source": "artificial_analysis_api",
                "total_models": stats["total_models"],
                "models_with_gpqa": stats["models_with_gpqa"],
                "coverage_percentage": stats["coverage_percentage"]
            },
            "models": models
        }
        
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n✓ Saved {len(models)} models to {output_file}")
        
        # Show top performers
        models_with_scores = [m for m in models if m.get("gpqa") is not None]
        models_with_scores.sort(key=lambda x: x["gpqa"], reverse=True)
        
        print(f"\nTop 10 GPQA Performers:")
        print(f"{'Rank':<6} {'Score':<8} {'Model':<40} {'Creator':<20}")
        print("-" * 80)
        for i, m in enumerate(models_with_scores[:10], 1):
            score_pct = f"{m['gpqa']*100:.1f}%"
            print(f"{i:<6} {score_pct:<8} {m['name'][:40]:<40} {m['creator_name'][:20]:<20}")
        
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
