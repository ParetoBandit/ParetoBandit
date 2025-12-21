#!/usr/bin/env python3
"""
Validate Benchmark Data Against Artificial Analysis

This script fetches the latest benchmark data from Artificial Analysis
and compares it to the values in models_cache.json to ensure accuracy.

Usage:
    python kdd_paper/scripts/validate_benchmarks.py
    python kdd_paper/scripts/validate_benchmarks.py --update  # Auto-fix discrepancies
"""

import json
import re
import sys
import argparse
from pathlib import Path

import requests

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_CACHE = PROJECT_ROOT / "banditgpt" / "data" / "models_cache.json"

# Tolerance for float comparison (percentage points)
TOLERANCE = 2.0


def fetch_aa_data():
    """Fetch model data from Artificial Analysis leaderboard."""
    print("📡 Fetching data from Artificial Analysis...")
    url = "https://artificialanalysis.ai/leaderboards/models"
    
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        print(f"   Received {len(resp.text):,} bytes")
        return resp.text
    except Exception as e:
        print(f"❌ Error fetching AA data: {e}")
        return None


def extract_model_benchmarks(aa_text, search_term):
    """Extract benchmark data by searching for a term in AA data."""
    idx = aa_text.find(search_term)
    if idx == -1:
        return None
    
    # Get context around the match
    context = aa_text[max(0, idx-200):idx+1500]
    
    data = {}
    
    # Extract fields - use escaped quotes since AA data has \"field\":value format
    fields = ["intelligence_index", "gpqa", "humaneval", "math_500", "mmlu_pro", "ifbench"]
    
    for field in fields:
        # Try both escaped and unescaped quote patterns
        for pattern in [
            field + r'["\s:]+(\d+\.?\d*)',  # Basic pattern
            field + r'\\?"\\?:(\d+\.?\d*)',  # Escaped quotes
            field + r'":(\d+\.?\d*)',  # Direct pattern
        ]:
            match = re.search(pattern, context)
            if match:
                try:
                    data[field] = float(match.group(1))
                    break
                except ValueError:
                    continue
    
    return data if data else None


def validate_and_report():
    """Main validation logic."""
    
    # Fetch AA data
    aa_text = fetch_aa_data()
    if not aa_text:
        return
    
    # Load our cache
    print(f"\n📂 Loading models_cache.json...")
    with open(MODELS_CACHE) as f:
        cache = json.load(f)
    
    # Define key models and their AA search terms
    # Multiple search terms per model to increase chance of finding
    key_models = {
        "deepseek/deepseek-chat-v3-0324": ["deepseek-v3", "deepseek-chat-v3"],
        "openai/gpt-4o": ['gpt-4o"', "gpt-4o-2024"],
        "openai/gpt-4o-mini": ['gpt-4o-mini"', "gpt-4o-mini-2024"],
    }
    
    print("\n" + "=" * 70)
    print("BENCHMARK VALIDATION REPORT")
    print("=" * 70)
    
    discrepancies = []
    
    for openrouter_id, aa_searches in key_models.items():
        # Get our data
        our_model = None
        for m in cache.get("models", []):
            if m.get("openrouter_id") == openrouter_id:
                our_model = m
                break
        
        if not our_model:
            print(f"\n⚠️  {openrouter_id}: NOT IN CACHE")
            continue
        
        # Get AA data - try multiple search terms
        aa_data = None
        used_search = None
        for aa_search in aa_searches:
            aa_data = extract_model_benchmarks(aa_text, aa_search)
            if aa_data and aa_data.get("intelligence_index"):
                used_search = aa_search
                break
        
        print(f"\n{openrouter_id}:")
        print(f"  AA search term: '{used_search or aa_searches[0]}'")
        
        if not aa_data:
            print(f"  ⚠️  NOT FOUND IN AA DATA")
            continue
        
        # Compare reasoning_score vs intelligence_index
        our_reasoning = our_model.get("reasoning_score")
        aa_intel = aa_data.get("intelligence_index")
        
        print(f"  reasoning_score (ours): {our_reasoning}")
        print(f"  intelligence_index (AA): {aa_intel}")
        
        if our_reasoning is not None and aa_intel is not None:
            diff = abs(float(our_reasoning) - float(aa_intel))
            if diff > TOLERANCE:
                print(f"  ❌ DISCREPANCY: diff = {diff:.2f} pts")
                discrepancies.append({
                    "model": openrouter_id,
                    "field": "reasoning_score",
                    "ours": our_reasoning,
                    "aa": aa_intel,
                })
            else:
                print(f"  ✅ OK (diff = {diff:.2f} pts)")
        
        # Show other AA fields for reference
        for field in ["gpqa", "humaneval", "math_500"]:
            if field in aa_data:
                val = aa_data[field]
                display = f"{val:.3f} ({val*100:.1f}%)" if val <= 1 else f"{val}"
                print(f"  {field} (AA): {display}")
    
    print("\n" + "=" * 70)
    if discrepancies:
        print(f"❌ {len(discrepancies)} DISCREPANCIES FOUND")
        print("\nTo fix, update models_cache.json with these values:")
        for d in discrepancies:
            print(f"  {d['model']}: {d['field']} = {d['aa']}")
    else:
        print("✅ ALL KEY MODELS VALIDATED!")
    print("=" * 70)
    
    return discrepancies


def main():
    parser = argparse.ArgumentParser(description="Validate benchmark data against Artificial Analysis")
    parser.add_argument("--update", action="store_true", help="Auto-update discrepancies")
    args = parser.parse_args()
    
    discrepancies = validate_and_report()
    
    if args.update and discrepancies:
        print("\n💾 Updating models_cache.json...")
        with open(MODELS_CACHE) as f:
            cache = json.load(f)
        
        for d in discrepancies:
            for m in cache.get("models", []):
                if m.get("openrouter_id") == d["model"]:
                    m[d["field"]] = d["aa"]
                    print(f"   Updated {d['model']}: {d['field']} = {d['aa']}")
        
        with open(MODELS_CACHE, "w") as f:
            json.dump(cache, f, indent=2)
        print("   Done!")


if __name__ == "__main__":
    main()
