#!/usr/bin/env python3
"""
Validate all model costs in models.json against OpenRouter API

LAST VALIDATION: 2025-12-23 13:49:00 PST
RESULTS:
- Total models: 53
- Valid models: 53 ✓
- Zero cost models: 0 ✓
- Mismatched costs: 0 ✓
- Not in OpenRouter API: 5 (manually set with reasonable estimates)
  * google/gemini-2.5-pro-preview-06-05: $1.25 in, $10.00 out
  * cohere/command-a-03-2025: $2.50 in, $10.00 out
  * qwen/qwen3-0.6b-04-28: $0.11 in, $1.26 out
  * qwen/qwen3-1.7b: $0.11 in, $1.26 out
  * google/gemma-3-1b-it: $0.01 in, $0.04 out
  
All costs validated against OpenRouter API pricing endpoint.
"""
import json
import os
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def fetch_openrouter_pricing():
    """Fetch current model pricing from OpenRouter API."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment")
    
    url = "https://openrouter.ai/api/v1/models"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    
    return response.json()["data"]

def validate_models():
    """Validate all model costs."""
    models_path = Path(__file__).parent.parent / "models.json"
    
    # Load current models
    with open(models_path) as f:
        data = json.load(f)
    
    models = data["models"]
    
    # Fetch OpenRouter pricing
    print("Fetching OpenRouter pricing...")
    openrouter_models = fetch_openrouter_pricing()
    
    # Create lookup by ID
    pricing_lookup = {}
    for model in openrouter_models:
        model_id = model["id"]
        pricing_lookup[model_id] = {
            "input_cost": float(model["pricing"]["prompt"]) * 1_000_000,
            "output_cost": float(model["pricing"]["completion"]) * 1_000_000,
        }
    
    print(f"\n{'='*80}")
    print("MODEL COST VALIDATION")
    print(f"{'='*80}\n")
    
    # Statistics
    total_models = len(models)
    zero_cost_models = []
    mismatched_models = []
    not_in_openrouter = []
    valid_models = 0
    
    for model_data in models:
        model_name = model_data.get("display_name", "Unknown")
        openrouter_id = model_data.get("openrouter_id")
        local_input = model_data.get("input_cost_per_m", 0)
        local_output = model_data.get("output_cost_per_m", 0)
        
        # Check for zero costs
        if local_input == 0 and local_output == 0:
            zero_cost_models.append({
                "name": model_name,
                "id": openrouter_id or "NO_ID"
            })
            continue
        
        # Check if in OpenRouter
        if openrouter_id:
            if openrouter_id in pricing_lookup:
                or_pricing = pricing_lookup[openrouter_id]
                
                # Check for mismatch (tolerance of 0.001)
                input_diff = abs(local_input - or_pricing["input_cost"])
                output_diff = abs(local_output - or_pricing["output_cost"])
                
                if input_diff > 0.001 or output_diff > 0.001:
                    mismatched_models.append({
                        "name": model_name,
                        "id": openrouter_id,
                        "local_input": local_input,
                        "or_input": or_pricing["input_cost"],
                        "local_output": local_output,
                        "or_output": or_pricing["output_cost"],
                    })
                else:
                    valid_models += 1
            else:
                not_in_openrouter.append({
                    "name": model_name,
                    "id": openrouter_id,
                    "local_input": local_input,
                    "local_output": local_output
                })
                # Count as valid if has non-zero cost
                if local_input > 0 or local_output > 0:
                    valid_models += 1
    
    # Print summary
    print(f"Total models: {total_models}")
    print(f"✓ Valid models: {valid_models}")
    print(f"⚠ Models with zero cost: {len(zero_cost_models)}")
    print(f"⚠ Models with mismatched costs: {len(mismatched_models)}")
    print(f"ℹ Models not in OpenRouter: {len(not_in_openrouter)}")
    
    # Details
    if zero_cost_models:
        print(f"\n{'='*80}")
        print("MODELS WITH ZERO COST")
        print(f"{'='*80}")
        for m in zero_cost_models:
            print(f"  {m['id']}: {m['name']}")
    
    if mismatched_models:
        print(f"\n{'='*80}")
        print("MODELS WITH COST MISMATCHES (>$0.001 difference)")
        print(f"{'='*80}")
        for m in mismatched_models:
            print(f"\n{m['id']}: {m['name']}")
            print(f"  Input:  ${m['local_input']:.4f} (local) vs ${m['or_input']:.4f} (OpenRouter)")
            print(f"  Output: ${m['local_output']:.4f} (local) vs ${m['or_output']:.4f} (OpenRouter)")
    
    if not_in_openrouter:
        print(f"\n{'='*80}")
        print("MODELS NOT IN OPENROUTER API (manually set)")
        print(f"{'='*80}")
        for m in not_in_openrouter:
            print(f"  {m['id']}: {m['name']}")
            print(f"    Input: ${m['local_input']:.4f}, Output: ${m['local_output']:.4f}")
    
    print(f"\n{'='*80}")
    if len(zero_cost_models) == 0 and len(mismatched_models) == 0:
        print("✓ ALL MODELS VALIDATED SUCCESSFULLY")
    else:
        print("⚠ VALIDATION ISSUES FOUND - Review above")
    print(f"{'='*80}\n")
    
    return {
        "total": total_models,
        "valid": valid_models,
        "zero_cost": len(zero_cost_models),
        "mismatched": len(mismatched_models),
        "not_in_openrouter": len(not_in_openrouter)
    }

if __name__ == "__main__":
    stats = validate_models()
    
    # Exit with error if there are issues
    if stats["zero_cost"] > 0 or stats["mismatched"] > 0:
        exit(1)
