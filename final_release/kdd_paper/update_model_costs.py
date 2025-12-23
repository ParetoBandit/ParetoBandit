#!/usr/bin/env python3
"""
Update model costs from OpenRouter API
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

def update_models_json():
    """Update models.json with latest OpenRouter pricing."""
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
            "input_cost": float(model["pricing"]["prompt"]) * 1_000_000,  # Convert to per-1M
            "output_cost": float(model["pricing"]["completion"]) * 1_000_000,
        }
    
    # Update models
    updated_count = 0
    for model_data in models:
        openrouter_id = model_data.get("openrouter_id")
        if openrouter_id and openrouter_id in pricing_lookup:
            pricing = pricing_lookup[openrouter_id]
            
            # Update costs
            old_input = model_data.get("input_cost_per_m", 0)
            old_output = model_data.get("output_cost_per_m", 0)
            
            model_data["input_cost_per_m"] = pricing["input_cost"]
            model_data["output_cost_per_m"] = pricing["output_cost"]
            model_data["price_1m_input"] = pricing["input_cost"]
            model_data["price_1m_output"] = pricing["output_cost"]
            model_data["price_1m_blended"] = (pricing["input_cost"] + pricing["output_cost"]) / 2
            
            if old_input != pricing["input_cost"] or old_output != pricing["output_cost"]:
                print(f"Updated {openrouter_id}:")
                print(f"  Input: ${old_input:.4f} -> ${pricing['input_cost']:.4f}")
                print(f"  Output: ${old_output:.4f} -> ${pricing['output_cost']:.4f}")
                updated_count += 1
    
    # Save updated models
    with open(models_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n✓ Updated {updated_count} models")
    print(f"✓ Saved to {models_path}")

if __name__ == "__main__":
    update_models_json()
