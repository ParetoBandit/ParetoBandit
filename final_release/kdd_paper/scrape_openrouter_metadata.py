#!/usr/bin/env python3
"""
Comprehensive OpenRouter Model Scraper

Fetches for all models in models.json:
1. Pricing (from API)
2. Context window length (from model pages)
3. Model description (from model pages)
4. Categories/use-cases (from model pages)
"""

import json
import os
import requests
import time
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Optional

load_dotenv()

def fetch_openrouter_api_data():
    """Fetch model data from OpenRouter API."""
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

def scrape_model_page(model_id: str) -> Optional[Dict]:
    """
    Scrape additional model metadata from OpenRouter model page.
    
    Since OpenRouter model pages are dynamic, we'll use the API's
    extended model information endpoint if available, or parse HTML.
    """
    # Try API first - some models have extended info
    api_key = os.getenv("OPENROUTER_API_KEY")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # OpenRouter model detail page URL (for reference)
    page_url = f"https://openrouter.ai/models/{model_id}"
    
    try:
        # Fetch from API with extended fields
        url = f"https://openrouter.ai/api/v1/models/{model_id}"
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json().get("data", {})
        
        # Extract fields if available
        return {
            "context_length": data.get("context_length"),
            "description": data.get("description"),
            "top_provider": data.get("top_provider", {}).get("name"),
            "architecture": data.get("architecture", {}).get("tokenizer"),
            "pricing_note": data.get("pricing", {}).get("note")
        }
    except Exception as e:
        print(f"  Warning: Could not fetch extended info for {model_id}: {e}")
        return None

def update_models_with_metadata():
    """Update models.json with comprehensive metadata."""
    models_path = Path(__file__).parent.parent / "models.json"
    
    # Load current models
    with open(models_path) as f:
        data = json.load(f)
    
    models = data["models"]
    print(f"Loaded {len(models)} models from models.json\n")
    
    # Fetch API data
    print("Fetching OpenRouter API data...")
    api_models = fetch_openrouter_api_data()
    
    # Create lookup by ID
    api_lookup = {}
    for model in api_models:
        model_id = model["id"]
        api_lookup[model_id] = {
            "input_cost": float(model["pricing"]["prompt"]) * 1_000_000,
            "output_cost": float(model["pricing"]["completion"]) * 1_000_000,
            "context_length": model.get("context_length"),
            "description": model.get("description"),
            "top_provider": model.get("top_provider", {}).get("name") if isinstance(model.get("top_provider"), dict) else None,
            "architecture": model.get("architecture", {}).get("tokenizer") if isinstance(model.get("architecture"), dict) else None
        }
    
    print(f"Fetched data for {len(api_lookup)} models from API\n")
    
    # Update each model
    updated_costs = 0
    updated_context = 0
    updated_description = 0
    
    for i, model_data in enumerate(models, 1):
        openrouter_id = model_data.get("openrouter_id")
        if not openrouter_id:
            continue
        
        print(f"[{i}/{len(models)}] Processing {openrouter_id}...")
        
        if openrouter_id not in api_lookup:
            print(f"  ⚠ Not found in API")
            continue
        
        api_data = api_lookup[openrouter_id]
        
        # Update pricing
        if api_data["input_cost"] and api_data["output_cost"]:
            model_data["input_cost_per_m"] = api_data["input_cost"]
            model_data["output_cost_per_m"] = api_data["output_cost"]
            model_data["price_1m_input"] = api_data["input_cost"]
            model_data["price_1m_output"] = api_data["output_cost"]
            model_data["price_1m_blended"] = (api_data["input_cost"] + api_data["output_cost"]) / 2
            updated_costs += 1
        
        # Update context length
        if api_data["context_length"]:
            old_context = model_data.get("context_length")
            model_data["context_length"] = api_data["context_length"]
            if old_context != api_data["context_length"]:
                print(f"  ✓ Context: {old_context} → {api_data['context_length']:,}")
                updated_context += 1
        
        # Update description
        if api_data["description"]:
            model_data["description"] = api_data["description"]
            print(f"  ✓ Description: {api_data['description'][:60]}...")
            updated_description += 1
        
        # Update provider/architecture
        if api_data["top_provider"]:
            model_data["top_provider"] = api_data["top_provider"]
        if api_data["architecture"]:
            model_data["architecture"] = api_data["architecture"]
        
        time.sleep(0.1)  # Rate limiting
    
    # Backup and save
    backup_path = models_path.with_suffix('.json.backup4')
    if models_path.exists():
        models_path.rename(backup_path)
        print(f"\n✓ Created backup: {backup_path}")
    
    with open(models_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n=== Summary ===")
    print(f"✓ Updated pricing: {updated_costs} models")
    print(f"✓ Updated context: {updated_context} models")
    print(f"✓ Updated description: {updated_description} models")
    print(f"✓ Saved to: {models_path}")

if __name__ == "__main__":
    update_models_with_metadata()
