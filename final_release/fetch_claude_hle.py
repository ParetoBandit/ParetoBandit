import os
import json
import requests
from pathlib import Path
from dotenv import load_dotenv

def main():
    # Load .env from project root
    project_root = Path(__file__).parent.parent
    load_dotenv(project_root / ".env")
    
    api_key = os.environ.get("AA_API_KEY") or os.environ.get("ARTIFICIAL_ANALYSIS_API_KEY")
    if not api_key:
        print("ERROR: API key not found in .env")
        return
    
    print("Fetching models from Artificial Analysis...")
    headers = {"x-api-key": api_key}
    url = "https://artificialanalysis.ai/api/v2/data/llms/models"
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        models = data if isinstance(data, list) else data.get("models", data.get("data", []))
        if models:
            print(f"Fetched {len(models)} models.")
            print(f"First model keys: {list(models[0].keys())}")
            print(f"First model name: {models[0].get('name') or models[0].get('model_name')}")
        
        target_name = "Claude 3.7 Sonnet (Reasoning)"
        found_model = None
        
        for m in models:
            name = m.get("name") or m.get("model_name")
            if name == target_name:
                found_model = m
                break
        
        if not found_model:
            # Try fuzzy match
            for m in models:
                name = m.get("name") or m.get("model_name", "")
                if name and "Claude 3.7 Sonnet" in name and "Reasoning" in name:
                    found_model = m
                    break
                    
        if found_model:
            name = found_model.get("name") or found_model.get("model_name")
            print(f"Found model: {name}")
            evals = found_model.get("evaluations", {})
            hle = evals.get("hle")
            print(f"HLE Score: {hle}")
            
            if hle is not None:
                # Update models.json
                models_path = project_root / "final_release/models.json"
                with open(models_path, "r") as f:
                    final_data = json.load(f)
                
                updated = False
                for model in final_data.get("models", []):
                    if model.get("openrouter_id") == "anthropic/claude-3.7-sonnet:thinking":
                        model["hle"] = float(hle)
                        updated = True
                        break
                
                if updated:
                    with open(models_path, "w") as f:
                        json.dump(final_data, f, indent=2)
                    print(f"Successfully updated {models_path}")
                else:
                    print("ERROR: Could not find Claude 3.7 in models.json")
            else:
                print("ERROR: HLE score not found for this model in API response")
        else:
            print(f"ERROR: Could not find '{target_name}' in API response")
            # Print a few model names for debugging
            print("Available models (first 10):")
            for m in models[:10]:
                name = m.get("name") or m.get("model_name")
                print(f" - {name}")
                
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    main()
