import os
import json
import requests
from pathlib import Path
from dotenv import load_dotenv

def main():
    # Load .env from project root
    project_root = Path("/Users/annette/repostitories/llm_jury")
    load_dotenv(project_root / ".env")
    
    api_key = os.environ.get("AA_API_KEY") or os.environ.get("ARTIFICIAL_ANALYSIS_API_KEY")
    if not api_key:
        print("ERROR: API key not found in .env")
        return
    
    # Load local models.json
    models_path = project_root / "final_release/models.json"
    with open(models_path, "r") as f:
        local_data = json.load(f)
    
    local_models = {m.get("openrouter_id"): m for m in local_data.get("models", [])}
    
    print("Fetching live models from Artificial Analysis...")
    headers = {"x-api-key": api_key}
    url = "https://artificialanalysis.ai/api/v2/data/llms/models"
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        api_models = response.json()
        if not isinstance(api_models, list):
            api_models = api_models.get("models", api_models.get("data", []))
            
        print(f"Fetched {len(api_models)} models from API.")
        
        # Create lookup for API models
        # We'll try to match by name or slug since API might not have openrouter_id directly
        api_lookup = {}
        for m in api_models:
            name = m.get("name") or m.get("model_name")
            evals = m.get("evaluations", {})
            hle = evals.get("hle")
            if name:
                api_lookup[name] = hle

        discrepancies = []
        matched_count = 0
        
        print("\nComparing HLE scores...")
        print(f"{'Model Name':<50} | {'Local HLE':<10} | {'API HLE':<10} | {'Diff'}")
        print("-" * 85)
        
        for oid, m in local_models.items():
            local_hle = m.get("hle")
            display_name = m.get("display_name") or m.get("name")
            
            # Try to find in API lookup
            api_hle = api_lookup.get(display_name)
            
            # Fuzzy match if exact name fails
            if api_hle is None:
                for api_name, score in api_lookup.items():
                    if display_name in api_name or api_name in display_name:
                        api_hle = score
                        break
            
            if api_hle is not None:
                matched_count += 1
                diff = abs(float(local_hle or 0) - float(api_hle or 0))
                if diff > 1e-5:
                    discrepancies.append({
                        "name": display_name,
                        "local": local_hle,
                        "api": api_hle,
                        "diff": diff
                    })
                    print(f"{display_name[:50]:<50} | {local_hle:<10.4f} | {api_hle:<10.4f} | {diff:.4f} !!!")
                else:
                    # print(f"{display_name[:50]:<50} | {local_hle:<10.4f} | {api_hle:<10.4f} | {diff:.4f}")
                    pass
            else:
                # print(f"{display_name[:50]:<50} | {local_hle:<10.4f} | {'NOT FOUND':<10} | -")
                pass

        print("\n" + "=" * 80)
        print("Validation Summary")
        print("=" * 80)
        print(f"Total local models: {len(local_models)}")
        print(f"Models matched in API: {matched_count}")
        print(f"Discrepancies found: {len(discrepancies)}")
        
        if discrepancies:
            print("\nList of discrepancies:")
            for d in discrepancies:
                print(f" - {d['name']}: Local={d['local']}, API={d['api']} (Diff: {d['diff']:.4f})")
        else:
            print("\n✓ All matched models have identical HLE scores!")
            
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    main()
