import json
from pathlib import Path

def main():
    base_dir = Path("/Users/annette/repostitories/llm_jury")
    final_path = base_dir / "final_release/models.json"
    cache_path = base_dir / "banditgpt/data/models_cache_with_hle.json"
    
    print(f"Loading final models from {final_path}...")
    with open(final_path, "r") as f:
        final_data = json.load(f)
        
    print(f"Loading HLE cache from {cache_path}...")
    with open(cache_path, "r") as f:
        cache_data = json.load(f)
        
    # Create a lookup for HLE scores
    hle_lookup = {}
    for m in cache_data.get("models", []):
        oid = m.get("openrouter_id")
        hle = m.get("hle")
        if oid and hle is not None:
            hle_lookup[oid] = hle
            
    # Merge HLE into final models
    merged_count = 0
    for m in final_data.get("models", []):
        oid = m.get("openrouter_id")
        if oid in hle_lookup:
            m["hle"] = hle_lookup[oid]
            merged_count += 1
        else:
            # Fallback to 0 if not found
            m["hle"] = 0.0
            
    print(f"Merged HLE scores for {merged_count}/{len(final_data['models'])} models.")
    
    # Save back
    with open(final_path, "w") as f:
        json.dump(final_data, f, indent=2)
    print("Successfully updated models.json")

if __name__ == "__main__":
    main()
