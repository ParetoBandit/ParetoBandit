import json
from pathlib import Path

def normalize(name):
    return name.lower().replace("-", "").replace("_", "").replace(" ", "")

def restore_metrics():
    project_root = Path("/Users/annette/repostitories/llm_jury")
    models_path = project_root / "final_release/models.json"
    backup_path = project_root / "final_release/models.json.bak"
    cache_with_hle_path = project_root / "banditgpt/data/models_cache_with_hle.json"
    
    # Load current (incorrect) models
    with open(models_path, "r") as f:
        current_data = json.load(f)
    
    # Load backups
    backups = []
    if backup_path.exists():
        with open(backup_path, "r") as f:
            backups.append(json.load(f))
    
    if cache_with_hle_path.exists():
        with open(cache_with_hle_path, "r") as f:
            backups.append(json.load(f))
    
    # Map for original HLE scores (from all backups)
    # Strategy: id match first, then normalized name match
    id_map = {}
    name_map = {}
    
    for bdata in backups:
        for m in bdata.get("models", []):
            oid = m.get("openrouter_id")
            val = m.get("hle")
            if val is not None:
                if oid:
                    id_map[oid] = val
                name = m.get("display_name") or m.get("name")
                if name:
                    name_map[normalize(name)] = val

    updated_count = 0
    restored_count = 0
    for model in current_data.get("models", []):
        oid = model.get("openrouter_id")
        d_name = model.get("display_name", "")
        
        # 1. Capture current hallucination rate
        current_hle_field = model.get("hle")
        
        # 2. Restore original HLE Reasoning score
        restored_val = id_map.get(oid)
        if restored_val is None:
            restored_val = name_map.get(normalize(d_name))
        
        if restored_val is not None:
            model["hle"] = restored_val
            restored_count += 1
            print(f"Restored: {d_name} -> HLE {restored_val}")
        else:
            # If still None and it's a known model type we added
            if any(x in d_name for x in ["Claude 3.7", "Grok 3", "Claude 4.5", "GPT-5"]):
                # Check if we have it in current_hle_field but it's likely a hallucination rate
                # We should leave hle as None or find a better source
                if model.get("hle") is not None and model.get("hle") > 0.1: # Likely hallucination rate
                     model["hle"] = None
                     print(f"Cleared incorrect HLE for new model: {d_name}")
            else:
                print(f"WARNING: Could not find original HLE for {d_name}")

        # 3. Ensure hallucination_rate is correctly populated as a percentage
        # If current_hle_field was updated by me, it's 0-1.0
        if current_hle_field is not None and 0 <= current_hle_field <= 1.0:
             model["hallucination_rate"] = round(current_hle_field * 100, 2)
    
    # Save results
    with open(models_path, "w") as f:
        json.dump(current_data, f, indent=2)
    
    print(f"Restored {restored_count} models. Successfully restored HLE scores and maintained hallucination rates.")

if __name__ == "__main__":
    restore_metrics()
