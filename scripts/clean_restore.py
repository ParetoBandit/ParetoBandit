import json
from pathlib import Path

def normalize(name):
    return name.lower().replace("-", "").replace("_", "").replace(" ", "")

def clean_restore():
    project_root = Path("/Users/annette/repostitories/llm_jury")
    models_path = project_root / "final_release/models.json"
    backup_path = project_root / "final_release/models.json.bak"
    cache_with_hle_path = project_root / "banditgpt/data/models_cache_with_hle.json"
    
    # Load current models
    with open(models_path, "r") as f:
        models_data = json.load(f)
    
    # Load backups
    backups = []
    if backup_path.exists():
        with open(backup_path, "r") as f:
            backups.append(json.load(f))
    if cache_with_hle_path.exists():
        with open(cache_with_hle_path, "r") as f:
            backups.append(json.load(f))
    
    # Build recovery maps
    id_map = {}
    name_map = {}
    for bdata in backups:
        for m in bdata.get("models", []):
            oid = m.get("openrouter_id")
            val = m.get("hle")
            if val is not None:
                if oid: id_map[oid] = val
                name = m.get("display_name") or m.get("name")
                if name: name_map[normalize(name)] = val

    # RESTORE PHASE
    print("--- Restoring HLE Reasoning Scores ---")
    current_models = models_data.get("models", [])
    restored_count = 0
    cleared_halluc_count = 0
    
    for model in current_models:
        oid = model.get("openrouter_id")
        d_name = model.get("display_name", "")
        
        # 1. Restore Reasoning HLE
        restored_val = id_map.get(oid)
        if restored_val is None:
            restored_val = name_map.get(normalize(d_name))
        
        if restored_val is not None:
            model["hle"] = restored_val
            restored_count += 1
        else:
            # If still None, it's a model added recently that wasn't in backup
            # We clear HLE if it's currently a hallucination rate (likely > 0.1 for reasoning-tier)
            # Actually, most HLE scores are < 0.3, while many hallucination rates are > 0.5.
            # Best to set to None if we aren't sure.
            model["hle"] = None
        
        # 2. CLEAR Hallucination Rates (Blank Slate)
        if "hallucination_rate" in model:
            del model["hallucination_rate"]
            cleared_halluc_count += 1

    # Save reset state
    with open(models_path, "w") as f:
        json.dump(models_data, f, indent=2)
    
    print(f"Restored {restored_count} HLE scores.")
    print(f"Cleared {cleared_halluc_count} hallucination rates.")
    print("Models are now in a CLEAN state. Proceeding to re-run update scripts...")

if __name__ == "__main__":
    clean_restore()
