import json
from pathlib import Path

def merge_models_config(current_path, full_path):
    print(f"🚀 Merging models from {full_path.name} into {current_path.name}...")
    
    # 1. Load Configurations
    with open(current_path, 'r') as f:
        current_config = json.load(f)
        
    with open(full_path, 'r') as f:
        full_config = json.load(f)
        
    # 2. Index Current Models
    current_models_map = {m["openrouter_id"]: m for m in current_config["models"]}
    initial_count = len(current_models_map)
    print(f"   Current models count: {initial_count}")
    
    # 3. Find and Add Missing Models
    added_count = 0
    for m in full_config["models"]:
        mid = m["openrouter_id"]
        if mid not in current_models_map:
            print(f"   + Adding missing model: {mid}")
            # Append to list
            current_config["models"].append(m)
            # Update index (optional, but good for tracking)
            current_models_map[mid] = m
            added_count += 1
            
    # 4. Save
    if added_count > 0:
        print(f"\n   Saving {len(current_config['models'])} models to {current_path}...")
        with open(current_path, 'w') as f:
            json.dump(current_config, f, indent=2)
        print("   ✓ Update complete!")
    else:
        print("\n   ✓ No missing models found. content is identical regarding model list.")

if __name__ == "__main__":
    base_dir = Path("src/bandit_gpt/config")
    merge_models_config(base_dir / "models.json", base_dir / "models_full.json")
