import json
from pathlib import Path

def filter_models_hle(config_path):
    print(f"🚀 Filtering {config_path.name} to keep only models with HLE data...")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    original_count = len(config["models"])
    
    # Filter: Keep if 'hle' exists and is not None
    filtered_models = [m for m in config["models"] if m.get("hle") is not None]
    
    filtered_count = len(filtered_models)
    
    print(f"   Original count: {original_count}")
    print(f"   Filtered count: {filtered_count}")
    
    removed_count = original_count - filtered_count
    if removed_count > 0:
        print(f"   Removed {removed_count} models without HLE data.")
        
        config["models"] = filtered_models
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
            
        print(f"   ✓ Saved updated config to {config_path}")
        
        if filtered_count == 39:
            print(f"   ✅ SUCCESS: Confirmed exactly 39 models remaining.")
        else:
            print(f"   ⚠️ WARNING: Expected 39 models, but found {filtered_count}.")
    else:
        print("   No models removed. All existing models have HLE data.")

if __name__ == "__main__":
    config_path = Path("src/bandit_gpt/config/models.json")
    filter_models_hle(config_path)
