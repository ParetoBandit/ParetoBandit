import json
from pathlib import Path

def main():
    # Paths
    source_path = Path(__file__).parent / "data/models_cache_with_hle.json"
    dest_path = Path(__file__).parent / "models.json"
    
    # Load
    with open(source_path) as f:
        data = json.load(f)
        
    # Filter
    filtered_models = []
    for m in data["models"]:
        hle = m.get("hle")
        if hle is not None and hle > 0:
            filtered_models.append(m)
            
    # Save
    output_data = {"models": filtered_models}
    with open(dest_path, "w") as f:
        json.dump(output_data, f, indent=2)
        
    print(f"Filtered {len(data['models'])} models down to {len(filtered_models)} with HLE scores.")
    print(f"Saved to {dest_path}")

if __name__ == "__main__":
    main()
