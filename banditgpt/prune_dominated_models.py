import json
import os

def main():
    # Load the full list of models
    # We backed up the original to models.json.bak and moved the working copy to models_full.json
    # Let's use models.json.bak as the source of truth if available, otherwise models_full.json
    source_file = 'models.json.bak'
    if not os.path.exists(source_file):
        source_file = 'models_full.json'
        
    if not os.path.exists(source_file):
        print(f"Error: Could not find source file {source_file}")
        return

    print(f"Reading from {source_file}...")
    with open(source_file, 'r') as f:
        data = json.load(f)
    
    models = data['models']
    print(f"Total models loaded: {len(models)}")

    # List of openrouter_ids or names to remove
    # Note: 'Nova Pro' is a name, others are IDs or names. We check both.
    dominated_list = [
        "google/gemini-2.5-pro-preview-06-05",
        "meta-llama/llama-3.1-405b-instruct",
        "amazon/nova-lite-v1",
        "google/gemini-2.5-flash-preview-09-2025",
        "google/gemini-3-pro-preview",
        "google/gemini-2.5-flash-lite",
        "Nova Pro"
    ]
    
    filtered_models = []
    removed_count = 0
    
    for m in models:
        name = m.get('name')
        or_id = m.get('openrouter_id')
        display_name = m.get('display_name')
        
        # Check if any identifier matches our dominated list
        is_dominated = False
        if name in dominated_list: is_dominated = True
        if or_id in dominated_list: is_dominated = True
        if display_name in dominated_list: is_dominated = True
        
        if is_dominated:
            print(f"Removing: {name or or_id}")
            removed_count += 1
        else:
            filtered_models.append(m)
            
    print(f"\nRemoved {removed_count} models.")
    print(f"Remaining models: {len(filtered_models)}")
    
    # Save to models.json
    output_data = {"models": filtered_models}
    with open('models.json', 'w') as f:
        json.dump(output_data, f, indent=2)
        
    print("Saved clean list to models.json")

if __name__ == "__main__":
    main()
