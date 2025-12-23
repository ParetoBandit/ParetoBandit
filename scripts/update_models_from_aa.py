import json
import csv
import argparse
from pathlib import Path
from typing import List, Dict, Optional

def normalize_name(name: str) -> str:
    """Normalize model name for matching."""
    import re
    # Remove common suffixes/tags in parentheses or as standalone words
    name = re.sub(r'\(.*?\)', '', name)
    name = re.sub(r'\b(instruct|reasoning|thinking|preview|non-reasoning|high|medium|low|codex|vision)\b', '', name, flags=re.IGNORECASE)
    # Remove dates like Nov '24
    name = re.sub(r"\b[A-Za-z]+\s+'\d{2}\b", '', name)
    return name.lower().replace("-", "").replace("_", "").replace(" ", "").replace(".", "").strip()

def update_models(models_path: Path, csv_path: Path, dry_run: bool = False):
    print(f"Loading models from {models_path}")
    with open(models_path, 'r') as f:
        data = json.load(f)
    
    models = data.get("models", [])
    
    print(f"Loading AA data from {csv_path}")
    aa_data = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row['modelName']
            rate = row['omniscienceHallucinationRate']
            aa_data[name] = float(rate)
    
    updated_count = 0
    missing_models = []
    
    for model in models:
        display_name = model.get("display_name", "")
        # Try exact match first
        if display_name in aa_data:
            old_val = model.get("hle", "N/A")
            new_val = aa_data[display_name]
            model["hle"] = new_val
            print(f"✓ {display_name}: {old_val} -> {new_val}")
            updated_count += 1
            continue
            
        # Try normalized match
        found = False
        norm_display = normalize_name(display_name)
        for aa_name, aa_rate in aa_data.items():
            if norm_display == normalize_name(aa_name):
                old_val = model.get("hle", "N/A")
                model["hle"] = aa_rate
                print(f"✓ {display_name} (matched as {aa_name}): {old_val} -> {aa_rate}")
                updated_count += 1
                found = True
                break
        
        if not found:
            # Check if it has a version mismatch (e.g. Gemini 2.0 vs 2.5)
            # For now, just mark as missing
            missing_models.append(display_name)
            print(f"✗ {display_name}: Not found in CSV")

    print("\n" + "="*40)
    print(f"Summary:")
    print(f"Total models in registry: {len(models)}")
    print(f"Updated: {updated_count}")
    print(f"Missing: {len(missing_models)}")
    print("="*40)
    
    if dry_run:
        print("Dry run - changes not saved.")
    else:
        with open(models_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Successfully updated {models_path}")
    
    return missing_models

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, default="final_release/models.json")
    parser.add_argument("--csv", type=str, default="scripts/aa_omniscience_data.csv")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    missing = update_models(project_root / args.models, project_root / args.csv, args.dry_run)
    
    if missing:
        print("\nMissing models that might need manual entry:")
        for m in missing:
            print(f"- {m}")
