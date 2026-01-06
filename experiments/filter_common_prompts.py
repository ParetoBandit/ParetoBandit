import json
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

def filter_dataset(input_files, output_file, expected_count=43):
    print(f"🚀 Processing {output_file.name}...")
    
    # 1. Load Data
    data_by_prompt = defaultdict(list)
    total_lines = 0
    
    for filepath in input_files:
        print(f"   Reading {filepath}...")
        with open(filepath, 'r') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    # Use cluster_id + prompt text as unique key if possible, or just prompt
                    # But prompt text is safer if cluster_ids differ across files (unlikely but possible)
                    prompt_text = record.get("prompt")
                    model_id = record.get("model_id")
                    
                    if prompt_text and model_id:
                        data_by_prompt[prompt_text].append(record)
                        total_lines += 1
                except:
                    continue
                    
    print(f"   Loaded {total_lines} records across {len(data_by_prompt)} unique prompts.")

    # 2. Filter
    valid_prompts = []
    
    for prompt, records in data_by_prompt.items():
        # Deduplicate by model_id for this prompt
        unique_models = {}
        for r in records:
            unique_models[r["model_id"]] = r
            
        if len(unique_models) == expected_count:
            # We have exactly the expected number of models!
            for m_id, record in unique_models.items():
                valid_prompts.append(record)
                
    print(f"   Found {len(valid_prompts) // expected_count} prompts with full {expected_count}-model coverage.")
    
    # 3. Save
    print(f"   Saving {len(valid_prompts)} records to {output_file}...")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Sort for deterministic output (by cluster_id, then prompt, then model)
    valid_prompts.sort(key=lambda x: (x.get("cluster_id", 0), x.get("prompt", ""), x.get("model_id", "")))
    
    with open(output_file, 'w') as f:
        for record in valid_prompts:
            f.write(json.dumps(record) + "\n")
            
    print("   ✓ Done!\n")

if __name__ == "__main__":
    base_dir = Path("src/bandit_gpt/data")
    
    # 1. Training Set Intersection
    # Combine original 1k + missing 7 models
    train_files = [
        base_dir / "train_rewards_1k.jsonl",
        base_dir / "offline_dataset/train_rewards_missing_7models_final.jsonl"
    ]
    train_out = base_dir / "train_rewards_43models.jsonl"
    
    filter_dataset(train_files, train_out, expected_count=43)
    
    # 2. Test Set Intersection
    # Use the backup file which is known to be the source of truth
    test_files = [
        base_dir / "offline_dataset/test_rewards_pareto_dedup_backup.jsonl"
    ]
    test_out = base_dir / "test_rewards_43models.jsonl"
    
    filter_dataset(test_files, test_out, expected_count=43)
