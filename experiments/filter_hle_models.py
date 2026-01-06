import json
from pathlib import Path
from collections import defaultdict

def filter_by_hle(models_file, train_in, test_in, train_out, test_out):
    print(f"🚀 Filtering for HLE models...")
    
    # 1. Identify HLE models
    with open(models_file, 'r') as f:
        config = json.load(f)
        
    hle_models = []
    print("   Scanning models.json...")
    for m in config["models"]:
        # Check if 'hle' exists and is not None. 
        # Some models might have hle=0.0 which is valid? 
        # The prompt said "have HLE data available".
        if "hle" in m and m["hle"] is not None:
            hle_models.append(m["openrouter_id"])
            
    print(f"   Found {len(hle_models)} models with HLE data: {hle_models}")
    
    # 2. Filter Datasets
    for infile, outfile in [(train_in, train_out), (test_in, test_out)]:
        print(f"\n   Processing {outfile.name}...")
        
        data_by_prompt = defaultdict(list)
        total_lines = 0
        
        with open(infile, 'r') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    model_id = record.get("model_id")
                    
                    if model_id in hle_models:
                        prompt_text = record.get("prompt")
                        if prompt_text:
                            data_by_prompt[prompt_text].append(record)
                            total_lines += 1
                except:
                    continue
        
        print(f"   Loaded {total_lines} records for HLE models.")
        
        # 3. Intersection (All HLE models must be present)
        valid_records = []
        expected_count = len(hle_models)
        
        for prompt, records in data_by_prompt.items():
            # Deduplicate by model_id
            unique_models = {}
            for r in records:
                unique_models[r["model_id"]] = r
                
            if len(unique_models) == expected_count:
                for m_id, record in unique_models.items():
                    valid_records.append(record)
        
        # Sort
        valid_records.sort(key=lambda x: (x.get("cluster_id", 0), x.get("prompt", ""), x.get("model_id", "")))
        
        # Save
        outfile.parent.mkdir(parents=True, exist_ok=True)
        with open(outfile, 'w') as f:
            for record in valid_records:
                f.write(json.dumps(record) + "\n")
                
        print(f"   Saved {len(valid_records)} records ({len(valid_records)//expected_count} prompts) to {outfile}")
        
    print("\n   ✓ Done!")

if __name__ == "__main__":
    base_dir = Path("src/bandit_gpt/data")
    
    models_path = Path("src/bandit_gpt/config/models_full.json")
    
    train_in = base_dir / "train_rewards_43models.jsonl"
    test_in = base_dir / "test_rewards_43models.jsonl"
    
    train_out = base_dir / "train_rewards_hle_models.jsonl"
    test_out = base_dir / "test_rewards_hle_models.jsonl"
    
    filter_by_hle(models_path, train_in, test_in, train_out, test_out)
