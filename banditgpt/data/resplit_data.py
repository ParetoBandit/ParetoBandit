
import json
import numpy as np
from pathlib import Path
import random

def main():
    base_dir = Path(__file__).parent
    source_path = base_dir / "lmsys_all_prompts.jsonl"
    
    # 1. Load UNIQUE prompts only
    print(f"Loading from {source_path}...")
    unique_prompts = set()
    train_prompts_text = set() # To preserve current train set if needed, or we just overwrite
    
    # We will exclude the OLD train/test prompts if we want to be safe, 
    # but actually the user wants a clean split from the 26k unique.
    # So let's load ALL unique prompts first.
    
    raw_rows = []
    with open(source_path) as f:
        for line in f:
            try:
                data = json.loads(line)
                if isinstance(data, dict):
                    text = data.get("prompt") or data.get("text") or data.get("content")
                else:
                    text = str(data)
                
                if text:
                    text = text.strip()
                    if text not in unique_prompts:
                        unique_prompts.add(text)
                        # Create a standard object
                        raw_rows.append({"prompt": text})
            except:
                pass
                
    print(f"Total Unique Prompts: {len(raw_rows)}")
    
    # 2. Shuffle Deterministically
    random.seed(42)
    random.shuffle(raw_rows)
    
    # 3. Split: 1k Test, 4k Train, Rest for Prior
    TEST_SIZE = 1000
    TRAIN_SIZE = 4000
    
    test_set = raw_rows[:TEST_SIZE]
    train_set = raw_rows[TEST_SIZE:TEST_SIZE + TRAIN_SIZE]
    prior_set = raw_rows[TEST_SIZE + TRAIN_SIZE:]
    
    print(f"Test Set: {len(test_set)}")
    print(f"Train Set: {len(train_set)}")
    print(f"Prior Set: {len(prior_set)}")
    
    # 4. Save Test Set
    test_path = base_dir / "test_prompts.jsonl"
    with open(test_path, 'w') as f:
        for item in test_set:
            f.write(json.dumps(item) + "\n")
    print(f"Saved {test_path}")
    
    # 5. Save Train Set
    train_path = base_dir / "train_prompts.jsonl"
    with open(train_path, 'w') as f:
        for item in train_set:
            f.write(json.dumps(item) + "\n")
    print(f"Saved {train_path}")
    
    # 5. Note for User
    print("\nNext Steps:")
    print("1. Run `python3 final_release/data/create_clusters.py` to label the new test set.")
    print("2. Run `python3 final_release/calc_priors_large.py` to regenerate priors (it will automatically exclude the new test set).")

if __name__ == "__main__":
    main()
