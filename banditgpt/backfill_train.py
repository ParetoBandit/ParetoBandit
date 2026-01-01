import json
from pathlib import Path
from rejudge_cot import CoTRewardGenerator
from tqdm import tqdm

def backfill_train():
    """Backfill missing entries in training data"""
    base = Path(__file__).parent
    missing_file = base / "data/train_missing_tasks.json"
    output_file = base / "data/train_rewards_1k.jsonl"
    
    print("=" * 60)
    print("BACKFILLING TRAINING DATA GAPS")
    print("=" * 60)
    
    # Load missing tasks
    with open(missing_file, 'r') as f:
        missing = json.load(f)
    
    print(f"\nFound {len(missing)} missing entries to backfill")
    
    # Initialize generator with moderate concurrency
    gen = CoTRewardGenerator(max_workers=10)  # Balance speed vs stability
    
    # Process
    successes = []
    failures = []
    
    for task in tqdm(missing, desc="Backfilling"):
        result = gen.process_task(tuple(task))
        
        if result.get("ok"):
            successes.append(result)
        else:
            failures.append(result)
    
    # Append to file
    if successes:
        print(f"\n✓ Backfilled {len(successes)} entries")
        with open(output_file, 'a') as f:
            for s in successes:
                f.write(json.dumps(s) + '\n')
    
    if failures:
        print(f"⚠️  Still missing after retry: {len(failures)}")
        from collections import Counter
        model_fails = Counter(f["model_id"] for f in failures)
        for mid, count in model_fails.most_common(5):
            print(f"  {mid}: {count}")
    
    print("\n✅ Backfill complete!")
    print("Run: python validate_train_data.py (to verify)")

if __name__ == "__main__":
    backfill_train()
