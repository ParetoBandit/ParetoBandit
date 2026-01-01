import json
from pathlib import Path
from collections import defaultdict, Counter

def validate_and_backfill():
    """Validate training data completeness and identify missing entries"""
    base = Path(__file__).parent
    rewards_file = base / "data/train_rewards_1k.jsonl"
    prompts_file = base / "data/train_prompts_sampled_1k.jsonl"
    models_file = base / "models.json"
    
    print("=" * 60)
    print("TRAINING DATA VALIDATION & GAP ANALYSIS")
    print("=" * 60)
    
    # Load expected scope
    print("\n[1/4] Loading expected scope...")
    prompts = []
    with open(prompts_file, 'r') as f:
        for line in f:
            prompts.append(json.loads(line))
    
    with open(models_file, 'r') as f:
        registry = json.load(f)
    models = [m["openrouter_id"] for m in registry["models"]]
    
    expected_total = len(prompts) * len(models)
    print(f"  Expected: {len(prompts)} prompts × {len(models)} models = {expected_total} entries")
    
    # Load actual data
    print("\n[2/4] Analyzing actual data...")
    coverage = defaultdict(set)  # prompt_text -> set of models
    failures = []
    successes = 0
    
    with open(rewards_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            prompt = data.get("prompt")
            model = data["model_id"]
            
            if data.get("ok"):
                coverage[prompt].add(model)
                successes += 1
            else:
                failures.append((prompt, model, data.get("cluster_id")))
    
    print(f"  Loaded: {successes} successful entries")
    print(f"  Failures: {len(failures)}")
    
    # Identify gaps
    print("\n[3/4] Identifying gaps...")
    missing = []
    prompt_to_cluster = {p["prompt"]: p["cluster_id"] for p in prompts}
    
    for p in prompts:
        prompt_text = p["prompt"]
        cluster_id = p["cluster_id"]
        
        for model_id in models:
            if model_id not in coverage.get(prompt_text, set()):
                missing.append((cluster_id, prompt_text, model_id))
    
    print(f"  Missing entries: {len(missing)}")
    
    # Coverage by model
    model_missing = Counter(m for _, _, m in missing)
    if model_missing:
        print(f"\n  Missing by model (top 10):")
        for model_id, count in model_missing.most_common(10):
            print(f"    {model_id}: {count}")
    
    # Coverage by prompt
    prompt_complete = sum(1 for p in prompts if len(coverage.get(p["prompt"], set())) == len(models))
    print(f"\n  Prompts with complete coverage: {prompt_complete} / {len(prompts)}")
    
    # Summary
    print("\n[4/4] Summary")
    coverage_pct = (successes / expected_total) * 100
    print(f"  Coverage: {successes} / {expected_total} ({coverage_pct:.2f}%)")
    print(f"  Failures: {len(failures)}")
    print(f"  Gaps: {len(missing)}")
    
    if len(missing) > 0:
        print("\n⚠️  Action required: Run backfill")
        
        # Save missing tasks
        backfill_file = base / "data/train_missing_tasks.json"
        with open(backfill_file, 'w') as f:
            json.dump(missing, f, indent=2)
        print(f"  Saved missing tasks to: {backfill_file.name}")
        print(f"  Run: python backfill_train.py")
        return False
    else:
        print("\n✅ Data is complete! Ready to update models.json")
        print("  Next: python fix_data_leakage.py")
        return True

if __name__ == "__main__":
    validate_and_backfill()
