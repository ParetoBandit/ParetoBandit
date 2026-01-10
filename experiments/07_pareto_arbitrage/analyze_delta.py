#!/usr/bin/env python3
"""
Delta Analysis: Find prompts where Max Quality beats gpt-oss.
This reveals if there's a learnable classification boundary for arbitrage.
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.utils.data_loader import load_oracle_rewards, load_model_registry

def main():
    print("="*70)
    print("DELTA ANALYSIS: Max Quality vs gpt-oss")
    print("="*70)
    
    # Load data
    train_rewards = load_oracle_rewards("lmsys_train_final_rewards_1k_clean.jsonl.gz")
    test_rewards = load_oracle_rewards("lmsys_test_final_rewards_1k_clean.jsonl.gz")
    all_rewards = {**train_rewards, **test_rewards}
    
    # Load registry
    registry = load_model_registry()
    
    # Load test split
    splits_path = Path(__file__).parent.parent / "01_effectiveness" / "results" / "splits.json"
    with open(splits_path) as f:
        splits = json.load(f)
    test_prompts = splits["holdout_pool"]
    
    # Find hard prompts (using corrected filter)
    hard_prompts = []
    for prompt in test_prompts:
        oracle_scores = [all_rewards.get(prompt, {}).get(mid) 
                       for mid in registry.keys() 
                       if all_rewards.get(prompt, {}).get(mid) is not None]
        if oracle_scores:
            failures = sum(1 for r in oracle_scores if r == 0.0)
            is_solvable = max(oracle_scores) == 1.0
            if failures >= 3 and is_solvable:
                hard_prompts.append(prompt)
    
    print(f"\nHard Prompts: {len(hard_prompts)}")
    
    # Compare gpt-oss vs best models
    gpt_oss_id = "openai/gpt-oss-120b"
    high_quality_models = ["openai/gpt-5.1", "openai/gpt-5-chat", 
                          "anthropic/claude-opus-4.5", "google/gemini-3-pro-preview"]
    
    # Find the delta
    gpt_oss_fails_but_solvable = []
    
    for prompt in hard_prompts:
        rewards = all_rewards.get(prompt, {})
        gpt_oss_reward = rewards.get(gpt_oss_id, 0.0)
        
        # Check if any high-quality model succeeded
        any_hq_success = any(rewards.get(mid, 0.0) == 1.0 for mid in high_quality_models)
        
        if gpt_oss_reward == 0.0 and any_hq_success:
            gpt_oss_fails_but_solvable.append({
                "prompt": prompt,
                "gpt_oss": gpt_oss_reward,
                "successes": [mid for mid in high_quality_models if rewards.get(mid, 0.0) == 1.0]
            })
    
    print(f"\n🎯 THE DELTA:")
    print(f"Prompts where gpt-oss FAILS but flagship models SUCCEED: {len(gpt_oss_fails_but_solvable)}")
    print(f"Percentage of hard set: {len(gpt_oss_fails_but_solvable) / len(hard_prompts) * 100:.1f}%")
    
    if gpt_oss_fails_but_solvable:
        print(f"\n📋 These {len(gpt_oss_fails_but_solvable)} prompts are your ARBITRAGE OPPORTUNITY:")
        for i, item in enumerate(gpt_oss_fails_but_solvable[:10], 1):
            prompt = item["prompt"]
            # Truncate for display
            display = prompt[:100] + "..." if len(prompt) > 100 else prompt
            print(f"\n{i}. {display}")
            print(f"   GPT-OSS: {item['gpt_oss']} | Succeeded: {', '.join(item['successes'])}")
            print(f"   Length: {len(prompt)} chars")
        
        # Basic analysis
        lengths = [len(item["prompt"]) for item in gpt_oss_fails_but_solvable]
        avg_len = sum(lengths) / len(lengths)
        print(f"\n📊 Prompt Statistics:")
        print(f"Average Length: {avg_len:.0f} chars")
        print(f"Min Length: {min(lengths)}")
        print(f"Max Length: {max(lengths)}")
        
        # Check if all hard prompts exist in the file
        print(f"\n💾 Saving delta prompts to file for further analysis...")
        output_path = Path(__file__).parent / "results" / "delta_prompts.json"
        output_path.parent.mkdir(exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(gpt_oss_fails_but_solvable, f, indent=2)
        print(f"Saved to: {output_path}")

if __name__ == "__main__":
    main()
