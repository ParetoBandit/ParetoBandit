import os
import json
from pathlib import Path
from final_release.data.generate_rewards_parallel import ParallelRewardGenerator

def test_tiny_run():
    base_dir = Path("final_release/data")
    
    # Tiny config
    models = ["openai/gpt-4o-mini"] # Use a cheap, fast model for testing
    generator = ParallelRewardGenerator(max_workers=2)
    
    test_out = base_dir / "test_rewards_verify.jsonl"
    if test_out.exists():
        test_out.unlink()
        
    print("Running tiny verification...")
    generator.generate_rewards_parallel(
        prompts_file=base_dir / "test_prompts.jsonl",
        output_file=test_out,
        models=models,
        max_prompts=1
    )
    
    if test_out.exists():
        print("Success! Reading result...")
        with open(test_out, 'r') as f:
            for line in f:
                data = json.loads(line)
                print(json.dumps(data, indent=2))
    else:
        print("Failed to produce output.")

if __name__ == "__main__":
    test_tiny_run()
