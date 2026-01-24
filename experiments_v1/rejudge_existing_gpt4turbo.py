#!/usr/bin/env python3
"""
Rejudge existing GPT-4-Turbo responses using multi-judge CoT system.
This reuses existing responses to save on generation costs.
"""

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from rejudge_cot import CoTRewardGenerator

def load_existing_responses(file_path):
    """Load existing GPT-4-Turbo responses."""
    responses = {}
    with open(file_path) as f:
        for line in f:
            entry = json.loads(line)
            if entry.get('ok') and entry['model_id'] == 'openai/gpt-4-turbo':
                responses[entry['prompt']] = entry['response']
    return responses

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Existing GPT-4-Turbo responses file")
    parser.add_argument("--output", type=str, required=True, help="Output file for rejudged rewards")
    parser.add_argument("--workers", type=int, default=5, help="Number of parallel workers")
    args = parser.parse_args()
    
    print(f"Loading existing GPT-4-Turbo responses from: {args.input}")
    responses = load_existing_responses(args.input)
    print(f"  Found {len(responses)} responses")
    
    # Initialize generator
    gen = CoTRewardGenerator(max_workers=args.workers)
    
    # Populate cache with existing responses
    model_id = "openai/gpt-4-turbo"
    for prompt, response in responses.items():
        gen.response_cache[(model_id, prompt)] = response
    
    print(f"\n✅ Loaded {len(responses)} responses into cache")
    print(f"   Now judging with multi-judge CoT system...")
    
    # Create tasks
    tasks = [(prompt, model_id) for prompt in responses.keys()]
    
    # Process
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm
    
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    with open(output_path, 'w') as outfile:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(gen.process_task, t): t for t in tasks}
            
            with tqdm(total=len(tasks), desc="Rejudging") as pbar:
                for f in as_completed(futures):
                    res = f.result()
                    outfile.write(json.dumps(res) + "\n")
                    pbar.update(1)
    
    print(f"\n✅ Rejudging complete!")
    print(f"   Output: {output_path}")

if __name__ == "__main__":
    main()

