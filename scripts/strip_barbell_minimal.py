#!/usr/bin/env python3
"""
Strip barbell dataset to minimal fields needed for N-tuning.

Original size: 68MB
Expected size after stripping: ~5-10MB

Fields kept:
- prompt: The actual prompt text (required for routing)
- subcategory: Category label (for verification/debugging)

Fields removed:
- conversation: Full chat history (redundant with prompt)
- conversation_id: Not needed for N-tuning
- model: Not needed (we simulate all models)
- openai_moderation: Not needed
- timestamp: Not needed
- language: Not needed
- turn: Already filtered to turn=1
- redacted: Not needed
"""

import json
from pathlib import Path

def strip_barbell_dataset():
    """Strip barbell dataset to minimal fields."""
    
    input_file = Path("src/bandit_gpt/data/lmsys_barbell_20k.jsonl")
    output_file = Path("src/bandit_gpt/data/lmsys_barbell_20k_minimal.jsonl")
    
    print("🔧 Stripping barbell dataset to minimal fields...")
    print(f"   Input: {input_file}")
    
    # Get original size
    orig_size_mb = input_file.stat().st_size / 1024 / 1024
    print(f"   Original size: {orig_size_mb:.1f} MB")
    print()
    
    prompts_processed = 0
    
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            data = json.loads(line.strip())
            
            # Extract minimal fields
            minimal = {
                'prompt': data.get('prompt', ''),
                'subcategory': data.get('subcategory', 'unknown')
            }
            
            # Validate prompt exists
            if minimal['prompt']:
                json.dump(minimal, f_out)
                f_out.write('\n')
                prompts_processed += 1
    
    # Get new size
    new_size_mb = output_file.stat().st_size / 1024 / 1024
    reduction_pct = (1 - new_size_mb / orig_size_mb) * 100
    
    print(f"✅ Stripped dataset created:")
    print(f"   Output: {output_file}")
    print(f"   Prompts: {prompts_processed}")
    print(f"   New size: {new_size_mb:.1f} MB")
    print(f"   Reduction: {reduction_pct:.1f}%")
    print()
    
    # Test compression
    import subprocess
    result = subprocess.run(
        ['gzip', '-c', str(output_file)],
        capture_output=True
    )
    comp_size_mb = len(result.stdout) / 1024 / 1024
    
    print(f"📦 Compressed size: {comp_size_mb:.1f} MB (gzip)")
    print(f"   Compression ratio: {comp_size_mb/new_size_mb*100:.1f}%")
    print()
    
    if comp_size_mb < 5:
        print("✅ Small enough to commit directly!")
    elif comp_size_mb < 20:
        print("⚠️  Recommend Git LFS or external storage")
    else:
        print("❌ Too large even compressed - use external storage")

if __name__ == '__main__':
    strip_barbell_dataset()
