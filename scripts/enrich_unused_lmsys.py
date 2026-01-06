#!/usr/bin/env python3
"""
Enrich unused LMSYS prompts with full metadata from Hugging Face dataset.

This script:
1. Loads the unused LMSYS prompts (just text)
2. Downloads the full LMSYS dataset from Hugging Face
3. Matches prompts and extracts full records with all metadata
4. Saves enriched data to a new file

The HuggingFace LMSYS dataset contains rich metadata like:
- conversation_id
- model
- conversation (full chat history)
- turn (conversation turn number)
- language
- openai_moderation (safety scores)
- redacted (whether PII was removed)
"""

import json
from pathlib import Path
from typing import Set, Dict, List
from datasets import load_dataset
from tqdm import tqdm

def load_unused_prompts(filepath: Path) -> Set[str]:
    """Load the set of unused prompts."""
    prompts = set()
    with open(filepath, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            prompts.add(data['prompt'])
    return prompts

def download_and_match_lmsys(unused_prompts: Set[str], output_file: Path):
    """
    Download LMSYS dataset from HuggingFace and match unused prompts.
    
    Args:
        unused_prompts: Set of prompt texts to find
        output_file: Where to save enriched data
    """
    print("📦 Downloading LMSYS dataset from Hugging Face...")
    print("   Dataset: lmsys/lmsys-chat-1m")
    print("   This may take a few minutes on first run (cached afterward)...")
    
    try:
        # Load the LMSYS Chat 1M dataset
        dataset = load_dataset("lmsys/lmsys-chat-1m", split="train")
        print(f"✅ Loaded {len(dataset)} conversations from HuggingFace")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("\nTrying alternative dataset name...")
        try:
            dataset = load_dataset("lmsys/chatbot_arena_conversations", split="train")
            print(f"✅ Loaded {len(dataset)} conversations from HuggingFace")
        except Exception as e2:
            print(f"❌ Error: {e2}")
            return
    
    print(f"\n🔍 Matching {len(unused_prompts)} unused prompts to dataset...")
    
    matched_records = []
    matched_prompts = set()
    
    # Iterate through dataset and match prompts
    for record in tqdm(dataset, desc="Processing"):
        # Extract the user prompt from the conversation
        # The conversation field is a list of messages
        conversation = record.get('conversation', [])
        
        if not conversation:
            continue
        
        # Get the first user message (usually the prompt)
        for message in conversation:
            if message.get('role') == 'user':
                user_prompt = message.get('content', '')
                
                # Check if this prompt is in our unused set
                if user_prompt in unused_prompts:
                    # Found a match! Save the full record
                    matched_records.append({
                        'prompt': user_prompt,
                        'conversation_id': record.get('conversation_id'),
                        'model': record.get('model'),
                        'conversation': conversation,
                        'turn': record.get('turn'),
                        'language': record.get('language'),
                        'openai_moderation': record.get('openai_moderation'),
                        'redacted': record.get('redacted', False),
                        'timestamp': record.get('tstamp')
                    })
                    matched_prompts.add(user_prompt)
                    
                    # Only match first user message per conversation
                    break
    
    print(f"\n📊 Matching Results:")
    print(f"   Unused prompts to find: {len(unused_prompts)}")
    print(f"   Matched in HF dataset: {len(matched_prompts)}")
    print(f"   Not found: {len(unused_prompts - matched_prompts)}")
    print(f"   Match rate: {len(matched_prompts)/len(unused_prompts)*100:.1f}%")
    
    # Save matched records
    print(f"\n💾 Saving {len(matched_records)} enriched records to {output_file}...")
    with open(output_file, 'w') as f:
        for record in matched_records:
            json.dump(record, f)
            f.write('\n')
    
    print(f"✅ Saved enriched data")
    
    # Save unmatched prompts for debugging
    unmatched = unused_prompts - matched_prompts
    if unmatched:
        unmatched_file = output_file.parent / "lmsys_unmatched_prompts.jsonl"
        print(f"\n⚠️  Saving {len(unmatched)} unmatched prompts to {unmatched_file}")
        with open(unmatched_file, 'w') as f:
            for prompt in sorted(unmatched):
                json.dump({'prompt': prompt}, f)
                f.write('\n')
    
    # Show sample of enriched data
    print(f"\n📝 Sample enriched record:")
    if matched_records:
        sample = matched_records[0]
        print(f"   Conversation ID: {sample['conversation_id']}")
        print(f"   Model: {sample['model']}")
        print(f"   Language: {sample['language']}")
        print(f"   Prompt: {sample['prompt'][:100]}...")
        print(f"   Full conversation turns: {len(sample['conversation'])}")

def main():
    """Main execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enrich LMSYS prompts with HuggingFace metadata")
    parser.add_argument(
        '--input',
        type=str,
        default='lmsys_unused_prompts.jsonl',
        help='Input JSONL file with prompts (default: lmsys_unused_prompts.jsonl)'
    )
    parser.add_argument(
        '--output-suffix',
        type=str,
        default='enriched',
        help='Output file suffix (default: enriched)'
    )
    args = parser.parse_args()
    
    print("="*70)
    print("LMSYS Prompt Enrichment Tool")
    print("="*70)
    print()
    
    data_dir = Path('src/bandit_gpt/data')
    
    # Load unused prompts
    unused_file = data_dir / args.input
    if not unused_file.exists():
        print(f"❌ Error: {unused_file} not found")
        return
    
    print(f"📂 Loading prompts from {unused_file}...")
    unused_prompts = load_unused_prompts(unused_file)
    print(f"   Loaded {len(unused_prompts)} prompts")
    print()
    
    # Generate output filename
    input_stem = Path(args.input).stem  # e.g., 'lmsys_unused_20k'
    output_file = data_dir / f"{input_stem}_{args.output_suffix}.jsonl"
    
    # Download and match
    download_and_match_lmsys(unused_prompts, output_file)
    
    print(f"\n🎉 Done! Enriched data saved to:")
    print(f"   {output_file}")

if __name__ == '__main__':
    main()
