#!/usr/bin/env python3
"""
Download 1M Prompt Dataset from LMSYS Chat-1M

This script downloads the full 1M prompt dataset from LMSYS Chatbot Arena
(lmsys/lmsys-chat-1m) which is the same source used for dev/holdout datasets.

The lmsys-chat-1m dataset contains real-world conversations from the Vicuna demo
and Chatbot Arena website collected from April to August 2023.

Usage:
    python3 experiments_v1/01_figure_1M/download_1M_dataset.py
"""

import sys
from pathlib import Path

# Add project root and src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import json
import gzip
import os
from tqdm import tqdm
from datasets import load_dataset
from dotenv import load_dotenv

# Load environment variables
load_dotenv(project_root / ".env")


def download_and_process_lmsys_1M():
    """Download and process the full LMSYS Chat-1M dataset."""
    
    print("="*80)
    print("DOWNLOAD LMSYS CHAT-1M DATASET FROM HUGGINGFACE")
    print("="*80)
    print("\n📋 Dataset: lmsys/lmsys-chat-1m")
    print("   Source: LMSYS Chatbot Arena conversations")
    print("   Period: April-August 2023")
    print("   Size: ~1M conversations from 210K unique IPs")
    
    # Configuration
    output_dir = Path(__file__).parent / "data"
    output_file = output_dir / "lmsys_chat_1M.jsonl.gz"
    hf_token = os.getenv('HUGGINGFACE_API_KEY') or os.getenv('HF_TOKEN')
    
    print(f"\n📋 Configuration:")
    print(f"   Output: {output_file}")
    print(f"   HF token: {'✓ Found' if hf_token else '✗ Not found (public access)'}")
    
    # Step 1: Download dataset
    print(f"\n📥 Step 1/3: Downloading LMSYS Chat-1M dataset from HuggingFace...")
    print(f"   Dataset: lmsys/lmsys-chat-1m")
    print(f"   Note: This may take a while for 1M conversations")
    
    try:
        ds = load_dataset(
            "lmsys/lmsys-chat-1m",
            split="train",
            token=hf_token,
            streaming=False
        )
        print(f"   ✅ Downloaded {len(ds):,} conversations")
    except Exception as e:
        print(f"   ❌ Error downloading: {e}")
        print("\n💡 Troubleshooting:")
        print("   1. Check internet connection")
        print("   2. Set HUGGINGFACE_API_KEY in .env file")
        print("   3. Accept dataset license at: https://huggingface.co/datasets/lmsys/lmsys-chat-1m")
        print("   4. Verify dataset name: lmsys/lmsys-chat-1m")
        return
    
    # Step 2: Process conversations
    print(f"\n⚙️  Step 2/3: Processing conversations...")
    print(f"   Extracting user prompts from conversations...")
    
    prompts = []
    skipped = 0
    
    for row in tqdm(ds, desc="   Processing"):
        try:
            # Extract conversation data
            # The dataset has 'conversation' field with list of turns
            conversation = row.get('conversation', [])
            
            if not conversation or len(conversation) == 0:
                skipped += 1
                continue
            
            # Get the first user message (the prompt)
            first_turn = conversation[0]
            
            # Handle different possible formats
            if isinstance(first_turn, dict):
                prompt = first_turn.get('content', '') or first_turn.get('text', '')
            elif isinstance(first_turn, str):
                prompt = first_turn
            else:
                skipped += 1
                continue
            
            if not prompt or not isinstance(prompt, str):
                skipped += 1
                continue
            
            # Clean prompt
            prompt = prompt.strip()
            if len(prompt) < 10 or len(prompt) > 10000:
                skipped += 1
                continue
            
            prompts.append(prompt)
            
        except Exception as e:
            skipped += 1
            continue
    
    print(f"   ✅ Processed {len(prompts):,} prompts")
    print(f"   ⚠️  Skipped {skipped:,} conversations (invalid or too short/long)")
    
    # Step 3: Deduplicate and analyze
    print(f"\n📊 Step 3/3: Deduplicating and analyzing...")
    
    # Deduplicate prompts
    unique_prompts = list(set(prompts))
    print(f"   Unique prompts: {len(unique_prompts):,}")
    print(f"   Duplicates removed: {len(prompts) - len(unique_prompts):,}")
    
    # Sample statistics
    prompt_lengths = [len(p) for p in unique_prompts]
    print(f"\n   Prompt length statistics:")
    print(f"      Mean: {sum(prompt_lengths) / len(prompt_lengths):.1f} characters")
    print(f"      Min: {min(prompt_lengths)} characters")
    print(f"      Max: {max(prompt_lengths)} characters")
    
    # Save to compressed file (simple format: one prompt per line)
    print(f"\n💾 Saving to: {output_file}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with gzip.open(output_file, 'wt') as f:
        for prompt in unique_prompts:
            # Save as JSON for consistency
            f.write(json.dumps({'prompt': prompt}) + '\n')
    
    print(f"   ✅ Saved {len(unique_prompts):,} unique prompts")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"   ✅ Downloaded and processed LMSYS Chat-1M dataset")
    print(f"   ✅ Total conversations: {len(ds):,}")
    print(f"   ✅ Unique prompts extracted: {len(unique_prompts):,}")
    print(f"   ✅ Output: {output_file}")
    print(f"   ✅ Format: JSONL.GZ with prompts")
    
    print("\n📋 Next steps:")
    print(f"   1. Run PCA analysis:")
    print(f"      python experiments_v1/01_figure_1M/plot_lmsys_1M_pca.py")
    print("="*80)


def main():
    download_and_process_lmsys_1M()


if __name__ == "__main__":
    main()
