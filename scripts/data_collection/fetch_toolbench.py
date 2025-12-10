#!/usr/bin/env python3
"""
Fetch ToolBench dataset for agentic execution prompts.

ToolBench is the academic standard for "Chained Tool Use" - it contains complex
instructions that require multiple steps (e.g., "Find a restaurant in Seattle 
and book a table for two").

Sources:
- GitHub: https://github.com/OpenBMB/ToolBench
- Dataset: Contains 16,000+ real APIs (Google Sheets, Weather, Klarna, etc.)
"""

import json
import os
import requests
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
TOOLBENCH_DIR = DATA_DIR / "toolbench"

# ToolBench raw data URLs (from GitHub)
TOOLBENCH_GITHUB_RAW = "https://raw.githubusercontent.com/OpenBMB/ToolBench/master/data"

# Alternative: Use the smaller instruction files
INSTRUCTION_FILES = [
    "instruction/G1_instruction.json",
    "instruction/G2_instruction.json", 
    "instruction/G3_instruction.json",
]


def download_file(url: str, output_path: Path) -> bool:
    """Download a file from URL."""
    try:
        print(f"  Downloading: {url}")
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(response.text)
        
        return True
    except Exception as e:
        print(f"  ❌ Error downloading {url}: {e}")
        return False


def fetch_toolbench_instructions():
    """Fetch ToolBench instruction files from GitHub."""
    print("=" * 60)
    print("FETCHING TOOLBENCH INSTRUCTIONS")
    print("=" * 60)
    
    TOOLBENCH_DIR.mkdir(parents=True, exist_ok=True)
    
    all_prompts = []
    
    for instruction_file in INSTRUCTION_FILES:
        url = f"{TOOLBENCH_GITHUB_RAW}/{instruction_file}"
        filename = instruction_file.split("/")[-1]
        output_path = TOOLBENCH_DIR / filename
        
        if output_path.exists():
            print(f"  ✓ Already have: {filename}")
            with open(output_path) as f:
                data = json.load(f)
        else:
            if not download_file(url, output_path):
                continue
            with open(output_path) as f:
                data = json.load(f)
        
        # Extract prompts
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    # Look for query/instruction field
                    prompt = item.get("query") or item.get("instruction") or item.get("user_request")
                    if prompt:
                        all_prompts.append({
                            "text": prompt,
                            "source": filename,
                            "category": item.get("category", "unknown"),
                            "api": item.get("api", "unknown"),
                        })
        elif isinstance(data, dict):
            for key, item in data.items():
                if isinstance(item, dict):
                    prompt = item.get("query") or item.get("instruction") or item.get("user_request")
                    if prompt:
                        all_prompts.append({
                            "text": prompt,
                            "source": filename,
                            "category": item.get("category", "unknown"),
                            "api": item.get("api", "unknown"),
                        })
    
    print(f"\n  Total prompts extracted: {len(all_prompts)}")
    return all_prompts


def fetch_toolbench_from_hf_api():
    """Try fetching ToolBench using HuggingFace API directly."""
    print("\n" + "=" * 60)
    print("TRYING HUGGINGFACE API")
    print("=" * 60)
    
    # Try different dataset variants
    datasets_to_try = [
        ("Tool-COLT/ToolBenchG2", "train"),
        ("Tool-COLT/ToolBenchG3", "train"),
    ]
    
    all_prompts = []
    
    for dataset_name, split in datasets_to_try:
        try:
            from datasets import load_dataset
            print(f"\n  Loading {dataset_name}...")
            ds = load_dataset(dataset_name, split=split)
            
            print(f"  ✓ Loaded {len(ds)} examples")
            print(f"  Columns: {ds.column_names}")
            
            # Extract prompts based on available columns
            for item in ds:
                prompt = None
                for field in ["query", "instruction", "user_request", "input", "prompt"]:
                    if field in item and item[field]:
                        prompt = item[field]
                        break
                
                if prompt:
                    all_prompts.append({
                        "text": prompt,
                        "source": dataset_name,
                        "category": item.get("category", "toolbench"),
                    })
            
            print(f"  Extracted {len(all_prompts)} prompts so far")
            
        except Exception as e:
            print(f"  ❌ Error with {dataset_name}: {e}")
    
    return all_prompts


def create_agentic_dataset(prompts: List[Dict], output_path: Path):
    """Create a dataset file with agentic prompts."""
    print(f"\n  Saving to: {output_path}")
    
    # Format for our labeling pipeline
    formatted = []
    for p in prompts:
        formatted.append({
            "text": p["text"],
            "source": f"toolbench_{p.get('source', 'unknown')}",
            "expected_label": "AGENTIC_EXECUTION",  # All ToolBench prompts are agentic
            "category": p.get("category", "unknown"),
        })
    
    with open(output_path, 'w') as f:
        json.dump(formatted, f, indent=2)
    
    print(f"  ✓ Saved {len(formatted)} agentic prompts")
    return formatted


def main():
    print("=" * 60)
    print("TOOLBENCH DATA FETCHER")
    print("=" * 60)
    print("ToolBench: Academic standard for Chained Tool Use")
    print("Contains complex instructions requiring multiple API calls")
    print()
    
    all_prompts = []
    
    # Method 1: Try GitHub raw files
    github_prompts = fetch_toolbench_instructions()
    all_prompts.extend(github_prompts)
    
    # Method 2: Try HuggingFace datasets
    if len(all_prompts) < 100:
        hf_prompts = fetch_toolbench_from_hf_api()
        all_prompts.extend(hf_prompts)
    
    # Deduplicate by text
    seen = set()
    unique_prompts = []
    for p in all_prompts:
        text = p["text"].strip()[:200]  # Use first 200 chars for dedup
        if text not in seen:
            seen.add(text)
            unique_prompts.append(p)
    
    print(f"\n  Unique prompts after dedup: {len(unique_prompts)}")
    
    if unique_prompts:
        # Save the dataset
        output_path = DATA_DIR / "toolbench_agentic_prompts.json"
        create_agentic_dataset(unique_prompts, output_path)
        
        # Show some examples
        print("\n" + "=" * 60)
        print("SAMPLE AGENTIC PROMPTS")
        print("=" * 60)
        for i, p in enumerate(unique_prompts[:10]):
            print(f"\n{i+1}. {p['text'][:150]}...")
    else:
        print("\n⚠️  No prompts fetched. Try manually downloading from:")
        print("   https://github.com/OpenBMB/ToolBench/tree/master/data")
    
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()

