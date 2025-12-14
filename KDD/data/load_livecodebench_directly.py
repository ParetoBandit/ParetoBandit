#!/usr/bin/env python3
"""
Load LiveCodeBench data directly from GitHub (bypassing HuggingFace)

Since HuggingFace deprecated custom loading scripts, we download
and parse LiveCodeBench JSON files directly from their GitHub repo.
"""

import requests
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict


def download_livecodebench_problems() -> pd.DataFrame:
    """
    Download LiveCodeBench problems directly from GitHub.
    
    Returns:
        DataFrame with columns: question_id, question_title, question_content, difficulty, etc.
    """
    print("Downloading LiveCodeBench problems from GitHub...")
    
    # LiveCodeBench maintains their data on GitHub
    # Check their repo: https://github.com/LiveCodeBench/LiveCodeBench
    
    # Option 1: Download from releases
    base_url = "https://github.com/LiveCodeBench/LiveCodeBench/raw/main/lcb_runner/data"
    
    # They organize by scenario - for agentic, we want code_execution and test_output
    scenarios = [
        'code_execution',  # Agentic: Running code
        'test_output_prediction',  # Agentic: Predicting test results
    ]
    
    all_problems = []
    
    for scenario in scenarios:
        try:
            # Try to download the problems file
            url = f"{base_url}/{scenario}/questions.jsonl"
            
            print(f"  Attempting: {url}")
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                # Parse JSONL
                for line in response.text.strip().split('\n'):
                    if line:
                        problem = json.loads(line)
                        problem['scenario'] = scenario
                        all_problems.append(problem)
                
                print(f"  ✓ Loaded {len(all_problems)} problems from {scenario}")
            else:
                print(f"  ⚠️  HTTP {response.status_code} for {scenario}")
                
        except Exception as e:
            print(f"  ⚠️  Error loading {scenario}: {e}")
    
    if all_problems:
        df = pd.DataFrame(all_problems)
        print(f"\n✓ Total problems loaded: {len(df)}")
        return df
    else:
        print("\n⚠️  No problems loaded - trying alternative approach...")
        return try_alternative_download()


def try_alternative_download() -> pd.DataFrame:
    """
    Alternative: Download from HuggingFace datasets repo (raw files).
    
    Even though the loader is deprecated, the raw data files might still be accessible.
    """
    print("\nTrying to access raw HuggingFace data files...")
    
    try:
        from huggingface_hub import hf_hub_download
        import os
        from dotenv import load_dotenv
        
        load_dotenv()
        HF_TOKEN = os.getenv('HUGGINGFACE_API_KEY')
        
        # Try to download the raw data file directly (not using the loader)
        repo_id = "livecodebench/code_generation_lite"
        
        # Common filenames in dataset repos
        possible_files = [
            "data/test.jsonl",
            "data/test.json", 
            "test.jsonl",
            "test.json",
            "questions.jsonl",
        ]
        
        for filename in possible_files:
            try:
                print(f"  Trying: {filename}")
                local_path = hf_hub_download(
                    repo_id, 
                    filename, 
                    repo_type='dataset',
                    token=HF_TOKEN
                )
                
                # Read the file
                if filename.endswith('.jsonl'):
                    data = []
                    with open(local_path) as f:
                        for line in f:
                            data.append(json.loads(line))
                else:
                    with open(local_path) as f:
                        data = json.load(f)
                
                df = pd.DataFrame(data)
                print(f"  ✓ Success! Loaded {len(df)} problems")
                return df
                
            except Exception as e:
                print(f"    Not found: {e}")
                continue
        
        print("\n❌ Could not find raw data files")
        return pd.DataFrame()
        
    except Exception as e:
        print(f"❌ Alternative download failed: {e}")
        return pd.DataFrame()


def load_agentic_prompts_for_join() -> pd.DataFrame:
    """
    Load LiveCodeBench prompts specifically for joining with OpenCompass predictions.
    
    Returns:
        DataFrame with question_id column for joining
    """
    df = download_livecodebench_problems()
    
    if df.empty:
        print("\n❌ FAILED: Could not load LiveCodeBench data")
        print("\nManual workaround options:")
        print("1. Clone LiveCodeBench repo: git clone https://github.com/LiveCodeBench/LiveCodeBench")
        print("2. Look for data/ folder with question files")
        print("3. Parse manually and create CSV")
        return df
    
    # Ensure we have a question_id for joining
    if 'question_id' not in df.columns:
        if 'id' in df.columns:
            df['question_id'] = df['id']
        else:
            df['question_id'] = df.index.astype(str)
    
    # Extract question text
    if 'question_content' not in df.columns:
        if 'content' in df.columns:
            df['question_content'] = df['content']
        elif 'prompt' in df.columns:
            df['question_content'] = df['prompt']
        elif 'question' in df.columns:
            df['question_content'] = df['question']
    
    print(f"\n✓ Prepared {len(df)} prompts for joining")
    print(f"  Columns: {list(df.columns)[:10]}")
    
    return df


def main():
    """Test the loading functions."""
    print("="*80)
    print("LIVECODEBENCH DIRECT LOADER TEST")
    print("="*80)
    
    df = load_agentic_prompts_for_join()
    
    if not df.empty:
        print(f"\n✅ SUCCESS!")
        print(f"   Loaded {len(df)} problems")
        print(f"\n   Sample row:")
        print(df.iloc[0].to_dict())
        
        # Save for later use
        output_path = Path(__file__).parent / 'livecodebench_prompts.csv'
        df.to_csv(output_path, index=False)
        print(f"\n✓ Saved to: {output_path}")
    else:
        print(f"\n❌ FAILED - See manual workaround options above")


if __name__ == '__main__':
    main()
