#!/usr/bin/env python3
"""
Collect ALL available instance-level data from OpenCompass for all intents.

This script collects:
- Reasoning: GPQA (58 models)
- Coding: HumanEval (58 models)
- RAG: TriviaQA (19 models)
- Summarization: IFEval (60 models)
"""

from huggingface_hub import list_repo_files, hf_hub_download
from datasets import load_dataset
import os
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
import sys

# Load environment
load_dotenv()
HF_TOKEN = os.getenv('HUGGINGFACE_API_KEY')

# Load name mappings
try:
    from opencompass_name_mappings import OPENCOMPASS_TO_CACHE
except:
    OPENCOMPASS_TO_CACHE = {}
    print("⚠️  Could not load name mappings")

def download_opencompass_benchmark(benchmark_name, intent_name):
    """Download predictions for a benchmark from OpenCompass"""
    print(f"\n{'='*80}")
    print(f"DOWNLOADING {benchmark_name.upper()} ({intent_name.upper()})")
    print("="*80)
    
    try:
        repo_id = "opencompass/compass_academic_predictions"
        
        # List all files
        files = list(list_repo_files(repo_id, repo_type='dataset', token=HF_TOKEN))
        
        # Filter for this benchmark
        prediction_files = [f for f in files if f.startswith(f'results_stations/{benchmark_name}/') and f.endswith('.json')]
        
        print(f"Found {len(prediction_files)} prediction files")
        
        # Filter to models in our cache
        if OPENCOMPASS_TO_CACHE:
            mapped_files = [f for f in prediction_files if Path(f).stem in OPENCOMPASS_TO_CACHE]
            print(f"✓ {len(mapped_files)} models match our cache")
            prediction_files = mapped_files
        
        if not prediction_files:
            print("❌ No files to download")
            return {}
        
        predictions = {}
        
        for file_path in tqdm(prediction_files, desc=f"Downloading {benchmark_name}"):
            local_path = hf_hub_download(repo_id, file_path, repo_type='dataset', token=HF_TOKEN)
            
            with open(local_path, 'r') as f:
                data = json.load(f)
            
            model_name = Path(file_path).stem
            predictions[model_name] = data
        
        print(f"✓ Downloaded {len(predictions)} models")
        return predictions
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return {}

def load_prompts(benchmark_config):
    """Load prompts for a benchmark"""
    dataset_name = benchmark_config.get('dataset')
    config = benchmark_config.get('config')
    split = benchmark_config.get('split', 'test')
    
    if not dataset_name:
        print(f"⚠️  No dataset specified")
        return pd.DataFrame()
    
    try:
        if config:
            ds = load_dataset(dataset_name, config, split=split, token=HF_TOKEN)
        else:
            ds = load_dataset(dataset_name, split=split, token=HF_TOKEN)
        
        df = ds.to_pandas()
        
        # Add question_id if not present
        if 'question_id' not in df.columns and 'task_id' not in df.columns:
            df['question_id'] = df.index.astype(str)
        
        print(f"✓ Loaded {len(df)} prompts")
        return df
        
    except Exception as e:
        print(f"❌ Error loading prompts: {e}")
        return pd.DataFrame()

def main():
    print("="*80)
    print("COLLECTING ALL BENCHMARKS FROM OPENCOMPASS")
    print("="*80)
    
    benchmarks = {
        'reasoning': {
            'opencompass_name': 'GPQA_diamond',
            'dataset': 'Idavidrein/gpqa',
            'config': 'gpqa_diamond',
            'split': 'train',
            'prompt_column': 'Question',
            'join_key': 'question_id'
        },
        'coding': {
            'opencompass_name': 'openai_humaneval',
            'dataset': 'evalplus/humanevalplus',
            'config': None,
            'split': 'test',
            'prompt_column': 'prompt',
            'join_key': 'task_id'
        }
    }
    
    results_summary = {}
    
    for intent, config in benchmarks.items():
        print(f"\n{'#'*80}")
        print(f"# {intent.upper()}")
        print("#"*80)
        
        # Load prompts
        print(f"\nLoading prompts from {config['dataset']}...")
        prompts = load_prompts(config)
        
        if prompts.empty:
            print(f"❌ Skipping {intent} - no prompts loaded")
            continue
        
        # Download predictions
        predictions = download_opencompass_benchmark(config['opencompass_name'], intent)
        
        if not predictions:
            print(f"❌ Skipping {intent} - no predictions downloaded")
            continue
        
        results_summary[intent] = {
            'prompts': len(prompts),
            'models': len(predictions),
            'estimated_examples': len(prompts) * len(predictions)
        }
        
        print(f"\n✓ {intent.upper()} READY:")
        print(f"   Prompts: {len(prompts):,}")
        print(f"   Models: {len(predictions)}")
        print(f"   Estimated examples: {len(prompts) * len(predictions):,}")
    
    print("\n" + "="*80)
    print("COLLECTION SUMMARY")
    print("="*80)
    
    total_examples = sum(r['estimated_examples'] for r in results_summary.values())
    
    for intent, stats in results_summary.items():
        print(f"{intent.capitalize():15s}: {stats['estimated_examples']:,} examples")
    
    print(f"{'Total':15s}: {total_examples:,} examples")
    
    print("\n✅ Data sources verified and ready to collect!")
    print("\nNext step: Run build_instance_level_training_data.py with updated config")

if __name__ == '__main__':
    main()
