#!/usr/bin/env python3
"""
Collect Ground Truth Labels for Proprietary Models

This script downloads ACTUAL pass/fail labels for proprietary models from:
1. LiveCodeBench submissions (Coding) - has pass@1 labels
2. Stanford HELM / OpenCompass (Reasoning) - has is_correct labels
3. GAIA traces (Agentic) - has success labels
4. RAGBench (RAG) - has hallucination labels

These labeled datasets enable proper validation of zero-shot transfer!
"""

import pandas as pd
from datasets import load_dataset
from pathlib import Path
import json
import os
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
HF_TOKEN = os.getenv('HUGGINGFACE_API_KEY')

# Proprietary models to validate
PROPRIETARY_MODELS_CODING = [
    'gpt-4-0125-preview',
    'gpt-4o-2024-05-13',
    'claude-3-opus-20240229',
    'claude-3-5-sonnet-20240620',
    'gemini-1.5-pro-preview-0409'
]

def collect_coding_labels():
    """Collect coding labels from LiveCodeBench submissions."""
    print("="*80)
    print("COLLECTING CODING LABELS (LIVECODEBENCH)")
    print("="*80)
    
    all_data = []
    
    for model_name in PROPRIETARY_MODELS_CODING:
        print(f"\nLoading {model_name}...")
        
        try:
            # Load submissions for this model
            ds = load_dataset(
                "livecodebench/submissions",
                model_name,
                split="test",
                token=HF_TOKEN,
                trust_remote_code=True
            )
            df = ds.to_pandas()
            
            # Extract relevant columns
            df_clean = df[['question_id', 'pass@1', 'code']].copy()
            df_clean.rename(columns={'pass@1': 'success'}, inplace=True)
            df_clean['model'] = model_name
            df_clean['intent'] = 'coding'
            
            # Convert pass@1 (0.0 or 1.0) to binary
            df_clean['success'] = (df_clean['success'] >= 0.5).astype(int)
            
            print(f"  ✓ Loaded {len(df_clean)} examples")
            print(f"    Success rate: {df_clean['success'].mean():.1%}")
            
            all_data.append(df_clean)
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            print(f"     Try alternate model name or check availability")
    
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        print(f"\n✓ Total coding labels: {len(combined)} ({combined['model'].nunique()} models)")
        return combined
    else:
        print("\n⚠️  No coding labels collected")
        return pd.DataFrame()

def collect_reasoning_labels():
    """
    Collect reasoning labels from OpenCompass predictions.
    For GPQA, we already have these - just need to mark as validation set.
    """
    print("\n" + "="*80)
    print("COLLECTING REASONING LABELS (GPQA)")
    print("="*80)
    
    # We already have GPQA data with GPT-4o, Claude-3.5, Gemini
    # Just need to load and mark as proprietary
    
    print("\n✓ Reasoning labels already collected in instance_level_training_data.csv")
    print("   Models: gpt-4o, claude-3.5-sonnet, gemini-1.5-pro")
    print("   These will be used as validation set (not training)")
    
    return None  # Already in main dataset

def collect_agentic_labels():
    """Collect agentic labels from GAIA benchmark."""
    print("\n" + "="*80)
    print("COLLECTING AGENTIC LABELS (GAIA)")
    print("="*80)
    
    try:
        # Load GAIA validation set
        ds = load_dataset("gaia-benchmark/GAIA", "2023_all", split="validation", token=HF_TOKEN)
        df = ds.to_pandas()
        
        print(f"✓ Loaded GAIA validation set: {len(df)} examples")
        print(f"  Columns: {list(df.columns)[:10]}")
        
        # GAIA structure: question, final_answer, annotator_metadata
        # We'd need model predictions separately
        print("\n⚠️  GAIA provides questions, but model predictions need to be downloaded separately")
        print("   Check GAIA leaderboard 'Traces' column for specific model results")
        
        return df
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return pd.DataFrame()

def collect_rag_labels():
    """Collect RAG labels from RAGBench/RAGTruth."""
    print("\n" + "="*80)
    print("COLLECTING RAG LABELS (RAGBENCH)")
    print("="*80)
    
    try:
        # Try RAGBench
        ds = load_dataset("rungalileo/ragbench", "ragtruth", split="test", token=HF_TOKEN)
        df = ds.to_pandas()
        
        print(f"✓ Loaded RAGBench: {len(df)} examples")
        print(f"  Columns: {list(df.columns)}")
        
        # Check for success/label column
        if 'label' in df.columns:
            success_rate = df['label'].mean()
            print(f"  Success rate: {success_rate:.1%}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("  Trying alternative datasets...")
        return pd.DataFrame()

def save_proprietary_labels(coding_df, output_dir):
    """Save collected proprietary labels."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if not coding_df.empty:
        coding_path = output_dir / 'proprietary_coding_labels.csv'
        coding_df.to_csv(coding_path, index=False)
        print(f"\n✓ Saved coding labels: {coding_path}")
    
    # Summary
    summary_path = output_dir / 'proprietary_labels_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("PROPRIETARY MODEL LABELED DATA SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        if not coding_df.empty:
            f.write("CODING (LiveCodeBench):\n")
            f.write(f"  Total examples: {len(coding_df):,}\n")
            f.write(f"  Models: {coding_df['model'].nunique()}\n")
            for model in coding_df['model'].unique():
                model_df = coding_df[coding_df['model'] == model]
                f.write(f"    - {model}: {len(model_df)} examples, {model_df['success'].mean():.1%} success\n")
        else:
            f.write("CODING: No data collected\n")
    
    print(f"✓ Saved summary: {summary_path}")

def main():
    print("="*80)
    print("COLLECTING GROUND TRUTH LABELS FOR PROPRIETARY MODELS")
    print("="*80)
    print("\nThis provides actual pass/fail labels for validation!")
    
    output_dir = Path(__file__).parent / 'proprietary_labels'
    
    # Collect each intent
    coding_df = collect_coding_labels()
    reasoning_df = collect_reasoning_labels()  # Already have it
    agentic_df = collect_agentic_labels()
    rag_df = collect_rag_labels()
    
    # Save
    save_proprietary_labels(coding_df, output_dir)
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print("="*80)
    
    collected = []
    if not coding_df.empty:
        collected.append(f"Coding: {len(coding_df)} examples, {coding_df['model'].nunique()} models")
    
    if collected:
        print(f"\n✅ Collected:")
        for item in collected:
            print(f"   - {item}")
    else:
        print("\n⚠️  No proprietary labels collected")
        print("   Check model names and dataset availability")
    
    print(f"\nNext steps:")
    print(f"  1. Review collected labels in {output_dir}")
    print(f"  2. Run validation script to compare predictions vs. actual")
    print(f"  3. Calculate correlation, accuracy, AUC for paper")

if __name__ == '__main__':
    main()
