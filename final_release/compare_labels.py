import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score
from dotenv import load_dotenv

# Ensure we can import from current directory
sys.path.append(str(Path(__file__).parent))

from llm_judge import LLMJudge

load_dotenv()

def compare_labels():
    print("Loading HelpSteer2 data...")
    try:
        from datasets import load_dataset
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
        from datasets import load_dataset

    dataset = load_dataset("nvidia/HelpSteer2", split="train")
    df = pd.DataFrame(dataset)
    
    # Deduplicate by prompt (keep first)
    initial_len = len(df)
    df = df.drop_duplicates(subset=['prompt'], keep='first')
    print(f"Deduplication: {initial_len} -> {len(df)} unique prompts.")
    
    SAMPLE_SIZE = 100
    df = df.head(SAMPLE_SIZE)
    print(f"Running comparison on first {SAMPLE_SIZE} unique prompts...")

    # Initialize Judge (Try multiple models if one fails)
    models_to_try = [
        "openai/gpt-4o-mini", # Try OpenAI first as baseline
        "google/gemini-flash-1.5",
        "google/gemini-pro-1.5",
    ]
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key:
        print(f"API Key loaded: {api_key[:4]}...{api_key[-4:]}")
    else:
        print("API Key NOT loaded!")

    judge = None
    for model_name in models_to_try:
        print(f"Trying model: {model_name}...")
        try:
            # Test with a dummy request
            j = LLMJudge(model=model_name)
            score = j.grade("test", "test")
            if score == 0.5:
                print(f"Failed to use {model_name} (returned 0.5 on test)")
                continue
                
            judge = j
            print(f"Successfully initialized judge with {model_name}")
            break
        except Exception as e:
            print(f"Failed to use {model_name}: {e}")
            
    if judge is None:
        print("CRITICAL: Could not initialize any judge model. Exiting.")
        return 
    
    results = []
    
    print(f"\n{'='*60}")
    print(f"Comparing HelpSteer2 Labels vs {judge.model}")
    print(f"{'='*60}")
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        prompt = row['prompt']
        response = row['response']
        
        # Ground Truth: HelpSteer correctness is 0-4. 
        # We need to binarize it to match our Judge (0.0-1.0).
        # Usually > 2.5 is Good, <= 2.5 is Bad.
        gt_score = row['correctness']
        gt_label = 1 if gt_score > 2.5 else 0
        
        # Gemini Prediction
        try:
            pred_score = judge.grade(prompt, response)
            pred_label = 1 if pred_score > 0.5 else 0
        except Exception as e:
            print(f"Error grading: {e}")
            continue
            
        results.append({
            "prompt": prompt,
            "response": response,
            "gt_score": gt_score,
            "gt_label": gt_label,
            "pred_score": pred_score,
            "pred_label": pred_label,
            "match": gt_label == pred_label
        })
        
    results_df = pd.DataFrame(results)
    
    # Metrics
    acc = accuracy_score(results_df['gt_label'], results_df['pred_label'])
    # Use labels=[0, 1] to force 2x2 matrix even if some classes are missing
    cm = confusion_matrix(results_df['gt_label'], results_df['pred_label'], labels=[0, 1])
    
    print(f"\n{'='*60}")
    print(f"RESULTS (n={len(results_df)})")
    print(f"{'='*60}")
    print(f"Accuracy: {acc:.3f}")
    print(f"Confusion Matrix:\n{cm}")
    print(f"  TN={cm[0,0]} | FP={cm[0,1]}")
    print(f"  FN={cm[1,0]} | TP={cm[1,1]}")
    
    # Disagreements
    disagreements = results_df[results_df['match'] == False]
    print(f"\nDisagreements ({len(disagreements)}):")
    for _, row in disagreements.head(5).iterrows():
        print("-" * 40)
        print(f"Prompt: {row['prompt'][:100]}...")
        print(f"Response: {row['response'][:100]}...")
        print(f"GT: {row['gt_label']} (Score: {row['gt_score']}) | Pred: {row['pred_label']} (Score: {row['pred_score']})")

if __name__ == "__main__":
    compare_labels()
