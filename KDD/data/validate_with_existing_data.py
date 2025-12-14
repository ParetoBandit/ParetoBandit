#!/usr/bin/env python3
"""
Validate Zero-Shot Transfer Using Existing OpenCompass Data

This script validates transfer to proprietary models using any existing
OpenCompass prediction data we already have. This is cheaper and faster than
running new evaluations.

Strategy:
1. Check which proprietary models have OpenCompass predictions
2. Use those as "held-out" test set (exclude from training)
3. Compare predicted vs. actual performance

Usage:
    python3 validate_with_existing_data.py
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.stats import pearsonr
import joblib
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from llm_jury.routing.nvidia_complexity_classifier import NvidiaComplexityClassifier


# Proprietary models we might have data for
PROPRIETARY_MODELS = [
    'gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo', 'gpt-4',
    'claude-3-5-sonnet-20241022', 'claude-3-opus', 'claude-3-sonnet',
    'gemini-2.0-flash', 'gemini-1.5-pro', 'gemini-1.5-flash',
    'o1-preview', 'o1-mini'
]


def load_opencompass_predictions(benchmark: str) -> Dict:
    """Load OpenCompass predictions for a benchmark."""
    from huggingface_hub import hf_hub_download, list_repo_files
    from dotenv import load_dotenv
    import os
    
    load_dotenv()
    HF_TOKEN = os.getenv('HUGGINGFACE_API_KEY')
    
    repo_id = "opencompass/compass_academic_predictions"
    
    # List files
    files = list(list_repo_files(repo_id, repo_type='dataset', token=HF_TOKEN))
    
    # Filter for benchmark
    prediction_files = [f for f in files if benchmark in f and f.endswith('.json')]
    
    # Find proprietary models
    predictions = {}
    
    for file_path in prediction_files:
        model_name = Path(file_path).stem
        
        # Check if this is a proprietary model
        is_proprietary = any(prop in model_name.lower() for prop in PROPRIETARY_MODELS)
        
        if is_proprietary:
            try:
                local_path = hf_hub_download(repo_id, file_path, repo_type='dataset', token=HF_TOKEN)
                with open(local_path) as f:
                    data = json.load(f)
                predictions[model_name] = data
                print(f"  ✓ Found proprietary model: {model_name} ({len(data)} predictions)")
            except Exception as e:
                print(f"  ⚠️  Error loading {model_name}: {e}")
    
    return predictions


def load_prompts(benchmark: str) -> pd.DataFrame:
    """Load prompts for a benchmark."""
    from datasets import load_dataset
    from dotenv import load_dotenv
    import os
    
    load_dotenv()
    HF_TOKEN = os.getenv('HUGGINGFACE_API_KEY')
    
    if benchmark == 'GPQA_diamond':
        ds = load_dataset('Idavidrein/gpqa', 'gpqa_diamond', split='train', token=HF_TOKEN)
        df = ds.to_pandas()
        df['question_id'] = df.index.astype(str)
        df['prompt'] = df['Question']
    elif benchmark == 'openai_humaneval':
        ds = load_dataset('evalplus/humanevalplus', split='test', token=HF_TOKEN)
        df = ds.to_pandas()
        df['question_id'] = df.index.astype(str)
        df['prompt'] = df['prompt']
    elif benchmark == 'IFEval':
        ds = load_dataset('google/IFEval', split='train', token=HF_TOKEN)
        df = ds.to_pandas()
        df['question_id'] = df.index.astype(str)
        df['prompt'] = df['prompt']
    else:
        return pd.DataFrame()
    
    return df


def map_model_name_to_cache(opencompass_name: str) -> str:
    """Map OpenCompass model name to models_cache.json name."""
    # Load mappings
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from opencompass_name_mappings import OPENCOMPASS_TO_CACHE
        return OPENCOMPASS_TO_CACHE.get(opencompass_name, opencompass_name)
    except:
        return opencompass_name


def get_model_capability(model_name: str, benchmark: str) -> float:
    """Get model's capability proxy from cache."""
    cache_path = Path(__file__).parent.parent.parent / 'data' / 'models_cache.json'
    
    with open(cache_path) as f:
        cache = json.load(f)
    
    # Map name
    cache_name = map_model_name_to_cache(model_name)
    
    for model in cache:
        if model['name'].lower() == cache_name.lower():
            benchmarks = model.get('benchmarks', {})
            score = benchmarks.get(benchmark) or benchmarks.get(benchmark.replace('model_', ''))
            if score:
                return score
    
    print(f"⚠️  Model {model_name} -> {cache_name} not found in cache")
    return None


def validate_benchmark(benchmark: str, intent: str, benchmark_feature: str):
    """Validate predictions for one benchmark."""
    print(f"\n{'='*80}")
    print(f"VALIDATING: {benchmark} ({intent})")
    print("="*80)
    
    # Load OpenCompass predictions for proprietary models
    print("\nChecking for proprietary model predictions in OpenCompass...")
    predictions_by_model = load_opencompass_predictions(benchmark)
    
    if not predictions_by_model:
        print(f"❌ No proprietary model data found for {benchmark}")
        return None
    
    print(f"\n✓ Found {len(predictions_by_model)} proprietary models with predictions")
    
    # Load prompts
    print(f"\nLoading prompts...")
    prompts = load_prompts(benchmark)
    print(f"✓ Loaded {len(prompts)} prompts")
    
    # Load trained XGBoost model
    model_path = Path(__file__).parent / 'trained_models' / f'xgboost_{intent}.joblib'
    if not model_path.exists():
        print(f"⚠️  No trained model found for {intent}")
        print(f"   Looking for: {model_path}")
        print(f"   Train models first using: python3 train_xgboost_tuned.py")
        return None
    
    print(f"\nLoading trained XGBoost model...")
    xgboost_model = joblib.load(model_path)
    
    # Initialize NVIDIA classifier
    print(f"Initializing NVIDIA classifier...")
    nvidia_classifier = NvidiaComplexityClassifier()
    
    # Validate each proprietary model
    results = {}
    
    for model_name, model_predictions in predictions_by_model.items():
        print(f"\n{'-'*80}")
        print(f"Model: {model_name}")
        print("-"*80)
        
        # Get model's capability proxy
        capability = get_model_capability(model_name, benchmark_feature)
        if capability is None:
            print(f"  ⚠️  Skipping - no capability score in cache")
            continue
        
        print(f"  Capability proxy ({benchmark_feature}): {capability:.2f}")
        
        # Prepare data
        all_predictions = []
        all_actuals = []
        
        print(f"  Computing predictions for {len(model_predictions)} prompts...")
        
        for idx, pred_data in enumerate(tqdm(model_predictions, desc="  Processing")):
            # Get actual result
            if 'prediction' in pred_data and 'gold' in pred_data:
                actual = 1 if pred_data['prediction'] == pred_data['gold'] else 0
            elif 'is_correct' in pred_data:
                actual = int(pred_data['is_correct'])
            else:
                continue
            
            # Get prompt
            if idx >= len(prompts):
                break
            prompt_text = prompts.iloc[idx]['prompt']
            
            # Compute NVIDIA features
            try:
                nvidia_result = nvidia_classifier.classify(prompt_text)
                nvidia_features = [
                    nvidia_result.get('creativity_scope', 0),
                    nvidia_result.get('reasoning', 0),
                    nvidia_result.get('constraint_ct', 0),
                    nvidia_result.get('domain_knowledge', 0),
                    nvidia_result.get('contextual_knowledge', 0),
                    nvidia_result.get('number_of_few_shots', 0)
                ]
            except:
                continue
            
            # Combine features
            X = nvidia_features + [capability]
            
            # Predict
            try:
                pred_prob = xgboost_model.predict_proba([X])[0][1]
                all_predictions.append(pred_prob)
                all_actuals.append(actual)
            except:
                continue
        
        if len(all_predictions) == 0:
            print(f"  ⚠️  No valid predictions")
            continue
        
        # Calculate metrics
        predictions_arr = np.array(all_predictions)
        actuals_arr = np.array(all_actuals)
        
        # Correlation
        corr, p_value = pearsonr(predictions_arr, actuals_arr)
        
        # Accuracy (threshold at 0.5)
        binary_preds = (predictions_arr >= 0.5).astype(int)
        accuracy = accuracy_score(actuals_arr, binary_preds)
        
        # AUC
        if len(np.unique(actuals_arr)) > 1:
            auc = roc_auc_score(actuals_arr, predictions_arr)
        else:
            auc = None
        
        # Calibration
        calibration_error = np.mean(np.abs(predictions_arr - actuals_arr))
        
        # Actual success rate
        actual_success_rate = actuals_arr.mean()
        predicted_success_rate = predictions_arr.mean()
        
        print(f"\n  RESULTS:")
        print(f"    N: {len(all_predictions)}")
        print(f"    Correlation: r = {corr:.3f} (p = {p_value:.4f})")
        print(f"    Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        if auc:
            print(f"    AUC: {auc:.3f}")
        print(f"    Calibration Error: ±{calibration_error:.3f} ({calibration_error*100:.1f}%)")
        print(f"    Actual success rate: {actual_success_rate:.3f}")
        print(f"    Predicted success rate: {predicted_success_rate:.3f}")
        print(f"    Difference: {abs(actual_success_rate - predicted_success_rate):.3f}")
        
        results[model_name] = {
            'n_samples': len(all_predictions),
            'correlation': corr,
            'p_value': p_value,
            'accuracy': accuracy,
            'auc': auc,
            'calibration_error': calibration_error,
            'actual_success_rate': actual_success_rate,
            'predicted_success_rate': predicted_success_rate,
            'capability_proxy': capability
        }
    
    return results


def main():
    print("="*80)
    print("VALIDATION: Zero-Shot Transfer Using Existing OpenCompass Data")
    print("="*80)
    
    benchmarks_to_check = [
        ('GPQA_diamond', 'reasoning', 'model_hle'),
        ('openai_humaneval', 'coding', 'model_livecodebench'),
        ('IFEval', 'summarization', 'model_ifbench')
    ]
    
    all_results = {}
    
    for benchmark, intent, feature in benchmarks_to_check:
        results = validate_benchmark(benchmark, intent, feature)
        if results:
            all_results[benchmark] = results
    
    # Summary
    print(f"\n{'='*80}")
    print("VALIDATION SUMMARY")
    print("="*80)
    
    if not all_results:
        print("\n❌ No validation results obtained")
        print("\nPossible reasons:")
        print("  1. No proprietary model predictions in OpenCompass")
        print("  2. Models not in our cache")
        print("  3. XGBoost models not trained yet")
        print("\nNext steps:")
        print("  - Train XGBoost models: python3 train_xgboost_tuned.py")
        print("  - Or run manual validation: python3 validate_proprietary_transfer.py")
        return
    
    summary_rows = []
    for benchmark, models in all_results.items():
        for model_name, metrics in models.items():
            summary_rows.append({
                'Benchmark': benchmark,
                'Model': model_name,
                'N': metrics['n_samples'],
                'Correlation': f"{metrics['correlation']:.3f}",
                'P-value': f"{metrics['p_value']:.4f}",
                'Accuracy': f"{metrics['accuracy']:.3f}",
                'AUC': f"{metrics['auc']:.3f}" if metrics['auc'] else 'N/A',
                'Cal.Error': f"±{metrics['calibration_error']:.3f}"
            })
    
    summary_df = pd.DataFrame(summary_rows)
    print("\n" + summary_df.to_string(index=False))
    
    # Save results
    output_dir = Path(__file__).parent / 'validation_results'
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / 'proprietary_validation_summary.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✓ Saved detailed results to {output_file}")
    
    # Calculate overall statistics
    all_correlations = [m['correlation'] for models in all_results.values() for m in models.values()]
    all_accuracies = [m['accuracy'] for models in all_results.values() for m in models.values()]
    
    print(f"\nOVERALL STATISTICS:")
    print(f"  Mean correlation: {np.mean(all_correlations):.3f}")
    print(f"  Mean accuracy: {np.mean(all_accuracies):.3f}")
    print(f"  Total proprietary models validated: {len(summary_rows)}")
    print(f"  Total predictions validated: {sum(row['N'] for _, models in all_results.items() for _, row in models.items())}")


if __name__ == '__main__':
    main()
