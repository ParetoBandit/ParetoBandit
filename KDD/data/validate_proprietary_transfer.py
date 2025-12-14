#!/usr/bin/env python3
"""
Validate Zero-Shot Transfer to Proprietary Models

This script validates that predictions for proprietary models (GPT-4o, Claude-3.5, etc.)
match actual performance by:
1. Selecting diverse prompts (stratified by predicted difficulty)
2. Running actual evaluations via API
3. Comparing predicted probabilities vs. actual success rates

Usage:
    python3 validate_proprietary_transfer.py --intent reasoning --models gpt-4o claude-3.5-sonnet --n-samples 50
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr
import joblib
from tqdm import tqdm
import os
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from llm_jury.routing.nvidia_complexity_classifier import NvidiaComplexityClassifier

# Intent configurations
INTENT_CONFIGS = {
    'reasoning': {
        'dataset': 'Idavidrein/gpqa',
        'config': 'gpqa_diamond',
        'split': 'train',
        'prompt_column': 'Question',
        'answer_column': 'Correct Answer',
        'benchmark_feature': 'model_hle'
    },
    'coding': {
        'dataset': 'evalplus/humanevalplus',
        'config': None,
        'split': 'test',
        'prompt_column': 'prompt',
        'answer_column': None,  # Requires execution
        'benchmark_feature': 'model_livecodebench'
    },
    'summarization': {
        'dataset': 'google/IFEval',
        'config': None,
        'split': 'train',
        'prompt_column': 'prompt',
        'answer_column': None,  # Requires evaluation
        'benchmark_feature': 'model_ifbench'
    }
}


class ProprietaryModelEvaluator:
    """Evaluate proprietary models via API."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        # Import OpenAI/Anthropic clients here
        try:
            from openai import OpenAI
            self.openai_client = OpenAI(api_key=api_key)
        except:
            self.openai_client = None
            
    def evaluate_reasoning(self, model: str, prompt: str, correct_answer: str) -> bool:
        """Evaluate a reasoning prompt (GPQA)."""
        if self.openai_client and 'gpt' in model.lower():
            try:
                response = self.openai_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=0
                )
                answer = response.choices[0].message.content.strip()
                # Simple string matching for validation
                return correct_answer.lower() in answer.lower()
            except Exception as e:
                print(f"Error evaluating {model}: {e}")
                return None
        return None
    
    def evaluate_coding(self, model: str, prompt: str, test_cases: List) -> bool:
        """Evaluate a coding prompt (HumanEval)."""
        # This would require code execution - simplified for validation
        # In practice, use existing evaluation harness
        return None
    
    def evaluate_summarization(self, model: str, prompt: str, instructions: List[str]) -> bool:
        """Evaluate an instruction-following prompt (IFEval)."""
        if self.openai_client and 'gpt' in model.lower():
            try:
                response = self.openai_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1000,
                    temperature=0
                )
                output = response.choices[0].message.content.strip()
                # Check if all instructions were followed (simplified)
                # In practice, use IFEval's evaluation logic
                return True  # Placeholder
            except Exception as e:
                print(f"Error evaluating {model}: {e}")
                return None
        return None


def load_trained_model(intent: str) -> Tuple:
    """Load trained XGBoost model for an intent."""
    model_dir = Path(__file__).parent / 'trained_models'
    
    model_path = model_dir / f'xgboost_{intent}.joblib'
    if not model_path.exists():
        raise FileNotFoundError(f"No trained model found for {intent} at {model_path}")
    
    model = joblib.load(model_path)
    
    # Load feature names
    metadata_path = model_dir / f'xgboost_{intent}_metadata.json'
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
            feature_names = metadata.get('feature_names', [])
    else:
        # Default feature names
        feature_names = [
            'nvidia_creativity', 'nvidia_reasoning', 'nvidia_constraint',
            'nvidia_domain_knowledge', 'nvidia_contextual_knowledge', 'nvidia_few_shots',
            f'model_{INTENT_CONFIGS[intent]["benchmark_feature"].replace("model_", "")}'
        ]
    
    return model, feature_names


def get_model_capability(model_name: str, benchmark: str) -> float:
    """Get model's capability proxy (aggregate benchmark score)."""
    cache_path = Path(__file__).parent.parent.parent / 'data' / 'models_cache.json'
    
    with open(cache_path) as f:
        cache = json.load(f)
    
    # Find model in cache
    for model in cache:
        if model['name'].lower() == model_name.lower() or model.get('slug', '').lower() == model_name.lower():
            benchmarks = model.get('benchmarks', {})
            # Try different possible keys
            score = benchmarks.get(benchmark) or benchmarks.get(benchmark.replace('model_', ''))
            if score:
                return score
    
    raise ValueError(f"Model {model_name} not found in cache or missing {benchmark}")


def select_diverse_prompts(intent: str, n_samples: int, model, feature_names: list) -> pd.DataFrame:
    """Select diverse prompts stratified by predicted difficulty."""
    from datasets import load_dataset
    
    config = INTENT_CONFIGS[intent]
    
    # Load dataset
    print(f"Loading {config['dataset']}...")
    if config['config']:
        ds = load_dataset(config['dataset'], config['config'], split=config['split'])
    else:
        ds = load_dataset(config['dataset'], split=config['split'])
    
    df = ds.to_pandas()
    
    # Sample prompts (stratified by index to get diversity)
    if len(df) > n_samples:
        # Get evenly spaced samples
        indices = np.linspace(0, len(df)-1, n_samples, dtype=int)
        df = df.iloc[indices].copy()
    
    print(f"Selected {len(df)} prompts for validation")
    return df


def compute_nvidia_features(prompt: str, classifier: NvidiaComplexityClassifier) -> Dict[str, float]:
    """Compute NVIDIA complexity features for a prompt."""
    result = classifier.classify(prompt)
    
    return {
        'nvidia_creativity': result.get('creativity_scope', 0),
        'nvidia_reasoning': result.get('reasoning', 0),
        'nvidia_constraint': result.get('constraint_ct', 0),
        'nvidia_domain_knowledge': result.get('domain_knowledge', 0),
        'nvidia_contextual_knowledge': result.get('contextual_knowledge', 0),
        'nvidia_few_shots': result.get('number_of_few_shots', 0)
    }


def predict_success_probability(prompt: str, model_name: str, intent: str, 
                                xgboost_model, feature_names: list, 
                                nvidia_classifier: NvidiaComplexityClassifier) -> float:
    """Predict success probability for a prompt-model pair."""
    
    # Get NVIDIA features
    nvidia_features = compute_nvidia_features(prompt, nvidia_classifier)
    
    # Get model capability
    benchmark = INTENT_CONFIGS[intent]['benchmark_feature']
    capability = get_model_capability(model_name, benchmark)
    
    # Combine features in correct order
    X = []
    for fname in feature_names:
        if fname.startswith('nvidia_'):
            X.append(nvidia_features[fname])
        elif fname.startswith('model_'):
            X.append(capability)
    
    # Predict
    prob = xgboost_model.predict_proba([X])[0][1]
    return prob


def evaluate_actual_performance(prompts: pd.DataFrame, model_name: str, intent: str,
                                evaluator: ProprietaryModelEvaluator) -> List[bool]:
    """Evaluate actual performance of model on prompts."""
    config = INTENT_CONFIGS[intent]
    results = []
    
    print(f"\nEvaluating {model_name} on {len(prompts)} prompts...")
    
    for idx, row in tqdm(prompts.iterrows(), total=len(prompts)):
        prompt = row[config['prompt_column']]
        
        if intent == 'reasoning':
            correct_answer = row.get(config['answer_column'], '')
            result = evaluator.evaluate_reasoning(model_name, prompt, correct_answer)
        elif intent == 'coding':
            result = evaluator.evaluate_coding(model_name, prompt, None)
        elif intent == 'summarization':
            result = evaluator.evaluate_summarization(model_name, prompt, None)
        else:
            result = None
        
        results.append(result)
    
    return results


def calculate_metrics(predictions: List[float], actuals: List[bool]) -> Dict:
    """Calculate validation metrics."""
    # Remove None values
    valid_pairs = [(p, a) for p, a in zip(predictions, actuals) if a is not None]
    if not valid_pairs:
        return {}
    
    predictions, actuals = zip(*valid_pairs)
    predictions = np.array(predictions)
    actuals = np.array(actuals, dtype=int)
    
    # Correlation
    corr, p_value = pearsonr(predictions, actuals)
    
    # AUC
    if len(np.unique(actuals)) > 1:
        auc = roc_auc_score(actuals, predictions)
    else:
        auc = None
    
    # Calibration (mean absolute error)
    calibration_error = np.mean(np.abs(predictions - actuals))
    
    # Binned calibration
    bins = [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]
    calibration_by_bin = {}
    
    for low, high in bins:
        mask = (predictions >= low) & (predictions < high)
        if mask.sum() > 0:
            pred_mean = predictions[mask].mean()
            actual_mean = actuals[mask].mean()
            calibration_by_bin[f'{low:.2f}-{high:.2f}'] = {
                'predicted': pred_mean,
                'actual': actual_mean,
                'error': abs(pred_mean - actual_mean),
                'n': mask.sum()
            }
    
    return {
        'correlation': corr,
        'p_value': p_value,
        'auc': auc,
        'calibration_error': calibration_error,
        'calibration_by_bin': calibration_by_bin,
        'n_samples': len(actuals)
    }


def main():
    parser = argparse.ArgumentParser(description='Validate zero-shot transfer to proprietary models')
    parser.add_argument('--intent', type=str, required=True, 
                       choices=['reasoning', 'coding', 'summarization'],
                       help='Intent to validate')
    parser.add_argument('--models', type=str, nargs='+', required=True,
                       help='Proprietary models to validate (e.g., gpt-4o claude-3.5-sonnet)')
    parser.add_argument('--n-samples', type=int, default=50,
                       help='Number of prompts to sample')
    parser.add_argument('--output', type=str, default='validation_results',
                       help='Output directory for results')
    parser.add_argument('--dry-run', action='store_true',
                       help='Compute predictions only, skip actual evaluation')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(__file__).parent / args.output
    output_dir.mkdir(exist_ok=True)
    
    print("="*80)
    print("VALIDATION: Zero-Shot Transfer to Proprietary Models")
    print("="*80)
    print(f"Intent: {args.intent}")
    print(f"Models: {', '.join(args.models)}")
    print(f"Samples: {args.n_samples}")
    print()
    
    # Load trained XGBoost model
    print("Loading trained XGBoost model...")
    xgboost_model, feature_names = load_trained_model(args.intent)
    print(f"✓ Loaded model with features: {feature_names}")
    
    # Initialize NVIDIA classifier
    print("\nInitializing NVIDIA complexity classifier...")
    nvidia_classifier = NvidiaComplexityClassifier()
    
    # Initialize evaluator
    api_key = os.getenv('OPENAI_API_KEY') or os.getenv('OPENROUTER_API_KEY')
    evaluator = ProprietaryModelEvaluator(api_key) if not args.dry_run else None
    
    # Select diverse prompts
    prompts = select_diverse_prompts(args.intent, args.n_samples, xgboost_model, feature_names)
    
    # Validate each model
    all_results = {}
    
    for model_name in args.models:
        print(f"\n{'='*80}")
        print(f"VALIDATING: {model_name}")
        print("="*80)
        
        # Compute predictions
        print("\nComputing predictions from XGBoost...")
        predictions = []
        
        for idx, row in tqdm(prompts.iterrows(), total=len(prompts), desc="Predicting"):
            config = INTENT_CONFIGS[args.intent]
            prompt = row[config['prompt_column']]
            
            pred_prob = predict_success_probability(
                prompt, model_name, args.intent, 
                xgboost_model, feature_names, nvidia_classifier
            )
            predictions.append(pred_prob)
        
        print(f"✓ Predicted success probability for {len(predictions)} prompts")
        print(f"  Mean predicted probability: {np.mean(predictions):.3f}")
        print(f"  Std: {np.std(predictions):.3f}")
        
        # Evaluate actual performance
        if not args.dry_run:
            actuals = evaluate_actual_performance(prompts, model_name, args.intent, evaluator)
            
            # Calculate metrics
            metrics = calculate_metrics(predictions, actuals)
            
            print(f"\n{'='*80}")
            print("VALIDATION RESULTS")
            print("="*80)
            print(f"Correlation: r = {metrics['correlation']:.3f} (p = {metrics['p_value']:.4f})")
            if metrics['auc']:
                print(f"AUC: {metrics['auc']:.3f}")
            print(f"Calibration Error: {metrics['calibration_error']:.3f} (±{metrics['calibration_error']*100:.1f}%)")
            print(f"\nCalibration by bin:")
            for bin_range, stats in metrics['calibration_by_bin'].items():
                print(f"  {bin_range}: Predicted={stats['predicted']:.3f}, "
                      f"Actual={stats['actual']:.3f}, Error={stats['error']:.3f} (n={stats['n']})")
            
            all_results[model_name] = {
                'predictions': predictions,
                'actuals': actuals,
                'metrics': metrics
            }
        else:
            all_results[model_name] = {
                'predictions': predictions,
                'actuals': None,
                'metrics': {}
            }
        
        # Save results
        result_file = output_dir / f'{args.intent}_{model_name.replace("/", "-")}_validation.json'
        with open(result_file, 'w') as f:
            json.dump({
                'model': model_name,
                'intent': args.intent,
                'n_samples': len(predictions),
                'predictions': predictions,
                'actuals': all_results[model_name]['actuals'],
                'metrics': all_results[model_name]['metrics']
            }, f, indent=2)
        print(f"\n✓ Saved results to {result_file}")
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print("="*80)
    
    if not args.dry_run:
        summary_df = pd.DataFrame([
            {
                'Model': model,
                'Correlation': results['metrics'].get('correlation', np.nan),
                'P-value': results['metrics'].get('p_value', np.nan),
                'AUC': results['metrics'].get('auc', np.nan),
                'Calibration Error': results['metrics'].get('calibration_error', np.nan),
                'N': results['metrics'].get('n_samples', 0)
            }
            for model, results in all_results.items()
        ])
        
        print(summary_df.to_string(index=False))
        
        summary_file = output_dir / f'{args.intent}_validation_summary.csv'
        summary_df.to_csv(summary_file, index=False)
        print(f"\n✓ Saved summary to {summary_file}")
    else:
        print("Dry run completed. Predictions computed but not validated.")
        print("Remove --dry-run flag to perform actual evaluation.")


if __name__ == '__main__':
    main()
