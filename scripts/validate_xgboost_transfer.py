#!/usr/bin/env python3
"""
Validate XGBoost Model Predictions Against Actual Benchmark Performance.

This script tests if our XGBoost quality predictions correlate with
actual benchmark performance across proprietary and open-source models.

Validation approach:
1. Load production XGBoost models (trained on open-source data)
2. Generate predictions for test prompts
3. Compare predictions against actual benchmark performance
4. Measure correlation for proprietary vs open-source models
"""

import json
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

# Add repo root to path
import sys
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))


def load_models_cache():
    """Load models cache with benchmark data."""
    cache_path = repo_root / 'data' / 'models_cache.json'
    with open(cache_path) as f:
        data = json.load(f)
    return data.get('models', data)


def load_test_prompts():
    """Load coding prompts for testing."""
    prompts_path = repo_root / 'data' / 'coding_samples_500.json'
    with open(prompts_path) as f:
        data = json.load(f)
    return data['samples']


def classify_model(model_data):
    """Classify model as proprietary or open-source."""
    license_val = model_data.get('license', model_data.get('openlm_license', '')).lower()
    if 'proprietary' in license_val:
        return 'proprietary'
    return 'open_source'


def get_actual_benchmark_score(model_data, intent):
    """Get actual benchmark score for an intent."""
    benchmark_fields = {
        'coding': 'livecodebench',
        'reasoning': 'gpqa',
        'summarization': 'summedits_score',
        'rag': 'mmlu_pro',
    }
    
    field = benchmark_fields.get(intent)
    if not field:
        return None
    
    score = model_data.get(field)
    if score is None or score == 'N/A':
        return None
    
    try:
        score = float(score)
        # Normalize to 0-100 if needed
        if score <= 1.0:
            score *= 100
        return score
    except:
        return None


def validate_intent(intent, models, predictor, test_prompts, n_prompts=20):
    """Validate zero-shot transfer for a specific intent."""
    print(f"\n{'='*80}")
    print(f"VALIDATING: {intent.upper()}")
    print(f"{'='*80}")
    
    # Get models with valid benchmark scores
    valid_models = []
    for m in models:
        score = get_actual_benchmark_score(m, intent)
        if score is not None:
            valid_models.append({
                'name': m['name'],
                'data': m,
                'actual_score': score,
                'model_type': classify_model(m),
            })
    
    print(f"\nModels with {intent} benchmark data: {len(valid_models)}")
    
    proprietary_models = [m for m in valid_models if m['model_type'] == 'proprietary']
    open_source_models = [m for m in valid_models if m['model_type'] == 'open_source']
    
    print(f"  Proprietary: {len(proprietary_models)}")
    print(f"  Open Source: {len(open_source_models)}")
    
    if len(valid_models) < 5:
        print(f"  ⚠️  Not enough models for validation!")
        return None
    
    # Generate predictions for each model across test prompts
    print(f"\nGenerating predictions for {n_prompts} prompts...")
    
    results = []
    for model_info in valid_models:
        model_data = model_info['data']
        predictions = []
        
        for prompt_data in test_prompts[:n_prompts]:
            prompt = prompt_data['prompt']
            try:
                pred = predictor.predict_quality(prompt, model_data, intent)
                predictions.append(pred)
            except:
                pass
        
        if predictions:
            avg_prediction = np.mean(predictions)
            results.append({
                'name': model_info['name'],
                'actual': model_info['actual_score'],
                'predicted': avg_prediction * 100,  # Scale to 0-100
                'model_type': model_info['model_type'],
            })
    
    if len(results) < 5:
        print(f"  ⚠️  Not enough predictions for validation!")
        return None
    
    # Convert to arrays
    actuals = np.array([r['actual'] for r in results])
    predictions = np.array([r['predicted'] for r in results])
    
    # Overall correlation
    corr, p_value = pearsonr(predictions, actuals)
    spearman_corr, spearman_p = spearmanr(predictions, actuals)
    
    print(f"\nOVERALL RESULTS (N={len(results)} models):")
    print(f"  Pearson Correlation:  r = {corr:.3f} (p = {p_value:.4f})")
    print(f"  Spearman Correlation: ρ = {spearman_corr:.3f} (p = {spearman_p:.4f})")
    
    # Proprietary models only
    prop_results = [r for r in results if r['model_type'] == 'proprietary']
    if len(prop_results) >= 3:
        prop_actuals = np.array([r['actual'] for r in prop_results])
        prop_predictions = np.array([r['predicted'] for r in prop_results])
        prop_corr, prop_p = pearsonr(prop_predictions, prop_actuals)
        print(f"\nPROPRIETARY MODELS ONLY (N={len(prop_results)}):")
        print(f"  Pearson Correlation:  r = {prop_corr:.3f} (p = {prop_p:.4f})")
    else:
        prop_corr, prop_p = np.nan, np.nan
        print(f"\nPROPRIETARY MODELS: Not enough data ({len(prop_results)} models)")
    
    # Open source models only
    os_results = [r for r in results if r['model_type'] == 'open_source']
    if len(os_results) >= 3:
        os_actuals = np.array([r['actual'] for r in os_results])
        os_predictions = np.array([r['predicted'] for r in os_results])
        os_corr, os_p = pearsonr(os_predictions, os_actuals)
        print(f"\nOPEN SOURCE MODELS ONLY (N={len(os_results)}):")
        print(f"  Pearson Correlation:  r = {os_corr:.3f} (p = {os_p:.4f})")
    else:
        os_corr, os_p = np.nan, np.nan
    
    # Quality assessment
    if corr > 0.60:
        quality = "✅ EXCELLENT"
    elif corr > 0.50:
        quality = "✅ GOOD"
    elif corr > 0.40:
        quality = "⚠️  MODERATE"
    elif corr > 0.30:
        quality = "⚠️  WEAK"
    else:
        quality = "❌ POOR"
    
    print(f"\n  Transfer Quality: {quality}")
    
    # Show top/bottom predictions vs actuals
    print(f"\nTOP 5 by PREDICTED:")
    sorted_by_pred = sorted(results, key=lambda x: x['predicted'], reverse=True)
    for r in sorted_by_pred[:5]:
        print(f"  {r['predicted']:.1f}% pred, {r['actual']:.1f}% actual - {r['name']} ({r['model_type']})")
    
    print(f"\nBOTTOM 5 by PREDICTED:")
    for r in sorted_by_pred[-5:]:
        print(f"  {r['predicted']:.1f}% pred, {r['actual']:.1f}% actual - {r['name']} ({r['model_type']})")
    
    return {
        'intent': intent,
        'n_models': len(results),
        'n_proprietary': len(prop_results),
        'n_open_source': len(os_results),
        'overall_correlation': corr,
        'overall_p_value': p_value,
        'spearman_correlation': spearman_corr,
        'proprietary_correlation': prop_corr,
        'proprietary_p_value': prop_p,
        'open_source_correlation': os_corr,
        'quality': quality,
        'results': results,
    }


def main():
    """Run validation for all intents."""
    print("="*80)
    print("XGBOOST QUALITY PREDICTION VALIDATION")
    print("Testing correlation with actual benchmark performance")
    print("="*80)
    
    # Load data
    print("\nLoading models cache...")
    models = load_models_cache()
    print(f"  Loaded {len(models)} models")
    
    print("\nLoading test prompts...")
    test_prompts = load_test_prompts()
    print(f"  Loaded {len(test_prompts)} prompts")
    
    # Initialize predictor
    print("\nInitializing XGBoost predictor...")
    from llm_jury.optimization.xgboost_quality import XGBoostQualityPredictor
    predictor = XGBoostQualityPredictor()
    
    # Validate each intent
    intents = ['coding', 'reasoning', 'rag', 'summarization']
    all_results = {}
    
    for intent in intents:
        result = validate_intent(intent, models, predictor, test_prompts)
        if result:
            all_results[intent] = result
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY: ZERO-SHOT TRANSFER VALIDATION")
    print(f"{'='*80}")
    
    print(f"\n{'Intent':<15} {'N Models':<10} {'Correlation':<15} {'Significance':<15} {'Quality':<15}")
    print("-"*70)
    
    for intent, result in all_results.items():
        sig = "***" if result['overall_p_value'] < 0.001 else ("**" if result['overall_p_value'] < 0.01 else ("*" if result['overall_p_value'] < 0.05 else ""))
        print(f"{intent:<15} {result['n_models']:<10} r={result['overall_correlation']:.3f}{sig:<8} p={result['overall_p_value']:.4f}{'':<6} {result['quality']}")
    
    print(f"\n{'='*80}")
    print("PROPRIETARY MODEL TRANSFER (Key Validation)")
    print(f"{'='*80}")
    
    for intent, result in all_results.items():
        if not np.isnan(result['proprietary_correlation']):
            sig = "***" if result['proprietary_p_value'] < 0.001 else ("**" if result['proprietary_p_value'] < 0.01 else ("*" if result['proprietary_p_value'] < 0.05 else ""))
            print(f"{intent:<15} N={result['n_proprietary']:<5} r={result['proprietary_correlation']:.3f}{sig}")
        else:
            print(f"{intent:<15} Insufficient proprietary model data")
    
    # Overall assessment
    avg_corr = np.mean([r['overall_correlation'] for r in all_results.values()])
    print(f"\n{'='*80}")
    print(f"OVERALL AVERAGE CORRELATION: r = {avg_corr:.3f}")
    
    if avg_corr > 0.50:
        print("✅ ZERO-SHOT TRANSFER VALIDATED!")
        print("   XGBoost predictions correlate significantly with actual benchmark performance")
    elif avg_corr > 0.40:
        print("⚠️  MODERATE TRANSFER - Consider improvements")
    else:
        print("❌ WEAK TRANSFER - Needs significant improvement")
    
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
