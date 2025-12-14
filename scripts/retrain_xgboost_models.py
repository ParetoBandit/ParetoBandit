#!/usr/bin/env python3
"""
Retrain XGBoost Models with Updated Capability Fields

This script retrains the XGBoost models for all 4 intents using:
- New capability fields: livecodebench, summedits_score, gpqa, mmlu_pro
- Proper k-fold cross-validation (5-fold)
- Held-out test set (15%)
- Group-based splitting (models as groups to prevent leakage)

Output:
- 4 trained XGBoost models (.joblib files)
- Model cards with performance metrics
- Training summary
"""

# CRITICAL: Import torch FIRST to avoid segfaults on Mac
# This must happen before any other imports that might trigger torch loading
try:
    import torch
except ImportError:
    pass

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import joblib
from datetime import datetime
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from xgboost import XGBClassifier
from sklearn.model_selection import (
    StratifiedKFold, 
    GroupKFold,
    cross_val_score,
    train_test_split
)
from sklearn.metrics import (
    accuracy_score, 
    roc_auc_score, 
    classification_report,
    precision_recall_fscore_support
)
from scipy.stats import pearsonr, spearmanr

# Try to import from llm_jury
try:
    from llm_jury.prediction.models import OPENCOMPASS_TO_CACHE
    print(f"✓ Loaded {len(OPENCOMPASS_TO_CACHE)} OpenCompass->Cache mappings")
except ImportError as e:
    print(f"⚠️  Could not import OPENCOMPASS_TO_CACHE: {e}")
    OPENCOMPASS_TO_CACHE = {}


@dataclass
class IntentConfig:
    """Configuration for each intent."""
    name: str
    capability_field: str
    benchmark_name: str
    
    
# Intent configurations with NEW capability fields
INTENT_CONFIGS = {
    'reasoning': IntentConfig(
        name='reasoning',
        capability_field='gpqa',
        benchmark_name='GPQA Diamond'
    ),
    'coding': IntentConfig(
        name='coding', 
        capability_field='livecodebench',
        benchmark_name='LiveCodeBench'
    ),
    'summarization': IntentConfig(
        name='summarization',
        capability_field='summedits_score',
        benchmark_name='SummEdits'
    ),
    'rag': IntentConfig(
        name='rag',
        capability_field='mmlu_pro',
        benchmark_name='MMLU-Pro'
    )
}


def load_models_cache() -> Dict:
    """Load the models cache with capability scores."""
    cache_paths = [
        Path(__file__).parent.parent / 'data' / 'models_cache.json',
        Path(__file__).parent.parent / 'llm_jury' / 'data' / 'models_cache.json',
    ]
    
    for cache_path in cache_paths:
        if cache_path.exists():
            with open(cache_path) as f:
                data = json.load(f)
            print(f"✓ Loaded models cache from {cache_path}")
            return data
    
    raise FileNotFoundError("Could not find models_cache.json")


def get_capability_scores(models_cache: Dict, capability_field: str) -> Dict[str, float]:
    """Extract capability scores for all models."""
    capability_map = {}
    
    for model in models_cache.get('models', []):
        name = model.get('name', '')
        
        # Get the capability score
        score = model.get(capability_field)
        
        if score is not None and score != 'N/A':
            # Handle percentage vs decimal
            score = float(score)
            if capability_field in ['livecodebench', 'gpqa'] and score < 1:
                score = score * 100  # Convert to percentage
            capability_map[name] = score
    
    return capability_map


def download_livecodebench_real_results() -> Optional[pd.DataFrame]:
    """
    Download REAL LiveCodeBench evaluated results from the official leaderboard.
    
    This downloads actual pass@1 scores from livecodebench.github.io
    which contains real code execution results, not synthetic labels.
    
    Returns DataFrame with columns: model, prompt, success, difficulty
    """
    import requests
    from huggingface_hub import hf_hub_download
    
    print("Downloading REAL LiveCodeBench results from leaderboard...")
    
    # Step 1: Download evaluated results from leaderboard
    url = 'https://livecodebench.github.io/performances_generation.json'
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        
        df_results = pd.DataFrame(data['performances'])
        print(f"✓ Downloaded {len(df_results)} evaluated results from {df_results['model'].nunique()} models")
    except Exception as e:
        print(f"❌ Error downloading leaderboard: {e}")
        return None
    
    # Step 2: Load prompts from LiveCodeBench
    try:
        local_path = hf_hub_download('livecodebench/code_generation', 'test.jsonl', repo_type='dataset')
        
        prompts = []
        with open(local_path) as f:
            for line in f:
                prompts.append(json.loads(line))
        
        df_prompts = pd.DataFrame(prompts)
        print(f"✓ Loaded {len(df_prompts)} prompts")
    except Exception as e:
        print(f"❌ Error loading prompts: {e}")
        return None
    
    # Step 3: Join results with prompts on question_id
    df_merged = df_results.merge(
        df_prompts[['question_id', 'question_content']], 
        on='question_id', 
        how='inner'
    )
    
    print(f"✓ Merged: {len(df_merged)} examples")
    
    # Step 4: Convert pass@1 (0-100) to binary success
    # pass@1 > 50 means more likely to pass than fail
    df_merged['success'] = (df_merged['pass@1'] >= 50).astype(int)
    
    # Prepare output DataFrame
    df_out = df_merged.rename(columns={'question_content': 'prompt'})[
        ['model', 'prompt', 'success', 'difficulty', 'question_id']
    ]
    
    print(f"\n✓ Final dataset:")
    print(f"  Total examples: {len(df_out)}")
    print(f"  Models: {df_out['model'].nunique()}")
    print(f"  Problems: {df_out['question_id'].nunique()}")
    print(f"  Success rate: {df_out['success'].mean():.1%}")
    print(f"  By difficulty:")
    for d in ['easy', 'medium', 'hard']:
        rate = df_out[df_out['difficulty'] == d]['success'].mean()
        print(f"    {d}: {rate:.1%}")
    
    return df_out


def download_training_data(intent: str) -> Optional[pd.DataFrame]:
    """
    Download instance-level training data from OpenCompass.
    
    Returns DataFrame with columns: model, prompt, success, nvidia_features
    """
    print(f"\n{'='*60}")
    print(f"DOWNLOADING TRAINING DATA FOR: {intent.upper()}")
    print(f"{'='*60}")
    
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
        from datasets import load_dataset
        import os
        
        # Note: Don't use HF_TOKEN for OpenCompass - it's public and token causes 401 errors
        HF_TOKEN = None  # os.getenv('HUGGINGFACE_API_KEY')
        
        repo_id = "opencompass/compass_academic_predictions"
        
        # Map intent to benchmark name in OpenCompass
        benchmark_map = {
            'reasoning': 'GPQA_diamond',
            'coding': 'lcb_code_generation',  # LiveCodeBench - harder than HumanEval
            'summarization': 'IFEval',
            'rag': 'triviaqa_wiki_1shot'
        }
        
        benchmark_name = benchmark_map.get(intent)
        if not benchmark_name:
            print(f"⚠️  Unknown intent: {intent}")
            return None
        
        # List available prediction files
        print(f"Looking for {benchmark_name} predictions...")
        files = list_repo_files(repo_id, repo_type='dataset', token=HF_TOKEN)
        prediction_files = [f for f in files if f.endswith('.json') and benchmark_name in f]
        
        print(f"Found {len(prediction_files)} prediction files")
        
        if not prediction_files:
            print(f"⚠️  No predictions found for {benchmark_name}")
            return None
        
        # Filter to models we can map
        if OPENCOMPASS_TO_CACHE:
            mapped_files = [f for f in prediction_files 
                          if Path(f).stem in OPENCOMPASS_TO_CACHE]
            print(f"✓ {len(mapped_files)} models match our cache")
            prediction_files = mapped_files if mapped_files else prediction_files[:20]
        else:
            prediction_files = prediction_files[:20]  # Limit if no mapping
        
        # Load prompts dataset
        print(f"\nLoading prompts...")
        if intent == 'reasoning':
            dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train", token=HF_TOKEN)
            prompts_df = dataset.to_pandas()
            prompt_col = 'Question'
        elif intent == 'coding':
            # Use REAL LiveCodeBench results from official leaderboard
            return download_livecodebench_real_results()
        elif intent == 'summarization':
            dataset = load_dataset("google/IFEval", split="train", token=HF_TOKEN)
            prompts_df = dataset.to_pandas()
            prompt_col = 'prompt'
        elif intent == 'rag':
            dataset = load_dataset("trivia_qa", "unfiltered.nocontext", split="validation[:5000]", token=HF_TOKEN)
            prompts_df = dataset.to_pandas()
            prompt_col = 'question'
        
        prompts_df['question_id'] = prompts_df.index.astype(str)
        print(f"✓ Loaded {len(prompts_df)} prompts")
        
        # Download and process predictions
        all_data = []
        
        for file_path in prediction_files[:30]:  # Limit to 30 models
            try:
                local_path = hf_hub_download(repo_id, file_path, repo_type='dataset', token=HF_TOKEN)
                
                with open(local_path) as f:
                    predictions = json.load(f)
                
                model_name = Path(file_path).stem
                
                # Grade predictions - use min of predictions and prompts length
                n_examples = min(len(predictions), len(prompts_df))
                if n_examples == 0:
                    continue
                
                for idx, pred in enumerate(predictions[:n_examples]):
                    if idx >= len(prompts_df):
                        break
                    
                    # Extract success label based on intent
                    if intent == 'reasoning':
                        # Multiple choice: compare extracted answer to gold
                        pred_text = pred.get('prediction', '')
                        gold = pred.get('gold', '')
                        
                        # Extract A/B/C/D from prediction
                        import re
                        match = re.search(r'\b([A-D])\b', str(pred_text)[-50:] if pred_text else '')
                        extracted = match.group(1) if match else None
                        success = 1 if extracted == gold else 0
                        
                    # Note: coding uses download_livecodebench_real_results() - returns early above
                    
                    elif intent == 'summarization':
                        # Heuristic: check if response is substantial
                        text = pred.get('prediction', '')
                        success = 1 if (text and len(text.split()) > 20) else 0
                        
                    elif intent == 'rag':
                        # Check if answer contains gold
                        pred_text = str(pred.get('prediction', '')).lower()
                        gold = pred.get('gold', {})
                        
                        if isinstance(gold, dict):
                            gold_answers = gold.get('aliases', []) + [gold.get('value', '')]
                        elif isinstance(gold, list):
                            gold_answers = gold
                        else:
                            gold_answers = [str(gold)]
                        
                        success = 0
                        for ans in gold_answers:
                            if ans and str(ans).lower() in pred_text:
                                success = 1
                                break
                    
                    all_data.append({
                        'model': model_name,
                        'prompt': prompts_df.iloc[idx][prompt_col] if idx < len(prompts_df) else '',
                        'question_id': str(idx),
                        'success': success,
                        'intent': intent
                    })
                
                print(f"  ✓ {model_name}: {len([d for d in all_data if d['model'] == model_name])} examples")
                
            except Exception as e:
                print(f"  ⚠️  Error with {file_path}: {str(e)[:50]}")
                continue
        
        if not all_data:
            print(f"⚠️  No training data collected for {intent}")
            return None
        
        df = pd.DataFrame(all_data)
        print(f"\n✓ Total: {len(df)} training examples from {df['model'].nunique()} models")
        print(f"  Success rate: {df['success'].mean():.1%}")
        
        return df
        
    except Exception as e:
        print(f"⚠️  Error downloading data: {e}")
        import traceback
        traceback.print_exc()
        return None


def compute_nvidia_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute NVIDIA complexity features for prompts."""
    print(f"\nComputing NVIDIA features for {df['prompt'].nunique()} unique prompts...")
    
    try:
        # Import with path already set
        from llm_jury.routing.nvidia_complexity_classifier import NvidiaComplexityClassifier
        print("✓ NVIDIA classifier imported successfully")
        
        classifier = NvidiaComplexityClassifier()
        
        # Get unique prompts
        unique_prompts = df['prompt'].unique().tolist()
        valid_prompts = [p for p in unique_prompts if p and isinstance(p, str) and len(str(p).strip()) > 0]
        
        print(f"Processing {len(valid_prompts)} valid prompts...")
        
        # Process in batches
        batch_size = 16
        prompt_to_features = {}
        
        from tqdm import tqdm
        for i in tqdm(range(0, len(valid_prompts), batch_size), desc="NVIDIA features"):
            batch = valid_prompts[i:i+batch_size]
            try:
                results = classifier.classify_batch(batch)
                for prompt, result in zip(batch, results):
                    prompt_to_features[prompt] = {
                        'nvidia_creativity': result.creativity_scope,
                        'nvidia_reasoning': result.reasoning,
                        'nvidia_constraint': result.constraint_ct,
                        'nvidia_domain_knowledge': result.domain_knowledge,
                        'nvidia_contextual_knowledge': result.contextual_knowledge,
                        'nvidia_few_shots': result.number_of_few_shots,
                    }
            except Exception as e:
                print(f"  ⚠️  Batch error: {e}")
                # Use defaults for failed batch
                for prompt in batch:
                    prompt_to_features[prompt] = {
                        'nvidia_creativity': 0.5,
                        'nvidia_reasoning': 0.5,
                        'nvidia_constraint': 0.5,
                        'nvidia_domain_knowledge': 0.5,
                        'nvidia_contextual_knowledge': 0.5,
                        'nvidia_few_shots': 0.0,
                    }
        
        # Add features to dataframe
        feature_cols = ['nvidia_creativity', 'nvidia_reasoning', 'nvidia_constraint',
                       'nvidia_domain_knowledge', 'nvidia_contextual_knowledge', 'nvidia_few_shots']
        
        for col in feature_cols:
            df[col] = df['prompt'].map(lambda p: prompt_to_features.get(p, {}).get(col, 0.5))
        
        coverage = df['nvidia_creativity'].notna().sum()
        print(f"✓ NVIDIA features added: {coverage}/{len(df)} rows")
        
        return df
        
    except ImportError as e:
        print(f"⚠️  Could not import NVIDIA classifier: {e}")
        print("   Using default features...")
        
        # Add default features
        for col in ['nvidia_creativity', 'nvidia_reasoning', 'nvidia_constraint',
                   'nvidia_domain_knowledge', 'nvidia_contextual_knowledge', 'nvidia_few_shots']:
            df[col] = 0.5
        df['nvidia_few_shots'] = 0.0
        
        return df


def add_capability_scores(df: pd.DataFrame, capability_map: Dict[str, float], 
                          intent_config: IntentConfig) -> pd.DataFrame:
    """Add capability scores to training data."""
    print(f"\nAdding capability scores ({intent_config.capability_field})...")
    print(f"  Available in capability_map: {len(capability_map)} models")
    print(f"  Sample cache names: {list(capability_map.keys())[:5]}")
    print(f"  Sample OpenCompass names: {df['model'].unique()[:5].tolist()}")
    
    # Map OpenCompass model names to cache names
    def get_capability(model_name):
        # Try via explicit mapping first (most reliable)
        if model_name in OPENCOMPASS_TO_CACHE:
            cache_name = OPENCOMPASS_TO_CACHE[model_name]
            if cache_name in capability_map:
                return capability_map[cache_name]
        
        # Try direct match (cache name == opencompass name)
        if model_name in capability_map:
            return capability_map[model_name]
        
        # Try fuzzy matching on common patterns
        model_lower = model_name.lower()
        for cache_name, score in capability_map.items():
            cache_lower = cache_name.lower()
            # Check if significant parts match
            if 'claude' in model_lower and 'claude' in cache_lower:
                if '3.5' in model_lower and '3.5' in cache_lower:
                    if 'sonnet' in model_lower and 'sonnet' in cache_lower:
                        return score
            if 'gpt-4o' in model_lower and 'gpt-4o' in cache_lower:
                return score
            if 'deepseek' in model_lower and 'deepseek' in cache_lower:
                if 'r1' in model_lower and 'r1' in cache_lower:
                    return score
                if 'v3' in model_lower and 'v3' in cache_lower:
                    return score
            if 'gemini' in model_lower and 'gemini' in cache_lower:
                return score
            if 'qwq' in model_lower and 'qwq' in cache_lower:
                return score
        
        return None
    
    # Test the mapping
    print(f"\n  Testing capability lookups:")
    for model_name in df['model'].unique()[:5]:
        cap = get_capability(model_name)
        mapped_name = OPENCOMPASS_TO_CACHE.get(model_name, 'NO MAPPING')
        print(f"    {model_name} -> {mapped_name} -> {cap}")
    
    df['model_capability'] = df['model'].apply(get_capability)
    
    # Report coverage
    before = len(df)
    df_with_cap = df.dropna(subset=['model_capability']).copy()
    after = len(df_with_cap)
    
    print(f"\n  Capability coverage: {after}/{before} examples ({after/before*100:.1f}%)")
    print(f"  Models with scores: {df_with_cap['model'].nunique() if after > 0 else 0}")
    
    if after == 0:
        print(f"⚠️  No models with {intent_config.capability_field} scores!")
        print("   Falling back to model aggregate calculated from training data...")
        
        # Calculate aggregate from training data itself
        model_aggregates = df.groupby('model')['success'].mean() * 100
        df['model_capability'] = df['model'].map(model_aggregates)
        print(f"   Calculated aggregates for {len(model_aggregates)} models")
        return df
    
    return df_with_cap


def train_xgboost_with_cv(
    X: np.ndarray, 
    y: np.ndarray, 
    groups: Optional[np.ndarray] = None,
    capabilities: Optional[np.ndarray] = None,
    n_folds: int = 5,
    test_size: float = 0.15
) -> Tuple[XGBClassifier, Dict]:
    """
    Train XGBoost with k-fold cross-validation and held-out test set.
    
    Uses STRATIFIED splitting by capability score to ensure train/test
    have similar capability distributions, while ensuring no model
    appears in both train and test (no data leakage).
    
    Args:
        X: Feature matrix
        y: Labels
        groups: Group labels for GroupKFold (e.g., model names)
        capabilities: Capability scores for stratified splitting
        n_folds: Number of CV folds
        test_size: Fraction for held-out test set
    
    Returns:
        Trained model and metrics dictionary
    """
    print(f"\n{'='*60}")
    print(f"TRAINING XGBOOST MODEL")
    print(f"{'='*60}")
    
    print(f"Total samples: {len(X)}")
    print(f"Features: {X.shape[1]}")
    print(f"Success rate: {y.mean():.1%}")
    
    # Split into train and held-out test set
    print(f"\nSplitting into train/test ({1-test_size:.0%}/{test_size:.0%})...")
    
    if groups is not None and capabilities is not None:
        # STRATIFIED GROUP SPLIT: 
        # 1. Group models by capability bins
        # 2. From each bin, select ~test_size% of models for test
        # 3. This ensures similar capability distribution in train/test
        
        # Get unique models and their capabilities
        unique_groups = np.unique(groups)
        group_to_cap = {}
        for g in unique_groups:
            mask = groups == g
            group_to_cap[g] = capabilities[mask][0]  # All samples from same model have same cap
        
        # Bin capabilities into quantiles for stratification
        cap_values = np.array([group_to_cap[g] for g in unique_groups])
        n_bins = min(5, len(unique_groups) // 2)  # At least 2 models per bin
        
        if n_bins >= 2:
            # Create bins based on quantiles
            bin_edges = np.percentile(cap_values, np.linspace(0, 100, n_bins + 1))
            bin_edges[-1] += 0.01  # Ensure max value is included
            group_bins = np.digitize(cap_values, bin_edges[:-1])
            
            # Select test models from each bin (stratified)
            np.random.seed(42)
            test_groups = set()
            
            for bin_id in range(1, n_bins + 1):
                bin_groups = unique_groups[group_bins == bin_id]
                n_test_from_bin = max(1, int(len(bin_groups) * test_size))
                if len(bin_groups) > 0:
                    selected = np.random.choice(bin_groups, min(n_test_from_bin, len(bin_groups)), replace=False)
                    test_groups.update(selected)
            
            print(f"  Stratified split: {len(test_groups)} test models from {n_bins} capability bins")
        else:
            # Fallback to random if too few models
            n_test_groups = max(1, int(len(unique_groups) * test_size))
            np.random.seed(42)
            test_group_indices = np.random.choice(len(unique_groups), n_test_groups, replace=False)
            test_groups = set(unique_groups[test_group_indices])
        
        # Create masks
        test_mask = np.array([g in test_groups for g in groups])
        train_mask = ~test_mask
        
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        groups_train = groups[train_mask]
        cap_train = capabilities[train_mask]
        cap_test = capabilities[test_mask]
        
        # Report capability distribution
        print(f"  Train capabilities: mean={cap_train.mean():.1f}%, range=[{cap_train.min():.1f}%, {cap_train.max():.1f}%]")
        print(f"  Test capabilities:  mean={cap_test.mean():.1f}%, range=[{cap_test.min():.1f}%, {cap_test.max():.1f}%]")
        
        # Verify no overlap
        train_groups = set(groups[train_mask])
        test_groups_actual = set(groups[test_mask])
        overlap = train_groups & test_groups_actual
        if overlap:
            print(f"  ⚠️  WARNING: {len(overlap)} models appear in both train and test!")
        else:
            print(f"  ✓ No overlap: {len(train_groups)} train models, {len(test_groups_actual)} test models")
            
    elif groups is not None:
        # Group-aware split without stratification
        unique_groups = np.unique(groups)
        n_test_groups = max(1, int(len(unique_groups) * test_size))
        
        np.random.seed(42)
        test_group_indices = np.random.choice(len(unique_groups), n_test_groups, replace=False)
        test_groups = set(unique_groups[test_group_indices])
        
        test_mask = np.array([g in test_groups for g in groups])
        train_mask = ~test_mask
        
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        groups_train = groups[train_mask]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        groups_train = None
    
    print(f"  Train: {len(X_train)} samples ({y_train.mean():.1%} success)")
    print(f"  Test:  {len(X_test)} samples ({y_test.mean():.1%} success)")
    
    # K-Fold Cross-Validation on training set
    print(f"\n{n_folds}-Fold Cross-Validation...")
    
    xgb_params = {
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'random_state': 42,
        'eval_metric': 'logloss',
        'tree_method': 'hist'
    }
    
    if groups_train is not None and len(np.unique(groups_train)) >= n_folds:
        cv = GroupKFold(n_splits=n_folds)
        cv_iterator = cv.split(X_train, y_train, groups_train)
    else:
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        cv_iterator = cv.split(X_train, y_train)
    
    cv_aucs = []
    cv_accs = []
    
    for fold, (train_idx, val_idx) in enumerate(cv_iterator):
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
        
        fold_model = XGBClassifier(**xgb_params)
        fold_model.fit(X_fold_train, y_fold_train, verbose=False)
        
        y_val_pred = fold_model.predict(X_fold_val)
        y_val_proba = fold_model.predict_proba(X_fold_val)[:, 1]
        
        fold_auc = roc_auc_score(y_fold_val, y_val_proba)
        fold_acc = accuracy_score(y_fold_val, y_val_pred)
        
        cv_aucs.append(fold_auc)
        cv_accs.append(fold_acc)
        
        print(f"  Fold {fold+1}: AUC={fold_auc:.3f}, Acc={fold_acc:.1%}")
    
    print(f"\n  CV AUC: {np.mean(cv_aucs):.3f} ± {np.std(cv_aucs):.3f}")
    print(f"  CV Acc: {np.mean(cv_accs):.1%} ± {np.std(cv_accs):.1%}")
    
    # Train final model on full training set
    print(f"\nTraining final model on {len(X_train)} samples...")
    final_model = XGBClassifier(**xgb_params)
    final_model.fit(X_train, y_train, verbose=False)
    
    # Evaluate on held-out test set
    print(f"\nEvaluating on held-out test set ({len(X_test)} samples)...")
    
    y_test_pred = final_model.predict(X_test)
    y_test_proba = final_model.predict_proba(X_test)[:, 1]
    
    test_acc = accuracy_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_proba)
    test_corr, test_p = pearsonr(y_test_proba, y_test)
    test_spearman, test_sp = spearmanr(y_test_proba, y_test)
    
    print(f"\n  Test Accuracy: {test_acc:.1%}")
    print(f"  Test AUC: {test_auc:.3f}")
    print(f"  Test Correlation: r={test_corr:.3f} (p={test_p:.2e})")
    print(f"  Test Spearman: ρ={test_spearman:.3f} (p={test_sp:.2e})")
    
    # Classification report
    print(f"\n  Classification Report:")
    report = classification_report(y_test, y_test_pred, target_names=['Fail', 'Success'])
    for line in report.split('\n'):
        if line.strip():
            print(f"    {line}")
    
    # Feature importance
    print(f"\nFeature Importance:")
    feature_names = [
        'nvidia_creativity', 'nvidia_reasoning', 'nvidia_constraint',
        'nvidia_domain_knowledge', 'nvidia_contextual_knowledge', 
        'nvidia_few_shots', 'model_capability'
    ]
    importances = final_model.feature_importances_
    for name, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
        print(f"  {name:30s}: {imp:.1%}")
    
    # Compile metrics
    metrics = {
        'n_total': len(X),
        'n_train': len(X_train),
        'n_test': len(X_test),
        'train_success_rate': float(y_train.mean()),
        'test_success_rate': float(y_test.mean()),
        'cv_auc_mean': float(np.mean(cv_aucs)),
        'cv_auc_std': float(np.std(cv_aucs)),
        'cv_acc_mean': float(np.mean(cv_accs)),
        'cv_acc_std': float(np.std(cv_accs)),
        'test_accuracy': float(test_acc),
        'test_auc': float(test_auc),
        'test_correlation': float(test_corr),
        'test_p_value': float(test_p),
        'test_spearman': float(test_spearman),
        'feature_importance': {name: float(imp) for name, imp in zip(feature_names, importances)},
        'xgboost_params': xgb_params,
        'feature_names': feature_names
    }
    
    return final_model, metrics


def save_model(model: XGBClassifier, metrics: Dict, intent_config: IntentConfig, output_dir: Path):
    """Save model and metadata."""
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save model
    model_path = output_dir / f'{intent_config.name}_xgboost_model.joblib'
    joblib.dump(model, model_path)
    print(f"\n✅ Model saved: {model_path}")
    
    # Save model card
    model_card = {
        'intent': intent_config.name,
        'capability_proxy': intent_config.capability_field,
        'benchmark_name': intent_config.benchmark_name,
        'trained_at': datetime.now().isoformat(),
        **metrics
    }
    
    card_path = output_dir / f'{intent_config.name}_model_card.json'
    with open(card_path, 'w') as f:
        json.dump(model_card, f, indent=2)
    print(f"✅ Model card saved: {card_path}")
    
    return model_path, card_path


def train_intent(intent: str, models_cache: Dict, output_dir: Path) -> Optional[Dict]:
    """Train XGBoost model for a single intent."""
    print(f"\n{'#'*80}")
    print(f"# TRAINING: {intent.upper()}")
    print(f"{'#'*80}")
    
    intent_config = INTENT_CONFIGS.get(intent)
    if not intent_config:
        print(f"⚠️  Unknown intent: {intent}")
        return None
    
    print(f"Capability field: {intent_config.capability_field}")
    print(f"Benchmark: {intent_config.benchmark_name}")
    
    # Step 1: Download training data
    df = download_training_data(intent)
    
    if df is None or len(df) == 0:
        print(f"⚠️  No training data for {intent}")
        return None
    
    # Step 2: Compute NVIDIA features
    df = compute_nvidia_features(df)
    
    # Step 3: Add capability scores
    capability_map = get_capability_scores(models_cache, intent_config.capability_field)
    print(f"\nFound {len(capability_map)} models with {intent_config.capability_field} scores")
    
    df = add_capability_scores(df, capability_map, intent_config)
    
    if len(df) == 0:
        print(f"⚠️  No data after adding capability scores")
        return None
    
    # Step 4: Prepare features
    feature_cols = [
        'nvidia_creativity', 'nvidia_reasoning', 'nvidia_constraint',
        'nvidia_domain_knowledge', 'nvidia_contextual_knowledge',
        'nvidia_few_shots', 'model_capability'
    ]
    
    X = df[feature_cols].values
    y = df['success'].values
    groups = df['model'].values
    capabilities = df['model_capability'].values  # For stratified splitting
    
    # Step 5: Train with k-fold CV and STRATIFIED test split
    model, metrics = train_xgboost_with_cv(X, y, groups, capabilities, n_folds=5, test_size=0.15)
    
    # Step 6: Save model
    model_path, card_path = save_model(model, metrics, intent_config, output_dir)
    
    return {
        'intent': intent,
        'status': 'success',
        'model_path': str(model_path),
        'metadata_path': str(card_path),
        'n_train': metrics['n_train'],
        'n_test': metrics['n_test'],
        'cv_auc': metrics['cv_auc_mean'],
        'test_auc': metrics['test_auc'],
        'test_acc': metrics['test_accuracy']
    }


def main():
    """Main training pipeline."""
    print("="*80)
    print("RETRAINING XGBOOST MODELS WITH NEW CAPABILITY FIELDS")
    print("="*80)
    print(f"\nTimestamp: {datetime.now().isoformat()}")
    print("\nNew capability fields:")
    for intent, config in INTENT_CONFIGS.items():
        print(f"  {intent:15s} -> {config.capability_field:20s} ({config.benchmark_name})")
    
    # Load models cache
    print("\n" + "-"*60)
    models_cache = load_models_cache()
    print(f"Loaded {len(models_cache.get('models', []))} models from cache")
    
    # Output directory
    output_dir = Path(__file__).parent.parent / 'llm_jury' / 'models' / 'production'
    print(f"\nOutput directory: {output_dir}")
    
    # Train models for each intent
    results = []
    
    for intent in ['reasoning', 'coding', 'summarization', 'rag']:
        try:
            result = train_intent(intent, models_cache, output_dir)
            if result:
                results.append(result)
        except Exception as e:
            print(f"\n❌ Error training {intent}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'intent': intent,
                'status': 'failed',
                'error': str(e)
            })
    
    # Save training summary
    summary_path = output_dir / 'training_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    
    print(f"\n{'Intent':<15s} | {'Train':>8s} | {'Test':>7s} | {'CV AUC':>8s} | {'Test AUC':>9s} | {'Test Acc':>9s}")
    print("-" * 75)
    
    for result in results:
        if result['status'] == 'success':
            print(f"  ✅ {result['intent']:<12s} | {result['n_train']:>8,} | {result['n_test']:>7,} | "
                  f"{result['cv_auc']:>8.3f} | {result['test_auc']:>9.3f} | {result['test_acc']:>9.1%}")
        else:
            print(f"  ❌ {result['intent']:<12s} | ERROR: {result.get('error', 'Unknown')[:40]}")
    
    print(f"\n✅ Training summary saved: {summary_path}")
    print(f"\nModels saved to: {output_dir}")


if __name__ == '__main__':
    main()
