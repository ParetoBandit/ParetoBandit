#!/usr/bin/env python3
"""
CRS × Reasoning Score Evaluation

This script:
1. Loads the 20 selected models and 149 downsampled prompts
2. Runs each model on each prompt to get actual accuracy
3. Fits a regression model: P(correct) = f(CRS, reasoning_score, CRS×reasoning)
4. Reports results by complexity bucket (Simple, Medium, Complex)

API Calls: 20 models × 149 prompts = 2,980 calls
"""

import os
import sys
import json
import time
import re
import random
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets import load_dataset
from dotenv import load_dotenv

# Load environment
for env_path in ['.env', '../.env', '../../.env', str(PROJECT_ROOT / '.env')]:
    if os.path.exists(env_path):
        load_dotenv(env_path)
        break

OUTPUT_DIR = Path(__file__).parent
RESULTS_FILE = OUTPUT_DIR / "evaluation_results.json"
CHECKPOINT_FILE = OUTPUT_DIR / "evaluation_checkpoint.json"


# =============================================================================
# DATA LOADING
# =============================================================================

def load_selected_models() -> pd.DataFrame:
    """Load the 20 selected models."""
    df = pd.read_csv(OUTPUT_DIR / "selected_models_20.csv")
    # Normalize CRS for regression
    df['crs_norm'] = (df['crs'] - df['crs'].min()) / (df['crs'].max() - df['crs'].min())
    print(f"✓ Loaded {len(df)} models")
    return df


def load_downsampled_prompts() -> pd.DataFrame:
    """Load the 149 downsampled prompts with metadata."""
    df = pd.read_csv(OUTPUT_DIR / "downsampled_prompts_150.csv")
    print(f"✓ Loaded {len(df)} prompts")
    return df


def load_prompts_with_text(prompts_df: pd.DataFrame, seed: int = 42) -> List[Dict]:
    """
    Reload prompts from original sources to get full text and ground truth.
    
    This matches each row in prompts_df with actual prompt text and correct answer.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    print("\n📚 Loading prompt text and ground truth from datasets...")
    
    # Group by source to batch load
    sources = prompts_df['source'].unique()
    
    # Load all prompts from each source
    source_prompts = {}
    
    for source in sources:
        n_needed = len(prompts_df[prompts_df['source'] == source])
        
        if source == 'ARC-Easy':
            source_prompts[source] = _load_arc_easy(n_needed * 2, seed)
        elif source == 'ARC-Challenge':
            source_prompts[source] = _load_arc_challenge(n_needed * 2, seed)
        elif source == 'GSM8K':
            source_prompts[source] = _load_gsm8k(n_needed * 2, seed)
        elif source == 'HellaSwag':
            source_prompts[source] = _load_hellaswag(n_needed * 2, seed)
        elif source == 'Winogrande':
            source_prompts[source] = _load_winogrande(n_needed * 2, seed)
        elif source.startswith('MMLU-'):
            subject = source.replace('MMLU-', '')
            source_prompts[source] = _load_mmlu_subject(subject, n_needed * 2, seed)
        elif source.startswith('LiveBench-'):
            category = source.replace('LiveBench-', '')
            source_prompts[source] = _load_livebench(category, n_needed * 2, seed)
        else:
            print(f"   ⚠️ Unknown source: {source}")
            source_prompts[source] = []
    
    # Match prompts_df rows with loaded prompts
    final_prompts = []
    source_indices = {s: 0 for s in sources}
    
    for _, row in prompts_df.iterrows():
        source = row['source']
        idx = source_indices[source]
        
        if idx < len(source_prompts.get(source, [])):
            prompt_data = source_prompts[source][idx].copy()
            prompt_data['reasoning_score'] = row['reasoning_score']
            prompt_data['complexity_level'] = row['complexity_level']
            final_prompts.append(prompt_data)
            source_indices[source] += 1
        else:
            print(f"   ⚠️ Ran out of prompts for {source}")
    
    print(f"   ✓ Matched {len(final_prompts)} prompts with text and ground truth")
    return final_prompts


def _load_arc_easy(n_samples: int, seed: int) -> List[Dict]:
    """Load ARC-Easy with ground truth."""
    dataset = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    samples = list(dataset)
    random.shuffle(samples)
    samples = samples[:n_samples]
    
    prompts = []
    for item in samples:
        prompt = f"{item['question']}\n\nOptions:\n"
        for label, text in zip(item['choices']['label'], item['choices']['text']):
            prompt += f"{label}. {text}\n"
        prompt += "\nAnswer with just the letter."
        
        prompts.append({
            'source': 'ARC-Easy',
            'prompt_text': prompt,
            'ground_truth': item['answerKey'],
            'eval_type': 'letter_match',
        })
    
    print(f"   ✓ ARC-Easy: {len(prompts)}")
    return prompts


def _load_arc_challenge(n_samples: int, seed: int) -> List[Dict]:
    """Load ARC-Challenge with ground truth."""
    dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    samples = list(dataset)
    random.shuffle(samples)
    samples = samples[:n_samples]
    
    prompts = []
    for item in samples:
        prompt = f"{item['question']}\n\nOptions:\n"
        for label, text in zip(item['choices']['label'], item['choices']['text']):
            prompt += f"{label}. {text}\n"
        prompt += "\nAnswer with just the letter."
        
        prompts.append({
            'source': 'ARC-Challenge',
            'prompt_text': prompt,
            'ground_truth': item['answerKey'],
            'eval_type': 'letter_match',
        })
    
    print(f"   ✓ ARC-Challenge: {len(prompts)}")
    return prompts


def _load_gsm8k(n_samples: int, seed: int) -> List[Dict]:
    """Load GSM8K with ground truth (numeric answer)."""
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    samples = list(dataset)
    random.shuffle(samples)
    samples = samples[:n_samples]
    
    prompts = []
    for item in samples:
        prompt = f"Solve this math problem. Show your work, then give the final answer as a number.\n\n{item['question']}"
        
        # Extract numeric answer from solution
        answer_text = item['answer']
        # GSM8K answers end with "#### <number>"
        match = re.search(r'####\s*(-?[\d,]+)', answer_text)
        if match:
            ground_truth = match.group(1).replace(',', '')
        else:
            ground_truth = answer_text.split()[-1]
        
        prompts.append({
            'source': 'GSM8K',
            'prompt_text': prompt,
            'ground_truth': ground_truth,
            'eval_type': 'numeric_match',
        })
    
    print(f"   ✓ GSM8K: {len(prompts)}")
    return prompts


def _load_hellaswag(n_samples: int, seed: int) -> List[Dict]:
    """Load HellaSwag with ground truth."""
    dataset = load_dataset("Rowan/hellaswag", split="validation")
    samples = list(dataset)
    random.shuffle(samples)
    samples = samples[:n_samples]
    
    prompts = []
    for item in samples:
        context = item['ctx']
        endings = item['endings']
        
        prompt = f"Complete the following:\n\n{context}\n\nOptions:\n"
        for i, ending in enumerate(endings):
            prompt += f"{chr(65+i)}. {ending}\n"
        prompt += "\nAnswer with just the letter (A, B, C, or D)."
        
        # Ground truth is index (0-3), convert to letter
        ground_truth = chr(65 + int(item['label']))
        
        prompts.append({
            'source': 'HellaSwag',
            'prompt_text': prompt,
            'ground_truth': ground_truth,
            'eval_type': 'letter_match',
        })
    
    print(f"   ✓ HellaSwag: {len(prompts)}")
    return prompts


def _load_winogrande(n_samples: int, seed: int) -> List[Dict]:
    """Load Winogrande with ground truth."""
    dataset = load_dataset("allenai/winogrande", "winogrande_xl", split="validation")
    samples = list(dataset)
    random.shuffle(samples)
    samples = samples[:n_samples]
    
    prompts = []
    for item in samples:
        sentence = item['sentence']
        option1 = item['option1']
        option2 = item['option2']
        
        prompt = f"Fill in the blank:\n\n{sentence}\n\nOptions:\nA. {option1}\nB. {option2}\n\nAnswer with just A or B."
        
        # Ground truth is "1" or "2", convert to letter
        ground_truth = 'A' if item['answer'] == '1' else 'B'
        
        prompts.append({
            'source': 'Winogrande',
            'prompt_text': prompt,
            'ground_truth': ground_truth,
            'eval_type': 'letter_match',
        })
    
    print(f"   ✓ Winogrande: {len(prompts)}")
    return prompts


def _load_mmlu_subject(subject: str, n_samples: int, seed: int) -> List[Dict]:
    """Load MMLU subject with ground truth."""
    try:
        dataset = load_dataset("cais/mmlu", subject, split="test")
        samples = list(dataset)
        random.shuffle(samples)
        samples = samples[:n_samples]
        
        prompts = []
        for item in samples:
            prompt = f"{item['question']}\n\nOptions:\n"
            for i, choice in enumerate(item['choices']):
                prompt += f"{chr(65+i)}. {choice}\n"
            prompt += "\nAnswer with just the letter."
            
            # Ground truth is index (0-3), convert to letter
            ground_truth = chr(65 + int(item['answer']))
            
            prompts.append({
                'source': f'MMLU-{subject}',
                'prompt_text': prompt,
                'ground_truth': ground_truth,
                'eval_type': 'letter_match',
            })
        
        print(f"   ✓ MMLU-{subject}: {len(prompts)}")
        return prompts
    except Exception as e:
        print(f"   ⚠️ MMLU-{subject} failed: {e}")
        return []


def _load_livebench(category: str, n_samples: int, seed: int) -> List[Dict]:
    """Load LiveBench category."""
    category_map = {
        'Math': 'livebench/math',
        'Reasoning': 'livebench/reasoning',
        'Coding': 'livebench/coding',
        'Language': 'livebench/language',
        'DataAnalysis': 'livebench/data_analysis',
        'InstructFollow': 'livebench/instruction_following',
    }
    
    hf_name = category_map.get(category)
    if not hf_name:
        print(f"   ⚠️ Unknown LiveBench category: {category}")
        return []
    
    try:
        dataset = load_dataset(hf_name, split="test")
        samples = list(dataset)
        random.shuffle(samples)
        samples = samples[:n_samples]
        
        prompts = []
        for item in samples:
            # Extract prompt from 'turns' field
            if 'turns' in item and item['turns']:
                prompt = item['turns'][0] if isinstance(item['turns'], list) else str(item['turns'])
            else:
                continue
            
            # Get ground truth
            ground_truth = item.get('ground_truth', '')
            if isinstance(ground_truth, list):
                ground_truth = str(ground_truth[0]) if ground_truth else ''
            
            # Truncate very long prompts
            if len(prompt) > 4000:
                prompt = prompt[:4000] + "\n\n[Truncated for length]"
            
            prompts.append({
                'source': f'LiveBench-{category}',
                'prompt_text': prompt,
                'ground_truth': str(ground_truth),
                'eval_type': 'livebench',  # Special handling needed
            })
        
        print(f"   ✓ LiveBench-{category}: {len(prompts)}")
        return prompts
    except Exception as e:
        print(f"   ⚠️ LiveBench-{category} failed: {e}")
        return []


# =============================================================================
# MODEL CALLING
# =============================================================================

def call_model(openrouter_id: str, prompt: str, max_retries: int = 3) -> Tuple[str, bool]:
    """
    Call OpenRouter API and return (response, success).
    """
    from openai import OpenAI
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set in environment")
    
    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )
    
    # Adjust tokens for reasoning/advanced models
    model_lower = openrouter_id.lower()
    is_advanced = any(x in model_lower for x in [
        'reasoning', 'thinking', 'r1', 'o1', 'o3', 'o4',
        'gemini-3', 'gemini-2.5', 'gpt-4', 'claude', 'preview'
    ])
    max_tokens = 16000 if is_advanced else 4000
    timeout_secs = 180 if is_advanced else 120
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=openrouter_id,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
                timeout=timeout_secs,
            )
            
            content = response.choices[0].message.content
            
            if isinstance(content, str) and content.strip():
                return content.strip(), True
            
            if isinstance(content, list):
                parts = [str(p.get("text", p.get("content", p))) if isinstance(p, dict) else str(p) for p in content]
                if parts:
                    return "\n".join(parts), True
            
            time.sleep(2 ** attempt)
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return f"ERROR: {str(e)[:100]}", False
    
    return "ERROR: Max retries exceeded", False


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_response(response: str, ground_truth: str, eval_type: str) -> bool:
    """
    Check if model response is correct.
    """
    response = response.strip().upper()
    ground_truth = str(ground_truth).strip().upper()
    
    if eval_type == 'letter_match':
        # Extract letter from response
        # Look for standalone letter or "The answer is X"
        patterns = [
            r'\b([A-D])\b',  # Standalone letter
            r'answer[:\s]+([A-D])',  # "Answer: X"
            r'\(([A-D])\)',  # (X)
        ]
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).upper() == ground_truth
        
        # Check if response starts with the letter
        if response and response[0] in 'ABCD':
            return response[0] == ground_truth
        
        return False
    
    elif eval_type == 'numeric_match':
        # Extract numbers from response
        # Look for final number or "the answer is X"
        response_nums = re.findall(r'-?[\d,]+\.?\d*', response.replace(',', ''))
        ground_nums = re.findall(r'-?[\d,]+\.?\d*', ground_truth.replace(',', ''))
        
        if not response_nums or not ground_nums:
            return False
        
        try:
            # Compare last number in response to ground truth
            resp_num = float(response_nums[-1])
            gt_num = float(ground_nums[-1])
            return abs(resp_num - gt_num) < 0.01
        except:
            return False
    
    elif eval_type == 'livebench':
        # For LiveBench, check if ground truth appears in response
        # This is a simplified check - real LiveBench uses more sophisticated evaluation
        if not ground_truth:
            return True  # No ground truth to check
        
        return ground_truth.lower() in response.lower()
    
    return False


# =============================================================================
# MAIN EVALUATION LOOP
# =============================================================================

def run_evaluation(models_df: pd.DataFrame, prompts: List[Dict], 
                   checkpoint_every: int = 50) -> pd.DataFrame:
    """
    Run all model-prompt evaluations.
    
    Returns DataFrame with columns:
    - model_name, openrouter_id, crs, crs_norm, crs_quartile
    - source, reasoning_score, complexity_level
    - response, is_correct
    """
    
    # Load checkpoint if exists
    completed = set()
    results = []
    
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            checkpoint = json.load(f)
            completed = set(checkpoint.get('completed', []))
            results = checkpoint.get('results', [])
            print(f"📂 Loaded checkpoint: {len(completed)} completed evaluations")
    
    total = len(models_df) * len(prompts)
    print(f"\n🚀 Running evaluation: {len(models_df)} models × {len(prompts)} prompts = {total} calls")
    print(f"   Already completed: {len(completed)}")
    print(f"   Remaining: {total - len(completed)}")
    
    start_time = time.time()
    call_count = 0
    
    for model_idx, model in models_df.iterrows():
        for prompt_idx, prompt in enumerate(prompts):
            key = f"{model['openrouter_id']}:{prompt_idx}"
            
            if key in completed:
                continue
            
            # Call model
            response, success = call_model(model['openrouter_id'], prompt['prompt_text'])
            
            # Evaluate
            is_correct = evaluate_response(response, prompt['ground_truth'], prompt['eval_type']) if success else False
            
            # Record result
            result = {
                'model_name': model['name'],
                'openrouter_id': model['openrouter_id'],
                'crs': model['crs'],
                'crs_norm': model['crs_norm'],
                'crs_quartile': model['crs_quartile'],
                'source': prompt['source'],
                'reasoning_score': prompt['reasoning_score'],
                'complexity_level': prompt['complexity_level'],
                'ground_truth': prompt['ground_truth'],
                'response': response[:500] if len(response) > 500 else response,  # Truncate for storage
                'is_correct': is_correct,
                'success': success,
            }
            results.append(result)
            completed.add(key)
            call_count += 1
            
            # Progress
            if call_count % 10 == 0:
                elapsed = time.time() - start_time
                rate = call_count / elapsed * 60  # calls per minute
                remaining = total - len(completed)
                eta = remaining / (rate / 60) if rate > 0 else 0
                print(f"   Progress: {len(completed)}/{total} ({len(completed)/total*100:.1f}%) | "
                      f"Rate: {rate:.1f}/min | ETA: {eta/60:.1f}min")
            
            # Checkpoint
            if call_count % checkpoint_every == 0:
                _save_checkpoint(completed, results)
            
            # Rate limiting
            time.sleep(0.5)
    
    # Final save
    _save_checkpoint(completed, results)
    
    print(f"\n✅ Evaluation complete: {len(results)} results")
    return pd.DataFrame(results)


def _save_checkpoint(completed: set, results: List[Dict]):
    """Save checkpoint to disk."""
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump({
            'completed': list(completed),
            'results': results,
            'timestamp': datetime.now().isoformat(),
        }, f)


# =============================================================================
# REGRESSION ANALYSIS
# =============================================================================

def fit_regression_model(results_df: pd.DataFrame) -> Dict:
    """
    Fit logistic regression: P(correct) = f(CRS, reasoning_score, CRS×reasoning)
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    print(f"\n{'='*70}")
    print("FITTING REGRESSION MODEL")
    print(f"{'='*70}")
    
    # Prepare features
    X = results_df[['crs_norm', 'reasoning_score']].copy()
    X['crs_x_reasoning'] = X['crs_norm'] * X['reasoning_score']
    y = results_df['is_correct'].astype(int).values
    
    print(f"\n📊 Dataset: {len(results_df)} observations")
    print(f"   Accuracy: {y.mean()*100:.1f}%")
    
    # VIF check
    print(f"\n📊 Variance Inflation Factor (VIF):")
    for i, name in enumerate(['crs_norm', 'reasoning_score', 'crs_x_reasoning']):
        vif = variance_inflation_factor(X.values, i)
        status = "✅" if vif < 5 else "⚠️" if vif < 10 else "❌"
        print(f"   {name:<20} VIF = {vif:>6.2f}  {status}")
    
    # Fit model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X, y)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
    
    coef = {
        'intercept': model.intercept_[0],
        'crs': model.coef_[0][0],
        'reasoning': model.coef_[0][1],
        'interaction': model.coef_[0][2],
    }
    
    print(f"\n📈 FITTED COEFFICIENTS:")
    print(f"   Intercept (β₀):        {coef['intercept']:>+8.3f}")
    print(f"   CRS (β₁):              {coef['crs']:>+8.3f}")
    print(f"   reasoning_score (β₂):  {coef['reasoning']:>+8.3f}")
    print(f"   CRS × reasoning (β₃):  {coef['interaction']:>+8.3f}")
    
    print(f"\n📊 Model Performance:")
    print(f"   ROC-AUC (5-fold CV): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    return {
        'model': model,
        'coef': coef,
        'cv_auc': cv_scores.mean(),
        'cv_std': cv_scores.std(),
    }


def compute_bucket_scores(models_df: pd.DataFrame, fitted: Dict) -> pd.DataFrame:
    """
    Compute regression scores for Simple, Medium, Complex buckets.
    """
    coef = fitted['coef']
    
    # Representative reasoning scores for each bucket
    buckets = {
        'Simple': 0.10,    # Low reasoning
        'Medium': 0.45,    # Mid reasoning
        'Complex': 0.75,   # High reasoning
    }
    
    results = []
    for _, model in models_df.iterrows():
        for bucket_name, reasoning_score in buckets.items():
            # Compute score
            score = (coef['intercept'] 
                    + coef['crs'] * model['crs_norm']
                    + coef['reasoning'] * reasoning_score
                    + coef['interaction'] * model['crs_norm'] * reasoning_score)
            
            # Convert to probability
            prob = 1 / (1 + np.exp(-score))
            
            results.append({
                'model_name': model['name'],
                'crs': model['crs'],
                'crs_quartile': model['crs_quartile'],
                'bucket': bucket_name,
                'reasoning_score': reasoning_score,
                'regression_score': score,
                'predicted_accuracy': prob * 100,
            })
    
    return pd.DataFrame(results)


def show_results(results_df: pd.DataFrame, bucket_scores: pd.DataFrame, fitted: Dict):
    """
    Display final results.
    """
    print(f"\n{'='*70}")
    print("RESULTS BY COMPLEXITY BUCKET")
    print(f"{'='*70}")
    
    # Actual accuracy by bucket
    print(f"\n📊 ACTUAL ACCURACY BY BUCKET:")
    for bucket in ['Simple', 'Medium', 'Complex']:
        subset = results_df[results_df['complexity_level'] == bucket]
        acc = subset['is_correct'].mean() * 100
        n = len(subset)
        print(f"   {bucket:<10} {acc:>6.1f}%  (n={n})")
    
    # Actual accuracy by CRS quartile
    print(f"\n📊 ACTUAL ACCURACY BY CRS QUARTILE:")
    for quartile in ['Q4 (High)', 'Q3 (Mid-High)', 'Q2 (Mid-Low)', 'Q1 (Low)']:
        subset = results_df[results_df['crs_quartile'] == quartile]
        acc = subset['is_correct'].mean() * 100
        n = len(subset)
        print(f"   {quartile:<15} {acc:>6.1f}%  (n={n})")
    
    # Regression scores by model and bucket
    print(f"\n{'='*70}")
    print("REGRESSION SCORES BY MODEL AND BUCKET")
    print(f"{'='*70}")
    
    pivot = bucket_scores.pivot(index='model_name', columns='bucket', values='regression_score')
    pivot = pivot[['Simple', 'Medium', 'Complex']]  # Reorder columns
    
    # Add CRS for sorting
    model_crs = bucket_scores.drop_duplicates('model_name')[['model_name', 'crs']].set_index('model_name')
    pivot = pivot.join(model_crs).sort_values('crs', ascending=False)
    
    print(f"\n{'Model':<35} {'CRS':>8} {'Simple':>10} {'Medium':>10} {'Complex':>10}")
    print(f"{'-'*35} {'-'*8} {'-'*10} {'-'*10} {'-'*10}")
    
    for model_name, row in pivot.iterrows():
        print(f"{model_name[:33]:<35} {row['crs']:>+7.2f} {row['Simple']:>+9.3f} {row['Medium']:>+9.3f} {row['Complex']:>+9.3f}")
    
    # Summary by CRS tier
    print(f"\n{'='*70}")
    print("SUMMARY: AVERAGE REGRESSION SCORES BY CRS QUARTILE")
    print(f"{'='*70}")
    
    tier_summary = bucket_scores.groupby(['crs_quartile', 'bucket'])['regression_score'].mean().unstack()
    tier_summary = tier_summary[['Simple', 'Medium', 'Complex']]
    
    print(f"\n{'CRS Quartile':<20} {'Simple':>10} {'Medium':>10} {'Complex':>10}")
    print(f"{'-'*20} {'-'*10} {'-'*10} {'-'*10}")
    
    for quartile in ['Q4 (High)', 'Q3 (Mid-High)', 'Q2 (Mid-Low)', 'Q1 (Low)']:
        if quartile in tier_summary.index:
            row = tier_summary.loc[quartile]
            print(f"{quartile:<20} {row['Simple']:>+9.3f} {row['Medium']:>+9.3f} {row['Complex']:>+9.3f}")
    
    # Final formula
    coef = fitted['coef']
    print(f"\n{'='*70}")
    print("FINAL REGRESSION FORMULA")
    print(f"{'='*70}")
    print(f"""
    Score = {coef['intercept']:+.3f} + {coef['crs']:+.3f}×CRS + {coef['reasoning']:+.3f}×reasoning + {coef['interaction']:+.3f}×CRS×reasoning
    
    Interpretation:
    • β_CRS = {coef['crs']:+.3f}: {"Higher CRS → better accuracy" if coef['crs'] > 0 else "CRS effect unclear"}
    • β_reasoning = {coef['reasoning']:+.3f}: {"Higher reasoning → harder prompts" if coef['reasoning'] < 0 else "Reasoning effect unclear"}
    • β_interaction = {coef['interaction']:+.3f}: {"High-CRS models handle complex prompts better" if coef['interaction'] > 0 else "No differential advantage"}
    """)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print("CRS × REASONING SCORE EVALUATION")
    print("="*70)
    
    # Load data
    models_df = load_selected_models()
    prompts_df = load_downsampled_prompts()
    
    # Load prompts with full text
    prompts = load_prompts_with_text(prompts_df)
    
    if len(prompts) == 0:
        print("❌ No prompts loaded!")
        return
    
    # Check for existing results
    if RESULTS_FILE.exists():
        print(f"\n📂 Found existing results: {RESULTS_FILE}")
        with open(RESULTS_FILE) as f:
            data = json.load(f)
        results_df = pd.DataFrame(data['results'])
        fitted = data['fitted']
        fitted['coef'] = data['fitted']['coef']
    else:
        # Run evaluation
        results_df = run_evaluation(models_df, prompts)
        
        # Fit regression
        fitted = fit_regression_model(results_df)
        
        # Save results
        with open(RESULTS_FILE, 'w') as f:
            json.dump({
                'results': results_df.to_dict(orient='records'),
                'fitted': {
                    'coef': fitted['coef'],
                    'cv_auc': fitted['cv_auc'],
                    'cv_std': fitted['cv_std'],
                },
                'timestamp': datetime.now().isoformat(),
            }, f, indent=2)
        print(f"\n💾 Saved results to: {RESULTS_FILE}")
    
    # Compute bucket scores
    bucket_scores = compute_bucket_scores(models_df, fitted)
    
    # Show results
    show_results(results_df, bucket_scores, fitted)
    
    # Save bucket scores
    bucket_scores.to_csv(OUTPUT_DIR / "bucket_regression_scores.csv", index=False)
    print(f"\n💾 Saved bucket scores to: bucket_regression_scores.csv")


if __name__ == "__main__":
    main()
