#!/usr/bin/env python3
"""
Generate warmup priors for RouteLLM comparison models using REAL reward data.

Models:
- mistralai/mixtral-8x7b-instruct (weak)
- openai/gpt-4-turbo (strong)

Strategy:
1. Load prompts with REAL rewards from RouteLLM battles dataset
2. Build warmup priors via LinUCB updates using ground truth rewards
3. Use Model Performance Gap (Reward_Strong - Reward_Weak) as difficulty measure
"""

import sys
import json
import numpy as np
import joblib
from pathlib import Path
from tqdm import tqdm
import argparse

# Add project root (script is in data/routellm/scripts/)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
ROUTELLM_DATA_DIR = PROJECT_ROOT / "data" / "routellm" / "data"
sys.path.insert(0, str(PROJECT_ROOT))

from sentence_transformers import SentenceTransformer
from bandit_gpt.config_legacy import (
    DEFAULT_SENTENCE_TRANSFORMER,
    DEFAULT_WARMUP_PRIORS_PATH,
    ROUTELLM_BATTLES_REWARDS_PATH
)

# Models for RouteLLM comparison
WEAK_MODEL = "mistralai/mixtral-8x7b-instruct"
STRONG_MODEL = "openai/gpt-4-turbo"


def load_rewards_from_file(rewards_file: Path, limit: int = 80000) -> list:
    """
    Load prompts with REAL rewards from RouteLLM battles dataset.
    
    Returns:
        list of dicts with keys: prompt, reward_weak, reward_strong
    """
    print(f"\n📥 Loading REAL rewards from: {rewards_file}")
    
    if not rewards_file.exists():
        print(f"   ❌ File not found: {rewards_file}")
        return []
    
    data = []
    with open(rewards_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            try:
                entry = json.loads(line)
                
                # Check if models match our target pair
                model_a = entry.get('model_a')
                model_b = entry.get('model_b')
                
                # Determine rewards based on model mapping
                reward_weak = None
                reward_strong = None
                
                if model_a == WEAK_MODEL and model_b == STRONG_MODEL:
                    reward_weak = entry['reward_a']
                    reward_strong = entry['reward_b']
                elif model_a == STRONG_MODEL and model_b == WEAK_MODEL:
                    reward_weak = entry['reward_b']
                    reward_strong = entry['reward_a']
                else:
                    # Skip if models don't match our target pair
                    continue

                # Parse prompt (handle stringified list)
                prompt_raw = entry['prompt']
                if isinstance(prompt_raw, str) and prompt_raw.startswith('['):
                    try:
                        prompt_list = json.loads(prompt_raw)
                        prompt = prompt_list[0] if prompt_list else ""
                    except:
                        prompt = prompt_raw
                else:
                    prompt = prompt_raw

                if prompt:
                    data.append({
                        'prompt': prompt,
                        'reward_weak': reward_weak,
                        'reward_strong': reward_strong
                    })
            except Exception as e:
                continue
    
    print(f"   ✅ Loaded {len(data):,} prompts with real rewards")
    return data


def estimate_prompt_difficulty(reward_weak: float, reward_strong: float) -> float:
    """
    Estimate prompt difficulty using the Model Performance Gap.
    
    Difficulty = Reward_Strong - Reward_Weak
    
    This uses GROUND TRUTH performance data instead of heuristics.
    - Easy (Gap ≈ 0): Weak model is sufficient.
    - Hard (Gap > 0.2): Strong model provides significantly better value.
    - Anomalous (Gap < 0): Weak model outperformed Strong model (treated as Easy).
    
    Returns:
        Difficulty score normalized to [0, 1]
    """
    # Ground truth difficulty: The Gap
    gap = reward_strong - reward_weak
    
    # Normalize for bandit logic:
    # - Negative gap (Weak > Strong) -> 0.0 (Easy)
    # - Zero gap (Weak == Strong) -> 0.0 (Easy)
    # - Positive gap (Strong > Weak) -> Scaled [0, 1]
    return max(0.0, min(1.0, gap))


def build_warmup_priors(
    reward_data: list,
    encoder: SentenceTransformer,
    pca,
    output_path: Path,
    plasticity: float = 0.1,
    seed: int = 42,
):
    """Build warmup priors using REAL rewards.
    
    Args:
        reward_data: List of dicts with 'prompt', 'reward_weak', 'reward_strong'
        encoder: SentenceTransformer encoder
        pca: PCA model for dimensionality reduction
        output_path: Where to save the priors
        plasticity: Plasticity factor to apply
        seed: Random seed
    """
    import warnings
    
    # Suppress runtime warnings about NaN values (we handle them explicitly)
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    np.random.seed(seed)
    
    print(f"\n🔧 Building warmup priors...")
    print(f"   Data points: {len(reward_data):,}")
    print(f"   Models: {WEAK_MODEL}, {STRONG_MODEL}")
    print(f"   Reward source: REAL (RouteLLM battles)")
    print(f"   Plasticity factor: {plasticity}")
    print(f"   PCA components: {pca.n_components_} (variance={np.sum(pca.explained_variance_ratio_):.1%})")
    
    # Initialize LinUCB matrices
    # Context dimension: PCA components + 1 bias term
    context_dim = pca.n_components_ + 1
    
    A = {
        WEAK_MODEL: np.eye(context_dim),
        STRONG_MODEL: np.eye(context_dim)
    }
    
    b = {
        WEAK_MODEL: np.zeros(context_dim),
        STRONG_MODEL: np.zeros(context_dim)
    }
    
    # Process data
    print(f"\n   Processing data...")
    
    skipped_count = 0
    processed_count = 0
    problematic_prompts = []  # Track problematic prompts
    
    for idx, item in enumerate(tqdm(reward_data, desc="   Warmup")):
        # Extract prompt and rewards
        prompt = item['prompt']
        reward_weak = item['reward_weak']
        reward_strong = item['reward_strong']
        
        try:
            # Encode prompt
            embedding = encoder.encode(prompt, convert_to_numpy=True, show_progress_bar=False)
            
            # Check for NaN/Inf in embedding (BEFORE PCA)
            if np.isnan(embedding).any() or np.isinf(embedding).any():
                skipped_count += 1
                problematic_prompts.append({
                    'index': idx,
                    'reason': 'nan_inf_in_embedding',
                    'prompt_length': len(prompt),
                    'prompt_preview': prompt[:200]
                })
                continue
            
            # Apply PCA
            embedding = pca.transform(embedding.reshape(1, -1)).flatten()
            
            # Check for NaN/Inf in PCA output (AFTER PCA)
            if np.isnan(embedding).any() or np.isinf(embedding).any():
                skipped_count += 1
                problematic_prompts.append({
                    'index': idx,
                    'reason': 'nan_inf_after_pca',
                    'prompt_length': len(prompt),
                    'prompt_preview': prompt[:200]
                })
                continue
            
            # Add bias term
            context = np.append(embedding, 1.0).reshape(-1, 1)  # Column vector with bias
            
            # LinUCB update for both models using REAL rewards
            A[WEAK_MODEL] += context @ context.T
            b[WEAK_MODEL] += (reward_weak * context).flatten()
            
            A[STRONG_MODEL] += context @ context.T
            b[STRONG_MODEL] += (reward_strong * context).flatten()
            
            processed_count += 1
            
        except Exception as e:
            # Skip problematic prompts
            skipped_count += 1
            problematic_prompts.append({
                'index': idx,
                'reason': f'exception: {str(e)}',
                'prompt_length': len(prompt),
                'prompt_preview': prompt[:200]
            })
            continue
    
    # Apply plasticity factor
    print(f"\n   Applying plasticity factor...")
    for model_id in [WEAK_MODEL, STRONG_MODEL]:
        A[model_id] *= plasticity
        b[model_id] *= plasticity
    
    # Report statistics
    print(f"\n   Processed: {processed_count:,}/{len(reward_data):,} prompts")
    if skipped_count > 0:
        print(f"   ⚠️  Skipped {skipped_count:,} prompts due to NaN/Inf values ({skipped_count/len(reward_data)*100:.2f}%)")
        
        # Categorize problems
        from collections import Counter
        reasons = Counter(p['reason'] for p in problematic_prompts)
        print(f"\n   Problem breakdown:")
        for reason, count in reasons.most_common():
            print(f"      {reason}: {count:,}")
        
        # Save problematic prompts to file
        problems_file = output_path.parent / "warmup_problematic_prompts.jsonl"
        with open(problems_file, 'w') as f:
            for problem in problematic_prompts:
                f.write(json.dumps(problem) + '\n')
        print(f"\n   📝 Saved problematic prompts to: {problems_file}")
        
        # Show a few examples
        print(f"\n   First 3 problematic prompts:")
        for i, p in enumerate(problematic_prompts[:3]):
            print(f"      [{i+1}] Index {p['index']}: {p['reason']}")
            print(f"          Length: {p['prompt_length']} chars")
            print(f"          Preview: {p['prompt_preview'][:100]}...")
    
    # Save
    state = {
        'A': A,
        'b': b,
        'models': [WEAK_MODEL, STRONG_MODEL],
        'n_prompts': processed_count,  # Use actual processed count
        'n_total': len(reward_data),
        'n_skipped': skipped_count,
        'plasticity': plasticity,
        'context_dim': context_dim,
        'seed': seed,
        'pca_applied': True,
        'pca_components': pca.n_components_,
        'reward_source': 'real_routellm_battles',
        'problematic_prompt_indices': [p['index'] for p in problematic_prompts]
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(state, output_path)
    
    print(f"\n   ✅ Saved warmup priors to: {output_path}")
    
    # Diagnostics
    print(f"\n📊 Warmup Statistics:")
    for model_id in [WEAK_MODEL, STRONG_MODEL]:
        A_norm = np.linalg.norm(A[model_id])
        b_norm = np.linalg.norm(b[model_id])
        print(f"   {model_id}:")
        print(f"      ||A|| = {A_norm:.2f}")
        print(f"      ||b|| = {b_norm:.2f}")


def check_data_leakage(train_prompts: set, eval_file: Path):
    """
    Check if any prompts in the training set appear in the evaluation set.
    
    Ensures warmup data is disjoint from test data.
    """
    print(f"\n🕵️ Checking for Data Leakage against {eval_file.name}...")
    
    if not eval_file.exists():
        print(f"   ⚠️  Eval file not found: {eval_file}")
        return

    leakage_count = 0
    total_eval = 0
    
    with open(eval_file, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line)
                prompt = entry.get('prompt', '')
                
                # Handle list-based prompts if necessary
                if isinstance(prompt, str) and prompt.startswith('['):
                    try:
                        p_list = json.loads(prompt)
                        prompt = p_list[0] if p_list else ""
                    except:
                        pass
                
                if prompt and prompt in train_prompts:
                    leakage_count += 1
                
                total_eval += 1
            except:
                continue
                
    print(f"   Eval set size: {total_eval:,}")
    if leakage_count > 0:
        print(f"   ❌ CRITICAL: Found {leakage_count:,} leaking prompts! ({leakage_count/total_eval:.1%})")
        print(f"      The bandit has seen these test cases during warmup.")
        raise ValueError(f"Data leakage detected: {leakage_count} prompts overlap with evaluation set.")
    else:
        print(f"   ✅ No leakage detected. Warmup data is disjoint from evaluation.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate warmup priors for RouteLLM comparison using REAL rewards",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--prompts", type=int, default=80000,
        help="Number of prompts to use for warmup (default: 80000)"
    )
    parser.add_argument(
        "--rewards-file", type=str,
        default=str(ROUTELLM_BATTLES_REWARDS_PATH),
        help="Path to file with real rewards (corrected RouteLLM battles dataset)"
    )
    parser.add_argument(
        "--output", type=str, 
        default=str(DEFAULT_WARMUP_PRIORS_PATH),
        help="Output path for warmup priors (relative to scripts/)"
    )
    parser.add_argument(
        "--plasticity", type=float, default=0.1,
        help="Plasticity factor (default: 0.1)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--pca", type=str, required=True,
        help="Path to PCA model (REQUIRED). Must match the PCA used in the live router."
    )
    parser.add_argument(
        "--check-leakage-dev", type=str, default=None,
        help="Path to dev evaluation file to check for data leakage"
    )
    parser.add_argument(
        "--check-leakage-holdout", type=str, default=None,
        help="Path to holdout evaluation file to check for data leakage"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("Generate Warmup Priors for RouteLLM Comparison")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Weak model:  {WEAK_MODEL}")
    print(f"  Strong model: {STRONG_MODEL}")
    print(f"  Target prompts: {args.prompts:,}")
    print(f"  Rewards file: {args.rewards_file}")
    print(f"  ✅ Using REAL rewards from RouteLLM battles")
    print(f"  Output: {args.output}")
    print(f"  Plasticity: {args.plasticity}")
    print(f"  Seed: {args.seed}")
    print(f"  PCA model: {args.pca}")
    
    # Load PCA model (REQUIRED)
    pca_path = Path(args.pca)
    if not pca_path.exists():
        print(f"\n❌ PCA model not found: {pca_path}")
        print("   PCA is REQUIRED to ensure embedding consistency with the live router.")
        return
    
    print(f"\n📐 Loading PCA model...")
    pca = joblib.load(pca_path)
    print(f"   ✅ Loaded PCA: {pca.n_components_} components")
    
    # Load encoder
    print(f"\n🔤 Loading sentence encoder...")
    # NOTE: Using default sentence transformer from config_legacy
    # This ensures consistency with the live router embeddings.
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
    print(f"   ✅ Loaded (embedding dim: 384)")
    
    # Load REAL rewards from file
    rewards_path = Path(args.rewards_file)
    if not rewards_path.is_absolute():
        rewards_path = Path(__file__).parent / args.rewards_file
    reward_data = load_rewards_from_file(rewards_path, limit=args.prompts)
    
    if not reward_data:
        print(f"\n❌ Failed to load real rewards from {rewards_path}")
        return
    
    # Analyze real reward distribution
    print(f"\n📊 Analyzing real reward distribution (Model Performance Gap)...")
    gaps = [d['reward_strong'] - d['reward_weak'] for d in reward_data]
    print(f"   Mean gap: {np.mean(gaps):.3f}")
    print(f"   Std gap: {np.std(gaps):.3f}")
    easy_count = sum(1 for g in gaps if g < 0.2)
    moderate_count = sum(1 for g in gaps if 0.2 <= g <= 0.6)
    hard_count = sum(1 for g in gaps if g > 0.6)
    print(f"   Easy (<0.2 gap): {easy_count/len(gaps)*100:.1f}%")
    print(f"   Moderate (0.2-0.6 gap): {moderate_count/len(gaps)*100:.1f}%")
    print(f"   Hard (>0.6 gap): {hard_count/len(gaps)*100:.1f}%")
    
    print(f"\n✅ Total data points for warmup: {len(reward_data):,}")
    
    # Check for data leakage if requested
    train_prompts = {item['prompt'] for item in reward_data}
    if args.check_leakage_dev:
        check_data_leakage(train_prompts, Path(args.check_leakage_dev))
    if args.check_leakage_holdout:
        check_data_leakage(train_prompts, Path(args.check_leakage_holdout))
    
    # Build priors
    output_path = Path(__file__).parent / args.output
    build_warmup_priors(
        reward_data=reward_data,
        encoder=encoder,
        pca=pca,
        output_path=output_path,
        plasticity=args.plasticity,
        seed=args.seed,
    )
    
    print(f"\n{'='*80}")
    print("✅ Warmup Priors Generated Successfully!")
    print(f"{'='*80}")
    
    print(f"\n🚀 Next Steps:")
    print(f"   1. Verify embedding consistency:")
    print(f"      - Encoder: {DEFAULT_SENTENCE_TRANSFORMER}")
    print(f"      - PCA: {args.pca}")
    print(f"      - These MUST match your live router configuration")
    print(f"")
    print(f"   2. Use priors in comparison:")
    print(f"      python run_comparison.py --warmup-priors {output_path}")


if __name__ == "__main__":
    main()
