#!/usr/bin/env python3
"""
Calibrate Router for Your Domain (CLI Tool)

After finding your optimal gamma with find_gamma.py, use this script to
calibrate a router for production use.

Usage:
    python3 calibrate_router.py \
        --calibration-data my_data.jsonl \
        --gamma 0.01 \
        --output my_calibrated_router.joblib

Input format (calibration-data):
    {"prompt": "...", "rewards": {"model_a": 0.85, "model_b": 0.95}}

Output:
    Calibrated router ready for inference (joblib file)
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import argparse
import json
import joblib
import gzip
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from bandit_gpt.calibration import CalibratedRouter, apply_gamma_scaling, embed_prompt
from bandit_gpt.config_legacy import (
    DEFAULT_SENTENCE_TRANSFORMER,
    DEFAULT_WARMUP_PRIORS_PATH,
    DEFAULT_PCA_PATH,
    CANONICAL_DEV_DATA_PATH,
    CANONICAL_CALIBRATED_ROUTER_PATH
)


def main():
    parser = argparse.ArgumentParser(description="Calibrate router for your domain")
    parser.add_argument(
        "--calibration-data", type=str,
        default=str(CANONICAL_DEV_DATA_PATH),
        help="Path to calibration data (JSONL with 'prompt' and 'rewards' fields)"
    )
    parser.add_argument(
        "--warmup-priors", type=str,
        default=str(DEFAULT_WARMUP_PRIORS_PATH),
        help="Path to warmup priors"
    )
    parser.add_argument(
        "--pca", type=str,
        default=str(DEFAULT_PCA_PATH),
        help="Path to PCA model (default: 32-component PCA)"
    )
    parser.add_argument(
        "--gamma", type=float, default=0.010,
        help="Gamma calibration factor (default: 0.010, optimal from find_gamma.py)"
    )
    parser.add_argument(
        "--alpha", type=float, default=1.0,
        help="Exploration parameter (default: 1.0)"
    )
    parser.add_argument(
        "--lambda-cost", type=float, default=0.0,
        help="Cost penalty for strong model (default: 0.0 = quality-first)"
    )
    parser.add_argument(
        "--output", type=str,
        default=str(CANONICAL_CALIBRATED_ROUTER_PATH),
        help="Output file for calibrated router (.joblib, default: canonical path)"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("CALIBRATE ROUTER FOR YOUR DOMAIN")
    print("="*80)
    
    # Load resources
    print("\n📥 Loading resources...")
    warmup_priors_original = joblib.load(Path(args.warmup_priors))
    pca_model = joblib.load(Path(args.pca))
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
    print(f"   ✅ Warmup priors: {warmup_priors_original['n_prompts']:,} samples")
    print(f"   ✅ PCA: {pca_model.n_components} components")
    print(f"   ✅ Models: {', '.join(warmup_priors_original['models'])}")
    
    # Apply gamma scaling
    print(f"\n🔧 Applying gamma scaling:")
    print(f"   Gamma (γ): {args.gamma}")
    print(f"   Original effective N: {warmup_priors_original['n_prompts']:,}")
    print(f"   New effective N: {int(warmup_priors_original['n_prompts'] * args.gamma):,}")
    
    warmup_priors = apply_gamma_scaling(warmup_priors_original, args.gamma)
    
    # Load calibration data
    print(f"\n📊 Loading calibration data from: {args.calibration_data}")
    if args.calibration_data.endswith('.gz'):
        with gzip.open(args.calibration_data, 'rt') as f:
            raw_data = [json.loads(line) for line in f]
    else:
        with open(args.calibration_data) as f:
            raw_data = [json.loads(line) for line in f]
    
    # Transform data: group by prompt and create rewards dict
    from collections import defaultdict
    by_prompt = defaultdict(list)
    for item in raw_data:
        by_prompt[item['prompt']].append(item)
    
    calibration_data = []
    for prompt, items in by_prompt.items():
        rewards = {}
        for item in items:
            # Use raw_score as reward (0.0-1.0 scale)
            rewards[item['model_id']] = item['raw_score']
        calibration_data.append({
            'prompt': prompt,
            'rewards': rewards
        })
    
    print(f"   ✅ Loaded {len(calibration_data)} calibration samples ({len(raw_data)} model responses)")
    
    # Validate
    if not calibration_data:
        print("❌ No calibration data found!")
        return
    
    first_item = calibration_data[0]
    if 'prompt' not in first_item or 'rewards' not in first_item:
        print("❌ Invalid format! Expected: {'prompt': '...', 'rewards': {'model': 0.0}}")
        return
    
    # Initialize router
    print(f"\n🤖 Initializing router...")
    print(f"   Alpha (exploration): {args.alpha}")
    print(f"   Lambda (cost penalty): {args.lambda_cost}")
    
    router = CalibratedRouter(
        warmup_priors=warmup_priors,
        encoder=encoder,
        pca_model=pca_model,
        alpha=args.alpha,
        lambda_cost=args.lambda_cost
    )
    
    # Calibration loop
    print(f"\n🔄 Calibrating on {len(calibration_data)} samples...")
    
    model_selections = {m: 0 for m in router.models}
    total_reward = 0.0
    
    for item in tqdm(calibration_data, desc="Calibration"):
        # Embed and select
        context = embed_prompt(item['prompt'], encoder, pca_model)
        selected_model = router.select_model(item['prompt'])
        
        # Get observed reward
        reward = item['rewards'].get(selected_model, 0.0)
        
        # Update router
        context_col = context.reshape(-1, 1)
        router.A[selected_model] += context_col @ context_col.T
        router.b[selected_model] += (reward * context_col).flatten()
        
        # Track stats
        model_selections[selected_model] += 1
        total_reward += reward
    
    router.metadata['n_calibration_samples'] = len(calibration_data)
    
    # Report calibration results
    print(f"\n📊 Calibration Results:")
    print(f"   Total reward: {total_reward:.2f}")
    print(f"   Average reward: {total_reward/len(calibration_data):.4f}")
    print(f"\n   Model usage:")
    for model, count in model_selections.items():
        pct = (count / len(calibration_data)) * 100
        print(f"      {model}: {count} ({pct:.1f}%)")
    
    # Save calibrated router
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    router.save(output_file)
    
    print(f"\n✅ Calibrated router saved to: {output_file}")
    print(f"   Size: {output_file.stat().st_size / 1024:.1f} KB")
    
    # Usage instructions
    print("\n" + "="*80)
    print("📋 USING YOUR CALIBRATED ROUTER")
    print("="*80)
    print("\nPython example:")
    print(f"""
import joblib
from sentence_transformers import SentenceTransformer
from bandit_gpt.calibration import CalibratedRouter
from bandit_gpt.config_legacy import DEFAULT_SENTENCE_TRANSFORMER

# Load router
encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
pca_model = joblib.load("{args.pca}")
router = CalibratedRouter.load("{output_file}", encoder, pca_model)

# Route a query
user_prompt = "Explain quantum computing"
selected_model = router.select_model(user_prompt)
print(f"Route to: {{selected_model}}")

# Call LLM
response = call_llm(selected_model, user_prompt)
""")
    
    print("="*80)
    print("✅ CALIBRATION COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()

