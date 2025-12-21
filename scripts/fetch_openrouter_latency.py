#!/usr/bin/env python3
"""
Fetch TTFT (Time to First Token) and latency metrics from OpenRouter API.

This script:
1. Loads the models_cache.json
2. For each model, sends 100 test prompts to measure TTFT with confidence intervals
3. Updates the cache with mean latency + 95% CI
4. Saves the updated cache

Usage:
    export OPENROUTER_API_KEY=your_key_here
    
    # Measure only models missing latency data (recommended)
    python scripts/fetch_openrouter_latency.py --skip-existing
    
    # Test mode (only measure 5 models, 10 samples each)
    python scripts/fetch_openrouter_latency.py --test --n-samples 10
    
    # Measure specific models with 100 samples each
    python scripts/fetch_openrouter_latency.py --models "openai/gpt-4o" "anthropic/claude-3.5-sonnet"
    
    # Custom sample size
    python scripts/fetch_openrouter_latency.py --skip-existing --n-samples 50

Note: Gemini models automatically use max_tokens=4000 (others use 50).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from dotenv import load_dotenv

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from openai import OpenAI


# Test prompt (short to minimize cost)
TEST_PROMPT = "Count from 1 to 5."

# Model-specific max_tokens (some models need higher limits)
MODEL_MAX_TOKENS = {
    "google/gemini-3-pro-preview": 4000,
    "google/gemini-2.0-flash-exp": 4000,
    "google/gemini-2.0-flash-thinking-exp": 4000,
    "google/gemini-exp-1206": 4000,
    "google/gemini-flash-1.5": 4000,
    "google/gemini-pro-1.5": 4000,
}
DEFAULT_MAX_TOKENS = 50


def get_max_tokens_for_model(model_id: str) -> int:
    """Get appropriate max_tokens for a model."""
    return MODEL_MAX_TOKENS.get(model_id, DEFAULT_MAX_TOKENS)


def measure_latency_single(
    model_id: str,
    prompt: str = TEST_PROMPT,
    max_tokens: Optional[int] = None,
    timeout: float = 30.0,
) -> Optional[Dict[str, float]]:
    """
    Measure TTFT and generation speed for a single request.
    
    Returns:
        Dict with:
            - time_to_first_token_seconds: TTFT in seconds
            - output_tokens_per_second: Generation speed
            - total_latency_seconds: Total request time
            - tokens_generated: Number of output tokens
    """
    # Use model-specific max_tokens if not provided
    if max_tokens is None:
        max_tokens = get_max_tokens_for_model(model_id)
    
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
    )
    
    try:
        start_time = time.perf_counter()
        
        # Stream to measure TTFT
        stream = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            timeout=timeout,
            stream=True,
        )
        
        ttft = None
        tokens_received = 0
        first_token_time = None
        
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                tokens_received += 1
                if ttft is None:
                    first_token_time = time.perf_counter()
                    ttft = first_token_time - start_time
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        if ttft is None:
            # No tokens received (empty response)
            return None
        
        generation_time = total_time - ttft
        tokens_per_second = tokens_received / generation_time if generation_time > 0 else 0
        
        return {
            "time_to_first_token_seconds": ttft,
            "output_tokens_per_second": tokens_per_second,
            "total_latency_seconds": total_time,
            "tokens_generated": tokens_received,
        }
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def measure_latency_with_ci(
    model_id: str,
    n_samples: int = 100,
    prompt: str = TEST_PROMPT,
    max_tokens: Optional[int] = None,
    timeout: float = 30.0,
    delay_between_requests: float = 0.1,
) -> Optional[Dict[str, float]]:
    """
    Measure TTFT and generation speed with confidence intervals.
    
    Args:
        model_id: OpenRouter model ID
        n_samples: Number of measurements to take (default 100)
        prompt: Test prompt
        max_tokens: Max tokens to generate
        timeout: Request timeout
        delay_between_requests: Delay between measurements (seconds)
    
    Returns:
        Dict with mean, std, and 95% CI for each metric:
            - time_to_first_token_seconds: mean TTFT
            - time_to_first_token_ci_lower: 95% CI lower bound
            - time_to_first_token_ci_upper: 95% CI upper bound
            - output_tokens_per_second: mean generation speed
            - output_tokens_per_second_ci_lower: 95% CI lower bound
            - output_tokens_per_second_ci_upper: 95% CI upper bound
            - n_samples: number of successful measurements
            - n_failures: number of failed measurements
    """
    # Use model-specific max_tokens if not provided
    if max_tokens is None:
        max_tokens = get_max_tokens_for_model(model_id)
    
    ttft_samples = []
    otps_samples = []
    failures = 0
    
    print(f"  Measuring {n_samples} samples (max_tokens={max_tokens})...")
    
    for i in range(n_samples):
        metrics = measure_latency_single(model_id, prompt, max_tokens, timeout)
        
        if metrics:
            ttft_samples.append(metrics["time_to_first_token_seconds"])
            otps_samples.append(metrics["output_tokens_per_second"])
        else:
            failures += 1
        
        # Progress indicator every 10 samples
        if (i + 1) % 10 == 0:
            success_rate = len(ttft_samples) / (i + 1) * 100
            print(f"  Progress: {i+1}/{n_samples} ({success_rate:.0f}% success)")
        
        # Rate limiting
        if i < n_samples - 1:
            time.sleep(delay_between_requests)
    
    if not ttft_samples:
        return None
    
    # Calculate statistics
    ttft_arr = np.array(ttft_samples)
    otps_arr = np.array(otps_samples)
    
    # Mean
    ttft_mean = float(np.mean(ttft_arr))
    otps_mean = float(np.mean(otps_arr))
    
    # Standard deviation
    ttft_std = float(np.std(ttft_arr, ddof=1))
    otps_std = float(np.std(otps_arr, ddof=1))
    
    # 95% Confidence Interval (using t-distribution for small samples)
    from scipy import stats
    confidence = 0.95
    df = len(ttft_samples) - 1
    t_crit = stats.t.ppf((1 + confidence) / 2, df)
    
    ttft_margin = t_crit * ttft_std / np.sqrt(len(ttft_samples))
    otps_margin = t_crit * otps_std / np.sqrt(len(otps_samples))
    
    return {
        "time_to_first_token_seconds": ttft_mean,
        "time_to_first_token_std": ttft_std,
        "time_to_first_token_ci_lower": ttft_mean - ttft_margin,
        "time_to_first_token_ci_upper": ttft_mean + ttft_margin,
        "output_tokens_per_second": otps_mean,
        "output_tokens_per_second_std": otps_std,
        "output_tokens_per_second_ci_lower": otps_mean - otps_margin,
        "output_tokens_per_second_ci_upper": otps_mean + otps_margin,
        "n_samples": len(ttft_samples),
        "n_failures": failures,
        "measured_at": time.time(),
    }


def load_models_cache(cache_path: Path) -> Dict:
    """Load models_cache.json."""
    with open(cache_path) as f:
        return json.load(f)


def save_models_cache(cache_path: Path, data: Dict) -> None:
    """Save models_cache.json with backup."""
    # Create backup
    backup_path = cache_path.with_suffix('.json.bak')
    if cache_path.exists():
        import shutil
        shutil.copy(cache_path, backup_path)
        print(f"Backed up cache to {backup_path}")
    
    # Save updated cache
    with open(cache_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Updated cache: {cache_path}")


def main():
    parser = argparse.ArgumentParser(description="Fetch latency metrics from OpenRouter")
    parser.add_argument(
        "--cache",
        type=str,
        default=str(PROJECT_ROOT / "banditgpt" / "data" / "models_cache.json"),
        help="Path to models_cache.json",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: only measure 5 models",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Specific model IDs to measure (e.g., openai/gpt-4o)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip models that already have latency data",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Delay between models (seconds) to avoid rate limits",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Number of measurements per model (default: 100)",
    )
    parser.add_argument(
        "--sample-delay",
        type=float,
        default=0.1,
        help="Delay between samples for the same model (seconds)",
    )
    args = parser.parse_args()
    
    # Check API key
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY not set!")
        print("Set it in .env file or export OPENROUTER_API_KEY=your_key")
        return 1
    
    cache_path = Path(args.cache)
    if not cache_path.exists():
        print(f"ERROR: Cache file not found: {cache_path}")
        return 1
    
    print("=" * 80)
    print("Fetching Latency Metrics from OpenRouter")
    print("=" * 80)
    print()
    
    # Load cache
    data = load_models_cache(cache_path)
    models = data.get("models", [])
    
    print(f"Loaded {len(models)} models from cache")
    
    # Filter models to measure
    if args.models:
        # Specific models requested
        models_to_measure = [m for m in models if m.get("openrouter_id") in args.models]
        print(f"Measuring {len(models_to_measure)} specified models")
    elif args.test:
        # Test mode: first 5 models
        models_to_measure = models[:5]
        print(f"TEST MODE: Measuring first {len(models_to_measure)} models")
    else:
        # All models
        models_to_measure = models
        print(f"Measuring all {len(models_to_measure)} models")
    
    # Skip models with existing latency data if requested
    if args.skip_existing:
        models_to_measure = [
            m for m in models_to_measure
            if not (m.get("time_to_first_token_seconds") and m.get("output_tokens_per_second"))
        ]
        print(f"Skipping models with existing data, {len(models_to_measure)} remaining")
    
    if not models_to_measure:
        print("No models to measure!")
        return 0
    
    print()
    
    # Measure latency for each model
    updated_count = 0
    failed_count = 0
    
    for i, model in enumerate(models_to_measure):
        model_id = model.get("openrouter_id")
        display_name = model.get("display_name", model.get("name", model_id))
        
        print(f"[{i+1}/{len(models_to_measure)}] {display_name}")
        print(f"  Model ID: {model_id}")
        
        # Measure latency with confidence intervals
        metrics = measure_latency_with_ci(
            model_id,
            n_samples=args.n_samples,
            delay_between_requests=args.sample_delay,
        )
        
        if metrics:
            # Update model entry with mean values
            model["time_to_first_token_seconds"] = metrics["time_to_first_token_seconds"]
            model["time_to_first_token_std"] = metrics["time_to_first_token_std"]
            model["time_to_first_token_ci_lower"] = metrics["time_to_first_token_ci_lower"]
            model["time_to_first_token_ci_upper"] = metrics["time_to_first_token_ci_upper"]
            model["output_tokens_per_second"] = metrics["output_tokens_per_second"]
            model["output_tokens_per_second_std"] = metrics["output_tokens_per_second_std"]
            model["output_tokens_per_second_ci_lower"] = metrics["output_tokens_per_second_ci_lower"]
            model["output_tokens_per_second_ci_upper"] = metrics["output_tokens_per_second_ci_upper"]
            model["measured_at"] = metrics["measured_at"]
            model["latency_n_samples"] = metrics["n_samples"]
            model["latency_n_failures"] = metrics["n_failures"]
            
            ttft_mean = metrics["time_to_first_token_seconds"]
            ttft_ci = (metrics["time_to_first_token_ci_lower"], metrics["time_to_first_token_ci_upper"])
            otps_mean = metrics["output_tokens_per_second"]
            otps_ci = (metrics["output_tokens_per_second_ci_lower"], metrics["output_tokens_per_second_ci_upper"])
            
            print(f"  ✓ TTFT: {ttft_mean:.3f}s (95% CI: [{ttft_ci[0]:.3f}, {ttft_ci[1]:.3f}])")
            print(f"  ✓ Speed: {otps_mean:.1f} tok/s (95% CI: [{otps_ci[0]:.1f}, {otps_ci[1]:.1f}])")
            print(f"  ✓ Samples: {metrics['n_samples']}/{args.n_samples} successful")
            updated_count += 1
        else:
            print(f"  ✗ Failed to measure latency (all {args.n_samples} attempts failed)")
            failed_count += 1
        
        print()
        
        # Rate limiting
        if i < len(models_to_measure) - 1:
            time.sleep(args.delay)
    
    # Save updated cache
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total models measured: {len(models_to_measure)}")
    print(f"Successfully updated: {updated_count}")
    print(f"Failed: {failed_count}")
    print()
    
    if updated_count > 0:
        save_models_cache(cache_path, data)
        print()
        print("✓ Cache updated successfully!")
    else:
        print("No updates to save.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

