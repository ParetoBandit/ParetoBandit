"""
Model Manager: Add new models to the router and cache.

This module provides utilities for:
1. Fetching model info from OpenRouter API
2. Measuring TTFT (Time To First Token)
3. Updating models_cache.json
4. Calling models via OpenRouter

Usage:
    # Add a new model to the cache
    python -m banditgpt.core.model_manager add openai/gpt-5
    
    # List all cached models
    python -m banditgpt.core.model_manager list
    
    # Measure TTFT for a model
    python -m banditgpt.core.model_manager ttft openai/gpt-4o
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from banditgpt._resources import get_models_cache_path


def call_openrouter(
    model_id: str,
    prompt: str,
    *,
    max_tokens: int = 100,
    timeout_s: float = 60.0,
    stream: bool = False,
) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Call an OpenRouter model and return (response_text, metadata).
    
    Args:
        model_id: OpenRouter model ID (e.g., "openai/gpt-4o")
        prompt: User prompt text
        max_tokens: Maximum tokens to generate
        timeout_s: Request timeout in seconds
        stream: If True, stream the response (for TTFT measurement)
    
    Returns:
        (response_text, metadata) where metadata includes timing info
    
    Raises:
        ImportError: If openai package not installed
        RuntimeError: If OPENROUTER_API_KEY not set
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Install openai: pip install openai")
    
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")
    
    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        timeout=timeout_s,
    )
    
    meta: Dict[str, Any] = {
        "model_id": model_id,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    
    start_time = time.perf_counter()
    
    try:
        if stream:
            # Streaming for TTFT measurement
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                stream=True,
            )
            
            ttft = None
            chunks = []
            for chunk in response:
                if ttft is None:
                    ttft = time.perf_counter() - start_time
                if chunk.choices and chunk.choices[0].delta.content:
                    chunks.append(chunk.choices[0].delta.content)
            
            meta["ttft_seconds"] = ttft
            meta["total_time_seconds"] = time.perf_counter() - start_time
            return "".join(chunks), meta
        else:
            # Non-streaming
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
            )
            
            meta["total_time_seconds"] = time.perf_counter() - start_time
            content = response.choices[0].message.content or ""
            
            if response.usage:
                meta["prompt_tokens"] = response.usage.prompt_tokens
                meta["completion_tokens"] = response.usage.completion_tokens
            
            return content, meta
            
    except Exception as e:
        meta["error"] = f"{type(e).__name__}: {e}"
        meta["total_time_seconds"] = time.perf_counter() - start_time
        return None, meta


def measure_ttft(
    model_id: str,
    *,
    prompt: str = "Hi",  # Minimal prompt (1 token) for fastest, cheapest measurement
    samples: int = 100,  # 100 samples provides good precision with reasonable time
    max_tokens: int = 5,  # We only need first token; keep response tiny
    warmup: int = 3,
    sleep_between: float = 0.05,  # 50ms between calls (rate limit protection)
    progress_every: int = 25,
) -> Tuple[float, Dict[str, Any]]:
    """
    Measure Time To First Token for a model with statistical rigor.
    
    Uses minimal prompt ("Hi") and low max_tokens (5) to minimize cost.
    Cost per model: ~0.0001$ for 100 samples.
    
    Args:
        model_id: OpenRouter model ID
        prompt: Test prompt (default "Hi" = 1 token, minimal cost)
        samples: Number of samples (default 100 for good precision)
        max_tokens: Max tokens per sample (default 5, we only need first token)
        warmup: Number of warmup calls to discard (cold start effects)
        sleep_between: Seconds between calls (rate limit protection)
        progress_every: Print progress every N samples
    
    Returns:
        (mean_ttft_seconds, metadata with confidence intervals)
    """
    import numpy as np
    
    ttfts = []
    errors = []
    
    total_calls = warmup + samples
    
    for i in range(total_calls):
        _, meta = call_openrouter(
            model_id, prompt, max_tokens=max_tokens, stream=True
        )
        
        if "error" in meta:
            errors.append(meta["error"])
        elif "ttft_seconds" in meta:
            # Skip warmup samples
            if i >= warmup:
                ttfts.append(meta["ttft_seconds"])
        
        # Progress reporting
        if progress_every > 0 and (i + 1) % progress_every == 0:
            print(f"    [{i + 1}/{total_calls}] collected {len(ttfts)} samples...")
        
        if i < total_calls - 1 and sleep_between > 0:
            time.sleep(sleep_between)
    
    if not ttfts:
        return 0.0, {"errors": errors, "samples": 0}
    
    # Statistical analysis
    arr = np.array(ttfts)
    n = len(arr)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1))  # Sample std
    
    # 95% confidence interval (t-distribution for small samples, z for large)
    if n >= 30:
        # Large sample: use z = 1.96
        ci_half = 1.96 * std / np.sqrt(n)
    else:
        # Small sample: use t-distribution
        from scipy import stats
        t_val = stats.t.ppf(0.975, df=n-1)
        ci_half = t_val * std / np.sqrt(n)
    
    ci_lower = mean - ci_half
    ci_upper = mean + ci_half
    
    # Percentiles
    p50 = float(np.percentile(arr, 50))
    p95 = float(np.percentile(arr, 95))
    p99 = float(np.percentile(arr, 99))
    
    return mean, {
        "samples": n,
        "mean": mean,
        "std": std,
        "ci_95_lower": ci_lower,
        "ci_95_upper": ci_upper,
        "ci_95_half_width": float(ci_half),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p50": p50,
        "p95": p95,
        "p99": p99,
        "errors": errors[:10] if errors else [],  # Limit error list
        "error_count": len(errors),
        "warmup_discarded": warmup,
    }


def fetch_model_info(model_id: str) -> Dict[str, Any]:
    """
    Fetch model info from OpenRouter API.
    
    Args:
        model_id: OpenRouter model ID (e.g., "openai/gpt-4o")
    
    Returns:
        Dict with model info (pricing, context length, etc.)
    """
    try:
        import requests
    except ImportError:
        # Fall back to urllib
        import urllib.request
        import urllib.error
        
        url = f"https://openrouter.ai/api/v1/models/{model_id}"
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                data = json.loads(resp.read().decode())
                return data.get("data", data)
        except urllib.error.HTTPError as e:
            return {"error": f"HTTP {e.code}: {e.reason}"}
        except Exception as e:
            return {"error": str(e)}
    
    # Use requests if available
    try:
        resp = requests.get(
            f"https://openrouter.ai/api/v1/models/{model_id}",
            timeout=10,
        )
        if resp.ok:
            data = resp.json()
            return data.get("data", data)
        else:
            return {"error": f"HTTP {resp.status_code}"}
    except Exception as e:
        return {"error": str(e)}


def get_model_pricing(model_id: str) -> Tuple[float, float]:
    """
    Get input/output pricing for a model from OpenRouter.
    
    Returns:
        (input_cost_per_m, output_cost_per_m) in USD per million tokens
    """
    info = fetch_model_info(model_id)
    
    if "error" in info:
        print(f"Warning: Could not fetch pricing for {model_id}: {info['error']}")
        return 0.0, 0.0
    
    # OpenRouter pricing is in $ per token, we want per million
    pricing = info.get("pricing", {})
    input_cost = float(pricing.get("prompt", 0)) * 1_000_000
    output_cost = float(pricing.get("completion", 0)) * 1_000_000
    
    return input_cost, output_cost


def load_models_cache(cache_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load models_cache.json."""
    path = cache_path or get_models_cache_path()
    if not path.exists():
        return {"models": []}
    return json.loads(path.read_text())


def save_models_cache(data: Dict[str, Any], cache_path: Optional[Path] = None) -> None:
    """Save models_cache.json."""
    path = cache_path or get_models_cache_path()
    path.write_text(json.dumps(data, indent=2))


def add_model_to_cache(
    model_id: str,
    *,
    cache_path: Optional[Path] = None,
    measure_ttft_samples: int = 100,
    fetch_pricing: bool = True,
    ttft_warmup: int = 3,
) -> Dict[str, Any]:
    """
    Add a new model to models_cache.json.
    
    Args:
        model_id: OpenRouter model ID (e.g., "openai/gpt-5")
        cache_path: Optional path to models_cache.json
        measure_ttft_samples: Number of TTFT samples (default 1000, 0 to skip)
        fetch_pricing: Whether to fetch pricing from OpenRouter
        ttft_warmup: Number of warmup calls to discard
    
    Returns:
        The new model entry
    """
    print(f"Adding model: {model_id}")
    
    # Load existing cache
    cache = load_models_cache(cache_path)
    models = cache.get("models", [])
    
    # Check if model already exists
    for m in models:
        if m.get("openrouter_id") == model_id:
            print(f"Model {model_id} already exists in cache")
            return m
    
    # Build new entry
    entry: Dict[str, Any] = {
        "openrouter_id": model_id,
        "display_name": model_id.split("/")[-1].replace("-", " ").title(),
        "name": model_id.split("/")[-1],
    }
    
    # Fetch pricing
    if fetch_pricing:
        print(f"  Fetching pricing from OpenRouter...")
        input_cost, output_cost = get_model_pricing(model_id)
        entry["input_cost_per_m"] = input_cost
        entry["output_cost_per_m"] = output_cost
        entry["price_1m_input"] = input_cost
        entry["price_1m_output"] = output_cost
        entry["price_1m_blended"] = input_cost * 0.75 + output_cost * 0.25
        print(f"  Pricing: ${input_cost:.4f} input, ${output_cost:.4f} output (per 1M tokens)")
    
    # Measure TTFT with statistical rigor
    if measure_ttft_samples > 0:
        print(f"  Measuring TTFT ({measure_ttft_samples} samples, {ttft_warmup} warmup)...")
        try:
            ttft_mean, meta = measure_ttft(
                model_id, 
                samples=measure_ttft_samples,
                warmup=ttft_warmup,
            )
            
            # Store mean and confidence interval
            entry["time_to_first_token_seconds"] = ttft_mean
            entry["measured_ttft_seconds"] = ttft_mean
            entry["ttft_mean"] = meta.get("mean", ttft_mean)
            entry["ttft_std"] = meta.get("std", 0)
            entry["ttft_ci_95_lower"] = meta.get("ci_95_lower", ttft_mean)
            entry["ttft_ci_95_upper"] = meta.get("ci_95_upper", ttft_mean)
            entry["ttft_p50"] = meta.get("p50", ttft_mean)
            entry["ttft_p95"] = meta.get("p95", ttft_mean)
            entry["ttft_p99"] = meta.get("p99", ttft_mean)
            entry["ttft_samples"] = meta.get("samples", 0)
            
            print(f"  TTFT Results:")
            print(f"    Mean: {ttft_mean:.3f}s")
            print(f"    95% CI: [{meta.get('ci_95_lower', 0):.3f}, {meta.get('ci_95_upper', 0):.3f}]")
            print(f"    P50/P95/P99: {meta.get('p50', 0):.3f} / {meta.get('p95', 0):.3f} / {meta.get('p99', 0):.3f}")
            print(f"    Min/Max: {meta.get('min', 0):.3f} / {meta.get('max', 0):.3f}")
            print(f"    Samples: {meta.get('samples', 0)}")
            
            if meta.get("error_count", 0) > 0:
                print(f"    Errors: {meta['error_count']}")
        except Exception as e:
            print(f"  TTFT measurement failed: {e}")
    
    # Add throughput estimate (default)
    if "output_tokens_per_second" not in entry:
        entry["output_tokens_per_second"] = 0  # Unknown
    
    # Add to cache
    models.append(entry)
    cache["models"] = models
    save_models_cache(cache, cache_path)
    
    print(f"  Added to cache: {get_models_cache_path()}")
    return entry


def list_models(cache_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """List all models in the cache."""
    cache = load_models_cache(cache_path)
    return cache.get("models", [])


def remove_model_from_cache(
    model_id: str,
    cache_path: Optional[Path] = None,
) -> bool:
    """
    Remove a model from the cache.
    
    Returns:
        True if model was removed, False if not found
    """
    cache = load_models_cache(cache_path)
    models = cache.get("models", [])
    
    new_models = [m for m in models if m.get("openrouter_id") != model_id]
    
    if len(new_models) == len(models):
        return False
    
    cache["models"] = new_models
    save_models_cache(cache, cache_path)
    return True


def update_model_ttft(
    model_id: str,
    *,
    cache_path: Optional[Path] = None,
    samples: int = 100,
    warmup: int = 3,
) -> Optional[Dict[str, Any]]:
    """
    Update TTFT measurements for an existing model in the cache.
    
    Args:
        model_id: OpenRouter model ID
        cache_path: Optional path to models_cache.json
        samples: Number of TTFT samples
        warmup: Number of warmup calls to discard
    
    Returns:
        Updated model entry, or None if not found
    """
    cache = load_models_cache(cache_path)
    models = cache.get("models", [])
    
    # Find the model
    model_idx = None
    for i, m in enumerate(models):
        if m.get("openrouter_id") == model_id:
            model_idx = i
            break
    
    if model_idx is None:
        return None
    
    print(f"Updating TTFT for {model_id} ({samples} samples)...")
    
    ttft_mean, meta = measure_ttft(
        model_id,
        samples=samples,
        warmup=warmup,
    )
    
    # Update the entry
    entry = models[model_idx]
    entry["time_to_first_token_seconds"] = ttft_mean
    entry["measured_ttft_seconds"] = ttft_mean
    entry["ttft_mean"] = meta.get("mean", ttft_mean)
    entry["ttft_std"] = meta.get("std", 0)
    entry["ttft_ci_95_lower"] = meta.get("ci_95_lower", ttft_mean)
    entry["ttft_ci_95_upper"] = meta.get("ci_95_upper", ttft_mean)
    entry["ttft_p50"] = meta.get("p50", ttft_mean)
    entry["ttft_p95"] = meta.get("p95", ttft_mean)
    entry["ttft_p99"] = meta.get("p99", ttft_mean)
    entry["ttft_samples"] = meta.get("samples", 0)
    
    # Save
    cache["models"] = models
    save_models_cache(cache, cache_path)
    
    print(f"  Mean: {ttft_mean:.3f}s, 95% CI: [{meta.get('ci_95_lower', 0):.3f}, {meta.get('ci_95_upper', 0):.3f}]")
    return entry


def batch_update_ttft(
    *,
    cache_path: Optional[Path] = None,
    samples: int = 100,
    warmup: int = 3,
    skip_existing: bool = False,
    model_filter: Optional[List[str]] = None,
    workers: int = 10,
) -> Dict[str, Any]:
    """
    Batch update TTFT measurements for all models in the cache (parallel).
    
    With 10 workers and 100 samples per model, 81 models takes ~15-20 minutes.
    
    Args:
        cache_path: Optional path to models_cache.json
        samples: Number of TTFT samples per model (default 100)
        warmup: Number of warmup calls to discard
        skip_existing: If True, skip models that already have ttft_samples field
        model_filter: Optional list of model IDs to update (None = all)
        workers: Number of parallel workers (default 10)
    
    Returns:
        Summary dict with success/failure counts
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading
    
    cache = load_models_cache(cache_path)
    models = cache.get("models", [])
    
    results = {
        "total": 0,
        "updated": 0,
        "skipped": 0,
        "failed": 0,
        "errors": [],
    }
    
    # Lock for thread-safe cache updates
    cache_lock = threading.Lock()
    
    to_update = []
    for i, m in enumerate(models):
        model_id = m.get("openrouter_id")
        if not model_id:
            continue
        
        # Filter by model list
        if model_filter and model_id not in model_filter:
            continue
        
        # Skip if already has precise measurements
        if skip_existing and m.get("ttft_samples", 0) >= samples:
            results["skipped"] += 1
            continue
        
        to_update.append((i, model_id))
    
    results["total"] = len(to_update)
    est_time = len(to_update) * (samples + warmup) * 0.15 / 60 / workers
    print(f"Batch TTFT update: {len(to_update)} models, {samples} samples each")
    print(f"Workers: {workers}")
    print(f"Estimated time: {est_time:.1f} minutes")
    print("=" * 60)
    
    def measure_one(model_idx: int, model_id: str) -> Tuple[int, str, Optional[Dict], Optional[str]]:
        """Measure TTFT for one model. Returns (idx, model_id, result_meta, error)."""
        try:
            ttft_mean, meta = measure_ttft(
                model_id,
                samples=samples,
                warmup=warmup,
                progress_every=0,  # Suppress per-sample progress in parallel mode
            )
            return (model_idx, model_id, {"mean": ttft_mean, **meta}, None)
        except Exception as e:
            return (model_idx, model_id, None, str(e))
    
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(measure_one, idx, mid): (idx, mid) 
            for idx, mid in to_update
        }
        
        # Process as they complete
        for future in as_completed(futures):
            model_idx, model_id, meta, error = future.result()
            completed += 1
            
            if error:
                print(f"[{completed}/{len(to_update)}] {model_id}: ERROR - {error}")
                results["failed"] += 1
                results["errors"].append({"model": model_id, "error": error})
            else:
                ttft_mean = meta["mean"]
                print(f"[{completed}/{len(to_update)}] {model_id}: {ttft_mean:.3f}s "
                      f"(95% CI: [{meta.get('ci_95_lower', 0):.3f}, {meta.get('ci_95_upper', 0):.3f}])")
                
                # Thread-safe update
                with cache_lock:
                    entry = models[model_idx]
                    entry["time_to_first_token_seconds"] = ttft_mean
                    entry["measured_ttft_seconds"] = ttft_mean
                    entry["ttft_mean"] = meta.get("mean", ttft_mean)
                    entry["ttft_std"] = meta.get("std", 0)
                    entry["ttft_ci_95_lower"] = meta.get("ci_95_lower", ttft_mean)
                    entry["ttft_ci_95_upper"] = meta.get("ci_95_upper", ttft_mean)
                    entry["ttft_p50"] = meta.get("p50", ttft_mean)
                    entry["ttft_p95"] = meta.get("p95", ttft_mean)
                    entry["ttft_p99"] = meta.get("p99", ttft_mean)
                    entry["ttft_samples"] = meta.get("samples", 0)
                    
                    results["updated"] += 1
                    
                    # Save periodically (every 5 models)
                    if results["updated"] % 5 == 0:
                        cache["models"] = models
                        save_models_cache(cache, cache_path)
    
    # Final save
    with cache_lock:
        cache["models"] = models
        save_models_cache(cache, cache_path)
    
    print("\n" + "=" * 60)
    print(f"Batch update complete:")
    print(f"  Updated: {results['updated']}")
    print(f"  Skipped: {results['skipped']}")
    print(f"  Failed: {results['failed']}")
    
    return results


def initialize_ttft_estimates(cache_path: Optional[Path] = None) -> int:
    """
    Initialize statistical TTFT fields with estimates based on existing point measurements.
    
    This allows the router to work with the new fields before running full measurements.
    Uses conservative estimates: std = 20% of mean, CI = mean ± 10%.
    
    Returns:
        Number of models updated
    """
    cache = load_models_cache(cache_path)
    models = cache.get("models", [])
    
    updated = 0
    for m in models:
        ttft = m.get("time_to_first_token_seconds", 0) or m.get("measured_ttft_seconds", 0)
        
        if ttft <= 0:
            continue
        
        # Skip if already has precise measurements
        if m.get("ttft_samples", 0) >= 30:
            continue
        
        # Conservative estimates based on typical TTFT variability
        std_estimate = ttft * 0.20  # 20% of mean
        ci_half = ttft * 0.10  # 10% of mean (conservative)
        
        m["ttft_mean"] = ttft
        m["ttft_std"] = std_estimate
        m["ttft_ci_95_lower"] = max(0, ttft - ci_half)
        m["ttft_ci_95_upper"] = ttft + ci_half
        m["ttft_p50"] = ttft
        m["ttft_p95"] = ttft * 1.3  # Estimate: P95 ≈ 1.3× mean
        m["ttft_p99"] = ttft * 1.5  # Estimate: P99 ≈ 1.5× mean
        m["ttft_samples"] = 0  # Mark as estimated, not measured
        
        updated += 1
    
    cache["models"] = models
    save_models_cache(cache, cache_path)
    
    return updated


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Manage models in the BanditGPT cache"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Add model
    add_parser = subparsers.add_parser("add", help="Add a new model to the cache")
    add_parser.add_argument("model_id", type=str, help="OpenRouter model ID (e.g., openai/gpt-5)")
    add_parser.add_argument("--no-ttft", action="store_true", help="Skip TTFT measurement")
    add_parser.add_argument("--no-pricing", action="store_true", help="Skip pricing fetch")
    add_parser.add_argument("--ttft-samples", type=int, default=1000, help="Number of TTFT samples (default 1000)")
    add_parser.add_argument("--ttft-warmup", type=int, default=5, help="TTFT warmup calls to discard")
    
    # Update TTFT for existing model
    update_parser = subparsers.add_parser("update-ttft", help="Update TTFT for an existing model")
    update_parser.add_argument("model_id", type=str, help="OpenRouter model ID")
    update_parser.add_argument("--samples", type=int, default=100, help="Number of TTFT samples")
    update_parser.add_argument("--warmup", type=int, default=3, help="Warmup calls to discard")
    
    # Batch update all models
    batch_parser = subparsers.add_parser("batch-update", help="Update TTFT for all models (parallel)")
    batch_parser.add_argument("--samples", type=int, default=100, help="Samples per model")
    batch_parser.add_argument("--warmup", type=int, default=3, help="Warmup calls to discard")
    batch_parser.add_argument("--workers", type=int, default=10, help="Parallel workers")
    batch_parser.add_argument("--skip-existing", action="store_true", help="Skip models with existing measurements")
    
    # Initialize estimates (no API calls)
    init_parser = subparsers.add_parser("init-estimates", help="Initialize TTFT estimates from existing data (no API calls)")
    
    # List models
    list_parser = subparsers.add_parser("list", help="List all cached models")
    list_parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    # Remove model
    rm_parser = subparsers.add_parser("remove", help="Remove a model from cache")
    rm_parser.add_argument("model_id", type=str, help="OpenRouter model ID")
    
    # Measure TTFT (without saving)
    ttft_parser = subparsers.add_parser("ttft", help="Measure TTFT for a model (without saving)")
    ttft_parser.add_argument("model_id", type=str, help="OpenRouter model ID")
    ttft_parser.add_argument("--samples", type=int, default=100, help="Number of samples")
    ttft_parser.add_argument("--warmup", type=int, default=3, help="Warmup calls to discard")
    
    # Call model (for testing)
    call_parser = subparsers.add_parser("call", help="Call a model (for testing)")
    call_parser.add_argument("model_id", type=str, help="OpenRouter model ID")
    call_parser.add_argument("--prompt", type=str, default="Say hello.", help="Prompt text")
    call_parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens")
    
    args = parser.parse_args()
    
    # Load environment for API key
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    if args.command == "add":
        add_model_to_cache(
            args.model_id,
            measure_ttft_samples=0 if args.no_ttft else args.ttft_samples,
            fetch_pricing=not args.no_pricing,
            ttft_warmup=args.ttft_warmup,
        )
        return 0
    
    elif args.command == "update-ttft":
        result = update_model_ttft(
            args.model_id,
            samples=args.samples,
            warmup=args.warmup,
        )
        if result is None:
            print(f"Model not found: {args.model_id}")
            return 1
        return 0
    
    elif args.command == "batch-update":
        results = batch_update_ttft(
            samples=args.samples,
            warmup=args.warmup,
            workers=args.workers,
            skip_existing=args.skip_existing,
        )
        return 0 if results["failed"] == 0 else 1
    
    elif args.command == "init-estimates":
        count = initialize_ttft_estimates()
        print(f"Initialized TTFT estimates for {count} models")
        return 0
    
    elif args.command == "list":
        models = list_models()
        if args.json:
            print(json.dumps(models, indent=2))
        else:
            print(f"Models in cache: {len(models)}\n")
            print(f"{'Model ID':<45} {'Cost/1M':>10} {'TTFT Mean':>10} {'95% CI':>20}")
            print("-" * 90)
            for m in models:
                oid = m.get("openrouter_id", "?")
                cost = m.get("price_1m_blended", 0)
                ttft = m.get("ttft_mean", m.get("time_to_first_token_seconds", 0))
                ci_lo = m.get("ttft_ci_95_lower", ttft)
                ci_hi = m.get("ttft_ci_95_upper", ttft)
                ci_str = f"[{ci_lo:.2f}, {ci_hi:.2f}]" if ci_lo != ci_hi else "N/A"
                print(f"{oid:<45} ${cost:>8.4f} {ttft:>9.3f}s {ci_str:>20}")
        return 0
    
    elif args.command == "remove":
        if remove_model_from_cache(args.model_id):
            print(f"Removed: {args.model_id}")
            return 0
        else:
            print(f"Model not found: {args.model_id}")
            return 1
    
    elif args.command == "ttft":
        print(f"Measuring TTFT for {args.model_id}")
        print(f"  Samples: {args.samples}, Warmup: {args.warmup}")
        print()
        
        ttft, meta = measure_ttft(
            args.model_id,
            samples=args.samples,
            warmup=args.warmup,
        )
        
        print(f"\n{'='*50}")
        print(f"TTFT Results for {args.model_id}")
        print(f"{'='*50}")
        print(f"  Mean:     {meta.get('mean', 0):.4f}s")
        print(f"  Std Dev:  {meta.get('std', 0):.4f}s")
        print(f"  95% CI:   [{meta.get('ci_95_lower', 0):.4f}, {meta.get('ci_95_upper', 0):.4f}]")
        print(f"  P50:      {meta.get('p50', 0):.4f}s")
        print(f"  P95:      {meta.get('p95', 0):.4f}s")
        print(f"  P99:      {meta.get('p99', 0):.4f}s")
        print(f"  Min:      {meta.get('min', 0):.4f}s")
        print(f"  Max:      {meta.get('max', 0):.4f}s")
        print(f"  Samples:  {meta.get('samples', 0)}")
        if meta.get("error_count", 0) > 0:
            print(f"  Errors:   {meta['error_count']}")
        return 0
    
    elif args.command == "call":
        print(f"Calling {args.model_id}...")
        response, meta = call_openrouter(
            args.model_id,
            args.prompt,
            max_tokens=args.max_tokens,
        )
        
        if response:
            print(f"\nResponse:\n{response}")
        else:
            print(f"\nError: {meta.get('error', 'Unknown')}")
        
        print(f"\nTime: {meta.get('total_time_seconds', 0):.2f}s")
        return 0
    
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    exit(main())
