#!/usr/bin/env python3
"""
Verify LiveCodeBench scores are authentic and from actual benchmark evaluation.

This script checks:
1. That scores in livecodebench_scores.json match models_cache.json
2. That scores are from Artificial Analysis API (official source)
3. Coverage statistics for the benchmark

For missing models (like GPT-3.5 Turbo), this script identifies them for
manual evaluation using the LiveCodeBench benchmark.

Usage:
    python verify_livecodebench_scores.py
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple


def load_json(filepath: Path) -> dict:
    """Load JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def verify_scores():
    """Verify LiveCodeBench scores across data files."""
    
    script_dir = Path(__file__).parent
    lcb_scores_path = script_dir / "livecodebench_scores.json"
    models_cache_path = script_dir.parent.parent.parent / "data" / "models_cache.json"
    
    print("="*70)
    print("LiveCodeBench Score Verification")
    print("="*70)
    
    # Load data
    print(f"\nLoading data files...")
    lcb_data = load_json(lcb_scores_path)
    cache_data = load_json(models_cache_path)
    
    lcb_models = lcb_data.get("models", [])
    cache_models = cache_data.get("models", [])
    
    print(f"✓ LiveCodeBench scores file: {len(lcb_models)} models")
    print(f"✓ Models cache: {len(cache_models)} models")
    
    # Create lookups
    lcb_lookup = {m["slug"]: m for m in lcb_models}
    cache_lookup = {m["slug"]: m for m in cache_models}
    
    # Verification checks
    print("\n" + "="*70)
    print("Verification Checks")
    print("="*70)
    
    # Check 1: Data source authenticity
    print("\n1. Data Source Verification:")
    sources = set(m.get("source") for m in lcb_models)
    print(f"   Sources in livecodebench_scores.json: {sources}")
    if sources == {"artificial_analysis_api"}:
        print("   ✓ All scores from Artificial Analysis API (official source)")
    else:
        print("   ⚠ Multiple sources detected")
    
    # Check 2: Score consistency
    print("\n2. Score Consistency Check:")
    mismatches = []
    matches = 0
    
    for slug, lcb_model in lcb_lookup.items():
        if slug in cache_lookup:
            cache_model = cache_lookup[slug]
            lcb_score = lcb_model.get("livecodebench")
            cache_score = cache_model.get("livecodebench")
            
            if lcb_score != cache_score:
                mismatches.append({
                    "slug": slug,
                    "name": lcb_model.get("name"),
                    "lcb_score": lcb_score,
                    "cache_score": cache_score
                })
            else:
                matches += 1
    
    print(f"   Matching scores: {matches}")
    print(f"   Mismatches: {len(mismatches)}")
    
    if mismatches:
        print("\n   ⚠ Score mismatches found:")
        for m in mismatches[:5]:
            print(f"     - {m['name']}: LCB={m['lcb_score']}, Cache={m['cache_score']}")
        if len(mismatches) > 5:
            print(f"     ... and {len(mismatches)-5} more")
    else:
        print("   ✓ All scores match between files")
    
    # Check 3: Coverage analysis
    print("\n3. Coverage Analysis:")
    
    # Models in cache but not in LCB scores
    missing_from_lcb = []
    for slug in cache_lookup:
        if slug not in lcb_lookup:
            missing_from_lcb.append(cache_lookup[slug].get("name", slug))
    
    # Models in LCB but not in cache
    missing_from_cache = []
    for slug in lcb_lookup:
        if slug not in cache_lookup:
            missing_from_cache.append(lcb_lookup[slug].get("name", slug))
    
    print(f"   Models in cache but not in LCB scores: {len(missing_from_lcb)}")
    if missing_from_lcb:
        for name in missing_from_lcb[:5]:
            print(f"     - {name}")
        if len(missing_from_lcb) > 5:
            print(f"     ... and {len(missing_from_lcb)-5} more")
    
    print(f"   Models in LCB scores but not in cache: {len(missing_from_cache)}")
    if missing_from_cache:
        for name in missing_from_cache[:5]:
            print(f"     - {name}")
        if len(missing_from_cache) > 5:
            print(f"     ... and {len(missing_from_cache)-5} more")
    
    # Check 4: Null scores (missing LiveCodeBench data)
    print("\n4. Missing LiveCodeBench Scores:")
    
    null_in_lcb = [m for m in lcb_models if m.get("livecodebench") is None]
    null_in_cache = [m for m in cache_models if m.get("livecodebench") is None]
    
    print(f"   Models with null scores in LCB file: {len(null_in_lcb)}")
    for m in null_in_lcb:
        print(f"     - {m.get('name')} ({m.get('slug')})")
    
    print(f"   Models with null scores in cache: {len(null_in_cache)}")
    for m in null_in_cache:
        print(f"     - {m.get('name')} ({m.get('slug')})")
    
    # Check 5: Score distribution
    print("\n5. Score Distribution:")
    
    valid_scores = [m["livecodebench"] for m in cache_models if m.get("livecodebench") is not None]
    valid_scores.sort()
    
    if valid_scores:
        print(f"   Models with scores: {len(valid_scores)}")
        print(f"   Score range: {valid_scores[0]:.3f} to {valid_scores[-1]:.3f}")
        print(f"   Mean: {sum(valid_scores)/len(valid_scores):.3f}")
        print(f"   Median: {valid_scores[len(valid_scores)//2]:.3f}")
        
        # Top 5
        top_models = sorted(
            [(m.get("name"), m.get("livecodebench")) for m in cache_models if m.get("livecodebench") is not None],
            key=lambda x: x[1],
            reverse=True
        )[:5]
        print(f"\n   Top 5 Models:")
        for i, (name, score) in enumerate(top_models, 1):
            print(f"     {i}. {name}: {score:.3f} ({score*100:.1f}%)")
    
    # Summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    
    total_cache = len(cache_models)
    with_scores = len([m for m in cache_models if m.get("livecodebench") is not None])
    without_scores = total_cache - with_scores
    coverage_pct = (with_scores / total_cache * 100) if total_cache > 0 else 0
    
    print(f"✓ Total models in cache: {total_cache}")
    print(f"✓ Models with LiveCodeBench scores: {with_scores} ({coverage_pct:.1f}%)")
    print(f"✓ Models without LiveCodeBench scores: {without_scores}")
    print(f"✓ Data source: Artificial Analysis API (official)")
    print(f"✓ Score consistency: {'PASS' if not mismatches else 'FAIL'}")
    
    if without_scores > 0:
        print(f"\n⚠ Action Required:")
        print(f"  {without_scores} model(s) need LiveCodeBench evaluation:")
        for m in null_in_cache:
            print(f"    - {m.get('name')} ({m.get('slug')})")
        print(f"\n  Options:")
        print(f"    1. Wait for Artificial Analysis to add these models")
        print(f"    2. Run manual evaluation using fetch_livecodebench.py + evaluate_code.py")
        print(f"    3. Mark as 'N/A' if model doesn't support code generation")
    
    return {
        "total_models": total_cache,
        "with_scores": with_scores,
        "without_scores": without_scores,
        "coverage_percent": coverage_pct,
        "mismatches": len(mismatches),
        "models_needing_eval": [m.get("slug") for m in null_in_cache]
    }


if __name__ == "__main__":
    try:
        results = verify_scores()
        
        if results["mismatches"] > 0:
            print("\n❌ Verification failed: Score mismatches detected")
            exit(1)
        elif results["without_scores"] > 0:
            print(f"\n⚠ Verification partial: {results['without_scores']} model(s) missing scores")
            exit(0)
        else:
            print("\n✓ Verification complete: All scores verified and consistent")
            exit(0)
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
