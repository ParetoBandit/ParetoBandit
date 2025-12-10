#!/usr/bin/env python3
"""
Validate MATH-500 Score Sources
================================

This script validates the MATH-500 scores in models_cache.json by:
1. Fetching data from Artificial Analysis API
2. Comparing against cached values
3. Checking VALS benchmark data (secondary source)
4. Reporting discrepancies

Sources:
- artificial_analysis: Artificial Analysis API (https://artificialanalysis.ai)
- vals_benchmark: Verified AI Labs Studio benchmark data

Usage:
    python scripts/validate_math500_sources.py --all
    python scripts/validate_math500_sources.py --fetch-aa
    python scripts/validate_math500_sources.py --list-sources
    python scripts/validate_math500_sources.py --update-sources
"""

import json
import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    # If dotenv not installed, try manual loading
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ.setdefault(key.strip(), value.strip().strip('"\''))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
CACHE_PATH = PROJECT_ROOT / "data" / "models_cache.json"
VALS_SCORES_PATH = PROJECT_ROOT / "data" / "vals_math500_scores.json"

# Artificial Analysis API
AA_API_URL = "https://artificialanalysis.ai/api/v2/data/llms/models"


@dataclass
class ValidationResult:
    """Result of validating a single model's MATH-500 score."""
    model_name: str
    source: str
    cached_score: float
    fetched_score: Optional[float]
    match: bool
    difference: Optional[float]
    notes: str = ""


def load_models_cache() -> List[Dict]:
    """Load models from the cache file."""
    with open(CACHE_PATH) as f:
        data = json.load(f)
    return data.get('models', data) if isinstance(data, dict) else data


def load_vals_scores() -> Dict[str, float]:
    """Load VALS benchmark scores."""
    if not VALS_SCORES_PATH.exists():
        logger.warning(f"VALS scores file not found: {VALS_SCORES_PATH}")
        return {}
    
    with open(VALS_SCORES_PATH) as f:
        data = json.load(f)
    
    # Build mapping from model name to score
    scores = {}
    for entry in data:
        model_name = entry.get('model', '')
        score = entry.get('math_500_vals')
        if model_name and score is not None:
            scores[model_name] = score
            # Also store normalized versions
            scores[model_name.lower()] = score
    
    logger.info(f"Loaded {len(data)} VALS MATH-500 scores")
    return scores


def fetch_artificial_analysis_data(api_key: Optional[str] = None) -> Dict[str, Dict]:
    """
    Fetch MATH-500 scores from Artificial Analysis API.
    
    Args:
        api_key: Artificial Analysis API key. If not provided, tries env var.
    
    Returns:
        Dict mapping model names to their data including math_500 score
    """
    import requests
    
    if api_key is None:
        api_key = os.environ.get('ARTIFICIAL_ANALYSIS_API_KEY')
    
    if not api_key:
        logger.error("No Artificial Analysis API key provided. Set ARTIFICIAL_ANALYSIS_API_KEY env var.")
        return {}
    
    logger.info("Fetching data from Artificial Analysis API...")
    
    try:
        headers = {
            "x-api-key": api_key,
            "Accept": "application/json"
        }
        response = requests.get(AA_API_URL, headers=headers, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if data.get("status") != 200:
            logger.warning(f"API returned non-200 status: {data.get('status')}")
            return {}
        
        models = data.get("data", [])
        logger.info(f"Fetched {len(models)} models from Artificial Analysis")
        
        # Build mapping
        result = {}
        for m in models:
            name = m.get('name', '')
            slug = m.get('slug', '')
            evaluations = m.get('evaluations', {})
            math_500 = evaluations.get('math_500')
            
            if name:
                result[name] = {
                    'name': name,
                    'slug': slug,
                    'math_500': math_500,
                    'evaluations': evaluations
                }
                # Also store by slug
                if slug:
                    result[slug] = result[name]
        
        return result
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            logger.error("Invalid or missing Artificial Analysis API key")
        elif e.response.status_code == 429:
            logger.error("Rate limit exceeded")
        else:
            logger.error(f"HTTP error: {e}")
        return {}
    except Exception as e:
        logger.error(f"Failed to fetch from Artificial Analysis: {e}")
        return {}


def get_aa_model_mapping_from_cache(cache: List[Dict]) -> Dict[str, str]:
    """
    Extract Artificial Analysis model name mappings from the cache.
    Uses aa_id or tries to match by name.
    
    Returns:
        Dict mapping cache model names to AA model names
    """
    mapping = {}
    
    for model in cache:
        name = model.get('name', '')
        aa_id = model.get('aa_id')
        slug = model.get('slug', '')
        
        # Store potential mappings
        if aa_id:
            mapping[name] = aa_id
        elif slug:
            mapping[name] = slug
    
    return mapping


def get_models_by_source(cache: List[Dict]) -> Dict[str, List[Dict]]:
    """Group models by their math_500 source."""
    by_source = {}
    
    for model in cache:
        math_500 = model.get('math_500')
        if math_500 is None:
            continue
            
        source = model.get('math_500_source', 'unknown')
        if source not in by_source:
            by_source[source] = []
        by_source[source].append(model)
    
    return by_source


def validate_aa_source(cache: List[Dict], aa_data: Dict[str, Dict]) -> List[ValidationResult]:
    """Validate models against Artificial Analysis data."""
    results = []
    
    if not aa_data:
        logger.error("No Artificial Analysis data available for validation")
        return results
    
    for model in cache:
        math_500 = model.get('math_500')
        if math_500 is None:
            continue
        
        model_name = model.get('name', '')
        slug = model.get('slug', '')
        
        # Try to find in AA data
        aa_model = None
        matched_name = None
        
        # Strategy 1: Direct name match
        if model_name in aa_data:
            aa_model = aa_data[model_name]
            matched_name = model_name
        
        # Strategy 2: Slug match
        if not aa_model and slug in aa_data:
            aa_model = aa_data[slug]
            matched_name = slug
        
        # Strategy 3: Partial name match
        if not aa_model:
            model_name_lower = model_name.lower()
            for aa_name, aa_info in aa_data.items():
                aa_name_lower = aa_name.lower()
                if model_name_lower in aa_name_lower or aa_name_lower in model_name_lower:
                    aa_model = aa_info
                    matched_name = aa_name
                    break
        
        if aa_model:
            fetched_score = aa_model.get('math_500')
            if fetched_score is not None:
                diff = abs(math_500 - fetched_score)
                match = diff < 0.02  # 2% tolerance
                
                results.append(ValidationResult(
                    model_name=model_name,
                    source='artificial_analysis',
                    cached_score=math_500,
                    fetched_score=fetched_score,
                    match=match,
                    difference=diff,
                    notes=f"AA: {matched_name}" if matched_name != model_name else ""
                ))
            else:
                results.append(ValidationResult(
                    model_name=model_name,
                    source='artificial_analysis',
                    cached_score=math_500,
                    fetched_score=None,
                    match=False,
                    difference=None,
                    notes="No math_500 in AA data"
                ))
        else:
            results.append(ValidationResult(
                model_name=model_name,
                source='artificial_analysis',
                cached_score=math_500,
                fetched_score=None,
                match=False,
                difference=None,
                notes="Not found in Artificial Analysis"
            ))
    
    return results


def validate_vals_source(cache: List[Dict], vals_data: Dict[str, float]) -> List[ValidationResult]:
    """Validate models against VALS benchmark data."""
    results = []
    
    if not vals_data:
        logger.error("No VALS data available for validation")
        return results
    
    for model in cache:
        math_500 = model.get('math_500')
        if math_500 is None:
            continue
        
        model_name = model.get('name', '')
        
        # Try to find in VALS data
        fetched_score = None
        matched_name = None
        
        # Try direct match
        if model_name in vals_data:
            fetched_score = vals_data[model_name]
            matched_name = model_name
        
        # Try lowercase match
        if fetched_score is None and model_name.lower() in vals_data:
            fetched_score = vals_data[model_name.lower()]
            matched_name = model_name.lower()
        
        # Try partial match
        if fetched_score is None:
            for vals_name, score in vals_data.items():
                if isinstance(vals_name, str):
                    if model_name.lower() in vals_name.lower() or vals_name.lower() in model_name.lower():
                        fetched_score = score
                        matched_name = vals_name
                        break
        
        if fetched_score is not None:
            diff = abs(math_500 - fetched_score)
            match = diff < 0.02  # 2% tolerance
            
            results.append(ValidationResult(
                model_name=model_name,
                source='vals_benchmark',
                cached_score=math_500,
                fetched_score=fetched_score,
                match=match,
                difference=diff,
                notes=f"VALS: {matched_name}" if matched_name != model_name else ""
            ))
    
    return results


def print_validation_results(results: List[ValidationResult], source_name: str):
    """Print validation results in a formatted way."""
    if not results:
        print(f"\n{'=' * 70}")
        print(f"VALIDATION: {source_name}")
        print("=" * 70)
        print("No results to display")
        return
    
    matches = [r for r in results if r.match]
    mismatches = [r for r in results if not r.match and r.fetched_score is not None]
    not_found = [r for r in results if r.fetched_score is None]
    
    print(f"\n{'=' * 70}")
    print(f"VALIDATION: {source_name}")
    print("=" * 70)
    print(f"\n✅ Matches: {len(matches)}")
    print(f"❌ Mismatches: {len(mismatches)}")
    print(f"❓ Not Found: {len(not_found)}")
    
    if matches:
        print(f"\n{'-' * 70}")
        print("MATCHES:")
        for r in matches[:5]:
            print(f"  ✅ {r.model_name}: cache={r.cached_score:.3f}, source={r.fetched_score:.3f}")
        if len(matches) > 5:
            print(f"  ... and {len(matches) - 5} more matches")
    
    if mismatches:
        print(f"\n{'-' * 70}")
        print("MISMATCHES:")
        for r in mismatches:
            print(f"  ❌ {r.model_name}: cache={r.cached_score:.3f}, source={r.fetched_score:.3f} (diff={r.difference:.3f})")
            if r.notes:
                print(f"     Note: {r.notes}")
    
    if not_found:
        print(f"\n{'-' * 70}")
        print("NOT FOUND IN SOURCE:")
        for r in not_found[:10]:
            print(f"  ❓ {r.model_name}: cache={r.cached_score:.3f}")
            if r.notes:
                print(f"     Note: {r.notes}")
        if len(not_found) > 10:
            print(f"  ... and {len(not_found) - 10} more")


def print_source_summary(cache: List[Dict]):
    """Print summary of MATH-500 sources in cache."""
    # Count models with math_500 scores
    with_math500 = [m for m in cache if m.get('math_500') is not None]
    without_math500 = [m for m in cache if m.get('math_500') is None]
    
    # Group by source
    by_source = {}
    for m in with_math500:
        source = m.get('math_500_source', 'unknown')
        if source not in by_source:
            by_source[source] = []
        by_source[source].append(m)
    
    print("\n" + "=" * 70)
    print("MATH-500 SCORE SOURCES IN CACHE")
    print("=" * 70)
    
    print(f"\nModels with MATH-500 scores: {len(with_math500)}")
    print(f"Models without MATH-500 scores: {len(without_math500)}")
    
    for source, models in sorted(by_source.items(), key=lambda x: -len(x[1])):
        print(f"\n{source}: {len(models)} models")
        
        if source == 'artificial_analysis':
            print(f"  📍 Source: https://artificialanalysis.ai")
            print(f"  📝 Method: Artificial Analysis API")
        elif source == 'vals_benchmark':
            print(f"  📍 Source: {VALS_SCORES_PATH}")
            print(f"  📝 Method: VALS benchmark data file")
        elif source == 'unknown':
            print(f"  📍 Source: Unknown (no source field)")
        
        # Show sample models sorted by score
        sorted_models = sorted(models, key=lambda x: x.get('math_500', 0), reverse=True)
        for m in sorted_models[:3]:
            score = m.get('math_500', 0)
            print(f"    • {m['name']}: {score:.3f}")
        if len(models) > 3:
            print(f"    ... and {len(models) - 3} more")
    
    print(f"\n{'─' * 70}")
    print(f"Total models with MATH-500 data: {len(with_math500)}")


def update_sources_in_cache():
    """Update math_500_source field for all models in cache."""
    with open(CACHE_PATH) as f:
        data = json.load(f)
    
    models = data.get('models', data) if isinstance(data, dict) else data
    
    updated = 0
    for m in models:
        if m.get('math_500') is not None and not m.get('math_500_source'):
            m['math_500_source'] = 'artificial_analysis'
            m['math_500_source_url'] = 'https://artificialanalysis.ai'
            updated += 1
    
    with open(CACHE_PATH, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Updated {updated} models with math_500_source='artificial_analysis'")


def main():
    parser = argparse.ArgumentParser(
        description="Validate MATH-500 scores against original sources"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Validate all sources"
    )
    parser.add_argument(
        "--fetch-aa", action="store_true",
        help="Just fetch and display Artificial Analysis data"
    )
    parser.add_argument(
        "--list-sources", action="store_true",
        help="List all sources and their models"
    )
    parser.add_argument(
        "--update-sources", action="store_true",
        help="Update math_500_source field in cache for models without it"
    )
    parser.add_argument(
        "--api-key", type=str,
        help="Artificial Analysis API key (or set ARTIFICIAL_ANALYSIS_API_KEY env var)"
    )
    
    args = parser.parse_args()
    
    # Load cache
    cache = load_models_cache()
    print(f"Loaded {len(cache)} models from cache")
    
    # Update sources
    if args.update_sources:
        update_sources_in_cache()
        return
    
    # Just fetch AA data
    if args.fetch_aa:
        aa_data = fetch_artificial_analysis_data(args.api_key)
        if aa_data:
            print(f"\nFetched {len(aa_data)} models from Artificial Analysis")
            print("\nModels with MATH-500 scores:")
            for name, info in sorted(aa_data.items(), key=lambda x: x[1].get('math_500') or 0, reverse=True):
                score = info.get('math_500')
                if score is not None:
                    print(f"  {name}: {score:.3f}")
        return
    
    # List sources
    if args.list_sources:
        print_source_summary(cache)
        return
    
    # Validate all
    if args.all:
        print_source_summary(cache)
        
        print("\n\n" + "=" * 70)
        print("RUNNING VALIDATIONS")
        print("=" * 70)
        
        # Artificial Analysis validation
        aa_data = fetch_artificial_analysis_data(args.api_key)
        if aa_data:
            aa_results = validate_aa_source(cache, aa_data)
            print_validation_results(aa_results, 'Artificial Analysis')
        else:
            print("\n⚠️ Skipping Artificial Analysis validation (no API key or fetch failed)")
            aa_results = []
        
        # VALS validation
        vals_data = load_vals_scores()
        vals_results = validate_vals_source(cache, vals_data)
        print_validation_results(vals_results, 'VALS Benchmark')
        
        # Summary
        print("\n\n" + "=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)
        
        all_results = aa_results + vals_results
        if all_results:
            total_validated = len(all_results)
            total_matches = sum(1 for r in all_results if r.match)
            total_mismatches = sum(1 for r in all_results if not r.match and r.fetched_score is not None)
            total_not_found = sum(1 for r in all_results if r.fetched_score is None)
            
            print(f"\nTotal validations: {total_validated}")
            print(f"  ✅ Matches: {total_matches} ({100*total_matches/total_validated:.1f}%)" if total_validated else "")
            print(f"  ❌ Mismatches: {total_mismatches}")
            print(f"  ❓ Not found in source: {total_not_found}")
        else:
            print("\nNo validations performed")
        
        return
    
    # Default: show help
    parser.print_help()


if __name__ == "__main__":
    main()
