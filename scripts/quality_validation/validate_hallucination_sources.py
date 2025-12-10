#!/usr/bin/env python3
"""
Validate Hallucination Score Sources
=====================================

This script validates the hallucination scores in models_cache.json by:
1. Fetching data from original sources
2. Comparing against cached values
3. Reporting discrepancies

Sources:
- vectara_leaderboard: GitHub repo vectara/hallucination-leaderboard
- Voronoi 2025: Visual Capitalist infographics (manual/archived data)
- llm_jury_evaluation: Internal TruthfulQA evaluations
- Kaggle SimpleQA: OpenAI's SimpleQA benchmark
- visual_capitalist_benchmark: Visual Capitalist article

Usage:
    python scripts/validate_hallucination_sources.py --all
    python scripts/validate_hallucination_sources.py --source vectara
    python scripts/validate_hallucination_sources.py --source truthfulqa
    python scripts/validate_hallucination_sources.py --list-sources
"""

import json
import argparse
import logging
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
CACHE_PATH = PROJECT_ROOT / "data" / "models_cache.json"
TRUTHFULQA_RESULTS_DIR = PROJECT_ROOT / "kdd_paper" / "truthfulqa_results"

# Source URLs
VECTARA_README_URL = "https://raw.githubusercontent.com/vectara/hallucination-leaderboard/main/README.md"
VECTARA_OLD_DATASET_URL = "https://raw.githubusercontent.com/vectara/hallucination-leaderboard/hhem-2.3-old-dataset/README.md"
VISUAL_CAPITALIST_URL = "https://www.visualcapitalist.com/ranked-ai-models-with-the-lowest-hallucination-rates/"
VORONOI_URL = "https://voronoiapp.com/technology/Ranking-AI-Models-by-Hallucination-Rate-3076"
KAGGLE_SIMPLEQA_URL = "https://www.kaggle.com/datasets/openai/simple-qa"


def load_models_cache() -> List[Dict]:
    """Load models from the cache file."""
    with open(CACHE_PATH) as f:
        data = json.load(f)
    return data.get('models', data) if isinstance(data, dict) else data


@dataclass
class ValidationResult:
    """Result of validating a single model's hallucination score."""
    model_name: str
    source: str
    cached_rate: float
    fetched_rate: Optional[float]
    match: bool
    difference: Optional[float]
    notes: str = ""


# ============================================================================
# SOURCE: VECTARA LEADERBOARD
# ============================================================================

def fetch_vectara_leaderboard() -> Dict[str, Dict]:
    """
    Fetch hallucination rates from Vectara's GitHub leaderboard.
    
    Source: https://github.com/vectara/hallucination-leaderboard
    
    Returns:
        Dict mapping model names to their hallucination data
    """
    logger.info("Fetching Vectara Hallucination Leaderboard...")
    
    try:
        response = requests.get(VECTARA_README_URL, timeout=30)
        response.raise_for_status()
        readme_content = response.text
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch Vectara leaderboard: {e}")
        return {}
    
    models = {}
    lines = readme_content.split('\n')
    in_table = False
    
    for line in lines:
        line = line.strip()
        
        # Detect table header
        if line.startswith('|Model|') and 'Hallucination Rate' in line:
            in_table = True
            continue
        
        # Skip separator line
        if in_table and line.startswith('|') and '----' in line:
            continue
        
        # End of table
        if in_table and (line.startswith('#') or line.startswith('<!--') or (line and not line.startswith('|'))):
            break
        
        # Parse data rows
        if in_table and line.startswith('|'):
            parts = [p.strip() for p in line.split('|')]
            parts = [p for p in parts if p]
            
            if len(parts) >= 4:
                try:
                    model_name = parts[0]
                    halluc_rate = float(parts[1].replace('%', '').strip())
                    factual_rate = float(parts[2].replace('%', '').strip())
                    answer_rate = float(parts[3].replace('%', '').strip())
                    
                    models[model_name] = {
                        'hallucination_rate': halluc_rate,
                        'factual_consistency_rate': factual_rate,
                        'answer_rate': answer_rate,
                    }
                except (ValueError, IndexError):
                    continue
    
    logger.info(f"Fetched {len(models)} models from Vectara leaderboard")
    return models


def get_vectara_mapping_from_cache(cache: List[Dict]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Extract Vectara model name mappings from the cache.
    
    Returns:
        Tuple of (vectara_to_cache, cache_to_vectara) mappings
    """
    vectara_to_cache = {}
    cache_to_vectara = {}
    
    for model in cache:
        name = model.get('name', '')
        vectara_name = model.get('vectara_model_name')
        
        if vectara_name:
            vectara_to_cache[vectara_name] = name
            cache_to_vectara[name] = vectara_name
            
            # Also add lowercase version for matching
            vectara_to_cache[vectara_name.lower()] = name
    
    return vectara_to_cache, cache_to_vectara


def get_vectara_model_mapping(cache: Optional[List[Dict]] = None) -> Dict[str, str]:
    """
    Mapping from Vectara model names (API-style) to our cache model names.
    Uses mappings stored in the cache's vectara_model_name field.
    
    Args:
        cache: Optional list of models from cache. If not provided, loads from file.
    
    Returns:
        Dict mapping Vectara names to cache names
    """
    if cache is None:
        cache = load_models_cache()
    
    vectara_to_cache, _ = get_vectara_mapping_from_cache(cache)
    return vectara_to_cache


def get_cache_to_vectara_mapping(cache: Optional[List[Dict]] = None) -> Dict[str, str]:
    """
    Reverse mapping from cache model names to Vectara model names.
    Uses mappings stored in the cache's vectara_model_name field.
    
    Args:
        cache: Optional list of models from cache. If not provided, loads from file.
    
    Returns:
        Dict mapping cache names to Vectara names
    """
    if cache is None:
        cache = load_models_cache()
    
    _, cache_to_vectara = get_vectara_mapping_from_cache(cache)
    return cache_to_vectara


# ============================================================================
# SOURCE: VECTARA LEADERBOARD (OLD DATASET - hhem-2.3-old-dataset)
# ============================================================================

def fetch_vectara_old_dataset() -> Dict[str, Dict]:
    """
    Fetch hallucination rates from Vectara's OLD dataset (hhem-2.3-old-dataset branch).
    
    This contains models that were removed from the main leaderboard when 
    Vectara updated to a new dataset, including Gemini 2.0 models.
    
    Source: https://github.com/vectara/hallucination-leaderboard/tree/hhem-2.3-old-dataset
    
    Returns:
        Dict mapping model names to their hallucination data
    """
    logger.info("Fetching Vectara OLD Dataset Leaderboard...")
    
    try:
        response = requests.get(VECTARA_OLD_DATASET_URL, timeout=30)
        response.raise_for_status()
        readme_content = response.text
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch Vectara old dataset: {e}")
        return {}
    
    models = {}
    lines = readme_content.split('\n')
    in_table = False
    
    for line in lines:
        line = line.strip()
        
        # Detect table header
        if line.startswith('|') and 'Hallucination Rate' in line:
            in_table = True
            continue
        
        # Skip separator line
        if in_table and line.startswith('|') and '----' in line:
            continue
        
        # End of table
        if in_table and (line.startswith('#') or line.startswith('<!--') or (line and not line.startswith('|'))):
            break
        
        # Parse data rows
        if in_table and line.startswith('|'):
            parts = [p.strip() for p in line.split('|')]
            parts = [p for p in parts if p]
            
            if len(parts) >= 4:
                try:
                    model_name = parts[0]
                    halluc_rate = float(parts[1].replace('%', '').strip())
                    factual_rate = float(parts[2].replace('%', '').strip())
                    answer_rate = float(parts[3].replace('%', '').strip())
                    
                    models[model_name] = {
                        'hallucination_rate': halluc_rate,
                        'factual_consistency_rate': factual_rate,
                        'answer_rate': answer_rate,
                    }
                except (ValueError, IndexError):
                    continue
    
    logger.info(f"Fetched {len(models)} models from Vectara old dataset")
    return models


def get_vectara_old_dataset_mapping(cache: Optional[List[Dict]] = None) -> Dict[str, str]:
    """
    Mapping from Vectara OLD dataset model names to our cache model names.
    Uses mappings stored in the cache's vectara_model_name field for models
    with vectara_leaderboard_old_dataset source.
    
    Args:
        cache: Optional list of models from cache. If not provided, loads from file.
    
    Returns:
        Dict mapping Vectara old dataset names to cache names
    """
    if cache is None:
        cache = load_models_cache()
    
    mapping = {}
    for model in cache:
        if model.get('hallucination_source') == 'vectara_leaderboard_old_dataset':
            name = model.get('name', '')
            vectara_name = model.get('vectara_model_name')
            
            if vectara_name:
                mapping[vectara_name] = name
                # Also add lowercase version for matching
                mapping[vectara_name.lower()] = name
    
    return mapping


def validate_vectara_old_dataset_source(cache: List[Dict]) -> List[ValidationResult]:
    """Validate models with vectara_leaderboard_old_dataset source."""
    results = []
    
    # Fetch old dataset
    vectara_data = fetch_vectara_old_dataset()
    if not vectara_data:
        logger.error("Could not fetch Vectara old dataset for validation")
        return results
    
    # Find models in cache with vectara_leaderboard_old_dataset source
    for model in cache:
        if model.get('hallucination_source') != 'vectara_leaderboard_old_dataset':
            continue
        
        model_name = model.get('name', '')
        cached_rate = model.get('hallucination_rate')
        
        if cached_rate is None:
            continue
        
        # Use vectara_model_name from cache if available
        vectara_name = model.get('vectara_model_name', model_name)
        
        fetched_data = vectara_data.get(vectara_name)
        
        if fetched_data:
            fetched_rate = fetched_data['hallucination_rate']
            diff = abs(cached_rate - fetched_rate)
            match = diff < 0.5  # Allow 0.5% tolerance
            
            results.append(ValidationResult(
                model_name=model_name,
                source='vectara_leaderboard_old_dataset',
                cached_rate=cached_rate,
                fetched_rate=fetched_rate,
                match=match,
                difference=diff,
                notes=f"Vectara old dataset: {vectara_name}" if vectara_name != model_name else ""
            ))
        else:
            results.append(ValidationResult(
                model_name=model_name,
                source='vectara_leaderboard_old_dataset',
                cached_rate=cached_rate,
                fetched_rate=None,
                match=False,
                difference=None,
                notes="Not found in Vectara old dataset"
            ))
    
    return results


# ============================================================================
# SOURCE: TRUTHFULQA (Internal Evaluations)
# ============================================================================

def load_truthfulqa_results() -> Dict[str, Dict]:
    """
    Load TruthfulQA evaluation results from our local evaluations.
    
    These were generated by: kdd_paper/run_truthfulqa_evaluation.py
    Results stored in: kdd_paper/truthfulqa_results/
    
    Returns:
        Dict mapping model names to their TruthfulQA results
    """
    logger.info("Loading TruthfulQA results from local files...")
    
    results = {}
    
    if not TRUTHFULQA_RESULTS_DIR.exists():
        logger.warning(f"TruthfulQA results directory not found: {TRUTHFULQA_RESULTS_DIR}")
        return results
    
    for result_file in TRUTHFULQA_RESULTS_DIR.glob("*_results.json"):
        try:
            with open(result_file) as f:
                data = json.load(f)
            
            model_name = data.get("model", "")
            accuracy = data.get("accuracy", 0)
            correct = data.get("correct", 0)
            total = data.get("total", 0)
            
            if model_name:
                # TruthfulQA accuracy -> estimated hallucination rate
                # Hallucination ≈ (100 - accuracy) / 2 (conservative estimate)
                estimated_halluc = (100 - accuracy) / 2
                
                results[model_name] = {
                    'truthfulqa_accuracy': accuracy,
                    'truthfulqa_correct': correct,
                    'truthfulqa_total': total,
                    'hallucination_rate_estimated': estimated_halluc,
                    'source_file': result_file.name
                }
        except Exception as e:
            logger.warning(f"Error loading {result_file}: {e}")
    
    logger.info(f"Loaded {len(results)} TruthfulQA results")
    return results


# ============================================================================
# SOURCE: VORONOI 2025 / VISUAL CAPITALIST
# ============================================================================

# Voronoi 2025 data is typically from infographics that can't be easily scraped.
# We store the known values here for validation.
# Source: https://voronoiapp.com/technology/Ranking-AI-Models-by-Hallucination-Rate-3076
# Last updated: Dec 2025
VORONOI_2025_DATA = {
    # Model name (as in our cache) -> hallucination_rate
    'Llama 3.3 Nemotron Super 49B v1 (Reasoning)': 76.0,
    'Claude Opus 4.5 (Reasoning)': 43.0,
    'gpt-oss-20B (high)': 93.2,
    'GPT-5.1 (high)': 51.0,
    'Grok 4': 64.0,
    'DeepSeek R1 0528 Qwen3 8B': 83.0,
    'Kimi K2 0905': 69.0,
    'Claude 4.5 Sonnet (Reasoning)': 31.0,
    'Llama 4 Maverick': 87.6,
    'Claude 4.5 Haiku (Reasoning)': 16.0,
    'Qwen3 235B A22B (Reasoning)': 89.6,
    'Kimi K2 Thinking': 74.0,
    'MiniMax-M2': 88.9,
}

VISUAL_CAPITALIST_DATA = {
    # Source: https://www.visualcapitalist.com/ranked-ai-models-with-the-lowest-hallucination-rates/
    'GPT-3.5 Turbo': 1.9,
}


def get_voronoi_data() -> Dict[str, float]:
    """
    Get Voronoi 2025 hallucination data.
    
    Note: Voronoi infographics are visual and can't be scraped programmatically.
    This returns manually archived data from the source.
    
    To update: Visit https://voronoiapp.com/ and manually extract values.
    """
    logger.info("Loading Voronoi 2025 archived data...")
    return VORONOI_2025_DATA


def get_visual_capitalist_data() -> Dict[str, float]:
    """
    Get Visual Capitalist hallucination data.
    
    Source: https://www.visualcapitalist.com/ranked-ai-models-with-the-lowest-hallucination-rates/
    """
    logger.info("Loading Visual Capitalist archived data...")
    return VISUAL_CAPITALIST_DATA


# ============================================================================
# SOURCE: KAGGLE SIMPLEQA
# ============================================================================

KAGGLE_SIMPLEQA_DATA = {
    # Source: OpenAI's SimpleQA benchmark on Kaggle
    # https://www.kaggle.com/datasets/openai/simple-qa
    # Values represent 100 - correctness (inverted to get hallucination-like metric)
    # Note: SimpleQA Correctness measures factual accuracy
    # Last updated: Dec 2025
    'Claude 3.7 Sonnet (Reasoning)': 62.5,  # 100 - 37.5% correct
    'Gemini 3 Pro Preview (high)': 29.5,    # 100 - 70.5% correct
    'GPT-5 (ChatGPT)': 48.9,                # 100 - 51.1% correct
}


def get_simpleqa_data() -> Dict[str, float]:
    """
    Get Kaggle SimpleQA benchmark data.
    
    Note: SimpleQA requires Kaggle API access or manual download.
    This returns known values from the benchmark.
    
    To update: 
    1. Download from https://www.kaggle.com/datasets/openai/simple-qa
    2. Parse the results CSV
    """
    logger.info("Loading Kaggle SimpleQA archived data...")
    return KAGGLE_SIMPLEQA_DATA


# ============================================================================
# VALIDATION LOGIC
# ============================================================================

def load_cache() -> List[Dict]:
    """Load the models cache."""
    with open(CACHE_PATH) as f:
        data = json.load(f)
    return data.get('models', data) if isinstance(data, dict) else data


def get_models_by_source(cache: List[Dict]) -> Dict[str, List[Dict]]:
    """Group cached models by their hallucination source."""
    by_source = defaultdict(list)
    
    for model in cache:
        source = model.get('hallucination_source') or model.get('truthfulqa_source', 'unknown')
        rate = model.get('hallucination_rate') or model.get('hallucination_rate_estimated')
        
        if rate is not None:
            by_source[source].append({
                'name': model.get('name', 'Unknown'),
                'slug': model.get('slug', ''),
                'hallucination_rate': rate,
                'hallucination_source': source,
                'truthfulqa_accuracy': model.get('truthfulqa_accuracy'),
            })
    
    return dict(by_source)


def validate_vectara_source(cache: List[Dict]) -> List[ValidationResult]:
    """Validate models with vectara_leaderboard source."""
    results = []
    
    # Fetch current data
    vectara_data = fetch_vectara_leaderboard()
    if not vectara_data:
        logger.error("Could not fetch Vectara data for validation")
        return results
    
    # Build a normalized name lookup for Vectara data
    vectara_normalized = {}
    for vname, vdata in vectara_data.items():
        # Store with original name
        vectara_normalized[vname] = vdata
        # Also store normalized (lowercase, no special chars)
        normalized = vname.lower().replace('-', '').replace('_', '').replace('/', '').replace('.', '')
        vectara_normalized[normalized] = vdata
    
    # Find models in cache with vectara source
    for model in cache:
        if model.get('hallucination_source') != 'vectara_leaderboard':
            continue
        
        model_name = model.get('name', '')
        cached_rate = model.get('hallucination_rate')
        
        if cached_rate is None:
            continue
        
        # Try to find in Vectara data using multiple strategies
        fetched_data = None
        matched_vectara_name = None
        
        # Strategy 1: Use vectara_model_name from cache (primary method)
        vectara_model_name = model.get('vectara_model_name')
        if vectara_model_name and vectara_model_name in vectara_data:
            fetched_data = vectara_data[vectara_model_name]
            matched_vectara_name = vectara_model_name
        
        # Strategy 2: Direct name lookup
        if not fetched_data and model_name in vectara_data:
            fetched_data = vectara_data[model_name]
            matched_vectara_name = model_name
        
        # Strategy 3: Normalized name lookup
        if not fetched_data:
            normalized_cache = model_name.lower().replace(' ', '').replace('-', '').replace('_', '').replace('.', '')
            if normalized_cache in vectara_normalized:
                fetched_data = vectara_normalized[normalized_cache]
                matched_vectara_name = normalized_cache
        
        # Strategy 4: Partial match on model name
        if not fetched_data:
            # Try to find by matching key parts
            for vname, vdata in vectara_data.items():
                # Extract model family from vectara name (e.g., "DeepSeek-R1" from "deepseek-ai/DeepSeek-R1")
                vname_parts = vname.split('/')[-1].lower().replace('-', ' ').replace('_', ' ')
                model_parts = model_name.lower().replace('-', ' ').replace('_', ' ')
                
                # Check if key parts match
                if vname_parts in model_parts or model_parts in vname_parts:
                    fetched_data = vdata
                    matched_vectara_name = vname
                    break
        
        if fetched_data:
            fetched_rate = fetched_data['hallucination_rate']
            diff = abs(cached_rate - fetched_rate)
            match = diff < 1.0  # Allow 1% tolerance
            
            results.append(ValidationResult(
                model_name=model_name,
                source='vectara_leaderboard',
                cached_rate=cached_rate,
                fetched_rate=fetched_rate,
                match=match,
                difference=diff,
                notes=f"Vectara: {matched_vectara_name}" if matched_vectara_name != model_name else ""
            ))
        else:
            results.append(ValidationResult(
                model_name=model_name,
                source='vectara_leaderboard',
                cached_rate=cached_rate,
                fetched_rate=None,
                match=False,
                difference=None,
                notes="Not found in current Vectara leaderboard (may need mapping update)"
            ))
    
    return results


def validate_truthfulqa_source(cache: List[Dict]) -> List[ValidationResult]:
    """Validate models with llm_jury_evaluation source."""
    results = []
    
    # Load local TruthfulQA results
    truthfulqa_data = load_truthfulqa_results()
    
    for model in cache:
        if model.get('truthfulqa_source') != 'llm_jury_evaluation':
            continue
        
        model_name = model.get('name', '')
        cached_accuracy = model.get('truthfulqa_accuracy')
        
        if cached_accuracy is None:
            continue
        
        # Find in local results
        local_data = truthfulqa_data.get(model_name)
        
        if local_data:
            fetched_accuracy = local_data['truthfulqa_accuracy']
            diff = abs(cached_accuracy - fetched_accuracy)
            match = diff < 0.5
            
            results.append(ValidationResult(
                model_name=model_name,
                source='llm_jury_evaluation',
                cached_rate=cached_accuracy,
                fetched_rate=fetched_accuracy,
                match=match,
                difference=diff,
                notes=f"File: {local_data['source_file']}"
            ))
        else:
            results.append(ValidationResult(
                model_name=model_name,
                source='llm_jury_evaluation',
                cached_rate=cached_accuracy,
                fetched_rate=None,
                match=False,
                difference=None,
                notes="Result file not found"
            ))
    
    return results


def validate_voronoi_source(cache: List[Dict]) -> List[ValidationResult]:
    """Validate models with Voronoi 2025 source."""
    results = []
    voronoi_data = get_voronoi_data()
    
    for model in cache:
        if model.get('hallucination_source') != 'Voronoi 2025':
            continue
        
        model_name = model.get('name', '')
        cached_rate = model.get('hallucination_rate')
        
        if cached_rate is None:
            continue
        
        fetched_rate = voronoi_data.get(model_name)
        
        if fetched_rate is not None:
            diff = abs(cached_rate - fetched_rate)
            match = diff < 0.5
            
            results.append(ValidationResult(
                model_name=model_name,
                source='Voronoi 2025',
                cached_rate=cached_rate,
                fetched_rate=fetched_rate,
                match=match,
                difference=diff,
                notes=""
            ))
        else:
            results.append(ValidationResult(
                model_name=model_name,
                source='Voronoi 2025',
                cached_rate=cached_rate,
                fetched_rate=None,
                match=False,
                difference=None,
                notes="Not in archived Voronoi data - may need manual update"
            ))
    
    return results


def validate_simpleqa_source(cache: List[Dict]) -> List[ValidationResult]:
    """Validate models with Kaggle SimpleQA source."""
    results = []
    simpleqa_data = get_simpleqa_data()
    
    for model in cache:
        if model.get('hallucination_source') != 'Kaggle SimpleQA':
            continue
        
        model_name = model.get('name', '')
        cached_rate = model.get('hallucination_rate')
        
        if cached_rate is None:
            continue
        
        # Try direct match first
        fetched_rate = simpleqa_data.get(model_name)
        
        # Try partial matches
        if fetched_rate is None:
            for sqa_name, sqa_rate in simpleqa_data.items():
                if model_name.lower() in sqa_name.lower() or sqa_name.lower() in model_name.lower():
                    fetched_rate = sqa_rate
                    break
        
        if fetched_rate is not None:
            diff = abs(cached_rate - fetched_rate)
            match = diff < 1.0  # Allow 1% tolerance
            
            results.append(ValidationResult(
                model_name=model_name,
                source='Kaggle SimpleQA',
                cached_rate=cached_rate,
                fetched_rate=fetched_rate,
                match=match,
                difference=diff,
                notes=""
            ))
        else:
            results.append(ValidationResult(
                model_name=model_name,
                source='Kaggle SimpleQA',
                cached_rate=cached_rate,
                fetched_rate=None,
                match=False,
                difference=None,
                notes="Not in archived SimpleQA data"
            ))
    
    return results


# ============================================================================
# MAIN
# ============================================================================

def print_source_summary(cache: List[Dict]):
    """Print summary of hallucination sources in cache."""
    by_source = get_models_by_source(cache)
    
    print("\n" + "=" * 70)
    print("HALLUCINATION SCORE SOURCES IN CACHE")
    print("=" * 70)
    
    total = 0
    for source, models in sorted(by_source.items(), key=lambda x: -len(x[1])):
        print(f"\n{source}: {len(models)} models")
        total += len(models)
        
        # Show source details
        if source == 'vectara_leaderboard':
            print(f"  📍 Source: {VECTARA_README_URL}")
            print(f"  📝 Method: Scrape markdown table from GitHub README")
        elif source == 'vectara_leaderboard_old_dataset':
            print(f"  📍 Source: {VECTARA_OLD_DATASET_URL}")
            print(f"  📝 Method: Scrape markdown table from GitHub (hhem-2.3-old-dataset branch)")
        elif source == 'Voronoi 2025':
            print(f"  📍 Source: {VORONOI_URL}")
            print(f"  📝 Method: Manual extraction from infographic")
        elif source == 'llm_jury_evaluation':
            print(f"  📍 Source: {TRUTHFULQA_RESULTS_DIR}")
            print(f"  📝 Method: TruthfulQA benchmark evaluation")
        elif source == 'Kaggle SimpleQA':
            print(f"  📍 Source: {KAGGLE_SIMPLEQA_URL}")
            print(f"  📝 Method: SimpleQA correctness benchmark")
        elif source == 'visual_capitalist_benchmark':
            print(f"  📍 Source: {VISUAL_CAPITALIST_URL}")
            print(f"  📝 Method: Manual extraction from article")
        
        # Show sample models
        for m in models[:3]:
            rate = m['hallucination_rate']
            print(f"    • {m['name']}: {rate}%")
        if len(models) > 3:
            print(f"    ... and {len(models) - 3} more")
    
    print(f"\n{'─' * 70}")
    print(f"Total models with hallucination data: {total}")


def print_validation_results(results: List[ValidationResult], source_name: str):
    """Print validation results for a source."""
    if not results:
        print(f"\n⚠️  No models to validate for {source_name}")
        return
    
    print(f"\n{'=' * 70}")
    print(f"VALIDATION: {source_name}")
    print(f"{'=' * 70}")
    
    matches = [r for r in results if r.match]
    mismatches = [r for r in results if not r.match and r.fetched_rate is not None]
    not_found = [r for r in results if r.fetched_rate is None]
    
    print(f"\n✅ Matches: {len(matches)}")
    print(f"❌ Mismatches: {len(mismatches)}")
    print(f"❓ Not Found: {len(not_found)}")
    
    if matches:
        print(f"\n{'─' * 70}")
        print("MATCHES:")
        for r in matches[:5]:
            print(f"  ✅ {r.model_name}: cache={r.cached_rate}%, source={r.fetched_rate}%")
        if len(matches) > 5:
            print(f"  ... and {len(matches) - 5} more matches")
    
    if mismatches:
        print(f"\n{'─' * 70}")
        print("MISMATCHES:")
        for r in mismatches:
            print(f"  ❌ {r.model_name}: cache={r.cached_rate}%, source={r.fetched_rate}% (diff={r.difference:.1f})")
            if r.notes:
                print(f"     Note: {r.notes}")
    
    if not_found:
        print(f"\n{'─' * 70}")
        print("NOT FOUND IN SOURCE:")
        for r in not_found:
            print(f"  ❓ {r.model_name}: cache={r.cached_rate}%")
            if r.notes:
                print(f"     Note: {r.notes}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate hallucination scores against original sources"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Validate all sources"
    )
    parser.add_argument(
        "--source", type=str,
        choices=['vectara', 'truthfulqa', 'voronoi', 'simpleqa', 'visual_capitalist'],
        help="Validate a specific source"
    )
    parser.add_argument(
        "--list-sources", action="store_true",
        help="List all sources and their models"
    )
    parser.add_argument(
        "--fetch-vectara", action="store_true",
        help="Just fetch and display current Vectara leaderboard"
    )
    
    args = parser.parse_args()
    
    # Load cache
    cache = load_cache()
    print(f"Loaded {len(cache)} models from cache")
    
    # Just fetch Vectara
    if args.fetch_vectara:
        data = fetch_vectara_leaderboard()
        print(f"\n{'Model':<50} {'Halluc %':<10} {'Factual %':<10}")
        print("-" * 70)
        for name, vals in sorted(data.items(), key=lambda x: x[1]['hallucination_rate']):
            print(f"{name:<50} {vals['hallucination_rate']:<10.1f} {vals['factual_consistency_rate']:<10.1f}")
        return
    
    # List sources
    if args.list_sources:
        print_source_summary(cache)
        return
    
    # Validate specific source
    if args.source:
        source_validators = {
            'vectara': ('vectara_leaderboard', validate_vectara_source),
            'truthfulqa': ('llm_jury_evaluation', validate_truthfulqa_source),
            'voronoi': ('Voronoi 2025', validate_voronoi_source),
            'simpleqa': ('Kaggle SimpleQA', validate_simpleqa_source),
        }
        
        if args.source in source_validators:
            source_name, validator = source_validators[args.source]
            results = validator(cache)
            print_validation_results(results, source_name)
        else:
            print(f"Source '{args.source}' validation not implemented yet")
        return
    
    # Validate all
    if args.all:
        print_source_summary(cache)
        
        print("\n\n" + "=" * 70)
        print("RUNNING VALIDATIONS")
        print("=" * 70)
        
        # Vectara (current)
        vectara_results = validate_vectara_source(cache)
        print_validation_results(vectara_results, 'Vectara Leaderboard (Current)')
        
        # Vectara (old dataset)
        vectara_old_results = validate_vectara_old_dataset_source(cache)
        print_validation_results(vectara_old_results, 'Vectara Leaderboard (Old Dataset)')
        
        # Voronoi
        voronoi_results = validate_voronoi_source(cache)
        print_validation_results(voronoi_results, 'Voronoi 2025')
        
        # SimpleQA
        simpleqa_results = validate_simpleqa_source(cache)
        print_validation_results(simpleqa_results, 'Kaggle SimpleQA')
        
        # Summary
        print("\n\n" + "=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)
        
        all_results = vectara_results + vectara_old_results + voronoi_results + simpleqa_results
        total_validated = len(all_results)
        total_matches = sum(1 for r in all_results if r.match)
        total_mismatches = sum(1 for r in all_results if not r.match and r.fetched_rate is not None)
        total_not_found = sum(1 for r in all_results if r.fetched_rate is None)
        
        print(f"\nTotal models validated: {total_validated}")
        print(f"  ✅ Matches: {total_matches} ({100*total_matches/total_validated:.1f}%)" if total_validated else "")
        print(f"  ❌ Mismatches: {total_mismatches}")
        print(f"  ❓ Not found in source: {total_not_found}")
        
        return
    
    # Default: show help
    parser.print_help()


if __name__ == "__main__":
    main()
