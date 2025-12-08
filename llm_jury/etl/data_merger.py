"""Merge data from Artificial Analysis API into unified model cache."""

import json
import logging
from typing import Dict, List
from pathlib import Path

logger = logging.getLogger(__name__)


class DataMerger:
    """Merges data from Artificial Analysis API into unified cache."""

    def merge_aa_data(
        self,
        aa_models: List[Dict],
        existing_cache: List[Dict] = None,
    ) -> List[Dict]:
        """Merge data from Artificial Analysis API.

        Args:
            aa_models: Models from Artificial Analysis API (already normalized)
            existing_cache: Existing model cache to preserve custom fields (optional)

        Returns:
            Merged list of model dictionaries
        """
        logger.info(f"Merging data from {len(aa_models)} Artificial Analysis models")

        # Index existing cache by model ID for preserving custom fields
        cache_index = {}
        if existing_cache:
            for model in existing_cache:
                aa_id = model.get("aa_id")
                name = model.get("name")
                slug = model.get("slug")
                
                if aa_id:
                    cache_index[aa_id] = model
                if name:
                    cache_index[name] = model
                if slug:
                    cache_index[slug] = model

        # Process AA models
        merged_models = []
        for aa_model in aa_models:
            aa_id = aa_model.get("aa_id")
            name = aa_model.get("name")
            
            # Check if we have existing data to preserve
            existing = None
            if aa_id and aa_id in cache_index:
                existing = cache_index[aa_id]
            elif name and name in cache_index:
                existing = cache_index[name]
            
            if existing:
                # Start with existing and update with new AA data
                merged = existing.copy()
                merged.update(aa_model)
            else:
                # Brand new model
                merged = aa_model.copy()

            # All AA models have benchmarks
            merged["has_benchmarks"] = True
            merged["data_source"] = "artificial_analysis"

            # Ensure required fields
            merged = self._normalize_aa_model(merged)

            merged_models.append(merged)

        logger.info(f"Merged {len(merged_models)} models (all with benchmarks)")

        return merged_models

    def _normalize_aa_model(self, model: Dict) -> Dict:
        """Normalize Artificial Analysis model data.

        Args:
            model: Model dictionary from AA API

        Returns:
            Normalized model dictionary
        """
        # Ensure ID fields
        if "id" not in model and "aa_id" in model:
            model["id"] = model["aa_id"]
        
        if "display_name" not in model and "name" in model:
            model["display_name"] = model["name"]

        # Map pricing to standard fields for compatibility
        if "price_1m_input" in model and "input_cost_per_m" not in model:
            model["input_cost_per_m"] = model["price_1m_input"]
        
        if "price_1m_output" in model and "output_cost_per_m" not in model:
            model["output_cost_per_m"] = model["price_1m_output"]

        # Ensure has_benchmarks flag
        model["has_benchmarks"] = True

        return model

    def merge_hallucination_data(
        self,
        models: List[Dict],
        hallucination_data: List[Dict]
    ) -> List[Dict]:
        """Merge hallucination data from Vectara leaderboard into models.

        Args:
            models: List of model dictionaries
            hallucination_data: Hallucination data from Vectara leaderboard

        Returns:
            Updated list of model dictionaries with hallucination data
        """
        logger.info(f"Merging hallucination data for {len(hallucination_data)} models")

        # Build index of hallucination data by normalized model name
        hallucination_index = {}
        for h in hallucination_data:
            # Store by original model identifier
            model_id = h.get('model', '').lower()
            hallucination_index[model_id] = h
            
            # Also extract just the model name part (after /)
            if '/' in model_id:
                short_name = model_id.split('/')[-1].lower()
                hallucination_index[short_name] = h

        matched = 0
        for model in models:
            name = model.get('name', '').lower()
            slug = model.get('slug', '').lower()
            
            # Try various matching strategies
            h_data = None
            
            # Direct match on name
            for key in hallucination_index:
                # Check if the key matches part of our model name or vice versa
                if key in name or name in key:
                    h_data = hallucination_index[key]
                    break
                if slug and (key in slug or slug in key):
                    h_data = hallucination_index[key]
                    break
            
            # Try specific mappings for common patterns
            if not h_data:
                h_data = self._match_hallucination_model(model, hallucination_index)
            
            if h_data:
                model['hallucination_rate'] = h_data.get('hallucination_rate')
                model['factual_consistency_rate'] = h_data.get('factual_consistency_rate')
                model['hallucination_answer_rate'] = h_data.get('answer_rate')
                model['hallucination_source'] = 'vectara_leaderboard'
                matched += 1

        logger.info(f"Matched hallucination data for {matched}/{len(models)} models")
        return models

    def _match_hallucination_model(self, model: Dict, hallucination_index: Dict) -> Dict:
        """Try to match a model to hallucination data using various strategies.

        Args:
            model: Model dictionary
            hallucination_index: Index of hallucination data

        Returns:
            Matching hallucination data or None
        """
        name = model.get('name', '').lower()
        
        # Common mappings
        mappings = {
            'gemini 2.5 flash-lite': 'gemini-2.5-flash-lite',
            'gemini 2.5 flash': 'gemini-2.5-flash',
            'gemini 2.5 pro': 'gemini-2.5-pro',
            'gemini 3 pro preview': 'gemini-3-pro-preview',
            'gemma 3 12b instruct': 'gemma-3-12b-it',
            'gemma 3 27b instruct': 'gemma-3-27b-it',
            'gemma 3 4b instruct': 'gemma-3-4b-it',
            'llama 3.3 instruct 70b': 'llama-3.3-70b-instruct',
            'llama 3.1 instruct 70b': 'llama-3.1-70b-instruct',
            'llama 3.1 instruct 8b': 'llama-3.1-8b-instruct',
            'llama 3.1 instruct 405b': 'llama-3.1-405b-instruct',
            'llama 4 maverick': 'llama-4-maverick',
            'llama 4 scout': 'llama-4-scout',
            'qwen3 8b': 'qwen3-8b',
            'qwen3 14b': 'qwen3-14b',
            'qwen3 32b': 'qwen3-32b',
            'qwen3 4b': 'qwen3-4b',
            'deepseek v3': 'deepseek-v3',
            'deepseek v3.1': 'deepseek-v3.1',
            'deepseek v3.2 exp': 'deepseek-v3.2-exp',
            'deepseek r1': 'deepseek-r1',
            'grok 3': 'grok-3',
            'grok 4': 'grok-4',
            'grok 4 fast': 'grok-4-fast',
            'grok 4.1 fast': 'grok-4-1-fast',
            'phi-4': 'phi-4',
            'phi-4 mini': 'phi-4-mini',
            'granite 4.0 h small': 'granite-4.0-h-small',
            'granite 3.3 8b': 'granite-3.3-8b',
            'mistral large': 'mistral-large',
            'mistral small': 'mistral-small',
            'command a': 'command-a',
            'glm-4.5-air': 'glm-4.5-air',
            'glm-4.6': 'glm-4.6',
            'gpt-5 mini': 'gpt-5-mini',
            'gpt-5 nano': 'gpt-5-nano',
            'gpt-5.1': 'gpt-5.1',
            'gpt-oss-120b': 'gpt-oss-120b',
            'claude opus 4.5': 'claude-opus-4-5',
            'claude 4.5 sonnet': 'claude-sonnet-4-5',
            'claude 4 opus': 'claude-opus-4',
            'claude 4 sonnet': 'claude-sonnet-4',
            'claude 4.5 haiku': 'claude-haiku-4-5',
        }
        
        for our_name, their_name in mappings.items():
            if our_name in name:
                if their_name in hallucination_index:
                    return hallucination_index[their_name]
                # Try with provider prefix
                for key in hallucination_index:
                    if their_name in key:
                        return hallucination_index[key]
        
        return None

    def save_cache(self, models: List[Dict], output_file: Path):
        """Save merged models to cache file.

        Args:
            models: List of model dictionaries
            output_file: Path to output cache file
        """
        logger.info(f"Saving {len(models)} models to {output_file}")

        # Create backup if file exists
        if output_file.exists():
            backup_file = output_file.with_suffix(output_file.suffix + ".bak")
            output_file.rename(backup_file)
            logger.info(f"Created backup: {backup_file}")

        # Sort models by ID for consistency
        sorted_models = sorted(models, key=lambda m: m.get("id", ""))

        # Save to file
        with open(output_file, "w") as f:
            json.dump(sorted_models, f, indent=2)

        logger.info(f"Cache saved successfully: {output_file}")

    def load_cache(self, cache_file: Path) -> List[Dict]:
        """Load existing cache from file.

        Args:
            cache_file: Path to cache file

        Returns:
            List of model dictionaries
        """
        if not cache_file.exists():
            logger.info(f"Cache file not found: {cache_file}")
            return []

        try:
            with open(cache_file) as f:
                models = json.load(f)
            logger.info(f"Loaded {len(models)} models from cache")
            return models
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
            return []

