"""
Benchmark estimator for models without direct benchmark data.

Uses similar models (same family, similar size) to estimate benchmark scores.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd

logger = logging.getLogger(__name__)


class BenchmarkEstimator:
    """Estimate benchmarks for models without direct data using similar models."""
    
    def __init__(self, reference_models: List[Dict]):
        """
        Initialize estimator with reference models that have benchmark data.
        
        Args:
            reference_models: List of dicts with model_name and benchmark scores
        """
        self.reference_models = reference_models
        self._build_family_index()
    
    def _build_family_index(self):
        """Build index of models by family and size."""
        self.family_index = {}
        
        for model in self.reference_models:
            name = model.get('model_name', '')
            family = self._extract_family(name)
            size = self._extract_size(name)
            
            if family and size:
                key = f"{family}_{size}b"
                if key not in self.family_index:
                    self.family_index[key] = []
                self.family_index[key].append(model)
        
        logger.info(f"  Built benchmark index with {len(self.family_index)} family/size combinations")
    
    def _extract_family(self, name: str) -> Optional[str]:
        """Extract model family (llama, mistral, qwen, etc.)."""
        name_lower = name.lower()
        
        families = [
            'llama', 'mistral', 'mixtral', 'qwen', 'yi', 'gemma', 
            'phi', 'deepseek', 'falcon', 'mpt', 'stablelm', 'solar',
            'openchat', 'starling', 'zephyr', 'nous', 'hermes'
        ]
        
        for family in families:
            if family in name_lower:
                return family
        
        return None
    
    def _extract_size(self, name: str) -> Optional[int]:
        """Extract model size in billions of parameters."""
        # Common patterns: 7b, 70b, 7B, 70B, 7-b, 7_b
        match = re.search(r'(\d+)[-_]?b(?:illion)?', name.lower())
        if match:
            return int(match.group(1))
        
        return None
    
    def estimate_benchmarks(self, target_model_name: str) -> Optional[Dict]:
        """
        Estimate benchmarks for a target model based on similar models.
        
        Args:
            target_model_name: Name of model to estimate benchmarks for
            
        Returns:
            Dict with estimated benchmark scores, or None if no similar models found
        """
        family = self._extract_family(target_model_name)
        size = self._extract_size(target_model_name)
        
        if not family or not size:
            return None
        
        # Try exact match first (same family and size)
        exact_key = f"{family}_{size}b"
        if exact_key in self.family_index:
            similar_models = self.family_index[exact_key]
            return self._average_benchmarks(similar_models, target_model_name, match_type='exact')
        
        # Try close size match (within 2x)
        close_matches = []
        for key, models in self.family_index.items():
            key_family, key_size = key.rsplit('_', 1)
            key_size_num = int(key_size.rstrip('b'))
            
            if key_family == family and 0.5 * size <= key_size_num <= 2 * size:
                close_matches.extend(models)
        
        if close_matches:
            return self._average_benchmarks(close_matches, target_model_name, match_type='close')
        
        return None
    
    def _average_benchmarks(self, models: List[Dict], target_name: str, match_type: str) -> Dict:
        """Average benchmark scores from similar models."""
        benchmark_fields = ['mmlu_score', 'gpqa_score', 'math_score', 'ifeval_score']
        
        averages = {}
        for field in benchmark_fields:
            scores = [m.get(field, 0) for m in models if m.get(field, 0) > 0]
            if scores:
                averages[field] = sum(scores) / len(scores)
            else:
                averages[field] = 0
        
        # Add metadata
        averages['estimated'] = True
        averages['estimation_method'] = match_type
        averages['reference_count'] = len(models)
        averages['reference_models'] = [m.get('model_name', '') for m in models[:3]]
        
        logger.debug(f"    Estimated benchmarks for {target_name}: {match_type} match using {len(models)} references")
        
        return averages
    
    def enrich_models(self, target_models: List[Dict]) -> List[Dict]:
        """
        Enrich target models with estimated benchmarks.
        
        Args:
            target_models: List of models to enrich
            
        Returns:
            List of enriched models with estimated benchmarks
        """
        enriched = []
        estimated_count = 0
        
        for model in target_models:
            name = model.get('model_name', '')
            has_benchmarks = model.get('mmlu_score', 0) > 0
            
            # Make sure to explicitly mark models without estimates
            model_copy = model.copy()
            
            if not has_benchmarks:
                # Try to estimate
                estimated = self.estimate_benchmarks(name)
                if estimated:
                    # Only update benchmark fields
                    model_copy['mmlu_score'] = estimated.get('mmlu_score', 0)
                    model_copy['gpqa_score'] = estimated.get('gpqa_score', 0)
                    model_copy['math_score'] = estimated.get('math_score', 0)
                    model_copy['ifeval_score'] = estimated.get('ifeval_score', 0)
                    model_copy['is_estimated'] = True
                    model_copy['estimation_method'] = estimated.get('estimation_method', '')
                    model_copy['reference_count'] = estimated.get('reference_count', 0)
                    estimated_count += 1
                else:
                    model_copy['is_estimated'] = False
            else:
                model_copy['is_estimated'] = False
            
            enriched.append(model_copy)
        
        logger.info(f"  Enriched {estimated_count} models with estimated benchmarks")
        
        return enriched

