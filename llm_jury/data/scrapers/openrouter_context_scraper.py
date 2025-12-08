"""
Scraper for OpenRouter model context window information.

Collects:
- Context window sizes (max tokens)
- Model identifiers for matching
- Additional model metadata (pricing, capabilities)

Uses the public OpenRouter API: https://openrouter.ai/api/v1/models
"""

import json
import re
from typing import Dict, List, Optional, Any
import logging

from .base_scraper import BaseScraper

logger = logging.getLogger(__name__)


class OpenRouterContextScraper(BaseScraper):
    """Scrape OpenRouter for model context window information."""
    
    API_URL = "https://openrouter.ai/api/v1/models"
    
    # Manual mappings for models with non-obvious name matches
    # Maps (creator_slug, cache_slug) -> openrouter_id
    SLUG_TO_OPENROUTER = {
        ('deepseek', 'deepseek-v3-1-reasoning'): 'deepseek/deepseek-chat-v3.1',
        ('alibaba', 'qwen3-4b-2507-instruct-reasoning'): 'qwen/qwen3-4b:free',
    }
    
    def __init__(self, rate_limit_delay: float = 1.0):
        """
        Initialize scraper.
        
        Args:
            rate_limit_delay: Seconds between requests
        """
        super().__init__(rate_limit_delay)
    
    def get_source_name(self) -> str:
        return "OpenRouter"
    
    def scrape(self) -> List[Dict]:
        """
        Scrape OpenRouter API for model context window information.
        
        Returns:
            List of dicts with model_id, name, context_length, and other metadata
        """
        logger.info(f"Scraping {self.get_source_name()} for context window data...")
        
        response = self._make_request(self.API_URL)
        if not response:
            logger.warning(f"{self.get_source_name()} unavailable")
            return []
        
        try:
            data = response.json()
            models = data.get('data', [])
            
            if not models:
                logger.warning("No models found in OpenRouter response")
                return []
            
            results = []
            for model in models:
                normalized = self._normalize_model_data(model)
                if normalized:
                    results.append(normalized)
            
            logger.info(f"Scraped context window data for {len(results)} models")
            return results
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse OpenRouter response: {e}")
            return []
        except Exception as e:
            logger.error(f"Scraping failed: {e}")
            return []
    
    def _normalize_model_data(self, model: Dict[str, Any]) -> Optional[Dict]:
        """
        Normalize OpenRouter model data into standard format.
        
        Args:
            model: Raw model data from OpenRouter API
            
        Returns:
            Normalized dict with context window info, or None if invalid
        """
        model_id = model.get('id')
        if not model_id:
            return None
        
        context_length = model.get('context_length')
        
        # Parse pricing if available
        pricing = model.get('pricing', {})
        input_price = self._parse_price(pricing.get('prompt'))
        output_price = self._parse_price(pricing.get('completion'))
        
        # Extract provider from model ID (e.g., "anthropic/claude-3-opus" -> "anthropic")
        provider = model_id.split('/')[0] if '/' in model_id else None
        
        # Get model name (short part after provider)
        model_name = model_id.split('/')[-1] if '/' in model_id else model_id
        
        return {
            'openrouter_id': model_id,
            'name': model.get('name', model_name),
            'model_name': model_name,
            'provider': provider,
            'context_length': context_length,
            'context_window_k': context_length // 1000 if context_length else None,
            'input_price_per_token': input_price,
            'output_price_per_token': output_price,
            'input_cost_per_m': input_price * 1_000_000 if input_price else None,
            'output_cost_per_m': output_price * 1_000_000 if output_price else None,
            'top_provider': model.get('top_provider'),
            'architecture': model.get('architecture', {}),
            'source': self.get_source_name(),
        }
    
    def _parse_price(self, price_str: Optional[str]) -> Optional[float]:
        """
        Parse price string to float.
        
        Args:
            price_str: Price as string (e.g., "0.00001")
            
        Returns:
            Float price per token, or None if invalid
        """
        if price_str is None:
            return None
        
        try:
            return float(price_str)
        except (ValueError, TypeError):
            return None
    
    def get_context_length_map(self) -> Dict[str, int]:
        """
        Get a mapping of model IDs to context lengths.
        
        Returns:
            Dict mapping openrouter_id -> context_length
        """
        models = self.scrape()
        return {
            m['openrouter_id']: m['context_length']
            for m in models
            if m.get('context_length')
        }
    
    def match_to_cache(
        self,
        cache_models: List[Dict],
        match_fields: List[str] = None
    ) -> Dict[str, Dict]:
        """
        Match scraped OpenRouter data to models in the cache.
        
        Args:
            cache_models: List of model dicts from the cache
            match_fields: Fields to use for matching (default: slug, name)
            
        Returns:
            Dict mapping cache model identifier to OpenRouter data
        """
        if match_fields is None:
            match_fields = ['slug', 'name', 'display_name']
        
        openrouter_models = self.scrape()
        
        # Build lookup indices
        by_id = {m['openrouter_id']: m for m in openrouter_models}
        by_name_lower = {m['name'].lower(): m for m in openrouter_models}
        by_model_name_lower = {m['model_name'].lower(): m for m in openrouter_models}
        
        matches = {}
        
        for cache_model in cache_models:
            # Try different matching strategies
            matched = None
            cache_id = cache_model.get('id') or cache_model.get('aa_id')
            slug = cache_model.get('slug', '')
            creator_slug = cache_model.get('creator_slug', '')
            
            # Try custom mapping first (for known edge cases)
            custom_key = (creator_slug, slug)
            if custom_key in self.SLUG_TO_OPENROUTER:
                custom_or_id = self.SLUG_TO_OPENROUTER[custom_key]
                if custom_or_id in by_id:
                    matched = by_id[custom_or_id]
            
            # Try direct OpenRouter ID match
            if not matched:
                openrouter_id = cache_model.get('openrouter_id')
                if openrouter_id and openrouter_id in by_id:
                    matched = by_id[openrouter_id]
            
            # Try slug-based match
            if not matched:
                if slug and creator_slug:
                    # Construct potential OpenRouter ID
                    potential_id = f"{creator_slug}/{slug}"
                    if potential_id in by_id:
                        matched = by_id[potential_id]
            
            # Try name matching
            if not matched:
                for field in match_fields:
                    value = cache_model.get(field, '')
                    if value:
                        value_lower = value.lower()
                        if value_lower in by_name_lower:
                            matched = by_name_lower[value_lower]
                            break
                        if value_lower in by_model_name_lower:
                            matched = by_model_name_lower[value_lower]
                            break
            
            # Try fuzzy matching on name
            if not matched:
                matched = self._fuzzy_match(cache_model, openrouter_models)
            
            if matched and cache_id:
                matches[cache_id] = matched
        
        logger.info(f"Matched {len(matches)}/{len(cache_models)} cache models to OpenRouter")
        return matches
    
    def _fuzzy_match(
        self,
        cache_model: Dict,
        openrouter_models: List[Dict]
    ) -> Optional[Dict]:
        """
        Fuzzy match a cache model to OpenRouter models.
        
        Args:
            cache_model: Model from cache
            openrouter_models: List of OpenRouter models
            
        Returns:
            Best matching OpenRouter model, or None
        """
        cache_name = cache_model.get('name', '').lower()
        cache_slug = cache_model.get('slug', '').lower()
        
        if not cache_name and not cache_slug:
            return None
        
        # Normalize for comparison
        cache_normalized = self._normalize_for_matching(cache_name or cache_slug)
        
        best_match = None
        best_score = 0
        
        for or_model in openrouter_models:
            or_name = or_model.get('name', '').lower()
            or_id = or_model.get('openrouter_id', '').lower()
            
            or_normalized = self._normalize_for_matching(or_name)
            
            # Calculate similarity score
            score = self._similarity_score(cache_normalized, or_normalized)
            
            # Also check against the model ID
            or_id_normalized = self._normalize_for_matching(or_id.split('/')[-1] if '/' in or_id else or_id)
            id_score = self._similarity_score(cache_normalized, or_id_normalized)
            score = max(score, id_score)
            
            if score > best_score and score > 0.7:  # Threshold for fuzzy matching
                best_score = score
                best_match = or_model
        
        return best_match
    
    def _normalize_for_matching(self, name: str) -> str:
        """Normalize name for fuzzy matching."""
        # Remove common suffixes and prefixes
        name = name.lower()
        name = re.sub(r'[^a-z0-9]', '', name)
        return name
    
    def _similarity_score(self, s1: str, s2: str) -> float:
        """Calculate similarity score between two strings."""
        if not s1 or not s2:
            return 0.0
        
        # Simple containment check
        if s1 in s2 or s2 in s1:
            return 0.8 + 0.2 * (min(len(s1), len(s2)) / max(len(s1), len(s2)))
        
        # Character overlap
        set1, set2 = set(s1), set(s2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0


def update_cache_with_context_lengths(
    cache_path: str,
    output_path: Optional[str] = None,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Update the model cache with context window information from OpenRouter.
    
    Args:
        cache_path: Path to the models_cache.json file
        output_path: Path to write updated cache (defaults to cache_path)
        dry_run: If True, don't write changes, just return stats
        
    Returns:
        Dict with update statistics
    """
    import json
    from pathlib import Path
    
    output_path = output_path or cache_path
    
    # Load existing cache
    with open(cache_path, 'r') as f:
        cache_models = json.load(f)
    
    logger.info(f"Loaded {len(cache_models)} models from cache")
    
    # Scrape OpenRouter
    scraper = OpenRouterContextScraper()
    matches = scraper.match_to_cache(cache_models)
    
    # Update cache models
    updated_count = 0
    already_had_context = 0
    no_match_count = 0
    
    for model in cache_models:
        model_id = model.get('id') or model.get('aa_id')
        
        if model_id and model_id in matches:
            or_data = matches[model_id]
            
            # Update context length if not already set or if OpenRouter has it
            if or_data.get('context_length'):
                old_context = model.get('context_length') or model.get('context_window_k')
                if old_context:
                    already_had_context += 1
                
                model['context_length'] = or_data['context_length']
                model['context_window_k'] = or_data['context_window_k']
                model['openrouter_id'] = or_data['openrouter_id']
                updated_count += 1
        else:
            no_match_count += 1
    
    stats = {
        'total_cache_models': len(cache_models),
        'matched_to_openrouter': len(matches),
        'updated_with_context': updated_count,
        'already_had_context': already_had_context,
        'no_match': no_match_count,
    }
    
    logger.info(f"Update stats: {stats}")
    
    if not dry_run:
        with open(output_path, 'w') as f:
            json.dump(cache_models, f, indent=2)
        logger.info(f"Wrote updated cache to {output_path}")
    
    return stats


if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    # Simple CLI for testing
    if len(sys.argv) > 1:
        cache_path = sys.argv[1]
        dry_run = '--dry-run' in sys.argv
        
        stats = update_cache_with_context_lengths(cache_path, dry_run=dry_run)
        print(f"\nUpdate Statistics:")
        for k, v in stats.items():
            print(f"  {k}: {v}")
    else:
        # Just scrape and print sample
        scraper = OpenRouterContextScraper()
        models = scraper.scrape()
        
        print(f"\nScraped {len(models)} models from OpenRouter")
        print("\nSample models with context lengths:")
        for model in models[:10]:
            print(f"  {model['openrouter_id']}: {model['context_length']:,} tokens")

