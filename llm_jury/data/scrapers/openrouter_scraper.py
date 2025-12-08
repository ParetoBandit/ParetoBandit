"""
Scraper for OpenRouter model data via their public API.

This scraper fetches model information from the OpenRouter API including:
- Model IDs and names
- Pricing information
- Context window sizes
- Model capabilities and metadata

Uses the public OpenRouter API: https://openrouter.ai/api/v1/models
"""

import json
from typing import Dict, List, Optional, Any
import logging

from .base_scraper import BaseScraper

logger = logging.getLogger(__name__)


class OpenRouterScraper(BaseScraper):
    """Scrape OpenRouter API for model data."""
    
    API_URL = "https://openrouter.ai/api/v1/models"
    
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
        Scrape OpenRouter API for model data.
        
        Returns:
            List of dicts with model data including context windows
        """
        logger.info(f"Scraping {self.get_source_name()} API...")
        
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
            
            logger.info(f"Scraped {len(results)} models from OpenRouter")
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
            Normalized dict, or None if invalid
        """
        model_id = model.get('id')
        if not model_id:
            return None
        
        # Parse pricing
        pricing = model.get('pricing', {})
        input_price = self._parse_price(pricing.get('prompt'))
        output_price = self._parse_price(pricing.get('completion'))
        
        # Extract provider from model ID
        provider = model_id.split('/')[0] if '/' in model_id else None
        model_name = model_id.split('/')[-1] if '/' in model_id else model_id
        
        # Get context length
        context_length = model.get('context_length')
        
        return {
            'model_id': model_id,
            'openrouter_id': model_id,
            'model_name': model.get('name', model_name),
            'provider': provider,
            'context_length': context_length,
            'context_window_k': context_length // 1000 if context_length else None,
            'input_price_per_token': input_price,
            'output_price_per_token': output_price,
            'input_cost_per_m': input_price * 1_000_000 if input_price else None,
            'output_cost_per_m': output_price * 1_000_000 if output_price else None,
            'top_provider': model.get('top_provider'),
            'architecture': model.get('architecture', {}),
            'description': model.get('description', ''),
            'source': self.get_source_name(),
        }
    
    def _parse_price(self, price_str: Optional[str]) -> Optional[float]:
        """Parse price string to float."""
        if price_str is None:
            return None
        try:
            return float(price_str)
        except (ValueError, TypeError):
            return None
    
    def get_model_names(self) -> List[str]:
        """Get list of all model IDs/names."""
        models = self.scrape()
        return [m['model_id'] for m in models if m.get('model_id')]
    
    def get_context_lengths(self) -> Dict[str, int]:
        """Get mapping of model IDs to context lengths."""
        models = self.scrape()
        return {
            m['model_id']: m['context_length']
            for m in models
            if m.get('context_length')
        }


# Keep backward compatibility - OpenRouterWebScraper is now just an alias
OpenRouterWebScraper = OpenRouterScraper


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    scraper = OpenRouterScraper()
    models = scraper.scrape()
    
    print(f"\nScraped {len(models)} models from OpenRouter")
    print("\nSample models:")
    for model in models[:10]:
        ctx = model.get('context_length', 'N/A')
        ctx_str = f'{ctx:,}' if isinstance(ctx, int) else str(ctx)
        print(f"  {model['model_id']}: {ctx_str} tokens")

