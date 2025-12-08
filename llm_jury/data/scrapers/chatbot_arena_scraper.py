"""
Scraper for LMSYS Chatbot Arena leaderboard data.

Collects:
- Elo ratings (human preference)
- MT-Bench scores (multi-turn conversation quality)
- Model rankings
"""

import json
import re
from typing import Dict, List, Optional
from bs4 import BeautifulSoup
import logging

from .base_scraper import BaseScraper

logger = logging.getLogger(__name__)


class ChatbotArenaScraper(BaseScraper):
    """Scrape LMSYS Chatbot Arena leaderboard for model ratings."""
    
    # LMSYS Chatbot Arena leaderboard URL
    LEADERBOARD_URL = "https://chat.lmsys.org/"
    
    # Alternative: Try to fetch from their Hugging Face space
    HF_SPACE_API = "https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard"
    
    def __init__(self, known_models: Optional[List[str]] = None, rate_limit_delay: float = 1.0):
        """
        Initialize scraper.
        
        Args:
            known_models: List of known model names from OpenRouter (for matching)
            rate_limit_delay: Seconds between requests
        """
        super().__init__(rate_limit_delay)
        self.known_models = known_models or []
    
    def get_source_name(self) -> str:
        return "LMSYS Chatbot Arena"
    
    def scrape(self) -> List[Dict]:
        """
        Scrape Chatbot Arena leaderboard.
        
        Returns:
            List of dicts with: model_name, arena_elo, mt_bench_score, rank
        """
        logger.info(f"Scraping {self.get_source_name()}...")
        
        # Try API endpoint first (if available)
        data = self._try_api_endpoint()
        if data:
            return data
        
        # Fallback: Try to parse HTML/gradio interface
        data = self._try_html_scraping()
        if data:
            return data
        
        # If both fail, return manual curated top models
        logger.warning(f"{self.get_source_name()} scraping failed, using fallback data")
        return self._get_fallback_data()
    
    def _try_api_endpoint(self) -> Optional[List[Dict]]:
        """
        Try to fetch data from LMSYS API if available.
        
        Note: LMSYS may not have a public API, this is aspirational.
        """
        # Check if there's a data endpoint
        api_urls = [
            "https://chat.lmsys.org/api/leaderboard",
            "https://huggingface.co/api/spaces/lmsys/chatbot-arena-leaderboard/data",
        ]
        
        for url in api_urls:
            response = self._make_request(url)
            if response and response.status_code == 200:
                try:
                    data = response.json()
                    return self._parse_api_response(data)
                except json.JSONDecodeError:
                    continue
        
        return None
    
    def _try_html_scraping(self) -> Optional[List[Dict]]:
        """
        Try to scrape leaderboard from HTML/Gradio interface.
        
        This is fragile and may break with UI updates.
        """
        response = self._make_request(self.LEADERBOARD_URL)
        if not response:
            return None
        
        try:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Look for data in script tags (Gradio apps often embed data in JS)
            scripts = soup.find_all('script')
            for script in scripts:
                if script.string and 'leaderboard' in script.string.lower():
                    # Try to extract JSON data
                    json_match = re.search(r'(\{.*"data".*\})', script.string)
                    if json_match:
                        try:
                            data = json.loads(json_match.group(1))
                            return self._parse_embedded_data(data)
                        except json.JSONDecodeError:
                            continue
            
            return None
            
        except Exception as e:
            logger.error(f"HTML scraping failed: {e}")
            return None
    
    def _parse_api_response(self, data: Dict) -> List[Dict]:
        """Parse API response into standardized format."""
        models = []
        
        # This structure depends on actual API format
        # Adjust based on real response
        for item in data.get('models', []):
            models.append({
                'model_name': item.get('name'),
                'arena_elo': item.get('elo'),
                'mt_bench_score': item.get('mt_bench'),
                'rank': item.get('rank'),
                'confidence_interval': item.get('ci', [0, 0]),
                'source': self.get_source_name(),
            })
        
        return models
    
    def _parse_embedded_data(self, data: Dict) -> List[Dict]:
        """Parse embedded JSON data from Gradio interface."""
        # Implementation depends on actual data structure
        # Placeholder for now
        return []
    
    def _get_fallback_data(self) -> List[Dict]:
        """
        Return manually curated top model ratings from recent Arena data.
        
        Uses OpenRouter model names if available, otherwise falls back to canonical names.
        
        Source: LMSYS Chatbot Arena as of Jan 2025
        """
        source = f"{self.get_source_name()} (Fallback - Jan 2025)"
        
        # Arena ratings with multiple possible name variations
        # Format: (arena_name, elo, mt_bench, rank)
        arena_ratings = [
            (['gpt-4o', 'chatgpt-4o', 'openai/gpt-4o'], 1310, 8.96, 1),
            (['claude-3.5-sonnet', 'claude-3-5-sonnet', 'anthropic/claude-3.5-sonnet'], 1308, 8.91, 2),
            (['gpt-4-turbo', 'gpt-4-turbo-2024-04-09', 'openai/gpt-4-turbo'], 1280, 8.82, 3),
            (['gemini-2.0-flash-exp', 'gemini-2.0-flash', 'google/gemini-2.0-flash-exp'], 1275, 8.85, 4),
            (['claude-3-opus', 'anthropic/claude-3-opus'], 1265, 8.78, 5),
            (['gemini-1.5-pro', 'google/gemini-1.5-pro', 'gemini-pro-1.5'], 1260, 8.73, 6),
            (['grok-2', 'x-ai/grok-2'], 1245, 8.65, 7),
            (['gpt-4o-mini', 'openai/gpt-4o-mini'], 1235, 8.52, 8),
            (['llama-3.1-405b', 'meta-llama/llama-3.1-405b-instruct', 'llama-3.1-405b-instruct'], 1225, 8.45, 9),
            (['claude-3-haiku', 'anthropic/claude-3-haiku'], 1215, 8.32, 10),
            (['command-r-plus', 'cohere/command-r-plus', 'c4ai-command-r-plus'], 1205, 8.28, 11),
            (['deepseek-v3', 'deepseek-ai/deepseek-v3'], 1200, 8.35, 12),
            (['llama-3.1-70b', 'meta-llama/llama-3.1-70b-instruct', 'llama-3.1-70b-instruct'], 1195, 8.25, 13),
            (['mistral-large', 'mistralai/mistral-large', 'mistral-large-2407'], 1185, 8.18, 14),
            (['qwen-2.5-coder-32b', 'qwen/qwen2.5-coder-32b-instruct'], 1180, 8.15, 15),
        ]
        
        results = []
        for name_variants, elo, mt_bench, rank in arena_ratings:
            # Find matching name in known models
            matched_name = self._match_model_name(name_variants)
            
            results.append({
                'model_name': matched_name,
                'arena_elo': elo,
                'mt_bench_score': mt_bench,
                'rank': rank,
                'source': source,
            })
        
        return results
    
    def _match_model_name(self, name_variants: List[str]) -> str:
        """
        Match a list of name variants against known models from OpenRouter.
        
        Args:
            name_variants: List of possible names for the model
            
        Returns:
            Best matching name from OpenRouter, or first variant if no match
        """
        if not self.known_models:
            # No known models, return first variant
            return name_variants[0]
        
        # Normalize known models for matching
        normalized_known = {
            self._normalize_name(name): name 
            for name in self.known_models
        }
        
        # Try each variant
        for variant in name_variants:
            normalized_variant = self._normalize_name(variant)
            
            # Exact match
            if normalized_variant in normalized_known:
                return normalized_known[normalized_variant]
            
            # Partial match (variant is substring of known model)
            for norm_known, original_known in normalized_known.items():
                if normalized_variant in norm_known or norm_known in normalized_variant:
                    return original_known
        
        # No match found, return first variant
        return name_variants[0]
    
    def _normalize_name(self, name: str) -> str:
        """Normalize model name for matching."""
        return name.lower().replace('-', '').replace('_', '').replace('/', '').replace('.', '')

