"""
Scraper for Artificial Analysis benchmark data.

Collects:
- Latency (time to first token, tokens per second)
- Throughput metrics
- Quality-price analysis
- API performance data
"""

import json
import re
from typing import Dict, List, Optional
from bs4 import BeautifulSoup
import logging

from .base_scraper import BaseScraper

logger = logging.getLogger(__name__)


class ArtificialAnalysisScraper(BaseScraper):
    """Scrape Artificial Analysis for latency and performance metrics."""
    
    BASE_URL = "https://artificialanalysis.ai"
    LEADERBOARD_URL = f"{BASE_URL}/models"
    
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
        return "Artificial Analysis"
    
    def scrape(self) -> List[Dict]:
        """
        Scrape Artificial Analysis leaderboard.
        
        Returns:
            List of dicts with: model_name, latency_ms, throughput_tps, 
                               quality_index, price_per_token
        """
        logger.info(f"Scraping {self.get_source_name()}...")
        
        response = self._make_request(self.LEADERBOARD_URL)
        if not response:
            logger.warning(f"{self.get_source_name()} unavailable, using fallback")
            return self._get_fallback_data()
        
        try:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Look for data in script tags or tables
            data = self._extract_model_data(soup)
            if data:
                return data
            
            # Fallback to manual data
            return self._get_fallback_data()
            
        except Exception as e:
            logger.error(f"Scraping failed: {e}")
            return self._get_fallback_data()
    
    def _extract_model_data(self, soup: BeautifulSoup) -> Optional[List[Dict]]:
        """Extract model performance data from page."""
        models = []
        
        # Look for embedded JSON data
        scripts = soup.find_all('script')
        for script in scripts:
            if script.string and 'model' in script.string.lower():
                # Try to find JSON data
                json_matches = re.findall(r'\{[^{}]*"model"[^{}]*\}', script.string)
                for match in json_matches:
                    try:
                        data = json.loads(match)
                        if 'model' in data or 'name' in data:
                            models.append(self._normalize_model_data(data))
                    except json.JSONDecodeError:
                        continue
        
        # Also try to find tables
        tables = soup.find_all('table')
        for table in tables:
            table_data = self._parse_table(table)
            if table_data:
                models.extend(table_data)
        
        return models if models else None
    
    def _parse_table(self, table) -> List[Dict]:
        """Parse HTML table for model data."""
        models = []
        
        try:
            headers = [th.text.strip().lower() for th in table.find_all('th')]
            rows = table.find_all('tr')[1:]  # Skip header row
            
            for row in rows:
                cells = [td.text.strip() for td in row.find_all('td')]
                if len(cells) >= 2:
                    model_dict = {}
                    for i, header in enumerate(headers):
                        if i < len(cells):
                            model_dict[header] = cells[i]
                    
                    if model_dict:
                        models.append(self._normalize_model_data(model_dict))
        except Exception as e:
            logger.debug(f"Table parsing failed: {e}")
        
        return models
    
    def _normalize_model_data(self, data: Dict) -> Dict:
        """Normalize data into standard format."""
        return {
            'model_name': data.get('model') or data.get('name'),
            'latency_ms': self._parse_numeric(data.get('latency') or data.get('ttft')),
            'throughput_tps': self._parse_numeric(data.get('throughput') or data.get('tps')),
            'quality_index': self._parse_numeric(data.get('quality')),
            'price_per_mtok': self._parse_numeric(data.get('price')),
            'source': self.get_source_name(),
        }
    
    def _parse_numeric(self, value) -> Optional[float]:
        """Parse numeric value from string."""
        if value is None:
            return None
        
        if isinstance(value, (int, float)):
            return float(value)
        
        # Extract number from string
        match = re.search(r'[\d.]+', str(value))
        if match:
            try:
                return float(match.group())
            except ValueError:
                return None
        
        return None
    
    def _get_fallback_data(self) -> List[Dict]:
        """
        Return manually curated performance data.
        
        Uses OpenRouter model names if available.
        
        Source: Artificial Analysis + various benchmarks (Jan 2025)
        """
        # Performance data with name variants
        # Format: ([name_variants], latency, throughput, quality, price)
        performance_data = [
            (['gpt-4o', 'chatgpt-4o', 'openai/gpt-4o'], 450, 95, 85, 5.0),
            (['gpt-4o-mini', 'openai/gpt-4o-mini'], 280, 125, 75, 0.30),
            (['gpt-4-turbo', 'gpt-4-turbo-2024-04-09', 'openai/gpt-4-turbo'], 550, 85, 83, 15.0),
            (['claude-3.5-sonnet', 'claude-3-5-sonnet', 'anthropic/claude-3.5-sonnet'], 420, 90, 86, 6.0),
            (['claude-3-opus', 'anthropic/claude-3-opus'], 650, 75, 84, 30.0),
            (['claude-3-haiku', 'anthropic/claude-3-haiku'], 320, 110, 72, 0.50),
            (['gemini-2.0-flash-exp', 'gemini-2.0-flash', 'google/gemini-2.0-flash-exp'], 380, 105, 82, 0.20),
            (['gemini-1.5-pro', 'google/gemini-1.5-pro'], 500, 88, 81, 3.50),
            (['llama-3.1-405b', 'meta-llama/llama-3.1-405b-instruct'], 2100, 45, 80, 0.0),
            (['llama-3.1-70b', 'meta-llama/llama-3.1-70b-instruct'], 400, 98, 74, 0.40),
            (['llama-3.1-8b', 'meta-llama/llama-3.1-8b-instruct'], 90, 180, 65, 0.02),
            (['deepseek-v3', 'deepseek-ai/deepseek-v3'], 420, 92, 76, 0.24),
            (['deepseek-r1', 'deepseek-ai/deepseek-r1'], 1200, 55, 78, 0.0),
            (['mistral-large', 'mistralai/mistral-large-2407'], 480, 85, 75, 3.0),
            (['command-r-plus', 'cohere/command-r-plus'], 570, 80, 73, 2.5),
            (['grok-2', 'x-ai/grok-2'], 1620, 60, 79, 5.0),
            (['qwen-2.5-coder-32b', 'qwen/qwen2.5-coder-32b-instruct'], 210, 115, 71, 0.20),
        ]
        
        results = []
        for name_variants, latency, throughput, quality, price in performance_data:
            matched_name = self._match_model_name(name_variants)
            
            results.append({
                'model_name': matched_name,
                'latency_ms': latency,
                'throughput_tps': throughput,
                'quality_index': quality,
                'price_per_mtok': price,
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
            return name_variants[0]
        
        # Normalize known models
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
            
            # Partial match
            for norm_known, original_known in normalized_known.items():
                if normalized_variant in norm_known or norm_known in normalized_variant:
                    return original_known
        
        return name_variants[0]
    
    def _normalize_name(self, name: str) -> str:
        """Normalize model name for matching."""
        return name.lower().replace('-', '').replace('_', '').replace('/', '').replace('.', '')

