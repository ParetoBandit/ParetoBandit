"""
Comprehensive benchmark scraper that aggregates data from artificialanalysis.ai

This site provides:
- Quality Index (composite benchmark score)
- MMLU scores
- Arena Elo ratings
- Latency and throughput
- Context length
- Pricing

All in one place, updated regularly.
"""

import json
import re
from typing import Dict, List, Optional
from bs4 import BeautifulSoup
import logging

from .base_scraper import BaseScraper

logger = logging.getLogger(__name__)


class ComprehensiveBenchmarksScraper(BaseScraper):
    """
    Scrape comprehensive benchmarks from artificialanalysis.ai
    
    This aggregator site collects data from multiple sources:
    - LMSYS Arena
    - OpenAI Evals
    - Anthropic Evals  
    - Community benchmarks
    """
    
    # Main leaderboard page
    LEADERBOARD_URL = "https://artificialanalysis.ai/models"
    
    # API endpoint (if available)
    API_URL = "https://artificialanalysis.ai/api/models"
    
    def __init__(self, known_models: Optional[List[str]] = None, rate_limit_delay: float = 2.0):
        """
        Initialize scraper.
        
        Args:
            known_models: List of known model names from OpenRouter (for matching)
            rate_limit_delay: Seconds between requests
        """
        super().__init__(rate_limit_delay)
        self.known_models = known_models or []
    
    def get_source_name(self) -> str:
        return "Artificial Analysis (Comprehensive)"
    
    def scrape(self) -> List[Dict]:
        """
        Scrape comprehensive benchmark data.
        
        Returns:
            List of dicts with model benchmarks
        """
        logger.info(f"Scraping {self.get_source_name()}...")
        
        # Try API first
        data = self._try_api()
        if data:
            logger.info(f"  ✅ Collected {len(data)} models from API")
            return data
        
        # Fallback to HTML scraping
        data = self._scrape_html()
        if data:
            logger.info(f"  ✅ Collected {len(data)} models from HTML")
            return data
        
        logger.warning(f"  ❌ Failed to scrape {self.get_source_name()}")
        return []
    
    def _try_api(self) -> Optional[List[Dict]]:
        """Try to fetch data from API."""
        response = self._make_request(self.API_URL)
        if not response:
            return None
        
        try:
            data = response.json()
            return self._parse_api_data(data)
        except Exception as e:
            logger.debug(f"API parsing failed: {e}")
            return None
    
    def _parse_api_data(self, data: Dict) -> List[Dict]:
        """Parse API response."""
        models = []
        
        # API format may vary - handle different structures
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict) and 'models' in data:
            items = data['models']
        elif isinstance(data, dict) and 'data' in data:
            items = data['data']
        else:
            return []
        
        for item in items:
            try:
                model = self._parse_model_item(item)
                if model:
                    models.append(model)
            except Exception as e:
                logger.debug(f"Failed to parse item: {e}")
                continue
        
        return models
    
    def _parse_model_item(self, item: Dict) -> Optional[Dict]:
        """Parse a single model item from API/HTML data."""
        # Extract model name
        name = item.get('model_name') or item.get('name') or item.get('model')
        if not name:
            return None
        
        # Match to OpenRouter canonical name
        matched_name = self._match_to_openrouter(name)
        
        # Extract benchmarks
        model_data = {
            'model_name': matched_name,
            'original_name': str(name),
            'source': self.get_source_name(),
        }
        
        # Quality metrics
        if 'quality_index' in item:
            model_data['quality_index'] = self._safe_float(item['quality_index'])
        
        if 'mmlu' in item or 'mmlu_score' in item:
            model_data['mmlu_score'] = self._safe_float(item.get('mmlu') or item.get('mmlu_score'))
        
        if 'arena_elo' in item or 'elo' in item:
            model_data['arena_elo'] = self._safe_float(item.get('arena_elo') or item.get('elo'))
        
        if 'mt_bench' in item or 'mt_bench_score' in item:
            model_data['mt_bench_score'] = self._safe_float(item.get('mt_bench') or item.get('mt_bench_score'))
        
        # Performance metrics
        if 'latency' in item or 'latency_ms' in item:
            model_data['latency_ms'] = self._safe_float(item.get('latency') or item.get('latency_ms'))
        
        if 'throughput' in item or 'throughput_tps' in item:
            model_data['throughput_tps'] = self._safe_float(item.get('throughput') or item.get('throughput_tps'))
        
        # Pricing
        if 'input_cost' in item or 'input_price' in item:
            model_data['input_cost_per_m'] = self._safe_float(item.get('input_cost') or item.get('input_price'))
        
        if 'output_cost' in item or 'output_price' in item:
            model_data['output_cost_per_m'] = self._safe_float(item.get('output_cost') or item.get('output_price'))
        
        # Context length
        if 'context_length' in item or 'context' in item:
            model_data['context_length'] = self._safe_int(item.get('context_length') or item.get('context'))
        
        return model_data
    
    def _scrape_html(self) -> List[Dict]:
        """Scrape data from HTML page."""
        response = self._make_request(self.LEADERBOARD_URL)
        if not response:
            return []
        
        try:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Look for data in various formats
            models = []
            
            # Method 1: Look for JSON data in script tags
            script_data = self._extract_json_from_scripts(soup)
            if script_data:
                models.extend(script_data)
            
            # Method 2: Parse HTML tables
            table_data = self._parse_tables(soup)
            if table_data:
                models.extend(table_data)
            
            return models
            
        except Exception as e:
            logger.error(f"HTML parsing failed: {e}")
            return []
    
    def _extract_json_from_scripts(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract JSON data from script tags."""
        models = []
        
        for script in soup.find_all('script'):
            if not script.string:
                continue
            
            # Look for JSON data
            try:
                # Common patterns for embedded data
                patterns = [
                    r'const\s+models\s*=\s*(\[.*?\]);',
                    r'var\s+data\s*=\s*(\{.*?\});',
                    r'window\.__DATA__\s*=\s*(\{.*?\});',
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, script.string, re.DOTALL)
                    if match:
                        json_str = match.group(1)
                        data = json.loads(json_str)
                        
                        if isinstance(data, list):
                            for item in data:
                                model = self._parse_model_item(item)
                                if model:
                                    models.append(model)
                        elif isinstance(data, dict):
                            model = self._parse_model_item(data)
                            if model:
                                models.append(model)
            except Exception as e:
                logger.debug(f"JSON extraction failed: {e}")
                continue
        
        return models
    
    def _parse_tables(self, soup: BeautifulSoup) -> List[Dict]:
        """Parse HTML tables for model data."""
        models = []
        
        for table in soup.find_all('table'):
            try:
                # Get headers
                headers = []
                header_row = table.find('thead') or table.find('tr')
                if header_row:
                    headers = [th.get_text(strip=True).lower() for th in header_row.find_all(['th', 'td'])]
                
                # Get data rows
                rows = table.find_all('tr')[1:] if header_row else table.find_all('tr')
                
                for row in rows:
                    cells = [td.get_text(strip=True) for td in row.find_all(['td', 'th'])]
                    if len(cells) < 2:
                        continue
                    
                    # Map to dict
                    row_dict = {}
                    for i, cell in enumerate(cells):
                        if i < len(headers):
                            row_dict[headers[i]] = cell
                    
                    # Parse as model
                    model = self._parse_model_item(row_dict)
                    if model:
                        models.append(model)
                        
            except Exception as e:
                logger.debug(f"Table parsing failed: {e}")
                continue
        
        return models
    
    def _match_to_openrouter(self, name: str) -> str:
        """Match model name to OpenRouter canonical name."""
        if not self.known_models:
            return self._clean_name(name)
        
        # Try exact match
        if name in self.known_models:
            return name
        
        # Try fuzzy match
        try:
            from fuzzywuzzy import process
            best_match, score = process.extractOne(name, self.known_models)
            if score >= 85:
                return best_match
        except ImportError:
            pass
        
        # Try partial matching
        name_lower = name.lower()
        for known in self.known_models:
            if name_lower in known.lower() or known.lower() in name_lower:
                return known
        
        return self._clean_name(name)
    
    def _clean_name(self, name: str) -> str:
        """Clean model name."""
        # Remove common prefixes/suffixes
        name = re.sub(r'\s+\(.*?\)', '', name)  # Remove parenthetical info
        name = name.strip()
        name = name.lower()
        name = name.replace(' ', '-')
        return name
    
    def _safe_float(self, value) -> float:
        """Safely convert to float."""
        if value is None or value == '':
            return 0.0
        try:
            # Remove any non-numeric characters except . and -
            if isinstance(value, str):
                value = re.sub(r'[^\d.-]', '', value)
            return float(value)
        except (ValueError, TypeError):
            return 0.0
    
    def _safe_int(self, value) -> int:
        """Safely convert to int."""
        try:
            return int(self._safe_float(value))
        except (ValueError, TypeError):
            return 0

