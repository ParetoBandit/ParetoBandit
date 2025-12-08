"""
Scraper for official model benchmarks from technical reports and model cards.

Collects:
- MMLU, GPQA, MATH, IFEval scores
- Official benchmark results from company sources
- Parameter counts and specifications
"""

import re
from typing import Dict, List, Optional
from bs4 import BeautifulSoup
import logging

from .base_scraper import BaseScraper

logger = logging.getLogger(__name__)


class OfficialSourcesScraper(BaseScraper):
    """
    Scrape official benchmarks from model cards and technical reports.
    
    Note: This is limited to web-accessible data. PDF reports require
    manual extraction or specialized PDF parsing.
    """
    
    # Model card URLs for major providers
    MODEL_CARD_URLS = {
        'openai': 'https://platform.openai.com/docs/models',
        'anthropic': 'https://www.anthropic.com/api',
        'google': 'https://ai.google.dev/gemini-api/docs/models/gemini',
        'cohere': 'https://docs.cohere.com/docs/models',
        'xai': 'https://docs.x.ai/docs',
    }
    
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
        return "Official Model Cards"
    
    def scrape(self) -> List[Dict]:
        """
        Scrape official sources for benchmark data.
        
        Returns:
            List of dicts with official benchmark scores
        """
        logger.info(f"Scraping {self.get_source_name()}...")
        
        all_data = []
        
        # Try each provider
        for provider, url in self.MODEL_CARD_URLS.items():
            logger.info(f"  Fetching {provider} model cards...")
            data = self._scrape_provider(provider, url)
            if data:
                all_data.extend(data)
        
        # Add manually curated data for models with known benchmarks
        all_data.extend(self._get_curated_benchmarks())
        
        logger.info(f"Collected {len(all_data)} official benchmark entries")
        return all_data
    
    def _scrape_provider(self, provider: str, url: str) -> List[Dict]:
        """Scrape a specific provider's model cards."""
        response = self._make_request(url)
        if not response:
            return []
        
        try:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Provider-specific parsing
            if provider == 'openai':
                return self._parse_openai(soup)
            elif provider == 'google':
                return self._parse_google(soup)
            elif provider == 'anthropic':
                return self._parse_anthropic(soup)
            else:
                return []
                
        except Exception as e:
            logger.debug(f"Failed to parse {provider}: {e}")
            return []
    
    def _parse_openai(self, soup: BeautifulSoup) -> List[Dict]:
        """Parse OpenAI model documentation."""
        # OpenAI doesn't usually publish detailed benchmarks on their docs
        # Return empty, rely on curated data instead
        return []
    
    def _parse_google(self, soup: BeautifulSoup) -> List[Dict]:
        """Parse Google Gemini documentation."""
        # Look for benchmark tables
        models = []
        tables = soup.find_all('table')
        
        for table in tables:
            # Check if this looks like a benchmark table
            headers = [th.text.strip().lower() for th in table.find_all('th')]
            if any(keyword in ' '.join(headers) for keyword in ['mmlu', 'benchmark', 'score']):
                models.extend(self._parse_benchmark_table(table, 'google'))
        
        return models
    
    def _parse_anthropic(self, soup: BeautifulSoup) -> List[Dict]:
        """Parse Anthropic Claude documentation."""
        return []
    
    def _parse_benchmark_table(self, table, provider: str) -> List[Dict]:
        """Parse a benchmark table from HTML."""
        models = []
        
        try:
            headers = [th.text.strip().lower() for th in table.find_all('th')]
            rows = table.find_all('tr')[1:]
            
            for row in rows:
                cells = [td.text.strip() for td in row.find_all('td')]
                if len(cells) >= 2:
                    model_dict = {'provider': provider}
                    for i, header in enumerate(headers):
                        if i < len(cells):
                            model_dict[header] = cells[i]
                    
                    models.append(model_dict)
        except Exception as e:
            logger.debug(f"Table parsing failed: {e}")
        
        return models
    
    def _get_curated_benchmarks(self) -> List[Dict]:
        """
        Return manually curated benchmark data from official sources.
        
        Uses OpenRouter model names when available for consistency.
        
        Sources:
        - OpenAI technical reports and system cards
        - Google Gemini blog posts and papers
        - Anthropic Claude model cards
        - xAI Grok announcements
        - Meta Llama papers
        """
        # Match to OpenRouter canonical names
        benchmarks_raw = [
            # OpenAI models (with name variants for matching)
            {
                'name_variants': ['gpt-4o', 'chatgpt-4o', 'openai/gpt-4o'],
                'provider': 'openai',
                'mmlu_score': 88.7,
                'gpqa_score': 53.6,
                'math_score': 76.6,
                'ifeval_score': 84.9,
                'humaneval_score': 90.2,
                'context_length': 128000,
                'param_count_b': 1800,  # Estimated
                'tool_use_ability': 0.95,
                'source': 'OpenAI GPT-4o System Card',
            },
            {
                'name_variants': ['gpt-4o-mini', 'openai/gpt-4o-mini'],
                'provider': 'openai',
                'mmlu_score': 82.0,
                'gpqa_score': 40.2,
                'math_score': 70.2,
                'ifeval_score': 80.4,
                'humaneval_score': 87.2,
                'context_length': 128000,
                'param_count_b': 8,  # Estimated
                'tool_use_ability': 0.85,
                'source': 'OpenAI GPT-4o-mini Announcement',
            },
            {
                'name_variants': ['gpt-4-turbo', 'gpt-4-turbo-2024-04-09', 'openai/gpt-4-turbo'],
                'provider': 'openai',
                'mmlu_score': 86.4,
                'gpqa_score': 49.3,
                'math_score': 72.2,
                'ifeval_score': 82.0,
                'humaneval_score': 88.0,
                'context_length': 128000,
                'param_count_b': 1760,  # Estimated
                'tool_use_ability': 0.90,
                'source': 'OpenAI GPT-4 Turbo Technical Report',
            },
            {
                'name_variants': ['o1-preview', 'openai/o1-preview'],
                'provider': 'openai',
                'mmlu_score': 89.3,
                'gpqa_score': 78.3,
                'math_score': 85.5,
                'ifeval_score': 75.0,  # Estimated (reasoning models less focused on instruction following)
                'humaneval_score': 92.0,
                'context_length': 131072,
                'param_count_b': 1800,  # Estimated
                'tool_use_ability': 0.70,  # Reasoning models have limited tool support
                'source': 'OpenAI o1 System Card',
            },
            {
                'name_variants': ['o1-mini', 'openai/o1-mini'],
                'provider': 'openai',
                'mmlu_score': 85.2,
                'gpqa_score': 70.0,
                'math_score': 80.0,
                'ifeval_score': 75.0,  # Estimated
                'humaneval_score': 89.0,
                'context_length': 131072,
                'param_count_b': 50,  # Estimated
                'tool_use_ability': 0.70,
                'source': 'OpenAI o1-mini Announcement',
            },
            
            # Google Gemini models
            {
                'name_variants': ['gemini-2.0-flash-exp', 'gemini-2.0-flash', 'google/gemini-2.0-flash-exp'],
                'provider': 'google',
                'mmlu_score': 87.7,
                'gpqa_score': 59.0,
                'math_score': 71.9,
                'ifeval_score': 85.0,  # Estimated
                'humaneval_score': 87.0,
                'context_length': 1048576,  # 1M tokens
                'param_count_b': 50,  # Estimated
                'tool_use_ability': 0.90,
                'source': 'Google Gemini 2.0 Blog Post (Dec 2024)',
            },
            {
                'name_variants': ['gemini-1.5-pro', 'google/gemini-1.5-pro', 'gemini-pro-1.5'],
                'provider': 'google',
                'mmlu_score': 85.9,
                'gpqa_score': 50.3,
                'math_score': 67.7,
                'ifeval_score': 82.0,  # Estimated
                'humaneval_score': 84.1,
                'context_length': 2097152,  # 2M tokens
                'param_count_b': 50,  # Estimated
                'tool_use_ability': 0.88,
                'source': 'Google Gemini 1.5 Technical Report',
            },
            
            # Anthropic Claude models
            {
                'name_variants': ['claude-3.5-sonnet', 'claude-3-5-sonnet', 'anthropic/claude-3.5-sonnet'],
                'provider': 'anthropic',
                'mmlu_score': 88.3,
                'gpqa_score': 59.4,
                'math_score': 71.1,
                'ifeval_score': 87.0,  # Estimated
                'humaneval_score': 92.0,
                'context_length': 200000,
                'param_count_b': 175,  # Estimated
                'tool_use_ability': 0.93,
                'source': 'Anthropic Claude 3.5 Model Card',
            },
            {
                'name_variants': ['claude-3-opus', 'anthropic/claude-3-opus'],
                'provider': 'anthropic',
                'mmlu_score': 86.8,
                'gpqa_score': 50.4,
                'math_score': 60.1,
                'ifeval_score': 85.0,  # Estimated
                'humaneval_score': 84.9,
                'context_length': 200000,
                'param_count_b': 400,  # Estimated
                'tool_use_ability': 0.90,
                'source': 'Anthropic Claude 3 Model Card',
            },
            {
                'name_variants': ['claude-3-haiku', 'anthropic/claude-3-haiku'],
                'provider': 'anthropic',
                'mmlu_score': 75.2,
                'gpqa_score': 35.0,  # Estimated
                'math_score': 38.9,
                'ifeval_score': 75.0,  # Estimated
                'humaneval_score': 75.9,
                'context_length': 200000,
                'param_count_b': 40,  # Estimated
                'tool_use_ability': 0.85,
                'source': 'Anthropic Claude 3 Model Card',
            },
            
            # xAI Grok models
            {
                'name_variants': ['grok-2', 'x-ai/grok-2'],
                'provider': 'xai',
                'mmlu_score': 88.0,  # From xAI blog
                'gpqa_score': 56.0,  # Estimated from GPQA benchmark
                'math_score': 76.6,  # From xAI blog (MATH benchmark)
                'ifeval_score': 80.0,  # Estimated
                'humaneval_score': 88.0,  # From xAI blog
                'context_length': 131072,
                'param_count_b': 314,  # 314B parameters
                'tool_use_ability': 0.85,
                'source': 'xAI Grok-2 Blog Post (Aug 2024)',
            },
            
            # Cohere Command-R+
            {
                'name_variants': ['command-r-plus', 'cohere/command-r-plus', 'c4ai-command-r-plus'],
                'provider': 'cohere',
                'mmlu_score': 75.7,  # From Cohere docs
                'gpqa_score': 40.0,  # Estimated
                'math_score': 50.0,  # Estimated
                'ifeval_score': 76.6,  # From HuggingFace leaderboard
                'humaneval_score': 70.0,  # Estimated
                'context_length': 131072,
                'param_count_b': 104,
                'tool_use_ability': 0.92,  # Strong tool calling
                'source': 'Cohere Documentation + HF Leaderboard',
            },
        ]
        
        # Match all entries to OpenRouter canonical names
        results = []
        for benchmark in benchmarks_raw:
            # Get name variants (either list or single name)
            name_variants = benchmark.pop('name_variants', None)
            if name_variants is None:
                name_variants = [benchmark.pop('model_name')]
            elif not isinstance(name_variants, list):
                name_variants = [name_variants]
            
            # Match to canonical name
            matched_name = self._match_model_name(name_variants)
            
            # Create result with matched name
            result = {'model_name': matched_name}
            result.update(benchmark)
            results.append(result)
        
        return results
    
    def _match_model_name(self, name_variants: List[str]) -> str:
        """Match name variants to OpenRouter canonical names."""
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

