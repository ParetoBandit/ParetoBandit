"""
Multi-source scrapers for comprehensive LLM benchmark data collection.

This package provides scrapers for:
- LMSYS Chatbot Arena (Elo ratings, MT-Bench)
- Artificial Analysis (latency, throughput, pricing)
- OpenRouter API + Web (pricing, availability, specs, canonical names, context windows)
- HuggingFace Hub (open source models, benchmarks)
- Official sources (model cards, technical reports)
"""

from .chatbot_arena_scraper import ChatbotArenaScraper
from .artificial_analysis_scraper import ArtificialAnalysisScraper
from .openrouter_scraper import OpenRouterScraper, OpenRouterWebScraper
from .official_sources_scraper import OfficialSourcesScraper
from .huggingface_scraper import HuggingFaceLeaderboardScraper
from .aggregate_scraper import ComprehensiveBenchmarkAggregator
from .openrouter_context_scraper import OpenRouterContextScraper, update_cache_with_context_lengths

__all__ = [
    'ChatbotArenaScraper',
    'ArtificialAnalysisScraper',
    'OpenRouterScraper',
    'OpenRouterWebScraper',
    'OfficialSourcesScraper',
    'HuggingFaceLeaderboardScraper',
    'ComprehensiveBenchmarkAggregator',
    'OpenRouterContextScraper',
    'update_cache_with_context_lengths',
]

