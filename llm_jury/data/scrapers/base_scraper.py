"""
Base scraper class with common functionality.
"""

import time
import requests
from typing import Dict, List, Optional
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class BaseScraper(ABC):
    """Base class for all scrapers with common utilities."""
    
    def __init__(self, rate_limit_delay: float = 1.0):
        """
        Initialize base scraper.
        
        Args:
            rate_limit_delay: Seconds to wait between requests
        """
        self.rate_limit_delay = rate_limit_delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'LLM-Jury-Benchmark-Collector/1.0 (Educational/Research)'
        })
        self.last_request_time = 0
    
    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()
    
    def _make_request(self, url: str, timeout: int = 15) -> Optional[requests.Response]:
        """
        Make HTTP request with rate limiting and error handling.
        
        Args:
            url: URL to request
            timeout: Request timeout in seconds
            
        Returns:
            Response object or None on error
        """
        self._rate_limit()
        
        try:
            response = self.session.get(url, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {url}: {e}")
            return None
    
    @abstractmethod
    def scrape(self) -> List[Dict]:
        """
        Scrape data from source.
        
        Returns:
            List of model data dictionaries
        """
        pass
    
    @abstractmethod
    def get_source_name(self) -> str:
        """Return the name of this data source."""
        pass

