"""Client for Artificial Analysis API.

Uses the official free API: https://artificialanalysis.ai/documentation
"""

import requests
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class ArtificialAnalysisClient:
    """Client for Artificial Analysis free API."""

    BASE_URL = "https://artificialanalysis.ai/api/v2"
    LLM_ENDPOINT = f"{BASE_URL}/data/llms/models"

    def __init__(self, api_key: str):
        """Initialize client with API key.

        Args:
            api_key: Artificial Analysis API key
        """
        self.api_key = api_key
        self.headers = {
            "x-api-key": api_key,
            "Accept": "application/json"
        }

    def get_llm_models(self) -> List[Dict]:
        """Fetch LLM model data from Artificial Analysis API.

        Returns:
            List of model dictionaries with benchmarks, pricing, and speed metrics

        Raises:
            requests.exceptions.RequestException: If API request fails
        """
        logger.info(f"Fetching LLM data from Artificial Analysis API")

        try:
            response = requests.get(
                self.LLM_ENDPOINT,
                headers=self.headers,
                timeout=30
            )
            response.raise_for_status()

            data = response.json()

            if data.get("status") != 200:
                logger.warning(f"API returned non-200 status: {data.get('status')}")
                return []

            models = data.get("data", [])
            logger.info(f"Successfully fetched {len(models)} models from Artificial Analysis")

            # Normalize the data
            normalized_models = [self._normalize_model(m) for m in models]

            return normalized_models

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                logger.error("Invalid or missing Artificial Analysis API key")
                raise ValueError("Invalid Artificial Analysis API key")
            elif e.response.status_code == 429:
                logger.error("Rate limit exceeded (1000 requests/day)")
                raise ValueError("Rate limit exceeded. Try again later.")
            else:
                logger.error(f"HTTP error fetching from Artificial Analysis: {e}")
                raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch from Artificial Analysis API: {e}")
            raise

    def _normalize_model(self, raw_model: Dict) -> Dict:
        """Normalize API response model data to internal format.

        Args:
            raw_model: Raw model data from API

        Returns:
            Normalized model dictionary
        """
        evaluations = raw_model.get("evaluations", {})
        pricing = raw_model.get("pricing", {})
        model_creator = raw_model.get("model_creator", {})

        return {
            # Identification
            "aa_id": raw_model.get("id"),
            "name": raw_model.get("name"),
            "slug": raw_model.get("slug"),
            
            # Creator info
            "creator_name": model_creator.get("name"),
            "creator_slug": model_creator.get("slug"),
            
            # Quality indices (Artificial Analysis proprietary)
            "intelligence_index": evaluations.get("artificial_analysis_intelligence_index"),
            "coding_index": evaluations.get("artificial_analysis_coding_index"),
            "math_index": evaluations.get("artificial_analysis_math_index"),
            
            # Standard benchmarks
            "mmlu_pro": evaluations.get("mmlu_pro"),
            "gpqa": evaluations.get("gpqa"),
            "hle": evaluations.get("hle"),
            "livecodebench": evaluations.get("livecodebench"),
            "scicode": evaluations.get("scicode"),
            "math_500": evaluations.get("math_500"),
            "aime": evaluations.get("aime"),
            
            # Pricing (per 1M tokens in USD)
            "price_1m_input": pricing.get("price_1m_input_tokens"),
            "price_1m_output": pricing.get("price_1m_output_tokens"),
            "price_1m_blended": pricing.get("price_1m_blended_3_to_1"),
            
            # Speed metrics
            "output_tokens_per_second": raw_model.get("median_output_tokens_per_second"),
            "time_to_first_token_seconds": raw_model.get("median_time_to_first_token_seconds"),
            
            # Raw data for reference
            "raw_data": raw_model,
            
            # Source
            "source": "artificial_analysis_api"
        }

    def get_model_by_name(self, model_name: str) -> Optional[Dict]:
        """Get a specific model by name.

        Args:
            model_name: Model name to search for

        Returns:
            Model dictionary or None if not found
        """
        models = self.get_llm_models()
        
        model_name_lower = model_name.lower()
        
        # Try exact match first
        for model in models:
            if model.get("name", "").lower() == model_name_lower:
                return model
        
        # Try slug match
        for model in models:
            if model.get("slug", "").lower() == model_name_lower:
                return model
        
        # Try partial match
        for model in models:
            name = model.get("name", "").lower()
            if model_name_lower in name or name in model_name_lower:
                return model
        
        return None

