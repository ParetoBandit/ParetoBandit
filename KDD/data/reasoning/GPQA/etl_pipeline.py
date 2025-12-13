"""Main ETL pipeline for LLM Jury."""

import logging
from typing import Optional, List
from pathlib import Path

from llm_jury.config import get_config
from llm_jury.etl.artificial_analysis_client import ArtificialAnalysisClient
from llm_jury.etl.hallucination_leaderboard_client import HallucinationLeaderboardClient
from llm_jury.etl.openrouter_ttft_client import OpenRouterTTFTClient
from llm_jury.etl.data_merger import DataMerger

logger = logging.getLogger(__name__)


class ETLPipeline:
    """Main ETL pipeline for fetching and evaluating models."""

    def __init__(
        self,
        artificial_analysis_api_key: Optional[str] = None,
        openrouter_api_key: Optional[str] = None,
        output_file: Optional[Path] = None,
    ):
        """Initialize ETL pipeline.

        Args:
            artificial_analysis_api_key: Artificial Analysis API key (uses config if not provided)
            openrouter_api_key: OpenRouter API key for TTFT measurement (uses env if not provided)
            output_file: Output cache file path (uses config if not provided)
        """
        config = get_config()

        # Get API key from parameter or config
        self.api_key = artificial_analysis_api_key or config.artificial_analysis_api_key
        if not self.api_key:
            raise ValueError(
                "Artificial Analysis API key not provided. "
                "Set it with: export ARTIFICIAL_ANALYSIS_API_KEY='your-key' or add to .env file"
            )

        # Initialize components
        self.client = ArtificialAnalysisClient(self.api_key)
        self.hallucination_client = HallucinationLeaderboardClient()
        self.merger = DataMerger()
        
        # TTFT client (optional - uses OPENROUTER_API_KEY from env if not provided)
        try:
            self.ttft_client = OpenRouterTTFTClient(openrouter_api_key)
        except ValueError:
            logger.warning("OpenRouter API key not found - TTFT measurement will be skipped")
            self.ttft_client = None

        # Output file
        self.output_file = output_file or config.cache_file

    def run(
        self, 
        model_filter: Optional[List[str]] = None,
        require_complete_benchmarks: bool = False,
        include_hallucination_data: bool = True,
        include_ttft_data: bool = True,
        require_ttft: bool = True,
        ttft_samples: int = 2,
    ) -> Path:
        """Run the complete ETL pipeline using Artificial Analysis API.

        Args:
            model_filter: Optional list of model names to filter (all if None)
            require_complete_benchmarks: If True, only include models with all composite indices
                                        (intelligence_index, coding_index, math_index)
            include_hallucination_data: If True, fetch and merge hallucination rates from Vectara
            include_ttft_data: If True, measure TTFT via OpenRouter for each model
            require_ttft: If True, only include models with successful TTFT measurement
            ttft_samples: Number of TTFT samples to average per model

        Returns:
            Path to output cache file
        """
        logger.info("Starting ETL pipeline (Artificial Analysis API)")

        # Step 1: Fetch models from Artificial Analysis
        logger.info("Step 1: Fetching models with benchmarks from Artificial Analysis API")
        aa_models = self.client.get_llm_models()
        
        logger.info(f"Fetched {len(aa_models)} models from Artificial Analysis")

        # Filter by name if specified
        if model_filter:
            aa_models = [
                m for m in aa_models
                if m.get("name") in model_filter or m.get("slug") in model_filter
            ]
            logger.info(f"Filtered to {len(aa_models)} models by name")

        # Filter for complete benchmarks if requested
        if require_complete_benchmarks:
            complete_models = [
                m for m in aa_models
                if all([
                    m.get("intelligence_index") is not None,
                    m.get("coding_index") is not None,
                    m.get("math_index") is not None
                ])
            ]
            filtered_count = len(aa_models) - len(complete_models)
            aa_models = complete_models
            logger.info(f"Filtered to {len(aa_models)} models with complete benchmarks")
            logger.info(f"(Excluded {filtered_count} models with incomplete data)")

        # Step 1b: Fetch hallucination data from Vectara
        hallucination_data = []
        if include_hallucination_data:
            logger.info("Step 1b: Fetching hallucination rates from Vectara leaderboard")
            try:
                hallucination_data = self.hallucination_client.fetch_leaderboard()
                logger.info(f"Fetched hallucination data for {len(hallucination_data)} models")
            except Exception as e:
                logger.warning(f"Failed to fetch hallucination data: {e}")
                hallucination_data = []

        # Step 2: Load existing cache
        logger.info("Step 2: Loading existing cache")
        existing_cache = self.merger.load_cache(self.output_file)

        # Step 3: Merge with existing cache
        logger.info("Step 3: Merging data")
        merged_models = self.merger.merge_aa_data(
            aa_models,
            existing_cache,
        )

        # Step 3b: Merge hallucination data
        if hallucination_data:
            logger.info("Step 3b: Merging hallucination data")
            merged_models = self.merger.merge_hallucination_data(
                merged_models,
                hallucination_data
            )

        # Step 3c: Measure TTFT via OpenRouter
        if include_ttft_data and self.ttft_client:
            logger.info("Step 3c: Measuring TTFT via OpenRouter")
            ttft_results = self.ttft_client.measure_all_models(
                merged_models,
                num_samples=ttft_samples,
                delay_between_models=0.5
            )
            
            # Add TTFT to models
            for model in merged_models:
                name = model.get('name', '')
                if name in ttft_results:
                    model['measured_ttft_seconds'] = round(ttft_results[name], 4)
            
            ttft_count = sum(1 for m in merged_models if m.get('measured_ttft_seconds') is not None)
            logger.info(f"Measured TTFT for {ttft_count}/{len(merged_models)} models")
            
            # Filter to only models with TTFT if required
            if require_ttft:
                before_count = len(merged_models)
                merged_models = [m for m in merged_models if m.get('measured_ttft_seconds') is not None]
                filtered_count = before_count - len(merged_models)
                if filtered_count > 0:
                    logger.info(f"Filtered out {filtered_count} models without TTFT data")

        # Step 4: Save cache
        logger.info("Step 4: Saving cache")
        self.merger.save_cache(merged_models, self.output_file)

        logger.info(f"ETL pipeline complete! Output: {self.output_file}")
        logger.info(f"Total models: {len(merged_models)}")
        if require_complete_benchmarks:
            logger.info(f"All models have COMPLETE benchmark data (intelligence + coding + math) ✓")
        else:
            logger.info(f"All models have benchmark data ✓")
        if hallucination_data:
            logger.info(f"Hallucination data merged for matching models ✓")
        if include_ttft_data and self.ttft_client:
            logger.info(f"TTFT data measured via OpenRouter ✓")
            if require_ttft:
                logger.info(f"All models have TTFT data ✓")

        return self.output_file

    def update_cache(self) -> Path:
        """Update the model cache with latest data from Artificial Analysis.

        Returns:
            Path to updated cache file
        """
        logger.info("Updating cache from Artificial Analysis")
        return self.run()

