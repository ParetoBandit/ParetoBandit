"""Direct OpenAI API Client for Complete Model Data Collection.

This ETL client:
1. Fetches model info from OpenAI API
2. Measures actual TTFT latency via streaming calls
3. Collects pricing from OpenAI documentation
4. Populates all fields needed for HYBRID optimization:
   - Quality: From external benchmarks (Artificial Analysis)
   - Cost: From OpenAI pricing
   - Latency: Measured TTFT via API
   - Hallucination: From Vectara leaderboard
   - Refusal: From model testing

Usage:
    python -m llm_jury.etl.openai_direct_client --models gpt-3.5-turbo gpt-4o --samples 5
"""

import os
import time
import json
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class OpenAIDirectClient:
    """Client for collecting complete model data from OpenAI API."""

    # OpenAI Pricing (as of Dec 2024) - per 1M tokens
    PRICING = {
        "gpt-4o": {
            "input": 2.50,
            "output": 10.00,
            "context_window": 128000,
        },
        "gpt-4o-2024-11-20": {
            "input": 2.50,
            "output": 10.00,
            "context_window": 128000,
        },
        "gpt-4o-mini": {
            "input": 0.15,
            "output": 0.60,
            "context_window": 128000,
        },
        "gpt-4-turbo": {
            "input": 10.00,
            "output": 30.00,
            "context_window": 128000,
        },
        "gpt-4-turbo-2024-04-09": {
            "input": 10.00,
            "output": 30.00,
            "context_window": 128000,
        },
        "gpt-4": {
            "input": 30.00,
            "output": 60.00,
            "context_window": 8192,
        },
        "gpt-3.5-turbo": {
            "input": 0.50,
            "output": 1.50,
            "context_window": 16385,
        },
        "gpt-3.5-turbo-0125": {
            "input": 0.50,
            "output": 1.50,
            "context_window": 16385,
        },
        "gpt-3.5-turbo-1106": {
            "input": 1.00,
            "output": 2.00,
            "context_window": 16385,
        },
        "o1-preview": {
            "input": 15.00,
            "output": 60.00,
            "context_window": 128000,
        },
        "o1-mini": {
            "input": 3.00,
            "output": 12.00,
            "context_window": 128000,
        },
        "o1": {
            "input": 15.00,
            "output": 60.00,
            "context_window": 200000,
        },
    }

    # Map to OpenRouter IDs for consistency with cache
    OPENROUTER_ID_MAP = {
        "gpt-4o": "openai/gpt-4o",
        "gpt-4o-2024-11-20": "openai/gpt-4o-2024-11-20",
        "gpt-4o-mini": "openai/gpt-4o-mini",
        "gpt-4-turbo": "openai/gpt-4-turbo",
        "gpt-4": "openai/gpt-4",
        "gpt-3.5-turbo": "openai/gpt-3.5-turbo",
        "gpt-3.5-turbo-0125": "openai/gpt-3.5-turbo-0125",
        "o1-preview": "openai/o1-preview",
        "o1-mini": "openai/o1-mini",
        "o1": "openai/o1",
    }

    def __init__(self, api_key: Optional[str] = None):
        """Initialize OpenAI client.

        Args:
            api_key: OpenAI API key (uses OPENAI_API_KEY env var if not provided)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. "
                "Set OPENAI_API_KEY in .env or pass api_key parameter"
            )
        
        self.client = OpenAI(api_key=self.api_key)
        logger.info("Initialized OpenAI client")

    def get_available_models(self) -> List[str]:
        """Fetch available chat models from OpenAI API.

        Returns:
            List of model IDs available for chat completions
        """
        try:
            models = self.client.models.list()
            # Filter to gpt models (chat models)
            chat_models = [
                m.id for m in models.data 
                if m.id.startswith(("gpt-", "o1"))
            ]
            logger.info(f"Found {len(chat_models)} OpenAI chat models")
            return sorted(chat_models)
        except Exception as e:
            logger.error(f"Failed to fetch OpenAI models: {e}")
            return []

    def measure_ttft(
        self, 
        model_id: str, 
        num_samples: int = 5,
        test_prompt: str = "Say 'Hello' once."
    ) -> Optional[float]:
        """Measure Time To First Token for a model.

        Args:
            model_id: OpenAI model ID (e.g., "gpt-3.5-turbo")
            num_samples: Number of measurements to average
            test_prompt: Prompt to use for testing

        Returns:
            Average TTFT in seconds, or None if failed
        """
        ttfts = []

        for i in range(num_samples):
            try:
                start_time = time.time()

                # Use streaming to measure TTFT
                stream = self.client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": test_prompt}],
                    stream=True,
                    max_tokens=10,
                    temperature=0.0,
                )

                # Get first chunk
                first_chunk = next(stream)
                first_token_time = time.time()
                
                ttft = first_token_time - start_time
                ttfts.append(ttft)

                # Consume rest of stream
                for _ in stream:
                    pass

                logger.debug(f"  Sample {i+1}/{num_samples}: {ttft:.4f}s")

                # Small delay between samples to avoid rate limiting
                if i < num_samples - 1:
                    time.sleep(0.5)

            except Exception as e:
                logger.warning(f"Failed TTFT measurement {i+1} for {model_id}: {e}")
                continue

        if ttfts:
            avg_ttft = sum(ttfts) / len(ttfts)
            std_ttft = (sum((x - avg_ttft) ** 2 for x in ttfts) / len(ttfts)) ** 0.5
            logger.info(f"  TTFT: {avg_ttft:.4f}s ± {std_ttft:.4f}s ({len(ttfts)} samples)")
            return round(avg_ttft, 4)
        
        return None

    def get_model_data(self, model_id: str, measure_latency: bool = True) -> Dict:
        """Get complete data for a model.

        Args:
            model_id: OpenAI model ID
            measure_latency: Whether to measure TTFT (costs API calls)

        Returns:
            Dictionary with all model data for cache
        """
        logger.info(f"\nCollecting data for {model_id}...")

        # Get pricing
        pricing = self.PRICING.get(model_id)
        if not pricing:
            # Try base model name (e.g., "gpt-4o" for "gpt-4o-2024-11-20")
            base_name = model_id.split("-202")[0]  # Remove date suffix
            pricing = self.PRICING.get(base_name)
        
        if not pricing:
            logger.warning(f"  No pricing data for {model_id}")
            return {}

        # Measure TTFT if requested
        ttft = None
        if measure_latency:
            ttft = self.measure_ttft(model_id)
            if ttft is None:
                logger.warning(f"  Failed to measure TTFT for {model_id}")

        # Build complete model data
        model_data = {
            "name": self._format_display_name(model_id),
            "slug": model_id,
            "creator_name": "OpenAI",
            "creator_slug": "openai",
            
            # Pricing (per 1M tokens)
            "price_1m_input": pricing["input"],
            "price_1m_output": pricing["output"],
            "input_cost_per_m": pricing["input"],
            "output_cost_per_m": pricing["output"],
            
            # Latency
            "measured_ttft_seconds": ttft,
            
            # Context
            "context_length": pricing["context_window"],
            "context_window_k": pricing["context_window"] // 1000,
            
            # Identifiers
            "openrouter_id": self.OPENROUTER_ID_MAP.get(model_id, f"openai/{model_id}"),
            "data_source": "openai_direct",
            
            # Note: Quality scores (benchmarks) and hallucination rates
            # should be populated from other sources (Artificial Analysis, Vectara)
        }

        logger.info(f"  ✓ Collected: ${pricing['input']}/{pricing['output']} per 1M, "
                   f"TTFT={ttft or 'N/A'}s, Context={pricing['context_window']}")

        return model_data

    def _format_display_name(self, model_id: str) -> str:
        """Format model ID into display name.

        Args:
            model_id: OpenAI model ID

        Returns:
            Formatted display name
        """
        # Simple formatting - can be enhanced
        name_map = {
            "gpt-4o": "GPT-4o",
            "gpt-4o-2024-11-20": "GPT-4o (Nov '24)",
            "gpt-4o-mini": "GPT-4o mini",
            "gpt-4-turbo": "GPT-4 Turbo",
            "gpt-4": "GPT-4",
            "gpt-3.5-turbo": "GPT-3.5 Turbo",
            "gpt-3.5-turbo-0125": "GPT-3.5 Turbo (0125)",
            "o1-preview": "o1-preview",
            "o1-mini": "o1-mini",
            "o1": "o1",
        }
        return name_map.get(model_id, model_id)

    def collect_models_data(
        self, 
        model_ids: Optional[List[str]] = None,
        measure_latency: bool = True,
        delay_between_models: float = 2.0
    ) -> List[Dict]:
        """Collect complete data for multiple models.

        Args:
            model_ids: List of model IDs to collect (None = all with pricing)
            measure_latency: Whether to measure TTFT
            delay_between_models: Delay between models (seconds)

        Returns:
            List of model data dictionaries
        """
        if model_ids is None:
            # Default to models with pricing
            model_ids = list(self.PRICING.keys())

        logger.info(f"\n{'='*80}")
        logger.info(f"Collecting data for {len(model_ids)} OpenAI models")
        logger.info(f"Measuring latency: {measure_latency}")
        logger.info(f"{'='*80}")

        results = []
        for i, model_id in enumerate(model_ids):
            logger.info(f"\n[{i+1}/{len(model_ids)}] Processing {model_id}...")
            
            model_data = self.get_model_data(model_id, measure_latency)
            
            if model_data:
                results.append(model_data)
            else:
                logger.warning(f"  Skipped {model_id} (no data)")

            # Delay to avoid rate limiting
            if i < len(model_ids) - 1:
                time.sleep(delay_between_models)

        logger.info(f"\n{'='*80}")
        logger.info(f"✅ Successfully collected data for {len(results)} models")
        logger.info(f"{'='*80}\n")

        return results


def update_cache_with_openai_data(
    cache_path: Path, 
    openai_data: List[Dict],
    merge_strategy: str = "update"
) -> Tuple[int, int]:
    """Update model cache with OpenAI data.

    Args:
        cache_path: Path to models_cache.json
        openai_data: List of model data dictionaries from OpenAI
        merge_strategy: "update" (update existing) or "add" (add new only)

    Returns:
        Tuple of (models_updated, models_added)
    """
    # Load existing cache
    with open(cache_path) as f:
        models = json.load(f)

    # Create lookup by openrouter_id
    openai_by_id = {m["openrouter_id"]: m for m in openai_data if "openrouter_id" in m}
    
    updated = 0
    added = 0

    # Update existing models
    for model in models:
        or_id = model.get("openrouter_id")
        if or_id in openai_by_id:
            openai_model = openai_by_id[or_id]
            
            # Update fields
            if merge_strategy == "update":
                # Update pricing
                if "input_cost_per_m" in openai_model:
                    model["input_cost_per_m"] = openai_model["input_cost_per_m"]
                    model["price_1m_input"] = openai_model["input_cost_per_m"]
                
                if "output_cost_per_m" in openai_model:
                    model["output_cost_per_m"] = openai_model["output_cost_per_m"]
                    model["price_1m_output"] = openai_model["output_cost_per_m"]
                
                # Update latency if measured
                if openai_model.get("measured_ttft_seconds"):
                    model["measured_ttft_seconds"] = openai_model["measured_ttft_seconds"]
                
                # Update context
                if "context_length" in openai_model:
                    model["context_length"] = openai_model["context_length"]
                    model["context_window_k"] = openai_model["context_window_k"]
                
                updated += 1
                logger.debug(f"  Updated: {model['name']}")
            
            # Remove from lookup (already processed)
            del openai_by_id[or_id]

    # Add new models
    if merge_strategy == "add" or merge_strategy == "update":
        for openai_model in openai_by_id.values():
            models.append(openai_model)
            added += 1
            logger.debug(f"  Added: {openai_model['name']}")

    # Save updated cache
    with open(cache_path, 'w') as f:
        json.dump(models, f, indent=2)

    return updated, added


if __name__ == '__main__':
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser(
        description='Collect complete model data from OpenAI API'
    )
    parser.add_argument(
        '--models', 
        nargs='+',
        help='Specific model IDs to collect (default: all with pricing)'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=5,
        help='Number of TTFT samples per model (default: 5)'
    )
    parser.add_argument(
        '--no-latency',
        action='store_true',
        help='Skip TTFT measurement (faster, no API calls)'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=2.0,
        help='Delay between models in seconds (default: 2.0)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path("data/openai_models_data.json"),
        help='Output file for collected data'
    )
    parser.add_argument(
        '--update-cache',
        action='store_true',
        help='Update models_cache.json with collected data'
    )
    parser.add_argument(
        '--list-models',
        action='store_true',
        help='List available OpenAI models and exit'
    )

    args = parser.parse_args()

    try:
        # Initialize client
        client = OpenAIDirectClient()

        if args.list_models:
            # Just list models
            models = client.get_available_models()
            print("\nAvailable OpenAI Models:")
            print("=" * 60)
            for model in models:
                pricing = client.PRICING.get(model)
                if pricing:
                    print(f"✓ {model:<40} ${pricing['input']}/{pricing['output']} per 1M")
                else:
                    print(f"  {model:<40} (no pricing data)")
            print(f"\nTotal: {len(models)} models")
            exit(0)

        # Collect data
        model_ids = args.models
        openai_data = client.collect_models_data(
            model_ids=model_ids,
            measure_latency=not args.no_latency,
            delay_between_models=args.delay
        )

        if not openai_data:
            print("\n❌ No data collected")
            exit(1)

        # Save to output file
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(openai_data, f, indent=2)
        
        print(f"\n✅ Saved data to {args.output}")

        # Update cache if requested
        if args.update_cache:
            cache_path = Path("data/models_cache.json")
            if not cache_path.exists():
                print(f"\n⚠️  Cache file not found: {cache_path}")
                print("Run with --output only to save data separately")
            else:
                updated, added = update_cache_with_openai_data(
                    cache_path,
                    openai_data,
                    merge_strategy="update"
                )
                print(f"\n✅ Updated cache: {updated} models updated, {added} models added")

        # Summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        for model_data in openai_data:
            name = model_data['name']
            input_cost = model_data.get('input_cost_per_m', 'N/A')
            output_cost = model_data.get('output_cost_per_m', 'N/A')
            ttft = model_data.get('measured_ttft_seconds', 'N/A')
            context = model_data.get('context_window_k', 'N/A')
            
            print(f"\n{name}:")
            print(f"  Pricing: ${input_cost}/{output_cost} per 1M tokens")
            print(f"  Latency: {ttft}s TTFT")
            print(f"  Context: {context}K tokens")

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        exit(130)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        exit(1)

