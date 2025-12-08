"""Client for measuring Time To First Token (TTFT) via OpenRouter API."""

import os
import time
import json
import logging
import requests
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class OpenRouterTTFTClient:
    """Client for measuring TTFT latency via OpenRouter."""

    OPENROUTER_API_URL = "https://openrouter.ai/api/v1"
    MODELS_ENDPOINT = f"{OPENROUTER_API_URL}/models"
    CHAT_ENDPOINT = f"{OPENROUTER_API_URL}/chat/completions"

    def __init__(self, api_key: Optional[str] = None):
        """Initialize client with API key.

        Args:
            api_key: OpenRouter API key (uses env var if not provided)
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key not provided. "
                "Set OPENROUTER_API_KEY in .env or pass api_key parameter"
            )

    def get_available_models(self) -> Dict[str, str]:
        """Fetch available models from OpenRouter.

        Returns:
            Dictionary mapping model ID to model name
        """
        try:
            response = requests.get(self.MODELS_ENDPOINT, timeout=30)
            response.raise_for_status()
            data = response.json()
            models = data.get('data', [])
            return {m['id']: m['name'] for m in models}
        except Exception as e:
            logger.error(f"Failed to fetch OpenRouter models: {e}")
            return {}

    def map_model_to_openrouter(self, our_model_name: str, openrouter_models: Dict[str, str]) -> Optional[str]:
        """Map our model name to OpenRouter model ID.

        Args:
            our_model_name: Our internal model name
            openrouter_models: Dictionary of OpenRouter model IDs to names

        Returns:
            OpenRouter model ID or None if not found
        """
        name_lower = our_model_name.lower()

        # Direct mappings for known models
        direct_mappings = {
            # OpenAI
            'gpt-5 (high)': 'openai/gpt-5',
            'gpt-5 (medium)': 'openai/gpt-5',
            'gpt-5 (minimal)': 'openai/gpt-5',
            'gpt-5 mini (high)': 'openai/gpt-5-mini',
            'gpt-5 mini (medium)': 'openai/gpt-5-mini',
            'gpt-5 mini (minimal)': 'openai/gpt-5-mini',
            'gpt-5 nano (high)': 'openai/gpt-5-nano',
            'gpt-5 nano (medium)': 'openai/gpt-5-nano',
            'gpt-5 nano (minimal)': 'openai/gpt-5-nano',
            'gpt-5.1 (high)': 'openai/gpt-5.1',
            'gpt-5.1 (non-reasoning)': 'openai/gpt-5.1',
            'gpt-4.1': 'openai/gpt-4.1',
            'gpt-4o (nov \'24)': 'openai/gpt-4o',
            'o3': 'openai/o3',
            'o4-mini (high)': 'openai/o4-mini',
            'gpt-oss-120b (high)': 'openai/gpt-oss-120b',
            'gpt-oss-120b (low)': 'openai/gpt-oss-120b',

            # Anthropic
            'claude opus 4.5 (reasoning)': 'anthropic/claude-opus-4.5',
            'claude opus 4.5 (non-reasoning)': 'anthropic/claude-opus-4.5',
            'claude 4.5 sonnet (reasoning)': 'anthropic/claude-sonnet-4.5',
            'claude 4.5 sonnet (non-reasoning)': 'anthropic/claude-sonnet-4.5',
            'claude 4 opus (reasoning)': 'anthropic/claude-opus-4',
            'claude 4 sonnet (reasoning)': 'anthropic/claude-sonnet-4',
            'claude 4 sonnet (non-reasoning)': 'anthropic/claude-sonnet-4',
            'claude 4.5 haiku (reasoning)': 'anthropic/claude-haiku-4.5',
            'claude 4.5 haiku (non-reasoning)': 'anthropic/claude-haiku-4.5',

            # Google
            'gemini 3 pro preview (high)': 'google/gemini-3-pro-preview',
            'gemini 2.5 pro': 'google/gemini-2.5-pro-preview-06-05',
            'gemini 2.5 flash (reasoning)': 'google/gemini-2.5-flash-preview',
            'gemini 2.5 flash (non-reasoning)': 'google/gemini-2.5-flash-preview',
            'gemini 2.5 flash preview (sep \'25) (reasoning)': 'google/gemini-2.5-flash-preview',
            'gemini 2.5 flash preview (sep \'25) (non-reasoning)': 'google/gemini-2.5-flash-preview',
            'gemini 2.5 flash-lite (reasoning)': 'google/gemini-2.5-flash-lite-preview',
            'gemini 2.5 flash-lite (non-reasoning)': 'google/gemini-2.5-flash-lite-preview',
            'gemini 2.5 flash-lite preview (sep \'25) (reasoning)': 'google/gemini-2.5-flash-lite-preview',
            'gemini 2.5 flash-lite preview (sep \'25) (non-reasoning)': 'google/gemini-2.5-flash-lite-preview',
            'gemma 3 4b instruct': 'google/gemma-3-4b-it',
            'gemma 3 12b instruct': 'google/gemma-3-12b-it',
            'gemma 3 27b instruct': 'google/gemma-3-27b-it',

            # xAI
            'grok 3': 'x-ai/grok-3',
            'grok 3 mini reasoning (high)': 'x-ai/grok-3-mini',
            'grok 4': 'x-ai/grok-4',
            'grok 4 fast (reasoning)': 'x-ai/grok-4-fast',
            'grok 4 fast (non-reasoning)': 'x-ai/grok-4-fast',
            'grok 4.1 fast (reasoning)': 'x-ai/grok-4.1-fast',
            'grok 4.1 fast (non-reasoning)': 'x-ai/grok-4.1-fast',

            # DeepSeek
            'deepseek v3 (dec \'24)': 'deepseek/deepseek-chat-v3-0324',
            'deepseek v3 0324': 'deepseek/deepseek-chat-v3-0324',
            'deepseek v3.1 (reasoning)': 'deepseek/deepseek-r1',
            'deepseek v3.1 (non-reasoning)': 'deepseek/deepseek-chat',
            'deepseek v3.1 terminus (reasoning)': 'deepseek/deepseek-r1',
            'deepseek v3.1 terminus (non-reasoning)': 'deepseek/deepseek-chat',
            'deepseek v3.2 exp (reasoning)': 'deepseek/deepseek-r1',
            'deepseek v3.2 exp (non-reasoning)': 'deepseek/deepseek-chat',
            'deepseek r1 (jan \'25)': 'deepseek/deepseek-r1',
            'deepseek r1 0528 (may \'25)': 'deepseek/deepseek-r1',
            'deepseek r1 0528 qwen3 8b': 'deepseek/deepseek-r1-distill-qwen-8b',
            'deepseek r1 distill llama 70b': 'deepseek/deepseek-r1-distill-llama-70b',

            # Mistral
            'mistral large 2 (nov \'24)': 'mistralai/mistral-large-2411',
            'mistral small 3.1': 'mistralai/mistral-small-2503',
            'mistral small 3.2': 'mistralai/mistral-small-2503',
            'ministral 3b': 'mistralai/ministral-3b',
            'ministral 8b': 'mistralai/ministral-8b',

            # Qwen
            'qwen3 8b (reasoning)': 'qwen/qwen3-8b',
            'qwen3 8b (non-reasoning)': 'qwen/qwen3-8b',
            'qwen3 14b (reasoning)': 'qwen/qwen3-14b',
            'qwen3 14b (non-reasoning)': 'qwen/qwen3-14b',
            'qwen3 32b (reasoning)': 'qwen/qwen3-32b',
            'qwen3 4b 2507 (reasoning)': 'qwen/qwen3-4b',
            'qwen3 4b 2507 instruct': 'qwen/qwen3-4b',

            # Meta
            'llama 3.3 instruct 70b': 'meta-llama/llama-3.3-70b-instruct',
            'llama 4 maverick': 'meta-llama/llama-4-maverick',
            'llama 4 scout': 'meta-llama/llama-4-scout',

            # Cohere
            'command a': 'cohere/command-a-03-2025',
            'aya expanse 8b': 'cohere/aya-expanse-8b',
            'aya expanse 32b': 'cohere/aya-expanse-32b',

            # IBM
            'granite 4.0 h small': 'ibm-granite/granite-4.0-h-small',
            'granite 3.3 8b (non-reasoning)': 'ibm-granite/granite-3.3-8b-instruct',

            # Microsoft
            'phi-4': 'microsoft/phi-4',
            'phi-4 mini instruct': 'microsoft/phi-4-mini-instruct',

            # Z AI / GLM
            'glm-4.5 (reasoning)': 'z-ai/glm-4.5',
            'glm-4.5-air': 'z-ai/glm-4.5-air',
            'glm-4.6 (reasoning)': 'z-ai/glm-4.6',
            'glm-4.6 (non-reasoning)': 'z-ai/glm-4.6',

            # Moonshot
            'kimi k2': 'moonshotai/kimi-k2-instruct',

            # Additional Gemini mappings
            'gemini 2.5 flash-lite (reasoning)': 'google/gemini-2.5-flash-lite-preview-06-17',
            'gemini 2.5 flash-lite (non-reasoning)': 'google/gemini-2.5-flash-lite-preview-06-17',
            'gemini 2.5 flash-lite preview (sep \'25) (reasoning)': 'google/gemini-2.5-flash-lite-preview-06-17',
            'gemini 2.5 flash-lite preview (sep \'25) (non-reasoning)': 'google/gemini-2.5-flash-lite-preview-06-17',
            'gemini 2.5 flash (reasoning)': 'google/gemini-2.5-flash-preview-05-20',
            'gemini 2.5 flash (non-reasoning)': 'google/gemini-2.5-flash-preview-05-20',
            'gemini 2.5 flash preview (sep \'25) (reasoning)': 'google/gemini-2.5-flash-preview-05-20',
            'gemini 2.5 flash preview (sep \'25) (non-reasoning)': 'google/gemini-2.5-flash-preview-05-20',

            # Grok 4.1
            'grok 4.1 fast (reasoning)': 'x-ai/grok-4.1-fast',
            'grok 4.1 fast (non-reasoning)': 'x-ai/grok-4.1-fast',

            # Cohere Aya
            'aya expanse 8b': 'cohere/aya-expanse-8b',
            'aya expanse 32b': 'cohere/aya-expanse-32b',

            # IBM Granite
            'granite 4.0 h small': 'ibm-granite/granite-4.0-h-small',
            'granite 3.3 8b (non-reasoning)': 'ibm-granite/granite-3.3-8b-instruct',

            # Qwen additional
            'qwen3 4b 2507 (reasoning)': 'qwen/qwen3-4b',
            'qwen3 4b 2507 instruct': 'qwen/qwen3-4b',
        }

        # Check direct mapping first
        if name_lower in direct_mappings:
            mapped_id = direct_mappings[name_lower]
            if mapped_id in openrouter_models:
                return mapped_id

        # Try fuzzy matching on OpenRouter model names
        for or_id, or_name in openrouter_models.items():
            or_name_lower = or_name.lower()
            # Check if our model name is contained in OpenRouter name or vice versa
            if name_lower in or_name_lower or or_name_lower in name_lower:
                return or_id

        return None

    def measure_ttft(self, model_id: str, num_samples: int = 3) -> Optional[float]:
        """Measure Time To First Token for a model.

        Args:
            model_id: OpenRouter model ID
            num_samples: Number of measurements to average

        Returns:
            Average TTFT in seconds, or None if failed
        """
        ttfts = []

        for i in range(num_samples):
            try:
                start_time = time.time()

                response = requests.post(
                    self.CHAT_ENDPOINT,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model_id,
                        "messages": [{"role": "user", "content": "Say 'Test'."}],
                        "stream": True,
                        "max_tokens": 16,  # Some providers require >= 16
                    },
                    stream=True,
                    timeout=60,
                )

                if response.status_code != 200:
                    logger.warning(f"Model {model_id} returned status {response.status_code}")
                    return None

                # Wait for first chunk
                for chunk in response.iter_lines():
                    if chunk:
                        first_token_time = time.time()
                        ttft = first_token_time - start_time
                        ttfts.append(ttft)
                        break

                # Small delay between samples
                if i < num_samples - 1:
                    time.sleep(0.5)

            except Exception as e:
                logger.warning(f"Failed to measure TTFT for {model_id}: {e}")
                continue

        if ttfts:
            return sum(ttfts) / len(ttfts)
        return None

    def measure_all_models(
        self,
        our_models: List[Dict],
        num_samples: int = 3,
        delay_between_models: float = 1.0
    ) -> Dict[str, float]:
        """Measure TTFT for all models in our cache.

        Args:
            our_models: List of our model dictionaries
            num_samples: Number of samples per model
            delay_between_models: Delay between models in seconds

        Returns:
            Dictionary mapping our model names to TTFT values
        """
        # Get OpenRouter models
        openrouter_models = self.get_available_models()
        if not openrouter_models:
            logger.error("Failed to fetch OpenRouter models")
            return {}

        logger.info(f"Found {len(openrouter_models)} models on OpenRouter")

        results = {}
        mapped_count = 0
        measured_count = 0

        for i, model in enumerate(our_models):
            our_name = model.get('name', '')

            # Map to OpenRouter ID
            or_id = self.map_model_to_openrouter(our_name, openrouter_models)

            if not or_id:
                logger.debug(f"No OpenRouter mapping for: {our_name}")
                continue

            mapped_count += 1
            logger.info(f"[{i+1}/{len(our_models)}] Measuring TTFT for {our_name} ({or_id})...")

            # Measure TTFT
            ttft = self.measure_ttft(or_id, num_samples)

            if ttft is not None:
                results[our_name] = ttft
                measured_count += 1
                logger.info(f"  TTFT: {ttft:.3f}s")
            else:
                logger.warning(f"  Failed to measure TTFT")

            # Delay between models to avoid rate limiting
            if i < len(our_models) - 1:
                time.sleep(delay_between_models)

        logger.info(f"Mapped {mapped_count}/{len(our_models)} models to OpenRouter")
        logger.info(f"Successfully measured TTFT for {measured_count} models")

        return results


def update_cache_with_ttft(cache_path: Path, ttft_results: Dict[str, float]) -> int:
    """Update model cache with TTFT measurements.

    Args:
        cache_path: Path to model cache file
        ttft_results: Dictionary mapping model names to TTFT values

    Returns:
        Number of models updated
    """
    with open(cache_path) as f:
        models = json.load(f)

    updated = 0
    for model in models:
        name = model.get('name', '')
        if name in ttft_results:
            model['measured_ttft_seconds'] = round(ttft_results[name], 4)
            updated += 1

    with open(cache_path, 'w') as f:
        json.dump(models, f, indent=2)

    return updated


if __name__ == '__main__':
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser(description='Measure TTFT for models via OpenRouter')
    parser.add_argument('--samples', type=int, default=3, help='Number of samples per model')
    parser.add_argument('--delay', type=float, default=1.0, help='Delay between models (seconds)')
    parser.add_argument('--dry-run', action='store_true', help='Only show mappings, do not measure')
    args = parser.parse_args()

    # Load models
    cache_path = Path("data/models_complete_composite_indices.json")
    with open(cache_path) as f:
        models = json.load(f)

    print(f"Loaded {len(models)} models from cache")

    # Initialize client
    client = OpenRouterTTFTClient()

    if args.dry_run:
        # Just show mappings
        openrouter_models = client.get_available_models()
        print(f"\nOpenRouter has {len(openrouter_models)} models")
        print("\nModel Mappings:")
        print("=" * 80)

        mapped = 0
        for model in models:
            our_name = model['name']
            or_id = client.map_model_to_openrouter(our_name, openrouter_models)
            if or_id:
                print(f"✓ {our_name:<50} -> {or_id}")
                mapped += 1
            else:
                print(f"✗ {our_name:<50} -> NOT FOUND")

        print(f"\nMapped {mapped}/{len(models)} models")
    else:
        # Measure TTFT
        print(f"\nMeasuring TTFT with {args.samples} samples per model...")
        results = client.measure_all_models(models, args.samples, args.delay)

        if results:
            # Update cache
            updated = update_cache_with_ttft(cache_path, results)
            print(f"\n✅ Updated {updated} models with TTFT data")

            # Also update the main cache
            main_cache = Path("data/models_cache.json")
            if main_cache.exists():
                update_cache_with_ttft(main_cache, results)

            # Show results
            print("\nTTFT Results (sorted by latency):")
            print("=" * 60)
            for name, ttft in sorted(results.items(), key=lambda x: x[1]):
                print(f"{name:<50} {ttft:.3f}s")

