"""
Multi-source pricing client for fetching real-time model pricing.

Uses a three-tier approach for comprehensive pricing coverage:

1. **LiteLLM** (Primary): Pricing for commercial models
   - OpenAI (GPT-4, GPT-3.5, etc.)
   - Anthropic (Claude)
   - Google (Gemini)
   - And many more

2. **OpenRouter** (Secondary): Pricing for open-source models
   - Meta (Llama variants)
   - Qwen models
   - DeepSeek models
   - Mistral/Mixtral models
   - And 300+ additional models

3. **Heuristic Fallback** (Tertiary): Physics-based estimation
   - For models not in any database
   - Based on parameter count and model characteristics

This ensures maximum pricing coverage while maintaining HuggingFace
as the exclusive source for benchmark scores.
"""

import time
from typing import Dict, Optional
from functools import lru_cache


class LiteLLMPricingClient:
    """
    Fetches real-time pricing using LiteLLM's pricing database.
    
    LiteLLM maintains an up-to-date pricing database for all major providers.
    Falls back to heuristics if a model is not found.
    """
    
    _pricing_cache: Dict[str, Dict[str, float]] = {}
    _last_update: float = 0
    _cache_ttl: int = 3600  # Refresh every hour

    @staticmethod
    def get_pricing(model_name: str) -> Dict[str, float]:
        """
        Get pricing for a model using multiple sources.
        
        Priority:
        1. LiteLLM database (for commercial models)
        2. OpenRouter API (for open-source and additional models)
        3. Heuristic fallback (when no pricing found)
        
        Args:
            model_name: Model name (e.g., "gpt-4o", "claude-3-5-sonnet")
            
        Returns:
            Dict with 'input', 'output' costs per million tokens and 'source' indicating where pricing came from
        """
        # Determine priority based on model name
        is_commercial = any(p in model_name.lower() for p in ['gpt-', 'claude-', 'gemini-', 'mistral-large', 'o1-'])
        
        if is_commercial:
            # 1. Try LiteLLM first for commercial models
            try:
                import litellm
                normalized_name = LiteLLMPricingClient._normalize_model_name(model_name)
                try:
                    model_info = litellm.get_model_info(normalized_name)
                    if model_info and 'input_cost_per_token' in model_info:
                        input_cost = model_info['input_cost_per_token'] * 1_000_000
                        output_cost = model_info.get('output_cost_per_token', input_cost * 3) * 1_000_000
                        return {
                            "input": round(input_cost, 4),
                            "output": round(output_cost, 4),
                            "source": "litellm"
                        }
                except Exception:
                    pass
            except ImportError:
                pass

        # 2. Try OpenRouter (Primary for open source, fallback for commercial)
        try:
            openrouter_pricing = LiteLLMPricingClient._get_openrouter_pricing(model_name)
            if openrouter_pricing:
                openrouter_pricing["source"] = "openrouter"
                return openrouter_pricing
        except Exception:
            pass
            
        # 3. Try LiteLLM as fallback for non-commercial (if not found in OpenRouter)
        # SKIP this step to avoid slow lookups and "Provider List" spam for thousands of HF models
        # If it's not a known commercial provider and not in OpenRouter, it's likely not in LiteLLM either.
        
        # 4. Final fallback to heuristics
        fallback_pricing = LiteLLMPricingClient._estimate_pricing_fallback(model_name)
        fallback_pricing["source"] = "estimated"
        return fallback_pricing
    
    @staticmethod
    def _normalize_model_name(name: str) -> str:
        """
        Normalize model name to match LiteLLM's naming convention.
        
        LiteLLM uses specific model names for each provider.
        Examples:
        - "GPT-4o" -> "gpt-4o"
        - "Claude-3.5-Sonnet" -> "claude-3-5-sonnet-20240620"
        - "Llama-3.1-70B" -> "meta-llama/Llama-3.1-70B"
        """
        name_lower = name.lower().replace('_', '-')
        
        # OpenAI models
        if 'gpt-4o' in name_lower:
            if 'mini' in name_lower:
                return 'gpt-4o-mini'
            return 'gpt-4o'
        elif 'gpt-4' in name_lower:
            if 'turbo' in name_lower:
                return 'gpt-4-turbo'
            return 'gpt-4'
        elif 'gpt-3.5' in name_lower:
            return 'gpt-3.5-turbo'
        
        # Anthropic models
        elif 'claude-3.5-sonnet' in name_lower or 'claude-3-5-sonnet' in name_lower:
            return 'claude-3-5-sonnet-20240620'
        elif 'claude-3-opus' in name_lower:
            return 'claude-3-opus-20240229'
        elif 'claude-3-sonnet' in name_lower:
            return 'claude-3-sonnet-20240229'
        elif 'claude-3-haiku' in name_lower:
            return 'claude-3-haiku-20240307'
        
        # Google models
        elif 'gemini-1.5-pro' in name_lower:
            return 'gemini-1.5-pro'
        elif 'gemini-1.5-flash' in name_lower:
            return 'gemini-1.5-flash'
        elif 'gemini-pro' in name_lower:
            return 'gemini-pro'
        
        # Meta Llama models (via various providers)
        elif 'llama-3.1-405b' in name_lower or 'llama-3-1-405b' in name_lower:
            return 'meta-llama/Meta-Llama-3.1-405B-Instruct'
        elif 'llama-3.1-70b' in name_lower or 'llama-3-1-70b' in name_lower:
            return 'meta-llama/Meta-Llama-3.1-70B-Instruct'
        elif 'llama-3.1-8b' in name_lower or 'llama-3-1-8b' in name_lower:
            return 'meta-llama/Meta-Llama-3.1-8B-Instruct'
        elif 'llama-3.3-70b' in name_lower or 'llama-3-3-70b' in name_lower:
            return 'meta-llama/Llama-3.3-70B-Instruct'
        
        # DeepSeek models
        elif 'deepseek-coder' in name_lower:
            return 'deepseek-coder'
        elif 'deepseek-v3' in name_lower:
            return 'deepseek-chat'
        
        # Mistral models
        elif 'mistral-large' in name_lower:
            return 'mistral-large-latest'
        elif 'mixtral-8x22b' in name_lower:
            return 'mistral/mixtral-8x22b-instruct-v0.1'
        
        # Qwen models
        elif 'qwen-2.5-72b' in name_lower or 'qwen-2-5-72b' in name_lower:
            return 'qwen/qwen-2.5-72b-instruct'
        
        # Return original if no match
        return name
    
    @staticmethod
    def _get_openrouter_pricing(model_name: str) -> Optional[Dict[str, float]]:
        """
        Get pricing from OpenRouter API.
        
        OpenRouter provides pricing for many open-source models that may not
        be in LiteLLM's database. This is especially useful for Llama, Qwen,
        and other open-source models.
        
        Args:
            model_name: Model name
            
        Returns:
            Dict with 'input' and 'output' costs or None if not found
        """
        import requests
        
        try:
            # Refresh cache if needed
            if time.time() - LiteLLMPricingClient._last_update > LiteLLMPricingClient._cache_ttl:
                response = requests.get("https://openrouter.ai/api/v1/models", timeout=5)
                if response.status_code == 200:
                    models = response.json().get('data', [])
                    
                    # Cache all models
                    LiteLLMPricingClient._pricing_cache.clear()
                    for model in models:
                        pricing = model.get('pricing', {})
                        model_id = model['id']
                        model_display_name = model.get('name', model_id)
                        
                        input_cost = float(pricing.get('prompt', 0)) * 1_000_000
                        output_cost = float(pricing.get('completion', 0)) * 1_000_000
                        
                        # Store by both ID and display name for easier lookup
                        LiteLLMPricingClient._pricing_cache[model_id.lower()] = {
                            'input': round(input_cost, 4),
                            'output': round(output_cost, 4)
                        }
                        LiteLLMPricingClient._pricing_cache[model_display_name.lower()] = {
                            'input': round(input_cost, 4),
                            'output': round(output_cost, 4)
                        }
                    
                    LiteLLMPricingClient._last_update = time.time()
            
            # Try to find pricing in cache
            name_lower = model_name.lower().replace('_', '-').replace(' ', '-')
            
            # 1. Direct lookup
            if name_lower in LiteLLMPricingClient._pricing_cache:
                return LiteLLMPricingClient._pricing_cache[name_lower]
            
            # 2. Normalize "meta-llama" to "llama" (common mismatch)
            name_normalized = name_lower.replace('meta-llama', 'llama')
            if name_normalized in LiteLLMPricingClient._pricing_cache:
                return LiteLLMPricingClient._pricing_cache[name_normalized]
            
            # 3. Try matching without provider prefix (e.g. "llama-3.1-8b-instruct" matches "meta-llama/llama-3.1-8b-instruct")
            if '/' in name_lower:
                model_suffix = name_lower.split('/')[-1]
                
                # Also normalize suffix (e.g. meta-llama-3.1 -> llama-3.1)
                model_suffix_norm = model_suffix.replace('meta-llama', 'llama')
                
                # Look for keys ending with this suffix
                for cached_name, pricing in LiteLLMPricingClient._pricing_cache.items():
                    if cached_name.endswith(f"/{model_suffix}") or cached_name == model_suffix:
                        return pricing
                    # Try normalized suffix
                    if cached_name.endswith(f"/{model_suffix_norm}") or cached_name == model_suffix_norm:
                        return pricing

            # 4. Fuzzy lookup - check if model name is a substring of any cached model
            # Be more strict: ensure the model name part matches
            for cached_name, pricing in LiteLLMPricingClient._pricing_cache.items():
                if name_lower in cached_name or cached_name in name_lower:
                    return pricing
            
            return None
            
        except Exception as e:
            # Silently fail and return None to try next fallback
            return None
    
    @staticmethod
    def _estimate_pricing_fallback(model_name: str) -> Dict[str, float]:
        """
        Fallback pricing estimation based on model characteristics.
        
        Used when LiteLLM doesn't have pricing info for a model.
        """
        # Estimate parameter count from model name
        import re
        name_lower = model_name.lower()
        
        # Look for explicit parameter counts
        match = re.search(r'(\d+)b', name_lower)
        if match:
            params = float(match.group(1))
        # Known models
        elif 'gpt-4' in name_lower and 'mini' not in name_lower:
            params = 1800.0
        elif 'claude-3' in name_lower and 'opus' in name_lower:
            params = 175.0
        elif 'gemini-1.5-pro' in name_lower:
            params = 175.0
        elif 'mini' in name_lower or 'small' in name_lower:
            params = 10.0
        else:
            params = 70.0
        
        # Base pricing on parameter count
        name_lower = model_name.lower()
        
        if params < 10:
            input_cost = 0.10 + (params / 10) * 0.20  # $0.10-0.30
        elif params < 100:
            input_cost = 0.30 + ((params - 10) / 90) * 0.70  # $0.30-1.00
        else:
            input_cost = 1.00 + ((params - 100) / 1000) * 4.00  # $1.00-5.00
        
        # Adjust for proprietary models
        if any(x in name_lower for x in ['gpt-4', 'claude-3', 'gemini']):
            input_cost *= 2.0
        
        # Adjust for specialized models
        if 'mini' in name_lower or 'small' in name_lower:
            input_cost *= 0.5
        elif 'turbo' in name_lower or 'flash' in name_lower:
            input_cost *= 0.7
        
        output_cost = input_cost * 3.0
        
        return {
            'input': round(input_cost, 4),
            'output': round(output_cost, 4)
        }
    
    @staticmethod
    @lru_cache(maxsize=1000)
    def get_cached_pricing(model_name: str) -> Dict[str, float]:
        """
        Get pricing with caching to avoid repeated lookups.
        
        Args:
            model_name: Model name
            
        Returns:
            Dict with 'input' and 'output' costs per million tokens
        """
        return LiteLLMPricingClient.get_pricing(model_name)


# Convenience functions for backwards compatibility
def get_pricing(model_name: str) -> Dict[str, float]:
    """Get pricing for a model."""
    return LiteLLMPricingClient.get_pricing(model_name)
