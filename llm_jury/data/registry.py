"""
Model registry for managing and creating model metadata.

Consolidates registry logic from:
- llm_router.py (get_model_registry with estimated pricing)
- llm_recommendation_orchestrator.py (get_model_registry with static data)
- LLM_Capability.py (ModelRegistry class with HF leaderboard)

Pricing is estimated using heuristics based on model size.
"""

import json
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from llm_jury.core.models import ModelMetadata, ProductArchetype, ModelSpecs
from llm_jury.data.huggingface import HuggingFaceDataSource


class ModelRegistry:
    """Centralized model registry with multiple data sources."""
    
    @staticmethod
    def _get_pricing(name: str) -> dict:
        """
        Get real-time pricing using LiteLLM.
        
        LiteLLM provides pricing for all major LLM providers including:
        - OpenAI (GPT-4, GPT-3.5, etc.)
        - Anthropic (Claude)
        - Google (Gemini)
        - Meta (Llama)
        - And many more
        
        Falls back to heuristic estimation if model not found in LiteLLM.
        """
        from llm_jury.data.litellm_pricing import LiteLLMPricingClient
        
        return LiteLLMPricingClient.get_pricing(name)
    
    @staticmethod
    def create_model(
        name: str, mmlu: float, gpqa: float, math: float, ifeval: float,
        tool: float, ctx: int, arch: ProductArchetype, lat: float, params: float,
        hallu: float = 0.0, ethics: float = 0.0,
        downloads: int = 0, likes: int = 0, created: str = "",
        daily_requests: int = 0, is_top_10: bool = False,
        avg_uptime: float = 0.0, num_apps: int = 0, num_notable: int = 0,
        proprietary: bool = False
    ) -> ModelMetadata:
        """
        Create a ModelMetadata instance with estimated pricing.
        
        Args:
            name: Model name
            mmlu, gpqa, math, ifeval: Benchmark scores (0-100)
            tool: Tool use ability (0-1)
            ctx: Context window in thousands
            arch: Product archetype
            lat: Median latency in ms
            params: Parameter count in billions
            hallu: Hallucination rate (0-1)
            ethics: Ethics score (0-100)
            downloads, likes, created: HuggingFace metrics
            daily_requests: Daily request count
            is_top_10: Whether in top 10 most used
            avg_uptime: Average uptime percentage
            num_apps, num_notable: App usage metrics
            proprietary: Whether model is proprietary
            
        Returns:
            ModelMetadata instance with estimated pricing
        """
        # Get pricing from LiteLLM
        costs = ModelRegistry._get_pricing(name)
        
        return ModelMetadata(
            name=name,
            mmlu_score=mmlu,
            gpqa_score=gpqa,
            math_score=math,
            ifeval_score=ifeval,
            tool_use_ability=tool,
            context_window_k=ctx,
            hallucination_rate=hallu,
            ethics_score=ethics,
            hf_downloads=downloads,
            hf_likes=likes,
            hf_created_at=created,
            archetype=arch,
            median_latency_ms=lat,
            param_count_b=params,
            input_cost_per_m=costs["input"],
            output_cost_per_m=costs["output"],
            pricing_source=costs.get("source", "unknown"),
            is_top_10_used=is_top_10,
            date_created=created,
            avg_uptime_90d=avg_uptime,
            num_apps_using=num_apps,
            num_notable_apps=num_notable,
            daily_requests=daily_requests
        )
    
    @staticmethod
    def get_static_models() -> List[ModelMetadata]:
        """
        Get hardcoded models with known metadata.
        Merged from llm_router.py and llm_recommendation_orchestrator.py.
        
        Returns:
            List of well-known models with static metadata
        """
        return [
            # Frontier Models
            ModelRegistry.create_model(
                "GPT-4o", 88.7, 73.0, 75.0, 88.0, 0.95, 128,
                ProductArchetype.FRONTIER, 450, 1800,
                hallu=0.03, ethics=95.0,
                downloads=10000000, likes=50000, created="2024-05-13",
                daily_requests=10000000, is_top_10=True,
                avg_uptime=99.95, num_apps=50000, num_notable=500,
                proprietary=True
            ),
            ModelRegistry.create_model(
                "Claude-3.5-Sonnet", 88.3, 65.0, 70.0, 89.0, 0.98, 200,
                ProductArchetype.FRONTIER, 800, 200,
                hallu=0.02, ethics=98.0,
                downloads=5000000, likes=25000, created="2024-06-20",
                daily_requests=5000000, is_top_10=True,
                avg_uptime=99.90, num_apps=25000, num_notable=300,
                proprietary=True
            ),
            ModelRegistry.create_model(
                "Llama-3.1-70B", 82.0, 55.0, 50.0, 82.0, 0.80, 128,
                ProductArchetype.FRONTIER, 300, 70,
                hallu=0.10, ethics=85.0,
                downloads=15000000, likes=30000, created="2024-07-23",
                daily_requests=2000000, is_top_10=True,
                avg_uptime=99.50, num_apps=15000, num_notable=100
            ),
            
            # Reasoning Specialists
            ModelRegistry.create_model(
                "DeepSeek-Coder-V2", 80.0, 65.0, 85.0, 80.0, 0.90, 128,
                ProductArchetype.REASONING_SPECIALIST, 600, 236,
                hallu=0.08, ethics=70.0,
                downloads=800000, likes=15000, created="2024-06-15",
                daily_requests=500000, is_top_10=False,
                avg_uptime=99.00, num_apps=5000, num_notable=20
            ),
            ModelRegistry.create_model(
                "Phi-3-Medium", 78.0, 45.0, 60.0, 75.0, 0.60, 128,
                ProductArchetype.REASONING_SPECIALIST, 150, 14,
                hallu=0.12, ethics=75.0,
                downloads=500000, likes=8000, created="2024-04-23",
                daily_requests=800000
            ),
            
            # RAG Specialists
            ModelRegistry.create_model(
                "GPT-4o-mini", 82.0, 50.0, 60.0, 80.0, 0.85, 128,
                ProductArchetype.RAG_SPECIALIST, 200, 20,
                hallu=0.05, ethics=90.0,
                downloads=8000000, likes=40000, created="2024-07-18",
                daily_requests=8000000, is_top_10=True,
                avg_uptime=99.99, num_apps=40000, num_notable=200,
                proprietary=True
            ),
            
            # Small models (now RAG Specialist)
            ModelRegistry.create_model(
                "Llama-3.1-8B", 68.4, 30.0, 30.0, 70.0, 0.65, 128,
                ProductArchetype.RAG_SPECIALIST, 100, 8,
                hallu=0.15, ethics=80.0,
                downloads=15000000, likes=30000, created="2024-07-23",
                daily_requests=5000000, is_top_10=True
            ),
            
            # Creative/Uncensored
            ModelRegistry.create_model(
                "Dolphin-Mixtral-Uncensored", 70.0, 40.0, 40.0, 60.0, 0.50, 32,
                ProductArchetype.FRONTIER, 250, 47,
                hallu=0.20, ethics=20.0,
                downloads=500000, likes=2000, created="2023-12-20",
                daily_requests=100000
            ),
        ]
    
    @staticmethod
    def get_live_models(limit: int = 150, include_anchors: bool = True) -> List[ModelSpecs]:
        """
        Fetch live models from HuggingFace leaderboard.
        
        Args:
            limit: Maximum number of models to fetch
            include_anchors: Whether to include reference anchors
            
        Returns:
            List of ModelSpecs from HuggingFace leaderboard
        """
        return HuggingFaceDataSource.fetch_leaderboard(limit=limit, include_anchors=include_anchors)
    
    
    @staticmethod
    def load_cache(
        cache_path: Optional[Union[str, Path]] = None,
        verbose: bool = True
    ) -> List[ModelMetadata]:
        """
        Load models from a cache file.
        
        This is the main method for loading model data. It supports:
        - Default cache (data/models_cache.json)
        - Custom cache files (user-provided path)
        - Both the standard cache format and enhanced cache format
        
        Args:
            cache_path: Path to cache file. If None, uses default from config.
            verbose: Whether to print loading messages.
            
        Returns:
            List of ModelMetadata from the cache
            
        Example:
            # Load from default cache
            models = ModelRegistry.load_cache()
            
            # Load from custom cache file
            models = ModelRegistry.load_cache("/path/to/my_models.json")
            
            # Load silently
            models = ModelRegistry.load_cache(verbose=False)
        """
        from llm_jury.config import get_config
        
        # Determine cache path
        if cache_path is None:
            config = get_config()
            cache_file = config.cache_file
        else:
            cache_file = Path(cache_path)
        
        if not cache_file.exists():
            if verbose:
                print(f"⚠️  Cache not found at {cache_file}")
                print("   Run 'python run_etl.py --complete-only' to generate it")
                print("   Or provide a custom cache path: ModelRegistry.load_cache('/path/to/cache.json')")
            return []
        
        with open(cache_file, 'r') as f:
            cache_data = json.load(f)
        
        # Handle both formats: list of models or dict with 'models' key
        if isinstance(cache_data, list):
            model_list = cache_data
        else:
            model_list = cache_data.get('models', [])
        
        models = []
        for model_dict in model_list:
            model = ModelRegistry._dict_to_model(model_dict)
            if model:
                models.append(model)
        
        if verbose:
            print(f"✅ Loaded {len(models)} models from {cache_file}")
        
        return models
    
    @staticmethod
    def load_raw_cache(
        cache_path: Optional[Union[str, Path]] = None
    ) -> List[Dict[str, Any]]:
        """
        Load raw model data from cache as dictionaries.
        
        Useful for passing to QualityScorer or custom processing.
        
        Args:
            cache_path: Path to cache file. If None, uses default from config.
            
        Returns:
            List of model dictionaries from the cache
            
        Example:
            # Get raw data for QualityScorer
            raw_data = ModelRegistry.load_raw_cache()
            scorer = QualityScorer(all_models_data=raw_data)
        """
        from llm_jury.config import get_config
        
        if cache_path is None:
            config = get_config()
            cache_file = config.cache_file
        else:
            cache_file = Path(cache_path)
        
        if not cache_file.exists():
            return []
        
        with open(cache_file, 'r') as f:
            cache_data = json.load(f)
        
        # Handle both formats
        if isinstance(cache_data, list):
            return cache_data
        return cache_data.get('models', cache_data)
    
    @staticmethod
    def _dict_to_model(model_dict: Dict[str, Any]) -> Optional[ModelMetadata]:
        """Convert a model dictionary to ModelMetadata."""
        try:
            # Determine archetype
            arch_str = model_dict.get('archetype', 'Frontier')
            if isinstance(arch_str, str):
                if "Frontier" in arch_str:
                    archetype = ProductArchetype.FRONTIER
                elif "Reasoning" in arch_str:
                    archetype = ProductArchetype.REASONING_SPECIALIST
                elif "RAG" in arch_str:
                    archetype = ProductArchetype.RAG_SPECIALIST
                elif "Bulk" in arch_str:
                    archetype = ProductArchetype.BULK_OPS
                elif "Edge" in arch_str or "Privacy" in arch_str:
                    archetype = ProductArchetype.RAG_SPECIALIST
                else:
                    archetype = ProductArchetype.FRONTIER
            else:
                archetype = ProductArchetype.FRONTIER
            
            # Create model with all available fields
            model = ModelMetadata(
                name=model_dict.get('name', model_dict.get('display_name', 'Unknown')),
                mmlu_score=model_dict.get('mmlu_score'),
                gpqa_score=model_dict.get('gpqa_score'),
                math_score=model_dict.get('math_score'),
                ifeval_score=model_dict.get('ifeval_score'),
                tool_use_ability=model_dict.get('tool_use_ability'),
                context_window_k=model_dict.get('context_window_k'),
                hallucination_rate=model_dict.get('hallucination_rate'),
                ethics_score=model_dict.get('ethics_score'),
                hf_downloads=model_dict.get('hf_downloads'),
                hf_likes=model_dict.get('hf_likes'),
                hf_created_at=model_dict.get('hf_created_at'),
                archetype=archetype,
                median_latency_ms=model_dict.get('median_latency_ms'),
                param_count_b=model_dict.get('param_count_b'),
                input_cost_per_m=model_dict.get('input_cost_per_m', model_dict.get('price_1m_input')),
                output_cost_per_m=model_dict.get('output_cost_per_m', model_dict.get('price_1m_output')),
                pricing_source=model_dict.get('pricing_source', model_dict.get('data_source')),
                daily_requests=model_dict.get('daily_requests'),
                # Artificial Analysis benchmark indices
                intelligence_index=model_dict.get('intelligence_index'),
                coding_index=model_dict.get('coding_index'),
                math_index=model_dict.get('math_index'),
                # Raw benchmarks
                mmlu_pro=model_dict.get('mmlu_pro'),
                gpqa=model_dict.get('gpqa'),
                hle=model_dict.get('hle'),
                livecodebench=model_dict.get('livecodebench'),
                scicode=model_dict.get('scicode'),
                math_500=model_dict.get('math_500'),
                aime=model_dict.get('aime'),
                # Performance metrics
                output_tokens_per_second=model_dict.get('output_tokens_per_second'),
                measured_ttft_seconds=model_dict.get('measured_ttft_seconds'),
                time_to_first_token_seconds=model_dict.get('time_to_first_token_seconds'),
                # Reliability metrics
                refusal_rate=model_dict.get('refusal_rate'),
                factual_consistency_rate=model_dict.get('factual_consistency_rate'),
            )
            
            # Transfer additional fields for custom objectives
            for key, value in model_dict.items():
                if not hasattr(model, key) and value is not None:
                    setattr(model, key, value)
            
            return model
        except Exception as e:
            return None
    
    @staticmethod
    def load_enhanced_cache(
        cache_path: Optional[Union[str, Path]] = None
    ) -> List[ModelMetadata]:
        """
        Load models from the enhanced cache file.
        
        DEPRECATED: Use load_cache() instead, which handles both formats.
        
        Args:
            cache_path: Path to cache file. If None, uses default.
            
        Returns:
            List of ModelMetadata from the cache
        """
        return ModelRegistry.load_cache(cache_path=cache_path)
