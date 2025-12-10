"""LLM-powered model name matching using OpenRouter.

Uses an LLM to match model names between different sources (HuggingFace, benchmark
datasets, etc.) and your models cache.

This is more accurate than regex-based matching because LLMs understand:
- Semantic similarity ("Llama 3.3 Instruct 70B" == "meta-llama/Llama-3.3-70B-Instruct")
- Model family relationships
- Version numbering conventions
- Common naming variations

Cost: ~$0.01-0.05 per batch of 50 models using GPT-4o-mini

Note: As of December 2025, most benchmarks are fetched directly from canonical sources.
This matcher is primarily used for HuggingFace benchmark integration.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Tuple
import time

import requests

logger = logging.getLogger(__name__)


class LLMModelMatcher:
    """Use an LLM to match model names between sources."""
    
    # Default model for matching (cheap and fast)
    DEFAULT_MODEL = "openai/gpt-4o-mini"
    
    # System prompt for matching
    SYSTEM_PROMPT = """You are an expert at matching LLM model names across different naming conventions.

Your task is to match model names from a SOURCE list to a TARGET list.

STRICT RULES (MUST follow all):
1. SAME MODEL ONLY: Only match if it's clearly the EXACT SAME model
2. SAME SIZE: Model sizes MUST match exactly (7B, 8B, 14B, 32B, 70B, 72B, 405B etc.)
   - 14B != 32B, 70B != 72B (unless they're the same model with different rounding)
   - 8B != 7B
3. SAME VERSION: Version numbers MUST match exactly
   - Llama 3 != Llama 3.1 != Llama 3.2 != Llama 3.3
   - Qwen2 != Qwen2.5
   - GPT-4 != GPT-4o != GPT-4.1
4. NO FINE-TUNE MATCHING: Do NOT match fine-tuned/community variants to base models
   - "MaziyarPanahi/calme-2.1-qwen2.5-72b" is NOT "Qwen2.5 72B"
   - "SuperNova-8B" is NOT "Llama 3.1 8B"
5. OFFICIAL ONLY: Only match official releases (meta-llama/*, Qwen/*, mistralai/*, google/*, etc.)

When in doubt, DO NOT match. It's better to miss a match than to create a wrong one.

Output format: JSON array of matches only
[
  {"source": "exact_source_name", "target": "exact_target_name", "confidence": "high"}
]

Only include very confident matches. Omit uncertain matches entirely."""

    def __init__(
        self, 
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
    ):
        """Initialize the LLM matcher.
        
        Args:
            api_key: OpenRouter API key (uses env var if not provided)
            model: Model to use for matching
        """
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")
        
        self.model = model
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        })
    
    def match_models(
        self,
        source_models: List[str],
        target_models: List[str],
        batch_size: int = 50,
        source_name: str = "Source",
        target_name: str = "Target",
    ) -> List[Dict]:
        """Match source model names to target model names using LLM.
        
        Args:
            source_models: List of model names from source (e.g., HuggingFace)
            target_models: List of model names from target (e.g., your cache)
            batch_size: Number of source models to process per LLM call
            source_name: Name of source for logging
            target_name: Name of target for logging
            
        Returns:
            List of matches: [{"source": "...", "target": "...", "confidence": "..."}]
        """
        all_matches = []
        
        # Process in batches
        for i in range(0, len(source_models), batch_size):
            batch = source_models[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1} ({len(batch)} models)")
            
            matches = self._match_batch(batch, target_models, source_name, target_name)
            all_matches.extend(matches)
            
            # Rate limiting
            time.sleep(1)
        
        logger.info(f"Found {len(all_matches)} matches total")
        return all_matches
    
    def _match_batch(
        self,
        source_batch: List[str],
        target_models: List[str],
        source_name: str,
        target_name: str,
    ) -> List[Dict]:
        """Match a batch of source models to targets.
        
        Args:
            source_batch: Batch of source model names
            target_models: All target model names
            source_name: Name of source
            target_name: Name of target
            
        Returns:
            List of matches for this batch
        """
        user_prompt = f"""Match these {source_name} model names to {target_name} model names.

{source_name.upper()} MODELS (to match FROM):
{json.dumps(source_batch, indent=2)}

{target_name.upper()} MODELS (to match TO):
{json.dumps(target_models, indent=2)}

Return JSON array of matches. Only include clear matches."""

        try:
            response = self.session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0,
                    "max_tokens": 4096,
                },
                timeout=60,
            )
            response.raise_for_status()
            
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            
            # Parse JSON from response
            # Handle markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            matches = json.loads(content.strip())
            
            # Validate matches
            valid_matches = []
            target_set = set(target_models)
            source_set = set(source_batch)
            
            for match in matches:
                if match.get("source") in source_set and match.get("target") in target_set:
                    valid_matches.append(match)
                else:
                    logger.warning(f"Invalid match: {match}")
            
            return valid_matches
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.debug(f"Response content: {content}")
            return []
        except Exception as e:
            logger.error(f"Error in LLM matching: {e}")
            return []
    
    def match_and_apply(
        self,
        source_data: List[Dict],
        cache_models: List[Dict],
        source_name_key: str = "model_name",
        cache_name_key: str = "name",
        score_keys: List[str] = None,
        source_label: str = "Source",
    ) -> Tuple[List[Dict], int]:
        """Match source data to cache and apply scores.
        
        Args:
            source_data: List of dicts with model data from source
            cache_models: List of model dicts from cache
            source_name_key: Key for model name in source data
            cache_name_key: Key for model name in cache
            score_keys: List of keys to copy from source to cache
            source_label: Label for the source (for logging)
            
        Returns:
            Tuple of (updated cache models, number of matches)
        """
        # Get names
        source_names = [m.get(source_name_key, "") for m in source_data if m.get(source_name_key)]
        cache_names = [m.get(cache_name_key, "") for m in cache_models if m.get(cache_name_key)]
        
        logger.info(f"Matching {len(source_names)} {source_label} models to {len(cache_names)} cache models")
        
        # Get LLM matches
        matches = self.match_models(
            source_names, 
            cache_names,
            source_name=source_label,
            target_name="Cache",
        )
        
        # Create lookup tables
        source_lookup = {m.get(source_name_key): m for m in source_data}
        cache_lookup = {m.get(cache_name_key): m for m in cache_models}
        
        # Apply matches
        matched_count = 0
        for match in matches:
            source_name = match.get("source")
            target_name = match.get("target")
            confidence = match.get("confidence", "medium")
            
            if source_name in source_lookup and target_name in cache_lookup:
                source_model = source_lookup[source_name]
                cache_model = cache_lookup[target_name]
                
                # Copy score keys
                if score_keys:
                    for key in score_keys:
                        if key in source_model:
                            cache_model[key] = source_model[key]
                
                # Track source
                cache_model[f"{source_label.lower()}_match_source"] = source_name
                cache_model[f"{source_label.lower()}_match_confidence"] = confidence
                
                matched_count += 1
                logger.debug(f"Matched: {source_name} -> {target_name} ({confidence})")
        
        logger.info(f"Applied {matched_count} matches from {source_label}")
        return cache_models, matched_count


def llm_match_ifeval(
    ifeval_data: List[Dict],
    cache_models: List[Dict],
    api_key: Optional[str] = None,
) -> List[Dict]:
    """Use LLM to match IFEval models to cache.
    
    Args:
        ifeval_data: IFEval leaderboard data
        cache_models: Models from cache
        api_key: OpenRouter API key
        
    Returns:
        Updated cache models with IFEval scores
    """
    matcher = LLMModelMatcher(api_key=api_key)
    
    updated, count = matcher.match_and_apply(
        source_data=ifeval_data,
        cache_models=cache_models,
        source_name_key="model_name",
        cache_name_key="name",
        score_keys=["ifeval_score", "ifeval_raw", "bbh_score", "math_score", "gpqa_score"],
        source_label="IFEval",
    )
    
    # Also set source tracking
    for model in updated:
        if model.get("ifeval_match_source"):
            model["ifeval_score_source"] = model["ifeval_match_source"]
            model["ifeval_source"] = "open-llm-leaderboard"
    
    return updated


def llm_match_wildbench(
    wb_data: List[Dict],
    cache_models: List[Dict],
    api_key: Optional[str] = None,
) -> List[Dict]:
    """Use LLM to match WildBench models to cache.
    
    Args:
        wb_data: WildBench leaderboard data
        cache_models: Models from cache
        api_key: OpenRouter API key
        
    Returns:
        Updated cache models with WildBench scores
    """
    matcher = LLMModelMatcher(api_key=api_key)
    
    updated, count = matcher.match_and_apply(
        source_data=wb_data,
        cache_models=cache_models,
        source_name_key="model_name",
        cache_name_key="name",
        score_keys=[
            "wb_score", "wb_elo", "wb_creative_tasks", "wb_coding_debugging",
            "wb_planning_reasoning", "wb_information_seeking", "wb_math_data_analysis",
        ],
        source_label="WildBench",
    )
    
    # Also set source tracking
    for model in updated:
        if model.get("wildbench_match_source"):
            model["wb_score_source"] = model["wildbench_match_source"]
    
    return updated


def llm_match_arena_hard(
    arena_data: List[Dict],
    cache_models: List[Dict],
    api_key: Optional[str] = None,
) -> List[Dict]:
    """Use LLM to match Arena-Hard-Auto models to cache.
    
    Args:
        arena_data: Arena-Hard-Auto leaderboard data
        cache_models: Models from cache
        api_key: OpenRouter API key
        
    Returns:
        Updated cache models with Arena-Hard-Auto scores
    """
    matcher = LLMModelMatcher(api_key=api_key)
    
    updated, count = matcher.match_and_apply(
        source_data=arena_data,
        cache_models=cache_models,
        source_name_key="model_name",
        cache_name_key="name",
        score_keys=["arena_hard_auto_score", "arena_hard_auto_ci_lower", "arena_hard_auto_ci_upper"],
        source_label="ArenaHard",
    )
    
    # Also set source tracking
    for model in updated:
        if model.get("arenahard_match_source"):
            model["arena_hard_auto_score_source"] = model["arenahard_match_source"]
    
    return updated

