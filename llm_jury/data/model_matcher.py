#!/usr/bin/env python3
"""
Intelligent model matching between HuggingFace and OpenRouter.

Uses multiple strategies to match model names across different platforms:
1. Direct exact matching
2. Normalized name matching  
3. Fuzzy string matching
4. Alias-based matching
5. Parameter-based validation
6. Multi-word tokenization matching

This handles common naming variations like:
- Case differences (llama vs Llama vs LLAMA)
- Separator differences (- vs _ vs space)
- Namespace differences (meta-llama/llama vs llama)
- Version suffixes (v1 vs v2 vs -latest)
- Instruction variants (instruct vs chat vs base)
"""

import difflib
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class MatchResult:
    """Result of a model matching attempt."""
    openrouter_id: str
    confidence: float  # 0.0 to 1.0
    strategy: str
    hf_name: str
    or_name: str


class ModelMatcher:
    """Intelligent model matcher between HuggingFace and OpenRouter."""
    
    # Known model aliases and variations
    KNOWN_ALIASES = {
        # OpenAI variations
        'gpt-4': ['gpt4', 'gpt-4.0', 'gpt-4-turbo', 'chatgpt-4'],
        'gpt-4o': ['gpt4o', 'gpt-4-omni', 'gpt-4o-latest'],
        'gpt-4o-mini': ['gpt4o-mini', 'gpt-4-omni-mini'],
        'gpt-3.5-turbo': ['gpt35-turbo', 'gpt-3.5', 'chatgpt'],
        
        # Anthropic variations
        'claude-3-opus': ['claude-3-opus-20240229'],
        'claude-3-sonnet': ['claude-3-sonnet-20240229'],
        'claude-3-haiku': ['claude-3-haiku-20240307'],
        'claude-3.5-sonnet': ['claude-3.5-sonnet-20241022'],
        
        # Meta Llama variations
        'llama-2-7b': ['llama2-7b', 'llama-2-7b-hf'],
        'llama-2-13b': ['llama2-13b', 'llama-2-13b-hf'],
        'llama-2-70b': ['llama2-70b', 'llama-2-70b-hf'],
        'llama-3-8b': ['llama3-8b', 'llama-3-8b-hf'],
        'llama-3-70b': ['llama3-70b', 'llama-3-70b-hf'],
        'llama-3.1-8b': ['llama3.1-8b', 'llama-3.1-8b-hf'],
        'llama-3.1-70b': ['llama3.1-70b', 'llama-3.1-70b-hf'],
        'llama-3.1-405b': ['llama3.1-405b', 'llama-3.1-405b-hf'],
        'llama-3.3-70b': ['llama3.3-70b'],
        
        # Google variations
        'gemini-pro': ['gemini-1.0-pro', 'gemini-pro-latest'],
        'gemini-1.5-pro': ['gemini-1.5-pro-latest'],
        'gemini-1.5-flash': ['gemini-1.5-flash-latest'],
        
        # Mistral variations
        'mistral-7b': ['mistral-7b-v0.1', 'mistral-7b-instruct'],
        'mixtral-8x7b': ['mixtral-8x7b-instruct'],
        'mixtral-8x22b': ['mixtral-8x22b-instruct'],
        
        # Qwen variations  
        'qwen-2.5-72b': ['qwen2.5-72b', 'qwen-2.5-72b-instruct'],
        'qwen-2.5-32b': ['qwen2.5-32b', 'qwen-2.5-32b-instruct'],
        'qwq-32b': ['qwen-qwq-32b', 'qwen-qwq-32b-preview'],
        
        # DeepSeek variations
        'deepseek-v3': ['deepseek-v3-base'],
        'deepseek-r1': ['deepseek-r1-lite-preview'],
        'deepseek-coder-v2': ['deepseek-coder-v2-instruct'],
    }
    
    # Namespace mappings (HF -> OpenRouter patterns)
    NAMESPACE_MAPPINGS = {
        'meta-llama': ['meta-llama', 'llama', 'facebook'],
        'microsoft': ['microsoft', 'msft'],
        'google': ['google', 'googleai'],
        'anthropic': ['anthropic'],
        'openai': ['openai'],
        'mistralai': ['mistralai', 'mistral'],
        'qwen': ['qwen', 'alibaba'],
        'deepseek-ai': ['deepseek-ai', 'deepseek'],
        'nousresearch': ['nousresearch', 'nous'],
        'togethercomputer': ['together', 'togethercomputer'],
        '01-ai': ['01-ai', 'yi'],
    }
    
    def __init__(self):
        self._alias_map = self._build_alias_map()
    
    def _build_alias_map(self) -> Dict[str, str]:
        """Build bidirectional alias mapping."""
        alias_map = {}
        for canonical, aliases in self.KNOWN_ALIASES.items():
            alias_map[canonical] = canonical
            for alias in aliases:
                alias_map[alias] = canonical
        return alias_map
    
    def normalize_name(self, name: str) -> str:
        """
        Normalize model name for matching.
        
        - Convert to lowercase
        - Replace underscores and spaces with hyphens
        - Remove common suffixes
        - Handle version numbers consistently
        """
        name = name.lower().strip()
        
        # Replace separators
        name = re.sub(r'[_\s]+', '-', name)
        
        # Handle version numbers
        name = re.sub(r'[-.]v(\d+)', r'-v\1', name)  # .v1 -> -v1
        name = re.sub(r'[-.](\d+)b\b', r'-\1b', name)  # .70b -> -70b
        
        # Remove common suffixes that don't affect matching
        suffixes_to_remove = [
            '-hf', '-huggingface', '-base', '-latest', '-preview', 
            '-instruct', '-chat', '-it', '-dpo', '-sft'
        ]
        
        for suffix in suffixes_to_remove:
            if name.endswith(suffix):
                name = name[:-len(suffix)]
                break
        
        return name
    
    def extract_model_tokens(self, name: str) -> List[str]:
        """Extract meaningful tokens from model name."""
        # Remove namespace/organization
        if '/' in name:
            name = name.split('/')[-1]
        
        # Normalize
        name = self.normalize_name(name)
        
        # Split on common separators and extract tokens
        tokens = re.split(r'[-_\s]+', name)
        
        # Filter out very short tokens unless they're numbers
        filtered_tokens = []
        for token in tokens:
            if len(token) >= 2 or token.isdigit():
                filtered_tokens.append(token)
        
        return filtered_tokens
    
    def calculate_token_similarity(self, hf_tokens: List[str], or_tokens: List[str]) -> float:
        """Calculate similarity based on token overlap."""
        if not hf_tokens or not or_tokens:
            return 0.0
        
        hf_set = set(hf_tokens)
        or_set = set(or_tokens)
        
        intersection = hf_set & or_set
        union = hf_set | or_set
        
        if not union:
            return 0.0
        
        # Jaccard similarity with bonus for key tokens
        jaccard = len(intersection) / len(union)
        
        # Bonus for important tokens (model family, size)
        important_tokens = {'llama', 'gpt', 'claude', 'gemini', 'qwen', 'mistral', 'deepseek'}
        important_matches = len(intersection & important_tokens)
        
        size_tokens = [t for t in intersection if re.match(r'\d+[bm]?$', t)]
        size_bonus = 0.1 if size_tokens else 0
        
        return min(1.0, jaccard + (important_matches * 0.1) + size_bonus)
    
    def find_best_match(self, hf_name: str, or_models: Dict[str, Any]) -> Optional[MatchResult]:
        """
        Find the best OpenRouter match for a HuggingFace model.
        
        Uses multiple strategies in order of confidence:
        1. Exact match
        2. Normalized exact match
        3. Alias match
        4. High-confidence fuzzy match
        5. Token-based match
        """
        if not or_models:
            return None
        
        hf_name_clean = hf_name.lower().strip()
        or_ids = list(or_models.keys())
        
        # Strategy 1: Exact match
        for or_id in or_ids:
            if or_id.lower() == hf_name_clean:
                return MatchResult(or_id, 1.0, "exact_match", hf_name, or_id)
        
        # Strategy 2: Normalized exact match
        hf_normalized = self.normalize_name(hf_name)
        for or_id in or_ids:
            or_normalized = self.normalize_name(or_id)
            if or_normalized == hf_normalized:
                return MatchResult(or_id, 0.95, "normalized_exact", hf_name, or_id)
        
        # Strategy 3: Check if HF name is contained in OR name (namespace match)
        for or_id in or_ids:
            if hf_name_clean in or_id.lower():
                return MatchResult(or_id, 0.9, "containment_match", hf_name, or_id)
        
        # Strategy 4: Check if OR name is contained in HF name
        hf_model_only = hf_name.split('/')[-1].lower() if '/' in hf_name else hf_name_clean
        for or_id in or_ids:
            or_model_only = or_id.split('/')[-1].lower() if '/' in or_id else or_id.lower()
            if or_model_only == hf_model_only:
                return MatchResult(or_id, 0.85, "model_name_match", hf_name, or_id)
        
        # Strategy 5: Alias matching
        hf_alias_key = self._alias_map.get(hf_normalized)
        if hf_alias_key:
            for or_id in or_ids:
                or_alias_key = self._alias_map.get(self.normalize_name(or_id))
                if or_alias_key == hf_alias_key:
                    return MatchResult(or_id, 0.8, "alias_match", hf_name, or_id)
        
        # Strategy 6: High-confidence fuzzy match
        hf_for_fuzzy = hf_name.split('/')[-1] if '/' in hf_name else hf_name
        fuzzy_matches = difflib.get_close_matches(
            hf_for_fuzzy.lower(), 
            [or_id.lower() for or_id in or_ids], 
            n=1, 
            cutoff=0.8
        )
        
        if fuzzy_matches:
            # Find the original OR ID
            for or_id in or_ids:
                if or_id.lower() == fuzzy_matches[0]:
                    similarity = difflib.SequenceMatcher(
                        None, hf_for_fuzzy.lower(), or_id.lower()
                    ).ratio()
                    return MatchResult(or_id, similarity, "fuzzy_match", hf_name, or_id)
        
        # Strategy 7: Token-based matching
        hf_tokens = self.extract_model_tokens(hf_name)
        best_token_match = None
        best_token_score = 0.0
        
        for or_id in or_ids:
            or_tokens = self.extract_model_tokens(or_id)
            token_sim = self.calculate_token_similarity(hf_tokens, or_tokens)
            
            if token_sim > best_token_score and token_sim >= 0.6:  # Minimum threshold
                best_token_score = token_sim
                best_token_match = or_id
        
        if best_token_match:
            return MatchResult(best_token_match, best_token_score, "token_match", hf_name, best_token_match)
        
        # Strategy 8: Namespace + model matching
        if '/' in hf_name:
            hf_org, hf_model = hf_name.split('/', 1)
            hf_org_normalized = self.normalize_name(hf_org)
            
            # Check if we have namespace mappings
            possible_namespaces = []
            for ns, variants in self.NAMESPACE_MAPPINGS.items():
                if hf_org_normalized in [self.normalize_name(v) for v in variants]:
                    possible_namespaces.extend(variants)
            
            if possible_namespaces:
                hf_model_normalized = self.normalize_name(hf_model)
                for or_id in or_ids:
                    if '/' in or_id:
                        or_org, or_model = or_id.split('/', 1)
                        or_org_normalized = self.normalize_name(or_org)
                        or_model_normalized = self.normalize_name(or_model)
                        
                        if (or_org_normalized in [self.normalize_name(ns) for ns in possible_namespaces] and
                            or_model_normalized == hf_model_normalized):
                            return MatchResult(or_id, 0.75, "namespace_match", hf_name, or_id)
        
        return None
    
    def match_models(self, hf_models: List[str], or_models: Dict[str, Any]) -> Dict[str, MatchResult]:
        """
        Match a list of HuggingFace models to OpenRouter models.
        
        Returns:
            Dict mapping HF model names to their best OR matches
        """
        matches = {}
        
        for hf_model in hf_models:
            match = self.find_best_match(hf_model, or_models)
            if match:
                matches[hf_model] = match
        
        return matches
    
    def get_match_statistics(self, matches: Dict[str, MatchResult]) -> Dict[str, Any]:
        """Get statistics about matching results."""
        if not matches:
            return {}
        
        strategy_counts = {}
        confidence_scores = []
        
        for match in matches.values():
            strategy_counts[match.strategy] = strategy_counts.get(match.strategy, 0) + 1
            confidence_scores.append(match.confidence)
        
        return {
            'total_matches': len(matches),
            'avg_confidence': sum(confidence_scores) / len(confidence_scores),
            'min_confidence': min(confidence_scores),
            'max_confidence': max(confidence_scores),
            'strategy_breakdown': strategy_counts,
            'high_confidence_matches': len([c for c in confidence_scores if c >= 0.8]),
            'medium_confidence_matches': len([c for c in confidence_scores if 0.6 <= c < 0.8]),
            'low_confidence_matches': len([c for c in confidence_scores if c < 0.6]),
        }


def test_matcher():
    """Test the model matcher with some examples."""
    print("="*80)
    print("🧪 Testing Improved Model Matcher")
    print("="*80)
    
    matcher = ModelMatcher()
    
    # Sample HuggingFace models
    hf_models = [
        "meta-llama/Llama-3-8B-Instruct",
        "meta-llama/Llama-3.1-70B",
        "Qwen/QwQ-32B",
        "microsoft/Phi-3-medium-4k-instruct",
        "NousResearch/Hermes-2-Pro-Llama-3-8B",
        "deepseek-ai/DeepSeek-V3",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "01-ai/Yi-1.5-34B-Chat"
    ]
    
    # Mock OpenRouter models (simplified)
    or_models = {
        "meta-llama/llama-3-8b-instruct": {"pricing": {"prompt": "0.0001", "completion": "0.0002"}},
        "meta-llama/llama-3.1-70b": {"pricing": {"prompt": "0.0005", "completion": "0.001"}},
        "qwen/qwq-32b": {"pricing": {"prompt": "0.0003", "completion": "0.0006"}},
        "microsoft/phi-3-medium-4k-instruct": {"pricing": {"prompt": "0.0002", "completion": "0.0004"}},
        "nousresearch/hermes-2-pro-llama-3-8b": {"pricing": {"prompt": "0.0001", "completion": "0.0002"}},
        "deepseek-ai/deepseek-v3": {"pricing": {"prompt": "0.0001", "completion": "0.0002"}},
        "mistralai/mixtral-8x7b-instruct": {"pricing": {"prompt": "0.0004", "completion": "0.0008"}},
        "01-ai/yi-1.5-34b-chat": {"pricing": {"prompt": "0.0003", "completion": "0.0006"}},
        
        # Additional OR models that might match
        "openai/gpt-4o": {"pricing": {"prompt": "0.005", "completion": "0.015"}},
        "anthropic/claude-3-sonnet": {"pricing": {"prompt": "0.003", "completion": "0.015"}},
        "google/gemini-1.5-pro": {"pricing": {"prompt": "0.002", "completion": "0.006"}},
    }
    
    print(f"📊 Matching {len(hf_models)} HF models against {len(or_models)} OR models...")
    
    matches = matcher.match_models(hf_models, or_models)
    
    print(f"\n✅ Found {len(matches)} matches:")
    for hf_model, match in matches.items():
        print(f"\n🔗 {hf_model}")
        print(f"   → {match.openrouter_id}")
        print(f"   Confidence: {match.confidence:.2f}")
        print(f"   Strategy: {match.strategy}")
    
    # Show unmatched models
    unmatched = [m for m in hf_models if m not in matches]
    if unmatched:
        print(f"\n❌ Unmatched models ({len(unmatched)}):")
        for model in unmatched:
            print(f"   - {model}")
    
    # Statistics
    stats = matcher.get_match_statistics(matches)
    print(f"\n📈 Matching Statistics:")
    print(f"   Total matches: {stats['total_matches']}")
    print(f"   Average confidence: {stats['avg_confidence']:.3f}")
    print(f"   High confidence (≥0.8): {stats['high_confidence_matches']}")
    print(f"   Medium confidence (0.6-0.8): {stats['medium_confidence_matches']}")
    print(f"   Low confidence (<0.6): {stats['low_confidence_matches']}")
    
    print(f"\n📋 Strategy breakdown:")
    for strategy, count in stats['strategy_breakdown'].items():
        print(f"   {strategy}: {count}")


if __name__ == "__main__":
    test_matcher()