"""
Feature Engineering (Regex/Transforms)

Feature extraction and transformation utilities for the BanditRouter.
Implements linearization strategies to ensure LinUCB linearity assumption compliance.

**KDD Review Critique: "The Linearity Assumption"**
LinUCB assumes Reward ≈ θ · x (linear relationship). Features like 'latex_density'
combine two distinct signals:
1. Step Function: "Is this math?" (massive jump when LaTeX present)
2. Continuous Slope: "How hard is the math?" (gradual increase with density)

A single linear coefficient cannot capture both. Solution: Split into two features:
- Binary: Captures the intercept shift (the "step")
- Log-scaled: Captures the incremental difficulty (the "slope")

This allows: Reward = θ_step * has_feature + θ_slope * log(density)
"""

from __future__ import annotations

import re
from typing import Tuple, Optional

import numpy as np


def _fast_toxicity_heuristic(text: str) -> float:
    """
    Ultra-fast regex-based toxicity proxy (<1ms).
    
    Replaces heavy ML scanner (llm-guard, 100-300ms) in hot path.
    Heavy scanner moved to async audit (Tier 2).
    
    Pattern-based triggers derived from common toxicity categories:
    - Violence: Physical harm language
    - Hate Speech: Discrimination, slurs
    - Explicit Content: Sexual/graphic content
    - Security Threats: Hacking, exploitation
    
    Returns:
        Toxicity score in [0, 1], compatible with LinUCB feature vector
        
    Performance: <1ms (vs 100-300ms for ML scanner)
    
    Production Note: For scale, replace with Bloom Filter (O(1) lookup)
    """
    if not text:
        return 0.0
    
    # Trigger patterns by category
    triggers = {
        'violence': ['kill', 'attack', 'murder', 'destroy', 'shoot', 'stab', 'bomb', 'weapon'],
        'hate': ['hate', 'racist', 'nazi', 'terrorist', 'slur', 'bigot'],
        'explicit': ['porn', 'xxx', 'sex', 'nude', 'nsfw'],
        'security': ['hack', 'exploit', 'crack', 'stolen', 'leak', 'bypass', 'jailbreak'],
        'self_harm': ['suicide', 'self-harm', 'cutting']
    }
    
    text_lower = text.lower()
    score = 0.0
    
    # Score accumulation (each trigger adds 0.15, caps at 1.0)
    for category, words in triggers.items():
        for word in words:
            if word in text_lower:
                score += 0.15
                break  # Only count category once
    
    return min(1.0, score)


class FeatureTransformer:
    """
    Linearizes non-linear signals for LinUCB compatibility.
    
    **CRITICAL: Numerical Stability**
    All features MUST be normalized to [0,1] to prevent matrix inversion instability.
    Raw log values can range from 0 to 10+, causing numerical issues.
    """
    
    @staticmethod
    def binarize(x: float) -> float:
        """Convert to presence indicator (0 or 1)."""
        return 1.0 if x > 0 else 0.0
    
    @staticmethod
    def log1p(x: float) -> float:
        """Log-scale for continuous intensity: log(1 + x)."""
        return float(np.log1p(max(0, x)))
    
    @staticmethod
    def normalize_log(log_value: float, max_expected: float = 10.0) -> float:
        """
        Normalize log-scaled values to [0,1] for numerical stability.
        
        Args:
            log_value: The log-transformed value
            max_expected: Expected maximum (e.g., log(20000) ≈ 10 for token counts)
        
        Returns:
            Normalized value in [0,1]
        
        **Why this matters:**
        LinUCB inverts the matrix A. If features have wildly different scales
        (e.g., binary=1, log_length=10), the matrix becomes ill-conditioned.
        Normalization prevents numerical instability.
        """
        return float(np.clip(log_value / max_expected, 0.0, 1.0))
    
    @staticmethod
    def split_signal(raw_count: float, max_log: float = 5.0) -> Tuple[float, float]:
        """
        Split a raw count into linearizable AND normalized components.
        
        Args:
            raw_count: The raw count value
            max_log: Maximum expected log value for normalization
        
        Returns:
            (binary_presence, normalized_log_intensity)
        
        Example:
            0 → (0.0, 0.0)
            1 → (1.0, 0.14)  # log(2)/5 ≈ 0.14
            10 → (1.0, 0.48) # log(11)/5 ≈ 0.48
            100 → (1.0, 0.92) # log(101)/5 ≈ 0.92
        """
        # Inline to avoid NameError in static method
        binary = 1.0 if raw_count > 0 else 0.0
        log_val = float(np.log1p(max(0, raw_count)))
        normalized = float(np.clip(log_val / max_log, 0.0, 1.0))
        return (binary, normalized)


class FeatureExtractor:
    """
    Extracts handcrafted features from text prompts.
    
    **SIMPLIFIED FEATURE SET (Based on KDD Feature Significance Analysis):**
    Analysis showed that most handcrafted features have minimal predictive power.
    Keeping only the most significant feature to reduce dimensionality and improve
    sample efficiency.
    
    **1 Feature Total:**
    1. length_penalty_log: Normalized log-scaled prompt length [0,1]
       - Statistically significant predictor (p < 0.05) for multiple models
       - Captures prompt complexity proxy with minimal overhead
    
    **Removed Features (low significance):**
    - is_code_heavy, requires_json, list_density, instruction_density
    - flesch_kincaid, toxicity_score
    - has_code_block, code_block_count_log
    - has_latex, latex_density_log
    - has_question, question_count_log
    - length_penalty_bin (redundant with log version)
    """
    
    def __init__(self, toxicity_scanner=None):
        """
        Initialize feature extractor.
        
        Args:
            toxicity_scanner: Optional toxicity scanner instance for safety features
        """
        self._toxicity_scanner = toxicity_scanner
    
    @staticmethod
    def count_syllables(word: str) -> int:
        """Heuristic syllable counter for Flesch-Kincaid."""
        word = word.lower().strip(".:;?!")
        if not word: return 0
        if len(word) <= 3: return 1
        
        # Count vowel groups
        count = len(re.findall(r'[aeiouy]+', word))
        # Subtract silent 'e' at end
        if word.endswith('e'):
            count -= 1
        # Subtract consecutive vowels (already handled by regex group)
        return max(1, count)
    
    def extract_features(self, text: str) -> np.ndarray:
        """
        Extract simplified features for routing logic.
        
        **SIMPLIFIED FEATURE SET:**
        Based on feature significance analysis, keeping only the most predictive feature.
        
        Args:
            text: Input text prompt
        
        Returns:
            Feature vector of shape (1,) containing:
            - length_penalty_log: Normalized log-scaled prompt length
        """
        if not text:
            return np.zeros(1)
        
        # Count tokens (rough estimate: words * 1.3)
        words = re.findall(r'\b\w+\b', text.lower())
        n_tokens = len(words) * 1.3
        
        # Length: Log-scaled length, normalized to [0,1]
        # Max expected: log(10000) ≈ 9.2
        length_penalty_log = FeatureTransformer.normalize_log(
            np.log1p(n_tokens), max_expected=10.0
        )
        
        return np.array([length_penalty_log])

    def extract_trap_features(self, text: str) -> np.ndarray:
        """
        Extract trap detection features for arbitrage routing.
        
        These features create "shortcuts" for the bandit to detect failure modes
        where cheap models (gpt-oss, gemma) fail but flagships succeed.
        
        **Single Source of Truth:**
        This method is used by BOTH router.py and generate_warmup.py to ensure
        identical feature extraction logic. This prevents "simulation gaps" where
        the warmup trains on different signals than the router uses.
        
        **6 Features Total:**
        1. is_non_english (binary): Non-ASCII characters (Korean, Chinese, etc.)
        2. is_adversarial (binary): Jailbreak/system override patterns
        3. is_tool_use (binary): Real-time/search requests
        4. log_length (continuous): Log-normalized prompt length
        5. code_density (continuous): Code character density
        6. math_density (continuous): Math indicator density
        
        Args:
            text: Input text prompt
            
        Returns:
            6-dimensional feature vector for trap detection
        """
        if not text:
            return np.zeros(6)
        
        text_lower = text.lower()
        
        # 1. Kill Switches (Binary Flags)
        # Non-English detection (often breaks OSS tokenization)
        # CRITICAL: Must match router.py logic exactly
        is_non_english = 1.0 if any(ord(c) > 255 for c in text[:1000]) else 0.0
        
        # Adversarial pattern detection
        jailbreak_patterns = [
            r"ignore previous", r"system instruction", r"role play", 
            r"act as", r"jailbreak", r"unfiltered", r"system:"
        ]
        is_adversarial = 1.0 if any(re.search(p, text_lower) for p in jailbreak_patterns) else 0.0
        
        # Tool use / real-time query detection
        tool_patterns = [
            r"weather", r"stock price", r"current time", 
            r"latest news", r"search for", r"who won",
            r"date today", r"what is the date", r"current", 
            r"price of", r"who is the president"
        ]
        is_tool_use = 1.0 if any(re.search(p, text_lower) for p in tool_patterns) else 0.0
        
        # 2. Complexity Proxies (Continuous)
        # Log-normalized length (stabilizes variance)
        log_length = np.log(len(text) + 1.0) / 10.0  # Scale to ~0-1 range
        
        # Code density heuristic
        code_chars = set("{}[]();=<>!_")
        code_count = sum(1 for c in text if c in code_chars)
        code_density = code_count / (len(text) + 1)
        
        # Math density (LaTeX-ish)
        math_indicators = ["sum", "int", "frac", "sqrt", "theta", "pi", "=", "+", "-"]
        math_count = sum(text_lower.count(m) for m in math_indicators)
        words = text_lower.split()
        math_density = math_count / (len(words) + 1)  # Normalize by word count
        
        return np.array([
            is_non_english,
            is_adversarial,
            is_tool_use,
            log_length,
            code_density,
            math_density
        ], dtype=np.float32)

