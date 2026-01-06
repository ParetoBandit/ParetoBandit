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
    
    **LINEARIZATION STRATEGY (Addressing KDD Critique):**
    Features that combine step functions + continuous slopes are split:
    - Binary: Presence indicator (captures intercept shift)
    - Log-scaled: Intensity (captures gradual difficulty increase)
    
    **14 Features Total:**
    1. is_code_heavy (continuous: code length / total length)
    2. requires_json (binary: JSON keyword presence)
    3. list_density (continuous: list items / lines)
    4. instruction_density (continuous: imperatives / words)
    5. flesch_kincaid (continuous: reading grade level)
    6. toxicity_score (continuous: LLM Guard score)
    7-8. Code blocks: has_code_block (binary) + code_block_count_log (continuous)
    9-10. LaTeX: has_latex (binary) + latex_density_log (continuous)
    11-12. Questions: has_question (binary) + question_count_log (continuous)
    13-14. Length: length_penalty_bin (binary: >500 tokens) + length_penalty_log (continuous)
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
        Extract linearized features for routing logic.
        
        Args:
            text: Input text prompt
        
        Returns:
            Feature vector of shape (14,) containing:
            - 6 continuous features
            - 4 binary/log pairs (8 features total)
        """
        if not text:
            return np.zeros(14)
        
        # --- BASICS ---
        total_len = len(text)
        words = re.findall(r'\b\w+\b', text.lower())
        n_words = len(words)
        lines = text.split('\n')
        n_lines = len(lines)
        n_tokens = n_words * 1.3
        
        # 1. Code Heavy (continuous)
        code_blocks = re.findall(r'`{1,3}(.*?)`{1,3}', text, re.DOTALL)
        code_len = sum(len(c) for c in code_blocks)
        is_code_heavy = (code_len / total_len) if total_len > 0 else 0.0
        
        # 2. Requires JSON (binary)
        json_keywords = ["json", "valid format", "schema", "output format"]
        requires_json = 1.0 if any(k in text.lower() for k in json_keywords) else 0.0
        
        # 3. List Density (continuous)
        list_markers = [l for l in lines if l.strip().startswith(('-', '*', '1.', '2.'))]
        list_density = (len(list_markers) / n_lines) if n_lines > 0 else 0.0
        
        # --- COMPLEXITY ---
        
        # 4. Instruction Density (continuous)
        imperatives = {"create", "write", "solve", "analyze", "explain", "summarize", "find", "calculate", "implement", "design"}
        n_imperatives = sum(1 for w in words if w in imperatives)
        instruction_density = (n_imperatives / n_words) if n_words > 0 else 0.0
        
        # 5. Flesch-Kincaid Grade (continuous)
        sentences = re.split(r'[.!?]+', text)
        n_sentences = max(1, len([s for s in sentences if s.strip()]))
        
        if n_words > 0:
            n_syllables = sum(self.count_syllables(w) for w in words)
            fk_grade = 0.39 * (n_words / n_sentences) + 11.8 * (n_syllables / n_words) - 15.59
        else:
            fk_grade = 0.0
        fk_normalized = max(0.0, min(fk_grade, 20.0)) / 20.0
        
        # --- SECURITY ---
        
        # 6. Toxicity Score (continuous)
        # TIER 1: Fast heuristic (<1ms) for feature vector
        # Heavy ML scanner (llm-guard) moved to async audit (Tier 2)
        from bandit_gpt.router import BanditRouter
        toxicity_score = BanditRouter._fast_toxicity_heuristic(text)
        
        # --- LINEARIZED FEATURES (Split Step + Slope) ---
        
        # 7-8. Code Blocks: Binary presence + Log intensity
        code_block_count = float(text.count('```'))
        has_code_block, code_block_count_log = FeatureTransformer.split_signal(code_block_count)
        
        # 9-10. LaTeX Symbols: Binary presence + Log density
        latex_count = float(text.count('$') + text.count('\\') + text.count('^') + text.count('_{'))
        has_latex, latex_density_log = FeatureTransformer.split_signal(latex_count)
        
        # 11-12. Questions: Binary presence + Log count
        question_count = float(text.count('?'))
        has_question, question_count_log = FeatureTransformer.split_signal(question_count)
        
        # 13-14. Length: Binary threshold + Log scaling (normalized)
        # Binary: Is this a "long" prompt? (>500 tokens)
        length_penalty_bin = 1.0 if n_tokens > 500 else 0.0
        # Continuous: Log-scaled length, normalized to [0,1]
        # Max expected: log(10000) ≈ 9.2
        length_penalty_log = FeatureTransformer.normalize_log(
            np.log1p(n_tokens), max_expected=10.0
        )
        
        return np.array([
            # Original continuous features (1-6)
            is_code_heavy, requires_json, list_density,
            instruction_density, fk_normalized, toxicity_score,
            # Linearized features (7-14): Binary + Log pairs
            has_code_block, code_block_count_log,
            has_latex, latex_density_log,
            has_question, question_count_log,
            length_penalty_bin, length_penalty_log
        ])
