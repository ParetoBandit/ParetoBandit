"""
Scraper for HuggingFace Open LLM Leaderboard.

Collects:
- MMLU, GPQA, MATH, IFEval benchmark scores
- Parameter counts
- Model metadata from HuggingFace leaderboard
"""

from typing import Dict, List, Optional
from datasets import load_dataset
import pandas as pd
import logging

from .base_scraper import BaseScraper

logger = logging.getLogger(__name__)


class HuggingFaceLeaderboardScraper(BaseScraper):
    """Scrape HuggingFace Open LLM Leaderboard for benchmark scores."""
    
    LEADERBOARD_DATASET = "open-llm-leaderboard/contents"
    
    def __init__(self, known_models: Optional[List[str]] = None, rate_limit_delay: float = 1.0):
        """
        Initialize scraper.
        
        Args:
            known_models: List of known model names from OpenRouter (for matching)
            rate_limit_delay: Seconds between requests
        """
        super().__init__(rate_limit_delay)
        self.known_models = known_models or []
    
    def get_source_name(self) -> str:
        return "HuggingFace Open LLM Leaderboard"
    
    def scrape(self) -> List[Dict]:
        """
        Scrape HuggingFace Open LLM Leaderboard.
        
        Returns:
            List of dicts with: model_name, mmlu_score, gpqa_score, 
                               math_score, ifeval_score, param_count_b
        """
        logger.info(f"Scraping {self.get_source_name()}...")
        
        try:
            # Load the leaderboard dataset
            logger.info("  Loading HuggingFace leaderboard dataset...")
            ds = load_dataset(self.LEADERBOARD_DATASET, split="train", trust_remote_code=True)
            df = ds.to_pandas()
            
            logger.info(f"  ✅ Loaded {len(df)} models from leaderboard")
            
            # Parse and normalize the data
            models = self._parse_leaderboard(df)
            
            logger.info(f"  ✅ Parsed {len(models)} models with benchmarks")
            return models
            
        except Exception as e:
            logger.error(f"  ❌ Failed to scrape HuggingFace leaderboard: {e}")
            logger.error(f"     Error details: {type(e).__name__}")
            return []
    
    def _parse_leaderboard(self, df: pd.DataFrame) -> List[Dict]:
        """Parse HuggingFace leaderboard DataFrame."""
        models = []
        
        # Map column names (HF leaderboard format)
        column_mapping = {
            'fullname': 'model_name',
            'IFEval': 'ifeval_score',
            'GPQA': 'gpqa_score',
            'MATH Lvl 5': 'math_score',
            'MMLU-PRO': 'mmlu_score',
            '#Params (B)': 'param_count_b',
        }
        
        # Rename columns
        df_renamed = df.rename(columns=column_mapping)
        
        # Filter to rows with meaningful data (at least one non-zero score)
        # This avoids processing 4576 rows one by one
        df_valid = df_renamed[
            (df_renamed['mmlu_score'].notna() & (df_renamed['mmlu_score'] > 0)) |
            (df_renamed['gpqa_score'].notna() & (df_renamed['gpqa_score'] > 0)) |
            (df_renamed['math_score'].notna() & (df_renamed['math_score'] > 0)) |
            (df_renamed['ifeval_score'].notna() & (df_renamed['ifeval_score'] > 0))
        ]
        
        logger.info(f"  Found {len(df_valid)} models with benchmark scores")
        
        # Process each model
        for idx, row in df_valid.iterrows():
            try:
                model_data = self._parse_model_row(row)
                if model_data:
                    models.append(model_data)
                    
                    # Progress indicator
                    if len(models) % 500 == 0:
                        logger.info(f"    Processed {len(models)} models...")
                        
            except Exception as e:
                logger.debug(f"  Failed to parse row {idx}: {e}")
                continue
        
        return models
    
    def _parse_model_row(self, row: pd.Series) -> Optional[Dict]:
        """Parse a single model row from the leaderboard."""
        # Get model name - use direct indexing for pandas Series
        model_name = row['model_name'] if 'model_name' in row.index else None
        
        if pd.isna(model_name) or not model_name:
            return None
        
        # Extract benchmark scores - use direct indexing
        mmlu_score = self._safe_float(row['mmlu_score']) if 'mmlu_score' in row.index else 0.0
        gpqa_score = self._safe_float(row['gpqa_score']) if 'gpqa_score' in row.index else 0.0
        math_score = self._safe_float(row['math_score']) if 'math_score' in row.index else 0.0
        ifeval_score = self._safe_float(row['ifeval_score']) if 'ifeval_score' in row.index else 0.0
        param_count = self._safe_float(row['param_count_b']) if 'param_count_b' in row.index else 0.0
        
        # Skip if no meaningful data (all zeros)
        if all(score == 0 for score in [mmlu_score, gpqa_score, math_score, ifeval_score]):
            return None
        
        # Match to OpenRouter canonical name
        matched_name = self._match_model_name(str(model_name))
        
        return {
            'model_name': matched_name,
            'hf_model_name': str(model_name),  # Keep original HF name for reference
            'mmlu_score': mmlu_score,
            'gpqa_score': gpqa_score,
            'math_score': math_score,
            'ifeval_score': ifeval_score,
            'param_count_b': param_count,
            'source': self.get_source_name(),
        }
    
    def _safe_float(self, value) -> float:
        """Safely convert value to float."""
        if value is None:
            return 0.0
        
        if pd.isna(value):
            return 0.0
        
        try:
            result = float(value)
            return result if not pd.isna(result) else 0.0
        except (ValueError, TypeError):
            return 0.0
    
    def _match_model_name(self, hf_name: str) -> str:
        """
        Match HuggingFace model name to OpenRouter canonical name using fuzzy matching.
        
        Args:
            hf_name: HuggingFace model name (e.g., "meta-llama/Llama-3.1-70B-Instruct")
            
        Returns:
            Matched OpenRouter name or original HF name
        """
        if not self.known_models:
            # No known models, clean up HF name
            return self._clean_hf_name(hf_name)
        
        cleaned_hf = self._clean_hf_name(hf_name)
        
        # Try exact match first
        if cleaned_hf in self.known_models:
            return cleaned_hf
        
        # Try fuzzy matching with fuzzywuzzy
        try:
            from fuzzywuzzy import process
            best_match, score = process.extractOne(cleaned_hf, self.known_models)
            
            # If confidence is high enough (>85), use the match
            if score >= 85:
                logger.debug(f"    Fuzzy matched: {cleaned_hf} -> {best_match} (score: {score})")
                return best_match
        except ImportError:
            pass
        
        # Try partial matching by removing version numbers
        import re
        # Remove version patterns like "3.1", "v0.3", "2.5"
        hf_base = re.sub(r'[-.]?v?\d+\.\d+', '', cleaned_hf)
        
        for known in self.known_models:
            known_base = re.sub(r'[-.]?v?\d+\.\d+', '', known)
            if hf_base == known_base or (len(hf_base) > 10 and hf_base in known_base):
                logger.debug(f"    Version-agnostic match: {cleaned_hf} -> {known}")
                return known
        
        # No match found - return cleaned HF name as-is
        return cleaned_hf
    
    def _is_similar(self, name1: str, name2: str) -> bool:
        """Check if two normalized names are similar enough to match."""
        # Remove all non-alphanumeric except dots for version numbers
        import re
        clean1 = re.sub(r'[^a-z0-9.]', '', name1)
        clean2 = re.sub(r'[^a-z0-9.]', '', name2)
        
        # Check if one contains the other
        if clean1 in clean2 or clean2 in clean1:
            return True
        
        # Check if they share at least 70% of characters (fuzzy match)
        if len(clean1) > 5 and len(clean2) > 5:
            # Simple overlap check
            shorter = clean1 if len(clean1) < len(clean2) else clean2
            longer = clean2 if len(clean1) < len(clean2) else clean1
            
            # Count matching character sequences
            match_score = sum(1 for i in range(len(shorter)) if i < len(longer) and shorter[i] == longer[i])
            similarity = match_score / len(shorter)
            
            if similarity >= 0.7:
                return True
        
        return False
    
    def _clean_hf_name(self, hf_name: str) -> str:
        """
        Clean HuggingFace model name to match OpenRouter format.
        
        Examples:
        - "meta-llama/Llama-3.1-70B-Instruct" → "llama-3.1-70b-instruct"
        - "mistralai/Mistral-7B-v0.1" → "mistral-7b-v0.1"
        - "Qwen/Qwen2.5-VL-72B-Instruct" → "qwen2.5-vl-72b-instruct"
        """
        # Remove provider prefix
        if '/' in hf_name:
            name = hf_name.split('/')[-1]
        else:
            name = hf_name
        
        # Lowercase
        name = name.lower()
        
        # Normalize separators (replace underscores with hyphens for consistency)
        name = name.replace('_', '-')
        
        # Remove common prefixes/patterns
        patterns_to_remove = [
            'models-',
            'model-',
        ]
        for pattern in patterns_to_remove:
            if name.startswith(pattern):
                name = name[len(pattern):]
        
        # Remove common suffixes
        suffixes_to_remove = ['-hf', '-base', '-fp16', '-gptq', '-awq']
        for suffix in suffixes_to_remove:
            if name.endswith(suffix):
                name = name[:-len(suffix)]
        
        return name
    
    def _normalize_name(self, name: str) -> str:
        """Normalize model name for matching."""
        return name.lower().replace('-', '').replace('_', '').replace('/', '').replace('.', '')

