"""
Safety Classifier Module for BanditGPT
======================================
Uses Meta's LLaMA Guard 3 (8B) for prompt safety classification.
Maps risk categories to sensitivity levels for the Risk Gating architecture.

Model: meta-llama/Llama-Guard-3-8B
License: Llama 3.1 Community License (Open Source)

Usage:
    from safety_classifier import SafetyClassifier
    classifier = SafetyClassifier()
    level = classifier.classify("What is the dosage for ibuprofen?")
    # Returns: "HIGH"
"""

import logging
from enum import Enum
from typing import Optional, Tuple
import torch

logger = logging.getLogger(__name__)

class SensitivityLevel(Enum):
    """Three-tier sensitivity classification for Risk Gating."""
    LOW = "LOW"       # Creative, Casual → No gate
    MID = "MID"       # Factual, Coding, Math → ≤5% risk
    HIGH = "HIGH"     # Medical, Legal, Financial → ≤2.5% risk


# LLaMA Guard 3 Risk Categories (from model card)
# https://huggingface.co/meta-llama/Llama-Guard-3-8B
LLAMA_GUARD_CATEGORIES = {
    "S1": "Violent Crimes",
    "S2": "Non-Violent Crimes", 
    "S3": "Sex-Related Crimes",
    "S4": "Child Sexual Exploitation",
    "S5": "Defamation",
    "S6": "Specialized Advice",        # Medical, Legal, Financial
    "S7": "Privacy",
    "S8": "Intellectual Property",
    "S9": "Indiscriminate Weapons",
    "S10": "Hate",
    "S11": "Suicide & Self-Harm",
    "S12": "Sexual Content",
    "S13": "Elections",
    "S14": "Code Interpreter Abuse",
}

# Map LLaMA Guard categories to our sensitivity levels
CATEGORY_TO_SENSITIVITY = {
    # HIGH: Medical, Legal, Financial, Weapons, Self-Harm
    "S1": SensitivityLevel.HIGH,   # Violent Crimes
    "S3": SensitivityLevel.HIGH,   # Sex Crimes
    "S4": SensitivityLevel.HIGH,   # CSAM
    "S6": SensitivityLevel.HIGH,   # Specialized Advice (Medical/Legal/Financial)
    "S9": SensitivityLevel.HIGH,   # Weapons
    "S11": SensitivityLevel.HIGH,  # Suicide/Self-Harm
    
    # MID: Privacy, IP, Elections, Code (factually costly if wrong)
    "S2": SensitivityLevel.MID,    # Non-Violent Crimes
    "S5": SensitivityLevel.MID,    # Defamation
    "S7": SensitivityLevel.MID,    # Privacy
    "S8": SensitivityLevel.MID,    # Intellectual Property
    "S10": SensitivityLevel.MID,   # Hate
    "S12": SensitivityLevel.MID,   # Sexual Content
    "S13": SensitivityLevel.MID,   # Elections
    "S14": SensitivityLevel.MID,   # Code Interpreter Abuse
}


class SafetyClassifier:
    """
    LLaMA Guard 3 based safety classifier for prompt sensitivity detection.
    
    Lazily loads the model on first use to minimize startup overhead.
    Falls back to regex-based classification if model loading fails.
    """
    
    MODEL_ID = "meta-llama/Llama-Guard-3-8B"
    
    def __init__(self, device: Optional[str] = None, use_fallback: bool = True):
        """
        Initialize the safety classifier.
        
        Args:
            device: Device to run inference on ('cuda', 'mps', 'cpu', or None for auto)
            use_fallback: If True, use regex fallback when model unavailable
        """
        self._model = None
        self._tokenizer = None
        self._device = device
        self._use_fallback = use_fallback
        self._load_attempted = False
    
    def _detect_device(self) -> str:
        """Auto-detect the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    
    def _load_model(self) -> bool:
        """
        Lazily load the LLaMA Guard model and tokenizer.
        Returns True if successful, False otherwise.
        """
        if self._load_attempted:
            return self._model is not None
        
        self._load_attempted = True
        
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            logger.info(f"Loading LLaMA Guard 3 from {self.MODEL_ID}...")
            
            device = self._device or self._detect_device()
            
            self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.MODEL_ID,
                torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
                device_map=device if device != "cpu" else None,
            )
            
            if device == "cpu":
                self._model = self._model.to("cpu")
            
            logger.info(f"LLaMA Guard 3 loaded successfully on {device}")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load LLaMA Guard 3: {e}")
            logger.warning("Falling back to regex-based classification")
            return False
    
    def _regex_fallback(self, prompt: str) -> Tuple[SensitivityLevel, str]:
        """
        Fallback regex-based classification when ML model is unavailable.
        
        Returns:
            Tuple of (SensitivityLevel, reason_string)
        """
        text = prompt.lower()
        
        # HIGH sensitivity keywords
        high_triggers = [
            "medical", "doctor", "medicine", "symptom", "dose", "drug", "dosage",
            "legal", "lawyer", "court", "contract", "statute", "sue",
            "suicide", "self-harm", "kill myself", "end my life",
            "weapon", "bomb", "poison", "lethal"
        ]
        
        # MID sensitivity keywords (factual, coding, math)
        mid_triggers = [
            "code", "python", "java", "function", "bug", "error", "traceback",
            "math", "solve", "equation", "calculate", "derivative", "integral",
            "fact", "history", "date", "who is", "what is", "define",
            "privacy", "personal data", "copyright"
        ]
        
        for trigger in high_triggers:
            if trigger in text:
                return SensitivityLevel.HIGH, f"regex_match:{trigger}"
        
        for trigger in mid_triggers:
            if trigger in text:
                return SensitivityLevel.MID, f"regex_match:{trigger}"
        
        return SensitivityLevel.LOW, "no_match"
    
    def classify(self, prompt: str) -> SensitivityLevel:
        """
        Classify a prompt's sensitivity level using LLaMA Guard 3.
        
        Args:
            prompt: The user prompt to classify
            
        Returns:
            SensitivityLevel enum (HIGH, MID, or LOW)
        """
        # Try to load the model
        if not self._load_model():
            if self._use_fallback:
                level, _ = self._regex_fallback(prompt)
                return level
            raise RuntimeError("LLaMA Guard model not available and fallback disabled")
        
        # Format prompt for LLaMA Guard
        # LLaMA Guard uses a specific chat format
        conversation = [
            {"role": "user", "content": prompt}
        ]
        
        input_ids = self._tokenizer.apply_chat_template(
            conversation,
            return_tensors="pt"
        ).to(self._model.device)
        
        # Generate classification
        with torch.no_grad():
            output = self._model.generate(
                input_ids,
                max_new_tokens=100,
                pad_token_id=self._tokenizer.eos_token_id
            )
        
        # Decode output
        response = self._tokenizer.decode(
            output[0][input_ids.shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        # Parse LLaMA Guard output
        # Format: "safe" or "unsafe\nS1,S6" (comma-separated categories)
        if response.lower().startswith("safe"):
            return SensitivityLevel.LOW
        
        # Extract categories from "unsafe\nS1,S6"
        lines = response.split("\n")
        if len(lines) > 1:
            categories = [c.strip() for c in lines[1].split(",")]
            
            # Find highest sensitivity from detected categories
            max_sensitivity = SensitivityLevel.LOW
            for cat in categories:
                cat_sensitivity = CATEGORY_TO_SENSITIVITY.get(cat, SensitivityLevel.LOW)
                if cat_sensitivity == SensitivityLevel.HIGH:
                    return SensitivityLevel.HIGH
                if cat_sensitivity == SensitivityLevel.MID:
                    max_sensitivity = SensitivityLevel.MID
            
            return max_sensitivity
        
        # Default to MID if unsafe but no categories parsed
        return SensitivityLevel.MID
    
    def classify_with_details(self, prompt: str) -> Tuple[SensitivityLevel, str, list]:
        """
        Classify a prompt and return detailed information.
        
        Returns:
            Tuple of (SensitivityLevel, raw_response, detected_categories)
        """
        if not self._load_model():
            if self._use_fallback:
                level, reason = self._regex_fallback(prompt)
                return level, f"fallback:{reason}", []
            raise RuntimeError("LLaMA Guard model not available")
        
        # Full classification with response parsing
        conversation = [{"role": "user", "content": prompt}]
        
        input_ids = self._tokenizer.apply_chat_template(
            conversation,
            return_tensors="pt"
        ).to(self._model.device)
        
        with torch.no_grad():
            output = self._model.generate(
                input_ids,
                max_new_tokens=100,
                pad_token_id=self._tokenizer.eos_token_id
            )
        
        response = self._tokenizer.decode(
            output[0][input_ids.shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        if response.lower().startswith("safe"):
            return SensitivityLevel.LOW, response, []
        
        lines = response.split("\n")
        categories = []
        if len(lines) > 1:
            categories = [c.strip() for c in lines[1].split(",")]
        
        level = SensitivityLevel.LOW
        for cat in categories:
            cat_sensitivity = CATEGORY_TO_SENSITIVITY.get(cat, SensitivityLevel.LOW)
            if cat_sensitivity == SensitivityLevel.HIGH:
                level = SensitivityLevel.HIGH
                break
            if cat_sensitivity == SensitivityLevel.MID:
                level = SensitivityLevel.MID
        
        return level, response, categories


# Singleton instance for easy import
_classifier_instance: Optional[SafetyClassifier] = None

def get_classifier() -> SafetyClassifier:
    """Get the global SafetyClassifier instance (lazy singleton)."""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = SafetyClassifier()
    return _classifier_instance


def classify_sensitivity(prompt: str) -> str:
    """
    Convenience function for classifying prompt sensitivity.
    
    Returns the sensitivity level as a string: "HIGH", "MID", or "LOW".
    """
    return get_classifier().classify(prompt).value
