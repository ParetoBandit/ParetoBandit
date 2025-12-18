"""
Prompt complexity classifiers for intelligent model routing.

This module provides two classifier implementations:
  - LocalComplexityClassifier: Uses a locally-trained DistilBERT model (fast, no downloads)
  - NvidiaComplexityClassifier: Uses NVIDIA's prompt-task-and-complexity-classifier (more features)

Usage:
    from llm_jury.async_bandit.complexity import (
        LocalComplexityClassifier,
        NvidiaComplexityClassifier,
        get_complexity_classifier,
    )

    # Auto-detect best available classifier
    classifier = get_complexity_classifier()
    result = classifier.classify("Write a recursive fibonacci function")
    print(result.prompt_complexity_score)  # 0.45
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Union

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract Interface
# ---------------------------------------------------------------------------


class ComplexityResult(Protocol):
    """Protocol for complexity classification results."""

    prompt: str
    prompt_complexity_score: float


@dataclass
class LocalComplexityResult:
    """Result from local DistilBERT complexity classifier."""

    prompt: str
    label: int  # 0..4 (increasing complexity)
    confidence: float  # probability assigned to predicted label
    prompt_complexity_score: float  # expected complexity in [0,1]
    probs: List[float]  # all class probabilities (len=5)


@dataclass
class NvidiaComplexityResult:
    """Result from NVIDIA complexity classifier with multi-dimensional scores."""

    prompt: str
    task_type_1: str
    task_type_2: str
    task_type_prob: float
    creativity_scope: float
    reasoning: float
    constraint_ct: float
    domain_knowledge: float
    contextual_knowledge: float
    number_of_few_shots: float
    prompt_complexity_score: float

    @property
    def is_complex(self) -> bool:
        """True if prompt is considered complex (score >= 0.4)."""
        return self.prompt_complexity_score >= 0.4

    @property
    def is_reasoning_heavy(self) -> bool:
        """True if prompt requires significant reasoning."""
        return self.reasoning >= 0.5

    @property
    def complexity_level(self) -> str:
        """Categorical complexity level."""
        score = self.prompt_complexity_score
        if score < 0.2:
            return "trivial"
        elif score < 0.35:
            return "simple"
        elif score < 0.5:
            return "moderate"
        elif score < 0.7:
            return "complex"
        return "expert"


# ---------------------------------------------------------------------------
# Local Complexity Classifier
# ---------------------------------------------------------------------------


def _default_local_model_path() -> Path:
    """Default path for local complexity model."""
    return Path(__file__).resolve().parent.parent.parent / "data" / "complexity_classifier_final"


class LocalComplexityClassifier:
    """
    Local DistilBERT complexity classifier.

    Loads a trained model from disk (no HuggingFace downloads at runtime).
    Fast and deterministic, ideal for production use.

    The model outputs 5 complexity classes (0-4), converted to a 0-1 score
    using expected class index.
    """

    def __init__(
        self,
        model_dir: Optional[Union[str, Path]] = None,
        *,
        device: str = "cpu",
        max_length: int = 512,
    ):
        """
        Initialize the local complexity classifier.

        Args:
            model_dir: Path to model directory (default: data/complexity_classifier_final/)
            device: Device to run on ("cpu", "cuda", "mps")
            max_length: Max input token length
        """
        self.model_dir = Path(model_dir) if model_dir else _default_local_model_path()
        self.device = str(device)
        self.max_length = int(max_length)
        self._tokenizer = None
        self._model = None

    def is_available(self) -> bool:
        """Check if the local model is available on disk."""
        return self.model_dir.exists() and (self.model_dir / "config.json").exists()

    def _ensure_loaded(self) -> None:
        """Lazy-load the model."""
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        self._model = AutoModelForSequenceClassification.from_pretrained(str(self.model_dir))
        self._model.eval()

        if self.device != "cpu":
            if self.device == "cuda" and torch.cuda.is_available():
                self._model = self._model.to("cuda")
            elif self.device == "mps" and torch.backends.mps.is_available():
                self._model = self._model.to("mps")

    def classify(self, prompt: str) -> LocalComplexityResult:
        """
        Classify a single prompt's complexity.

        Args:
            prompt: The prompt text to classify

        Returns:
            LocalComplexityResult with label, confidence, and complexity score
        """
        import torch

        self._ensure_loaded()
        assert self._tokenizer is not None and self._model is not None

        enc = self._tokenizer(
            str(prompt),
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        if next(self._model.parameters()).device.type != "cpu":
            enc = {k: v.to(next(self._model.parameters()).device) for k, v in enc.items()}

        with torch.no_grad():
            logits = self._model(**enc).logits.squeeze(0)
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy().astype(np.float64)

        n = int(probs.shape[0])
        idxs = np.arange(n, dtype=np.float64)
        exp_class = float(np.sum(idxs * probs))
        score01 = float(exp_class / max(n - 1, 1))

        label = int(np.argmax(probs))
        conf = float(probs[label])

        return LocalComplexityResult(
            prompt=str(prompt),
            label=label,
            confidence=conf,
            prompt_complexity_score=score01,
            probs=[float(x) for x in probs.tolist()],
        )


# ---------------------------------------------------------------------------
# NVIDIA Complexity Classifier
# ---------------------------------------------------------------------------


# Module-level cache for NVIDIA model (expensive to load)
_nvidia_model = None
_nvidia_tokenizer = None


def _load_nvidia_model():
    """Lazy-load the NVIDIA model and tokenizer."""
    global _nvidia_model, _nvidia_tokenizer

    if _nvidia_model is not None:
        return _nvidia_model, _nvidia_tokenizer

    import torch
    import torch.nn as nn
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    logger.info("Loading NVIDIA prompt-task-and-complexity-classifier...")

    config = AutoConfig.from_pretrained("nvidia/prompt-task-and-complexity-classifier")
    _nvidia_tokenizer = AutoTokenizer.from_pretrained("nvidia/prompt-task-and-complexity-classifier")

    class MeanPooling(nn.Module):
        def forward(self, last_hidden_state, attention_mask):
            mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * mask, 1)
            sum_mask = torch.clamp(mask.sum(1), min=1e-9)
            return sum_embeddings / sum_mask

    class MulticlassHead(nn.Module):
        def __init__(self, input_size, num_classes):
            super().__init__()
            self.fc = nn.Linear(input_size, num_classes)

        def forward(self, x):
            return self.fc(x)

    class CustomModel(nn.Module):
        def __init__(self, target_sizes, task_type_map, weights_map, divisor_map):
            super().__init__()
            self.backbone = AutoModel.from_pretrained("microsoft/DeBERTa-v3-base")
            self.target_sizes = list(target_sizes.values())
            self.task_type_map = task_type_map
            self.weights_map = weights_map
            self.divisor_map = divisor_map

            self.heads = []
            for i, sz in enumerate(self.target_sizes):
                head = MulticlassHead(self.backbone.config.hidden_size, sz)
                self.add_module(f"head_{i}", head)
                self.heads.append(head)
            self.pool = MeanPooling()

        def _compute_results(self, preds, target):
            if target == "task_type":
                top2_indices = torch.topk(preds, k=2, dim=1).indices
                softmax_probs = torch.softmax(preds, dim=1)
                top2_probs = softmax_probs.gather(1, top2_indices)
                top2 = top2_indices.detach().cpu().tolist()
                top2_prob = top2_probs.detach().cpu().tolist()

                top2_strings = [[self.task_type_map[str(idx)] for idx in sample] for sample in top2]
                top2_prob_rounded = [[round(v, 3) for v in sublist] for sublist in top2_prob]

                for i, probs in enumerate(top2_prob_rounded):
                    if probs[1] < 0.1:
                        top2_strings[i][1] = "NA"

                return (
                    [s[0] for s in top2_strings],
                    [s[1] for s in top2_strings],
                    [p[0] for p in top2_prob_rounded],
                )

            preds = torch.softmax(preds, dim=1)
            weights = np.array(self.weights_map[target])
            weighted_sum = np.sum(np.array(preds.detach().cpu()) * weights, axis=1)
            scores = [round(float(s), 4) for s in weighted_sum / self.divisor_map[target]]
            if target == "number_of_few_shots":
                scores = [x if x >= 0.05 else 0 for x in scores]
            return scores

        def forward(self, batch):
            outputs = self.backbone(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            pooled = self.pool(outputs.last_hidden_state, batch["attention_mask"])
            logits = [h(pooled) for h in self.heads]

            result = {}
            tt = self._compute_results(logits[0], "task_type")
            result["task_type_1"], result["task_type_2"], result["task_type_prob"] = tt
            result["creativity_scope"] = self._compute_results(logits[1], "creativity_scope")
            result["reasoning"] = self._compute_results(logits[2], "reasoning")
            result["contextual_knowledge"] = self._compute_results(logits[3], "contextual_knowledge")
            result["number_of_few_shots"] = self._compute_results(logits[4], "number_of_few_shots")
            result["domain_knowledge"] = self._compute_results(logits[5], "domain_knowledge")
            result["constraint_ct"] = self._compute_results(logits[7], "constraint_ct")

            result["prompt_complexity_score"] = [
                round(
                    0.35 * c + 0.25 * r + 0.15 * cn + 0.15 * d + 0.05 * ctx + 0.05 * f,
                    5,
                )
                for c, r, cn, d, ctx, f in zip(
                    result["creativity_scope"],
                    result["reasoning"],
                    result["constraint_ct"],
                    result["domain_knowledge"],
                    result["contextual_knowledge"],
                    result["number_of_few_shots"],
                )
            ]
            return result

    _nvidia_model = CustomModel(
        target_sizes=config.target_sizes,
        task_type_map=config.task_type_map,
        weights_map=config.weights_map,
        divisor_map=config.divisor_map,
    )

    weights_path = hf_hub_download(
        repo_id="nvidia/prompt-task-and-complexity-classifier",
        filename="model.safetensors",
    )
    state_dict = load_file(weights_path)
    _nvidia_model.load_state_dict(state_dict, strict=True)
    _nvidia_model.eval()

    logger.info("NVIDIA classifier loaded successfully")
    return _nvidia_model, _nvidia_tokenizer


class NvidiaComplexityClassifier:
    """
    NVIDIA prompt-task-and-complexity-classifier wrapper.

    Provides multi-dimensional complexity scoring:
    - creativity_scope (35% weight)
    - reasoning (25% weight)
    - constraint_ct (15% weight)
    - domain_knowledge (15% weight)
    - contextual_knowledge (5% weight)
    - number_of_few_shots (5% weight)

    Also classifies into 11 task types.
    """

    def __init__(self, device: str = "cpu"):
        """
        Initialize the NVIDIA complexity classifier.

        Args:
            device: Device to run on ("cpu" or "cuda")
        """
        self.device = device
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        """Lazy-load the model."""
        if self._model is None:
            import torch

            self._model, self._tokenizer = _load_nvidia_model()
            if self.device != "cpu" and torch.cuda.is_available():
                self._model = self._model.to(self.device)

    def classify(self, prompt: str) -> NvidiaComplexityResult:
        """
        Classify a single prompt's complexity.

        Args:
            prompt: The prompt text to classify

        Returns:
            NvidiaComplexityResult with all complexity dimensions
        """
        return self.classify_batch([prompt])[0]

    def classify_batch(self, prompts: List[str]) -> List[NvidiaComplexityResult]:
        """
        Classify a batch of prompts.

        Args:
            prompts: List of prompt texts

        Returns:
            List of NvidiaComplexityResult objects
        """
        import torch

        self._ensure_loaded()

        formatted = [f"Prompt: {p}" for p in prompts]
        encoded = self._tokenizer(
            formatted,
            return_tensors="pt",
            add_special_tokens=True,
            max_length=512,
            padding="max_length",
            truncation=True,
        )

        if self.device != "cpu":
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.no_grad():
            raw = self._model(encoded)

        results = []
        for i, prompt in enumerate(prompts):
            results.append(
                NvidiaComplexityResult(
                    prompt=prompt,
                    task_type_1=raw["task_type_1"][i],
                    task_type_2=raw["task_type_2"][i],
                    task_type_prob=raw["task_type_prob"][i],
                    creativity_scope=raw["creativity_scope"][i],
                    reasoning=raw["reasoning"][i],
                    constraint_ct=raw["constraint_ct"][i],
                    domain_knowledge=raw["domain_knowledge"][i],
                    contextual_knowledge=raw["contextual_knowledge"][i],
                    number_of_few_shots=raw["number_of_few_shots"][i],
                    prompt_complexity_score=raw["prompt_complexity_score"][i],
                )
            )
        return results

    def filter_by_complexity(
        self,
        prompts: List[str],
        min_score: float = 0.4,
        min_reasoning: Optional[float] = None,
    ) -> List[str]:
        """Filter prompts to only include complex ones."""
        results = self.classify_batch(prompts)
        return [
            r.prompt
            for r in results
            if r.prompt_complexity_score >= min_score
            and (min_reasoning is None or r.reasoning >= min_reasoning)
        ]

    def get_complexity_distribution(self, prompts: List[str]) -> Dict:
        """Get complexity distribution statistics for a set of prompts."""
        results = self.classify_batch(prompts)
        scores = [r.prompt_complexity_score for r in results]
        reasoning = [r.reasoning for r in results]

        levels = {"trivial": 0, "simple": 0, "moderate": 0, "complex": 0, "expert": 0}
        for r in results:
            levels[r.complexity_level] += 1

        task_types: Dict[str, int] = {}
        for r in results:
            task_types[r.task_type_1] = task_types.get(r.task_type_1, 0) + 1

        return {
            "count": len(prompts),
            "complexity_score": {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "min": float(min(scores)),
                "max": float(max(scores)),
            },
            "reasoning_score": {"mean": float(np.mean(reasoning)), "std": float(np.std(reasoning))},
            "complexity_levels": levels,
            "task_types": task_types,
        }


# ---------------------------------------------------------------------------
# Factory Function
# ---------------------------------------------------------------------------


def get_complexity_classifier(
    prefer: str = "local",
    device: str = "cpu",
) -> Union[LocalComplexityClassifier, NvidiaComplexityClassifier]:
    """
    Get the best available complexity classifier.

    Args:
        prefer: Preferred classifier ("local" or "nvidia")
        device: Device to run on

    Returns:
        Complexity classifier instance

    Raises:
        RuntimeError: If no classifier is available
    """
    if prefer == "local":
        local = LocalComplexityClassifier(device=device)
        if local.is_available():
            return local
        logger.warning("Local classifier not available, falling back to NVIDIA")
        return NvidiaComplexityClassifier(device=device)

    return NvidiaComplexityClassifier(device=device)


# Convenience function
def classify_prompt_complexity(prompt: str) -> ComplexityResult:
    """Quick function to classify a single prompt using the best available classifier."""
    classifier = get_complexity_classifier()
    return classifier.classify(prompt)
