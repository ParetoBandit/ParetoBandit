"""
Local (in-repo) prompt complexity classifier.

This loads the trained model artifact stored under:
  data/complexity_classifier_final/

That artifact is a DistilBERT sequence classifier with 5 labels (0..4). We
convert it into a 0..1 complexity score by taking the expected class index and
dividing by 4.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


@dataclass
class LocalComplexityResult:
    prompt: str
    # 0..4 (increasing complexity)
    label: int
    # probability assigned to the predicted label
    confidence: float
    # expected complexity in [0,1]
    prompt_complexity_score: float
    # all class probabilities (len=5)
    probs: List[float]


def default_local_complexity_model_path() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "data" / "complexity_classifier_final"


class LocalComplexityClassifier:
    """
    Loads the local DistilBERT complexity classifier from disk.

    This avoids any HuggingFace model downloads at runtime (fast & deterministic).
    """

    def __init__(self, model_dir: Optional[str | Path] = None, *, device: str = "cpu", max_length: int = 512):
        self.model_dir = Path(model_dir) if model_dir is not None else default_local_complexity_model_path()
        self.device = str(device)
        self.max_length = int(max_length)
        self._tokenizer = None
        self._model = None

    def is_available(self) -> bool:
        return self.model_dir.exists() and (self.model_dir / "config.json").exists()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        self._model = AutoModelForSequenceClassification.from_pretrained(str(self.model_dir))
        self._model.eval()

        if self.device != "cpu":
            if self.device == "cuda" and torch.cuda.is_available():
                self._model = self._model.to("cuda")
            elif self.device == "mps" and torch.backends.mps.is_available():
                self._model = self._model.to("mps")

    def classify(self, prompt: str) -> LocalComplexityResult:
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

        # Expect 5 classes, but don't hard-crash if config changes.
        n = int(probs.shape[0])
        idxs = np.arange(n, dtype=np.float64)
        exp_class = float(np.sum(idxs * probs))
        denom = float(max(n - 1, 1))
        score01 = float(exp_class / denom)

        label = int(np.argmax(probs))
        conf = float(probs[label])

        return LocalComplexityResult(
            prompt=str(prompt),
            label=label,
            confidence=conf,
            prompt_complexity_score=score01,
            probs=[float(x) for x in probs.tolist()],
        )

