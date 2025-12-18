#!/usr/bin/env python3
"""
Abstract Judge Interface for the Async Bandit Router.

This module provides:
  - `Judge`: Abstract protocol for grading (prompt, response) pairs
  - `PriorManager`: Utilities for loading, updating, and generating priors
  - Factory functions for common judge configurations

Users can:
  1. Use bundled priors that ship with the library
  2. Update priors incrementally with new observations
  3. Generate new priors from scratch using their own data
  4. Dynamically add new models via "brain surgery"

Prior Storage Locations:
  - BUNDLED (read-only): <package>/data/priors/shippable_priors.npz
  - USER (read-write):   ~/.llm_jury/priors/user_priors.npz
  - CUSTOM:              User-specified path

When you update priors (e.g., add a new model), the changes are saved to the
USER location by default. This preserves the bundled priors while allowing
personalization.

References (clustering & data efficiency):
  - LIMA: Less Is More for Alignment (Zhou et al., 2023)
  - #InsTag: Instruction Tagging for Analyzing Supervised Fine-tuning (Lu et al., 2024)
  - AlpaGasus (Chen et al., 2024)
"""

from __future__ import annotations

import json
import os
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# Default Paths
# ---------------------------------------------------------------------------

def _get_user_priors_dir() -> Path:
    """Get user-specific priors directory (~/.llm_jury/priors/)."""
    return Path.home() / ".llm_jury" / "priors"


def _get_user_priors_path() -> Path:
    """Get default user priors file path."""
    return _get_user_priors_dir() / "user_priors.npz"


def _get_bundled_priors_path() -> Path:
    """Get bundled priors path (read-only, ships with library)."""
    return Path(__file__).parent.parent.parent / "data" / "priors" / "shippable_priors.npz"


# ---------------------------------------------------------------------------
# Abstract Judge Protocol
# ---------------------------------------------------------------------------


class Judge(Protocol):
    """
    Abstract interface for grading (prompt, response) pairs.

    All judges must implement `grade()` which returns:
      - reward: float in some consistent scale (e.g., logit-transformed P(correct))
      - metadata: dict with debug info (e.g., which sub-grader was used)

    The router uses `reward` for bandit updates; `metadata` is for logging/debugging.
    """

    def grade(self, prompt: str, response: str) -> Tuple[float, Dict[str, Any]]:
        """
        Grade a (prompt, response) pair.

        Returns:
            (reward, metadata)
            - reward: scalar signal for bandit learning
            - metadata: arbitrary debug/audit info
        """
        ...


class JudgeWithComplexity(Protocol):
    """
    Extended judge that also provides prompt complexity classification.

    Useful for:
      - Tiered grading (soft vs hard path)
      - Complexity-gated routing
    """

    def grade(self, prompt: str, response: str) -> Tuple[float, Dict[str, Any]]: ...

    def classify_complexity(self, prompt: str) -> Tuple[str, float]:
        """
        Classify prompt complexity.

        Returns:
            (complexity_label, confidence)
            - complexity_label: e.g., "simple", "moderate", "complex", "domain"
            - confidence: float in [0, 1]
        """
        ...


# ---------------------------------------------------------------------------
# Prior Configuration
# ---------------------------------------------------------------------------


@dataclass
class PriorConfig:
    """
    Configuration for how priors are loaded/generated.

    Attributes:
        source: One of "bundled", "file", "generate", "none"
        path: Path to priors file (for "file" source)
        cluster_k: Number of clusters for archetype grid (for "generate")
        dataset: HuggingFace dataset name (for "generate")
        dataset_split: Dataset split (for "generate")
        max_prompts: Max prompts to sample from dataset (for "generate")
    """

    source: str = "bundled"  # "bundled", "file", "generate", "none"
    path: Optional[Path] = None
    cluster_k: int = 500
    dataset: str = "lmsys/chatbot_arena_conversations"
    dataset_split: str = "train"
    max_prompts: int = 50000

    def __post_init__(self):
        if self.source not in ("bundled", "file", "generate", "none"):
            raise ValueError(f"Invalid prior source: {self.source}")
        if self.source == "file" and self.path is None:
            raise ValueError("Must provide path when source='file'")


# ---------------------------------------------------------------------------
# Prior Manager
# ---------------------------------------------------------------------------


class PriorManager:
    """
    Manages prior loading, updating, saving, and generation for the bandit router.

    Storage Locations:
        - BUNDLED (read-only):  <package>/data/priors/shippable_priors.npz
        - USER (read-write):    ~/.llm_jury/priors/user_priors.npz
        - CUSTOM:               User-specified path

    When you call save(), changes go to USER location by default (not bundled).
    This allows users to customize priors while preserving library defaults.

    Usage:
        # Use bundled priors (read-only)
        manager = PriorManager.bundled()
        priors = manager.load()

        # Use user priors (read-write, falls back to bundled)
        manager = PriorManager.user()
        priors = manager.load()
        # ... modify priors ...
        manager.save(priors)  # Saves to ~/.llm_jury/priors/

        # Use custom priors file
        manager = PriorManager.from_file("my_priors.npz")
        priors = manager.load()

        # Generate new priors (expensive)
        manager = PriorManager.generate(cluster_k=500, dataset="lmsys/chatbot_arena_conversations")
        priors = manager.build(judge=my_judge, models=model_list)
    """

    # Default locations
    BUNDLED_PRIORS_PATH = _get_bundled_priors_path()
    USER_PRIORS_PATH = _get_user_priors_path()

    def __init__(self, config: PriorConfig, *, save_path: Optional[Path] = None):
        self.config = config
        # Where to save updated priors (defaults to user location)
        self._save_path = save_path or self.USER_PRIORS_PATH

    @classmethod
    def bundled(cls) -> "PriorManager":
        """Use the priors that ship with the library (read-only)."""
        return cls(PriorConfig(source="bundled"))

    @classmethod
    def user(cls) -> "PriorManager":
        """
        Use user-specific priors (~/.llm_jury/priors/user_priors.npz).

        Falls back to bundled priors if user priors don't exist.
        Saves go to user location, preserving bundled priors.
        """
        user_path = _get_user_priors_path()
        if user_path.exists():
            return cls(PriorConfig(source="file", path=user_path), save_path=user_path)
        # Fall back to bundled, but save to user location
        return cls(PriorConfig(source="bundled"), save_path=user_path)

    @classmethod
    def merged(cls) -> "PriorManager":
        """
        Load bundled priors as base, then layer user additions on top.

        This is the recommended mode for most users:
          - Start with library's bundled priors (all models)
          - Add/override with user's custom models and updates

        Example workflow:
            manager = PriorManager.merged()
            priors = manager.load()  # bundled + user merged
            priors = manager.add_model(priors, "new/model-v1")
            manager.save(priors)  # Only saves user additions to ~/.llm_jury/

        Merge strategy:
            - A_shared: Use user's if available, else bundled
            - b_vectors: Union of both, user takes precedence on conflicts
            - model_ids: Union of both sets
        """
        return cls(PriorConfig(source="bundled"), save_path=_get_user_priors_path())

    @classmethod
    def from_file(cls, path: Union[str, Path], *, save_in_place: bool = True) -> "PriorManager":
        """
        Load priors from a custom file.

        Args:
            path: Path to the priors NPZ file
            save_in_place: If True, save() overwrites this file. If False, saves to user location.
        """
        p = Path(path)
        save_path = p if save_in_place else _get_user_priors_path()
        return cls(PriorConfig(source="file", path=p), save_path=save_path)

    @classmethod
    def generate(
        cls,
        cluster_k: int = 500,
        dataset: str = "lmsys/chatbot_arena_conversations",
        dataset_split: str = "train",
        max_prompts: int = 50000,
    ) -> "PriorManager":
        """
        Configure for generating new priors.

        Note: Call `.build()` to actually generate (requires judge and models).
        """
        return cls(
            PriorConfig(
                source="generate",
                cluster_k=cluster_k,
                dataset=dataset,
                dataset_split=dataset_split,
                max_prompts=max_prompts,
            )
        )

    @classmethod
    def none(cls) -> "PriorManager":
        """No priors (cold start)."""
        return cls(PriorConfig(source="none"))

    def load(self) -> Optional[Dict[str, Any]]:
        """
        Load priors based on configuration.

        Returns:
            Dict with keys: "A_shared", "b_vectors", "model_ids", "dim", "alpha", etc.
            Or None if source="none" or file doesn't exist.
        """
        if self.config.source == "none":
            return None

        if self.config.source == "bundled":
            path = self.BUNDLED_PRIORS_PATH
        elif self.config.source == "file":
            path = self.config.path
        else:
            # "generate" source - priors don't exist yet
            return None

        if path is None or not path.exists():
            return None

        return self._load_npz(path)

    def _load_npz(self, path: Path) -> Dict[str, Any]:
        """Load priors from NPZ file."""
        data = np.load(path, allow_pickle=True)
        result: Dict[str, Any] = {}

        # Load model IDs first (needed to convert b_vectors)
        model_ids: List[str] = []
        if "model_ids" in data:
            model_ids = [str(x) for x in list(data["model_ids"])]
            result["model_ids"] = model_ids

        # Load A_shared
        if "A_shared" in data:
            result["A_shared"] = data["A_shared"].astype(np.float64)

        # Load b_vectors - may be 2D array (n_models x dim) or already a dict
        if "b_vectors" in data:
            b_raw = data["b_vectors"]
            if isinstance(b_raw, np.ndarray) and b_raw.ndim == 2:
                # Convert 2D array to dict: {model_id: b_vector}
                result["b_vectors"] = {
                    model_ids[i]: b_raw[i].astype(np.float64)
                    for i in range(min(len(model_ids), len(b_raw)))
                }
            else:
                # Already a dict-like structure
                result["b_vectors"] = {str(k): np.asarray(v, dtype=np.float64) for k, v in dict(b_raw).items()}

        # Load metadata
        if "meta" in data:
            meta = data["meta"].item() if hasattr(data["meta"], "item") else dict(data["meta"])
            result.update(meta)

        return result

    def exists(self) -> bool:
        """Check if priors exist (for bundled/file sources)."""
        if self.config.source == "none":
            return False
        if self.config.source == "generate":
            return False  # Not built yet

        if self.config.source == "bundled":
            return self.BUNDLED_PRIORS_PATH.exists()
        elif self.config.source == "file":
            return self.config.path is not None and self.config.path.exists()
        return False

    def load_merged(self) -> Optional[Dict[str, Any]]:
        """
        Load bundled priors as base, then layer user additions on top.

        Returns:
            Merged priors dict, or None if no priors exist.

        Merge strategy:
            - A_shared: Use user's if user priors exist, else bundled
            - b_vectors: Union of both, user takes precedence on conflicts
            - model_ids: Union of both sets (bundled + user additions)
        """
        bundled = None
        user = None

        if self.BUNDLED_PRIORS_PATH.exists():
            bundled = self._load_npz(self.BUNDLED_PRIORS_PATH)

        if self.USER_PRIORS_PATH.exists():
            user = self._load_npz(self.USER_PRIORS_PATH)

        if bundled is None and user is None:
            return None
        if bundled is None:
            return user
        if user is None:
            return bundled

        # Merge: bundled as base, user as overlay
        return self.merge_priors(bundled, user)

    @staticmethod
    def merge_priors(
        base: Dict[str, Any],
        overlay: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Merge two prior bundles.

        Args:
            base: Base priors (e.g., bundled library priors)
            overlay: Overlay priors (e.g., user additions/updates)

        Returns:
            Merged priors with:
                - A_shared: overlay's if present, else base's
                - b_vectors: union, overlay takes precedence on conflicts
                - model_ids: union of both

        Example:
            # Bundled has models A, B, C
            # User added model D and updated B
            # Result: A (bundled), B (user), C (bundled), D (user)
        """
        base_models = set(base.get("model_ids", []))
        overlay_models = set(overlay.get("model_ids", []))
        all_models = list(base_models | overlay_models)

        base_b = base.get("b_vectors", {})
        overlay_b = overlay.get("b_vectors", {})

        # Merge b_vectors: overlay takes precedence
        merged_b: Dict[str, Any] = {}
        for m in all_models:
            if m in overlay_b:
                merged_b[m] = overlay_b[m]
            elif m in base_b:
                merged_b[m] = base_b[m]
            else:
                # New model with no b vector - initialize to zeros
                dim = int(base.get("dim", overlay.get("dim", 384)))
                merged_b[m] = np.zeros(dim, dtype=np.float64)

        # A_shared: prefer overlay if it exists
        a_shared = overlay.get("A_shared")
        if a_shared is None:
            a_shared = base.get("A_shared")

        return {
            "A_shared": a_shared,
            "b_vectors": merged_b,
            "model_ids": all_models,
            "dim": overlay.get("dim", base.get("dim", 384)),
            "alpha": overlay.get("alpha", base.get("alpha", 0.5)),
            "_merged_from": ["base", "overlay"],
        }

    @property
    def save_path(self) -> Path:
        """Where save() will write priors."""
        return self._save_path

    def save(self, priors: Dict[str, Any], *, path: Optional[Path] = None) -> Path:
        """
        Save priors to disk.

        By default, saves to the user location (~/.llm_jury/priors/user_priors.npz)
        to preserve bundled priors. Use `path` to override.

        Args:
            priors: Prior bundle dict with keys like "A_shared", "b_vectors", "model_ids"
            path: Optional override path (default: self.save_path)

        Returns:
            Path where priors were saved
        """
        out_path = Path(path) if path else self._save_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Extract components
        A_shared = priors.get("A_shared")
        b_vectors = priors.get("b_vectors", {})
        model_ids = priors.get("model_ids", list(b_vectors.keys()))
        dim = priors.get("dim", 384)
        alpha = priors.get("alpha", 0.5)

        # Stack b vectors
        b_matrix = np.stack([
            np.asarray(b_vectors.get(m, np.zeros(dim)), dtype=np.float16)
            for m in model_ids
        ], axis=0)

        # Save
        meta = {"dim": int(dim), "alpha": float(alpha)}
        np.savez_compressed(
            out_path,
            A_shared=np.asarray(A_shared, dtype=np.float16) if A_shared is not None else np.eye(dim, dtype=np.float16),
            b_vectors=b_matrix,
            model_ids=np.asarray(model_ids, dtype=object),
            meta=meta,
        )

        return out_path

    def copy_bundled_to_user(self) -> Optional[Path]:
        """
        Copy bundled priors to user location for customization.

        Returns:
            Path to user priors, or None if bundled priors don't exist.
        """
        if not self.BUNDLED_PRIORS_PATH.exists():
            return None

        self.USER_PRIORS_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.BUNDLED_PRIORS_PATH, self.USER_PRIORS_PATH)
        return self.USER_PRIORS_PATH

    def build(
        self,
        judge: Judge,
        models: List[str],
        *,
        call_model: Callable[[str, str, int], Optional[str]],
        out_path: Optional[Path] = None,
        workers: int = 10,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Dict[str, Any]:
        """
        Generate new priors using the Archetype Grid strategy.

        This is an expensive operation that:
          1. Clusters prompts from the configured dataset
          2. Runs all models on the representative prompts
          3. Grades responses with the provided judge
          4. Builds and returns the prior bundle

        Args:
            judge: Judge to grade (prompt, response) pairs
            models: List of model IDs to include
            call_model: Function(model_id, prompt, max_tokens) -> response_text
            out_path: Optional path to save the priors NPZ
            workers: Number of parallel workers for API calls
            progress_callback: Optional callback(completed, total) for progress

        Returns:
            Prior bundle dict (same format as load())

        Raises:
            ValueError: If config.source != "generate"
        """
        if self.config.source != "generate":
            raise ValueError("build() only works with source='generate'")

        # Import heavy dependencies only when needed
        from concurrent.futures import ThreadPoolExecutor, as_completed

        from sentence_transformers import SentenceTransformer
        from sklearn.cluster import MiniBatchKMeans

        from llm_jury.async_bandit.bandit_router import (
            SharedCovarianceLinUCBPolicy,
            l2_normalize,
        )

        # Step 1: Load and cluster prompts
        prompts = self._load_prompts_from_dataset()
        if not prompts:
            raise ValueError(f"No prompts loaded from {self.config.dataset}")

        encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        embeddings = encoder.encode(prompts, show_progress_bar=True, normalize_embeddings=True)

        kmeans = MiniBatchKMeans(n_clusters=self.config.cluster_k, random_state=42, batch_size=256)
        labels = kmeans.fit_predict(embeddings)

        # Select representative prompt per cluster (closest to centroid)
        archetypes: List[Tuple[int, str]] = []
        for k in range(self.config.cluster_k):
            mask = labels == k
            if not mask.any():
                continue
            cluster_embeds = embeddings[mask]
            cluster_prompts = [prompts[i] for i in range(len(prompts)) if labels[i] == k]
            centroid = kmeans.cluster_centers_[k]
            dists = np.linalg.norm(cluster_embeds - centroid, axis=1)
            best_idx = int(np.argmin(dists))
            archetypes.append((k, cluster_prompts[best_idx]))

        # Step 2: Dense run - all models × all archetypes
        policy = SharedCovarianceLinUCBPolicy(models, dim=384, alpha=0.5)
        total_pairs = len(archetypes) * len(models)
        completed = 0

        def process_one(cid: int, prompt: str, model_id: str, x: np.ndarray):
            resp = call_model(model_id, prompt, 800)
            if resp is None or (isinstance(resp, str) and resp.startswith("[ERROR")):
                return None
            reward, _ = judge.grade(prompt, resp)
            return {"cid": cid, "model_id": model_id, "reward": reward, "x": x}

        for cid, prompt in archetypes:
            x = encoder.encode(prompt, normalize_embeddings=True)
            x = l2_normalize(np.asarray(x, dtype=np.float64))

            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(process_one, cid, prompt, mid, x): mid
                    for mid in models
                }
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result is not None:
                            policy.update(result["model_id"], result["x"], result["reward"])
                    except Exception:
                        pass
                    completed += 1
                    if progress_callback:
                        progress_callback(completed, total_pairs)

        # Step 3: Export
        if out_path:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            policy.to_shippable_priors_npz(out_path, dtype=np.float16)

        # Return the bundle
        return {
            "A_shared": policy.A_shared.copy(),
            "b_vectors": {mid: policy.b[mid].copy() for mid in models},
            "model_ids": models,
            "dim": 384,
            "alpha": 0.5,
        }

    def _load_prompts_from_dataset(self) -> List[str]:
        """Load prompts from configured HuggingFace dataset."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Install 'datasets' to generate priors: pip install datasets")

        ds = load_dataset(self.config.dataset, split=self.config.dataset_split)

        prompts: List[str] = []
        for ex in ds:
            prompt = self._extract_prompt(ex)
            if prompt:
                prompts.append(prompt)
            if len(prompts) >= self.config.max_prompts:
                break

        return prompts

    def _extract_prompt(self, ex: Any) -> Optional[str]:
        """Extract prompt from various dataset schemas."""
        # LMSYS Chatbot Arena format
        if "conversation_a" in ex:
            conv = ex.get("conversation_a") or ex.get("conversation_b")
            if isinstance(conv, list) and len(conv) > 0:
                first = conv[0]
                if isinstance(first, dict) and first.get("role") == "user":
                    return str(first.get("content", "")).strip() or None
        # Standard formats
        for key in ("prompt", "instruction", "question", "input", "text"):
            if key in ex:
                val = ex[key]
                if isinstance(val, str) and val.strip():
                    return val.strip()
        return None

    def update(
        self,
        current_priors: Dict[str, Any],
        observations: List[Tuple[str, str, str, float]],  # (prompt, response, model_id, reward)
        *,
        encoder: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Incrementally update priors with new observations.

        Args:
            current_priors: Existing prior bundle from load()
            observations: List of (prompt, response, model_id, reward) tuples
            encoder: Optional SentenceTransformer encoder (will load default if None)

        Returns:
            Updated prior bundle
        """
        if encoder is None:
            from sentence_transformers import SentenceTransformer

            encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        from llm_jury.async_bandit.bandit_router import l2_normalize

        # Reconstruct policy from priors
        model_ids = current_priors.get("model_ids", [])
        A_shared = current_priors.get("A_shared")
        b_vectors = current_priors.get("b_vectors", {})
        dim = int(current_priors.get("dim", 384))
        alpha = float(current_priors.get("alpha", 0.5))

        # Apply updates
        for prompt, response, model_id, reward in observations:
            if model_id not in b_vectors:
                # New model - initialize
                b_vectors[model_id] = np.zeros(dim, dtype=np.float64)
                if model_id not in model_ids:
                    model_ids.append(model_id)

            x = encoder.encode(prompt, normalize_embeddings=True)
            x = l2_normalize(np.asarray(x, dtype=np.float64))

            # Update shared A and per-model b
            # A_shared += x @ x.T  (but we keep A_shared as the inverse for efficiency)
            # For simplicity, we just update b here; full A update would require re-inversion
            b_vectors[model_id] = b_vectors[model_id] + reward * x

        return {
            "A_shared": A_shared,
            "b_vectors": b_vectors,
            "model_ids": model_ids,
            "dim": dim,
            "alpha": alpha,
        }

    def add_model(
        self,
        current_priors: Dict[str, Any],
        new_model_id: str,
        *,
        clone_from: Optional[str] = None,
        clone_decay: float = 0.9,
    ) -> Dict[str, Any]:
        """
        Add a new model to existing priors ("brain surgery").

        This is how you "install" a new model (e.g., DeepSeek-V3) into the router
        without retraining from scratch.

        Strategies:
            1. CLONING (recommended): Clone weights from a similar model (e.g., DeepSeek-V2).
               The new model inherits the old model's "knowledge" with slight uncertainty boost.
            2. COLD START: Initialize with zeros (shared A, zero b). The bandit will
               explore this model to learn its capabilities.

        Args:
            current_priors: Existing prior bundle from load()
            new_model_id: Model ID to add (e.g., "deepseek/deepseek-v3")
            clone_from: Optional model ID to clone from (e.g., "deepseek/deepseek-v2")
            clone_decay: Multiply cloned weights by this factor (default 0.9) to
                         increase uncertainty and encourage exploration of the new model.

        Returns:
            Updated prior bundle with the new model

        Example:
            # Clone from a similar model (recommended for model upgrades)
            priors = manager.add_model(priors, "deepseek/deepseek-v3", clone_from="deepseek/deepseek-r1")

            # Cold start (for completely new models)
            priors = manager.add_model(priors, "brand-new/model-v1")
        """
        model_ids = list(current_priors.get("model_ids", []))
        b_vectors = dict(current_priors.get("b_vectors", {}))
        dim = int(current_priors.get("dim", 384))

        if new_model_id in model_ids:
            # Model already exists
            return current_priors

        # Add to model list
        model_ids.append(new_model_id)

        if clone_from and clone_from in b_vectors:
            # CLONING: Copy weights from existing model with decay
            source_b = np.asarray(b_vectors[clone_from], dtype=np.float64)
            b_vectors[new_model_id] = source_b * float(clone_decay)
        else:
            # COLD START: Zero initialization
            b_vectors[new_model_id] = np.zeros(dim, dtype=np.float64)

        return {
            "A_shared": current_priors.get("A_shared"),
            "b_vectors": b_vectors,
            "model_ids": model_ids,
            "dim": dim,
            "alpha": current_priors.get("alpha", 0.5),
        }

    def remove_model(
        self,
        current_priors: Dict[str, Any],
        model_id: str,
    ) -> Dict[str, Any]:
        """
        Remove a model from priors.

        Args:
            current_priors: Existing prior bundle
            model_id: Model ID to remove

        Returns:
            Updated prior bundle without the model
        """
        model_ids = [m for m in current_priors.get("model_ids", []) if m != model_id]
        b_vectors = {k: v for k, v in current_priors.get("b_vectors", {}).items() if k != model_id}

        return {
            "A_shared": current_priors.get("A_shared"),
            "b_vectors": b_vectors,
            "model_ids": model_ids,
            "dim": current_priors.get("dim", 384),
            "alpha": current_priors.get("alpha", 0.5),
        }


# ---------------------------------------------------------------------------
# Factory Functions for Common Judge Configurations
# ---------------------------------------------------------------------------


def create_soft_judge(
    model_path: Optional[Path] = None,
) -> Judge:
    """
    Create a soft (local, fast) judge using the QualityCostPredictor.

    This is the default DeBERTa-based grader that runs locally.
    Good for: style, fluency, general quality.
    Not good for: factual correctness, math, code execution.
    """
    from llm_jury.async_bandit.quality_cost_predictor import QualityCostPredictor

    default_path = Path(__file__).parent.parent.parent / "data" / "quality_predictor" / "best_quality_predictor.pt"
    path = model_path or default_path

    predictor = QualityCostPredictor.load(path)
    predictor.eval()

    class SoftJudge:
        def grade(self, prompt: str, response: str) -> Tuple[float, Dict[str, Any]]:
            result = predictor.predict_production(prompt, response)
            reward = float(result.get("reward_logit", 0.0))
            return reward, {"source": "soft", "raw": result}

    return SoftJudge()


def create_tiered_judge(
    soft_model_path: Optional[Path] = None,
    teacher_model: str = "openai/gpt-4o",
    teacher_max_tokens: int = 64,
    use_teacher: bool = True,
) -> Judge:
    """
    Create a tiered judge (soft + optional hard teacher).

    Automatically uses the teacher (LLM-as-a-Judge) for hard prompts
    (math, code, logic) and the soft local grader for easy prompts.
    """
    from llm_jury.async_bandit.quality_cost_predictor import QualityCostPredictor
    from llm_jury.async_bandit.tiered_grader import (
        OpenRouterTeacherVerifier,
        TieredGrader,
    )

    default_path = Path(__file__).parent.parent.parent / "data" / "quality_predictor" / "best_quality_predictor.pt"
    path = soft_model_path or default_path

    soft = QualityCostPredictor.load(path)
    soft.eval()

    teacher = OpenRouterTeacherVerifier(model_id=teacher_model, max_tokens=teacher_max_tokens) if use_teacher else None
    grader = TieredGrader(soft_grader=soft, teacher_verifier=teacher)

    class TieredJudge:
        def grade(self, prompt: str, response: str) -> Tuple[float, Dict[str, Any]]:
            result = grader.predict_production(prompt, response, reward_normalizer=None)
            reward = float(result.get("reward_logit", 0.0))
            return reward, {"source": "tiered", "used_teacher": result.get("tiered_used_teacher", False), "raw": result}

    return TieredJudge()


def create_custom_judge(
    grade_fn: Callable[[str, str], Tuple[float, Dict[str, Any]]],
) -> Judge:
    """
    Create a judge from a custom grading function.

    Example:
        def my_grader(prompt, response):
            # Your custom logic
            score = some_api_call(prompt, response)
            return score, {"custom": True}

        judge = create_custom_judge(my_grader)
    """

    class CustomJudge:
        def grade(self, prompt: str, response: str) -> Tuple[float, Dict[str, Any]]:
            return grade_fn(prompt, response)

    return CustomJudge()
