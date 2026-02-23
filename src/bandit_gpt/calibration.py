"""
Calibration API: Generate PCA and warmup priors for custom sentence transformers.

When using a non-default encoder model, the shipped PCA artifact and warmup
priors become semantically invalid.  This module provides the two functions
needed to produce compatible artifacts for any SentenceTransformer model:

    from bandit_gpt.calibration import train_pca, generate_warmup_priors

    pca = train_pca(prompts, encoder_model="your-model", output_path="pca.joblib")
    priors = generate_warmup_priors(
        rewards_data, encoder_model="your-model", pca=pca, output_path="priors.joblib"
    )

Both functions return the artifact in-memory and optionally persist it to disk.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import joblib
import numpy as np
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


def train_pca(
    prompts: list[str],
    encoder_model: str,
    n_components: int = 32,
    output_path: Path | str | None = None,
    batch_size: int = 64,
) -> PCA:
    """Train a PCA artifact from prompt embeddings.

    Encodes *prompts* with the given SentenceTransformer *encoder_model*,
    fits PCA, and optionally saves the result to *output_path*.

    Args:
        prompts: Corpus of representative text samples (>= 100 recommended).
        encoder_model: HuggingFace SentenceTransformer model name or path.
        n_components: Number of PCA components to retain.
        output_path: If provided, the fitted PCA is persisted via ``joblib``.
        batch_size: Batch size for the encoder's ``.encode()`` call.

    Returns:
        The fitted ``sklearn.decomposition.PCA`` object.

    Raises:
        ValueError: If *prompts* is empty or too short for the requested
            number of components.
    """
    if not prompts:
        raise ValueError("prompts must be a non-empty list of strings")
    if len(prompts) < n_components:
        raise ValueError(
            f"Need at least {n_components} prompts to fit {n_components} PCA "
            f"components, got {len(prompts)}"
        )

    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(encoder_model)
    logger.info("Encoding %d prompts with '%s'...", len(prompts), encoder_model)

    embeddings = encoder.encode(
        prompts,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=batch_size,
        convert_to_numpy=True,
    )
    logger.info("Embeddings shape: %s", embeddings.shape)

    pca = PCA(n_components=n_components)
    pca.fit(embeddings)

    explained = float(np.sum(pca.explained_variance_ratio_))
    logger.info(
        "PCA trained: %dD -> %dD (explained variance %.2f%%)",
        embeddings.shape[1],
        n_components,
        explained * 100,
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pca, output_path)
        logger.info("PCA saved to %s", output_path)

    return pca


def generate_warmup_priors(
    rewards_data: list[dict],
    encoder_model: str,
    pca: Union[PCA, Path, str],
    plasticity: float = 0.1,
    output_path: Path | str | None = None,
    batch_size: int = 64,
) -> dict:
    """Generate warmup priors (A, b matrices) for LinUCB.

    Processes a labelled dataset of prompts with per-model rewards, encodes
    each prompt, projects through PCA, and accumulates LinUCB sufficient
    statistics.

    Args:
        rewards_data: List of dicts, each containing::

                {"prompt": str, "rewards": {"model_id": float, ...}}

        encoder_model: HuggingFace SentenceTransformer model name or path.
        pca: A fitted PCA object **or** a path to a joblib-serialised one.
            Must have been trained with the same *encoder_model*.
        plasticity: Scaling factor applied to A and b after accumulation.
            Lower values yield softer priors that are faster to override
            with online observations.  Default ``0.1``.
        output_path: If provided, the priors dict is persisted via ``joblib``.
        batch_size: Batch size for the encoder's ``.encode()`` call.

    Returns:
        A dict with keys ``A``, ``b``, ``models``, ``n_prompts``,
        ``context_dim``, ``pca_components``, ``plasticity``, and
        ``reward_source``.

    Raises:
        ValueError: If *rewards_data* is empty or malformed.
    """
    if not rewards_data:
        raise ValueError("rewards_data must be a non-empty list")

    # Resolve PCA ----------------------------------------------------------
    if isinstance(pca, (str, Path)):
        pca = joblib.load(Path(pca))

    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(encoder_model)

    # Discover model set ---------------------------------------------------
    all_models: set[str] = set()
    for entry in rewards_data:
        all_models.update(entry["rewards"].keys())
    all_models_list = sorted(all_models)

    context_dim = pca.n_components_ + 1  # PCA features + bias

    # Initialise LinUCB sufficient statistics ------------------------------
    A: dict[str, np.ndarray] = {m: np.eye(context_dim) for m in all_models_list}
    b: dict[str, np.ndarray] = {m: np.zeros(context_dim) for m in all_models_list}

    processed = 0
    skipped = 0

    logger.info(
        "Building warmup priors for %d models from %d samples...",
        len(all_models_list),
        len(rewards_data),
    )

    for entry in rewards_data:
        prompt = entry["prompt"]
        rewards = entry["rewards"]

        try:
            embedding = encoder.encode(
                prompt,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            if np.isnan(embedding).any() or np.isinf(embedding).any():
                skipped += 1
                continue

            embedding = pca.transform(embedding.reshape(1, -1)).flatten()

            if np.isnan(embedding).any() or np.isinf(embedding).any():
                skipped += 1
                continue

            context = np.append(embedding, 1.0)  # bias term
            x_col = context.reshape(-1, 1)

            for model_id, reward in rewards.items():
                A[model_id] += x_col @ x_col.T
                b[model_id] += reward * context

            processed += 1

        except Exception:
            skipped += 1
            continue

    # Plasticity scaling ---------------------------------------------------
    for model_id in all_models_list:
        A[model_id] *= plasticity
        b[model_id] *= plasticity

    logger.info(
        "Warmup priors built: %d processed, %d skipped, plasticity=%.2f",
        processed,
        skipped,
        plasticity,
    )

    state = {
        "A": A,
        "b": b,
        "models": all_models_list,
        "n_prompts": processed,
        "n": processed,
        "context_dim": context_dim,
        "pca_components": pca.n_components_,
        "plasticity": plasticity,
        "reward_source": "calibration_api",
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(state, output_path)
        logger.info("Warmup priors saved to %s", output_path)

    return state
