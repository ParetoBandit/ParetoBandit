"""
Experiment Router Factory
=========================

Creates ``BanditRouter`` instances that exercise the **full production code
path** (including Corralling and prior loading) while accepting pre-computed
embeddings so experiments avoid reloading the ~2 GB sentence-transformer on
every trial.

Usage
-----
    from experiments.utils.router_factory import create_experiment_router

    router = create_experiment_router(
        model_registry=registry,
        feature_dim=33,
    )
    model, log = router.route(embedding, total_steps=1000)
    router.process_feedback(log.request_id, reward)
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bandit_gpt.feature_service import FeatureService
from bandit_gpt.storage import EphemeralContextStore
from bandit_gpt.router import BanditRouter


def create_experiment_router(
    model_registry: Dict[str, Any],
    feature_dim: int = 33,
    *,
    prior_n_effective: float = 10.0,
    alpha: float = 0.5,
    warmup_path: Optional[str] = None,
    use_corralling: bool = True,
    corralling_learning_rate: float = 0.1,
    corralling_gamma: float = 0.05,
    cost_penalty: float = 0.3,
    forgetting_factor: float = 1.0,
    tabula_rasa_alpha: Optional[float] = None,
    tabula_rasa_forgetting_factor: Optional[float] = None,
    policy: str = "disjoint",
) -> BanditRouter:
    """Build a production ``BanditRouter`` suitable for offline experiments.

    The router uses a lightweight :class:`FeatureService` (no model loading)
    and an :class:`EphemeralContextStore` (RAM-only, no SQLite).  All other
    behaviour—Corralling expert creation, prior loading/scaling—mirrors the
    production ``BanditRouter.create()`` path.

    Parameters
    ----------
    model_registry:
        ``{model_id: config_dict}`` in the same format as ``models.json``.
    feature_dim:
        Total feature-vector length (PCA components + 1 bias).
    prior_n_effective:
        Effective sample size for prior scaling (passed to ``create()``).
        Only affects the warmup expert's priors.
    alpha:
        Exploration coefficient for the warmup expert inside Corralling.
    warmup_path:
        Path to the ``.joblib`` warmup priors file.  ``None`` uses the
        library default.
    use_corralling:
        Whether to enable the Corralling meta-learner.
    corralling_learning_rate:
        Meta-learning rate for expert weight updates.
    corralling_gamma:
        Mixing parameter for Corralling safety.
    cost_penalty:
        Lambda for UCB cost penalty (paper Eq. 4).  Applied at selection
        time in both Corralling experts and the singleton fallback.
    forgetting_factor:
        Exponential decay for past observations in the canonical bandit
        (warmup expert).  ``1.0`` = stationary (no decay).
    tabula_rasa_alpha:
        Per-expert exploration coefficient for the tabula-rasa expert
        inside Corralling.  When ``None``, the legacy schedule
        (``2 * alpha`` decaying to 0.01) is used.
    tabula_rasa_forgetting_factor:
        Per-expert forgetting factor for the tabula-rasa expert inside
        Corralling.  When ``None``, inherits ``forgetting_factor``.
    policy:
        Bandit policy type.  ``"disjoint"`` (default) uses independent
        per-arm ridge regression.  ``"hybrid"`` adds a global shared
        beta parameter (Li et al. 2010 Algorithm 2) that captures
        cross-arm signal and improves sample efficiency.

    Returns
    -------
    BanditRouter
        Fully initialised router ready for ``route()`` / ``process_feedback()``.
    """
    fs = FeatureService.for_precomputed(feature_dim)
    store = EphemeralContextStore()

    router = BanditRouter.create(
        model_registry=model_registry,
        feature_service=fs,
        context_store=store,
        priors="warmup",
        prior_n_effective=prior_n_effective,
        alpha=alpha,
        use_corralling=use_corralling,
        corralling_learning_rate=corralling_learning_rate,
        corralling_gamma=corralling_gamma,
        cost_penalty=cost_penalty,
        forgetting_factor=forgetting_factor,
        tabula_rasa_alpha=tabula_rasa_alpha,
        tabula_rasa_forgetting_factor=tabula_rasa_forgetting_factor,
        policy=policy,
        **({"warmup_path": warmup_path} if warmup_path else {}),
    )
    return router
