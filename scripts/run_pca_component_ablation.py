#!/usr/bin/env python3
"""
PCA component-count ablation for ParetoBandit.

This script answers the reviewer question: "How many PCA components do we need
before routing performance saturates?"

Protocol (train-then-freeze, matching `experiments/03_figure/run_prequential.py`)
----------------------------------------------------------------------------
1) Fit PCA on an *external* prompt corpus (offline battles) to avoid leakage.
2) For each component count k (for a fixed K-model portfolio):
   - Project prompt embeddings to k dims (+ bias => k+1 context dim).
   - Generate warmup priors for the portfolio models using the fixed
     prior-train pool from `src/artifacts/splits_three_way.json` (created from
     the full-coverage 43-model data).
   - Train BanditRouter on the online-learn pool and evaluate greedily
     (alpha=0) on the holdout pool.

Outputs
-------
- JSON results file with explained variance, holdout reward mean/std, and
  metadata needed for reproducibility.
- Optional PNG plot (reward + explained variance vs k).

Notes
-----
- Uses the current default SentenceTransformer from `pareto_bandit.config`
  (default: `all-MiniLM-L6-v2`) unless overridden.
- Runs on CPU by default for reproducibility and to avoid device-dependent
  numerical differences.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import joblib
import numpy as np
from sklearn.decomposition import PCA

# Local imports (script is executed from repo root).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys

sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from pareto_bandit.config import (
    DEFAULT_SENTENCE_TRANSFORMER,
    DEV_DATA_PATH_ALL_MODELS,
    HOLDOUT_DATA_PATH_ALL_MODELS,
    LMSYS_BATTLES_PATH,
)
from pareto_bandit.rewards import extract_reward

# Reuse the canonical train-then-freeze utilities used in the paper code.
from utils.router_factory import create_experiment_router
from utils.multimodel import MODEL_CATALOG, PORTFOLIO_K10, PORTFOLIO_K5, build_model_registry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Train-then-freeze utilities (kept local to avoid importing numbered folders)
# ---------------------------------------------------------------------------

REWARD_THEORETICAL_MIN: float = 0.0
REWARD_THEORETICAL_MAX: float = 1.0


def compute_reward_normalization() -> Tuple[float, float]:
    """Return theoretical reward bounds for normalization.

    ParetoBandit's canonical reward is mean(vote × confidence), which lies in [0, 1].
    Using theoretical bounds avoids any counterfactual leakage from the reward matrix.
    """
    return REWARD_THEORETICAL_MIN, REWARD_THEORETICAL_MAX


def train_router(
    router: Any,
    *,
    train_data: Sequence[Mapping[str, Any]],
    train_embeddings: Sequence[np.ndarray],
    r_min: float,
    r_range: float,
    shuffle: bool,
) -> int:
    """Train router online via `route()` + `process_feedback()`."""
    n_steps = len(train_data)
    order = np.random.permutation(n_steps) if shuffle else np.arange(n_steps)
    for idx in order:
        p, x = train_data[idx], train_embeddings[idx]
        model, log = router.route(x, total_steps=n_steps)
        raw_reward = float(p["rewards"][model])
        norm_reward = (raw_reward - r_min) / r_range if r_range > 1e-6 else 0.5
        router.process_feedback(log.request_id, norm_reward)
    return n_steps


def evaluate_router_frozen(
    router: Any,
    *,
    eval_data: Sequence[Mapping[str, Any]],
    eval_embeddings: Sequence[np.ndarray],
    costs: Mapping[str, float],
    total_steps: int,
) -> Tuple[float, float]:
    """Evaluate a trained router greedily on holdout (no feedback updates)."""
    rng_state = np.random.get_state()

    r_total = 0.0
    c_total = 0.0
    with router.exploit():
        for p, x in zip(eval_data, eval_embeddings):
            model, _log = router.route(x, total_steps=total_steps)
            r_total += float(p["rewards"][model])
            c_total += float(costs.get(model, 0.0))

    np.random.set_state(rng_state)

    n = max(1, len(eval_data))
    return r_total / n, c_total / n


# ---------------------------------------------------------------------------
# Reproducibility helpers
# ---------------------------------------------------------------------------


def set_all_seeds(seed: int) -> None:
    """Set global seeds for strict reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_rewards_grouped_by_prompt(
    data_path: Path,
    *,
    min_models: int,
) -> Dict[str, Dict[str, float]]:
    """Load rewards from a `*.jsonl.gz` file grouped as {prompt: {model: reward}}.

    Keeps only prompts that have at least *min_models* model rewards.
    """
    rewards: Dict[str, Dict[str, float]] = {}
    with gzip.open(data_path, "rt") as f:
        for line in f:
            entry = json.loads(line)
            if not entry.get("ok"):
                continue
            prompt = entry["prompt"]
            model_id = entry["model_id"]
            rewards.setdefault(prompt, {})[model_id] = extract_reward(entry)

    full = {p: r for p, r in rewards.items() if len(r) >= min_models}
    return full


def _load_rewards_for_model_set(
    data_path: Path,
    *,
    models: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    """Load rewards grouped by prompt, restricted to an explicit model set.

    Keeps only prompts that have rewards for **all** models in *models*.
    """
    model_set = set(models)
    K = len(models)
    rewards: Dict[str, Dict[str, float]] = {}
    with gzip.open(data_path, "rt") as f:
        for line in f:
            entry = json.loads(line)
            if not entry.get("ok"):
                continue
            model_id = entry["model_id"]
            if model_id not in model_set:
                continue
            prompt = entry["prompt"]
            rewards.setdefault(prompt, {})[model_id] = extract_reward(entry)

    full = {p: r for p, r in rewards.items() if len(r) == K}
    return full


def _intersection_of_models(rewards_by_prompt: Mapping[str, Mapping[str, float]]) -> List[str]:
    """Return the model IDs present for *every* prompt in the mapping."""
    prompts = list(rewards_by_prompt.keys())
    if not prompts:
        return []
    common: set[str] = set(rewards_by_prompt[prompts[0]].keys())
    for p in prompts[1:]:
        common &= set(rewards_by_prompt[p].keys())
    return sorted(common)


def _load_three_way_splits(splits_path: Path) -> Tuple[List[str], List[str]]:
    """Load (prior_train_pool, online_learn_pool) prompt lists."""
    splits = json.loads(splits_path.read_text())
    prior_train = list(splits["prior_train_pool"])
    online_learn = list(splits["online_learn_pool"])
    return prior_train, online_learn


def _load_battle_prompts(
    battles_path: Path,
    *,
    max_prompts: int,
) -> List[str]:
    """Load up to *max_prompts* unique prompts from offline battles JSONL."""
    prompts_seen: set[str] = set()
    prompts: List[str] = []
    with open(battles_path, "r", encoding="utf-8") as f:
        for line in f:
            if len(prompts) >= max_prompts:
                break
            try:
                entry = json.loads(line)
                prompt = entry.get("prompt", "")
                if isinstance(prompt, list):
                    prompt = prompt[0] if prompt else ""
                if isinstance(prompt, str) and prompt.startswith('["'):
                    try:
                        prompt_list = json.loads(prompt)
                        prompt = prompt_list[0] if prompt_list else ""
                    except Exception:
                        pass
                prompt = str(prompt).strip()
                if not prompt or prompt in prompts_seen:
                    continue
                prompts_seen.add(prompt)
                prompts.append(prompt)
            except Exception:
                continue
    return prompts


# ---------------------------------------------------------------------------
# PCA projector (prefix PCA)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PcaPrefixProjector:
    """A PCA projector that supports fast top-k projection by slicing components."""

    mean_: np.ndarray  # shape (D,)
    components_: np.ndarray  # shape (Kmax, D)
    explained_variance_ratio_: np.ndarray  # shape (Kmax,)

    @property
    def max_components(self) -> int:
        return int(self.components_.shape[0])

    def cumulative_explained_variance(self, k: int) -> float:
        if k < 1 or k > self.max_components:
            raise ValueError(f"k must be in [1, {self.max_components}], got {k}")
        return float(np.sum(self.explained_variance_ratio_[:k]))

    def project(self, x: np.ndarray, k: int) -> np.ndarray:
        """Project a 2D matrix of embeddings to k dims.

        Args:
            x: 2D array of shape (n, D)
            k: Number of components to project onto (<= max_components)

        Returns:
            2D array of shape (n, k)
        """
        if x.ndim != 2:
            raise ValueError(f"x must be 2D, got shape {x.shape}")
        if k < 1 or k > self.max_components:
            raise ValueError(f"k must be in [1, {self.max_components}], got {k}")
        centered = x - self.mean_[None, :]
        return centered @ self.components_[:k].T


def fit_pca_prefix(
    embeddings: np.ndarray,
    *,
    max_components: int,
    seed: int,
) -> PcaPrefixProjector:
    """Fit PCA with max_components and return a projector supporting top-k slicing."""
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2D, got {embeddings.shape}")
    if embeddings.shape[0] < max_components:
        raise ValueError(
            f"Need at least {max_components} samples to fit {max_components} components, "
            f"got {embeddings.shape[0]}"
        )

    # Randomized PCA is dramatically faster at (80k x 1024). Fix random_state for determinism.
    pca = PCA(
        n_components=max_components,
        svd_solver="randomized",
        random_state=seed,
    )
    pca.fit(embeddings)
    return PcaPrefixProjector(
        mean_=np.asarray(pca.mean_, dtype=np.float64),
        components_=np.asarray(pca.components_, dtype=np.float64),
        explained_variance_ratio_=np.asarray(pca.explained_variance_ratio_, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# Warmup priors (in-memory -> joblib)
# ---------------------------------------------------------------------------


def build_warmup_priors_from_embeddings(
    *,
    prompts: Sequence[str],
    models: Sequence[str],
    rewards_by_prompt: Mapping[str, Mapping[str, float]],
    prompt_embeddings: Mapping[str, np.ndarray],
    projector: PcaPrefixProjector,
    k: int,
    plasticity: float,
) -> Dict[str, Any]:
    """Build LinUCB warmup priors for a given PCA component count k.

    Produces a joblib-serialisable dict with keys compatible with
    `BanditRouter.create(..., warmup_path=...)`.
    """
    context_dim = k + 1  # PCA dims + bias
    A: Dict[str, np.ndarray] = {m: np.eye(context_dim) for m in models}
    b: Dict[str, np.ndarray] = {m: np.zeros(context_dim) for m in models}

    processed_prompts = 0
    processed_obs = 0
    for p in prompts:
        rewards = rewards_by_prompt.get(p)
        if rewards is None:
            continue
        emb = prompt_embeddings.get(p)
        if emb is None:
            continue
        if np.isnan(emb).any() or np.isinf(emb).any():
            continue

        z = projector.project(emb[None, :], k=k).reshape(-1)
        if np.isnan(z).any() or np.isinf(z).any():
            continue
        x = np.append(z, 1.0).reshape(-1, 1)

        for m in models:
            r = float(rewards[m])
            A[m] += x @ x.T
            b[m] += (r * x).reshape(-1)
            processed_obs += 1
        processed_prompts += 1

    for m in models:
        A[m] *= plasticity
        b[m] *= plasticity

    state: Dict[str, Any] = {
        "A": A,
        "b": b,
        "models": list(models),
        "context_dim": context_dim,
        "pca_components": k,
        "plasticity": plasticity,
        "n_prompts": processed_prompts,
        # IMPORTANT: Router scales priors by prior_n_effective / n.
        "n": processed_prompts,
        "n_observations": processed_obs,
        "reward_source": "component_ablation",
    }
    return state


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------


def _encode_prompts(
    encoder_model: str,
    prompts: Sequence[str],
    *,
    device: str,
    batch_size: int,
) -> np.ndarray:
    """Encode prompts to a 2D numpy array using SentenceTransformer."""
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(encoder_model, device=device)
    emb = encoder.encode(
        list(prompts),
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=len(prompts) >= 200,
    )
    return np.asarray(emb, dtype=np.float32)


def _build_minimal_registry(models: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """Build a minimal router registry for offline evaluation (no cost penalty)."""
    reg: Dict[str, Dict[str, Any]] = {}
    for m in models:
        reg[m] = {
            "model_id": m,
            # Provide costs so router init never raises MissingCostError.
            "input_cost_per_m": 1.0,
            "output_cost_per_m": 1.0,
            "initial_quality": 0.5,
        }
    return reg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PCA component-count ablation (train-then-freeze).")
    p.add_argument("--encoder-model", type=str, default=DEFAULT_SENTENCE_TRANSFORMER)
    p.add_argument("--device", type=str, default="cpu", help="torch device (default: cpu)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument(
        "--portfolio",
        type=str,
        default="k10",
        choices=("k10", "k5"),
        help="Which experiment portfolio to evaluate (default: k10).",
    )
    p.add_argument(
        "--models",
        type=str,
        default="",
        help="Optional comma-separated model IDs (overrides --portfolio).",
    )
    p.add_argument(
        "--components",
        type=str,
        default="8,16,32,64,96,128,192,256",
        help="Comma-separated list of PCA component counts to evaluate.",
    )
    p.add_argument("--max-components", type=int, default=256, help="Max PCA components to fit once.")
    p.add_argument("--max-pca-prompts", type=int, default=50000)
    p.add_argument("--prior-plasticity", type=float, default=0.1)
    p.add_argument("--prior-n-effective", type=float, default=10.0)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument(
        "--forgetting-factor",
        type=float,
        default=1.0,
        help="Exponential decay for past observations (1.0 = stationary, no forgetting).",
    )
    p.add_argument(
        "--corralling-gamma",
        type=float,
        default=0.05,
        help="Corralling mixing/safety parameter (not the same as forgetting).",
    )
    p.add_argument("--n-trials", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--splits",
        type=str,
        default=str(PROJECT_ROOT / "src" / "artifacts" / "splits_three_way.json"),
    )
    p.add_argument("--out-dir", type=str, default=str(PROJECT_ROOT / "results" / "pca_components_ablation"))
    p.add_argument("--plot", action="store_true", help="Also save a PNG plot.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    set_all_seeds(int(args.seed))

    portfolio_models: List[str]
    if str(args.models).strip():
        portfolio_models = [m.strip() for m in str(args.models).split(",") if m.strip()]
    else:
        portfolio_models = list(PORTFOLIO_K10 if args.portfolio == "k10" else PORTFOLIO_K5)
    if len(portfolio_models) < 2:
        raise ValueError("Need at least 2 models for routing.")

    ks = [int(x.strip()) for x in str(args.components).split(",") if x.strip()]
    if not ks:
        raise ValueError("No component counts provided via --components")
    if any(k <= 0 for k in ks):
        raise ValueError(f"All component counts must be positive, got {ks}")
    max_k = int(args.max_components)
    if max(ks) > max_k:
        raise ValueError(f"max(components)={max(ks)} exceeds --max-components={max_k}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("PCA COMPONENT-COUNT ABLATION")
    logger.info("=" * 70)
    logger.info(f"encoder_model : {args.encoder_model}")
    logger.info(f"device        : {args.device}")
    logger.info(f"portfolio     : {args.portfolio}")
    logger.info(f"models (K)    : {len(portfolio_models)}")
    logger.info(f"components    : {ks}")
    logger.info(f"seed          : {args.seed}")
    logger.info(f"n_trials      : {args.n_trials}")
    logger.info(f"prior_n_eff   : {args.prior_n_effective}")
    logger.info(f"forget_factor : {args.forgetting_factor}")
    logger.info(f"corral_gamma  : {args.corralling_gamma}")
    logger.info(f"out_dir       : {out_dir}")

    # ---------------------------------------------------------------------
    # 1) Load reward data (restricted to the chosen portfolio models).
    # ---------------------------------------------------------------------
    logger.info("\n1) Loading reward data for portfolio models ...")
    dev_rewards = _load_rewards_for_model_set(Path(DEV_DATA_PATH_ALL_MODELS), models=portfolio_models)
    holdout_rewards = _load_rewards_for_model_set(Path(HOLDOUT_DATA_PATH_ALL_MODELS), models=portfolio_models)
    logger.info(f"  dev prompts with full K coverage    : {len(dev_rewards)}")
    logger.info(f"  holdout prompts with full K coverage: {len(holdout_rewards)}")

    models = list(portfolio_models)
    missing_from_catalog = [m for m in models if m not in MODEL_CATALOG]
    if missing_from_catalog:
        raise ValueError(
            "Some portfolio models are missing from MODEL_CATALOG in experiments/utils/multimodel.py: "
            f"{missing_from_catalog}"
        )

    # ---------------------------------------------------------------------
    # 2) Load fixed three-way splits.
    # ---------------------------------------------------------------------
    splits_path = Path(args.splits)
    prior_train_pool, online_learn_pool = _load_three_way_splits(splits_path)
    prior_train_pool = [p for p in prior_train_pool if p in dev_rewards]
    online_learn_pool = [p for p in online_learn_pool if p in dev_rewards]
    if not prior_train_pool or not online_learn_pool:
        raise ValueError(
            "Splits do not intersect with dev full-coverage prompts. "
            "Regenerate splits or check --splits path."
        )
    logger.info("\n2) Splits")
    logger.info(f"  prior_train_pool : {len(prior_train_pool)}")
    logger.info(f"  online_learn_pool: {len(online_learn_pool)}")

    # Build train/eval datasets (prompt + rewards dict).
    train_data = [{"prompt": p, "rewards": {m: float(dev_rewards[p][m]) for m in models}} for p in online_learn_pool]
    eval_prompts = sorted(holdout_rewards.keys())
    eval_data = [{"prompt": p, "rewards": {m: float(holdout_rewards[p][m]) for m in models}} for p in eval_prompts]

    # ---------------------------------------------------------------------
    # 3) Fit PCA on offline battles prompts (external corpus).
    # ---------------------------------------------------------------------
    logger.info("\n3) Fitting PCA prefix on offline battles prompts (no leakage) ...")
    t0 = time.time()
    battle_prompts = _load_battle_prompts(Path(LMSYS_BATTLES_PATH), max_prompts=int(args.max_pca_prompts))
    if len(battle_prompts) < max_k:
        raise ValueError(f"Need at least {max_k} battle prompts, got {len(battle_prompts)}")
    logger.info(f"  loaded battle prompts: {len(battle_prompts)}")

    # Encode battle prompts once.
    emb_cache_path = out_dir / f"battle_embeddings_{args.encoder_model.replace('/', '_')}_{len(battle_prompts)}.npy"
    if emb_cache_path.exists():
        logger.info(f"  loading cached embeddings: {emb_cache_path.name}")
        battle_emb = np.load(emb_cache_path)
    else:
        logger.info("  encoding battle prompts (this is the slow step) ...")
        battle_emb = _encode_prompts(
            str(args.encoder_model),
            battle_prompts,
            device=str(args.device),
            batch_size=int(args.batch_size),
        )
        np.save(emb_cache_path, battle_emb)
        logger.info(f"  cached embeddings to: {emb_cache_path.name}")

    projector = fit_pca_prefix(battle_emb, max_components=max_k, seed=int(args.seed))
    logger.info(f"  PCA fit complete in {time.time() - t0:.1f}s (max_k={max_k})")

    # ---------------------------------------------------------------------
    # 4) Encode dev/holdout prompts once (raw embeddings), then sweep k.
    # ---------------------------------------------------------------------
    logger.info("\n4) Encoding dev/holdout prompts once ...")
    all_needed_prompts = list(dict.fromkeys(prior_train_pool + online_learn_pool + eval_prompts))
    raw_all = _encode_prompts(
        str(args.encoder_model),
        all_needed_prompts,
        device=str(args.device),
        batch_size=int(args.batch_size),
    )
    raw_by_prompt: Dict[str, np.ndarray] = {p: raw_all[i] for i, p in enumerate(all_needed_prompts)}

    # Use the experiment catalog for registry + costs (cost penalty is zero here).
    model_registry = build_model_registry(models)
    costs = {m: float(MODEL_CATALOG[m]["cost"]) for m in models}

    r_min, r_max = compute_reward_normalization()
    r_range = r_max - r_min

    results: List[Dict[str, Any]] = []
    for k in ks:
        logger.info("\n" + "-" * 70)
        logger.info(f"k = {k}")
        explained = projector.cumulative_explained_variance(k)
        logger.info(f"  cumulative explained variance: {explained:.2%}")

        # Build warmup priors for this k (prior-train pool only).
        priors = build_warmup_priors_from_embeddings(
            prompts=prior_train_pool,
            models=models,
            rewards_by_prompt=dev_rewards,
            prompt_embeddings=raw_by_prompt,
            projector=projector,
            k=k,
            plasticity=float(args.prior_plasticity),
        )
        priors_path = out_dir / f"priors_43model_k{k}.joblib"
        joblib.dump(priors, priors_path)
        logger.info(f"  priors saved: {priors_path.name} (n={priors['n']})")

        # Precompute features for training/eval at this k.
        train_raw = np.stack([raw_by_prompt[p["prompt"]] for p in train_data], axis=0)
        eval_raw = np.stack([raw_by_prompt[p["prompt"]] for p in eval_data], axis=0)
        train_z = projector.project(train_raw, k=k)
        eval_z = projector.project(eval_raw, k=k)
        train_emb = [np.append(train_z[i], 1.0).astype(np.float64) for i in range(train_z.shape[0])]
        eval_emb = [np.append(eval_z[i], 1.0).astype(np.float64) for i in range(eval_z.shape[0])]

        trial_rewards: List[float] = []
        trial_costs: List[float] = []
        for trial in range(int(args.n_trials)):
            seed = int(args.seed) + 10_000 * trial + k
            set_all_seeds(seed)
            np.random.seed(seed)

            router = create_experiment_router(
                model_registry=model_registry,
                feature_dim=k + 1,
                prior_n_effective=float(args.prior_n_effective),
                alpha=float(args.alpha),
                warmup_path=str(priors_path),
                use_corralling=True,
                corralling_gamma=float(args.corralling_gamma),
                cost_penalty=0.0,
                forgetting_factor=float(args.forgetting_factor),
            )
            total_steps = train_router(
                router,
                train_data=train_data,
                train_embeddings=train_emb,
                r_min=r_min,
                r_range=r_range,
                shuffle=True,
            )
            r, c = evaluate_router_frozen(
                router,
                eval_data=eval_data,
                eval_embeddings=eval_emb,
                costs=costs,
                total_steps=total_steps,
            )
            trial_rewards.append(float(r))
            trial_costs.append(float(c))

        entry = {
            "k": int(k),
            "context_dim": int(k + 1),
            "encoder_model": str(args.encoder_model),
            "device": str(args.device),
            "seed": int(args.seed),
            "n_trials": int(args.n_trials),
            "prior_plasticity": float(args.prior_plasticity),
            "prior_n_effective": float(args.prior_n_effective),
            "alpha": float(args.alpha),
            "explained_variance_cum": explained,
            "holdout_reward_mean": float(np.mean(trial_rewards)),
            "holdout_reward_std": float(np.std(trial_rewards, ddof=1)) if len(trial_rewards) > 1 else 0.0,
            "per_trial_holdout_rewards": trial_rewards,
            "warmup_priors_path": str(priors_path),
            "n_prior_prompts_used": int(priors["n_prompts"]),
        }
        results.append(entry)
        logger.info(
            f"  holdout reward: {entry['holdout_reward_mean']:.4f} "
            f"+/- {entry['holdout_reward_std']:.4f} (n={len(trial_rewards)})"
        )

    out_json = out_dir / "pca_component_ablation_results.json"
    payload = {
        "metadata": {
            "encoder_model": str(args.encoder_model),
            "device": str(args.device),
            "portfolio": str(args.portfolio),
            "models": list(models),
            "components": ks,
            "max_components_fit": max_k,
            "max_pca_prompts": int(args.max_pca_prompts),
            "splits_path": str(splits_path),
            "dev_data_path": str(DEV_DATA_PATH_ALL_MODELS),
            "holdout_data_path": str(HOLDOUT_DATA_PATH_ALL_MODELS),
            "lmsys_battles_path": str(LMSYS_BATTLES_PATH),
            "seed": int(args.seed),
            "n_trials": int(args.n_trials),
            "prior_n_effective": float(args.prior_n_effective),
            "forgetting_factor": float(args.forgetting_factor),
            "corralling_gamma": float(args.corralling_gamma),
        },
        "results": results,
    }
    out_json.write_text(json.dumps(payload, indent=2))
    logger.info(f"\nSaved results: {out_json}")

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        xs = [r["k"] for r in results]
        ys = [r["holdout_reward_mean"] for r in results]
        yerr = [r["holdout_reward_std"] for r in results]
        ev = [r["explained_variance_cum"] for r in results]

        fig, ax1 = plt.subplots(figsize=(7.5, 4.5))
        ax1.errorbar(xs, ys, yerr=yerr, marker="o", linewidth=1.5, capsize=3)
        ax1.set_xlabel("PCA components (k)")
        ax1.set_ylabel("Holdout reward (mean ± std)")
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        ax2.plot(xs, ev, marker="s", linestyle="--", linewidth=1.2, color="tab:orange")
        ax2.set_ylabel("Cumulative explained variance")
        ax2.set_ylim(0.0, 1.0)

        plt.title("PCA component-count ablation (train-then-freeze)")
        fig.tight_layout()
        out_png = out_dir / "pca_component_ablation.png"
        fig.savefig(out_png, dpi=200)
        logger.info(f"Saved plot: {out_png}")


if __name__ == "__main__":
    main()

