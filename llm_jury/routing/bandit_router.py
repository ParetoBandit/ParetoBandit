"""
Production-grade contextual bandit router (Hot Path) + async grader loop (Cold Path).

Design goals (per our architecture discussion):
  - Router (hot path): millisecond-scale model selection using LinUCB.
  - Grader (cold path): asynchronous competence grading using QualityCostPredictor.
  - Learning signal: reward_z (clipped z-score) derived from reward_raw (clipped P_correct).
  - Persistence: store bandit parameters AND reward normalizer state so the system
    is "primed" immediately on startup (no day-1 cold start).

Important terminology:
  - This is Type B "competence risk" routing (avoid low-competence answers).
  - Policy safety (Type A) should be handled by provider / separate safety filter.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:  # pragma: no cover
    raise ImportError("Missing dependency: sentence-transformers") from e

from llm_jury.neural_routing.quality_cost_predictor import (
    QualityCostPredictor,
    LogitReward,
    RunningZScoreNormalizer,
)


DEFAULT_CONTEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # fast 384-dim


@dataclass
class RoutingLog:
    """
    The golden tuple recorded per request.

    Store this (and the response later) to enable the asynchronous learning loop.
    """

    request_id: str
    timestamp_s: float
    prompt: str
    context_vector: List[float]
    selected_model: str
    # Bandit prediction (quality signal) for the chosen model.
    # Note: selection uses predicted_utility, not just predicted_quality.
    predicted_quality: float
    # Utility score used for routing:
    #   U = quality_hat - lambda_cost * cost - lambda_latency * latency
    predicted_utility: float
    propensity: float  # P(selected_model | policy, x) needed for off-policy eval later
    # Filled in by the cold path:
    response_text: Optional[str] = None
    reward_raw: Optional[float] = None
    reward_logit: Optional[float] = None
    reward_z: Optional[float] = None
    grader_meta: Optional[Dict[str, Any]] = None


class DisjointLinUCBPolicy:
    """
    Disjoint LinUCB: one ridge regression per arm.

    We learn directly in normalized reward space (reward_z), which gives stronger
    signal and faster convergence under score compression.
    """

    def __init__(
        self,
        model_names: List[str],
        dim: int = 384,
        alpha: float = 0.5,
        ridge_lambda: float = 1.0,
        recompute_inv_every: int = 50,
    ):
        self.models = list(model_names)
        self.dim = int(dim)
        self.alpha = float(alpha)
        self.ridge_lambda = float(ridge_lambda)
        self.recompute_inv_every = int(recompute_inv_every)

        self.A: Dict[str, np.ndarray] = {m: np.eye(self.dim) * self.ridge_lambda for m in self.models}
        self.b: Dict[str, np.ndarray] = {m: np.zeros(self.dim, dtype=np.float64) for m in self.models}
        self.A_inv: Dict[str, np.ndarray] = {m: np.linalg.inv(self.A[m]) for m in self.models}
        self._updates: Dict[str, int] = {m: 0 for m in self.models}

    def select_arm(
        self,
        x: np.ndarray,
        *,
        candidate_models: Optional[List[str]] = None,
        epsilon: float = 0.0,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[str, float, float]:
        """
        Returns (model_name, predicted_ucb, propensity).

        - epsilon-greedy exploration is supported (propensity is computed accordingly).
        """
        rng = rng or np.random.default_rng()
        candidates = candidate_models if candidate_models else self.models
        candidates = [m for m in candidates if m in self.A]
        if not candidates:
            raise ValueError("No candidate models available for selection.")

        eps = float(max(0.0, min(1.0, epsilon)))
        if eps > 0 and rng.random() < eps:
            chosen = candidates[int(rng.integers(0, len(candidates)))]
            return chosen, float("nan"), (eps / len(candidates))

        best_model = candidates[0]
        best_ucb = -float("inf")

        for m in candidates:
            theta = self.A_inv[m] @ self.b[m]
            mean = float(theta.dot(x))
            var = float(x.dot(self.A_inv[m]).dot(x))
            std = float(np.sqrt(max(var, 1e-12)))
            ucb = mean + self.alpha * std
            if ucb > best_ucb:
                best_ucb = ucb
                best_model = m

        # propensity under epsilon-greedy:
        #   P(best) = (1-eps) + eps/|candidates|
        #   P(other) = eps/|candidates|
        prop = (1.0 - eps) + (eps / len(candidates))
        return best_model, float(best_ucb), float(prop)

    def update(self, model: str, x: np.ndarray, reward_z: float) -> None:
        if model not in self.A:
            return
        self.A[model] += np.outer(x, x)
        self.b[model] += float(reward_z) * x
        self._updates[model] += 1
        if self._updates[model] % self.recompute_inv_every == 0:
            self.A_inv[model] = np.linalg.inv(self.A[model])

    def to_state_dict(self) -> Dict[str, Any]:
        return {
            "dim": self.dim,
            "alpha": self.alpha,
            "ridge_lambda": self.ridge_lambda,
            "recompute_inv_every": self.recompute_inv_every,
            "models": self.models,
            "A": {m: self.A[m].tolist() for m in self.models},
            "b": {m: self.b[m].tolist() for m in self.models},
            "updates": dict(self._updates),
        }

    @classmethod
    def from_state_dict(cls, d: Dict[str, Any]) -> "DisjointLinUCBPolicy":
        obj = cls(
            model_names=list(d["models"]),
            dim=int(d["dim"]),
            alpha=float(d["alpha"]),
            ridge_lambda=float(d.get("ridge_lambda", 1.0)),
            recompute_inv_every=int(d.get("recompute_inv_every", 50)),
        )
        for m in obj.models:
            obj.A[m] = np.asarray(d["A"][m], dtype=np.float64)
            obj.b[m] = np.asarray(d["b"][m], dtype=np.float64)
            obj.A_inv[m] = np.linalg.inv(obj.A[m])
        obj._updates = {k: int(v) for k, v in d.get("updates", {}).items()}
        return obj


class BanditRouter:
    """
    Public entrypoint for production routing.

    Hot path:
      - embed prompt -> x
      - choose model via LinUCB (in reward_z space)
      - return choice + request_id (and a log record)

    Cold path:
      - given (request_id -> response_text), grade and update bandit
    """

    def __init__(
        self,
        model_registry: Dict[str, Dict[str, Any]],
        *,
        context_model: str = DEFAULT_CONTEXT_MODEL,
        alpha: float = 0.5,
        state_path: Optional[Path] = None,
        normalizer_init: Optional[RunningZScoreNormalizer] = None,
        reward_mode: str = "z",
        embedding_dim: int = 384,
    ):
        self.registry = dict(model_registry)
        self.encoder = SentenceTransformer(context_model)
        self.bandit = DisjointLinUCBPolicy(list(self.registry.keys()), dim=embedding_dim, alpha=alpha)

        # Reward normalizer for competence reward_z. Persist this state in production.
        self.normalizer = normalizer_init or RunningZScoreNormalizer(
            mean_init=0.65,
            std_init=0.05,
            alpha=0.01,
            clamp=3.0,
            auto_init_from_first_sample=True,
        )

        # Optional: logit reward transform (stationary stretch). This is safe due to strict clipping.
        self.reward_mode = str(reward_mode).lower().strip()
        if self.reward_mode not in {"z", "logit"}:
            raise ValueError("reward_mode must be 'z' or 'logit'")
        self.logit_reward = LogitReward(epsilon=1e-4)

        self.state_path = Path(state_path) if state_path is not None else None
        self.logs: List[RoutingLog] = []

    def route(
        self,
        prompt: str,
        *,
        lambda_cost: float = 0.0,
        lambda_latency: float = 0.0,
        max_latency_s: Optional[float] = None,
        epsilon: float = 0.05,
        candidate_models: Optional[List[str]] = None,
        use_ucb_for_quality: bool = True,
    ) -> Tuple[str, RoutingLog]:
        """
        Hot path: route by maximizing real-time utility.

        For each candidate model:
            U_model = quality_hat - lambda_cost * cost - lambda_latency * latency

        - quality_hat is predicted by the bandit (mean or UCB).
        - cost/latency are deterministic lookups from the registry.
        - Learning still uses the grader's reward (quality-only).

        Args:
            lambda_cost: penalty per unit cost (user/business knob)
            lambda_latency: penalty per second (user/business knob)
            max_latency_s: optional hard constraint (filter models exceeding this)
            epsilon: epsilon-greedy exploration rate
            candidate_models: optional allowlist of models (e.g. compliance mask)
            use_ucb_for_quality: if True use UCB as quality_hat, else use mean.
        """
        x = self.encoder.encode(prompt)
        x = np.asarray(x, dtype=np.float64)

        candidates = candidate_models if candidate_models else list(self.registry.keys())
        candidates = [m for m in candidates if m in self.bandit.A]

        # Optional hard constraint: latency ceiling
        if max_latency_s is not None:
            cap = float(max_latency_s)
            filtered: List[str] = []
            for m in candidates:
                lat = float(self.registry.get(m, {}).get("latency_s", 0.0) or 0.0)
                if lat <= cap:
                    filtered.append(m)
            candidates = filtered

        if not candidates:
            raise ValueError("No candidate models available after applying constraints.")

        eps = float(max(0.0, min(1.0, epsilon)))
        rng = np.random.default_rng()
        explore = eps > 0 and rng.random() < eps

        best_model = candidates[0]
        best_utility = -float("inf")
        best_quality = float("nan")

        # Compute utilities for all candidates (fast: O(#arms * d^2) dominated by dot products).
        for m in candidates:
            theta = self.bandit.A_inv[m] @ self.bandit.b[m]
            mean = float(theta.dot(x))
            var = float(x.dot(self.bandit.A_inv[m]).dot(x))
            std = float(np.sqrt(max(var, 1e-12)))
            ucb = mean + self.bandit.alpha * std
            quality_hat = float(ucb if use_ucb_for_quality else mean)

            # Deterministic penalties (not learned)
            reg = self.registry.get(m, {})
            cost = float(reg.get("cost", 0.0) or 0.0)
            latency = float(reg.get("latency_s", 0.0) or 0.0)
            utility = quality_hat - float(lambda_cost) * cost - float(lambda_latency) * latency

            if utility > best_utility:
                best_utility = float(utility)
                best_model = m
                best_quality = float(quality_hat)

        if explore:
            best_model = candidates[int(rng.integers(0, len(candidates)))]
            # Recompute predicted quality/utility for the randomly chosen model (for logging).
            theta = self.bandit.A_inv[best_model] @ self.bandit.b[best_model]
            mean = float(theta.dot(x))
            var = float(x.dot(self.bandit.A_inv[best_model]).dot(x))
            std = float(np.sqrt(max(var, 1e-12)))
            ucb = mean + self.bandit.alpha * std
            best_quality = float(ucb if use_ucb_for_quality else mean)
            reg = self.registry.get(best_model, {})
            cost = float(reg.get("cost", 0.0) or 0.0)
            latency = float(reg.get("latency_s", 0.0) or 0.0)
            best_utility = float(best_quality - float(lambda_cost) * cost - float(lambda_latency) * latency)

        # propensity under epsilon-greedy with deterministic argmax utility
        prop = (eps / len(candidates)) if explore else ((1.0 - eps) + (eps / len(candidates)))

        model = best_model
        pred_quality = best_quality
        pred_utility = best_utility
        req_id = str(time.time_ns())
        log = RoutingLog(
            request_id=req_id,
            timestamp_s=time.time(),
            prompt=prompt,
            context_vector=x.tolist(),
            selected_model=model,
            predicted_quality=float(pred_quality),
            predicted_utility=float(pred_utility),
            propensity=float(prop),
        )
        self.logs.append(log)
        return model, log

    def process_feedback(
        self,
        grader: QualityCostPredictor,
        *,
        responses_by_request_id: Dict[str, str],
    ) -> List[RoutingLog]:
        """
        Cold path: attach responses, grade, compute reward_z, update bandit.

        Returns the updated logs for storage.
        """
        updated: List[RoutingLog] = []
        for log in self.logs:
            resp = responses_by_request_id.get(log.request_id)
            if resp is None:
                continue

            prod = grader.predict_production(log.prompt, resp, reward_normalizer=self.normalizer)
            if self.reward_mode == "logit":
                reward_for_update = float(prod["reward_logit"])
            else:
                reward_for_update = prod.get("reward_z")
                if reward_for_update is None:
                    # If no normalizer used, fall back to raw reward (not recommended).
                    reward_for_update = float(prod["reward_raw"])

            x = np.asarray(log.context_vector, dtype=np.float64)
            self.bandit.update(log.selected_model, x, float(reward_for_update))

            log.response_text = resp
            log.reward_raw = float(prod["reward_raw"])
            log.reward_logit = float(prod.get("reward_logit")) if prod.get("reward_logit") is not None else None
            log.reward_z = float(prod.get("reward_z")) if prod.get("reward_z") is not None else None
            log.grader_meta = {
                k: v
                for k, v in prod.items()
                if k not in {"reward_raw", "reward_logit", "reward_z"}
            }
            updated.append(log)

        # Clear buffer (production would stream these logs to storage instead).
        self.logs = []
        return updated

    def save_state(self, path: Optional[Path] = None) -> None:
        p = Path(path) if path is not None else self.state_path
        if p is None:
            raise ValueError("No state_path provided.")
        p.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "version": 1,
            "reward_mode": self.reward_mode,
            "bandit": self.bandit.to_state_dict(),
            "normalizer": self.normalizer.state_dict(),
            "registry": self.registry,
        }
        p.write_text(json.dumps(state, indent=2))

    @classmethod
    def load_state(
        cls,
        path: Path,
        *,
        context_model: str = DEFAULT_CONTEXT_MODEL,
    ) -> "BanditRouter":
        p = Path(path)
        d = json.loads(p.read_text())
        router = cls(
            model_registry=d["registry"],
            context_model=context_model,
            alpha=float(d["bandit"]["alpha"]),
            state_path=p,
            normalizer_init=RunningZScoreNormalizer.from_state_dict(d["normalizer"]),
            reward_mode=str(d.get("reward_mode", "z")),
            embedding_dim=int(d["bandit"]["dim"]),
        )
        router.bandit = DisjointLinUCBPolicy.from_state_dict(d["bandit"])
        return router

