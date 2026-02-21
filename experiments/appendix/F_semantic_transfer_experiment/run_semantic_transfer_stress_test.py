#!/usr/bin/env python3
"""
Semantic Transfer Stress Test
=============================

Validates banditGPT's key differentiator: zero-cost onboarding of new models
via semantic transfer of learned priors from similar existing models.

Three complementary experiments:

  Part 1 — Transfer Quality (offline):
      Leave-one-out over K=20. For each model, compare ground-truth theta
      (learned from data) against theta transferred from its nearest neighbor.
      Measures directional accuracy (cosine) and relative error.

  Part 2 — Online Onboarding (bandit simulation):
      Leave-one-out over K=10. For each model, run LinUCB with three
      initialization strategies (Oracle / Semantic Transfer / Tabula Rasa)
      at horizons N=50, 100, 200, 500.  Measures convergence speed and
      cumulative regret.

  Part 3 — Portfolio Growth Simulation:
      Start at K=2, grow to K=5 → K=10 → K=20 by adding new models.
      Compare semantic transfer vs tabula rasa initialization across growth
      phases.  Measures cumulative reward during each expansion.

Protocol:
  - Dev data split 50/50 into prior (full-information) and burn-in (bandit)
  - Holdout for evaluation (750 prompts)
  - 10 seeds per configuration
  - Real PCA embeddings (norm ~1.15), consistent reward normalization
"""

import sys
import ast as ast_module
import copy
import gzip
import json
import time
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "04_figure"))

from bandit_gpt.router import CostAwareLinUCBRouter, CostAwareTabulaRasaRouter
from bandit_gpt.calibration import embed_prompt
from bandit_gpt.config_legacy import (
    DEFAULT_SENTENCE_TRANSFORMER, DEFAULT_PCA_PATH,
    DEV_DATA_PATH_ALL_MODELS, HOLDOUT_DATA_PATH_ALL_MODELS,
)
from bandit_gpt.utils import safe_inv
from sentence_transformers import SentenceTransformer
import joblib

# Import cost data from K-scaling experiment
from run_k_scaling_experiment import MODEL_COSTS_PER_M, compute_cost_per_request

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_SEEDS = 10
SEED_OFFSET = 42
TARGET_PRIOR_NEFF = 2.0  # Tuned via Part 4 n_eff sweep: lower confidence avoids "confident wrong prior"
ALPHA_START = 2.0
ALPHA_END = 0.1
MIN_SIMILARITY = 0.5
RESULTS_DIR = Path(__file__).parent / "results"

PORTFOLIOS = {
    2: [
        "mistralai/mixtral-8x7b-instruct",
        "openai/gpt-4-turbo",
    ],
    5: [
        "meta-llama/llama-3.1-8b-instruct",
        "mistralai/mixtral-8x7b-instruct",
        "google/gemma-3-27b-it",
        "openai/gpt-4o",
        "openai/gpt-5-chat",
    ],
    10: [
        "meta-llama/llama-3.2-1b-instruct",
        "meta-llama/llama-3.1-8b-instruct",
        "mistralai/mixtral-8x7b-instruct",
        "amazon/nova-micro-v1",
        "google/gemma-3-12b-it",
        "anthropic/claude-haiku-4.5",
        "openai/gpt-4o",
        "deepseek/deepseek-chat-v3-0324",
        "anthropic/claude-sonnet-4",
        "openai/gpt-5-chat",
    ],
    20: [
        "meta-llama/llama-3.2-1b-instruct",
        "mistralai/ministral-3b",
        "meta-llama/llama-3.1-8b-instruct",
        "mistralai/ministral-8b",
        "microsoft/phi-4",
        "mistralai/mixtral-8x7b-instruct",
        "amazon/nova-lite-v1",
        "google/gemma-3-4b-it",
        "meta-llama/llama-3.1-70b-instruct",
        "google/gemma-3-12b-it",
        "anthropic/claude-haiku-4.5",
        "google/gemini-2.5-flash-lite",
        "deepseek/deepseek-chat-v3-0324",
        "openai/gpt-4o",
        "anthropic/claude-sonnet-4",
        "cohere/command-a-03-2025",
        "x-ai/grok-3",
        "openai/gpt-5-chat",
        "anthropic/claude-opus-4.5",
        "openai/o3",
    ],
}

EXCLUDED_MODELS = {"openai/gpt-5", "google/gemini-2.5-flash-preview-09-2025"}

# Realistic family-based growth: each new model has a high-similarity neighbor
# already in the portfolio. Simulates natural deployment expansion.
REALISTIC_GROWTH = [
    (2, [
        "openai/gpt-4o",
        "anthropic/claude-sonnet-4",
    ]),
    (5, [
        "openai/gpt-4o",
        "anthropic/claude-sonnet-4",
        "openai/gpt-5-chat",           # neighbor: gpt-4o (same family)
        "openai/o3",                    # neighbor: gpt-4o (same provider)
        "anthropic/claude-haiku-4.5",   # neighbor: claude-sonnet-4 (same family)
    ]),
    (10, [
        "openai/gpt-4o",
        "anthropic/claude-sonnet-4",
        "openai/gpt-5-chat",
        "openai/o3",
        "anthropic/claude-haiku-4.5",
        "anthropic/claude-opus-4.5",          # neighbor: claude-sonnet-4
        "google/gemma-3-12b-it",              # new provider, first entry
        "deepseek/deepseek-chat-v3-0324",     # new provider, first entry
        "mistralai/mixtral-8x7b-instruct",    # new provider, first entry
        "meta-llama/llama-3.1-8b-instruct",   # new provider, first entry
    ]),
    (15, [
        "openai/gpt-4o",
        "anthropic/claude-sonnet-4",
        "openai/gpt-5-chat",
        "openai/o3",
        "anthropic/claude-haiku-4.5",
        "anthropic/claude-opus-4.5",
        "google/gemma-3-12b-it",
        "deepseek/deepseek-chat-v3-0324",
        "mistralai/mixtral-8x7b-instruct",
        "meta-llama/llama-3.1-8b-instruct",
        "google/gemma-3-4b-it",               # neighbor: gemma-3-12b-it (same family)
        "mistralai/ministral-8b",             # neighbor: mixtral-8x7b (same provider)
        "mistralai/ministral-3b",             # neighbor: ministral-8b (same family)
        "meta-llama/llama-3.1-70b-instruct",  # neighbor: llama-3.1-8b (same family)
        "meta-llama/llama-3.2-1b-instruct",   # neighbor: llama-3.1-8b (same family)
    ]),
]

# Adversarial cross-family growth: uses the existing PORTFOLIOS ordering
# where new models may be very different from the base.
ADVERSARIAL_GROWTH = [
    (K, PORTFOLIOS[K]) for K in [2, 5, 10, 20]
]


# =============================================================================
# DATA LOADING (mirrors run_k_scaling_experiment.py)
# =============================================================================

def _judge_weighted_reward(entry: Dict) -> float:
    """Compute continuous reward from judge votes weighted by confidence."""
    jd_raw = entry.get("judge_details", "[]")
    try:
        judges = ast_module.literal_eval(jd_raw) if isinstance(jd_raw, str) else jd_raw
    except (ValueError, SyntaxError):
        judges = []
    if not judges:
        return float(entry.get("raw_score", 0))
    votes = [j["vote"] for j in judges]
    confs = [j.get("confidence", 1.0) for j in judges]
    total_conf = sum(confs)
    return sum(v * c for v, c in zip(votes, confs)) / total_conf if total_conf > 0 else 0.0


def load_all_models_data(filepath: Path, min_models: int = 2,
                         reward_mode: str = "continuous") -> List[Dict]:
    """Load per-prompt, per-model rewards.

    reward_mode:
      "binary"     — use raw_score (0 or 1)
      "continuous"  — use judge-confidence-weighted score (0–1, 170+ distinct values)
    """
    prompt_rewards = defaultdict(dict)
    with gzip.open(filepath, "rt") as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("ok"):
                if reward_mode == "continuous":
                    score = _judge_weighted_reward(entry)
                else:
                    score = float(entry["raw_score"])
                prompt_rewards[entry["prompt"]][entry["model_id"]] = score
    return [
        {"prompt": p, "rewards": r}
        for p, r in prompt_rewards.items()
        if len(r) >= min_models
    ]


def filter_portfolio(data: List[Dict], models: List[str]) -> List[Dict]:
    return [
        {"prompt": item["prompt"], "rewards": {m: item["rewards"][m] for m in models}}
        for item in data
        if all(m in item["rewards"] for m in models)
    ]


def compute_reward_bounds(data: List[Dict], models: List[str]) -> Tuple[float, float]:
    all_r = [item["rewards"][m] for item in data for m in models if m in item["rewards"]]
    return min(all_r), max(all_r)


# =============================================================================
# PRIOR CONSTRUCTION
# =============================================================================

def build_warmup_priors(
    data: List[Dict],
    embeddings: List[np.ndarray],
    models: List[str],
    ridge_lambda: float = 1.0,
    reward_bounds: Optional[Tuple[float, float]] = None,
) -> Dict:
    dim = len(embeddings[0])
    A = {m: ridge_lambda * np.eye(dim) for m in models}
    b = {m: np.zeros(dim) for m in models}

    if reward_bounds is not None:
        r_min, r_max = reward_bounds
        r_range = r_max - r_min if (r_max - r_min) > 1e-6 else 1.0
    else:
        r_min, r_range = 0.0, 1.0

    for i, item in enumerate(data):
        x = embeddings[i]
        xxT = np.outer(x, x)
        for m in models:
            if m in item["rewards"]:
                reward = (item["rewards"][m] - r_min) / r_range
                A[m] += xxT
                b[m] += reward * x

    return {"A": A, "b": b, "models": models, "context_dim": dim,
            "n_prompts": len(data), "reward_bounds": (r_min, r_min + r_range)}


def normalize_prior_strength(priors: Dict, target: float = 10.0) -> Dict:
    dim = priors["context_dim"]
    new = copy.deepcopy(priors)
    for m in priors["A"]:
        mass = np.trace(priors["A"][m]) / dim
        scale = target / max(mass, 1e-6)
        new["A"][m] = priors["A"][m] * scale
        new["b"][m] = priors["b"][m] * scale
    return new


def extract_theta(priors: Dict, model: str) -> np.ndarray:
    dim = priors["context_dim"]
    A_inv = safe_inv(priors["A"][model])
    return A_inv @ priors["b"][model]


# =============================================================================
# DNA EMBEDDING & SEMANTIC SIMILARITY (mirrors router.py)
# =============================================================================

def get_model_dna(model_id: str) -> str:
    return model_id.replace("-", " ").replace("/", " ").replace("_", " ").lower()


def compute_dna_embeddings(models: List[str], encoder: SentenceTransformer) -> Dict[str, np.ndarray]:
    dna_strings = [get_model_dna(m) for m in models]
    vecs = encoder.encode(dna_strings, convert_to_numpy=True)
    return {m: vecs[i] for i, m in enumerate(models)}


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def find_nearest_neighbor(
    model: str,
    candidates: List[str],
    dna_embeddings: Dict[str, np.ndarray],
) -> Tuple[Optional[str], float]:
    best, best_sim = None, -1.0
    for c in candidates:
        if c == model:
            continue
        sim = cosine_similarity(dna_embeddings[model], dna_embeddings[c])
        if sim > best_sim:
            best_sim = sim
            best = c
    return best, best_sim


def semantic_transfer_prior(
    new_model: str,
    neighbor: str,
    neighbor_priors: Dict,
    n_eff: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """First-Child Bias Correction: transfer theta, reset A."""
    theta_neighbor = extract_theta(neighbor_priors, neighbor)
    dim = neighbor_priors["context_dim"]
    A_new = n_eff * np.eye(dim)
    b_new = n_eff * theta_neighbor
    return A_new, b_new


def classify_transfer(model: str, neighbor: str) -> str:
    m_provider = model.split("/")[0] if "/" in model else model
    n_provider = neighbor.split("/")[0] if "/" in neighbor else neighbor
    m_family = model.split("/")[1].split("-")[0] if "/" in model else model.split("-")[0]
    n_family = neighbor.split("/")[1].split("-")[0] if "/" in neighbor else neighbor.split("-")[0]
    if m_provider == n_provider and m_family == n_family:
        return "Same Family"
    elif m_provider == n_provider:
        return "Same Provider"
    else:
        return "Cross-Provider"


def find_provider_neighbor(
    model: str,
    candidates: List[str],
    dna_embeddings: Dict[str, np.ndarray],
) -> Tuple[Optional[str], float]:
    """Find the best same-provider candidate; fall back to random if none."""
    provider = model.split("/")[0] if "/" in model else model
    same_provider = [c for c in candidates if c != model and c.split("/")[0] == provider]
    if same_provider:
        return find_nearest_neighbor(model, same_provider, dna_embeddings)
    return None, 0.0


def find_random_neighbor(
    model: str,
    candidates: List[str],
    rng: np.random.RandomState,
) -> str:
    """Pick a uniformly random candidate (not the model itself)."""
    pool = [c for c in candidates if c != model]
    return pool[rng.randint(len(pool))]


# =============================================================================
# PART 1: TRANSFER QUALITY ANALYSIS (OFFLINE)
# =============================================================================

def run_part1_transfer_quality(
    models: List[str],
    prior_data: List[Dict],
    prior_emb: List[np.ndarray],
    dna_embeddings: Dict[str, np.ndarray],
    reward_bounds: Tuple[float, float],
) -> List[Dict]:
    """Leave-one-out transfer quality analysis over K=20."""
    logger.info("\n" + "=" * 70)
    logger.info("PART 1: TRANSFER QUALITY ANALYSIS (OFFLINE)")
    logger.info("=" * 70)

    ground_truth_priors = build_warmup_priors(
        prior_data, prior_emb, models, reward_bounds=reward_bounds,
    )

    results = []
    logger.info(f"\n  {'Model':<40} {'Neighbor':<40} {'Sim':>6} {'cos(θ)':>8} {'RelErr':>8} {'Type':<16}")
    logger.info("  " + "-" * 120)

    for model in models:
        others = [m for m in models if m != model]
        neighbor, sim = find_nearest_neighbor(model, others, dna_embeddings)

        if neighbor is None or sim < MIN_SIMILARITY:
            results.append({
                "model": model, "neighbor": "N/A", "similarity": 0.0,
                "theta_cosine": 0.0, "relative_error": float("inf"),
                "transfer_type": "None", "theta_gt_norm": 0.0,
                "theta_transfer_norm": 0.0,
            })
            continue

        # Ground truth theta for model (learned from data)
        theta_gt = extract_theta(ground_truth_priors, model)
        # Transferred theta (from neighbor)
        theta_neighbor = extract_theta(ground_truth_priors, neighbor)

        gt_norm = np.linalg.norm(theta_gt)
        tr_norm = np.linalg.norm(theta_neighbor)
        theta_cos = cosine_similarity(theta_gt, theta_neighbor)
        rel_err = np.linalg.norm(theta_gt - theta_neighbor) / max(gt_norm, 1e-12)
        transfer_type = classify_transfer(model, neighbor)

        results.append({
            "model": model, "neighbor": neighbor, "similarity": sim,
            "theta_cosine": theta_cos, "relative_error": rel_err,
            "transfer_type": transfer_type, "theta_gt_norm": float(gt_norm),
            "theta_transfer_norm": float(tr_norm),
        })
        logger.info(
            f"  {model:<40} {neighbor:<40} {sim:>6.3f} {theta_cos:>8.3f} "
            f"{rel_err:>8.3f} {transfer_type:<16}"
        )

    # Summary statistics
    sims = [r["similarity"] for r in results if r["similarity"] > 0]
    cosines = [r["theta_cosine"] for r in results if r["similarity"] > 0]
    errs = [r["relative_error"] for r in results if r["relative_error"] < float("inf")]

    logger.info(f"\n  Summary over {len(sims)} successful transfers:")
    logger.info(f"    DNA embedding similarity: mean={np.mean(sims):.3f}, range=[{min(sims):.3f}, {max(sims):.3f}]")
    logger.info(f"    Theta cosine similarity:  mean={np.mean(cosines):.3f}, range=[{min(cosines):.3f}, {max(cosines):.3f}]")
    logger.info(f"    Relative error:           mean={np.mean(errs):.3f}, range=[{min(errs):.3f}, {max(errs):.3f}]")

    # By transfer type
    for ttype in ["Same Family", "Same Provider", "Cross-Provider"]:
        subset = [r for r in results if r["transfer_type"] == ttype]
        if subset:
            avg_sim = np.mean([r["similarity"] for r in subset])
            avg_cos = np.mean([r["theta_cosine"] for r in subset])
            avg_err = np.mean([r["relative_error"] for r in subset])
            logger.info(f"    {ttype:<16}: n={len(subset)}, sim={avg_sim:.3f}, cos(θ)={avg_cos:.3f}, err={avg_err:.3f}")

    return results


# =============================================================================
# PART 2: ONLINE ONBOARDING (BANDIT SIMULATION)
# =============================================================================

def run_single_onboarding_trial(
    new_model: str,
    all_models: List[str],
    train_data: List[Dict],
    eval_data: List[Dict],
    train_emb: List[np.ndarray],
    eval_emb: List[np.ndarray],
    full_priors: Dict,
    transferred_A: np.ndarray,
    transferred_b: np.ndarray,
    init_mode: str,
    n_steps: int,
    reward_bounds: Tuple[float, float],
    seed: int,
    model_costs_norm: Optional[Dict] = None,
    model_costs_raw: Optional[Dict] = None,
    cost_penalty: float = 0.0,
) -> Dict:
    """Run a single bandit trial with a specific init mode for the new model."""
    rng = np.random.RandomState(seed)
    dim = full_priors["context_dim"]
    r_min, r_max = reward_bounds
    r_range = r_max - r_min if (r_max - r_min) > 1e-6 else 1.0

    priors = copy.deepcopy(full_priors)
    priors = normalize_prior_strength(priors, TARGET_PRIOR_NEFF)

    if init_mode == "oracle":
        pass
    elif init_mode == "transfer":
        priors["A"][new_model] = transferred_A.copy()
        priors["b"][new_model] = transferred_b.copy()
    elif init_mode == "transfer_b_only":
        priors["A"][new_model] = np.eye(dim)
        priors["b"][new_model] = transferred_b.copy() / max(transferred_A[0, 0], 1e-6)
    elif init_mode == "true_oracle":
        pass
    elif init_mode == "tabula_rasa":
        priors["A"][new_model] = np.eye(dim)
        priors["b"][new_model] = np.zeros(dim)

    if model_costs_norm is None:
        model_costs_norm = {m: {"normalized_cost": 0.0} for m in all_models}
    if model_costs_raw is None:
        model_costs_raw = {m: 0.0 for m in all_models}

    router = CostAwareLinUCBRouter(
        models=all_models, warmup_priors=priors, model_costs=model_costs_norm,
        alpha_start=ALPHA_START, alpha_end=ALPHA_END, cost_penalty=cost_penalty,
    )

    actual_steps = min(n_steps, len(train_data))
    cumulative_reward = 0.0
    cumulative_cost = 0.0
    step_rewards = []
    selections = []
    new_model_selected_burnin = 0

    # Shuffle burn-in order per seed for statistical independence
    indices = rng.permutation(len(train_data))[:actual_steps]

    for idx in indices:
        x = train_emb[idx]
        if init_mode == "true_oracle":
            best_r = -1.0
            sel = all_models[0]
            for m in all_models:
                mr = train_data[idx]["rewards"].get(m, 0.0)
                if mr > best_r:
                    best_r = mr
                    sel = m
        else:
            sel = router.select_model(x, total_steps=actual_steps)
        raw_r = train_data[idx]["rewards"].get(sel, 0.0)
        norm_r = (raw_r - r_min) / r_range
        router.update(x, sel, norm_r)
        cumulative_reward += raw_r
        cumulative_cost += model_costs_raw.get(sel, 0.0)
        step_rewards.append(raw_r)
        selections.append(sel)
        if sel == new_model:
            new_model_selected_burnin += 1

    eval_reward = 0.0
    eval_cost = 0.0
    new_model_selected_eval = 0
    per_prompt_regret = 0.0

    for i, p in enumerate(eval_data):
        x = eval_emb[i]
        sel = router.select_model(x, total_steps=actual_steps)
        sel_r = p["rewards"].get(sel, 0.0)
        eval_reward += sel_r
        eval_cost += model_costs_raw.get(sel, 0.0)
        if sel == new_model:
            new_model_selected_eval += 1
        best_r = max(p["rewards"].get(m, 0.0) for m in all_models)
        per_prompt_regret += (best_r - sel_r)

    n_eval = len(eval_data)
    return {
        "eval_reward": eval_reward / n_eval,
        "eval_cost": eval_cost / n_eval,
        "eval_regret": per_prompt_regret / n_eval,
        "cumulative_train_reward": cumulative_reward / max(actual_steps, 1),
        "cumulative_train_cost": cumulative_cost / max(actual_steps, 1),
        "new_model_selection_rate_burnin": new_model_selected_burnin / max(actual_steps, 1),
        "new_model_selection_rate_eval": new_model_selected_eval / n_eval,
        "step_rewards": step_rewards,
        "selections": selections,
    }


COST_PENALTY = 0.5  # moderate: makes cost matter but doesn't dominate quality

def run_part2_online_onboarding(
    models: List[str],
    prior_data: List[Dict],
    burnin_data: List[Dict],
    eval_data: List[Dict],
    prior_emb: List[np.ndarray],
    burnin_emb: List[np.ndarray],
    eval_emb: List[np.ndarray],
    dna_embeddings: Dict[str, np.ndarray],
    reward_bounds: Tuple[float, float],
    model_costs_norm: Optional[Dict] = None,
    model_costs_raw: Optional[Dict] = None,
    k_label: str = "",
) -> Dict:
    """Leave-one-out onboarding experiment with transfer ablation."""
    K = len(models)
    logger.info("\n" + "=" * 70)
    logger.info(f"PART 2: ONLINE ONBOARDING PERFORMANCE (K={K}) {k_label}")
    logger.info("=" * 70)
    if model_costs_norm:
        logger.info(f"  Cost penalty = {COST_PENALTY}")

    MODES = [
        "true_oracle",
        "oracle",
        "transfer_dna",
        "transfer_dna_b_only",
        "transfer_random",
        "tabula_rasa",
    ]
    MODE_LABELS = {
        "true_oracle": "True Oracle",
        "oracle": "Prior Oracle",
        "transfer_dna": "DNA Transfer",
        "transfer_dna_b_only": "DNA Direction Only",
        "transfer_random": "Random Transfer",
        "tabula_rasa": "Tabula Rasa",
    }
    horizons = [10, 20, 50, 100, 200, 500]
    all_results = {}

    full_priors = build_warmup_priors(
        prior_data, prior_emb, models, reward_bounds=reward_bounds,
    )
    dim = full_priors["context_dim"]
    identity_A = np.eye(dim)
    zero_b = np.zeros(dim)

    for mi, new_model in enumerate(models):
        logger.info(f"\n  [{mi+1}/{len(models)}] New model: {new_model}")
        others = [m for m in models if m != new_model]

        dna_neighbor, dna_sim = find_nearest_neighbor(new_model, others, dna_embeddings)
        dna_type = classify_transfer(new_model, dna_neighbor) if dna_neighbor else "None"
        logger.info(f"    DNA neighbor: {dna_neighbor} (sim={dna_sim:.3f}, {dna_type})")

        A_dna, b_dna = semantic_transfer_prior(
            new_model, dna_neighbor, full_priors, n_eff=TARGET_PRIOR_NEFF,
        ) if dna_neighbor and dna_sim >= MIN_SIMILARITY else (identity_A.copy(), zero_b.copy())

        model_results = {
            "model": new_model,
            "dna_neighbor": dna_neighbor, "dna_similarity": dna_sim,
            "dna_transfer_type": dna_type,
            "horizons": {},
        }

        for N in horizons:
            horizon_results = {}
            for mode in MODES:
                trial_rewards = []
                trial_costs = []
                trial_train_rewards = []
                trial_regrets = []
                trial_sel_burnin = []
                trial_sel_eval = []
                for seed in range(N_SEEDS):
                    if mode == "transfer_random":
                        rng_t = np.random.RandomState(SEED_OFFSET + seed + hash(new_model) % 10000)
                        rand_neighbor = find_random_neighbor(new_model, others, rng_t)
                        A_rand, b_rand = semantic_transfer_prior(
                            new_model, rand_neighbor, full_priors, n_eff=TARGET_PRIOR_NEFF,
                        )
                    else:
                        A_rand, b_rand = identity_A, zero_b

                    if mode in ("oracle", "true_oracle", "tabula_rasa"):
                        A_trial, b_trial = identity_A, zero_b
                    elif mode == "transfer_dna":
                        A_trial, b_trial = A_dna, b_dna
                    elif mode == "transfer_dna_b_only":
                        A_trial, b_trial = A_dna, b_dna
                    elif mode == "transfer_random":
                        A_trial, b_trial = A_rand, b_rand

                    if mode == "transfer_dna_b_only":
                        init_mode_key = "transfer_b_only"
                    elif mode == "true_oracle":
                        init_mode_key = "true_oracle"
                    elif mode.startswith("transfer"):
                        init_mode_key = "transfer"
                    else:
                        init_mode_key = mode

                    res = run_single_onboarding_trial(
                        new_model, models,
                        burnin_data, eval_data, burnin_emb, eval_emb,
                        full_priors, A_trial, b_trial,
                        init_mode=init_mode_key, n_steps=N,
                        reward_bounds=reward_bounds, seed=SEED_OFFSET + seed,
                        model_costs_norm=model_costs_norm,
                        model_costs_raw=model_costs_raw,
                        cost_penalty=COST_PENALTY if model_costs_norm else 0.0,
                    )
                    trial_rewards.append(res["eval_reward"])
                    trial_costs.append(res["eval_cost"])
                    trial_train_rewards.append(res["cumulative_train_reward"])
                    trial_regrets.append(res["eval_regret"])
                    trial_sel_burnin.append(res["new_model_selection_rate_burnin"])
                    trial_sel_eval.append(res["new_model_selection_rate_eval"])

                horizon_results[mode] = {
                    "mean": float(np.mean(trial_rewards)),
                    "std": float(np.std(trial_rewards, ddof=1)) if N_SEEDS > 1 else 0.0,
                    "ci95": float(1.96 * np.std(trial_rewards, ddof=1) / np.sqrt(N_SEEDS)) if N_SEEDS > 1 else 0.0,
                    "cost_mean": float(np.mean(trial_costs)),
                    "regret_mean": float(np.mean(trial_regrets)),
                    "train_reward_mean": float(np.mean(trial_train_rewards)),
                    "sel_rate_burnin": float(np.mean(trial_sel_burnin)),
                    "sel_rate_eval": float(np.mean(trial_sel_eval)),
                    "raw_evals": trial_rewards,
                }
            model_results["horizons"][N] = horizon_results

            true_oracle_r = horizon_results["true_oracle"]["mean"]
            tabula_r = horizon_results["tabula_rasa"]["mean"]
            gap = true_oracle_r - tabula_r
            parts = [f"N={N:>3}:"]
            for mode in MODES:
                r = horizon_results[mode]["mean"]
                c = horizon_results[mode]["cost_mean"]
                sel = horizon_results[mode]["sel_rate_eval"]
                closed = (r - tabula_r) / gap * 100 if abs(gap) > 1e-6 else 0.0
                if mode.startswith("transfer"):
                    parts.append(f"{MODE_LABELS[mode]}={r:.4f}({closed:+.0f}%,sel={sel:.2f})")
                else:
                    parts.append(f"{MODE_LABELS[mode]}={r:.4f}(${c*1e3:.2f}m)")
            logger.info("    " + "  ".join(parts))

        all_results[new_model] = model_results

    return all_results


# =============================================================================
# PART 3: PORTFOLIO GROWTH SIMULATION
# =============================================================================

def run_part3_portfolio_growth(
    burnin_data_raw: List[Dict],
    eval_data_raw: List[Dict],
    prior_data_raw: List[Dict],
    burnin_emb_all: List[np.ndarray],
    eval_emb_all: List[np.ndarray],
    prior_emb_all: List[np.ndarray],
    dna_embeddings: Dict[str, np.ndarray],
    burnin_prompt_idx: Dict[str, int],
    eval_prompt_idx: Dict[str, int],
    prior_prompt_idx: Dict[str, int],
    growth_path: Optional[List[Tuple]] = None,
    scenario_name: str = "Default",
) -> Dict:
    """Simulate portfolio growth along a specified growth path."""
    logger.info(f"\n  --- Scenario: {scenario_name} ---")

    phases = growth_path if growth_path is not None else [
        (K, PORTFOLIOS[K]) for K in [2, 5, 10, 20]
    ]
    STEPS_PER_PHASE = 200
    growth_results = {"transfer": {}, "tabula_rasa": {}, "oracle": {}}

    for seed in range(N_SEEDS):
        np.random.seed(SEED_OFFSET + seed)
        dim = 33  # PCA 32 + bias

        # Each init strategy gets its own running state
        states = {}
        for mode in ["transfer", "tabula_rasa", "oracle"]:
            states[mode] = {
                "A": {}, "b": {},
                "phase_rewards": [],
                "phase_eval_rewards": [],
            }

        known_models = set()

        for phase_idx, (K, models_k) in enumerate(phases):
            new_models = [m for m in models_k if m not in known_models]
            logger.info(f"\n  Phase {phase_idx}: K={K}, adding {len(new_models)} new models"
                        f" (seed {seed})" if seed == 0 else "")

            # Filter data for this portfolio
            burnin_k = filter_portfolio(burnin_data_raw, models_k)
            eval_k = filter_portfolio(eval_data_raw, models_k)
            prior_k = filter_portfolio(prior_data_raw, models_k)

            if len(burnin_k) < 10 or len(eval_k) < 10:
                logger.warning(f"    Insufficient data for K={K}, skipping")
                continue

            # Get embeddings aligned to filtered data
            burnin_emb_k = [burnin_emb_all[burnin_prompt_idx[p["prompt"]]] for p in burnin_k]
            eval_emb_k = [eval_emb_all[eval_prompt_idx[p["prompt"]]] for p in eval_k]
            prior_emb_k = [prior_emb_all[prior_prompt_idx[p["prompt"]]] for p in prior_k]

            # Build ground truth priors for this portfolio
            r_bounds = compute_reward_bounds(prior_k, models_k)
            gt_priors = build_warmup_priors(prior_k, prior_emb_k, models_k, reward_bounds=r_bounds)
            gt_priors_norm = normalize_prior_strength(gt_priors, TARGET_PRIOR_NEFF)
            r_min, r_max = r_bounds
            r_range = r_max - r_min if (r_max - r_min) > 1e-6 else 1.0

            for mode in ["transfer", "tabula_rasa", "oracle"]:
                s = states[mode]
                # Initialize new models
                for m in new_models:
                    if mode == "oracle":
                        s["A"][m] = gt_priors_norm["A"][m].copy()
                        s["b"][m] = gt_priors_norm["b"][m].copy()
                    elif mode == "transfer" and known_models:
                        neighbor, sim = find_nearest_neighbor(m, list(known_models), dna_embeddings)
                        if neighbor and sim >= MIN_SIMILARITY and neighbor in s["A"]:
                            theta_n = safe_inv(s["A"][neighbor]) @ s["b"][neighbor]
                            s["A"][m] = TARGET_PRIOR_NEFF * np.eye(dim)
                            s["b"][m] = TARGET_PRIOR_NEFF * theta_n
                        else:
                            s["A"][m] = np.eye(dim)
                            s["b"][m] = np.zeros(dim)
                    else:
                        s["A"][m] = np.eye(dim)
                        s["b"][m] = np.zeros(dim)

                # Build priors dict for router
                priors_dict = {
                    "A": {m: s["A"][m].copy() for m in models_k},
                    "b": {m: s["b"][m].copy() for m in models_k},
                    "context_dim": dim,
                }
                model_costs = {m: {"normalized_cost": 0.0} for m in models_k}

                router = CostAwareLinUCBRouter(
                    models=models_k, warmup_priors=priors_dict,
                    model_costs=model_costs, alpha_start=ALPHA_START,
                    alpha_end=ALPHA_END, cost_penalty=0.0,
                )

                # Run burn-in steps
                actual_steps = min(STEPS_PER_PHASE, len(burnin_k))
                phase_reward = 0.0
                indices = np.random.permutation(len(burnin_k))[:actual_steps]
                for idx in indices:
                    x = burnin_emb_k[idx]
                    sel = router.select_model(x, total_steps=actual_steps)
                    raw_r = burnin_k[idx]["rewards"].get(sel, 0.0)
                    norm_r = (raw_r - r_min) / r_range
                    router.update(x, sel, norm_r)
                    phase_reward += raw_r

                # Evaluate
                eval_reward = 0.0
                for i, p in enumerate(eval_k):
                    x = eval_emb_k[i]
                    sel = router.select_model(x, total_steps=actual_steps)
                    eval_reward += p["rewards"].get(sel, 0.0)
                eval_reward /= len(eval_k)

                s["phase_rewards"].append(phase_reward / actual_steps)
                s["phase_eval_rewards"].append(eval_reward)

                # Save updated state from router back
                for m in models_k:
                    if hasattr(router, 'policy'):
                        s["A"][m] = router.policy.A[m].copy()
                        s["b"][m] = router.policy.b[m].copy()
                    else:
                        s["A"][m] = router.A[m].copy()
                        s["b"][m] = router.b[m].copy()

            known_models.update(new_models)

        # Store per-seed results
        for mode in ["transfer", "tabula_rasa", "oracle"]:
            if seed == 0:
                growth_results[mode] = {
                    "phase_rewards": [],
                    "phase_eval_rewards": [],
                }
            growth_results[mode]["phase_rewards"].append(
                states[mode]["phase_rewards"][:])
            growth_results[mode]["phase_eval_rewards"].append(
                states[mode]["phase_eval_rewards"][:])

    # Print summary
    logger.info(f"\n  {scenario_name} Growth Summary (eval reward, {N_SEEDS} seeds):")
    logger.info(f"  {'Phase':<20} {'Oracle':>16} {'Transfer':>16} {'Tabula Rasa':>16} {'Gap Closed':>14}")
    logger.info("  " + "-" * 85)
    for pi, (K, _) in enumerate(phases):
        vals = {}
        for mode in ["oracle", "transfer", "tabula_rasa"]:
            evals = [growth_results[mode]["phase_eval_rewards"][s][pi]
                     for s in range(N_SEEDS)
                     if pi < len(growth_results[mode]["phase_eval_rewards"][s])]
            vals[mode] = (np.mean(evals), np.std(evals))

        gap = vals["oracle"][0] - vals["tabula_rasa"][0]
        closed = (vals["transfer"][0] - vals["tabula_rasa"][0]) / gap * 100 if abs(gap) > 1e-6 else 0.0
        logger.info(
            f"  K={K:<2} ({len(phases[pi][1]):2d} models)  "
            f"{vals['oracle'][0]:.4f}±{vals['oracle'][1]:.3f}  "
            f"{vals['transfer'][0]:.4f}±{vals['transfer'][1]:.3f}  "
            f"{vals['tabula_rasa'][0]:.4f}±{vals['tabula_rasa'][1]:.3f}  "
            f"{closed:>12.1f}%"
        )

    growth_results["_phases"] = phases
    growth_results["_scenario_name"] = scenario_name
    return growth_results


# =============================================================================
# FIGURE GENERATION
# =============================================================================

def plot_part1_scatter(results: List[Dict], output_dir: Path):
    """Scatter: DNA similarity vs theta cosine similarity."""
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = {"Same Family": "#2ecc71", "Same Provider": "#3498db", "Cross-Provider": "#e74c3c", "None": "#95a5a6"}
    markers = {"Same Family": "o", "Same Provider": "s", "Cross-Provider": "^", "None": "x"}

    for r in results:
        if r["similarity"] < MIN_SIMILARITY:
            continue
        ax.scatter(r["similarity"], r["theta_cosine"],
                   c=colors.get(r["transfer_type"], "#95a5a6"),
                   marker=markers.get(r["transfer_type"], "o"),
                   s=80, edgecolors="white", linewidths=0.5, zorder=3,
                   label=r["transfer_type"])

    # De-duplicate legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="lower right", fontsize=10)

    ax.set_xlabel("DNA Embedding Similarity (cosine)", fontsize=12)
    ax.set_ylabel("Theta Cosine Similarity (transferred vs ground truth)", fontsize=12)
    ax.set_title("Semantic Transfer Quality: Does Embedding Similarity Predict Prior Quality?", fontsize=13)
    ax.axhline(y=0.0, color="#bdc3c7", ls="--", lw=0.8)
    ax.set_xlim(0.3, 1.05)
    ax.set_ylim(-0.2, 1.05)
    ax.grid(True, alpha=0.3)

    # Annotate each point with short model name
    for r in results:
        if r["similarity"] < MIN_SIMILARITY:
            continue
        short = r["model"].split("/")[-1][:12]
        ax.annotate(short, (r["similarity"], r["theta_cosine"]),
                    fontsize=6, alpha=0.7, xytext=(3, 3), textcoords="offset points")

    fig.tight_layout()
    fig.savefig(output_dir / "part1_transfer_quality_scatter.png", dpi=200)
    fig.savefig(output_dir / "part1_transfer_quality_scatter.pdf")
    plt.close(fig)
    logger.info(f"  Saved Part 1 scatter: {output_dir / 'part1_transfer_quality_scatter.png'}")


def plot_part2_onboarding(results: Dict, output_dir: Path):
    """Three-panel figure: reward lines, cost lines, cost-quality scatter."""
    MODES = ["true_oracle", "oracle", "transfer_dna", "transfer_dna_b_only", "transfer_random", "tabula_rasa"]
    MODE_LABELS = {
        "true_oracle": "True Oracle",
        "oracle": "Prior Oracle", "transfer_dna": "DNA Transfer",
        "transfer_dna_b_only": "DNA Direction Only",
        "transfer_random": "Random Transfer",
        "tabula_rasa": "Tabula Rasa",
    }
    COLORS = {
        "true_oracle": "#1abc9c",
        "oracle": "#2ecc71", "transfer_dna": "#3498db",
        "transfer_dna_b_only": "#8e44ad",
        "transfer_random": "#f39c12",
        "tabula_rasa": "#e74c3c",
    }
    MARKERS = {"true_oracle": "D", "oracle": "s", "transfer_dna": "o",
               "transfer_dna_b_only": "P", "transfer_random": "v", "tabula_rasa": "x"}

    # Detect available horizons from data
    sample_model = next(iter(results))
    horizons = sorted(results[sample_model]["horizons"].keys())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # --- Panel 1: Holdout Reward vs Horizon (line plot) ---
    ax = axes[0]
    for mode in MODES:
        means, ci95s = [], []
        for N in horizons:
            vals = [results[m]["horizons"][N][mode]["mean"] for m in results
                    if mode in results[m]["horizons"][N]]
            means.append(np.mean(vals) if vals else 0)
            ci95s.append(1.96 * np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0)
        means, ci95s = np.array(means), np.array(ci95s)
        ax.plot(horizons, means, f"{MARKERS[mode]}-", color=COLORS[mode],
                label=MODE_LABELS[mode], lw=2, markersize=6)
        ax.fill_between(horizons, means - ci95s, means + ci95s,
                        color=COLORS[mode], alpha=0.12)

    ax.set_xlabel("Burn-in Horizon (N steps)", fontsize=11)
    ax.set_ylabel("Holdout Reward", fontsize=11)
    ax.set_title("Quality: Eval Reward vs Horizon", fontsize=12)
    ax.set_xscale("log")
    ax.set_xticks(horizons)
    ax.set_xticklabels([str(h) for h in horizons])
    ax.legend(fontsize=7.5, ncol=1)
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Holdout Cost vs Horizon ---
    ax = axes[1]
    has_cost = any(results[m]["horizons"][horizons[0]].get("oracle", {}).get("cost_mean", 0) > 0
                   for m in results)
    if has_cost:
        for mode in MODES:
            means = []
            for N in horizons:
                vals = [results[m]["horizons"][N][mode]["cost_mean"] for m in results
                        if mode in results[m]["horizons"][N]]
                means.append(np.mean(vals) * 1000 if vals else 0)  # convert to millicents
            ax.plot(horizons, means, f"{MARKERS[mode]}-", color=COLORS[mode],
                    label=MODE_LABELS[mode], lw=2, markersize=6)
        ax.set_xlabel("Burn-in Horizon (N steps)", fontsize=11)
        ax.set_ylabel("Avg Cost per Request ($×10³)", fontsize=11)
        ax.set_title("Cost: Avg Cost vs Horizon", fontsize=12)
        ax.set_xscale("log")
        ax.set_xticks(horizons)
        ax.set_xticklabels([str(h) for h in horizons])
        ax.legend(fontsize=7.5, ncol=1)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No cost data\n(cost_penalty=0)", transform=ax.transAxes,
                ha="center", va="center", fontsize=14, color="#bdc3c7")
        ax.set_title("Cost: Avg Cost vs Horizon", fontsize=12)

    # --- Panel 3: Cost-Quality Pareto at N=50 (scatter) ---
    ax = axes[2]
    ref_N = 50 if 50 in horizons else horizons[len(horizons) // 2]
    for mode in MODES:
        rewards_all, costs_all = [], []
        for m in results:
            h = results[m]["horizons"][ref_N]
            if mode in h:
                rewards_all.append(h[mode]["mean"])
                costs_all.append(h[mode].get("cost_mean", 0) * 1000)
        if rewards_all:
            ax.scatter(np.mean(costs_all), np.mean(rewards_all),
                       color=COLORS[mode], marker=MARKERS[mode], s=120,
                       label=MODE_LABELS[mode], edgecolors="white", linewidths=0.5,
                       zorder=3)

    ax.set_xlabel("Avg Cost per Request ($×10³)", fontsize=11)
    ax.set_ylabel("Holdout Reward", fontsize=11)
    ax.set_title(f"Cost-Quality Tradeoff (N={ref_N})", fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "part2_onboarding_performance.png", dpi=200)
    fig.savefig(output_dir / "part2_onboarding_performance.pdf")
    plt.close(fig)
    logger.info(f"  Saved Part 2 figures: {output_dir / 'part2_onboarding_performance.png'}")


# =============================================================================
# PART 4: n_effective SENSITIVITY SWEEP
# =============================================================================

NEFF_VALUES = [1, 2, 5, 10, 20, 50]

def run_part4_neff_sweep(
    models_k10: List[str],
    prior_data: List[Dict],
    burnin_data: List[Dict],
    eval_data: List[Dict],
    prior_emb: List[np.ndarray],
    burnin_emb: List[np.ndarray],
    eval_emb: List[np.ndarray],
    dna_embeddings: Dict[str, np.ndarray],
    reward_bounds: Tuple[float, float],
    model_costs_norm: Optional[Dict] = None,
    model_costs_raw: Optional[Dict] = None,
) -> Dict:
    """Sweep n_effective for DNA transfer at multiple horizons."""
    logger.info("\n" + "=" * 70)
    logger.info("PART 4: n_effective SENSITIVITY SWEEP")
    logger.info("=" * 70)

    horizons = [10, 50, 200]
    full_priors = build_warmup_priors(
        prior_data, prior_emb, models_k10, reward_bounds=reward_bounds,
    )
    dim = full_priors["context_dim"]
    results = {}

    for N in horizons:
        results[N] = {}

        # Oracle and Tabula Rasa baselines (run once per horizon)
        for baseline in ["oracle", "tabula_rasa"]:
            all_evals = []
            for mi, new_model in enumerate(models_k10):
                for seed in range(N_SEEDS):
                    res = run_single_onboarding_trial(
                        new_model, models_k10,
                        burnin_data, eval_data, burnin_emb, eval_emb,
                        full_priors, np.eye(dim), np.zeros(dim),
                        init_mode=baseline, n_steps=N,
                        reward_bounds=reward_bounds, seed=SEED_OFFSET + seed,
                        model_costs_norm=model_costs_norm,
                        model_costs_raw=model_costs_raw,
                        cost_penalty=COST_PENALTY if model_costs_norm else 0.0,
                    )
                    all_evals.append(res["eval_reward"])
            results[N][baseline] = {
                "mean": float(np.mean(all_evals)),
                "std": float(np.std(all_evals, ddof=1)),
                "ci95": float(1.96 * np.std(all_evals, ddof=1) / np.sqrt(len(all_evals))),
            }

        # DNA transfer at each n_eff
        for n_eff in NEFF_VALUES:
            all_evals = []
            for mi, new_model in enumerate(models_k10):
                others = [m for m in models_k10 if m != new_model]
                neighbor, sim = find_nearest_neighbor(new_model, others, dna_embeddings)
                if neighbor and sim >= MIN_SIMILARITY:
                    A_t, b_t = semantic_transfer_prior(
                        new_model, neighbor, full_priors, n_eff=n_eff,
                    )
                else:
                    A_t, b_t = np.eye(dim), np.zeros(dim)

                for seed in range(N_SEEDS):
                    res = run_single_onboarding_trial(
                        new_model, models_k10,
                        burnin_data, eval_data, burnin_emb, eval_emb,
                        full_priors, A_t, b_t,
                        init_mode="transfer", n_steps=N,
                        reward_bounds=reward_bounds, seed=SEED_OFFSET + seed,
                        model_costs_norm=model_costs_norm,
                        model_costs_raw=model_costs_raw,
                        cost_penalty=COST_PENALTY if model_costs_norm else 0.0,
                    )
                    all_evals.append(res["eval_reward"])
            results[N][f"neff_{n_eff}"] = {
                "mean": float(np.mean(all_evals)),
                "std": float(np.std(all_evals, ddof=1)),
                "ci95": float(1.96 * np.std(all_evals, ddof=1) / np.sqrt(len(all_evals))),
                "n_eff": n_eff,
            }

        # Log summary for this horizon
        oracle_r = results[N]["oracle"]["mean"]
        tabula_r = results[N]["tabula_rasa"]["mean"]
        gap = oracle_r - tabula_r
        logger.info(f"\n  N={N}: Oracle={oracle_r:.4f}, Tabula={tabula_r:.4f}, gap={gap:.4f}")
        for n_eff in NEFF_VALUES:
            r = results[N][f"neff_{n_eff}"]["mean"]
            gc = (r - tabula_r) / gap * 100 if abs(gap) > 1e-6 else 0.0
            logger.info(f"    n_eff={n_eff:<3}: R={r:.4f}  gap_closed={gc:+.1f}%")

    return results


def plot_part4_neff(results: Dict, output_dir: Path):
    """Line plot: reward vs n_effective at each horizon."""
    horizons = sorted(results.keys())
    fig, ax = plt.subplots(figsize=(9, 5.5))

    colors_h = {10: "#e74c3c", 50: "#3498db", 200: "#2ecc71"}
    for N in horizons:
        color = colors_h.get(N, "#7f8c8d")
        means = [results[N][f"neff_{ne}"]["mean"] for ne in NEFF_VALUES]
        ci95s = [results[N][f"neff_{ne}"]["ci95"] for ne in NEFF_VALUES]
        means, ci95s = np.array(means), np.array(ci95s)
        ax.plot(NEFF_VALUES, means, "o-", color=color, lw=2, markersize=7,
                label=f"DNA Transfer (N={N})")
        ax.fill_between(NEFF_VALUES, means - ci95s, means + ci95s,
                        color=color, alpha=0.12)

        # Reference lines for this horizon
        oracle_r = results[N]["oracle"]["mean"]
        tabula_r = results[N]["tabula_rasa"]["mean"]
        ax.axhline(oracle_r, color=color, ls="--", lw=1, alpha=0.5)
        ax.axhline(tabula_r, color=color, ls=":", lw=1, alpha=0.5)

    # Legend annotations
    ax.plot([], [], "k--", lw=1, alpha=0.5, label="Oracle (per horizon)")
    ax.plot([], [], "k:", lw=1, alpha=0.5, label="Tabula Rasa (per horizon)")

    ax.set_xlabel("n_effective (prior confidence)", fontsize=12)
    ax.set_ylabel("Holdout Reward", fontsize=12)
    ax.set_title("Prior Confidence Sensitivity: n_effective Sweep", fontsize=13)
    ax.set_xscale("log")
    ax.set_xticks(NEFF_VALUES)
    ax.set_xticklabels([str(v) for v in NEFF_VALUES])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "part4_neff_sweep.png", dpi=200)
    fig.savefig(output_dir / "part4_neff_sweep.pdf")
    plt.close(fig)
    logger.info(f"  Saved Part 4 figure: {output_dir / 'part4_neff_sweep.png'}")


def plot_part3_growth(results_realistic: Dict, results_adversarial: Dict, output_dir: Path):
    """Side-by-side portfolio growth: realistic vs adversarial scenarios."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

    colors = {"oracle": "#2ecc71", "transfer": "#3498db", "tabula_rasa": "#e74c3c"}
    mode_labels = {"oracle": "Oracle Priors", "transfer": "Semantic Transfer", "tabula_rasa": "Tabula Rasa"}

    for ax_idx, (results, title_suffix) in enumerate([
        (results_realistic, "Realistic Growth (family-based)"),
        (results_adversarial, "Adversarial Growth (cross-family)"),
    ]):
        ax = axes[ax_idx]
        phases = results["_phases"]
        phase_ks = [K for K, _ in phases]

        for mode in ["oracle", "transfer", "tabula_rasa"]:
            means = []
            ci95s = []
            for pi in range(len(phases)):
                evals = [results[mode]["phase_eval_rewards"][s][pi]
                         for s in range(len(results[mode]["phase_eval_rewards"]))
                         if pi < len(results[mode]["phase_eval_rewards"][s])]
                means.append(np.mean(evals))
                ci95s.append(1.96 * np.std(evals) / np.sqrt(len(evals)) if len(evals) > 1 else 0)

            means = np.array(means)
            ci95s = np.array(ci95s)
            ax.plot(phase_ks, means, "o-", color=colors[mode], label=mode_labels[mode],
                    lw=2, markersize=8)
            ax.fill_between(phase_ks, means - ci95s, means + ci95s,
                            color=colors[mode], alpha=0.15)

        ax.set_xlabel("Portfolio Size (K)", fontsize=12)
        ax.set_title(title_suffix, fontsize=12)
        ax.set_xticks(phase_ks)
        ax.grid(True, alpha=0.3)
        if ax_idx == 0:
            ax.set_ylabel("Holdout Eval Reward", fontsize=12)
            ax.legend(fontsize=9)

    fig.suptitle("Portfolio Growth: Semantic Transfer vs Tabula Rasa", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "part3_portfolio_growth.png", dpi=200, bbox_inches="tight")
    fig.savefig(output_dir / "part3_portfolio_growth.pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved Part 3 figure: {output_dir / 'part3_portfolio_growth.png'}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    logger.info("=" * 70)
    logger.info("SEMANTIC TRANSFER STRESS TEST")
    logger.info("=" * 70)
    t_total = time.time()

    # ---- Load data ----
    REWARD_MODE = "continuous"  # "continuous" = judge-weighted, "binary" = raw_score
    logger.info("\n--- Loading data ---")
    logger.info(f"  Reward mode: {REWARD_MODE}")
    dev_data_raw = load_all_models_data(DEV_DATA_PATH_ALL_MODELS, min_models=2,
                                         reward_mode=REWARD_MODE)
    holdout_data_raw = load_all_models_data(HOLDOUT_DATA_PATH_ALL_MODELS, min_models=2,
                                             reward_mode=REWARD_MODE)
    logger.info(f"  Dev: {len(dev_data_raw)} prompts, Holdout: {len(holdout_data_raw)} prompts")

    # Split dev 50/50 (same seed as K-scaling experiment for reproducibility)
    np.random.seed(SEED_OFFSET)
    n_dev = len(dev_data_raw)
    perm = np.random.permutation(n_dev)
    split_idx = n_dev // 2
    prior_data = [dev_data_raw[i] for i in sorted(perm[:split_idx])]
    burnin_data = [dev_data_raw[i] for i in sorted(perm[split_idx:])]
    logger.info(f"  Prior split: {len(prior_data)}, Burn-in split: {len(burnin_data)}")

    # ---- Load encoder, PCA, compute embeddings ----
    logger.info("\n--- Loading encoder and PCA ---")
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
    pca = joblib.load(DEFAULT_PCA_PATH)

    logger.info("--- Computing embeddings (this may take a minute) ---")
    t0 = time.time()
    prior_emb = [embed_prompt(p["prompt"], encoder, pca) for p in prior_data]
    burnin_emb = [embed_prompt(p["prompt"], encoder, pca) for p in burnin_data]
    holdout_emb = [embed_prompt(p["prompt"], encoder, pca) for p in holdout_data_raw]
    logger.info(f"  Encoded {len(prior_emb)}+{len(burnin_emb)}+{len(holdout_emb)} "
                f"prompts in {time.time()-t0:.1f}s")

    # Build prompt→index maps
    prior_prompt_idx = {p["prompt"]: i for i, p in enumerate(prior_data)}
    burnin_prompt_idx = {p["prompt"]: i for i, p in enumerate(burnin_data)}
    holdout_prompt_idx = {p["prompt"]: i for i, p in enumerate(holdout_data_raw)}

    # Compute DNA embeddings for all models
    logger.info("\n--- Computing model DNA embeddings ---")
    all_models_in_data = sorted(
        set().union(*(set(p["rewards"].keys()) for p in dev_data_raw))
    )
    all_models_in_data = [m for m in all_models_in_data if m not in EXCLUDED_MODELS]
    dna_embeddings = compute_dna_embeddings(all_models_in_data, encoder)
    logger.info(f"  DNA embeddings for {len(dna_embeddings)} models")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ================================================================
    # PART 1: Transfer Quality (K=20)
    # ================================================================
    models_k20 = PORTFOLIOS[20]
    prior_k20 = filter_portfolio(prior_data, models_k20)
    prior_emb_k20 = [prior_emb[prior_prompt_idx[p["prompt"]]] for p in prior_k20]
    r_bounds_k20 = compute_reward_bounds(prior_k20, models_k20)

    part1_results = run_part1_transfer_quality(
        models_k20, prior_k20, prior_emb_k20, dna_embeddings, r_bounds_k20,
    )
    plot_part1_scatter(part1_results, RESULTS_DIR)

    # ================================================================
    # PART 2: Online Onboarding at multiple K values
    # ================================================================
    def _prepare_portfolio(k_val):
        models_k = PORTFOLIOS[k_val]
        prior_k = filter_portfolio(prior_data, models_k)
        burnin_k = filter_portfolio(burnin_data, models_k)
        eval_k = filter_portfolio(holdout_data_raw, models_k)
        prior_emb_k = [prior_emb[prior_prompt_idx[p["prompt"]]] for p in prior_k]
        burnin_emb_k = [burnin_emb[burnin_prompt_idx[p["prompt"]]] for p in burnin_k]
        eval_emb_k = [holdout_emb[holdout_prompt_idx[p["prompt"]]] for p in eval_k]
        r_bounds_k = compute_reward_bounds(prior_k, models_k)
        raw_costs_k = {m: compute_cost_per_request(m) for m in models_k}
        c_min, c_max = min(raw_costs_k.values()), max(raw_costs_k.values())
        c_rng = c_max - c_min if c_max > c_min else 1.0
        norm_costs_k = {m: {"normalized_cost": (raw_costs_k[m] - c_min) / c_rng} for m in models_k}
        logger.info(f"\n  K={k_val} cost range: ${c_min:.6f} – ${c_max:.6f} ({c_max/max(c_min,1e-9):.0f}×)")
        return models_k, prior_k, burnin_k, eval_k, prior_emb_k, burnin_emb_k, eval_emb_k, r_bounds_k, norm_costs_k, raw_costs_k

    part2_all = {}
    for k_val in [5, 10]:
        models_k, prior_k, burnin_k, eval_k, prior_emb_k, burnin_emb_k, eval_emb_k, r_bounds_k, norm_costs_k, raw_costs_k = _prepare_portfolio(k_val)
        part2_all[k_val] = run_part2_online_onboarding(
            models_k, prior_k, burnin_k, eval_k,
            prior_emb_k, burnin_emb_k, eval_emb_k,
            dna_embeddings, r_bounds_k,
            model_costs_norm=norm_costs_k,
            model_costs_raw=raw_costs_k,
            k_label=f"(K={k_val})",
        )

    # Use K=10 for plotting and Part 4
    models_k10 = PORTFOLIOS[10]
    prior_k10 = filter_portfolio(prior_data, models_k10)
    burnin_k10 = filter_portfolio(burnin_data, models_k10)
    eval_k10 = filter_portfolio(holdout_data_raw, models_k10)
    prior_emb_k10 = [prior_emb[prior_prompt_idx[p["prompt"]]] for p in prior_k10]
    burnin_emb_k10 = [burnin_emb[burnin_prompt_idx[p["prompt"]]] for p in burnin_k10]
    eval_emb_k10 = [holdout_emb[holdout_prompt_idx[p["prompt"]]] for p in eval_k10]
    r_bounds_k10 = compute_reward_bounds(prior_k10, models_k10)
    raw_costs_k10 = {m: compute_cost_per_request(m) for m in models_k10}
    min_c, max_c = min(raw_costs_k10.values()), max(raw_costs_k10.values())
    c_range = max_c - min_c if max_c > min_c else 1.0
    norm_costs_k10 = {m: {"normalized_cost": (raw_costs_k10[m] - min_c) / c_range} for m in models_k10}
    part2_results = part2_all[10]
    plot_part2_onboarding(part2_results, RESULTS_DIR)

    # ================================================================
    # PART 3: Portfolio Growth — two scenarios
    # ================================================================
    logger.info("\n" + "=" * 70)
    logger.info("PART 3: PORTFOLIO GROWTH SIMULATION")
    logger.info("=" * 70)

    common_kwargs = dict(
        burnin_data_raw=burnin_data,
        eval_data_raw=holdout_data_raw,
        prior_data_raw=prior_data,
        burnin_emb_all=burnin_emb,
        eval_emb_all=holdout_emb,
        prior_emb_all=prior_emb,
        dna_embeddings=dna_embeddings,
        burnin_prompt_idx=burnin_prompt_idx,
        eval_prompt_idx=holdout_prompt_idx,
        prior_prompt_idx=prior_prompt_idx,
    )

    part3_realistic = run_part3_portfolio_growth(
        **common_kwargs,
        growth_path=REALISTIC_GROWTH,
        scenario_name="Realistic (family-based)",
    )
    part3_adversarial = run_part3_portfolio_growth(
        **common_kwargs,
        growth_path=ADVERSARIAL_GROWTH,
        scenario_name="Adversarial (cross-family)",
    )
    plot_part3_growth(part3_realistic, part3_adversarial, RESULTS_DIR)

    # ================================================================
    # PART 4: n_effective Sensitivity Sweep
    # ================================================================
    part4_results = run_part4_neff_sweep(
        models_k10, prior_k10, burnin_k10, eval_k10,
        prior_emb_k10, burnin_emb_k10, eval_emb_k10,
        dna_embeddings, r_bounds_k10,
        model_costs_norm=norm_costs_k10,
        model_costs_raw=raw_costs_k10,
    )
    plot_part4_neff(part4_results, RESULTS_DIR)

    # ================================================================
    # Save all results to JSON
    # ================================================================
    def _serialize_part2(results_dict):
        return {
            m: {
                "model": r["model"],
                "dna_neighbor": r.get("dna_neighbor", r.get("neighbor")),
                "dna_similarity": r.get("dna_similarity", r.get("similarity")),
                "dna_transfer_type": r.get("dna_transfer_type", r.get("transfer_type")),
                "horizons": {str(k): v for k, v in r["horizons"].items()},
            }
            for m, r in results_dict.items()
        }

    output_data = {
        "metadata": {
            "description": "Semantic Transfer Stress Test (fixed: shuffled seeds, b-only control, true oracle, multi-K)",
            "n_seeds": N_SEEDS,
            "seed_offset": SEED_OFFSET,
            "prior_neff": TARGET_PRIOR_NEFF,
            "portfolios": {str(k): v for k, v in PORTFOLIOS.items()},
            "fixes": [
                "burn-in data shuffled per seed for statistical independence",
                "added transfer_b_only mode (direction only, full exploration)",
                "added true_oracle (per-prompt best model selection)",
                "report new-model selection rates and per-prompt regret",
                "run at K=5 and K=10",
            ],
        },
        "part1_transfer_quality": part1_results,
        "part2_onboarding": {
            str(k_val): _serialize_part2(k_results)
            for k_val, k_results in part2_all.items()
        },
        "part3_growth": {
            "realistic": {
                mode: {"phase_eval_rewards": vals["phase_eval_rewards"]}
                for mode, vals in part3_realistic.items()
                if not mode.startswith("_")
            },
            "adversarial": {
                mode: {"phase_eval_rewards": vals["phase_eval_rewards"]}
                for mode, vals in part3_adversarial.items()
                if not mode.startswith("_")
            },
        },
        "part4_neff_sweep": {
            str(N): {
                k: v for k, v in horizon_data.items()
            }
            for N, horizon_data in part4_results.items()
        },
    }

    with open(RESULTS_DIR / "semantic_transfer_results.json", "w") as f:
        json.dump(output_data, f, indent=2, default=str)
    logger.info(f"\n  Saved results: {RESULTS_DIR / 'semantic_transfer_results.json'}")

    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    logger.info("\n" + "=" * 70)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 70)

    # Part 1 summary
    sims = [r["similarity"] for r in part1_results if r["similarity"] > MIN_SIMILARITY]
    cosines = [r["theta_cosine"] for r in part1_results if r["similarity"] > MIN_SIMILARITY]
    logger.info(f"\n  Part 1 — Transfer Quality (K=20, {len(sims)} models):")
    logger.info(f"    Mean DNA similarity:  {np.mean(sims):.3f}")
    logger.info(f"    Mean theta cosine:    {np.mean(cosines):.3f}")
    logger.info(f"    Correlation (sim→cos): {np.corrcoef(sims, cosines)[0,1]:.3f}")

    # Part 2 summary — transfer ablation at each K
    SUMMARY_MODES = ["true_oracle", "oracle", "transfer_dna", "transfer_dna_b_only", "transfer_random", "tabula_rasa"]
    for k_val, p2_res in part2_all.items():
        sample_model = next(iter(p2_res))
        p2_horizons = sorted(p2_res[sample_model]["horizons"].keys())
        logger.info(f"\n  Part 2 — Transfer Ablation (K={k_val}, leave-one-out, cost_penalty={COST_PENALTY}):")
        logger.info(f"    {'N':>5}  {'TrueOracl':>10}  {'PriorOrcl':>10}  {'DNA Xfer':>10}  {'DNA BOnly':>10}  {'RandXfer':>10}  {'TabRasa':>10}  {'SelRate':>8}")
        logger.info("    " + "-" * 90)
        for N in p2_horizons:
            vals = {}
            sel_rates = {}
            for mode in SUMMARY_MODES:
                rs = [p2_res[m]["horizons"][N][mode]["mean"] for m in p2_res
                      if mode in p2_res[m]["horizons"][N]]
                vals[mode] = np.mean(rs) if rs else 0.0
                srs = [p2_res[m]["horizons"][N][mode]["sel_rate_eval"] for m in p2_res
                       if mode in p2_res[m]["horizons"][N]]
                sel_rates[mode] = np.mean(srs) if srs else 0.0
            logger.info(
                f"    N={N:>3}  {vals['true_oracle']:>10.4f}  {vals['oracle']:>10.4f}  "
                f"{vals['transfer_dna']:>10.4f}  {vals['transfer_dna_b_only']:>10.4f}  "
                f"{vals['transfer_random']:>10.4f}  {vals['tabula_rasa']:>10.4f}  "
                f"{sel_rates['transfer_dna']:>7.3f}"
            )

    # Part 3 summary — both scenarios
    for label, part3_res in [("Realistic", part3_realistic), ("Adversarial", part3_adversarial)]:
        phases = part3_res["_phases"]
        logger.info(f"\n  Part 3 — {label} Growth ({N_SEEDS} seeds):")
        logger.info(f"    {'K':<5} {'Oracle':>12} {'Transfer':>12} {'Tab.Rasa':>12} {'GapClosed':>12}")
        logger.info("    " + "-" * 55)
        for pi, (K, _) in enumerate(phases):
            vals = {}
            for mode in ["oracle", "transfer", "tabula_rasa"]:
                evals = [part3_res[mode]["phase_eval_rewards"][s][pi]
                         for s in range(N_SEEDS)
                         if pi < len(part3_res[mode]["phase_eval_rewards"][s])]
                vals[mode] = np.mean(evals) if evals else 0.0
            gap = vals["oracle"] - vals["tabula_rasa"]
            closed = (vals["transfer"] - vals["tabula_rasa"]) / gap * 100 if abs(gap) > 1e-6 else 0.0
            logger.info(
                f"    K={K:<3} {vals['oracle']:>10.4f}  {vals['transfer']:>10.4f}  "
                f"{vals['tabula_rasa']:>10.4f}  {closed:>10.1f}%"
            )

    # Part 4 summary — n_effective sweep
    logger.info(f"\n  Part 4 — n_effective Sensitivity Sweep:")
    p4_horizons = sorted(part4_results.keys())
    for N in p4_horizons:
        oracle_r = part4_results[N]["oracle"]["mean"]
        tabula_r = part4_results[N]["tabula_rasa"]["mean"]
        gap = oracle_r - tabula_r
        best_neff = None
        best_gc = -float("inf")
        logger.info(f"    N={N}: Oracle={oracle_r:.4f}, Tabula={tabula_r:.4f}")
        header = "      " + "  ".join(f"neff={ne:<3}" for ne in NEFF_VALUES)
        logger.info(header)
        vals_str = "      "
        for ne in NEFF_VALUES:
            r = part4_results[N][f"neff_{ne}"]["mean"]
            gc = (r - tabula_r) / gap * 100 if abs(gap) > 1e-6 else 0.0
            vals_str += f"{gc:>+7.1f}%  "
            if gc > best_gc:
                best_gc = gc
                best_neff = ne
        logger.info(vals_str)
        logger.info(f"      Best: n_eff={best_neff} ({best_gc:+.1f}% gap closed)")

    elapsed = time.time() - t_total
    logger.info(f"\n  Total time: {elapsed:.0f}s")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
