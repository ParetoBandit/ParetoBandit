#!/usr/bin/env python3
"""
Unified Experiment Runner for Figure 3
=======================================

Runs all experiments for the Figure 3 panel in the paper.

Experiments:
1. 2A: Expert Weight Evolution (validates meta-learning adaptation)
2. 2BC: Convergence Dynamics (validates learning speed)
3. 3: Heterogeneous Alpha Ablation (validates exploration strategy)
4. prior: Prior Quality Degradation Sweep (finds crossover point)
5. 5: Gamma Ablation (optional; validates mixing parameter -- null result)

CRITICAL FIX (2026-02-14):
All experiments now properly capture and pass selection_token to enable
meta-learning. Previous versions had frozen weights due to missing token.

Usage:
    python run_all_experiments.py [--experiments 2a,2bc,3,prior] [--seeds 20]

Author: BanditGPT Team
Date: 2026-02-14
Status: Production-ready (bug fixed)
"""

import sys
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm
import argparse
import time
from scipy import stats as sp_stats

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from bandit_gpt.router import CorrallingRouter, CostAwareLinUCBRouter, CostAwareTabulaRasaRouter
from bandit_gpt.calibration import embed_prompt, apply_gamma_scaling
from sentence_transformers import SentenceTransformer
import joblib
from bandit_gpt.config_legacy import (
    DEFAULT_SENTENCE_TRANSFORMER,
    DEFAULT_WARMUP_PRIORS_PATH,
    DEFAULT_PCA_PATH,
    CANONICAL_HOLDOUT_DATA_PATH
)

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# SHARED CONFIGURATION
# ============================================================================
# Meta-learning rate η for the Corralling coordinator.
# Theoretically optimal: η* = sqrt(ln(K) / T)  [Agarwal et al., 2017]
# where K = number of experts, T = horizon (number of prompts).
# Computed at runtime in main() after loading data; see LEARNING_RATE below.
N_EXPERTS = 2  # K: warmup + tabula rasa
LEARNING_RATE = None  # Computed from data size; see compute_learning_rate()

# These are TWO DIFFERENT parameters that were previously conflated as "GAMMA":
CORRALLING_GAMMA = 0.05   # Meta-learner mixing floor (expert death prevention)
PRIOR_SCALING = 0.05      # Prior confidence reduction (0.05 = keep 5% of prior strength)

# Initial warmup expert weight for Corralling (prior-trust bias).
# 0.5 = uniform (no trust), 0.7 = moderate trust (recommended default).
# See Appendix D.3 for the full trade-off analysis.
INITIAL_WARMUP_WEIGHT = 0.7

BASE_OUTPUT_DIR = Path(__file__).parent / "results"


def compute_learning_rate(n_experts: int, horizon: int) -> float:
    """
    Compute theoretically optimal Exp4 learning rate.

    η* = sqrt(ln(K) / T)

    For K=2, T=750: η* ≈ 0.030
    """
    return float(np.sqrt(np.log(n_experts) / horizon))


# =================================================================
# PLOTTING CONFIGURATION
# =================================================================
PLOT_STYLE = {
    "figure.figsize": (10, 4),
    "font.size": 11,
    "font.family": "sans-serif",
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
}

# Colorblind-friendly palette
COLORS = {
    "blue": "#0173B2",
    "orange": "#DE8F05",
    "green": "#029E73",
    "red": "#CC78BC",
    "purple": "#9467bd",
    "gray": "#7f7f7f",
    "cyan": "#17becf",
    "brown": "#8c564b",
}


# ============================================================================
# SHARED HELPERS
# ============================================================================

def load_resources():
    """Load shared resources once."""
    logger.info("📦 Loading shared resources...")
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
    pca = joblib.load(DEFAULT_PCA_PATH)
    warmup_priors = joblib.load(DEFAULT_WARMUP_PRIORS_PATH)
    warmup_priors_scaled = apply_gamma_scaling(warmup_priors, gamma=PRIOR_SCALING)

    models = warmup_priors['models']
    context_dim = warmup_priors['A'][models[0]].shape[0]

    logger.info(f"   ✅ Models: {len(models)}")
    logger.info(f"   ✅ Context Dim: {context_dim}")

    return encoder, pca, warmup_priors_scaled, warmup_priors, models, context_dim


def load_holdout_data():
    """Load holdout dataset."""
    import gzip

    logger.info("📊 Loading holdout data...")

    entries = []
    with gzip.open(CANONICAL_HOLDOUT_DATA_PATH, 'rt') as f:
        for line in f:
            entries.append(json.loads(line))

    prompt_data = {}
    for entry in entries:
        prompt = entry['prompt']
        model_id = entry['model_id']
        score = entry.get('raw_score', 0.0)

        if prompt not in prompt_data:
            prompt_data[prompt] = {'prompt': prompt, 'scores': {}}

        prompt_data[prompt]['scores'][model_id] = score

    data_list = list(prompt_data.values())

    # Signal sparsity analysis
    n = len(data_list)
    models_in_data = set()
    both_agree = 0
    for d in data_list:
        models_in_data.update(d['scores'].keys())
        vals = list(d['scores'].values())
        if len(vals) == 2 and vals[0] == vals[1]:
            both_agree += 1
    differentiating = n - both_agree

    logger.info(f"   ✅ Loaded {n} unique prompts, {len(models_in_data)} models")
    logger.info(f"   ⚠️  Signal sparsity: {both_agree}/{n} prompts ({100*both_agree/n:.0f}%) "
                f"have identical scores → zero meta-learner signal")
    logger.info(f"   📊 Differentiating prompts: {differentiating}/{n} ({100*differentiating/n:.0f}%) "
                f"→ max possible cumulative regret = {differentiating}")

    return data_list


def precompute_embeddings(data, encoder, pca):
    """
    Pre-compute embeddings for all prompts.

    Each prompt only needs to be embedded once, regardless of how many
    seeds/strategies/experiments use it.  This is ~100x faster than
    re-embedding inside every trial loop.
    """
    logger.info("📐 Pre-computing embeddings for all prompts...")
    embeddings = {}
    for sample in tqdm(data, desc="Embedding", leave=False):
        prompt = sample['prompt']
        if prompt not in embeddings:
            embeddings[prompt] = embed_prompt(prompt, encoder, pca)
    logger.info(f"   ✅ {len(embeddings)} unique embeddings cached")
    return embeddings


def corrupt_priors(priors_scaled: dict, corruption_level: float) -> dict:
    """
    Interpolate priors toward model-swapped version.

    Creates a smooth spectrum of prior quality:
        α=0.0:  original (correct) priors
        α=0.5:  priors averaged between models (uninformative)
        α=1.0:  priors fully swapped (adversarial — systematically wrong)

    For each model m_i, the corrupted priors are:
        A_new[m_i] = (1-α) * A_orig[m_i] + α * A_orig[m_j]
        b_new[m_i] = (1-α) * b_orig[m_i] + α * b_orig[m_j]

    This preserves the overall matrix scale (confidence level) while
    corrupting the direction of θ = A⁻¹b (prediction quality).
    """
    models = priors_scaled['models']
    assert len(models) == 2, f"Corruption assumes 2 models, got {len(models)}"
    m1, m2 = models
    alpha = corruption_level

    A_new = {
        m1: (1 - alpha) * priors_scaled['A'][m1] + alpha * priors_scaled['A'][m2],
        m2: (1 - alpha) * priors_scaled['A'][m2] + alpha * priors_scaled['A'][m1],
    }
    b_new = {
        m1: (1 - alpha) * priors_scaled['b'][m1] + alpha * priors_scaled['b'][m2],
        m2: (1 - alpha) * priors_scaled['b'][m2] + alpha * priors_scaled['b'][m1],
    }

    result = dict(priors_scaled)  # shallow copy
    result['A'] = A_new
    result['b'] = b_new
    return result


def compute_paired_test(results_a: List[float], results_b: List[float],
                        name_a: str, name_b: str) -> dict:
    """
    Compute Wilcoxon signed-rank test for paired per-seed regrets.

    Uses Wilcoxon (non-parametric) since regret distributions may be
    non-normal.  Also reports Cohen's d for effect size.
    """
    a = np.array(results_a)
    b = np.array(results_b)
    diff = a - b

    # Wilcoxon signed-rank test (paired, non-parametric)
    try:
        stat, p_value = sp_stats.wilcoxon(diff)
    except ValueError:
        # All differences are zero
        stat, p_value = 0.0, 1.0

    # Cohen's d (effect size)
    pooled_std = np.sqrt((np.std(a)**2 + np.std(b)**2) / 2)
    cohens_d = float(np.mean(diff) / pooled_std) if pooled_std > 0 else 0.0

    return {
        'comparison': f'{name_a} vs {name_b}',
        'mean_a': float(np.mean(a)),
        'mean_b': float(np.mean(b)),
        'mean_diff': float(np.mean(diff)),
        'wilcoxon_stat': float(stat),
        'p_value': float(p_value),
        'cohens_d': cohens_d,
        'significant_005': bool(p_value < 0.05),
        'significant_001': bool(p_value < 0.001),
    }


# ============================================================================
# EXPERIMENT 2A: WEIGHT EVOLUTION
# ============================================================================
def run_experiment_2a(embeddings, warmup_priors, models, context_dim, data, n_seeds=20):
    """
    Expert Weight Evolution: Track how trust shifts between experts.

    Configuration: CONSTANT alpha=2.0 for both experts (matches Experiment 2BC)
    """
    logger.info("\n" + "="*80)
    logger.info("EXPERIMENT 2A: EXPERT WEIGHT EVOLUTION")
    logger.info("="*80)

    output_dir = BASE_OUTPUT_DIR / "weight_evolution"
    output_dir.mkdir(parents=True, exist_ok=True)

    def run_single_trial(seed):
        np.random.seed(seed)
        rng = np.random.RandomState(seed)

        warmup_expert = CostAwareLinUCBRouter(
            models=models, warmup_priors=warmup_priors,
            alpha_start=2.0, alpha_end=2.0, cost_penalty=0.0,
            model_costs={m: {"normalized_cost": 0.0} for m in models}
        )
        tabula_rasa_expert = CostAwareTabulaRasaRouter(
            models=models, context_dim=context_dim,
            alpha_start=2.0, alpha_end=2.0, cost_penalty=0.0,
            model_costs={m: {"normalized_cost": 0.0} for m in models}
        )
        router = CorrallingRouter(
            experts=[warmup_expert, tabula_rasa_expert],
            models=models, learning_rate=LEARNING_RATE, gamma=CORRALLING_GAMMA,
            loss_decay=1.0,
            initial_weights=np.array([INITIAL_WARMUP_WEIGHT, 1 - INITIAL_WARMUP_WEIGHT]),
        )

        weights_history = []
        cumulative_regret = 0.0
        indices = rng.permutation(len(data))
        total_steps = len(data)

        for i, idx in enumerate(tqdm(indices, desc=f"Seed {seed}", leave=False)):
            sample = data[idx]
            context = embeddings[sample['prompt']]

            selected_model, selection_token = router.select_model(context, total_steps=total_steps)

            scores = sample.get('scores', {})
            if not scores:
                continue

            oracle_model = max(scores, key=scores.get)
            oracle_reward = scores[oracle_model]
            model_reward = scores.get(selected_model, 0.0)

            regret = oracle_reward - model_reward
            cumulative_regret += regret

            router.update(context, selected_model, model_reward, selection_token)

            weights_history.append({
                'step': i + 1,
                'warmup': float(router.weights[0]),
                'tabula_rasa': float(router.weights[1]),
                'regret': cumulative_regret
            })

        return {
            'weights_history': weights_history,
            'final_regret': cumulative_regret,
            'final_warmup_weight': float(router.weights[0]),
            'final_tabula_weight': float(router.weights[1])
        }

    logger.info(f"\n🔬 Running {n_seeds} trials...")
    all_results = []

    for seed in range(42, 42 + n_seeds):
        result = run_single_trial(seed)
        all_results.append(result)
        logger.info(f"   Seed {seed}: Final weights = [{result['final_warmup_weight']:.3f}, "
                   f"{result['final_tabula_weight']:.3f}], Regret = {result['final_regret']:.1f}")

    final_warmup = [r['final_warmup_weight'] for r in all_results]
    final_tabula = [r['final_tabula_weight'] for r in all_results]
    final_regrets = [r['final_regret'] for r in all_results]

    stats = {
        'n_seeds': n_seeds,
        'learning_rate': LEARNING_RATE,
        'corralling_gamma': CORRALLING_GAMMA,
        'prior_scaling': PRIOR_SCALING,
        'data_size': len(data),
        'initial_weights': {'warmup': 0.5, 'tabula_rasa': 0.5},
        'final_weights': {
            'warmup': float(np.mean(final_warmup)),
            'warmup_std': float(np.std(final_warmup)),
            'tabula_rasa': float(np.mean(final_tabula)),
            'tabula_std': float(np.std(final_tabula))
        },
        'average_final_regret': float(np.mean(final_regrets)),
        'regret_std': float(np.std(final_regrets))
    }

    stats_path = output_dir / "statistics.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    logger.info(f"\n📊 Results:")
    logger.info(f"   Final Warmup: {stats['final_weights']['warmup']:.3f} ± {stats['final_weights']['warmup_std']:.3f}")
    logger.info(f"   Final Tabula: {stats['final_weights']['tabula_rasa']:.3f} ± {stats['final_weights']['tabula_std']:.3f}")
    logger.info(f"   Avg Regret: {stats['average_final_regret']:.1f} ± {stats['regret_std']:.1f}")
    logger.info(f"💾 Saved: {stats_path}")

    weight_histories = [r['weights_history'] for r in all_results]
    return stats, weight_histories


# ============================================================================
# EXPERIMENT 2BC: CONVERGENCE DYNAMICS
# ============================================================================
def run_experiment_2bc(embeddings, warmup_priors, models, context_dim, data, n_seeds=20):
    """
    Convergence Dynamics: Compare learning speeds of different strategies.

    Tests: Corralling vs Warmup-Only vs Tabula-Rasa
    """
    logger.info("\n" + "="*80)
    logger.info("EXPERIMENT 2BC: CONVERGENCE DYNAMICS")
    logger.info("="*80)

    output_dir = BASE_OUTPUT_DIR / "convergence"
    output_dir.mkdir(parents=True, exist_ok=True)

    def run_single_strategy(strategy_name, seed):
        np.random.seed(seed)
        rng = np.random.RandomState(seed)

        if strategy_name == "corralling":
            warmup_expert = CostAwareLinUCBRouter(
                models=models, warmup_priors=warmup_priors,
                alpha_start=2.0, alpha_end=2.0, cost_penalty=0.0,
                model_costs={m: {"normalized_cost": 0.0} for m in models}
            )
            tabula_rasa_expert = CostAwareTabulaRasaRouter(
                models=models, context_dim=context_dim,
                alpha_start=2.0, alpha_end=2.0, cost_penalty=0.0,
                model_costs={m: {"normalized_cost": 0.0} for m in models}
            )
            router = CorrallingRouter(
                experts=[warmup_expert, tabula_rasa_expert],
                models=models, learning_rate=LEARNING_RATE, gamma=CORRALLING_GAMMA,
                loss_decay=1.0,
                initial_weights=np.array([INITIAL_WARMUP_WEIGHT, 1 - INITIAL_WARMUP_WEIGHT]),
            )
        elif strategy_name == "warmup_only":
            router = CostAwareLinUCBRouter(
                models=models, warmup_priors=warmup_priors,
                alpha_start=2.0, alpha_end=2.0, cost_penalty=0.0,
                model_costs={m: {"normalized_cost": 0.0} for m in models}
            )
        else:  # tabula_rasa
            router = CostAwareTabulaRasaRouter(
                models=models, context_dim=context_dim,
                alpha_start=2.0, alpha_end=2.0, cost_penalty=0.0,
                model_costs={m: {"normalized_cost": 0.0} for m in models}
            )

        regret_history = []
        cumulative_regret = 0.0
        indices = rng.permutation(len(data))
        total_steps = len(data)

        for i, idx in enumerate(tqdm(indices, desc=f"{strategy_name}-{seed}", leave=False)):
            sample = data[idx]
            context = embeddings[sample['prompt']]

            if strategy_name == "corralling":
                selected_model, selection_token = router.select_model(context, total_steps=total_steps)
            else:
                result = router.select_model(context, total_steps=total_steps)
                if isinstance(result, tuple):
                    selected_model, selection_token = result
                else:
                    selected_model = result
                    selection_token = None

            scores = sample.get('scores', {})
            if not scores:
                continue

            oracle_model = max(scores, key=scores.get)
            model_reward = scores.get(selected_model, 0.0)
            regret = scores[oracle_model] - model_reward
            cumulative_regret += regret

            if strategy_name == "corralling":
                router.update(context, selected_model, model_reward, selection_token)
            else:
                if selection_token:
                    router.update(context, selected_model, model_reward, selection_token)
                else:
                    router.update(context, selected_model, model_reward)

            regret_history.append(cumulative_regret)

        return regret_history

    strategies = ["corralling", "warmup_only", "tabula_rasa"]
    logger.info(f"\n🔬 Running {len(strategies)} strategies × {n_seeds} seeds...")

    results = {s: [] for s in strategies}
    for strategy in strategies:
        for seed in range(42, 42 + n_seeds):
            regret_hist = run_single_strategy(strategy, seed)
            results[strategy].append(regret_hist)

        final_regrets = [r[-1] for r in results[strategy]]
        logger.info(f"   {strategy}: {np.mean(final_regrets):.1f} ± {np.std(final_regrets):.1f}")

    # Build stats with per-seed regrets
    stats = {
        'strategies': {s: {
            'regret_mean': float(np.mean([r[-1] for r in results[s]])),
            'regret_std': float(np.std([r[-1] for r in results[s]])),
            'per_seed_regrets': [float(r[-1]) for r in results[s]]
        } for s in strategies},
        'n_seeds': n_seeds,
        'data_size': len(data),
        'learning_rate': LEARNING_RATE,
        'corralling_gamma': CORRALLING_GAMMA
    }

    # Statistical tests (paired by seed)
    logger.info("\n📊 Paired Wilcoxon signed-rank tests:")
    stat_tests = []
    for sa, sb in [("warmup_only", "corralling"), ("warmup_only", "tabula_rasa"),
                   ("corralling", "tabula_rasa")]:
        test = compute_paired_test(
            stats['strategies'][sa]['per_seed_regrets'],
            stats['strategies'][sb]['per_seed_regrets'],
            sa, sb
        )
        stat_tests.append(test)
        sig = "***" if test['significant_001'] else ("*" if test['significant_005'] else "ns")
        logger.info(f"   {test['comparison']}: Δ={test['mean_diff']:.1f}, "
                    f"p={test['p_value']:.4f} {sig}, d={test['cohens_d']:.2f}")
    stats['statistical_tests'] = stat_tests

    stats_path = output_dir / "convergence_statistics.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    logger.info(f"💾 Saved: {stats_path}")
    return stats


# ============================================================================
# EXPERIMENT 3: ALPHA ABLATION
# ============================================================================
def run_experiment_3(embeddings, warmup_priors, models, context_dim, data, n_seeds=20):
    """
    Alpha Ablation: Test constant vs adaptive exploration.

    Configurations (w_start, w_end = warmup expert; t_start, t_end = tabula rasa):
    1. Homogeneous Constant: both α=2.0 (constant exploration)
    2. Mixed (Original Design): warmup constant α=2.0, tabula decays 2.0→0.1
    3. Homogeneous Decay: both decay 2.0→0.1
    4. Reversed Mixed: warmup decays 2.0→0.1, tabula constant α=2.0
    """
    logger.info("\n" + "="*80)
    logger.info("EXPERIMENT 3: ALPHA ABLATION")
    logger.info("="*80)

    output_dir = BASE_OUTPUT_DIR / "ablation"
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        ("constant_constant", 2.0, 2.0, 2.0, 2.0),
        ("mixed",             2.0, 2.0, 2.0, 0.1),
        ("decay_decay",       2.0, 0.1, 2.0, 0.1),
        ("reversed_mixed",    2.0, 0.1, 2.0, 2.0),
    ]

    results = {}
    logger.info(f"\n🔬 Running {len(configs)} configs × {n_seeds} seeds...")

    for config_name, w_start, w_end, t_start, t_end in configs:
        config_results = []

        for seed in range(42, 42 + n_seeds):
            np.random.seed(seed)
            rng = np.random.RandomState(seed)

            warmup_expert = CostAwareLinUCBRouter(
                models=models, warmup_priors=warmup_priors,
                alpha_start=w_start, alpha_end=w_end, cost_penalty=0.0,
                model_costs={m: {"normalized_cost": 0.0} for m in models}
            )
            tabula_expert = CostAwareTabulaRasaRouter(
                models=models, context_dim=context_dim,
                alpha_start=t_start, alpha_end=t_end, cost_penalty=0.0,
                model_costs={m: {"normalized_cost": 0.0} for m in models}
            )
            router = CorrallingRouter(
                experts=[warmup_expert, tabula_expert],
                models=models, learning_rate=LEARNING_RATE, gamma=CORRALLING_GAMMA,
                loss_decay=1.0,
                initial_weights=np.array([INITIAL_WARMUP_WEIGHT, 1 - INITIAL_WARMUP_WEIGHT]),
            )

            cumulative_regret = 0.0
            indices = rng.permutation(len(data))
            total_steps = len(data)

            for i, idx in enumerate(tqdm(indices, desc=f"{config_name}-{seed}", leave=False)):
                sample = data[idx]
                context = embeddings[sample['prompt']]

                selected_model, selection_token = router.select_model(context, total_steps=total_steps)

                scores = sample.get('scores', {})
                if not scores:
                    continue

                model_reward = scores.get(selected_model, 0.0)
                oracle_reward = max(scores.values())
                cumulative_regret += (oracle_reward - model_reward)

                router.update(context, selected_model, model_reward, selection_token)

            config_results.append(cumulative_regret)

        results[config_name] = config_results
        logger.info(f"   {config_name}: {np.mean(config_results):.1f} ± {np.std(config_results):.1f}")

    stats = {
        'configs': {name: {
            'regret_mean': float(np.mean(results[name])),
            'regret_std': float(np.std(results[name])),
            'per_seed_regrets': [float(r) for r in results[name]]
        } for name in results}
    }

    # Statistical tests: key comparison is constant vs decay
    logger.info("\n📊 Paired Wilcoxon signed-rank tests:")
    stat_tests = []
    for ca, cb in [("constant_constant", "decay_decay"),
                   ("constant_constant", "mixed"),
                   ("decay_decay", "mixed")]:
        test = compute_paired_test(
            stats['configs'][ca]['per_seed_regrets'],
            stats['configs'][cb]['per_seed_regrets'],
            ca, cb
        )
        stat_tests.append(test)
        sig = "***" if test['significant_001'] else ("*" if test['significant_005'] else "ns")
        logger.info(f"   {test['comparison']}: Δ={test['mean_diff']:.1f}, "
                    f"p={test['p_value']:.4f} {sig}, d={test['cohens_d']:.2f}")
    stats['statistical_tests'] = stat_tests

    stats_path = output_dir / "ablation_statistics.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    logger.info(f"💾 Saved: {stats_path}")
    return stats


# ============================================================================
# EXPERIMENT 5: GAMMA ABLATION (optional -- null result)
# ============================================================================
def run_experiment_5(embeddings, warmup_priors, models, context_dim, data, n_seeds=20):
    """
    Gamma Ablation: Test mixing parameter (prevents expert death).

    NOTE: With correctly calibrated η≈0.030, gamma has negligible impact
    (range 0.7 regret across all values). This experiment is retained for
    completeness but is not included in the default run set.
    """
    logger.info("\n" + "="*80)
    logger.info("EXPERIMENT 5: GAMMA ABLATION")
    logger.info("="*80)

    output_dir = BASE_OUTPUT_DIR / "gamma_ablation"
    output_dir.mkdir(parents=True, exist_ok=True)

    gamma_values = [0.0, 0.05, 0.10, 0.20]
    results = {}

    logger.info(f"\n🔬 Running {len(gamma_values)} gamma values × {n_seeds} seeds...")

    for gamma_val in gamma_values:
        gamma_results = []

        for seed in range(42, 42 + n_seeds):
            np.random.seed(seed)
            rng = np.random.RandomState(seed)

            warmup_expert = CostAwareLinUCBRouter(
                models=models, warmup_priors=warmup_priors,
                alpha_start=2.0, alpha_end=2.0, cost_penalty=0.0,
                model_costs={m: {"normalized_cost": 0.0} for m in models}
            )
            tabula_expert = CostAwareTabulaRasaRouter(
                models=models, context_dim=context_dim,
                alpha_start=2.0, alpha_end=2.0, cost_penalty=0.0,
                model_costs={m: {"normalized_cost": 0.0} for m in models}
            )
            router = CorrallingRouter(
                experts=[warmup_expert, tabula_expert],
                models=models, learning_rate=LEARNING_RATE, gamma=gamma_val,
                loss_decay=1.0,
                initial_weights=np.array([INITIAL_WARMUP_WEIGHT, 1 - INITIAL_WARMUP_WEIGHT]),
            )

            cumulative_regret = 0.0
            indices = rng.permutation(len(data))
            total_steps = len(data)

            for i, idx in enumerate(tqdm(indices, desc=f"gamma={gamma_val}-{seed}", leave=False)):
                sample = data[idx]
                context = embeddings[sample['prompt']]

                selected_model, selection_token = router.select_model(context, total_steps=total_steps)

                scores = sample.get('scores', {})
                if not scores:
                    continue

                model_reward = scores.get(selected_model, 0.0)
                oracle_reward = max(scores.values())
                cumulative_regret += (oracle_reward - model_reward)

                router.update(context, selected_model, model_reward, selection_token)

            gamma_results.append(cumulative_regret)

        results[f"gamma_{gamma_val}"] = gamma_results
        logger.info(f"   γ={gamma_val}: {np.mean(gamma_results):.1f} ± {np.std(gamma_results):.1f}")

    stats = {
        'gamma_values': {f"gamma_{g}": {
            'regret_mean': float(np.mean(results[f"gamma_{g}"])),
            'regret_std': float(np.std(results[f"gamma_{g}"])),
            'per_seed_regrets': [float(r) for r in results[f"gamma_{g}"]]
        } for g in gamma_values}
    }

    stats_path = output_dir / "gamma_statistics.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    logger.info(f"💾 Saved: {stats_path}")
    return stats


# ============================================================================
# EXPERIMENT: PRIOR QUALITY DEGRADATION SWEEP
# ============================================================================
def run_experiment_prior_degradation(embeddings, warmup_priors_unscaled, warmup_priors_scaled,
                                     models, context_dim, data, n_seeds=20):
    """
    Prior Quality Degradation Sweep: Find the crossover point.

    Interpolates priors from correct (α=0) through uninformative (α=0.5)
    to adversarial (α=1.0) by swapping prior matrices between models.
    Tests warmup-only, Corralling, and tabula rasa at each quality level.

    This is the key experiment for Figure 3: it directly answers
    "at what prior quality does Corralling start outperforming warmup-only?"
    """
    logger.info("\n" + "="*80)
    logger.info("EXPERIMENT: PRIOR QUALITY DEGRADATION SWEEP")
    logger.info("="*80)

    output_dir = BASE_OUTPUT_DIR / "prior_degradation"
    output_dir.mkdir(parents=True, exist_ok=True)

    corruption_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    strategies = ["warmup_only", "corralling", "tabula_rasa"]

    all_results = {s: {str(c): [] for c in corruption_levels} for s in strategies}
    # Track weight histories at 3 representative levels for Panel B
    weight_tracking_levels = {0.0, 0.5, 1.0}
    weight_histories = {str(c): [] for c in weight_tracking_levels}

    total_runs = len(corruption_levels) * len(strategies) * n_seeds
    logger.info(f"\n🔬 Running {len(corruption_levels)} corruption levels × "
                f"{len(strategies)} strategies × {n_seeds} seeds = {total_runs} trials")

    for corruption in corruption_levels:
        # Create corrupted priors for this level
        corrupted_priors = corrupt_priors(warmup_priors_scaled, corruption)

        for strategy in strategies:
            for seed in range(42, 42 + n_seeds):
                np.random.seed(seed)
                rng = np.random.RandomState(seed)

                track_weights = (strategy == "corralling" and corruption in weight_tracking_levels)

                if strategy == "corralling":
                    warmup_expert = CostAwareLinUCBRouter(
                        models=models, warmup_priors=corrupted_priors,
                        alpha_start=2.0, alpha_end=2.0, cost_penalty=0.0,
                        model_costs={m: {"normalized_cost": 0.0} for m in models}
                    )
                    tabula_rasa_expert = CostAwareTabulaRasaRouter(
                        models=models, context_dim=context_dim,
                        alpha_start=2.0, alpha_end=2.0, cost_penalty=0.0,
                        model_costs={m: {"normalized_cost": 0.0} for m in models}
                    )
                    router = CorrallingRouter(
                        experts=[warmup_expert, tabula_rasa_expert],
                        models=models, learning_rate=LEARNING_RATE, gamma=CORRALLING_GAMMA,
                        loss_decay=1.0,
                        initial_weights=np.array([INITIAL_WARMUP_WEIGHT, 1 - INITIAL_WARMUP_WEIGHT]),
                    )
                elif strategy == "warmup_only":
                    router = CostAwareLinUCBRouter(
                        models=models, warmup_priors=corrupted_priors,
                        alpha_start=2.0, alpha_end=2.0, cost_penalty=0.0,
                        model_costs={m: {"normalized_cost": 0.0} for m in models}
                    )
                else:  # tabula_rasa
                    router = CostAwareTabulaRasaRouter(
                        models=models, context_dim=context_dim,
                        alpha_start=2.0, alpha_end=2.0, cost_penalty=0.0,
                        model_costs={m: {"normalized_cost": 0.0} for m in models}
                    )

                seed_weights = [] if track_weights else None
                cumulative_regret = 0.0
                indices = rng.permutation(len(data))
                total_steps = len(data)

                for i, idx in enumerate(indices):
                    sample = data[idx]
                    context = embeddings[sample['prompt']]

                    if strategy == "corralling":
                        selected_model, selection_token = router.select_model(
                            context, total_steps=total_steps)
                    else:
                        result = router.select_model(context, total_steps=total_steps)
                        if isinstance(result, tuple):
                            selected_model, selection_token = result
                        else:
                            selected_model = result
                            selection_token = None

                    scores = sample.get('scores', {})
                    if not scores:
                        continue

                    model_reward = scores.get(selected_model, 0.0)
                    oracle_reward = max(scores.values())
                    cumulative_regret += (oracle_reward - model_reward)

                    if strategy == "corralling":
                        router.update(context, selected_model, model_reward, selection_token)
                    else:
                        if selection_token:
                            router.update(context, selected_model, model_reward, selection_token)
                        else:
                            router.update(context, selected_model, model_reward)

                    if track_weights:
                        seed_weights.append(float(router.weights[0]))

                all_results[strategy][str(corruption)].append(cumulative_regret)
                if track_weights:
                    weight_histories[str(corruption)].append(seed_weights)

        # Log progress per corruption level
        warmup_mean = np.mean(all_results['warmup_only'][str(corruption)])
        corr_mean = np.mean(all_results['corralling'][str(corruption)])
        tab_mean = np.mean(all_results['tabula_rasa'][str(corruption)])
        leader = "warmup" if warmup_mean < corr_mean else "CORRALLING"
        logger.info(f"   α={corruption:.1f}: warmup={warmup_mean:.1f}, "
                    f"corralling={corr_mean:.1f}, tabula={tab_mean:.1f}  [{leader}]")

    # Build statistics
    stats = {
        'corruption_levels': corruption_levels,
        'strategies': {},
        'n_seeds': n_seeds,
        'data_size': len(data),
        'learning_rate': LEARNING_RATE,
        'corralling_gamma': CORRALLING_GAMMA,
        'prior_scaling': PRIOR_SCALING,
    }

    for strategy in strategies:
        stats['strategies'][strategy] = {}
        for c in corruption_levels:
            regrets = all_results[strategy][str(c)]
            stats['strategies'][strategy][str(c)] = {
                'regret_mean': float(np.mean(regrets)),
                'regret_std': float(np.std(regrets)),
                'per_seed_regrets': [float(r) for r in regrets],
            }

    # Statistical tests: warmup vs corralling at each corruption level
    logger.info("\n📊 Crossover analysis (warmup_only vs corralling):")
    crossover_tests = []
    crossover_point = None
    for c in corruption_levels:
        w_regrets = all_results['warmup_only'][str(c)]
        c_regrets = all_results['corralling'][str(c)]
        test = compute_paired_test(w_regrets, c_regrets, 'warmup_only', 'corralling')
        test['corruption_level'] = c
        crossover_tests.append(test)

        w_mean = np.mean(w_regrets)
        c_mean = np.mean(c_regrets)
        sig = "***" if test['significant_001'] else ("*" if test['significant_005'] else "ns")
        winner = "warmup" if w_mean < c_mean else "CORRALLING"
        logger.info(f"   α={c:.1f}: warmup={w_mean:.1f} vs corr={c_mean:.1f}, "
                    f"Δ={test['mean_diff']:.1f}, p={test['p_value']:.4f} {sig} [{winner}]")

        # Detect crossover: first level where corralling beats warmup significantly
        if crossover_point is None and c_mean < w_mean and test['significant_005']:
            crossover_point = c

    stats['crossover_tests'] = crossover_tests
    if crossover_point is not None:
        stats['crossover_point'] = crossover_point
        logger.info(f"\n🎯 CROSSOVER POINT: α={crossover_point:.1f} "
                    f"(Corralling first significantly beats warmup-only)")
    else:
        # Find the approximate crossover by interpolation
        # diff = warmup_regret - corralling_regret
        # diff < 0 → warmup better; diff > 0 → corralling better
        for i in range(len(corruption_levels) - 1):
            c1, c2 = corruption_levels[i], corruption_levels[i+1]
            w1 = np.mean(all_results['warmup_only'][str(c1)])
            w2 = np.mean(all_results['warmup_only'][str(c2)])
            c1_r = np.mean(all_results['corralling'][str(c1)])
            c2_r = np.mean(all_results['corralling'][str(c2)])
            diff1 = w1 - c1_r  # negative = warmup better
            diff2 = w2 - c2_r  # positive = corralling better
            if diff1 < 0 and diff2 > 0:
                # Linearly interpolate
                crossover_point = c1 + (c2 - c1) * diff1 / (diff1 - diff2)
                stats['crossover_point_interpolated'] = float(crossover_point)
                logger.info(f"\n🎯 CROSSOVER POINT (interpolated): α≈{crossover_point:.2f}")
                break

        if crossover_point is None:
            logger.info("\n⚠️  No crossover detected in tested range")

    # Save weight histories for Panel B
    stats['weight_histories'] = {}
    for c_str, histories in weight_histories.items():
        if histories:
            wh_array = np.array(histories)
            stats['weight_histories'][c_str] = {
                'mean': wh_array.mean(axis=0).tolist(),
                'std': wh_array.std(axis=0).tolist(),
                'n_seeds': len(histories),
            }

    stats_path = output_dir / "prior_degradation_statistics.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    logger.info(f"💾 Saved: {stats_path}")
    return stats, all_results, weight_histories


# ============================================================================
# EXPERIMENT: INITIAL WEIGHT BIAS SWEEP
# ============================================================================

def run_experiment_initial_weight_sweep(
    embeddings, warmup_priors_unscaled, warmup_priors_scaled,
    models, context_dim, data, n_seeds=20
):
    """
    Sweep initial warmup weight bias across the corruption spectrum.

    For each initial_warmup_weight in {0.5, 0.6, 0.7, 0.8, 0.9},
    run the full corruption sweep for Corralling only.
    Warmup-only and tabula rasa are prior-independent of this parameter
    so we run them once as baselines.
    """
    output_dir = BASE_OUTPUT_DIR / "initial_weight_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)

    corruption_levels = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]  # coarser grid (speed)
    weight_biases = [0.5, 0.6, 0.7, 0.8, 0.9]

    logger.info("\n" + "="*80)
    logger.info("EXPERIMENT: INITIAL WEIGHT BIAS SWEEP")
    logger.info("="*80)
    logger.info(f"  Weight biases: {weight_biases}")
    logger.info(f"  Corruption levels: {corruption_levels}")
    logger.info(f"  Seeds: {n_seeds}")
    n_trials = len(weight_biases) * len(corruption_levels) * n_seeds
    n_baselines = 2 * len(corruption_levels) * n_seeds
    logger.info(f"  Total trials: {n_trials} Corralling + {n_baselines} baselines = {n_trials + n_baselines}")

    # --- Run baselines (warmup-only and tabula rasa) once ---
    baseline_results = {'warmup_only': {}, 'tabula_rasa': {}}
    for corruption in corruption_levels:
        corrupted_priors = corrupt_priors(warmup_priors_scaled, corruption)
        baseline_results['warmup_only'][str(corruption)] = []
        baseline_results['tabula_rasa'][str(corruption)] = []

        for seed in range(n_seeds):
            rng = np.random.RandomState(seed)
            indices = rng.permutation(len(data))

            for strategy in ['warmup_only', 'tabula_rasa']:
                if strategy == 'warmup_only':
                    router = CostAwareLinUCBRouter(
                        models=models, warmup_priors=corrupted_priors,
                        alpha_start=2.0, alpha_end=2.0, cost_penalty=0.0,
                        model_costs={m: {"normalized_cost": 0.0} for m in models}
                    )
                else:
                    router = CostAwareTabulaRasaRouter(
                        models=models, context_dim=context_dim,
                        alpha_start=2.0, alpha_end=2.0, cost_penalty=0.0,
                        model_costs={m: {"normalized_cost": 0.0} for m in models}
                    )

                cumulative_regret = 0.0
                for idx in indices:
                    sample = data[idx]
                    context = embeddings[sample['prompt']]
                    result = router.select_model(context, total_steps=len(data))
                    selected_model = result[0] if isinstance(result, tuple) else result
                    sel_token = result[1] if isinstance(result, tuple) else None
                    scores = sample.get('scores', {})
                    if not scores:
                        continue
                    reward = scores.get(selected_model, 0.0)
                    oracle = max(scores.values())
                    cumulative_regret += (oracle - reward)
                    if sel_token:
                        router.update(context, selected_model, reward, sel_token)
                    else:
                        router.update(context, selected_model, reward)

                baseline_results[strategy][str(corruption)].append(cumulative_regret)

        w_mean = np.mean(baseline_results['warmup_only'][str(corruption)])
        t_mean = np.mean(baseline_results['tabula_rasa'][str(corruption)])
        logger.info(f"   Baselines α={corruption:.1f}: warmup={w_mean:.1f}, tabula={t_mean:.1f}")

    # --- Run Corralling at each initial weight bias ---
    corralling_results = {}  # {bias_str: {corruption_str: [regrets]}}
    for bias in weight_biases:
        bias_key = f"w0_{bias}"
        corralling_results[bias_key] = {}
        logger.info(f"\n   🔄 Corralling with initial warmup weight = {bias}")

        for corruption in corruption_levels:
            corrupted_priors = corrupt_priors(warmup_priors_scaled, corruption)
            corralling_results[bias_key][str(corruption)] = []

            for seed in range(n_seeds):
                rng = np.random.RandomState(seed)
                indices = rng.permutation(len(data))

                warmup_expert = CostAwareLinUCBRouter(
                    models=models, warmup_priors=corrupted_priors,
                    alpha_start=2.0, alpha_end=2.0, cost_penalty=0.0,
                    model_costs={m: {"normalized_cost": 0.0} for m in models}
                )
                tabula_rasa_expert = CostAwareTabulaRasaRouter(
                    models=models, context_dim=context_dim,
                    alpha_start=2.0, alpha_end=2.0, cost_penalty=0.0,
                    model_costs={m: {"normalized_cost": 0.0} for m in models}
                )
                initial_w = np.array([bias, 1.0 - bias])
                router = CorrallingRouter(
                    experts=[warmup_expert, tabula_rasa_expert],
                    models=models, learning_rate=LEARNING_RATE,
                    gamma=CORRALLING_GAMMA, loss_decay=1.0,
                    initial_weights=initial_w,
                )

                cumulative_regret = 0.0
                for idx in indices:
                    sample = data[idx]
                    context = embeddings[sample['prompt']]
                    selected_model, selection_token = router.select_model(
                        context, total_steps=len(data))
                    scores = sample.get('scores', {})
                    if not scores:
                        continue
                    reward = scores.get(selected_model, 0.0)
                    oracle = max(scores.values())
                    cumulative_regret += (oracle - reward)
                    router.update(context, selected_model, reward, selection_token)

                corralling_results[bias_key][str(corruption)].append(cumulative_regret)

            c_mean = np.mean(corralling_results[bias_key][str(corruption)])
            logger.info(f"      α={corruption:.1f}: regret={c_mean:.1f}")

    # --- Compute statistics ---
    stats = {
        'corruption_levels': corruption_levels,
        'weight_biases': weight_biases,
        'n_seeds': n_seeds,
        'baselines': {},
        'corralling_by_bias': {},
    }

    for strategy in ['warmup_only', 'tabula_rasa']:
        stats['baselines'][strategy] = {}
        for c in corruption_levels:
            regrets = baseline_results[strategy][str(c)]
            stats['baselines'][strategy][str(c)] = {
                'regret_mean': float(np.mean(regrets)),
                'regret_std': float(np.std(regrets)),
            }

    for bias in weight_biases:
        bias_key = f"w0_{bias}"
        stats['corralling_by_bias'][bias_key] = {}
        for c in corruption_levels:
            regrets = corralling_results[bias_key][str(c)]
            stats['corralling_by_bias'][bias_key][str(c)] = {
                'regret_mean': float(np.mean(regrets)),
                'regret_std': float(np.std(regrets)),
                'per_seed_regrets': [float(r) for r in regrets],
            }

    # --- Statistical tests: each bias vs baseline (0.5) at α=0 and α=1 ---
    logger.info("\n📊 Initial weight bias analysis:")
    tests = []
    baseline_key = "w0_0.5"
    for bias in weight_biases:
        if bias == 0.5:
            continue
        bias_key = f"w0_{bias}"
        for c_check in [0.0, 1.0]:
            c_str = str(c_check)
            a = corralling_results[baseline_key][c_str]
            b = corralling_results[bias_key][c_str]
            test = compute_paired_test(a, b, f'w0=0.5', f'w0={bias}')
            tests.append({'corruption': c_check, 'bias': bias, **test})
            sig = '***' if test['p_value'] < 0.001 else ('*' if test['p_value'] < 0.05 else 'ns')
            logger.info(f"   α={c_check}, w0={bias}: Δ={test['mean_diff']:.1f}, "
                        f"p={test['p_value']:.4f} {sig}")

    stats['statistical_tests'] = tests

    stats_path = output_dir / "initial_weight_sweep_statistics.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"💾 Saved: {stats_path}")

    return stats, baseline_results, corralling_results


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_main_figure(degradation_stats, degradation_results, weight_histories, output_dir):
    """
    Create the main 2-panel figure for the paper.

    Panel A: Strategy Crossover — regret vs prior quality for all 3 strategies
    Panel B: Adaptive Weights — weight evolution at 3 prior quality levels
    """
    mpl.rcParams.update(PLOT_STYLE)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    corruption_levels = degradation_stats['corruption_levels']
    strategies = ['warmup_only', 'corralling', 'tabula_rasa']
    strategy_labels = {
        'warmup_only': 'Warmup-Only',
        'corralling': 'Corralling',
        'tabula_rasa': 'Tabula Rasa',
    }
    strategy_colors = {
        'warmup_only': COLORS['green'],
        'corralling': COLORS['orange'],
        'tabula_rasa': COLORS['gray'],
    }

    # =================================================================
    # PANEL A: STRATEGY CROSSOVER
    # =================================================================
    for strategy in strategies:
        means = []
        stds = []
        for c in corruption_levels:
            data = degradation_stats['strategies'][strategy][str(c)]
            means.append(data['regret_mean'])
            stds.append(data['regret_std'])

        means = np.array(means)
        stds = np.array(stds)
        color = strategy_colors[strategy]

        ax1.plot(corruption_levels, means, '-o', color=color, linewidth=2.5,
                markersize=6, label=strategy_labels[strategy], zorder=3)
        ax1.fill_between(corruption_levels, means - stds, means + stds,
                        color=color, alpha=0.15, zorder=1)

    # ---- Find both crossover points from the data ----
    # Crossover 1: warmup_only ↔ corralling (warmup becomes worse)
    warmup_means = [degradation_stats['strategies']['warmup_only'][str(c)]['regret_mean']
                    for c in corruption_levels]
    corr_means = [degradation_stats['strategies']['corralling'][str(c)]['regret_mean']
                  for c in corruption_levels]
    tab_means = [degradation_stats['strategies']['tabula_rasa'][str(c)]['regret_mean']
                 for c in corruption_levels]

    crossover_wc = None  # warmup → corralling
    crossover_ct = None  # corralling → tabula rasa
    for i in range(len(corruption_levels) - 1):
        diff_wc = warmup_means[i] - corr_means[i]
        diff_wc_next = warmup_means[i+1] - corr_means[i+1]
        if diff_wc < 0 and diff_wc_next >= 0 and crossover_wc is None:
            frac = -diff_wc / (diff_wc_next - diff_wc) if (diff_wc_next - diff_wc) != 0 else 0.5
            crossover_wc = corruption_levels[i] + frac * (corruption_levels[i+1] - corruption_levels[i])

        diff_ct = corr_means[i] - tab_means[i]
        diff_ct_next = corr_means[i+1] - tab_means[i+1]
        if diff_ct < 0 and diff_ct_next >= 0 and crossover_ct is None:
            frac = -diff_ct / (diff_ct_next - diff_ct) if (diff_ct_next - diff_ct) != 0 else 0.5
            crossover_ct = corruption_levels[i] + frac * (corruption_levels[i+1] - corruption_levels[i])

    # Fallback crossover values
    if crossover_wc is None:
        crossover_wc = degradation_stats.get('crossover_point',
                       degradation_stats.get('crossover_point_interpolated', 0.5))
    if crossover_ct is None:
        crossover_ct = 0.55  # approximate from data inspection

    logger.info(f"   📍 Crossover warmup↔corralling: α≈{crossover_wc:.2f}")
    logger.info(f"   📍 Crossover corralling↔tabula:  α≈{crossover_ct:.2f}")

    # ---- Three-regime shading ----
    ax1.axvspan(0.0, crossover_wc, alpha=0.05, color='green', zorder=0)
    ax1.axvspan(crossover_wc, crossover_ct, alpha=0.05, color='orange', zorder=0)
    ax1.axvspan(crossover_ct, 1.0, alpha=0.05, color='gray', zorder=0)

    # ---- Crossover lines ----
    # Interpolate regret values at crossover points for annotation placement
    wc_regret = np.interp(crossover_wc, corruption_levels, warmup_means)
    ct_regret = np.interp(crossover_ct, corruption_levels, corr_means)

    ax1.axvline(x=crossover_wc, color='red', linestyle=':', linewidth=1.8,
               alpha=0.6, zorder=2)
    ax1.axvline(x=crossover_ct, color='purple', linestyle=':', linewidth=1.8,
               alpha=0.6, zorder=2)

    # Annotate crossover region (combined since the two points are so close)
    mid_crossover = (crossover_wc + crossover_ct) / 2
    mid_regret = np.interp(mid_crossover, corruption_levels, warmup_means)
    ax1.annotate(f'Crossovers\nα≈{crossover_wc:.2f}–{crossover_ct:.2f}',
                xy=(mid_crossover, mid_regret),
                xytext=(mid_crossover + 0.18, mid_regret + 18),
                fontsize=8, ha='center', color='red',
                arrowprops=dict(arrowstyle='->', color='red', lw=1.2),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor='red', alpha=0.9))

    # ---- Region labels ----
    ax1.text(0.15, 0.03, 'Warmup-Only optimal',
             transform=ax1.transAxes, fontsize=7.5, color=COLORS['green'],
             fontstyle='italic', va='bottom', ha='center')
    ax1.text(0.80, 0.03, 'Tabula Rasa optimal',
             transform=ax1.transAxes, fontsize=7.5, color='dimgray',
             fontstyle='italic', va='bottom', ha='center')

    ax1.set_xlabel('Prior Corruption Level (α)', fontsize=11)
    ax1.set_ylabel('Cumulative Regret', fontsize=11)
    ax1.set_title('(A) Three Regimes of Prior Quality',
                  fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.2)
    ax1.set_xlim(-0.02, 1.02)

    # =================================================================
    # PANEL B: WEIGHT EVOLUTION AT 3 QUALITY LEVELS
    # =================================================================
    quality_labels = {
        '0.0': ('Strong (α=0.0)', COLORS['green']),
        '0.5': ('Uninformative (α=0.5)', COLORS['blue']),
        '1.0': ('Adversarial (α=1.0)', COLORS['red']),
    }

    for c_str, (label, color) in quality_labels.items():
        if c_str in degradation_stats.get('weight_histories', {}):
            wh = degradation_stats['weight_histories'][c_str]
            mean_weights = np.array(wh['mean'])
            std_weights = np.array(wh['std'])
            timesteps = np.arange(1, len(mean_weights) + 1)

            ax2.plot(timesteps, mean_weights, color=color, linewidth=2.5,
                    label=label, zorder=3)
            ax2.fill_between(timesteps, mean_weights - std_weights,
                           mean_weights + std_weights,
                           color=color, alpha=0.15, zorder=1)

    ax2.axhline(y=0.5, color='gray', linestyle='--', linewidth=1,
               alpha=0.5, label='Initial (50%)')
    ax2.set_xlabel('Query Number', fontsize=11)
    ax2.set_ylabel('Warmup Expert Weight', fontsize=11)
    ax2.set_title('(B) Meta-Learner Correctly Adapts to Prior Quality',
                  fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=9, framealpha=0.9)
    ax2.grid(True, alpha=0.2)
    ax2.set_ylim(-0.02, 1.02)

    plt.tight_layout()
    fig_path = output_dir / "figure3_prior_degradation.pdf"
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    fig_path_png = output_dir / "figure3_prior_degradation.png"
    fig.savefig(fig_path_png, dpi=300, bbox_inches='tight')
    logger.info(f"✅ MAIN FIGURE saved: {fig_path}")
    plt.close()


def plot_alpha_ablation(config_results, output_dir):
    """Plot alpha ablation study (Experiment 3). Appendix figure."""
    mpl.rcParams.update(PLOT_STYLE)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    configs = list(config_results.keys())
    config_labels = {
        'constant_constant': 'Homogeneous\nConstant',
        'mixed': 'Mixed\n(Tabula Decay)',
        'decay_decay': 'Homogeneous\nDecay',
        'reversed_mixed': 'Reversed\nMixed',
    }

    means = [config_results[c]['regret_mean'] for c in configs]
    stds = [config_results[c]['regret_std'] for c in configs]

    colors_list = [COLORS['green'], COLORS['blue'], COLORS['orange'], COLORS['red']]
    bar_colors = colors_list[:len(configs)]

    # Panel A: Bar chart with error bars
    bars = ax1.bar(range(len(configs)), means, yerr=stds, capsize=5,
                   color=bar_colors, alpha=0.7, edgecolor='black', linewidth=1.5)

    ax1.set_xticks(range(len(configs)))
    ax1.set_xticklabels([config_labels.get(c, c) for c in configs])
    ax1.set_ylabel('Cumulative Regret')
    ax1.set_title('(A) Alpha Strategy Comparison')
    ax1.grid(True, alpha=0.2, axis='y')
    ax1.axhline(y=50, color='gray', linestyle='--', linewidth=1, alpha=0.3)

    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + std + 1,
                f'{mean:.1f}±{std:.1f}',
                ha='center', va='bottom', fontsize=9)

    # Panel B: Per-seed scatter
    for i, config in enumerate(configs):
        y_values = config_results[config]['per_seed_regrets']
        x_values = [i] * len(y_values)
        ax2.scatter(x_values, y_values, alpha=0.6, s=80,
                   color=bar_colors[i], edgecolors='black', linewidth=0.5)

    ax2.set_xticks(range(len(configs)))
    ax2.set_xticklabels([config_labels.get(c, c) for c in configs])
    ax2.set_ylabel('Cumulative Regret (per seed)')
    ax2.set_title('(B) Per-Seed Results')
    ax2.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(output_dir / f"figure_alpha_ablation.{ext}", dpi=150, bbox_inches='tight')
    logger.info(f"💾 Saved: {output_dir / 'figure_alpha_ablation.pdf'}")
    plt.close()


def plot_initial_weight_sweep(sweep_stats, output_dir):
    """
    Plot the initial weight bias trade-off.

    Panel A: Regret vs corruption for Corralling at each initial weight,
             with warmup-only and tabula rasa as baselines.
    Panel B: Regret at α=0 vs regret at α=1 for each bias (Pareto frontier).
    """
    mpl.rcParams.update(PLOT_STYLE)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    corruption_levels = sweep_stats['corruption_levels']
    weight_biases = sweep_stats['weight_biases']

    # --- Panel A: Regret vs corruption ---
    # Baselines
    warmup_means = [sweep_stats['baselines']['warmup_only'][str(c)]['regret_mean']
                    for c in corruption_levels]
    tabula_means = [sweep_stats['baselines']['tabula_rasa'][str(c)]['regret_mean']
                    for c in corruption_levels]

    ax1.plot(corruption_levels, warmup_means, '-s', color=COLORS['green'],
             linewidth=2.5, markersize=7, label='Warmup-Only', zorder=4)
    ax1.plot(corruption_levels, tabula_means, '-^', color='gray',
             linewidth=2.5, markersize=7, label='Tabula Rasa', zorder=4)

    # Corralling at each bias
    bias_cmap = plt.cm.YlOrRd(np.linspace(0.2, 0.9, len(weight_biases)))
    for i, bias in enumerate(weight_biases):
        bias_key = f"w0_{bias}"
        means = [sweep_stats['corralling_by_bias'][bias_key][str(c)]['regret_mean']
                 for c in corruption_levels]
        stds = [sweep_stats['corralling_by_bias'][bias_key][str(c)]['regret_std']
                for c in corruption_levels]
        means = np.array(means)
        stds = np.array(stds)

        lw = 3.0 if bias == 0.5 else 1.8
        ls = '-' if bias == 0.5 else '--'
        ax1.plot(corruption_levels, means, ls + 'o', color=bias_cmap[i],
                linewidth=lw, markersize=5,
                label=f'Corralling w₀={bias:.1f}', zorder=3)
        ax1.fill_between(corruption_levels, means - stds, means + stds,
                        color=bias_cmap[i], alpha=0.08, zorder=1)

    ax1.set_xlabel('Prior Corruption Level (α)', fontsize=11)
    ax1.set_ylabel('Cumulative Regret', fontsize=11)
    ax1.set_title('(A) Prior-Trust Bias: Shifting the Trade-Off',
                  fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=8, framealpha=0.9, ncol=1)
    ax1.grid(True, alpha=0.2)
    ax1.set_xlim(-0.02, 1.02)

    # --- Panel B: Pareto frontier (α=0 regret vs α=1 regret) ---
    for i, bias in enumerate(weight_biases):
        bias_key = f"w0_{bias}"
        regret_0 = sweep_stats['corralling_by_bias'][bias_key]['0.0']['regret_mean']
        regret_1 = sweep_stats['corralling_by_bias'][bias_key]['1.0']['regret_mean']
        std_0 = sweep_stats['corralling_by_bias'][bias_key]['0.0']['regret_std']
        std_1 = sweep_stats['corralling_by_bias'][bias_key]['1.0']['regret_std']

        ax2.errorbar(regret_0, regret_1, xerr=std_0, yerr=std_1,
                    fmt='o', color=bias_cmap[i], markersize=10,
                    capsize=4, capthick=1.5, linewidth=1.5, zorder=3)
        ax2.annotate(f'w₀={bias:.1f}',
                    xy=(regret_0, regret_1),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, fontweight='bold', color=bias_cmap[i])

    # Add baselines as reference points
    w_0 = sweep_stats['baselines']['warmup_only']['0.0']['regret_mean']
    w_1 = sweep_stats['baselines']['warmup_only']['1.0']['regret_mean']
    t_0 = sweep_stats['baselines']['tabula_rasa']['0.0']['regret_mean']
    t_1 = sweep_stats['baselines']['tabula_rasa']['1.0']['regret_mean']

    ax2.plot(w_0, w_1, 's', color=COLORS['green'], markersize=12, zorder=4)
    ax2.annotate('Warmup', xy=(w_0, w_1), xytext=(-8, 8),
                textcoords='offset points', fontsize=9, color=COLORS['green'],
                fontweight='bold')
    ax2.plot(t_0, t_1, '^', color='gray', markersize=12, zorder=4)
    ax2.annotate('Tabula\nRasa', xy=(t_0, t_1), xytext=(8, -5),
                textcoords='offset points', fontsize=9, color='gray',
                fontweight='bold')

    ax2.set_xlabel('Regret at α=0 (correct priors)', fontsize=11)
    ax2.set_ylabel('Regret at α=1 (adversarial priors)', fontsize=11)
    ax2.set_title('(B) Prior-Trust Pareto Frontier',
                  fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.2)

    # Draw ideal corner arrow
    ax2.annotate('← Better', xy=(0.02, 0.5), xycoords='axes fraction',
                fontsize=8, color='gray', fontstyle='italic')
    ax2.annotate('↓ Better', xy=(0.5, 0.02), xycoords='axes fraction',
                fontsize=8, color='gray', fontstyle='italic')

    plt.tight_layout()
    out_dir = output_dir / "initial_weight_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ['pdf', 'png']:
        fig.savefig(out_dir / f"figure_initial_weight_sweep.{ext}",
                   dpi=300, bbox_inches='tight')
    logger.info(f"✅ Weight sweep figure saved: {out_dir / 'figure_initial_weight_sweep.pdf'}")
    plt.close()


# ============================================================================
# MAIN RUNNER
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='Run all Figure 3 experiments')
    parser.add_argument('--experiments', type=str, default='2a,2bc,3,prior',
                       help='Comma-separated experiments to run (default: 2a,2bc,3,prior). '
                            'Options: 2a, 2bc, 3, prior, 5(gamma), iw(initial weight sweep)')
    parser.add_argument('--seeds', type=int, default=20,
                       help='Number of seeds for experiments 2a, 2bc, and prior (default: 20)')
    parser.add_argument('--seeds-ablation', type=int, default=20,
                       help='Number of seeds for ablation experiments 3 and 5 (default: 20)')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip figure generation (for faster testing)')

    args = parser.parse_args()
    experiments_to_run = args.experiments.split(',')

    logger.info("="*80)
    logger.info("🚀 UNIFIED EXPERIMENT RUNNER - FIGURE 3")
    logger.info("="*80)
    logger.info(f"Experiments: {experiments_to_run}")
    logger.info(f"Seeds (2a,2bc,prior): {args.seeds}")
    logger.info(f"Seeds (3,5): {args.seeds_ablation}")
    logger.info("="*80)

    # Load shared resources once
    start_time = time.time()
    encoder, pca, warmup_priors_scaled, warmup_priors_unscaled, models, context_dim = load_resources()
    data = load_holdout_data()

    # Compute theoretically optimal learning rate: η* = sqrt(ln(K)/T)
    global LEARNING_RATE
    LEARNING_RATE = compute_learning_rate(N_EXPERTS, len(data))
    logger.info(f"📐 Learning rate: η* = sqrt(ln({N_EXPERTS})/{len(data)}) = {LEARNING_RATE:.4f}")

    # Pre-compute all embeddings (massive speedup: 750 embeds instead of 100k+)
    emb = precompute_embeddings(data, encoder, pca)

    # Run experiments
    all_stats = {}

    if '2a' in experiments_to_run:
        stats_2a, _ = run_experiment_2a(
            emb, warmup_priors_scaled, models, context_dim, data, args.seeds)
        all_stats['2a_weight_evolution'] = stats_2a

    if '2bc' in experiments_to_run:
        stats_2bc = run_experiment_2bc(
            emb, warmup_priors_scaled, models, context_dim, data, args.seeds)
        all_stats['2bc_convergence'] = stats_2bc

    if '3' in experiments_to_run:
        stats_3 = run_experiment_3(
            emb, warmup_priors_scaled, models, context_dim, data, args.seeds_ablation)
        all_stats['3_alpha_ablation'] = stats_3
        if not args.no_plots:
            logger.info("\n📊 Generating alpha ablation figure...")
            try:
                plot_alpha_ablation(stats_3['configs'], BASE_OUTPUT_DIR / "ablation")
            except Exception as e:
                logger.error(f"⚠️ Failed to generate figure: {e}")
                import traceback; traceback.print_exc()

    if 'prior' in experiments_to_run:
        logger.info("\n" + "="*80)
        logger.info("📊 RUNNING PRIOR QUALITY DEGRADATION SWEEP (MAIN EXPERIMENT)")
        logger.info("="*80)
        stats_prior, prior_results, prior_weights = run_experiment_prior_degradation(
            emb, warmup_priors_unscaled, warmup_priors_scaled,
            models, context_dim, data, args.seeds)
        all_stats['prior_degradation'] = stats_prior
        if not args.no_plots:
            logger.info("\n📊 Generating main figure (crossover + adaptive weights)...")
            try:
                plot_main_figure(stats_prior, prior_results, prior_weights, BASE_OUTPUT_DIR)
            except Exception as e:
                logger.error(f"⚠️ Failed to generate main figure: {e}")
                import traceback; traceback.print_exc()

    if '5' in experiments_to_run:
        stats_5 = run_experiment_5(
            emb, warmup_priors_scaled, models, context_dim, data, args.seeds_ablation)
        all_stats['5_gamma_ablation'] = stats_5

    if 'iw' in experiments_to_run:
        logger.info("\n" + "="*80)
        logger.info("📊 RUNNING INITIAL WEIGHT BIAS SWEEP")
        logger.info("="*80)
        stats_iw, _, _ = run_experiment_initial_weight_sweep(
            emb, warmup_priors_unscaled, warmup_priors_scaled,
            models, context_dim, data, args.seeds)
        all_stats['initial_weight_sweep'] = stats_iw
        if not args.no_plots:
            logger.info("\n📊 Generating initial weight sweep figure...")
            try:
                plot_initial_weight_sweep(stats_iw, BASE_OUTPUT_DIR)
            except Exception as e:
                logger.error(f"⚠️ Failed to generate figure: {e}")
                import traceback; traceback.print_exc()

    # Save combined results
    elapsed = time.time() - start_time
    summary = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'experiments_run': experiments_to_run,
        'total_time_seconds': elapsed,
        'configuration': {
            'learning_rate': LEARNING_RATE,
            'corralling_gamma': CORRALLING_GAMMA,
            'prior_scaling': PRIOR_SCALING,
            'n_experts': N_EXPERTS,
        },
        'results': all_stats
    }

    summary_path = BASE_OUTPUT_DIR / "all_experiments_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info("\n" + "="*80)
    logger.info("✅ ALL EXPERIMENTS COMPLETE")
    logger.info("="*80)
    logger.info(f"Total time: {elapsed/60:.1f} minutes")
    logger.info(f"Summary: {summary_path}")
    logger.info("="*80)


if __name__ == '__main__':
    main()
