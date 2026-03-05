"""
Metrics computation for BanditGPT experiments.

Provides standard metric calculations used across all experiments,
including regret metrics, router comparison metrics following
conventions from LLMRouterBench (2026), and paired-test statistics
for ablation comparisons.
"""

import numpy as np
from typing import List, Sequence, Union, Tuple


def calculate_cumulative_regret(
    selected_rewards: Union[List[float], np.ndarray],
    oracle_rewards: Union[List[float], np.ndarray]
) -> np.ndarray:
    """
    Calculate cumulative regret over time.
    
    Regret at timestep t = sum_{i=1}^{t} (oracle_reward_i - selected_reward_i)
    
    Args:
        selected_rewards: rewards actually received by the bandit
        oracle_rewards: rewards that would have been received by oracle
    
    Returns:
        cumulative regret at each timestep
    
    Example:
        >>> selected = [0.5, 0.7, 0.6]
        >>> oracle = [0.9, 0.9, 0.9]
        >>> calculate_cumulative_regret(selected, oracle)
        array([0.4, 0.6, 0.9])
    """
    selected = np.array(selected_rewards)
    oracle = np.array(oracle_rewards)
    
    instantaneous_regret = oracle - selected
    cumulative_regret = np.cumsum(instantaneous_regret)
    
    return cumulative_regret


def calculate_simple_regret(
    selected_rewards: Union[List[float], np.ndarray],
    oracle_rewards: Union[List[float], np.ndarray]
) -> float:
    """
    Calculate simple regret (final timestep regret / T).
    
    Args:
        selected_rewards: rewards received
        oracle_rewards: oracle rewards
    
    Returns:
        simple regret (average per-step regret)
    """
    cumulative = calculate_cumulative_regret(selected_rewards, oracle_rewards)
    return cumulative[-1] / len(selected_rewards)


def calculate_convergence_mse(
    policy_weights: np.ndarray,
    oracle_weights: np.ndarray
) -> float:
    """
    Calculate MSE between policy weights and oracle weights.
    
    Used to measure how quickly the policy converges to optimal.
    
    Args:
        policy_weights: learned weights (shape: [d])
        oracle_weights: true optimal weights (shape: [d])
    
    Returns:
        mean squared error
    """
    return np.mean((policy_weights - oracle_weights) ** 2)


def bootstrap_ci(
    data: np.ndarray,
    confidence: float = 0.95,
    n_bootstrap: int = 1000
) -> Tuple[float, float]:
    """
    Calculate bootstrap confidence interval.
    
    Args:
        data: observations (shape: [n_samples])
        confidence: confidence level (default: 0.95)
        n_bootstrap: number of bootstrap samples
    
    Returns:
        (lower_bound, upper_bound)
    """
    n = len(data)
    bootstrap_means = []
    
    rng = np.random.RandomState(42)  # Fixed seed for reproducibility
    
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))
    
    bootstrap_means = np.array(bootstrap_means)
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    
    return lower, upper


def calculate_normalized_regret(
    selected_rewards: Union[List[float], np.ndarray],
    oracle_rewards: Union[List[float], np.ndarray],
    random_rewards: Union[List[float], np.ndarray]
) -> float:
    """
    Calculate normalized regret: (R_policy - R_oracle) / (R_random - R_oracle).
    
    Result in [0, 1] where:
    - 0 = optimal (oracle performance)
    - 1 = random (worst performance)
    
    Args:
        selected_rewards: policy rewards
        oracle_rewards: oracle rewards
        random_rewards: random baseline rewards
    
    Returns:
        normalized regret
    """
    selected = np.array(selected_rewards)
    oracle = np.array(oracle_rewards)
    random = np.array(random_rewards)
    
    policy_cumulative = np.sum(oracle - selected)
    random_cumulative = np.sum(oracle - random)
    
    if random_cumulative == 0:
        return 0.0  # Edge case: all strategies are equally good
    
    return policy_cumulative / random_cumulative


# =========================================================================
# Router comparison metrics (LLMRouterBench conventions)
# =========================================================================


def perfgain(
    router_reward: float,
    baseline_reward: float,
) -> float:
    """Performance gain of a router over a baseline at matched cost.

    Positive means the router achieves higher reward than the baseline
    at the same cost level (isocost comparison).

    Ref: LLMRouterBench (2026), "PerfGain" metric.

    Args:
        router_reward: Router's interpolated reward at the baseline's
            operating cost.
        baseline_reward: Baseline's reward at its operating cost.

    Returns:
        Absolute reward difference (router - baseline).
    """
    return router_reward - baseline_reward


def costsave(
    router_cost: float,
    baseline_cost: float,
) -> Tuple[float, float]:
    """Cost savings of a router over a baseline at matched quality.

    Positive values mean the router achieves the same quality at
    lower cost (iso-quality comparison).

    Ref: LLMRouterBench (2026), "CostSave" metric.

    Args:
        router_cost: Router's interpolated cost at the baseline's
            reward level.
        baseline_cost: Baseline's cost at its operating point.

    Returns:
        ``(absolute_saving, percentage_saving)`` where percentage is
        relative to the baseline cost.  Both are positive when the
        router is cheaper.
    """
    absolute = baseline_cost - router_cost
    pct = (absolute / baseline_cost * 100.0) if baseline_cost != 0 else 0.0
    return absolute, pct


def gap_at_oracle(
    oracle_reward: float,
    method_reward: float,
) -> Tuple[float, float]:
    """Remaining reward gap between a method and the oracle.

    Measures how much headroom remains between the method's
    performance and instance-wise optimal routing.

    Ref: LLMRouterBench (2026), "Gap@Oracle" metric.

    Args:
        oracle_reward: Per-instance optimal (oracle) mean reward.
        method_reward: Method's mean reward.

    Returns:
        ``(absolute_gap, relative_gap_pct)`` where relative gap is
        ``(oracle - method) / oracle * 100``.  Both are non-negative
        when the method under-performs the oracle.
    """
    absolute = oracle_reward - method_reward
    pct = (absolute / oracle_reward * 100.0) if oracle_reward != 0 else 0.0
    return absolute, pct


# =========================================================================
# Paired-test statistics (used by ablation experiments)
# =========================================================================


def holm_bonferroni(p_values: Sequence[float]) -> List[float]:
    """Holm-Bonferroni step-down correction for multiple comparisons.

    Args:
        p_values: Uncorrected p-values (one per hypothesis).

    Returns:
        Adjusted p-values in the same order as the input, each
        clamped to [0, 1].
    """
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * n
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        corrected = p * (n - rank)
        running_max = max(running_max, corrected)
        adjusted[orig_idx] = min(running_max, 1.0)
    return adjusted


def cohens_d_paired(
    a: Sequence[float],
    b: Sequence[float],
) -> float:
    """Cohen's d for paired samples.

    Computed as ``mean(a - b) / std(a - b, ddof=1)``.

    Args:
        a: Observations under condition A.
        b: Observations under condition B (same length, paired).

    Returns:
        Effect size (positive when A > B).
    """
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    sd = float(np.std(diff, ddof=1))
    if sd < 1e-15:
        return 0.0
    return float(np.mean(diff) / sd)
