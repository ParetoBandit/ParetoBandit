"""
Metrics computation for BanditGPT experiments.

Provides standard metric calculations used across all experiments.
"""

import numpy as np
from typing import List, Union, Tuple


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


# Placeholder for future implementations
def calculate_feature_lift(baseline_regret, ablated_regret):
    """
    [PLACEHOLDER] Calculate performance lift from adding a feature.
    
    Args:
        baseline_regret: regret with feature
        ablated_regret: regret without feature
    
    Returns:
        lift percentage
    """
    # TODO: Implement
    pass


def calculate_pruning_false_positive_rate(pruned_models, true_dominated):
    """
    [PLACEHOLDER] Calculate false positive rate in pruning.
    
    Args:
        pruned_models: models that were pruned
        true_dominated: models that are truly dominated
    
    Returns:
        FPR (false positives / total negatives)
    """
    # TODO: Implement
    pass


if __name__ == "__main__":
    # Test metrics
    selected = np.array([0.5, 0.7, 0.6, 0.8])
    oracle = np.array([0.9, 0.9, 0.9, 0.9])
    random_baseline = np.array([0.4, 0.5, 0.4, 0.5])
    
    cum_regret = calculate_cumulative_regret(selected, oracle)
    print(f"Cumulative regret: {cum_regret}")
    print(f"Simple regret: {calculate_simple_regret(selected, oracle):.3f}")
    print(f"Normalized regret: {calculate_normalized_regret(selected, oracle, random_baseline):.3f}")
    
    # Test CI
    data = np.random.randn(100)
    lower, upper = bootstrap_ci(data)
    print(f"95% CI: [{lower:.3f}, {upper:.3f}]")
    
    print("✓ Metrics working correctly!")
