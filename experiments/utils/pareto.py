"""
Pareto frontier analysis utilities for BanditGPT experiments.

Provides shared functions for computing Pareto hulls, AUC metrics,
interpolation along frontiers, and bootstrap confidence intervals
for Pareto AUC differences.  These are used across multiple experiment
scripts (03_figure, 04_figure, appendix experiments).
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def pareto_hull(
    costs: List[float],
    rewards: List[float],
) -> Tuple[List[float], List[float]]:
    """Compute the monotone upper envelope sorted by ascending cost.

    Filters the (cost, reward) point cloud down to the Pareto-optimal
    frontier: for each point retained, no other point has both lower
    cost and higher reward.

    Args:
        costs: Per-configuration mean costs.
        rewards: Per-configuration mean rewards.

    Returns:
        ``(hull_costs, hull_rewards)`` lists sorted by ascending cost,
        with strictly increasing reward.
    """
    pairs = sorted(zip(costs, rewards), key=lambda x: (x[0], -x[1]))
    hull_c: List[float] = []
    hull_r: List[float] = []
    best_r = -np.inf
    for c, r in pairs:
        if r > best_r:
            hull_c.append(c)
            hull_r.append(r)
            best_r = r
    return hull_c, hull_r


def pareto_auc(
    costs: List[float],
    rewards: List[float],
    cost_lo: float,
    cost_hi: float,
) -> float:
    """Area under the Pareto frontier (trapezoidal) over ``[cost_lo, cost_hi]``.

    Normalised by the cost range so the result is in reward units.
    The hull is linearly interpolated at both boundaries so the
    integral always covers the full ``[cost_lo, cost_hi]`` span,
    regardless of whether the hull extends past, falls short, or
    exactly matches the boundaries.

    Returns 0 if the hull has no overlap with the range.

    Args:
        costs: Per-configuration mean costs (need not be sorted).
        rewards: Per-configuration mean rewards.
        cost_lo: Lower bound of the integration range.
        cost_hi: Upper bound of the integration range.

    Returns:
        Normalised AUC in reward units.
    """
    hull_c, hull_r = pareto_hull(costs, rewards)
    if len(hull_c) < 1:
        return 0.0
    hc = np.array(hull_c, dtype=float)
    hr = np.array(hull_r, dtype=float)
    if hc[-1] < cost_lo or hc[0] > cost_hi:
        return 0.0

    r_lo = float(np.interp(cost_lo, hc, hr))
    r_hi = float(np.interp(cost_hi, hc, hr))

    interior_mask = (hc > cost_lo) & (hc < cost_hi)
    clip_c = [cost_lo] + hc[interior_mask].tolist() + [cost_hi]
    clip_r = [r_lo] + hr[interior_mask].tolist() + [r_hi]

    cc = np.array(clip_c)
    cr = np.array(clip_r)
    if len(cc) < 2:
        return float(cr.mean())
    return float(np.trapezoid(cr, cc) / (cost_hi - cost_lo))


def interpolate_pareto_reward(
    hull_c: List[float],
    hull_r: List[float],
    target_cost: float,
) -> Optional[float]:
    """Linearly interpolate reward on the Pareto hull at a target cost.

    The hull must be sorted by ascending cost (as returned by
    :func:`pareto_hull`).

    Args:
        hull_c: Pareto hull costs (ascending).
        hull_r: Pareto hull rewards (corresponding).
        target_cost: The cost at which to interpolate.

    Returns:
        Interpolated reward, or ``None`` if *target_cost* is outside
        the hull's cost range.
    """
    if not hull_c or target_cost < hull_c[0] or target_cost > hull_c[-1]:
        return None
    return float(np.interp(target_cost, hull_c, hull_r))


def interpolate_pareto_cost(
    hull_c: List[float],
    hull_r: List[float],
    target_reward: float,
) -> Optional[float]:
    """Linearly interpolate cost on the Pareto hull at a target reward.

    This is the *inverse* of :func:`interpolate_pareto_reward`: given a
    desired quality level, find the cheapest cost that achieves it on
    the Pareto frontier.  Used for CostSave computation.

    The hull must be sorted by ascending cost (as returned by
    :func:`pareto_hull`).  Internally re-sorts by ascending reward
    before interpolating.

    Args:
        hull_c: Pareto hull costs (ascending by cost).
        hull_r: Pareto hull rewards (corresponding).
        target_reward: The reward level at which to interpolate cost.

    Returns:
        Interpolated cost, or ``None`` if *target_reward* is outside
        the hull's reward range.
    """
    if not hull_c:
        return None
    pairs = sorted(zip(hull_r, hull_c))
    sorted_r = [p[0] for p in pairs]
    sorted_c = [p[1] for p in pairs]
    if target_reward < sorted_r[0] or target_reward > sorted_r[-1]:
        return None
    return float(np.interp(target_reward, sorted_r, sorted_c))


def find_closest_pareto_point(
    pareto: List[Dict[str, Any]],
    target_cost: float,
    cost_key: str = "mean_cost",
) -> Dict[str, Any]:
    """Find the Pareto sweep point whose cost is closest to *target_cost*.

    Args:
        pareto: List of sweep-result dicts.
        target_cost: Target cost to match.
        cost_key: Dict key for the cost field.

    Returns:
        The sweep dict with the smallest ``|cost - target_cost|``.
    """
    return min(pareto, key=lambda p: abs(p[cost_key] - target_cost))


def dev_pareto_indices(
    sweep_results: List[Dict],
    dev_cost_key: str = "dev_mean_cost",
    dev_reward_key: str = "dev_mean_reward",
) -> List[int]:
    """Identify sweep indices that lie on the dev-set Pareto frontier.

    The hull is built strictly from dev-set metrics (dev_cost,
    dev_reward) -- no holdout information is used.  This selects the
    hyperparameters a practitioner would consider optimal based
    solely on historical (dev) data.

    Args:
        sweep_results: List of dicts, each with dev-set cost/reward keys.
        dev_cost_key: Dict key for dev-set cost.
        dev_reward_key: Dict key for dev-set reward.

    Returns:
        Sorted list of indices into *sweep_results* that form the
        dev-optimal Pareto frontier.
    """
    n = len(sweep_results)
    pairs = [
        (sweep_results[i][dev_cost_key], sweep_results[i][dev_reward_key], i)
        for i in range(n)
    ]
    pairs.sort(key=lambda x: (x[0], -x[1]))
    hull_idx: List[int] = []
    best_r = -np.inf
    for _, r, idx in pairs:
        if r > best_r:
            hull_idx.append(idx)
            best_r = r
    return hull_idx


def dev_selected_pareto_auc(
    sweep_results: List[Dict],
    cost_lo: float,
    cost_hi: float,
    *,
    dev_cost_key: str = "dev_mean_cost",
    dev_reward_key: str = "dev_mean_reward",
    holdout_cost_key: str = "mean_cost",
    holdout_reward_key: str = "mean_reward",
) -> Tuple[float, List[float], List[float], List[int]]:
    """Pareto AUC of the dev-selected deployable frontier.

    **Hyperparameter selection is strictly partitioned from holdout
    evaluation.** The procedure:

    1. Build the Pareto hull from ``(dev_cost, dev_reward)`` to
       identify which hyperparameter settings a practitioner would
       consider optimal using *only* dev-set information.
    2. For those dev-optimal settings, extract the corresponding
       ``(holdout_cost, holdout_reward)`` -- the performance the
       practitioner would actually observe after deployment.
    3. Take the Pareto hull of the resulting holdout points (since
       dev-optimal points may not be monotone on the holdout set)
       and compute AUC.

    Args:
        sweep_results: List of dicts from ``run_pareto_sweep``,
            each with dev and holdout metrics.
        cost_lo: Lower bound of the shared cost range for AUC.
        cost_hi: Upper bound of the shared cost range for AUC.
        dev_cost_key: Dict key for dev-set cost.
        dev_reward_key: Dict key for dev-set reward.
        holdout_cost_key: Dict key for holdout cost.
        holdout_reward_key: Dict key for holdout reward.

    Returns:
        ``(auc, hull_holdout_costs, hull_holdout_rewards, dev_hull_indices)``.
    """
    dev_idx = dev_pareto_indices(
        sweep_results, dev_cost_key, dev_reward_key,
    )
    holdout_costs = [sweep_results[i][holdout_cost_key] for i in dev_idx]
    holdout_rewards = [sweep_results[i][holdout_reward_key] for i in dev_idx]
    hull_c, hull_r = pareto_hull(holdout_costs, holdout_rewards)
    auc = pareto_auc(hull_c, hull_r, cost_lo, cost_hi)
    return auc, hull_c, hull_r, dev_idx


def bootstrap_pareto_auc_difference(
    bg_pp_rewards: List[np.ndarray],
    bg_pp_costs: List[np.ndarray],
    bl_pp_rewards: List[np.ndarray],
    bl_pp_costs: List[np.ndarray],
    cost_lo: float,
    cost_hi: float,
    n_holdout: int,
    *,
    n_bootstrap: int = 1_000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Paired bootstrap CI for the difference in dev-selected Pareto AUC.

    **Caller must pre-filter to dev-Pareto-optimal points.** This
    function receives per-prompt reward *and cost* arrays for the
    hyperparameters identified as optimal on the dev set.  Both axes
    of the Pareto frontier are resampled jointly, correctly capturing
    variance in both reward and cost.

    Args:
        bg_pp_rewards: Per-prompt holdout rewards for each dev-optimal
            BanditGPT setting (list of 1-D arrays, length n_holdout).
        bg_pp_costs: Per-prompt holdout costs (same structure).
        bl_pp_rewards: Per-prompt holdout rewards for each dev-optimal
            baseline setting.
        bl_pp_costs: Per-prompt holdout costs (same structure).
        cost_lo: Lower bound of shared cost range.
        cost_hi: Upper bound of shared cost range.
        n_holdout: Number of holdout prompts.
        n_bootstrap: Number of bootstrap resamples.
        seed: RNG seed for reproducibility.

    Returns:
        Dict with observed AUC difference, 95% CI, and bootstrap
        p-value.
    """
    rng = np.random.default_rng(seed)

    def _auc_for_resample(
        idx: np.ndarray,
        pp_cost_arrays: List[np.ndarray],
        pp_reward_arrays: List[np.ndarray],
    ) -> float:
        costs = [float(np.mean(c[idx])) for c in pp_cost_arrays]
        rewards = [float(np.mean(r[idx])) for r in pp_reward_arrays]
        hull_c, hull_r = pareto_hull(costs, rewards)
        return pareto_auc(hull_c, hull_r, cost_lo, cost_hi)

    all_idx = np.arange(n_holdout)
    obs_bg = _auc_for_resample(all_idx, bg_pp_costs, bg_pp_rewards)
    obs_bl = _auc_for_resample(all_idx, bl_pp_costs, bl_pp_rewards)
    obs_diff = obs_bg - obs_bl

    boot_diffs: List[float] = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n_holdout, size=n_holdout, replace=True)
        bg_auc = _auc_for_resample(idx, bg_pp_costs, bg_pp_rewards)
        bl_auc = _auc_for_resample(idx, bl_pp_costs, bl_pp_rewards)
        boot_diffs.append(bg_auc - bl_auc)

    boot_arr = np.array(boot_diffs)
    centred = boot_arr - obs_diff
    p_value = float(np.mean(np.abs(centred) >= np.abs(obs_diff)))

    return {
        "observed_diff": obs_diff,
        "bg_auc": obs_bg,
        "baseline_auc": obs_bl,
        "ci_95_lower": float(np.percentile(boot_arr, 2.5)),
        "ci_95_upper": float(np.percentile(boot_arr, 97.5)),
        "p_value": p_value,
        "n_bootstrap": n_bootstrap,
        "n_bg_dev_optimal": len(bg_pp_rewards),
        "n_bl_dev_optimal": len(bl_pp_rewards),
        "note": (
            "Paired bootstrap over holdout prompts.  Dev-Pareto-optimal "
            "indices fixed before bootstrapping.  Both costs and rewards "
            "are resampled jointly."
        ),
    }


def extract_dev_optimal_per_prompt(
    sweep: List[Dict],
    dev_idx: List[int],
    per_prompt_reward_map: Dict,
    per_prompt_cost_map: Dict,
    hparam_key: str = "lambda",
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Extract seed-averaged per-prompt arrays for dev-optimal sweep points.

    Args:
        sweep: Full sweep results.
        dev_idx: Indices of dev-Pareto-optimal points.
        per_prompt_reward_map: Maps hyperparameter value to per-prompt
            reward array (shape ``(n_seeds, n_holdout)`` or
            ``(n_holdout,)``).
        per_prompt_cost_map: Maps hyperparameter value to per-prompt
            cost array (same shapes as reward map).
        hparam_key: Key to extract the hyperparameter value from each
            sweep dict.

    Returns:
        ``(per_prompt_reward_arrays, per_prompt_cost_arrays)`` for the
        dev-optimal subset, each seed-averaged to shape
        ``(n_holdout,)``.
    """
    pp_r_arrays: List[np.ndarray] = []
    pp_c_arrays: List[np.ndarray] = []
    for i in dev_idx:
        hval = sweep[i][hparam_key]
        r_arr = per_prompt_reward_map[hval]
        c_arr = per_prompt_cost_map[hval]
        pp_r_arrays.append(
            np.mean(r_arr, axis=0) if r_arr.ndim == 2 else r_arr
        )
        pp_c_arrays.append(
            np.mean(c_arr, axis=0) if c_arr.ndim == 2 else c_arr
        )
    return pp_r_arrays, pp_c_arrays
