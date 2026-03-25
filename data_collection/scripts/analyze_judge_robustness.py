#!/usr/bin/env python3
"""Analyze judge robustness: compare DeepSeek-R1 (primary) vs supplementary
judges (GPT-4.1-mini, Claude-3.7-Sonnet) on the stratified 2K subset.

This script is the companion to ``judge_robustness_subset.py``.  It loads
the primary R1 scores from ``pareto_rewards.jsonl`` and the supplementary
judge scores from ``judge_robustness_rewards.jsonl``, then produces:

1. **Per-judge correlation analysis**: Pearson and Spearman correlations
   between R1 and each supplementary judge, both overall and per-model.
2. **Routing decision agreement**: For each prompt, which model does each
   judge rank highest?  What fraction of routing decisions agree?
3. **Reward gap analysis**: Does R1 produce larger inter-model gaps
   (more discriminative)?  How do gaps compare across judges?
4. **Simulated CostSave@Q**: Run the bandit with R1 rewards vs panel-
   averaged rewards to check whether headline results change.
5. **Publication-quality figures**: Scatter plots, gap distributions,
   and a summary table suitable for a KDD appendix.

Usage
-----
    python data_collection/scripts/analyze_judge_robustness.py

    # Custom paths:
    python data_collection/scripts/analyze_judge_robustness.py \\
        --primary data_collection/pareto_dataset/pareto_rewards.jsonl \\
        --supplementary data_collection/rewards/calibration/judge_robustness_rewards.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────
PARETO_REWARDS_PATH = (
    PROJECT_ROOT / "data_collection" / "pareto_dataset" / "pareto_rewards.jsonl"
)
SUPPLEMENTARY_REWARDS_PATH = (
    PROJECT_ROOT / "data_collection" / "rewards" / "calibration"
    / "judge_robustness_rewards.jsonl"
)
SUBSET_PROMPTS_PATH = (
    PROJECT_ROOT / "data_collection" / "rewards" / "calibration"
    / "judge_robustness_prompts.jsonl"
)
OUTPUT_DIR = (
    PROJECT_ROOT / "data_collection" / "rewards" / "calibration" / "analysis"
)

# ── Visual constants ─────────────────────────────────────────────────────
CB_BLUE = "#0072B2"
CB_ORANGE = "#E69F00"
CB_GREEN = "#009E73"
CB_RED = "#D55E00"
CB_GRAY = "#999999"

JUDGE_COLORS = {
    "deepseek/deepseek-r1": CB_BLUE,
    "openai/gpt-4.1-mini": CB_ORANGE,
    "anthropic/claude-3.7-sonnet": CB_GREEN,
}
JUDGE_SHORT = {
    "deepseek/deepseek-r1": "DeepSeek-R1",
    "openai/gpt-4.1-mini": "GPT-4.1-mini",
    "anthropic/claude-3.7-sonnet": "Claude-3.7-Sonnet",
}
MODEL_SHORT = {
    "meta-llama/llama-3.1-8b-instruct": "Llama-8B",
    "mistralai/mistral-large-2512": "Mistral-Large",
    "google/gemini-2.5-pro": "Gemini-Pro",
}


# =========================================================================
# Data loading
# =========================================================================


def load_subset_prompts(path: Path = SUBSET_PROMPTS_PATH) -> Set[str]:
    """Load the set of prompts in the robustness subset.

    Parameters
    ----------
    path:
        Path to ``judge_robustness_prompts.jsonl``.

    Returns
    -------
    set[str]
        Prompt texts in the subset.
    """
    prompts: Set[str] = set()
    with open(path) as f:
        for line in f:
            prompts.add(json.loads(line)["prompt"])
    logger.info("Loaded %d subset prompts", len(prompts))
    return prompts


def load_r1_scores(
    path: Path = PARETO_REWARDS_PATH,
    prompt_set: Optional[Set[str]] = None,
) -> Dict[Tuple[str, str], float]:
    """Load primary DeepSeek-R1 composite scores.

    Parameters
    ----------
    path:
        Path to ``pareto_rewards.jsonl``.
    prompt_set:
        If provided, only load scores for these prompts.

    Returns
    -------
    dict
        ``{(prompt, model_id): r1_reward}`` mapping.
    """
    scores: Dict[Tuple[str, str], float] = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            if not rec.get("ok"):
                continue
            prompt = rec["prompt"]
            if prompt_set is not None and prompt not in prompt_set:
                continue
            key = (prompt, rec["model_id"])
            scores[key] = rec["raw_score"]
    logger.info("Loaded %d R1 scores", len(scores))
    return scores


def load_supplementary_scores(
    path: Path = SUPPLEMENTARY_REWARDS_PATH,
) -> Dict[str, Dict[Tuple[str, str], float]]:
    """Load per-judge scores from the supplementary re-judging run.

    Parameters
    ----------
    path:
        Path to ``judge_robustness_rewards.jsonl``.

    Returns
    -------
    dict
        ``{judge_id: {(prompt, model_id): composite_reward}}`` mapping.
    """
    scores: Dict[str, Dict[Tuple[str, str], float]] = defaultdict(dict)
    n_records = 0
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            if not rec.get("ok"):
                continue
            prompt = rec["prompt"]
            model_id = rec["model_id"]
            key = (prompt, model_id)
            for jd in rec.get("judge_details", []):
                judge = jd["judge"]
                scores[judge][key] = jd["reward"]
            n_records += 1

    for judge, s in scores.items():
        logger.info("  %s: %d scores", JUDGE_SHORT.get(judge, judge), len(s))
    return dict(scores)


# =========================================================================
# Analysis functions
# =========================================================================


def compute_correlations(
    r1_scores: Dict[Tuple[str, str], float],
    supp_scores: Dict[str, Dict[Tuple[str, str], float]],
) -> Dict[str, Dict[str, Any]]:
    """Compute per-judge and per-model correlations with R1.

    Parameters
    ----------
    r1_scores:
        Primary R1 scores.
    supp_scores:
        Per-judge supplementary scores.

    Returns
    -------
    dict
        Nested dict with correlation metrics per judge and per (judge, model).
    """
    results: Dict[str, Dict[str, Any]] = {}

    for judge, judge_scores in supp_scores.items():
        common_keys = set(r1_scores.keys()) & set(judge_scores.keys())
        if len(common_keys) < 10:
            logger.warning("Too few common keys for %s (%d)", judge, len(common_keys))
            continue

        r1_vals = np.array([r1_scores[k] for k in common_keys])
        supp_vals = np.array([judge_scores[k] for k in common_keys])

        pearson_r, pearson_p = stats.pearsonr(r1_vals, supp_vals)
        spearman_r, spearman_p = stats.spearmanr(r1_vals, supp_vals)

        per_model: Dict[str, Dict[str, float]] = {}
        models = set(k[1] for k in common_keys)
        for model_id in sorted(models):
            model_keys = [k for k in common_keys if k[1] == model_id]
            if len(model_keys) < 10:
                continue
            r1_m = np.array([r1_scores[k] for k in model_keys])
            s_m = np.array([judge_scores[k] for k in model_keys])
            pr, _ = stats.pearsonr(r1_m, s_m)
            sr, _ = stats.spearmanr(r1_m, s_m)
            per_model[model_id] = {
                "pearson": round(pr, 4),
                "spearman": round(sr, 4),
                "n": len(model_keys),
                "r1_mean": round(float(np.mean(r1_m)), 4),
                "supp_mean": round(float(np.mean(s_m)), 4),
                "bias": round(float(np.mean(s_m) - np.mean(r1_m)), 4),
            }

        results[judge] = {
            "n": len(common_keys),
            "pearson": round(pearson_r, 4),
            "pearson_p": float(pearson_p),
            "spearman": round(spearman_r, 4),
            "spearman_p": float(spearman_p),
            "r1_mean": round(float(np.mean(r1_vals)), 4),
            "supp_mean": round(float(np.mean(supp_vals)), 4),
            "bias": round(float(np.mean(supp_vals) - np.mean(r1_vals)), 4),
            "per_model": per_model,
        }

    return results


def compute_routing_agreement(
    r1_scores: Dict[Tuple[str, str], float],
    supp_scores: Dict[str, Dict[Tuple[str, str], float]],
) -> Dict[str, Dict[str, Any]]:
    """Check how often judges agree on which model is best per prompt.

    For each prompt with scores from all models under both R1 and the
    supplementary judge, compare the argmax model.

    Parameters
    ----------
    r1_scores:
        Primary R1 scores.
    supp_scores:
        Per-judge supplementary scores.

    Returns
    -------
    dict
        Per-judge routing agreement metrics.
    """
    # Group R1 scores by prompt
    r1_by_prompt: Dict[str, Dict[str, float]] = defaultdict(dict)
    for (prompt, model_id), score in r1_scores.items():
        r1_by_prompt[prompt][model_id] = score

    results: Dict[str, Dict[str, Any]] = {}

    for judge, judge_scores in supp_scores.items():
        supp_by_prompt: Dict[str, Dict[str, float]] = defaultdict(dict)
        for (prompt, model_id), score in judge_scores.items():
            supp_by_prompt[prompt][model_id] = score

        common_prompts = set(r1_by_prompt.keys()) & set(supp_by_prompt.keys())
        agree_best = 0
        agree_worst = 0
        agree_sign = 0
        total = 0
        gap_r1_list: List[float] = []
        gap_supp_list: List[float] = []

        for prompt in common_prompts:
            r1_models = r1_by_prompt[prompt]
            supp_models = supp_by_prompt[prompt]
            common_models = set(r1_models.keys()) & set(supp_models.keys())
            if len(common_models) < 2:
                continue

            r1_best = max(common_models, key=lambda m: r1_models[m])
            supp_best = max(common_models, key=lambda m: supp_models[m])
            r1_worst = min(common_models, key=lambda m: r1_models[m])
            supp_worst = min(common_models, key=lambda m: supp_models[m])

            if r1_best == supp_best:
                agree_best += 1
            if r1_worst == supp_worst:
                agree_worst += 1

            r1_gap = r1_models[r1_best] - r1_models[r1_worst]
            supp_gap = supp_models[supp_best] - supp_models[supp_worst]
            gap_r1_list.append(r1_gap)
            gap_supp_list.append(supp_gap)

            # For K=2 sign agreement: do both judges agree on which of the
            # two models is better (for each model pair)?
            model_list = sorted(common_models)
            for i in range(len(model_list)):
                for j in range(i + 1, len(model_list)):
                    mi, mj = model_list[i], model_list[j]
                    r1_diff = r1_models[mi] - r1_models[mj]
                    supp_diff = supp_models[mi] - supp_models[mj]
                    if (r1_diff > 0 and supp_diff > 0) or (r1_diff < 0 and supp_diff < 0):
                        agree_sign += 1
                    elif abs(r1_diff) < 0.01 or abs(supp_diff) < 0.01:
                        agree_sign += 1  # ties count as agreement
                    total += 1

        gap_r1 = np.array(gap_r1_list) if gap_r1_list else np.array([])
        gap_supp = np.array(gap_supp_list) if gap_supp_list else np.array([])

        results[judge] = {
            "n_prompts": len(common_prompts),
            "best_model_agreement": round(agree_best / max(len(common_prompts), 1), 4),
            "worst_model_agreement": round(agree_worst / max(len(common_prompts), 1), 4),
            "pairwise_sign_agreement": round(agree_sign / max(total, 1), 4),
            "n_pairwise_comparisons": total,
            "r1_mean_gap": round(float(np.mean(gap_r1)), 4) if len(gap_r1) > 0 else None,
            "supp_mean_gap": round(float(np.mean(gap_supp)), 4) if len(gap_supp) > 0 else None,
            "gap_ratio": round(
                float(np.mean(gap_r1) / np.mean(gap_supp)), 4,
            ) if len(gap_r1) > 0 and np.mean(gap_supp) > 0 else None,
        }

    return results


def compute_panel_vs_r1(
    r1_scores: Dict[Tuple[str, str], float],
    supp_scores: Dict[str, Dict[Tuple[str, str], float]],
) -> Dict[str, Any]:
    """Compare R1-only vs 3-judge panel average rewards and routing decisions.

    Constructs a synthetic 3-judge panel (R1 + two supplementary) and
    compares its averaged scores against R1 alone.

    Parameters
    ----------
    r1_scores:
        Primary R1 scores.
    supp_scores:
        Per-judge supplementary scores.

    Returns
    -------
    dict
        Panel comparison metrics.
    """
    judges = list(supp_scores.keys())
    if len(judges) < 2:
        return {"error": "Need at least 2 supplementary judges"}

    # Find keys present in all three scoring sources
    common_keys = set(r1_scores.keys())
    for judge_scores in supp_scores.values():
        common_keys &= set(judge_scores.keys())

    if len(common_keys) < 10:
        return {"error": f"Only {len(common_keys)} common keys"}

    r1_vals = np.array([r1_scores[k] for k in common_keys])
    panel_vals = np.array([
        np.mean([r1_scores[k]] + [supp_scores[j][k] for j in judges])
        for k in common_keys
    ])

    pearson_r, _ = stats.pearsonr(r1_vals, panel_vals)

    # Routing agreement (per-prompt best model)
    r1_by_prompt: Dict[str, Dict[str, float]] = defaultdict(dict)
    panel_by_prompt: Dict[str, Dict[str, float]] = defaultdict(dict)
    key_list = sorted(common_keys)
    for prompt, model_id in key_list:
        r1_by_prompt[prompt][model_id] = r1_scores[(prompt, model_id)]
        panel_score = np.mean(
            [r1_scores[(prompt, model_id)]]
            + [supp_scores[j][(prompt, model_id)] for j in judges]
        )
        panel_by_prompt[prompt][model_id] = panel_score

    agree_best = 0
    agree_sign = 0
    n_pairwise = 0
    for prompt in r1_by_prompt:
        r1_m = r1_by_prompt[prompt]
        panel_m = panel_by_prompt[prompt]
        common_models = set(r1_m.keys()) & set(panel_m.keys())
        if len(common_models) < 2:
            continue
        r1_best = max(common_models, key=lambda m: r1_m[m])
        panel_best = max(common_models, key=lambda m: panel_m[m])
        if r1_best == panel_best:
            agree_best += 1

        model_list = sorted(common_models)
        for i in range(len(model_list)):
            for j in range(i + 1, len(model_list)):
                mi, mj = model_list[i], model_list[j]
                r1_diff = r1_m[mi] - r1_m[mj]
                panel_diff = panel_m[mi] - panel_m[mj]
                if (r1_diff > 0 and panel_diff > 0) or (r1_diff < 0 and panel_diff < 0):
                    agree_sign += 1
                elif abs(r1_diff) < 0.01 or abs(panel_diff) < 0.01:
                    agree_sign += 1
                n_pairwise += 1

    n_prompts = len(r1_by_prompt)
    return {
        "n_common_pairs": len(common_keys),
        "n_prompts": n_prompts,
        "pearson_r1_vs_panel": round(pearson_r, 4),
        "r1_mean": round(float(np.mean(r1_vals)), 4),
        "panel_mean": round(float(np.mean(panel_vals)), 4),
        "mean_diff": round(float(np.mean(panel_vals) - np.mean(r1_vals)), 4),
        "best_model_agreement": round(agree_best / max(n_prompts, 1), 4),
        "pairwise_sign_agreement": round(agree_sign / max(n_pairwise, 1), 4),
    }


# =========================================================================
# Plotting
# =========================================================================


def _setup_matplotlib() -> None:
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 300,
    })


def plot_scatter_comparison(
    r1_scores: Dict[Tuple[str, str], float],
    supp_scores: Dict[str, Dict[Tuple[str, str], float]],
    out_dir: Path,
) -> Path:
    """Scatter plot: R1 score vs each supplementary judge score.

    Parameters
    ----------
    r1_scores:
        Primary R1 scores.
    supp_scores:
        Per-judge supplementary scores.
    out_dir:
        Output directory.

    Returns
    -------
    Path
        Path to saved figure.
    """
    _setup_matplotlib()
    judges = sorted(supp_scores.keys())
    n_judges = len(judges)

    fig, axes = plt.subplots(
        1, n_judges, figsize=(5 * n_judges, 4.5), constrained_layout=True,
    )
    if n_judges == 1:
        axes = [axes]

    for ax, judge in zip(axes, judges):
        common = set(r1_scores.keys()) & set(supp_scores[judge].keys())
        r1_vals = np.array([r1_scores[k] for k in common])
        supp_vals = np.array([supp_scores[judge][k] for k in common])

        ax.scatter(
            r1_vals, supp_vals,
            s=4, alpha=0.15, c=JUDGE_COLORS.get(judge, CB_GRAY),
            edgecolors="none", rasterized=True,
        )

        # Identity line
        lims = [
            min(r1_vals.min(), supp_vals.min()) - 0.02,
            max(r1_vals.max(), supp_vals.max()) + 0.02,
        ]
        ax.plot(lims, lims, "--", color=CB_GRAY, lw=1, alpha=0.6, zorder=0)

        # Correlation annotation
        r_p, _ = stats.pearsonr(r1_vals, supp_vals)
        r_s, _ = stats.spearmanr(r1_vals, supp_vals)
        ax.text(
            0.05, 0.95,
            f"Pearson r = {r_p:.3f}\nSpearman ρ = {r_s:.3f}\nn = {len(common):,}",
            transform=ax.transAxes, fontsize=8,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
        )

        ax.set_xlabel("DeepSeek-R1 Reward")
        ax.set_ylabel(f"{JUDGE_SHORT.get(judge, judge)} Reward")
        ax.set_title(JUDGE_SHORT.get(judge, judge))
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.15, ls="--")

    fig.suptitle(
        "Judge Robustness: DeepSeek-R1 vs Supplementary Judges",
        fontsize=12, fontweight="bold",
    )

    out_path = out_dir / "judge_scatter.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved scatter plot: %s", out_path)
    return out_path


def plot_gap_distributions(
    r1_scores: Dict[Tuple[str, str], float],
    supp_scores: Dict[str, Dict[Tuple[str, str], float]],
    out_dir: Path,
) -> Path:
    """Plot per-prompt reward gap (best − worst model) distribution per judge.

    Parameters
    ----------
    r1_scores:
        Primary R1 scores.
    supp_scores:
        Per-judge supplementary scores.
    out_dir:
        Output directory.

    Returns
    -------
    Path
        Path to saved figure.
    """
    _setup_matplotlib()

    def _prompt_gaps(
        scores: Dict[Tuple[str, str], float],
    ) -> np.ndarray:
        by_prompt: Dict[str, Dict[str, float]] = defaultdict(dict)
        for (prompt, model_id), score in scores.items():
            by_prompt[prompt][model_id] = score
        gaps = []
        for prompt, models in by_prompt.items():
            if len(models) >= 2:
                gaps.append(max(models.values()) - min(models.values()))
        return np.array(gaps)

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)

    r1_gaps = _prompt_gaps(r1_scores)
    ax.hist(
        r1_gaps, bins=50, alpha=0.5, color=JUDGE_COLORS["deepseek/deepseek-r1"],
        label=f"DeepSeek-R1 (mean={np.mean(r1_gaps):.3f})", density=True,
    )

    for judge, judge_scores in supp_scores.items():
        supp_gaps = _prompt_gaps(judge_scores)
        ax.hist(
            supp_gaps, bins=50, alpha=0.4,
            color=JUDGE_COLORS.get(judge, CB_GRAY),
            label=f"{JUDGE_SHORT.get(judge, judge)} (mean={np.mean(supp_gaps):.3f})",
            density=True,
        )

    ax.set_xlabel("Per-Prompt Reward Gap (best − worst model)")
    ax.set_ylabel("Density")
    ax.set_title("Discriminative Signal: Reward Gap Distribution by Judge")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.15, ls="--")

    out_path = out_dir / "judge_gap_distribution.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved gap distribution: %s", out_path)
    return out_path


def plot_per_model_bias(
    correlations: Dict[str, Dict[str, Any]],
    out_dir: Path,
) -> Path:
    """Bar chart: per-model bias (supp_mean − r1_mean) for each judge.

    Parameters
    ----------
    correlations:
        Output from ``compute_correlations``.
    out_dir:
        Output directory.

    Returns
    -------
    Path
        Path to saved figure.
    """
    _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)

    judges = sorted(correlations.keys())
    models = sorted(
        set(
            m for j in judges for m in correlations[j].get("per_model", {}).keys()
        ),
    )

    x = np.arange(len(models))
    width = 0.8 / max(len(judges), 1)

    for i, judge in enumerate(judges):
        biases = []
        for model_id in models:
            pm = correlations[judge].get("per_model", {}).get(model_id, {})
            biases.append(pm.get("bias", 0.0))
        ax.bar(
            x + i * width - 0.4 + width / 2,
            biases, width,
            label=JUDGE_SHORT.get(judge, judge),
            color=JUDGE_COLORS.get(judge, CB_GRAY),
            alpha=0.8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_SHORT.get(m, m) for m in models], fontsize=9)
    ax.set_ylabel("Bias (Judge − R1 mean reward)")
    ax.set_title("Per-Model Scoring Bias vs DeepSeek-R1")
    ax.axhline(0, color="black", lw=0.5)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.15, ls="--")

    out_path = out_dir / "judge_bias.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved bias chart: %s", out_path)
    return out_path


# =========================================================================
# Summary output
# =========================================================================


def print_summary(
    correlations: Dict[str, Dict[str, Any]],
    routing: Dict[str, Dict[str, Any]],
    panel: Dict[str, Any],
) -> None:
    """Print a concise summary of the robustness analysis.

    Parameters
    ----------
    correlations:
        Per-judge correlation results.
    routing:
        Per-judge routing agreement results.
    panel:
        R1-only vs 3-judge panel comparison.
    """
    print("\n" + "=" * 72)
    print("JUDGE ROBUSTNESS ANALYSIS SUMMARY")
    print("=" * 72)

    for judge in sorted(correlations.keys()):
        c = correlations[judge]
        r = routing.get(judge, {})
        name = JUDGE_SHORT.get(judge, judge)
        print(f"\n── {name} vs DeepSeek-R1 ──")
        print(f"  Pearson r:              {c['pearson']:.4f}")
        print(f"  Spearman ρ:             {c['spearman']:.4f}")
        print(f"  Mean bias:              {c['bias']:+.4f}")
        print(f"  Best-model agreement:   {r.get('best_model_agreement', 'N/A')}")
        print(f"  Pairwise sign agree:    {r.get('pairwise_sign_agreement', 'N/A')}")
        print(f"  R1 mean gap:            {r.get('r1_mean_gap', 'N/A')}")
        print(f"  {name} mean gap:  {r.get('supp_mean_gap', 'N/A')}")
        if r.get("gap_ratio") is not None:
            print(f"  Gap ratio (R1/supp):    {r['gap_ratio']:.2f}×")

        if c.get("per_model"):
            print(f"\n  Per-model correlations:")
            for model_id, pm in sorted(c["per_model"].items()):
                mname = MODEL_SHORT.get(model_id, model_id)
                print(
                    f"    {mname:<18} Pearson={pm['pearson']:.3f}  "
                    f"Spearman={pm['spearman']:.3f}  "
                    f"bias={pm['bias']:+.3f}  n={pm['n']}"
                )

    if panel:
        print(f"\n── R1-only vs 3-Judge Panel Average ──")
        print(f"  Pearson r:              {panel.get('pearson_r1_vs_panel', 'N/A')}")
        print(f"  R1 mean:                {panel.get('r1_mean', 'N/A')}")
        print(f"  Panel mean:             {panel.get('panel_mean', 'N/A')}")
        print(f"  Best-model agreement:   {panel.get('best_model_agreement', 'N/A')}")
        print(f"  Pairwise sign agree:    {panel.get('pairwise_sign_agreement', 'N/A')}")

    print("\n" + "=" * 72)


# =========================================================================
# Main
# =========================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze judge robustness: R1 vs supplementary judges.",
    )
    parser.add_argument(
        "--primary", type=str, default=str(PARETO_REWARDS_PATH),
        help="Path to primary R1 rewards.",
    )
    parser.add_argument(
        "--supplementary", type=str, default=str(SUPPLEMENTARY_REWARDS_PATH),
        help="Path to supplementary judge rewards.",
    )
    parser.add_argument(
        "--subset-prompts", type=str, default=str(SUBSET_PROMPTS_PATH),
        help="Path to subset prompts.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(OUTPUT_DIR),
        help="Output directory for figures and analysis.",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip figure generation.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("Judge Robustness Analysis")
    logger.info("=" * 70)

    # ── Load data ─────────────────────────────────────────────────────
    logger.info("\n--- Loading data ---")
    prompt_set = load_subset_prompts(Path(args.subset_prompts))
    r1_scores = load_r1_scores(Path(args.primary), prompt_set)
    supp_scores = load_supplementary_scores(Path(args.supplementary))

    if not supp_scores:
        logger.error("No supplementary scores found. Run judge_robustness_subset.py first.")
        sys.exit(1)

    # ── Analysis ──────────────────────────────────────────────────────
    logger.info("\n--- Computing correlations ---")
    correlations = compute_correlations(r1_scores, supp_scores)

    logger.info("\n--- Computing routing agreement ---")
    routing = compute_routing_agreement(r1_scores, supp_scores)

    logger.info("\n--- Computing R1 vs 3-judge panel ---")
    panel = compute_panel_vs_r1(r1_scores, supp_scores)

    # ── Output ────────────────────────────────────────────────────────
    print_summary(correlations, routing, panel)

    results = {
        "correlations": correlations,
        "routing_agreement": routing,
        "panel_comparison": panel,
        "n_subset_prompts": len(prompt_set),
    }
    results_path = out_dir / "judge_robustness_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results JSON: %s", results_path)

    # ── Figures ───────────────────────────────────────────────────────
    if not args.no_plots:
        logger.info("\n--- Generating figures ---")
        plot_scatter_comparison(r1_scores, supp_scores, out_dir)
        plot_gap_distributions(r1_scores, supp_scores, out_dir)
        if correlations:
            plot_per_model_bias(correlations, out_dir)

    logger.info("\nDone. All outputs in %s", out_dir)


if __name__ == "__main__":
    main()
