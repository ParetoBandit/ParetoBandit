#!/usr/bin/env python3
"""Generate publication-quality judge robustness figures for experiments_v2.

Produces three figures from pre-existing calibration data (no API calls):

1. **Two-panel scatter** — R1 vs each supplementary judge, annotated with
   Lin's CCC and Kendall's tau-b (appropriate for ceiling-compressed,
   tied data).
2. **Two-panel Bland-Altman** — difference vs mean for each judge pair,
   showing how disagreement varies with score level.
3. **Gap distribution overlay** — Per-prompt reward gaps by judge.

Data sources
------------
- Primary R1 scores: ``data_collection/pareto_dataset/pareto_rewards.jsonl``
- Stratified subset:  ``data_collection/rewards/calibration/judge_robustness_prompts.jsonl``
- Supplementary:      ``data_collection/rewards/calibration/judge_robustness_rewards.jsonl``

Usage
-----
    python experiments_v2/appendix/judge_robustness/generate_figure.py
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from bandit_gpt.config import CALIBRATION_DIR, PARETO_REWARDS_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────
SUBSET_PROMPTS_PATH = CALIBRATION_DIR / "judge_robustness_prompts.jsonl"
SUPPLEMENTARY_REWARDS_PATH = CALIBRATION_DIR / "judge_robustness_rewards.jsonl"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# ── Visual constants ──────────────────────────────────────────────────────
CB_BLUE = "#0072B2"
CB_ORANGE = "#E69F00"
CB_GREEN = "#009E73"
CB_RED = "#D55E00"
CB_GRAY = "#999999"

JUDGE_META = {
    "openai/gpt-4.1-mini": {"short": "GPT-4.1-mini", "color": CB_ORANGE},
    "anthropic/claude-3.7-sonnet": {"short": "Claude-3.7-Sonnet", "color": CB_GREEN},
}


def _setup_matplotlib() -> None:
    """Configure matplotlib for publication-quality output."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10.5,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })


# ══════════════════════════════════════════════════════════════════════════
# Agreement metrics
# ══════════════════════════════════════════════════════════════════════════


def lins_ccc(x: np.ndarray, y: np.ndarray) -> float:
    """Lin's Concordance Correlation Coefficient.

    Measures agreement on the identity line, combining precision (Pearson r)
    with accuracy (how far the best-fit line deviates from y=x).  Unlike
    Pearson, CCC is penalised by both scale shift and location shift.

    Parameters
    ----------
    x, y:
        Paired measurements of equal length.

    Returns
    -------
    float
        CCC in [-1, 1].  Values near 1 indicate near-perfect agreement.

    References
    ----------
    Lin, L.I. (1989). A concordance correlation coefficient to evaluate
    reproducibility. *Biometrics*, 45(1), 255-268.
    """
    mx, my = np.mean(x), np.mean(y)
    sx2, sy2 = np.var(x, ddof=1), np.var(y, ddof=1)
    sxy = np.cov(x, y, ddof=1)[0, 1]
    return float(2.0 * sxy / (sx2 + sy2 + (mx - my) ** 2))


def compute_agreement_metrics(
    x: np.ndarray, y: np.ndarray,
) -> Dict[str, float]:
    """Compute a comprehensive suite of inter-judge agreement metrics.

    Parameters
    ----------
    x:
        Reference judge scores (R1).
    y:
        Supplementary judge scores.

    Returns
    -------
    Dict[str, float]
        Dictionary of metric name -> value.
    """
    pr, pr_p = stats.pearsonr(x, y)
    sr, sr_p = stats.spearmanr(x, y)
    tau, tau_p = stats.kendalltau(x, y, variant="b")
    ccc = lins_ccc(x, y)
    bias = float(np.mean(y) - np.mean(x))
    mad = float(np.mean(np.abs(y - x)))
    diff = y - x
    loa_mean = float(np.mean(diff))
    loa_std = float(np.std(diff, ddof=1))

    return {
        "n": len(x),
        "pearson_r": pr,
        "pearson_p": pr_p,
        "spearman_rho": sr,
        "spearman_p": sr_p,
        "kendall_tau_b": tau,
        "kendall_p": tau_p,
        "lins_ccc": ccc,
        "mean_bias": bias,
        "mad": mad,
        "bland_altman_mean": loa_mean,
        "bland_altman_lower": loa_mean - 1.96 * loa_std,
        "bland_altman_upper": loa_mean + 1.96 * loa_std,
    }


# ══════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════


def load_subset_prompts() -> Set[str]:
    """Load prompt texts from the stratified subset.

    Returns
    -------
    Set[str]
        The unique prompt strings in the robustness subset.
    """
    prompts: Set[str] = set()
    with open(SUBSET_PROMPTS_PATH) as f:
        for line in f:
            prompts.add(json.loads(line)["prompt"])
    return prompts


def load_r1_scores(prompt_set: Set[str]) -> Dict[Tuple[str, str], float]:
    """Load primary DeepSeek-R1 scores for the subset.

    Parameters
    ----------
    prompt_set:
        Prompts to include.

    Returns
    -------
    Dict[Tuple[str, str], float]
        Mapping ``(prompt, model_id)`` -> raw_score.
    """
    scores: Dict[Tuple[str, str], float] = {}
    with open(PARETO_REWARDS_PATH) as f:
        for line in f:
            rec = json.loads(line)
            if not rec.get("ok"):
                continue
            if rec["prompt"] not in prompt_set:
                continue
            scores[(rec["prompt"], rec["model_id"])] = rec["raw_score"]
    return scores


def load_supplementary_scores() -> Dict[str, Dict[Tuple[str, str], float]]:
    """Load per-judge supplementary scores.

    Returns
    -------
    Dict[str, Dict[Tuple[str, str], float]]
        Outer key is judge model id; inner maps ``(prompt, model_id)`` to
        reward.
    """
    scores: Dict[str, Dict[Tuple[str, str], float]] = defaultdict(dict)
    with open(SUPPLEMENTARY_REWARDS_PATH) as f:
        for line in f:
            rec = json.loads(line)
            if not rec.get("ok"):
                continue
            key = (rec["prompt"], rec["model_id"])
            for jd in rec.get("judge_details", []):
                scores[jd["judge"]][key] = jd["reward"]
    return dict(scores)


def _get_paired_arrays(
    r1: Dict[Tuple[str, str], float],
    judge_scores: Dict[Tuple[str, str], float],
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, str]]]:
    """Align R1 and supplementary scores on common keys.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, List[Tuple[str, str]]]
        (r1_values, supp_values, common_keys) — all sorted by key.
    """
    common = sorted(set(r1.keys()) & set(judge_scores.keys()))
    r1_vals = np.array([r1[k] for k in common])
    s_vals = np.array([judge_scores[k] for k in common])
    return r1_vals, s_vals, common


# ══════════════════════════════════════════════════════════════════════════
# Figure 1: Two-panel scatter with agreement metrics
# ══════════════════════════════════════════════════════════════════════════


def plot_scatter_panels(
    r1: Dict[Tuple[str, str], float],
    supp: Dict[str, Dict[Tuple[str, str], float]],
) -> Tuple[Path, Dict[str, Dict[str, float]]]:
    """Two-panel scatter: R1 vs each supplementary judge.

    Annotated with Lin's CCC and Kendall's tau-b rather than Pearson r,
    since the data exhibits ceiling compression and discretisation ties.

    Parameters
    ----------
    r1:
        Primary R1 scores keyed by (prompt, model_id).
    supp:
        Per-judge supplementary scores.

    Returns
    -------
    Tuple[Path, Dict[str, Dict[str, float]]]
        (path_to_pdf, {judge_id: metrics_dict}).
    """
    _setup_matplotlib()
    judges = sorted(
        supp.keys(), key=lambda j: JUDGE_META.get(j, {}).get("short", j)
    )

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0), constrained_layout=True)
    all_metrics: Dict[str, Dict[str, float]] = {}

    for ax, judge in zip(axes, judges):
        meta = JUDGE_META.get(judge, {"short": judge, "color": CB_GRAY})
        r1_vals, s_vals, _ = _get_paired_arrays(r1, supp[judge])
        m = compute_agreement_metrics(r1_vals, s_vals)
        all_metrics[judge] = m

        ax.scatter(
            r1_vals, s_vals,
            s=2.5, alpha=0.12, c=meta["color"],
            edgecolors="none", rasterized=True,
        )

        lims = [
            min(r1_vals.min(), s_vals.min()) - 0.03,
            max(r1_vals.max(), s_vals.max()) + 0.03,
        ]
        ax.plot(lims, lims, "--", color=CB_GRAY, lw=0.8, alpha=0.5, zorder=0)

        ax.text(
            0.04, 0.96,
            (
                f"CCC = {m['lins_ccc']:.3f}\n"
                f"Kendall $\\tau_b$ = {m['kendall_tau_b']:.3f}\n"
                f"Bias = {m['mean_bias']:+.3f}\n"
                f"MAD = {m['mad']:.3f}\n"
                f"$n$ = {m['n']:,}"
            ),
            transform=ax.transAxes, fontsize=7.5,
            verticalalignment="top",
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor="white",
                edgecolor="#cccccc", alpha=0.9,
            ),
        )

        ax.set_xlabel("DeepSeek-R1 Reward")
        ax.set_ylabel(f"{meta['short']} Reward")
        ax.set_title(f"R1 vs {meta['short']}", fontweight="bold")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.12, ls="--", lw=0.5)

    out = RESULTS_DIR / "judge_robustness.pdf"
    fig.savefig(out, facecolor="white")
    fig.savefig(out.with_suffix(".png"), facecolor="white")
    plt.close(fig)
    logger.info("Saved scatter panels: %s", out)
    return out, all_metrics


# ══════════════════════════════════════════════════════════════════════════
# Figure 2: Two-panel Bland-Altman
# ══════════════════════════════════════════════════════════════════════════


def plot_bland_altman(
    r1: Dict[Tuple[str, str], float],
    supp: Dict[str, Dict[Tuple[str, str], float]],
) -> Path:
    """Two-panel Bland-Altman: difference vs mean for each judge pair.

    Reveals how disagreement varies with score level — the ceiling
    compression that attenuates correlation is directly visible here.

    Parameters
    ----------
    r1:
        Primary R1 scores.
    supp:
        Per-judge supplementary scores.

    Returns
    -------
    Path
        Path to saved PDF.
    """
    _setup_matplotlib()
    judges = sorted(
        supp.keys(), key=lambda j: JUDGE_META.get(j, {}).get("short", j)
    )

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0), constrained_layout=True)

    for ax, judge in zip(axes, judges):
        meta = JUDGE_META.get(judge, {"short": judge, "color": CB_GRAY})
        r1_vals, s_vals, _ = _get_paired_arrays(r1, supp[judge])

        mean_vals = (r1_vals + s_vals) / 2.0
        diff_vals = s_vals - r1_vals

        ax.scatter(
            mean_vals, diff_vals,
            s=2.5, alpha=0.12, c=meta["color"],
            edgecolors="none", rasterized=True,
        )

        md = float(np.mean(diff_vals))
        sd = float(np.std(diff_vals, ddof=1))
        ax.axhline(md, color="black", lw=1.0, ls="-", alpha=0.7)
        ax.axhline(md + 1.96 * sd, color=CB_RED, lw=0.8, ls="--", alpha=0.6)
        ax.axhline(md - 1.96 * sd, color=CB_RED, lw=0.8, ls="--", alpha=0.6)

        ax.text(
            0.97, 0.96,
            (
                f"Mean diff = {md:+.3f}\n"
                f"95% LoA: [{md - 1.96*sd:+.3f}, {md + 1.96*sd:+.3f}]"
            ),
            transform=ax.transAxes, fontsize=7,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor="white",
                edgecolor="#cccccc", alpha=0.9,
            ),
        )

        ax.set_xlabel("Mean Reward (R1, Judge) / 2")
        ax.set_ylabel(f"{meta['short']} $-$ R1")
        ax.set_title(f"Bland-Altman: {meta['short']}", fontweight="bold")
        ax.grid(True, alpha=0.12, ls="--", lw=0.5)

    out = RESULTS_DIR / "judge_bland_altman.pdf"
    fig.savefig(out, facecolor="white")
    fig.savefig(out.with_suffix(".png"), facecolor="white")
    plt.close(fig)
    logger.info("Saved Bland-Altman panels: %s", out)
    return out


# ══════════════════════════════════════════════════════════════════════════
# Figure 3: Gap distribution comparison
# ══════════════════════════════════════════════════════════════════════════


def _prompt_gaps(scores: Dict[Tuple[str, str], float]) -> np.ndarray:
    """Compute per-prompt reward gap (best - worst model).

    Parameters
    ----------
    scores:
        Mapping ``(prompt, model_id)`` -> score.

    Returns
    -------
    np.ndarray
        Array of gap values, one per prompt with >= 2 scored models.
    """
    by_prompt: Dict[str, Dict[str, float]] = defaultdict(dict)
    for (prompt, model_id), score in scores.items():
        by_prompt[prompt][model_id] = score
    gaps = []
    for models in by_prompt.values():
        if len(models) >= 2:
            gaps.append(max(models.values()) - min(models.values()))
    return np.array(gaps)


def plot_gap_distribution(
    r1: Dict[Tuple[str, str], float],
    supp: Dict[str, Dict[Tuple[str, str], float]],
) -> Path:
    """Overlaid histogram of per-prompt reward gaps by judge.

    Parameters
    ----------
    r1:
        Primary R1 scores.
    supp:
        Per-judge supplementary scores.

    Returns
    -------
    Path
        Path to saved PDF.
    """
    _setup_matplotlib()

    fig, ax = plt.subplots(figsize=(4.5, 2.8), constrained_layout=True)

    r1_gaps = _prompt_gaps(r1)
    ax.hist(
        r1_gaps, bins=50, alpha=0.45, color=CB_BLUE, density=True,
        label=f"DeepSeek-R1 ($\\mu$={np.mean(r1_gaps):.3f})",
    )

    for judge in sorted(supp.keys()):
        meta = JUDGE_META.get(judge, {"short": judge, "color": CB_GRAY})
        g = _prompt_gaps(supp[judge])
        ax.hist(
            g, bins=50, alpha=0.35, color=meta["color"], density=True,
            label=f"{meta['short']} ($\\mu$={np.mean(g):.3f})",
        )

    ax.set_xlabel("Per-Prompt Reward Gap (best $-$ worst model)")
    ax.set_ylabel("Density")
    ax.set_title("Discriminative Signal by Judge", fontweight="bold")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.12, ls="--", lw=0.5)

    out = RESULTS_DIR / "judge_gap_distribution.pdf"
    fig.savefig(out, facecolor="white")
    fig.savefig(out.with_suffix(".png"), facecolor="white")
    plt.close(fig)
    logger.info("Saved gap distribution: %s", out)
    return out


# ══════════════════════════════════════════════════════════════════════════
# Summary export
# ══════════════════════════════════════════════════════════════════════════


def export_summary(
    per_judge_metrics: Dict[str, Dict[str, float]],
    r1: Dict[Tuple[str, str], float],
    supp: Dict[str, Dict[Tuple[str, str], float]],
) -> Path:
    """Export a JSON summary with all agreement metrics.

    Parameters
    ----------
    per_judge_metrics:
        Agreement metrics from :func:`plot_scatter_panels`.
    r1:
        Primary R1 scores.
    supp:
        Per-judge supplementary scores.

    Returns
    -------
    Path
        Path to saved JSON.
    """
    src = CALIBRATION_DIR / "analysis" / "judge_robustness_results.json"
    data: Dict[str, Any] = {}
    if src.exists():
        with open(src) as f:
            data = json.load(f)

    data["agreement_metrics"] = {
        judge: {k: round(v, 6) if isinstance(v, float) else v
                for k, v in metrics.items()}
        for judge, metrics in per_judge_metrics.items()
    }

    r1_gaps = _prompt_gaps(r1)
    gap_stats: Dict[str, Dict[str, float]] = {
        "deepseek-r1": {
            "mean_gap": round(float(np.mean(r1_gaps)), 4),
            "median_gap": round(float(np.median(r1_gaps)), 4),
        }
    }
    for judge, scores in supp.items():
        g = _prompt_gaps(scores)
        short = JUDGE_META.get(judge, {}).get("short", judge)
        gap_stats[short] = {
            "mean_gap": round(float(np.mean(g)), 4),
            "median_gap": round(float(np.median(g)), 4),
        }
    data["gap_statistics"] = gap_stats

    out = RESULTS_DIR / "judge_robustness_summary.json"
    with open(out, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Exported summary: %s", out)
    return out


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Generate all judge robustness figures and export summary."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading data...")
    prompt_set = load_subset_prompts()
    r1 = load_r1_scores(prompt_set)
    supp = load_supplementary_scores()
    logger.info(
        "R1: %d scores, Judges: %s",
        len(r1),
        {j: len(s) for j, s in supp.items()},
    )

    logger.info("Generating figures...")
    _, per_judge = plot_scatter_panels(r1, supp)
    plot_bland_altman(r1, supp)
    plot_gap_distribution(r1, supp)
    export_summary(per_judge, r1, supp)

    for judge, m in per_judge.items():
        short = JUDGE_META.get(judge, {}).get("short", judge)
        logger.info(
            "%s — CCC=%.3f  tau_b=%.3f  Pearson=%.3f  "
            "Bias=%+.3f  MAD=%.3f  LoA=[%+.3f, %+.3f]",
            short, m["lins_ccc"], m["kendall_tau_b"], m["pearson_r"],
            m["mean_bias"], m["mad"],
            m["bland_altman_lower"], m["bland_altman_upper"],
        )

    logger.info("Done. All outputs in %s", RESULTS_DIR)


if __name__ == "__main__":
    main()
