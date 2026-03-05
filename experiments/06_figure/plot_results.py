#!/usr/bin/env python3
"""
Generate Figure 6: Semantic Transfer Evaluation.

Two-panel figure:
  (a) Forest plot of per-target Delta (transfer minus tabula rasa) with
      95% CI.  Targets ordered by tetrachoric similarity to their
      neighbor.  Targets without same-provider peers shown grayed out.
  (b) Learning curves (holdout reward vs online steps) for targets that
      received transfer, comparing condition A (transfer) vs B (tabula
      rasa).
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).parent / "results"

BLUE = "#0173B2"
ORANGE = "#DE8F05"
GREEN = "#029E73"
GRAY = "#949494"
LIGHT_GRAY = "#cccccc"
RED = "#CC78BC"

TRANSFER_COLOR = BLUE
TABULA_COLOR = ORANGE
NO_PEER_COLOR = LIGHT_GRAY


def load_results() -> Dict[str, Any]:
    """Load semantic transfer results."""
    with open(RESULTS_DIR / "semantic_transfer_results.json") as f:
        return json.load(f)


def plot_figure6(data: Dict[str, Any], output_dir: Path) -> None:
    """Generate the two-panel Figure 6."""
    per_target = data["per_target"]
    learning_curves = data.get("learning_curves", {})
    n_seeds = data["metadata"]["n_seeds"]

    has_lc = len(learning_curves) > 0
    n_cols = 2 if has_lc else 1
    fig, axes = plt.subplots(
        1, n_cols,
        figsize=(7 * n_cols, 5),
        gridspec_kw={"wspace": 0.35},
    )
    if n_cols == 1:
        axes = [axes]

    # ------------------------------------------------------------------
    # Panel (a): Forest plot of per-target Delta
    # ------------------------------------------------------------------
    ax = axes[0]

    entries = []
    for model_id, r in per_target.items():
        entries.append({
            "display": r["display"],
            "delta": r["delta_mean"],
            "ci95": r["delta_ci95"],
            "r_tet": r["tetrachoric_similarity"],
            "has_peer": r["tetrachoric_neighbor"] is not None,
            "p_holm": r.get("p_value_holm", 1.0),
            "n_xfer": r["n_trials_with_actual_transfer"],
            "n_total": r["n_trials_total"],
        })

    entries.sort(key=lambda e: (not e["has_peer"], -e["r_tet"]))

    y_positions = list(range(len(entries)))
    for i, e in enumerate(entries):
        color = TRANSFER_COLOR if e["has_peer"] else NO_PEER_COLOR
        ax.errorbar(
            e["delta"], i,
            xerr=e["ci95"],
            fmt="o" if e["has_peer"] else "s",
            color=color,
            capsize=4,
            markersize=6,
            linewidth=1.5,
        )
        sig = ""
        if e["p_holm"] < 0.01:
            sig = " **"
        elif e["p_holm"] < 0.05:
            sig = " *"
        label = e["display"]
        if e["has_peer"]:
            label += f" (r={e['r_tet']:.2f}){sig}"
        else:
            label += " (no peer)"

        ax.annotate(
            label,
            (e["delta"], i),
            textcoords="offset points",
            xytext=(8, 0),
            fontsize=8.5,
            va="center",
            color=color if not e["has_peer"] else "black",
        )

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_yticks([])
    ax.set_xlabel("$\\Delta$ Holdout Reward (transfer $-$ tabula rasa)")
    ax.set_title(
        "(a) Per-Target Semantic Transfer Effect\n"
        f"(K=3, {n_seeds} seeds, 95% CI, Holm-corrected)",
        fontsize=11,
    )

    agg = data["aggregate"]
    ax.annotate(
        f"Aggregate: $\\Delta$={agg['mean_delta']:+.4f}, "
        f"d={agg['cohens_d']:.3f}, p={agg['p_value']:.3g}",
        xy=(0.02, 0.02),
        xycoords="axes fraction",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8),
    )

    x_margin = max(abs(e["delta"]) + e["ci95"] for e in entries) * 1.8
    ax.set_xlim(-x_margin, x_margin)
    ax.set_ylim(-0.5, len(entries) - 0.5)
    ax.invert_yaxis()

    # ------------------------------------------------------------------
    # Panel (b): Learning curves for transferred targets
    # ------------------------------------------------------------------
    if has_lc:
        ax = axes[1]
        ci_z = 1.96 / np.sqrt(n_seeds)

        lc_colors = [BLUE, GREEN, RED, ORANGE, GRAY]
        for idx, (model_id, lc) in enumerate(learning_curves.items()):
            color = lc_colors[idx % len(lc_colors)]
            display = lc["display"]
            checkpoints = lc["checkpoints"]
            steps = np.array(checkpoints)

            t_means = np.array(
                [lc["transfer"][str(s)]["mean"] for s in checkpoints]
            )
            t_stds = np.array(
                [lc["transfer"][str(s)]["std"] for s in checkpoints]
            )
            b_means = np.array(
                [lc["tabula_rasa"][str(s)]["mean"] for s in checkpoints]
            )
            b_stds = np.array(
                [lc["tabula_rasa"][str(s)]["std"] for s in checkpoints]
            )

            ax.plot(
                steps, t_means, "-o",
                color=color, markersize=3, linewidth=1.5,
                label=f"{display} (transfer)",
            )
            ax.fill_between(
                steps,
                t_means - ci_z * t_stds,
                t_means + ci_z * t_stds,
                alpha=0.15, color=color,
            )

            ax.plot(
                steps, b_means, "--s",
                color=color, markersize=3, linewidth=1.0, alpha=0.6,
                label=f"{display} (tabula rasa)",
            )
            ax.fill_between(
                steps,
                b_means - ci_z * b_stds,
                b_means + ci_z * b_stds,
                alpha=0.08, color=color,
            )

        ax.set_xlabel("Online Training Steps")
        ax.set_ylabel("Holdout Reward")
        ax.set_title(
            "(b) Learning Curves: Transfer vs Tabula Rasa\n"
            f"({n_seeds} seeds, 95% CI)",
            fontsize=11,
        )
        ax.legend(fontsize=7, loc="lower right")

    fig.tight_layout()

    out_png = output_dir / "figure6_semantic_transfer.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")

    out_pdf = output_dir / "figure6_semantic_transfer.pdf"
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")

    plt.close(fig)
    print(f"Figure saved: {out_png}")
    print(f"Figure saved: {out_pdf}")


def main() -> None:
    output_dir = RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_results()
    plot_figure6(data, output_dir)


if __name__ == "__main__":
    main()
