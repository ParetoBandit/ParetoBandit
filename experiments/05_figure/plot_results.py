#!/usr/bin/env python3
"""
Generate Figure 5: Corralling Enables Recovery from Catastrophic Failure.

Three-panel figure:
  (a) Online reward over time — BanditGPT vs baselines with phase shading.
      Inset: frozen holdout evaluation at checkpoints.
  (b) Corralling expert weights — warmup vs tabula rasa expert.
  (c) Model selection fractions — traffic redistribution during failure.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

RESULTS_DIR = Path(__file__).parent / "results"

BLUE = "#2c3e50"
RED = "#e74c3c"
ORANGE = "#e67e22"
GREEN = "#27ae60"
GRAY = "#95a5a6"
PURPLE = "#8e44ad"
TEAL = "#1abc9c"

MODEL_COLORS = [
    "#3498db", "#2ecc71", "#f39c12", "#e74c3c", "#9b59b6",
    "#1abc9c", "#e67e22", "#34495e", "#d35400", "#7f8c8d",
]


def load_results() -> dict:
    """Load catastrophic failure results."""
    with open(RESULTS_DIR / "catastrophic_failure_results.json") as f:
        return json.load(f)


def _smooth(arr: np.ndarray, window: int = 20) -> np.ndarray:
    """Running mean along axis 0."""
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")


def plot_figure5(data: dict, output_dir: Path) -> None:
    """Generate three-panel Figure 5."""
    n_steps = data["n_steps"]
    phase_b = data["phase_boundaries"]
    ts = data["time_series"]
    n_train = data.get("n_train", phase_b[0])
    window = max(20, n_train // 50)

    fig = plt.figure(figsize=(13, 11))
    gs = fig.add_gridspec(
        4, 1, height_ratios=[0.04, 1, 0.8, 0.8],
        hspace=0.08, top=0.94, bottom=0.05,
    )
    ax_phase = fig.add_subplot(gs[0, 0])
    ax_reward = fig.add_subplot(gs[1, 0])
    ax_weights = fig.add_subplot(gs[2, 0], sharex=ax_reward)
    ax_selection = fig.add_subplot(gs[3, 0], sharex=ax_reward)

    t = np.arange(n_steps)
    t_smooth = t[window - 1:]

    # Phase strip
    ax_phase.set_xlim(0, n_steps)
    ax_phase.set_ylim(0, 1)
    ax_phase.axvspan(0, phase_b[0], color="#2ecc71", alpha=0.25)
    ax_phase.axvspan(phase_b[0], phase_b[1], color="#e74c3c", alpha=0.25)
    ax_phase.axvspan(phase_b[1], n_steps, color="#3498db", alpha=0.25)
    ax_phase.text(
        phase_b[0] / 2, 0.5, "All Healthy", ha="center", va="center",
        fontsize=12, fontweight="bold", color="#1a7a3a",
    )
    failing_display = data.get("failing_model", "GPT-4.1").split("/")[-1]
    ax_phase.text(
        (phase_b[0] + phase_b[1]) / 2, 0.5,
        f"{failing_display} Fails",
        ha="center", va="center", fontsize=12, fontweight="bold", color="#a8201a",
    )
    ax_phase.text(
        (phase_b[1] + n_steps) / 2, 0.5,
        f"{failing_display} Recovers",
        ha="center", va="center", fontsize=12, fontweight="bold", color="#1a5276",
    )
    ax_phase.set_axis_off()

    for ax in (ax_reward, ax_weights, ax_selection):
        ax.axvspan(0, phase_b[0], color="#2ecc71", alpha=0.04)
        ax.axvspan(phase_b[0], phase_b[1], color="#e74c3c", alpha=0.04)
        ax.axvspan(phase_b[1], n_steps, color="#3498db", alpha=0.04)
        ax.axvline(phase_b[0], color="gray", ls="--", lw=1, alpha=0.4)
        ax.axvline(phase_b[1], color="gray", ls="--", lw=1, alpha=0.4)

    # ================================================================
    # Panel (a): Online reward over time (dev-train stream)
    # ================================================================
    methods = [
        ("oracle", "Oracle", GRAY, 1.5, ":"),
        ("banditgpt", "BanditGPT (Corralling)", BLUE, 2.5, "-"),
        ("warmup_only", "Warmup-only (no Corralling)", PURPLE, 1.8, "-."),
        ("tabula_rasa", "Tabula rasa", TEAL, 1.5, "--"),
        ("ema", "EMA Tracker", ORANGE, 1.8, "-."),
        ("static", f"Static ({failing_display})", RED, 1.8, "--"),
    ]

    for method, label, color, lw, ls in methods:
        mu = _smooth(np.array(ts[method]["mean"]), window)
        std = _smooth(np.array(ts[method]["std"]), window)
        zorder = 5 if method == "banditgpt" else 3
        ax_reward.plot(t_smooth, mu, color=color, lw=lw, ls=ls,
                       label=label, zorder=zorder)
        ax_reward.fill_between(t_smooth, mu - std, mu + std,
                               color=color, alpha=0.10)

    # Holdout evaluation markers (if available)
    holdout_eval = data.get("holdout_eval", {})
    if holdout_eval and "banditgpt" in holdout_eval:
        h_steps = np.array(holdout_eval["steps"])
        h_bg = np.array(holdout_eval["banditgpt"]["mean"])
        ax_reward.scatter(
            h_steps, h_bg, marker="D", s=15, color=BLUE,
            zorder=6, alpha=0.5, label="Holdout eval (BanditGPT)",
        )

    # Failure phase annotation
    bg_fail = data["banditgpt_failure_mean"]
    ema_fail = data["ema_failure_mean"]
    static_fail = data["static_failure_mean"]
    ax_reward.text(
        0.98, 0.03,
        (
            f"Failure phase (dev-train):\n"
            f"  BanditGPT: {bg_fail:.3f}  |  "
            f"EMA: {ema_fail:.3f}  |  "
            f"Static: {static_fail:.3f}"
        ),
        transform=ax_reward.transAxes,
        fontsize=9, fontfamily="monospace", va="bottom", ha="right",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.9),
    )

    ax_reward.set_ylabel(
        f"Online reward ({window}-step avg, dev-train)", fontsize=11,
    )
    ax_reward.set_ylim(0.0, 1.05)
    ax_reward.grid(True, alpha=0.2, ls=":")
    ax_reward.legend(loc="center left", bbox_to_anchor=(0.01, 0.30),
                     fontsize=8.5, framealpha=0.85)
    plt.setp(ax_reward.get_xticklabels(), visible=False)

    sep = data.get("data_separation", {})
    n_h = sep.get("n_holdout", "?")
    fig.suptitle(
        f"Corralling Enables Recovery from Catastrophic LLM Failure "
        f"(K=3, dev-train={n_train}, holdout={n_h})",
        fontsize=14, fontweight="bold", y=0.97,
    )

    # ================================================================
    # Panel (b): Expert weights
    # ================================================================
    ew = data.get("expert_weights", {})
    if ew:
        w_warmup = np.array(ew["warmup_mean"])
        w_warmup_std = np.array(ew["warmup_std"])
        w_tr = np.array(ew["tabula_rasa_mean"])
        w_tr_std = np.array(ew["tabula_rasa_std"])

        ax_weights.plot(t, w_warmup, color=RED, lw=2,
                        label="Warmup expert (priors)")
        ax_weights.fill_between(
            t, w_warmup - w_warmup_std, w_warmup + w_warmup_std,
            color=RED, alpha=0.12,
        )
        ax_weights.plot(t, w_tr, color=GREEN, lw=2,
                        label="Tabula rasa expert")
        ax_weights.fill_between(
            t, w_tr - w_tr_std, w_tr + w_tr_std,
            color=GREEN, alpha=0.12,
        )
        ax_weights.axhline(0.5, color="gray", ls=":", alpha=0.3)

    ax_weights.set_ylabel(r"Expert weight $p_{i,t}$", fontsize=11)
    ax_weights.set_ylim(-0.05, 1.05)
    ax_weights.grid(True, alpha=0.2, ls=":")
    ax_weights.legend(loc="upper left", fontsize=9, framealpha=0.85)
    plt.setp(ax_weights.get_xticklabels(), visible=False)

    # ================================================================
    # Panel (c): Model selection fractions (BanditGPT)
    # ================================================================
    sel = data.get("model_selection_banditgpt", {})
    for i, (display_name, frac_list) in enumerate(sel.items()):
        frac = np.array(frac_list)
        smoothed = _smooth(frac, window)
        color = MODEL_COLORS[i % len(MODEL_COLORS)]
        ax_selection.plot(
            t_smooth, smoothed, color=color, lw=1.5,
            label=display_name,
        )

    ax_selection.set_ylabel("Selection fraction", fontsize=11)
    ax_selection.set_xlabel("Online routing step (dev-train prompts)", fontsize=11)
    ax_selection.set_ylim(-0.05, 1.05)
    ax_selection.grid(True, alpha=0.2, ls=":")

    ncol = 3 if len(sel) > 5 else 2
    ax_selection.legend(
        loc="upper right", fontsize=7.5, framealpha=0.85,
        ncol=ncol, columnspacing=1.0,
    )

    fig.align_ylabels([ax_reward, ax_weights, ax_selection])
    out_path = output_dir / "figure5_catastrophic_failure.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    data = load_results()
    plot_figure5(data, RESULTS_DIR)
