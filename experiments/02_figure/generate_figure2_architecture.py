#!/usr/bin/env python3
"""
Figure 2: The banditGPT Router Architecture

Publication-quality architecture diagram for KDD. Clean, minimal design
with no overlapping elements. Shows the three-layer routing pipeline,
data sources, and two-level feedback loop.

Usage:
    python3 experiments/02_figure/generate_figure2_architecture.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════
#  COLORS — Wong (2011) colorblind-safe palette (Nature Methods)
#  Verified for protanopia, deuteranopia, and tritanopia.
# ═══════════════════════════════════════════════════════════════════

PAL = dict(
    blue="#0072B2",       # input / output boxes
    sky="#56B4E9",        # feature extraction pipeline
    vermillion="#D55E00", # coordinator (key innovation)
    teal="#009E73",       # warmup expert
    orange="#E69F00",     # tabula rasa expert
    rpur="#CC79A7",       # feedback loop
    gray="#505050",       # model registry
    dk="#1a1a1a", md="#333333", lt="#555555",
)


def create_figure():
    fig, ax = plt.subplots(figsize=(7.2, 10.0))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # ─── Helpers ──────────────────────────────────────────────

    def box(x, y, w, h, label, fc="white", ec="#333", tc="#1a1a1a",
            lw=1.4, fs=9, fw="bold", z=3):
        """Simple rounded box with ONE centered label."""
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.4",
            fc=fc, ec=ec, lw=lw, zorder=z))
        ax.text(x + w/2, y + h/2, label, ha="center", va="center",
                fontsize=fs, fontweight=fw, color=tc, zorder=z+1)

    def box2(x, y, w, h, line1, line2, fc="white", ec="#333",
             tc="#1a1a1a", tc2=None, lw=1.4, fs=9, fs2=7, fw="bold", z=3):
        """Rounded box with title + subtitle."""
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.4",
            fc=fc, ec=ec, lw=lw, zorder=z))
        ax.text(x + w/2, y + h*0.62, line1, ha="center", va="center",
                fontsize=fs, fontweight=fw, color=tc, zorder=z+1)
        ax.text(x + w/2, y + h*0.28, line2, ha="center", va="center",
                fontsize=fs2, color=tc2 or PAL["md"], zorder=z+1,
                fontstyle="italic")

    def arrow(x1, y1, x2, y2, c="#333", lw=1.4, ls="-", rad=0):
        con = f"arc3,rad={rad}" if rad else "arc3,rad=0"
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>",
            connectionstyle=con, color=c, lw=lw,
            mutation_scale=11, zorder=2, linestyle=ls))

    def line(x1, y1, x2, y2, c="#333", lw=1.2, ls="--"):
        ax.plot([x1, x2], [y1, y2], color=c, lw=lw, ls=ls, zorder=2)

    def txt(x, y, s, fs=7, c=None, fw="normal", ha="center", va="center",
            fam=None, sty="normal"):
        ax.text(x, y, s, fontsize=fs, color=c or PAL["md"],
                fontweight=fw, ha=ha, va=va, fontfamily=fam or "sans-serif",
                fontstyle=sty, zorder=5)

    # ─── Layout coordinates ───────────────────────────────────
    cx = 50              # pipeline center
    bw, bh = 32, 5.0    # standard box size

    # Y positions (top to bottom, well-spaced)
    y_prompt  = 93
    y_embed   = 84
    y_pca     = 75
    y_pareto  = 64
    y_coord   = 53
    y_exp     = 40       # expert row
    y_ucb     = 27
    y_model   = 11
    y_reward  = 1

    # ═════════════════════════════════════════════════════════
    #  MAIN PIPELINE (top → bottom)
    # ═════════════════════════════════════════════════════════

    # 1. User Prompt
    box2(cx-bw/2, y_prompt, bw, bh,
         "User Prompt", "natural language query",
         fc="#d4e6f1", ec=PAL["blue"], tc=PAL["blue"], lw=1.8)

    # 2. Sentence Transformer
    box2(cx-bw/2, y_embed, bw, bh,
         "Sentence Transformer", "all-MiniLM-L6-v2  \u2192  384-D",
         ec=PAL["sky"], tc=PAL["sky"])

    arrow(cx, y_prompt, cx, y_embed + bh, c=PAL["dk"])

    # 3. PCA
    box2(cx-bw/2, y_pca, bw, bh,
         "Domain-Adapted PCA", "384-D  \u2192  33-D context vector $x_t$",
         ec=PAL["sky"], tc=PAL["sky"])

    arrow(cx, y_embed, cx, y_pca + bh, c=PAL["dk"])

    # PCA note (to the left, out of the way)
    txt(cx - bw/2 - 2, y_pca + bh/2,
        "Trained on 80K\nRouteLLM prompts\n(unsupervised)",
        fs=6, c=PAL["lt"], ha="right", sty="italic")

    # ═════════════════════════════════════════════════════════
    #  CONTEXT VECTOR ANNOTATION (right side)
    # ═════════════════════════════════════════════════════════

    ctx_x, ctx_y, ctx_w, ctx_h = 70, 74, 28, 10
    ax.add_patch(FancyBboxPatch(
        (ctx_x, ctx_y), ctx_w, ctx_h, boxstyle="round,pad=0.3",
        fc="#eaf4fb", ec=PAL["sky"], lw=0.8, zorder=3, linestyle="--"))
    txt(ctx_x + ctx_w/2, ctx_y + ctx_h - 1.3,
        "Context Vector $x_t \\in \\mathbb{R}^{33}$",
        fs=6.5, fw="bold", c=PAL["sky"])
    txt(ctx_x + ctx_w/2, ctx_y + ctx_h - 3.0,
        "[PCA$_0$, \u2026, PCA$_{31}$, bias]",
        fs=6, c=PAL["md"], fam="serif")

    for i, feat in enumerate([
        "\u2022  task type (math, code, creative, \u2026)",
        "\u2022  query complexity & difficulty",
        "\u2022  semantic style & specificity",
    ]):
        txt(ctx_x + 1.5, ctx_y + ctx_h - 5.0 - i*1.7,
            feat, fs=5.5, c=PAL["md"], ha="left")

    # Dashed connector from x_t label area to annotation box
    line(cx + bw/2 + 1, y_pca + bh/2, ctx_x - 0.5, ctx_y + ctx_h/2,
         c=PAL["sky"], lw=0.8, ls="--")

    # 4. Dynamic Pareto Filter (L1)
    pw = 40
    box2(cx-pw/2, y_pareto, pw, bh + 1,
         "Layer 1:  Dynamic Pareto Filter",
         "prune dominated models per-context",
         ec=PAL["dk"], tc=PAL["dk"], fs=8.5)

    arrow(cx, y_pca, cx, y_pareto + bh + 1, c=PAL["dk"])
    txt(cx + 3, (y_pca + y_pareto + bh + 1)/2 + 0.3,
        "$x_t$", fs=9, c=PAL["md"], ha="left", fam="serif")

    # 5. Corralling Coordinator (L2)
    cw = 40
    box2(cx-cw/2, y_coord, cw, bh + 1.5,
         "Layer 2:  Corralling Coordinator",
         "meta-learner over expert portfolio",
         fc="#fde8d0", ec=PAL["vermillion"], tc=PAL["vermillion"],
         lw=2.0, fs=8.5)

    arrow(cx, y_pareto, cx, y_coord + bh + 1.5, c=PAL["dk"])
    txt(cx + 3, (y_pareto + y_coord + bh + 1.5)/2 + 0.3,
        "$\\mathcal{P}_{x_t}$", fs=8, c=PAL["lt"], ha="left", fam="serif")

    # Mixing equation to the left of coordinator
    txt(cx - cw/2 - 1.5, y_coord + (bh+1.5)*0.35,
        r"$p_{i,t} = (1{-}\gamma)\frac{w_{i,t}}{\Sigma_j w_j} + \frac{\gamma}{K}$",
        fs=8, c=PAL["vermillion"], fam="serif", ha="right")

    # 6. Expert Bandits (two side-by-side)
    ew, eh = 26, 7.5
    egap = 5
    e1x = cx - ew - egap/2
    e2x = cx + egap/2

    # Expert 1: Warmup
    ax.add_patch(FancyBboxPatch(
        (e1x, y_exp), ew, eh, boxstyle="round,pad=0.4",
        fc="#d5f0db", ec=PAL["teal"], lw=1.6, zorder=3))
    txt(e1x + ew/2, y_exp + eh*0.78,
        "Expert 1: Warmup", fs=8.5, fw="bold", c=PAL["teal"])
    txt(e1x + ew/2, y_exp + eh*0.52,
        "LinUCB with offline priors", fs=6.5, c=PAL["md"], sty="italic")
    txt(e1x + ew/2, y_exp + eh*0.25,
        "$A_m, b_m$ from RouteLLM + $\\lambda I$",
        fs=6, c=PAL["md"], fam="serif")

    # Expert 2: Tabula Rasa
    ax.add_patch(FancyBboxPatch(
        (e2x, y_exp), ew, eh, boxstyle="round,pad=0.4",
        fc="#fdf2d0", ec=PAL["orange"], lw=1.6, zorder=3))
    txt(e2x + ew/2, y_exp + eh*0.78,
        "Expert 2: Tabula Rasa", fs=8.5, fw="bold", c="#b07d00")
    txt(e2x + ew/2, y_exp + eh*0.52,
        "LinUCB, no priors", fs=6.5, c=PAL["md"], sty="italic")
    txt(e2x + ew/2, y_exp + eh*0.25,
        "$A_m = \\lambda I$,  $b_m = 0$",
        fs=6, c=PAL["md"], fam="serif")

    # Arrows: Coordinator → Experts
    arrow(cx - 5, y_coord, e1x + ew/2, y_exp + eh,
          c=PAL["teal"], lw=1.2, rad=0.12)
    arrow(cx + 5, y_coord, e2x + ew/2, y_exp + eh,
          c=PAL["orange"], lw=1.2, rad=-0.12)

    # Probability labels above experts
    txt(e1x + ew/2 - 1, y_exp + eh + 1.8,
        "$p_t(1)$", fs=8, c=PAL["teal"], fw="bold", fam="serif")
    txt(e2x + ew/2 + 1, y_exp + eh + 1.8,
        "$p_t(2)$", fs=8, c="#b07d00", fw="bold", fam="serif")

    # 7. UCB Selection (L3)
    uw = 50
    box2(cx-uw/2, y_ucb, uw, bh + 1.5,
         "Layer 3:  Cost-Aware LinUCB Selection",
         r"$a_t = \arg\max_a\; \hat{\theta}_a^\top x_t"
         r" + \alpha\sqrt{x_t^\top A_a^{-1} x_t} - \lambda c_a$",
         ec=PAL["dk"], tc=PAL["dk"], fs=8.5, fs2=6.5)

    arrow(e1x + ew/2, y_exp, cx - 5, y_ucb + bh + 1.5,
          c=PAL["dk"], lw=1.1, rad=0.08)
    arrow(e2x + ew/2, y_exp, cx + 5, y_ucb + bh + 1.5,
          c=PAL["dk"], lw=1.1, rad=-0.08)

    # 8. Selected Model
    mw = 34
    box2(cx-mw/2, y_model, mw, bh,
         "Selected LLM  \u2192  Response",
         "GPT-4o / Claude / Mixtral / Llama / \u2026",
         fc="#d4e6f1", ec=PAL["blue"], tc=PAL["blue"], lw=1.8)

    # ─── Arm Nodes (bandit arms between UCB and selected LLM) ───
    y_arms = 21
    aw, ah = 9, 3.0
    agap = 1.8
    arms_data = [
        # (label,    colour,        status,   selected?)
        ("Llama",    PAL["teal"],   "active",  False),
        ("Claude",   PAL["sky"],    "active",  False),
        ("GPT-4o",   PAL["blue"],   "active",  True),   # ← selected
        ("Mixtral",  PAL["orange"], "active",  False),
        ("Gemini",   PAL["gray"],   "pruned",  False),  # ← Pareto-pruned
    ]
    n_arms = len(arms_data)
    total_aw = n_arms * aw + (n_arms - 1) * agap
    arms_x0 = cx - total_aw / 2

    for i, (name, col, status, selected) in enumerate(arms_data):
        bx = arms_x0 + i * (aw + agap)
        by = y_arms - ah / 2
        if status == "pruned":
            fc_, ec_, lw_ = "#f5f5f5", "#ccc", 0.8
            tc_ = "#bbb"
        elif selected:
            fc_, ec_, lw_ = "#d4e6f1", PAL["blue"], 2.2
            tc_ = PAL["blue"]
        else:
            fc_, ec_, lw_ = "white", col, 1.2
            tc_ = col
        ax.add_patch(FancyBboxPatch(
            (bx, by), aw, ah, boxstyle="round,pad=0.3",
            fc=fc_, ec=ec_, lw=lw_, zorder=3))
        txt(bx + aw/2, by + ah/2, name,
            fs=6, fw="bold" if selected else "normal", c=tc_)
        # Strikethrough on pruned arm
        if status == "pruned":
            ax.plot([bx + 1.0, bx + aw - 1.0],
                    [by + ah/2, by + ah/2],
                    color="#bbb", lw=1.2, zorder=6)

    # "pruned" annotation under the Pareto-filtered arm
    pruned_cx = arms_x0 + 4 * (aw + agap) + aw / 2
    txt(pruned_cx, y_arms - ah/2 - 1.3,
        "pruned", fs=5, c="#aaa", sty="italic")

    # Side label
    txt(arms_x0 + total_aw + 1.5, y_arms,
        "bandit arms", fs=6, c=PAL["lt"], sty="italic", ha="left")

    # Arrow:  UCB → arm row
    arrow(cx, y_ucb, cx, y_arms + ah/2 + 0.8, c=PAL["dk"], lw=1.4)
    # Arrow:  selected arm → model box
    arrow(cx, y_arms - ah/2 - 0.5, cx, y_model + bh,
          c=PAL["blue"], lw=1.6)
    txt(cx + 3, (y_arms - ah/2 + y_model + bh) / 2,
        "$a_t$", fs=9, c=PAL["md"], fw="bold", ha="left", fam="serif")

    # 9. Reward
    rw = 30
    box2(cx-rw/2, y_reward, rw, bh,
         "Reward Observation",
         "$r_t \\in [0,1]$  (quality score)",
         fc="#f2e2ef", ec=PAL["rpur"], tc=PAL["rpur"], lw=1.6)

    arrow(cx, y_model, cx, y_reward + bh, c=PAL["dk"], lw=1.4)

    # ═════════════════════════════════════════════════════════
    #  FEEDBACK LOOP (right side, going up)
    # ═════════════════════════════════════════════════════════

    fx = 88  # vertical feedback line

    # Horizontal from reward box → right
    line(cx + rw/2 + 0.5, y_reward + bh/2, fx, y_reward + bh/2,
         c=PAL["rpur"])

    # Vertical up to coordinator
    line(fx, y_reward + bh/2, fx, y_coord + (bh+1.5)/2,
         c=PAL["rpur"])

    # → Expert 2 (expert update)
    arrow(fx, y_exp + eh*0.5, e2x + ew + 0.5, y_exp + eh*0.5,
          c=PAL["rpur"], lw=1.1, ls="--")

    # → Coordinator (meta-weight update)
    arrow(fx, y_coord + (bh+1.5)*0.4,
          cx + cw/2 + 0.5, y_coord + (bh+1.5)*0.4,
          c=PAL["rpur"], lw=1.1, ls="--")

    # Annotation: meta-weight update
    txt(fx + 1, y_coord + (bh+1.5)*0.4 + 2,
        "Meta-Weight", fs=6.5, c=PAL["rpur"], fw="bold", ha="left")
    txt(fx + 1, y_coord + (bh+1.5)*0.4 - 1.5,
        "$\\hat{\\ell}_t = (1{-}r_t)/p_t$",
        fs=7, c=PAL["rpur"], fam="serif", ha="left")

    # Annotation: expert update
    txt(fx + 1, y_exp + eh*0.5 + 2,
        "Expert Update", fs=6.5, c=PAL["rpur"], fw="bold", ha="left")
    txt(fx + 1, y_exp + eh*0.5 - 1.5,
        "$A \\!\\leftarrow\\! A + xx^\\top$",
        fs=7, c=PAL["rpur"], fam="serif", ha="left")

    # ═════════════════════════════════════════════════════════
    #  DATA SOURCES (left side, well-separated)
    # ═════════════════════════════════════════════════════════

    dw, dh = 15, 4.5

    # RouteLLM Battles → Expert 1  (positioned at expert midpoint, shifted down)
    dx1 = 1
    dy1 = y_exp + eh*0.25 - dh/2
    ax.add_patch(FancyBboxPatch(
        (dx1, dy1), dw, dh, boxstyle="round,pad=0.3",
        fc="#e4e4e4", ec=PAL["gray"], lw=1.0, zorder=3, linestyle="--"))
    txt(dx1 + dw/2, dy1 + dh*0.62,
        "RouteLLM Battles", fs=6, fw="bold", c=PAL["dk"])
    txt(dx1 + dw/2, dy1 + dh*0.25,
        "80K offline pairs", fs=5.5, c=PAL["md"], sty="italic")

    arrow(dx1 + dw, dy1 + dh/2, e1x - 0.5, y_exp + eh*0.35,
          c=PAL["gray"], lw=0.9, ls="-.")

    # Model Registry → Pareto Filter
    dx2 = 1
    dy2 = y_pareto + (bh+1)/2 - dh/2
    ax.add_patch(FancyBboxPatch(
        (dx2, dy2), dw, dh, boxstyle="round,pad=0.3",
        fc="#e4e4e4", ec=PAL["gray"], lw=1.0, zorder=3, linestyle="--"))
    txt(dx2 + dw/2, dy2 + dh*0.62,
        "Model Registry", fs=6, fw="bold", c=PAL["dk"])
    txt(dx2 + dw/2, dy2 + dh*0.25,
        "costs & metadata", fs=5.5, c=PAL["md"], sty="italic")

    arrow(dx2 + dw, dy2 + dh/2, cx - pw/2 - 0.5, y_pareto + (bh+1)/2,
          c=PAL["gray"], lw=0.9, ls="-.")

    # ═════════════════════════════════════════════════════════
    #  KEY PARAMETERS (top-right, compact)
    # ═════════════════════════════════════════════════════════

    kx, ky, kw_, kh_ = 71, 90, 28, 9
    ax.add_patch(FancyBboxPatch(
        (kx, ky), kw_, kh_, boxstyle="round,pad=0.3",
        fc="#f0f0f0", ec=PAL["lt"], lw=0.8, zorder=3))
    txt(kx + kw_/2, ky + kh_ - 1.2,
        "Key Parameters", fs=7, fw="bold", c=PAL["dk"])

    for i, (sym, desc) in enumerate([
        ("$\\alpha \\geq 0$",   "exploration"),
        ("$\\eta \\in \\{0.1, 1.0\\}$", "meta-learning rate"),
        ("$\\gamma \\in (0,1)$", "mixing floor"),
        ("$\\lambda \\geq 0$",  "cost sensitivity"),
    ]):
        py = ky + kh_ - 3.2 - i*1.6
        txt(kx + 1.5, py, sym, fs=6, c=PAL["dk"], ha="left", fam="serif")
        txt(kx + 12, py, desc, fs=5.5, c=PAL["lt"], ha="left")

    # ═════════════════════════════════════════════════════════
    #  LEGEND (bottom-right)
    # ═════════════════════════════════════════════════════════

    lx, ly, llw, llh = 69, 1, 30, 6.5
    ax.add_patch(FancyBboxPatch(
        (lx, ly), llw, llh, boxstyle="round,pad=0.3",
        fc="#f0f0f0", ec="#999", lw=0.8, zorder=3))
    txt(lx + llw/2, ly + llh - 1,
        "Legend", fs=6.5, fw="bold", c=PAL["dk"])

    for i, (col, ls, desc) in enumerate([
        (PAL["dk"],         "-",  "Forward pass (routing)"),
        (PAL["rpur"],       "--", "Feedback (online learning)"),
        (PAL["gray"],       "-.", "Data source (offline)"),
    ]):
        yy = ly + llh - 2.8 - i*1.5
        ax.plot([lx+2, lx+7], [yy, yy], color=col, lw=1.5, ls=ls, zorder=5)
        txt(lx + 8.5, yy, desc, fs=5.5, c=PAL["dk"], ha="left")

    return fig


# ═══════════════════════════════════════════════════════════════════
#  SAVE
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("FIGURE 2: banditGPT ROUTER ARCHITECTURE")
    print("=" * 60)

    out = Path(__file__).parent / "results"
    out.mkdir(parents=True, exist_ok=True)
    fig = create_figure()

    for name, dpi in [("figure2_architecture.png", 300),
                      ("figure2_architecture_hires.png", 600)]:
        p = out / name
        fig.savefig(p, dpi=dpi, bbox_inches="tight", facecolor="white",
                    pad_inches=0.08)
        print(f"Saved: {p}")

    p = out / "figure2_architecture.pdf"
    fig.savefig(p, bbox_inches="tight", facecolor="white", pad_inches=0.08)
    print(f"Saved: {p}")

    paper_dir = Path(__file__).parent.parent.parent / "paper" / "figures"
    if paper_dir.exists():
        import shutil
        for src, dst in [
            (out / "figure2_architecture.png",
             paper_dir / "figure3_architecture.png"),
            (out / "figure2_architecture.pdf",
             paper_dir / "figure2_architecture.pdf"),
        ]:
            shutil.copy2(src, dst)
            print(f"Copied to: {dst}")

    plt.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
