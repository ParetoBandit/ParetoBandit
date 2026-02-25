#!/usr/bin/env python3
"""
Figure 2: The banditGPT Router Architecture

Clean vertical pipeline diagram for KDD.
Wong (2011) colorblind-safe palette throughout.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path
import numpy as np

PAL = dict(
    blue="#0072B2",
    sky="#2171B5",        # darkened from #56B4E9 for text contrast on white
    sky_fill="#d0e4f5",   # light fill for sky-themed boxes
    vermillion="#D55E00",
    teal="#006D5B",       # darkened from #009E73 for text contrast on white
    teal_fill="#c8ead3",  # light fill
    orange="#C67A00",     # darkened from #E69F00 for text contrast on white
    orange_fill="#fdf2d0",
    rpur="#9B4577",       # darkened from #CC79A7 for text contrast on white
    rpur_fill="#f2dcea",
    gray="#555555",       # darkened from #888888
    dk="#1a1a1a", md="#333333", lt="#555555",
)


def create_figure():
    fig, ax = plt.subplots(figsize=(10, 11.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 120)
    ax.axis("off")

    # ─── Drawing Primitives ──────────────────────────────────

    def rbox(cx, cy, w, h, title, subtitle=None, fc="white", ec="#333",
             tc=None, lw=1.6, fs=11.5, fs2=9.5, z=3, ls="-"):
        tc = tc or PAL["dk"]
        ax.add_patch(FancyBboxPatch(
            (cx - w/2, cy - h/2), w, h,
            boxstyle="round,pad=0.4", fc=fc, ec=ec,
            lw=lw, zorder=z, linestyle=ls))
        if subtitle:
            ax.text(cx, cy + h*0.15, title, ha="center", va="center",
                    fontsize=fs, fontweight="bold", color=tc, zorder=z+1)
            ax.text(cx, cy - h*0.22, subtitle, ha="center", va="center",
                    fontsize=fs2, color=PAL["md"], fontstyle="italic",
                    zorder=z+1)
        else:
            ax.text(cx, cy, title, ha="center", va="center",
                    fontsize=fs, fontweight="bold", color=tc, zorder=z+1)

    def arr(x1, y1, x2, y2, c="#333", lw=1.6, ls="-", rad=0.0, ms=14):
        con = f"arc3,rad={rad}" if rad else "arc3,rad=0"
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>",
            connectionstyle=con, color=c, lw=lw,
            mutation_scale=ms, zorder=10, linestyle=ls))

    def seg(x1, y1, x2, y2, c="#333", lw=1.6, ls="-"):
        ax.plot([x1, x2], [y1, y2], color=c, lw=lw, ls=ls,
                zorder=10, solid_capstyle="round")

    def label(x, y, s, fs=10, c=None, fw="normal", ha="center",
              va="center", fam=None, sty="normal", bg=False):
        kw = dict(fontsize=fs, color=c or PAL["md"], fontweight=fw,
                  ha=ha, va=va, fontfamily=fam or "sans-serif",
                  fontstyle=sty, zorder=12)
        if bg:
            kw["bbox"] = dict(boxstyle="round,pad=0.2", fc="white",
                              ec="none", alpha=0.92)
        ax.text(x, y, s, **kw)

    # ─── Layout Constants ────────────────────────────────────
    cx = 50                # pipeline center x
    pw = 42                # pipeline box width
    bh = 7                 # standard box height
    fb_x = 92              # feedback spine x

    # Y positions (top to bottom)
    y_prompt = 113
    y_feat   = 101
    y_filter = 89
    y_coord  = 76
    y_exp    = 60          # experts row (taller boxes)
    y_ucb    = 44
    y_llms   = 30          # LLM portfolio row
    y_reward = 14

    # ═════════════════════════════════════════════════════════
    #  1. USER PROMPT
    # ═════════════════════════════════════════════════════════
    rbox(cx, y_prompt, 30, bh, "User Prompt",
         r"Query $q_t$  +  constraints",
         fc=PAL["sky_fill"], ec=PAL["sky"], tc=PAL["sky"])

    arr(cx, y_prompt - bh/2, cx, y_feat + bh/2, c=PAL["dk"])

    # ═════════════════════════════════════════════════════════
    #  2. FEATURE EXTRACTION
    # ═════════════════════════════════════════════════════════
    rbox(cx, y_feat, pw, bh, "Feature Extraction",
         r"Sentence-BERT  +  PCA  $\rightarrow\;  x_t \in \mathbb{R}^{33}$",
         fc=PAL["sky_fill"], ec=PAL["sky"], tc=PAL["sky"])

    arr(cx, y_feat - bh/2, cx, y_filter + bh/2, c=PAL["dk"])

    # ═════════════════════════════════════════════════════════
    #  3. CONSTRAINT & PARETO FILTER
    # ═════════════════════════════════════════════════════════
    rbox(cx, y_filter, pw, bh, "Constraint & Pareto Filter",
         r"Cost, latency, quality  $\rightarrow$  candidate set $\mathcal{A}_t$",
         ec=PAL["dk"], tc=PAL["dk"])

    # Model Registry (small, to the right)
    mr_x = cx + pw/2 + 12
    rbox(mr_x, y_filter, 14, 5, "Model\nRegistry",
         fc="#e8e8e8", ec=PAL["gray"], tc=PAL["gray"],
         lw=1.0, fs=9, ls="--")
    arr(mr_x - 7, y_filter, cx + pw/2, y_filter, c=PAL["gray"], lw=1.0, ls="-.")

    arr(cx, y_filter - bh/2, cx, y_coord + bh/2, c=PAL["dk"])

    # ═════════════════════════════════════════════════════════
    #  4. CORRALLING COORDINATOR
    # ═════════════════════════════════════════════════════════
    rbox(cx, y_coord, pw, bh, "Corralling Coordinator",
         "Exp4 meta-learner over expert portfolio",
         fc="#fde8d0", ec=PAL["vermillion"], tc=PAL["vermillion"], lw=2.2)

    # Mixing equation (left side, y-centered with the coordinator box)
    label(cx - pw/2 - 3, y_coord,
          r"$p_{i,t} = (1{-}\gamma)\,\frac{w_{i,t}}{\sum_j w_j} + \frac{\gamma}{K}$",
          fs=14, c=PAL["vermillion"], fam="serif", ha="right", bg=True)

    # Coordinator → Experts (fan out)
    e1_x, e2_x = 33, 67
    ew, e_h = 28, 10

    arr(cx - 6, y_coord - bh/2, e1_x + 3, y_exp + e_h/2,
        c=PAL["teal"], lw=1.5, rad=-0.08)
    arr(cx + 6, y_coord - bh/2, e2_x - 3, y_exp + e_h/2,
        c=PAL["orange"], lw=1.5, rad=0.08)

    # Probability labels
    label(e1_x + 8, y_exp + e_h/2 + 2.5, "$p_t(1)$",
          fs=11, c=PAL["teal"], fw="bold", bg=True)
    label(e2_x - 8, y_exp + e_h/2 + 2.5, "$p_t(2)$",
          fs=11, c=PAL["orange"], fw="bold", bg=True)

    # ═════════════════════════════════════════════════════════
    #  5. EXPERTS  (side by side)
    # ═════════════════════════════════════════════════════════
    rbox(e1_x, y_exp, ew, e_h, "Expert 1: Warmup",
         "Hybrid LinUCB\nwith offline priors",
         fc=PAL["teal_fill"], ec=PAL["teal"], tc=PAL["teal"])

    rbox(e2_x, y_exp, ew, e_h, "Expert 2: Tabula Rasa",
         r"Hybrid LinUCB" + "\n" + r"$A_m\!=\!\lambda I,\; b_m\!=\!0$",
         fc=PAL["orange_fill"], ec=PAL["orange"], tc=PAL["orange"])

    ucb_w = 56
    ucb_h = 10

    # Experts → UCB scoring (fan in)
    arr(e1_x + 3, y_exp - e_h/2, cx - 8, y_ucb + ucb_h/2,
        c=PAL["teal"], lw=1.5, rad=0.08)
    arr(e2_x - 3, y_exp - e_h/2, cx + 8, y_ucb + ucb_h/2,
        c=PAL["orange"], lw=1.5, rad=-0.08)

    # ═════════════════════════════════════════════════════════
    #  6. COST-AWARE HYBRID LinUCB
    # ═════════════════════════════════════════════════════════
    ax.add_patch(FancyBboxPatch(
        (cx - ucb_w/2, y_ucb - ucb_h/2), ucb_w, ucb_h,
        boxstyle="round,pad=0.4", fc="white", ec=PAL["dk"],
        lw=1.6, zorder=3))
    ax.text(cx, y_ucb + ucb_h*0.26, "Cost-Aware Hybrid LinUCB",
            ha="center", va="center", fontsize=13, fontweight="bold",
            color=PAL["dk"], zorder=4)
    ax.text(cx, y_ucb - ucb_h*0.18,
            r"$a_t = \arg\max_{a \in \mathcal{A}_t}"
            r"\!\left[\, x_t^\top\!(\hat{\beta}_F \!+\! \hat{\theta}_a)"
            r" + \alpha\!\sqrt{x_t^\top\! A_a^{-1} x_t}"
            r" - \lambda c_a \,\right]$",
            ha="center", va="center", fontsize=11,
            color=PAL["md"], fontstyle="italic", zorder=4)

    arr(cx, y_ucb - ucb_h/2, cx, y_llms + 3.5, c=PAL["dk"])

    # ═════════════════════════════════════════════════════════
    #  7. LLM PORTFOLIO  (row of model cards)
    # ═════════════════════════════════════════════════════════
    models = [
        ("Llama-3",  PAL["teal"],   False),
        ("Claude-3", PAL["sky"],    False),
        ("GPT-4o",   PAL["blue"],   True),
        ("Mixtral",  PAL["orange"], False),
        ("Gemini-3", PAL["gray"],   False),
    ]
    card_w, card_h = 14.5, 5.5
    gap = 2.0
    n = len(models)
    total_w = n * card_w + (n - 1) * gap
    x0 = cx - total_w / 2

    # "LLM Portfolio" label (left-aligned, off the center arrow)
    label(x0 - 1, y_llms, "LLM Portfolio\n(example)",
          fs=9.5, c=PAL["lt"], sty="italic", ha="right")

    for i, (name, col, selected) in enumerate(models):
        card_cx = x0 + card_w/2 + i * (card_w + gap)
        card_cy = y_llms

        if selected:
            fc_, ec_, lw_, tc_ = "#d4e6f1", PAL["blue"], 2.5, PAL["blue"]
            fs_ = 10
        else:
            fc_, ec_, lw_, tc_ = "white", col, 1.2, col
            fs_ = 9.5

        ax.add_patch(FancyBboxPatch(
            (card_cx - card_w/2, card_cy - card_h/2), card_w, card_h,
            boxstyle="round,pad=0.3", fc=fc_, ec=ec_, lw=lw_, zorder=3))
        ax.text(card_cx, card_cy, name, ha="center", va="center",
                fontsize=fs_, fontweight="bold" if selected else "normal",
                color=tc_, zorder=4)

        if selected:
            pass  # bold border + fill already marks the selected card

    # Arrow from selected card to reward (offset right to avoid card center)
    arr(cx, y_llms - card_h/2 - 2.5, cx, y_reward + bh/2,
        c=PAL["blue"], lw=2.0)
    label(cx + 3.5, (y_llms - card_h/2 - 2.5 + y_reward + bh/2)/2,
          "$a_t$", fs=12, c=PAL["blue"], fw="bold", fam="serif")

    # ═════════════════════════════════════════════════════════
    #  8. REWARD
    # ═════════════════════════════════════════════════════════
    rbox(cx, y_reward, 30, bh, "Reward Signal",
         r"Quality score $r_t \in [0,1]$",
         fc=PAL["rpur_fill"], ec=PAL["rpur"], tc=PAL["rpur"])

    # ═════════════════════════════════════════════════════════
    #  FEEDBACK LOOP  (single clean path up the right side)
    # ═════════════════════════════════════════════════════════

    # Reward → spine (horizontal)
    arr(cx + 15, y_reward, fb_x, y_reward, c=PAL["rpur"], lw=2.0, ls="--")

    # Spine up (vertical to coordinator level)
    seg(fb_x, y_reward, fb_x, y_coord, c=PAL["rpur"], lw=2.0, ls="--")

    # → Coordinator (single arrow into right edge)
    arr(fb_x, y_coord, cx + pw/2, y_coord,
        c=PAL["rpur"], lw=1.8, ls="--")

    # Label the feedback loop clearly (spaced apart)
    label(fb_x + 1.5, (y_reward + y_coord) / 2 + 6,
          "Online\nFeedback\n$r_t$", fs=10, c=PAL["rpur"],
          fw="bold", ha="left")
    label(fb_x + 1.5, (y_reward + y_coord) / 2 - 6,
          "Updates coordinator\nweights & expert\nparameters", fs=10.5,
          c=PAL["rpur"], ha="left", sty="italic")

    # (Context x_t flows through the pipeline implicitly —
    #  no separate side-channel needed.)

    # ═════════════════════════════════════════════════════════
    #  LEGEND  (bottom-left)
    # ═════════════════════════════════════════════════════════
    lx, ly = 3, 3
    ax.add_patch(FancyBboxPatch(
        (lx, ly), 28, 7, boxstyle="round,pad=0.3",
        fc="white", ec="#ccc", lw=0.8, zorder=3))
    label(lx + 14, ly + 5.8, "Legend", fs=9.5, fw="bold", c=PAL["dk"])

    for i, (col, ls_, desc) in enumerate([
        (PAL["dk"],   "-",  "Forward pass"),
        (PAL["rpur"], "--", "Online feedback"),
    ]):
        yy = ly + 3.8 - i * 2.0
        ax.plot([lx + 1.5, lx + 5.5], [yy, yy], color=col,
                lw=1.8, ls=ls_, zorder=5, solid_capstyle="round")
        label(lx + 6.5, yy, desc, fs=9, c=PAL["dk"], ha="left")

    return fig


def main():
    out = Path(__file__).parent / "results"
    out.mkdir(parents=True, exist_ok=True)
    fig = create_figure()

    for name, dpi in [("figure2_architecture.png", 300),
                      ("figure2_architecture_hires.png", 600)]:
        p = out / name
        fig.savefig(p, dpi=dpi, bbox_inches="tight", facecolor="white",
                    pad_inches=0.05)
        print(f"Saved: {p}")

    p = out / "figure2_architecture.pdf"
    fig.savefig(p, bbox_inches="tight", facecolor="white", pad_inches=0.05)
    print(f"Saved: {p}")

    paper_dir = Path(__file__).parent.parent.parent / "paper" / "figures"
    if paper_dir.exists():
        import shutil
        for src, dst in [
            (out / "figure2_architecture.png",
             paper_dir / "figure2_architecture.png"),
            (out / "figure2_architecture.pdf",
             paper_dir / "figure2_architecture.pdf"),
        ]:
            shutil.copy2(src, dst)
            print(f"  -> {dst}")

    plt.close()
    print("Done.")


if __name__ == "__main__":
    main()
