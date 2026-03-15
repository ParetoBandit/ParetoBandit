#!/usr/bin/env python3
"""Monitor mid-tier candidate reward collection and report early statistics.

Reads the partial JSONL files and computes running mean reward, comparing
against the Llama-8B and Gemini-Pro reference values from the pareto dataset.

Usage
-----
    python data_collection/scripts/check_midtier_progress.py
    watch -n 60 python data_collection/scripts/check_midtier_progress.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

PARETO_CLASSIFIED = (
    PROJECT_ROOT / "data_collection" / "pareto_dataset" / "pareto_classified.jsonl"
)
MIDTIER_DIR = PROJECT_ROOT / "data_collection" / "midtier_candidates"

CANDIDATES = [
    "microsoft/phi-4",
    "meta-llama/llama-3.1-70b-instruct",
    "google/gemma-3-27b-it",
]


def get_reference_rewards() -> dict[str, float]:
    """Compute mean rewards for Llama-8B and Gemini-Pro from pareto data."""
    llama, gemini = [], []
    with open(PARETO_CLASSIFIED) as f:
        for line in f:
            r = json.loads(line)
            arms = r.get("arms", {})
            li = arms.get("meta-llama/llama-3.1-8b-instruct", {})
            gi = arms.get("google/gemini-2.5-pro", {})
            if "reward" in li:
                llama.append(li["reward"])
            if "reward" in gi:
                gemini.append(gi["reward"])
    return {
        "llama": np.mean(llama),
        "gemini": np.mean(gemini),
    }


def load_candidate_rewards(model_id: str) -> list[float]:
    """Load valid rewards from a candidate's partial JSONL."""
    slug = model_id.replace("/", "_")
    path = MIDTIER_DIR / f"{slug}_rewards.jsonl"
    if not path.exists():
        return []
    rewards = []
    with open(path) as f:
        for line in f:
            try:
                entry = json.loads(line)
                if entry.get("ok") and "raw_score" in entry:
                    rewards.append(entry["raw_score"])
            except (json.JSONDecodeError, KeyError):
                continue
    return rewards


def main() -> None:
    ref = get_reference_rewards()
    llama_r = ref["llama"]
    gemini_r = ref["gemini"]
    gap = gemini_r - llama_r

    print(f"\n{'=' * 78}")
    print("MID-TIER CANDIDATE COLLECTION PROGRESS")
    print(f"{'=' * 78}")
    print(f"  Llama-8B ref:   {llama_r:.4f}")
    print(f"  Gemini-Pro ref: {gemini_r:.4f}")
    print(f"  Total gap:      {gap:.4f}")
    print()
    print(
        f"  {'Model':<38s} {'N':>6s} {'Mean':>7s} {'±SE':>7s} "
        f"{'Δ_Llama':>8s} {'Δ_Gem':>7s} {'Mid%':>6s}  Verdict"
    )
    print(f"  {'-' * 76}")

    for model_id in CANDIDATES:
        rewards = load_candidate_rewards(model_id)
        short = model_id.split("/")[-1]
        if len(rewards) < 5:
            print(f"  {short:<38s} {len(rewards):6d}  [waiting for data...]")
            continue

        arr = np.array(rewards)
        mean_r = float(arr.mean())
        se = float(arr.std(ddof=1) / np.sqrt(len(arr)))
        d_llama = mean_r - llama_r
        d_gemini = gemini_r - mean_r
        mid_pct = (mean_r - llama_r) / gap * 100

        if d_llama < 0.03:
            verdict = "TOO CLOSE to Llama"
        elif d_gemini < 0.03:
            verdict = "TOO CLOSE to Gemini"
        elif 25 < mid_pct < 75:
            verdict = "GOOD separation"
        elif 15 < mid_pct < 85:
            verdict = "OK separation"
        else:
            verdict = "POOR separation"

        print(
            f"  {short:<38s} {len(arr):6d} {mean_r:7.4f} {se:7.4f} "
            f"{d_llama:+8.4f} {-d_gemini:+7.4f} {mid_pct:5.1f}%  {verdict}"
        )

    print(f"\n  Target: 11,983 prompts per model. Mid% 30-60% is ideal.")
    print(f"{'=' * 78}\n")


if __name__ == "__main__":
    main()
