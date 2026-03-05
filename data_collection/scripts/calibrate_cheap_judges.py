#!/usr/bin/env python3
"""
Calibrate and validate LLM judge panels.

Provides two workflows:

**run** — Cheap-only calibration: re-judge sampled records with four
cheaper models and compare against the original v1 scores.

**validate** — Apples-to-apples validation: re-judge sampled records
with *both* the expensive panel and the recommended panel using the
same v2 rubric, eliminating the v1-vs-v2 format confound.

Usage
-----
    # Cheap-only calibration (~$1-2)
    python calibrate_cheap_judges.py run --n 200 --seed 42

    # Apples-to-apples validation (~$5-8, 7 judges per record)
    python calibrate_cheap_judges.py validate --n 200 --seed 42

    # Analyze saved results (offline, no API calls)
    python calibrate_cheap_judges.py analyze
    python calibrate_cheap_judges.py analyze --path path/to/file.jsonl
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Model pools
# ---------------------------------------------------------------------------

EXPENSIVE_POOL: Dict[str, str] = {
    "openai": "openai/gpt-4o",
    "anthropic": "anthropic/claude-3.5-sonnet",
    "meta": "meta-llama/llama-3.1-405b-instruct",
    "google": "google/gemini-2.5-pro-preview-06-05",
}

CHEAP_POOL: Dict[str, str] = {
    "openai": "openai/gpt-4o-mini",
    "anthropic": "anthropic/claude-3.5-haiku",
    "meta": "meta-llama/llama-3.3-70b-instruct",
    "google": "google/gemini-2.5-flash",
}

RECOMMENDED_POOL: Dict[str, str] = {
    "openai": "openai/gpt-4o",
    "anthropic": "anthropic/claude-3.5-haiku",
    "meta": "meta-llama/llama-3.3-70b-instruct",
    "google": "google/gemini-2.5-flash",
}

FAMILY_MAP: Dict[str, str] = {
    "gpt": "openai", "o1": "openai", "o3": "openai", "o4": "openai",
    "claude": "anthropic",
    "llama": "meta",
    "gemini": "google", "gemma": "google",
}

# ---------------------------------------------------------------------------
# V2 rubric prompt — identical to rejudge_cot.py
# ---------------------------------------------------------------------------

RUBRIC_SYSTEM_PROMPT: str = (
    "You are a Discriminative Router Judge. Your goal is to find the "
    "failure points in LLM responses.\n\n"
    "Score the response on three factors:\n\n"
    "1. **Logical Integrity (50 %)** — Does the model show its work? "
    "If there is a single calculation or logical-step error, this "
    "factor is 0. No partial credit.\n"
    "2. **Constraint Adherence (30 %)** — Did the model follow ALL "
    "formatting and negative constraints (e.g. \"Do not use the word "
    "'AI'\")? If one constraint is missed, this factor is 0.\n"
    "3. **Utility & Tone (20 %)** — Is the answer helpful and "
    "professional? Score continuously from 0.0 (useless / rude) to "
    "1.0 (maximally helpful and professional).\n\n"
    "Format your response EXACTLY as follows:\n\n"
    "## Reasoning\n"
    "<Concise chain-of-thought analysis identifying any errors or "
    "constraint violations>\n\n"
    "## Logical Integrity\n"
    "<0 or 1>\n\n"
    "## Constraint Adherence\n"
    "<0 or 1>\n\n"
    "## Utility & Tone\n"
    "<0.0 to 1.0>"
)

W_LOGIC: float = 0.5
W_CONSTRAINT: float = 0.3
W_UTILITY: float = 0.2

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
_DATA_COLLECTION_DIR = _SCRIPT_DIR.parent
_PROJECT_ROOT = _DATA_COLLECTION_DIR.parent

DEFAULT_DATA_PATH: Path = (
    _DATA_COLLECTION_DIR / "rewards" / "dev_rewards_complete_all_models.jsonl.gz"
)
DEFAULT_OUTPUT_DIR: Path = _DATA_COLLECTION_DIR / "rewards" / "calibration"


# ===================================================================
# Calibrator
# ===================================================================


class CheapJudgeCalibrator:
    """Sample existing reward records and re-judge with cheaper models."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_workers: int = 32,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            try:
                from dotenv import load_dotenv
                env_path = _PROJECT_ROOT / ".env"
                if env_path.exists():
                    load_dotenv(env_path)
                self.api_key = os.getenv("OPENROUTER_API_KEY")
            except ImportError:
                pass
        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not found. Set it as an env var or in .env"
            )

        self.base_url = "https://openrouter.ai/api/v1"
        self.max_workers = max_workers
        self._lock = threading.Lock()

    # ---- sampling --------------------------------------------------------

    @staticmethod
    def sample_stratified(
        data_path: Path,
        n: int = 200,
        seed: int = 42,
        n_strata: int = 4,
    ) -> List[Dict[str, Any]]:
        """Load reward data and return *n* records stratified by raw_score.

        Parameters
        ----------
        data_path:
            Gzipped (or plain) JSONL reward file.
        n:
            Total records to sample.
        seed:
            Random seed for reproducibility.
        n_strata:
            Number of score strata (quartiles by default).

        Returns
        -------
        list[dict]
            Sampled records with all original fields preserved.
        """
        records: List[Dict[str, Any]] = []
        opener = gzip.open if str(data_path).endswith(".gz") else open
        with opener(data_path, "rt") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not rec.get("ok"):
                    continue
                raw = rec.get("raw_score")
                if raw is None or (isinstance(raw, float) and math.isnan(raw)):
                    continue
                if not rec.get("response"):
                    continue
                records.append(rec)

        print(f"  Loaded {len(records)} valid records from {data_path.name}")

        rng = np.random.default_rng(seed)
        scores = np.array([r["raw_score"] for r in records])

        boundaries = np.percentile(scores, np.linspace(0, 100, n_strata + 1))

        per_stratum = n // n_strata
        remainder = n % n_strata
        sampled: List[Dict[str, Any]] = []

        for i in range(n_strata):
            lo, hi = boundaries[i], boundaries[i + 1]
            if i == n_strata - 1:
                mask = (scores >= lo) & (scores <= hi)
            else:
                mask = (scores >= lo) & (scores < hi)

            stratum_indices = np.where(mask)[0]
            take = per_stratum + (1 if i < remainder else 0)
            take = min(take, len(stratum_indices))
            chosen = rng.choice(stratum_indices, size=take, replace=False)
            sampled.extend(records[int(idx)] for idx in chosen)

        rng.shuffle(sampled)

        score_arr = np.array([r["raw_score"] for r in sampled])
        print(
            f"  Score distribution: "
            f"min={score_arr.min():.2f}  median={np.median(score_arr):.2f}  "
            f"max={score_arr.max():.2f}"
        )
        return sampled

    # ---- single judge call -----------------------------------------------

    def _call_judge(
        self,
        judge_model: str,
        prompt: str,
        response: str,
    ) -> Optional[Dict[str, Any]]:
        """Send (prompt, response) to one cheap judge and parse the rubric.

        Returns
        -------
        dict or None
            Parsed rubric: logic, constraint, utility, reward, reasoning,
            raw_output.  ``None`` on API or parse failure.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/banditgpt/llm-jury",
        }
        payload = {
            "model": judge_model,
            "messages": [
                {"role": "system", "content": RUBRIC_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"PROMPT: {prompt}\n\nRESPONSE: {response}",
                },
            ],
            "temperature": 0.0,
            "max_tokens": 4000,
        }

        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=90,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            return None

        logic = 0
        m = re.search(
            r"##\s*Logical Integrity\s*[:\-]?\s*(\d)", content, re.IGNORECASE,
        )
        if m:
            logic = 1 if int(m.group(1)) == 1 else 0

        constraint = 0
        m = re.search(
            r"##\s*Constraint Adherence\s*[:\-]?\s*(\d)", content, re.IGNORECASE,
        )
        if m:
            constraint = 1 if int(m.group(1)) == 1 else 0

        utility = 0.5
        m = re.search(
            r"##\s*Utility\s*(?:&|and)?\s*Tone\s*[:\-]?\s*(\d+\.?\d*)",
            content,
            re.IGNORECASE,
        )
        if m:
            val = float(m.group(1))
            if val > 1.0:
                val /= 100.0
            utility = max(0.0, min(1.0, val))

        reward = logic * W_LOGIC + constraint * W_CONSTRAINT + utility * W_UTILITY

        reasoning = content
        rm = re.search(
            r"##\s*Reasoning\s*(.*?)(\n##|$)", content, re.DOTALL | re.IGNORECASE,
        )
        if rm:
            reasoning = rm.group(1).strip()

        return {
            "judge": judge_model,
            "logic": logic,
            "constraint": constraint,
            "utility": round(utility, 4),
            "reward": round(reward, 4),
            "reasoning": reasoning,
            "raw_output": content,
        }

    # ---- family resolution -----------------------------------------------

    @staticmethod
    def _get_model_family(model_id: str) -> Optional[str]:
        """Resolve a model_id to a provider family string."""
        lower = model_id.lower()
        for key, family in FAMILY_MAP.items():
            if key in lower:
                return family
        for prefix, family in [
            ("openai/", "openai"),
            ("anthropic/", "anthropic"),
            ("google/", "google"),
            ("meta-llama/", "meta"),
        ]:
            if prefix in lower:
                return family
        return None

    # ---- judge one record ------------------------------------------------

    def judge_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Judge one record with all 4 cheap judges.

        Returns a dict containing original fields, all cheap judge details,
        and two composite rewards (all-judges and family-excluded).
        """
        prompt = record["prompt"]
        response = record["response"]
        model_id = record["model_id"]
        family = self._get_model_family(model_id)

        all_results: List[Dict[str, Any]] = []
        excluded_results: List[Dict[str, Any]] = []

        with ThreadPoolExecutor(max_workers=len(CHEAP_POOL)) as executor:
            futures = {
                executor.submit(
                    self._call_judge, judge_id, prompt, response,
                ): org
                for org, judge_id in CHEAP_POOL.items()
            }
            for future in as_completed(futures):
                org = futures[future]
                parsed = future.result()
                if parsed is None:
                    continue
                parsed["provider"] = org
                all_results.append(parsed)
                if org != family:
                    excluded_results.append(parsed)

        cheap_reward_all = (
            float(np.mean([r["reward"] for r in all_results]))
            if all_results
            else float("nan")
        )
        cheap_reward_excluded = (
            float(np.mean([r["reward"] for r in excluded_results]))
            if excluded_results
            else float("nan")
        )

        return {
            "model_id": model_id,
            "model_family": family,
            "prompt": prompt,
            "response": response,
            "original_judge_details": record.get("judge_details", []),
            "original_raw_score": record.get("raw_score"),
            "original_reward_logit": record.get("reward_logit"),
            "cheap_judge_details": all_results,
            "cheap_reward_all_judges": round(cheap_reward_all, 4),
            "cheap_reward_family_excluded": round(cheap_reward_excluded, 4),
            "ts": time.time(),
        }

    # ---- judge with arbitrary pool ----------------------------------------

    def _judge_with_pool(
        self,
        pool: Dict[str, str],
        prompt: str,
        response: str,
        model_family: Optional[str],
    ) -> Tuple[List[Dict[str, Any]], float, float]:
        """Send (prompt, response) to every judge in *pool*.

        Returns
        -------
        tuple[list[dict], float, float]
            (all_results, reward_all, reward_family_excluded)
        """
        all_results: List[Dict[str, Any]] = []
        excluded_results: List[Dict[str, Any]] = []

        with ThreadPoolExecutor(max_workers=len(pool)) as executor:
            futures = {
                executor.submit(
                    self._call_judge, judge_id, prompt, response,
                ): org
                for org, judge_id in pool.items()
            }
            for future in as_completed(futures):
                org = futures[future]
                parsed = future.result()
                if parsed is None:
                    continue
                parsed["provider"] = org
                all_results.append(parsed)
                if org != model_family:
                    excluded_results.append(parsed)

        reward_all = (
            float(np.mean([r["reward"] for r in all_results]))
            if all_results
            else float("nan")
        )
        reward_excluded = (
            float(np.mean([r["reward"] for r in excluded_results]))
            if excluded_results
            else float("nan")
        )
        return all_results, reward_all, reward_excluded

    # ---- validation: both panels on same record --------------------------

    def judge_record_validation(
        self, record: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Judge one record with both the expensive and recommended panels.

        Sends gpt-4o only once (shared between panels) and fans out the
        remaining 6 unique judges in parallel.
        """
        prompt = record["prompt"]
        response = record["response"]
        model_id = record["model_id"]
        family = self._get_model_family(model_id)

        unique_judges: Dict[str, Tuple[str, str]] = {}
        for org, judge_id in EXPENSIVE_POOL.items():
            unique_judges[f"expensive_{org}"] = (org, judge_id)
        for org, judge_id in RECOMMENDED_POOL.items():
            key = f"recommended_{org}"
            if judge_id not in {v[1] for v in unique_judges.values()}:
                unique_judges[key] = (org, judge_id)

        raw_results: Dict[str, Dict[str, Any]] = {}

        with ThreadPoolExecutor(max_workers=len(unique_judges)) as executor:
            futures = {
                executor.submit(
                    self._call_judge, judge_id, prompt, response,
                ): (key, org)
                for key, (org, judge_id) in unique_judges.items()
            }
            for future in as_completed(futures):
                key, org = futures[future]
                parsed = future.result()
                if parsed is not None:
                    parsed["provider"] = org
                    raw_results[key] = parsed

        def _assemble_panel(
            pool: Dict[str, str], panel_prefix: str,
        ) -> Tuple[List[Dict[str, Any]], float, float]:
            all_res: List[Dict[str, Any]] = []
            excl_res: List[Dict[str, Any]] = []
            for org, judge_id in pool.items():
                key = f"{panel_prefix}_{org}"
                if key not in raw_results:
                    for other_key, res in raw_results.items():
                        if res["judge"] == judge_id:
                            key = other_key
                            break
                if key in raw_results:
                    r = raw_results[key]
                    all_res.append(r)
                    if org != family:
                        excl_res.append(r)
            rwd_all = (
                float(np.mean([r["reward"] for r in all_res]))
                if all_res else float("nan")
            )
            rwd_excl = (
                float(np.mean([r["reward"] for r in excl_res]))
                if excl_res else float("nan")
            )
            return all_res, rwd_all, rwd_excl

        exp_details, exp_all, exp_excl = _assemble_panel(
            EXPENSIVE_POOL, "expensive",
        )
        rec_details, rec_all, rec_excl = _assemble_panel(
            RECOMMENDED_POOL, "recommended",
        )

        return {
            "model_id": model_id,
            "model_family": family,
            "prompt": prompt,
            "response": response,
            "original_v1_raw_score": record.get("raw_score"),
            "expensive_v2_judge_details": exp_details,
            "expensive_v2_reward_all": round(exp_all, 4),
            "expensive_v2_reward_excluded": round(exp_excl, 4),
            "recommended_judge_details": rec_details,
            "recommended_reward_all": round(rec_all, 4),
            "recommended_reward_excluded": round(rec_excl, 4),
            "ts": time.time(),
        }

    # ---- run validation --------------------------------------------------

    def run_validation(
        self,
        data_path: Path = DEFAULT_DATA_PATH,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
        n: int = 200,
        seed: int = 42,
    ) -> Path:
        """Run both expensive and recommended panels on sampled records.

        Returns
        -------
        Path
            Path to the output JSONL file.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"validation_n{n}_seed{seed}.jsonl"

        completed_keys: set[Tuple[str, str]] = set()
        if output_path.exists():
            with open(output_path, "r") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        completed_keys.add(
                            (rec["model_id"], rec["prompt"][:200])
                        )
                    except (json.JSONDecodeError, KeyError):
                        continue
            if completed_keys:
                print(
                    f"Resuming: {len(completed_keys)} records already judged."
                )

        print(f"Sampling {n} records (stratified) from {data_path.name} ...")
        sampled = self.sample_stratified(data_path, n=n, seed=seed)
        print(f"Sampled {len(sampled)} records across score quartiles.\n")

        remaining = [
            r for r in sampled
            if (r["model_id"], r["prompt"][:200]) not in completed_keys
        ]
        print(
            f"To judge: {len(remaining)}  "
            f"(skipping {len(sampled) - len(remaining)} already done)"
        )

        if not remaining:
            print(
                "All records already judged. "
                "Run 'analyze --mode validate' to see results."
            )
            return output_path

        print(
            f"Running 7 unique judges per record "
            f"(4 expensive + 3 new cheap, gpt-4o shared) ...\n"
        )

        with open(output_path, "a") as outfile:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self.judge_record_validation, rec): rec
                    for rec in remaining
                }
                with tqdm(
                    total=len(remaining),
                    desc="Validation (both panels)",
                ) as pbar:
                    for future in as_completed(futures):
                        result = future.result()
                        with self._lock:
                            outfile.write(json.dumps(result) + "\n")
                            outfile.flush()
                        pbar.update(1)

        print(f"\nResults saved to {output_path}")
        return output_path

    # ---- main run loop ---------------------------------------------------

    def run(
        self,
        data_path: Path = DEFAULT_DATA_PATH,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
        n: int = 200,
        seed: int = 42,
    ) -> Path:
        """Full pipeline: sample, judge, save.

        Returns
        -------
        Path
            Path to the output JSONL file.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"calibration_n{n}_seed{seed}.jsonl"

        completed_keys: set[Tuple[str, str]] = set()
        if output_path.exists():
            with open(output_path, "r") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        completed_keys.add(
                            (rec["model_id"], rec["prompt"][:200])
                        )
                    except (json.JSONDecodeError, KeyError):
                        continue
            if completed_keys:
                print(f"Resuming: {len(completed_keys)} records already judged.")

        print(f"Sampling {n} records (stratified) from {data_path.name} ...")
        sampled = self.sample_stratified(data_path, n=n, seed=seed)
        print(f"Sampled {len(sampled)} records across score quartiles.\n")

        remaining = [
            r for r in sampled
            if (r["model_id"], r["prompt"][:200]) not in completed_keys
        ]
        print(
            f"To judge: {len(remaining)}  "
            f"(skipping {len(sampled) - len(remaining)} already done)"
        )

        if not remaining:
            print("All records already judged. Run 'analyze' to see results.")
            return output_path

        with open(output_path, "a") as outfile:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self.judge_record, rec): rec
                    for rec in remaining
                }
                with tqdm(
                    total=len(remaining), desc="Cheap-judge calibration",
                ) as pbar:
                    for future in as_completed(futures):
                        result = future.result()
                        with self._lock:
                            outfile.write(json.dumps(result) + "\n")
                            outfile.flush()
                        pbar.update(1)

        print(f"\nResults saved to {output_path}")
        return output_path


# ===================================================================
# Analysis
# ===================================================================


def _cohens_kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's kappa for two binary label arrays."""
    n = len(a)
    if n == 0:
        return float("nan")
    p_observed = float(np.sum(a == b)) / n
    p_a1 = float(np.mean(a))
    p_b1 = float(np.mean(b))
    p_expected = p_a1 * p_b1 + (1 - p_a1) * (1 - p_b1)
    if p_expected == 1.0:
        return 1.0 if p_observed == 1.0 else 0.0
    return (p_observed - p_expected) / (1.0 - p_expected)


def analyze_calibration(calibration_path: Path) -> None:
    """Load calibration results and print agreement metrics.

    Compares cheap-panel composite rewards against the original panel's
    ``raw_score``.  Reports Pearson r, Spearman rho, Cohen kappa
    (binarized at median), MAE, and per-provider rubric breakdowns.

    Parameters
    ----------
    calibration_path:
        Path to the JSONL file produced by ``CheapJudgeCalibrator.run()``.
    """
    from scipy import stats

    records: List[Dict[str, Any]] = []
    with open(calibration_path, "r") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not records:
        print("No records found.")
        return

    print(f"\n{'=' * 64}")
    print("  Cheap Judge Calibration Report")
    print(f"  {len(records)} records from {calibration_path.name}")
    print(f"{'=' * 64}\n")

    # -- Panel-level comparison --------------------------------------------

    original = np.array([r["original_raw_score"] for r in records])
    cheap_all = np.array([r["cheap_reward_all_judges"] for r in records])
    cheap_excl = np.array([r["cheap_reward_family_excluded"] for r in records])

    valid_all = ~(np.isnan(original) | np.isnan(cheap_all))
    valid_excl = ~(np.isnan(original) | np.isnan(cheap_excl))

    print("── Panel Composite: Cheap vs. Original ──\n")

    for label, cheap, valid_mask in [
        ("All 4 cheap judges", cheap_all, valid_all),
        ("Family-excluded (3 judges)", cheap_excl, valid_excl),
    ]:
        o, c = original[valid_mask], cheap[valid_mask]
        if len(o) < 5:
            print(f"  {label}: too few valid records ({len(o)})\n")
            continue

        pearson_r, pearson_p = stats.pearsonr(o, c)
        spearman_r, spearman_p = stats.spearmanr(o, c)
        mae = float(np.mean(np.abs(o - c)))

        median_orig = float(np.median(o))
        o_bin = (o >= median_orig).astype(int)
        c_bin = (c >= median_orig).astype(int)
        kappa = _cohens_kappa(o_bin, c_bin)

        print(f"  {label} (n={len(o)}):")
        print(f"    Pearson r  = {pearson_r:.3f}  (p={pearson_p:.1e})")
        print(f"    Spearman ρ = {spearman_r:.3f}  (p={spearman_p:.1e})")
        print(
            f"    Cohen κ    = {kappa:.3f}  "
            f"(binarized at median={median_orig:.2f})"
        )
        print(f"    MAE        = {mae:.4f}")
        print(
            f"    Mean orig  = {np.mean(o):.3f}  "
            f"Mean cheap = {np.mean(c):.3f}"
        )
        print()

    # -- Per-provider breakdown --------------------------------------------

    print("── Per Cheap Judge Breakdown ──\n")

    provider_scores: Dict[str, List[Tuple[float, float]]] = {}
    for rec in records:
        orig = rec["original_raw_score"]
        if isinstance(orig, float) and math.isnan(orig):
            continue
        for jd in rec.get("cheap_judge_details", []):
            prov = jd.get("provider", "unknown")
            provider_scores.setdefault(prov, []).append((orig, jd["reward"]))

    for prov in sorted(provider_scores):
        pairs = provider_scores[prov]
        o = np.array([p[0] for p in pairs])
        c = np.array([p[1] for p in pairs])
        if len(o) < 5:
            continue

        pearson_r, _ = stats.pearsonr(o, c)
        spearman_r, _ = stats.spearmanr(o, c)
        mae = float(np.mean(np.abs(o - c)))
        cheap_model = CHEAP_POOL.get(prov, prov)

        print(f"  {cheap_model} (n={len(pairs)}):")
        print(f"    Pearson r  = {pearson_r:.3f}")
        print(f"    Spearman ρ = {spearman_r:.3f}")
        print(f"    MAE        = {mae:.4f}")
        print(
            f"    Mean score = {np.mean(c):.3f}  "
            f"vs orig {np.mean(o):.3f}"
        )
        print()

    # -- Rubric factor distributions ---------------------------------------

    print("── Rubric Factor Pass Rates (Cheap Judges) ──\n")

    for prov in sorted(CHEAP_POOL):
        logic_vals: List[int] = []
        constraint_vals: List[int] = []
        utility_vals: List[float] = []

        for rec in records:
            for jd in rec.get("cheap_judge_details", []):
                if jd.get("provider") == prov:
                    logic_vals.append(jd.get("logic", 0))
                    constraint_vals.append(jd.get("constraint", 0))
                    utility_vals.append(jd.get("utility", 0.5))

        if not logic_vals:
            continue

        cheap_model = CHEAP_POOL[prov]
        print(f"  {cheap_model} (n={len(logic_vals)}):")
        print(f"    Logic pass rate       = {np.mean(logic_vals):.1%}")
        print(f"    Constraint pass rate  = {np.mean(constraint_vals):.1%}")
        print(f"    Utility mean          = {np.mean(utility_vals):.3f}")
        print()


def analyze_validation(validation_path: Path) -> None:
    """Analyze apples-to-apples validation: expensive-v2 vs recommended-v2.

    Both panels used the same v2 rubric, so this comparison isolates the
    effect of swapping models without a format confound.

    Parameters
    ----------
    validation_path:
        Path to the JSONL file produced by
        ``CheapJudgeCalibrator.run_validation()``.
    """
    from scipy import stats

    records: List[Dict[str, Any]] = []
    with open(validation_path, "r") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not records:
        print("No records found.")
        return

    print(f"\n{'=' * 64}")
    print("  Apples-to-Apples Validation Report")
    print(f"  {len(records)} records from {validation_path.name}")
    print(f"{'=' * 64}\n")

    # -- Panel-level comparison --------------------------------------------

    exp_all = np.array([r["expensive_v2_reward_all"] for r in records])
    exp_excl = np.array([r["expensive_v2_reward_excluded"] for r in records])
    rec_all = np.array([r["recommended_reward_all"] for r in records])
    rec_excl = np.array([r["recommended_reward_excluded"] for r in records])

    print("── Panel Composite: Recommended vs. Expensive (both v2 rubric) ──\n")

    for label, expensive, recommended in [
        ("All 4 judges", exp_all, rec_all),
        ("Family-excluded (3 judges)", exp_excl, rec_excl),
    ]:
        valid = ~(np.isnan(expensive) | np.isnan(recommended))
        e, r = expensive[valid], recommended[valid]
        if len(e) < 5:
            print(f"  {label}: too few valid records ({len(e)})\n")
            continue

        pearson_r, pearson_p = stats.pearsonr(e, r)
        spearman_r, spearman_p = stats.spearmanr(e, r)
        mae = float(np.mean(np.abs(e - r)))

        median_exp = float(np.median(e))
        e_bin = (e >= median_exp).astype(int)
        r_bin = (r >= median_exp).astype(int)
        kappa = _cohens_kappa(e_bin, r_bin)

        print(f"  {label} (n={len(e)}):")
        print(f"    Pearson r  = {pearson_r:.3f}  (p={pearson_p:.1e})")
        print(f"    Spearman ρ = {spearman_r:.3f}  (p={spearman_p:.1e})")
        print(
            f"    Cohen κ    = {kappa:.3f}  "
            f"(binarized at median={median_exp:.2f})"
        )
        print(f"    MAE        = {mae:.4f}")
        print(
            f"    Mean expensive = {np.mean(e):.3f}  "
            f"Mean recommended = {np.mean(r):.3f}"
        )
        print()

    # -- Per-factor agreement across panels --------------------------------

    print("── Per-Factor Agreement (Expensive vs. Recommended) ──\n")

    factor_names = ["logic", "constraint", "utility"]

    for factor in factor_names:
        exp_vals: List[float] = []
        rec_vals: List[float] = []

        for record in records:
            exp_by_prov: Dict[str, float] = {}
            for jd in record.get("expensive_v2_judge_details", []):
                exp_by_prov[jd.get("provider", "")] = jd.get(factor, 0)
            rec_by_prov: Dict[str, float] = {}
            for jd in record.get("recommended_judge_details", []):
                rec_by_prov[jd.get("provider", "")] = jd.get(factor, 0)

            shared_provs = set(exp_by_prov) & set(rec_by_prov)
            if not shared_provs:
                for ep in exp_by_prov.values():
                    exp_vals.append(ep)
                for rp in rec_by_prov.values():
                    rec_vals.append(rp)
            else:
                for prov in shared_provs:
                    exp_vals.append(exp_by_prov[prov])
                    rec_vals.append(rec_by_prov[prov])

        if len(exp_vals) < 5:
            continue

        e_arr = np.array(exp_vals)
        r_arr = np.array(rec_vals)

        if factor == "utility":
            pearson_r, _ = stats.pearsonr(e_arr, r_arr)
            mae = float(np.mean(np.abs(e_arr - r_arr)))
            print(f"  {factor.title()} & Tone (continuous):")
            print(f"    Pearson r = {pearson_r:.3f}")
            print(f"    MAE       = {mae:.4f}")
            print(
                f"    Mean exp  = {np.mean(e_arr):.3f}  "
                f"Mean rec  = {np.mean(r_arr):.3f}"
            )
        else:
            agreement = float(np.mean(e_arr == r_arr))
            kappa = _cohens_kappa(
                e_arr.astype(int), r_arr.astype(int),
            )
            print(f"  {factor.title()} (binary):")
            print(f"    Agreement = {agreement:.1%}")
            print(f"    Cohen κ   = {kappa:.3f}")
            print(
                f"    Pass rate exp = {np.mean(e_arr):.1%}  "
                f"rec = {np.mean(r_arr):.1%}"
            )
        print()

    # -- Per-judge rubric breakdown ----------------------------------------

    print("── Rubric Factor Pass Rates by Judge ──\n")

    all_pools = {
        "Expensive": EXPENSIVE_POOL,
        "Recommended": RECOMMENDED_POOL,
    }
    detail_keys = {
        "Expensive": "expensive_v2_judge_details",
        "Recommended": "recommended_judge_details",
    }

    for panel_label in ["Expensive", "Recommended"]:
        pool = all_pools[panel_label]
        detail_key = detail_keys[panel_label]
        print(f"  [{panel_label} Panel]")

        for prov in sorted(pool):
            logic_vals: List[int] = []
            constraint_vals: List[int] = []
            utility_vals: List[float] = []

            for rec in records:
                for jd in rec.get(detail_key, []):
                    if jd.get("provider") == prov:
                        logic_vals.append(jd.get("logic", 0))
                        constraint_vals.append(jd.get("constraint", 0))
                        utility_vals.append(jd.get("utility", 0.5))

            if not logic_vals:
                continue

            print(f"    {pool[prov]} (n={len(logic_vals)}):")
            print(f"      Logic pass rate       = {np.mean(logic_vals):.1%}")
            print(
                f"      Constraint pass rate  = "
                f"{np.mean(constraint_vals):.1%}"
            )
            print(f"      Utility mean          = {np.mean(utility_vals):.3f}")
            print()

    # -- Also compare against original v1 for context ----------------------

    print("── Context: Original v1 Scores ──\n")

    orig_v1 = np.array([
        r.get("original_v1_raw_score", float("nan")) for r in records
    ])
    valid_v1_exp = ~(np.isnan(orig_v1) | np.isnan(exp_excl))

    if np.sum(valid_v1_exp) >= 5:
        o, e = orig_v1[valid_v1_exp], exp_excl[valid_v1_exp]
        spearman_r, _ = stats.spearmanr(o, e)
        print(f"  v1 original vs expensive-v2 (family-excluded):")
        print(f"    Spearman ρ = {spearman_r:.3f}")
        print(
            f"    Mean v1 = {np.mean(o):.3f}  "
            f"Mean expensive-v2 = {np.mean(e):.3f}"
        )
        print()


# ===================================================================
# CLI
# ===================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate cheap LLM judges against the expensive panel.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")
    sub.required = True

    # -- shared arguments for run/validate ---------------------------------

    for name, help_text in [
        ("run", "Sample records and judge with cheap models only"),
        ("validate", "Run both expensive and recommended panels (apples-to-apples)"),
    ]:
        p = sub.add_parser(name, help=help_text)
        p.add_argument(
            "--n", type=int, default=200,
            help="Number of records to sample (default: 200)",
        )
        p.add_argument(
            "--seed", type=int, default=42,
            help="Random seed for stratified sampling (default: 42)",
        )
        p.add_argument(
            "--data", type=str, default=str(DEFAULT_DATA_PATH),
            help="Path to gzipped reward JSONL",
        )
        p.add_argument(
            "--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
            help="Directory for calibration output",
        )
        p.add_argument(
            "--workers", type=int, default=32,
            help="Max parallel workers for outer loop (default: 32)",
        )

    # -- analyze -----------------------------------------------------------

    analyze_p = sub.add_parser(
        "analyze", help="Analyze saved results (offline)",
    )
    analyze_p.add_argument(
        "--path", type=str, default=None,
        help="Path to calibration/validation JSONL (default: latest)",
    )
    analyze_p.add_argument(
        "--mode", type=str, default="auto",
        choices=["auto", "calibration", "validation"],
        help="Analysis type (default: auto-detect from filename)",
    )

    args = parser.parse_args()

    if args.command == "run":
        cal = CheapJudgeCalibrator(max_workers=args.workers)
        output_path = cal.run(
            data_path=Path(args.data),
            output_dir=Path(args.output_dir),
            n=args.n,
            seed=args.seed,
        )
        print("\nRunning analysis on results...\n")
        analyze_calibration(output_path)

    elif args.command == "validate":
        cal = CheapJudgeCalibrator(max_workers=args.workers)
        output_path = cal.run_validation(
            data_path=Path(args.data),
            output_dir=Path(args.output_dir),
            n=args.n,
            seed=args.seed,
        )
        print("\nRunning analysis on results...\n")
        analyze_validation(output_path)

    elif args.command == "analyze":
        if args.path:
            path = Path(args.path)
        else:
            cal_dir = DEFAULT_OUTPUT_DIR
            candidates = sorted(
                list(cal_dir.glob("calibration_*.jsonl"))
                + list(cal_dir.glob("validation_*.jsonl"))
            )
            if not candidates:
                print(f"No result files found in {cal_dir}")
                return
            path = candidates[-1]
            print(f"Using latest file: {path.name}")

        mode = args.mode
        if mode == "auto":
            mode = (
                "validation" if "validation" in path.name else "calibration"
            )

        if mode == "validation":
            analyze_validation(path)
        else:
            analyze_calibration(path)


if __name__ == "__main__":
    main()
