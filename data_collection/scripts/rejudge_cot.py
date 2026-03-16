import json
import logging
import os
import time
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PoLL-style judge-panel diagnostics
# ---------------------------------------------------------------------------

class JudgePanelDiagnostics:
    """Online tracker for per-judge bias and variance.

    Accumulates per-(judge, candidate_model) reward statistics so that
    systematic over-/under-scoring can be detected and optionally
    compensated for via inverse-variance or bias-corrected weighting.

    Reference: Verga et al., "Replacing Judges with Juries: Evaluating
    LLM Generations with a Panel of Diverse Models" (2024).
    """

    def __init__(self) -> None:
        # {judge_id: {model_id: [composite_reward, ...]}}
        self._scores: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list),
        )

    # -- recording ----------------------------------------------------------

    def record(
        self,
        judge_id: str,
        candidate_model: str,
        reward: float,
    ) -> None:
        """Record a single judge score for a candidate model."""
        if np.isfinite(reward):
            self._scores[judge_id][candidate_model].append(reward)

    def record_panel(
        self,
        judge_details: List[Dict[str, Any]],
        candidate_model: str,
    ) -> None:
        """Convenience: record every judge from a panel result list."""
        for jd in judge_details:
            judge_id = jd.get("judge", "unknown")
            reward = jd.get("reward", float("nan"))
            self.record(judge_id, candidate_model, reward)

    # -- analysis -----------------------------------------------------------

    def per_judge_summary(self) -> Dict[str, Dict[str, Any]]:
        """Return per-judge aggregate stats.

        Returns
        -------
        dict
            ``{judge_id: {model_id: {n, mean, std}, ..., "_overall": {n, mean, std}}}``
        """
        summary: Dict[str, Dict[str, Any]] = {}
        for judge, models in self._scores.items():
            all_scores: List[float] = []
            per_model: Dict[str, Any] = {}
            for model, scores in models.items():
                arr = np.array(scores)
                per_model[model] = {
                    "n": len(arr),
                    "mean": float(arr.mean()),
                    "std": float(arr.std()),
                }
                all_scores.extend(scores)
            arr_all = np.array(all_scores)
            per_model["_overall"] = {
                "n": len(arr_all),
                "mean": float(arr_all.mean()),
                "std": float(arr_all.std()),
            }
            summary[judge] = per_model
        return summary

    def bias_matrix(self) -> Dict[str, Dict[str, float]]:
        """Compute per-(judge, model) bias relative to the panel mean.

        ``bias[j][m] = mean_j(m) - panel_mean(m)``

        A large positive value means judge *j* is systematically more
        lenient toward model *m* than the panel average.
        """
        # Panel mean per model (across all judges).
        model_panel_mean: Dict[str, float] = defaultdict(float)
        model_panel_n: Dict[str, int] = defaultdict(int)
        for _judge, models in self._scores.items():
            for model, scores in models.items():
                model_panel_mean[model] += sum(scores)
                model_panel_n[model] += len(scores)
        for m in model_panel_mean:
            model_panel_mean[m] /= max(model_panel_n[m], 1)

        bias: Dict[str, Dict[str, float]] = {}
        for judge, models in self._scores.items():
            bias[judge] = {}
            for model, scores in models.items():
                judge_mean = float(np.mean(scores))
                bias[judge][model] = round(judge_mean - model_panel_mean[model], 4)
        return bias

    def compute_weights(
        self,
        method: str = "equal",
    ) -> Dict[str, float]:
        """Compute per-judge ensemble weights.

        Parameters
        ----------
        method
            ``"equal"`` — uniform 1/J weights (default).
            ``"inverse_variance"`` — weight inversely proportional to
            each judge's overall score variance (down-weights noisy
            judges).
            ``"inverse_bias"`` — weight inversely proportional to the
            judge's maximum absolute per-model bias (down-weights
            judges that systematically favor/penalize specific models).
        """
        judges = list(self._scores.keys())
        n_judges = len(judges)
        if n_judges == 0:
            return {}

        if method == "equal":
            w = 1.0 / n_judges
            return {j: w for j in judges}

        if method == "inverse_variance":
            variances: Dict[str, float] = {}
            for j in judges:
                all_s = []
                for scores in self._scores[j].values():
                    all_s.extend(scores)
                variances[j] = float(np.var(all_s)) if all_s else 1.0
            inv = {j: 1.0 / max(v, 1e-8) for j, v in variances.items()}
            total = sum(inv.values())
            return {j: inv[j] / total for j in judges}

        if method == "inverse_bias":
            bias = self.bias_matrix()
            max_abs_bias: Dict[str, float] = {}
            for j in judges:
                abs_biases = [abs(b) for b in bias.get(j, {}).values()]
                max_abs_bias[j] = max(abs_biases) if abs_biases else 0.0
            inv = {j: 1.0 / max(b, 1e-8) for j, b in max_abs_bias.items()}
            total = sum(inv.values())
            return {j: inv[j] / total for j in judges}

        raise ValueError(f"Unknown weighting method: {method!r}")

    def log_report(self, min_samples: int = 50) -> None:
        """Log a human-readable diagnostics report.

        Only emits output once every judge × model cell has at least
        *min_samples* observations to avoid noisy early reports.
        """
        ready = all(
            len(scores) >= min_samples
            for models in self._scores.values()
            for scores in models.values()
        )
        if not ready:
            return

        summary = self.per_judge_summary()
        bias = self.bias_matrix()
        weights_iv = self.compute_weights("inverse_variance")
        weights_ib = self.compute_weights("inverse_bias")

        lines = ["", "=== Judge Panel Diagnostics ==="]
        for judge in sorted(summary):
            overall = summary[judge]["_overall"]
            lines.append(
                f"  {judge:<45} "
                f"N={overall['n']:>5}  "
                f"mean={overall['mean']:.3f}  "
                f"std={overall['std']:.3f}  "
                f"w_iv={weights_iv.get(judge, 0):.3f}  "
                f"w_ib={weights_ib.get(judge, 0):.3f}"
            )
            for model in sorted(summary[judge]):
                if model == "_overall":
                    continue
                ms = summary[judge][model]
                b = bias.get(judge, {}).get(model, 0.0)
                flag = " ⚠" if abs(b) > 0.05 else ""
                lines.append(
                    f"    → {model:<40} "
                    f"mean={ms['mean']:.3f}  "
                    f"bias={b:+.3f}{flag}"
                )
        lines.append("")
        logger.info("\n".join(lines))


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

# Reuse the class structure but modify for CoT / Re-judging
class CoTRewardGenerator:
    def __init__(self, api_key: str = None, max_workers: int = 10):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            # Try loading from .env manually if not in env
             try:
                from dotenv import load_dotenv
                # paretobandit/rejudge_cot.py -> parent = paretobandit -> parent = root
                env_path = Path(__file__).parent.parent.parent / '.env'
                if env_path.exists():
                    load_dotenv(env_path)
                self.api_key = os.getenv("OPENROUTER_API_KEY")
             except: pass
             
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found")
        
        self.base_url = "https://openrouter.ai/api/v1"
        
        # Fixed 3-judge panel — intentionally disjoint from every K=5
        # candidate family (Meta, Google, OpenAI, Mistral) to eliminate
        # panel-composition bias.
        self.judge_panel: List[str] = [
            "deepseek/deepseek-r1",
            "openai/gpt-4.1-mini",
            "anthropic/claude-3.5-haiku",
        ]

        self.judge_max_tokens: int = 4000
        # DeepSeek-R1 produces verbose chain-of-thought; cap its output to
        # keep costs and latency under control.
        self._judge_max_tokens_override: Dict[str, int] = {
            "deepseek/deepseek-r1": 2048,
        }

        self.max_workers = max_workers
        self.lock = threading.Lock()

        # Cache for existing responses: (model_id, prompt) -> response_text
        self.response_cache = {}

        # PoLL-style per-judge diagnostics (accumulated across the run).
        self.diagnostics = JudgePanelDiagnostics()

        # Aggregation strategy: "equal", "inverse_variance", or
        # "inverse_bias".  Early in a run (few samples) the weights
        # degenerate to equal; they become meaningful once every
        # (judge, model) cell has enough observations.
        self.judge_weighting: str = "equal"

    def load_cache(self, cache_file: Path):
        """Load existing responses from a previous run."""
        if not cache_file.exists():
            return
        
        print(f"Loading cache from {cache_file}...")
        count = 0
        with open(cache_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if data.get("ok") and data.get("response"):
                        key = (data["model_id"], data["prompt"])
                        self.response_cache[key] = data["response"]
                        count += 1
                except:
                    continue
        print(f"Loaded {count} cached responses.")

    def get_judges_for_model(self, model_id: str) -> List[str]:
        """Return the fixed judge panel.

        The panel is intentionally the same for every model so that
        reward comparisons across models are never confounded by
        judge-composition differences.
        """
        return list(self.judge_panel)

    def get_model_response(self, model_id: str, prompt: str) -> str:
        # Check cache first
        if (model_id, prompt) in self.response_cache:
            return self.response_cache[(model_id, prompt)]
            
        # Fetch fresh
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/paretobandit/llm-jury",
        }
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 4000
        }
        try:
            resp = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=payload, timeout=300)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            return None

    # Rubric v3 weights — all-continuous dimensions.  Must stay in sync
    # with ``_W_REASONING``, ``_W_INSTRUCTION``, ``_W_COMMUNICATION`` in
    # ``src/pareto_bandit/rewards.py``.
    _W_REASONING: float = 0.40
    _W_INSTRUCTION: float = 0.30
    _W_COMMUNICATION: float = 0.30

    def _parse_continuous_score(
        self, content: str, heading: str, *, default: float = 0.5,
    ) -> float:
        """Extract a continuous 0.0–1.0 score from a markdown heading.

        Handles common formatting variants: ``## Heading: 0.8``,
        ``## Heading\n0.8``, and percentage-style ``80`` (mapped to 0.8).
        """
        import re
        pattern = (
            r"##\s*" + heading + r"\s*[:\-]?\s*(\d+\.?\d*)"
        )
        m = re.search(pattern, content, re.IGNORECASE)
        if m:
            val = float(m.group(1))
            if val > 1.0:
                val = val / 100.0
            return max(0.0, min(1.0, val))
        return default

    _MAX_RETRIES: int = 3
    _RETRY_BACKOFF_BASE: float = 2.0

    _TIMEOUT_DEFAULT: float = 90.0
    _TIMEOUT_OVERRIDE: Dict[str, float] = {
        "deepseek/deepseek-r1": 180.0,
    }

    def judge_single_cot(
        self, judge_model: str, system_prompt: str, user_content: str,
    ) -> Dict[str, Any] | None:
        """Query a single judge and parse the three-factor rubric response.

        Retries up to ``_MAX_RETRIES`` times with exponential backoff on
        transient failures (timeouts, 429/5xx).

        Returns
        -------
        dict | None
            Keys: ``reasoning_quality`` (float 0-1),
            ``instruction_following`` (float 0-1),
            ``communication_quality`` (float 0-1), ``reward`` (composite
            float), ``reasoning`` (str).  ``None`` on API failure.

            For backward compatibility the dict also contains the legacy
            keys ``logic``, ``constraint``, ``utility`` mapped from the
            new dimensions.
        """
        import re

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/paretobandit/llm-jury",
        }

        max_tok = self._judge_max_tokens_override.get(
            judge_model, self.judge_max_tokens,
        )
        timeout = self._TIMEOUT_OVERRIDE.get(judge_model, self._TIMEOUT_DEFAULT)

        effective_prompt = system_prompt
        if "deepseek" in judge_model.lower():
            effective_prompt += (
                "\n\nIMPORTANT: Keep the ## Reasoning section to 3–5 "
                "sentences. Be direct — identify errors or confirm "
                "correctness, then move to scoring."
            )

        payload = {
            "model": judge_model,
            "messages": [
                {"role": "system", "content": effective_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.0,
            "max_tokens": max_tok,
        }

        last_exc: Exception | None = None
        for attempt in range(1, self._MAX_RETRIES + 1):
            try:
                resp = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers, json=payload, timeout=timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                raw_content = data["choices"][0]["message"]["content"]
                if raw_content is None:
                    raise ValueError("API returned null content")
                content = raw_content.strip()

                reasoning_quality = self._parse_continuous_score(
                    content, r"Reasoning\s+Quality",
                )
                instruction_following = self._parse_continuous_score(
                    content, r"Instruction\s+Following",
                )
                communication_quality = self._parse_continuous_score(
                    content, r"Communication\s+Quality",
                )

                reward = (
                    reasoning_quality * self._W_REASONING
                    + instruction_following * self._W_INSTRUCTION
                    + communication_quality * self._W_COMMUNICATION
                )

                # --- Extract Reasoning block ---
                reasoning = content
                rm = re.search(
                    r"##\s*Reasoning\s*(.*?)(\n##|$)",
                    content, re.DOTALL | re.IGNORECASE,
                )
                if rm:
                    reasoning = rm.group(1).strip()

                return {
                    "reasoning_quality": round(reasoning_quality, 4),
                    "instruction_following": round(instruction_following, 4),
                    "communication_quality": round(communication_quality, 4),
                    "reward": round(reward, 4),
                    "reasoning": reasoning,
                    # Legacy keys for backward-compatible downstream code.
                    "logic": round(reasoning_quality, 4),
                    "constraint": round(instruction_following, 4),
                    "utility": round(communication_quality, 4),
                }

            except (
                requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
                ValueError,
                KeyError,
                IndexError,
            ) as e:
                last_exc = e
            except requests.exceptions.HTTPError as e:
                last_exc = e
                status = e.response.status_code if e.response is not None else 0
                if status in (429, 502, 503, 504):
                    pass  # retryable
                else:
                    logger.warning("Judge %s non-retryable HTTP %d", judge_model, status)
                    return None
            except Exception as e:
                logger.warning("Judge %s unexpected error: %s", judge_model, e)
                return None

            backoff = self._RETRY_BACKOFF_BASE ** attempt
            logger.debug(
                "Judge %s attempt %d/%d failed (%s), retrying in %.1fs",
                judge_model, attempt, self._MAX_RETRIES, last_exc, backoff,
            )
            time.sleep(backoff)

        logger.warning(
            "Judge %s failed after %d attempts: %s",
            judge_model, self._MAX_RETRIES, last_exc,
        )
        return None

    def judge_with_panel_cot(self, prompt: str, response: str, model_id: str) -> Tuple[float, List[Dict]]:
        """Run the multi-judge panel and aggregate rubric scores.

        Each judge independently scores three continuous factors (Reasoning
        Quality, Instruction Following, Communication Quality).  The final
        reward is the mean of per-judge composite scores.
        """
        judges = self.get_judges_for_model(model_id)

        system_prompt = (
            "You are a Discriminative Router Judge. Your goal is to evaluate "
            "how well an LLM response addresses the given prompt.\n\n"
            "Score on three continuous dimensions (0.0–1.0). Use the FULL "
            "range; do NOT default to 0 or 1.\n\n"
            "1. **Reasoning Quality (40 %)** — How sound is the reasoning?\n"
            "   0.9–1.0 Flawless; every step correct and clearly justified.\n"
            "   0.7–0.8 Sound overall; minor inefficiency or a trivial error "
            "that does not change the conclusion.\n"
            "   0.5–0.6 Partially correct; approach is reasonable but "
            "important steps are wrong or missing.\n"
            "   0.3–0.4 Weak; only fragments of correct logic.\n"
            "   0.0–0.2 No coherent reasoning, or completely wrong approach.\n"
            "   If the prompt needs no multi-step reasoning, score factual "
            "accuracy and depth of explanation.\n\n"
            "2. **Instruction Following (30 %)** — Were all explicit and "
            "implicit constraints satisfied?\n"
            "   0.9–1.0 Every constraint followed precisely.\n"
            "   0.7–0.8 All major constraints met; one minor instruction "
            "partially missed.\n"
            "   0.5–0.6 Some important instructions missed or only partially "
            "addressed.\n"
            "   0.3–0.4 Multiple instructions ignored or misinterpreted.\n"
            "   0.0–0.2 Response largely ignores the prompt's requirements.\n\n"
            "3. **Communication Quality (30 %)** — How clear, well-structured, "
            "and useful is the response?\n"
            "   0.9–1.0 Exceptionally clear, well-organized, appropriate "
            "detail.\n"
            "   0.7–0.8 Clear and competent; minor improvements possible.\n"
            "   0.5–0.6 Adequate but noticeably unclear, verbose, or poorly "
            "organized.\n"
            "   0.3–0.4 Hard to follow; significant clarity issues.\n"
            "   0.0–0.2 Unintelligible, unhelpful, or inappropriate tone.\n\n"
            "Format your response EXACTLY as follows:\n\n"
            "## Reasoning\n"
            "<Concise chain-of-thought analysis>\n\n"
            "## Reasoning Quality\n"
            "<0.0 to 1.0>\n\n"
            "## Instruction Following\n"
            "<0.0 to 1.0>\n\n"
            "## Communication Quality\n"
            "<0.0 to 1.0>"
        )
        user_content = f"PROMPT: {prompt}\n\nRESPONSE: {response}"

        results: List[Dict] = []

        with ThreadPoolExecutor(max_workers=len(judges)) as executor:
            futures = {
                executor.submit(
                    self.judge_single_cot, judge, system_prompt, user_content,
                ): judge
                for judge in judges
            }

            for future in as_completed(futures):
                parsed = future.result()
                if parsed is None:
                    continue
                judge = futures[future]
                results.append({
                    "judge": judge,
                    "reasoning_quality": parsed["reasoning_quality"],
                    "instruction_following": parsed["instruction_following"],
                    "communication_quality": parsed["communication_quality"],
                    "reward": parsed["reward"],
                    "reasoning": parsed["reasoning"],
                    # Legacy keys for backward-compatible extract_reward().
                    "logic": parsed["logic"],
                    "constraint": parsed["constraint"],
                    "utility": parsed["utility"],
                })

        if not results:
            return float("nan"), results

        self.diagnostics.record_panel(results, model_id)

        weights = self.diagnostics.compute_weights(self.judge_weighting)
        weighted_sum = 0.0
        weight_total = 0.0
        for r in results:
            j = r["judge"]
            w = weights.get(j, 1.0 / max(len(results), 1))
            weighted_sum += w * r["reward"]
            weight_total += w
        final_reward = weighted_sum / weight_total if weight_total > 0 else float("nan")

        return float(final_reward), results

    def logit_transform(self, score: float) -> float:
        if np.isnan(score):
            return float("nan")
        score = np.clip(score, 0.01, 0.99)
        return float(np.log(score / (1 - score)))

    def process_task(self, task):
        prompt_text, model_id = task
        
        # 1. Get Response (Cached or New)
        response = self.get_model_response(model_id, prompt_text)
        
        if not response:
            return {
                "model_id": model_id, "ok": False, "ts": time.time()
            }

        # 2. Judge with CoT Panel
        final_score, judge_details = self.judge_with_panel_cot(prompt_text, response, model_id)
        reward_logit = self.logit_transform(final_score)
        
        return {
            "model_id": model_id,
            "prompt": prompt_text,
            "response": response,
            "ok": True,
            "teacher_used": True,
            "judge_details": judge_details, # Contains individual reasoning/scores
            "reward_logit": reward_logit,
            "raw_score": final_score,
            "ts": time.time()
        }

    def _process_rejudge_task(
        self, task: Tuple[str, str, str],
    ) -> Dict[str, Any]:
        """Re-judge a single (prompt, response, model_id) triple.

        Skips the response-generation step entirely — the response is
        taken verbatim from the source file.
        """
        prompt_text, response, model_id = task

        final_score, judge_details = self.judge_with_panel_cot(
            prompt_text, response, model_id,
        )
        reward_logit = self.logit_transform(final_score)

        return {
            "model_id": model_id,
            "prompt": prompt_text,
            "response": response,
            "ok": True,
            "teacher_used": False,
            "judge_details": judge_details,
            "reward_logit": reward_logit,
            "raw_score": final_score,
            "ts": time.time(),
        }

    def rejudge_from_file(
        self,
        source_file: Path,
        output_file: Path,
        *,
        limit: Optional[int] = None,
    ) -> None:
        """Re-judge existing (prompt, response) pairs with the current panel.

        Reads *source_file* (JSONL with ``prompt``, ``response``,
        ``model_id`` fields), passes each response through the new judge
        panel, and writes fresh reward records to *output_file*.

        Supports resume: already-completed (prompt, model_id) pairs in
        *output_file* are skipped.

        Args:
            source_file: Path to the existing rewards JSONL.
            output_file: Destination for re-judged records.
            limit: If set, only process the first *limit* source records.
        """
        # 1. Load source records.
        tasks: List[Tuple[str, str, str]] = []
        skipped_bad = 0
        with open(source_file) as f:
            for line in f:
                rec = json.loads(line)
                if not rec.get("ok") or not rec.get("response"):
                    skipped_bad += 1
                    continue
                tasks.append((rec["prompt"], rec["response"], rec["model_id"]))

        if limit is not None:
            tasks = tasks[:limit]

        print(
            f"Loaded {len(tasks)} valid records from {source_file} "
            f"(skipped {skipped_bad} bad records)"
        )

        # 2. Resume support — skip fully-completed pairs, re-process partial ones.
        n_expected_judges = len(self.judge_panel)
        completed: set = set()
        partial_keys: set = set()
        if output_file.exists():
            with open(output_file) as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        key = (entry.get("prompt", ""), entry.get("model_id", ""))
                        n_judges = len(entry.get("judge_details", []))
                        if n_judges >= n_expected_judges:
                            completed.add(key)
                        else:
                            partial_keys.add(key)
                    except json.JSONDecodeError:
                        continue
            if completed:
                print(f"Resuming: {len(completed)} fully completed, skipping.")
            if partial_keys:
                print(
                    f"  {len(partial_keys)} partial records (<{n_expected_judges} judges) "
                    f"will be re-processed."
                )

            # Remove partial records from the output file so they can be
            # cleanly re-written after re-judging.
            if partial_keys:
                kept_lines: list[str] = []
                with open(output_file) as f:
                    for line in f:
                        try:
                            entry = json.loads(line)
                            key = (entry.get("prompt", ""), entry.get("model_id", ""))
                            if key not in partial_keys:
                                kept_lines.append(line)
                        except json.JSONDecodeError:
                            kept_lines.append(line)
                with open(output_file, "w") as f:
                    f.writelines(kept_lines)
                print(f"  Removed {len(partial_keys)} partial records from output for re-judging.")

        remaining = [t for t in tasks if (t[0], t[2]) not in completed]
        print(f"Tasks to run: {len(remaining)} (skipped {len(tasks) - len(remaining)})")

        if not remaining:
            print("Nothing to do.")
            return

        # 3. Run parallel re-judging.
        completed_count = 0
        diag_interval = max(100, len(remaining) // 10)

        with open(output_file, "a") as outfile:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self._process_rejudge_task, t): t
                    for t in remaining
                }
                with tqdm(total=len(remaining), desc="Re-judging") as pbar:
                    for fut in as_completed(futures):
                        res = fut.result()
                        with self.lock:
                            outfile.write(json.dumps(res) + "\n")
                            outfile.flush()
                            completed_count += 1
                        pbar.update(1)

                        if completed_count % diag_interval == 0:
                            self.diagnostics.log_report(min_samples=30)

        # 4. Final diagnostics.
        print("\n" + "=" * 60)
        print("FINAL JUDGE PANEL DIAGNOSTICS")
        print("=" * 60)
        self.diagnostics.log_report(min_samples=1)

        diag_path = output_file.with_suffix(".judge_diagnostics.json")
        diag_payload = {
            "summary": self.diagnostics.per_judge_summary(),
            "bias_matrix": self.diagnostics.bias_matrix(),
            "weights_equal": self.diagnostics.compute_weights("equal"),
            "weights_inverse_variance": self.diagnostics.compute_weights(
                "inverse_variance",
            ),
            "weights_inverse_bias": self.diagnostics.compute_weights(
                "inverse_bias",
            ),
        }
        with open(diag_path, "w") as df:
            json.dump(diag_payload, df, indent=2)
        print(f"Diagnostics written to {diag_path}")

    def run(self, prompts_file, models_file, output_file, cache_file, is_lmsys=False, limit=None):
        # 1. Load Cache
        self.load_cache(cache_file)
        
        # 2. Load Prompts
        prompts = []
        with open(prompts_file) as f:
            for line in f:
                data = json.loads(line)
                if is_lmsys:
                    # Check for direct 'prompt' key first (cleaned format)
                    if 'prompt' in data:
                        prompts.append(data)
                    else:
                        # Fallback to raw LMSYS format: conversation[0]['content']
                        try:
                            prompt_text = data['conversation'][0]['content']
                            prompts.append({"prompt": prompt_text})
                        except:
                            continue
                else:
                    prompts.append(data)
        
        if limit:
            prompts = prompts[:limit]
            print(f"Limiting to first {limit} prompts.")
        
        # 3. Load Models
        with open(models_file) as f:
            registry = json.load(f)
        models = [m["model_id"] for m in registry["models"]]
        
        print(f"Processing {len(prompts)} prompts x {len(models)} models = {len(prompts)*len(models)} tasks")
        
        # 4. Create Tasks
        tasks = []
        for p in prompts:
            for m in models:
                tasks.append((p["prompt"], m))
                
        # 5. Load already-completed tasks from output (resume support)
        completed = set()
        if output_file.exists():
            with open(output_file, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        completed.add((entry.get("prompt", ""), entry.get("model_id", "")))
                    except json.JSONDecodeError:
                        continue
            if completed:
                print(f"Resuming: {len(completed)} tasks already completed, skipping them.")

        remaining = [t for t in tasks if t not in completed]
        print(f"Tasks to run: {len(remaining)} (skipped {len(tasks) - len(remaining)})")

        # 6. Run Parallel — flush each result to disk immediately
        print(f"Saving to {output_file} (append + flush per entry)")
        completed_count = 0
        diag_interval = max(100, len(remaining) // 10)

        with open(output_file, 'a') as outfile:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(self.process_task, t): t for t in remaining}

                with tqdm(total=len(remaining), desc="CoT Judging") as pbar:
                    for f in as_completed(futures):
                        res = f.result()
                        with self.lock:
                            outfile.write(json.dumps(res) + "\n")
                            outfile.flush()
                            completed_count += 1
                        pbar.update(1)

                        if completed_count % diag_interval == 0:
                            self.diagnostics.log_report(min_samples=30)

        # Final diagnostics report.
        print("\n" + "=" * 60)
        print("FINAL JUDGE PANEL DIAGNOSTICS")
        print("=" * 60)
        self.diagnostics.log_report(min_samples=1)

        # Write diagnostics to a sidecar file for later analysis.
        diag_path = output_file.with_suffix(".judge_diagnostics.json")
        diag_payload = {
            "summary": self.diagnostics.per_judge_summary(),
            "bias_matrix": self.diagnostics.bias_matrix(),
            "weights_equal": self.diagnostics.compute_weights("equal"),
            "weights_inverse_variance": self.diagnostics.compute_weights("inverse_variance"),
            "weights_inverse_bias": self.diagnostics.compute_weights("inverse_bias"),
        }
        with open(diag_path, "w") as df:
            json.dump(diag_payload, df, indent=2)
        print(f"Diagnostics written to {diag_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate multi-judge CoT rewards for (prompt, model) pairs.",
    )
    parser.add_argument("--mode", type=str, default="pareto",
                        choices=["pareto", "distribution", "custom", "rejudge"],
                        help="Preset mode or 'rejudge' to re-judge existing data")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of prompts to process")
    parser.add_argument("--prompts-file", type=str, default=None,
                        help="Path to prompts JSONL (required for --mode custom)")
    parser.add_argument("--models-file", type=str, default=None,
                        help="Path to models JSON (default: models.json)")
    parser.add_argument("--output-file", type=str, default=None,
                        help="Path to output JSONL (required for --mode custom/rejudge)")
    parser.add_argument("--cache-file", type=str, default=None,
                        help="Path to response cache JSONL (optional)")
    parser.add_argument(
        "--rejudge-from", type=str, default=None,
        help=(
            "Path to an existing rewards JSONL whose (prompt, response) "
            "pairs will be re-judged with the current panel.  Use with "
            "--mode rejudge."
        ),
    )
    parser.add_argument("--workers", type=int, default=10,
                        help="Max parallel workers (default: 10)")
    parser.add_argument(
        "--judge-weighting", type=str, default="equal",
        choices=["equal", "inverse_variance", "inverse_bias"],
        help=(
            "Panel aggregation strategy.  'equal' (default) uses uniform "
            "weights.  'inverse_variance' down-weights noisy judges.  "
            "'inverse_bias' down-weights judges that systematically "
            "favor/penalize specific candidate models."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    root = Path(__file__).parent.parent.parent
    gen = CoTRewardGenerator(max_workers=args.workers)
    gen.judge_weighting = args.judge_weighting

    models_file = Path(args.models_file) if args.models_file else root / "src/pareto_bandit/config/models.json"

    if args.mode == "rejudge":
        if not args.rejudge_from:
            parser.error("--mode rejudge requires --rejudge-from <source.jsonl>")
        source = Path(args.rejudge_from)
        output = (
            Path(args.output_file)
            if args.output_file
            else source.with_name(source.stem + "_v3.jsonl")
        )
        print(f"Re-judging from {source} → {output}")
        print(f"Judge panel: {gen.judge_panel}")
        print(f"Judge weighting: {gen.judge_weighting}")
        gen.rejudge_from_file(source, output, limit=args.limit)
    elif args.mode == "custom":
        if not args.prompts_file or not args.output_file:
            parser.error("--mode custom requires --prompts-file and --output-file")
        gen.run(
            prompts_file=Path(args.prompts_file),
            models_file=models_file,
            output_file=Path(args.output_file),
            cache_file=Path(args.cache_file) if args.cache_file else Path(args.output_file).with_suffix(".cache.jsonl"),
            is_lmsys=False,
            limit=args.limit,
        )
    elif args.mode == "distribution":
        gen.run(
            prompts_file=root / "data/lmsys_needs_rewards_combined.jsonl",
            models_file=models_file,
            output_file=root / "data/lmsys_new_rewards_888.jsonl",
            cache_file=root / "data/lmsys_rewards_cache.jsonl",
            is_lmsys=True,
            limit=args.limit,
        )
    else:
        gen.run(
            prompts_file=root / "src/pareto_bandit/data/test_prompts.jsonl",
            models_file=models_file,
            output_file=root / "src/pareto_bandit/data/test_rewards_pareto.jsonl",
            cache_file=root / "src/pareto_bandit/data/test_rewards_cache.jsonl",
            limit=args.limit,
        )
