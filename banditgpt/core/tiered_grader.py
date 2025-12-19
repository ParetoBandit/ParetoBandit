"""
Tiered grading: "vibe vs verifier".

Why:
  - A cheap local grader (our DeBERTa/MiniLM classifier) is good at fluency/style.
  - It is *not* a truth oracle for math/logic/coding constraint satisfaction.

This module provides a single drop-in grader interface that:
  - Uses a local "soft grader" for most prompts (cheap/fast).
  - Escalates "hard" prompts to an optional teacher/verifier (slow/costly).

Important:
  - The teacher path is optional. If you do not provide a teacher, this will
    behave like the soft grader only (no network calls).
"""

from __future__ import annotations

import os
import re
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, Tuple

import subprocess

from banditgpt.core.quality_cost_predictor import (
    LogitReward,
    RunningZScoreNormalizer,
    clip01,
    clipped_quality_reward,
)

logger = logging.getLogger(__name__)


class ProductionGrader(Protocol):
    """
    Minimal interface expected by `BanditRouter.process_feedback`.
    """

    def predict_production(
        self,
        prompt: str,
        response: str,
        *,
        reward_normalizer: Optional[RunningZScoreNormalizer] = None,
    ) -> Dict[str, Any]: ...


class TeacherVerifier(Protocol):
    """
    A "hard path" verifier that returns a truth-focused score in [0, 1].
    """

    def verify(self, prompt: str, response: str) -> Tuple[float, Dict[str, Any]]: ...

class CodeExecutionVerifier(Protocol):
    """
    Optional verifier for code prompts (stronger than LLM judging).

    IMPORTANT: only safe when executed inside a locked-down sandbox
    (Docker/Firecracker/VM). Do NOT run untrusted code on the host.
    """

    def verify(self, prompt: str, response: str) -> Tuple[float, Dict[str, Any]]: ...


@dataclass
class HardPromptHeuristics:
    """
    Conservative heuristic gate for prompts where "truth" dominates "vibe".
    """

    triggers: Tuple[str, ...] = (
        "calculate",
        "compute",
        "solve",
        "equation",
        "derivative",
        "integral",
        "prove",
        "pH",
        "molar",
        "mol/l",
        "unit",
        "json",
        "yaml",
        "schema",
        "constraint",
        "must",
        "exactly",
        "only",
        "code",
        "python",
        "javascript",
        "typescript",
        "sql",
        "function",
        "class",
        "regex",
        "compile",
        "error",
        "traceback",
    )

    def is_hard(self, prompt: str) -> bool:
        p = (prompt or "").lower()
        return any(t.lower() in p for t in self.triggers)


def _parse_float01(text: str) -> Optional[float]:
    """
    Robust parsing for LLM judge outputs.
    Accepts e.g. "0.73", "score: 0.73", "0.73\\n".
    """

    if not isinstance(text, str):
        return None
    m = re.search(r"(-?\d+(?:\.\d+)?)", text.strip())
    if not m:
        return None
    try:
        x = float(m.group(1))
    except Exception:
        return None
    if x != x:  # NaN
        return None
    return float(min(max(x, 0.0), 1.0))


class OpenRouterTeacherVerifier:
    """
    Truth-focused verifier using OpenRouter (optional network path).

    Notes:
      - Requires `OPENROUTER_API_KEY` in environment.
      - Uses the OpenAI SDK with OpenRouter base_url.
      - Returns a float score in [0, 1] and meta about the call.
    """

    def __init__(
        self,
        *,
        model_id: str = "openai/gpt-4o",
        max_tokens: int = 64,
        api_key_env: str = "OPENROUTER_API_KEY",
    ):
        self.model_id = str(model_id)
        self.max_tokens = int(max_tokens)
        self.api_key_env = str(api_key_env)

    def _judge_prompt(self, prompt: str, response: str) -> str:
        return (
            "You are an impartial verifier.\\n"
            "Evaluate ONLY truth / correctness / logical consistency / constraint satisfaction.\\n"
            "Ignore style, fluency, verbosity, and confidence.\\n\\n"
            f"PROMPT:\\n{prompt}\\n\\n"
            f"RESPONSE:\\n{response}\\n\\n"
            "Output STRICTLY one float number between 0.0 and 1.0.\\n"
            "0.0 = wrong/violates constraints; 1.0 = correct.\\n"
        )

    def verify(self, prompt: str, response: str) -> Tuple[float, Dict[str, Any]]:
        from openai import OpenAI

        api_key = os.getenv(self.api_key_env)
        if not api_key:
            return 0.0, {"ok": False, "error": f"{self.api_key_env} not set"}

        client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
        judge = self._judge_prompt(prompt, response)

        try:
            # Reasoning models use max_completion_tokens; keep it simple here.
            resp = client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": judge}],
                max_tokens=int(self.max_tokens),
            )
            content = (resp.choices[0].message.content or "").strip()
            s = _parse_float01(content)
            if s is None:
                return 0.0, {"ok": False, "error": "could_not_parse_score", "raw": content}
            return float(s), {"ok": True, "raw": content, "teacher_model": self.model_id}
        except Exception as e:
            return 0.0, {"ok": False, "error": f"{type(e).__name__}: {e}", "teacher_model": self.model_id}


class TieredGrader:
    """
    Combines:
      - soft_grader: local `QualityCostPredictor`-style model
      - optional teacher_verifier: slow truth checker for "hard" prompts

    Contract:
      - Exposes `predict_production(...)` so it can be passed where a
        `QualityCostPredictor` is expected (e.g. bandit feedback).
    """

    def __init__(
        self,
        *,
        soft_grader: ProductionGrader,
        hard_detector: Optional[HardPromptHeuristics] = None,
        teacher_verifier: Optional[TeacherVerifier] = None,
        code_verifier: Optional[CodeExecutionVerifier] = None,
        reward_clip_eps: float = 0.01,
        reward_logit_eps: float = 1e-4,
    ):
        self.soft_grader = soft_grader
        self.hard_detector = hard_detector or HardPromptHeuristics()
        self.teacher_verifier = teacher_verifier
        self.code_verifier = code_verifier
        self.reward_clip_eps = float(reward_clip_eps)
        self.logit = LogitReward(epsilon=float(reward_logit_eps))

    def predict_production(
        self,
        prompt: str,
        response: str,
        *,
        reward_normalizer: Optional[RunningZScoreNormalizer] = None,
    ) -> Dict[str, Any]:
        # Always compute the soft grader result (useful for meta and fallback).
        soft = self.soft_grader.predict_production(prompt, response, reward_normalizer=reward_normalizer)

        is_hard = self.hard_detector.is_hard(prompt)
        used_teacher = False
        used_code_verifier = False
        teacher_meta: Dict[str, Any] = {}
        code_meta: Dict[str, Any] = {}

        p_correct_raw = float(soft.get("p_correct_raw", soft.get("reward_raw", 0.5)))

        # If hard + teacher available: override p_correct with teacher's truth score.
        if is_hard:
            # Prefer execution verification for code-like prompts when provided.
            p = (prompt or "").lower()
            looks_like_code_task = any(k in p for k in ("python", "code", "function", "class", "sql", "javascript", "typescript"))
            if looks_like_code_task and self.code_verifier is not None:
                s, meta = self.code_verifier.verify(prompt, response)
                used_code_verifier = bool(meta.get("ok", False))
                code_meta = dict(meta)
                if used_code_verifier:
                    p_correct_raw = float(s)
            if (not used_code_verifier) and self.teacher_verifier is not None:
                s, meta = self.teacher_verifier.verify(prompt, response)
                used_teacher = bool(meta.get("ok", False))
                teacher_meta = dict(meta)
                if used_teacher:
                    p_correct_raw = float(s)
            try:
                logger.debug(
                    "tiered_grader_decision",
                    extra={
                        "is_hard": True,
                        "used_teacher": bool(used_teacher),
                        "used_code_verifier": bool(used_code_verifier),
                        "teacher_ok": teacher_meta.get("ok"),
                        "code_ok": code_meta.get("ok"),
                        "p_correct_raw": p_correct_raw,
                    },
                )
            except Exception:
                pass

        # Rebuild production rewards from the chosen p_correct_raw.
        p_correct_clipped = clip01(p_correct_raw, eps=self.reward_clip_eps)
        reward_raw = clipped_quality_reward(p_correct_raw, clip_eps=self.reward_clip_eps)
        reward_logit = float(self.logit.transform(p_correct_raw)) if (response or "").strip() else float(self.logit.min_val * 1.5)

        reward_z = None
        if reward_normalizer is not None:
            # Preserve the "empty response = hard penalty" behavior via soft.grader policy:
            # if response empty, the soft grader already returns reward_z=-3.0; mirror that.
            if (response or "").strip() == "":
                reward_z = -3.0
            else:
                reward_z = float(reward_normalizer.normalize(reward_raw, update=True))

        # Keep routing threshold semantics from the soft grader (calibration); but update
        # the competence_risk based on the chosen truth score.
        competence_risk = float(1.0 - p_correct_clipped)
        out: Dict[str, Any] = dict(soft)
        out.update(
            {
                "p_correct_raw": float(p_correct_raw),
                "p_correct_clipped": float(p_correct_clipped),
                "competence_risk": float(competence_risk),
                "reward_raw": float(reward_raw),
                "reward_logit": float(reward_logit),
                "reward_z": float(reward_z) if reward_z is not None else None,
                "tiered_is_hard": bool(is_hard),
                "tiered_used_teacher": bool(used_teacher),
                "tiered_teacher_meta": teacher_meta,
                "tiered_used_code_verifier": bool(used_code_verifier),
                "tiered_code_verifier_meta": code_meta,
            }
        )
        return out


class UnsafePythonSubprocessVerifier:
    """
    Minimal python code execution verifier using a subprocess.

    WARNING:
      - This is UNSAFE for untrusted code unless run inside a real sandbox.
      - Default behavior is to refuse to run unless allow_unsafe=True.
    """

    def __init__(self, *, allow_unsafe: bool = False, timeout_s: float = 2.0):
        self.allow_unsafe = bool(allow_unsafe)
        self.timeout_s = float(timeout_s)

    def verify(self, prompt: str, response: str) -> Tuple[float, Dict[str, Any]]:
        if not self.allow_unsafe:
            return 0.0, {"ok": False, "error": "unsafe_verifier_disabled"}

        code = (response or "").strip()
        if not code:
            return 0.0, {"ok": False, "error": "empty_response"}

        # Very small wrapper: just check that the code parses and runs without exception.
        wrapped = "try:\n"
        for line in code.splitlines():
            wrapped += f"    {line}\n"
        wrapped += "    print('SUCCESS')\n"
        wrapped += "except Exception as e:\n"
        wrapped += "    print('FAILURE')\n"

        try:
            r = subprocess.run(
                ["python3", "-c", wrapped],
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
            )
            ok = ("SUCCESS" in (r.stdout or "")) and (r.returncode == 0) and not (r.stderr or "").strip()
            return (1.0 if ok else 0.0), {"ok": True, "stdout": r.stdout, "stderr": r.stderr, "returncode": r.returncode}
        except subprocess.TimeoutExpired:
            return 0.0, {"ok": True, "error": "timeout"}

