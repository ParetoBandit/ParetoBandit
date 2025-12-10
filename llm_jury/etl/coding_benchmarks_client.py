"""Client for running HumanEval and MBPP coding benchmarks.

Evaluates models by generating code completions and executing them against test suites.

Sources (Official Repos - cloned to external/):
- HumanEval: github.com/openai/human-eval (164 Python problems)
- MBPP: github.com/google-research/google-research/tree/master/mbpp (~1000 problems)

Metrics:
- pass@1: Probability of generating correct solution in single attempt
- pass@k: Estimated probability with k samples (using unbiased estimator)

KDD Paper Justification:
"HumanEval and MBPP are standard benchmarks for evaluating code generation.
We use pass@1 which measures the probability of generating a correct solution
in a single attempt - the most practically relevant metric for real-world use."
"""

import gzip
import json
import logging
import os
import re
import tempfile
import contextlib
import io
import signal
import multiprocessing
from pathlib import Path
from typing import Dict, List, Optional, Iterable, Any
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
EXTERNAL_DIR = PROJECT_ROOT / "external"
HUMANEVAL_PATH = EXTERNAL_DIR / "human-eval" / "data" / "HumanEval.jsonl.gz"
MBPP_PATH = EXTERNAL_DIR / "google-research-mbpp" / "mbpp" / "mbpp.jsonl"
MBPP_SANITIZED_PATH = EXTERNAL_DIR / "google-research-mbpp" / "mbpp" / "sanitized-mbpp.json"


@dataclass
class CodingProblem:
    """Represents a coding problem from HumanEval or MBPP."""
    task_id: str
    prompt: str
    entry_point: str
    test_code: str
    canonical_solution: Optional[str] = None
    benchmark: str = "humaneval"
    
    def get_full_prompt(self, include_tests: bool = False) -> str:
        """Get the prompt to send to the model."""
        if include_tests and self.benchmark == "mbpp":
            # MBPP often benefits from showing test cases
            return f"{self.prompt}\n\nYour code should pass these tests:\n{self.test_code}"
        return self.prompt


class CodingBenchmarksClient:
    """Client for loading and evaluating HumanEval and MBPP benchmarks."""
    
    def __init__(self, timeout: float = 5.0):
        """Initialize the client.
        
        Args:
            timeout: Timeout in seconds for code execution
        """
        self.timeout = timeout
        self._humaneval_problems: Optional[Dict[str, CodingProblem]] = None
        self._mbpp_problems: Optional[Dict[str, CodingProblem]] = None
    
    def load_humaneval(self) -> Dict[str, CodingProblem]:
        """Load HumanEval problems from the cloned repo.
        
        Returns:
            Dictionary mapping task_id to CodingProblem
        """
        if self._humaneval_problems is not None:
            return self._humaneval_problems
        
        if not HUMANEVAL_PATH.exists():
            raise FileNotFoundError(
                f"HumanEval data not found at {HUMANEVAL_PATH}. "
                "Please clone: git clone https://github.com/openai/human-eval external/human-eval"
            )
        
        problems = {}
        for item in self._stream_jsonl(HUMANEVAL_PATH):
            task_id = item["task_id"]
            problems[task_id] = CodingProblem(
                task_id=task_id,
                prompt=item["prompt"],
                entry_point=item["entry_point"],
                test_code=item["test"],
                canonical_solution=item.get("canonical_solution"),
                benchmark="humaneval"
            )
        
        logger.info(f"Loaded {len(problems)} HumanEval problems")
        self._humaneval_problems = problems
        return problems
    
    def load_mbpp(self, use_sanitized: bool = True) -> Dict[str, CodingProblem]:
        """Load MBPP problems from the cloned repo.
        
        Args:
            use_sanitized: If True, use the sanitized subset (hand-verified)
        
        Returns:
            Dictionary mapping task_id to CodingProblem
        """
        if self._mbpp_problems is not None:
            return self._mbpp_problems
        
        path = MBPP_SANITIZED_PATH if use_sanitized else MBPP_PATH
        
        if not path.exists():
            raise FileNotFoundError(
                f"MBPP data not found at {path}. "
                "Please clone: cd external && git clone --sparse https://github.com/google-research/google-research.git google-research-mbpp && cd google-research-mbpp && git sparse-checkout set mbpp"
            )
        
        problems = {}
        
        if path.suffix == ".json":
            # Sanitized MBPP is a JSON array
            with open(path) as f:
                data = json.load(f)
        else:
            # Full MBPP is JSONL
            data = list(self._stream_jsonl(path))
        
        # Filter to test set (task_ids 11-510 per MBPP README)
        # Note: sanitized MBPP uses "prompt" field, regular MBPP uses "text"
        for item in data:
            task_id = str(item["task_id"])
            task_num = int(task_id)
            
            # For sanitized MBPP, include all (already filtered)
            # For full MBPP, only include test set (11-510)
            if not use_sanitized and (task_num < 11 or task_num > 510):
                continue
            
            # Build test code from test_list
            test_list = item.get("test_list", [])
            test_code = "\n".join(test_list)
            
            # Extract function name from the solution code
            code = item.get("code", "")
            entry_point = self._extract_function_name(code)
            
            # Build prompt from task description
            # Sanitized uses "prompt", full uses "text"
            task_text = item.get("prompt") or item.get("text", "")
            prompt = f'"""{task_text}"""\n'
            
            problems[task_id] = CodingProblem(
                task_id=task_id,
                prompt=prompt,
                entry_point=entry_point,
                test_code=test_code,
                canonical_solution=code,
                benchmark="mbpp"
            )
        
        logger.info(f"Loaded {len(problems)} MBPP problems (test set)")
        self._mbpp_problems = problems
        return problems
    
    def _extract_function_name(self, code: str) -> str:
        """Extract the main function name from Python code."""
        match = re.search(r'def\s+(\w+)\s*\(', code)
        return match.group(1) if match else "solution"
    
    def _stream_jsonl(self, filename: Path) -> Iterable[Dict]:
        """Parse JSONL file (optionally gzipped)."""
        filename_str = str(filename)
        if filename_str.endswith(".gz"):
            with open(filename, "rb") as gzfp:
                with gzip.open(gzfp, 'rt') as fp:
                    for line in fp:
                        if any(not x.isspace() for x in line):
                            yield json.loads(line)
        else:
            with open(filename, "r") as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        yield json.loads(line)
    
    def check_correctness(
        self,
        problem: CodingProblem,
        completion: str,
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """Check if a completion is correct by running tests.
        
        Args:
            problem: The coding problem
            completion: The model's code completion
            timeout: Timeout in seconds (uses instance default if not provided)
        
        Returns:
            Dict with task_id, passed (bool), result (str), completion
        """
        timeout = timeout or self.timeout
        
        # Build the check program
        if problem.benchmark == "humaneval":
            check_program = (
                problem.prompt
                + completion
                + "\n"
                + problem.test_code
                + "\n"
                + f"check({problem.entry_point})"
            )
        else:  # MBPP
            check_program = completion + "\n" + problem.test_code
        
        # Use standalone function for subprocess execution
        result = _check_code_in_subprocess(check_program, timeout)
        
        return {
            "task_id": problem.task_id,
            "passed": result == "passed",
            "result": result,
            "completion": completion
        }
    


class TimeoutException(Exception):
    """Exception raised when code execution times out."""
    pass


def _check_code_in_subprocess(check_program: str, timeout: float) -> str:
    """Execute code in a subprocess and return result.
    
    Uses subprocess.run with a temp file for reliable isolation.
    
    Args:
        check_program: The full Python program to execute
        timeout: Timeout in seconds
    
    Returns:
        "passed", "timed out", or "failed: <error>"
    """
    import subprocess
    
    # Write code to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(check_program)
        temp_file = f.name
    
    try:
        result = subprocess.run(
            ['python', temp_file],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, 'OMP_NUM_THREADS': '1'}
        )
        
        if result.returncode == 0:
            return "passed"
        else:
            # Get the last line of stderr for error message
            error = result.stderr.strip().split('\n')[-1] if result.stderr else "unknown error"
            return f"failed: {error}"
            
    except subprocess.TimeoutExpired:
        return "timed out"
    except Exception as e:
        return f"failed: {e}"
    finally:
        # Clean up temp file
        try:
            os.unlink(temp_file)
        except:
            pass


def estimate_pass_at_k(
    num_samples: int,
    num_correct: int,
    k: int
) -> float:
    """Estimate pass@k using the unbiased estimator.
    
    Calculates 1 - comb(n - c, k) / comb(n, k).
    
    Args:
        num_samples: Total number of samples (n)
        num_correct: Number of correct samples (c)
        k: The k in pass@k
    
    Returns:
        Estimated pass@k probability
    """
    if num_samples - num_correct < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(num_samples - num_correct + 1, num_samples + 1))


def calculate_pass_at_k(
    results: List[Dict[str, Any]],
    k: int = 1
) -> float:
    """Calculate pass@k from evaluation results.
    
    Args:
        results: List of check_correctness results
        k: The k in pass@k (default 1 for pass@1)
    
    Returns:
        pass@k score (0-100 scale)
    """
    # Group by task_id
    task_results = {}
    for r in results:
        task_id = r["task_id"]
        if task_id not in task_results:
            task_results[task_id] = {"total": 0, "correct": 0}
        task_results[task_id]["total"] += 1
        if r["passed"]:
            task_results[task_id]["correct"] += 1
    
    # Calculate pass@k for each task
    pass_at_k_values = []
    for task_id, counts in task_results.items():
        if counts["total"] >= k:
            p = estimate_pass_at_k(counts["total"], counts["correct"], k)
            pass_at_k_values.append(p)
    
    if not pass_at_k_values:
        return 0.0
    
    # Return mean pass@k as percentage
    return float(np.mean(pass_at_k_values) * 100)


# Code generation prompt templates
HUMANEVAL_PROMPT_TEMPLATE = """{prompt}"""

MBPP_PROMPT_TEMPLATE = """You are an expert Python programmer. Write a Python function to solve the following problem.

{prompt}

Your code should pass these tests:
{tests}

Write only the Python function, no explanations."""


def format_humaneval_prompt(problem: CodingProblem) -> str:
    """Format a HumanEval problem for the model."""
    return problem.prompt


def format_mbpp_prompt(problem: CodingProblem) -> str:
    """Format an MBPP problem for the model."""
    return MBPP_PROMPT_TEMPLATE.format(
        prompt=problem.prompt,
        tests=problem.test_code
    )


def extract_code_from_response(response: str, problem: CodingProblem) -> str:
    """Extract Python code from model response.
    
    Handles markdown code blocks and strips explanations.
    
    Args:
        response: Raw model response
        problem: The coding problem (for context)
    
    Returns:
        Extracted Python code
    """
    if not response:
        return ""
    
    # Try to extract from markdown code block
    code_block_pattern = r'```(?:python)?\s*\n(.*?)```'
    matches = re.findall(code_block_pattern, response, re.DOTALL)
    if matches:
        # Return the first (or longest) code block
        return max(matches, key=len).strip()
    
    # For HumanEval, the response should be just the completion
    if problem.benchmark == "humaneval":
        # Remove any leading/trailing markdown or explanations
        lines = response.strip().split('\n')
        code_lines = []
        in_code = False
        
        for line in lines:
            # Skip obvious non-code lines
            if line.strip().startswith(('#', '//', '/*', '*')):
                continue
            if 'explanation' in line.lower() or 'note:' in line.lower():
                continue
            
            # Heuristic: code lines typically start with whitespace or def/class/return/if/for/etc
            stripped = line.strip()
            if stripped and (line[0].isspace() or 
                           stripped.startswith(('def ', 'class ', 'return ', 'if ', 'for ', 'while ', 'try:', 'except', 'with ', 'import ', 'from ')) or
                           '=' in stripped or
                           stripped.startswith(('(', '[', '{', '"', "'"))):
                code_lines.append(line)
                in_code = True
            elif in_code and (line.strip() == '' or line[0].isspace()):
                code_lines.append(line)
        
        return '\n'.join(code_lines).strip()
    
    # For MBPP, we need the full function
    return response.strip()
