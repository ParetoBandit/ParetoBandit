#!/usr/bin/env python3
"""
Evaluate code generation with execution-based testing (Pass@1 metric).

This script evaluates model-generated code by running it against test cases
and computing the Pass@1 metric (percentage of problems where the first
generated solution passes all test cases).

Evaluation is FREE - runs locally on CPU with no API costs.

Usage:
    python evaluate_code.py --problems prompts.json --responses responses.json
    
Safety:
    - Runs code in subprocess with timeout
    - Can use Docker for isolation (optional)
    - Validates output format
"""

import json
import sys
import subprocess
import tempfile
import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import re


class CodeExecutor:
    """Execute Python code safely with test cases."""
    
    def __init__(
        self,
        timeout: int = 5,
        use_docker: bool = False,
        max_memory_mb: int = 512
    ):
        """
        Initialize code executor.
        
        Args:
            timeout: Execution timeout in seconds
            use_docker: Use Docker for isolation (requires Docker installed)
            max_memory_mb: Maximum memory limit in MB
        """
        self.timeout = timeout
        self.use_docker = use_docker
        self.max_memory_mb = max_memory_mb
    
    def execute_code(
        self,
        code: str,
        test_input: str,
        expected_output: str
    ) -> Tuple[bool, str]:
        """
        Execute code with a test case and check if output matches.
        
        Args:
            code: Python code to execute
            test_input: Input to pass to the code
            expected_output: Expected output
            
        Returns:
            (passed: bool, message: str)
        """
        # Create temporary file with code
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            # Wrap code to read input and print output
            wrapped_code = self._wrap_code(code, test_input)
            f.write(wrapped_code)
            temp_file = f.name
        
        try:
            if self.use_docker:
                result = self._execute_docker(temp_file, test_input)
            else:
                result = self._execute_subprocess(temp_file, test_input)
            
            success, output, error = result
            
            if not success:
                return False, f"Execution error: {error}"
            
            # Compare outputs
            if self._outputs_match(output, expected_output):
                return True, "PASS"
            else:
                return False, f"Output mismatch. Expected: {expected_output[:100]}, Got: {output[:100]}"
        
        finally:
            # Cleanup
            try:
                Path(temp_file).unlink()
            except:
                pass
    
    def _wrap_code(self, code: str, test_input: str) -> str:
        """Wrap code to handle input/output."""
        # Extract function name if it's a function definition
        func_match = re.search(r'def\s+(\w+)\s*\(', code)
        
        if func_match:
            func_name = func_match.group(1)
            # Parse input as function arguments
            wrapped = f"""{code}

if __name__ == "__main__":
    import sys
    import json
    try:
        # Parse input
        test_input = {repr(test_input)}
        # Try to evaluate as Python literal
        try:
            args = eval(test_input)
            if not isinstance(args, tuple):
                args = (args,)
        except:
            args = (test_input,)
        
        # Call function
        result = {func_name}(*args)
        print(result)
    except Exception as e:
        print(f"Error: {{e}}", file=sys.stderr)
        sys.exit(1)
"""
        else:
            # Standalone code
            wrapped = f"""import sys

{code}

if __name__ == "__main__":
    try:
        # Code should handle input/output itself
        pass
    except Exception as e:
        print(f"Error: {{e}}", file=sys.stderr)
        sys.exit(1)
"""
        
        return wrapped
    
    def _execute_subprocess(
        self,
        script_path: str,
        test_input: str
    ) -> Tuple[bool, str, str]:
        """Execute code in subprocess."""
        try:
            result = subprocess.run(
                [sys.executable, script_path],
                input=test_input,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            success = (result.returncode == 0)
            return success, result.stdout.strip(), result.stderr.strip()
            
        except subprocess.TimeoutExpired:
            return False, "", f"Timeout ({self.timeout}s exceeded)"
        except Exception as e:
            return False, "", str(e)
    
    def _execute_docker(
        self,
        script_path: str,
        test_input: str
    ) -> Tuple[bool, str, str]:
        """Execute code in Docker container (safer isolation)."""
        # TODO: Implement Docker execution
        # docker run --rm -v script_path:/code.py python:3.10 python /code.py
        raise NotImplementedError("Docker execution not yet implemented")
    
    def _outputs_match(self, actual: str, expected: str) -> bool:
        """Check if outputs match (with normalization)."""
        # Normalize whitespace
        actual_norm = ' '.join(actual.split())
        expected_norm = ' '.join(expected.split())
        
        if actual_norm == expected_norm:
            return True
        
        # Try parsing as numbers for numerical comparison
        try:
            actual_num = float(actual_norm)
            expected_num = float(expected_norm)
            return abs(actual_num - expected_num) < 1e-6
        except:
            pass
        
        # Try parsing as JSON
        try:
            actual_json = json.loads(actual)
            expected_json = json.loads(expected)
            return actual_json == expected_json
        except:
            pass
        
        return False


def evaluate_response(
    problem: Dict,
    response_code: str,
    executor: CodeExecutor
) -> Dict:
    """
    Evaluate a single response against all test cases.
    
    Args:
        problem: Problem dictionary with test cases
        response_code: Generated code
        executor: Code executor instance
        
    Returns:
        Evaluation results dictionary
    """
    test_cases = problem.get("test_cases", [])
    
    if not test_cases:
        return {
            "passed": False,
            "reason": "No test cases available",
            "tests_passed": 0,
            "tests_total": 0,
            "pass_rate": 0.0
        }
    
    results = []
    for i, tc in enumerate(test_cases):
        test_input = tc.get("input", "")
        expected_output = tc.get("output", "")
        
        passed, message = executor.execute_code(
            response_code,
            test_input,
            expected_output
        )
        
        results.append({
            "test_id": i,
            "passed": passed,
            "message": message,
            "input": test_input[:100],  # Truncate for storage
            "expected": expected_output[:100]
        })
    
    tests_passed = sum(1 for r in results if r["passed"])
    tests_total = len(results)
    all_passed = (tests_passed == tests_total)
    
    return {
        "passed": all_passed,
        "reason": "All tests passed" if all_passed else f"Failed {tests_total - tests_passed}/{tests_total} tests",
        "tests_passed": tests_passed,
        "tests_total": tests_total,
        "pass_rate": tests_passed / tests_total if tests_total > 0 else 0.0,
        "test_results": results
    }


def compute_pass_at_k(results: List[Dict], k: int = 1) -> float:
    """
    Compute Pass@k metric.
    
    For k=1: Percentage of problems where the first solution passed.
    
    Args:
        results: List of evaluation results
        k: Number of attempts considered (typically 1)
        
    Returns:
        Pass@k score (0.0 to 1.0)
    """
    if not results:
        return 0.0
    
    passed_count = sum(1 for r in results if r.get("passed", False))
    return passed_count / len(results)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate code generation with execution-based testing"
    )
    parser.add_argument(
        "--problems", type=str, required=True,
        help="JSON file with problems and test cases"
    )
    parser.add_argument(
        "--responses", type=str, required=True,
        help="JSON file with model responses (code)"
    )
    parser.add_argument(
        "--output", type=str, default="evaluation_results.json",
        help="Output file for results"
    )
    parser.add_argument(
        "--timeout", type=int, default=5,
        help="Execution timeout in seconds"
    )
    parser.add_argument(
        "--use-docker", action="store_true",
        help="Use Docker for isolation (safer)"
    )
    parser.add_argument(
        "--max-problems", type=int, default=None,
        help="Maximum problems to evaluate"
    )
    
    args = parser.parse_args()
    
    # Load problems
    print(f"Loading problems from {args.problems}...")
    with open(args.problems) as f:
        problems_data = json.load(f)
    
    problems = problems_data.get("problems", problems_data)
    if args.max_problems:
        problems = problems[:args.max_problems]
    
    print(f"✓ Loaded {len(problems)} problems")
    
    # Load responses
    print(f"Loading responses from {args.responses}...")
    with open(args.responses) as f:
        responses_data = json.load(f)
    
    # Responses format: {problem_id: code}
    responses = responses_data.get("responses", responses_data)
    print(f"✓ Loaded {len(responses)} responses")
    
    # Initialize executor
    executor = CodeExecutor(
        timeout=args.timeout,
        use_docker=args.use_docker
    )
    
    # Evaluate each response
    results = []
    print(f"\nEvaluating {len(problems)} problems...")
    print("="*60)
    
    for i, problem in enumerate(problems, 1):
        problem_id = problem.get("problem_id", f"problem_{i}")
        
        # Get response
        response_code = responses.get(problem_id)
        
        if not response_code:
            result = {
                "problem_id": problem_id,
                "passed": False,
                "reason": "No response found",
                "tests_passed": 0,
                "tests_total": len(problem.get("test_cases", [])),
                "pass_rate": 0.0
            }
        else:
            # Evaluate
            print(f"[{i}/{len(problems)}] {problem_id}...", end=" ", flush=True)
            
            result = evaluate_response(problem, response_code, executor)
            result["problem_id"] = problem_id
            
            status = "✓ PASS" if result["passed"] else "✗ FAIL"
            print(f"{status} ({result['tests_passed']}/{result['tests_total']})")
        
        results.append(result)
    
    # Compute metrics
    pass_at_1 = compute_pass_at_k(results, k=1)
    total_tests_passed = sum(r["tests_passed"] for r in results)
    total_tests = sum(r["tests_total"] for r in results)
    
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    print(f"Problems Evaluated:     {len(results)}")
    print(f"Problems Passed:        {sum(1 for r in results if r['passed'])}")
    print(f"Pass@1:                 {pass_at_1*100:.1f}%")
    print(f"Total Tests Passed:     {total_tests_passed}/{total_tests}")
    print(f"Overall Test Pass Rate: {total_tests_passed/total_tests*100:.1f}%")
    
    # Save results
    output_data = {
        "metadata": {
            "evaluation_date": datetime.now().isoformat(),
            "problems_file": args.problems,
            "responses_file": args.responses,
            "timeout": args.timeout,
            "use_docker": args.use_docker,
        },
        "metrics": {
            "pass_at_1": pass_at_1,
            "problems_evaluated": len(results),
            "problems_passed": sum(1 for r in results if r["passed"]),
            "total_tests_passed": total_tests_passed,
            "total_tests": total_tests,
            "overall_test_pass_rate": total_tests_passed / total_tests if total_tests > 0 else 0.0
        },
        "results": results
    }
    
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ Saved results to {output_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())
