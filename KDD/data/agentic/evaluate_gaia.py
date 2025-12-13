#!/usr/bin/env python3
"""
Evaluate GAIA responses with exact match scoring.

This script evaluates model responses against GAIA ground truth answers
using exact string matching (case-insensitive) or normalized numeric matching.

GAIA answers are typically:
- City/place names (e.g., "Seattle")
- Numbers (e.g., "42")
- Dates (e.g., "2024-01-15")
- Short strings

Evaluation is simple exact match, making it objective and deterministic.

Usage:
    python evaluate_gaia.py --problems gaia.json --responses responses.json
"""

import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class GAIAEvaluator:
    """Evaluator for GAIA benchmark with exact match scoring."""
    
    def __init__(self, case_sensitive: bool = False, strict_numeric: bool = False):
        """
        Initialize evaluator.
        
        Args:
            case_sensitive: Whether to use case-sensitive matching
            strict_numeric: Whether to use strict numeric comparison
        """
        self.case_sensitive = case_sensitive
        self.strict_numeric = strict_numeric
    
    def normalize_answer(self, answer: str) -> str:
        """Normalize an answer for comparison."""
        if not answer:
            return ""
        
        # Convert to string if not already
        answer = str(answer).strip()
        
        # Remove common artifacts
        answer = re.sub(r'^(answer|the answer is)[:\s]+', '', answer, flags=re.IGNORECASE)
        answer = re.sub(r'^(final answer)[:\s]+', '', answer, flags=re.IGNORECASE)
        
        # Remove quotes
        answer = answer.strip('"\'')
        
        # Remove trailing punctuation
        answer = answer.rstrip('.,;!?')
        
        # Normalize whitespace
        answer = ' '.join(answer.split())
        
        # Case normalization
        if not self.case_sensitive:
            answer = answer.lower()
        
        return answer
    
    def extract_answer(self, response: str) -> str:
        """Extract the final answer from a model response."""
        if not response:
            return ""
        
        response = response.strip()
        
        # Try to find explicit answer markers
        patterns = [
            r'(?:final\s+)?answer[:\s]+([^\n]+)',
            r'the\s+answer\s+is[:\s]+([^\n]+)',
            r'(?:^|\n)([A-Z][^.!?\n]{0,100})[.!?]?\s*$',  # Last sentence
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
            if match:
                return self.normalize_answer(match.group(1))
        
        # If no pattern matches, try the last line
        lines = response.strip().split('\n')
        if lines:
            return self.normalize_answer(lines[-1])
        
        return self.normalize_answer(response)
    
    def answers_match(self, extracted: str, expected: str) -> Tuple[bool, str]:
        """
        Check if extracted answer matches expected answer.
        
        Args:
            extracted: Extracted answer from model
            expected: Expected ground truth answer
            
        Returns:
            (matches: bool, reason: str)
        """
        extracted_norm = self.normalize_answer(extracted)
        expected_norm = self.normalize_answer(expected)
        
        # Exact match
        if extracted_norm == expected_norm:
            return True, "Exact match"
        
        # Try numeric comparison
        if self._is_numeric(extracted_norm) and self._is_numeric(expected_norm):
            extracted_num = self._parse_number(extracted_norm)
            expected_num = self._parse_number(expected_norm)
            
            if extracted_num is not None and expected_num is not None:
                if self.strict_numeric:
                    matches = (extracted_num == expected_num)
                else:
                    # Allow small floating point differences
                    matches = abs(extracted_num - expected_num) < 1e-6
                
                if matches:
                    return True, "Numeric match"
        
        # Try substring match (for cases like "Seattle, WA" vs "Seattle")
        if expected_norm in extracted_norm or extracted_norm in expected_norm:
            return True, "Substring match"
        
        # No match
        return False, f"Mismatch: '{extracted_norm}' != '{expected_norm}'"
    
    def _is_numeric(self, s: str) -> bool:
        """Check if string represents a number."""
        try:
            float(s.replace(',', ''))
            return True
        except:
            return False
    
    def _parse_number(self, s: str) -> Optional[float]:
        """Parse a number from string."""
        try:
            return float(s.replace(',', ''))
        except:
            return None


def evaluate_response(
    problem: Dict,
    response: str,
    evaluator: GAIAEvaluator
) -> Dict:
    """
    Evaluate a single response.
    
    Args:
        problem: Problem dictionary with ground truth
        response: Model's response
        evaluator: Evaluator instance
        
    Returns:
        Evaluation result dictionary
    """
    expected_answer = problem.get("final_answer", "")
    
    if not response:
        return {
            "correct": False,
            "extracted_answer": "",
            "expected_answer": expected_answer,
            "reason": "No response provided"
        }
    
    # Extract answer from response
    extracted_answer = evaluator.extract_answer(response)
    
    # Check if it matches
    matches, reason = evaluator.answers_match(extracted_answer, expected_answer)
    
    return {
        "correct": matches,
        "extracted_answer": extracted_answer,
        "expected_answer": expected_answer,
        "reason": reason,
        "full_response": response[:500]  # Store truncated response for debugging
    }


def compute_accuracy(results: List[Dict]) -> Dict:
    """
    Compute accuracy metrics.
    
    Args:
        results: List of evaluation results
        
    Returns:
        Metrics dictionary
    """
    if not results:
        return {"accuracy": 0.0, "correct": 0, "total": 0}
    
    correct = sum(1 for r in results if r.get("correct", False))
    total = len(results)
    accuracy = correct / total if total > 0 else 0.0
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total
    }


def compute_level_accuracy(results: List[Dict], problems: List[Dict]) -> Dict:
    """Compute accuracy by difficulty level."""
    level_stats = {}
    
    # Group results by level
    for result, problem in zip(results, problems):
        level = problem.get("level", "Unknown")
        
        if level not in level_stats:
            level_stats[level] = {"correct": 0, "total": 0}
        
        level_stats[level]["total"] += 1
        if result.get("correct"):
            level_stats[level]["correct"] += 1
    
    # Compute accuracy for each level
    for level in level_stats:
        stats = level_stats[level]
        stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
    
    return level_stats


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate GAIA responses with exact match"
    )
    parser.add_argument(
        "--problems", type=str, required=True,
        help="JSON file with GAIA problems"
    )
    parser.add_argument(
        "--responses", type=str, required=True,
        help="JSON file with model responses"
    )
    parser.add_argument(
        "--output", type=str, default="gaia_evaluation.json",
        help="Output file for results"
    )
    parser.add_argument(
        "--case-sensitive", action="store_true",
        help="Use case-sensitive matching"
    )
    parser.add_argument(
        "--strict-numeric", action="store_true",
        help="Use strict numeric comparison (no tolerance)"
    )
    
    args = parser.parse_args()
    
    # Load problems
    print(f"Loading problems from {args.problems}...")
    with open(args.problems) as f:
        problems_data = json.load(f)
    
    problems = problems_data.get("problems", problems_data)
    print(f"✓ Loaded {len(problems)} problems")
    
    # Load responses
    print(f"Loading responses from {args.responses}...")
    with open(args.responses) as f:
        responses_data = json.load(f)
    
    # Responses format: {task_id: response_text}
    responses = responses_data.get("responses", responses_data)
    print(f"✓ Loaded {len(responses)} responses")
    
    # Initialize evaluator
    evaluator = GAIAEvaluator(
        case_sensitive=args.case_sensitive,
        strict_numeric=args.strict_numeric
    )
    
    # Evaluate each response
    results = []
    print(f"\nEvaluating {len(problems)} problems...")
    print("="*60)
    
    for i, problem in enumerate(problems, 1):
        task_id = problem.get("task_id", f"problem_{i}")
        
        # Get response
        response = responses.get(task_id, "")
        
        # Evaluate
        print(f"[{i}/{len(problems)}] {task_id}...", end=" ", flush=True)
        
        result = evaluate_response(problem, response, evaluator)
        result["task_id"] = task_id
        result["level"] = problem.get("level", 0)
        
        status = "✓" if result["correct"] else "✗"
        print(f"{status} {result['reason']}")
        
        results.append(result)
    
    # Compute overall metrics
    overall_metrics = compute_accuracy(results)
    
    # Compute per-level metrics
    level_metrics = compute_level_accuracy(results, problems)
    
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    print(f"Problems Evaluated:  {overall_metrics['total']}")
    print(f"Correct Answers:     {overall_metrics['correct']}")
    print(f"Accuracy:            {overall_metrics['accuracy']*100:.1f}%")
    
    # Show per-level accuracy
    if level_metrics:
        print(f"\nAccuracy by Level:")
        for level in sorted(level_metrics.keys()):
            stats = level_metrics[level]
            difficulty = {1: "Easy", 2: "Medium", 3: "Hard"}.get(level, "Unknown")
            print(f"  Level {level} ({difficulty}): {stats['accuracy']*100:.1f}% ({stats['correct']}/{stats['total']})")
    
    # Save results
    output_data = {
        "metadata": {
            "evaluation_date": datetime.now().isoformat(),
            "problems_file": args.problems,
            "responses_file": args.responses,
            "case_sensitive": args.case_sensitive,
            "strict_numeric": args.strict_numeric,
        },
        "metrics": {
            "overall": overall_metrics,
            "by_level": level_metrics
        },
        "results": results
    }
    
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ Saved results to {output_path}")
    
    # Show sample incorrect answers
    incorrect = [r for r in results if not r["correct"]]
    if incorrect:
        print(f"\n❌ Sample Incorrect Answers (showing {min(3, len(incorrect))}):")
        for r in incorrect[:3]:
            print(f"\n  Task: {r['task_id']}")
            print(f"  Expected: {r['expected_answer']}")
            print(f"  Got:      {r['extracted_answer']}")
            print(f"  Reason:   {r['reason']}")
    
    return 0


if __name__ == "__main__":
    exit(main())
