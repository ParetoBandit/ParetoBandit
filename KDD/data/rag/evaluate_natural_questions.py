#!/usr/bin/env python3
"""
Natural Questions Exact Match Evaluator

Evaluates model responses against Natural Questions using exact match scoring.
Handles multiple valid answers and flexible matching strategies.

Metric: Exact Match
- Model response must contain at least one valid answer
- Case-insensitive, punctuation-normalized matching
"""

import re
import json
import string
from typing import List, Dict, Union, Tuple
from pathlib import Path
from dataclasses import dataclass


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    question: str
    ground_truth: List[str]
    model_response: str
    is_correct: bool
    matched_answer: Union[str, None]
    normalized_response: str


class NaturalQuestionsEvaluator:
    """
    Evaluator for Natural Questions with exact match scoring.
    """
    
    def __init__(self, case_sensitive: bool = False):
        """
        Initialize evaluator.
        
        Args:
            case_sensitive: Whether to use case-sensitive matching
        """
        self.case_sensitive = case_sensitive
    
    def normalize_answer(self, text: str) -> str:
        """
        Normalize answer text for comparison.
        
        Normalization steps:
        1. Convert to lowercase (unless case_sensitive)
        2. Remove articles (a, an, the)
        3. Remove punctuation
        4. Remove extra whitespace
        5. Strip leading/trailing spaces
        
        Args:
            text: Text to normalize
        
        Returns:
            Normalized text
        """
        if not self.case_sensitive:
            text = text.lower()
        
        # Remove articles
        text = re.sub(r'\b(a|an|the)\b', ' ', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def extract_answer_from_response(self, response: str) -> str:
        """
        Extract the answer from a model response.
        
        Handles various response formats:
        - "Answer: Rocky Mountains"
        - "The answer is Rocky Mountains."
        - "Rocky Mountains"
        
        Args:
            response: Full model response
        
        Returns:
            Extracted answer
        """
        # Try to extract after "Answer:" or "answer is"
        patterns = [
            r'(?:Answer|answer)\s*:\s*(.+?)(?:\.|$)',
            r'(?:The answer is|answer is)\s+(.+?)(?:\.|$)',
            r'^(.+?)(?:\.|$)'  # Fallback: first sentence
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return response.strip()
    
    def check_exact_match(
        self,
        response: str,
        valid_answers: List[str]
    ) -> Tuple[bool, Union[str, None]]:
        """
        Check if response matches any valid answer.
        
        Args:
            response: Model's response
            valid_answers: List of acceptable answers
        
        Returns:
            Tuple of (is_correct, matched_answer)
        """
        # Extract and normalize response
        extracted = self.extract_answer_from_response(response)
        normalized_response = self.normalize_answer(extracted)
        
        # Check against each valid answer
        for answer in valid_answers:
            normalized_answer = self.normalize_answer(answer)
            
            # Check if normalized answer appears in normalized response
            if normalized_answer in normalized_response or normalized_response in normalized_answer:
                return True, answer
        
        return False, None
    
    def evaluate(
        self,
        questions: List[Dict],
        responses: List[str]
    ) -> Tuple[List[EvaluationResult], Dict]:
        """
        Evaluate model responses against ground truth.
        
        Args:
            questions: List of question dicts with 'question' and 'answers'
            responses: List of model responses (same order as questions)
        
        Returns:
            Tuple of (detailed_results, summary_metrics)
        """
        if len(questions) != len(responses):
            raise ValueError(f"Mismatch: {len(questions)} questions but {len(responses)} responses")
        
        results = []
        correct_count = 0
        
        for q, response in zip(questions, responses):
            is_correct, matched = self.check_exact_match(
                response,
                q["answers"]
            )
            
            if is_correct:
                correct_count += 1
            
            result = EvaluationResult(
                question=q["question"],
                ground_truth=q["answers"],
                model_response=response,
                is_correct=is_correct,
                matched_answer=matched,
                normalized_response=self.normalize_answer(
                    self.extract_answer_from_response(response)
                )
            )
            results.append(result)
        
        # Compute metrics
        accuracy = correct_count / len(questions) if questions else 0.0
        
        metrics = {
            "total_questions": len(questions),
            "correct": correct_count,
            "incorrect": len(questions) - correct_count,
            "accuracy": accuracy,
            "exact_match_rate": accuracy  # Same as accuracy for this metric
        }
        
        return results, metrics
    
    def evaluate_from_file(
        self,
        questions_file: str,
        responses_file: str
    ) -> Tuple[List[EvaluationResult], Dict]:
        """
        Evaluate from JSON files.
        
        Args:
            questions_file: Path to questions JSON
            responses_file: Path to responses JSON (list of strings)
        
        Returns:
            Tuple of (detailed_results, summary_metrics)
        """
        with open(questions_file) as f:
            questions = json.load(f)
        
        with open(responses_file) as f:
            responses = json.load(f)
        
        return self.evaluate(questions, responses)


def save_evaluation_results(
    results: List[EvaluationResult],
    metrics: Dict,
    output_path: str = "evaluation_results.json"
) -> None:
    """Save evaluation results to JSON."""
    output_data = {
        "metrics": metrics,
        "detailed_results": [
            {
                "question": r.question,
                "ground_truth": r.ground_truth,
                "model_response": r.model_response,
                "normalized_response": r.normalized_response,
                "is_correct": r.is_correct,
                "matched_answer": r.matched_answer
            }
            for r in results
        ]
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"💾 Saved evaluation results to {output_path}")


def main():
    """Example usage with mock responses."""
    # Example questions
    questions = [
        {
            "question": "What mountain range runs through Colorado?",
            "answers": ["Rocky Mountains", "The Rockies", "Rockies"]
        },
        {
            "question": "Who wrote 'Romeo and Juliet'?",
            "answers": ["William Shakespeare", "Shakespeare"]
        },
        {
            "question": "What is the capital of France?",
            "answers": ["Paris"]
        }
    ]
    
    # Mock model responses (some correct, some incorrect)
    responses = [
        "Answer: The Rocky Mountains run through Colorado.",
        "The answer is William Shakespeare wrote that play.",
        "The capital of France is London."  # Incorrect
    ]
    
    # Evaluate
    evaluator = NaturalQuestionsEvaluator(case_sensitive=False)
    results, metrics = evaluator.evaluate(questions, responses)
    
    # Print results
    print("\n📊 Evaluation Results\n")
    print(f"Total Questions: {metrics['total_questions']}")
    print(f"Correct: {metrics['correct']}")
    print(f"Incorrect: {metrics['incorrect']}")
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"Exact Match Rate: {metrics['exact_match_rate']:.2%}")
    
    print("\n📝 Detailed Results:\n")
    for i, result in enumerate(results, 1):
        status = "✓" if result.is_correct else "✗"
        print(f"{status} Question {i}: {result.question}")
        print(f"  Valid Answers: {result.ground_truth}")
        print(f"  Model Response: {result.model_response}")
        if result.matched_answer:
            print(f"  Matched: '{result.matched_answer}'")
        print()


if __name__ == "__main__":
    main()
