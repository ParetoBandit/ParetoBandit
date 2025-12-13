#!/usr/bin/env python3
"""
Natural Questions (Open) Dataset Fetcher

This script fetches the Natural Questions Open dataset from HuggingFace,
which is the gold standard for evaluating RAG (Retrieval-Augmented Generation) systems.

Dataset: Natural Questions (Open)
HuggingFace ID: nq_open
Task: Open-domain question answering
Metric: Exact Match (any valid answer)
"""

import os
import json
from typing import List, Dict, Optional
from pathlib import Path
from datasets import load_dataset


def fetch_natural_questions(
    split: str = "validation",
    n_samples: Optional[int] = None,
    seed: int = 42
) -> List[Dict]:
    """
    Fetch Natural Questions Open dataset from HuggingFace.
    
    Args:
        split: Dataset split ('train' or 'validation')
        n_samples: Number of samples to fetch (None = all)
        seed: Random seed for sampling
    
    Returns:
        List of dictionaries with question and answer(s)
    """
    print(f"📥 Fetching Natural Questions ({split} split)...")
    
    # Load dataset from HuggingFace
    # Note: There are two versions:
    # 1. "google-research-datasets/natural_questions" (full with HTML)
    # 2. "nq_open" (simplified, open-domain version) <- We use this
    dataset = load_dataset("nq_open", split=split)
    
    print(f"✓ Loaded {len(dataset)} questions from Natural Questions")
    
    # Optionally sample
    if n_samples is not None and n_samples < len(dataset):
        dataset = dataset.shuffle(seed=seed).select(range(n_samples))
        print(f"✓ Sampled {n_samples} questions (seed={seed})")
    
    # Convert to list of dicts
    questions = []
    for idx, item in enumerate(dataset):
        questions.append({
            "id": idx,
            "question": item["question"],
            "answers": item["answer"],  # List of valid answer strings
            "source": "natural_questions_open"
        })
    
    return questions


def create_rag_prompt(
    question: str,
    retrieved_context: Optional[str] = None,
    style: str = "standard"
) -> str:
    """
    Create a prompt for RAG evaluation.
    
    Args:
        question: The question to answer
        retrieved_context: Optional retrieved context/documents
        style: Prompt style ('standard', 'with_context', 'instruction_following')
    
    Returns:
        Formatted prompt string
    """
    if style == "standard":
        # Simple question without context (tests model's parametric knowledge)
        return f"""Answer the following question concisely. Provide a short, factual answer.

Question: {question}

Answer:"""
    
    elif style == "with_context" and retrieved_context:
        # RAG-style with retrieved context
        return f"""Answer the following question using the provided context. If the answer is not in the context, say "I don't know."

Context: {retrieved_context}

Question: {question}

Answer:"""
    
    elif style == "instruction_following":
        # More detailed instructions
        return f"""You are a helpful assistant. Answer the following question accurately and concisely. Provide only the answer without additional explanation.

Question: {question}

Answer:"""
    
    else:
        return f"Question: {question}\nAnswer:"


def save_questions(
    questions: List[Dict],
    output_path: str = "natural_questions.json"
) -> None:
    """Save questions to JSON file."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(questions, f, indent=2)
    
    print(f"💾 Saved {len(questions)} questions to {output_file}")


def main():
    """Example usage."""
    # Fetch validation set (smaller, good for evaluation)
    questions = fetch_natural_questions(
        split="validation",
        n_samples=100,  # Sample 100 for quick testing
        seed=42
    )
    
    # Save to file
    save_questions(questions, "natural_questions_validation_100.json")
    
    # Print examples
    print("\n📝 Sample Questions:\n")
    for i, q in enumerate(questions[:3]):
        print(f"Question {i+1}: {q['question']}")
        print(f"Valid Answers: {q['answers']}")
        print()
    
    # Show prompt examples
    print("📋 Prompt Examples:\n")
    sample_q = questions[0]["question"]
    
    print("Style: standard")
    print(create_rag_prompt(sample_q, style="standard"))
    print("\n" + "="*80 + "\n")
    
    print("Style: with_context")
    mock_context = "The Rocky Mountains stretch over 3,000 miles from Canada to New Mexico."
    print(create_rag_prompt(sample_q, mock_context, style="with_context"))
    print("\n" + "="*80 + "\n")
    
    print("Style: instruction_following")
    print(create_rag_prompt(sample_q, style="instruction_following"))


if __name__ == "__main__":
    main()
