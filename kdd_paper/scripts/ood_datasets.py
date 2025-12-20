#!/usr/bin/env python3
"""
Out-of-Distribution (OOD) Dataset Loaders for KDD Rebuttal.

These datasets are used to evaluate generalization of LMSYS-trained priors
to held-out domains. Each loader returns prompts from a domain NOT present
in the LMSYS Chatbot Arena training data.

Datasets:
    - GSM8K: Grade school math word problems (maps to math_500 benchmark)
    - HumanEval: Code generation prompts (maps to humaneval_score benchmark)
    - MMLU: Multi-task language understanding (maps to mmlu_pro benchmark)

Usage:
    from kdd_paper.scripts.ood_datasets import load_gsm8k_prompts, load_humaneval_prompts, load_mmlu_prompts
    
    math_prompts = load_gsm8k_prompts(n=500)
    code_prompts = load_humaneval_prompts(n=164)
    knowledge_prompts = load_mmlu_prompts(n=500)

References:
    - GSM8K: https://huggingface.co/datasets/gsm8k
    - HumanEval: https://huggingface.co/datasets/openai/openai_humaneval
    - MMLU: https://huggingface.co/datasets/cais/mmlu
"""

from __future__ import annotations

import random
from typing import List, Optional, Tuple

# Lazy import to avoid dependency issues
def _get_datasets():
    try:
        from datasets import load_dataset
        return load_dataset
    except ImportError:
        raise ImportError(
            "OOD datasets require the 'datasets' library. "
            "Install with: pip install datasets"
        )


def load_gsm8k_prompts(
    n: int = 500,
    seed: int = 42,
    include_answer: bool = False,
) -> List[str]:
    """
    Load math word problems from GSM8K test set.
    
    GSM8K contains grade-school level math word problems that require
    multi-step reasoning. These are structurally different from general
    chat prompts in LMSYS.
    
    Args:
        n: Number of prompts to return (max 1319 in test set)
        seed: Random seed for reproducibility
        include_answer: If True, return (prompt, answer) tuples
        
    Returns:
        List of math problem prompts
        
    Example prompt:
        "Natalia sold clips to 48 of her friends in April, and then 
         she sold half as many clips in May. How many clips did 
         Natalia sell altogether in April and May?"
    """
    load_dataset = _get_datasets()
    
    ds = load_dataset("gsm8k", "main", split="test")
    
    # Shuffle for random sampling
    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    
    prompts = []
    for i in indices[:n]:
        question = ds[i]["question"]
        if include_answer:
            # Extract final answer from "#### <answer>" format
            answer_text = ds[i]["answer"]
            final_answer = answer_text.split("####")[-1].strip() if "####" in answer_text else ""
            prompts.append((question, final_answer))
        else:
            prompts.append(question)
    
    return prompts


def load_humaneval_prompts(
    n: Optional[int] = None,
    seed: int = 42,
) -> List[str]:
    """
    Load code generation prompts from HumanEval.
    
    HumanEval contains 164 hand-written Python programming problems
    with function signatures and docstrings. These are very different
    from general chat prompts.
    
    Args:
        n: Number of prompts (default: all 164)
        seed: Random seed for reproducibility
        
    Returns:
        List of code generation prompts (function signature + docstring)
        
    Example prompt:
        '''from typing import List

        def has_close_elements(numbers: List[float], threshold: float) -> bool:
            \"\"\" Check if in given list of numbers, are any two numbers 
                closer to each other than given threshold.
            >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
            False
            >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
            True
            \"\"\"'''
    """
    load_dataset = _get_datasets()
    
    ds = load_dataset("openai/openai_humaneval", split="test")
    
    # HumanEval has exactly 164 problems
    total = len(ds)
    n = n or total
    n = min(n, total)
    
    # Shuffle for random sampling if n < total
    rng = random.Random(seed)
    indices = list(range(total))
    rng.shuffle(indices)
    
    prompts = []
    for i in indices[:n]:
        # The "prompt" field contains the function signature and docstring
        prompts.append(ds[i]["prompt"])
    
    return prompts


def load_mmlu_prompts(
    n: int = 500,
    seed: int = 42,
    subjects: Optional[List[str]] = None,
    format_style: str = "question_only",
) -> List[str]:
    """
    Load knowledge QA prompts from MMLU test set.
    
    MMLU covers 57 subjects from STEM, humanities, social sciences, etc.
    These structured QA prompts are different from conversational chat.
    
    Args:
        n: Number of prompts to return
        seed: Random seed for reproducibility
        subjects: List of subject names to filter by (None = all subjects)
        format_style: How to format the prompt
            - "question_only": Just the question text
            - "with_choices": Question + A/B/C/D choices
            - "full": Question + choices + "Answer:" prompt
            
    Returns:
        List of knowledge QA prompts
        
    Example prompt (with_choices):
        "Which of the following is the longest river in the world?
        A. Amazon
        B. Nile
        C. Yangtze
        D. Mississippi"
    """
    load_dataset = _get_datasets()
    
    # Load all subjects
    ds = load_dataset("cais/mmlu", "all", split="test")
    
    # Filter by subjects if specified
    if subjects:
        subject_set = set(subjects)
        filtered_indices = [i for i in range(len(ds)) if ds[i]["subject"] in subject_set]
    else:
        filtered_indices = list(range(len(ds)))
    
    # Shuffle for random sampling
    rng = random.Random(seed)
    rng.shuffle(filtered_indices)
    
    prompts = []
    for i in filtered_indices[:n]:
        example = ds[i]
        question = example["question"]
        choices = example["choices"]
        
        if format_style == "question_only":
            prompts.append(question)
        elif format_style == "with_choices":
            choice_text = "\n".join([
                f"{chr(65+j)}. {choice}" 
                for j, choice in enumerate(choices)
            ])
            prompts.append(f"{question}\n{choice_text}")
        elif format_style == "full":
            choice_text = "\n".join([
                f"{chr(65+j)}. {choice}" 
                for j, choice in enumerate(choices)
            ])
            prompts.append(f"{question}\n{choice_text}\n\nAnswer:")
        else:
            raise ValueError(f"Unknown format_style: {format_style}")
    
    return prompts


def load_gpqa_prompts(
    n: int = 200,
    seed: int = 42,
    difficulty: str = "diamond",
) -> List[str]:
    """
    Load graduate-level QA prompts from GPQA.
    
    GPQA contains expert-level questions in biology, physics, and chemistry
    that are designed to be difficult even for domain experts.
    
    Args:
        n: Number of prompts to return
        seed: Random seed for reproducibility
        difficulty: Difficulty level ("diamond", "extended", "main")
        
    Returns:
        List of graduate-level QA prompts
    """
    load_dataset = _get_datasets()
    
    # GPQA diamond is the hardest subset
    ds = load_dataset("Idavidrein/gpqa", difficulty, split="train")
    
    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    
    prompts = []
    for i in indices[:n]:
        # GPQA has a "Question" field
        question = ds[i].get("Question", ds[i].get("question", ""))
        if question:
            prompts.append(question)
    
    return prompts


# ---------------------------------------------------------------------------
# Domain Configuration
# ---------------------------------------------------------------------------

DOMAIN_CONFIG = {
    "math": {
        "loader": load_gsm8k_prompts,
        "benchmark_key": "math_500",
        "description": "GSM8K math word problems → MATH-500 benchmark",
        "default_n": 500,
    },
    "code": {
        "loader": load_humaneval_prompts,
        "benchmark_key": "humaneval_score",
        "benchmark_scale": 100,  # HumanEval is 0-100, not 0-1
        "description": "HumanEval code generation → HumanEval benchmark",
        "default_n": 164,
    },
    "code_live": {
        "loader": load_humaneval_prompts,
        "benchmark_key": "livecodebench",
        "description": "HumanEval prompts → LiveCodeBench benchmark",
        "default_n": 164,
    },
    "knowledge": {
        "loader": load_mmlu_prompts,
        "benchmark_key": "mmlu_pro",
        "description": "MMLU knowledge QA → MMLU-Pro benchmark",
        "default_n": 500,
    },
    "graduate": {
        "loader": load_gpqa_prompts,
        "benchmark_key": "gpqa",
        "description": "GPQA graduate-level QA → GPQA benchmark",
        "default_n": 200,
    },
}


def get_domain_prompts(domain: str, n: Optional[int] = None, seed: int = 42) -> Tuple[List[str], dict]:
    """
    Load prompts for a domain and return domain configuration.
    
    Args:
        domain: One of "math", "code", "code_live", "knowledge", "graduate"
        n: Number of prompts (None = use domain default)
        seed: Random seed
        
    Returns:
        (prompts, config) where config contains benchmark_key, description, etc.
    """
    if domain not in DOMAIN_CONFIG:
        raise ValueError(f"Unknown domain: {domain}. Choose from: {list(DOMAIN_CONFIG.keys())}")
    
    config = DOMAIN_CONFIG[domain]
    loader = config["loader"]
    n = n or config["default_n"]
    
    prompts = loader(n=n, seed=seed)
    
    return prompts, config


# ---------------------------------------------------------------------------
# CLI for testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Load OOD datasets for testing")
    parser.add_argument("--domain", choices=list(DOMAIN_CONFIG.keys()), default="math")
    parser.add_argument("--n", type=int, default=5, help="Number of prompts to display")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    prompts, config = get_domain_prompts(args.domain, n=args.n, seed=args.seed)
    
    print(f"Domain: {args.domain}")
    print(f"Description: {config['description']}")
    print(f"Benchmark key: {config['benchmark_key']}")
    print(f"Loaded {len(prompts)} prompts")
    print("=" * 60)
    
    for i, prompt in enumerate(prompts):
        print(f"\n--- Prompt {i+1} ---")
        print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
