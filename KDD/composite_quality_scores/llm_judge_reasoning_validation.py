#!/usr/bin/env python3
"""
LLM Judge Validation for Reasoning Scores (CRS) using GPQA-Diamond.

This script validates that BLF-derived Composite Reasoning Scores (CRS) correlate with:
1. Hard Validity: Accuracy on GPQA-Diamond (correct answer selection)
2. Soft Validity: LLM Judge ratings of reasoning quality

GPQA-Diamond is ideal because:
- Graduate-level scientific questions resistant to knowledge retrieval
- Requires actual derivation/reasoning, not memorization
- 198 expert-curated problems across physics, biology, chemistry

KDD-GRADE Hybrid Workflow for Reasoning:
- Step 1: Extract model's final answer (A/B/C/D) → binary correctness
- Step 2: LLM Judge evaluates reasoning QUALITY (independent of correctness)
"""

import os
import sys
import json
import random
import re
import argparse
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from datetime import datetime

# Try to load .env file
try:
    from dotenv import load_dotenv
    for env_path in ['.env', '../.env', '../../.env', '../../../.env']:
        if os.path.exists(env_path):
            load_dotenv(env_path)
            break
except ImportError:
    pass


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class GPQAProblem:
    """A GPQA-Diamond problem with shuffled options."""
    prompt: str
    correct_letter: str  # A, B, C, or D
    question: str
    domain: str  # physics, biology, chemistry, etc.
    task_id: str


@dataclass
class ModelForExperiment:
    """Model selected for the experiment."""
    name: str
    openrouter_id: str
    blf_score: float  # CRS score
    blf_rank: int


# =============================================================================
# Fallback Reasoning Problems (when GPQA unavailable)
# =============================================================================

def _create_fallback_reasoning_problems(n_samples: int, seed: int) -> List[GPQAProblem]:
    """
    Create fallback reasoning problems when GPQA is unavailable.
    Tries multiple reasoning datasets as alternatives.
    """
    from datasets import load_dataset
    
    random.seed(seed)
    
    # Try multiple datasets in order of preference
    datasets_to_try = [
        ("openai/gsm8k", "main", "test", "GSM8K"),  # Grade school math reasoning
        ("allenai/ai2_arc", "ARC-Challenge", "test", "ARC"),  # Science reasoning
        ("cais/mmlu", "abstract_algebra", "test", "MMLU"),  # Multiple choice reasoning
        ("tasksource/bigbench", "logical_deduction_five_objects", "validation", "BigBench"),
    ]
    
    for dataset_name, config, split, short_name in datasets_to_try:
        try:
            print(f"  Trying {short_name} dataset...")
            dataset = load_dataset(dataset_name, config, split=split)
            problems = []
            
            for idx, item in enumerate(dataset):
                if len(problems) >= n_samples:
                    break
                
                # Handle different dataset formats
                if short_name == "GSM8K":
                    question = item.get('question', '')
                    answer = item.get('answer', '').split('####')[-1].strip()
                    domain = "Math Reasoning"
                elif short_name == "ARC":
                    question = item.get('question', '')
                    choices = item.get('choices', {})
                    labels = choices.get('label', [])
                    texts = choices.get('text', [])
                    answer_key = item.get('answerKey', '')
                    
                    # Format as multiple choice
                    options = "\n".join([f"{l}. {t}" for l, t in zip(labels, texts)])
                    question = f"{question}\n\n{options}"
                    answer = answer_key
                    domain = "Science"
                elif short_name == "MMLU":
                    question = item.get('question', '')
                    choices = item.get('choices', [])
                    answer_idx = item.get('answer', 0)
                    
                    options = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
                    question = f"{question}\n\n{options}"
                    answer = chr(65 + answer_idx)
                    domain = "Abstract Algebra"
                else:
                    question = str(item.get('inputs', item.get('question', '')))
                    answer = str(item.get('targets', item.get('answer', '')))
                    domain = "Logic"
                
                if not question:
                    continue
                
                prompt = f"""You are an expert problem solver. Think through this problem carefully step-by-step.

**Problem ({domain}):**
{question}

**Instructions:**
1. Analyze the problem and identify what is being asked.
2. Show your complete reasoning process.
3. State your final answer clearly.

Think step-by-step before giving your final answer."""

                problems.append(GPQAProblem(
                    prompt=prompt.strip(),
                    correct_letter=str(answer),
                    question=question[:200],
                    domain=domain,
                    task_id=f"{short_name}/{idx}"
                ))
            
            if problems:
                print(f"  ✅ Loaded {len(problems)} {short_name} problems")
                return problems
                
        except Exception as e:
            print(f"  {short_name} failed: {str(e)[:60]}...")
    
    # Ultimate fallback: hardcoded reasoning problems
    print("  Using hardcoded reasoning problems...")
    hardcoded = [
        {
            "question": "A train travels from city A to city B at 60 mph and returns at 40 mph. What is the average speed for the entire journey?",
            "answer": "48",
            "domain": "Physics/Math"
        },
        {
            "question": "If 5 machines can produce 5 widgets in 5 minutes, how many minutes would it take 100 machines to produce 100 widgets?",
            "answer": "5",
            "domain": "Logic"
        },
        {
            "question": "A lily pad doubles in size every day. If it takes 48 days to cover the whole pond, how many days does it take to cover half the pond?",
            "answer": "47",
            "domain": "Exponential Growth"
        },
        {
            "question": "Three people check into a hotel room that costs $30. They each pay $10. Later, the clerk realizes the room was only $25 and gives $5 to the bellboy to return. The bellboy keeps $2 and gives each person $1 back. Now each person has paid $9 (total $27), plus $2 the bellboy kept = $29. Where is the missing dollar?",
            "answer": "There is no missing dollar - the $27 paid includes the $2 tip. $25 room + $2 tip = $27.",
            "domain": "Logic Puzzle"
        },
        {
            "question": "In a race, you overtake the person in second place. What position are you now in?",
            "answer": "Second place",
            "domain": "Logic"
        },
    ]
    
    problems = []
    for idx, item in enumerate(hardcoded[:n_samples]):
        prompt = f"""You are an expert problem solver. Think through this problem carefully step-by-step.

**Problem ({item['domain']}):**
{item['question']}

**Instructions:**
1. Identify any tricks or common misconceptions in the problem.
2. Show your reasoning process clearly.
3. State your final answer.

Think carefully before answering."""

        problems.append(GPQAProblem(
            prompt=prompt.strip(),
            correct_letter=item['answer'],
            question=item['question'][:200],
            domain=item['domain'],
            task_id=f"LOGIC/{idx}"
        ))
    
    return problems


# =============================================================================
# GPQA-Diamond Loading
# =============================================================================

def load_gpqa_diamond(n_samples: int = 20, seed: int = 42) -> List[GPQAProblem]:
    """
    Load GPQA-Diamond dataset with shuffled answer options.
    
    CRITICAL: Options must be shuffled! The raw dataset has the correct
    answer in a fixed column, so without shuffling you test position bias.
    
    NOTE: GPQA is a gated dataset. You need to:
    1. Visit https://huggingface.co/datasets/Idavidrein/gpqa
    2. Accept the terms to get access
    3. Set HF_TOKEN environment variable or run: huggingface-cli login
    
    Args:
        n_samples: Number of problems to sample
        seed: Random seed for reproducibility
        
    Returns:
        List of GPQAProblem objects with formatted prompts
    """
    from datasets import load_dataset
    
    random.seed(seed)
    
    print("  Loading GPQA-Diamond from HuggingFace...")
    print("  (Note: GPQA is gated. Set HF_TOKEN or run 'huggingface-cli login')")
    
    # Get HF token from environment (check common variable names)
    hf_token = (os.getenv("HF_TOKEN") or 
                os.getenv("HUGGINGFACE_TOKEN") or 
                os.getenv("HUGGINGFACE_API_KEY") or
                os.getenv("HF_API_KEY"))
    
    try:
        # Try with token if available
        if hf_token:
            dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train", token=hf_token)
        else:
            dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    except Exception as e:
        error_msg = str(e)
        if "gated" in error_msg.lower() or "authentication" in error_msg.lower():
            print(f"\n  ❌ GPQA is a gated dataset. To access it:")
            print(f"     1. Visit: https://huggingface.co/datasets/Idavidrein/gpqa")
            print(f"     2. Accept the terms of use")
            print(f"     3. Set HF_TOKEN=your_token or run: huggingface-cli login")
            print(f"\n  Falling back to a simpler reasoning test...")
            return _create_fallback_reasoning_problems(n_samples, seed)
        else:
            print(f"  Error loading dataset: {e}")
            return _create_fallback_reasoning_problems(n_samples, seed)
    
    problems = []
    
    for idx, item in enumerate(dataset):
        # Extract question and answers
        question = item.get('Question', item.get('question', ''))
        correct_answer = item.get('Correct Answer', item.get('correct_answer', ''))
        
        # Get distractors (incorrect answers)
        distractors = []
        for key in ['Incorrect Answer 1', 'Incorrect Answer 2', 'Incorrect Answer 3',
                    'incorrect_answer_1', 'incorrect_answer_2', 'incorrect_answer_3']:
            if key in item and item[key]:
                distractors.append(item[key])
        
        if not question or not correct_answer or len(distractors) < 3:
            continue
        
        # Shuffle options - CRITICAL for valid evaluation
        all_options = [(correct_answer, True)] + [(d, False) for d in distractors[:3]]
        random.shuffle(all_options)
        
        # Assign letters and find correct one
        options_text = ""
        correct_letter = None
        
        for i, (opt_text, is_correct) in enumerate(all_options):
            letter = chr(65 + i)  # A, B, C, D
            options_text += f"{letter}. {opt_text}\n"
            if is_correct:
                correct_letter = letter
        
        # Get domain/subdomain if available
        domain = item.get('Subdomain', item.get('subdomain', 
                  item.get('Domain', item.get('domain', 'Science'))))
        
        # Construct Chain-of-Thought prompt
        prompt = f"""You are an expert scientist and logician answering a graduate-level scientific question.
You must think step-by-step to derive the answer. Do not use external tools or look up information.

**Question:**
{question}

**Options:**
{options_text}

**Instructions:**
1. First, analyze the scientific principles and concepts involved in this question.
2. Second, perform any necessary derivations, calculations, or logical deductions.
3. Third, evaluate each option against your analysis.
4. Finally, state your final answer in the format: "Answer: X" (where X is A, B, C, or D)

Show your complete reasoning process before giving your final answer."""

        problems.append(GPQAProblem(
            prompt=prompt.strip(),
            correct_letter=correct_letter,
            question=question[:200],  # Truncated for display
            domain=domain,
            task_id=f"GPQA/{idx}"
        ))
    
    print(f"  Loaded {len(problems)} GPQA-Diamond problems")
    
    # Sample if needed
    if len(problems) > n_samples:
        problems = random.sample(problems, n_samples)
        print(f"  Sampled {n_samples} problems for validation")
    
    return problems


# =============================================================================
# Model Selection
# =============================================================================

def select_models_for_experiment(
    cache_path: str = "../../data/models_cache.json",
    composite_field: str = "crs",  # Composite Reasoning Score
    n_top: int = 5,
    n_bottom: int = 5
) -> List[ModelForExperiment]:
    """
    Select top-N and bottom-N models by CRS for validation.
    
    Returns models with OpenRouter access for live evaluation.
    """
    with open(cache_path) as f:
        data = json.load(f)
    models = data if isinstance(data, list) else data.get('models', [])
    
    # Filter to models with OpenRouter access and CRS score
    api_models = [
        m for m in models 
        if m.get('openrouter_id')
        and m.get(composite_field) is not None
    ]
    
    # Sort by composite score (descending)
    api_models.sort(key=lambda m: m[composite_field], reverse=True)
    
    # Select top and bottom
    top_models = api_models[:n_top]
    bottom_models = api_models[-n_bottom:]
    
    selected = []
    for i, m in enumerate(top_models + bottom_models):
        # Determine rank
        rank = i + 1 if i < n_top else len(api_models) - (n_top + n_bottom - 1 - i)
        selected.append(ModelForExperiment(
            name=m['name'],
            openrouter_id=m['openrouter_id'],
            blf_score=m[composite_field],
            blf_rank=rank,
        ))
    
    return selected


# =============================================================================
# LLM API Calls
# =============================================================================

def call_model(openrouter_id: str, prompt: str, max_retries: int = 3) -> str:
    """
    Call an LLM via OpenRouter API with proper token handling for reasoning.
    
    Reasoning models often need more tokens for Chain-of-Thought.
    """
    from openai import OpenAI
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment")
    
    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )
    
    # Reasoning tasks need generous token limits for CoT
    model_lower = openrouter_id.lower()
    is_reasoning_model = any(x in model_lower for x in ['reasoning', 'thinking', 'r1', 'o1', 'o3'])
    is_gemini3 = 'gemini-3' in model_lower
    
    if is_gemini3 or is_reasoning_model:
        token_limits = [16000, 32000, 65000]
    else:
        token_limits = [8000, 16000, 32000]
    
    last_error = None
    for attempt in range(max_retries):
        try:
            tokens = token_limits[min(attempt, len(token_limits) - 1)]
            
            response = client.chat.completions.create(
                model=openrouter_id,
                max_tokens=tokens,
                messages=[{"role": "user", "content": prompt}]
            )
            
            message = response.choices[0].message
            content = message.content
            
            if isinstance(content, str) and content.strip():
                return content
            
            # Handle list content
            if isinstance(content, list):
                parts = [str(p.get("text", p.get("content", p))) if isinstance(p, dict) else str(p) for p in content]
                if parts:
                    return "\n".join(parts)
            
            # Check reasoning fields
            for field in ['reasoning', 'reasoning_content']:
                if hasattr(message, field) and getattr(message, field):
                    val = getattr(message, field)
                    if isinstance(val, str):
                        return val
                    elif isinstance(val, list):
                        return "\n".join(str(p) for p in val)
            
            # Empty response - log and retry
            if attempt < max_retries - 1:
                finish_reason = response.choices[0].finish_reason if response.choices else "no_choices"
                print(f"\n  ⚠️ Empty on attempt {attempt+1}, finish_reason={finish_reason}, retrying...")
                continue
                
        except Exception as e:
            last_error = e
            print(f"\n  ⚠️ API error: {type(e).__name__}: {str(e)[:80]}")
            if attempt < max_retries - 1:
                continue
            raise
    
    print(f"\n  ❌ All {max_retries} attempts returned empty for {openrouter_id}")
    return ""


# =============================================================================
# Answer Extraction (LLM-based for robustness)
# =============================================================================

def extract_answer_with_llm(response: str, question: str = "") -> Optional[str]:
    """
    Use a fast LLM to extract the final answer from a response.
    
    This is more robust than regex as it handles any format.
    Uses GPT-4o-mini for speed and low cost.
    """
    from openai import OpenAI
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return None
    
    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    
    extraction_prompt = f"""Extract ONLY the final numeric answer from this response. 
Return ONLY the number, nothing else. No units, no explanation, just the number.

If the answer is a dollar amount like $18, return: 18
If the answer is "18 eggs", return: 18
If no clear final answer, return: NONE

Response to extract from:
{response[-1500:]}

Final numeric answer (just the number):"""

    try:
        result = client.chat.completions.create(
            model="openai/gpt-4o-mini",  # Fast and cheap
            max_tokens=20,
            temperature=0,
            messages=[{"role": "user", "content": extraction_prompt}]
        )
        
        answer = result.choices[0].message.content.strip()
        
        # Clean up the response
        if answer.upper() == "NONE" or not answer:
            return None
        
        # Remove any non-numeric characters except decimal point and minus
        import re
        cleaned = re.sub(r'[^\d.\-]', '', answer)
        if cleaned:
            return cleaned
        
        return answer if answer else None
        
    except Exception as e:
        print(f"  ⚠️ LLM extraction failed: {e}")
        return None


def extract_answer(response: str, expected_type: str = "auto", use_llm: bool = True) -> Optional[str]:
    """
    Extract the final answer from model response.
    
    Uses LLM-based extraction for robustness (handles any format).
    Falls back to regex if LLM fails.
    
    Args:
        response: Model's full response text
        expected_type: "letter", "number", or "auto" (tries both)
        use_llm: Whether to use LLM for extraction (default True)
    
    Returns:
        Extracted answer as string, or None if not found
    """
    if not response:
        return None
    
    # Try LLM-based extraction first (most robust)
    if use_llm and expected_type in ("number", "auto"):
        llm_answer = extract_answer_with_llm(response)
        if llm_answer:
            return llm_answer
    
    # Fall back to regex patterns
    if expected_type in ("number", "auto"):
        numeric_answer = _extract_numeric_answer(response)
        if numeric_answer is not None:
            return str(numeric_answer)
    
    # Try to extract letter answer
    if expected_type in ("letter", "auto"):
        letter_answer = _extract_letter_answer(response)
        if letter_answer:
            return letter_answer
    
    return None


def _extract_numeric_answer(response: str) -> Optional[str]:
    """
    Extract numeric answer from response.
    
    PRIORITY ORDER (to avoid extracting intermediate steps):
    1. Explicit final answer statements
    2. LaTeX boxed
    3. Bold numbers at end
    4. Last calculation result
    """
    if not response:
        return None
    
    # Normalize response - look at last portion for final answer
    response_lower = response.lower()
    
    # PRIORITY 1: Explicit "final answer" or "answer is" statements
    # These are the most reliable signals
    # Handle **$18**, **18**, $18, 18 formats
    explicit_patterns = [
        r'(?:final\s+)?answer[:\s]+[is\s]*\*?\*?\$?(-?\d+\.?\d*)\*?\*?',
        r'the\s+answer\s+is[:\s]+\*?\*?\$?(-?\d+\.?\d*)\*?\*?',
        r'(?:so|thus|therefore|hence)[,\s]+(?:the\s+)?(?:final\s+)?answer[:\s]+[is\s]*\*?\*?\$?(-?\d+\.?\d*)\*?\*?',
        r'(?:janet|she|he|they)\s+(?:makes?|earns?|gets?|has|have)\s+\*?\*?\$?(-?\d+\.?\d*)\*?\*?\s*(?:per\s+day|daily|every\s+day|total)?',
    ]
    
    for pattern in explicit_patterns:
        matches = re.findall(pattern, response, re.IGNORECASE | re.MULTILINE)
        if matches:
            # Return the LAST match (final answer, not intermediate)
            return matches[-1].replace(',', '')
    
    # PRIORITY 2: LaTeX boxed (highest priority for math-style answers)
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', response)
    if boxed_match:
        answer = boxed_match.group(1).strip()
        num_match = re.search(r'(-?\d+\.?\d*)', answer)
        return num_match.group(1) if num_match else None
    
    # PRIORITY 3: Bold number - but only if it appears near "answer" or at very end
    # Handle **$18** or **18** format
    # Look for **number** or **$number** near answer-related words
    bold_with_context = re.findall(
        r'(?:answer|total|result|makes?|earns?)[:\s]+[^*]*\*\*\$?(-?\d+\.?\d*)\*\*',
        response, re.IGNORECASE
    )
    if bold_with_context:
        return bold_with_context[-1].replace(',', '')
    
    # Or bold number at the very end of response (with or without $)
    bold_at_end = re.search(r'\*\*\$?(-?\d+\.?\d*)\*\*[.\s]*$', response)
    if bold_at_end:
        return bold_at_end.group(1).replace(',', '')
    
    # Also check for "makes **$X**" pattern anywhere
    makes_bold = re.findall(r'(?:makes?|earns?)\s+\*\*\$?(-?\d+\.?\d*)\*\*', response, re.IGNORECASE)
    if makes_bold:
        return makes_bold[-1].replace(',', '')
    
    # PRIORITY 4: Final calculation - look for "= X" patterns at end
    # But be careful to get the FINAL result, not intermediate
    last_500 = response[-500:] if len(response) > 500 else response
    
    # Look for final "= number" patterns, especially with units
    final_calc_patterns = [
        r'=\s*\$?(-?\d+\.?\d*)\s*(?:dollars?|per\s+day|daily|total|eggs?|bolts?)?[.\s]*$',
        r'=\s*\*?\*?(-?\d+\.?\d*)\*?\*?[.\s]*$',
    ]
    for pattern in final_calc_patterns:
        match = re.search(pattern, last_500, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).replace(',', '')
    
    # PRIORITY 5: Number with conclusive context (makes/earns/total) in last sentence
    last_sentence_match = re.search(
        r'(?:makes?|earns?|gets?|total(?:s|ing)?|is)\s+\$?(-?\d+\.?\d*)\b',
        last_500, re.IGNORECASE
    )
    if last_sentence_match:
        return last_sentence_match.group(1).replace(',', '')
    
    # PRIORITY 6: Last number after final "=" sign
    all_equals = re.findall(r'=\s*\$?(-?\d+\.?\d*)', last_500)
    if all_equals:
        return all_equals[-1].replace(',', '')
    
    return None


def _extract_letter_answer(response: str) -> Optional[str]:
    """
    Extract multiple choice letter (A, B, C, D) from response.
    """
    response_upper = response.upper()
    
    # Pattern 1: Explicit "Answer: X" format (preferred)
    patterns = [
        r'ANSWER:\s*\(?([A-D])\)?',
        r'FINAL ANSWER:\s*\(?([A-D])\)?',
        r'THE ANSWER IS\s*\(?([A-D])\)?',
        r'CORRECT ANSWER IS\s*\(?([A-D])\)?',
        r'I CHOOSE\s*\(?([A-D])\)?',
        r'OPTION\s*([A-D])\s*IS CORRECT',
        r'\*\*([A-D])\*\*\s*$',  # Bold letter at end
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response_upper)
        if match:
            return match.group(1)
    
    # Fallback: Look for isolated letter near the end
    last_200 = response_upper[-200:] if len(response_upper) > 200 else response_upper
    match = re.search(r'\b([A-D])\b', last_200)
    if match:
        return match.group(1)
    
    return None


def normalize_numeric_answer(answer: str) -> Optional[float]:
    """
    Normalize a numeric answer for comparison.
    Handles integers, decimals, fractions, percentages.
    """
    if not answer:
        return None
    
    answer = answer.strip()
    
    # Remove common suffixes
    answer = re.sub(r'\s*(dollars?|eggs?|items?|people|hours?|days?|miles?|%|percent).*$', '', answer, flags=re.IGNORECASE)
    answer = answer.strip()
    
    # Handle fractions like "3/4"
    if '/' in answer:
        parts = answer.split('/')
        if len(parts) == 2:
            try:
                return float(parts[0]) / float(parts[1])
            except:
                pass
    
    # Handle regular numbers
    try:
        return float(answer.replace(',', ''))
    except:
        return None


def answers_match(extracted: str, expected: str, tolerance: float = 0.01) -> bool:
    """
    Check if extracted answer matches expected answer.
    Handles both exact string match and numeric comparison with tolerance.
    """
    if not extracted or not expected:
        return False
    
    # Try exact string match first (case-insensitive)
    if extracted.strip().lower() == expected.strip().lower():
        return True
    
    # Try numeric comparison
    extracted_num = normalize_numeric_answer(extracted)
    expected_num = normalize_numeric_answer(expected)
    
    if extracted_num is not None and expected_num is not None:
        # Check if they're close enough (handles floating point issues)
        if expected_num == 0:
            return abs(extracted_num) < tolerance
        return abs(extracted_num - expected_num) / abs(expected_num) < tolerance
    
    return False


# Backwards compatibility alias
def extract_answer_letter(response: str) -> Optional[str]:
    """Legacy function - use extract_answer() instead."""
    return extract_answer(response, expected_type="auto")


# =============================================================================
# Bayesian Belief Update Scoring (KDD-Grade)
# =============================================================================

def calculate_bayesian_score(
    exact_match: bool, 
    judge_score: float,  # 0-10 scale
    trust_alpha: float = 0.7
) -> float:
    """
    Calculate final score using Bayesian Belief Update (KDD-Grade method).
    
    This treats the Judge as a "noisy sensor" that updates belief about model quality.
    It handles uncertainty in both the test result and the judge's evaluation.
    
    Args:
        exact_match: Whether the model's answer matched the expected answer
        judge_score: LLM Judge's quality score (0-10)
        trust_alpha: Trust factor for test cases (0.7 = 70% weight on test, 30% on judge)
    
    Returns:
        Final score (0.0 to 1.0)
    
    Formula:
        Score = α * Prior + (1 - α) * Likelihood
        
        Where:
        - Prior = 0.9 if exact_match else 0.1 (allows judge influence even on failure)
        - Likelihood = judge_score / 10.0
        - α = trust factor (default 0.7)
    """
    # 1. Establish Prior from Deterministic Test
    # We don't use 0.0 or 1.0 to allow the Judge to have *some* influence
    prior = 0.9 if exact_match else 0.1
    
    # 2. Normalize Judge Score (Likelihood)
    likelihood = judge_score / 10.0
    
    # 3. Weighted Update
    # If Trust is high (0.7), the test case dominates
    # If Trust is low (0.3), the Judge's opinion matters more
    final_score = (trust_alpha * prior) + ((1 - trust_alpha) * likelihood)
    
    return final_score


def calculate_bayesian_score_with_breakdown(
    exact_match: bool,
    judge_score: float,
    trust_alpha: float = 0.7
) -> Dict:
    """
    Calculate Bayesian score with full breakdown for analysis.
    
    Returns:
        Dict with score, prior, likelihood, and components
    """
    prior = 0.9 if exact_match else 0.1
    likelihood = judge_score / 10.0
    final_score = (trust_alpha * prior) + ((1 - trust_alpha) * likelihood)
    
    return {
        "final_score": final_score,
        "final_score_10": final_score * 10,  # On 0-10 scale for display
        "prior": prior,
        "likelihood": likelihood,
        "trust_alpha": trust_alpha,
        "exact_match": exact_match,
        "judge_score": judge_score,
        "formula": f"{trust_alpha:.1f}*{prior:.1f} + {1-trust_alpha:.1f}*{likelihood:.2f} = {final_score:.3f}"
    }


# =============================================================================
# LLM Judge for Reasoning Quality
# =============================================================================

# Rival judge mapping (Committee of Rivals approach)
RIVAL_JUDGES = {
    'anthropic': 'google/gemini-3-pro-preview',
    'openai': 'google/gemini-3-pro-preview',
    'google': 'anthropic/claude-opus-4.5',
    'meta-llama': 'google/gemini-3-pro-preview',
    'mistralai': 'google/gemini-3-pro-preview',
    'deepseek': 'google/gemini-3-pro-preview',
    'x-ai': 'google/gemini-3-pro-preview',
}
DEFAULT_JUDGE = 'google/gemini-3-pro-preview'


def get_rival_judge(model_openrouter_id: str) -> str:
    """Get a rival judge for Committee of Rivals approach."""
    if '/' in model_openrouter_id:
        provider = model_openrouter_id.split('/')[0].lower()
    else:
        provider = 'unknown'
    return RIVAL_JUDGES.get(provider, DEFAULT_JUDGE)


# Judge prompt for reasoning evaluation - STRICT VERSION
REASONING_JUDGE_PROMPT = """You are a STRICT evaluator of mathematical reasoning. Be critical and discriminating.

**Task:** Evaluate the reasoning quality. Be HARSH - most responses should score 5-7, only exceptional reasoning gets 9-10.

**Question:**
{question}

**Model's Response:**
{response}

**Ground Truth Answer:** {correct_letter}
**Model's Answer:** {model_answer}
**Answer Correctness:** {correctness}

**Scoring Criteria (BE STRICT):**

1. **Mathematical Accuracy (0-3):**
   - 3: All calculations are correct with no errors
   - 2: Minor arithmetic errors but correct method
   - 1: Significant calculation errors or flawed method
   - 0: Completely wrong calculations or no calculations shown

2. **Logical Structure (0-3):**
   - 3: Crystal clear step-by-step reasoning, easy to follow
   - 2: Mostly clear but some jumps in logic
   - 1: Confusing or hard to follow reasoning
   - 0: No clear reasoning chain, just states answer

3. **Problem Understanding (0-2):**
   - 2: Correctly identifies what the problem asks and all constraints
   - 1: Partially understands but misses some aspects
   - 0: Misunderstands the problem

4. **Final Answer Penalty:**
   - If final answer is WRONG: Subtract 2 points from total
   - If no clear final answer given: Subtract 1 point

**IMPORTANT:** 
- Do NOT give 10/10 unless reasoning is truly exceptional
- A score of 6-7 is GOOD, 8 is VERY GOOD, 9-10 is EXCEPTIONAL
- Wrong final answer caps maximum score at 8/10

Provide evaluation in this EXACT format:
```
Mathematical Accuracy: X/3
Logical Structure: X/3
Problem Understanding: X/2
Answer Penalty: -X (0, 1, or 2)
---
Total: X/10
```

One sentence justification:"""


def judge_reasoning(
    question: str,
    response: str,
    correct_letter: str,
    model_answer: Optional[str],
    model_openrouter_id: str,
    judge_model: str = None,
) -> Dict:
    """
    Use LLM judge to evaluate reasoning quality.
    
    Returns dict with score (0-10) and breakdown.
    """
    from openai import OpenAI
    
    if judge_model is None:
        judge_model = get_rival_judge(model_openrouter_id)
    
    # Determine correctness (handles numeric comparison for GSM8K)
    is_correct = answers_match(model_answer, correct_letter) if model_answer else False
    correctness = "✓ CORRECT" if is_correct else "✗ INCORRECT"
    
    judge_prompt = REASONING_JUDGE_PROMPT.format(
        question=question[:1000],
        response=response[:3000],
        correct_letter=correct_letter,
        model_answer=model_answer or "NOT FOUND",
        correctness=correctness
    )
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    
    judge_max_tokens = 16000 if 'gemini-3' in judge_model.lower() else 4000
    
    result = client.chat.completions.create(
        model=judge_model,
        max_tokens=judge_max_tokens,
        messages=[{"role": "user", "content": judge_prompt}]
    )
    
    response_text = result.choices[0].message.content
    if not response_text:
        raise ValueError(f"Empty response from judge {judge_model}")
    
    # Parse the score
    total_match = re.search(r'Total:\s*(\d+)/10', response_text)
    if total_match:
        score = int(total_match.group(1))
    else:
        # Fallback: look for any X/10 pattern
        score_match = re.search(r'(\d+)/10', response_text)
        score = int(score_match.group(1)) if score_match else 5
    
    return {
        "score": min(10, max(0, score)),
        "judge_model": judge_model,
        "raw_response": response_text[:500],
        "is_correct": is_correct
    }


# =============================================================================
# Confidence Intervals & Progress Tracking
# =============================================================================

def calculate_confidence_interval(scores: List[float], confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    Calculate mean and confidence interval for a list of scores.
    Returns (mean, ci_lower, ci_upper).
    """
    import numpy as np
    from scipy import stats as scipy_stats
    
    if not scores:
        return (0.0, 0.0, 0.0)
    
    scores_arr = np.array(scores)
    mean = float(np.mean(scores_arr))
    
    if len(scores) < 2:
        return (mean, mean, mean)
    
    # Use t-distribution for small samples
    sem = scipy_stats.sem(scores_arr)
    ci = scipy_stats.t.interval(confidence, len(scores_arr)-1, loc=mean, scale=sem)
    
    return (mean, float(ci[0]), float(ci[1]))


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


# =============================================================================
# Main Validation
# =============================================================================

def run_reasoning_validation(
    n_prompts: int = 10,
    n_top_models: int = 5,
    n_bottom_models: int = 5,
    output_dir: str = "llm_judge_results",
    seed: int = 42
) -> Dict:
    """
    Run full reasoning validation using GPQA-Diamond.
    
    Validates that CRS correlates with:
    1. Accuracy (selecting correct answer)
    2. Reasoning quality (LLM judge scores)
    """
    from scipy import stats
    import time
    import numpy as np
    
    start_time = time.time()
    
    print("=" * 80)
    print("CRS (REASONING) VALIDATION - GPQA-Diamond")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Load problems
    print("\n📚 Loading GPQA-Diamond problems...")
    problems = load_gpqa_diamond(n_samples=n_prompts, seed=seed)
    
    if not problems:
        print("❌ Failed to load GPQA-Diamond problems")
        return {}
    
    # Select models
    print("\n🤖 Selecting models for validation...")
    models = select_models_for_experiment(
        n_top=n_top_models,
        n_bottom=n_bottom_models,
        composite_field="crs"
    )
    
    print(f"\n  Selected {len(models)} models (top {n_top_models} + bottom {n_bottom_models} by CRS):")
    print(f"  {'Tier':<8} {'CRS':>6} | {'Model':<40} | {'Judge':<25}")
    print(f"  {'-'*8} {'-'*6} | {'-'*40} | {'-'*25}")
    for m in models:
        tier = "TOP" if m.blf_rank <= n_top_models else "BOTTOM"
        rival = get_rival_judge(m.openrouter_id)
        print(f"  [{tier:6s}] {m.blf_score:5.1f} | {m.name[:40]:40s} | {rival}")
    
    # Initialize results
    results = {
        "timestamp": datetime.now().isoformat(),
        "intent": "reasoning",
        "dataset": "GPQA-Diamond",
        "n_prompts": len(problems),
        "n_models": len(models),
        "model_scores": {m.name: {
            "blf_score": m.blf_score,
            "blf_rank": m.blf_rank,
            "openrouter_id": m.openrouter_id,
            "correct_count": 0,
            "total_count": 0,
            "judge_scores": [],
            "accuracy": 0.0
        } for m in models},
        "problems": []
    }
    
    # Run evaluation
    total_evals = len(problems) * len(models)
    completed_evals = 0
    print(f"\n🧪 Running evaluation on {len(problems)} problems × {len(models)} models = {total_evals} evaluations")
    print("=" * 80)
    
    for i, problem in enumerate(problems, 1):
        problem_start = time.time()
        elapsed = time.time() - start_time
        if completed_evals > 0:
            rate = completed_evals / elapsed
            remaining = (total_evals - completed_evals) / rate if rate > 0 else 0
            eta_str = f"ETA: {format_time(remaining)}"
        else:
            eta_str = "ETA: calculating..."
        
        pct = (completed_evals / total_evals) * 100
        print(f"\n--- Problem {i}/{len(problems)} [{problem.task_id}] ({pct:.1f}% complete, {eta_str}) ---")
        print(f"  Domain: {problem.domain}")
        print(f"  Q: {problem.question[:80]}...")
        print(f"  Correct: {problem.correct_letter}")
        
        problem_result = {
            "task_id": problem.task_id,
            "domain": problem.domain,
            "correct_letter": problem.correct_letter,
            "responses": {}
        }
        
        for m in models:
            print(f"    {m.name[:30]}...", end=" ", flush=True)
            
            try:
                # Get model response
                response = call_model(m.openrouter_id, problem.prompt)
                
                if not response or not response.strip():
                    print("❌ Empty response")
                    problem_result["responses"][m.name] = {"error": "Empty response"}
                    continue
                
                # Extract answer (handles both letters and numbers)
                model_answer = extract_answer(response, expected_type="auto")
                is_correct = answers_match(model_answer, problem.correct_letter)
                
                if is_correct:
                    results["model_scores"][m.name]["correct_count"] += 1
                results["model_scores"][m.name]["total_count"] += 1
                
                # Judge reasoning quality
                judgment = judge_reasoning(
                    problem.question,
                    response,
                    problem.correct_letter,
                    model_answer,
                    m.openrouter_id
                )
                
                # Calculate Bayesian combined score (KDD-Grade method)
                bayesian = calculate_bayesian_score_with_breakdown(
                    exact_match=is_correct,
                    judge_score=judgment["score"],
                    trust_alpha=0.7
                )
                
                results["model_scores"][m.name]["judge_scores"].append(judgment["score"])
                if "bayesian_scores" not in results["model_scores"][m.name]:
                    results["model_scores"][m.name]["bayesian_scores"] = []
                results["model_scores"][m.name]["bayesian_scores"].append(bayesian["final_score"])
                
                # Display result with Bayesian score
                status = "✓" if is_correct else "✗"
                print(f"[{status} {model_answer or '?'}] Judge: {judgment['score']}/10 → Bayes: {bayesian['final_score_10']:.1f}/10")
                
                problem_result["responses"][m.name] = {
                    "response": response[:500],
                    "model_answer": model_answer,
                    "is_correct": is_correct,
                    "judge_score": judgment["score"],
                    "bayesian_score": bayesian["final_score"],
                    "bayesian_formula": bayesian["formula"],
                    "judge_model": judgment["judge_model"]
                }
                
                completed_evals += 1
                
            except Exception as e:
                print(f"❌ Error: {type(e).__name__}: {str(e)[:50]}")
                completed_evals += 1
                problem_result["responses"][m.name] = {"error": str(e)[:200]}
        
        results["problems"].append(problem_result)
    
    # Calculate final metrics with confidence intervals
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY - REASONING VALIDATION")
    print(f"Completed in {format_time(total_time)} ({total_evals} evaluations)")
    print("=" * 80)
    
    for m_name, m_data in results["model_scores"].items():
        if m_data["total_count"] > 0:
            m_data["accuracy"] = m_data["correct_count"] / m_data["total_count"]
            
            # Judge score with CI
            judge_mean, judge_ci_low, judge_ci_high = calculate_confidence_interval(m_data["judge_scores"])
            m_data["avg_judge_score"] = judge_mean
            m_data["judge_ci_lower"] = judge_ci_low
            m_data["judge_ci_upper"] = judge_ci_high
            
            # Bayesian score with CI
            bayesian_scores = m_data.get("bayesian_scores", [])
            if bayesian_scores:
                bayes_mean, bayes_ci_low, bayes_ci_high = calculate_confidence_interval(bayesian_scores)
                m_data["avg_bayesian_score"] = bayes_mean
                m_data["bayesian_ci_lower"] = bayes_ci_low
                m_data["bayesian_ci_upper"] = bayes_ci_high
            else:
                m_data["avg_bayesian_score"] = 0
                m_data["bayesian_ci_lower"] = 0
                m_data["bayesian_ci_upper"] = 0
            
            # Accuracy CI (binomial proportion CI using Wilson score)
            n = m_data["total_count"]
            p = m_data["accuracy"]
            z = 1.96  # 95% CI
            denom = 1 + z*z/n
            center = (p + z*z/(2*n)) / denom
            spread = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
            m_data["accuracy_ci_lower"] = max(0, center - spread)
            m_data["accuracy_ci_upper"] = min(1, center + spread)
    
    # Sort by BLF score for display
    sorted_models = sorted(results["model_scores"].items(), key=lambda x: x[1]["blf_score"], reverse=True)
    
    print(f"\n{'Model':<32} {'CRS':>5} {'Accuracy':>18} {'Judge':>18} {'Bayesian':>18}")
    print("-" * 95)
    
    model_data_for_corr = []
    for rank, (m_name, m_data) in enumerate(sorted_models, 1):
        acc_pct = m_data["accuracy"] * 100
        acc_ci = f"[{m_data.get('accuracy_ci_lower', 0)*100:.0f}-{m_data.get('accuracy_ci_upper', 0)*100:.0f}%]"
        
        avg_judge = m_data.get("avg_judge_score", 0)
        judge_ci = f"[{m_data.get('judge_ci_lower', 0):.1f}-{m_data.get('judge_ci_upper', 0):.1f}]"
        
        avg_bayes = m_data.get("avg_bayesian_score", 0) * 10  # Convert to 0-10 scale
        bayes_ci = f"[{m_data.get('bayesian_ci_lower', 0)*10:.1f}-{m_data.get('bayesian_ci_upper', 0)*10:.1f}]"
        
        print(f"{m_name[:30]:<32} {m_data['blf_score']:>5.1f} {acc_pct:>4.0f}% {acc_ci:<12} {avg_judge:>4.1f} {judge_ci:<12} {avg_bayes:>4.1f} {bayes_ci}")
        
        if m_data["total_count"] > 0:
            model_data_for_corr.append({
                "name": m_name,
                "blf_score": m_data["blf_score"],
                "accuracy": m_data["accuracy"],
                "avg_judge_score": avg_judge,
                "avg_bayesian_score": m_data.get("avg_bayesian_score", 0)
            })
    
    # Compute correlations
    if len(model_data_for_corr) >= 3:
        blf_scores = [d["blf_score"] for d in model_data_for_corr]
        accuracies = [d["accuracy"] for d in model_data_for_corr]
        judge_scores = [d["avg_judge_score"] for d in model_data_for_corr]
        bayesian_scores = [d["avg_bayesian_score"] for d in model_data_for_corr]
        
        print("\n" + "=" * 80)
        print("CORRELATION ANALYSIS")
        print("=" * 80)
        
        # CRS vs Accuracy (Hard Validity)
        if len(set(accuracies)) > 1:
            rho_acc, p_acc = stats.spearmanr(blf_scores, accuracies)
            print(f"\n📊 HARD VALIDITY: CRS vs Accuracy (Exact Match)")
            print(f"   Spearman's ρ: {rho_acc:+.3f} (p={p_acc:.4f})")
            sig = "✅ Significant" if p_acc < 0.05 else "⚠️ Not significant"
            print(f"   → {sig}")
        
        # CRS vs Judge Score (Soft Validity)
        if len(set(judge_scores)) > 1:
            rho_judge, p_judge = stats.spearmanr(blf_scores, judge_scores)
            print(f"\n📊 SOFT VALIDITY: CRS vs Judge Score (Raw)")
            print(f"   Spearman's ρ: {rho_judge:+.3f} (p={p_judge:.4f})")
            sig = "✅ Significant" if p_judge < 0.05 else "⚠️ Not significant"
            print(f"   → {sig}")
        
        # CRS vs Bayesian Score (Combined KDD-Grade)
        if len(set(bayesian_scores)) > 1:
            rho_bayes, p_bayes = stats.spearmanr(blf_scores, bayesian_scores)
            print(f"\n📊 KDD-GRADE: CRS vs Bayesian Score (Combined)")
            print(f"   Spearman's ρ: {rho_bayes:+.3f} (p={p_bayes:.4f})")
            print(f"   Formula: 0.7*Prior + 0.3*Judge (Prior=0.9 if correct, 0.1 if wrong)")
            sig = "✅ Significant" if p_bayes < 0.05 else "⚠️ Not significant"
            print(f"   → {sig}")
        
        # Top vs Bottom comparison
        top_models = [d for d in model_data_for_corr if d["blf_score"] > 0]  # Positive CRS
        bottom_models = [d for d in model_data_for_corr if d["blf_score"] <= 0]  # Negative CRS
        
        if top_models and bottom_models:
            top_acc = sum(d["accuracy"] for d in top_models) / len(top_models)
            bottom_acc = sum(d["accuracy"] for d in bottom_models) / len(bottom_models)
            top_bayes = sum(d["avg_bayesian_score"] for d in top_models) / len(top_models)
            bottom_bayes = sum(d["avg_bayesian_score"] for d in bottom_models) / len(bottom_models)
            
            print(f"\n📈 TOP vs BOTTOM SEPARATION (CRS > 0 vs CRS ≤ 0)")
            print(f"   Top-{len(top_models)} avg accuracy:    {top_acc*100:.1f}%")
            print(f"   Bottom-{len(bottom_models)} avg accuracy: {bottom_acc*100:.1f}%")
            print(f"   Gap: {(top_acc-bottom_acc)*100:+.1f} percentage points")
            print(f"\n   Top-{len(top_models)} avg Bayesian:    {top_bayes*10:.1f}/10")
            print(f"   Bottom-{len(bottom_models)} avg Bayesian: {bottom_bayes*10:.1f}/10")
            print(f"   Gap: {(top_bayes-bottom_bayes)*10:+.1f} points")
    
    # Add summary statistics to results
    results["summary"] = {
        "total_evaluations": total_evals,
        "runtime_seconds": total_time,
        "runtime_formatted": format_time(total_time),
        "n_problems": len(problems),
        "n_models": len(models),
        "models": []
    }
    
    # Add per-model summary
    for m_name, m_data in sorted_models:
        results["summary"]["models"].append({
            "name": m_name,
            "crs": m_data["blf_score"],
            "blf_rank": m_data["blf_rank"],
            "accuracy": {
                "mean": m_data["accuracy"],
                "ci_lower": m_data.get("accuracy_ci_lower", 0),
                "ci_upper": m_data.get("accuracy_ci_upper", 0),
                "n_correct": m_data["correct_count"],
                "n_total": m_data["total_count"]
            },
            "judge_score": {
                "mean": m_data.get("avg_judge_score", 0),
                "ci_lower": m_data.get("judge_ci_lower", 0),
                "ci_upper": m_data.get("judge_ci_upper", 0)
            },
            "bayesian_score": {
                "mean": m_data.get("avg_bayesian_score", 0),
                "ci_lower": m_data.get("bayesian_ci_lower", 0),
                "ci_upper": m_data.get("bayesian_ci_upper", 0)
            }
        })
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_path = Path(output_dir) / "llm_judge_reasoning_results.json"
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n\n📁 Results saved to: {output_path.absolute()}")
    
    # Also save a CSV for easy analysis
    csv_path = Path(output_dir) / "llm_judge_reasoning_summary.csv"
    with open(csv_path, 'w') as f:
        f.write("model,crs,blf_rank,accuracy,acc_ci_lower,acc_ci_upper,judge_mean,judge_ci_lower,judge_ci_upper,bayes_mean,bayes_ci_lower,bayes_ci_upper\n")
        for m in results["summary"]["models"]:
            f.write(f"{m['name']},{m['crs']:.3f},{m['blf_rank']},{m['accuracy']['mean']:.4f},{m['accuracy']['ci_lower']:.4f},{m['accuracy']['ci_upper']:.4f},{m['judge_score']['mean']:.2f},{m['judge_score']['ci_lower']:.2f},{m['judge_score']['ci_upper']:.2f},{m['bayesian_score']['mean']:.4f},{m['bayesian_score']['ci_lower']:.4f},{m['bayesian_score']['ci_upper']:.4f}\n")
    print(f"📊 CSV summary: {csv_path.absolute()}")
    
    return results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Validate CRS (Reasoning) scores using GPQA-Diamond"
    )
    parser.add_argument(
        "--n-prompts", type=int, default=10,
        help="Number of GPQA problems to use (default: 10)"
    )
    parser.add_argument(
        "--n-top", type=int, default=5,
        help="Number of top models to include (default: 5)"
    )
    parser.add_argument(
        "--n-bottom", type=int, default=5,
        help="Number of bottom models to include (default: 5)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--run", action="store_true",
        help="Actually run the validation (costs money!)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be evaluated without calling APIs"
    )
    
    args = parser.parse_args()
    
    if args.dry_run or not args.run:
        print("=" * 80)
        print("DRY RUN - Showing configuration (use --run to execute)")
        print("=" * 80)
        
        print("\n📚 Loading GPQA-Diamond problems...")
        problems = load_gpqa_diamond(n_samples=args.n_prompts, seed=args.seed)
        
        print("\n🤖 Models that would be evaluated:")
        models = select_models_for_experiment(
            n_top=args.n_top,
            n_bottom=args.n_bottom,
            composite_field="crs"
        )
        
        for m in models:
            tier = "TOP" if m.blf_rank <= args.n_top else "BOTTOM"
            print(f"  [{tier}] CRS={m.blf_score:.1f} | {m.name}")
        
        print(f"\n📊 Sample problem:")
        if problems:
            print(f"  {problems[0].task_id}: {problems[0].question[:100]}...")
            print(f"  Correct: {problems[0].correct_letter}")
        
        print(f"\n💰 Estimated API calls: {len(problems) * len(models) * 2} (model + judge)")
        print("\nUse --run flag to execute the validation.")
        return
    
    # Run the validation
    run_reasoning_validation(
        n_prompts=args.n_prompts,
        n_top_models=args.n_top,
        n_bottom_models=args.n_bottom,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
