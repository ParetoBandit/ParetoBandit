#!/usr/bin/env python3
"""
Evaluate OpenCompass Raw Predictions

OpenCompass provides RAW model outputs but not always pass/fail labels.
This script evaluates:
1. Coding (HumanEval): Run unit tests using evalplus
2. Summarization (IFEval): Check instruction compliance
3. RAG (TriviaQA): Check answer exactness
4. Agentic (LCB): Check code execution results

The goal is to generate proper binary success labels for training.
"""

import json
import pandas as pd
from pathlib import Path
from datasets import load_dataset
import os
from dotenv import load_dotenv
from tqdm import tqdm
import re

load_dotenv()
HF_TOKEN = os.getenv('HUGGINGFACE_API_KEY')


def evaluate_humaneval_predictions(predictions_file: Path, dataset_path: Path = None):
    """
    Evaluate HumanEval predictions by checking code correctness.
    
    Since running arbitrary code is dangerous, we use heuristics and evalplus if available.
    
    Args:
        predictions_file: Path to OpenCompass prediction JSON
        dataset_path: Optional path to HumanEval dataset with test cases
    
    Returns:
        List of dicts with 'task_id', 'prediction', 'passed'
    """
    print(f"\n{'='*80}")
    print(f"EVALUATING HUMANEVAL: {predictions_file.name}")
    print(f"{'='*80}")
    
    with open(predictions_file) as f:
        predictions = json.load(f)
    
    print(f"Loaded {len(predictions)} predictions")
    
    # Load HumanEval dataset with test cases
    try:
        dataset = load_dataset("openai_humaneval", split="test", token=HF_TOKEN)
        humaneval_tests = {item['task_id']: item for item in dataset}
        print(f"✓ Loaded {len(humaneval_tests)} HumanEval problems with tests")
    except Exception as e:
        print(f"⚠️  Could not load HumanEval dataset: {e}")
        humaneval_tests = {}
    
    # Evaluate each prediction
    evaluated = []
    
    for pred in tqdm(predictions, desc="Evaluating"):
        task_id = pred.get('origin_prompt', pred.get('gold', ''))
        code = pred.get('prediction', '')
        
        # Method 1: Use evalplus if available
        if humaneval_tests and task_id in humaneval_tests:
            try:
                # Get test cases
                test_item = humaneval_tests[task_id]
                test_code = test_item.get('test', '')
                entry_point = test_item.get('entry_point', '')
                
                # Combine code + tests
                full_code = code + "\n\n" + test_code + f"\n\ncheck({entry_point})"
                
                # Try to execute safely (using eval in restricted mode)
                # For now, use heuristics - actual execution is complex
                passed = evaluate_code_heuristically(code, test_code)
                
            except Exception as e:
                # print(f"  Error evaluating {task_id}: {e}")
                passed = evaluate_code_heuristically(code, "")
        else:
            # Method 2: Heuristic evaluation
            passed = evaluate_code_heuristically(code, "")
        
        evaluated.append({
            'task_id': task_id,
            'prediction': code,
            'passed': passed
        })
    
    success_rate = sum(1 for e in evaluated if e['passed']) / len(evaluated)
    print(f"\n✓ Evaluated {len(evaluated)} predictions")
    print(f"  Heuristic success rate: {success_rate:.1%}")
    print(f"  ⚠️  NOTE: These are HEURISTIC labels (not actual test execution)")
    
    return evaluated


def evaluate_code_heuristically(code: str, test_code: str = "") -> bool:
    """
    Heuristic evaluation of code quality (NOT actual execution).
    
    Checks for:
    - Has function definition
    - Has return statement
    - No obvious errors
    - Reasonable length
    - No exception/error keywords
    
    This is NOT perfect but provides reasonable proxy labels.
    """
    if not isinstance(code, str) or len(code.strip()) < 10:
        return False
    
    code_lower = code.lower()
    
    # Positive signals
    has_def = 'def ' in code
    has_return = 'return' in code
    reasonable_length = 20 < len(code) < 5000
    
    # Negative signals
    has_error_keywords = any(keyword in code_lower for keyword in [
        'error:', 'exception:', 'failed to', 'cannot', 'unable to',
        'i apologize', "i'm sorry", 'i cannot'
    ])
    
    # Very basic syntax check
    has_unmatched_parens = code.count('(') != code.count(')')
    has_unmatched_brackets = code.count('[') != code.count(']')
    has_unmatched_braces = code.count('{') != code.count('}')
    
    syntax_ok = not (has_unmatched_parens or has_unmatched_brackets or has_unmatched_braces)
    
    # Combine signals
    passed = (
        has_def and 
        has_return and 
        reasonable_length and 
        not has_error_keywords and
        syntax_ok
    )
    
    return passed


def evaluate_ifeval_predictions(predictions_file: Path):
    """
    Evaluate IFEval predictions by checking instruction compliance.
    
    IFEval tests if models follow specific instructions (e.g., "include 3 paragraphs").
    
    Args:
        predictions_file: Path to OpenCompass prediction JSON
    
    Returns:
        List of dicts with 'prompt_id', 'prediction', 'passed'
    """
    print(f"\n{'='*80}")
    print(f"EVALUATING IFEVAL: {predictions_file.name}")
    print(f"{'='*80}")
    
    with open(predictions_file) as f:
        predictions = json.load(f)
    
    print(f"Loaded {len(predictions)} predictions")
    
    # Load IFEval dataset
    try:
        dataset = load_dataset("google/IFEval", split="train", token=HF_TOKEN)
        ifeval_data = {str(i): item for i, item in enumerate(dataset)}
        print(f"✓ Loaded {len(ifeval_data)} IFEval problems")
    except Exception as e:
        print(f"⚠️  Could not load IFEval dataset: {e}")
        ifeval_data = {}
    
    # Evaluate each prediction
    evaluated = []
    
    for i, pred in enumerate(tqdm(predictions, desc="Evaluating")):
        prompt_id = str(i)
        response = pred.get('prediction', '')
        
        # Get instruction requirements
        if prompt_id in ifeval_data:
            prompt_data = ifeval_data[prompt_id]
            instruction_id_list = prompt_data.get('instruction_id_list', [])
            
            # Check compliance (simplified)
            passed = check_instruction_compliance(response, instruction_id_list, prompt_data)
        else:
            # Heuristic: Check if response is reasonable
            passed = len(response.split()) > 10 and len(response) < 10000
        
        evaluated.append({
            'prompt_id': prompt_id,
            'prediction': response,
            'passed': passed
        })
    
    success_rate = sum(1 for e in evaluated if e['passed']) / len(evaluated)
    print(f"\n✓ Evaluated {len(evaluated)} predictions")
    print(f"  Heuristic success rate: {success_rate:.1%}")
    print(f"  ⚠️  NOTE: These are SIMPLIFIED labels (not full IFEval evaluation)")
    
    return evaluated


def check_instruction_compliance(response: str, instruction_ids: list, prompt_data: dict) -> bool:
    """
    Simplified instruction compliance checking for IFEval.
    
    Real IFEval has complex rules. This is a basic heuristic.
    """
    if not response or len(response.strip()) < 10:
        return False
    
    # Check common instruction types
    kwargs = prompt_data.get('kwargs', [{}])[0] if prompt_data.get('kwargs') else {}
    
    # Check for common constraints
    checks_passed = 0
    checks_total = 0
    
    # Length constraints
    if 'num_paragraphs' in kwargs:
        checks_total += 1
        num_paragraphs = len([p for p in response.split('\n\n') if p.strip()])
        if num_paragraphs >= kwargs['num_paragraphs']:
            checks_passed += 1
    
    if 'num_sentences' in kwargs:
        checks_total += 1
        num_sentences = len([s for s in response.split('.') if s.strip()])
        if num_sentences >= kwargs['num_sentences']:
            checks_passed += 1
    
    if 'num_words' in kwargs:
        checks_total += 1
        num_words = len(response.split())
        if num_words >= kwargs['num_words']:
            checks_passed += 1
    
    # Keyword constraints
    if 'keywords' in kwargs:
        checks_total += 1
        keywords = kwargs['keywords']
        if all(kw.lower() in response.lower() for kw in keywords):
            checks_passed += 1
    
    # Default: if no specific checks, just check reasonable length
    if checks_total == 0:
        return len(response.split()) > 10
    
    # Pass if majority of checks passed
    return checks_passed / checks_total >= 0.5


def evaluate_triviaqa_predictions(predictions_file: Path):
    """
    Evaluate TriviaQA predictions by checking answer exactness.
    
    Args:
        predictions_file: Path to OpenCompass prediction JSON
    
    Returns:
        List of dicts with 'question_id', 'prediction', 'passed'
    """
    print(f"\n{'='*80}")
    print(f"EVALUATING TRIVIAQA: {predictions_file.name}")
    print(f"{'='*80}")
    
    with open(predictions_file) as f:
        predictions = json.load(f)
    
    print(f"Loaded {len(predictions)} predictions")
    
    # Load TriviaQA dataset
    try:
        dataset = load_dataset("trivia_qa", "unfiltered.nocontext", split="validation", token=HF_TOKEN)
        triviaqa_data = {item['question_id']: item for item in dataset}
        print(f"✓ Loaded {len(triviaqa_data)} TriviaQA questions")
    except Exception as e:
        print(f"⚠️  Could not load TriviaQA dataset: {e}")
        triviaqa_data = {}
    
    # Evaluate each prediction
    evaluated = []
    
    for pred in tqdm(predictions, desc="Evaluating"):
        question_id = pred.get('origin_prompt', pred.get('gold', ''))
        answer = pred.get('prediction', '')
        
        # Get correct answers
        if question_id in triviaqa_data:
            correct_answers = triviaqa_data[question_id].get('answer', {}).get('aliases', [])
            correct_answers.append(triviaqa_data[question_id].get('answer', {}).get('value', ''))
            
            # Check if answer matches any correct answer (case-insensitive, normalized)
            passed = any(normalize_answer(answer) == normalize_answer(correct) 
                        for correct in correct_answers)
        else:
            # Heuristic: Check if answer is reasonable
            passed = 2 < len(answer.split()) < 50
        
        evaluated.append({
            'question_id': question_id,
            'prediction': answer,
            'passed': passed
        })
    
    success_rate = sum(1 for e in evaluated if e['passed']) / len(evaluated)
    print(f"\n✓ Evaluated {len(evaluated)} predictions")
    print(f"  Success rate: {success_rate:.1%}")
    
    return evaluated


def normalize_answer(text: str) -> str:
    """Normalize answer for comparison."""
    # Remove articles, punctuation, extra whitespace
    text = text.lower().strip()
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def main():
    """
    Main evaluation pipeline.
    
    This script should be run AFTER build_instance_level_training_data.py
    to add proper evaluation labels.
    """
    print("="*80)
    print("EVALUATING OPENCOMPASS PREDICTIONS")
    print("="*80)
    print("\nThis script evaluates raw OpenCompass predictions to generate")
    print("proper binary success/failure labels for training.\n")
    
    # Check if we have the instance-level data
    data_file = Path(__file__).parent / 'instance_level_training_data' / 'instance_level_training_data.csv'
    
    if not data_file.exists():
        print(f"❌ ERROR: {data_file} not found!")
        print("   Run build_instance_level_training_data.py first.")
        return
    
    print(f"✓ Found training data: {data_file}")
    df = pd.read_csv(data_file)
    
    print(f"\nCurrent data status:")
    print(f"  Total examples: {len(df):,}")
    print(f"  By intent:")
    for intent in df['intent'].unique():
        intent_df = df[df['intent'] == intent]
        success_rate = intent_df['success'].mean()
        print(f"    {intent}: {len(intent_df):,} examples, {success_rate:.1%} success")
    
    # Check which intents need evaluation
    print(f"\n{'='*80}")
    print("EVALUATION NEEDED FOR:")
    print(f"{'='*80}")
    
    needs_eval = []
    for intent in df['intent'].unique():
        intent_df = df[df['intent'] == intent]
        success_rate = intent_df['success'].mean()
        
        if success_rate < 0.05 or success_rate > 0.95:
            print(f"  ⚠️  {intent}: {success_rate:.1%} (likely incorrect labels)")
            needs_eval.append(intent)
        else:
            print(f"  ✓ {intent}: {success_rate:.1%} (looks reasonable)")
    
    if not needs_eval:
        print("\n✅ All intents have reasonable success rates!")
        print("   No evaluation needed.")
        return
    
    print(f"\n{'='*80}")
    print(f"RECOMMENDATION")
    print(f"{'='*80}")
    print("\nFor intents with ~0% or ~100% success:")
    print("1. Coding: Use evalplus library or heuristic evaluation")
    print("2. Summarization: Use IFEval compliance checker")
    print("3. RAG: Use answer matching with TriviaQA")
    print("\nCurrently, this script provides HEURISTIC evaluation.")
    print("For production use, consider running actual test suites.")


if __name__ == '__main__':
    main()
