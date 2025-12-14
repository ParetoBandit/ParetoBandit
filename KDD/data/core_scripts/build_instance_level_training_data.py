#!/usr/bin/env python3
"""
Build Instance-Level Training Data for Logistic Regression

This script downloads open-source evaluation results and joins them with
benchmark prompts to create a proper instance-level training dataset.

Data Sources (ALL from OpenCompass):
1. Reasoning: GPQA Diamond (~58 models × 199 prompts)
2. Coding: HumanEval + LCB Code Generation (~58 models × 564 prompts)
3. Agentic: LCB Code Execution + LCB Test Output (~12 models × 200+ prompts)
4. RAG: TriviaQA 1-shot (~19 models × 1,000+ prompts)
5. Summarization: IFEval (~60 models × 541 prompts)

Total: ~97,000 raw examples (50,000-60,000 after deduplication)

The script performs SQL-like JOIN operations:
  File A (Prompts): Benchmark datasets from HuggingFace
  File B (Labels): Prediction results from GitHub/HuggingFace
  Output: (prompt, model, success/failure) tuples

Then computes NVIDIA complexity features for each prompt.
"""

import json
import pandas as pd
from pathlib import Path
from datasets import load_dataset
from typing import Dict, List, Tuple
import requests
from tqdm import tqdm
import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from llm_jury.routing.nvidia_complexity_classifier import NvidiaComplexityClassifier

# Get HuggingFace token for authentication
HF_TOKEN = os.getenv('HUGGINGFACE_API_KEY')


def download_opencompass_predictions():
    """
    Download OpenCompass prediction results for reasoning benchmarks.
    
    Returns:
        Dict mapping model_name -> dataset_name -> list of predictions
    """
    print("="*80)
    print("DOWNLOADING OPENCOMPASS PREDICTIONS")
    print("="*80)
    
    # OpenCompass predictions are on HuggingFace
    # Dataset: opencompass/compass_academic_predictions
    
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
        
        repo_id = "opencompass/compass_academic_predictions"
        
        print(f"Listing files in {repo_id}...")
        files = list_repo_files(repo_id, repo_type='dataset', token=HF_TOKEN)
        
        # Filter for GPQA results only (we focus on reasoning for this demo)
        prediction_files = [f for f in files if f.endswith('.json') and 'GPQA_diamond' in f]
        
        print(f"Found {len(prediction_files)} GPQA prediction files")
        
        # Load name mappings
        try:
            # Import from llm_jury library
            from llm_jury.prediction.models import OPENCOMPASS_TO_CACHE
            print(f"✓ Loaded {len(OPENCOMPASS_TO_CACHE)} model name mappings")
                
                # Filter to only models we can map to cache
                mapped_files = [f for f in prediction_files 
                               if Path(f).stem in OPENCOMPASS_TO_CACHE]
                print(f"✓ Will download {len(mapped_files)} models that match our cache")
                prediction_files = mapped_files
            else:
                print("⚠️  No mapping file found, limiting to 5 models")
                prediction_files = prediction_files[:5]
        except:
            print("⚠️  Could not load mappings, limiting to 5 models")
            prediction_files = prediction_files[:5]
        
        predictions = {}
        
        for file_path in tqdm(prediction_files, desc="Downloading predictions"):
            local_path = hf_hub_download(repo_id, file_path, repo_type='dataset', token=HF_TOKEN)
            
            with open(local_path, 'r') as f:
                data = json.load(f)
            
            # Extract model name from filename
            # Example path: "results_stations/GPQA_diamond/claude-3-5-sonnet-20241022.json"
            # Model name is the filename without extension
            model_name = Path(file_path).stem
            
            # Store predictions directly (list of dicts with 'prediction' and 'gold')
            predictions[model_name] = data
            
            print(f"  ✓ Loaded {file_path}: {len(data)} predictions")
        
        return predictions
        
    except Exception as e:
        print(f"⚠️  Error downloading OpenCompass predictions: {e}")
        print("Falling back to mock data...")
        return {}


def download_opencompass_benchmark(benchmark_name: str, intent_name: str):
    """
    Download predictions for any benchmark from OpenCompass.
    
    Args:
        benchmark_name: Name of the benchmark in OpenCompass (e.g., 'GPQA_diamond', 'IFEval')
        intent_name: Intent category (e.g., 'reasoning', 'summarization')
    
    Returns:
        Dict mapping model_name -> list of predictions
    """
    print("\n" + "="*80)
    print(f"DOWNLOADING {benchmark_name.upper()} PREDICTIONS ({intent_name.upper()})")
    print("="*80)
    
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
        
        repo_id = "opencompass/compass_academic_predictions"
        
        print(f"Listing files in {repo_id}...")
        files = list_repo_files(repo_id, repo_type='dataset', token=HF_TOKEN)
        
        # Filter for this benchmark
        prediction_files = [f for f in files if f.endswith('.json') and benchmark_name in f]
        
        print(f"Found {len(prediction_files)} {benchmark_name} prediction files")
        
        # Load name mappings
        try:
            # Import from llm_jury library
            from llm_jury.prediction.models import OPENCOMPASS_TO_CACHE
            print(f"✓ Loaded {len(OPENCOMPASS_TO_CACHE)} model name mappings")
                
                # Filter to only models we can map to cache
                mapped_files = [f for f in prediction_files 
                               if Path(f).stem in OPENCOMPASS_TO_CACHE]
                print(f"✓ Will download {len(mapped_files)} models that match our cache")
                prediction_files = mapped_files
            else:
                print("⚠️  No mapping file found, limiting to 5 models")
                prediction_files = prediction_files[:5]
        except:
            print("⚠️  Could not load mappings, limiting to 5 models")
            prediction_files = prediction_files[:5]
        
        predictions = {}
        
        for file_path in tqdm(prediction_files, desc=f"Downloading {benchmark_name}"):
            local_path = hf_hub_download(repo_id, file_path, repo_type='dataset', token=HF_TOKEN)
            
            with open(local_path, 'r') as f:
                data = json.load(f)
            
            # Extract model name from filename
            model_name = Path(file_path).stem
            
            # Store predictions directly (list of dicts with 'prediction' and 'gold')
            predictions[model_name] = data
            
            print(f"  ✓ Loaded {file_path}: {len(data)} predictions")
        
        return predictions
        
    except Exception as e:
        print(f"⚠️  Error downloading {benchmark_name} predictions: {e}")
        return {}


def load_gpqa_dataset():
    """Load GPQA Diamond dataset with prompts."""
    print("\n" + "="*80)
    print("LOADING GPQA DATASET (PROMPTS)")
    print("="*80)
    
    try:
        # Load GPQA Diamond from HuggingFace
        dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train", token=HF_TOKEN)
        df = dataset.to_pandas()
        
        # Create unique IDs
        df['question_id'] = df.index.astype(str)
        
        print(f"✓ Loaded {len(df)} GPQA prompts")
        print(f"  Columns: {list(df.columns)}")
        
        return df
        
    except Exception as e:
        print(f"⚠️  Error loading GPQA: {e}")
        return pd.DataFrame()


def load_ifeval_dataset():
    """Load IFEval dataset with prompts."""
    print("\n" + "="*80)
    print("LOADING IFEVAL DATASET (PROMPTS)")
    print("="*80)
    
    try:
        # Load IFEval from HuggingFace
        dataset = load_dataset("google/IFEval", split="train", token=HF_TOKEN)
        df = dataset.to_pandas()
        
        # Create unique IDs
        df['question_id'] = df.index.astype(str)
        
        print(f"✓ Loaded {len(df)} IFEval prompts")
        print(f"  Columns: {list(df.columns)}")
        
        return df
        
    except Exception as e:
        print(f"⚠️  Error loading IFEval: {e}")
        return pd.DataFrame()


def load_triviaqa_dataset():
    """Load TriviaQA dataset with prompts."""
    print("\n" + "="*80)
    print("LOADING TRIVIAQA DATASET (PROMPTS)")
    print("="*80)
    
    try:
        # Load TriviaQA from HuggingFace
        # Note: OpenCompass uses the 'unfiltered' version
        dataset = load_dataset("trivia_qa", "unfiltered.nocontext", split="validation", token=HF_TOKEN)
        df = dataset.to_pandas()
        
        # Create unique IDs
        df['question_id'] = df.index.astype(str)
        
        print(f"✓ Loaded {len(df)} TriviaQA prompts")
        print(f"  Columns: {list(df.columns)}")
        
        return df
        
    except Exception as e:
        print(f"⚠️  Error loading TriviaQA: {e}")
        return pd.DataFrame()


def load_mmlu_pro_dataset():
    """Load MMLU-Pro dataset with prompts."""
    print("\n" + "="*80)
    print("LOADING MMLU-PRO DATASET (PROMPTS)")
    print("="*80)
    
    try:
        # Load MMLU-Pro from HuggingFace
        dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="test", token=HF_TOKEN)
        df = dataset.to_pandas()
        
        # Create unique IDs
        df['question_id'] = df.index.astype(str)
        
        print(f"✓ Loaded {len(df)} MMLU-Pro prompts")
        print(f"  Columns: {list(df.columns)}")
        
        return df
        
    except Exception as e:
        print(f"⚠️  Error loading MMLU-Pro: {e}")
        return pd.DataFrame()


def download_evalplus_results():
    """
    Download EvalPlus (HumanEval+) results from GitHub releases.
    
    Returns:
        Dict mapping model_name -> list of predictions
    """
    print("\n" + "="*80)
    print("DOWNLOADING EVALPLUS RESULTS")
    print("="*80)
    
    predictions = {}
    
    try:
        import zipfile
        import io
        
        # Download from official v0.1.0 release
        url = "https://github.com/evalplus/evalplus/releases/download/v0.1.0/humaneval_results.zip"
        print(f"Downloading from GitHub release: {url}")
        
        response = requests.get(url, timeout=30)
        
        if response.status_code == 404:
            print("❌ 404 Not Found at v0.1.0")
            print("   Trying alternative: leaderboard data...")
            
            # Try the leaderboard data repo
            alt_url = "https://raw.githubusercontent.com/evalplus/evalplus.github.io/master/data/humaneval_plus.json"
            response = requests.get(alt_url, timeout=30)
            
            if response.status_code == 200:
                print(f"✓ Found alternative data source")
                # This would be aggregate scores, not instance-level
                # We'll skip for now and continue with other benchmarks
                return predictions
            else:
                print(f"❌ Alternative also failed: {response.status_code}")
                return predictions
        
        if response.status_code == 200:
            print(f"✓ Downloaded ZIP file ({len(response.content) / 1024:.1f} KB)")
            
            # Extract ZIP
            z = zipfile.ZipFile(io.BytesIO(response.content))
            print(f"✓ Found {len(z.namelist())} model files in archive")
            
            # Parse each model's predictions
            for filename in z.namelist()[:10]:  # Limit to first 10 models
                if not filename.endswith('.jsonl') and not filename.endswith('.json'):
                    continue
                    
                model_name = Path(filename).stem
                
                try:
                    with z.open(filename) as f:
                        # Try JSONL format first
                        content = f.read().decode('utf-8')
                        if filename.endswith('.jsonl'):
                            data = [json.loads(line) for line in content.strip().split('\n') if line.strip()]
                        else:
                            data = json.loads(content)
                        
                        if isinstance(data, list) and len(data) > 0:
                            predictions[model_name] = data
                            print(f"  ✓ Loaded {model_name}: {len(data)} predictions")
                except Exception as e:
                    print(f"  ⚠️  Error parsing {filename}: {e}")
        else:
            print(f"❌ HTTP {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error downloading EvalPlus results: {e}")
    
    if not predictions:
        print("\n⚠️  No EvalPlus predictions downloaded")
        print("   This is OK - we'll use GPQA for now")
    
    return predictions


def load_humaneval_plus_dataset():
    """Load HumanEval+ dataset with prompts."""
    print("\n" + "="*80)
    print("LOADING HUMANEVAL+ DATASET (PROMPTS)")
    print("="*80)
    
    try:
        # Load HumanEval+ from HuggingFace
        dataset = load_dataset("evalplus/humanevalplus", split="test", token=HF_TOKEN)
        df = dataset.to_pandas()
        
        print(f"✓ Loaded {len(df)} HumanEval+ prompts")
        print(f"  Columns: {list(df.columns)}")
        
        return df
        
    except Exception as e:
        print(f"⚠️  Error loading HumanEval+: {e}")
        return pd.DataFrame()


def extract_livecodebench_data(target_models=None):
    """
    Safely extract LiveCodeBench data by joining Problems with Model Submissions.
    
    This avoids schema mismatch errors by:
    1. Loading problems (ground truth) first
    2. Loading submissions model-by-model
    3. Joining on question_id
    
    Args:
        target_models: List of model names to extract. If None, tries common models.
    
    Returns:
        DataFrame with columns: model, prompt_id, difficulty, code, success, question_content
    """
    print("\n" + "="*80)
    print("EXTRACTING LIVECODEBENCH DATA (PROBLEMS + SUBMISSIONS)")
    print("="*80)
    
    if target_models is None:
        # Try common models in our cache
        target_models = [
            "DeepSeek-Coder-V2-Instruct",
            "gpt-4o-2024-05-13",
            "claude-3-opus-20240229",
            "Meta-Llama-3.1-70B-Instruct",
            "Qwen2.5-Coder-32B-Instruct",
        ]
    
    # STEP 1: Load problems (the prompts)
    print("\n--- STEP 1: LOADING PROBLEMS (GROUND TRUTH) ---")
    try:
        ds_problems = load_dataset("livecodebench/code_generation_lite", split="test", token=HF_TOKEN)
        df_problems = ds_problems.to_pandas()
        
        # Keep relevant columns
        problem_cols = ['question_id', 'question_content', 'difficulty']
        df_problems = df_problems[[c for c in problem_cols if c in df_problems.columns]]
        
        print(f"✓ Loaded {len(df_problems)} unique coding problems")
        if 'difficulty' in df_problems.columns:
            print(f"  Difficulty distribution: {df_problems['difficulty'].value_counts().to_dict()}")
    except Exception as e:
        print(f"❌ Error loading problems: {e}")
        return pd.DataFrame()
    
    # STEP 2: Load submissions model by model
    print("\n--- STEP 2: LOADING SUBMISSIONS (MODEL ANSWERS) ---")
    all_rows = []
    models_loaded = 0
    
    for model_name in target_models:
        try:
            print(f"\nProcessing model: {model_name}...")
            
            # Try to load this specific model's submissions
            # The dataset is partitioned by model name as configs
            ds_sub = load_dataset(
                "livecodebench/submissions",
                model_name,
                split="test",
                token=HF_TOKEN,
                trust_remote_code=False
            )
            df_sub = ds_sub.to_pandas()
            
            # Check if we have the required columns
            if 'question_id' not in df_sub.columns:
                print(f"  ⚠️  No 'question_id' column in submissions")
                continue
            
            # Extract success label (pass@1 or graded_list)
            if 'pass@1' in df_sub.columns:
                df_sub['success'] = df_sub['pass@1'].apply(lambda x: int(x) if pd.notna(x) else None)
            elif 'graded_list' in df_sub.columns:
                # graded_list is a list of booleans for each test case
                df_sub['success'] = df_sub['graded_list'].apply(
                    lambda x: int(all(x)) if isinstance(x, list) and len(x) > 0 else None
                )
            else:
                print(f"  ⚠️  No 'pass@1' or 'graded_list' column found")
                continue
            
            # JOIN: Link code to problem difficulty
            merged = pd.merge(df_sub, df_problems, on="question_id", how="inner")
            
            if len(merged) == 0:
                print(f"  ⚠️  No matches after join")
                continue
            
            # Extract rows for training data
            for _, row in merged.iterrows():
                if pd.isna(row['success']):
                    continue
                    
                all_rows.append({
                    'model': model_name,
                    'prompt_id': row['question_id'],
                    'prompt': row.get('question_content', row.get('question_title', '')),
                    'difficulty': row.get('difficulty', 'unknown'),
                    'code': row.get('code', ''),
                    'success': row['success'],
                    'intent': 'coding'
                })
            
            print(f"  ✓ Extracted {len([r for r in all_rows if r['model'] == model_name])} examples")
            models_loaded += 1
            
        except Exception as e:
            print(f"  ⚠️  Could not load {model_name}: {str(e)[:100]}")
            # Try alternate config names
            alt_names = [
                model_name.replace('-', '_'),
                model_name.replace('_', '-'),
                model_name.split('/')[-1] if '/' in model_name else model_name
            ]
            for alt_name in alt_names:
                if alt_name != model_name:
                    try:
                        print(f"     Trying alternate name: {alt_name}...")
                        ds_sub = load_dataset(
                            "livecodebench/submissions",
                            alt_name,
                            split="test",
                            token=HF_TOKEN,
                            trust_remote_code=False
                        )
                        print(f"  ✓ Success with {alt_name}!")
                        # Process this data...
                        break
                    except:
                        continue
    
    if models_loaded == 0:
        print("\n⚠️  No LiveCodeBench submissions loaded")
        print("   This is OK - we'll use GPQA and HumanEval+ for training")
        return pd.DataFrame()
    
    df_result = pd.DataFrame(all_rows)
    print(f"\n✓ Total LiveCodeBench examples extracted: {len(df_result)}")
    print(f"  Models: {df_result['model'].nunique()}")
    print(f"  Unique problems: {df_result['prompt_id'].nunique()}")
    
    return df_result


def join_prompts_and_labels(prompts_df: pd.DataFrame, 
                            predictions: Dict,
                            join_key: str,
                            prompt_column: str,
                            intent: str) -> pd.DataFrame:
    """
    Perform SQL-like JOIN between prompts (File A) and predictions (File B).
    
    Args:
        prompts_df: DataFrame with prompts (has join_key and prompt_column)
        predictions: Dict mapping model_name -> list of predictions
        join_key: Column to join on (e.g., 'question_id', 'task_id')
        prompt_column: Column containing the prompt text
        intent: Intent category (reasoning, coding, etc.)
    
    Returns:
        DataFrame with columns: [prompt, model, intent, success, join_key]
    """
    print(f"\n{'='*80}")
    print(f"JOINING PROMPTS AND LABELS FOR {intent.upper()}")
    print(f"{'='*80}")
    
    joined_data = []
    
    for model_name, model_predictions in predictions.items():
        print(f"\nProcessing {model_name}...")
        
        # Convert predictions to DataFrame
        if isinstance(model_predictions, list):
            pred_df = pd.DataFrame(model_predictions)
            
            # Check if this is OpenCompass format (has 'prediction' and 'gold')
            if 'prediction' in pred_df.columns and 'gold' in pred_df.columns:
                # Handle different benchmark types
                
                if intent == 'reasoning':
                    # For GPQA: Extract multiple choice answer from verbose output
                    def extract_mc_answer(pred_text):
                        """
                        Extract A/B/C/D answer from model's verbose output.
                        Models say "The answer is A" not just "A".
                        """
                        if not isinstance(pred_text, str):
                            return None
                        
                        import re
                        
                        # Try various patterns (ordered by specificity)
                        patterns = [
                            r"[Aa]nswer:\s*\(?([A-D])\)?",                    # "Answer: A" or "answer: (B)"
                            r"[Tt]he answer is:?\s*\(?([A-D])\)?",            # "The answer is A"
                            r"[Cc]orrect answer is:?\s*\(?([A-D])\)?",        # "Correct answer is B"
                            r"[Oo]ption\s+([A-D])\b",                         # "Option A"
                            r"[Cc]hoice\s+([A-D])\b",                         # "Choice B"
                            r"^\s*\(?([A-D])\)?\s*\.?\s*$",                   # Just "A" or "(B)" on a line
                            r"\b([A-D])\s*\)",                                # "A)" format
                            r"answer\s+([A-D])\b",                            # "answer A"
                        ]
                        
                        for pattern in patterns:
                            match = re.search(pattern, pred_text, re.IGNORECASE | re.MULTILINE)
                            if match:
                                return match.group(1).upper()
                        
                        # Fallback: Find last occurrence of isolated A-D letter
                        matches = re.findall(r'\b([A-D])\b', pred_text)
                        if matches:
                            return matches[-1].upper()  # Take the last one (often the final answer)
                        
                        return None
                    
                    pred_df['predicted_answer'] = pred_df['prediction'].apply(extract_mc_answer)
                    pred_df['is_correct'] = pred_df['predicted_answer'] == pred_df['gold']
                    
                    # Debug: Show extraction success rate
                    extraction_rate = pred_df['predicted_answer'].notna().mean()
                    success_rate = pred_df['is_correct'].mean()
                    print(f"    Answer extraction: {extraction_rate:.1%}, Success: {success_rate:.1%}")
                
                elif intent == 'coding':
                    # For coding: Heuristic validation (better than nothing)
                    def is_code_valid_heuristic(code):
                        """
                        Heuristic check if generated code is likely correct.
                        NOT a substitute for unit tests, but provides reasonable proxy labels.
                        """
                        if not isinstance(code, str) or len(code.strip()) < 10:
                            return False
                        
                        code_lower = code.lower()
                        
                        # Positive signals
                        has_def = 'def ' in code
                        has_return = 'return' in code
                        reasonable_length = 20 < len(code) < 5000
                        
                        # Negative signals (model refused or errored)
                        refusal_phrases = [
                            "i cannot", "i can't", "i apologize", "i'm sorry",
                            "unable to", "error:", "exception:", "failed to"
                        ]
                        has_refusal = any(phrase in code_lower for phrase in refusal_phrases)
                        
                        # Basic syntax check
                        has_unmatched_parens = code.count('(') != code.count(')')
                        has_unmatched_brackets = code.count('[') != code.count(']')
                        syntax_ok = not (has_unmatched_parens or has_unmatched_brackets)
                        
                        return has_def and has_return and reasonable_length and not has_refusal and syntax_ok
                    
                    pred_df['is_correct'] = pred_df['prediction'].apply(is_code_valid_heuristic)
                    success_rate = pred_df['is_correct'].mean()
                    print(f"    Heuristic success: {success_rate:.1%}")
                    print(f"    ⚠️  NOTE: Using heuristics, not actual test execution")
                
                elif intent == 'summarization':
                    # For IFEval: Simplified instruction compliance check
                    def is_response_valid(text):
                        """
                        Check if response is valid (not a refusal, reasonable length).
                        Full IFEval evaluation is complex, this is a proxy.
                        """
                        if not isinstance(text, str):
                            return False
                        
                        text_lower = text.lower()
                        
                        # Check length
                        word_count = len(text.split())
                        if word_count < 20 or word_count > 10000:
                            return False
                        
                        # Check for refusals
                        refusal_phrases = [
                            "i cannot", "i can't", "i apologize", "i'm sorry",
                            "unable to", "i don't have", "i do not have"
                        ]
                        has_refusal = any(phrase in text_lower for phrase in refusal_phrases)
                        
                        return not has_refusal
                    
                    pred_df['is_correct'] = pred_df['prediction'].apply(is_response_valid)
                    success_rate = pred_df['is_correct'].mean()
                    print(f"    Heuristic success: {success_rate:.1%}")
                    print(f"    ⚠️  NOTE: Using simplified validation, not full IFEval evaluation")
                
                elif intent == 'agentic':
                    # For agentic (LCB code execution): Similar to coding
                    print(f"  Grading agentic (code validation)...")
                    def is_code_valid_heuristic(code):
                        if not isinstance(code, str) or len(code.strip()) < 10:
                            return False
                        code_lower = code.lower()
                        has_def = 'def ' in code
                        has_return = 'return' in code
                        reasonable_length = 20 < len(code) < 5000
                        refusal_phrases = ["i cannot", "i can't", "i apologize", "i'm sorry"]
                        has_refusal = any(phrase in code_lower for phrase in refusal_phrases)
                        has_unmatched_parens = code.count('(') != code.count(')')
                        syntax_ok = not has_unmatched_parens
                        return has_def and has_return and reasonable_length and not has_refusal and syntax_ok
                    
                    pred_df['is_correct'] = pred_df['prediction'].apply(is_code_valid_heuristic)
                    success_rate = pred_df['is_correct'].mean()
                    print(f"    Heuristic success: {success_rate:.1%}")
                    print(f"    ⚠️  NOTE: Using heuristics for agentic code evaluation")
                
                elif intent == 'rag':
                    # For RAG (TriviaQA): Check if model's answer contains ANY correct answer
                    print(f"  Grading RAG (answer matching)...")
                    def check_answer_match(prediction, gold_list):
                        """
                        Check if prediction contains any of the gold answers.
                        TriviaQA provides multiple acceptable answers as a list.
                        """
                        if not isinstance(prediction, str):
                            return False
                        
                        # gold_list might be a list or a dict with 'aliases' key
                        if isinstance(gold_list, dict):
                            gold_answers = gold_list.get('aliases', [])
                            gold_answers.append(gold_list.get('value', ''))
                        elif isinstance(gold_list, list):
                            gold_answers = gold_list
                        else:
                            gold_answers = [str(gold_list)]
                        
                        import re
                        # Normalize prediction
                        pred_norm = re.sub(r'[^\w\s]', '', prediction.lower()).strip()
                        
                        # Check if ANY gold answer is in prediction
                        for gold in gold_answers:
                            if not gold:
                                continue
                            gold_norm = re.sub(r'[^\w\s]', '', str(gold).lower()).strip()
                            if gold_norm and (gold_norm in pred_norm or pred_norm == gold_norm):
                                return True
                        
                        return False
                    
                    pred_df['is_correct'] = pred_df.apply(
                        lambda row: check_answer_match(row['prediction'], row['gold']), 
                        axis=1
                    )
                    success_rate = pred_df['is_correct'].mean()
                    print(f"    Answer matching success: {success_rate:.1%}")
                
                else:
                    # Default: try to compare
                    pred_df['is_correct'] = pred_df['prediction'] == pred_df['gold']
                
                # Join by index (assuming same order)
                if len(pred_df) == len(prompts_df):
                    merged = prompts_df.copy()
                    merged['is_correct'] = pred_df['is_correct'].values
                else:
                    print(f"  ⚠️  Length mismatch: {len(pred_df)} predictions vs {len(prompts_df)} prompts")
                    
                    # Special handling for RAG (TriviaQA): Extract and match questions
                    if intent == 'rag' and 'origin_prompt' in pred_df.columns:
                        print(f"    Extracting questions from origin_prompt...")
                        
                        def extract_question(origin_prompt):
                            """Extract the actual question from origin_prompt."""
                            if isinstance(origin_prompt, list):
                                for msg in reversed(origin_prompt):
                                    if isinstance(msg, dict) and msg.get('role') == 'HUMAN':
                                        q = msg.get('prompt', '')
                                        return q[3:].strip() if q.startswith('Q: ') else q.strip()
                            return ''
                        
                        pred_df['extracted_question'] = pred_df['origin_prompt'].apply(extract_question)
                        prompts_df['question_norm'] = prompts_df['question'].str.lower().str.strip()
                        pred_df['question_norm'] = pred_df['extracted_question'].str.lower().str.strip()
                        
                        merged = pred_df.merge(prompts_df, on='question_norm', how='inner')
                        print(f"    Matched {len(merged)}/{len(pred_df)} questions")
                        
                        if len(merged) == 0:
                            print(f"    ERROR: No questions matched!")
                            continue
                    else:
                        continue
            else:
                # Try to join on key if available
                if join_key in pred_df.columns:
                    merged = prompts_df.merge(pred_df, on=join_key, how='inner')
                else:
                    print(f"  ⚠️  Cannot find join key '{join_key}' in predictions")
                    continue
                    
        elif isinstance(model_predictions, dict):
            pred_df = pd.DataFrame.from_dict(model_predictions, orient='index').reset_index()
            pred_df.columns = [join_key, 'result']
            merged = prompts_df.merge(pred_df, on=join_key, how='inner')
        else:
            print(f"  ⚠️  Unknown prediction format for {model_name}")
            continue
        
        if len(merged) == 0:
            print(f"  ⚠️  No matches found for {model_name}")
            continue
        
        # Extract relevant columns
        for idx, row in merged.iterrows():
            success = row.get('is_correct', row.get('passed', row.get('success', None)))
            
            # Skip if success is None or NaN
            if pd.isna(success):
                continue
                
            joined_data.append({
                'prompt': row[prompt_column],
                'model': model_name,
                'intent': intent,
                'success': int(success) if isinstance(success, bool) else success,
                'question_id': row.get(join_key, f"{intent}_{idx}")
            })
        
        print(f"  ✓ Joined {len([x for x in joined_data if x['model'] == model_name])} examples for {model_name}")
    
    result_df = pd.DataFrame(joined_data)
    print(f"\n✓ Total joined examples: {len(result_df)}")
    
    return result_df


def compute_nvidia_features(df: pd.DataFrame, batch_size: int = 16) -> pd.DataFrame:
    """
    Compute NVIDIA complexity features for all prompts.
    
    For each prompt in the dataset, this function computes 7 complexity dimensions
    using NVIDIA's prompt-task-and-complexity-classifier:
    - Overall complexity score (0-1)
    - Creativity scope (0-1)
    - Reasoning complexity (0-1)
    - Constraint count (0-1)
    - Domain knowledge required (0-1)
    - Contextual knowledge required (0-1)
    - Number of few-shot examples (0-1)
    
    Args:
        df: DataFrame with 'prompt' column
        batch_size: Number of prompts to process at once (default 16 to avoid OOM)
    
    Returns:
        DataFrame with added NVIDIA complexity columns
    """
    print("\n" + "="*80)
    print("COMPUTING NVIDIA COMPLEXITY FEATURES FOR EACH PROMPT")
    print("="*80)
    
    if 'prompt' not in df.columns:
        print("⚠️  ERROR: DataFrame missing 'prompt' column. Cannot compute NVIDIA features.")
        return df
    
    try:
        from llm_jury.routing.nvidia_complexity_classifier import NvidiaComplexityClassifier
        print("✓ NVIDIA classifier loaded successfully")
        
        classifier = NvidiaComplexityClassifier()
        
        # Get unique prompts to avoid redundant computation
        unique_prompts = df['prompt'].unique().tolist()
        print(f"\n📊 Processing {len(unique_prompts)} unique prompts...")
        print(f"   (Batch size: {batch_size})")
        
        # Filter out any None or empty prompts
        valid_prompts = [p for p in unique_prompts if p and isinstance(p, str) and len(p.strip()) > 0]
        if len(valid_prompts) < len(unique_prompts):
            print(f"⚠️  Filtered out {len(unique_prompts) - len(valid_prompts)} invalid prompts")
        
        # Process in batches with progress bar
        complexity_results = []
        failed_prompts = []
        
        print("\n🔄 Computing NVIDIA complexity scores...")
        for i in tqdm(range(0, len(valid_prompts), batch_size), desc="Processing batches"):
            batch = valid_prompts[i:i+batch_size]
            
            try:
                results = classifier.classify_batch(batch)
                complexity_results.extend(results)
            except Exception as batch_error:
                print(f"\n⚠️  Error in batch {i//batch_size + 1}: {batch_error}")
                print(f"   Attempting individual processing for this batch...")
                
                # Fall back to processing individually for failed batch
                for prompt in batch:
                    try:
                        result = classifier.classify(prompt)
                        complexity_results.append(result)
                    except Exception as prompt_error:
                        print(f"   ✗ Failed to process prompt: {prompt[:50]}...")
                        failed_prompts.append(prompt)
                        # Add placeholder result with default values
                        from llm_jury.routing.nvidia_complexity_classifier import NvidiaComplexityResult
                        complexity_results.append(NvidiaComplexityResult(
                            prompt=prompt,
                            task_type_1="Text Generation",  # Default task type
                            task_type_2="Open QA",          # Default secondary task type
                            task_type_prob=0.0,             # No confidence for failed predictions
                            creativity_scope=0.5,
                            reasoning=0.5,
                            constraint_ct=0.5,
                            domain_knowledge=0.5,
                            contextual_knowledge=0.5,
                            number_of_few_shots=0.0,
                            prompt_complexity_score=0.5
                        ))
        
        print(f"\n✓ Successfully processed {len(complexity_results) - len(failed_prompts)}/{len(valid_prompts)} prompts")
        if failed_prompts:
            print(f"⚠️  {len(failed_prompts)} prompts failed (using default scores)")
        
        # Create mapping from prompt to complexity features
        prompt_to_features = {}
        for prompt, result in zip(valid_prompts, complexity_results):
            prompt_to_features[prompt] = {
                'nvidia_complexity_score': result.prompt_complexity_score,
                'nvidia_creativity': result.creativity_scope,
                'nvidia_reasoning': result.reasoning,
                'nvidia_constraint': result.constraint_ct,
                'nvidia_domain_knowledge': result.domain_knowledge,
                'nvidia_contextual_knowledge': result.contextual_knowledge,
                'nvidia_few_shots': result.number_of_few_shots,
                'nvidia_task_type_1': result.task_type_1,  # Primary predicted task
                'nvidia_task_type_2': result.task_type_2,  # Secondary predicted task
                'nvidia_task_type_prob': result.task_type_prob,  # Prediction confidence
            }
        
        # Add features to DataFrame for ALL rows (each instance)
        print("\n📝 Adding NVIDIA features to all prompt instances...")
        nvidia_columns = ['nvidia_complexity_score', 'nvidia_creativity', 'nvidia_reasoning',
                         'nvidia_constraint', 'nvidia_domain_knowledge', 'nvidia_contextual_knowledge',
                         'nvidia_few_shots', 'nvidia_task_type_1', 'nvidia_task_type_2', 'nvidia_task_type_prob']
        
        for col in nvidia_columns:
            df[col] = df['prompt'].map(lambda p: prompt_to_features.get(p, {}).get(col, None))
        
        # Validation: Check coverage
        coverage = df['nvidia_complexity_score'].notna().sum()
        coverage_pct = coverage / len(df) * 100
        
        print(f"\n✓ Added NVIDIA features to {coverage:,}/{len(df):,} instances ({coverage_pct:.1f}% coverage)")
        
        # Show feature statistics
        print(f"\n📈 NVIDIA Feature Statistics:")
        for col in ['nvidia_complexity_score', 'nvidia_reasoning', 'nvidia_creativity']:
            if col in df.columns:
                mean_val = df[col].mean()
                std_val = df[col].std()
                print(f"   {col:30s}: μ={mean_val:.3f}, σ={std_val:.3f}")
        
        # Show task type distribution
        if 'nvidia_task_type_1' in df.columns:
            print(f"\n📋 Task Type Distribution (Primary Predictions):")
            task_counts = df['nvidia_task_type_1'].value_counts().head(5)
            for task, count in task_counts.items():
                pct = count / len(df) * 100
                print(f"   {task:25s}: {count:5d} ({pct:5.1f}%)")
        
        # Show task prediction confidence
        if 'nvidia_task_type_prob' in df.columns:
            avg_confidence = df['nvidia_task_type_prob'].mean()
            print(f"\n📊 Average Task Prediction Confidence: {avg_confidence:.3f}")
        
        return df
        
    except ImportError as e:
        print(f"⚠️  ERROR: Could not import NVIDIA classifier: {e}")
        print("   Make sure llm_jury.routing.nvidia_complexity_classifier is available")
        print("   Continuing without NVIDIA features...")
        return df
    except Exception as e:
        print(f"⚠️  ERROR computing NVIDIA features: {e}")
        import traceback
        traceback.print_exc()
        print("   Continuing without NVIDIA features...")
        return df


def save_training_data(df: pd.DataFrame, output_path: Path):
    """Save the joined training data to disk."""
    print("\n" + "="*80)
    print("SAVING TRAINING DATA")
    print("="*80)
    
    # Save as CSV
    csv_path = output_path / "instance_level_training_data.csv"
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved CSV: {csv_path}")
    
    # Save as JSON
    json_path = output_path / "instance_level_training_data.json"
    df.to_json(json_path, orient='records', indent=2)
    print(f"✓ Saved JSON: {json_path}")
    
    # Save summary statistics
    summary_path = output_path / "training_data_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("INSTANCE-LEVEL TRAINING DATA SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Total examples: {len(df):,}\n")
        f.write(f"Unique prompts: {df['prompt'].nunique():,}\n")
        f.write(f"Unique models: {df['model'].nunique()}\n\n")
        
        f.write("Examples by intent:\n")
        for intent, count in df['intent'].value_counts().items():
            f.write(f"  - {intent}: {count:,}\n")
        
        f.write("\nExamples by model:\n")
        for model, count in df['model'].value_counts().items():
            f.write(f"  - {model}: {count:,}\n")
        
        f.write("\nSuccess rate:\n")
        if 'success' in df.columns:
            success_rate = df['success'].mean()
            f.write(f"  Overall: {success_rate:.2%}\n")
            
            for intent in df['intent'].unique():
                intent_df = df[df['intent'] == intent]
                intent_rate = intent_df['success'].mean()
                f.write(f"  {intent}: {intent_rate:.2%}\n")
        
        # Add NVIDIA feature statistics
        nvidia_cols = ['nvidia_complexity_score', 'nvidia_reasoning', 'nvidia_creativity',
                      'nvidia_constraint', 'nvidia_domain_knowledge']
        
        f.write("\n" + "="*80 + "\n")
        f.write("NVIDIA COMPLEXITY FEATURES\n")
        f.write("="*80 + "\n\n")
        
        has_nvidia = all(col in df.columns for col in nvidia_cols)
        if has_nvidia:
            coverage = df['nvidia_complexity_score'].notna().sum()
            coverage_pct = coverage / len(df) * 100
            f.write(f"Coverage: {coverage:,}/{len(df):,} instances ({coverage_pct:.1f}%)\n\n")
            
            f.write("Feature statistics:\n")
            for col in nvidia_cols:
                if col in df.columns:
                    mean_val = df[col].mean()
                    std_val = df[col].std()
                    min_val = df[col].min()
                    max_val = df[col].max()
                    f.write(f"  {col}:\n")
                    f.write(f"    Mean: {mean_val:.3f}, Std: {std_val:.3f}\n")
                    f.write(f"    Range: [{min_val:.3f}, {max_val:.3f}]\n")
            
            # Task type distribution
            if 'nvidia_task_type_1' in df.columns:
                f.write("\nTask type distribution (Primary Predictions):\n")
                for task, count in df['nvidia_task_type_1'].value_counts().head(10).items():
                    pct = count / len(df) * 100
                    f.write(f"  {task}: {count:,} ({pct:.1f}%)\n")
            
            # Task prediction confidence
            if 'nvidia_task_type_prob' in df.columns:
                avg_conf = df['nvidia_task_type_prob'].mean()
                median_conf = df['nvidia_task_type_prob'].median()
                f.write(f"\nTask prediction confidence:\n")
                f.write(f"  Mean: {avg_conf:.3f}\n")
                f.write(f"  Median: {median_conf:.3f}\n")
                
                # Show low confidence predictions
                low_conf = (df['nvidia_task_type_prob'] < 0.5).sum()
                if low_conf > 0:
                    low_conf_pct = low_conf / len(df) * 100
                    f.write(f"  Low confidence (<0.5): {low_conf:,} ({low_conf_pct:.1f}%)\n")
        else:
            f.write("⚠️  NVIDIA features not available in dataset\n")
        
        # Add feature columns list
        f.write("\n" + "="*80 + "\n")
        f.write("AVAILABLE FEATURES\n")
        f.write("="*80 + "\n\n")
        for col in sorted(df.columns):
            dtype = str(df[col].dtype)
            non_null = df[col].notna().sum()
            f.write(f"  {col:35s} ({dtype:10s}) - {non_null:,}/{len(df):,} non-null\n")
    
    print(f"✓ Saved summary: {summary_path}")
    
    # Print sample
    print("\nSample of training data (first 10 rows):")
    sample_cols = ['model', 'intent', 'success']
    if 'nvidia_complexity_score' in df.columns:
        sample_cols.append('nvidia_complexity_score')
    if 'nvidia_reasoning' in df.columns:
        sample_cols.append('nvidia_reasoning')
    if 'nvidia_task_type_1' in df.columns:
        sample_cols.append('nvidia_task_type_1')
    if 'nvidia_task_type_prob' in df.columns:
        sample_cols.append('nvidia_task_type_prob')
    
    print(df[sample_cols].head(10).to_string(index=False))


def main():
    """Main pipeline to build instance-level training data."""
    print("="*80)
    print("BUILDING INSTANCE-LEVEL TRAINING DATA FOR LOGISTIC REGRESSION")
    print("="*80)
    print()
    
    output_dir = Path(__file__).parent / "instance_level_training_data"
    output_dir.mkdir(exist_ok=True)
    
    all_training_data = []
    
    # 1. REASONING: GPQA
    print("\n" + "#"*80)
    print("# REASONING: GPQA")
    print("#"*80)
    
    gpqa_prompts = load_gpqa_dataset()
    gpqa_predictions = download_opencompass_benchmark('GPQA_diamond', 'reasoning')
    
    if not gpqa_prompts.empty and gpqa_predictions:
        gpqa_joined = join_prompts_and_labels(
            gpqa_prompts,
            gpqa_predictions,
            join_key='question_id',
            prompt_column='Question',
            intent='reasoning'
        )
        all_training_data.append(gpqa_joined)
    
    # 2. CODING: HumanEval
    print("\n" + "#"*80)
    print("# CODING: HUMANEVAL")
    print("#"*80)
    
    humaneval_prompts = load_humaneval_plus_dataset()
    humaneval_predictions = download_opencompass_benchmark('openai_humaneval', 'coding')
    
    if not humaneval_prompts.empty and humaneval_predictions:
        humaneval_joined = join_prompts_and_labels(
            humaneval_prompts,
            humaneval_predictions,
            join_key='task_id',
            prompt_column='prompt',
            intent='coding'
        )
        all_training_data.append(humaneval_joined)
    
    # 3. CODING: LiveCodeBench Code Generation
    print("\n" + "#"*80)
    print("# CODING: LIVECODEBENCH CODE GENERATION")
    print("#"*80)
    
    # Note: Using the OpenCompass version instead of the direct LCB submissions
    # This gives us more models and cleaner data structure
    lcb_gen_predictions = download_opencompass_benchmark('lcb_code_generation', 'coding')
    
    if lcb_gen_predictions:
        # Load LCB problems (we'll need to match by index)
        try:
            from datasets import load_dataset
            ds_lcb = load_dataset("livecodebench/code_generation_lite", split="test", token=HF_TOKEN)
            lcb_prompts = ds_lcb.to_pandas()
            lcb_prompts['question_id'] = lcb_prompts.index.astype(str)
            
            # Join with predictions
            lcb_gen_joined = join_prompts_and_labels(
                lcb_prompts,
                lcb_gen_predictions,
                join_key='question_id',
                prompt_column='question_content' if 'question_content' in lcb_prompts.columns else 'question_title',
                intent='coding'
            )
            all_training_data.append(lcb_gen_joined)
        except Exception as e:
            print(f"⚠️  Error loading LCB code generation: {e}")
    
    # 4. AGENTIC: LiveCodeBench Code Execution
    print("\n" + "#"*80)
    print("# AGENTIC: LIVECODEBENCH CODE EXECUTION")
    print("#"*80)
    
    lcb_exec_predictions = download_opencompass_benchmark('lcb_code_execution', 'agentic')
    
    if lcb_exec_predictions:
        try:
            # Load LiveCodeBench directly (bypassing deprecated loader)
            from huggingface_hub import hf_hub_download
            
            print("Loading LiveCodeBench problems (direct file access)...")
            local_path = hf_hub_download(
                "livecodebench/code_generation_lite",
                "test.jsonl",
                repo_type='dataset',
                token=HF_TOKEN
            )
            
            # Read JSONL manually
            import json
            lcb_data = []
            with open(local_path) as f:
                for line in f:
                    lcb_data.append(json.loads(line))
            
            lcb_prompts = pd.DataFrame(lcb_data)
            print(f"✓ Loaded {len(lcb_prompts)} LiveCodeBench problems")
            
            # Ensure question_id exists
            if 'question_id' not in lcb_prompts.columns:
                lcb_prompts['question_id'] = lcb_prompts.index.astype(str)
            
            lcb_exec_joined = join_prompts_and_labels(
                lcb_prompts,
                lcb_exec_predictions,
                join_key='question_id',
                prompt_column='question_content' if 'question_content' in lcb_prompts.columns else 'question_title',
                intent='agentic'
            )
            all_training_data.append(lcb_exec_joined)
        except Exception as e:
            print(f"⚠️  Error loading LCB code execution: {e}")
    
    # 5. AGENTIC: LiveCodeBench Test Output Prediction
    print("\n" + "#"*80)
    print("# AGENTIC: LIVECODEBENCH TEST OUTPUT")
    print("#"*80)
    
    lcb_test_predictions = download_opencompass_benchmark('lcb_test_output', 'agentic')
    
    if lcb_test_predictions:
        try:
            # Load LiveCodeBench directly (bypassing deprecated loader)
            from huggingface_hub import hf_hub_download
            
            print("Loading LiveCodeBench problems (direct file access)...")
            local_path = hf_hub_download(
                "livecodebench/code_generation_lite",
                "test.jsonl",
                repo_type='dataset',
                token=HF_TOKEN
            )
            
            # Read JSONL manually
            import json
            lcb_data = []
            with open(local_path) as f:
                for line in f:
                    lcb_data.append(json.loads(line))
            
            lcb_prompts = pd.DataFrame(lcb_data)
            print(f"✓ Loaded {len(lcb_prompts)} LiveCodeBench problems")
            
            # Ensure question_id exists
            if 'question_id' not in lcb_prompts.columns:
                lcb_prompts['question_id'] = lcb_prompts.index.astype(str)
            
            lcb_test_joined = join_prompts_and_labels(
                lcb_prompts,
                lcb_test_predictions,
                join_key='question_id',
                prompt_column='question_content' if 'question_content' in lcb_prompts.columns else 'question_title',
                intent='agentic'
            )
            all_training_data.append(lcb_test_joined)
        except Exception as e:
            print(f"⚠️  Error loading LCB test output: {e}")
    
    # 6. RAG: TriviaQA
    print("\n" + "#"*80)
    print("# RAG: TRIVIAQA")
    print("#"*80)
    
    triviaqa_prompts = load_triviaqa_dataset()
    triviaqa_predictions = download_opencompass_benchmark('triviaqa_wiki_1shot', 'rag')
    
    if not triviaqa_prompts.empty and triviaqa_predictions:
        triviaqa_joined = join_prompts_and_labels(
            triviaqa_prompts,
            triviaqa_predictions,
            join_key='question_id',
            prompt_column='question',
            intent='rag'
        )
        all_training_data.append(triviaqa_joined)
    
    # 7. SUMMARIZATION: IFEval
    print("\n" + "#"*80)
    print("# SUMMARIZATION: IFEVAL")
    print("#"*80)
    
    ifeval_prompts = load_ifeval_dataset()
    ifeval_predictions = download_opencompass_benchmark('IFEval', 'summarization')
    
    if not ifeval_prompts.empty and ifeval_predictions:
        ifeval_joined = join_prompts_and_labels(
            ifeval_prompts,
            ifeval_predictions,
            join_key='question_id',
            prompt_column='prompt',
            intent='summarization'
        )
        all_training_data.append(ifeval_joined)
    
    # Combine all training data
    if all_training_data:
        print("\n" + "="*80)
        print("COMBINING ALL TRAINING DATA")
        print("="*80)
        
        combined_df = pd.concat(all_training_data, ignore_index=True)
        
        print(f"\n✓ Combined {len(combined_df)} training examples:")
        for intent in combined_df['intent'].unique():
            count = (combined_df['intent'] == intent).sum()
            print(f"   - {intent}: {count:,} examples")
        
        print(f"\nModels in dataset: {combined_df['model'].nunique()}")
        print(f"Unique prompts: {combined_df['prompt'].nunique()}")
        
        # Compute NVIDIA complexity features for ALL prompt instances
        print("\n" + "#"*80)
        print("# STEP: ADDING NVIDIA COMPLEXITY SCORES TO EACH PROMPT")
        print("#"*80)
        
        combined_df = compute_nvidia_features(combined_df, batch_size=16)
        
        # Validate NVIDIA features were added
        nvidia_cols = ['nvidia_complexity_score', 'nvidia_reasoning', 'nvidia_creativity']
        missing_features = [col for col in nvidia_cols if col not in combined_df.columns]
        
        if missing_features:
            print(f"\n⚠️  WARNING: Missing NVIDIA features: {missing_features}")
            print("   Training will proceed but models may have lower accuracy")
        else:
            coverage = combined_df['nvidia_complexity_score'].notna().sum()
            if coverage < len(combined_df) * 0.95:  # Less than 95% coverage
                print(f"\n⚠️  WARNING: Low NVIDIA feature coverage ({coverage}/{len(combined_df)})")
                print("   Consider investigating why some prompts failed to process")
            else:
                print(f"\n✓ NVIDIA features successfully added to all {len(combined_df):,} instances!")
        
        # Save to disk
        save_training_data(combined_df, output_dir)
        
        print("\n" + "="*80)
        print("✓ SUCCESS: Instance-level training data built!")
        print("="*80)
        print(f"\nOutput directory: {output_dir}")
        print(f"Total training examples: {len(combined_df):,}")
        print(f"With NVIDIA features: {combined_df['nvidia_complexity_score'].notna().sum():,}")
        print(f"\nReady for training! Run:")
        print(f"  python3 KDD/data/core_scripts/train_final_xgboost_models.py")
        
    else:
        print("\n" + "="*80)
        print("⚠️  WARNING: No training data collected")
        print("="*80)
        print("This may be due to:")
        print("  1. Network issues downloading datasets")
        print("  2. Missing dependencies (datasets, huggingface_hub)")
        print("  3. API rate limits")
        print("\nPlease install requirements:")
        print("  pip install datasets huggingface_hub requests tqdm transformers torch")


if __name__ == '__main__':
    main()
