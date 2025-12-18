#!/usr/bin/env python3
"""
Multi-Dimension Quality Predictor

Trains on THREE dimensions of difficulty:
1. STYLE (HelpSteer2): "Is this good assistant behavior?"
2. KNOWLEDGE (NaturalQuestions): "Does this require rare facts?"
3. REASONING (BBH): "Is the logic sound?"

This creates a router that can detect different failure modes:
- Hallucination (knowledge gaps)
- Logic errors (reasoning failures)
- Instruction drift (style violations)

Usage:
    python -m llm_jury.neural_routing.multi_dimension_predictor --epochs 3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import json
import copy
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import Counter

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, ConcatDataset

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(x, **kwargs):
        return x

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from datasets import load_dataset
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class MultiDimConfig:
    """Configuration for Multi-Dimension Predictor."""
    backbone: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # Training
    batch_size: int = 32
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    max_epochs: int = 3
    max_length: int = 256
    
    # Dataset sizes (balanced diet)
    helpsteer_max: int = 10000   # Style dimension
    nq_max: int = 5000           # Knowledge dimension  
    bbh_max: int = 5000          # Reasoning dimension
    gpqa_max: int = 500          # Expert/Hard dimension (graduate-level science)
    
    # Ordinal thresholds (0-4 scale)
    num_thresholds: int = 4

    # Include explicit task/dimension token in the prompt.
    # This is a major accuracy win for true multi-task training because otherwise
    # the encoder must infer which label semantics apply.
    # Default OFF to avoid requiring any runtime prefix selection. If you enable
    # this for research experiments, production must pass the intended dimension
    # tag (or you must run all heads).
    use_dimension_prefix: bool = False
    dimension_prefix_template: str = "[DIM={dim}]\n"

    # Loss shaping: cost-sensitive learning to reduce BAD->GOOD slip-through.
    # We apply per-sample weights to ordinal BCE.
    loss_weight_bad: float = 2.0
    loss_weight_mid: float = 1.0
    loss_weight_good: float = 1.0
    per_dim_loss_multiplier: Dict[str, float] = field(
        default_factory=lambda: {"expert": 2.5, "knowledge": 1.5, "reasoning": 1.2, "style": 1.0}
    )

    # -------------------------------------------------------------------------
    # Decision policy (post-hoc calibration)
    #
    # IMPORTANT:
    # - A single global GOOD/BAD threshold is brittle because dimensions differ
    #   in difficulty + label noise.
    # - We therefore calibrate per-dimension thresholds on a held-out set to
    #   target low "BAD -> predicted GOOD" false positives.
    # - We also keep an UNCERTAIN band (abstain) between T_bad and T_good.
    # -------------------------------------------------------------------------
    default_good_threshold: float = 2.5  # legacy fallback (0-4 score scale)
    default_bad_threshold: float = 1.5   # legacy fallback (0-4 score scale)
    target_good_fp_rate: float = 0.10    # P(predict GOOD | true BAD) target
    target_bad_fp_rate: float = 0.10     # P(predict BAD  | true GOOD) target
    min_uncertainty_band: float = 0.25   # enforce T_good - T_bad >= this

    # Per-dimension overrides (recommended: make EXPERT much stricter)
    per_dim_target_good_fp_rate: Dict[str, float] = field(
        default_factory=lambda: {"expert": 0.02}
    )
    per_dim_target_bad_fp_rate: Dict[str, float] = field(default_factory=dict)
    
    val_split: float = 0.1
    
    checkpoint_dir: Path = field(
        default_factory=lambda: Path(__file__).parent.parent.parent / "data" / "multi_dim_predictor"
    )


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _safe_quantile(values: np.ndarray, q: float, default: float) -> float:
    """Quantile helper that is robust to empty inputs."""
    if values is None:
        return float(default)
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float(default)
    return float(np.quantile(values, q))


def calibrate_thresholds_from_val(
    model: "MultiDimPredictor",
    val_loader: DataLoader,
    device: torch.device,
    config: Optional[MultiDimConfig] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Calibrate per-dimension thresholds from validation data.

    Policy:
      - If score >= T_good(dim) -> GOOD
      - If score <= T_bad(dim)  -> BAD
      - Else                   -> UNCERTAIN

    Thresholds chosen to approximately hit:
      - target_good_fp_rate: P(predict GOOD | true BAD)
      - target_bad_fp_rate:  P(predict BAD  | true GOOD)

    Borderline labels (correctness == 2) are ignored for calibration.
    """
    cfg = config or getattr(model, "config", MultiDimConfig())
    model.eval()

    dim_scores: Dict[str, List[float]] = {"style": [], "knowledge": [], "reasoning": [], "expert": []}
    dim_targets: Dict[str, List[int]] = {"style": [], "knowledge": [], "reasoning": [], "expert": []}

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Calibrating thresholds", leave=False):
            scores = model.predict_score(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
            ).cpu().numpy()

            for i in range(len(scores)):
                dim = str(batch["dimension"][i]).lower()
                if dim not in dim_scores:
                    continue
                dim_scores[dim].append(float(scores[i]))
                dim_targets[dim].append(int(batch["correctness"][i].item()))

    thresholds_by_dimension: Dict[str, Dict[str, float]] = {}
    score_min, score_max = 0.0, float(cfg.num_thresholds)  # e.g. 0..4

    for dim in ["style", "knowledge", "reasoning", "expert"]:
        scores = np.asarray(dim_scores[dim], dtype=np.float64)
        targets = np.asarray(dim_targets[dim], dtype=np.int64)

        # Ignore "borderline" correctness == 2 during calibration.
        bad_scores = scores[targets <= 1]
        good_scores = scores[targets >= 3]

        good_fp_target = float(cfg.per_dim_target_good_fp_rate.get(dim, cfg.target_good_fp_rate))
        bad_fp_target = float(cfg.per_dim_target_bad_fp_rate.get(dim, cfg.target_bad_fp_rate))

        # GOOD threshold: set using BAD distribution to bound BAD->GOOD slip-through.
        t_good = _safe_quantile(
            bad_scores,
            q=max(0.0, min(1.0, 1.0 - good_fp_target)),
            default=cfg.default_good_threshold,
        )
        # Break ties so that a BAD sample exactly at the quantile does not pass.
        # (Decision uses strict > / < below.)
        t_good = float(t_good) + 1e-6

        # BAD threshold: set using GOOD distribution to bound GOOD->BAD over-rejection.
        t_bad = _safe_quantile(
            good_scores,
            q=max(0.0, min(1.0, bad_fp_target)),
            default=cfg.default_bad_threshold,
        )
        t_bad = float(t_bad) - 1e-6

        # Ensure an uncertainty band exists.
        if (t_good - t_bad) < cfg.min_uncertainty_band:
            mid = 0.5 * (t_good + t_bad)
            t_bad = mid - 0.5 * cfg.min_uncertainty_band
            t_good = mid + 0.5 * cfg.min_uncertainty_band

        # Clip to valid range.
        t_bad = float(np.clip(t_bad, score_min, score_max))
        t_good = float(np.clip(t_good, score_min, score_max))

        thresholds_by_dimension[dim] = {
            "t_good": t_good,
            "t_bad": t_bad,
            "target_good_fp_rate": good_fp_target,
            "target_bad_fp_rate": bad_fp_target,
            "n_val": float(len(scores)),
            "n_bad": float(len(bad_scores)),
            "n_good": float(len(good_scores)),
        }

    return thresholds_by_dimension


def _format_prompt_with_dimension(prompt: str, dimension: Optional[str], config: MultiDimConfig) -> str:
    """
    Add an explicit dimension tag to the prompt to make the task observable.
    """
    p = str(prompt)
    if not config.use_dimension_prefix:
        return p
    if dimension is None:
        return p
    dim = str(dimension).strip().upper()
    prefix = config.dimension_prefix_template.format(dim=dim)
    return prefix + p


# =============================================================================
# Data Loaders for Each Dimension
# =============================================================================

def load_helpsteer_dimension(max_samples: int, seed: int = 42) -> pd.DataFrame:
    """
    DIMENSION 1: STYLE
    HelpSteer2 teaches "Is this good assistant behavior?"
    
    Failure mode: Instruction drift, chatty fluff, wrong format
    """
    print("\n[DIM 1: STYLE] Loading HelpSteer2...")
    
    dataset = load_dataset("nvidia/HelpSteer2", split="train")
    df = pd.DataFrame(dataset)
    df = df.drop_duplicates(subset=['prompt', 'response'])
    
    # Sample if needed
    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=seed)
    
    # Normalize to unified format
    df = df[['prompt', 'response', 'correctness']].copy()
    df['dimension'] = 'style'
    df['correctness'] = df['correctness'].astype(int)  # 0-4 scale
    
    print(f"  Loaded {len(df):,} samples")
    print(f"  Correctness dist: {dict(df['correctness'].value_counts().sort_index())}")
    
    return df


def _classify_question_subtype(question: str) -> str:
    """
    Classify question into FINE-GRAINED subtype for hard negative generation.
    
    Key insight: wrong answers should be from the SAME subtype:
    - "capital of X" → wrong answer should be another capital city
    - "who plays X" → wrong answer should be another actor
    - "when did X" → wrong answer should be another date
    """
    q_lower = question.lower()
    
    # GEOGRAPHY subtypes (more specific)
    if 'capital of' in q_lower or 'capital city' in q_lower:
        return 'geo_capital'
    elif any(kw in q_lower for kw in ['largest city', 'biggest city', 'major city']):
        return 'geo_city'
    elif any(kw in q_lower for kw in ['country', 'nation', 'which country']):
        return 'geo_country'
    elif any(kw in q_lower for kw in ['river', 'longest river', 'flows']):
        return 'geo_river'
    elif any(kw in q_lower for kw in ['mountain', 'highest', 'tallest peak']):
        return 'geo_mountain'
    elif any(kw in q_lower for kw in ['ocean', 'sea', 'largest ocean']):
        return 'geo_ocean'
    elif any(kw in q_lower for kw in ['continent', 'which continent']):
        return 'geo_continent'
    elif any(kw in q_lower for kw in ['where is', 'located', 'location']):
        return 'geo_location'
    
    # PERSON subtypes
    elif any(kw in q_lower for kw in ['who plays', 'who played', 'actor', 'actress', 'cast']):
        return 'person_actor'
    elif any(kw in q_lower for kw in ['who sang', 'who sings', 'singer', 'performed by']):
        return 'person_singer'
    elif any(kw in q_lower for kw in ['who wrote', 'author', 'written by', 'writer']):
        return 'person_writer'
    elif any(kw in q_lower for kw in ['who directed', 'director']):
        return 'person_director'
    elif any(kw in q_lower for kw in ['president', 'prime minister', 'leader', 'ruler', 'king', 'queen']):
        return 'person_leader'
    elif any(kw in q_lower for kw in ['ceo', 'founder', 'started', 'founded']):
        return 'person_business'
    elif any(kw in q_lower for kw in ['who is', 'who was']):
        return 'person_general'
    
    # DATE subtypes
    elif any(kw in q_lower for kw in ['what year', 'which year', 'in what year']):
        return 'date_year'
    elif any(kw in q_lower for kw in ['when did', 'when was', 'when is']):
        return 'date_when'
    elif any(kw in q_lower for kw in ['how old', 'age of', 'born in']):
        return 'date_age'
    
    # NUMBER subtypes
    elif any(kw in q_lower for kw in ['how many', 'number of', 'count']):
        return 'number_count'
    elif any(kw in q_lower for kw in ['how much', 'cost', 'price', 'worth']):
        return 'number_amount'
    elif any(kw in q_lower for kw in ['how long', 'length', 'duration', 'distance']):
        return 'number_measure'
    
    # SCIENCE subtypes
    elif any(kw in q_lower for kw in ['chemical', 'element', 'formula', 'compound']):
        return 'science_chemistry'
    elif any(kw in q_lower for kw in ['planet', 'star', 'galaxy', 'solar system']):
        return 'science_astronomy'
    elif any(kw in q_lower for kw in ['species', 'animal', 'mammal', 'bird']):
        return 'science_biology'
    
    # Fallback
    else:
        return 'other'


def load_nq_dimension(max_samples: int, seed: int = 42) -> pd.DataFrame:
    """
    DIMENSION 2: KNOWLEDGE
    NaturalQuestions + TriviaQA teaches "Does the answer contain correct facts?"
    
    KEY FIX: Creates HARD negatives by SUBTYPE:
    - "capital of X" → wrong answer is another capital city (not a date or person)
    - "who plays X" → wrong answer is another actor (not a singer)
    - "what year X" → wrong answer is another year (not a city)
    
    Failure mode: Hallucination, making up facts
    """
    print("\n[DIM 2: KNOWLEDGE] Loading NaturalQuestions + TriviaQA...")
    
    np.random.seed(seed)
    
    # Collect Q&A pairs grouped by FINE-GRAINED subtype
    subtype_qa = {}  # Will have keys like 'geo_capital', 'person_actor', etc.
    
    # Load NQ
    try:
        nq_dataset = load_dataset("nq_open", split="train")
        print(f"  Loaded NQ: {len(nq_dataset):,} samples")
        
        indices = np.random.permutation(len(nq_dataset))
        for idx in indices[:max_samples]:
            item = nq_dataset[int(idx)]
            question = item.get('question', '')
            answers = item.get('answer', [])
            
            if not question or not answers:
                continue
            
            correct_answer = answers[0] if isinstance(answers, list) else str(answers)
            if len(str(correct_answer).split()) > 10:
                continue
            
            subtype = _classify_question_subtype(question)
            if subtype not in subtype_qa:
                subtype_qa[subtype] = []
            subtype_qa[subtype].append({'question': question, 'answer': str(correct_answer)})
            
    except Exception as e:
        print(f"  NQ load failed: {e}")
    
    # Load TriviaQA for better coverage (especially geography)
    try:
        trivia_dataset = load_dataset("trivia_qa", "rc.nocontext", split="train")
        print(f"  Loaded TriviaQA: {len(trivia_dataset):,} samples")
        
        trivia_indices = np.random.permutation(len(trivia_dataset))
        for idx in trivia_indices[:max_samples // 2]:
            item = trivia_dataset[int(idx)]
            question = item.get('question', '')
            answer_info = item.get('answer', {})
            
            aliases = answer_info.get('aliases', [])
            if not aliases:
                normalized = answer_info.get('normalized_value', '')
                if normalized:
                    aliases = [normalized]
            
            if not aliases or not question:
                continue
            
            # Use shortest alias (usually cleanest)
            correct_answer = min(aliases, key=len)
            if len(correct_answer.split()) > 10:
                continue
            
            subtype = _classify_question_subtype(question)
            if subtype not in subtype_qa:
                subtype_qa[subtype] = []
            subtype_qa[subtype].append({'question': question, 'answer': correct_answer})
            
    except Exception as e:
        print(f"  TriviaQA not available: {e}")
    
    # Print subtype distribution
    print(f"  Subtypes: {', '.join(f'{k}={len(v)}' for k, v in sorted(subtype_qa.items(), key=lambda x: -len(x[1])) if len(v) > 10)}")
    
    # Create training samples with HARD negatives (same subtype)
    samples = []
    
    for subtype, qa_list in subtype_qa.items():
        if len(qa_list) < 2:
            continue
        
        for i, qa in enumerate(qa_list):
            # Add CORRECT sample
            samples.append({
                'prompt': qa['question'],
                'response': qa['answer'],
                'correctness': 4,
                'dimension': 'knowledge'
            })
            
            # Add INCORRECT sample using answer from SAME DOMAIN (key fix!)
            # This ensures "Paris" vs "Lyon" type confusion is in training data
            other_indices = [j for j in range(len(qa_list)) if j != i and qa_list[j]['answer'] != qa['answer']]
            if other_indices:
                wrong_idx = np.random.choice(other_indices)
                wrong_answer = qa_list[wrong_idx]['answer']
                samples.append({
                    'prompt': qa['question'],
                    'response': wrong_answer,
                    'correctness': 0,
                    'dimension': 'knowledge'
                })
    
    df = pd.DataFrame(samples)
    
    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=seed)
    
    print(f"  Loaded {len(df):,} samples")
    print(f"  Correctness dist: {dict(df['correctness'].value_counts().sort_index())}")
    
    return df


def load_gpqa_dimension(max_samples: int, seed: int = 42) -> pd.DataFrame:
    """
    DIMENSION 4: EXPERT/HARD
    GPQA teaches "Is this an extremely difficult question?"
    
    Graduate-level science questions (physics, chemistry, biology).
    Even domain experts struggle with these.
    Failure mode: Overconfidence on hard problems
    """
    print("\n[DIM 4: EXPERT] Loading GPQA (Graduate-level Q&A)...")
    
    # GPQA has multiple configs - try diamond (hardest) first
    try:
        dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    except Exception:
        try:
            dataset = load_dataset("Idavidrein/gpqa", "gpqa_main", split="train")
        except Exception:
            # Try alternative GPQA source
            dataset = load_dataset("openai/gpqa", split="train")
    
    samples = []
    np.random.seed(seed)
    
    for item in dataset:
        if len(samples) >= max_samples:
            break
        
        question = item.get('Question', item.get('question', ''))
        correct_answer = item.get('Correct Answer', item.get('correct_answer', ''))
        
        # Get incorrect options
        incorrect_answers = []
        for key in ['Incorrect Answer 1', 'Incorrect Answer 2', 'Incorrect Answer 3',
                    'incorrect_answer_1', 'incorrect_answer_2', 'incorrect_answer_3']:
            if key in item and item[key]:
                incorrect_answers.append(item[key])
        
        if not question or not correct_answer:
            continue
        
        # Add CORRECT sample (these are HARD questions, so correctness=4 means expert-level correct)
        samples.append({
            'prompt': question,
            'response': correct_answer,
            'correctness': 4,
            'dimension': 'expert'
        })
        
        # Add INCORRECT samples
        for wrong in incorrect_answers[:2]:
            samples.append({
                'prompt': question,
                'response': wrong,
                'correctness': 0,
                'dimension': 'expert'
            })
    
    df = pd.DataFrame(samples)
    
    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=seed)
    
    print(f"  Loaded {len(df):,} samples")
    print(f"  Correctness dist: {dict(df['correctness'].value_counts().sort_index())}")
    
    return df


def load_bbh_dimension(max_samples: int, seed: int = 42) -> pd.DataFrame:
    """
    DIMENSION 3: REASONING
    BIG-Bench Hard teaches "Is the logic sound?"
    
    Uses real BBH questions with correct answers, and creates wrong answers
    by using correct answers from OTHER questions (realistic wrong reasoning).
    
    Failure mode: Logic errors, wrong reasoning steps
    """
    print("\n[DIM 3: REASONING] Loading BBH (BIG-Bench Hard)...")
    
    np.random.seed(seed)
    
    try:
        # BBH has multiple tasks - load several for diversity
        bbh_tasks = [
            "logical_deduction_three_objects",
            "tracking_shuffled_objects_three_objects", 
            "date_understanding",
            "penguins_in_a_table",
            "reasoning_about_colored_objects",
            "boolean_expressions",
            "causal_judgement",
            "navigate",
        ]
        
        # First pass: collect all correct Q&A pairs
        correct_pairs = []
        
        for task in bbh_tasks:
            try:
                dataset = load_dataset("lukaemon/bbh", task, split="test")
                
                for item in dataset:
                    question = item.get('input', '')
                    correct_answer = item.get('target', '')
                    
                    if question and correct_answer:
                        correct_pairs.append({
                            'prompt': question,
                            'response': correct_answer,
                            'task': task
                        })
                        
            except Exception as e:
                print(f"    Warning: Could not load {task}: {e}")
                continue
        
        if len(correct_pairs) == 0:
            raise RuntimeError("No BBH samples loaded")
        
        print(f"  Loaded {len(correct_pairs):,} correct Q&A pairs from BBH")
        
        # Second pass: create training samples with correct AND incorrect answers
        all_samples = []
        
        for i, pair in enumerate(correct_pairs):
            # Add CORRECT sample
            all_samples.append({
                'prompt': pair['prompt'],
                'response': pair['response'],
                'correctness': 4,
                'dimension': 'reasoning'
            })
            
            # Add INCORRECT sample: use answer from a DIFFERENT question
            # This is a realistic "wrong reasoning" scenario
            other_indices = [j for j in range(len(correct_pairs)) 
                           if j != i and correct_pairs[j]['response'] != pair['response']]
            
            if other_indices:
                wrong_idx = np.random.choice(other_indices)
                wrong_answer = correct_pairs[wrong_idx]['response']
                
                all_samples.append({
                    'prompt': pair['prompt'],
                    'response': wrong_answer,
                    'correctness': 0,
                    'dimension': 'reasoning'
                })
        
        df = pd.DataFrame(all_samples)
        
    except Exception as e:
        raise RuntimeError(f"BBH load failed: {e}. Cannot proceed without real data.")
    
    if len(df) == 0:
        raise RuntimeError("BBH loaded 0 samples. Check dataset availability.")
    
    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=seed)
    
    print(f"  Final: {len(df):,} samples")
    print(f"  Correctness dist: {dict(df['correctness'].value_counts().sort_index())}")
    
    return df


# =============================================================================
# Unified Dataset
# =============================================================================

class MultiDimDataset(Dataset):
    """
    Unified dataset for multi-dimensional training.
    
    Key improvements:
    1. Uses PAIR tokenization (prompt, response) for proper segment handling
    2. Balances by dimension × correctness × length (critical for short answers)
    3. Stores prompt for pairwise ranking loss
    """
    
    def __init__(self, df: pd.DataFrame, tokenizer, config: MultiDimConfig):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.config = config
        
        # Add metadata for balanced sampling
        self.df['response_words'] = self.df['response'].apply(lambda x: len(str(x).split()))
        self.df['length_bucket'] = self.df['response_words'].apply(
            lambda x: 0 if x <= 3 else (1 if x <= 10 else (2 if x <= 30 else 3))
        )
        self.df['correctness_bin'] = self.df['correctness'].apply(
            lambda x: 0 if x <= 1 else (1 if x == 2 else 2)
        )
        # FIX: Group by dimension + length + correctness (include length!)
        self.df['sample_group'] = self.df.apply(
            lambda r: f"{r['dimension']}_{r['length_bucket']}_{r['correctness_bin']}", axis=1
        )
        
        # Build prompt-to-indices map for pairwise ranking
        self._build_prompt_pairs()
    
    def _build_prompt_pairs(self):
        """Build index of (correct, wrong) pairs for same prompt."""
        self.prompt_pairs = {}
        prompt_to_indices = {}
        
        for idx, row in self.df.iterrows():
            # IMPORTANT: include dimension in the key so we don't accidentally
            # create cross-task pairs for identical question strings.
            prompt = f"{row['dimension']}||{row['prompt']}"
            if prompt not in prompt_to_indices:
                prompt_to_indices[prompt] = {'correct': [], 'wrong': []}
            
            if row['correctness'] >= 3:
                prompt_to_indices[prompt]['correct'].append(idx)
            elif row['correctness'] <= 1:
                prompt_to_indices[prompt]['wrong'].append(idx)
        
        # Store only prompts with both correct and wrong examples
        self.contrastive_prompts = [
            p for p, indices in prompt_to_indices.items()
            if indices['correct'] and indices['wrong']
        ]
        self.prompt_to_indices = prompt_to_indices
        
        n_pairs = sum(
            len(v['correct']) * len(v['wrong']) 
            for v in prompt_to_indices.values()
        )
        print(f"  Built {len(self.contrastive_prompts):,} prompts with contrastive pairs ({n_pairs:,} total pairs)")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        dimension = str(row["dimension"])
        prompt = _format_prompt_with_dimension(str(row["prompt"]), dimension, self.config)
        response = str(row['response'])
        correctness = int(row['correctness'])
        
        # FIX: Use PAIR tokenization instead of string concatenation
        # This lets the model treat prompt and response as separate segments
        enc = self.tokenizer(
            prompt,                          # First segment
            response,                        # Second segment  
            max_length=self.config.max_length,
            padding='max_length',
            truncation=True,                 # Truncates the second segment (response) if needed
            return_tensors='pt'
        )
        
        # Ordinal targets
        ordinal_targets = torch.zeros(self.config.num_thresholds, dtype=torch.float32)
        for k in range(self.config.num_thresholds):
            if correctness > k:
                ordinal_targets[k] = 1.0
        
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'ordinal_targets': ordinal_targets,
            'correctness': torch.tensor(correctness, dtype=torch.long),
            'dimension': dimension,
            'length_bucket': row['length_bucket'],
            'prompt': f"{dimension}||{row['prompt']}",  # key for pairwise loss
            'idx': idx,
        }
    
    def get_balanced_sampler(self, verbose: bool = True) -> WeightedRandomSampler:
        """
        FIX: Balance across dimension × correctness × LENGTH.
        
        This ensures each batch contains:
        - Examples from all dimensions
        - Both correct and incorrect examples  
        - SHORT correct examples (critical for factual Q&A)
        """
        # FIX: Include length_bucket in balance group
        self.df['balance_group'] = self.df.apply(
            lambda r: f"{r['dimension']}_{r['correctness_bin']}_{r['length_bucket']}", axis=1
        )
        
        group_counts = Counter(self.df['balance_group'])
        
        if verbose:
            print("\nBalanced sampling (dim × correctness × length):")
            for group, count in sorted(group_counts.items()):
                weight = 1.0 / count
                print(f"  {group}: {count:,} samples → weight={weight:.4f}")
        
        # Inverse frequency weighting
        weights = [1.0 / group_counts[self.df.iloc[i]['balance_group']] for i in range(len(self.df))]
        
        return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# =============================================================================
# Model (same architecture, multi-dimensional training)
# =============================================================================

class MultiDimPredictor(nn.Module):
    """Cross-encoder predictor trained on multiple difficulty dimensions."""
    
    def __init__(self, config: MultiDimConfig = None):
        super().__init__()
        self.config = config or MultiDimConfig()
        # Optional post-hoc per-dimension thresholds.
        # Format: {"style": {"t_good": x, "t_bad": y}, ...}
        self.thresholds_by_dimension: Dict[str, Dict[str, float]] = {}
        
        self.encoder = AutoModelForSequenceClassification.from_pretrained(
            self.config.backbone,
            num_labels=self.config.num_thresholds,
            ignore_mismatched_sizes=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.backbone)
    
    def forward(self, input_ids, attention_mask):
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask).logits
    
    def predict_score(self, input_ids, attention_mask) -> torch.Tensor:
        logits = self.forward(input_ids, attention_mask)
        probs = torch.sigmoid(logits)
        return probs.sum(dim=1)

    def set_thresholds(self, thresholds_by_dimension: Dict[str, Dict[str, float]]) -> None:
        """Attach calibrated per-dimension thresholds to the model."""
        self.thresholds_by_dimension = thresholds_by_dimension or {}

    def decision_from_score(self, score: float, dimension: Optional[str] = None) -> Dict[str, Any]:
        """
        Convert a scalar score into GOOD/BAD/UNCERTAIN using per-dimension thresholds.

        If no dimension is provided (or thresholds are missing), falls back to the
        legacy global thresholds in config.
        """
        dim = str(dimension).lower() if dimension is not None else None
        t_good = self.config.default_good_threshold
        t_bad = self.config.default_bad_threshold

        if dim is not None:
            t_good = float(self.thresholds_by_dimension.get(dim, {}).get("t_good", t_good))
            t_bad = float(self.thresholds_by_dimension.get(dim, {}).get("t_bad", t_bad))

        # Use strict comparisons to avoid threshold tie edge-cases.
        if score > t_good:
            decision = "GOOD"
        elif score < t_bad:
            decision = "BAD"
        else:
            decision = "UNCERTAIN"

        return {
            "dimension": dim,
            "score": float(score),
            "decision": decision,
            "is_good": decision == "GOOD",
            "is_bad": decision == "BAD",
            "is_uncertain": decision == "UNCERTAIN",
            "t_good": float(t_good),
            "t_bad": float(t_bad),
        }
    
    def predict(self, prompt: str, response: str, dimension: Optional[str] = None) -> Dict[str, Any]:
        """
        Predict correctness score for a (prompt, response) pair.
        
        Uses PAIR tokenization (same as training) for consistent results.
        """
        device = next(self.parameters()).device

        prompt_fmt = _format_prompt_with_dimension(prompt, dimension, self.config)
        
        # FIX: Use PAIR tokenization, not string concatenation
        enc = self.tokenizer(
            prompt_fmt,                      # First segment (optionally with DIM prefix)
            response,                        # Second segment
            max_length=self.config.max_length, 
            padding='max_length', 
            truncation=True, 
            return_tensors='pt'
        )
        
        with torch.no_grad():
            logits = self.forward(enc['input_ids'].to(device), enc['attention_mask'].to(device))
            probs = torch.sigmoid(logits).squeeze()
            score = probs.sum().item()

        decision = self.decision_from_score(score, dimension=dimension)
        # Backwards compatibility: keep 'is_good'/'is_bad' keys, but now they
        # reflect per-dimension thresholds (when dimension is provided).
        return {
            "score": float(score),
            "decision": decision["decision"],
            "is_good": bool(decision["is_good"]),
            "is_bad": bool(decision["is_bad"]),
            "is_uncertain": bool(decision["is_uncertain"]),
            "t_good": float(decision["t_good"]),
            "t_bad": float(decision["t_bad"]),
            "dimension": decision["dimension"],
            "ordinal_probs": probs.tolist(),
        }
    
    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'config': {
                'backbone': self.config.backbone,
                'max_length': self.config.max_length,
                'num_thresholds': self.config.num_thresholds,
                'use_dimension_prefix': self.config.use_dimension_prefix,
                'dimension_prefix_template': self.config.dimension_prefix_template,
                'loss_weight_bad': self.config.loss_weight_bad,
                'loss_weight_mid': self.config.loss_weight_mid,
                'loss_weight_good': self.config.loss_weight_good,
                'per_dim_loss_multiplier': self.config.per_dim_loss_multiplier,
                'default_good_threshold': self.config.default_good_threshold,
                'default_bad_threshold': self.config.default_bad_threshold,
                'target_good_fp_rate': self.config.target_good_fp_rate,
                'target_bad_fp_rate': self.config.target_bad_fp_rate,
                'min_uncertainty_band': self.config.min_uncertainty_band,
                'per_dim_target_good_fp_rate': self.config.per_dim_target_good_fp_rate,
                'per_dim_target_bad_fp_rate': self.config.per_dim_target_bad_fp_rate,
            },
            'thresholds_by_dimension': self.thresholds_by_dimension,
            'state_dict': self.state_dict(),
        }, path)
        print(f"Saved to {path}")
    
    @classmethod
    def load(cls, path: Path) -> 'MultiDimPredictor':
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        config = MultiDimConfig(**checkpoint['config'])
        model = cls(config)
        model.load_state_dict(checkpoint['state_dict'])
        model.set_thresholds(checkpoint.get('thresholds_by_dimension', {}))
        return model


# =============================================================================
# Training
# =============================================================================

def load_all_dimensions(config: MultiDimConfig, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and combine all four dimensions with proper splitting."""
    from sklearn.model_selection import train_test_split
    
    # Load each dimension (all real data, no synthetic)
    style_df = load_helpsteer_dimension(config.helpsteer_max, seed)
    knowledge_df = load_nq_dimension(config.nq_max, seed)
    reasoning_df = load_bbh_dimension(config.bbh_max, seed)
    expert_df = load_gpqa_dimension(config.gpqa_max, seed)
    
    # Combine all four dimensions
    combined = pd.concat([style_df, knowledge_df, reasoning_df, expert_df], ignore_index=True)
    
    print(f"\n{'='*60}")
    print("COMBINED TRAINING DATA (All Real - No Synthetic)")
    print(f"{'='*60}")
    print(f"Total samples: {len(combined):,}")
    
    # Detailed class distribution by dimension
    print(f"\nClass distribution by dimension:")
    for dim in ['style', 'knowledge', 'reasoning', 'expert']:
        dim_df = combined[combined['dimension'] == dim]
        n_total = len(dim_df)
        if n_total == 0:
            print(f"  {dim.upper()}: 0 samples")
            continue
        
        # Count by correctness
        n_bad = len(dim_df[dim_df['correctness'] <= 1])  # 0-1 = BAD
        n_mid = len(dim_df[dim_df['correctness'] == 2])  # 2 = BORDERLINE
        n_good = len(dim_df[dim_df['correctness'] >= 3]) # 3-4 = GOOD
        
        print(f"  {dim.upper()}: {n_total:,} total | "
              f"BAD(0-1)={n_bad} ({100*n_bad/n_total:.0f}%) | "
              f"MID(2)={n_mid} ({100*n_mid/n_total:.0f}%) | "
              f"GOOD(3-4)={n_good} ({100*n_good/n_total:.0f}%)")
    
    # Create stratification key: dimension + correctness_class
    combined['correctness_bin'] = combined['correctness'].apply(
        lambda x: 'bad' if x <= 1 else ('mid' if x == 2 else 'good')
    )
    combined['strat_key'] = combined['dimension'] + '_' + combined['correctness_bin']
    
    # Check for classes with too few samples for stratification
    strat_counts = combined['strat_key'].value_counts()
    min_count = strat_counts.min()
    
    if min_count < 2:
        print(f"\n⚠️ Some strat groups have <2 samples, falling back to dimension-only stratification")
        stratify_col = combined['dimension']
    else:
        print(f"\n✓ Stratifying by dimension × correctness_class (min group size: {min_count})")
        stratify_col = combined['strat_key']
    
    # Split with stratification
    train_df, val_df = train_test_split(
        combined, test_size=config.val_split, 
        stratify=stratify_col, 
        random_state=seed
    )
    
    # Verify class balance in splits
    print(f"\nSplit: Train {len(train_df):,} | Val {len(val_df):,}")
    print(f"\nValidation set class balance:")
    for dim in ['style', 'knowledge', 'reasoning', 'expert']:
        val_dim = val_df[val_df['dimension'] == dim]
        if len(val_dim) == 0:
            continue
        n_bad = len(val_dim[val_dim['correctness'] <= 1])
        n_good = len(val_dim[val_dim['correctness'] >= 3])
        print(f"  {dim.upper()}: {len(val_dim)} samples | BAD={n_bad}, GOOD={n_good}")
    
    # Clean up temp columns
    train_df = train_df.drop(columns=['correctness_bin', 'strat_key'], errors='ignore')
    val_df = val_df.drop(columns=['correctness_bin', 'strat_key'], errors='ignore')
    
    return train_df, val_df


def evaluate_by_dimension(
    model: MultiDimPredictor,
    val_loader: DataLoader,
    device: torch.device
) -> Dict[str, Dict[str, float]]:
    """Evaluate performance by difficulty dimension."""
    model.eval()
    
    dim_preds = {'style': [], 'knowledge': [], 'reasoning': [], 'expert': []}
    dim_targets = {'style': [], 'knowledge': [], 'reasoning': [], 'expert': []}
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating", leave=False):
            scores = model.predict_score(
                batch['input_ids'].to(device),
                batch['attention_mask'].to(device)
            ).cpu()
            
            for i in range(len(scores)):
                dim = batch['dimension'][i]
                dim_preds[dim].append(scores[i].item())
                dim_targets[dim].append(batch['correctness'][i].item())
    
    results = {}
    for dim in ['style', 'knowledge', 'reasoning', 'expert']:
        preds = np.array(dim_preds[dim])
        targets = np.array(dim_targets[dim])
        
        if len(preds) == 0:
            results[dim.upper()] = {'n': 0, 'mae': float('nan'), 'acc': float('nan')}
            continue
        
        mae = np.mean(np.abs(preds - targets))
        acc = np.mean(np.round(preds) == targets)
        binary_acc = np.mean((preds >= 2.5) == (targets >= 3))
        corr = np.corrcoef(preds, targets)[0, 1] if len(set(targets)) > 1 else float('nan')
        
        results[dim.upper()] = {
            'n': len(preds),
            'mae': mae,
            'acc': acc,
            'binary_acc': binary_acc,
            'corr': corr,
        }
    
    # Overall
    all_preds = np.concatenate([np.array(dim_preds[d]) for d in dim_preds])
    all_targets = np.concatenate([np.array(dim_targets[d]) for d in dim_targets])
    results['OVERALL'] = {
        'n': len(all_preds),
        'mae': np.mean(np.abs(all_preds - all_targets)),
        'acc': np.mean(np.round(all_preds) == all_targets),
        'binary_acc': np.mean((all_preds >= 2.5) == (all_targets >= 3)),
        'corr': np.corrcoef(all_preds, all_targets)[0, 1] if len(set(all_targets)) > 1 else float('nan'),
    }
    
    return results


def _target_to_true_class(correctness: int) -> str:
    """Map 0-4 correctness label to {BAD, MID, GOOD}."""
    if correctness <= 1:
        return "BAD"
    if correctness == 2:
        return "MID"
    return "GOOD"


def _print_confusion_matrix(
    cm: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    title: str,
) -> None:
    """Pretty-print a confusion matrix with counts."""
    cm = np.asarray(cm, dtype=np.int64)
    row_w = max(len(r) for r in row_labels + ["True\\Pred"])
    col_w = max(9, max(len(c) for c in col_labels) + 2)

    print(f"\n{title}")
    print("-" * (row_w + col_w * len(col_labels)))
    header = "True\\Pred".ljust(row_w) + "".join(c.rjust(col_w) for c in col_labels)
    print(header)
    for i, r in enumerate(row_labels):
        line = r.ljust(row_w) + "".join(f"{cm[i, j]:d}".rjust(col_w) for j in range(len(col_labels)))
        print(line)
    print("-" * (row_w + col_w * len(col_labels)))


def evaluate_confusion_matrices(
    model: MultiDimPredictor,
    val_loader: DataLoader,
    device: torch.device,
    *,
    by_dimension: bool = True,
) -> Dict[str, Any]:
    """
    Compute confusion matrices after applying the model decision policy.

    True classes:  BAD / MID / GOOD     (from correctness 0-4)
    Pred classes:  BAD / UNCERTAIN / GOOD   (from calibrated thresholds)
    """
    model.eval()

    true_labels = ["BAD", "MID", "GOOD"]
    pred_labels = ["BAD", "UNCERTAIN", "GOOD"]
    true_to_idx = {k: i for i, k in enumerate(true_labels)}
    pred_to_idx = {k: i for i, k in enumerate(pred_labels)}

    overall_cm = np.zeros((len(true_labels), len(pred_labels)), dtype=np.int64)
    dim_cms: Dict[str, np.ndarray] = {d: overall_cm.copy() for d in ["style", "knowledge", "reasoning", "expert"]}
    dim_counts: Dict[str, int] = {d: 0 for d in ["style", "knowledge", "reasoning", "expert"]}

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Confusion matrix", leave=False):
            scores = model.predict_score(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
            ).detach().cpu().numpy()

            for i in range(len(scores)):
                dim = str(batch["dimension"][i]).lower()
                correctness = int(batch["correctness"][i].item())
                true_class = _target_to_true_class(correctness)

                decision = model.decision_from_score(float(scores[i]), dimension=dim)
                pred_class = str(decision["decision"])

                overall_cm[true_to_idx[true_class], pred_to_idx[pred_class]] += 1
                if by_dimension and dim in dim_cms:
                    dim_cms[dim][true_to_idx[true_class], pred_to_idx[pred_class]] += 1
                    dim_counts[dim] += 1

    # Extra diagnostics: coverage on non-MID labels
    def _coverage(cm: np.ndarray) -> Dict[str, float]:
        cm = np.asarray(cm, dtype=np.int64)
        bad_row = cm[true_to_idx["BAD"]]
        good_row = cm[true_to_idx["GOOD"]]
        denom = int(bad_row.sum() + good_row.sum())
        if denom == 0:
            return {"coverage_non_mid": float("nan")}
        decided = int((bad_row[pred_to_idx["BAD"]] + bad_row[pred_to_idx["GOOD"]]) +
                      (good_row[pred_to_idx["BAD"]] + good_row[pred_to_idx["GOOD"]]))
        return {"coverage_non_mid": decided / denom}

    return {
        "labels": {"true": true_labels, "pred": pred_labels},
        "overall": {"cm": overall_cm, **_coverage(overall_cm)},
        "by_dimension": {
            d.upper(): {"cm": cm, "n": int(dim_counts[d]), **_coverage(cm)}
            for d, cm in dim_cms.items()
        } if by_dimension else {},
    }


def summarize_fp_rates_from_confusion(
    cms: Dict[str, Any],
    config: Optional[MultiDimConfig] = None,
) -> Dict[str, Any]:
    """
    Summarize the key operational error rates:
      - FP(BAD->GOOD): predicted GOOD when true BAD
      - FN(GOOD->BAD): predicted BAD when true GOOD
      - Coverage(non-MID): fraction of (true BAD or true GOOD) that get a decided label
    """
    cfg = config or MultiDimConfig()
    true_labels = cms["labels"]["true"]
    pred_labels = cms["labels"]["pred"]
    t_bad_idx = true_labels.index("BAD")
    t_good_idx = true_labels.index("GOOD")
    p_bad_idx = pred_labels.index("BAD")
    p_good_idx = pred_labels.index("GOOD")
    p_unc_idx = pred_labels.index("UNCERTAIN")

    def _rates(cm: np.ndarray) -> Dict[str, float]:
        cm = np.asarray(cm, dtype=np.int64)
        bad_row = cm[t_bad_idx]
        good_row = cm[t_good_idx]
        bad_total = int(bad_row.sum())
        good_total = int(good_row.sum())
        fp_bad_to_good = float(bad_row[p_good_idx] / bad_total) if bad_total > 0 else float("nan")
        fn_good_to_bad = float(good_row[p_bad_idx] / good_total) if good_total > 0 else float("nan")
        abstain_bad = float(bad_row[p_unc_idx] / bad_total) if bad_total > 0 else float("nan")
        abstain_good = float(good_row[p_unc_idx] / good_total) if good_total > 0 else float("nan")
        return {
            "fp_bad_to_good": fp_bad_to_good,
            "fn_good_to_bad": fn_good_to_bad,
            "abstain_bad": abstain_bad,
            "abstain_good": abstain_good,
        }

    out: Dict[str, Any] = {"overall": _rates(cms["overall"]["cm"]), "by_dimension": {}}
    for dim_name, payload in cms.get("by_dimension", {}).items():
        dim_key = dim_name.lower()
        target_fp = float(cfg.per_dim_target_good_fp_rate.get(dim_key, cfg.target_good_fp_rate))
        out["by_dimension"][dim_name] = {**_rates(payload["cm"]), "target_fp_bad_to_good": target_fp}
    return out


def print_dimension_table(results: Dict[str, Dict[str, float]]):
    """Print evaluation by dimension."""
    print(f"\n{'='*75}")
    print("EVALUATION BY DIFFICULTY DIMENSION")
    print(f"{'='*75}")
    print(f"{'Dimension':<15} {'N':>8} {'MAE':>8} {'Acc':>8} {'BinAcc':>8} {'Corr':>8}")
    print(f"{'-'*75}")
    
    for dim, metrics in results.items():
        n = metrics['n']
        mae = f"{metrics['mae']:.3f}" if not np.isnan(metrics['mae']) else "N/A"
        acc = f"{metrics['acc']:.3f}" if not np.isnan(metrics['acc']) else "N/A"
        bin_acc = f"{metrics['binary_acc']:.3f}" if not np.isnan(metrics.get('binary_acc', float('nan'))) else "N/A"
        corr = f"{metrics['corr']:.3f}" if not np.isnan(metrics.get('corr', float('nan'))) else "N/A"
        print(f"{dim:<15} {n:>8} {mae:>8} {acc:>8} {bin_acc:>8} {corr:>8}")
    
    print(f"{'='*75}")


def print_labeled_sanity_examples(
    model: MultiDimPredictor,
    val_df: pd.DataFrame,
    *,
    n_per_dim: int = 5,
    seed: int = 42,
) -> None:
    """
    Print a small labeled sanity set from the *validation dataframe*.

    This avoids misleading "toy" prompts where we don't have ground-truth labels.
    """
    rng = np.random.default_rng(seed)
    model.eval()

    print("\n" + "=" * 80)
    print("LABELED SANITY CHECKS (sampled from validation set)")
    print("Legend: true_label in {BAD,MID,GOOD} vs pred_decision in {BAD,UNCERTAIN,GOOD}")
    print("=" * 80)

    for dim in ["style", "knowledge", "reasoning", "expert"]:
        sub = val_df[val_df["dimension"] == dim].copy()
        if len(sub) == 0:
            continue

        # Prefer extremes for sanity: BAD (<=1) and GOOD (>=3); include a couple MID if present.
        bad = sub[sub["correctness"] <= 1]
        good = sub[sub["correctness"] >= 3]
        mid = sub[sub["correctness"] == 2]

        # Sample from each bucket (as available)
        rows = []
        if len(bad) > 0:
            rows.append(bad.sample(n=min(len(bad), max(1, n_per_dim // 2)), random_state=seed))
        if len(good) > 0:
            rows.append(good.sample(n=min(len(good), max(1, n_per_dim // 2)), random_state=seed + 1))
        if len(mid) > 0 and n_per_dim >= 4:
            rows.append(mid.sample(n=min(len(mid), 1), random_state=seed + 2))

        if not rows:
            continue
        ex = pd.concat(rows, ignore_index=True)
        # Shuffle deterministically
        ex = ex.iloc[rng.permutation(len(ex))].reset_index(drop=True)

        mismatches = 0
        decided = 0
        total = len(ex)

        print(f"\n[{dim.upper()}] showing {total} labeled examples")
        for _, r in ex.iterrows():
            prompt = str(r["prompt"])
            response = str(r["response"])
            y = int(r["correctness"])
            true_label = _target_to_true_class(y)
            pred = model.predict(prompt, response, dimension=dim)
            pred_decision = pred["decision"]

            # Count mismatches only on extremes (BAD/GOOD), and only when decided.
            if true_label in {"BAD", "GOOD"} and pred_decision in {"BAD", "GOOD"}:
                decided += 1
                if true_label != pred_decision:
                    mismatches += 1

            print(f"  true={true_label:<4} pred={pred_decision:<9} score={pred['score']:.2f} "
                  f"(T_bad={pred['t_bad']:.2f}, T_good={pred['t_good']:.2f})")
            print(f"    Q: {prompt[:140].replace('\\n', ' ')}")
            print(f"    A: {response[:140].replace('\\n', ' ')}")

        if decided > 0:
            print(f"  decided_on_extremes={decided}/{total} | extreme_mismatch_rate={mismatches/decided:.1%}")
        else:
            print(f"  decided_on_extremes=0/{total} (all abstained on extremes)")


def train_multi_dim_predictor(
    config: MultiDimConfig = None,
    device: torch.device = None
) -> Tuple[MultiDimPredictor, DataLoader]:
    """Train the multi-dimensional predictor."""
    if config is None:
        config = MultiDimConfig()
    if device is None:
        device = get_device()
    
    print(f"\n{'='*60}")
    print("MULTI-DIMENSION PREDICTOR TRAINING")
    print("Four Dimensions: STYLE + KNOWLEDGE + REASONING + EXPERT")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Backbone: {config.backbone}")
    print(f"LR: {config.learning_rate}")
    print(f"{'='*60}")
    
    # Load all dimensions
    train_df, val_df = load_all_dimensions(config)
    
    # Create tokenizer and datasets
    tokenizer = AutoTokenizer.from_pretrained(config.backbone)
    train_ds = MultiDimDataset(train_df, tokenizer, config)
    val_ds = MultiDimDataset(val_df, tokenizer, config)
    
    # Show sampling distribution
    print("\nSampling groups (dimension × length × correctness):")
    group_counts = Counter(train_ds.df['sample_group'])
    for group, count in sorted(group_counts.items())[:10]:
        print(f"  {group}: {count:,}")
    if len(group_counts) > 10:
        print(f"  ... and {len(group_counts) - 10} more groups")
    
    # Data loaders
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size,
        sampler=train_ds.get_balanced_sampler(), num_workers=0
    )
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=0)
    
    # Model
    model = MultiDimPredictor(config)
    model.to(device)
    
    # Optimizer with warmup
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    total_steps = len(train_loader) * config.max_epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return 1.0
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Pairwise ranking loss helper
    def compute_pairwise_loss(batch, scores, margin=0.5):
        """
        Add ranking loss: score(correct) > score(wrong) + margin
        for samples with the same prompt.
        """
        prompts = batch['prompt']
        correctness = batch['correctness']
        
        # Group by prompt
        prompt_to_indices = {}
        for i, p in enumerate(prompts):
            if p not in prompt_to_indices:
                prompt_to_indices[p] = {'correct': [], 'wrong': []}
            if correctness[i] >= 3:
                prompt_to_indices[p]['correct'].append(i)
            elif correctness[i] <= 1:
                prompt_to_indices[p]['wrong'].append(i)
        
        # Compute margin ranking loss for each pair
        ranking_loss = 0.0
        n_pairs = 0
        
        for p, indices in prompt_to_indices.items():
            for c_idx in indices['correct']:
                for w_idx in indices['wrong']:
                    # We want: score[c_idx] > score[w_idx] + margin
                    # Loss = max(0, margin - (score_correct - score_wrong))
                    diff = scores[c_idx] - scores[w_idx]
                    pair_loss = F.relu(margin - diff)
                    ranking_loss += pair_loss
                    n_pairs += 1
        
        if n_pairs > 0:
            return ranking_loss / n_pairs
        return torch.tensor(0.0, device=scores.device)
    
    # Training loop with learning curve tracking
    best_mae = float('inf')
    best_state_dict = None
    learning_curve = []  # Track metrics per epoch
    ranking_weight = 0.3  # Weight for pairwise ranking loss
    
    for epoch in range(config.max_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_rank_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.max_epochs}")
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['ordinal_targets'].to(device)
            correctness = batch["correctness"].to(device)
            
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            
            # Primary loss: ordinal BCE (cost-sensitive to reduce BAD->GOOD slip-through)
            # Weight by label bucket + dimension.
            # - BAD (0-1) gets upweighted
            # - EXPERT/KNOWLEDGE get extra multiplier
            label_w = torch.where(
                correctness <= 1,
                torch.tensor(config.loss_weight_bad, device=device),
                torch.where(
                    correctness == 2,
                    torch.tensor(config.loss_weight_mid, device=device),
                    torch.tensor(config.loss_weight_good, device=device),
                ),
            ).to(torch.float32)

            dims = [str(d).lower() for d in batch["dimension"]]
            dim_w = torch.tensor(
                [float(config.per_dim_loss_multiplier.get(d, 1.0)) for d in dims],
                device=device,
                dtype=torch.float32,
            )
            sample_w = (label_w * dim_w).clamp_min(1e-6)

            bce_per = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
            bce_loss = (bce_per * sample_w.unsqueeze(1)).mean()
            
            # Secondary loss: pairwise ranking (score_correct > score_wrong)
            scores = model.predict_score(input_ids, attention_mask)
            rank_loss = compute_pairwise_loss(batch, scores)
            
            # Combined loss
            loss = bce_loss + ranking_weight * rank_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            
            epoch_loss += bce_loss.item()
            epoch_rank_loss += rank_loss.item()
            pbar.set_postfix({'bce': f'{bce_loss.item():.4f}', 'rank': f'{rank_loss.item():.4f}'})
        
        avg_loss = epoch_loss / len(train_loader)
        avg_rank_loss = epoch_rank_loss / len(train_loader)
        
        # Evaluate by dimension
        results = evaluate_by_dimension(model, val_loader, device)
        
        # Store learning curve data
        learning_curve.append({
            'epoch': epoch + 1,
            'train_loss': avg_loss,
            'overall_mae': results['OVERALL']['mae'],
            'overall_bin_acc': results['OVERALL']['binary_acc'],
            'style_bin_acc': results.get('STYLE', {}).get('binary_acc', float('nan')),
            'knowledge_bin_acc': results.get('KNOWLEDGE', {}).get('binary_acc', float('nan')),
            'reasoning_bin_acc': results.get('REASONING', {}).get('binary_acc', float('nan')),
            'expert_bin_acc': results.get('EXPERT', {}).get('binary_acc', float('nan')),
        })
        
        print(f"\nEpoch {epoch+1}/{config.max_epochs}")
        print(f"  BCE Loss: {avg_loss:.4f} | Ranking Loss: {avg_rank_loss:.4f}")
        print_dimension_table(results)
        
        # Save best
        if results['OVERALL']['mae'] < best_mae:
            best_mae = results['OVERALL']['mae']
            best_state_dict = copy.deepcopy(model.state_dict())
            print(f"  ✓ New best model (MAE={best_mae:.3f})")
    
    # Print learning curve summary
    print(f"\n{'='*90}")
    print("LEARNING CURVE SUMMARY")
    print(f"{'='*90}")
    print(f"{'Epoch':<7} {'Loss':<10} {'MAE':<8} {'Overall':<10} {'Style':<10} {'Knowledge':<10} {'Reasoning':<10} {'Expert':<10}")
    print(f"{'':<7} {'':<10} {'':<8} {'BinAcc':<10} {'BinAcc':<10} {'BinAcc':<10} {'BinAcc':<10} {'BinAcc':<10}")
    print(f"{'-'*90}")
    for lc in learning_curve:
        print(f"{lc['epoch']:<7} {lc['train_loss']:<10.4f} {lc['overall_mae']:<8.3f} "
              f"{lc['overall_bin_acc']:<10.3f} "
              f"{lc['style_bin_acc']:<10.3f} "
              f"{lc['knowledge_bin_acc']:<10.3f} "
              f"{lc['reasoning_bin_acc']:<10.3f} "
              f"{lc['expert_bin_acc']:<10.3f}")
    print(f"{'='*90}")
    
    print(f"\n{'='*60}")
    print(f"Training complete! Best Val MAE: {best_mae:.3f}")
    print(f"{'='*60}")

    # Restore best weights, calibrate per-dimension thresholds for low FP,
    # then save a single final artifact.
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    thresholds_by_dimension = calibrate_thresholds_from_val(
        model=model,
        val_loader=val_loader,
        device=device,
        config=config,
    )
    model.set_thresholds(thresholds_by_dimension)
    model.save(config.checkpoint_dir / "best_multi_dim_predictor.pt")

    print("\nCalibrated per-dimension thresholds (GOOD/BAD/UNCERTAIN):")
    for dim in ["style", "knowledge", "reasoning", "expert"]:
        t = thresholds_by_dimension.get(dim, {})
        if not t:
            continue
        print(
            f"  {dim.upper():<9} "
            f"T_bad={t['t_bad']:.2f} | T_good={t['t_good']:.2f} "
            f"(target FP BAD->GOOD={t['target_good_fp_rate']:.2%})"
        )

    # Confusion matrix on validation set using calibrated decision policy.
    cms = evaluate_confusion_matrices(model, val_loader, device, by_dimension=True)
    _print_confusion_matrix(
        cms["overall"]["cm"],
        row_labels=cms["labels"]["true"],
        col_labels=cms["labels"]["pred"],
        title=f"CONFUSION MATRIX (OVERALL) | coverage_non_mid={cms['overall']['coverage_non_mid']:.3f}",
    )
    for dim_name, payload in cms["by_dimension"].items():
        _print_confusion_matrix(
            payload["cm"],
            row_labels=cms["labels"]["true"],
            col_labels=cms["labels"]["pred"],
            title=f"CONFUSION MATRIX ({dim_name}) | n={payload['n']} | coverage_non_mid={payload['coverage_non_mid']:.3f}",
        )

    rates = summarize_fp_rates_from_confusion(cms, config=config)
    print("\nKey error rates (post-calibration):")
    for dim_name, r in rates["by_dimension"].items():
        fp = r["fp_bad_to_good"]
        tgt = r["target_fp_bad_to_good"]
        print(
            f"  {dim_name:<9} FP(BAD->GOOD)={fp:.3%} (target={tgt:.1%}) | "
            f"FN(GOOD->BAD)={r['fn_good_to_bad']:.3%} | "
            f"abstain_BAD={r['abstain_bad']:.3%} | abstain_GOOD={r['abstain_good']:.3%}"
        )

    # Labeled sanity examples from the validation set (avoids misleading toy prompts).
    print_labeled_sanity_examples(model, val_df, n_per_dim=5, seed=42)
    
    return model, val_loader


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Multi-Dimension Predictor")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    # Use all available data by default (high limits)
    parser.add_argument("--helpsteer-max", type=int, default=50000)  # All HelpSteer2
    parser.add_argument("--nq-max", type=int, default=50000)         # Large NQ subset
    parser.add_argument("--bbh-max", type=int, default=10000)        # All BBH tasks
    parser.add_argument("--gpqa-max", type=int, default=1000)        # All GPQA
    
    args = parser.parse_args()
    
    config = MultiDimConfig(
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        helpsteer_max=args.helpsteer_max,
        nq_max=args.nq_max,
        bbh_max=args.bbh_max,
        gpqa_max=args.gpqa_max,
    )
    
    model, val_loader = train_multi_dim_predictor(config)
    
    # Optional: ad-hoc prompts can be added here for *diagnostic only*.
    # IMPORTANT: Unless you have ground-truth labels, don't interpret these as
    # accuracy/correctness failures; use the labeled validation sanity checks above.
