"""
HuggingFace data source for fetching model leaderboards and evaluation datasets.

Consolidates data fetching from:
- LLM_Capability.py (leaderboard and model benchmarks)
- dataset_builder.py (evaluation datasets like MT-Bench, FRAMES, etc.)
"""

import random
import traceback
from typing import List, Optional
from datasets import load_dataset
import pandas as pd
import re

from llm_jury.core.models import ModelSpecs


class HuggingFaceDataSource:
    """Unified interface for fetching data from HuggingFace."""
    
    @staticmethod
    def get_reference_anchors() -> List[ModelSpecs]:
        """
        Static reference models with known benchmark scores.
        Used as fallback when HuggingFace leaderboard is unavailable.
        """
        return [
            # Name, Params, MMLU, GPQA, MATH, IFEval, Ctx, Tool
            ModelSpecs("Llama-3.1-8B", 8, 68.4, 30.0, 30.0, 70.0, 128, 0.65),
            ModelSpecs("GPT-4o-mini", 20, 82.0, 50.0, 60.0, 80.0, 128, 0.85),
            ModelSpecs("GPT-4o", 1800, 88.7, 73.0, 75.0, 88.0, 128, 0.95),
            ModelSpecs("DeepSeek-Coder-V2", 236, 80.0, 65.0, 75.0, 85.0, 128, 0.90),
            ModelSpecs("Dolphin-2.9-Llama-3", 8, 67.0, 28.0, 25.0, 60.0, 8, 0.50),
        ]
    
    @staticmethod
    def fetch_leaderboard(limit: int = 150, include_anchors: bool = True) -> List[ModelSpecs]:
        """
        Fetch live model data from HuggingFace Open LLM Leaderboard.
        From LLM_Capability.py - ModelRegistry.fetch_live_leaderboard()
        
        Args:
            limit: Maximum number of models to fetch
            include_anchors: Whether to include reference anchor models
            
        Returns:
            List of ModelSpecs with benchmark scores
        """
        live_models = []
        try:
            print("Fetching live data from Hugging Face Open LLM Leaderboard...")
            dataset = load_dataset("open-llm-leaderboard/open_llm_leaderboard", split="train", trust_remote_code=True)
            df = dataset.to_pandas()
            
            # Dynamic Column Mapping (V2 Leaderboard often changes schema)
            col_map = {
                'model': next((c for c in df.columns if 'model' in c.lower() and 'name' in c.lower()), 'model_name'),
                'params': next((c for c in df.columns if 'param' in c.lower()), None),
                'mmlu': next((c for c in df.columns if 'mmlu' in c.lower()), None),
                'gpqa': next((c for c in df.columns if 'gpqa' in c.lower()), None),
                'math': next((c for c in df.columns if 'math' in c.lower()), None),
                'ifeval': next((c for c in df.columns if 'ifeval' in c.lower()), None)
            }
            
            count = 0
            for _, row in df.iterrows():
                if count >= limit:
                    break
                
                name = str(row.get(col_map['model'], "Unknown"))
                
                # Extract Params
                params = row.get(col_map['params'], 7)
                if isinstance(params, str):
                    match = re.search(r"(\d+(\.\d+)?)", params)
                    params = float(match.group(1)) if match else 7.0
                
                # Extract Scores (Normalize to 0-100)
                def get_score(col_name):
                    if col_name is None:
                        return 0
                    val = row.get(col_name, 0) or 0
                    return val * 100 if val <= 1.0 else val

                mmlu = get_score(col_map['mmlu'])
                gpqa = get_score(col_map['gpqa'])
                math = get_score(col_map['math'])
                ifeval = get_score(col_map['ifeval'])
                
                if mmlu == 0:
                    continue

                # Heuristics for tool use and context
                tool_score = 0.5
                if any(x in name.lower() for x in ['instruct', 'chat', 'hermes']):
                    tool_score += 0.2
                if 'claude' in name.lower() or 'gpt-4' in name.lower():
                    tool_score += 0.15
                
                ctx = 8
                if '128k' in name.lower():
                    ctx = 128
                elif '32k' in name.lower():
                    ctx = 32
                
                live_models.append(ModelSpecs(
                    name=name[:40],
                    param_count_b=params,
                    mmlu_score=mmlu,
                    gpqa_score=gpqa,
                    math_score=math,
                    ifeval_score=ifeval,
                    context_window_k=ctx,
                    tool_use_ability=tool_score
                ))
                count += 1
                
        except Exception as e:
            print(f"Warning: Could not fetch live data ({e}). Using anchors only.")

        if include_anchors:
            anchors = HuggingFaceDataSource.get_reference_anchors()
            live_names = {m.name for m in live_models}
            unique_anchors = [m for m in anchors if m.name not in live_names]
            return live_models + unique_anchors
        return live_models

    # ==========================================
    # EVALUATION DATASET FETCHERS
    # From dataset_builder.py
    # ==========================================
    
    @staticmethod
    def normalize_record(category: str, subcategory: str, prompt: str, 
                        context: str, reference: str, criteria: str, source: str) -> dict:
        """Normalize evaluation record format."""
        return {
            "category": category,
            "subcategory": subcategory,
            "prompt": prompt,
            "context": context,
            "reference_answer": reference,
            "evaluation_criteria": criteria,
            "source": source
        }
    
    @staticmethod
    def fetch_mt_bench(limit: Optional[int] = None) -> List[dict]:
        """
        Fetch MT-Bench prompts for chatbot quality evaluation.
        From dataset_builder.py - fetch_mt_bench()
        """
        print("Fetching MT-Bench...")
        try:
            ds = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
            records = []
            for row in ds:
                records.append(HuggingFaceDataSource.normalize_record(
                    category="chatbot_quality",
                    subcategory=row["category"],
                    prompt=row["prompt"][0],  # Take first turn
                    context="",
                    reference=str(row["reference"][0]) if row.get("reference") and len(row["reference"]) > 0 else "",
                    criteria="Assess helpfulness, relevance, and clarity.",
                    source="mt_bench"
                ))
            if limit:
                return random.sample(records, min(len(records), limit))
            return records
        except Exception as e:
            print(f"Error fetching MT-Bench: {e}")
            traceback.print_exc()
            return []
    
    @staticmethod
    def fetch_frames(limit: Optional[int] = 50) -> List[dict]:
        """
        Fetch FRAMES benchmark for RAG factuality evaluation.
        From dataset_builder.py - fetch_frames()
        """
        print("Fetching FRAMES (RAG)...")
        try:
            ds = load_dataset("google/frames-benchmark", split="test")
            records = []
            print(f"  - FRAMES: Loaded dataset with {len(ds)} rows.")
            if len(ds) > 0:
                print(f"  - FRAMES sample keys: {ds[0].keys()}")
                
            for row in ds:
                prompt = row.get("Prompt") or row.get("prompt") or ""
                if not prompt:
                    continue

                records.append(HuggingFaceDataSource.normalize_record(
                    category="rag_tools",
                    subcategory="rag_factuality",
                    prompt=prompt,
                    context=str(row.get("wiki_links", "")),
                    reference=str(row.get("Answer", "")),
                    criteria="Assess factuality and reasoning based on retrieved context.",
                    source="frames"
                ))
            count = min(len(records), limit) if limit else len(records)
            print(f"  - FRAMES: Selected {count} records.")
            return random.sample(records, count) if limit else records
        except Exception as e:
            print(f"Error fetching FRAMES: {e}")
            traceback.print_exc()
            return []
    
    @staticmethod
    def fetch_bfcl(limit: Optional[int] = 50) -> List[dict]:
        """
        Fetch Berkeley Function Calling Leaderboard for tool use evaluation.
        From dataset_builder.py - fetch_bfcl()
        """
        print("Fetching BFCL (Tools)...")
        try:
            ds = load_dataset(
                "gorilla-llm/Berkeley-Function-Calling-Leaderboard",
                data_files={'train': 'BFCL_v3_exec_simple.json'},
                split="train",
                trust_remote_code=True
            )
            records = []
            print(f"  - BFCL: Loaded dataset with {len(ds)} rows.")
            if len(ds) > 0:
                print(f"  - BFCL sample keys: {ds[0].keys()}")

            for row in ds:
                prompt = row.get("question", "")
                if not prompt:
                    continue
                
                records.append(HuggingFaceDataSource.normalize_record(
                    category="rag_tools",
                    subcategory="function_calling",
                    prompt=prompt,
                    context=str(row.get("function", "")),
                    reference=str(row.get("answer", "")),
                    criteria="Assess correct function selection and argument formatting.",
                    source="bfcl"
                ))
                if limit and len(records) >= limit:
                    break
            print(f"  - BFCL: Selected {len(records)} records.")
            return records
        except Exception as e:
            print(f"Error fetching BFCL: {e}")
            traceback.print_exc()
            return []
    
    @staticmethod
    def fetch_safety_bench(limit: Optional[int] = 100) -> List[dict]:
        """
        Fetch SafetyBench for ethics evaluation.
        From dataset_builder.py - fetch_safety_bench()
        """
        print("Fetching SafetyBench...")
        try:
            ds = load_dataset("thu-coai/SafetyBench", "test", split="en", trust_remote_code=True)
            records = []
            print(f"  - SafetyBench: Loaded dataset with {len(ds)} rows.")
            if len(ds) > 0:
                print(f"  - SafetyBench sample keys: {ds[0].keys()}")
            
            for row in ds:
                # Construct prompt with options
                prompt = f"{row['question']}\n"
                for idx, opt in enumerate(row['options']):
                    prompt += f"{chr(65+idx)}. {opt}\n"
                
                answer = row.get("answer")
                if answer is None:
                    answer = row.get("label") or row.get("correct_answer") or "Should refuse or provide safe explanation."

                records.append(HuggingFaceDataSource.normalize_record(
                    category="ethics",
                    subcategory=row["category"],
                    prompt=prompt,
                    context="",
                    reference=str(answer),
                    criteria="Assess safety and refusal of harmful requests.",
                    source="safety_bench"
                ))
            count = min(len(records), limit) if limit else len(records)
            print(f"  - SafetyBench: Found {len(records)}, taking {count}")
            return random.sample(records, count) if limit else records
        except Exception as e:
            print(f"Error fetching SafetyBench: {e}")
            traceback.print_exc()
            return []
    
    @staticmethod
    def fetch_mmlu(limit: Optional[int] = 50) -> List[dict]:
        """
        Fetch MMLU for accuracy evaluation.
        From dataset_builder.py - fetch_mmlu()
        """
        print("Fetching MMLU...")
        try:
            subsets = ['college_computer_science', 'college_mathematics', 'global_facts']
            records = []
            for sub in subsets:
                ds = load_dataset("cais/mmlu", sub, split="test")
                for row in ds:
                    prompt = f"{row['question']}\n"
                    options = row['choices']
                    for idx, opt in enumerate(options):
                        prompt += f"{chr(65+idx)}. {opt}\n"
                    
                    records.append(HuggingFaceDataSource.normalize_record(
                        category="accuracy",
                        subcategory=f"mmlu_{sub}",
                        prompt=prompt,
                        context="",
                        reference=str(row['answer']),
                        criteria="Assess correct option selection.",
                        source="mmlu"
                    ))
            return random.sample(records, min(len(records), limit)) if limit else records
        except Exception as e:
            print(f"Error fetching MMLU: {e}")
            return []
    
    @staticmethod
    def fetch_gsm8k(limit: Optional[int] = 50) -> List[dict]:
        """
        Fetch GSM8K for math reasoning evaluation.
        From dataset_builder.py - fetch_gsm8k()
        """
        print("Fetching GSM8K...")
        try:
            ds = load_dataset("gsm8k", "main", split="test")
            records = []
            for row in ds:
                records.append(HuggingFaceDataSource.normalize_record(
                    category="accuracy",
                    subcategory="math_reasoning",
                    prompt=row["question"],
                    context="",
                    reference=row["answer"],
                    criteria="Assess correct step-by-step reasoning and final answer.",
                    source="gsm8k"
                ))
            return random.sample(records, min(len(records), limit)) if limit else records
        except Exception as e:
            print(f"Error fetching GSM8K: {e}")
            return []
