#!/usr/bin/env python3
"""
Fetch GAIA (General AI Assistants) benchmark data.

GAIA is a benchmark for real-world agentic tasks that require tool use,
reasoning, and multi-step problem solving. Questions include tasks like
analyzing files, searching the web, and multi-hop reasoning.

Dataset: GAIA (validation set)
Source: HuggingFace gaia-benchmark/GAIA
Evaluation: Exact match on final answer (string or number)
Scoring: Simple exact match (case-insensitive)

Usage:
    python fetch_gaia.py --split validation --output gaia_validation.json

Requirements:
    pip install datasets huggingface_hub
"""

import json
import argparse
import os
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime


def fetch_gaia(
    split: str = "validation",
    level: Optional[str] = None,
    n_samples: Optional[int] = None,
    seed: int = 42,
    download_files: bool = False,
    files_dir: str = "gaia_files"
) -> List[Dict]:
    """
    Fetch GAIA dataset from HuggingFace.
    
    Args:
        split: Dataset split ("validation" or "test")
        level: Filter by level (1, 2, 3) or None for all
        n_samples: Number of samples to fetch (None = all)
        seed: Random seed for sampling
        download_files: Whether to download associated files
        files_dir: Directory to save downloaded files
        
    Returns:
        List of GAIA problems
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "datasets library not found. Install with: pip install datasets"
        )
    
    print(f"Fetching GAIA ({split} split) from HuggingFace...")
    
    # GAIA dataset requires authentication
    hf_token = (os.getenv("HF_TOKEN") or 
                os.getenv("HUGGINGFACE_TOKEN") or 
                os.getenv("HUGGINGFACE_API_KEY") or
                os.getenv("HF_API_KEY"))
    
    try:
        # Load dataset
        if hf_token:
            dataset = load_dataset(
                "gaia-benchmark/GAIA",
                "2023_all",
                split=split,
                token=hf_token
            )
        else:
            dataset = load_dataset(
                "gaia-benchmark/GAIA",
                "2023_all",
                split=split
            )
        
        print(f"✓ Loaded {len(dataset)} problems from GAIA ({split})")
        
    except Exception as e:
        error_msg = str(e)
        if "gated" in error_msg.lower() or "authentication" in error_msg.lower():
            print(f"\n❌ GAIA is a gated dataset. To access it:")
            print(f"   1. Visit: https://huggingface.co/datasets/gaia-benchmark/GAIA")
            print(f"   2. Accept the terms of use")
            print(f"   3. Set HF_TOKEN=your_token or run: huggingface-cli login")
            raise
        else:
            print(f"❌ Error loading dataset: {e}")
            raise
    
    # Convert to list and process
    problems = []
    for idx, item in enumerate(dataset):
        problem = {
            "task_id": item.get("task_id", f"gaia_{split}_{idx}"),
            "question": item.get("Question", item.get("question", "")),
            "level": item.get("Level", item.get("level", 0)),
            "final_answer": item.get("Final answer", item.get("final_answer", "")),
            "file_name": item.get("file_name", ""),
            "file_path": item.get("file_path", ""),
            "annotator_metadata": item.get("Annotator Metadata", {}),
            "metadata": {
                "steps": item.get("Steps", item.get("steps", "")),
                "number_of_steps": item.get("Number of steps", item.get("number_of_steps", 0)),
                "tools": item.get("Tools", item.get("tools", [])),
            }
        }
        
        # Filter by level if specified
        if level is not None and problem["level"] != int(level):
            continue
        
        problems.append(problem)
    
    print(f"✓ Processed {len(problems)} problems")
    
    # Download associated files if requested
    if download_files and problems:
        print(f"\nDownloading associated files to {files_dir}/...")
        files_path = Path(files_dir)
        files_path.mkdir(exist_ok=True)
        
        files_downloaded = 0
        for problem in problems:
            if problem.get("file_name"):
                # Files would need to be extracted from dataset
                # This is a placeholder - actual implementation depends on dataset structure
                print(f"  Note: File downloading not yet implemented for: {problem['file_name']}")
        
        if files_downloaded > 0:
            print(f"✓ Downloaded {files_downloaded} files")
    
    # Sample if requested
    if n_samples and len(problems) > n_samples:
        import random
        random.seed(seed)
        problems = random.sample(problems, n_samples)
        print(f"✓ Sampled {n_samples} problems (seed={seed})")
    
    return problems


def create_prompt(problem: Dict, prompt_style: str = "simple") -> str:
    """
    Create a prompt from a GAIA problem.
    
    Args:
        problem: Problem dictionary
        prompt_style: "simple" or "detailed"
        
    Returns:
        Formatted prompt string
    """
    if prompt_style == "simple":
        prompt = problem["question"]
        
        if problem.get("file_name"):
            prompt += f"\n\n[Note: This question references a file: {problem['file_name']}]"
    
    elif prompt_style == "detailed":
        prompt = f"""You are a helpful AI assistant that can use tools and reason through complex problems.

Question: {problem['question']}

"""
        if problem.get("file_name"):
            prompt += f"Associated File: {problem['file_name']}\n"
        
        prompt += """
Please provide your final answer. For questions that ask for specific information:
- If it's a city, country, or place name, provide just the name
- If it's a number, provide just the number
- If it's a date, provide in YYYY-MM-DD format
- Be concise and specific

Your answer:"""
    
    else:
        raise ValueError(f"Unknown prompt_style: {prompt_style}")
    
    return prompt


def analyze_dataset(problems: List[Dict]) -> Dict:
    """Analyze the fetched dataset."""
    stats = {
        "total_problems": len(problems),
        "levels": {},
        "with_files": sum(1 for p in problems if p.get("file_name")),
        "avg_steps": 0,
    }
    
    total_steps = 0
    for p in problems:
        level = p.get("level", "Unknown")
        stats["levels"][level] = stats["levels"].get(level, 0) + 1
        
        steps = p.get("metadata", {}).get("number_of_steps", 0)
        total_steps += steps
    
    if problems:
        stats["avg_steps"] = total_steps / len(problems)
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Fetch GAIA benchmark data from HuggingFace"
    )
    parser.add_argument(
        "--split", type=str, default="validation",
        choices=["validation", "test"],
        help="Dataset split to fetch"
    )
    parser.add_argument(
        "--level", type=int, default=None,
        choices=[1, 2, 3],
        help="Filter by difficulty level (1=easy, 2=medium, 3=hard)"
    )
    parser.add_argument(
        "--n-samples", type=int, default=None,
        help="Number of problems to fetch (default: all)"
    )
    parser.add_argument(
        "--output", type=str, default="gaia_validation.json",
        help="Output JSON file"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for sampling"
    )
    parser.add_argument(
        "--prompt-style", type=str, default="simple",
        choices=["simple", "detailed"],
        help="Prompt formatting style"
    )
    parser.add_argument(
        "--download-files", action="store_true",
        help="Download associated files (if any)"
    )
    parser.add_argument(
        "--files-dir", type=str, default="gaia_files",
        help="Directory to save downloaded files"
    )
    
    args = parser.parse_args()
    
    try:
        # Check for HuggingFace token
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        if not hf_token:
            print("⚠️  Warning: HF_TOKEN not set. You may need to authenticate.")
            print("   Set with: export HF_TOKEN=your_token")
            print("   Or run: huggingface-cli login\n")
        
        # Fetch problems
        problems = fetch_gaia(
            split=args.split,
            level=args.level,
            n_samples=args.n_samples,
            seed=args.seed,
            download_files=args.download_files,
            files_dir=args.files_dir
        )
        
        if not problems:
            print("❌ No problems fetched")
            return 1
        
        # Add formatted prompts
        for problem in problems:
            problem["prompt"] = create_prompt(problem, args.prompt_style)
        
        # Analyze dataset
        stats = analyze_dataset(problems)
        
        print("\n" + "="*60)
        print("Dataset Statistics")
        print("="*60)
        print(f"Total Problems:        {stats['total_problems']}")
        print(f"Problems with Files:   {stats['with_files']}")
        print(f"Avg Steps per Problem: {stats['avg_steps']:.1f}")
        
        if stats['levels']:
            print(f"\nLevel Distribution:")
            for level in sorted(stats['levels'].keys()):
                count = stats['levels'][level]
                pct = (count / stats['total_problems']) * 100
                difficulty = {1: "Easy", 2: "Medium", 3: "Hard"}.get(level, "Unknown")
                print(f"  Level {level} ({difficulty}): {count} ({pct:.1f}%)")
        
        # Save to file
        output_data = {
            "metadata": {
                "dataset": "GAIA (General AI Assistants)",
                "source": "gaia-benchmark/GAIA",
                "split": args.split,
                "fetch_date": datetime.now().isoformat(),
                "n_problems": len(problems),
                "level_filter": args.level,
                "seed": args.seed,
                "prompt_style": args.prompt_style,
            },
            "statistics": stats,
            "problems": problems
        }
        
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n✓ Saved {len(problems)} problems to {output_path}")
        
        # Show sample problem
        if problems:
            print("\n" + "="*60)
            print("Sample Problem")
            print("="*60)
            sample = problems[0]
            print(f"Task ID: {sample['task_id']}")
            print(f"Level: {sample['level']}")
            print(f"Question: {sample['question'][:200]}...")
            print(f"Final Answer: {sample['final_answer']}")
            if sample.get("file_name"):
                print(f"File: {sample['file_name']}")
            print(f"\nPrompt Preview:")
            print(sample['prompt'][:300] + "..." if len(sample['prompt']) > 300 else sample['prompt'])
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
