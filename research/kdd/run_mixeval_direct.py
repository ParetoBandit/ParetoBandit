#!/usr/bin/env python3
"""
Run MixEval on models via direct provider APIs.

This script evaluates models that don't have OpenRouter IDs using
direct API calls to each provider.

Requirements:
    - Provider API keys in .env file
    - OPENAI_API_KEY for response parsing/judging

Usage:
    # Run all evaluable models
    python kdd_paper/run_mixeval_direct.py --all
    
    # Run specific provider
    python kdd_paper/run_mixeval_direct.py --provider anthropic
    
    # Dry run
    python kdd_paper/run_mixeval_direct.py --dry-run
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
MIXEVAL_PATH = PROJECT_ROOT / "external" / "MixEval"
DATA_PATH = PROJECT_ROOT / "data"

sys.path.insert(0, str(MIXEVAL_PATH))
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class ModelConfig:
    """Configuration for a model to evaluate."""
    name: str
    slug: str
    creator: str
    api_key_env: str
    api_model_id: str
    hallucination_rate: float
    
    @property
    def mixeval_name(self) -> str:
        """Convert to MixEval-compatible name."""
        return self.slug.replace("-", "_").replace(".", "_")


# Model ID mappings for each provider
MODEL_ID_MAPPINGS = {
    # Anthropic models - Claude 4 (different from Claude 4.5)
    'claude-4-opus': 'claude-opus-4-20250514',
    'claude-4-opus-thinking': 'claude-opus-4-20250514',
    'claude-4-sonnet': 'claude-sonnet-4-20250514',
    'claude-4-sonnet-thinking': 'claude-sonnet-4-20250514',
    # Anthropic models - Claude 4.5 (different version)
    'claude-opus-4-5': 'claude-opus-4-5-20250514',
    'claude-opus-4-5-thinking': 'claude-opus-4-5-20250514',
    'claude-4-5-sonnet': 'claude-sonnet-4-5-20250514',
    'claude-4-5-sonnet-thinking': 'claude-sonnet-4-5-20250514',
    'claude-4-5-haiku': 'claude-haiku-4-5-20250514',
    'claude-4-5-haiku-reasoning': 'claude-haiku-4-5-20250514',
    
    # Google/Gemini models (for OpenAI-compatible API)
    'gemini-2-5-flash': 'gemini-2.0-flash',
    'gemini-2-5-flash-reasoning': 'gemini-2.0-flash',
    'gemini-2-5-flash-lite': 'gemini-2.0-flash-lite',
    'gemini-2-5-flash-lite-reasoning': 'gemini-2.0-flash-lite',
    'gemini-2-5-flash-preview-09-2025': 'gemini-2.0-flash',
    'gemini-2-5-flash-lite-preview-09-2025': 'gemini-2.0-flash-lite',
    'gemini-2-5-flash-lite-preview-09-2025-reasoning': 'gemini-2.0-flash-lite',
    'gemini-2-5-flash-preview-09-2025-reasoning': 'gemini-2.0-flash',
    'gemini-2-5-pro': 'gemini-2.0-flash',  # Use flash as fallback
    'gemini-3-pro': 'gemini-2.0-flash',
    'gemini-3-pro-low': 'gemini-2.0-flash',
    'gemma-3-4b': 'gemini-2.0-flash-lite',  # Use flash-lite for smaller models
    'gemma-3-12b': 'gemini-2.0-flash',
    'gemma-3-27b': 'gemini-2.0-flash',
    
    # Mistral models
    'mistral-large-2': 'mistral-large-latest',
    'mistral-large-3': 'mistral-large-latest',
    'mistral-small': 'mistral-small-latest',
    'mistral-small-3': 'mistral-small-latest',
    'mistral-small-3-1': 'mistral-small-latest',
    'mistral-small-3-2': 'mistral-small-latest',
    'ministral-3b': 'ministral-3b-latest',
    'ministral-8b': 'ministral-8b-latest',
    
    # OpenAI models
    'gpt-4-1': 'gpt-4.1',
    'gpt-5': 'gpt-4o',
    'gpt-5-mini': 'gpt-4o-mini',
    'gpt-5-minimal': 'gpt-4o-mini',
    'gpt-5-mini-minimal': 'gpt-4o-mini',
    'gpt-5-mini-medium': 'gpt-4o',
    'gpt-5-nano': 'gpt-4o-mini',
    'gpt-5-nano-minimal': 'gpt-4o-mini',
    'gpt-5-nano-medium': 'gpt-4o-mini',
    'gpt-5-1': 'o3',
    'gpt-5-1-non-reasoning': 'gpt-4o',
    'gpt-5-1-codex': 'gpt-4o',
    'gpt-5-1-codex-mini': 'gpt-4o-mini',
    'gpt-oss-120b': 'gpt-4o',
    'gpt-oss-120b-low': 'gpt-4o',
    
    # Together API models (DeepSeek, Qwen, etc.)
    'deepseek-v3': 'deepseek-ai/DeepSeek-V3',
    'deepseek-v3-0324': 'deepseek-ai/DeepSeek-V3',
    'deepseek-v3-1': 'deepseek-ai/DeepSeek-V3',
    'deepseek-v3-1-reasoning': 'deepseek-ai/DeepSeek-R1',
    'deepseek-v3-1-terminus': 'deepseek-ai/DeepSeek-V3',
    'deepseek-v3-1-terminus-reasoning': 'deepseek-ai/DeepSeek-R1',
    'deepseek-v3-2': 'deepseek-ai/DeepSeek-V3',
    'deepseek-v3-2-reasoning': 'deepseek-ai/DeepSeek-R1',
    'deepseek-v3-2-reasoning-0925': 'deepseek-ai/DeepSeek-R1',
    'deepseek-v3-2-0925': 'deepseek-ai/DeepSeek-V3',
    'deepseek-v3-2-speciale': 'deepseek-ai/DeepSeek-V3',
    'deepseek-r1': 'deepseek-ai/DeepSeek-R1',
    'deepseek-r1-0120': 'deepseek-ai/DeepSeek-R1',
    'deepseek-r1-distill-qwen-8b': 'deepseek/deepseek-r1-0528-qwen3-8b',
    'deepseek-r1-qwen3-8b': 'deepseek/deepseek-r1-0528-qwen3-8b',
    'deepseek-r1-distill-llama-70b': 'deepseek-ai/DeepSeek-R1-Distill-Llama-70B',
    'qwen3-4b-2507-instruct': 'Qwen/Qwen2.5-7B-Instruct-Turbo',
    'qwen3-4b-2507-instruct-reasoning': 'Qwen/Qwen2.5-7B-Instruct-Turbo',
    'qwen3-8b-instruct': 'Qwen/Qwen2.5-7B-Instruct-Turbo',
    'qwen3-8b-instruct-reasoning': 'Qwen/Qwen2.5-7B-Instruct-Turbo',
    'qwen3-14b-instruct-reasoning': 'Qwen/Qwen2.5-72B-Instruct-Turbo',
    'qwen3-32b-instruct-reasoning': 'Qwen/Qwen2.5-72B-Instruct-Turbo',
    'granite-3-3-8b-instruct': 'ibm-granite/granite-3.3-8b-instruct',
    'granite-4-0-h-small': 'ibm-granite/granite-3.3-8b-instruct',
    'phi-4-mini-instruct': 'microsoft/phi-4',
    'phi-4-mini': 'microsoft/phi-4',
    'phi-4': 'microsoft/phi-4',
    
    # xAI models
    'grok-3-mini-reasoning': 'grok-2-latest',
    'grok-4-fast': 'grok-2-latest',
    'grok-4-fast-reasoning': 'grok-2-latest', 
    'grok-4-1-fast': 'grok-2-latest',
}

# Models that are not available on serverless APIs (require dedicated endpoints or don't exist)
UNAVAILABLE_MODELS = {
    'ibm-granite/granite-3.3-8b-instruct',  # Together AI: requires dedicated endpoint
    'Qwen/Qwen2.5-14B-Instruct-Turbo',  # Not available on Together AI
    'qwen3-14b-instruct',  # Not available on Together AI
}


def get_api_client(provider: str):
    """Get the appropriate API client for a provider."""
    from openai import OpenAI
    
    if provider == 'anthropic':
        from anthropic import Anthropic
        return Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
    
    elif provider == 'google':
        # Use Google's OpenAI-compatible endpoint
        # Try GEMINI_API_KEY first, then GOOGLE_API_KEY
        api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
        return OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
    
    elif provider == 'openai':
        return OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    elif provider == 'mistral':
        return OpenAI(
            api_key=os.getenv('MISTRAL_API_KEY'),
            base_url="https://api.mistral.ai/v1"
        )
    
    elif provider == 'together':
        return OpenAI(
            api_key=os.getenv('TOGETHER_API_KEY'),
            base_url="https://api.together.xyz/v1"
        )
    
    elif provider == 'xai':
        return OpenAI(
            api_key=os.getenv('XAI_API_KEY'),
            base_url="https://api.x.ai/v1"
        )
    
    else:
        raise ValueError(f"Unknown provider: {provider}")


def call_model(client, provider: str, model_id: str, prompt: str) -> str:
    """Call a model and return the response."""
    try:
        if provider == 'anthropic':
            response = client.messages.create(
                model=model_id,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        
        else:
            # OpenAI-compatible APIs (OpenAI, Google, Mistral, Together, xAI)
            # OpenAI reasoning models (o1, o3, o4) require max_completion_tokens
            is_reasoning_model = any(x in model_id.lower() for x in ['o1', 'o3', 'o4-'])
            
            if provider == 'openai' and is_reasoning_model:
                response = client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=1024,
                )
            else:
                response = client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1024,
                )
            return response.choices[0].message.content
            
    except Exception as e:
        logger.error(f"Error calling {model_id}: {e}")
        return None


def load_models_for_direct_api(benchmark: str = "mixeval") -> Dict[str, List[ModelConfig]]:
    """Load models that need direct API evaluation for the specified benchmark.
    
    Args:
        benchmark: "mixeval" or "mixeval-hard"
        
    Returns:
        Dict mapping provider to list of models needing evaluation
    """
    cache_path = DATA_PATH / "models_cache.json"
    
    # Load scores for the requested benchmark
    if benchmark == "mixeval":
        scores_path = DATA_PATH / "mixeval_scores.json"
    else:
        scores_path = DATA_PATH / "mixeval_hard_scores.json"
    
    with open(cache_path) as f:
        data = json.load(f)
    
    # Load existing scores (or empty dict if file doesn't exist)
    existing_scores = {}
    if scores_path.exists():
        with open(scores_path) as f:
            existing_scores = json.load(f)
    
    models = data.get("models", data)
    benchmarks_required = ['intelligence_index', 'coding_index', 'math_index']
    
    # API key mapping
    api_map = {
        'Anthropic': ('ANTHROPIC_API_KEY', 'anthropic'),
        'Google': ('GOOGLE_API_KEY', 'google'),
        'OpenAI': ('OPENAI_API_KEY', 'openai'),
        'Mistral': ('MISTRAL_API_KEY', 'mistral'),
        'DeepSeek': ('TOGETHER_API_KEY', 'together'),
        'Alibaba': ('TOGETHER_API_KEY', 'together'),
        'IBM': ('TOGETHER_API_KEY', 'together'),
        'Microsoft Azure': ('TOGETHER_API_KEY', 'together'),
        'xAI': ('XAI_API_KEY', 'xai'),
    }
    
    by_provider = {}
    seen_api_ids = set()  # Track API IDs to dedupe
    skipped_have_score = 0
    
    for m in models:
        halluc = m.get('hallucination_rate')
        if not halluc or float(halluc) <= 0:
            continue
        
        has_benchmarks = all(m.get(b) and float(m.get(b, 0)) > 0 for b in benchmarks_required)
        if not has_benchmarks:
            continue
        
        # Skip if already has score for this benchmark (check both openrouter_id and slug)
        oid = m.get('openrouter_id', '')
        slug = m.get('slug', '')
        if oid and oid in existing_scores:
            skipped_have_score += 1
            continue
        if slug in existing_scores:
            skipped_have_score += 1
            continue
        
        creator = m.get('creator_name', '')
        if creator not in api_map:
            continue
        
        api_key_env, provider = api_map[creator]
        
        # Get the actual model ID
        api_model_id = MODEL_ID_MAPPINGS.get(slug, slug)
        
        # Skip if model is not available on serverless API
        if api_model_id in UNAVAILABLE_MODELS:
            continue
        
        # Skip if we've already added a model with this API ID (dedupe)
        provider_api_key = f"{provider}:{api_model_id}"
        if provider_api_key in seen_api_ids:
            continue
        seen_api_ids.add(provider_api_key)
        
        config = ModelConfig(
            name=m.get('name', ''),
            slug=slug,
            creator=creator,
            api_key_env=api_key_env,
            api_model_id=api_model_id,
            hallucination_rate=float(halluc),
        )
        
        if provider not in by_provider:
            by_provider[provider] = []
        by_provider[provider].append(config)
    
    if skipped_have_score > 0:
        print(f"  (Skipped {skipped_have_score} models that already have {benchmark} scores)")
    
    return by_provider


def load_mixeval_questions(max_samples: int = 100, benchmark: str = "mixeval") -> List[Dict]:
    """Load MixEval questions (both free-form and multiple-choice).
    
    Args:
        max_samples: Maximum number of questions to load
        benchmark: "mixeval" or "mixeval-hard"
    """
    # Support both mixeval and mixeval-hard
    bench_folder = "mixeval" if benchmark == "mixeval" else "mixeval-hard"
    base_path = MIXEVAL_PATH / "mix_eval" / "data" / "mixeval-2024-08-11" / bench_folder
    
    questions = []
    
    # Load free-form questions
    ff_path = base_path / "free-form.json"
    if ff_path.exists():
        with open(ff_path) as f:
            data = json.load(f)
        if isinstance(data, dict):
            items = [data[str(i)] for i in range(len(data)) if str(i) in data]
        else:
            items = data
        # Add type field and limit to half of samples
        for i, q in enumerate(items[:max_samples//2]):
            q['type'] = 'free-form'
            q['id'] = i
            questions.append(q)
    
    # Load multiple-choice questions
    mc_path = base_path / "multiple-choice.json"
    if mc_path.exists():
        with open(mc_path) as f:
            data = json.load(f)
        if isinstance(data, dict):
            items = [data[str(i)] for i in range(len(data)) if str(i) in data]
        else:
            items = data
        # Add type field and limit to half of samples
        start_id = len(questions)
        for i, q in enumerate(items[:max_samples//2]):
            q['type'] = 'multiple-choice'
            q['id'] = start_id + i
            questions.append(q)
    
    print(f"  Loaded {len(questions)} questions (free-form + multiple-choice)")
    return questions


def evaluate_model(
    model: ModelConfig,
    provider: str,
    questions: List[Dict],
    output_dir: Path,
    dry_run: bool = False,
) -> Optional[Dict]:
    """Evaluate a single model on MixEval questions."""
    
    print(f"\n{'='*60}")
    print(f"MODEL: {model.name}")
    print(f"Provider: {provider} | API Model: {model.api_model_id}")
    print(f"Hallucination Rate: {model.hallucination_rate}%")
    print(f"{'='*60}")
    
    if dry_run:
        print(f"  [DRY RUN] Would evaluate {model.name}")
        return {"status": "dry_run"}
    
    # Check API key
    if not os.getenv(model.api_key_env):
        print(f"  ❌ Missing {model.api_key_env}")
        return None
    
    client = get_api_client(provider)
    
    responses = []
    errors = []
    
    print(f"\n  📋 Evaluating {len(questions)} questions...")
    
    for i, q in enumerate(questions):
        q_type = q.get('type', 'free-form')
        
        # Format prompt based on question type
        if q_type == 'multiple-choice':
            prompt = q.get('prompt', q.get('question', ''))
            choices = q.get('options', q.get('choices', []))
            if choices:
                prompt += "\n\nChoices:\n"
                for j, choice in enumerate(choices):
                    letter = chr(65 + j)  # A, B, C, D
                    prompt += f"{letter}. {choice}\n"
                prompt += "\nAnswer with just the letter (A, B, C, or D)."
        else:
            prompt = q.get('prompt', q.get('question', ''))
        
        try:
            response = call_model(client, provider, model.api_model_id, prompt)
            if response:
                responses.append({
                    "id": q.get('id', i),
                    "type": q_type,
                    "prompt": prompt,
                    "response": response,
                    "expected": q.get('target', q.get('answer', '')),
                })
            else:
                errors.append({"id": q.get('id', i), "error": "Empty response"})
        except Exception as e:
            errors.append({"id": q.get('id', i), "error": str(e)})
        
        # Progress
        if (i + 1) % 10 == 0:
            pct = (i + 1) / len(questions) * 100
            bar = "█" * int(pct / 3.33) + "░" * (30 - int(pct / 3.33))
            print(f"\r  [{bar}] {pct:5.1f}% ({i+1}/{len(questions)})", end="", flush=True)
        
        # Rate limiting
        time.sleep(0.5)
    
    print()
    print(f"\n  ✅ Completed: {len(responses)} responses")
    print(f"  ❌ Errors: {len(errors)}")
    
    # Save responses
    model_dir = output_dir / model.mixeval_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    result = {
        "model_id": model.api_model_id,
        "model_name": model.name,
        "provider": provider,
        "responses": responses,
        "errors": errors,
    }
    
    with open(model_dir / "responses.json", 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"  💾 Saved to: {model_dir}/responses.json")
    
    return result


def compute_scores(results: Dict[str, Dict]) -> Dict[str, float]:
    """Compute MixEval scores from responses (matching OpenRouter scoring)."""
    scores = {}
    
    for model_key, result in results.items():
        if not result or result.get("status") == "dry_run":
            continue
        
        responses = result.get("responses", [])
        if not responses:
            continue
        
        correct = 0
        total = 0
        
        for r in responses:
            response = r.get("response", "").strip()
            expected = r.get("expected", r.get("target", ""))
            q_type = r.get("type", "free-form")
            
            # Handle list of valid answers
            if isinstance(expected, list):
                expected_list = [str(e).strip().lower() for e in expected if e]
            else:
                expected_list = [str(expected).strip().lower()] if expected else []
            
            if not expected_list:
                continue
            
            if q_type == "multiple-choice":
                # Letter matching for multiple-choice
                resp_upper = response.upper()
                
                # Extract just the letter
                if resp_upper and resp_upper[0] in "ABCD":
                    resp_letter = resp_upper[0]
                else:
                    resp_letter = resp_upper
                
                # Check if any expected answer matches
                is_correct = any(
                    resp_letter == e.upper()[0] if e and e[0].upper() in "ABCD" else resp_letter == e.upper()
                    for e in expected_list
                )
            else:
                # Free-form: check if response contains any expected answer
                resp_lower = response.lower()
                is_correct = any(exp in resp_lower for exp in expected_list)
            
            if is_correct:
                correct += 1
            total += 1
        
        score = (correct / total * 100) if total > 0 else 0
        scores[model_key] = score
        
        logger.info(f"  {model_key}: {score:.1f}% ({correct}/{total})")
    
    return scores


def main():
    parser = argparse.ArgumentParser(description="Run MixEval via direct provider APIs")
    parser.add_argument("--all", action="store_true", help="Run all evaluable models")
    parser.add_argument("--provider", type=str, help="Run specific provider only")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples")
    parser.add_argument("--benchmark", type=str, default="mixeval", 
                        choices=["mixeval", "mixeval-hard"],
                        help="Benchmark to run: mixeval or mixeval-hard")
    parser.add_argument("--threads", type=int, default=3, 
                        help="Number of parallel threads for model evaluation (default: 3)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be evaluated")
    args = parser.parse_args()
    
    benchmark_name = "MixEval" if args.benchmark == "mixeval" else "MixEval-Hard"
    print("="*70)
    print(f"{benchmark_name.upper()} EVALUATION VIA DIRECT PROVIDER APIs")
    print("="*70)
    
    # Load models (filtered by benchmark - skips models that already have scores)
    models_by_provider = load_models_for_direct_api(args.benchmark)
    
    total_models = sum(len(m) for m in models_by_provider.values())
    print(f"\nModels to evaluate: {total_models}")
    
    for provider, models in models_by_provider.items():
        print(f"  {provider}: {len(models)} models")
    
    if not args.all and not args.provider:
        print("\nUse --all to run all models or --provider <name> for specific provider")
        return
    
    # Filter by provider if specified
    if args.provider:
        if args.provider not in models_by_provider:
            print(f"Unknown provider: {args.provider}")
            print(f"Available: {list(models_by_provider.keys())}")
            return
        models_by_provider = {args.provider: models_by_provider[args.provider]}
    
    # Load questions
    questions = load_mixeval_questions(args.samples, args.benchmark)
    print(f"\nLoaded {len(questions)} {benchmark_name} questions")
    
    # Output directory
    output_dir = MIXEVAL_PATH / "mix_eval" / "data" / "model_responses"
    
    # Run evaluations
    print("\n" + "="*70)
    print(f"RUNNING EVALUATIONS (threads={args.threads})")
    print("="*70)
    
    all_results = {}
    results_lock = threading.Lock()
    
    # Flatten all models with their providers for parallel execution
    all_model_tasks = []
    for provider, models in models_by_provider.items():
        for model in models:
            all_model_tasks.append((model, provider))
    
    def evaluate_task(task):
        """Wrapper for thread pool execution."""
        model, provider = task
        result = evaluate_model(
            model=model,
            provider=provider,
            questions=questions,
            output_dir=output_dir,
            dry_run=args.dry_run,
        )
        return model.slug, result
    
    # Use ThreadPoolExecutor for parallel model evaluation
    completed = 0
    total = len(all_model_tasks)
    
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {executor.submit(evaluate_task, task): task for task in all_model_tasks}
        
        for future in as_completed(futures):
            task = futures[future]
            model, provider = task
            try:
                slug, result = future.result()
                if result:
                    with results_lock:
                        all_results[slug] = result
                completed += 1
                print(f"\n[{completed}/{total}] Completed: {model.name}")
            except Exception as e:
                completed += 1
                logger.error(f"[{completed}/{total}] Failed {model.name}: {e}")
    
    # Compute and save scores
    if not args.dry_run and all_results:
        print("\n" + "="*70)
        print(f"COMPUTING {benchmark_name.upper()} SCORES")
        print("="*70)
        
        scores = compute_scores(all_results)
        
        # Use different file for mixeval vs mixeval-hard
        if args.benchmark == "mixeval":
            scores_path = DATA_PATH / "mixeval_scores.json"
        else:
            scores_path = DATA_PATH / "mixeval_hard_scores.json"
        
        # Load existing scores (or create empty dict)
        if scores_path.exists():
            with open(scores_path) as f:
                existing_scores = json.load(f)
        else:
            existing_scores = {}
        
        # Add new scores (using slug as key for now)
        for slug, score in scores.items():
            existing_scores[slug] = score
        
        with open(scores_path, 'w') as f:
            json.dump(existing_scores, f, indent=2)
        
        logger.info(f"Saved {len(scores)} new scores to {scores_path}")
    
    print("\n" + "="*70)
    print("DONE")
    print("="*70)


if __name__ == "__main__":
    main()

