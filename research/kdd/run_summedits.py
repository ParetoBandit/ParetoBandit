#!/usr/bin/env python3
"""
SummEdits Evaluation Script

Evaluates models on the SummEdits benchmark for factual consistency in summarization.
SummEdits is a binary classification task: given a (Document, Summary) pair, 
determine if the summary is factually consistent with the document.

Benchmark: https://github.com/salesforce/factualNLG
Metric: Balanced Accuracy (0.0 to 1.0)

This is efficient as it only requires generating 1 token per sample ("Yes" or "No").
"""

import os
import sys
import json
import argparse
import time
import random
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from collections import defaultdict

# Setup paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # research/kdd -> research -> project root
DATA_PATH = PROJECT_ROOT / "data"
SUMMEDITS_PATH = PROJECT_ROOT / "factualNLG" / "data" / "summedits"
SUMMEDITS_PROMPTS = PROJECT_ROOT / "factualNLG" / "prompts" / "summedits"

sys.path.insert(0, str(PROJECT_ROOT))

# Load environment
load_dotenv(PROJECT_ROOT / ".env")

# Logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

# Reasoning model patterns (require max_completion_tokens instead of max_tokens)
REASONING_PATTERNS = [
    '/o1',
    '/o3',
    '/o4-',
    '/gpt-5',
    # Note: Gemini 3 works better with max_tokens, not max_completion_tokens
    '/deepseek-r1',          # DeepSeek R1 and variants
    '/distill-llama',        # DeepSeek R1 Distill Llama family
    '/distill-qwen',         # DeepSeek R1 Distill Qwen family
]

# Available SummEdits domains
SUMMEDITS_DOMAINS = [
    "news",
    "podcast", 
    "billsum",
    "samsum",
    "sales_call",
    "sales_email",
    "shakespeare",
    "scitldr",
    "qmsumm",
    "ectsum"
]


@dataclass
class ModelConfig:
    """Configuration for a model to evaluate."""
    name: str
    slug: str
    openrouter_id: str
    hallucination_rate: float
    
    @property
    def score_key(self) -> str:
        """Get the key used for storing scores."""
        return self.openrouter_id


def is_reasoning_model(model_id: str) -> bool:
    """Check if a model ID indicates a reasoning model."""
    model_lower = model_id.lower()
    return any(pattern in model_lower for pattern in REASONING_PATTERNS)


class OpenRouterClient:
    """OpenRouter API client with singleton pattern and retry logic."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._client = None
        return cls._instance
    
    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=os.getenv('OPENROUTER_API_KEY'),
                base_url="https://openrouter.ai/api/v1"
            )
        return self._client
    
    def _extract_text(self, choice: Any) -> str:
        """Normalize OpenAI/OpenRouter message content into plain text."""
        # Get message from choice
        message = getattr(choice, "message", None)
        if not message:
            return ""
        
        # Extract primary content
        content = getattr(message, "content", None)
        
        # Handle string content (most common case)
        if isinstance(content, str) and content.strip():
            return content
        
        # Newer SDKs/models may return list-of-parts; join any text parts
        if isinstance(content, list):
            parts: List[str] = []
            for part in content:
                if isinstance(part, dict):
                    text_val = part.get("text") or part.get("content") or ""
                else:
                    text_val = str(part)
                if text_val:
                    parts.append(text_val)
            if parts:
                return "\n".join(parts)
        
        # Some reasoning models (DeepSeek, Gemini 3) put output in 'reasoning' field
        # when given high token limits, leaving 'content' empty
        if hasattr(message, "reasoning"):
            reasoning = getattr(message, "reasoning", None)
            if isinstance(reasoning, str) and reasoning.strip():
                # For yes/no questions, extract the final answer from reasoning
                reasoning_lower = reasoning.lower().strip()
                # Look for final answer patterns
                if reasoning_lower.endswith("yes") or reasoning_lower.endswith("yes."):
                    return "yes"
                elif reasoning_lower.endswith("no") or reasoning_lower.endswith("no."):
                    return "no"
                # Return the full reasoning if no clear yes/no found
                return reasoning
        
        # Reasoning models may also use reasoning_content
        if hasattr(message, "reasoning_content"):
            reasoning_content = getattr(message, "reasoning_content")
            if isinstance(reasoning_content, str):
                return reasoning_content
            elif isinstance(reasoning_content, list):
                parts: List[str] = []
                for part in reasoning_content:
                    if isinstance(part, dict):
                        text_val = part.get("text") or part.get("content") or ""
                    else:
                        text_val = str(part)
                    if text_val:
                        parts.append(text_val)
                if parts:
                    return "\n".join(parts)
        
        # Fallback: no content found
        return ""
    
    def call(self, model_id: str, prompt: str, max_retries: int = 3) -> Optional[str]:
        """Call a model via OpenRouter with retry logic."""
        # Token limits to try - escalate if we get empty responses
        # Some models need extra tokens for internal processing
        is_reasoning = is_reasoning_model(model_id)
        is_gemini3 = '/gemini-3' in model_id.lower()
        is_minimax_m2 = '/minimax-m2' in model_id.lower()
        is_deepseek_distill = '/distill-' in model_id.lower()  # DeepSeek R1 Distill models
        
        if is_gemini3:
            token_limits = [4000, 8000, 16000]  # Gemini 3 needs significantly more tokens
        elif is_minimax_m2:
            token_limits = [2000, 4000, 8000]  # MiniMax M2 needs more tokens
        elif is_deepseek_distill:
            token_limits = [500, 1000, 2000]  # DeepSeek Distill works better with lower limits
        else:
            token_limits = [4000, 8000, 16000]  # Default for other models
        
        for attempt in range(max_retries):
            try:
                # Use higher token limit on retries if previous was empty
                tokens = token_limits[min(attempt, len(token_limits) - 1)]
                
                if is_reasoning:
                    response = self.client.chat.completions.create(
                        model=model_id,
                        messages=[{"role": "user", "content": prompt}],
                        max_completion_tokens=tokens,
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=model_id,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=tokens,
                        temperature=0,
                    )
                
                content = self._extract_text(response.choices[0])
                
                # If response is empty, retry with higher token limit
                if not content or not content.strip():
                    if attempt < max_retries - 1:
                        logger.warning(f"Empty response from {model_id}, retrying with more tokens...")
                        continue
                
                return content
            except Exception as e:
                error_str = str(e).lower()
                
                # Rate limit - exponential backoff
                if 'rate' in error_str or '429' in error_str:
                    wait_time = (2 ** attempt) * (1 + random.random())
                    logger.warning(f"Rate limit hit for {model_id}, waiting {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue
                
                # Other errors
                if attempt < max_retries - 1:
                    logger.warning(f"Error calling {model_id} (attempt {attempt + 1}/{max_retries}): {e}")
                    time.sleep(1)
                    continue
                else:
                    logger.error(f"Failed calling {model_id} after {max_retries} attempts: {e}")
                    return None
        
        return None


# Global client
_client = OpenRouterClient()


class DataManager:
    """Manages loading and caching of data files."""
    
    def __init__(self):
        self._models_cache: Optional[List[Dict]] = None
        self._scores_cache: Dict[str, Dict[str, float]] = {}
        self._summedits_cache: Dict[str, List[Dict]] = {}
        self._lock = threading.Lock()
    
    def get_models(self) -> List[Dict]:
        """Load and cache models from models_cache.json."""
        with self._lock:
            if self._models_cache is None:
                cache_path = DATA_PATH / "models_cache.json"
                with open(cache_path) as f:
                    data = json.load(f)
                self._models_cache = data.get("models", data)
            return self._models_cache
    
    def get_scores(self, domain: str) -> Dict[str, float]:
        """Load and cache scores for a SummEdits domain."""
        with self._lock:
            scores_file = f"summedits_{domain}_scores.json"
            if domain not in self._scores_cache:
                scores_path = DATA_PATH / scores_file
                if scores_path.exists():
                    with open(scores_path) as f:
                        self._scores_cache[domain] = json.load(f)
                else:
                    self._scores_cache[domain] = {}
            return self._scores_cache[domain]
    
    def save_scores(self, scores: Dict[str, float], domain: str):
        """Save scores and update cache."""
        with self._lock:
            # Always reload from disk to ensure we have the latest data
            # This prevents data loss if cache is stale or not populated
            scores_file = f"summedits_{domain}_scores.json"
            output_path = DATA_PATH / scores_file
            
            # Load existing scores from disk
            if output_path.exists():
                try:
                    with open(output_path, "r") as f:
                        existing = json.load(f)
                except (json.JSONDecodeError, IOError) as e:
                    logger.error(f"Failed to load {scores_file}: {e}")
                    existing = {}
            else:
                existing = {}
            
            # Update with new scores
            existing.update(scores)
            
            # Write to disk with explicit flush
            try:
                with open(output_path, "w") as f:
                    json.dump(existing, f, indent=2)
                    f.flush()  # Explicit flush to ensure write
                    os.fsync(f.fileno())  # Force OS to write to disk
                logger.info(f"Saved {len(scores)} scores to {scores_file}")
            except (IOError, OSError) as e:
                logger.error(f"Failed to save {scores_file}: {e}")
                raise  # Re-raise to make failure visible
            
            # Update cache after successful save
            self._scores_cache[domain] = existing
    
    def get_summedits_data(self, domain: str, split: str = "evaluation", max_samples: Optional[int] = None) -> List[Dict]:
        """Load SummEdits data for a specific domain."""
        cache_key = f"{domain}:{split}:{max_samples}"
        
        with self._lock:
            if cache_key not in self._summedits_cache:
                self._summedits_cache[cache_key] = self._load_summedits(domain, split, max_samples)
            return self._summedits_cache[cache_key]
    
    def _load_summedits(self, domain: str, split: str, max_samples: Optional[int]) -> List[Dict]:
        """Load SummEdits data from disk."""
        file_path = SUMMEDITS_PATH / f"summedits_{domain}.json"
        
        if not file_path.exists():
            logger.error(f"SummEdits file not found: {file_path}")
            return []
        
        with open(file_path) as f:
            all_data = json.load(f)
        
        # Filter by split
        data = [item for item in all_data if item.get("split") == split]
        
        # Sample if needed
        if max_samples and len(data) > max_samples:
            random.seed(42)
            data = random.sample(data, max_samples)
        
        logger.info(f"Loaded {len(data)} {domain} samples (split: {split})")
        return data


# Global data manager
_data = DataManager()


def load_prompt_template() -> str:
    """Load the standard zero-shot prompt template."""
    prompt_path = SUMMEDITS_PROMPTS / "standard_zs_prompt.txt"
    with open(prompt_path) as f:
        return f.read()


def format_prompt(doc: str, summary: str, template: str) -> str:
    """Format a document-summary pair as a prompt."""
    prompt = template.replace("[ARTICLE]", doc)
    prompt = prompt.replace("[SUMMARY_SENTENCES]", summary)
    return prompt


def parse_response(response: str) -> Optional[bool]:
    """Parse model response to extract Yes/No answer.
    
    Returns:
        True if consistent (Yes), False if inconsistent (No), None if unclear
    """
    if not response:
        return None
    
    import re
    
    # Strip various reasoning model tags
    cleaned = response
    for tag in ['think', 'thinking', 'reasoning', 'internal_thoughts', 'scratchpad']:
        cleaned = re.sub(rf'<{tag}>.*?</{tag}>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = cleaned.strip()
    
    # If response was entirely thinking, use original
    if not cleaned:
        cleaned = response
    
    response_lower = cleaned.strip().lower()
    
    # Look for explicit Yes/No at the start
    if response_lower.startswith("yes"):
        return True
    elif response_lower.startswith("no"):
        return False
    
    # Check if Yes/No appears early in response (first 50 chars)
    first_50 = response_lower[:50]
    if "yes" in first_50 and "no" not in first_50:
        return True
    elif "no" in first_50 and "yes" not in first_50:
        return False
    
    # Look for "Answer: Yes" or similar patterns anywhere
    answer_patterns = [
        r'\b(?:answer|conclusion|verdict|response)[:\s]+\**\s*(yes|no)\**',
        r'\*\*(yes|no)\*\*',  # Bold markdown
        r'^(yes|no)\s*[,.]',  # Yes/No at line start
        r'\n(yes|no)\s*$',    # Yes/No at end of line
        r'is\s+(yes|no)\b',   # "is Yes" or "is No"
        r':\s*(yes|no)\s*$',  # Colon followed by Yes/No at end
    ]
    
    for pattern in answer_patterns:
        match = re.search(pattern, response_lower, re.MULTILINE)
        if match:
            return match.group(1) == "yes"
    
    # Check for final lines/sentences starting with Yes/No
    lines = response_lower.strip().split('\n')
    for line in reversed(lines[-5:]):  # Check last 5 lines
        line = line.strip()
        if line.startswith("yes"):
            return True
        elif line.startswith("no"):
            return False
    
    sentences = response_lower.split('.')
    for sent in reversed(sentences[-5:]):  # Check last 5 sentences
        sent = sent.strip()
        if sent.startswith("yes"):
            return True
        elif sent.startswith("no"):
            return False
    
    # Last resort: check if the response ends with Yes or No (within last 20 chars)
    last_20 = response_lower[-20:].strip()
    if last_20.endswith("yes") or last_20.endswith("yes."):
        return True
    elif last_20.endswith("no") or last_20.endswith("no."):
        return False
    
    # Final fallback: count occurrences (only if clear majority)
    # Use stricter pattern to avoid matching "cannot", "know", etc.
    yes_count = len(re.findall(r'(?<![a-z])\byes\b(?![a-z])', response_lower))
    no_count = len(re.findall(r'(?<![a-z])\bno\b(?![a-z])', response_lower))
    
    # Exclude common phrases where "no" doesn't mean "No"
    # e.g., "no issues", "no errors" actually support consistency
    negative_no_phrases = len(re.findall(r'\bno\s+(?:issues?|errors?|problems?|inconsistenc)', response_lower))
    no_count = max(0, no_count - negative_no_phrases)
    
    # Only use count if there's a clear winner and at least one mention
    if yes_count > 0 and no_count == 0:
        return True
    elif no_count > 0 and yes_count == 0:
        return False
    
    return None


def calculate_balanced_accuracy(predictions: List[bool], labels: List[int]) -> float:
    """Calculate balanced accuracy (average of sensitivity and specificity).
    
    Args:
        predictions: List of predicted labels (True/False)
        labels: List of ground truth labels (1=consistent, 0=inconsistent)
    
    Returns:
        Balanced accuracy between 0.0 and 1.0
    """
    if len(predictions) != len(labels):
        raise ValueError("Predictions and labels must have same length")
    
    # Convert predictions to 1/0
    pred_labels = [1 if p else 0 for p in predictions]
    
    # Calculate confusion matrix
    tp = sum(1 for p, l in zip(pred_labels, labels) if p == 1 and l == 1)
    tn = sum(1 for p, l in zip(pred_labels, labels) if p == 0 and l == 0)
    fp = sum(1 for p, l in zip(pred_labels, labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(pred_labels, labels) if p == 0 and l == 1)
    
    # Sensitivity (recall for positive class)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # Specificity (recall for negative class)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # Balanced accuracy
    balanced_acc = (sensitivity + specificity) / 2.0
    
    return balanced_acc


def load_qualified_models(domains: List[str], force: bool = False) -> List[ModelConfig]:
    """Load models that need evaluation for the specified domains.
    
    Args:
        domains: List of domains to evaluate
        force: If True, include all models even if they have existing scores
    """
    models_data = _data.get_models()
    
    # Load existing scores
    existing_scores = {domain: _data.get_scores(domain) for domain in domains}
    
    qualified = []
    seen_ids = set()
    skipped_have_scores = 0
    skipped_no_api = 0
    
    for m in models_data:
        openrouter_id = m.get('openrouter_id', '')
        if not openrouter_id:
            skipped_no_api += 1
            continue
        
        # Check if model needs any of the requested domains (skip if --force is set)
        if not force:
            needs_domain = False
            for domain in domains:
                if openrouter_id not in existing_scores.get(domain, {}):
                    needs_domain = True
                    break
            
            if not needs_domain:
                skipped_have_scores += 1
                continue
        
        # Dedupe by OpenRouter ID
        if openrouter_id in seen_ids:
            continue
        seen_ids.add(openrouter_id)
        
        qualified.append(ModelConfig(
            name=m.get('name', ''),
            slug=m.get('slug', ''),
            openrouter_id=openrouter_id,
            hallucination_rate=float(m.get('hallucination_rate', 0) or 0),
        ))
    
    if skipped_have_scores > 0:
        logger.info(f"Skipped {skipped_have_scores} models that already have scores")
    if skipped_no_api > 0:
        logger.info(f"Skipped {skipped_no_api} models with no OpenRouter ID")
    
    return qualified


def run_evaluation(
    model: ModelConfig, 
    samples: List[Dict], 
    domain: str,
    prompt_template: str,
    dry_run: bool = False
) -> Optional[Dict]:
    """Run evaluation for a single model on a domain."""
    
    if dry_run:
        print(f"  [DRY RUN] Would evaluate {model.name} via {model.openrouter_id}")
        return {"balanced_accuracy": 0, "accuracy": 0, "dry_run": True}
    
    total = len(samples)
    predictions = []
    labels = []
    errors = 0
    
    print(f"\n{'='*70}")
    print(f"Evaluating: {model.name}")
    print(f"Model ID: {model.openrouter_id}")
    print(f"Domain: {domain}")
    print(f"Samples: {total}")
    print(f"{'='*70}")
    
    for i, sample in enumerate(samples):
        # Progress bar
        pct = (i + 1) / total * 100
        filled = int(30 * (i + 1) / total)
        bar = "█" * filled + "░" * (30 - filled)
        print(f"\r  [{bar}] {pct:5.1f}% ({i+1}/{total}) ", end="", flush=True)
        
        try:
            doc = sample.get("doc", "")
            summary = sample.get("summary", "")
            label = sample.get("label", 0)
            
            # Format prompt
            prompt = format_prompt(doc, summary, prompt_template)
            
            # Call model
            response = _client.call(model.openrouter_id, prompt)
            
            # Parse response
            if response:
                prediction = parse_response(response)
                if prediction is not None:
                    predictions.append(prediction)
                    labels.append(label)
                else:
                    errors += 1
                    # Log first few unparseable responses for debugging
                    if errors <= 3:
                        logger.warning(f"Unparseable response from {model.name}: {response[:200]}...")
                    # For unparseable responses, use a conservative guess
                    predictions.append(True)  # Assume consistent if unclear
                    labels.append(label)
            else:
                errors += 1
            
            # Rate limiting
            time.sleep(0.05)  # Very short delay since we only generate 1 token
            
        except Exception as e:
            errors += 1
            logger.error(f"Error on sample {i}: {e}")
    
    print()  # New line after progress bar
    
    if len(predictions) == 0:
        logger.error(f"No valid predictions for {model.name}")
        return None
    
    # Calculate metrics
    balanced_acc = calculate_balanced_accuracy(predictions, labels)
    
    # Also calculate simple accuracy for reference
    correct = sum(1 for p, l in zip(predictions, labels) if (p and l == 1) or (not p and l == 0))
    accuracy = correct / len(predictions) if predictions else 0
    
    print(f"  ✅ Balanced Accuracy: {balanced_acc*100:.1f}%")
    print(f"  📊 Simple Accuracy: {accuracy*100:.1f}% ({correct}/{len(predictions)})")
    if errors:
        print(f"  ❌ Errors/Unparseable: {errors}")
    
    return {
        "balanced_accuracy": balanced_acc * 100,  # Store as percentage
        "accuracy": accuracy * 100,
        "correct": correct,
        "total": len(predictions),
        "errors": errors
    }


def main():
    parser = argparse.ArgumentParser(description="Run SummEdits evaluation via OpenRouter")
    parser.add_argument("--all", action="store_true", help="Run on all qualified models")
    parser.add_argument("--models", nargs="+", help="Specific model slugs to evaluate")
    parser.add_argument("--model", type=str, help="Single model to evaluate (matches name or openrouter_id, case-insensitive)")
    parser.add_argument("--domains", nargs="+", choices=SUMMEDITS_DOMAINS + ["all"],
                        default=["news"], help="Which domain(s) to evaluate (default: news)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be evaluated")
    parser.add_argument("--max-models", type=int, default=None, help="Limit number of models")
    parser.add_argument("--max-samples", type=int, default=None, 
                        help="Max samples per domain (default: all)")
    parser.add_argument("--threads", type=int, default=5, 
                        help="Parallel threads (default: 5, can be higher since only 1 token per call)")
    parser.add_argument("--split", default="evaluation", choices=["evaluation", "test"],
                        help="Which data split to use (default: evaluation)")
    parser.add_argument("--force", action="store_true",
                        help="Force re-evaluation even if scores already exist")
    
    args = parser.parse_args()
    
    # Determine domains to evaluate
    if "all" in args.domains:
        domains = SUMMEDITS_DOMAINS
    else:
        domains = args.domains
    
    print("=" * 70)
    print("SUMMEDITS EVALUATION VIA OPENROUTER")
    print("=" * 70)
    print(f"Domains: {', '.join(domains)}")
    print(f"Split: {args.split}")
    print(f"Max samples per domain: {args.max_samples if args.max_samples else 'all'}")
    print(f"Threads: {args.threads}")
    if args.force:
        print(f"⚠️  Force mode: Will re-evaluate models even if scores exist")
    print()
    
    # Load prompt template
    prompt_template = load_prompt_template()
    print(f"✓ Loaded prompt template")
    
    # Load qualified models
    all_models = load_qualified_models(domains, force=args.force)
    print(f"\n✓ Qualified models: {len(all_models)}")
    
    # Filter to requested models
    if args.model:
        # Single model selection - match by name or openrouter_id (case-insensitive)
        search = args.model.lower()
        
        # Try exact matches first
        exact_matches = [
            m for m in all_models 
            if search == m.openrouter_id.lower() 
            or search == m.name.lower()
            or search == m.slug.lower()
        ]
        
        if exact_matches:
            models = exact_matches
        else:
            # Fall back to substring match
            models = [m for m in all_models if search in m.name.lower() or search in m.openrouter_id.lower()]
            
            # If multiple matches, prefer the shortest (most specific) match
            if len(models) > 1:
                # Sort by total length of matched fields (prefer shorter = more specific)
                models_with_score = []
                for m in models:
                    # Score based on how much extra text beyond the search term
                    name_extra = len(m.name) - len(search) if search in m.name.lower() else 9999
                    id_extra = len(m.openrouter_id) - len(search) if search in m.openrouter_id.lower() else 9999
                    min_extra = min(name_extra, id_extra)
                    models_with_score.append((min_extra, m))
                
                # Sort by score (lower is better)
                models_with_score.sort(key=lambda x: x[0])
                
                # If the best match is significantly better than second best, use it
                if len(models_with_score) > 1:
                    best_score = models_with_score[0][0]
                    second_score = models_with_score[1][0]
                    
                    # If best match has at least 3 fewer extra chars, it's unambiguous
                    if best_score + 3 <= second_score:
                        models = [models_with_score[0][1]]
        
        if not models:
            print(f"\n❌ No model found matching '{args.model}'")
            print("\nAvailable models:")
            for m in all_models:
                print(f"  {m.name} ({m.openrouter_id})")
            return
        elif len(models) > 1:
            print(f"\n⚠️  Multiple models match '{args.model}':")
            for m in models:
                print(f"  {m.name} ({m.openrouter_id})")
            print("\nBe more specific or use the full openrouter_id")
            return
        print(f"  Selected: {models[0].name} ({models[0].openrouter_id})")
    elif args.models:
        models = [m for m in all_models if m.slug in args.models]
        print(f"  Filtered to {len(models)} requested models")
    elif args.all:
        models = all_models
    else:
        print("\nSpecify --all, --model <name>, or --models <slugs>")
        print("\nAvailable models:")
        for m in all_models[:15]:
            print(f"  {m.name} ({m.openrouter_id})")
        if len(all_models) > 15:
            print(f"  ... and {len(all_models) - 15} more")
        return
    
    if args.max_models:
        models = models[:args.max_models]
    
    print(f"\n✓ Models to evaluate: {len(models)}")
    
    # Dry run mode
    if args.dry_run:
        print("\n[DRY RUN MODE]")
        for m in models:
            print(f"  {m.name} -> {m.openrouter_id}")
        return
    
    # Run evaluations for each domain
    for domain in domains:
        print("\n" + "=" * 70)
        print(f"DOMAIN: {domain.upper()}")
        print("=" * 70)
        
        # Load samples
        samples = _data.get_summedits_data(domain, args.split, args.max_samples)
        
        if not samples:
            logger.warning(f"No samples loaded for domain: {domain}")
            continue
        
        # Count positive/negative samples
        pos_count = sum(1 for s in samples if s.get("label") == 1)
        neg_count = len(samples) - pos_count
        print(f"  Samples: {len(samples)} (Consistent: {pos_count}, Inconsistent: {neg_count})")
        
        results = {}
        results_lock = threading.Lock()
        
        def evaluate_model(model: ModelConfig):
            return model, run_evaluation(model, samples, domain, prompt_template)
        
        # Run evaluations in parallel
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            futures = {executor.submit(evaluate_model, m): m for m in models}
            
            for future in as_completed(futures):
                model = futures[future]
                try:
                    m, result = future.result()
                    if result and result.get("balanced_accuracy") is not None:
                        with results_lock:
                            results[m.score_key] = result["balanced_accuracy"]
                except Exception as e:
                    logger.error(f"Failed {model.name}: {e}")
        
        # Print and save results
        if results:
            # Print summary first
            print(f"\n{'='*70}")
            print(f"RESULTS SUMMARY - {domain.upper()}")
            print(f"{'='*70}")
            sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
            for model_id, score in sorted_results[:10]:
                print(f"  {model_id:<50} {score:>6.1f}%")
            if len(sorted_results) > 10:
                print(f"  ... and {len(sorted_results) - 10} more")
            
            # Save to disk immediately after printing (ensures screen output and file write are in sync)
            _data.save_scores(results, domain)
    
    print("\n" + "=" * 70)
    print("✅ EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

