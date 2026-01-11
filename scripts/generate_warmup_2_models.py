#!/usr/bin/env python3
"""
generate_warmup_2_models.py

Specialized Warmup Generator for 2-Model Baseline (Gemini 3 Pro + GPT-OSS-120B).

Strategically identical to generate_warmup.py but restricted to 2 models.
"""

import sys
import numpy as np
import joblib
from pathlib import Path
from tqdm import tqdm
import math

# Add project root to path so we can import src
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.bandit_gpt.router import BanditRouter
from experiments.utils.data_loader import load_model_registry
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from src.bandit_gpt.router import BanditRouter
from src.bandit_gpt.utils.heuristics import HeuristicService

# CONFIGURATION
# Export to root/data directory (absolute path for clarity)
OUTPUT_PATH = PROJECT_ROOT / "data" / "priors_warmup_fair_test.joblib"
N_SAMPLES = 20000
SEED = 42

# Bucket allocation ("Best of Both Worlds" strategy)
N_ROUTELLM_HARD = 7000      # Hard prompts from RouteLLM (famous for tricking weak models)
N_DOMAIN_SPECIFIC = 7000    # Synthetic domain coverage (Math, Code, Reasoning)
N_SIMPLE_NOISE = 6000       # Synthetic easy prompts (Chat, simple questions)

def ir_theory_reward(model_skill: float, difficulty: float) -> float:
    """Same IRT logic as main script."""
    theta = (model_skill - 0.5) * 6.0      # Model ability in logit space
    beta = (difficulty - 0.5) * 6.0        # Task difficulty in logit space
    discriminability = 1.5
    logit = discriminability * (theta - beta)
    probability = 1.0 / (1.0 + math.exp(-logit))
    return probability

def mine_hard_prompts_from_routellm(n: int = 7000, seed: int = 42) -> list:
    """Same mining logic."""
    import random
    import re
    
    random.seed(seed)
    
    print(f"   ⛏️  Mining {n} hard prompts from RouteLLM (gpt4_judge_battles)...")
    print("      (Using streaming mode to avoid full download)")
    
    # Load dataset in streaming mode
    ds = load_dataset("routellm/gpt4_judge_battles", split="train", streaming=True)
    
    hard_prompts = []
    candidates_seen = 0
    
    # Feature detection patterns (lightweight)
    code_pattern = re.compile(r'```|def |class |import |function|\bcode\b', re.IGNORECASE)
    math_pattern = re.compile(r'\\frac|\\int|derivative|integral|theorem|prove|equation|calculate', re.IGNORECASE)
    
    for row in ds:
        # Extract prompt (handle both string and list formats)
        prompt = row['prompt'][0] if isinstance(row['prompt'], list) else row['prompt']
        
        # Apply filtering: code, math, or long reasoning
        has_code = bool(code_pattern.search(prompt))
        has_math = bool(math_pattern.search(prompt))
        is_long = len(prompt) > 200  # Long prompts often indicate complex reasoning
        
        # Only keep if it has structural complexity
        if has_code or has_math or is_long:
            hard_prompts.append(prompt)
            
            if len(hard_prompts) >= n:
                break
        
        candidates_seen += 1
        
        # Progress updates
        if candidates_seen % 5000 == 0:
            print(f"      Scanned {candidates_seen} prompts, found {len(hard_prompts)} hard ones...")
    
    print(f"   ✓ Mined {len(hard_prompts)} hard prompts (scanned {candidates_seen} total)")
    
    return hard_prompts

def generate_domain_specific_prompts(n: int = 7000, seed: int = 42) -> list:
    """Same generation logic."""
    import random
    random.seed(seed + 1)  # Different seed from routellm mining
    
    print(f"   ⚙️  Generating {n} domain-specific synthetic prompts...")
    
    # Domain-specific templates (Math, Code, Reasoning)
    templates = {
        "math": [
            "Solve the integral of {expr} with respect to {var}",
            "Prove that {theorem} using mathematical induction",
            "Find the derivative of {function} and explain each step",
            "Calculate the eigenvalues of the matrix {matrix}",
            "Determine if the series {series} converges or diverges"
        ],
        "coding": [
            "Write a Python function to {task} using {library}",
            "Implement {algorithm} in {language} with time complexity analysis",
            "Debug this {language} code that {problem}",
            "Create a {language} class for {task} with unit tests",
            "Optimize this {algorithm} implementation for {constraint}"
        ],
        "reasoning": [
            "Analyze the logical structure of {argument} and identify fallacies",
            "Develop a step-by-step solution for {problem}",
            "Compare and contrast {concept_a} with {concept_b}",
            "Explain the causal relationship between {cause} and {effect}",
            "Evaluate the validity of {claim} given {evidence}"
        ]
    }
    
    # Fill values for placeholders
    fill_values = {
        "expr": ["x^2 + 3x + 2", "sin(x)cos(x)", "e^(2x)", "ln(x^2)"],
        "var": ["x", "y", "t", "theta"],
        "theorem": ["Fermat's Last Theorem", "the Pythagorean identity", "Euler's formula"],
        "function": ["f(x) = x^3 + 2x", "g(x) = sqrt(x+1)", "h(x) = e^x / x"],
        "matrix": ["[[1,2],[3,4]]", "a 3x3 identity matrix", "[[2,-1],[4,3]]"],
        "series": ["sum(1/n^2)", "sum((-1)^n/n)", "sum(1/n!)"],
        "task": ["parse JSON", "sort a list", "find duplicates", "merge dictionaries"],
        "library": ["pandas", "numpy", "requests", "pathlib"],
        "algorithm": ["binary search", "quicksort", "dijkstra's", "BFS"],
        "language": ["Python", "JavaScript", "Java", "C++"],
        "problem": ["throws TypeError", "has memory leak", "returns wrong output"],
        "constraint": ["memory", "speed", "readability"],
        "argument": ["this logical claim", "the premise that AI is conscious"],
        "concept_a": ["AI", "machine learning", "neural networks"],
        "concept_b": ["automation", "deep learning", "decision trees"],
        "cause": ["climate change", "urbanization", "technology adoption"],
        "effect": ["sea level rise", "habitat loss", "social transformation"],
        "claim": ["this hypothesis", "the assertion", "the theory"],
        "evidence": ["the data", "experimental results", "historical records"]
    }
    
    prompts = []
    archetype_keys = list(templates.keys())
    
    for _ in range(n):
        archetype = random.choice(archetype_keys)
        template = random.choice(templates[archetype])
        
        # Fill placeholders
        result = template
        for placeholder, options in fill_values.items():
            if "{" + placeholder + "}" in result:
                result = result.replace("{" + placeholder + "}", random.choice(options))
        
        prompts.append(result)
    
    print(f"   ✓ Generated {len(prompts)} domain-specific prompts")
    return prompts

def generate_simple_prompts(n: int = 6000, seed: int = 42) -> list:
    """Same generation logic."""
    import random
    random.seed(seed + 2)  # Different seed
    
    print(f"   💬 Generating {n} simple/noise synthetic prompts...")
    
    # Simple chat templates
    templates = {
        "chat": [
            "What is {simple_concept} and why is it important?",
            "Can you explain {topic} in simple terms?",
            "Tell me about {subject}",
            "Why does {phenomenon} happen?",
            "What's the difference between {concept_a} and {concept_b}?",
            "Hi",
            "Hello",
            "How are you?",
            "Tell me a joke",
            "What's the weather like?"
        ]
    }
    
    fill_values = {
        "simple_concept": ["photosynthesis", "gravity", "democracy", "inflation"],
        "topic": ["climate change", "artificial intelligence", "the internet"],
        "subject": ["cats", "history", "cooking", "music"],
        "phenomenon": ["rain", "lightning", "the aurora borealis"],
        "concept_a": ["coffee", "cats", "summer"],
        "concept_b": ["tea", "dogs", "winter"]
    }
    
    prompts = []
    
    for _ in range(n):
        template = random.choice(templates["chat"])
        
        # Fill placeholders if any
        result = template
        for placeholder, options in fill_values.items():
            if "{" + placeholder + "}" in result:
                result = result.replace("{" + placeholder + "}", random.choice(options))
        
        prompts.append(result)
    
    print(f"   ✓ Generated {len(prompts)} simple prompts")
    return prompts

def simulate_irt_reward(model_hle: float, difficulty_score: float, is_trap: bool = False) -> float:
    """Simulates outcome using IRT logic (P = Sigmoid(Ability - Difficulty))."""
    if is_trap:
        return model_hle
    
    # Map HLE (0.7-0.98) to Ability Logit
    ability_logit = (model_hle - 0.65) * 20.0
    # Map Difficulty (0.0-1.0) to Difficulty Logit
    difficulty_logit = (difficulty_score - 0.2) * 6.0
    
    logit = ability_logit - difficulty_logit
    prob = 1 / (1 + math.exp(-logit))
    return prob

def main():
    print(f"🚀 Starting Synthetic Warmup Generator (2-Model Subset, N={N_SAMPLES})...")
    
    # 1. Setup
    full_registry = load_model_registry()
    
    # --- FILTER TO 2 MODELS ---
    target_models = ["google/gemini-2.5-flash-preview-09-2025", "anthropic/claude-opus-4.5"]
    registry = {k: v for k, v in full_registry.items() if k in target_models}
    
    if len(registry) != 2:
        print(f"❌ Error: Expected 2 models, found {len(registry)}")
        print(f"Available in registry: {list(full_registry.keys())}")
        return
        
    print(f"   ℹ️ Filtered registry to: {list(registry.keys())}")
    
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    # Initialize a "Blank" Router
    print("   Initializing blank router (cold start)...")
    router = BanditRouter.create(
        registry,
        exploration="safe",
        priors="none", 
        context_encoder=encoder
    )
    
    # 2. Generate Mixed Dataset (Three Buckets)
    print("\n📦 Building Mixed Warmup Dataset (20,000 prompts)...")
    
    # Bucket 1: RouteLLM Hard Prompts (Augmented Mining)
    routellm_prompts = mine_hard_prompts_from_routellm(n=N_ROUTELLM_HARD, seed=SEED)
    
    # Bucket 2: Domain-Specific Synthetic (Controlled Coverage)
    domain_prompts = generate_domain_specific_prompts(n=N_DOMAIN_SPECIFIC, seed=SEED)
    
    # Bucket 3: Simple/Noise Synthetic (Easy Baselines)
    simple_prompts = generate_simple_prompts(n=N_SIMPLE_NOISE, seed=SEED)
    
    # Combine and shuffle for IID training
    print("\n   🔀 Combining and shuffling buckets...")
    all_prompts = routellm_prompts + domain_prompts + simple_prompts
    
    import random
    random.seed(SEED)
    random.shuffle(all_prompts)
    
    prompts = all_prompts
    
    print(f"   Simulating {len(prompts)} interactions across {len(router.bandit.models)} models...")
    
    # 3. Analyze HLE Score Coverage
    models_with_hle = 0
    models_with_fallback = 0
    hle_scores = []
    
    for model_id in router.bandit.models:
        hle = router.registry.get(model_id, {}).get("hle", None)
        if hle is not None:
            models_with_hle += 1
            hle_scores.append(hle)
        else:
            models_with_fallback += 1
    
    print(f"   HLE Coverage:")
    print(f"     ✓ Models with HLE scores: {models_with_hle}/{len(router.bandit.models)}")
    if hle_scores:
        print(f"     HLE range: [{min(hle_scores):.3f}, {max(hle_scores):.3f}], mean={np.mean(hle_scores):.3f}")
    
    # 4. Training Loop
    print("   🚀 Processing updates in batches for speed...")
    BATCH_SIZE = 100
    updates_count = 0
    
    # Pre-calculate HLE map for fast lookup
    model_hle_map = {}
    for model_id in router.bandit.models:
        model_hle_map[model_id] = router.registry.get(model_id, {}).get("hle", 0.5) or 0.5

    for i in tqdm(range(0, len(prompts), BATCH_SIZE), desc="Processing Batches"):
        batch_prompts = prompts[i:i+BATCH_SIZE]
        
        for prompt in batch_prompts:
            # 1. Pre-compute context vector ONCE
            context_vector = router._get_context_vector(prompt)
        
            # A. Analyze Context (The "Map")
            difficulty = HeuristicService.detect_difficulty(prompt)
            is_trap = HeuristicService.detect_trap(prompt)
            
            # B. Update Every Model (The "Compass")
            for model_id in router.bandit.models:
                base_hle = model_hle_map[model_id]
                
                # Use IRT for standard prompts
                prob_success = simulate_irt_reward(base_hle, difficulty, is_trap=is_trap)
                
                # Bernoulli Sampling (Thompson Style)
                reward = 1.0 if random.random() < prob_success else 0.0
                
                # C. Update the Bandit State
                router.update(model_id, context_vector, reward)
                updates_count += 1

    # 5. Save the Artifact
    print(f"✅ Training complete. Processed {updates_count} simulated updates.")
    
    state_to_save = {
        "A": router.bandit.A,
        "b": router.bandit.b,
        "n": N_SAMPLES
    }
    
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(state_to_save, OUTPUT_PATH)
    
    print(f"💾 Saved Warmup Priors to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
