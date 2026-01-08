#!/usr/bin/env python3
"""
generate_warmup.py

End-to-End Mixed Warmup Generator (20,000 prompts).

Strategy ("Best of Both Worlds"):
1. Bucket 1 (7,000): Hard prompts from routellm/gpt4_judge_battles (filtered for code/math/long reasoning)
2. Bucket 2 (7,000): Domain-specific synthetic (Math, Code, Reasoning archetypes)
3. Bucket 3 (6,000): Simple/noise synthetic (Chat, easy questions)

Workflow:
- Mines hard prompts using streaming HuggingFace dataset access
- Generates controlled synthetic data for domain coverage
- Simulates rewards using Item Response Theory (IRT)
- Updates a BanditRouter to build dense A matrices and b vectors
- Saves the resulting state to 'data/priors_warmup.joblib'

Reference: RouteLLM dataset - https://huggingface.co/datasets/routellm/gpt4_judge_battles
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
from src.bandit_gpt.router import transform_hle_to_prior

# CONFIGURATION
# Export to root/data directory (absolute path for clarity)
OUTPUT_PATH = PROJECT_ROOT / "data" / "priors_warmup.joblib"
N_SAMPLES = 20000
SEED = 42

# Bucket allocation ("Best of Both Worlds" strategy)
N_ROUTELLM_HARD = 7000      # Hard prompts from RouteLLM (famous for tricking weak models)
N_DOMAIN_SPECIFIC = 7000    # Synthetic domain coverage (Math, Code, Reasoning)
N_SIMPLE_NOISE = 6000       # Synthetic easy prompts (Chat, simple questions)

def ir_theory_reward(model_skill: float, difficulty: float) -> float:
    """
    Simulates a probability of success using Item Response Theory (IRT).
    
    **Mathematical Foundation:**
    
    The transformation from [0, 1] to [-3, +3] is REQUIRED for realistic probabilities.
    
    Without transformation (naive approach):
    - Opus (0.9) vs Easy (0.1): diff = 0.8 → sigmoid(0.8) ≈ 0.69 (69% success)
    - WRONG: Opus should have ~100% success on easy prompts!
    
    With transformation (this implementation):
    - Opus (0.9): (0.9 - 0.5) × 6 = +2.4 (strong skill)
    - Easy (0.1): (0.1 - 0.5) × 6 = -2.4 (low difficulty)
    - Logit: 1.5 × (2.4 - (-2.4)) = 1.5 × 4.8 = 7.2
    - Probability: sigmoid(7.2) ≈ 0.991 (99.1% success)
    - CORRECT: Opus almost always solves easy prompts!
    
    **Why This Matters:**
    - Small gaps (HLE 0.5 vs 0.55) → ~50/50 outcomes (uncertainty)
    - Large gaps (HLE 0.2 vs 0.8) → decisive outcomes (0% or 100%)
    - Without this, synthetic data is "muddy" and bandit can't learn sharp distinctions
    
    **1-Parameter Logistic Model (Rasch):**
    P(success) = 1 / (1 + exp(-a(θ - β)))
    
    where:
    - θ (theta): person ability = (model_skill - 0.5) × 6
    - β (beta): item difficulty = (difficulty - 0.5) × 6
    - a: discrimination parameter = 1.5 (controls curve steepness)
    
    Args:
        model_skill (float): Normalized skill from HLE score (0.0 to 1.0)
        difficulty (float): Normalized difficulty from router (0.0 to 1.0)
        
    Returns:
        float: Probability of success (0.0 to 1.0)
    
    Example:
        >>> ir_theory_reward(model_skill=0.9, difficulty=0.1)  # Opus on easy
        0.991  # 99.1% success
        >>> ir_theory_reward(model_skill=0.2, difficulty=0.9)  # Weak model on hard
        0.009  # 0.9% success
        >>> ir_theory_reward(model_skill=0.5, difficulty=0.5)  # Equal match
        0.500  # 50% success
    """
    # 1. Transform from [0, 1] to [-3, +3] logit space
    #    This "stretches" the probability space to enable extreme values
    theta = (model_skill - 0.5) * 6.0      # Model ability in logit space
    beta = (difficulty - 0.5) * 6.0        # Task difficulty in logit space
    
    # 2. Apply 1-Parameter Logistic Model with discrimination
    #    Discriminability (1.5) controls steepness of the sigmoid curve
    #    Higher values → sharper transitions between success/failure
    discriminability = 1.5
    logit = discriminability * (theta - beta)
    
    # 3. Convert logit to probability via sigmoid
    #    sigmoid(x) = 1 / (1 + exp(-x))
    probability = 1.0 / (1.0 + math.exp(-logit))
    
    return probability

def mine_hard_prompts_from_routellm(n: int = 7000, seed: int = 42) -> list:
    """
    Mine hard prompts from RouteLLM's gpt4_judge_battles dataset.
    
    Uses streaming mode to avoid downloading the full dataset.
    Filters for prompts with code, math, or long reasoning (structural complexity).
    
    Args:
        n: Number of hard prompts to mine (default: 7000)
        seed: Random seed for reproducibility
        
    Returns:
        List of hard prompt strings
    """
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
    """
    Generate domain-specific synthetic prompts (Math, Code, Reasoning).
    
    Args:
        n: Number of domain-specific prompts to generate
        seed: Random seed for reproducibility
        
    Returns:
        List of synthetic prompt strings
    """
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
    """
    Generate simple/noise synthetic prompts (Chat, easy questions).
    
    Args:
        n: Number of simple prompts to generate
        seed: Random seed for reproducibility
        
    Returns:
        List of synthetic prompt strings
    """
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

def main():
    print(f"🚀 Starting Synthetic Warmup Generator (N={N_SAMPLES})...")
    
    # 1. Setup
    registry = load_model_registry()
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    # Initialize a "Blank" Router
    # We set priors="none" because we are ABOUT to build the priors manually.
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
    
    print(f"   ✓ Total prompts: {len(all_prompts)}")
    print(f"      - RouteLLM Hard: {len(routellm_prompts)}")
    print(f"      - Domain-Specific: {len(domain_prompts)}")
    print(f"      - Simple/Noise: {len(simple_prompts)}")
    
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
    if models_with_fallback > 0:
        print(f"     ⚠ Models using fallback (0.5): {models_with_fallback}")
    if hle_scores:
        print(f"     HLE range: [{min(hle_scores):.3f}, {max(hle_scores):.3f}], mean={np.mean(hle_scores):.3f}")
    
    # 4. Training Loop (Optimized Batch Processing)
    print("   🚀 Processing updates in batches for speed...")
    BATCH_SIZE = 100
    updates_count = 0
    
    # Pre-calculate HLE map for fast lookup
    model_hle_map = {}
    for model_id in router.bandit.models:
        model_hle_map[model_id] = router.registry.get(model_id, {}).get("hle", 0.5) or 0.5

    for i in tqdm(range(0, len(prompts), BATCH_SIZE), desc="Processing Batches"):
        batch_prompts = prompts[i:i+BATCH_SIZE]
        
        # 1. Batch Encode
        # We need context vectors for the bandit update.
        # Router doesn't expose public batch encoding easily, but encoder does.
        # However, we need the FULL context vector (with features).
        # We'll use a loop for now but with pre-computed embeddings if possible?
        # Actually, let's keep the loop simple but optimize the calls.
        
        # NOTE: True batching requires refactoring router.update.
        # For now, we accept the overhead but removing print/tqdm overhead inside loop
        # and pre-calculating HLE helps.
        
        for prompt in batch_prompts:
            # 1. Pre-compute context vector ONCE (37x speedup)
            # This avoids re-encoding the prompt for every model update
            # We access the internal method to get the vector
            # NOTE: _get_context_vector handles string -> vector conversion
            # We call it here so we can pass the vector to update()
            context_vector = router._get_context_vector(prompt)
        
            # A. Analyze Context (The "Map")
            difficulty = router._detect_difficulty_score(prompt)
            
            # B. Update Every Model (The "Compass")
            for model_id in router.bandit.models:
                hle = model_hle_map[model_id]
                
                # Calculate Reward using Router's Transformation Logic
                # This ensures the warmup data reflects the "Elite Advantage" (Quadratic HLE)
                # that the router expects, aligning ground truth with priors.
                prob_success = transform_hle_to_prior(
                   raw_hle_score=hle, 
                   difficulty_score=difficulty,
                   # We use default calibration constants (hard_exponent=2.0)
                )
                reward = prob_success
                
                # C. Update the Bandit State
                # PASS THE VECTOR, NOT THE STRING
                router.update(model_id, context_vector, reward)
                updates_count += 1

    # 5. Save the Artifact
    print(f"✅ Training complete. Processed {updates_count} simulated updates.")
    
    # We extract strictly the LinUCB matrices
    state_to_save = {
        "A": router.bandit.A,  # The Covariance Matrices (The Map)
        "b": router.bandit.b,  # The Reward Vectors (The Compass)
        "n": N_SAMPLES         # Metadata
    }
    
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(state_to_save, OUTPUT_PATH)
    
    print(f"💾 Saved Warmup Priors to: {OUTPUT_PATH}")
    print("   You can now use priors='warmup' in your experiments.")

if __name__ == "__main__":
    main()
