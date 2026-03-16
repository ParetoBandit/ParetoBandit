"""Synthetic prompt generation for PCA calibration and procedural warmup.

Template patterns and fill values are aligned with the procedural warmup
archetypes to ensure consistency between the PCA manifold and the warmup
covariance structure.
"""

from __future__ import annotations

import random
from typing import Dict, List

# ---------------------------------------------------------------------------
# Archetype templates
# ---------------------------------------------------------------------------

ARCHETYPE_TEMPLATES: Dict[str, List[str]] = {
    "math": [
        "Solve the integral of {expr} with respect to {var}",
        "Prove that {theorem} using mathematical induction",
        "Find the derivative of {function} and explain each step",
        "Calculate the eigenvalues of the matrix {matrix}",
        "Determine if the series {series} converges or diverges",
    ],
    "coding": [
        "Write a Python function to {task} using {library}",
        "Implement {algorithm} in {language} with time complexity analysis",
        "Debug this {language} code that {problem}",
        "Create a {language} class for {task} with unit tests",
        "Optimize this {algorithm} implementation for {constraint}",
    ],
    "reasoning": [
        "Analyze the logical structure of {argument} and identify fallacies",
        "Develop a step-by-step solution for {problem}",
        "Compare and contrast {concept_a} with {concept_b}",
        "Explain the causal relationship between {cause} and {effect}",
        "Evaluate the validity of {claim} given {evidence}",
    ],
    "creative": [
        "Write a {genre} story about {topic} in {style}",
        "Compose a poem about {subject} using {form}",
        "Create a dialogue between {character_a} and {character_b} about {topic}",
        "Describe {scene} from the perspective of {viewpoint}",
        "Develop a plot outline for a {genre} involving {element}",
    ],
    "chat": [
        "What is {simple_concept} and why is it important?",
        "Can you explain {topic} in simple terms?",
        "Tell me about {subject}",
        "Why does {phenomenon} happen?",
        "What's the difference between {concept_a} and {concept_b}?",
    ],
}

PLACEHOLDER_VALUES: Dict[str, List[str]] = {
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
    "evidence": ["the data", "experimental results", "historical records"],
    "genre": ["science fiction", "mystery", "romance", "thriller"],
    "topic": ["time travel", "AI", "space exploration", "ancient civilizations"],
    "style": ["Hemingway's style", "a humorous tone", "dark and moody"],
    "subject": ["autumn", "technology", "love", "nature"],
    "form": ["haiku", "sonnet", "free verse"],
    "character_a": ["a scientist", "an AI", "a detective"],
    "character_b": ["a philosopher", "a child", "a criminal"],
    "scene": ["a futuristic city", "a quiet forest", "a busy marketplace"],
    "viewpoint": ["a bird", "an alien observer", "a time traveler"],
    "element": ["time loops", "parallel universes", "mind reading"],
    "simple_concept": ["photosynthesis", "gravity", "democracy"],
    "phenomenon": ["rain", "lightning", "the aurora borealis"],
}


def generate_synthetic_prompts(n: int = 1000, seed: int = 42) -> List[str]:
    """Generate *n* synthetic prompts by sampling archetype templates.

    Each prompt is built by choosing a random archetype (math, coding,
    reasoning, creative, chat), selecting a template, and filling
    placeholders with randomly sampled values.

    Args:
        n: Number of prompts to generate.  For robust PCA fitting,
            use at least 10x the target dimensionality (e.g. 320 for
            32 PCA components).
        seed: RNG seed for reproducibility.

    Returns:
        List of synthetic prompt strings.
    """
    rng = random.Random(seed)
    archetype_keys = list(ARCHETYPE_TEMPLATES.keys())
    prompts: List[str] = []

    for _ in range(n):
        archetype = rng.choice(archetype_keys)
        template = rng.choice(ARCHETYPE_TEMPLATES[archetype])
        prompt = template
        for placeholder, values in PLACEHOLDER_VALUES.items():
            token = f"{{{placeholder}}}"
            if token in prompt:
                prompt = prompt.replace(token, rng.choice(values))
        prompts.append(prompt)

    return prompts
