import numpy as np
import os
import logging
from pathlib import Path
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from dotenv import load_dotenv
from bandit_gpt.config_legacy import DEFAULT_SENTENCE_TRANSFORMER

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

def generate_complexity_vector(output_path: str = "../../priors/complexity_vector.npz"):
    """
    Generates the 'Gold Standard' Reference Complexity Vector (H-vector).
    
    Architecture:
    - Use a Contrastive Axis: H = UnitVector(Mean(Hard) - Mean(Easy))
    - Hard: 120+ samples from GSM8k (Math), MBPP (Code), and Curated Technical Reasoning.
    - Easy: 120+ samples from simple conversational datasets and common greetings.
    """
    logger.info("=== Generating Final Reference Complexity Vector ===")
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
    
    hard_prompts = []
    easy_prompts = []
    
    # 1. FETCH HARD PROMPTS (Target: 120+)
    try:
        # GSM8k (Math Reasoning) - reduced to balance
        logger.info("Fetching GSM8k samples...")
        ds = load_dataset("gsm8k", "main", split="train", streaming=True)
        it = iter(ds)
        for _ in range(30):
            hard_prompts.append(next(it)['question'])
            
        # MBPP (Coding) - reduced to balance
        logger.info("Fetching MBPP samples...")
        ds = load_dataset("mbpp", split="train", streaming=True)
        it = iter(ds)
        for _ in range(30):
            hard_prompts.append(next(it)['text'])
            
        # Curated Technical Reasoning (Calculus, Physics, Logic, Systems) - expanded to 40
        pro_seeds = [
            # Calculus & Analysis (12)
            "Calculate the indefinite integral of sin(x)*exp(-x).",
            "Find the limit as x approaches infinity of (1 + 1/x)^x.",
            "Compute the Fourier transform of the Gaussian function.",
            "Solve the differential equation d²y/dx² + 4y = 0.",
            "Evaluate the integral of 1/(1+x²) from 0 to infinity.",
            "Determine the convergence of the series sum(1/n²) from n=1 to infinity.",
            "Calculate the partial derivative of f(x,y) = x²y + sin(xy).",
            "Find the Taylor series expansion of ln(1+x) around x=0.",
            "Compute the line integral of F along the curve C.",
            "Determine if the function f(x) = |x| is differentiable at x=0.",
            "Find the critical points of f(x,y) = x³ - 3xy + y².",
            "Evaluate the double integral of e^(-x²-y²) over the entire plane.",
            
            # Physics & Engineering (8)
            "Derive the field equations of general relativity.",
            "Analyze the quantum mechanical wavefunction for a particle in a box.",
            "Calculate the Lorentz transformation for time dilation.",
            "Solve the Navier-Stokes equations for incompressible flow.",
            "Determine the eigenvalues of the Hamiltonian operator.",
            "Derive the Schrödinger equation from first principles.",
            "Calculate the electric field of a uniformly charged sphere.",
            "Analyze the thermodynamic efficiency of a Carnot engine.",
            
            # Computer Science Theory (10)
            "Prove the halting problem is undecidable.",
            "Implement a lock-free concurrent hash map in Rust.",
            "Analyze the amortized time complexity of dynamic array resizing.",
            "Design a globally distributed Raft-based consensus algorithm.",
            "Compare Dijkstra vs A* pathfinding on sparse graphs.",
            "Debug a race condition in a multi-threaded memory allocator.",
            "Prove that SAT is NP-complete using Cook's theorem.",
            "Implement a persistent red-black tree with path copying.",
            "Analyze the space-time tradeoff in Merkle tree construction.",
            "Design a Byzantine Fault Tolerant state machine replication protocol.",
            
            # Pure Mathematics (10)
            "Explain the mathematical proof of the Prime Number Theorem.",
            "Prove that the set of real numbers is uncountable using Cantor's diagonalization.",
            "Demonstrate Gödel's first incompleteness theorem.",
            "Prove Fermat's Last Theorem for n=3.",
            "Show that every finite group of prime order is cyclic.",
            "Prove the Banach fixed-point theorem.",
            "Demonstrate the fundamental theorem of algebra.",
            "Prove that π is irrational using continued fractions.",
            "Show that the Riemann hypothesis implies the Prime Number Theorem.",
            "Prove the Cauchy-Schwarz inequality in n-dimensional space."
        ]
        hard_prompts.extend(pro_seeds)
        logger.info(f"✓ Collected {len(hard_prompts)} HARD prompts.")
    except Exception as e:
        logger.warning(f"Hard data fetch partially failed: {e}")

    # 2. FETCH EASY PROMPTS (Target: 120+)
    try:
        # Using simple snippets from a large corpus to define 'regular English'
        logger.info("Fetching conversational baseline (TinyStories/Chat)...")
        # TinyStories is great for very simple, generic English
        ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
        it = iter(ds)
        for _ in range(100):
            # Take just the first sentence of the story as a 'prompt'
            story = next(it)['text']
            first_sentence = story.split('.')[0] + "."
            if len(first_sentence) > 10:
                easy_prompts.append(first_sentence[:150])
            
        # Add basic greetings and commands
        greetings = [
            "Hello", "Hi there", "How are you?", "What time is it?", "Tell me a joke",
            "What's the weather today?", "Nice to meet you!", "Good morning", "Bye",
            "Thanks for the help", "Can you help me with a quick question?", "Yes",
            "No thank you", "Repeat that please", "I am happy", "The sky is blue"
        ]
        easy_prompts.extend(greetings)
        logger.info(f"✓ Collected {len(easy_prompts)} EASY prompts.")
    except Exception as e:
        logger.warning(f"Easy data fetch partially failed: {e}")
        if not easy_prompts:
            easy_prompts = ["Hello", "Hi", "How are you?", "What's the weather?", "Help me."]

    # 3. COMPUTE AXIS
    logger.info("Embedding prompts and computing contrastive axis...")
    hard_embs = encoder.encode(hard_prompts, normalize_embeddings=True)
    easy_embs = encoder.encode(easy_prompts, normalize_embeddings=True)
    
    # H-Axis = Mean(Hard) - Mean(Easy)
    # This isolates the technical complexity while removing 'common text' features
    h_axis = np.mean(hard_embs, axis=0) - np.mean(easy_embs, axis=0)
    h_axis /= (np.linalg.norm(h_axis) + 1e-12)
    
    # 4. SAVE
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(out_file, complexity_vector=h_axis, metadata={
        "samples_hard": len(hard_prompts),
        "samples_easy": len(easy_prompts),
        "method": "contrastive_axis_v4",
        "timestamp": "2026-01-04"
    })
    
    logger.info(f"✓ Reference Complexity Vector saved to {out_file}")
    return h_axis

if __name__ == "__main__":
    generate_complexity_vector()
