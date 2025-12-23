import json
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Paths
MODELS_PATH = Path("/Users/annette/repostitories/llm_jury/final_release/models.json")
CACHE_PATH = Path("/Users/annette/repostitories/llm_jury/banditgpt/data/models_cache.json")

def calculate_harmonic_risk(v_risk, a_risk, beta=1.0):
    """
    Calculate Composite Risk by taking the Harmonic Mean of Truthfulness rates.
    This biases the result toward the 'worst' (highest risk) metric.
    
    Args:
        v_risk: Vectara Hallucination Rate (0-100)
        a_risk: AA-Omniscience Hallucination Rate (0-100)
        beta: Weighting factor (default 1.0)
        
    Returns:
        Composite Risk Rate (0-100)
    """
    # Convert risks to truthfulness (success) rates
    # Ensure values are within [0, 100] and avoid absolute zero for division
    t_v = max(0.001, 100.0 - v_risk)
    t_a = max(0.001, 100.0 - a_risk)
    
    # Harmonic Mean of Truthfulness: 
    # T_composite = (1 + beta^2) / (1/T_v + beta^2/T_a)
    t_composite = (1 + beta**2) / (1/t_v + beta**2/t_a)
    
    # Back to Risk
    return round(100.0 - t_composite, 2)

def main():
    if not MODELS_PATH.exists():
        logger.error(f"Registry not found at {MODELS_PATH}")
        return

    with open(MODELS_PATH, 'r') as f:
        data = json.load(f)

    updated_count = 0
    for model in data.get('models', []):
        v = model.get('hallucination_vectara')
        a = model.get('hallucination_rate') # AA-Omniscience
        
        if v is not None and a is not None:
            composite = calculate_harmonic_risk(v, a)
            model['hallucination_composite'] = composite
            updated_count += 1
        else:
            logger.warning(f"Skipping {model.get('display_name')}: Missing one or more hallucination metrics.")

    # Save to models.json
    with open(MODELS_PATH, 'w') as f:
        json.dump(data, f, indent=2)
    logger.info(f"Updated {updated_count} models in {MODELS_PATH}")

    # Save to models_cache.json
    if CACHE_PATH.exists():
        with open(CACHE_PATH, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Updated {updated_count} models in {CACHE_PATH}")
    else:
        logger.warning(f"Cache file not found at {CACHE_PATH}")

if __name__ == "__main__":
    main()
