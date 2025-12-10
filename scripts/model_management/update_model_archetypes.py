import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from llm_jury.core.models import ProductArchetype

def classify_model(model: Dict[str, Any]) -> str:
    """
    Classify a model into a ProductArchetype based on its metrics.
    Returns the string value of the archetype.
    """
    name = model.get("name", "").lower()
    slug = model.get("slug", "").lower()
    
    # Metrics (handle None)
    math_index = model.get("math_index") or 0
    coding_index = model.get("coding_index") or 0
    intelligence_index = model.get("intelligence_index") or 0
    mmlu_pro = model.get("mmlu_pro") or 0
    gpqa = model.get("gpqa") or 0
    
    price_blended = model.get("price_1m_blended") or 0
    context_window = model.get("context_window_k") or 0
    speed = model.get("output_tokens_per_second") or 0
    
    # 1. REASONING_SPECIALIST
    # Explicit naming or extremely high reasoning scores
    if "reasoning" in name or "thinking" in name or "r1" in slug or "o1" in slug:
        return ProductArchetype.REASONING_SPECIALIST.value
    
    if math_index > 85 or gpqa > 0.75:
        return ProductArchetype.REASONING_SPECIALIST.value

    # 2. FRONTIER
    # High general intelligence, expensive or flagship
    if intelligence_index > 60 or mmlu_pro > 0.70:
        return ProductArchetype.FRONTIER.value

    # 3. BULK_OPS
    # Cheap and Fast
    # Price < $1.00 blended AND Speed > 70 tps
    if price_blended < 1.0 and speed > 70:
        return ProductArchetype.BULK_OPS.value
        
    # 4. RAG_SPECIALIST
    # Large context, reasonable price, decent intelligence
    if context_window >= 128 and price_blended < 5.0 and intelligence_index > 30:
        return ProductArchetype.RAG_SPECIALIST.value

    # Default fallback (usually to Bulk or RAG depending on context)
    if price_blended < 2.0:
        return ProductArchetype.BULK_OPS.value
    
    return ProductArchetype.RAG_SPECIALIST.value

def main():
    cache_path = Path("data/models_cache.json")
    if not cache_path.exists():
        print(f"Error: Cache file not found at {cache_path}")
        return

    print(f"Reading cache from {cache_path}...")
    with open(cache_path, "r") as f:
        models = json.load(f)

    updated_count = 0
    for model in models:
        current_archetype = model.get("archetype")
        new_archetype = classify_model(model)
        
        # Update the model
        model["archetype"] = new_archetype
        
        # Print changes for verification
        if current_archetype != new_archetype:
            print(f"[{model['name']}] -> {new_archetype}")
            updated_count += 1

    print(f"\nUpdated {updated_count} models.")
    
    print(f"Saving to {cache_path}...")
    with open(cache_path, "w") as f:
        json.dump(models, f, indent=2)
    print("Done.")

if __name__ == "__main__":
    main()
