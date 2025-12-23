import json
import os
from pathlib import Path

def finalize_manual_updates():
    project_root = Path("/Users/annette/repostitories/llm_jury")
    models_path = project_root / "final_release/models.json"
    
    # Load models
    with open(models_path, "r") as f:
        models_data = json.load(f)
    
    # Manual Rates Provided by User
    manual_rates = {
        "Gemini 3 Pro Preview": 0.88,
        "GPT-5": 0.81,
        "Qwen3 Max": 0.89,
        "DeepSeek V3.2": 0.91,
        "Nova 2.0 Pro Preview": 0.90,
        "Llama Nemotron Ultra": 0.81,
        "Grok 4.1 Fast": 0.72,
        "Claude 4.5 Sonnet": 0.48,
        "Claude 4.5 Haiku": 0.26,
        "gpt-oss-120B": 0.90,
        "gpt-oss-20B": 0.93,
        "GPT-4o (Nov '24)": 0.38,
        "o4-mini": 0.79,
        "Llama 3.1 Instruct 70B": 0.78,
        "Llama 3.1 Instruct 405B": 0.51,
        "Llama 3.2 Instruct 1B": 0.66,
        "DeepSeek R1 0528": 0.83
    }
    
    updated_count = 0
    models = models_data.get("models", [])
    
    # Update existing entries
    for model in models:
        d_name = model.get("display_name", "")
        for m_name, rate in manual_rates.items():
            if m_name.lower() in d_name.lower():
                # ONLY update hallucination_rate, never touch hle
                model["hallucination_rate"] = round(rate * 100, 2)
                print(f"Updated: {d_name} -> Hallucination {rate*100}%")
                updated_count += 1
                break

    # Save results
    with open(models_path, "w") as f:
        json.dump(models_data, f, indent=2)
    
    print(f"Successfully finalized {updated_count} manual updates.")

if __name__ == "__main__":
    finalize_manual_updates()
