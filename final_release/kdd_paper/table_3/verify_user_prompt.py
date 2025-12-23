
import os
import json
import requests
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
env_path = Path(".env")
load_dotenv(dotenv_path=env_path)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

def call_openrouter(model, prompt):
    # Determine timeout based on model type
    is_reasoning = "qwq" in model.lower() or "deepseek" in model.lower()
    timeout = 300 if is_reasoning else 80
    
    print(f"Calling {model} (timeout: {timeout}s)...")
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/atabernermiller/banditgpt",
        "X-Title": "BanditGPT Verification",
    }
    
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
    }
    
    try:
        response = requests.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers=headers,
            data=json.dumps(data),
            timeout=timeout
        )
        if response.status_code == 408:
            return "Error: Request Timeout from OpenRouter (408)"
        response.raise_for_status()
        result = response.json()
        if 'choices' not in result or not result['choices']:
             return f"Error: No choices in response. Raw: {result}"
        return result['choices'][0]['message']['content']
    except requests.exceptions.Timeout:
        return f"Error: Global Python Timeout ({timeout}s reached)"
    except Exception as e:
        return f"Error: {str(e)}"

# Define the NEW HARD PROMPT
new_scenarios = [
    {
        "name": "User-Provided Metrics Prompt (Mid-Hard)",
        "prompt": """You are given a binary classifier evaluated on a test set of 1,000 samples.
The classifier predicted “positive” on 260 samples.
Of those predicted positives, 195 were actually positive.
In total, there are 240 actual positive samples in the dataset.
Questions:
How many true positives (TP), false positives (FP), true negatives (TN), and false negatives (FN) did the classifier produce?
What are the precision, recall, and F1 score of the classifier?
Provide the values of TP, FP, TN, FN, precision, recall, and F1 score, each rounded to two decimal places where applicable, with no intermediate explanation.""",
        "models": {
            "BanditGPT": "deepseek/deepseek-v3.1-terminus",
            "LiteLLM": "google/gemini-2.0-flash-001",
            "RouteLLM": "openai/gpt-4o",
        },
        "ground_truth": {
            "TP": "195",
            "FP": "65",
            "TN": "695",
            "FN": "45",
            "Precision": "0.75",
            "Recall": "0.81",
            "F1": "0.78"
        }
    }
]

def main():
    if not OPENROUTER_API_KEY:
        print("Error: OPENROUTER_API_KEY not found in .env")
        return

    all_results = {}
    
    for scen in new_scenarios:
        print(f"\n{'='*80}")
        print(f"SCENARIO: {scen['name']}")
        print(f"PROMPT:\n{scen['prompt']}")
        print(f"{'='*80}")
        print(f"\nGROUND TRUTH EXPECTED:")
        for k, v in scen['ground_truth'].items():
            print(f" - {k}: {v}")
        
        scen_results = {}
        for router, model in scen['models'].items():
            print(f"\n--- {router} selected: {model} ---")
            response = call_openrouter(model, scen['prompt'])
            print(f"RESPONSE:\n{response}")
            scen_results[router] = {
                "model": model,
                "response": response
            }
        all_results[scen['name']] = scen_results

    # Save results
    with open("table_3_user_verification.json", "w") as f:
        json.dump(all_results, f, indent=2)

if __name__ == "__main__":
    main()
