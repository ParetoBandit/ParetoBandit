
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

# Define Scenarios and Prompts
scenarios = [
    {
        "name": "Simple Query (Easy)",
        "prompt": "Write a python function to print 'Hello World'.",
        "models": {
            "BanditGPT": "mistralai/ministral-3b",
            "LiteLLM": "google/gemini-2.0-flash-001",
            "RouteLLM": "openai/gpt-4o",
        }
    },
    {
        "name": "Standard Logic (Mid)",
        "prompt": "Solve for x: 3x + 5 = 20. Explain your steps.",
        "models": {
            "BanditGPT": "qwen/qwq-32b",
            "LiteLLM": "google/gemini-2.0-flash-001",
            "RouteLLM": "openai/gpt-4o",
        }
    },
    {
        "name": "Complex Reasoning (Hard)",
        "prompt": """You are an expert in regulating heat networks and you need to decide on the supply temperature of the heat exchange station based on the known information:
Current time: 2023-01-20 12:03;
Predicted heat load for the next hour: 33 w/m2;
Weather forecast for the next hour: 13 °C;
current supply water temperature: 48°C;
You need to decide the water supply temperature to ensure that 80% of your customers have a room temperature between 18 and 22°C. Provide a specific temperature and justify it briefly.""",
        "models": {
            "BanditGPT": "deepseek/deepseek-chat-v3-0324",
            "LiteLLM": "google/gemini-2.0-flash-001",
            "FrugalGPT": "openai/gpt-4o",
        }
    }
]

def main():
    if not OPENROUTER_API_KEY:
        print("Error: OPENROUTER_API_KEY not found in .env")
        return

    all_results = {}
    
    for scen in scenarios:
        print(f"\n{'='*80}")
        print(f"SCENARIO: {scen['name']}")
        print(f"PROMPT: {scen['prompt']}")
        print(f"{'='*80}")
        
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

    # Save results for artifact
    with open("table_3_live_verification.json", "w") as f:
        json.dump(all_results, f, indent=2)

if __name__ == "__main__":
    main()
