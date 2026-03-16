import os
import json
from flask import Flask, request, jsonify
from pathlib import Path

from pareto_bandit.router import BanditRouter

app = Flask(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
STATE_DIR = PROJECT_ROOT / "data"
STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_PATH = STATE_DIR / "bandit_state.npz"

router = BanditRouter.create(
    model_registry=None,
    state_path=STATE_PATH if STATE_PATH.exists() else None,
)

# --- Provider setup ---
# Build an LLMClient from environment variables.  Set PROVIDER to one of:
#   openrouter, openai, anthropic, gemini, ollama
_PROVIDER = os.environ.get("PROVIDER", "openrouter")
_API_KEY = os.environ.get("LLM_API_KEY", "")

llm_client = None  # lazy-init on first request


def _get_client():
    global llm_client
    if llm_client is not None:
        return llm_client
    from pareto_bandit.providers import (
        OpenRouterClient, OpenAIClient, AnthropicClient,
        GeminiClient, OllamaClient,
    )
    builders = {
        "openrouter": lambda: OpenRouterClient(api_key=_API_KEY),
        "openai": lambda: OpenAIClient(api_key=_API_KEY),
        "anthropic": lambda: AnthropicClient(api_key=_API_KEY),
        "gemini": lambda: GeminiClient(api_key=_API_KEY),
        "ollama": lambda: OllamaClient(),
    }
    llm_client = builders[_PROVIDER]()
    return llm_client


@app.route('/chat/completions', methods=['POST'])
def chat_endpoint():
    data = request.json

    if 'messages' in data:
        prompt = data['messages'][-1]['content']
    else:
        prompt = data.get('prompt', '')

    user_profile = data.get('profile', 'smart_shopper')

    model_id, log = router.route(prompt=prompt, profile=user_profile)

    client = _get_client()
    response_text = client.complete(
        model_id,
        [{"role": "user", "content": prompt}],
    )

    return jsonify({
        "id": "chatcmpl-" + log.request_id,
        "object": "chat.completion",
        "model": model_id,
        "choices": [{
            "message": {"role": "assistant", "content": response_text},
            "finish_reason": "stop"
        }],
        "system_fingerprint": log.request_id,
    })

@app.route('/feedback', methods=['POST'])
def feedback_endpoint():
    """
    Expects: { "request_id": "...", "outcome": "success" }
    """
    data = request.json
    
    # Use the Request ID to link feedback to the specific context vector (x)
    request_id = data.get('request_id') or data.get('system_fingerprint')
    outcome_type = data.get('outcome')
    
    if not request_id:
        return jsonify({"error": "Missing request_id"}), 400
    
    # Normalize Feedback (Binary Reward)
    # 1.0 = Success (Copy/Paste), 0.0 = Failure (Regenerate)
    reward = 1.0 if outcome_type in ['success', 'copy', 'thumbs_up'] else 0.0
    
    # 3. Update Bandit State (Closes the Loop)
    router.process_feedback(request_id, reward)
    
    # Optional: Periodically save state to disk
    router.save_state(STATE_PATH)
    
    return jsonify({"status": "ack", "reward": reward})

if __name__ == '__main__':
    print("🚀 Bandit Feedback Loop running on :5005")
    app.run(debug=True, port=5005)
