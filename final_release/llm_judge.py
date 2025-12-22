import os
import re
import json
import requests
from typing import Optional, Dict, Any, Tuple

class LLMJudge:
    """
    A lightweight, zero-config LLM Judge for BanditRouter.
    
    Uses standard HTTP requests to avoid adding 'openai' dependency.
    Default: Uses OpenRouter (compatible with OpenAI API).
    """
    
    def __init__(
        self, 
        model: str = "openai/gpt-4o",
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1"
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            print("WARNING: LLMJudge initialized without API Key. Grading will fail.")

    def grade(self, prompt: str, response: str) -> float:
        """
        Grade a response on a scale of 0.0 to 1.0 for correctness.
        
        Args:
            prompt: The user's original query.
            response: The model's response to grade.
            
        Returns:
            float: Score between 0.0 (Wrong) and 1.0 (Correct).
        """
        if not self.api_key:
            raise ValueError("No API key provided for LLMJudge. Set OPENROUTER_API_KEY.")

        system_prompt = (
            "You are an impartial judge. Rate the correctness of the response to the prompt.\n"
            "Output ONLY a single float number between 0.0 and 1.0.\n"
            "0.0 = Completely wrong or harmful.\n"
            "0.5 = Partially correct but missing key details.\n"
            "1.0 = Perfectly correct and helpful.\n"
            "Do not output any other text."
        )
        
        user_content = f"PROMPT: {prompt}\n\nRESPONSE: {response}"
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "temperature": 0.0,
            "max_tokens": 10
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/banditgpt/llm-jury",
        }
        
        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=10
            )
            resp.raise_for_status()
            data = resp.json()
            
            content = data["choices"][0]["message"]["content"].strip()
            score = self._parse_score(content)
            if score == 0.5:
                print(f"DEBUG: Failed to parse score from: '{content}'")
            return score
            
        except Exception as e:
            print(f"Judge Error: {e}")
            if 'data' in locals():
                print(f"DEBUG: Response data: {data}")
            return 0.5 # Fallback to neutral on error

    def _parse_score(self, text: str) -> float:
        """Extract float from text."""
        print(f"DEBUG: Raw LLM Output: '{text}'")
        try:
            # Look for number
            match = re.search(r"(\d+(\.\d+)?)", text)
            if match:
                val = float(match.group(1))
                return max(0.0, min(1.0, val))
        except:
            pass
        return 0.5
