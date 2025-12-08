#!/usr/bin/env python3
"""
Professional Visualization: Side-by-Side Model Response Comparison

Compares actual LLM responses between baseline (Gemini 3 Pro) and 
recommended value-optimized models, showing cost savings.

Uses OpenRouter API to get real responses.
"""

import os
import json
import requests
import textwrap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_jury.ranking.quality_scorer import QualityScorer
from llm_jury.core.models import PromptCategory

load_dotenv()

# Consistent color scheme with other plots
COLORS = {
    'bg': '#0a0e17',
    'panel': '#131a2a',
    'grid': '#1e2738',
    'text': '#e8eaed',
    'muted': '#7a8599',
    'accent': '#22c55e',
    'gold': '#ffd93d',
    'baseline': '#ff6b6b',
    'recommended': '#22c55e',
    'partial': '#f59e0b',
    'savings': '#10b981',
}


@dataclass
class ModelResponse:
    """Response from a model via OpenRouter."""
    model_name: str
    model_id: str
    response_text: str
    input_tokens: int
    output_tokens: int
    cost: float  # Total cost in dollars
    latency_ms: float


class OpenRouterClient:
    """Client for getting LLM responses via OpenRouter."""
    
    CHAT_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
    MODELS_ENDPOINT = "https://openrouter.ai/api/v1/models"
    
    # Model ID mappings for our cache names to OpenRouter IDs
    MODEL_MAPPINGS = {
        "Gemini 3 Pro Preview (high)": "google/gemini-3-pro-preview",
        "GPT-5 mini (high)": "openai/gpt-5-mini",
        "GPT-5.1 (high)": "openai/gpt-5.1",
        "Grok 4.1 Fast (Reasoning)": "x-ai/grok-4-fast",
        "Grok 4 Fast (Reasoning)": "x-ai/grok-4-fast",
        "gpt-oss-120B (high)": "openai/gpt-oss-120b",
        "DeepSeek V3.1 (Reasoning)": "deepseek/deepseek-chat-v3.1",
        "DeepSeek V3.1 Terminus (Reasoning)": "deepseek/deepseek-v3.1-terminus",
        "Gemini 2.5 Flash Preview (Sep '25) (Reasoning)": "google/gemini-2.5-flash-preview-09-2025",
        "Gemini 2.5 Flash (Reasoning)": "google/gemini-2.5-flash",
        "Mistral Small 3.2": "mistralai/mistral-small-3.2-24b-instruct",
        "Llama 4 Maverick": "nvidia/llama-3.3-nemotron-super-49b-v1.5",
        "Claude 4.5 Sonnet (Reasoning)": "anthropic/claude-sonnet-4.5",
        "Grok 3 mini Reasoning (high)": "x-ai/grok-3-mini",
    }
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not set in environment")
        self._models_cache = None
    
    def get_available_models(self) -> Dict[str, dict]:
        """Get available models and their pricing."""
        if self._models_cache is None:
            try:
                response = requests.get(self.MODELS_ENDPOINT, timeout=30)
                response.raise_for_status()
                data = response.json()
                self._models_cache = {m['id']: m for m in data.get('data', [])}
            except Exception as e:
                print(f"Failed to fetch models: {e}")
                self._models_cache = {}
        return self._models_cache
    
    def get_model_id(self, our_model_name: str) -> Optional[str]:
        """Map our model name to OpenRouter model ID."""
        # Direct mapping
        if our_model_name in self.MODEL_MAPPINGS:
            return self.MODEL_MAPPINGS[our_model_name]
        
        # Try to find by name matching
        models = self.get_available_models()
        name_lower = our_model_name.lower()
        
        for model_id, model_info in models.items():
            model_name = model_info.get('name', '').lower()
            if name_lower in model_name or model_name in name_lower:
                return model_id
        
        return None
    
    def get_response(self, model_name: str, prompt: str, max_tokens: int = 4000) -> Optional[ModelResponse]:
        """Get a response from a model."""
        model_id = self.get_model_id(model_name)
        if not model_id:
            print(f"Could not find model ID for: {model_name}")
            return None
        
        models = self.get_available_models()
        model_info = models.get(model_id, {})
        
        try:
            import time
            start_time = time.time()
            
            response = requests.post(
                self.CHAT_ENDPOINT,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://llm-jury.com",
                    "X-Title": "LLM Jury Comparison",
                },
                json={
                    "model": model_id,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                },
                timeout=180,  # Increased timeout for longer responses
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            if response.status_code != 200:
                print(f"API error for {model_name}: {response.status_code} - {response.text}")
                return None
            
            data = response.json()
            
            # Extract usage info
            usage = data.get('usage', {})
            input_tokens = usage.get('prompt_tokens', 0)
            output_tokens = usage.get('completion_tokens', 0)
            
            # Check finish reason for debugging
            choices = data.get('choices', [])
            if choices:
                finish_reason = choices[0].get('finish_reason', 'unknown')
                if finish_reason == 'length':
                    print(f"⚠️ {model_name} response truncated due to max_tokens limit")
                elif finish_reason != 'stop':
                    print(f"⚠️ {model_name} finish_reason: {finish_reason}")
            
            # Calculate cost
            pricing = model_info.get('pricing', {})
            input_cost = float(pricing.get('prompt', 0)) * input_tokens
            output_cost = float(pricing.get('completion', 0)) * output_tokens
            total_cost = input_cost + output_cost
            
            # Get response text
            response_text = choices[0]['message']['content'] if choices else ""
            
            print(f"   ✓ Got {output_tokens} tokens, {len(response_text)} chars")
            
            return ModelResponse(
                model_name=model_name,
                model_id=model_id,
                response_text=response_text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=total_cost,
                latency_ms=latency_ms,
            )
            
        except Exception as e:
            print(f"Error getting response from {model_name}: {e}")
            import traceback
            traceback.print_exc()
            return None


def wrap_text(text: str, width: int = 50) -> str:
    """Wrap text to specified width."""
    return '\n'.join(textwrap.wrap(text, width=width))


def extract_code_blocks(response_text: str) -> Tuple[List[str], str]:
    """
    Extract code blocks from a response.
    Returns (list of code blocks, brief summary text).
    """
    import re
    
    # Find all code blocks (```python ... ``` or ``` ... ```)
    code_pattern = r'```(?:python|py)?\s*\n(.*?)```'
    code_blocks = re.findall(code_pattern, response_text, re.DOTALL | re.IGNORECASE)
    
    # If no fenced blocks, try to find indented code or function definitions
    if not code_blocks:
        # Look for def statements
        def_pattern = r'(def\s+\w+\([^)]*\):[^\n]*(?:\n(?:[ \t]+[^\n]+|\s*$))*)'
        code_blocks = re.findall(def_pattern, response_text)
    
    # Clean up code blocks
    cleaned_blocks = []
    for block in code_blocks:
        # Strip leading/trailing whitespace but preserve internal indentation
        lines = block.strip().split('\n')
        if lines:
            # Find minimum indentation (excluding empty lines)
            min_indent = float('inf')
            for line in lines:
                if line.strip():
                    indent = len(line) - len(line.lstrip())
                    min_indent = min(min_indent, indent)
            
            if min_indent == float('inf'):
                min_indent = 0
            
            # Remove common indentation
            cleaned_lines = []
            for line in lines:
                if line.strip():
                    cleaned_lines.append(line[min_indent:])
                else:
                    cleaned_lines.append('')
            
            cleaned_blocks.append('\n'.join(cleaned_lines))
    
    # Extract a brief summary (first sentence or first line before code)
    summary = ""
    if code_blocks:
        # Get text before first code block
        first_code_pos = response_text.find('```')
        if first_code_pos > 0:
            pre_text = response_text[:first_code_pos].strip()
            # Get first sentence or first 100 chars
            sentences = re.split(r'[.!?]\s+', pre_text)
            if sentences:
                summary = sentences[0][:150]
                if len(sentences[0]) > 150:
                    summary += "..."
    
    return cleaned_blocks, summary


def format_response_for_display(response_text: str, max_code_lines: int = 35) -> Tuple[str, str]:
    """
    Format a model response for display in the plot.
    Returns (formatted_code, summary_text).
    """
    code_blocks, summary = extract_code_blocks(response_text)
    
    if code_blocks:
        # Take the first/main code block
        main_code = code_blocks[0]
        
        # Limit lines
        lines = main_code.split('\n')
        if len(lines) > max_code_lines:
            lines = lines[:max_code_lines]
            lines.append('    # ... (continued)')
        
        formatted_code = '\n'.join(lines)
        
        # If there are multiple code blocks, note that
        if len(code_blocks) > 1:
            summary += f"\n[+ {len(code_blocks)-1} more code block(s)]"
        
        return formatted_code, summary
    else:
        # No code found, return truncated text
        truncated = response_text[:800]
        if len(response_text) > 800:
            truncated += "\n\n... [response continues]"
        return "", truncated


def create_comparison_plot(
    prompt: str,
    baseline_response: ModelResponse,
    recommended_response: ModelResponse,
    optimization_config: Dict,
    output_path: str,
):
    """Create a side-by-side comparison plot."""
    
    plt.style.use('default')
    fig = plt.figure(figsize=(18, 18))  # Increased height for more response text
    fig.patch.set_facecolor(COLORS['bg'])
    
    # Calculate savings
    cost_savings = baseline_response.cost - recommended_response.cost
    cost_savings_pct = (cost_savings / baseline_response.cost * 100) if baseline_response.cost > 0 else 0
    latency_savings = baseline_response.latency_ms - recommended_response.latency_ms
    latency_savings_pct = (latency_savings / baseline_response.latency_ms * 100) if baseline_response.latency_ms > 0 else 0
    
    # =========================================================================
    # HEADER SECTION
    # =========================================================================
    
    # Main title
    fig.text(0.5, 0.96, '🔬 Model Response Comparison',
            fontsize=24, fontweight='bold', color=COLORS['text'], ha='center')
    fig.text(0.5, 0.93, f'Baseline vs Value-Optimized Recommendation',
            fontsize=14, color=COLORS['muted'], ha='center')
    
    # =========================================================================
    # PROMPT SECTION
    # =========================================================================
    
    prompt_ax = fig.add_axes([0.05, 0.82, 0.90, 0.08])
    prompt_ax.set_facecolor(COLORS['panel'])
    prompt_ax.axis('off')
    for spine in prompt_ax.spines.values():
        spine.set_visible(True)
        spine.set_color(COLORS['accent'])
        spine.set_linewidth(2)
    
    prompt_ax.text(0.5, 0.85, '📝 PROMPT', fontsize=12, fontweight='bold',
                  color=COLORS['accent'], ha='center', va='top', transform=prompt_ax.transAxes)
    
    wrapped_prompt = wrap_text(prompt, width=120)
    prompt_ax.text(0.5, 0.45, f'"{wrapped_prompt}"', fontsize=11, style='italic',
                  color=COLORS['text'], ha='center', va='center', transform=prompt_ax.transAxes,
                  linespacing=1.3)
    
    # =========================================================================
    # OPTIMIZATION CONFIG SECTION
    # =========================================================================
    
    config_ax = fig.add_axes([0.05, 0.74, 0.90, 0.06])
    config_ax.set_facecolor(COLORS['panel'])
    config_ax.axis('off')
    
    config_text = (
        f"⚙️ OPTIMIZATION: Quality ≥{optimization_config['quality']*100:.0f}%  |  "
        f"Cost ≤{optimization_config['cost']*100:.0f}%  |  "
        f"Latency ≤{optimization_config['latency']*100:.0f}%"
    )
    config_ax.text(0.5, 0.5, config_text, fontsize=12, fontweight='bold',
                  color=COLORS['gold'], ha='center', va='center', transform=config_ax.transAxes)
    
    # =========================================================================
    # SIDE-BY-SIDE RESPONSES
    # =========================================================================
    
    # LEFT: Baseline
    baseline_ax = fig.add_axes([0.05, 0.15, 0.44, 0.56])  # Taller panel for more text
    baseline_ax.set_facecolor(COLORS['panel'])
    baseline_ax.axis('off')
    for spine in baseline_ax.spines.values():
        spine.set_visible(True)
        spine.set_color(COLORS['baseline'])
        spine.set_linewidth(3)
    
    # Baseline header
    baseline_ax.text(0.5, 0.98, '⭐ BASELINE MODEL', fontsize=14, fontweight='bold',
                    color=COLORS['baseline'], ha='center', va='top', transform=baseline_ax.transAxes)
    baseline_ax.text(0.5, 0.94, baseline_response.model_name, fontsize=11,
                    color=COLORS['text'], ha='center', va='top', transform=baseline_ax.transAxes)
    
    # Baseline metrics
    baseline_ax.text(0.5, 0.89, f'Tokens: {baseline_response.input_tokens} in / {baseline_response.output_tokens} out  |  '
                    f'Cost: ${baseline_response.cost*1000:.4f}  |  Latency: {baseline_response.latency_ms:.0f}ms',
                    fontsize=9, color=COLORS['muted'], ha='center', va='top', transform=baseline_ax.transAxes)
    
    # Baseline response - extract and format code
    baseline_ax.plot([0.02, 0.98], [0.86, 0.86], color=COLORS['grid'], linewidth=1, transform=baseline_ax.transAxes)
    
    baseline_code, baseline_summary = format_response_for_display(baseline_response.response_text)
    
    if baseline_code:
        # Show summary first
        if baseline_summary:
            baseline_ax.text(0.03, 0.83, baseline_summary, fontsize=9,
                           color=COLORS['muted'], ha='left', va='top', transform=baseline_ax.transAxes,
                           style='italic', linespacing=1.2)
            code_start_y = 0.78
        else:
            code_start_y = 0.83
        
        # Show code with syntax-like styling
        baseline_ax.text(0.03, code_start_y, '📝 Python Code:', fontsize=9, fontweight='bold',
                        color=COLORS['accent'], ha='left', va='top', transform=baseline_ax.transAxes)
        
        # Add code background box
        code_bg = FancyBboxPatch((0.02, 0.03), 0.96, code_start_y - 0.08,
                                  boxstyle="round,pad=0.01,rounding_size=0.02",
                                  facecolor='#1a1a2e', edgecolor=COLORS['grid'],
                                  transform=baseline_ax.transAxes, zorder=1)
        baseline_ax.add_patch(code_bg)
        
        baseline_ax.text(0.03, code_start_y - 0.04, baseline_code, fontsize=10,
                        color='#98c379', ha='left', va='top', transform=baseline_ax.transAxes,
                        linespacing=1.25, family='monospace', zorder=2)
    else:
        # No code found, show text summary
        baseline_ax.text(0.03, 0.83, baseline_summary, fontsize=9,
                        color=COLORS['text'], ha='left', va='top', transform=baseline_ax.transAxes,
                        linespacing=1.3)
    
    # RIGHT: Recommended
    rec_ax = fig.add_axes([0.51, 0.15, 0.44, 0.56])  # Taller panel for more text
    rec_ax.set_facecolor(COLORS['panel'])
    rec_ax.axis('off')
    for spine in rec_ax.spines.values():
        spine.set_visible(True)
        spine.set_color(COLORS['recommended'])
        spine.set_linewidth(3)
    
    # Recommended header
    rec_ax.text(0.5, 0.98, '💰 RECOMMENDED MODEL', fontsize=14, fontweight='bold',
               color=COLORS['recommended'], ha='center', va='top', transform=rec_ax.transAxes)
    rec_ax.text(0.5, 0.94, recommended_response.model_name, fontsize=11,
               color=COLORS['text'], ha='center', va='top', transform=rec_ax.transAxes)
    
    # Recommended metrics
    rec_ax.text(0.5, 0.89, f'Tokens: {recommended_response.input_tokens} in / {recommended_response.output_tokens} out  |  '
               f'Cost: ${recommended_response.cost*1000:.4f}  |  Latency: {recommended_response.latency_ms:.0f}ms',
               fontsize=9, color=COLORS['muted'], ha='center', va='top', transform=rec_ax.transAxes)
    
    # Recommended response - extract and format code
    rec_ax.plot([0.02, 0.98], [0.86, 0.86], color=COLORS['grid'], linewidth=1, transform=rec_ax.transAxes)
    
    rec_code, rec_summary = format_response_for_display(recommended_response.response_text)
    
    if rec_code:
        # Show summary first
        if rec_summary:
            rec_ax.text(0.03, 0.83, rec_summary, fontsize=9,
                       color=COLORS['muted'], ha='left', va='top', transform=rec_ax.transAxes,
                       style='italic', linespacing=1.2)
            code_start_y = 0.78
        else:
            code_start_y = 0.83
        
        # Show code with syntax-like styling
        rec_ax.text(0.03, code_start_y, '📝 Python Code:', fontsize=9, fontweight='bold',
                   color=COLORS['accent'], ha='left', va='top', transform=rec_ax.transAxes)
        
        # Add code background box
        code_bg = FancyBboxPatch((0.02, 0.03), 0.96, code_start_y - 0.08,
                                  boxstyle="round,pad=0.01,rounding_size=0.02",
                                  facecolor='#1a1a2e', edgecolor=COLORS['grid'],
                                  transform=rec_ax.transAxes, zorder=1)
        rec_ax.add_patch(code_bg)
        
        rec_ax.text(0.03, code_start_y - 0.04, rec_code, fontsize=10,
                   color='#98c379', ha='left', va='top', transform=rec_ax.transAxes,
                   linespacing=1.25, family='monospace', zorder=2)
    else:
        # No code found, show text summary
        rec_ax.text(0.03, 0.83, rec_summary, fontsize=9,
                   color=COLORS['text'], ha='left', va='top', transform=rec_ax.transAxes,
                   linespacing=1.3)
    
    # =========================================================================
    # SAVINGS SECTION
    # =========================================================================
    
    savings_ax = fig.add_axes([0.05, 0.02, 0.90, 0.11])  # Adjusted for taller response panels
    savings_ax.set_facecolor(COLORS['panel'])
    savings_ax.axis('off')
    for spine in savings_ax.spines.values():
        spine.set_visible(True)
        spine.set_color(COLORS['savings'])
        spine.set_linewidth(2)
    
    # Savings header
    savings_ax.text(0.5, 0.92, '📊 COST & LATENCY SAVINGS', fontsize=14, fontweight='bold',
                   color=COLORS['savings'], ha='center', va='top', transform=savings_ax.transAxes)
    
    # Savings metrics in boxes
    # Cost savings
    savings_ax.add_patch(FancyBboxPatch((0.08, 0.25), 0.25, 0.55,
                         boxstyle="round,pad=0.02,rounding_size=0.02",
                         facecolor='#0f2318', edgecolor=COLORS['savings'],
                         linewidth=2, transform=savings_ax.transAxes))
    savings_ax.text(0.205, 0.7, 'COST SAVINGS', fontsize=10, fontweight='bold',
                   color=COLORS['savings'], ha='center', va='center', transform=savings_ax.transAxes)
    savings_ax.text(0.205, 0.45, f'{cost_savings_pct:.1f}%', fontsize=28, fontweight='bold',
                   color=COLORS['gold'], ha='center', va='center', transform=savings_ax.transAxes)
    savings_ax.text(0.205, 0.3, f'(${cost_savings*1000:.4f} per request)', fontsize=9,
                   color=COLORS['muted'], ha='center', va='center', transform=savings_ax.transAxes)
    
    # Latency change
    latency_color = COLORS['savings'] if latency_savings > 0 else COLORS['baseline']
    savings_ax.add_patch(FancyBboxPatch((0.38, 0.25), 0.25, 0.55,
                         boxstyle="round,pad=0.02,rounding_size=0.02",
                         facecolor='#0f2318', edgecolor=latency_color,
                         linewidth=2, transform=savings_ax.transAxes))
    savings_ax.text(0.505, 0.7, 'LATENCY', fontsize=10, fontweight='bold',
                   color=latency_color, ha='center', va='center', transform=savings_ax.transAxes)
    latency_sign = '+' if latency_savings < 0 else '-'
    savings_ax.text(0.505, 0.45, f'{latency_sign}{abs(latency_savings_pct):.1f}%', fontsize=28, fontweight='bold',
                   color=COLORS['gold'], ha='center', va='center', transform=savings_ax.transAxes)
    savings_ax.text(0.505, 0.3, f'({recommended_response.latency_ms:.0f}ms vs {baseline_response.latency_ms:.0f}ms)', fontsize=9,
                   color=COLORS['muted'], ha='center', va='center', transform=savings_ax.transAxes)
    
    # Value summary
    savings_ax.add_patch(FancyBboxPatch((0.68, 0.25), 0.25, 0.55,
                         boxstyle="round,pad=0.02,rounding_size=0.02",
                         facecolor='#1a1a2e', edgecolor=COLORS['gold'],
                         linewidth=2, transform=savings_ax.transAxes))
    savings_ax.text(0.805, 0.7, 'VALUE SCORE', fontsize=10, fontweight='bold',
                   color=COLORS['gold'], ha='center', va='center', transform=savings_ax.transAxes)
    value_score = (100 - cost_savings_pct) / 100  # Lower cost = higher value
    savings_ax.text(0.805, 0.45, f'{(1/value_score):.1f}x', fontsize=28, fontweight='bold',
                   color=COLORS['gold'], ha='center', va='center', transform=savings_ax.transAxes)
    savings_ax.text(0.805, 0.3, 'better value', fontsize=9,
                   color=COLORS['muted'], ha='center', va='center', transform=savings_ax.transAxes)
    
    plt.savefig(output_path, dpi=200, facecolor=COLORS['bg'], edgecolor='none', bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()


def create_mock_comparison_plot(
    prompt: str,
    baseline_name: str,
    recommended_name: str,
    optimization_config: Dict,
    output_path: str,
):
    """Create a comparison plot with mock data (when API is not available)."""
    
    # Mock responses for demonstration
    baseline_response = ModelResponse(
        model_name=baseline_name,
        model_id="google/gemini-2.5-pro",
        response_text="Here's a Python function to calculate the Fibonacci sequence:\n\n```python\ndef fibonacci(n):\n    if n <= 0:\n        return []\n    elif n == 1:\n        return [0]\n    elif n == 2:\n        return [0, 1]\n    \n    fib = [0, 1]\n    for i in range(2, n):\n        fib.append(fib[i-1] + fib[i-2])\n    return fib\n```\n\nThis function takes an integer n and returns a list containing the first n Fibonacci numbers. The time complexity is O(n) and space complexity is O(n).",
        input_tokens=45,
        output_tokens=120,
        cost=0.000825,  # $4.50/M * 0.75 + $15/M * 0.25 for output
        latency_ms=1250,
    )
    
    # Recommended model - lower cost
    cost_ratio = optimization_config['cost']
    recommended_response = ModelResponse(
        model_name=recommended_name,
        model_id="deepseek/deepseek-chat",
        response_text="Here's an efficient Fibonacci implementation:\n\n```python\ndef fibonacci(n):\n    if n <= 0:\n        return []\n    fib = [0, 1]\n    while len(fib) < n:\n        fib.append(fib[-1] + fib[-2])\n    return fib[:n]\n```\n\nThis returns the first n Fibonacci numbers using dynamic programming with O(n) time and space complexity.",
        input_tokens=45,
        output_tokens=95,
        cost=baseline_response.cost * cost_ratio,
        latency_ms=baseline_response.latency_ms * optimization_config['latency'],
    )
    
    create_comparison_plot(prompt, baseline_response, recommended_response, optimization_config, output_path)


def run_live_comparison(
    prompt: str,
    baseline_name: str,
    recommended_name: str,
    optimization_config: Dict,
    output_path: str,
):
    """Run a live comparison using OpenRouter API."""
    
    try:
        client = OpenRouterClient()
        
        print(f"🔄 Getting response from baseline: {baseline_name}")
        baseline_response = client.get_response(baseline_name, prompt)
        
        if not baseline_response:
            print("❌ Failed to get baseline response, using mock data")
            create_mock_comparison_plot(prompt, baseline_name, recommended_name, optimization_config, output_path)
            return
        
        print(f"🔄 Getting response from recommended: {recommended_name}")
        recommended_response = client.get_response(recommended_name, prompt)
        
        if not recommended_response:
            print("❌ Failed to get recommended response, using mock data")
            create_mock_comparison_plot(prompt, baseline_name, recommended_name, optimization_config, output_path)
            return
        
        create_comparison_plot(prompt, baseline_response, recommended_response, optimization_config, output_path)
        
    except ValueError as e:
        print(f"⚠️ API not configured: {e}")
        print("📋 Creating plot with mock data for demonstration")
        create_mock_comparison_plot(prompt, baseline_name, recommended_name, optimization_config, output_path)


def main():
    """Generate comparison plots with different optimization configurations."""
    
    print("=" * 70)
    print("MODEL RESPONSE COMPARISON VISUALIZATIONS")
    print("=" * 70)
    
    baseline = "Gemini 3 Pro Preview (high)"
    
    # Test prompts
    prompts = [
        {
            'text': "Write a Python function to calculate the Fibonacci sequence efficiently.",
            'name': 'coding',
        },
        {
            'text': "Explain the concept of machine learning to a 10-year-old.",
            'name': 'explanation',
        },
    ]
    
    # Different optimization configurations to demonstrate value trade-offs
    configs = [
        # Aggressive cost savings
        {'quality': 0.80, 'cost': 0.20, 'latency': 1.0, 'recommended': 'Mistral Small 3.2'},
        # Higher quality threshold
        {'quality': 0.90, 'cost': 0.30, 'latency': 1.0, 'recommended': 'DeepSeek V3.1 (Reasoning)'},
        # Balanced approach
        {'quality': 0.85, 'cost': 0.50, 'latency': 0.80, 'recommended': 'Grok 4 Fast (Reasoning)'},
    ]
    
    # Generate multiple comparison plots
    for prompt_info in prompts[:1]:  # Just coding for now
        for config in configs:
            print(f"\n📝 Prompt: {prompt_info['text'][:50]}...")
            print(f"⚙️ Config: Q≥{config['quality']*100:.0f}%, C≤{config['cost']*100:.0f}%, L≤{config['latency']*100:.0f}%")
            
            run_live_comparison(
                prompt=prompt_info['text'],
                baseline_name=baseline,
                recommended_name=config['recommended'],
                optimization_config=config,
                output_path=f"blog/comparison_{prompt_info['name']}_q{int(config['quality']*100)}_c{int(config['cost']*100)}.png"
            )
    
    print("\n" + "=" * 70)
    print("COMPARISON VISUALIZATIONS COMPLETE")
    print("=" * 70)
    print("\nGenerated files:")
    for config in configs:
        print(f"  - blog/comparison_coding_q{int(config['quality']*100)}_c{int(config['cost']*100)}.png")


if __name__ == "__main__":
    main()

