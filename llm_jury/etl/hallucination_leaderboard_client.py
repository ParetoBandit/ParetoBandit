"""Client for Vectara Hallucination Leaderboard.

Fetches hallucination rates from: https://github.com/vectara/hallucination-leaderboard
Uses the raw README.md to parse the leaderboard table.
"""

import requests
import logging
import re
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class HallucinationLeaderboardClient:
    """Client for fetching hallucination rates from Vectara's leaderboard."""

    # Raw README URL from GitHub
    README_URL = "https://raw.githubusercontent.com/vectara/hallucination-leaderboard/main/README.md"

    def __init__(self):
        """Initialize client."""
        pass

    def fetch_leaderboard(self) -> List[Dict]:
        """Fetch hallucination leaderboard data.

        Returns:
            List of dictionaries with model hallucination data:
            - model: Model name/identifier
            - hallucination_rate: Percentage (float, e.g., 3.3 for 3.3%)
            - factual_consistency_rate: Percentage (float)
            - answer_rate: Percentage (float)
            - avg_summary_length: Average summary length in words (float)

        Raises:
            requests.exceptions.RequestException: If request fails
        """
        logger.info("Fetching hallucination leaderboard from Vectara GitHub")

        try:
            response = requests.get(self.README_URL, timeout=30)
            response.raise_for_status()
            readme_content = response.text
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch hallucination leaderboard: {e}")
            raise

        return self._parse_leaderboard_table(readme_content)

    def _parse_leaderboard_table(self, readme_content: str) -> List[Dict]:
        """Parse the markdown table from README.

        Args:
            readme_content: Raw README.md content

        Returns:
            List of parsed model data dictionaries
        """
        models = []

        # Find the table between LEADERBOARD_START and LEADERBOARD_END markers
        # Table format:
        # |Model|Hallucination Rate|Factual Consistency Rate|Answer Rate|Average Summary Length (Words)|
        
        lines = readme_content.split('\n')
        in_table = False
        header_found = False

        for line in lines:
            line = line.strip()

            # Detect table header - format is |Model|Hallucination Rate|...
            if line.startswith('|Model|') and 'Hallucination Rate' in line:
                in_table = True
                header_found = True
                continue

            # Skip separator line (|----|----:|...)
            if in_table and line.startswith('|') and '----' in line:
                continue

            # Detect end of table
            if in_table and header_found:
                # End markers
                if line.startswith('<!--') or line.startswith('#') or not line.startswith('|'):
                    if line:  # Non-empty non-table line
                        break
                    continue  # Skip empty lines

            # Parse data rows
            if in_table and line.startswith('|') and header_found:
                parts = [p.strip() for p in line.split('|')]
                # Filter out empty parts from leading/trailing |
                parts = [p for p in parts if p]

                if len(parts) >= 5:
                    try:
                        model_name = parts[0]
                        hallucination_rate = self._parse_percentage(parts[1])
                        factual_consistency = self._parse_percentage(parts[2])
                        answer_rate = self._parse_percentage(parts[3])
                        avg_length = self._parse_float(parts[4])

                        if model_name and hallucination_rate is not None:
                            models.append({
                                'model': model_name,
                                'hallucination_rate': hallucination_rate,
                                'factual_consistency_rate': factual_consistency,
                                'answer_rate': answer_rate,
                                'avg_summary_length': avg_length,
                                'source': 'vectara_hallucination_leaderboard'
                            })
                    except (ValueError, IndexError) as e:
                        logger.debug(f"Skipping malformed row: {line} - {e}")
                        continue

        logger.info(f"Parsed {len(models)} models from hallucination leaderboard")
        return models

    def _parse_percentage(self, value: str) -> Optional[float]:
        """Parse percentage value like '3.3 %' to float 3.3.

        Args:
            value: String like '3.3 %' or '96.7%'

        Returns:
            Float value or None if parsing fails
        """
        if not value:
            return None
        # Remove % and whitespace
        cleaned = value.replace('%', '').strip()
        try:
            return float(cleaned)
        except ValueError:
            return None

    def _parse_float(self, value: str) -> Optional[float]:
        """Parse float value.

        Args:
            value: String representation of float

        Returns:
            Float value or None if parsing fails
        """
        if not value:
            return None
        try:
            return float(value.strip())
        except ValueError:
            return None

    def get_model_mapping(self) -> Dict[str, str]:
        """Get mapping from leaderboard model names to our standardized names.

        Returns:
            Dictionary mapping leaderboard names to our model names
        """
        # This maps the Vectara leaderboard model identifiers to our model names
        # Format: 'leaderboard_name': 'our_model_name'
        return {
            # Google models
            'google/gemini-2.5-flash-lite': 'Gemini 2.5 Flash-Lite',
            'google/gemini-2.5-flash': 'Gemini 2.5 Flash',
            'google/gemini-2.5-pro': 'Gemini 2.5 Pro',
            'google/gemini-2.0-flash': 'Gemini 2.0 Flash',
            'google/gemma-3-12b-it': 'Gemma 3 12B Instruct',
            'google/gemma-3-27b-it': 'Gemma 3 27B Instruct',
            
            # OpenAI models
            'openai/gpt-4o': 'GPT-4o',
            'openai/gpt-4o-mini': 'GPT-4o mini',
            'openai/gpt-4-turbo': 'GPT-4 Turbo',
            'openai/o1': 'o1',
            'openai/o1-mini': 'o1-mini',
            'openai/o3': 'o3',
            'openai/o3-mini': 'o3-mini',
            'openai/o4-mini': 'o4-mini',
            'openai/gpt-5': 'GPT-5',
            'openai/gpt-5-mini': 'GPT-5 mini',
            'openai/gpt-5-nano': 'GPT-5 nano',
            'openai/gpt-5.1': 'GPT-5.1',
            'openai/gpt-oss-120b': 'gpt-oss-120B',
            
            # Anthropic models
            'anthropic/claude-3.5-sonnet': 'Claude 3.5 Sonnet',
            'anthropic/claude-3.5-haiku': 'Claude 3.5 Haiku',
            'anthropic/claude-3-opus': 'Claude 3 Opus',
            'anthropic/claude-3-sonnet': 'Claude 3 Sonnet',
            'anthropic/claude-4-opus': 'Claude 4 Opus',
            'anthropic/claude-4-sonnet': 'Claude 4 Sonnet',
            'anthropic/claude-4.5-sonnet': 'Claude 4.5 Sonnet',
            'anthropic/claude-opus-4.5': 'Claude Opus 4.5',
            
            # Meta models
            'meta-llama/Llama-3.3-70B-Instruct': 'Llama 3.3 Instruct 70B',
            'meta-llama/Llama-3.3-70B-Instruct-Turbo': 'Llama 3.3 Instruct 70B',
            'meta-llama/Llama-3.1-70B-Instruct': 'Llama 3.1 Instruct 70B',
            'meta-llama/Llama-3.1-8B-Instruct': 'Llama 3.1 Instruct 8B',
            'meta-llama/Llama-3.1-405B-Instruct': 'Llama 3.1 Instruct 405B',
            'meta-llama/Llama-4-Maverick': 'Llama 4 Maverick',
            'meta-llama/Llama-4-Scout': 'Llama 4 Scout',
            
            # Mistral models
            'mistralai/mistral-large': 'Mistral Large',
            'mistralai/mistral-large-2411': 'Mistral Large 2',
            'mistralai/mistral-medium': 'Mistral Medium',
            'mistralai/mistral-small': 'Mistral Small',
            'mistralai/mistral-small-2501': 'Mistral Small 3.1',
            'mistralai/mixtral-8x7b': 'Mixtral 8x7B',
            'mistralai/magistral-medium': 'Magistral Medium',
            'mistralai/magistral-small': 'Magistral Small',
            
            # Qwen models
            'qwen/qwen3-8b': 'Qwen3 8B',
            'qwen/qwen3-14b': 'Qwen3 14B',
            'qwen/qwen3-32b': 'Qwen3 32B',
            'qwen/qwen3-4b': 'Qwen3 4B',
            'qwen/qwen3-80b-a3b-thinking': 'Qwen3 Next 80B A3B',
            'qwen/qwen2.5-72b-instruct': 'Qwen2.5 Instruct 72B',
            
            # DeepSeek models
            'deepseek-ai/DeepSeek-V3': 'DeepSeek V3',
            'deepseek-ai/DeepSeek-V3.1': 'DeepSeek V3.1',
            'deepseek-ai/DeepSeek-V3.2-Exp': 'DeepSeek V3.2 Exp',
            'deepseek-ai/DeepSeek-R1': 'DeepSeek R1',
            
            # xAI models
            'xai/grok-3': 'Grok 3',
            'xai/grok-4': 'Grok 4',
            'xai/grok-4-fast': 'Grok 4 Fast',
            'xai/grok-4.1-fast': 'Grok 4.1 Fast',
            
            # IBM models
            'ibm-granite/granite-4.0-h-small': 'Granite 4.0 H Small',
            'ibm-granite/granite-3.3-8b': 'Granite 3.3 8B',
            
            # Microsoft models
            'microsoft/Phi-4': 'Phi-4',
            'microsoft/Phi-3-medium': 'Phi-3 Medium Instruct 14B',
            'microsoft/Phi-3-mini': 'Phi-3 Mini Instruct 3.8B',
            
            # Cohere models
            'cohere/command-a': 'Command A',
            'cohere/command-r-plus': 'Command R+',
            
            # Other models
            'zhipu-ai/glm-4.5-air': 'GLM-4.5-Air',
            'zhipu-ai/glm-4.6': 'GLM-4.6',
            'ai21/jamba-1.5-large': 'Jamba 1.5 Large',
            'ai21/jamba-1.5-mini': 'Jamba 1.5 Mini',
        }


def fetch_hallucination_data() -> List[Dict]:
    """Convenience function to fetch hallucination data.

    Returns:
        List of model hallucination data dictionaries
    """
    client = HallucinationLeaderboardClient()
    return client.fetch_leaderboard()


if __name__ == '__main__':
    # Test the client
    logging.basicConfig(level=logging.INFO)
    
    client = HallucinationLeaderboardClient()
    models = client.fetch_leaderboard()
    
    print(f"\nFetched {len(models)} models from Hallucination Leaderboard")
    print("=" * 80)
    print(f"{'Model':<50} {'Halluc. Rate':<15} {'Factual Consist.':<15}")
    print("-" * 80)
    
    for m in models[:20]:  # Show first 20
        print(f"{m['model']:<50} {m['hallucination_rate']:<15.1f} {m['factual_consistency_rate']:<15.1f}")
    
    if len(models) > 20:
        print(f"... and {len(models) - 20} more models")

