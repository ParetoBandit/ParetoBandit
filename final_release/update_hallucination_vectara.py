import json
import requests
import re
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Constants
BASE_DIR = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent
MODELS_PATH = BASE_DIR / "models.json"
RAW_DATA_PATH = BASE_DIR / "data" / "vectara_data.json"
VECTARA_MAIN_URL = "https://raw.githubusercontent.com/vectara/hallucination-leaderboard/main/README.md"
VECTARA_OLD_URL = "https://raw.githubusercontent.com/vectara/hallucination-leaderboard/hhem-2.3-old-dataset/README.md"

# Manual Overrides (Vectara Hallucination Rate %)
MANUAL_OVERRIDES = {
    "openai/gpt-5-chat": 1.4,
    "openai/gpt-5": 1.4,
    "moonshotai/kimi-k2-0905:exacto": 6.2,
    "moonshotai/kimi-k2-0905": 6.2
}

def normalize_name(name):
    """Normalize names for fuzzy matching."""
    return re.sub(r'[^a-zA-Z0-9]', '', name).lower()

def fetch_and_parse_vectara(url):
    """Fetch and parse models from a Vectara README URL."""
    logger.info(f"Fetching Vectara data from {url}...")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        content = response.text
    except Exception as e:
        logger.error(f"Error fetching Vectara data: {e}")
        return {}

    models = {}
    lines = content.split('\n')
    in_table = False
    for line in lines:
        line = line.strip()
        if '|Model|' in line and 'Hallucination Rate' in line:
            in_table = True
            continue
        if in_table and line.startswith('|---'):
            continue
        if in_table and line.startswith('|'):
            parts = [p.strip() for p in line.split('|')]
            # Header line: |Model|Hallucination Rate|...
            # Parts: ['', 'Model', 'Hallucination Rate', ...]
            if len(parts) >= 3:
                name = parts[1].replace('`', '')
                try:
                    # Look for the percentage value
                    rate_str = parts[2].replace('%', '').strip()
                    # Some entries might have ranges or descriptions
                    rate_match = re.search(r'(\d+\.?\d*)', rate_str)
                    if rate_match:
                        rate = float(rate_match.group(1))
                        models[name] = rate
                except ValueError:
                    continue
        elif in_table and line == '':
            in_table = False
    
    logger.info(f"Extracted {len(models)} models from table.")
    return models

def update_models():
    """Update models.json with Vectara hallucination data."""
    # 1. Fetch data from both sources
    vectara_main = fetch_and_parse_vectara(VECTARA_MAIN_URL)
    vectara_old = fetch_and_parse_vectara(VECTARA_OLD_URL)
    
    # Merge datasets, main takes precedence
    vectara_data = {**vectara_old, **vectara_main}
    logger.info(f"Total unique Vectara models: {len(vectara_data)}")

    # 2. Save raw data for reproducibility
    logger.info(f"Saving raw Vectara data to {RAW_DATA_PATH}...")
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RAW_DATA_PATH, 'w') as f:
        json.dump(vectara_data, f, indent=2)

    # 3. Load current registry
    if not MODELS_PATH.exists():
        logger.error(f"Models file not found at {MODELS_PATH}")
        return

    with open(MODELS_PATH, 'r') as f:
        data = json.load(f)

    # 3. Perform matching
    updated_count = 0
    not_found = []

    # Map for normalization
    vectara_map = {normalize_name(name): (name, rate) for name, rate in vectara_data.items()}

    for model in data.get('models', []):
        display_name = model.get('display_name', '')
        openrouter_id = model.get('openrouter_id', '')
        
        # 1. Check for manual overrides first
        if openrouter_id in MANUAL_OVERRIDES:
            model['hallucination_vectara'] = MANUAL_OVERRIDES[openrouter_id]
            updated_count += 1
            continue

        match_found = False
        
        # Strategy: Match normalized display name or OpenRouter ID parts
        norm_display = normalize_name(display_name)
        norm_oid = normalize_name(openrouter_id)

        # Direct normalized match
        if norm_display in vectara_map:
            model['hallucination_vectara'] = vectara_map[norm_display][1]
            match_found = True
        elif norm_oid in vectara_map:
            model['hallucination_vectara'] = vectara_map[norm_oid][1]
            match_found = True
        else:
            # Partial match for common variations
            for norm_vname, (vname, rate) in vectara_map.items():
                if norm_vname in norm_display or norm_display in norm_vname or \
                   norm_vname in norm_oid or norm_oid in norm_vname:
                    model['hallucination_vectara'] = rate
                    match_found = True
                    break
        
        if match_found:
            updated_count += 1
        else:
            not_found.append(display_name)

    # 4. Save results
    with open(MODELS_PATH, 'w') as f:
        json.dump(data, f, indent=2)

    logger.info(f"Updated {updated_count} models with Vectara data.")
    if not_found:
        logger.info(f"No Vectara data found for {len(not_found)} models.")

if __name__ == "__main__":
    update_models()
