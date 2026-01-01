import json
import re
from pathlib import Path
import shutil

DATA_DIR = Path('/Users/annette/repostitories/llm_jury/final_release/data')

# Keywords that suggest a SAFETY refusal (Standard/Medical/Copyright/Naughty)
SAFETY_KEYWORDS = [
    "bomb", "kill", "suicide", "dead", "death", "murder", "weapon", "gun", "knife", "attack",
    "porn", "xxx", "sex", "erotic", "nude", "naked", "dick", "cock", "pussy", "vagina",
    "drug", "cocaine", "heroin", "meth", "weed", "cannabis", "pill", "overdose",
    "hack", "exploit", "vulnerability", "virus", "malware", "steal", "crack", "bypass",
    "medical", "doctor", "treatment", "cure", "disease", "symptom", "diagnosis", "prescribe",
    "copyright", "lyrics", "song", "movie", "book", "harry potter", "batman", "marvel", # Copyright traps
    "jewish", "black", "white", "racist", "hate", "nazi", "hitler", # Hate speech
    "diss", "roast", "insult", # Harassment
    "david", "jailbreak", "roleplay", # Adversarial
    "admin", "root", "password", "key", # Security
    "joke", "humor" # Often triggers safety if the joke is offensive
]

def is_likely_safety(prompt):
    p_lower = prompt.lower()
    for kw in SAFETY_KEYWORDS:
        # Simple substring match for now, could be regex for whole words
        if kw in p_lower:
            return True
    return False

def classify_and_update():
    # 1. Load all failure files
    files = [
        "safety_refusal_prompts.json",        # Claude (small)
        "safety_refusal_prompts_gemini.json", # Gemini (medium)
        "safety_refusal_prompts_openai.json"  # OpenAI (large)
    ]
    
    all_failures = []
    for fname in files:
        fpath = DATA_DIR / fname
        if fpath.exists():
            with open(fpath) as f:
                data = json.load(f)
                print(f"Loaded {len(data)} items from {fname}")
                all_failures.extend(data)
                
    print(f"Total failures to process: {len(all_failures)}")
    
    # 2. Classify
    safety_refusals = []
    technical_failures = []
    
    # Map for dataset patching: (model_id, prompt) -> Correct Reason
    patch_map = {}
    
    for item in all_failures:
        prompt = item.get('prompt', '')
        model = item.get('model_id', '')
        
        if is_likely_safety(prompt):
            item['classification'] = "Safety Refusal"
            safety_refusals.append(item)
            patch_map[(model, prompt)] = "Forced Safety Refusal"
        else:
            item['classification'] = "Technical Failure"
            technical_failures.append(item)
            patch_map[(model, prompt)] = "Persistent Technical Failure"
            
    print(f"Classified: {len(safety_refusals)} Safety Refusals, {len(technical_failures)} Technical Failures")
    
    # 3. Save Master Lists
    with open(DATA_DIR / 'master_safety_refusals.json', 'w') as f:
        json.dump(safety_refusals, f, indent=2)
        
    with open(DATA_DIR / 'master_technical_failures.json', 'w') as f:
        json.dump(technical_failures, f, indent=2)
        
    # 4. Patch Dataset Labels in Place
    # We read line by line, if we find a matching record, we update the 'note'
    for dataset in ['train', 'test']:
        fpath = DATA_DIR / f'{dataset}_rewards.jsonl'
        if not fpath.exists(): continue
        
        print(f"Patching {dataset}_rewards.jsonl...")
        temp_path = fpath.with_suffix('.tmp')
        
        updated_count = 0
        with open(fpath, 'r') as fin, open(temp_path, 'w') as fout:
            for line in fin:
                try:
                    record = json.loads(line)
                    # Check if this record needs patching
                    key = (record.get('model_id'), record.get('prompt'))
                    if key in patch_map:
                        # Only update if it currently says "Forced Safety Refusal" or generic
                        # or just overwrite to be sure
                        if record.get('note') == "Forced Safety Refusal":
                            new_note = patch_map[key]
                            if record.get('note') != new_note:
                                record['note'] = new_note
                                updated_count += 1
                                
                    fout.write(json.dumps(record) + "\n")
                except:
                    fout.write(line) # preserve corrupted lines if any
        
        # Atomic swap
        shutil.move(temp_path, fpath)
        print(f"Updated {updated_count} records in {dataset}_rewards.jsonl")

if __name__ == "__main__":
    classify_and_update()
