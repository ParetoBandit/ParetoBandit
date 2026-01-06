#!/usr/bin/env python3
"""
Create a barbell-distributed dataset for N-tuning with stress tests.

Target Distribution (20K total):
- STEM (33.3%): 6,660 prompts
  - Deep Calculus (16.7%): 3,330 Hard→Hard (aligned)
  - Arithmetic Trick (16.7%): 3,330 Easy→Hard (stress test)
- CODE (33.3%): 6,660 prompts
  - Kernel Debugging (16.7%): 3,330 Hard→Hard (aligned)
  - HTML Boilerplate (16.7%): 3,330 Easy→Hard (stress test)
- GENERAL (33.3%): 6,660 prompts
  - Email Draft (16.7%): 3,330 Easy→Easy (aligned)
  - Nuanced Haiku (16.7%): 3,330 Easy→Hard (stress test)

Key Learning Goals:
1. "Easy that looks hard" (HTML, long code) → Don't overpay
2. "Hard that looks easy" (riddles, creative writing) → Don't underpay
3. "Aligned" examples → Build correct priors
"""

import json
import re
import random
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

# ============================================================================
# Classification Heuristics
# ============================================================================

def classify_stem(prompt: str) -> Tuple[str, str]:
    """
    Classify STEM prompts into Deep Calculus vs Arithmetic Trick.
    
    Returns:
        (category, subcategory) e.g., ('STEM', 'deep_calculus')
    """
    prompt_lower = prompt.lower()
    
    # Deep Calculus indicators (Hard→Hard)
    deep_math_patterns = [
        r'\b(integral|derivative|differential|theorem|proof|lemma|corollary)\b',
        r'\$\\int',  # LaTeX integrals
        r'\$\\frac',  # LaTeX fractions
        r'\$\\sum',  # LaTeX summations
        r'\b(riemann|fourier|laplace|topology|manifold|eigenvalue)\b',
        r'\b(prove|disprove) that\b',
        r'\bcalculus\b',
        r'\bdifferential equation\b',
    ]
    
    # Arithmetic Trick indicators (Easy→Hard - looks simple but tricky)
    arithmetic_patterns = [
        r'\b\d+\s*[\+\-\*\/]\s*\d+',  # Simple arithmetic
        r'\bhow many\b.*\b(apples|coins|marbles)\b',  # Word problems
        r'\bif\b.*\bthen how (much|many)\b',  # Conditional arithmetic
        r'\bpercentage|percent\b.*\bof\b',  # Percentage problems
        r'\bwhat is\b.*\b[\+\-\*\/]\b',  # Direct calculation questions
    ]
    
    # Check for deep math first
    for pattern in deep_math_patterns:
        if re.search(pattern, prompt_lower):
            return ('STEM', 'deep_calculus')
    
    # Then check for arithmetic tricks
    for pattern in arithmetic_patterns:
        if re.search(pattern, prompt_lower):
            return ('STEM', 'arithmetic_trick')
    
    return (None, None)


def classify_code(prompt: str) -> Tuple[str, str]:
    """
    Classify CODE prompts into Kernel Debugging vs HTML Boilerplate.
    
    Returns:
        (category, subcategory)
    """
    prompt_lower = prompt.lower()
    
    # Kernel/System Debugging indicators (Hard→Hard)
    kernel_patterns = [
        r'\b(kernel|driver|syscall|interrupt|memory leak|race condition)\b',
        r'\b(deadlock|mutex|semaphore|thread|process)\b',
        r'\bdebug\b.*\b(segfault|seg fault|core dump)\b',
        r'\b(pointer|malloc|free|buffer overflow)\b',
        r'\bassembly\b',
        r'\b(rust|c\+\+|c language)\b.*\b(unsafe|pointer)\b',
    ]
    
    # HTML/Boilerplate indicators (Hard→Easy - looks complex but simple)
    html_patterns = [
        r'```html',
        r'\b(html|css|bootstrap|tailwind)\b',
        r'\bboilerplate\b',
        r'\btemplate\b.*\b(html|css|webpage)\b',
        r'\bcreate\b.*\b(website|webpage|landing page)\b',
        r'\bstyle\b.*\b(button|div|navbar)\b',
    ]
    
    # Check for kernel/system programming first
    for pattern in kernel_patterns:
        if re.search(pattern, prompt_lower):
            return ('CODE', 'kernel_debugging')
    
    # Then check for HTML/boilerplate
    for pattern in html_patterns:
        if re.search(pattern, prompt_lower):
            return ('CODE', 'html_boilerplate')
    
    # General code patterns (fallback - distribute based on complexity)
    code_patterns = [
        r'```(python|javascript|java|cpp|rust|go)',
        r'\b(def|class|function|import|return)\b',
        r'\bwrite\b.*\b(function|class|algorithm)\b',
        r'\bimplement\b',
    ]
    
    for pattern in code_patterns:
        if re.search(pattern, prompt_lower):
            # Use length as proxy: longer = kernel, shorter = boilerplate
            if len(prompt) > 200:
                return ('CODE', 'kernel_debugging')
            else:
                return ('CODE', 'html_boilerplate')
    
    return (None, None)


def classify_general(prompt: str) -> Tuple[str, str]:
    """
    Classify GENERAL prompts into Email Draft vs Nuanced Haiku.
    
    Returns:
        (category, subcategory)
    """
    prompt_lower = prompt.lower()
    
    # Email Draft indicators (Easy→Easy)
    email_patterns = [
        r'\b(email|letter|message)\b',
        r'\bwrite\b.*\b(email|letter|note)\b',
        r'\bdraft\b',
        r'\brequest\b.*\b(time off|vacation|meeting)\b',
        r'\bpolitely\b.*\b(ask|decline|respond)\b',
        r'\bthank you\b.*\bnote\b',
    ]
    
    # Nuanced Haiku indicators (Easy→Hard - looks simple but requires skill)
    haiku_patterns = [
        r'\b(haiku|poem|poetry|verse)\b',
        r'\bcreative\b.*\b(writing|story)\b',
        r'\bmetaphor\b',
        r'\bsymbolism\b',
        r'\b(philosophical|existential)\b',
        r'\bexplain\b.*\b(meaning of life|purpose|consciousness)\b',
        r'\bnuance\b',
    ]
    
    # Check for simple email/draft first
    for pattern in email_patterns:
        if re.search(pattern, prompt_lower):
            return ('GENERAL', 'email_draft')
    
    # Then check for nuanced/creative
    for pattern in haiku_patterns:
        if re.search(pattern, prompt_lower):
            return ('GENERAL', 'nuanced_haiku')
    
    # Fallback: use length as proxy
    # Very short casual prompts = email
    # Moderate creative prompts = haiku
    if len(prompt) < 50 and not any(char in prompt for char in ['?', '$', '```']):
        return ('GENERAL', 'email_draft')
    
    return (None, None)


def classify_prompt(prompt: str) -> Tuple[str, str]:
    """
    Classify a prompt into one of 6 categories.
    
    Returns:
        (category, subcategory) or (None, None) if unclassifiable
    """
    # Try STEM first
    cat, subcat = classify_stem(prompt)
    if cat:
        return (cat, subcat)
    
    # Try CODE
    cat, subcat = classify_code(prompt)
    if cat:
        return (cat, subcat)
    
    # Try GENERAL
    cat, subcat = classify_general(prompt)
    if cat:
        return (cat, subcat)
    
    return (None, None)


# ============================================================================
# Stratified Sampling
# ============================================================================

def stratified_sample_prompts(
    enriched_file: Path,
    target_per_category: int = 3330
) -> Dict[str, List[Dict]]:
    """
    Sample prompts to match the barbell distribution.
    
    Args:
        enriched_file: Path to lmsys_unused_20k_enriched.jsonl
        target_per_category: Target size for each of 6 subcategories
    
    Returns:
        Dict mapping subcategory -> list of prompt records
    """
    # Buckets for classification
    buckets = defaultdict(list)
    
    print("📂 Loading and classifying prompts...")
    total_processed = 0
    first_turn_count = 0
    
    with open(enriched_file, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            total_processed += 1
            
            # Only first turn
            if data.get('turn', 1) != 1:
                continue
            
            first_turn_count += 1
            
            # Extract prompt from conversation
            conversation = data.get('conversation', [])
            prompt = None
            for message in conversation:
                if message.get('role') == 'user':
                    prompt = message.get('content', '')
                    break
            
            if not prompt:
                continue
            
            # Classify
            category, subcategory = classify_prompt(prompt)
            if subcategory:
                buckets[subcategory].append(data)
            
            if total_processed % 10000 == 0:
                print(f"  Processed {total_processed} records...")
    
    print(f"✓ Processed {total_processed} total records")
    print(f"✓ Found {first_turn_count} first-turn conversations")
    print()
    
    # Show distribution
    print("📊 Classification Results:")
    for subcat in ['deep_calculus', 'arithmetic_trick', 'kernel_debugging', 
                   'html_boilerplate', 'email_draft', 'nuanced_haiku']:
        count = len(buckets[subcat])
        print(f"  {subcat}: {count} prompts")
    print()
    
    # Sample from each bucket
    print(f"🎯 Sampling {target_per_category} from each category...")
    sampled = {}
    random.seed(42)  # Reproducibility
    
    for subcat in ['deep_calculus', 'arithmetic_trick', 'kernel_debugging',
                   'html_boilerplate', 'email_draft', 'nuanced_haiku']:
        available = buckets[subcat]
        target = min(target_per_category, len(available))
        
        if len(available) < target_per_category:
            print(f"  ⚠️  {subcat}: Only {len(available)} available (target: {target_per_category})")
            sampled[subcat] = available
        else:
            sampled[subcat] = random.sample(available, target)
            print(f"  ✓ {subcat}: Sampled {target}")
    
    return sampled


def main():
    print("="*70)
    print("Barbell Distribution Dataset Creator")
    print("="*70)
    print()
    
    data_dir = Path('src/bandit_gpt/data')
    enriched_file = data_dir / 'lmsys_unused_20k_enriched.jsonl'
    
    if not enriched_file.exists():
        print(f"❌ Error: {enriched_file} not found")
        print("   Run: python scripts/enrich_unused_lmsys.py --input lmsys_unused_20k.jsonl")
        return
    
    # Sample prompts
    sampled = stratified_sample_prompts(enriched_file, target_per_category=3330)
    
    # Save to output file
    output_file = data_dir / 'lmsys_unused_20k_barbell.jsonl'
    print(f"\n💾 Saving barbell dataset to {output_file}...")
    
    total_saved = 0
    with open(output_file, 'w') as f:
        for subcat in ['deep_calculus', 'arithmetic_trick', 'kernel_debugging',
                       'html_boilerplate', 'email_draft', 'nuanced_haiku']:
            for record in sampled[subcat]:
                json.dump(record, f)
                f.write('\n')
                total_saved += 1
    
    print(f"✅ Saved {total_saved} prompts")
    print()
    
    # Summary
    print("="*70)
    print("📊 Final Distribution:")
    print()
    print("STEM (33.3%):")
    print(f"  Deep Calculus (Hard→Hard): {len(sampled['deep_calculus'])}")
    print(f"  Arithmetic Trick (Easy→Hard): {len(sampled['arithmetic_trick'])}")
    print()
    print("CODE (33.3%):")
    print(f"  Kernel Debugging (Hard→Hard): {len(sampled['kernel_debugging'])}")
    print(f"  HTML Boilerplate (Hard→Easy): {len(sampled['html_boilerplate'])}")
    print()
    print("GENERAL (33.3%):")
    print(f"  Email Draft (Easy→Easy): {len(sampled['email_draft'])}")
    print(f"  Nuanced Haiku (Easy→Hard): {len(sampled['nuanced_haiku'])}")
    print()
    print(f"Total: {total_saved} prompts")
    print("="*70)

if __name__ == '__main__':
    main()
