#!/usr/bin/env python3
"""
Script to switch all experiments from GPT-4o to GPT-4-turbo.

This eliminates the model substitution confound by using the same model
in both warmup (RouteLLM battles) and evaluation (dev/holdout).

Scientific justification:
- Warmup priors trained on mixtral vs. gpt-4-turbo battles
- Evaluation should use mixtral vs. gpt-4-turbo (not gpt-4o)  
- This isolates algorithmic contribution from model capability shifts
"""

import re
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

def replace_in_file(filepath: Path, old_pattern: str, new_pattern: str) -> bool:
    """Replace pattern in file. Returns True if changes made."""
    try:
        content = filepath.read_text()
        new_content = re.sub(old_pattern, new_pattern, content)
        
        if new_content != content:
            filepath.write_text(new_content)
            print(f"  ✓ Updated: {filepath.relative_to(PROJECT_ROOT)}")
            return True
        return False
    except Exception as e:
        print(f"  ✗ Error in {filepath}: {e}")
        return False


def main():
    print("=" * 70)
    print("SWITCHING FROM GPT-4O TO GPT-4-TURBO")
    print("=" * 70)
    print("\nThis eliminates the model substitution confound.")
    print("\nSearching for files that reference gpt-4o...")
    
    # Patterns to replace
    replacements = [
        # Python strings
        (r'"openai/gpt-4-turbo"', '"openai/gpt-4-turbo"'),
        (r"'openai/gpt-4-turbo'", "'openai/gpt-4-turbo'"),
        
        # Comments and documentation
        (r'gpt-4-turbo evaluations', 'gpt-4-turbo evaluations'),
        (r'uses gpt-4-turbo', 'uses gpt-4-turbo'),
        (r'Model: gpt-4-turbo', 'Model: gpt-4-turbo'),
        
        # Variable names (be careful)
        (r'gpt4turbo_', 'gpt4turbo_'),
    ]
    
    # Find all Python files in experiments
    files_to_check = []
    for pattern in ["**/*.py", "**/*.tex", "**/*.md"]:
        files_to_check.extend((PROJECT_ROOT / "experiments").glob(pattern))
    
    # Also check src and scripts
    for pattern in ["**/*.py"]:
        files_to_check.extend((PROJECT_ROOT / "src").glob(pattern))
        files_to_check.extend((PROJECT_ROOT / "scripts").glob(pattern))
    
    # Track changes
    files_changed = 0
    total_files = len(files_to_check)
    
    print(f"\nChecking {total_files} files...")
    print()
    
    for filepath in files_to_check:
        changed = False
        for old_pattern, new_pattern in replacements:
            if replace_in_file(filepath, old_pattern, new_pattern):
                changed = True
        
        if changed:
            files_changed += 1
    
    print()
    print("=" * 70)
    print(f"SUMMARY: Updated {files_changed}/{total_files} files")
    print("=" * 70)
    print()
    
    if files_changed > 0:
        print("✓ Successfully switched to gpt-4-turbo!")
        print()
        print("Next steps:")
        print("  1. Review changes: git diff")
        print("  2. Re-run key experiments to verify")
        print("  3. Update experiments.tex (already done)")
        print("  4. Verify results match expected behavior")
    else:
        print("ℹ No files needed updating (already using gpt-4-turbo?)")
    
    print()


if __name__ == "__main__":
    main()
