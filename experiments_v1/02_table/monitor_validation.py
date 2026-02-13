#!/usr/bin/env python3
"""
Real-time monitoring dashboard for validation progress.

Usage:
    python monitor_validation.py

Press Ctrl+C to exit.
"""

import json
import time
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta


def clear_screen():
    """Clear terminal screen."""
    os.system('clear' if os.name != 'nt' else 'cls')


def load_json_safe(path):
    """Safely load JSON, return None if file doesn't exist."""
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def format_duration(seconds):
    """Format duration in human-readable format."""
    return str(timedelta(seconds=int(seconds)))


def get_file_age(path):
    """Get file age in seconds, or None if doesn't exist."""
    try:
        mtime = os.path.getmtime(path)
        return time.time() - mtime
    except FileNotFoundError:
        return None


def print_header():
    """Print dashboard header."""
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "TABLE 2 VALIDATION MONITOR" + " " * 32 + "║")
    print("╠" + "═" * 78 + "╣")
    print(f"║ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}" + " " * 55 + "║")
    print("╚" + "═" * 78 + "╝")
    print()


def print_section(title):
    """Print section header."""
    print(f"\n┌─ {title} " + "─" * (75 - len(title)) + "┐")


def print_end_section():
    """Print section footer."""
    print("└" + "─" * 77 + "┘")


def check_step(step_name, json_path, num_seeds_expected=10):
    """Check status of a validation step."""
    data = load_json_safe(json_path)
    file_age = get_file_age(json_path)
    
    if data is None:
        if file_age is None:
            status = "⏳ NOT STARTED"
            details = ""
        else:
            status = "⚠️  ERROR (file exists but invalid)"
            details = f"Last modified: {format_duration(file_age)} ago"
    else:
        # Check number of seeds completed
        try:
            hybrid = data.get('Hybrid (Corralling)', {})
            num_seeds = hybrid.get('num_seeds', 0)
            
            if num_seeds >= num_seeds_expected:
                status = "✅ COMPLETE"
                
                # Extract key metrics
                stats = hybrid.get('statistics', {})
                cum_regret = stats.get('cumulative_regret', {})
                mean = cum_regret.get('mean', 0)
                std = cum_regret.get('std', 0)
                median = cum_regret.get('median', 0)
                
                details = f"Median: {median:.1f}, Mean: {mean:.1f} ± {std:.1f}"
            else:
                status = f"⏳ IN PROGRESS ({num_seeds}/{num_seeds_expected} seeds)"
                details = f"Last updated: {format_duration(file_age)} ago" if file_age else ""
        except Exception as e:
            status = "⚠️  ERROR"
            details = str(e)
    
    print(f"│ {step_name:<20} {status:<30} {details:<25}│")


def monitor_loop():
    """Main monitoring loop."""
    try:
        iteration = 0
        start_time = time.time()
        
        while True:
            iteration += 1
            clear_screen()
            
            # Header
            print_header()
            
            # Runtime
            runtime = time.time() - start_time
            print(f"Runtime: {format_duration(runtime)}")
            
            # Step 1: η=0.1 validation
            print_section("Step 1/3: η=0.1 Validation")
            check_step(
                "η=0.1 Results",
                "data/eta_0.1_holdout_multiseed/results_multiseed.json"
            )
            print_end_section()
            
            # Step 2: η=1.0 validation
            print_section("Step 2/3: η=1.0 Validation")
            check_step(
                "η=1.0 Results",
                "data/eta_1.0_holdout_multiseed/results_multiseed.json"
            )
            print_end_section()
            
            # Step 3: Comparison
            print_section("Step 3/3: Statistical Comparison")
            comparison_path = "data/statistical_comparison/comparison_results.json"
            comparison_data = load_json_safe(comparison_path)
            
            if comparison_data:
                print("│ " + "✅ COMPLETE" + " " * 64 + "│")
                
                # Extract key results
                try:
                    hybrid = comparison_data.get('Hybrid (Corralling)', {})
                    cum = hybrid.get('cumulative_regret', {})
                    
                    t_test = cum.get('t_test', {})
                    effect = cum.get('effect_size', {})
                    
                    p_val = t_test.get('p_value', 0)
                    cohens_d = effect.get('cohens_d', 0)
                    sig = t_test.get('significant_bonferroni_0.05', False)
                    
                    print(f"│ p-value: {p_val:.4f} {'✅ Significant' if sig else '❌ Not significant'}" + " " * 32 + "│")
                    print(f"│ Cohen's d: {cohens_d:.3f} ({effect.get('interpretation', 'unknown')})" + " " * 32 + "│")
                except Exception as e:
                    print(f"│ Error extracting stats: {str(e)}" + " " * 42 + "│")
            else:
                file_age = get_file_age(comparison_path)
                if file_age is None:
                    print("│ " + "⏳ PENDING (waiting for both η results)" + " " * 33 + "│")
                else:
                    print("│ " + "⚠️  ERROR (file exists but invalid)" + " " * 37 + "│")
            
            print_end_section()
            
            # Log files
            print_section("Log Files")
            
            for log_file in ['validation_eta_01.log', 'validation_eta_10.log', 'validation_full.log']:
                log_path = Path(log_file)
                if log_path.exists():
                    age = get_file_age(log_file)
                    size_kb = log_path.stat().st_size / 1024
                    print(f"│ {log_file:<30} Size: {size_kb:>6.1f} KB    Age: {format_duration(age)}" + " " * 8 + "│")
                else:
                    print(f"│ {log_file:<30} {'Not found':<40}│")
            
            print_end_section()
            
            # Instructions
            print("\n" + "─" * 80)
            print("Press Ctrl+C to exit | Refreshing every 5 seconds")
            print("─" * 80)
            
            # Check if all complete
            eta_01 = load_json_safe("data/eta_0.1_holdout_multiseed/results_multiseed.json")
            eta_10 = load_json_safe("data/eta_1.0_holdout_multiseed/results_multiseed.json")
            comparison = load_json_safe("data/statistical_comparison/comparison_results.json")
            
            if eta_01 and eta_10 and comparison:
                print("\n🎉 VALIDATION COMPLETE! 🎉")
                print("\nNext steps:")
                print("  1. Run: python visualize_variance.py")
                print("  2. Run: python generate_table_from_results.py")
                print("  3. See: NEXT_STEPS_AFTER_VALIDATION.md")
                print("\nPress Ctrl+C to exit or wait 30 seconds for auto-exit...")
                time.sleep(30)
                break
            
            # Wait before next iteration
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\n\n✅ Monitoring stopped by user")
        sys.exit(0)


def main():
    print("Starting validation monitor...")
    print("Make sure validation is running in another terminal!")
    print("")
    time.sleep(2)
    
    monitor_loop()


if __name__ == '__main__':
    main()
