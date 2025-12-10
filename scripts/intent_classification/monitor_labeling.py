"""
Monitor Teacher Labeling Progress

Watches the output file and displays real-time progress and category distribution.
"""

import json
import time
import os
from pathlib import Path
from collections import Counter
from datetime import datetime, timedelta


def clear_screen():
    """Clear the terminal screen."""
    os.system('clear' if os.name == 'posix' else 'cls')


def format_time(seconds):
    """Format seconds into human-readable time."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def display_progress(output_file: str, total_samples: int = 764, refresh_interval: int = 10):
    """
    Monitor and display labeling progress.
    
    Args:
        output_file: Path to the output JSON file
        total_samples: Total number of samples to label
        refresh_interval: How often to refresh (seconds)
    """
    output_path = Path(output_file)
    error_path = output_path.with_suffix('.errors.json')
    
    start_time = time.time()
    last_count = 0
    last_update_time = start_time
    
    print(f"🔍 Monitoring: {output_file}")
    print(f"📊 Total samples: {total_samples}")
    print(f"🔄 Refresh interval: {refresh_interval}s")
    print("\nPress Ctrl+C to stop monitoring\n")
    print("="*80)
    
    try:
        while True:
            clear_screen()
            
            # Header
            print("="*80)
            print("  GEMINI 3 PRO TEACHER LABELING - LIVE MONITOR")
            print("="*80)
            
            current_time = time.time()
            elapsed = current_time - start_time
            
            # Check if file exists
            if not output_path.exists():
                print("\n⏳ Waiting for output file to be created...")
                print(f"   Expected: {output_file}")
                print(f"   Elapsed: {format_time(elapsed)}")
                time.sleep(refresh_interval)
                continue
            
            # Load current results
            try:
                with open(output_path, 'r') as f:
                    data = json.load(f)
                
                samples = data.get('samples', [])
                labeled_count = len(samples)
                
                # Load errors if they exist
                error_count = 0
                if error_path.exists():
                    try:
                        with open(error_path, 'r') as f:
                            errors = json.load(f)
                            error_count = len(errors)
                    except:
                        pass
                
                # Calculate stats
                progress_pct = (labeled_count / total_samples) * 100
                success_rate = (labeled_count / (labeled_count + error_count) * 100) if (labeled_count + error_count) > 0 else 0
                
                # Calculate speed
                if labeled_count > last_count:
                    time_diff = current_time - last_update_time
                    samples_diff = labeled_count - last_count
                    current_rate = samples_diff / time_diff if time_diff > 0 else 0
                    last_count = labeled_count
                    last_update_time = current_time
                else:
                    current_rate = labeled_count / elapsed if elapsed > 0 else 0
                
                # Estimate remaining time
                remaining_samples = total_samples - labeled_count - error_count
                eta_seconds = remaining_samples / current_rate if current_rate > 0 else 0
                
                # Display progress
                print(f"\n📊 PROGRESS")
                print("-"*80)
                print(f"Labeled:       {labeled_count:>6} / {total_samples} ({progress_pct:>5.1f}%)")
                print(f"Errors:        {error_count:>6}")
                print(f"Remaining:     {remaining_samples:>6}")
                print(f"Success Rate:  {success_rate:>5.1f}%")
                
                # Progress bar
                bar_width = 50
                filled = int(bar_width * progress_pct / 100)
                bar = "█" * filled + "░" * (bar_width - filled)
                print(f"\n[{bar}] {progress_pct:.1f}%")
                
                print(f"\n⏱️  TIMING")
                print("-"*80)
                print(f"Elapsed:       {format_time(elapsed)}")
                print(f"Rate:          {current_rate:.2f} samples/sec ({current_rate*60:.1f}/min)")
                print(f"Est. Complete: {format_time(eta_seconds)} remaining")
                
                if eta_seconds > 0:
                    completion_time = datetime.now() + timedelta(seconds=eta_seconds)
                    print(f"ETA:           {completion_time.strftime('%I:%M:%S %p')}")
                
                # Category distribution
                if labeled_count > 0:
                    category_counts = Counter(s['label'] for s in samples)
                    
                    print(f"\n📋 CATEGORY DISTRIBUTION (Current)")
                    print("-"*80)
                    
                    # Sort by count
                    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
                        pct = count / labeled_count * 100
                        bar_len = int(pct / 2)
                        bar = "█" * bar_len
                        print(f"{cat:<25} {count:>4} ({pct:>5.1f}%) {bar}")
                
                # Recent errors
                if error_count > 0 and error_path.exists():
                    try:
                        with open(error_path, 'r') as f:
                            errors = json.load(f)
                        
                        print(f"\n⚠️  ERRORS (Last 3)")
                        print("-"*80)
                        for i, err in enumerate(errors[-3:], 1):
                            error_type = err.get('error_type', 'Unknown')
                            error_msg = err.get('error', 'No message')[:60]
                            print(f"{i}. {error_type}: {error_msg}")
                    except:
                        pass
                
                print(f"\n{'='*80}")
                print(f"Last updated: {datetime.now().strftime('%I:%M:%S %p')} | Refreshing every {refresh_interval}s")
                print(f"Press Ctrl+C to stop monitoring")
                
                # Check if complete
                if labeled_count + error_count >= total_samples:
                    print("\n🎉 LABELING COMPLETE!")
                    print(f"\n✓ Successfully labeled: {labeled_count}")
                    print(f"✗ Errors: {error_count}")
                    print(f"Total time: {format_time(elapsed)}")
                    break
                
            except json.JSONDecodeError:
                print("\n⏳ File is being written... waiting...")
            except Exception as e:
                print(f"\n❌ Error reading file: {e}")
            
            time.sleep(refresh_interval)
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Monitoring stopped by user")
        print(f"\nFinal status:")
        print(f"  Labeled: {labeled_count}/{total_samples}")
        print(f"  Time: {format_time(time.time() - start_time)}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Monitor teacher labeling progress"
    )
    parser.add_argument(
        '--output',
        default='data/real_intent_labeled_gemini3_v2.json',
        help='Path to output file'
    )
    parser.add_argument(
        '--total',
        type=int,
        default=764,
        help='Total number of samples'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=10,
        help='Refresh interval in seconds'
    )
    
    args = parser.parse_args()
    
    display_progress(args.output, args.total, args.interval)


if __name__ == '__main__':
    main()

