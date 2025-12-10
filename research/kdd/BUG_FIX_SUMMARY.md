# SummEdits Silent Save Failure - Bug Fix Summary

## Problem

**58 evaluation scores were lost** across 11 models due to a silent save failure in `run_summedits.py`. The script would log "Saved X scores to summedits_xxx_scores.json" but the files were not actually updated with the new scores.

### Impact
- **44 scores recovered** from terminal logs
- **58 scores lost** due to terminal truncation (only ~546 lines captured)
- **43.1% recovery rate**

### Most affected domains (lost to terminal truncation):
- `billsum` and `samsum`: 11 models each
- `podcast`: 10 models
- `news`, `sales_email`, `sales_call`: 7-8 models each

## Root Cause

The bug was in the `DataManager.save_scores()` method (lines 283-295):

```python
def save_scores(self, scores: Dict[str, float], domain: str):
    """Save scores and update cache."""
    with self._lock:
        existing = self._scores_cache.get(domain, {}).copy()  # ❌ BUG HERE
        existing.update(scores)
        
        scores_file = f"summedits_{domain}_scores.json"
        output_path = DATA_PATH / scores_file
        with open(output_path, "w") as f:
            json.dump(existing, f, indent=2)
        
        self._scores_cache[domain] = existing
        logger.info(f"Saved {len(scores)} scores to {scores_file}")
```

**The issue:** Line 286 gets scores from the in-memory cache with `self._scores_cache.get(domain, {})`, which returns an empty dict if the domain isn't in the cache yet. This could happen if:

1. The cache was never populated for that domain (no prior `get_scores()` call)
2. The cache was stale or out of sync with disk
3. Concurrent threads had timing issues

When this happened, `existing` would be an empty dict, so `existing.update(scores)` would only contain the new scores, and the file write would **overwrite all previous scores**, causing data loss.

The log message "Saved X scores..." would still print, making the failure silent and hard to detect.

## The Fix

### 1. Always Read from Disk Before Saving

```python
def save_scores(self, scores: Dict[str, float], domain: str):
    """Save scores and update cache."""
    with self._lock:
        # Always reload from disk to ensure we have the latest data
        # This prevents data loss if cache is stale or not populated
        scores_file = f"summedits_{domain}_scores.json"
        output_path = DATA_PATH / scores_file
        
        # Load existing scores from disk
        if output_path.exists():
            try:
                with open(output_path, "r") as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Failed to load {scores_file}: {e}")
                existing = {}
        else:
            existing = {}
        
        # Update with new scores
        existing.update(scores)
        
        # Write to disk with explicit flush
        try:
            with open(output_path, "w") as f:
                json.dump(existing, f, indent=2)
                f.flush()  # Explicit flush to ensure write
                os.fsync(f.fileno())  # Force OS to write to disk
            logger.info(f"Saved {len(scores)} scores to {scores_file}")
        except (IOError, OSError) as e:
            logger.error(f"Failed to save {scores_file}: {e}")
            raise  # Re-raise to make failure visible
        
        # Update cache after successful save
        self._scores_cache[domain] = existing
```

**Key improvements:**
- ✅ Always read from disk first (don't trust cache)
- ✅ Explicit error handling with try/except
- ✅ Explicit `f.flush()` and `os.fsync()` to force write to disk
- ✅ Re-raise exceptions to make failures visible (not silent)
- ✅ Update cache only after successful save

### 2. Added `--force` Flag

```python
parser.add_argument("--force", action="store_true",
                    help="Force re-evaluation even if scores already exist")
```

This allows re-running evaluations for models that already have scores, which is useful for:
- Verifying results
- Re-running after model improvements
- Recovering from corrupted data

## Testing

Created a test that verifies:
1. Existing data is preserved when adding new scores
2. New scores are correctly added
3. File writes are properly flushed to disk

```bash
✅ Save verification PASSED
   Existing data preserved: 2 entries
   New data added: 1 entries
   Total: 3 entries
```

## Recovery Actions Taken

1. **Extracted scores from terminal logs:**
   - Terminal 10: GPT-5 (high) - 4 domains
   - Terminal 12: o3 - 4 domains  
   - Terminal 13: o3-mini (high) - 4 domains
   - Terminals 14-23: 8 models - 4 domains each

2. **Manually added recovered scores** to respective JSON files using correct OpenRouter IDs (not UUIDs)

3. **Created recovery file:** `data/terminal_recovery_scores.json` for audit trail

## Current Status

After recovery and bug fix:
- ✅ **71 models complete** (10/10 domains)
- ⚠️ **13 models incomplete** (missing 1-6 domains)
- ❌ **0 models with no scores**

### Models still needing evaluation:
1. Gemini 3 Pro Preview (high) - 9/10 (missing: news)
2. o3 - 8/10 (missing: billsum, samsum)
3. DeepSeek R1 Distill Qwen 32B - 6/10
4. o3-mini (high) - 5/10
5. GPT-5 (high) - 5/10
6. Qwen3 14B (Reasoning) - 5/10
7. o1 - 5/10
8. DeepSeek R1 Distill Llama 70B - 5/10
9-13. Five more models with 4 domains each

## Recommendations

1. **Re-run missing evaluations** with the fixed code
2. **Monitor logs carefully** for any new save failures
3. **Consider adding automated verification** that scores were actually saved after each evaluation
4. **Increase terminal buffer size** or implement proper logging to file to prevent output truncation
5. **Add checksums or verification** to detect silent corruption

## Usage

```bash
# Normal run (skips models with existing scores)
python research/kdd/run_summedits.py --model "GPT-5 (high)" --domains all

# Force re-evaluation even if scores exist
python research/kdd/run_summedits.py --model "GPT-5 (high)" --domains billsum samsum --force

# Run multiple incomplete models
python research/kdd/run_summedits.py --all --domains billsum samsum --threads 10
```
