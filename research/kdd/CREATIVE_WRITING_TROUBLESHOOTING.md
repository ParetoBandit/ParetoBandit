# Creative Writing Benchmark - Troubleshooting

## Common Issues and Solutions

### ❌ Error: `ModuleNotFoundError: No module named 'trueskill'`

**Problem:**
```
File "creative-writing-bench/core/trueskill_solver_cw.py", line 5, in <module>
    import trueskill # type: ignore
    ^^^^^^^^^^^^^^^^
ModuleNotFoundError: No module named 'trueskill'
```

**Solution:**
The `trueskill` package was missing from the original requirements.txt. Install it:

```bash
pip install trueskill
```

**Why this happened:**
The creative-writing-bench repository's requirements.txt didn't include `trueskill`, even though it's used in the Elo calculation system.

**Fixed:**
We've updated the requirements.txt in the cloned repository to include `trueskill`.

---

### ❌ Error: `OPENROUTER_API_KEY not found`

**Problem:**
```
ValueError: OPENROUTER_API_KEY not found in environment
```

**Solution:**
Add your OpenRouter API key to `.env` file in project root:

```bash
OPENROUTER_API_KEY=your_key_here
```

---

### ⚠️ Warning: Evaluation takes 15-30 minutes

**Not an error** - this is expected behavior.

The Creative Writing Benchmark is comprehensive:
- 96 generation calls (32 prompts × 3 iterations)
- Rubric scoring for each piece
- Multiple pairwise comparisons
- Elo calculation

**If it's too slow:**
- For testing: Use `--iterations 1` (not leaderboard-comparable)
- For production: Be patient - quality evaluation takes time

---

### ❌ Error: `Benchmark failed` with subprocess errors

**Check:**
1. Sufficient disk space for results files
2. Write permissions in creative-writing-bench/ directory
3. OpenRouter API key is valid
4. Model is available on OpenRouter

**Debug:**
Run the benchmark directly to see detailed errors:

```bash
cd creative-writing-bench
python3 creative_writing_bench.py \
  --test-model "openai/gpt-4o" \
  --judge-model "anthropic/claude-sonnet-4" \
  --runs-file "creative_bench_runs.json" \
  --run-id "test_run" \
  --iterations 1 \
  --verbosity DEBUG
```

---

### ❌ Error: `Run key not found in results`

**Problem:**
The benchmark completed but couldn't extract Elo score.

**Solution:**
Check if `creative_bench_runs.json` was created and contains your run.

```bash
# List run keys
python -c "
import json
with open('creative-writing-bench/creative_bench_runs.json') as f:
    runs = json.load(f)
    print(list(runs.keys())[-10:])  # Last 10 runs
"
```

---

### ⚠️ Scores don't match EQ-Bench leaderboard

**Check:**
1. Using recommended judge: `anthropic/claude-sonnet-4`
2. Using 3 iterations (standard)
3. Canonical data was unzipped (auto-done by script)
4. Using latest version of creative-writing-bench

**Note:** Small variations are normal due to:
- Judge model version differences
- Random seed variations in generation
- Elo calculation differences with pool of compared models

---

### ❌ Error: `Rate limit exceeded`

**Problem:**
```
RateLimitError: Rate limit exceeded
```

**Solution:**
1. Wait a few minutes and retry
2. Reduce thread count: `--threads 50` (default is 100)
3. Check your OpenRouter usage dashboard

**Why this happens:**
The benchmark makes many API calls in parallel (generation + judging).

---

### ❌ Error: NLTK data not found

**Problem:**
```
LookupError: Resource punkt not found
```

**Solution:**
```python
import nltk
nltk.download('punkt')
nltk.download('cmudict')
```

Or run:
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('cmudict')"
```

---

### ⚠️ High API costs

**Not an error** - Creative Writing is more expensive than other benchmarks.

**Expected costs:**
- ~$10 per model for full evaluation
- $3-5 for generation (96 pieces)
- $5-7 for judging (rubric + pairwise)

**Reduce costs:**
- Test with `--iterations 1`: ~$3-4 per model
- Test fewer models: `--models gpt-4o`
- Use cheaper models (but scores won't be comparable)

---

### ❌ Error: Timeout after 2 hours

**Problem:**
```
subprocess.TimeoutExpired
```

**Possible causes:**
1. Very slow model (unlikely - most complete in 20-30 min)
2. API issues causing retries
3. Judge model bottleneck

**Solution:**
1. Check OpenRouter status
2. Retry the evaluation
3. Check specific model availability

---

## Verification Checklist

Before running evaluations, verify:

- [ ] `trueskill` installed: `python -c "import trueskill"`
- [ ] NLTK data downloaded: `python -c "import nltk; nltk.data.find('tokenizers/punkt')"`
- [ ] OpenRouter API key set: `echo $OPENROUTER_API_KEY`
- [ ] Canonical data unzipped: `ls creative-writing-bench/creative_bench_runs.json`
- [ ] Disk space available: `df -h .`

Quick verification:
```bash
python kdd_paper/run_creative_writing.py --all --dry-run --max-models 1
```

Should output model list without errors.

---

## Getting Help

If issues persist:

1. **Check documentation:**
   - `CREATIVE_WRITING_GUIDE.md` - Full guide
   - `CREATIVE_WRITING_QUICKSTART.md` - Quick reference
   - Original repo: https://github.com/EQ-bench/creative-writing-bench

2. **Verify installation:**
   ```bash
   pip list | grep -E "trueskill|glicko2|nltk"
   ```

3. **Test benchmark directly:**
   ```bash
   cd creative-writing-bench
   python3 creative_writing_bench.py --help
   ```

4. **Check logs:**
   - Terminal output shows detailed errors
   - Check `creative-writing-bench/` for any log files

---

## Summary of Fixed Issues

✅ **Added `trueskill` to requirements** - Most common issue
✅ **Updated requirements.txt** in creative-writing-bench/
✅ **Updated documentation** with trueskill installation
✅ **Created this troubleshooting guide**

All other dependencies were already included and working correctly.

