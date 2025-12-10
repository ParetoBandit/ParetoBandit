# Creative Writing Quick Start

## What is Creative Writing Benchmark?

Evaluates creative writing quality using 32 diverse prompts and pairwise Elo comparisons. Metric: **Normalized Elo Score** (200-1700 range, higher is better).

## Quick Commands

### Run Evaluation

```bash
# All models (recommended - leaderboard comparable)
python kdd_paper/run_creative_writing.py --all

# Specific models
python kdd_paper/run_creative_writing.py --models gpt-4o claude-sonnet

# Quick test (1 iteration, not leaderboard comparable)
python kdd_paper/run_creative_writing.py --models gpt-4o --iterations 1

# Dry run (preview without running)
python kdd_paper/run_creative_writing.py --all --dry-run
```

### Analyze Results

```bash
# Analysis report
python scripts/analyze_creative_writing.py
```

## Prompt Categories

32 prompts across diverse genres:
- Historical Fiction
- Romance
- Sci-Fi
- Fantasy
- Horror
- Mystery
- Humor
- Poetry
- Dialogue
- ... and more

## Output Location

Scores saved to `data/creative_writing_scores.json`

Full benchmark data in `creative-writing-bench/`:
- `creative_bench_runs.json` - All generated text
- `elo_results.json` - Detailed Elo analysis

## Performance Expectations

| Model Tier | Expected Elo | Example |
|------------|--------------|---------|
| Elite | 1600+ | GPT-4.5, Claude 3.7 |
| Excellent | 1400-1600 | GPT-4o, Claude 3.5 |
| Good | 1200-1400 | Gemini 2.0, Llama 405B |
| Fair | 1000-1200 | Mistral Nemo, Gemma 27B |
| Basic | <1000 | Smaller models |

## Cost & Time

**Cost**: ~$10 per model
- Generation: $3-5 (96 pieces × 1000 tokens)
- Judging: $5-7 (rubric + pairwise)

**Time**: 15-30 minutes per model
- Generation: 5-10 min
- Judging: 10-20 min

## Common Issues

**Takes too long**
→ Expected - comprehensive evaluation with 96 generations + judging

**OPENROUTER_API_KEY not found**
→ Add to `.env`: `OPENROUTER_API_KEY=your_key_here`

**Want faster testing**
→ Use `--iterations 1` (not leaderboard comparable)

## Integration Example

```python
# Load scores
import json
with open('data/creative_writing_scores.json') as f:
    scores = json.load(f)

# Normalize to 0-100
def normalize_elo(elo):
    return (elo - 200) / 15  # 200-1700 → 0-100

# Use in quality scoring
quality = 0.25*intelligence + 0.15*normalize_elo(scores[model]) + ...
```

## Full Documentation

- **Complete Guide**: `kdd_paper/CREATIVE_WRITING_GUIDE.md`
- **Data Source**: `creative-writing-bench/README.md`
- **Leaderboard**: https://eqbench.com/creative_writing.html

