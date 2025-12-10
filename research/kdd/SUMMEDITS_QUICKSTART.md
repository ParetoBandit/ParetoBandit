# SummEdits Quick Start

## What is SummEdits?

Binary classification benchmark that tests if AI models can detect factual inconsistencies in summaries. Metric: **Balanced Accuracy** (0-100%).

## Quick Commands

### Run Evaluation

```bash
# All models, all domains (recommended for full evaluation)
python kdd_paper/run_summedits.py --all --domains all --threads 10

# Quick test with one domain
python kdd_paper/run_summedits.py --all --domains news --max-samples 100

# Specific models
python kdd_paper/run_summedits.py --models gpt-4o claude-sonnet --domains news podcast

# Dry run (preview without running)
python kdd_paper/run_summedits.py --all --dry-run
```

### Analyze Results

```bash
# Full analysis report
python scripts/analyze_summedits.py

# Example usage
python examples/summedits_example.py
```

## Available Domains

| Domain | Description |
|--------|-------------|
| `news` | News articles |
| `podcast` | Podcast transcripts |
| `billsum` | Legal bills |
| `samsum` | Conversations |
| `sales_call` | Sales calls |
| `sales_email` | Sales emails |
| `shakespeare` | Shakespeare texts |
| `scitldr` | Scientific papers |
| `qmsumm` | Meetings |
| `ectsum` | Earnings calls |

## Output Location

Scores saved to `data/`:
- `summedits_news_scores.json`
- `summedits_podcast_scores.json`
- etc.

## Performance Expectations

| Model Class | Expected Score |
|-------------|----------------|
| GPT-4 | 80-85% |
| Claude 3+ | 75-80% |
| Gemini Pro | 70-75% |
| Smaller | 50-65% |
| Human | ~90% |

## Cost & Time

**Cost**: ~$0.05-0.50 per model for all 10 domains
**Time**: ~5-10 minutes per domain per model (with 10 threads)

## Common Issues

**No qualified models found**
→ Check `models_cache.json` has `openrouter_id` fields

**Rate limits**
→ Script auto-retries, or reduce threads: `--threads 3`

**Domain not evaluated**
→ Run: `python kdd_paper/run_summedits.py --all --domains <domain>`

## Integration Example

```python
# Load scores
import json
with open('data/summedits_aggregate_scores.json') as f:
    scores = json.load(f)

# Use in quality scoring
quality = 0.3*intelligence + 0.15*summedits[model_id] + ...
```

## Full Documentation

- **Complete Guide**: `kdd_paper/SUMMEDITS_GUIDE.md`
- **Implementation Details**: `SUMMEDITS_IMPLEMENTATION.md`
- **Data Source**: `factualNLG/README.md`

