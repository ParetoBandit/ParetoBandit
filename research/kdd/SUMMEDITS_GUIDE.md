# SummEdits Evaluation Guide

## Overview

The SummEdits benchmark evaluates factual consistency in summarization. It's a binary classification task where models must determine if a summary is factually consistent with its source document.

**Key Features:**
- 10 diverse domains (news, podcasts, legal documents, scientific papers, etc.)
- ~1,000 samples per domain
- Binary classification: "Yes" (consistent) or "No" (inconsistent)
- **Very cost-efficient**: Only 1 token generation per sample
- Metric: **Balanced Accuracy** (0.0 to 1.0)

## Source

Maintained by Salesforce AI Research: https://github.com/salesforce/factualNLG

## Setup

The repository has already been cloned and dependencies installed. If needed:

```bash
# Clone repository (already done)
git clone https://github.com/salesforce/factualNLG

# Install dependencies (already done)
cd factualNLG
pip install -r requirements.txt
```

## Usage

### Basic Usage

Evaluate all qualified models on the news domain:

```bash
python kdd_paper/run_summedits.py --all
```

### Evaluate Specific Models

```bash
python kdd_paper/run_summedits.py --models gpt-4o claude-sonnet-4 gemini-2-flash
```

### Evaluate Multiple Domains

```bash
# Evaluate on news and podcast domains
python kdd_paper/run_summedits.py --all --domains news podcast

# Evaluate on all 10 domains
python kdd_paper/run_summedits.py --all --domains all
```

### Control Sample Size

```bash
# Limit to 100 samples per domain (for quick testing)
python kdd_paper/run_summedits.py --all --max-samples 100
```

### Adjust Parallelism

```bash
# Use 10 parallel threads (safe since only 1 token per call)
python kdd_paper/run_summedits.py --all --threads 10
```

### Dry Run (Preview)

```bash
# See what would be evaluated without actually running
python kdd_paper/run_summedits.py --all --dry-run
```

## Available Domains

1. **news** - News article summaries
2. **podcast** - Podcast transcription summaries
3. **billsum** - Legal bill summaries
4. **samsum** - Conversation summaries
5. **sales_call** - Sales call transcripts
6. **sales_email** - Sales email summaries
7. **shakespeare** - Shakespeare text summaries
8. **scitldr** - Scientific paper summaries (TL;DR)
9. **qmsumm** - Meeting summaries
10. **ectsum** - Earnings call transcripts

## Output

### Scores Storage

Results are saved to:
- `data/summedits_news_scores.json`
- `data/summedits_podcast_scores.json`
- etc.

Format:
```json
{
  "openai/gpt-4o": 85.3,
  "anthropic/claude-sonnet-4": 82.1,
  ...
}
```

Values are **balanced accuracy percentages** (0-100).

### Console Output

During evaluation, you'll see:
- Progress bar for each model
- Balanced accuracy (primary metric)
- Simple accuracy (for reference)
- Error count

Example:
```
======================================================================
Evaluating: GPT-4o
Model ID: openai/gpt-4o
Domain: news
Samples: 1024
======================================================================
  [██████████████████████████████] 100.0% (1024/1024) 
  ✅ Balanced Accuracy: 83.3%
  📊 Simple Accuracy: 85.1% (871/1024)
  ❌ Errors/Unparseable: 2
```

## Metrics Explained

### Balanced Accuracy (Primary Metric)

Balanced accuracy is the average of:
- **Sensitivity**: Recall for consistent samples (True Positives / All Positives)
- **Specificity**: Recall for inconsistent samples (True Negatives / All Negatives)

This metric handles class imbalance better than simple accuracy.

**Formula:**
```
Balanced Accuracy = (Sensitivity + Specificity) / 2
```

### Why Balanced Accuracy?

SummEdits datasets can be imbalanced (e.g., 60% consistent, 40% inconsistent). A model that always predicts "consistent" would get 60% simple accuracy but only 50% balanced accuracy.

## Integration with models_cache.json

The script automatically:
1. Loads models from `data/models_cache.json`
2. Filters to models with:
   - Valid `hallucination_rate`
   - Required benchmark scores (intelligence, coding, math indices)
   - OpenRouter API access (`openrouter_id`)
3. Skips models that already have scores for the requested domains
4. Deduplicates by `openrouter_id`

## Cost Estimation

SummEdits is **extremely cost-efficient** compared to other benchmarks:

- **1 token** per sample (just "Yes" or "No")
- ~1,000 samples per domain
- 10 domains total = ~10,000 tokens per model

**Example cost per model:**
- GPT-4o: ~$0.50 for all 10 domains
- Claude Sonnet: ~$0.30 for all 10 domains
- Cheaper models: ~$0.05-0.15 for all 10 domains

Compare to:
- MixEval: ~1,000 tokens per sample
- IFEval: ~500 tokens per sample

## Performance Tips

### 1. Higher Thread Count

Since we only generate 1 token, rate limits are less of a concern:

```bash
# Use 10-15 threads for faster evaluation
python kdd_paper/run_summedits.py --all --threads 15
```

### 2. Batch Multiple Domains

Evaluate all domains in one run:

```bash
python kdd_paper/run_summedits.py --all --domains all
```

### 3. Sample for Testing

Use `--max-samples` for quick tests:

```bash
# Test with 50 samples per domain
python kdd_paper/run_summedits.py --all --max-samples 50
```

## Troubleshooting

### "No qualified models found"

Check that your `models_cache.json` has models with:
- `hallucination_rate > 0`
- `intelligence_index`, `coding_index`, `math_index` scores
- `openrouter_id` field

### Rate Limit Errors

The script includes automatic retry with exponential backoff. If you still hit rate limits:

1. Reduce thread count: `--threads 3`
2. Increase delay in code (edit `time.sleep(0.05)` to `time.sleep(0.2)`)

### Parsing Errors

The script expects responses starting with "Yes" or "No". If a model frequently fails to parse:

1. Check the prompt template in `factualNLG/prompts/summedits/standard_zs_prompt.txt`
2. Review unparseable responses in logs
3. Adjust `parse_response()` function if needed

## Example Workflows

### Quick Test on One Domain

```bash
# Test 3 models on 100 news samples
python kdd_paper/run_summedits.py \
  --models gpt-4o claude-sonnet gemini-flash \
  --domains news \
  --max-samples 100
```

### Full Evaluation on All Domains

```bash
# Evaluate all models on all domains (will take hours)
python kdd_paper/run_summedits.py \
  --all \
  --domains all \
  --threads 10
```

### Incremental Evaluation

```bash
# Run once - automatically skips models with existing scores
python kdd_paper/run_summedits.py --all --domains news

# Run again - only evaluates new models added to models_cache.json
python kdd_paper/run_summedits.py --all --domains news
```

## Expected Results

Based on the original SummEdits paper (see factualNLG README):

| Model Tier | Expected Balanced Accuracy |
|------------|---------------------------|
| GPT-4 class | 80-85% |
| Claude 3+ | 75-80% |
| Gemini Pro | 70-75% |
| Smaller models | 50-65% |
| Random baseline | 50% |

Human performance: ~90%

## Adding SummEdits to Quality Scoring

Once you have scores, you can integrate them into your quality scoring system:

```python
# In your quality scoring module
summedits_scores = load_scores("data/summedits_news_scores.json")

# Average across domains for overall summarization quality
all_domains = ["news", "podcast", "billsum", ...]
model_summedits_avg = average([
    load_scores(f"data/summedits_{domain}_scores.json")[model_id]
    for domain in all_domains
])

# Weight in your composite quality score
quality_score = (
    0.3 * intelligence_index +
    0.2 * coding_index +
    0.15 * math_index +
    0.15 * summedits_avg +  # Summarization quality
    0.1 * hallucination_rate_inv +
    0.1 * other_metrics
)
```

## References

- Paper: "Evaluating Factual Consistency in Summarization" (Salesforce AI Research)
- Repo: https://github.com/salesforce/factualNLG
- Benchmark details: See `factualNLG/README.md`

