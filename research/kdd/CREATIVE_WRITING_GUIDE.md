## Creative Writing Benchmark Guide

## Overview

The Creative Writing Benchmark v3 evaluates the creative writing capabilities of large language models using a hybrid rubric and Elo scoring system. It's designed for enhanced discrimination, especially at the top end of model performance.

**Key Features:**
- 32 diverse creative writing prompts across multiple genres
- 3 iterations per prompt (96 total generations)
- Rubric scoring (0-100) for absolute quality
- **Elo ratings** from pairwise comparisons for relative ranking
- Judge model: Claude Sonnet 4 (recommended for leaderboard parity)
- **Metric**: Normalized Elo Score (anchored to leaderboard)

## Source

Maintained by EQ-Bench: https://github.com/EQ-bench/creative-writing-bench  
Leaderboard: https://eqbench.com/creative_writing.html

## Setup

The repository has already been cloned and dependencies installed. If needed:

```bash
# Clone repository (already done)
git clone https://github.com/EQ-bench/creative-writing-bench.git

# Install dependencies (already done)
cd creative-writing-bench
pip install -r requirements.txt

# Install trueskill (if not in requirements.txt)
pip install trueskill

# Download NLTK data (already done)
python -c "import nltk; nltk.download('punkt'); nltk.download('cmudict')"
```

## Usage

### Basic Usage

Evaluate all qualified models:

```bash
python kdd_paper/run_creative_writing.py --all
```

### Evaluate Specific Models

```bash
python kdd_paper/run_creative_writing.py --models gpt-4o claude-sonnet-4 gemini-2-flash
```

### Custom Judge Model

```bash
# Use different judge model (not recommended for leaderboard comparison)
python kdd_paper/run_creative_writing.py --all --judge-model anthropic/claude-3-opus
```

### Adjust Iterations

```bash
# Quick test with 1 iteration (not recommended for leaderboard)
python kdd_paper/run_creative_writing.py --models gpt-4o --iterations 1

# Standard: 3 iterations (recommended)
python kdd_paper/run_creative_writing.py --all --iterations 3
```

### Dry Run (Preview)

```bash
# See what would be evaluated without actually running
python kdd_paper/run_creative_writing.py --all --dry-run
```

## How It Works

### Evaluation Process

1. **Generation**: Model generates responses to 32 prompts × 3 iterations = 96 pieces
   - Temperature: 0.7
   - Min_p: 0.1
   - Max tokens: ~1000 per piece

2. **Rubric Scoring**: Each piece is judged against comprehensive rubric
   - Criteria: originality, character depth, prose quality, humor, etc.
   - Score: 0-20 per piece, aggregated to 0-100 scale

3. **Initial Elo Inference**: Rubric score estimates initial Elo position

4. **Pairwise Matchups**: Model compared against neighbors on leaderboard
   - Judge picks better output
   - Win margin scored with `+` symbols

5. **Glicko-2 Calculation**: Elo ratings computed with Glicko-2 system
   - Accounts for rating uncertainty
   - Incorporates win margins

6. **Final Elo**: Definitive leaderboard score after all comparisons

7. **Normalization**: Anchored to reference models for comparability
   - DeepSeek-R1: 1500 Elo
   - Ministral-3B: 200 Elo

## Output

### Scores Storage

Results are saved to:
- `data/creative_writing_scores.json` - Normalized Elo scores

Format:
```json
{
  "openai/gpt-4o": 1543.2,
  "anthropic/claude-sonnet-4": 1612.8,
  ...
}
```

Values are **normalized Elo ratings** (higher is better).

### Full Benchmark Data

Additional data stored in creative-writing-bench/:
- `creative_bench_runs.json` - All run data with generated text
- `elo_results.json` - Detailed Elo analysis and matchups

### Console Output

During evaluation:
- Progress updates for generation and judging
- Rubric scores
- Pairwise comparison results
- Final normalized Elo score

Example:
```
======================================================================
Evaluating: GPT-4o
Model ID: openai/gpt-4o
Judge: anthropic/claude-sonnet-4
Iterations: 3
======================================================================

▶️  Running Creative Writing Benchmark...
   This will take ~15-30 minutes per model

✅ Benchmark completed in 23.4 minutes

🎨 Creative Writing Elo Score: 1543.2
```

## Metrics Explained

### Rubric Score (0-100)

Aggregate score based on judging each piece in isolation against detailed rubric.

**Criteria:**
- Originality and creativity
- Character depth and development
- Prose quality and style
- Humor and wit (where appropriate)
- Emotional resonance
- Spatial awareness
- Coherence and structure

**Range:**
- 90-100: Exceptional creative writing
- 75-90: Strong creative ability
- 60-75: Competent writing
- <60: Basic or weak creative skills

### Elo Score (Primary Metric)

Relative rating derived from pairwise comparisons against other models.

**Advantages:**
- More discriminative at high performance levels
- Accounts for comparative quality
- Less susceptible to scoring saturation

**Disadvantages:**
- Sensitive to pool of compared models
- Requires historical data for normalization

**Typical Ranges:**
- 1600+: Elite creative writers (GPT-4+ class)
- 1400-1600: Excellent creative ability
- 1200-1400: Good creative writing
- 1000-1200: Fair creative writing
- <1000: Limited creative ability

### Why Two Scores?

- **Rubric**: Good for absolute quality assessment
- **Elo**: Better for relative ranking and discrimination

The **normalized Elo score** is the primary metric used for leaderboard ranking and model selection.

## Integration with models_cache.json

The script automatically:
1. Loads models from `data/models_cache.json`
2. Filters to models with:
   - Valid `hallucination_rate`
   - Required benchmark scores (intelligence, coding, math indices)
   - OpenRouter API access (`openrouter_id`)
3. Skips models that already have creative writing scores
4. Deduplicates by `openrouter_id`

## Cost & Time Estimation

### Per Model

**Time**: 15-30 minutes
- Generation: ~5-10 minutes (96 pieces)
- Judging: ~10-20 minutes (rubric + pairwise)

**Cost**: ~$10 per model
- Test model generation: ~$3-5 (96 × 1000 tokens)
- Judge model scoring: ~$5-7 (rubric + pairwise comparisons)

**Total for 55 models**: ~$550 and ~20-30 hours

### Comparison to Other Benchmarks

| Benchmark | Cost per Model | Time per Model |
|-----------|----------------|----------------|
| Creative Writing | $10 | 20 minutes |
| SummEdits (all domains) | $0.50 | 50 minutes |
| MixEval | $50 | 60 minutes |
| IFEval | $20 | 30 minutes |

Creative Writing is mid-range in cost but provides valuable signal for creative tasks.

## Prompt Categories

The 32 prompts cover diverse genres and challenges:

1. **Historical Fiction** - Gladiator scene
2. **Epistolary** - Lighthouse keeper letters
3. **Romance** - Celebrity meets bookstore owner
4. **Sci-Fi** - Generation ship
5. **Fantasy** - Dragon negotiation
6. **Horror** - Psychological thriller
7. **Mystery** - Detective noir
8. **Humor** - Comedic situation
9. **Poetry** - Verse forms
10. **Dialogue** - Character interaction
11. **Description** - Sensory details
12. **Worldbuilding** - Setting creation
... (and 20 more)

Each prompt designed to challenge models in specific creative dimensions.

## Performance Tips

### 1. Use Recommended Settings

For leaderboard-comparable scores:
- Judge: `anthropic/claude-sonnet-4`
- Iterations: 3
- Use canonical runs file (auto-unzipped)

```bash
python kdd_paper/run_creative_writing.py --all \
  --judge-model anthropic/claude-sonnet-4 \
  --iterations 3
```

### 2. Sequential Evaluation

Don't try to parallelize - the benchmark is sequential by design:

```bash
# Good: Sequential evaluation
python kdd_paper/run_creative_writing.py --all

# Avoid: Parallel would waste API credits
# (script forces sequential anyway)
```

### 3. Monitor Progress

The benchmark is verbose - watch console output for:
- Generation progress
- Judging progress
- Elo calculations
- Any errors or retries

### 4. Quick Testing

For quick tests (not leaderboard comparable):

```bash
# Test with 1 iteration on one model
python kdd_paper/run_creative_writing.py \
  --models gpt-4o \
  --iterations 1
```

## Troubleshooting

### "OPENROUTER_API_KEY not found"

Set your OpenRouter API key in `.env`:
```bash
OPENROUTER_API_KEY=your_key_here
```

### Benchmark Takes Too Long

Expected behavior - this is a comprehensive evaluation:
- 96 generations per model
- Rubric scoring for each piece
- Pairwise comparisons
- Total: 15-30 minutes per model

Reduce iterations for faster testing (but not leaderboard-comparable):
```bash
--iterations 1
```

### "Run key not found" Error

The benchmark may have failed to save results. Check:
1. `creative-writing-bench/creative_bench_runs.json` exists
2. Sufficient disk space
3. No permission errors

### Rate Limit Errors

OpenRouter has generous rate limits, but if you hit them:
1. Reduce `--threads` (default 100)
2. Wait a few minutes and retry
3. Check OpenRouter dashboard for limits

### Scores Don't Match Leaderboard

Ensure you're using:
1. Canonical runs file (auto-unzipped from `creative_bench_runs.zip`)
2. Recommended judge: `anthropic/claude-sonnet-4`
3. Standard 3 iterations
4. Latest version of benchmark code

## Example Workflows

### Quick Test on One Model

```bash
# Fast test (not leaderboard-comparable)
python kdd_paper/run_creative_writing.py \
  --models gpt-4o \
  --iterations 1
```

### Full Evaluation

```bash
# Comprehensive evaluation (leaderboard-comparable)
python kdd_paper/run_creative_writing.py \
  --all \
  --judge-model anthropic/claude-sonnet-4 \
  --iterations 3
```

### Evaluate New Models

```bash
# Script automatically skips models with existing scores
python kdd_paper/run_creative_writing.py --all

# Only new models will be evaluated
```

## Expected Results

Based on EQ-Bench leaderboard:

| Model Tier | Expected Elo | Example Models |
|------------|--------------|----------------|
| Elite | 1600+ | GPT-4.5, Claude 3.7 Sonnet |
| Excellent | 1400-1600 | GPT-4o, Claude 3.5 Sonnet |
| Good | 1200-1400 | Gemini 2.0 Flash, Llama 3.1 405B |
| Fair | 1000-1200 | Mistral Nemo, Gemma 27B |
| Basic | <1000 | Smaller models |

## Adding Creative Writing to Quality Scoring

Once you have scores, integrate them into your quality scoring:

```python
# Load scores
with open('data/creative_writing_scores.json') as f:
    creative_scores = json.load(f)

# Normalize to 0-100 scale (Elo ~200-1700 → 0-100)
def normalize_elo(elo):
    return max(0, min(100, (elo - 200) / 15))

creative_normalized = normalize_elo(creative_scores[model_id])

# Use in composite quality score
quality_score = (
    0.25 * intelligence_index +
    0.15 * coding_index +
    0.15 * math_index +
    0.10 * summedits_avg +
    0.15 * creative_normalized +  # NEW!
    0.10 * (100 - hallucination_rate) +
    0.10 * other_metrics
)
```

## Bias Mitigation

The benchmark attempts to control for:

- ✅ **Length bias**: Outputs truncated to 4000 characters
- ✅ **Position bias**: A/B and B/A orders averaged
- ✅ **Verbosity bias**: Penalized in rubric
- ✅ **Poetic incoherence**: Specific judging criteria

**Not controlled:**
- Judge self-bias
- Stylistic preferences  
- "Slop" bias (overused tropes)
- NSFW aversion

Always supplement scores by reading sample outputs!

## Limitations

1. **Subjectivity**: Creative quality is inherently subjective
2. **Judge limitations**: Even Sonnet 4 may miss nuances
3. **Not for RP**: Doesn't assess conversational roleplay
4. **English only**: Currently English language only
5. **Cost**: $10 per model adds up for large evaluations
6. **Time**: 20+ minutes per model

**View scores as a guide, not absolute truth.**

## References

- **Paper**: "EQ-Bench Creative Writing Benchmark v3"
- **Repository**: https://github.com/EQ-bench/creative-writing-bench
- **Leaderboard**: https://eqbench.com/creative_writing.html
- **About**: https://eqbench.com/about.html#creative-writing-v3

## Citation

```bibtex
@misc{creative-writing-bench-v3,
  author = {Samuel J Paech},
  title = {EQ-Bench Creative Writing Benchmark v3},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/EQ-bench/creative-writing-bench}}
}
```

