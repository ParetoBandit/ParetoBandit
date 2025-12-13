# SummEdits Dataset

This directory contains the SummEdits benchmark datasets, prompts, and model evaluation results.

## Overview

SummEdits is a binary classification benchmark for evaluating factual consistency in summarization. Given a (Document, Summary) pair, models must determine if the summary is factually consistent with the document.

**Benchmark Source**: https://github.com/salesforce/factualNLG

**Evaluation Metric**: Balanced Accuracy (0.0 to 1.0)

## Directory Structure

```
sumedits/
├── README.md                           # This documentation file
├── run_summedits.py                    # Main evaluation script
├── aggregate_summedits_scores.py       # Score aggregation script
├── ground_truth/                       # Original SummEdits datasets with ground truth labels
├── prompts/                            # Prompt templates used for evaluation
└── model_scores/                       # Model evaluation results and scores
```

## Ground Truth Datasets

Location: `ground_truth/`

Contains 10 domain-specific SummEdits datasets:

1. **summedits_news.json** - News article summaries (3.0 MB)
2. **summedits_podcast.json** - Podcast transcript summaries (3.1 MB)
3. **summedits_billsum.json** - Bill summaries (8.6 MB)
4. **summedits_samsum.json** - Conversation summaries (770 KB)
5. **summedits_sales_call.json** - Sales call summaries (1.6 MB)
6. **summedits_sales_email.json** - Sales email summaries (2.0 MB)
7. **summedits_shakespeare.json** - Shakespeare text summaries (4.6 MB)
8. **summedits_scitldr.json** - Scientific paper summaries (762 KB)
9. **summedits_qmsumm.json** - Query-based summaries (1.6 MB)
10. **summedits_ectsum.json** - Earnings call summaries (1.1 MB)

Each dataset contains:
- `id`: Unique identifier for the example
- `doc`: Source document text
- `summary`: Summary text to evaluate
- `label`: Ground truth (1 = factually consistent, 0 = inconsistent)
- `original_summary`: Original unmodified summary
- `edit_types`: Types of edits applied (if any)
- `split`: Train/validation/evaluation split designation

## Prompts

Location: `prompts/`

Contains prompt templates used for model evaluation:

- **standard_zs_prompt.txt** - Standard zero-shot prompt
- **step2_consistent.txt** - Two-step prompt for consistent examples
- **step2_inconsistent.txt** - Two-step prompt for inconsistent examples
- **edit_typing_gpt4.txt** - GPT-4 edit typing prompt

## Model Scores

Location: `model_scores/`

Contains evaluation results for all tested models:

### Per-Domain Scores
- `summedits_news_scores.json`
- `summedits_podcast_scores.json`
- `summedits_billsum_scores.json`
- `summedits_samsum_scores.json`
- `summedits_sales_call_scores.json`
- `summedits_sales_email_scores.json`
- `summedits_shakespeare_scores.json`
- `summedits_scitldr_scores.json`
- `summedits_qmsumm_scores.json`
- `summedits_ectsum_scores.json`

Each file contains model-wise balanced accuracy scores (0-100 scale).

### Aggregated Results
- **summedits_aggregate_scores.json** - Average scores across all domains
- **summedits_aggregate_detailed.json** - Detailed breakdown with per-domain statistics

## How Model Scores Were Obtained

### Evaluation Process

The model scores were generated using the evaluation pipeline in `research/kdd/run_summedits.py`. Here's the detailed process:

#### 1. Task Format
- **Task Type**: Binary classification
- **Question**: "Is the summary factually consistent with the document?"
- **Expected Output**: "Yes" (consistent) or "No" (inconsistent)
- **Efficiency**: Only 1 token generation required per sample

#### 2. Prompt Template
Models were evaluated using the standard zero-shot prompt (`prompts/standard_zs_prompt.txt`):

```
Given the document below, you have to determine if "Yes" or "No", 
the summary is factually consistent with the document.

Document:
[ARTICLE]

Summary:
[SUMMARY_SENTENCES]

Is the summary factually consistent with the document? (Yes/No)
Start your answer explicitly with "Yes" or "No", and if you answer no, 
explain which sentence is inconsistent and why.
```

#### 3. Model Inference
- **API**: Models called via OpenRouter API
- **Parameters**: 
  - Temperature: 0 (for deterministic responses)
  - Max tokens: 100-500 (reasoning models use `max_completion_tokens`)
  - Rate limiting: 50ms delay between calls
- **Retry Logic**: Exponential backoff for rate limits, up to 3 retries per sample

#### 4. Response Parsing
Responses are parsed using a robust multi-strategy approach:
- First, check the first 50 characters for explicit "Yes" or "No"
- Look for patterns like "Answer: Yes", "**Yes**", "Conclusion: No"
- Check last 5 lines/sentences for Yes/No
- Fall back to counting occurrences (with filtering for phrases like "no issues")
- Unparseable responses default to "Yes" (conservative assumption)

#### 5. Metric Calculation
- **Primary Metric**: **Balanced Accuracy**
  - Formula: `(Sensitivity + Specificity) / 2`
  - Sensitivity: True Positive Rate = TP / (TP + FN)
  - Specificity: True Negative Rate = TN / (TN + FP)
  - Balanced accuracy is used because the dataset may have class imbalance
- **Secondary Metric**: Simple accuracy (for reference)
- **Scores Reported**: Percentages (0-100 scale)

#### 6. Data Splits
- **Split Used**: "evaluation" split from each domain dataset
- **Sampling**: Random sampling with seed=42 if `max_samples` specified
- **Coverage**: All 10 domains evaluated separately

#### 7. Model Selection
Models were selected from the models cache (`data/models_cache.json`) based on:
- Availability via OpenRouter API
- Hallucination rate thresholds
- Domain-specific completion requirements

#### 8. Score Aggregation
After per-domain evaluation:
- Scores saved to individual domain files (e.g., `summedits_news_scores.json`)
- Aggregate script (`research/kdd/aggregate_summedits_scores.py`) computes:
  - Mean balanced accuracy across all 10 domains
  - Per-domain breakdowns
  - Statistical summaries

### Reproducibility

To reproduce these scores:
```bash
# Navigate to this directory
cd KDD/data/sumedits/

# Evaluate a single model on all domains
python run_summedits.py --models <model-slug>

# Evaluate all qualified models
python run_summedits.py --all

# Aggregate results across domains
python aggregate_summedits_scores.py
```

**Note**: The scripts automatically handle paths relative to the project root, so they can be run from this directory or from `research/kdd/`.

### Quality Assurance
- Responses are logged for debugging
- Error counts tracked per model
- First 3 unparseable responses logged for inspection
- Scores written to disk with explicit flush and fsync for reliability

## Evaluation Scripts

### run_summedits.py
Main evaluation script for running SummEdits benchmark on models via OpenRouter API.

**Features**:
- Evaluates models on all 10 SummEdits domains
- Supports single model or batch evaluation
- Robust response parsing and error handling
- Progress tracking with visual progress bars
- Automatic retry logic for rate limits
- Saves results incrementally per domain

**Usage**:
```bash
# Evaluate a specific model on all domains
python run_summedits.py --models <model-slug>

# Evaluate multiple specific models
python run_summedits.py --models gpt-4 claude-3-opus

# Evaluate all qualified models
python run_summedits.py --all

# Evaluate on specific domains only
python run_summedits.py --models <model-slug> --domains news podcast

# Dry run (no API calls)
python run_summedits.py --models <model-slug> --dry-run

# Force re-evaluation (ignore existing scores)
python run_summedits.py --models <model-slug> --force
```

### aggregate_summedits_scores.py
Aggregates per-domain scores into overall statistics.

**Features**:
- Computes mean balanced accuracy across all domains
- Generates detailed per-domain breakdowns
- Creates summary statistics
- Produces both aggregate and detailed output files

**Usage**:
```bash
# Aggregate all existing scores
python aggregate_summedits_scores.py
```

**Output**:
- `model_scores/summedits_aggregate_scores.json` - Mean scores per model
- `model_scores/summedits_aggregate_detailed.json` - Full per-domain breakdown

## Data Files Source Locations

These files were consolidated from:
- Ground truth: `factualNLG/data/summedits/`
- Prompts: `factualNLG/prompts/summedits/`
- Model scores: `data/summedits_*_scores.json`

## Notes

- This is an efficient benchmark requiring only 1 token generation per sample ("Yes" or "No")
- Scores are reported as balanced accuracy percentages (0-100)
- The benchmark tests factual consistency, a key aspect of summarization quality
