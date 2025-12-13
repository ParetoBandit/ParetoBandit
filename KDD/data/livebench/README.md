# LiveBench - Contamination-Free Reasoning Benchmark

This directory contains LiveBench prompts, labels, model responses, and evaluation scripts used for reasoning complexity analysis.

## Overview

**LiveBench** is a contamination-free benchmark that releases new questions monthly (after model training cutoffs) to test genuine reasoning capabilities without memorization. This folder contains our evaluation data and analysis scripts.

**Key Features**:
- ✅ **Contamination-Free**: Questions released after model training
- ✅ **Auto-Updating**: New problems added monthly
- ✅ **Reasoning Focus**: Math, logic, spatial reasoning tasks
- ✅ **Complexity Threshold**: τ < 0.45 (45%) for complex classification
- ✅ **Model Responses**: Actual model outputs on LiveBench tasks

## Directory Contents

```
livebench/
├── README.md                              # This documentation
├── livebench_complexity.csv               # Prompts with complexity scores (301 rows)
├── livebench_simple_complex.csv           # Classified prompts (151 rows)
├── run_crs_reasoning_evaluation.py        # Evaluation script (CRS × Reasoning)
└── evaluation_results.json                # Model responses and results (2.0 MB)
```

## Data Files

### 1. livebench_complexity.csv (301 rows)

Prompts from LiveBench with computed complexity scores.

**Columns**:
- `source`: Dataset source (e.g., "LiveBench-Math")
- `task`: Specific task type (e.g., "olympiad")
- `reasoning_score`: Reasoning complexity score (0.0-1.0)
- `creativity_score`: Creativity component score
- `complexity_score`: Overall complexity score
- `is_complex`: Boolean complexity classification
- `level`: Difficulty level (Low/Mid/High)

**Sample**:
```csv
source,task,reasoning_score,creativity_score,complexity_score,is_complex,level
LiveBench-Math,olympiad,0.4272,0.0209,0.40414,True,Mid (0.3-0.6)
LiveBench-Math,olympiad,0.3324,0.0139,0.37911,False,Mid (0.3-0.6)
LiveBench-Math,olympiad,0.4833,0.026,0.42912,True,Mid (0.3-0.6)
```

**Statistics**:
- Total prompts: 301
- Complex prompts (is_complex=True): ~60%
- Primarily from LiveBench-Math olympiad tasks
- Reasoning scores range: 0.298 to 0.543

### 2. livebench_simple_complex.csv (151 rows)

Subset of prompts with explicit Simple/Complex labels for stratified sampling.

**Columns**:
- Same as `livebench_complexity.csv` plus:
- `complexity`: Explicit "Simple" or "Complex" label

**Sample**:
```csv
source,task,reasoning_score,creativity_score,complexity_score,is_complex,level,complexity
LiveBench-Math,olympiad,0.4272,0.0209,0.40414,True,Mid (0.3-0.6),Complex
LiveBench-Math,olympiad,0.3324,0.0139,0.37911,False,Mid (0.3-0.6),Simple
```

**Usage**: Used for stratified sampling in evaluation (ensures balanced representation of simple vs complex prompts)

### 3. run_crs_reasoning_evaluation.py

Main evaluation script that:
1. Loads 20 selected models with CRS scores
2. Loads 149 downsampled reasoning prompts (including LiveBench)
3. Runs each model on each prompt via OpenRouter API
4. Collects ground truth answers and model responses
5. Computes accuracy and analyzes CRS correlation
6. Saves results to `evaluation_results.json`

**Key Features**:
- Loads LiveBench prompts using `_load_livebench()` function
- Handles multiple LiveBench categories (Math, etc.)
- Includes retry logic and rate limiting
- Checkpointing for long-running evaluations
- Regression analysis: P(correct) = f(CRS, reasoning_score, interaction)

**API Calls**: 20 models × 149 prompts = 2,980 calls

### 4. evaluation_results.json (2.0 MB)

Complete evaluation results with model responses.

**Structure**:
```json
[
  {
    "model_name": "Gemini 3 Pro Preview (high)",
    "openrouter_id": "google/gemini-3-pro-preview",
    "crs": 1.829,
    "crs_norm": 1.0,
    "crs_quartile": "Q4 (High)",
    "source": "LiveBench-Math",
    "reasoning_score": 0.4272,
    "complexity_level": "Complex",
    "ground_truth": "B",
    "response": "B",
    "is_correct": true,
    "success": true
  }
]
```

**Fields**:
- `model_name`: Human-readable model name
- `openrouter_id`: Model identifier for API
- `crs`: Composite Reasoning Score (latent factor)
- `crs_norm`: Normalized CRS (0-1 scale)
- `crs_quartile`: CRS quartile (Q1-Q4)
- `source`: Dataset source (includes LiveBench)
- `reasoning_score`: Prompt complexity score
- `complexity_level`: Simple/Medium/Complex
- `ground_truth`: Correct answer
- `response`: Model's answer
- `is_correct`: Whether answer matches ground truth
- `success`: Whether API call succeeded

**Statistics**:
- Total responses: ~2,980 (20 models × 149 prompts)
- Includes LiveBench-Math responses
- Accuracy tracked by complexity level
- Full responses stored for analysis

## LiveBench Task Categories

### Math (Olympiad-Level)

**Description**: Competition-level mathematics problems requiring:
- Multi-step problem solving
- Abstract mathematical reasoning
- Pattern recognition
- Logical deduction

**Difficulty**: Most problems in "Mid" range (0.3-0.6 complexity)
- Easy: < 0.3 (basic calculations)
- **Mid: 0.3-0.6** (requires reasoning) ← Most LiveBench problems
- Hard: > 0.6 (expert-level)

**Complexity Classification**:
- **Complex** (is_complex=True): reasoning_score > threshold
- **Simple** (is_complex=False): reasoning_score < threshold

## Complexity Scoring Methodology

### Reasoning Score Calculation

Prompts are scored using the NVIDIA Complexity Classifier:

```python
# Pseudo-code
reasoning_score = nvidia_complexity_classifier(prompt)
is_complex = (reasoning_score > threshold)
```

**Threshold**: τ = 0.40 (empirically determined)
- Above 0.40: Classified as Complex
- Below 0.40: Classified as Simple

### Complexity Score Components

```
complexity_score ≈ reasoning_score + creativity_score
```

- **reasoning_score**: Logical/analytical complexity (dominant)
- **creativity_score**: Open-endedness component (minor for LiveBench)

### Level Categorization

| Level | Score Range | Description |
|-------|-------------|-------------|
| Low | < 0.3 | Basic reasoning, straightforward |
| **Mid** | **0.3-0.6** | **Multi-step reasoning** (most LiveBench) |
| High | > 0.6 | Expert-level, complex reasoning |

## Model Performance Analysis

### CRS × Complexity Interaction

The evaluation script tests the hypothesis:

**H**: Models with high CRS should perform better on complex prompts than simple prompts, relative to low-CRS models.

**Regression Model**:
```
P(correct) = β₀ + β₁·CRS + β₂·complexity + β₃·CRS×complexity
```

Where:
- β₃ > 0: High-CRS models excel on complex prompts
- β₃ < 0: High-CRS models struggle relative to their simple-prompt performance

### Results Access

Results are in `evaluation_results.json`:

```python
import json
import pandas as pd

# Load results
with open('evaluation_results.json') as f:
    results = json.load(f)

df = pd.DataFrame(results)

# Filter LiveBench responses
livebench_df = df[df['source'].str.contains('LiveBench')]

# Accuracy by complexity
accuracy_by_complexity = livebench_df.groupby('complexity_level')['is_correct'].mean()
print(accuracy_by_complexity)
```

## Usage

### Analyzing LiveBench Prompts

```python
import pandas as pd

# Load complexity scores
df = pd.read_csv('livebench_complexity.csv')

# Get complex prompts
complex_prompts = df[df['is_complex'] == True]
print(f"Complex prompts: {len(complex_prompts)} / {len(df)}")

# Distribution by level
print(df['level'].value_counts())
```

### Running Evaluation

```bash
# The evaluation script requires:
# - OPENROUTER_API_KEY environment variable
# - HuggingFace datasets library
# - Selected models and prompts CSV files

cd KDD/quality_score_complexity_models
python run_crs_reasoning_evaluation.py
```

**Note**: Script is in parent directory but references these data files.

### Extracting LiveBench Responses

```python
import json
import pandas as pd

# Load evaluation results
with open('evaluation_results.json') as f:
    results = json.load(f)

df = pd.DataFrame(results)

# Filter to LiveBench only
livebench = df[df['source'].str.startswith('LiveBench')]

# Accuracy by model
model_accuracy = livebench.groupby('model_name')['is_correct'].mean().sort_values(ascending=False)
print(model_accuracy)

# Accuracy by complexity
complexity_accuracy = livebench.groupby('complexity_level')['is_correct'].mean()
print(complexity_accuracy)
```

## LiveBench Categories in Data

Based on the CSV files, we primarily have:

### LiveBench-Math (Olympiad)
- **Count**: 301 prompts in complexity.csv, 151 in simple_complex.csv
- **Task Type**: Olympiad-level mathematics
- **Complexity Range**: 0.298 to 0.543 (reasoning scores)
- **Typical Level**: Mid (0.3-0.6)
- **Complex Rate**: ~60% classified as complex

## Integration with Project

### Connection to CRS

LiveBench is used to **validate** the Composite Reasoning Score (CRS):
1. CRS is computed from MATH-500, GPQA, HLE, AIME benchmarks
2. LiveBench provides **independent validation** (not used in CRS computation)
3. If CRS is valid, high-CRS models should perform better on complex LiveBench prompts

### Connection to Complexity Threshold

From `BENCHMARK_THRESHOLDS.md`:
- **Threshold**: τ < 0.45 (45%) for LiveBench reasoning tasks
- **Basis**: SOTA models (GPT-4o, Claude 3.5) score 53-58%
- **Interpretation**: Below 45% = below majority success = complex

### Union-Based Reasoning Classification

For the **Reasoning** intent:

```
C_reasoning(x) = 1 if P_GPQA(x) < 0.34 OR P_LiveBench(x) < 0.45
```

A prompt is complex if it fails either:
- **GPQA** (< 34%): Requires expert domain knowledge
- **LiveBench** (< 45%): Requires advanced logical reasoning

## Data Provenance

### Source
- **LiveBench Dataset**: https://livebench.ai
- **Access Method**: HuggingFace datasets library
- **Collection Date**: December 2024
- **Questions**: Monthly releases (contamination-free)

### Processing
1. Prompts fetched from LiveBench via HuggingFace
2. Complexity scores computed using NVIDIA classifier
3. Stratified sampling for balanced evaluation
4. Model responses collected via OpenRouter API
5. Ground truth comparison for accuracy

### Data Authenticity
✅ **Prompts from official LiveBench dataset**  
✅ **Complexity scores from validated classifier**  
✅ **Model responses from actual API calls**  
✅ **Ground truth from benchmark dataset**  

## References

### LiveBench Citation

```bibtex
@inproceedings{white2024livebench,
  title={LiveBench: A Challenging, Contamination-Free LLM Benchmark},
  author={White, Colin and Dooley, Samuel and Roberts, Manley and Pal, Arka and others},
  booktitle={NeurIPS Datasets and Benchmarks Track},
  year={2024},
  url={https://livebench.ai/}
}
```

### Related Documentation
- **Threshold Methodology**: `KDD/data/BENCHMARK_THRESHOLDS.md`
- **Complexity Analysis**: `KDD/quality_score_complexity_models/`
- **CRS Documentation**: `quality_scoring/docs/COMPOSITE_REASONING_SCORE.md`

## File Sizes

| File | Size | Rows | Description |
|------|------|------|-------------|
| livebench_complexity.csv | 21 KB | 301 | All prompts with scores |
| livebench_simple_complex.csv | 12 KB | 151 | Stratified sample |
| evaluation_results.json | 2.0 MB | ~2,980 | Model responses |
| run_crs_reasoning_evaluation.py | 28 KB | 801 lines | Evaluation script |

**Total**: ~2.05 MB of LiveBench evaluation data

## Next Steps

### For Analysis
1. Load `evaluation_results.json` to analyze model performance
2. Compare accuracy on complex vs simple LiveBench prompts
3. Validate CRS correlation with LiveBench performance
4. Compute per-model LiveBench scores

### For Extension
1. Add more LiveBench categories (Coding, Language, etc.)
2. Track performance over time as new questions release
3. Correlate with GPQA for union-based complexity
4. Expand model coverage beyond initial 20

## Contact

For questions about LiveBench data or evaluation methodology:
- Review evaluation script: `run_crs_reasoning_evaluation.py`
- Check threshold documentation: `BENCHMARK_THRESHOLDS.md`
- Consult CRS validation results: `KDD/composite_quality_scores/`

**Last Updated**: December 13, 2025  
**Data Collection**: December 2024  
**Evaluation**: 20 models × 149 prompts (including LiveBench-Math)
