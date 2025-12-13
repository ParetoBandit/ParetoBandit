# LiveCodeBench - Contamination-Free Code Generation Benchmark

This directory contains LiveCodeBench benchmark data, evaluation scripts, and model scores for code generation tasks.

## Overview

**LiveCodeBench** is a contamination-free coding benchmark containing problems from competitive programming platforms **after 2023**, minimizing the risk of data contamination in model training. Each problem includes test cases (inputs and expected outputs) for execution-based evaluation.

**Key Features**:
- ✅ **Contamination-Free**: Problems from after 2023
- ✅ **Execution-Based**: Unit test evaluation (Pass@1 metric)
- ✅ **Free Evaluation**: Local CPU execution, no API costs
- ✅ **Complete Test Cases**: Inputs and expected outputs included
- ✅ **Real-World Problems**: From actual competitive programming contests

## Directory Structure

```
coding/
├── README.md                      # This documentation
├── livecodebench_scores.json      # Pre-computed scores from Artificial Analysis
├── fetch_livecodebench.py         # Fetch problems with test cases from HuggingFace
├── evaluate_code.py               # Execute code and compute Pass@1 metric
└── run_evaluation.py              # End-to-end evaluation script (TODO)
```

## Benchmark Details

### Dataset Information

- **Name**: LiveCodeBench (Code Generation Lite)
- **Source**: HuggingFace `livecodebench/code_generation_lite`
- **Size**: ~400+ problems
- **Time Range**: Problems from 2024+ (contamination-free)
- **Languages**: Python primarily
- **Platforms**: LeetCode, Codeforces, AtCoder, etc.

### Problem Format

Each problem includes:
- **Title**: Problem name
- **Description**: Problem statement
- **Difficulty**: Easy, Medium, Hard
- **Test Cases**: Multiple input/output pairs
  - `input`: Test input
  - `output`: Expected output
  - `explanation`: Optional explanation
- **Metadata**: Platform, contest date, topics, URL

### Evaluation Metric

**Pass@1** (Pass-at-one):
- Generate one solution per problem
- Execute against all test cases
- Problem passes if ALL test cases pass
- Metric = % of problems passed

**Formula**:
```
Pass@1 = (# problems with all tests passing) / (# total problems)
```

## Data Files

### 1. livecodebench_scores.json

Pre-computed LiveCodeBench scores from Artificial Analysis API.

**Format**:
```json
{
  "models": [
    {
      "name": "Claude 3.5 Haiku",
      "slug": "claude-3-5-haiku",
      "creator_name": "Anthropic",
      "livecodebench": 0.314,
      "source": "artificial_analysis_api"
    }
  ]
}
```

**Coverage**: 82 out of 83 models (98.8%)

**Score Interpretation**:
- Scale: 0.0 to 1.0 (0% to 100%)
- Example: 0.314 = 31.4% of problems passed

## Scripts

### 1. fetch_livecodebench.py

Fetches LiveCodeBench problems with test cases from HuggingFace.

**Features**:
- Downloads problems from HuggingFace datasets
- Filters by date (default: 2024+)
- Extracts test cases (inputs/outputs)
- Creates formatted prompts
- Samples subset if desired

**Usage**:
```bash
# Fetch all problems
python fetch_livecodebench.py --output prompts.json

# Fetch 50 problems
python fetch_livecodebench.py --n-samples 50 --output prompts_50.json

# Customize prompt style
python fetch_livecodebench.py --prompt-style leetcode --output prompts.json

# Filter by date
python fetch_livecodebench.py --min-date 2024-06-01 --output prompts_recent.json
```

**Output Format**:
```json
{
  "metadata": {
    "dataset": "LiveCodeBench (Code Generation Lite)",
    "fetch_date": "2025-12-13T...",
    "n_problems": 100,
    "min_date": "2024-01-01"
  },
  "statistics": {
    "total_problems": 100,
    "problems_with_tests": 100,
    "avg_test_cases": 3.5,
    "difficulties": {"Easy": 20, "Medium": 50, "Hard": 30}
  },
  "problems": [
    {
      "problem_id": "lcb_001",
      "title": "Two Sum",
      "difficulty": "Easy",
      "description": "Given an array...",
      "prompt": "Solve the following coding problem:\n...",
      "test_cases": [
        {"input": "[2,7,11,15], 9", "output": "[0,1]", "explanation": "..."}
      ],
      "metadata": {
        "platform": "LeetCode",
        "contest_date": "2024-01-15",
        "topics": ["array", "hash-table"],
        "url": "https://..."
      }
    }
  ]
}
```

**Requirements**:
```bash
pip install datasets
```

### 2. evaluate_code.py

Evaluates model-generated code by executing it against test cases.

**Features**:
- Executes Python code safely with timeout
- Runs all test cases per problem
- Computes Pass@1 metric
- **FREE**: Local CPU execution, no API costs
- Safety: Subprocess isolation, optional Docker

**Usage**:
```bash
# Basic evaluation
python evaluate_code.py \
    --problems prompts.json \
    --responses model_responses.json \
    --output results.json

# With custom timeout
python evaluate_code.py \
    --problems prompts.json \
    --responses responses.json \
    --timeout 10

# Evaluate subset
python evaluate_code.py \
    --problems prompts.json \
    --responses responses.json \
    --max-problems 50

# Use Docker for better isolation (safer)
python evaluate_code.py \
    --problems prompts.json \
    --responses responses.json \
    --use-docker
```

**Input Format** (`model_responses.json`):
```json
{
  "responses": {
    "lcb_001": "def twoSum(nums, target):\n    ...",
    "lcb_002": "def solution(arr):\n    ..."
  }
}
```

**Output Format** (`results.json`):
```json
{
  "metadata": {
    "evaluation_date": "2025-12-13T...",
    "problems_file": "prompts.json",
    "responses_file": "responses.json",
    "timeout": 5
  },
  "metrics": {
    "pass_at_1": 0.65,
    "problems_evaluated": 100,
    "problems_passed": 65,
    "total_tests_passed": 195,
    "total_tests": 300,
    "overall_test_pass_rate": 0.65
  },
  "results": [
    {
      "problem_id": "lcb_001",
      "passed": true,
      "reason": "All tests passed",
      "tests_passed": 3,
      "tests_total": 3,
      "pass_rate": 1.0,
      "test_results": [...]
    }
  ]
}
```

**Execution Safety**:
- Subprocess with timeout (default 5s)
- Memory limits configurable
- Optional Docker isolation
- Input/output validation

## Evaluation Workflow

### Step 1: Fetch Problems

```bash
python fetch_livecodebench.py --n-samples 100 --output problems.json
```

### Step 2: Generate Code (Using your LLM)

Create responses JSON with model outputs:

```python
import json
import openai

# Load problems
with open('problems.json') as f:
    data = json.load(f)

# Generate responses
responses = {}
for problem in data['problems']:
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": problem['prompt']}]
    )
    responses[problem['problem_id']] = response.choices[0].message.content

# Save
with open('responses.json', 'w') as f:
    json.dump({"responses": responses}, f, indent=2)
```

### Step 3: Evaluate

```bash
python evaluate_code.py \
    --problems problems.json \
    --responses responses.json \
    --output evaluation.json
```

### Step 4: Analyze Results

```python
import json

with open('evaluation.json') as f:
    results = json.load(f)

print(f"Pass@1: {results['metrics']['pass_at_1']*100:.1f}%")
print(f"Problems Passed: {results['metrics']['problems_passed']}/{results['metrics']['problems_evaluated']}")
```

## Pre-computed Scores

### Coverage Statistics

From Artificial Analysis API:
- **Total Models**: 83
- **Models with LiveCodeBench**: 82 (98.8%)
- **Missing**: 1 model (GPT-4 Turbo)

### Score Distribution

| Percentile | LiveCodeBench Score |
|------------|---------------------|
| Max        | 0.734 (73.4%)      |
| 95th       | 0.673              |
| 75th       | 0.486              |
| Median     | 0.314              |
| 25th       | 0.157              |
| Min        | 0.0 (0%)           |

### Top Performers

Based on pre-computed scores:

1. **o3-mini (high)** - 0.734 (73.4%)
2. **Claude 3.7 Sonnet (Reasoning)** - 0.673 (67.3%)
3. **DeepSeek V3.1 (Reasoning)** - 0.609 (60.9%)
4. **DeepSeek V3.1 Terminus** - 0.551 (55.1%)
5. **Claude Opus 4.5** - 0.510 (51.0%)

## Why LiveCodeBench?

### Advantages

1. **Contamination-Free**
   - Problems from 2024+ (after most model training cutoffs)
   - Minimizes memorization, tests true reasoning

2. **Execution-Based Evaluation**
   - Objective: Code either works or doesn't
   - No need for LLM judges or subjective scoring

3. **Free to Run**
   - Local CPU execution
   - No API costs for evaluation
   - Reproducible

4. **Complete Test Cases**
   - Multiple test cases per problem
   - Inputs and expected outputs provided
   - Tests edge cases

5. **Real-World Problems**
   - From actual competitive programming contests
   - Diverse difficulty levels
   - Multiple algorithmic topics

### Limitations

1. **Python Only**
   - Current implementation focuses on Python
   - Can be extended to other languages

2. **Execution Safety**
   - Running untrusted code requires isolation
   - Docker recommended for production use

3. **Test Case Coverage**
   - Public test cases may not cover all edge cases
   - Hidden test cases not available

## Usage in the Project

### 1. Composite Coding Score (CCS)

LiveCodeBench is one of 3 benchmarks used for CCS:
- **LiveCodeBench** (~60% coverage) - Competitive programming
- **HumanEval** (~70% coverage) - Function-level generation
- **SciCode** (~50% coverage) - Scientific computing

Weight: Determined by Bayesian latent factor model (λ ≈ 0.8)

### 2. Model Comparison

LiveCodeBench scores used for:
- Ranking models by code generation ability
- Validating Composite Coding Score
- Analyzing reasoning model improvements

## Implementation Details

### Code Execution

The evaluator:
1. Wraps generated code with test harness
2. Creates temporary Python file
3. Executes in subprocess with timeout
4. Captures stdout/stderr
5. Compares output to expected
6. Handles numerical/JSON comparisons

### Output Matching

Multiple strategies:
- Exact string match (after whitespace normalization)
- Numerical comparison (with tolerance)
- JSON parsing and comparison
- List/array comparison

### Error Handling

- Syntax errors → Test fails
- Runtime errors → Test fails
- Timeout → Test fails
- Import errors → Test fails
- Output mismatch → Test fails

## Extending the Benchmark

### Adding More Problems

```python
# Custom problems format
custom_problems = {
    "problems": [
        {
            "problem_id": "custom_001",
            "title": "My Problem",
            "description": "...",
            "prompt": "...",
            "test_cases": [
                {"input": "5", "output": "25"}
            ]
        }
    ]
}
```

### Multi-Language Support

To add support for other languages:
1. Update `CodeExecutor` class
2. Add language-specific execution logic
3. Handle language-specific test harness
4. Update output matching logic

### Docker Integration

For production use:

```bash
# Build Docker image
docker build -t code-eval .

# Run evaluation in Docker
python evaluate_code.py --use-docker ...
```

## References

- **LiveCodeBench Paper**: "LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code"
- **HuggingFace Dataset**: https://huggingface.co/datasets/livecodebench/code_generation_lite
- **Official Repo**: https://github.com/LiveCodeBench/LiveCodeBench
- **Artificial Analysis**: https://artificialanalysis.ai (pre-computed scores)

## Related Files

- **Main Cache**: `data/models_cache.json` (full model data including LiveCodeBench)
- **CCS Documentation**: `quality_scoring/docs/COMPOSITE_CODING_SCORE.md`
- **Coding Score Script**: `scripts/quality_scoring/compute_coding_score.py`

## Requirements

### Python Packages

```bash
pip install datasets        # For fetching problems
pip install requests       # For API calls (optional)
```

### System Requirements

- Python 3.8+
- 1GB+ RAM for evaluation
- Docker (optional, for safer isolation)

## Troubleshooting

### Dataset Loading Issues

If HuggingFace dataset fails to load:
1. Check dataset name: `livecodebench/code_generation_lite`
2. Verify internet connection
3. Try with `trust_remote_code=True`

### Execution Timeouts

If code times out frequently:
```bash
python evaluate_code.py --timeout 10  # Increase to 10 seconds
```

### Memory Issues

For memory-intensive problems:
```bash
# Limit memory per execution
python evaluate_code.py --max-memory-mb 1024
```

### Docker Issues

If Docker fails:
```bash
# Fallback to subprocess
python evaluate_code.py  # No --use-docker flag
```

## Data Authenticity

✅ **Pre-computed scores are authentic from Artificial Analysis API**  
✅ **Zero imputation or estimation performed**  
✅ **Evaluation scripts are open-source and auditable**  
✅ **Execution-based evaluation is deterministic and reproducible**

Models without LiveCodeBench scores have `null` values. Only 1 model (GPT-4 Turbo) lacks a score.
