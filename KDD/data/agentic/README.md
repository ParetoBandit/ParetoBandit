# GAIA - General AI Assistants Benchmark

This directory contains the GAIA benchmark for evaluating general agentic capabilities, including tool use, reasoning, and multi-step problem solving with real-world questions.

## Overview

**GAIA** (General AI Assistants) is a benchmark designed to test AI assistants on realistic, complex tasks that require:
- **Tool use** (file analysis, web search, calculations)
- **Multi-step reasoning**
- **Real-world knowledge**
- **Information synthesis**

**Key Features**:
- ✅ **Real-World Questions**: "What city is in this file?", "Who is older, X or Y?"
- ✅ **Short Answers**: City names, numbers, dates (easy to verify)
- ✅ **Exact Match Scoring**: Objective, deterministic evaluation
- ✅ **Multiple Difficulty Levels**: Level 1 (Easy), Level 2 (Medium), Level 3 (Hard)
- ✅ **File-Based Tasks**: Many questions reference files (images, PDFs, spreadsheets)

## Directory Structure

```
agentic/
├── README.md                # This documentation
├── fetch_gaia.py            # Fetch GAIA problems from HuggingFace
├── evaluate_gaia.py         # Evaluate responses with exact match
└── example_usage.py         # End-to-end workflow example
```

## Benchmark Details

### Dataset Information

- **Name**: GAIA (General AI Assistants)
- **Source**: HuggingFace `gaia-benchmark/GAIA`
- **Paper**: "GAIA: a benchmark for General AI Assistants"
- **Splits**: 
  - **Validation**: ~165 problems (recommended for development)
  - **Test**: ~300 problems (for final evaluation, answers not public)
- **Difficulty Levels**:
  - Level 1: Easy (straightforward questions)
  - Level 2: Medium (requires 2-3 steps)
  - Level 3: Hard (complex multi-step reasoning)

### Question Types

1. **File Analysis**
   - "What city is mentioned in this PDF?"
   - "What's the total in column B of this spreadsheet?"
   
2. **Web Search + Reasoning**
   - "Who is older, X or Y?" (requires looking up birthdates)
   - "What year did X happen?"

3. **Multi-Hop Reasoning**
   - "Find the author of book X, then find their birthplace"
   - "Calculate the time difference between events A and B"

4. **Calculations**
   - "What's 15% of the value in cell A5?"
   - "How many days between these two dates?"

### Answer Format

Answers are **short and specific**:
- **City names**: "Seattle", "Paris", "Tokyo"
- **Numbers**: "42", "1234.56"
- **Dates**: "2024-01-15", "March 15, 2024"
- **Names**: "Albert Einstein"
- **Yes/No**: "Yes", "No"

This makes evaluation straightforward with **exact string matching**.

## Evaluation Metric

**Exact Match Accuracy**:
- Answer must exactly match the ground truth (case-insensitive)
- Numeric answers allow small tolerance (< 1e-6)
- Substring matching for cases like "Seattle, WA" vs "Seattle"

**Formula**:
```
Accuracy = (# Correct Answers) / (# Total Problems)
```

## Scripts

### 1. fetch_gaia.py

Fetches GAIA problems from HuggingFace with ground truth answers.

**Features**:
- Downloads validation or test split
- Filters by difficulty level
- Extracts questions and final answers
- Creates formatted prompts
- Handles file references

**Usage**:
```bash
# Fetch validation split (all levels)
python fetch_gaia.py --split validation --output gaia_val.json

# Fetch only Level 1 (easy) problems
python fetch_gaia.py --split validation --level 1 --output gaia_easy.json

# Fetch 50 random problems
python fetch_gaia.py --n-samples 50 --output gaia_50.json

# Use detailed prompts
python fetch_gaia.py --prompt-style detailed --output gaia_detailed.json
```

**Authentication**:
GAIA is a gated dataset. You need to:
1. Visit https://huggingface.co/datasets/gaia-benchmark/GAIA
2. Accept the terms
3. Set your token: `export HF_TOKEN=your_token`

**Output Format**:
```json
{
  "metadata": {
    "dataset": "GAIA (General AI Assistants)",
    "split": "validation",
    "fetch_date": "2025-12-13T...",
    "n_problems": 165
  },
  "statistics": {
    "total_problems": 165,
    "with_files": 89,
    "avg_steps": 2.3,
    "levels": {
      "1": 65,
      "2": 70,
      "3": 30
    }
  },
  "problems": [
    {
      "task_id": "gaia_val_001",
      "question": "What city is mentioned in the attached file?",
      "level": 1,
      "final_answer": "Seattle",
      "file_name": "document.pdf",
      "prompt": "What city is mentioned in the attached file?...",
      "metadata": {
        "steps": "Read file, identify city",
        "number_of_steps": 2,
        "tools": ["file_reader"]
      }
    }
  ]
}
```

**Requirements**:
```bash
pip install datasets huggingface_hub
```

### 2. evaluate_gaia.py

Evaluates model responses using exact match scoring.

**Features**:
- Extracts final answer from model response
- Normalizes answers (case, whitespace, punctuation)
- Exact match comparison
- Numeric matching with tolerance
- Per-level accuracy breakdown

**Usage**:
```bash
# Basic evaluation
python evaluate_gaia.py \
    --problems gaia_val.json \
    --responses model_responses.json \
    --output evaluation.json

# Case-sensitive matching
python evaluate_gaia.py \
    --problems gaia_val.json \
    --responses responses.json \
    --case-sensitive

# Strict numeric comparison
python evaluate_gaia.py \
    --problems gaia_val.json \
    --responses responses.json \
    --strict-numeric
```

**Input Format** (`model_responses.json`):
```json
{
  "responses": {
    "gaia_val_001": "The city mentioned in the file is Seattle.",
    "gaia_val_002": "42"
  }
}
```

**Output Format** (`evaluation.json`):
```json
{
  "metadata": {
    "evaluation_date": "2025-12-13T...",
    "problems_file": "gaia_val.json",
    "responses_file": "responses.json"
  },
  "metrics": {
    "overall": {
      "accuracy": 0.65,
      "correct": 107,
      "total": 165
    },
    "by_level": {
      "1": {"accuracy": 0.85, "correct": 55, "total": 65},
      "2": {"accuracy": 0.60, "correct": 42, "total": 70},
      "3": {"accuracy": 0.33, "correct": 10, "total": 30}
    }
  },
  "results": [
    {
      "task_id": "gaia_val_001",
      "level": 1,
      "correct": true,
      "extracted_answer": "seattle",
      "expected_answer": "Seattle",
      "reason": "Exact match",
      "full_response": "The city mentioned..."
    }
  ]
}
```

## Evaluation Workflow

### Step 1: Fetch GAIA Problems

```bash
python fetch_gaia.py --split validation --output problems.json
```

### Step 2: Generate Responses (Using your LLM/Agent)

Create a responses JSON file:

```python
import json

# Load problems
with open('problems.json') as f:
    data = json.load(f)

# Generate responses with your agent
responses = {}
for problem in data['problems']:
    # Your agent logic here
    response = your_agent(problem['question'], problem.get('file_name'))
    responses[problem['task_id']] = response

# Save
with open('responses.json', 'w') as f:
    json.dump({"responses": responses}, f, indent=2)
```

### Step 3: Evaluate

```bash
python evaluate_gaia.py \
    --problems problems.json \
    --responses responses.json \
    --output evaluation.json
```

### Step 4: Analyze Results

```python
import json

with open('evaluation.json') as f:
    results = json.load(f)

print(f"Overall Accuracy: {results['metrics']['overall']['accuracy']*100:.1f}%")

# Per-level breakdown
for level, stats in results['metrics']['by_level'].items():
    print(f"Level {level}: {stats['accuracy']*100:.1f}%")
```

## Answer Extraction

The evaluator uses multiple strategies to extract the final answer:

1. **Explicit Markers**:
   - "Answer: Seattle"
   - "Final answer: 42"
   - "The answer is: Paris"

2. **Last Sentence**:
   - If no explicit marker, uses last sentence

3. **Last Line**:
   - Falls back to last line if no sentence detected

4. **Normalization**:
   - Removes quotes, punctuation
   - Normalizes whitespace
   - Case-insensitive (by default)

## File-Based Questions

Many GAIA questions reference files (images, PDFs, spreadsheets, etc.):

```json
{
  "question": "What is the total in column B of the attached spreadsheet?",
  "file_name": "data.xlsx",
  "final_answer": "1234.56"
}
```

**Handling Files**:
1. Files are part of the GAIA dataset
2. Your agent needs file-reading capabilities
3. File paths are provided in the problem metadata
4. The evaluation script doesn't need the files (only checks answers)

## Difficulty Levels

### Level 1 (Easy) - ~65 problems
- Single-step questions
- Simple lookups
- Straightforward reasoning
- Example: "What year was X born?"

### Level 2 (Medium) - ~70 problems
- 2-3 step reasoning
- Requires combining information
- May need calculations
- Example: "Who is older, X or Y?" (lookup 2 birthdates, compare)

### Level 3 (Hard) - ~30 problems
- Complex multi-step reasoning
- Multiple tool uses
- Information synthesis
- Example: "Find the CEO of company X, then find their alma mater's founding year"

## Expected Performance

Based on GAIA paper benchmarks:

| Model Type | Level 1 | Level 2 | Level 3 | Overall |
|------------|---------|---------|---------|---------|
| GPT-4 (no tools) | ~60% | ~30% | ~10% | ~35% |
| GPT-4 + Tools | ~85% | ~60% | ~30% | ~60% |
| Humans | ~95% | ~92% | ~86% | ~92% |

**Key Insight**: This benchmark is hard! Even GPT-4 with tools only achieves ~60% overall accuracy, while humans achieve ~92%.

## Why GAIA?

### Advantages

1. **Real-World Tasks**
   - Questions feel natural and practical
   - Tests actual assistant capabilities

2. **Objective Evaluation**
   - Short, specific answers
   - Exact match scoring
   - No subjective judgment needed

3. **Tests Multiple Skills**
   - Tool use
   - Multi-step reasoning
   - Information synthesis
   - File handling

4. **Multiple Difficulty Levels**
   - Can test across capability spectrum
   - Level 3 is very challenging even for SOTA models

5. **File-Based Tasks**
   - Tests multimodal understanding
   - Realistic assistant scenarios

### Limitations

1. **Requires Tool Access**
   - Models need web search, file reading, etc.
   - Not just language modeling

2. **Files May Be Large**
   - Some questions reference PDFs, images
   - May need file handling infrastructure

3. **Exact Match Can Be Strict**
   - "New York" vs "New York City" might fail
   - Substring matching helps but isn't perfect

4. **Limited Size**
   - Only ~165 validation problems
   - May not be enough for full statistical analysis

## Integration with Project

### Agentic Quality Score

GAIA can be used to compute an "Agentic Quality Score" measuring:
- Tool use capability
- Multi-step reasoning
- Real-world task completion

### Comparison with Other Benchmarks

| Benchmark | Focus | Answer Format | Evaluation |
|-----------|-------|---------------|------------|
| **GAIA** | Agentic tasks | Short strings | Exact match |
| **GPQA** | Science reasoning | Multiple choice | Selection |
| **LiveCodeBench** | Code generation | Code | Execution |
| **SummEdits** | Summarization | Binary | Yes/No classification |

GAIA is unique in testing **general assistant capabilities** rather than a specific skill.

## References

- **GAIA Paper**: "GAIA: a benchmark for General AI Assistants"
- **HuggingFace Dataset**: https://huggingface.co/datasets/gaia-benchmark/GAIA
- **Official Site**: https://huggingface.co/gaia-benchmark
- **Leaderboard**: https://huggingface.co/spaces/gaia-benchmark/leaderboard

## Related Files

- **Main Cache**: `data/models_cache.json` (for other benchmarks)
- **Tool Use Examples**: Can integrate with agent frameworks
- **Multi-Modal**: Consider combining with vision models for image tasks

## Requirements

### Python Packages

```bash
pip install datasets        # For fetching GAIA
pip install huggingface_hub # For authentication
```

### Authentication

GAIA is gated. Set up access:

```bash
# Option 1: Environment variable
export HF_TOKEN=your_huggingface_token

# Option 2: CLI login
huggingface-cli login
```

### For Agents

To actually run GAIA tasks, your agent needs:
- Web search capability
- File reading (PDF, images, spreadsheets, etc.)
- Calculator/computation tools
- Multi-step reasoning
- Tool orchestration

## Troubleshooting

### Dataset Access Issues

If you get authentication errors:
```bash
# Make sure you've accepted the dataset terms
# Visit: https://huggingface.co/datasets/gaia-benchmark/GAIA

# Set your token
export HF_TOKEN=your_token

# Or use CLI
huggingface-cli login
```

### File Handling

For file-based questions:
- Files are in the dataset but may need special handling
- Consider extracting files to a separate directory
- Your agent needs file-reading tools

### Answer Extraction Fails

If the evaluator doesn't extract answers correctly:
- Check that responses have clear "Answer: X" format
- Adjust the extraction patterns in `evaluate_gaia.py`
- Use the `--case-sensitive` flag if needed

## Example Agent Implementation

```python
# Pseudo-code for a GAIA agent
class GAIAAgent:
    def __init__(self):
        self.tools = {
            'search': WebSearchTool(),
            'file_reader': FileReaderTool(),
            'calculator': CalculatorTool()
        }
    
    def solve(self, question, file_name=None):
        # 1. Analyze question to determine required tools
        plan = self.create_plan(question)
        
        # 2. Execute plan step by step
        for step in plan:
            result = self.execute_step(step, file_name)
        
        # 3. Synthesize final answer
        answer = self.synthesize_answer(results)
        
        return answer
```

## Data Authenticity

✅ **GAIA data comes directly from HuggingFace**  
✅ **Ground truth answers from official dataset**  
✅ **Evaluation is deterministic (exact match)**  
✅ **No imputation or estimation**

The validation split is recommended for development and testing, as it has public answers for verification.
