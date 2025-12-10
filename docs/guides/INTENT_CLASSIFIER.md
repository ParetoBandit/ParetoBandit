# Intent Classifier Documentation

## Overview

The Intent Classifier is a fast, regex-based classifier that categorizes user prompts into 5 core intent categories:

1. **REASONING** - Mathematical, logical, and analytical problem-solving
2. **CODING** - Programming, code generation, debugging, and software development
3. **FACTUAL_QA** - Knowledge retrieval, question answering, explanations, and learning
4. **AGENTIC_EXECUTION** - Multi-step tasks, workflows, tool usage, and autonomous operations
5. **GENERAL** - General conversation, unclear intent, and other interactions

## Architecture

### Pattern-Based Classification

The classifier uses a two-stage approach:

1. **Pattern Matching**: Regex patterns with associated confidence weights (0.0-1.0)
2. **Keyword Boosting**: Additional signals from keyword presence

This approach provides:
- **Fast inference**: ~1-2ms per classification
- **Transparency**: Clear signals for why a classification was made
- **No model dependencies**: Pure Python with regex
- **Easy to extend**: Add new patterns/keywords as needed

### Design Principles

- **Clear boundaries**: Categories are distinct with minimal overlap
- **Benchmark alignment**: Categories align with established NLP benchmarks
- **Explicit signals**: Code blocks, math symbols, question words, etc.
- **Conservative fallback**: Low confidence defaults to GENERAL

## Performance Metrics

### Overall Results

| Split | Accuracy | Macro F1 | Errors |
|-------|----------|----------|--------|
| **TRAIN** | 82.67% | 82.52% | 13/75 |
| **VAL** | 82.00% | 81.36% | 9/50 |
| **TEST** | 76.00% | 75.75% | 12/50 |

### Per-Category Performance (Test Set)

| Category | Precision | Recall | F1 | Support |
|----------|-----------|--------|-----|---------|
| **reasoning** | 100.0% | 70.0% | 82.4% | 10 |
| **coding** | 69.2% | 90.0% | 78.3% | 10 |
| **factual_qa** | 66.7% | 100.0% | 80.0% | 10 |
| **agentic_execution** | 100.0% | 60.0% | 75.0% | 10 |
| **general** | 66.7% | 60.0% | 63.2% | 10 |

## Key Findings

### Strengths

1. **High Precision Categories**
   - **Reasoning**: 100% precision across val/test - no false positives
   - **Agentic Execution**: 100% precision across val/test - when detected, always correct
   - Strong pattern matching for code blocks, math symbols

2. **High Recall Categories**
   - **Factual QA**: 100% recall across val/test - captures all questions
   - **Coding**: 90% recall on test set - rarely misses code-related prompts

3. **Fast and Efficient**
   - No model loading required
   - ~1-2ms classification latency
   - Runs on any Python environment

### Challenges

1. **Agentic Execution Recall**
   - 50-60% recall - many multi-step tasks not captured
   - Often confused with CODING or GENERAL
   - **Root cause**: Patterns too specific (e.g., "workflow", "pipeline", "agent")
   - **Improvement**: Add more natural language patterns like "first...then", "after that"

2. **General vs Factual QA Confusion**
   - Questions starting with "What" often misclassified
   - **Root cause**: Pattern `^(what|who|when|where|why)` triggers FACTUAL_QA too eagerly
   - **Improvement**: Add context signals (e.g., "what do you think" → GENERAL)

3. **Reasoning vs Coding Overlap**
   - Algorithm complexity analysis confused between categories
   - **Root cause**: Both mention "function", "algorithm"
   - **Current behavior**: Reasonable, as many algorithm problems span both

## Common Misclassification Patterns

### 1. GENERAL → FACTUAL_QA (4 errors in test)

```
"What's the best way to stay motivated?" → Predicted: factual_qa
"What are good habits to develop?" → Predicted: factual_qa
```

**Cause**: Question patterns (`^what`) trigger FACTUAL_QA even for subjective/opinion questions

### 2. AGENTIC_EXECUTION → CODING (2-3 errors per split)

```
"Use the calendar API to schedule a meeting" → Predicted: coding
"Implement a job queue system with retry logic" → Predicted: coding
```

**Cause**: API/implementation keywords trigger CODING before AGENTIC_EXECUTION patterns match

### 3. REASONING → CODING (2 errors in test)

```
"Analyze the Big O notation of this recursive function" → Predicted: coding
"Calculate the gradient of this multivariable function" → Predicted: coding
```

**Cause**: "function" keyword triggers CODING patterns strongly

## Usage

### Basic Usage

```python
from llm_jury.routing.intent_classifier import IntentClassifier

# Initialize classifier
classifier = IntentClassifier()

# Classify a prompt
result = classifier.classify("Write a Python function to sort a list")

print(result.category)      # IntentCategory.CODING
print(result.confidence)    # 0.95
print(result.signals)       # ['pattern:...', 'keywords:2']
```

### Batch Classification

```python
prompts = [
    "Solve for x: 2x + 5 = 13",
    "What is the capital of France?",
    "Plan a 7-day trip to Japan",
]

results = classifier.classify_batch(prompts)

for prompt, result in zip(prompts, results):
    print(f"{prompt[:50]:<50} → {result.category.value}")
```

### Accessing Detailed Scores

```python
result = classifier.classify("Debug this SQL injection vulnerability")

print(f"Category: {result.category.value}")
print(f"Confidence: {result.confidence:.2f}")
print(f"All scores: {result.all_scores}")
print(f"Signals: {result.signals}")
print(f"Latency: {result.latency_ms:.2f}ms")
```

## Evaluation

### Running Evaluation

```bash
# Evaluate on all splits
python scripts/evaluate_intent_classifier.py

# Show more misclassifications
python scripts/evaluate_intent_classifier.py --show-misclass 20

# Evaluate specific splits
python scripts/evaluate_intent_classifier.py --splits train val
```

### Generating Visualizations

```bash
# Create plots
python scripts/visualize_intent_results.py

# Custom output directory
python scripts/visualize_intent_results.py --output-dir my_plots/
```

### Output Files

- `results/intent_classifier_evaluation.json` - Detailed results
- `results/intent_classifier_plots/` - Visualization PNG files
  - `confusion_matrix_{split}.png` - Confusion matrices
  - `per_class_metrics.png` - Precision/Recall/F1 by category
  - `overall_comparison.png` - Cross-split comparison
  - `error_distribution.png` - Error rates by category

## Dataset

The evaluation dataset contains 200 labeled samples:
- **Training**: 75 samples (15 per category)
- **Validation**: 50 samples (10 per category)  
- **Test**: 50 samples (10 per category)

Samples cover diverse scenarios including:
- Math problems, logical reasoning, statistical analysis
- Code generation, debugging, refactoring across multiple languages
- Factual questions, explanations, educational queries
- Multi-step workflows, agent tasks, orchestration
- General chat, opinions, recommendations

Location: `data/intent_classification_dataset.json`

## Improvement Recommendations

### High Priority

1. **Improve Agentic Execution Recall**
   - Add patterns for natural multi-step language: "first...then", "after that"
   - Detect sequential/conditional language: "if...then", "once...proceed"
   - Lower API pattern weight to reduce false positives from CODING

2. **Disambiguate Subjective Questions**
   - Add negative patterns for opinion-seeking: "what do you think", "your opinion"
   - Check for subjective language: "best", "should I", "recommend"
   - Weight context over question words

3. **Handle Coding-Reasoning Overlap**
   - Create composite category or choose primary
   - Current behavior (favor CODING) is acceptable for routing

### Medium Priority

4. **Add Context Window Analysis**
   - Check surrounding words for question patterns
   - Example: "what is X" vs "what should I do"

5. **Confidence Calibration**
   - Review threshold for GENERAL fallback
   - Current: min_confidence = 0.60

6. **Pattern Refinement**
   - Review patterns that fired on misclassifications
   - Add negative patterns to exclude certain matches

## Testing

### Unit Tests

```python
# Add to tests/test_intent_classifier.py
from llm_jury.routing.intent_classifier import IntentClassifier, IntentCategory

def test_coding_classification():
    classifier = IntentClassifier()
    result = classifier.classify("Write a Python function")
    assert result.category == IntentCategory.CODING
    assert result.confidence > 0.8

def test_reasoning_classification():
    classifier = IntentClassifier()
    result = classifier.classify("Solve for x: 2x + 5 = 13")
    assert result.category == IntentCategory.REASONING
    assert result.confidence > 0.8
```

### Integration Testing

The classifier integrates with:
- Model selection/routing systems
- Query preprocessing pipelines
- Analytics and monitoring

## References

- Implementation: `llm_jury/routing/intent_classifier.py`
- Evaluation: `scripts/evaluate_intent_classifier.py`
- Visualization: `scripts/visualize_intent_results.py`
- Dataset: `data/intent_classification_dataset.json`
- Results: `results/intent_classifier_evaluation.json`

## Changelog

### v1.0.0 (2025-12-06)
- Initial implementation with 5 core categories
- Pattern-based classification with keyword boosting
- Comprehensive evaluation on 200-sample dataset
- Achieved 76-83% accuracy across splits
- Generated confusion matrices and performance visualizations

