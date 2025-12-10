# Intent Classifier - Quick Start Guide

## Overview

A fast, accurate classifier for categorizing user prompts into 5 core intent categories:

- **REASONING**: Math, logic, analytical problem-solving
- **CODING**: Programming, debugging, software development
- **FACTUAL_QA**: Knowledge retrieval, questions, explanations
- **AGENTIC_EXECUTION**: Multi-step tasks, workflows, automation
- **GENERAL**: Conversation, opinions, unclear intent

## Quick Start

### Installation

The classifier is part of the `llm_jury` package. No additional dependencies required for classification (only `scikit-learn` for evaluation).

### Basic Usage

```python
from llm_jury.routing import IntentClassifier

# Initialize
classifier = IntentClassifier()

# Classify a prompt
result = classifier.classify("Write a Python function to sort a list")

print(result.category.value)   # "coding"
print(result.confidence)        # 0.95
```

### Batch Processing

```python
prompts = [
    "Solve for x: 2x + 5 = 13",
    "What is the capital of France?",
    "Plan a 7-day trip to Japan",
]

results = classifier.classify_batch(prompts)

for prompt, result in zip(prompts, results):
    print(f"{prompt} → {result.category.value}")
```

## Performance

### Test Set Results (50 samples)

| Metric | Score |
|--------|-------|
| **Overall Accuracy** | 76.00% |
| **Macro F1** | 75.75% |
| **Inference Speed** | 1-2ms |

### Per-Category Performance

| Category | Precision | Recall | F1 |
|----------|-----------|--------|-----|
| reasoning | 100.0% | 70.0% | 82.4% |
| coding | 69.2% | 90.0% | 78.3% |
| factual_qa | 66.7% | 100.0% | 80.0% |
| agentic_execution | 100.0% | 60.0% | 75.0% |
| general | 66.7% | 60.0% | 63.2% |

## Evaluation

### Run Full Evaluation

```bash
# Evaluate on train/val/test splits
python scripts/evaluate_intent_classifier.py

# Generate visualizations
python scripts/visualize_intent_results.py

# Quick functionality test
python scripts/test_intent_classifier_quick.py
```

### Outputs

- **Results**: `results/intent_classifier_evaluation.json`
- **Plots**: `results/intent_classifier_plots/`
  - Confusion matrices
  - Per-category metrics
  - Cross-split comparison
  - Error distribution

## Files

```
llm_jury/
├── llm_jury/routing/
│   └── intent_classifier.py                    # Classifier implementation
├── data/
│   └── intent_classification_dataset.json       # 200 labeled samples
├── scripts/
│   ├── evaluate_intent_classifier.py            # Full evaluation
│   ├── visualize_intent_results.py              # Generate plots
│   └── test_intent_classifier_quick.py          # Quick test
├── docs/
│   └── INTENT_CLASSIFIER.md                     # Detailed documentation
└── INTENT_CLASSIFIER_SUMMARY.md                 # Executive summary
```

## Key Features

✅ **Fast**: 1-2ms classification (no model loading)  
✅ **Accurate**: 76-83% accuracy across splits  
✅ **Transparent**: Clear signals for each classification  
✅ **Simple**: Pure Python with regex patterns  
✅ **Tested**: 200-sample labeled dataset with train/val/test splits

## Examples

### High Confidence Cases

```python
# Reasoning
classifier.classify("Solve for x: 2x + 5 = 13")
# → reasoning (0.95)

# Coding
classifier.classify("Write a Python function to reverse a string")
# → coding (1.00)

# Factual QA
classifier.classify("What is the capital of France?")
# → factual_qa (0.90)

# Agentic Execution
classifier.classify("Plan a 7-day trip including flights and hotels")
# → agentic_execution (0.90)

# General
classifier.classify("Hello, how are you today?")
# → general (0.80)
```

### Edge Cases

Some prompts are harder to classify:

```python
# Reasoning vs Coding overlap
"Analyze the Big O notation of this function"
# → coding (often, due to "function" keyword)

# General vs Factual QA confusion
"What do you think about AI?"
# → factual_qa (question pattern triggers, should be general)

# Agentic Execution low recall
"Use the calendar API to schedule a meeting"
# → coding (API keyword triggers strongly)
```

## Improvements Roadmap

### High Priority
1. Better patterns for agentic execution (50% → 80% recall target)
2. Disambiguate opinion questions from factual QA
3. Add context-aware pattern matching

### Future Enhancements
4. Hybrid approach (regex + lightweight ML)
5. Active learning from misclassifications
6. Multi-label classification for overlapping intents

## Documentation

- **Quick Start**: This file
- **Full Documentation**: `docs/INTENT_CLASSIFIER.md`
- **Evaluation Summary**: `INTENT_CLASSIFIER_SUMMARY.md`
- **API Reference**: See docstrings in `llm_jury/routing/intent_classifier.py`

## Support

For issues or questions:
1. Check the documentation in `docs/INTENT_CLASSIFIER.md`
2. Review evaluation results in `results/`
3. Run quick test: `python scripts/test_intent_classifier_quick.py`

---

**Version**: 1.0.0  
**Last Updated**: 2025-12-06  
**Dataset Size**: 200 samples (75 train, 50 val, 50 test)

