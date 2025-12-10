

# Real Intent Classification Data Collection Guide

## Overview

This guide explains how to collect **real-world labeled data** for intent classification using the methodology from RouteLLM and KDD papers. This approach is superior to synthetic data as it uses authentic user prompts from established benchmarks.

## Methodology

### 1. Source Selection

We collect prompts from 5 established datasets, each representing one intent category:

| Intent Category | Dataset | HuggingFace ID | Samples |
|-----------------|---------|----------------|---------|
| **REASONING** | GSM8k (Grade School Math) | `openai/gsm8k` | 2,000 |
| **CODING** | MBPP (Python Problems) | `google-research-datasets/mbpp` | 2,000 |
| **FACTUAL_QA** | Natural Questions | `google-research-datasets/natural_questions` | 2,000 |
| **AGENTIC_EXECUTION** | Glaive Function Calling v2 | `glaiveai/glaive-function-calling-v2` | 2,000 |
| **GENERAL** | LMSYS-Chat-1M (filtered) | `lmsys/lmsys-chat-1m` | 2,000 |

**Total**: 10,000 prompts

### 2. Filtering Strategy

Each dataset requires specific filtering to ensure quality:

#### REASONING (GSM8k)
- ✅ Use training split directly
- ✅ These are clean, self-contained math problems
- No filtering needed

#### CODING (MBPP)
- ✅ Extract the `text` field (natural language task description)
- ✅ Examples: "Write a python function to..."
- No filtering needed

#### FACTUAL_QA (Natural Questions)
- ⚠️ Filter for questions starting with question words
- ✅ Keep: "What", "Who", "When", "Where", "Why", "How"
- ❌ Remove: Non-question text

#### AGENTIC_EXECUTION (Glaive)
- ⚠️ Extract user messages from conversations
- ✅ Take messages that precede function calls
- ❌ Skip system messages

#### GENERAL (LMSYS-Chat)
- ⚠️ **Heavy filtering required** (this dataset has everything)
- ❌ Remove code blocks (```)
- ❌ Remove math symbols ($, LaTeX)
- ❌ Remove messages > 50 words
- ❌ Remove coding keywords (python, function, implement, etc.)
- ✅ Result: Pure chitchat/general queries

### 3. Teacher Labeling (Oracle)

Instead of trusting the source dataset as the label (noisy), we use a strong model as an oracle:

**Teacher Models**:
- GPT-4o (OpenAI)
- Claude 3.5 Sonnet (Anthropic)

**Process**:
1. Send each prompt to the teacher model
2. Ask it to classify into one of 5 categories
3. Get confidence score and reasoning
4. Use these labels as ground truth

**Why This Works**:
- Strong models are highly accurate on classification
- Removes noise from dataset source assumptions
- Provides confidence scores for quality filtering
- Cited methodology in RouteLLM paper

### 4. Train/Val/Test Splitting

**Stratified splitting** to ensure balanced class distribution:
- Train: 70% (7,000 samples)
- Validation: 15% (1,500 samples)
- Test: 15% (1,500 samples)

Each category has proportional representation in all splits.

## Pipeline Steps

### Step 1: Collect Raw Prompts

```bash
python scripts/collect_real_intent_data.py \
    --samples-per-class 2000 \
    --output data/real_intent_prompts_raw.json
```

**Output**: `data/real_intent_prompts_raw.json`
- 10,000 prompts
- Source metadata
- Category hints (before teacher labeling)

**Requirements**:
```bash
pip install datasets  # HuggingFace datasets library
```

### Step 2: Teacher Labeling

```bash
# Using OpenAI GPT-4
export OPENAI_API_KEY=your-key-here
python scripts/teacher_label_intents.py \
    --input data/real_intent_prompts_raw.json \
    --output data/real_intent_labeled.json \
    --provider openai \
    --model gpt-4o

# OR using Anthropic Claude
export ANTHROPIC_API_KEY=your-key-here
python scripts/teacher_label_intents.py \
    --input data/real_intent_prompts_raw.json \
    --output data/real_intent_labeled.json \
    --provider anthropic \
    --model claude-3-5-sonnet-20241022
```

**Output**: `data/real_intent_labeled.json`
- 10,000 labeled prompts
- Teacher confidence scores
- Teacher reasoning

**Cost Estimate** (10,000 prompts):
- GPT-4o: ~$5-10
- Claude 3.5 Sonnet: ~$15-20

**Requirements**:
```bash
pip install openai      # For OpenAI
pip install anthropic   # For Anthropic
```

### Step 3: Split Data

```bash
python scripts/split_labeled_data.py \
    --input data/real_intent_labeled.json \
    --output data/real_intent_labeled_split.json \
    --train-ratio 0.7 \
    --val-ratio 0.15 \
    --test-ratio 0.15
```

**Output**: `data/real_intent_labeled_split.json`
- Train/val/test splits
- Stratified by category
- Metadata with counts

### Step 4: Train XGBoost

```bash
python scripts/train_xgboost_intent.py \
    --dataset data/real_intent_labeled_split.json \
    --model-path models/xgboost_intent_classifier.json \
    --n-estimators 200 \
    --max-depth 6
```

**Output**: 
- `models/xgboost_intent_classifier.json` - Trained model
- `models/xgboost_intent_classifier.meta.json` - Feature metadata
- Console output with accuracy metrics

**Requirements**:
```bash
pip install xgboost scikit-learn
```

### Step 5: Compare with Regex Baseline

```bash
python scripts/compare_classifiers.py \
    --dataset data/real_intent_labeled_split.json \
    --xgboost-model models/xgboost_intent_classifier.json
```

This compares:
- Regex-based classifier (baseline)
- XGBoost classifier (ML approach)

## Expected Results

### Synthetic Data (Current - 200 samples)

| Classifier | Test Accuracy |
|------------|---------------|
| Regex | 76.00% |

### Real Data (Expected - 10,000 samples)

| Classifier | Test Accuracy | Notes |
|------------|---------------|-------|
| Regex | 70-75% | Lower due to real-world complexity |
| XGBoost | **80-85%** | ML learns from data patterns |
| BERT/DeBERTa | 85-90% | Deep learning (future) |

## Dataset Quality Indicators

### Good Quality Signs
- ✅ Teacher confidence > 0.9 for >80% of samples
- ✅ Balanced class distribution (±10%)
- ✅ Low disagreement between source hint and teacher label
- ✅ Diverse prompt lengths and styles

### Quality Issues to Watch
- ⚠️ Teacher confidence < 0.7 → Review manually
- ⚠️ One class has <50% of target samples → Collection failed
- ⚠️ High error rate during labeling → API issues or prompt problems

## Cost Breakdown

| Step | Time | Cost | Requirements |
|------|------|------|--------------|
| **Collect Raw Prompts** | 30-60 min | Free | HuggingFace account |
| **Teacher Labeling** | 2-3 hours | $10-20 | API keys |
| **Split Data** | <1 min | Free | - |
| **Train XGBoost** | 5-10 min | Free | - |
| **Evaluate** | 2-5 min | Free | - |

**Total**: ~3-4 hours, $10-20

## Advantages Over Synthetic Data

| Aspect | Synthetic (Current) | Real (This Approach) |
|--------|---------------------|----------------------|
| **Authenticity** | AI-generated examples | Real user prompts |
| **Diversity** | Limited patterns | Natural language variety |
| **Edge Cases** | Missing | Included from real use |
| **Ambiguity** | Clear-cut | Realistic ambiguity |
| **Generalization** | May overfit | Better real-world performance |
| **Size** | 200 samples | 10,000 samples |

## Troubleshooting

### Dataset Download Issues

```python
# If streaming fails, try non-streaming
dataset = load_dataset("lmsys/lmsys-chat-1m", split="train[:10000]")
```

### API Rate Limits

```python
# Add delays in teacher_label_intents.py
time.sleep(1.0)  # Increase from 0.1 to 1.0
```

### Memory Issues

```python
# Process in smaller batches
python scripts/collect_real_intent_data.py --samples-per-class 500
```

### Cost Concerns

```python
# Test with smaller sample first
python scripts/teacher_label_intents.py --limit 100
```

## References

1. **RouteLLM**: [arXiv:2406.18665](https://arxiv.org/abs/2406.18665) - Model routing with learned cost-quality tradeoffs
2. **GSM8k**: [arXiv:2110.14168](https://arxiv.org/abs/2110.14168) - Grade school math reasoning
3. **MBPP**: [arXiv:2108.07732](https://arxiv.org/abs/2108.07732) - Python programming benchmark
4. **Natural Questions**: [TACL 2019](https://ai.google.com/research/NaturalQuestions) - Real search queries
5. **LMSYS-Chat**: [arXiv:2309.11998](https://arxiv.org/abs/2309.11998) - Real chatbot conversations

## Next Steps

After training on real data:

1. **Benchmark Performance**: Compare against synthetic-trained model
2. **Feature Analysis**: Identify most important features
3. **Error Analysis**: Find patterns in misclassifications
4. **Active Learning**: Iteratively improve with hard examples
5. **Production Deployment**: A/B test in real application

## Files Created

```
llm_jury/
├── scripts/
│   ├── collect_real_intent_data.py       # Step 1: Download datasets
│   ├── teacher_label_intents.py          # Step 2: Oracle labeling
│   ├── split_labeled_data.py             # Step 3: Train/val/test split
│   └── train_xgboost_intent.py          # Step 4: Train XGBoost
├── data/
│   ├── real_intent_prompts_raw.json      # After Step 1
│   ├── real_intent_labeled.json          # After Step 2
│   └── real_intent_labeled_split.json    # After Step 3
└── models/
    ├── xgboost_intent_classifier.json    # After Step 4
    └── xgboost_intent_classifier.meta.json
```

---

**Ready to start?** Run Step 1:

```bash
python scripts/collect_real_intent_data.py
```

