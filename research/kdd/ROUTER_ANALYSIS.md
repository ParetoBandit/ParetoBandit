# KDD Review: Open-Source LLM Router Analysis

## Executive Summary

This document provides a deep technical analysis of open-source LLM routing systems and assesses whether our prepared dataset is appropriate for fair comparison. The analysis covers:

1. **RouteLLM** - Matrix factorization binary routing
2. **FrugalGPT** - Learned cascading with confidence scoring
3. **LLM Jury** - Archetype-based multi-objective routing
4. **Not Diamond** - Meta-model API + custom router training
5. **Semantic Router** - Semantic embedding similarity routing

---

## Router Deep Dive

### 1. RouteLLM (LMSYS)

**Paper**: "RouteLLM: Learning to Route LLMs with Preference Data" (2024)

**Methodology**:
```
┌─────────────────────────────────────────────────────────────┐
│                      RouteLLM Architecture                   │
├─────────────────────────────────────────────────────────────┤
│  Input: User Prompt                                          │
│     ↓                                                        │
│  Matrix Factorization Model (MF)                             │
│     - Trained on LMSYS Arena battles (55K)                   │
│     - Embeddings: OpenAI text-embedding-3-small              │
│     - Predicts: P(strong_model_wins | prompt)                │
│     ↓                                                        │
│  Threshold Comparison                                        │
│     - If win_rate >= threshold → Route to Strong (GPT-4)     │
│     - If win_rate < threshold → Route to Weak (GPT-3.5)      │
└─────────────────────────────────────────────────────────────┘
```

**Training Data**: 
- `lmsys/lmsys-arena-human-preference-55k` (Hugging Face)
- `routellm/gpt4_judge_battles` (augmented with GPT-4 as judge)

**Key Code** (from routellm/routers/routers.py):
```python
class MatrixFactorizationRouter(Router):
    def calculate_strong_win_rate(self, prompt):
        winrate = self.model.pred_win_rate(
            self.strong_model_id, self.weak_model_id, prompt
        )
        return winrate  # Returns 0-1 probability
```

**What It Learns**: "Which prompts do humans prefer GPT-4's response over GPT-3.5?"

---

### 2. FrugalGPT (Stanford)

**Paper**: "FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance" (2023)

**Methodology**:
```
┌─────────────────────────────────────────────────────────────┐
│                    FrugalGPT Cascade                         │
├─────────────────────────────────────────────────────────────┤
│  Input: Query + 8-shot examples (task-specific)              │
│     ↓                                                        │
│  Stage 1: Call cheapest model (GPT-3.5)                      │
│     ↓                                                        │
│  DistilBERT Confidence Scorer                                │
│     - Trained per-task on (query, response, correct?) data   │
│     - Outputs: confidence score 0-1                          │
│     ↓                                                        │
│  Decision: confidence >= threshold?                          │
│     - Yes → Return response                                  │
│     - No → Escalate to next model (GPT-4)                    │
│     ↓                                                        │
│  Repeat until confidence met OR final model reached          │
└─────────────────────────────────────────────────────────────┘
```

**Training Data**: 
- Task-specific datasets (HEADLINES, SciQ, etc.)
- Requires: (query, model_response, is_correct) tuples

**Key Insight**: FrugalGPT's confidence scorer is trained **per-task**. Their HEADLINES model is specifically trained to predict gold price classification accuracy.

**What It Learns**: "How confident should I be that this model's answer is correct for THIS specific task?"

---

### 3. LLM Jury (This Project)

**Methodology**:
```
┌─────────────────────────────────────────────────────────────┐
│                    LLM Jury Architecture                     │
├─────────────────────────────────────────────────────────────┤
│  Input: User Prompt                                          │
│     ↓                                                        │
│  Stage 1: Intent Classification (Regex + HF Zero-Shot)       │
│     - Coding, Creative, Reasoning, QA, General               │
│     ↓                                                        │
│  Stage 2: Complexity Classification                          │
│     - DIRECT_ANSWER, SIMPLE_TASK, MULTI_STEP, COMPLEX_TASK   │
│     ↓                                                        │
│  Stage 3: Archetype Mapping                                  │
│     ┌────────────────────────────────────────────────────┐   │
│     │ Intent + Complexity → Archetype                    │   │
│     │                                                    │   │
│     │ Complex + Coding → REASONING_SPECIALIST           │   │
│     │ Complex + General → FRONTIER                      │   │
│     │ Simple + Any → BULK_OPS                           │   │
│     │ Moderate → RAG_SPECIALIST                         │   │
│     └────────────────────────────────────────────────────┘   │
│     ↓                                                        │
│  Stage 4: Model Selection (Multi-Objective Optimization)     │
│     - Quality, Cost, Latency, Hallucination, Refusal        │
└─────────────────────────────────────────────────────────────┘
```

**Training Data**: **NONE** - Pure heuristic + zero-shot classification

**What It Learns**: Nothing task-specific. Uses general patterns.

---

### 4. Not Diamond

**Methodology**: Two approaches

#### A. API Router (Proprietary)
```
┌─────────────────────────────────────────────────────────────┐
│                Not Diamond API (Proprietary)                 │
├─────────────────────────────────────────────────────────────┤
│  Input: Prompt + List of candidate models                    │
│     ↓                                                        │
│  Not Diamond Meta-Model (Black Box)                          │
│     - Trained on internal benchmark data                     │
│     - Considers: quality, cost, latency                      │
│     ↓                                                        │
│  Output: Best model selection + response                     │
└─────────────────────────────────────────────────────────────┘
```

**⚠️ NOT OPEN SOURCE**: The routing intelligence is in their cloud API.

#### B. Custom Router Training (Open Source)
```
┌─────────────────────────────────────────────────────────────┐
│              Not Diamond Custom Router                       │
├─────────────────────────────────────────────────────────────┤
│  Training Input:                                             │
│     - Dataset with prompts                                   │
│     - Responses from each candidate model                    │
│     - Scores for each response (quality metric)              │
│     ↓                                                        │
│  CustomRouter.fit(dataset, prompt_col, response_col, score)  │
│     - Uploads to Not Diamond API                             │
│     - Returns preference_id for routing                      │
│     ↓                                                        │
│  Inference: Still calls Not Diamond API                      │
└─────────────────────────────────────────────────────────────┘
```

**Training Data**: User-provided (prompts + model responses + scores)

**Key Code**:
```python
class CustomRouter:
    def fit(self, dataset, prompt_column, response_column, score_column):
        # Uploads to Not Diamond API for training
        # Returns preference_id for future routing calls
```

**What It Learns**: "Which model gets highest score on my custom metric?"

---

### 5. Semantic Router (Aurelio Labs)

**Methodology**:
```
┌─────────────────────────────────────────────────────────────┐
│                   Semantic Router                            │
├─────────────────────────────────────────────────────────────┤
│  Setup: Define Routes with example utterances                │
│     Route("weather", utterances=["what's the weather",       │
│                                   "is it raining"])          │
│     Route("booking", utterances=["book a flight",            │
│                                   "reserve a hotel"])        │
│     ↓                                                        │
│  Pre-compute: Embed all utterances → route embeddings        │
│     ↓                                                        │
│  Inference: User query                                       │
│     ↓                                                        │
│  Embed query → Find most similar route by cosine similarity  │
│     ↓                                                        │
│  Return: Route name + trigger associated action              │
└─────────────────────────────────────────────────────────────┘
```

**Training Data**: User-defined utterances per route (few-shot examples)

**Key Insight**: Semantic Router is designed for **intent classification / action routing**, NOT model selection for quality optimization.

**What It Learns**: "Which predefined category does this query belong to?"

---

## Dataset Appropriateness Matrix

| Router | Task Type | Training Paradigm | Our Dataset Appropriate? | Reason |
|--------|-----------|-------------------|--------------------------|--------|
| **RouteLLM** | Binary (strong/weak) | Supervised (preferences) | ✅ **YES** (held-out) | Uses LMSYS preference data; we use 20% held-out |
| **FrugalGPT** | Cascading | Supervised (task-specific) | ✅ **YES** (OOD) | Trained on HEADLINES; our data tests generalization |
| **LLM Jury** | Multi-archetype | Zero-shot | ✅ **YES** | No training = no contamination |
| **Not Diamond API** | Multi-model | Black-box | ⚠️ **PARTIAL** | Can't verify training data overlap |
| **Not Diamond Custom** | Multi-model | User-trained | ✅ **YES** | We could train on our train split |
| **Semantic Router** | Intent classification | Few-shot (utterances) | ❌ **NO** | Different task paradigm |

---

## Critical Analysis: Semantic Router

### Why Semantic Router is NOT directly comparable:

**Designed Task**: Route queries to **predefined actions/categories**
- Example: "book a flight" → `booking_handler()`
- Example: "what's 2+2" → `calculator_tool()`

**Our Task**: Select best **LLM model** for quality/cost optimization
- Example: "complex coding task" → GPT-4 (strong)
- Example: "simple greeting" → GPT-3.5 (weak)

### Could Semantic Router be adapted?

**Yes, but requires transformation**:
```python
# Define routes as model tiers
Route("strong_model", utterances=[
    "write complex algorithm",
    "analyze this research paper",
    "solve this math proof"
])

Route("weak_model", utterances=[
    "hello",
    "what's the capital of France",
    "translate hello to Spanish"
])

# Then route queries to model tiers
```

**Problem**: This becomes a **hand-crafted rule system** with example utterances. It's not learning from preference data - it's pattern matching.

---

## Recommendations for Fair Comparison

### 1. Primary Comparison (Well-Suited to Our Dataset)

| Router | Include | Notes |
|--------|---------|-------|
| RouteLLM | ✅ Yes | Use held-out LMSYS data |
| FrugalGPT | ✅ Yes | Tests OOD generalization |
| LLM Jury | ✅ Yes | Zero-shot baseline |
| Not Diamond Custom | ⚠️ Optional | Train on our train split |

### 2. Exclude from Primary Comparison

| Router | Reason |
|--------|--------|
| Not Diamond API | Black-box, can't verify fairness |
| Semantic Router | Different task paradigm (intent→action, not model selection) |

### 3. If Including Semantic Router

Create a **separate ablation study** that:
1. Defines "strong_model" and "weak_model" routes with example utterances
2. Uses our training data to craft those utterances
3. Evaluates as a few-shot intent classifier
4. **Clearly labels** this as a different routing paradigm

---

## Dataset Verification Checklist

✅ **Multi-domain coverage**: coding (14.6%), creative (10.6%), reasoning (21.4%), qa (18.6%), general (34.8%)

✅ **Difficulty distribution**: easy (5.0%), medium (55.9%), hard (39.1%)

✅ **Source diversity**: LMSYS Arena (49.3%), WildBench (50.7%)

✅ **Proper splits**: train (60%), val (20%), test (20%)

✅ **Held-out from RouteLLM training**: Using 20% hash-based subset of LMSYS

✅ **Out-of-domain for FrugalGPT**: No gold price classification data

✅ **No training contamination for LLM Jury**: Zero-shot approach

---

## Conclusion

Our dataset is **well-suited** for comparing:
- **RouteLLM** (with held-out precaution)
- **FrugalGPT** (as out-of-domain generalization test)
- **LLM Jury** (as zero-shot baseline)

**Not Diamond API** should be excluded (black-box) or clearly labeled as non-reproducible.

**Semantic Router** is a **different paradigm** (intent→action routing) and should either be:
- Excluded from primary comparison, OR
- Adapted with clear methodology and labeled as ablation study

This ensures our KDD paper maintains scientific rigor while providing actionable insights for practitioners choosing LLM routing solutions.

