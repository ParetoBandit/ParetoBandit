# Response to Minor Reviewer Notes

## Summary

Two minor notes received, both easily addressable with one-sentence additions:

1. ✅ **NVIDIA Features Assumption** - Add sentence about calibration/noise assumption
2. ✅ **"Free" Data Acknowledgment** - Move OpenCompass acknowledgment to main text

---

## Minor Note #1: NVIDIA Features Assumption

### Reviewer's Note
> **NVIDIA Features**: You treat these as ground truth. Briefly mention (one sentence) that you assume the NVIDIA classifier is calibrated, or that its noise is random.

### ✅ RESPONSE: Add One Sentence

**Where to add**: In the **Feature Engineering** subsection of Methods

**Suggested Language** (choose one):

#### Option 1: Calibration Assumption (Preferred)
> We use the NVIDIA Prompt Task and Complexity Classifier (version X.X) to extract six prompt-level features: creativity scope, reasoning depth, constraint count, domain knowledge, contextual knowledge, and number of few-shot examples. **We assume the NVIDIA classifier is well-calibrated on our task domains; any residual measurement noise is expected to be random and thus attenuated through aggregation across our large sample (N>130K).**

#### Option 2: Noise Assumption (Alternative)
> We use the NVIDIA Prompt Task and Complexity Classifier to extract prompt-level features. **We treat these features as informative signals while acknowledging that any classifier noise is assumed to be random rather than systematically biased, which our XGBoost model can handle robustly.**

#### Option 3: Brief Version (Most Concise)
> We extract prompt features using the NVIDIA Prompt Task and Complexity Classifier. **We assume the classifier is sufficiently calibrated, with any residual noise being random.**

**Recommendation**: Use **Option 1** - it's the most defensible because:
1. Acknowledges calibration assumption explicitly
2. Explains why noise is not a major concern (large sample size)
3. Sounds scientifically rigorous

---

## Minor Note #2: "Free" Data Acknowledgment

### Reviewer's Note
> **"Free" Data**: You list "Free" as an advantage. Be careful—OpenCompass data is free to download, but someone paid to generate it. Acknowledge OpenCompass's contribution more formally (which you do in Appendix, but maybe move up).

### ✅ RESPONSE: Two Actions

#### Action 1: Revise "Free" Language

**Current problematic phrasing** (likely in Introduction or Methods):
> ❌ "We use free, open-source benchmark data from OpenCompass..."
> ❌ "Our approach is cost-effective because the data is freely available..."

**Revised phrasing**:
> ✅ "We leverage publicly available benchmark results from OpenCompass [cite], a community-driven effort that has evaluated 100+ models across academic benchmarks. While the data is freely accessible for research, we acknowledge the significant computational resources invested by the OpenCompass team in generating these comprehensive evaluations."

#### Action 2: Move Acknowledgment to Main Text

**Where to add**: In the **Data Collection** subsection of Methods (before the detailed description)

**Suggested Language**:

> **Data Sources and Acknowledgments**
> 
> Our instance-level training data is sourced from OpenCompass [cite], an open evaluation platform that provides comprehensive benchmark results for 100+ language models. We acknowledge the OpenCompass team's substantial contribution in generating and publicly releasing these evaluation datasets, which enable reproducible research without requiring extensive computational resources. Their platform provides raw model predictions and ground-truth labels across GPQA (reasoning), HumanEval (coding), IFEval (instruction-following), and TriviaQA (question-answering), totaling 133,394 labeled instances.
> 
> While these datasets are publicly accessible for research purposes, we recognize that generating them required significant GPU hours and careful benchmark curation by the OpenComass consortium. This work would not be possible without their open science commitment.

**And in the Acknowledgments** (keep the existing text, but brief):
> We thank the OpenCompass team for providing public access to their comprehensive model evaluation results.

---

## Specific Edits by Section

### Methods Section: Feature Engineering

**Add this paragraph**:

```
Prompt-Level Features

We extract six prompt-level features using the NVIDIA Prompt Task and 
Complexity Classifier (nvidia/prompt-task-and-complexity-classifier-v1), 
a transformer-based model trained to assess prompt characteristics:

1. creativity_scope: Open-ended vs. constrained generation (0-1)
2. reasoning: Logical reasoning complexity (0-1)  
3. constraint_ct: Number of explicit constraints (integer)
4. domain_knowledge: Required domain expertise (0-1)
5. contextual_knowledge: Contextual information needed (0-1)
6. number_of_few_shots: Few-shot examples present (integer)

These features are computed once per prompt and combined with model capability
scores to create our feature vectors. We assume the NVIDIA classifier is 
well-calibrated on our task domains; any residual measurement noise is 
expected to be random and thus attenuated through aggregation across our 
large sample (N=133,394).                    ← ADD THIS SENTENCE
```

---

### Methods Section: Data Collection

**Add this before detailed data description**:

```
Data Sources and Acknowledgments

Our instance-level training data is sourced from OpenCompass (cite: 
https://opencompass.org.cn/), an open evaluation platform that provides 
comprehensive benchmark results for 100+ language models. We acknowledge 
the OpenCompass team's substantial contribution in generating and publicly 
releasing these evaluation datasets, which enable reproducible research 
without requiring extensive computational resources for model inference.

While these datasets are publicly accessible for research purposes, we 
recognize that generating them required significant GPU hours and careful 
benchmark curation. This work would not be possible without their open 
science commitment.

Specifically, we use OpenCompass predictions for:
- GPQA (Reasoning): 198 questions × 42 models = 8,316 instances
- HumanEval (Coding): 164 problems × 40 models = 6,560 instances
- IFEval (Summarization): 541 instructions × 42 models = 22,722 instances
- TriviaQA (RAG): 7,993 questions × 12 models = 95,796 instances

Total: 133,394 labeled training instances.
```

---

### Abstract/Introduction: Revise "Free" Claims

**Before** (if you have this):
> ❌ "Our approach is cost-effective, using freely available benchmark data..."

**After**:
> ✅ "Our approach leverages publicly available benchmark results from 
> OpenCompass, enabling reproducible research without requiring extensive 
> computational resources for model evaluation..."

---

### Related Work: Add OpenCompass Citation

**Add this paragraph** (if not already present):

```
Open Evaluation Platforms

Several initiatives have emerged to democratize LLM evaluation through 
open platforms. OpenCompass [cite] provides comprehensive benchmark results 
for 100+ models across academic tasks, releasing raw predictions and labels 
for reproducibility. HELM [cite] offers standardized evaluations with a 
focus on holistic assessment. Our work builds on OpenCompass's infrastructure, 
using their publicly released prediction logs to construct instance-level 
training data for performance prediction.
```

---

## Citation to Add

**In References**:

```
@misc{opencompass2024,
  title={OpenCompass: A Universal Evaluation Platform for Large Language Models},
  author={OpenCompass Contributors},
  year={2024},
  howpublished={\url{https://opencompass.org.cn/}},
  note={Accessed: 2024-12-13}
}
```

Or if there's a paper:

```
@inproceedings{opencompass2024,
  title={OpenCompass: A Universal Evaluation Platform for Foundation Models},
  author={OpenCompass Team},
  booktitle={NeurIPS Datasets and Benchmarks Track},
  year={2024}
}
```

*(Check if OpenCompass has a published paper to cite properly)*

---

## Summary of Changes

### Quick Checklist

- [ ] **Add NVIDIA calibration sentence** in Methods → Feature Engineering
- [ ] **Add OpenCompass acknowledgment** in Methods → Data Collection (main text)
- [ ] **Revise "free data" claims** to "publicly available" throughout
- [ ] **Add OpenCompass citation** to References
- [ ] **Keep brief acknowledgment** in Acknowledgments section
- [ ] **Check Introduction** for any "free" or "zero-cost" claims

---

## Example Combined Text (Methods Section)

Here's how the revised Methods section would read:

```
3. Methods

3.1 Data Collection

Data Sources and Acknowledgments

Our instance-level training data is sourced from OpenCompass [cite], an 
open evaluation platform that provides comprehensive benchmark results 
for 100+ language models. We acknowledge the OpenCompass team's substantial 
contribution in generating and publicly releasing these evaluation datasets, 
which enable reproducible research without requiring extensive computational 
resources. While these datasets are publicly accessible for research 
purposes, we recognize that generating them required significant GPU hours 
and careful benchmark curation.

We collect raw model predictions and ground-truth labels across four task 
categories:
[... rest of data description ...]

3.2 Feature Engineering

Prompt-Level Features

We extract six prompt-level features using the NVIDIA Prompt Task and 
Complexity Classifier, a transformer-based model trained to assess prompt 
characteristics: creativity scope, reasoning depth, constraint count, 
domain knowledge, contextual knowledge, and number of few-shot examples. 
We assume the NVIDIA classifier is well-calibrated on our task domains; 
any residual measurement noise is expected to be random and thus attenuated 
through aggregation across our large sample (N=133,394).

Model Capability Features

For each intent, we use task-specific capability proxies...
[... rest of feature description ...]
```

---

## Draft Responses for Rebuttal (If Needed)

### Response to NVIDIA Features Note
> **R: NVIDIA Features Assumption**
> 
> We have added a sentence in Section 3.2 acknowledging our assumption that the NVIDIA classifier is well-calibrated, with any residual noise being random and attenuated through our large sample size (N=133,394). This is a standard assumption for using pre-trained classifiers as feature extractors in machine learning pipelines.

### Response to "Free" Data Note
> **R: OpenCompass Acknowledgment**
> 
> We have moved the OpenCompass acknowledgment from the Appendix to the main text (Section 3.1) and revised all instances of "free data" to "publicly available data" to properly credit the computational resources invested by the OpenCompass team. We now explicitly acknowledge their contribution in the Methods section before describing our data collection process.

---

## Files to Update

1. **Main paper (LaTeX/Word)**:
   - Methods → Feature Engineering: Add NVIDIA calibration sentence
   - Methods → Data Collection: Add OpenCompass acknowledgment
   - Introduction: Check for "free" claims, revise to "publicly available"
   - References: Add OpenCompass citation

2. **Supplementary Materials**:
   - Can keep detailed OpenCompass description in Appendix
   - Reference main text acknowledgment

3. **Acknowledgments**:
   - Keep brief: "We thank the OpenCompass team for providing public access to their comprehensive evaluation results."

---

## Verification Checklist

- [ ] "Free" → "publicly available" throughout paper
- [ ] NVIDIA calibration assumption stated (1 sentence)
- [ ] OpenCompass acknowledgment in Methods (main text)
- [ ] OpenCompass cited properly in References
- [ ] Removed any "zero-cost" or "free" claims
- [ ] Checked Introduction, Abstract, Methods, Discussion for these issues

---

## Conclusion

✅ **Both minor notes are easily fixable with minimal text additions**

**Changes required**:
1. Add **1 sentence** about NVIDIA calibration (Methods)
2. Add **1 paragraph** acknowledging OpenCompass (Methods)
3. Replace "free" with "publicly available" (throughout)
4. Add proper citation (References)

**Total additional text**: ~100-150 words

These are standard good practices and actually **strengthen** the paper by being more precise and acknowledging contributions! 🎯
