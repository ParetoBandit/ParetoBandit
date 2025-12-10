# Figure Captions for Intent Classification Section

## Figure 1: Confusion Matrix

**Left panel**: Raw count confusion matrix showing classification results from 5-fold cross-validation (n=2,458). Diagonal elements represent correct predictions. The model achieves high accuracy across all classes, with SUMMARIZATION (99.8%) and REASONING (98.8%) showing near-perfect performance.

**Right panel**: Row-normalized confusion matrix showing the proportion of true labels assigned to each predicted class. The GENERAL class shows the highest confusion rate (11.6% misclassified as FACTUAL_QA), reflecting the inherent ambiguity of catch-all categories. SUMMARIZATION exhibits minimal confusion (99.8% diagonal), demonstrating clear semantic boundaries.

---

## Figure 2: Per-Class Classification Performance

Comparison of accuracy and F1-scores across five intent classes. Specialized task classes (SUMMARIZATION, REASONING, FACTUAL_QA) achieve >96% accuracy, while the catch-all GENERAL class reaches 84.1%. The red dashed line indicates overall cross-validation accuracy (94.47%). Error bars represent 95% confidence intervals from bootstrap resampling (n=1000).

Key observations:
- SUMMARIZATION: Near-perfect classification (99.8% accuracy, 98.6% F1)
- REASONING and FACTUAL_QA: Excellent performance (>96%)
- CODING: Strong performance (92.2%) with minor confusion with GENERAL
- GENERAL: Acceptable performance (84.1%) given inherent category ambiguity

---

## Figure 3: 5-Fold Cross-Validation Results

Fold-by-fold breakdown showing consistency across validation splits. Each fold maintains stratified class distribution (±0.1%) and exhibits stable performance (mean=94.47%, std=1.00%). The red dashed line indicates mean accuracy across all folds. Minimal variance demonstrates robust generalization independent of train/validation split.

Statistical analysis:
- Mean accuracy: 94.47% ± 1.00%
- Mean F1-score: 94.39% ± 1.06%
- Range: 92.89% - 95.73%
- All folds exceed 92% accuracy threshold

---

## Figure 4: Dataset Distribution and Prompt Length Analysis

**Left panel**: Sample counts per intent class showing near-balanced distribution. Final dataset contains 2,458 unique prompts after deduplication (42 duplicates removed). Class distribution ranges from 465 (GENERAL) to 500 (CODING, REASONING, FACTUAL_QA), representing 18.9%-20.3% of total samples.

**Right panel**: Box plots of prompt length distribution by intent (log scale). SUMMARIZATION exhibits significantly longer prompts (median=1,017 chars) due to inclusion of article text, while FACTUAL_QA shows compact queries (median=46 chars). Whiskers represent 1.5×IQR; outliers shown as circles. Despite length variance, classification performance remains strong, indicating semantic features dominate over length-based shortcuts.

---

## Figure 5: Data Source Breakdown

Hierarchical view of dataset composition showing 7 distinct sources mapping to 5 intent classes. Color coding indicates intent mapping: blue (CODING), orange (REASONING), green (FACTUAL_QA), red (SUMMARIZATION), purple (GENERAL). Each source contributes exclusively to a single intent class, ensuring no cross-contamination.

Source details:
- **CODING** (500 total): MBPP (120), HumanEval (164), CodeAlpaca (216)
- **REASONING** (500): GSM8k (500)
- **FACTUAL_QA** (500): Natural Questions (500)
- **SUMMARIZATION** (493): CNN/DailyMail (493)
- **GENERAL** (465): WildChat (465)

All sources are publicly available on HuggingFace, enabling full reproducibility.

---

## Supplementary Figures (Online Appendix)

### Figure S1: Learning Curves
Training and validation accuracy across increasing training set sizes (10%, 25%, 50%, 75%, 100%). Shows saturation at ~80% of data, suggesting dataset is sufficiently large for current task complexity.

### Figure S2: Feature Importance (Top 20)
XGBoost feature importance scores for top 20 embedding dimensions (out of 384). No single dimension dominates, indicating distributed semantic representation. Top features show 2-5% importance, consistent with ensemble learning from rich embedding space.

### Figure S3: Misclassification Examples
Annotated examples of the most common confusion patterns:
- GENERAL → FACTUAL_QA (11.6%)
- CODING → FACTUAL_QA (3.8%)
- CODING → GENERAL (2.4%)

Each example includes ground-truth label, predicted label, confidence score, and linguistic analysis explaining the confusion.

---

## Data Availability

All figures generated using open-source tools:
- Python 3.12, matplotlib 3.8.0, seaborn 0.13.0
- Source data: `data/real_intent_prompts_labeled.json`
- Generation script: `KDD/intent_classification/generate_figures.py`
- High-resolution versions (300 DPI) available in repository

## Reproduction

To regenerate all figures:
```bash
cd KDD/intent_classification
python generate_figures.py
```

Output files:
- `figure1_confusion_matrix.png`
- `figure2_per_class_performance.png`
- `figure3_cv_folds.png`
- `figure4_data_distribution.png`
- `figure5_data_sources.png`
