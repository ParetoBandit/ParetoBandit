# Trust Score: SummEdits & Hallucination Metrics

This document explains the two complementary trustworthiness metrics—**SummEdits** and **Hallucination Rate**—and the rationale for combining them into a unified **Trust Score**.

---

## Overview

Trustworthiness in LLMs is multidimensional. A model can be reliable in recalling general world knowledge but fail when asked to faithfully summarize a specific document—or vice versa. To capture this nuance, we measure two distinct dimensions:

| Dimension | Metric | What It Measures |
|-----------|--------|------------------|
| **Intrinsic Consistency** | SummEdits Score | Faithfulness to a provided source document |
| **Extrinsic Consistency** | Hallucination Score | Accuracy relative to general world knowledge |

By combining both metrics into a weighted Trust Score, we obtain a robust, singular measure of model reliability across different operational contexts.

---

## SummEdits Score

### What is SummEdits?

**SummEdits** (Summary Edits) is a benchmark that measures an LLM's ability to detect factual inconsistencies between a generated summary and its source document. The task presents the model with a source document and a summary that may contain subtle edits that violate the original facts.

### Primary Purpose

**Intrinsic Consistency (Faithfulness)**

SummEdits evaluates whether a model can maintain factual integrity when transforming or analyzing a source text. This is critical for:

- **RAG (Retrieval-Augmented Generation)**: Ensuring generated responses accurately reflect retrieved documents
- **Document Analysis**: Verifying that extracted information matches the source
- **Summarization**: Confirming summaries don't introduce fabricated details
- **Legal/Medical Applications**: Where source fidelity is paramount

### Measurement

The SummEdits score represents the **percentage of summaries correctly classified as factually consistent or inconsistent** with their source document.

```
SummEdits Score = (Correct Classifications / Total Samples) × 100
```

### Example of Intrinsic Inconsistency

| Component | Content |
|-----------|---------|
| **Source Document** | "The quarterly meeting was scheduled for Tuesday, March 15th." |
| **Generated Summary** | "The meeting was held on Wednesday." |
| **Issue** | The summary contradicts the source—this is an **intrinsic inconsistency** |

The model should detect that "Wednesday" violates the source which clearly states "Tuesday."

---

## Hallucination Score

### What is Hallucination Rate?

The **Hallucination Score** measures an LLM's tendency to generate information that contradicts verifiable world knowledge, independent of any provided source document.

### Primary Purpose

**Extrinsic Consistency (Factual Recall)**

Hallucination scoring evaluates the model's general factual reliability—its ability to produce accurate information from its training data without inventing false facts.

### Measurement

The hallucination score is typically the **percentage of generated facts that are inconsistent with general world knowledge**, often inverted for scoring purposes (higher = better).

```
Hallucination Score (inverted) = 100 - Hallucination Rate
```

### Test Context

Unlike SummEdits, hallucination is assessed in contexts where:
- No source document is provided
- The model must rely on parametric (trained) knowledge
- Open-ended generation or QA tasks are performed

### Example of Extrinsic Inconsistency

| Component | Content |
|-----------|---------|
| **User Query** | "What is the capital of Spain?" |
| **Model Response** | "The capital of Spain is Barcelona." |
| **Issue** | This contradicts world knowledge (Madrid is the capital)—an **extrinsic inconsistency** |

The model fabricated incorrect information from its training data, not from a misread source.

---

## Comparison: Two Dimensions of Trustworthiness

| Feature | Hallucination Score | SummEdits Score |
|---------|---------------------|-----------------|
| **Primary Purpose** | Extrinsic Consistency (Factual Recall) | Intrinsic Consistency (Faithfulness) |
| **Measurement** | % of facts inconsistent with world knowledge | % of summaries correctly classified vs. source |
| **Test Context** | General QA, open-ended generation (no source) | Always requires a source document |
| **What It Captures** | Model's tendency to invent information | Model's ability to maintain source fidelity |
| **Failure Mode** | Knowledge deficiency or confabulation | Source adherence failure |
| **Critical For** | General assistant tasks, factual QA | RAG, document analysis, summarization |

---

## The Unique Insight: Faithfulness to Source

The critical distinction lies in understanding **where** the inconsistency originates:

### Intrinsic Inconsistency (SummEdits Domain)

> The generated text contradicts the **provided source document**.

This failure mode is particularly dangerous in enterprise applications where users expect the model to work with their specific documents. A model might have excellent general knowledge but still:

- Swap dates, names, or numbers from the source
- Infer conclusions not supported by the document
- Mix information from its training data with source content

### Extrinsic Inconsistency (Hallucination Domain)

> The generated text contradicts **general, verifiable world knowledge**.

This occurs when the model:

- Fabricates facts not in its training data
- Confuses similar entities or concepts
- Generates plausible-sounding but incorrect information

### Why Both Matter

A model could score well on one dimension but poorly on the other:

| Scenario | Hallucination Score | SummEdits Score | Risk |
|----------|---------------------|-----------------|------|
| **Strong general knowledge, poor source adherence** | High | Low | Ignores user documents, substitutes its own "facts" |
| **Poor general knowledge, strong source adherence** | Low | High | Reliable for document tasks, unreliable for general QA |
| **High trust** | High | High | Reliable across both contexts |

---

## Combining into a Trust Score

### Rationale

By incorporating both metrics into a unified **Trust Score**, we capture:

1. **Source Adherence** (SummEdits): The model's ability to stick to facts given a source
2. **Factual Reliability** (Hallucination): The model's general accuracy outside a source

This produces a singular, robust **Trust Metric** suitable for:

- Model selection decisions
- Quality routing based on trust requirements
- Identifying models suitable for high-stakes applications

### Integration with Bayesian Latent Factor Model

When combined using the [Bayesian Latent Factor approach](LATENT_FACTOR_MODULE.md), the Trust Score benefits from:

- **Learned importance weights**: The model learns how each metric contributes to overall trust
- **Uncertainty quantification**: Posterior credible intervals reflect measurement noise
- **Missing data handling**: Models with partial benchmark coverage are scored appropriately

### Weighted Composite Formula

For a simpler weighted average approach:

```
Trust Score = w₁ × SummEdits_normalized + w₂ × Hallucination_normalized
```

Where:
- `w₁` and `w₂` are weights (e.g., 0.5 each for equal weighting)
- Scores are normalized to a common scale (typically z-scores or 0-100)

---

## Practical Applications

### Use Case: RAG Pipeline Selection

When building a RAG system, prioritize models with high SummEdits scores—they're more likely to faithfully represent retrieved documents.

### Use Case: General Assistant

For open-ended chatbot applications, hallucination score may be more critical—users expect accurate world knowledge.

### Use Case: High-Stakes Applications

Legal, medical, and financial applications require high scores on **both** dimensions:
- High SummEdits: Won't misrepresent client documents
- Low Hallucination: Won't fabricate case law, diagnoses, or figures

---

## Summary

| Metric | Measures | Critical For |
|--------|----------|--------------|
| **SummEdits** | Faithfulness to source documents | RAG, document analysis, summarization |
| **Hallucination** | General factual accuracy | Open-ended QA, assistant tasks |
| **Trust Score** | Combined reliability | Holistic model evaluation |

By measuring both intrinsic and extrinsic consistency, the Trust Score provides a comprehensive view of model reliability that neither metric alone can offer.

---

## Related Documentation

- [Bayesian Latent Factor Module](LATENT_FACTOR_MODULE.md)
- [Composite Reasoning Score (CRS)](COMPOSITE_REASONING_SCORE.md)
- [Composite Coding Score (CCS)](COMPOSITE_CODING_SCORE.md)
