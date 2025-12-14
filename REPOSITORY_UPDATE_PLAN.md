# Repository Update Plan - Post KDD/data Refactoring

**Date**: December 13, 2024  
**Context**: After refactoring and organizing `KDD/data/`, this document identifies what needs updating in the broader repository.

---

## ✅ What's Already Good

### The `llm_jury/` Package (Core Library)
- ✅ **No updates needed** - Package doesn't reference `KDD/data/` scripts
- ✅ Has its own intent classification system (`llm_jury/routing/xgboost_intent_classifier.py`)
- ✅ Has comprehensive README and documentation (`llm_jury/intent/README.md`)
- ✅ Well-organized: `routing/`, `optimization/`, `ranking/`, `data/`, etc.

### The `data/` Directory (Root-Level)
- ✅ `data/README.md` is comprehensive and up-to-date
- ✅ Contains `models_cache.json` (central model registry)
- ✅ No references to KDD/data scripts

### Tests
- ✅ `tests/` directory is independent
- ✅ 23/23 tests passing in `KDD/data/tests/`

---

## 📝 What Needs Updating

### 1. **KDD/README.md** ⚠️ CRITICAL

**Current Status**: Outdated - only mentions BLF section

**What's Missing**:
- No mention of the **4 production XGBoost models** (Reasoning, Coding, Summarization, RAG)
- No reference to `KDD/data/` refactoring and new structure
- No mention of **113K training examples** collected
- No link to `KDD/data/README.md` or `FINAL_SYSTEM_STATUS.md`
- Submission checklist doesn't include data collection work

**Recommended Updates**:
```markdown
## 📂 Directory Structure

KDD/
├── data/                  # ⭐ Intent Prediction Models & Training Data
│   ├── README.md          # Complete guide to data pipeline
│   ├── FINAL_SYSTEM_STATUS.md  # Production system summary
│   ├── core_scripts/      # Data collection & model training (4 scripts)
│   ├── validation/        # Validation scripts (2 scripts)
│   ├── production_models/ # 4 trained XGBoost models
│   ├── instance_level_training_data/  # 113K examples
│   ├── tests/             # 23 unit tests (all passing)
│   └── documentation/     # Methodology, validation, reviewer responses
│
├── BLF/                   # Bayesian Latent Factor Model Section
│   └── ... (11 files)
│
└── README.md              # This file
```

Add new section:

```markdown
## 🎯 Intent Prediction System (KDD/data/)

**Status**: ✅ Production Ready

The `data/` subdirectory contains a complete system for predicting LLM performance across different task intents using XGBoost models.

### Key Components
- **4 Production Models**: Reasoning, Coding, Summarization, RAG
- **113K Training Examples**: Instance-level data from OpenCompass benchmarks
- **23 Unit Tests**: All passing, comprehensive coverage
- **Zero-Shot Transfer Validation**: Proven on proprietary models

### Results

| Intent | Training N | Test AUC | Test Acc | Transfer r | Status |
|--------|-----------|----------|----------|------------|--------|
| Coding | 5,576 | 0.969 | 91.7% | 0.480*** | ✅ Ready |
| Summarization | 19,313 | 0.896 | 93.8% | 0.744*** | ✅ Ready |
| Reasoning | 7,068 | 0.824 | 75.7% | 0.580*** | ✅ Ready |
| RAG | 81,426 | 0.779 | 85.1% | 0.453*** | ✅ Ready |

***p < 0.0001*

### Quick Start
\```bash
cd KDD/data
cat README.md  # Complete documentation
cat FINAL_SYSTEM_STATUS.md  # System summary
python -m pytest tests/ -v  # Run all tests
\```

### Features
- **NVIDIA Prompt Complexity**: 6 features per prompt (creativity, reasoning, etc.)
- **Capability Proxies**: External benchmarks (MMLU-Pro for RAG, etc.)
- **Robust Training**: 5-fold CV, 85/15 train/test split
- **Validated Transfer**: Significant correlations with proprietary model performance

See `KDD/data/FINAL_SYSTEM_STATUS.md` for complete details.
```

---

### 2. **Root README.md** 📝 MISSING

**Current Status**: **Does not exist!**

**Recommended Creation**:
Create `/Users/annette/repostitories/llm_jury/README.md` with:

```markdown
# LLM Jury: Intent-Aware Multi-Model Routing

> Cost-effective LLM applications through intelligent model routing and quality prediction

[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.9+-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

---

## 🎯 Overview

LLM Jury is a research project for optimizing LLM application costs through:
1. **Intent-aware routing** - Classify prompts and route to specialized models
2. **Quality prediction** - Predict model performance without expensive API calls
3. **Bayesian aggregation** - Handle missing benchmark data gracefully
4. **Cost optimization** - Minimize costs while maintaining quality thresholds

---

## 📂 Project Structure

\```
llm_jury/
├── llm_jury/              # Core Python package
│   ├── routing/           # Intent classification & prompt routing
│   ├── ranking/           # Model quality scoring & ranking
│   ├── optimization/      # Cost optimization algorithms
│   ├── data/              # Data collection & model registry
│   └── etl/               # Data pipelines
│
├── KDD/                   # KDD 2025 submission materials
│   ├── data/              # ⭐ Intent prediction models (113K examples)
│   └── BLF/               # Bayesian latent factor model
│
├── data/                  # Shared data (models_cache.json, benchmarks)
├── tests/                 # Unit tests
├── scripts/               # Utility scripts
└── docs/                  # Documentation
\```

---

## 🚀 Quick Start

### Installation
\```bash
git clone https://github.com/yourusername/llm_jury.git
cd llm_jury
pip install -r requirements.txt
\```

### Basic Usage
\```python
from llm_jury.routing import IntentClassifier
from llm_jury.ranking import QualityScorer

# Classify intent
classifier = IntentClassifier()
intent = classifier.classify("Write a Python function to sort a list")
# Intent: "coding"

# Rank models for this intent
scorer = QualityScorer()
rankings = scorer.rank_for_intent(intent, budget=0.10)
# Returns: [(model_name, quality_score, cost), ...]
\```

---

## 📊 Key Results

### Intent Prediction (KDD/data/)
- **4 production models** (Reasoning, Coding, Summarization, RAG)
- **Test AUC**: 0.779-0.969 across intents
- **Transfer correlation**: 0.453-0.744 (all p<0.0001)
- **Training data**: 113K instance-level examples

### Bayesian Quality Aggregation (KDD/BLF/)
- **Spearman ρ = 0.89** with human preferences (Chatbot Arena)
- **95% model coverage** (vs 68% for baseline methods)
- **Principled missing data** handling (no ad-hoc imputation)

---

## 🎓 Research Papers

### KDD 2025 Submission
**Title**: "LLM Jury: Intent-Aware Multi-Model Routing for Cost-Effective LLM Applications"

**Key Contributions**:
1. Instance-level training data collection from OpenCompass
2. Intent-specific quality prediction models
3. Bayesian benchmark aggregation with missing data handling
4. Zero-shot transfer validation on proprietary models

**Status**: In preparation  
**Directories**: `KDD/data/`, `KDD/BLF/`

---

## 📚 Documentation

### For Users
- **Getting Started**: `docs/guides/`
- **API Reference**: `llm_jury/` (docstrings)
- **Examples**: `examples/`

### For Researchers
- **KDD Data Pipeline**: `KDD/data/README.md`
- **BLF Model**: `KDD/BLF/README.md`
- **Methodology**: `KDD/data/documentation/methodology/`
- **Validation Results**: `KDD/data/documentation/validation/`

---

## 🧪 Testing

\```bash
# Core library tests
cd tests/
pytest -v

# KDD data pipeline tests
cd KDD/data/
python -m pytest tests/ -v
# Expected: 23/23 passing
\```

---

## 📦 Key Datasets

### models_cache.json
Central registry of LLM models with:
- Benchmark scores (MMLU-Pro, HumanEval+, IFEval, etc.)
- Pricing (input/output costs)
- Context lengths
- Metadata

**Location**: `data/models_cache.json`  
**Documentation**: `data/README.md`

### Instance-Level Training Data
113K prompt-level examples with:
- Prompts from 4 intents
- Model predictions
- NVIDIA complexity features
- Success labels

**Location**: `KDD/data/instance_level_training_data/`  
**Documentation**: `KDD/data/FINAL_SYSTEM_STATUS.md`

---

## 🤝 Contributing

We welcome contributions! See `CONTRIBUTING.md` for guidelines.

---

## 📜 License

MIT License - see `LICENSE` for details.

---

## 📧 Contact

**Issues**: https://github.com/yourusername/llm_jury/issues  
**Discussions**: https://github.com/yourusername/llm_jury/discussions

---

## 🙏 Acknowledgments

- **OpenCompass** for open-source benchmark predictions
- **NVIDIA** for prompt complexity classifier
- **Artificial Analysis** for LLM pricing and performance data
- **Chatbot Arena** for human preference rankings

---

**Last Updated**: December 13, 2024
\```
```

---

### 3. **Integration Points** 🔗

Currently, the `llm_jury/` package and `KDD/data/` systems are **independent**:

**llm_jury/routing/xgboost_intent_classifier.py**:
- Uses **pattern-based features** (regex, word counts, etc.)
- 5 classes: reasoning, coding, factual_qa, agentic, general
- Trained on labeled prompt data

**KDD/data/ XGBoost models**:
- Use **NVIDIA complexity features** + **capability proxies**
- 4 intents: reasoning, coding, summarization, rag
- Predict **success probability** (not just intent)
- Trained on **113K instance-level examples** from OpenCompass

**Potential Integration** (Future Work):
```python
# Option 1: Use KDD models directly in llm_jury package
from KDD.data.production_models import load_model

model = load_model('rag')
success_prob = model.predict_proba(prompt_features)

# Option 2: Create adapter layer
from llm_jury.routing import KDDIntentPredictor

predictor = KDDIntentPredictor(
    model_dir='KDD/data/production_models/'
)
result = predictor.predict(prompt, model='GPT-4o mini')
# Returns: {intent: 'rag', success_prob: 0.87, should_use: True}
```

**Recommendation**: Keep separate for now (different use cases), but document relationship.

---

## 📋 Action Items

### High Priority
- [ ] **Update KDD/README.md** (add data/ section)
- [ ] **Create root README.md** (project overview)
- [ ] **Update KDD/data/README.md** (link to parent README)

### Medium Priority
- [ ] Document relationship between `llm_jury/routing/` and `KDD/data/` models
- [ ] Add "See also" links in both READMEs
- [ ] Update `CONTRIBUTING.md` (if exists) with new structure

### Low Priority (Future)
- [ ] Create integration adapter between systems
- [ ] Unify intent taxonomies (5 classes vs 4 intents)
- [ ] Add cross-references in docstrings

---

## 🎯 Summary

**What's Working Well**:
- ✅ `KDD/data/` is clean, organized, and well-documented
- ✅ `llm_jury/` package is independent and functional
- ✅ All tests passing (23/23 in KDD/data)

**What Needs Attention**:
- ⚠️ KDD/README.md doesn't mention data/ work
- ⚠️ No root README.md for overall project
- ⚠️ Integration points not documented

**Estimated Time**:
- KDD/README.md update: 15 minutes
- Root README.md creation: 30 minutes
- Documentation links: 15 minutes
**Total**: ~1 hour

---

**Status**: Ready to implement updates  
**Next Step**: Update KDD/README.md first (highest impact)
