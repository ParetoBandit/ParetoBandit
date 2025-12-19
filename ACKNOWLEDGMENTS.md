# Acknowledgments

BanditGPT builds upon the work of many open-source projects and datasets. We gratefully acknowledge the following contributions:

---

## Data Pipeline for Expert Priors

The bundled `expert_priors.npz` (which provides 63% regret reduction) was generated through the following pipeline:

| Step | Data Source | Output |
|------|-------------|--------|
| 1. **Prompt Collection** | LMSYS Chatbot Arena (50K samples) | Raw prompts |
| 2. **Clustering** | K-means on embeddings | 497 archetype prompts |
| 3. **Response Generation** | 81 models via OpenRouter | Model responses |
| 4. **Reward Grading** | TieredGrader (see below) | Quality scores |
| 5. **Expert Distillation** | Simulated oracle (80% optimal picks) | `expert_priors.npz` |

### TieredGrader Breakdown

The reward signals in `archetype_grid_dense_run.jsonl` were generated using:

| Grader | Samples | Percentage | When Used |
|--------|---------|------------|-----------|
| **QualityCostPredictor** (local neural model) | 55,834 | 85% | Default for most prompts |
| **GPT-4o** (via OpenRouter) | 10,010 | 15% | "Hard" prompts (math, code, logic) |

The local model was trained on HelpSteer2 + LMSYS preferences. Using 85% local grading mitigates "judge memorization" concerns.

### Generated Data Files

| File | Contents | Source |
|------|----------|--------|
| `archetype_grid_prompts.jsonl` | 497 clustered prompts | LMSYS → K-means |
| `archetype_grid_dense_run.jsonl` | Rewards for 81 models × 497 prompts | OpenRouter + TieredGrader |
| `expert_priors.npz` | LinUCB matrices (A_stack, b_stack) | Expert Distillation |

---

## Datasets

### LMSYS Chatbot Arena

**Source:** [lmsys/chatbot_arena_conversations](https://huggingface.co/datasets/lmsys/chatbot_arena_conversations)

**License:** CC-BY-4.0 / CC-BY-NC-4.0

**Usage in Priors:** We sample 50,000 prompts and cluster them into ~500 representative "archetypes" using K-means on sentence embeddings. These archetypes form the basis for our dense evaluation grid.

**Citation:**
```bibtex
@misc{zheng2023lmsyschat1m,
    title={LMSYS-Chat-1M: A Large-Scale Real-World LLM Conversation Dataset},
    author={Lianmin Zheng and Wei-Lin Chiang and Ying Sheng and Tianle Li and Siyuan Zhuang and Zhanghao Wu and Yonghao Zhuang and Zhuohan Li and Zi Lin and Eric P. Xing and Joseph E. Gonzalez and Ion Stoica and Hao Zhang},
    year={2023},
    eprint={2309.11998},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

---

### LMSYS Arena Human Preference 55K

**Source:** [lmsys/lmsys-arena-human-preference-55k](https://huggingface.co/datasets/lmsys/lmsys-arena-human-preference-55k)

**License:** CC-BY-4.0

**Usage in Quality Model:** We use human preference labels (winner/loser) to train our local quality prediction model ("Soft Grader"), which provides 85% of reward signals during prior generation.

**Citation:**
```bibtex
@misc{zheng2023judging,
    title={Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena},
    author={Lianmin Zheng and Wei-Lin Chiang and Ying Sheng and Siyuan Zhuang and Zhanghao Wu and Yonghao Zhuang and Zi Lin and Zhuohan Li and Dacheng Li and Eric P. Xing and Hao Zhang and Joseph E. Gonzalez and Ion Stoica},
    year={2023},
    eprint={2306.05685},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

---

### NVIDIA HelpSteer2

**Source:** [nvidia/HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2)

**License:** CC-BY-4.0

**Usage in Quality Model:** Primary training data for our quality prediction model, leveraging multi-dimensional annotations (helpfulness, correctness, coherence, complexity, verbosity).

**Citation:**
```bibtex
@misc{wang2024helpsteer2,
    title={HelpSteer2: Open-source dataset for training top-performing reward models},
    author={Zhilin Wang and Yi Dong and Olivier Delalleau and Jiaqi Zeng and Gerald Shen and Daniel Egert and Jimmy J. Zhang and Makesh Narsimhan Sreedhar and Oleksii Kuchaiev},
    year={2024},
    eprint={2406.08673},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

---

### Vectara Hallucination Leaderboard

**Source:** [github.com/vectara/hallucination-leaderboard](https://github.com/vectara/hallucination-leaderboard)

**License:** Apache-2.0

**Usage:** We incorporate hallucination rates into our model registry for hallucination-aware routing.

---

## APIs & Services

### OpenRouter

**Source:** [openrouter.ai](https://openrouter.ai/)

**Critical Usage in Prior Generation:**
1. **Response Generation**: All 81 model responses in `archetype_grid_dense_run.jsonl` were generated via OpenRouter's unified API
2. **Teacher Verifier**: GPT-4o grading for "hard" prompts (15% of reward signals)
3. **TTFT Measurement**: Real-world latency benchmarking

OpenRouter enabled us to evaluate 81 diverse models (OpenAI, Anthropic, Google, Meta, Amazon, Cohere, xAI, DeepSeek, etc.) through a single API, making the dense grid evaluation tractable.

---

### Artificial Analysis

**Source:** [artificialanalysis.ai](https://artificialanalysis.ai/)

**Usage:** We use their API to collect model benchmark scores (intelligence, coding, math indices), pricing, and latency metrics for 80+ models.

---

## Research Foundations

Our clustering methodology is grounded in the following research:

### LIMA: Less Is More for Alignment

**Citation:**
```bibtex
@misc{zhou2023lima,
    title={LIMA: Less Is More for Alignment},
    author={Chunting Zhou and Pengfei Liu and Puxin Xu and Srini Iyer and Jiao Sun and Yuning Mao and Xuezhe Ma and Avia Efrat and Ping Yu and Lili Yu and Susan Zhang and Gargi Ghosh and Mike Lewis and Luke Zettlemoyer and Omer Levy},
    year={2023},
    eprint={2305.11206},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

**Relevance:** Supports our use of ~500 representative prompts for prior generation (the "Superficial Alignment Hypothesis").

---

### #InsTag: Instruction Tagging

**Citation:**
```bibtex
@misc{lu2023instag,
    title={#InsTag: Instruction Tagging for Analyzing Supervised Fine-tuning of Large Language Models},
    author={Keming Lu and Hongyi Yuan and Zheng Yuan and Runji Lin and Junyang Lin and Chuanqi Tan and Chang Zhou and Jingren Zhou},
    year={2023},
    eprint={2308.07074},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

**Relevance:** Informs our understanding of instruction diversity and semantic clustering.

---

## Open-Source Libraries

BanditGPT relies on the following open-source libraries:

| Library | License | Usage |
|---------|---------|-------|
| [PyTorch](https://pytorch.org/) | BSD-3-Clause | Neural network operations |
| [Sentence-Transformers](https://www.sbert.net/) | Apache-2.0 | Prompt embedding |
| [Transformers](https://huggingface.co/transformers) | Apache-2.0 | Tokenizers, complexity classifiers |
| [NumPy](https://numpy.org/) | BSD-3-Clause | Matrix operations |
| [Pandas](https://pandas.pydata.org/) | BSD-3-Clause | Data processing |
| [Matplotlib](https://matplotlib.org/) | PSF | Experiment visualizations |

---

## Acknowledgment Statement

If you use BanditGPT in your research, please cite our paper and acknowledge the datasets above:

```bibtex
@inproceedings{banditgpt2025,
    title={Density-Based Warm-Start for Adaptive LLM Routing},
    author={...},
    booktitle={Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
    year={2025}
}
```

We thank the LMSYS team, NVIDIA, Vectara, Artificial Analysis, and the broader open-source community for making this research possible.
