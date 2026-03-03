# BanditGPT Data Directory

This directory contains all data files and preparation scripts for BanditGPT evaluation.

## Structure

```
banditgpt/data/
├── scripts/              # Data preparation and generation scripts
│   ├── data_manager.py          # Clustering & train/test splitting
│   ├── pca_manager.py           # PCA & hybrid feature extraction
│   ├── generate_cluster_priors.py   # Cluster-specific prior generation
│   ├── generate_prior_covariance.py # Prior covariance matrices
│   ├── generate_priors_meta.py      # Meta priors generation
│   ├── find_golden_prompts.py       # Golden prompt selection
│   └── find_optimal_clusters.py     # Cluster optimization analysis
│
├── Source Data (Prompts)
│   ├── lmsys_all_prompts.jsonl           # Raw LMSYS prompts (~26k)
│   ├── lmsys_all_prompts_clustered.jsonl # Clustered prompts (100 clusters)
│   ├── helpsteer_gemini_agreed.jsonl     # HelpSteer2 subset
│   └── golden_prompts.jsonl              # Golden cluster representatives
│
├── Evaluation Sets
│   ├── test_prompts.jsonl                # Test set (1,000 prompts)
│   ├── train_prompts.jsonl               # Original train set (4,000 prompts)
│   └── train_prompts_sampled_1k.jsonl    # Stratified 1K sample
│
├── Reward Data (3-4 Judge CoT)
│   ├── test_rewards_pareto.jsonl  (172M) # Test set, 36 models, 1K prompts
│   └── train_rewards_1k.jsonl     (174M) # Train set, 36 models, 1K prompts
│
├── Prior Matrices
│   ├── priors_meta_clusters.npz          # Cluster-based priors
│   ├── priors_meta_pca.npz               # PCA-based priors (32D)
│   ├── priors_meta_large.npz             # Full embedding priors
│   └── pca_32.joblib                     # Fitted PCA model (1024→32; `BAAI/bge-m3` → PCA)
│
└── Analysis Results
    ├── optimal_clusters_results.json
    └── optimal_clusters_analysis.png
```

## Data Preparation Pipeline

### 1. Clustering Prompts

```python
from banditgpt.data.scripts.data_manager import PromptClusterer

clusterer = PromptClusterer(n_clusters=100)
clusterer.cluster_file(
    input_path="lmsys_all_prompts.jsonl",
    output_path="lmsys_all_prompts_clustered.jsonl"
)
```

### 2. Train/Test Split

```python
from banditgpt.data.scripts.data_manager import DataSplitter

splitter = DataSplitter(test_size=1000, train_size=4000)
splitter.split_file(
    input_path="lmsys_all_prompts_clustered.jsonl",
test_path="test_prompts.jsonl",
    train_path="train_prompts.jsonl"
)
```

### 3. Generate Priors (PCA-based)

```python
from banditgpt.data.scripts.pca_manager import PCAManager

manager = PCAManager(n_components=32)
manager.fit_pca(prompts, pca_path="pca_32.joblib")
manager.generate_prior_covariance(
    prompts=prompts,
    output_path="priors_meta_pca.npz"
)
```

### 4. Evaluate with 3-Judge CoT

```python
from banditgpt.rejudge_cot import CoTRewardGenerator

gen = CoTRewardGenerator(max_workers=64)
gen.run(
    prompts_file="test_prompts.jsonl",
    models_file="../models.json",
    output_file="test_rewards_pareto.jsonl",
    cache_file="test_rewards_cache.jsonl"
)
```

## Data Lineage

1. **Source**: LMSYS Chat Arena prompts (~26k unique)
2. **Clustering**: MiniBatchKMeans (k=100) on SentenceTransformer embeddings (default: `BAAI/bge-m3`)
3. **Splitting**: Stratified by cluster (1K test, 4K train, rest for priors)
4. **Evaluation**: 3-4 judge panel (GPT-4o, Claude-3.5-Sonnet, Llama-405b, Gemini-2.5-Pro)
5. **Scoring**: Vote + Confidence tie-breaker → Binary reward (0/1)

## Leakage Prevention

- Prior covariance matrices use prompts **not** in train/test sets
- Strict train/test split with no overlap
- Cluster-stratified sampling maintains representativeness

## File Sizes

- **Total**: ~520MB
  - Reward data: ~346MB (172M + 174M)
  - Source prompts: ~28MB
  - Priors: ~1.5MB
  - Models: ~100KB
