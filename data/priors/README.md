# LLM Jury Priors

This folder contains **warm-start priors** for the async bandit router, enabling 63% regret reduction on Day 1.

## Quick Start: Generate Expert Priors

```bash
# Generate expert-distilled priors (recommended)
python -m llm_jury.experiment.generate_expert_priors generate --seed 42

# Verify existing priors
python -m llm_jury.experiment.generate_expert_priors verify
```

**Output:** `data/priors/expert_priors.npz` (~21 MB, 81 models, 384 dimensions)

## How Routing Works: Prompt → Model Selection

The router uses a **contextual bandit** (LinUCB) to match prompts to the best model:

### Step 1: The Mapper (Embedding)
The Sentence Transformer (`all-MiniLM-L6-v2`) converts text into a 384-dimensional vector:

```
Prompt: "Write a Python script..."  →  x = [0.05, -0.92, 0.44, ...]
```

This vector describes the prompt's position in "Meaning Space" — close to Coding, far from Poetry, etc.

### Step 2: The Memory (Learned Weights)
Each model has a weight vector θ learned from prior training:

```
θ_llama3 = A⁻¹ @ b
```

Models that performed well on coding prompts have high weights in "coding dimensions."

### Step 3: The Lookup (Dot Product)
Predicted quality is computed via dot product:

```
predicted_score = θ · x
```

- **Aligned vectors** → high score (0.95): "Llama-3 is great at Python"
- **Opposing vectors** → low score (0.20): "Llama-3 struggles with French History"

### Step 4: The Decision (UCB + Utility)
We add exploration bonus and apply cost/latency penalties:

```
UCB = (θ·x + prior) + α·√(x'A⁻¹x)
Utility = UCB - λ_cost·Cost - λ_latency·Latency
```

The model with highest **Utility** is selected.

## Prior Storage Locations

| Location | Path | Writable | Purpose |
|----------|------|----------|---------|
| **BUNDLED** | `<package>/data/priors/expert_priors.npz` | No | Expert-distilled (63% improvement) |
| **FALLBACK** | `<package>/data/priors/shippable_priors.npz` | No | Legacy shared priors |
| **USER** | `~/.llm_jury/priors/user_priors.npz` | Yes | User customizations, new models |
| **CUSTOM** | User-specified | Yes | Explicit path for special use cases |

When you update priors (e.g., add a new model), changes go to the **USER** location by default.
This preserves bundled priors while allowing personalization.

## What is `expert_priors.npz`?

`expert_priors.npz` is the **recommended** priors file (~21 MB), built using **Expert Distillation**:

| Property | Value |
|----------|-------|
| Format | Full Disjoint (A_stack, b_stack) |
| Compression | float16 (21 MB vs 86 MB float32) |
| Models | 81 |
| Dimension | 384 |
| Training | Oracle picks optimal 80% of time |
| Result | 63% regret reduction |

The library loads this by default with `prior_strength=50.0`.

## What is `shippable_priors.npz`? (Legacy)

`shippable_priors.npz` is the older "shared covariance" format (~300 KB):

- Store **one shared covariance** matrix (A_shared) in float16
- Store **per-model reward vectors** (b_model) in float16
- Smaller file but lower performance (~5% regret reduction)

The library falls back to this if `expert_priors.npz` is missing.


## What is the big `router_state_*.json` and why don’t we commit it?

A fully warmed router state produced by synthetic warmup can be **hundreds of MB** because it contains large matrices for every model.

We treat that large file as a **build artifact** and compress it into `shippable_priors.npz` for check‑in.

## How to Generate Priors

### Option A (Recommended): Expert Distillation

Expert Distillation trains priors where a "teacher oracle" picks the optimal model 80% of the time. This creates priors that encode **"the right answer"** rather than random noise.

**Why it works:**
| Prior Type | What It Encodes | Effect of Boosting |
|------------|-----------------|-------------------|
| Uniform | "Everything is average" | Bandit becomes stubborn |
| **Expert** | "Model A wins for code" | Bandit exploits correctly |

**Step 1:** Ensure you have the training data:
- `data/priors/archetype_grid_prompts.jsonl` (497 prompts)
- `data/priors/archetype_grid_dense_run.jsonl` (rewards from 81 models)

**Step 2:** Generate expert priors:
```bash
python -m llm_jury.experiment.generate_expert_priors generate \
    --seed 42 \
    --epochs 5 \
    --expert-rate 0.8
```

**Step 3:** Verify the output:
```bash
python -m llm_jury.experiment.generate_expert_priors verify
```

**Expected output:**
```
[Training Hyperparameters]
  expert_rate: 0.8
  n_epochs: 5
  seed: 42
  context_model: sentence-transformers/all-MiniLM-L6-v2

[Provenance]
  prompts_hash: 33790ea90fa66e6a
  rewards_hash: a0d3828174915a54
```

**Result:** 63.6% regret reduction with `prior_strength=50.0`

---

### Option B: Synthetic prior injection (round‑robin)

This is a cost-controlled warmup that still covers all models, but does **not** run a dense grid.

1) Produce a warmed router state (synthetic prior injection):

```bash
python -m llm_jury.async_bandit.synthetic_prior_injection \
  --dataset lmsys/chatbot_arena_conversations \
  --split train \
  --max-prompts 500 \
  --models-per-prompt 10 \
  --reward-mode logit \
  --alpha 1.5 \
  --use-teacher \
  --teacher-model openai/gpt-4o \
  --teacher-max-tokens 64 \
  --out-state data/router_state_synthetic.json
```

2) Compress the warmed state into a shippable priors bundle:

```bash
python -m llm_jury.async_bandit.compress_shippable_priors \
  --state data/router_state_synthetic.json
```

By default, this writes:

- `data/priors/shippable_priors.npz`

### Option B (recommended): Archetype Grid (clustering → dense K×M run)

This is the “dense representative data” approach:

#### Why we use ~500 clusters

In the LLM domain, the “intent space” is surprisingly consistent:

- **~50 clusters**: broad topics (Coding, Math, History, Roleplay)
- **~500 clusters**: specific tasks (Python Debugging, SQL Generation, Integral Calculus, French Translation)
- **~5,000 clusters**: specific entities (Taylor Swift Trivia, Django v4 vs v5)

For a router, you care about the **task level (~500)**. You generally don’t care about the entity level (~5,000), because a model that is good at “Django v4” is almost certainly good at “Django v5.” The router doesn’t need to distinguish them to make the right choice.

#### Citations (why this heuristic is reasonable)

You won’t find a single paper that contains the exact “50 / 500 / 5,000” table. It’s a **heuristic synthesis** grounded in established results on **data efficiency** and **instruction diversity**:

- **Task layer (~500–1,000 archetypes)**: *LIMA: Less Is More for Alignment* (Zhou et al., 2023) supports the idea that “task space” saturates with a relatively small number of diverse, curated prompts.  
  - Doc-friendly paraphrase from the paper: alignment can be achieved with as few as ~1,000 carefully curated examples because core capabilities come from pretraining.

- **Fine-grained entity/nuance layer (~6k tags)**: *#InsTag: Instruction Tagging for Analyzing Supervised Fine-tuning* (Lu et al., 2024) reports thousands of fine-grained instruction tags and shows that selecting diverse samples can outperform much larger random training sets. This supports the idea that entity-level diversity is larger but still plateaus.

- **Broad topic layer (<50 groups)**: the *Databricks Dolly 15k* taxonomy (Conover et al., 2023) illustrates that high-level user intent is low-dimensional (single-digit to low tens of broad categories).

- **Clustering validation (redundancy hurts)**: *AlpaGasus* (Chen et al., 2024) is a useful reference for the idea that K-means style selection can discard large fractions of redundant data while improving outcomes.

Suggested wording for docs/whitepaper:

> “Our clustering strategy is grounded in the Superficial Alignment Hypothesis proposed by Zhou et al. (LIMA, 2023), which suggests that task alignment saturates at approximately 1,000 diverse samples. Empirical analysis using the #InsTag method (Lu et al., 2024) further reveals that while high-level intents cluster into <50 groups, fine-grained semantic diversity plateaus around several thousand unique tags. Therefore, we set our default archetype count to 500, striking a balance between task coverage and computational efficiency.”

1) Build K representative prompts by clustering:

```bash
python -m llm_jury.async_bandit.archetype_grid \
  --dataset lmsys/chatbot_arena_conversations \
  --split train \
  --max-prompts 50000 \
  --k 500 \
  --out data/priors/archetype_grid_prompts.jsonl
```

2) Run a dense grid: all models × K prompts, grade with TieredGrader, and export priors:

```bash
python -m llm_jury.async_bandit.archetype_grid_dense_run \
  --grid data/priors/archetype_grid_prompts.jsonl \
  --use-teacher \
  --teacher-model openai/gpt-4o \
  --out data/priors/shippable_priors.npz \
  --resume
```

## How to use priors when recommending models

The recommend CLI will **auto-load** `data/priors/shippable_priors.npz` if it exists:

```bash
python -m llm_jury.async_bandit.recommend \
  --prompt 'Calculate the pH of a $10^{-8}$ M solution of HCl.' \
  --top-k 10 \
  --use-complexity-gating
```

You can also provide an explicit path:

```bash
python -m llm_jury.async_bandit.recommend \
  --prompt 'Calculate the pH of a $10^{-8}$ M solution of HCl.' \
  --top-k 10 \
  --use-complexity-gating \
  --shippable-priors data/priors/shippable_priors.npz
```

## How to add a new model ("Brain Surgery")

When a new model is released (e.g., GPT-5, DeepSeek-V3), you can install it into an existing router without retraining.

### Option 1: Clone from a similar model (recommended)

```python
from llm_jury.async_bandit import BanditRouter, PriorManager

# Load router with existing priors
router = BanditRouter.from_shippable_priors(priors_npz, model_registry)

# "Install" GPT-5 by cloning from GPT-4o
router.add_model(
    "openai/gpt-5",
    clone_from="openai/gpt-4o",
    clone_decay=0.9,  # Slightly increase uncertainty to encourage exploration
    registry_entry={"display_name": "GPT-5", "cost_per_1k_input": 0.005}
)

# Save updated state
router.save_state(Path("~/.llm_jury/priors/user_state.json").expanduser())
```

### Option 2: Cold start (for completely new models)

```python
# Add a brand-new model with no prior knowledge
router.add_model("brand-new/model-v1")
# The bandit will explore this model to learn its capabilities
```

### Option 3: Update priors via PriorManager

```python
from llm_jury.async_bandit import PriorManager

# Load user priors (falls back to bundled if not present)
manager = PriorManager.user()
priors = manager.load()

# Add new model
priors = manager.add_model(priors, "deepseek/deepseek-v3", clone_from="deepseek/deepseek-r1")

# Save to user location (~/.llm_jury/priors/)
manager.save(priors)
```

### The Lifecycle of a New Model

1. **Release Day**: OpenAI releases GPT-5
2. **Offline Step (Optional)**: Run the archetype grid on just this one model (~10 min, ~$2)
3. **Deploy**: Push updated priors in a library release
4. **Runtime**: Users' routers detect the new model key and call `add_model()`
5. **Result**: First query to GPT-5 already uses learned priors from the library or cloned from GPT-4o

## Notes

- **Quote prompts containing `$...$`** (LaTeX) in the shell, otherwise `$10` may be expanded by your shell.
- `shippable_priors.npz` is intended to provide a warm start; the router can still update online and diverge per-model.
- When you call `PriorManager.save()`, it writes to `~/.llm_jury/priors/user_priors.npz` by default, preserving bundled priors.

