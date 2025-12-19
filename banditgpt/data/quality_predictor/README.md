# Quality Predictor Model

This directory holds the trained `QualityCostPredictor` model used by the **TieredGrader**.

## Current Status

⚠️ **Model not bundled** — The pre-trained model is too large (~50-100MB) for the package.

## When You Need This

| Use Case | Need Model? |
|----------|-------------|
| Using `expert_priors.npz` for routing | ❌ No |
| Running experiments (RQ1, RQ2, RQ3) | ❌ No |
| Regenerating priors with `create_custom_judge()` | ❌ No |
| Regenerating priors with `create_tiered_judge()` | ✅ Yes |
| Using `TieredGrader` in production | ✅ Yes |
| **Exact replication of our prior generation pipeline** | ✅ Yes |

## Regenerating Priors (Without This Model)

You can regenerate priors using a **custom judge** (e.g., GPT-4o directly):

```python
from banditgpt.core import PriorManager, create_custom_judge
import openai
import os

def gpt4o_judge(prompt: str, response: str):
    """Grade using GPT-4o directly via OpenRouter."""
    client = openai.OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1"
    )
    result = client.chat.completions.create(
        model="openai/gpt-4o",
        messages=[{"role": "user", "content": f"Rate this response 0-1:\n\nPrompt: {prompt}\n\nResponse: {response}"}],
        max_tokens=10,
    )
    score = float(result.choices[0].message.content.strip())
    return score, {"source": "gpt-4o"}

judge = create_custom_judge(gpt4o_judge)

# Generate priors
manager = PriorManager.generate(cluster_k=500, dataset="lmsys/chatbot_arena_conversations")
priors = manager.build(
    judge=judge,
    models=["openai/gpt-4o", "anthropic/claude-3.5-sonnet", ...],
    call_model=my_openrouter_call_fn,  # Your function to call models
)
```

## How to Train (For Exact Pipeline Replication)

The model is trained on NVIDIA HelpSteer2 + LMSYS Arena Preferences:

```bash
# Train quality predictor (~30 min on GPU)
python -m banditgpt.core.quality_cost_predictor \
    --epochs 3 \
    --batch-size 32

# Output: banditgpt/data/quality_predictor/best_quality_predictor.pt
```

### Training Data

| Dataset | License | Purpose |
|---------|---------|---------|
| [nvidia/HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2) | CC-BY-4.0 | Primary training data |
| [lmsys/lmsys-arena-human-preference-55k](https://huggingface.co/datasets/lmsys/lmsys-arena-human-preference-55k) | CC-BY-4.0 | Augmentation |

## Architecture

The `QualityCostPredictor` is a neural model that predicts response quality:

- **Encoder**: DeBERTa-v3-small (frozen or fine-tuned)
- **Head**: MLP classifier
- **Output**: Quality probability [0, 1]

## TieredGrader Integration

The TieredGrader uses this model as the "soft grader" for 85% of prompts:

```python
from banditgpt.core.tiered_grader import TieredGrader, OpenRouterTeacherVerifier
from banditgpt.core.quality_cost_predictor import QualityCostPredictor

# Load soft grader
soft = QualityCostPredictor.load("banditgpt/data/quality_predictor/best_quality_predictor.pt")

# Optional teacher verifier for "hard" prompts (math, code, logic)
teacher = OpenRouterTeacherVerifier(model="openai/gpt-4o")

# Create tiered grader
grader = TieredGrader(soft_grader=soft, teacher_verifier=teacher)

# Grade a response
reward, meta = grader.grade(prompt="Explain quantum computing", response="...")
# reward: 0.0-1.0
# meta: {"source": "soft"} or {"source": "teacher"}
```

For "hard" prompts (math, code, logic), the grader escalates to GPT-4o via OpenRouter.

## Why TieredGrader?

During our large-scale prior generation (81 models × 500 prompts = 40,500 responses):

| Approach | API Calls | Cost |
|----------|-----------|------|
| GPT-4o for all | 40,500 | ~$40 |
| TieredGrader (85% soft) | 6,075 | ~$6 |

The soft grader handles "easy" prompts locally, saving ~85% on API costs.
