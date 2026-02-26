# Your LLM Router Is Frozen in Time. Here's One That Learns While It Runs.

**How contextual bandits discover which model works best for each prompt — with zero labels, adapting continuously from your own traffic.**

![Static routers freeze at training time. Adaptive routers learn continuously from deployment traffic.](images/hero_frozen_vs_adaptive.png)

---

You shipped an LLM routing layer. It sends hard prompts to the expensive model and easy ones to the cheap model. It worked great on your eval set. Then you deployed it.

Three weeks later:
- Anthropic released Claude Sonnet 4.6 — Opus-level intelligence at one-fifth the cost of older flagships — and your router doesn't know it exists.
- Your users shifted from single-turn code generation to multi-step agentic workflows, a trend now hitting [67% of enterprises](https://www.langchain.com/state-of-agent-engineering) in production.
- GPT-4o caught a [documented symbolic reasoning regression](https://community.openai.com/t/symbolic-reasoning-degradation-in-gpt-4o-a-dialog-based-study-q2-2025/1251113), silently degrading on the logical tasks you routed to it.

Your router doesn't know any of this happened. It's still making the same decisions it learned from a static dataset months ago.

This is the **frozen router problem**. Static routing — whether rule-based, semantic, or supervised — learns once and then stops. But the LLM landscape doesn't stop. Models change. Prices change. Your users change. The routing decision that was optimal in January is wrong by March.

What if your router could learn *continuously*, from *your own* outcomes, and adapt to all of this automatically?

That's what BanditGPT does.

---

## The Insight That Started This: Expensive Models Aren't Always Better

Before explaining the system, let me share the finding that motivated it.

We analyzed 750 held-out prompts from the LMSYS Chatbot Arena and measured which model actually produced the better response. The conventional wisdom — "when in doubt, send it to the expensive model" — turned out to be wrong more often than you'd expect:

| Outcome | Share |
|---------|-------|
| Frontier model wins | 13.1% |
| Tie (both equally good) | 72.8% |
| **Cheaper model wins** | **14.1%** |

On 14.1% of prompts, the cheaper model produced a *better* response than the frontier model. Not "about the same for less money." Better. And on nearly three-quarters of prompts, there was no quality difference at all — meaning a static router that defaults to the expensive model is paying up to 43× more for identical quality.

But here's the deeper finding — what we call **Quality Inversion**. When we scaled our evaluation to 10 models across six providers, we found that models which are strictly *worse on average* can still be the *best choice for specific prompts*. In our experiments, one model that was Pareto-dominated in aggregate (meaning another model was both cheaper and higher-quality overall) still received 32% of optimally-routed traffic because it outperformed on specific prompt types. A static router that ranks models by average benchmark scores would never route to it — and would leave quality on the table.

The deeper question is: **can we predict which prompts prefer which model?** We found that prompt embeddings carry a strong signal. The Spearman correlation between the first principal component of prompt embeddings and the reward gap between models is ρ = −0.370 (p < 0.0001) — exceeding all 100 random projection baselines. Different prompt types genuinely prefer different models, and that preference is learnable from the prompt itself.

This is the core insight behind BanditGPT: rather than applying a static decision boundary, let the router *discover* this preference structure online, from its own traffic, and adapt as it shifts.

---

## The Slot Machine Intuition

Imagine walking into a casino with 80 slot machines. Each has a different (unknown) payout rate. You have a limited budget. How do you maximize your total payout?

The naive approach: pull the first machine that pays well and never try anything else. But you might be leaving money on the table — a machine you never tried could pay twice as much.

The other extreme: try every machine equally. But that wastes pulls on machines you already know are bad.

The optimal strategy is a **bandit algorithm** — a mathematically principled way to balance **exploitation** (keep pulling what works) with **exploration** (try unknowns that might work better). After enough pulls, you converge on the best machine with high probability.

BanditGPT treats each LLM as a slot machine. But unlike a simple bandit, it uses **context** — the content of your prompt — to make different decisions for different prompt types. A coding question gets routed differently than a creative writing prompt, even if both could technically go to any model.

![Each LLM is a slot machine with unknown payouts. The bandit algorithm learns which machine pays best for each type of prompt — balancing exploitation of known winners with exploration of uncertain alternatives.](images/slot_machine_bandit_analogy.png)

This is a **contextual bandit**, the same algorithmic family behind Netflix recommendations and ad placement systems that serve billions of decisions per day. The key property: it converges on the optimal policy *while serving real traffic* — no separate training phase, no labeled dataset, no offline evaluation.

---

## The Architecture: Three Layers of Intelligent Routing

Here's how BanditGPT works end to end. The system is a three-layer pipeline, each layer solving a distinct problem:

![BanditGPT architecture: a user prompt flows through feature extraction, constraint filtering, a Corralling meta-learner coordinating two experts, and cost-aware UCB selection. Online reward feedback closes the learning loop.](images/architecture_figure2.png)

### Layer 1: Constraint Filtering

Before any learning happens, hard business constraints prune the candidate set. If you set `max_cost=1.00`, models exceeding $1 per 1K tokens are masked out. If you need sub-200ms latency, slow models are removed. If a model's known quality score falls below your floor, it's excluded.

This is deliberately simple — no machine learning here, just guardrails. The output is a filtered candidate set that respects your operational requirements.

### Layer 2: Corralling Meta-Learner

This is the safety layer. BanditGPT can maintain **two expert bandits** simultaneously:

- A **Warmup Expert** initialized with offline priors (feature correlations you've generated from your own historical data or any available preference dataset)
- A **Tabula Rasa Expert** that starts from scratch with no prior knowledge

A meta-learner (based on the Exp4 algorithm) tracks which expert is performing better on your actual traffic and dynamically shifts weight toward the winner. The mixing floor γ = 0.05 guarantees neither expert can ever be fully silenced — so if the dominant expert degrades, the system can recover.

**Why does this matter?** If the warm-start priors happen to match your deployment well, the Warmup Expert dominates and you get fast convergence. If your traffic is radically different from the prior training distribution, the Tabula Rasa Expert takes over and learns from scratch. Either way, you're covered.

One of our more surprising experimental findings: **it doesn't matter where your priors come from.** We tested priors from a completely different distribution (80,000 battle outcomes from a separate dataset, with a Population Stability Index of 0.763 — indicating severe feature-space divergence from deployment prompts) against same-distribution priors. Once online learning was enabled, the performance difference between the two was not statistically significant (p > 0.18). Online learning rapidly overwrites the prior. This means you can bootstrap from *any* available offline data — even cross-domain sources — without measurable penalty.

In experiments, Corralling delivered a **+1.3 percentage point steady-state advantage** (p < 0.0001, Cohen's d = 0.98) and **7–13% lower regret** compared to single-expert strategies.

![Corralling maintains two experts — a Warmup Expert (with offline priors) and a Tabula Rasa Expert (learning from scratch) — and a meta-learner that dynamically shifts weight toward whichever is performing better.](images/corralling_safety_mechanism.png)

### Layer 3: Cost-Aware Hybrid LinUCB

Each expert uses a **Hybrid LinUCB** bandit to score candidate models. For each model, the score is:

```
score = quality_estimate + exploration_bonus − cost_penalty
```

Three terms, each doing something different:

- **Quality estimate** — A linear prediction of expected reward given the prompt features. The "Hybrid" part means it decomposes into a **family-shared** component (what all GPT models have in common) and a **model-specific residual** (what GPT-4o does differently from GPT-4-mini). This family sharing accelerates learning when you have model families: updates to one GPT model inform its siblings.

- **Exploration bonus** — Large when the router hasn't seen many prompts like this one routed to this model. It's the mathematical mechanism for "try things you're uncertain about." The bonus shrinks automatically as the router gathers more data.

- **Cost penalty** — Discourages expensive models when cheaper ones perform equally well. The parameter λ controls the quality-cost tradeoff — set it to 0 for pure quality optimization, or higher to aggressively cut costs.

When feedback arrives (a quality score from 0 to 1), the router updates via rank-one matrix operations. This takes **microseconds**. No retraining, no gradient descent, no GPU required. Every routing decision makes the next one smarter.

---

## What This Achieves in Practice

We evaluated BanditGPT across portfolio sizes of 2, 5, and 10 models on held-out LMSYS Arena prompts, averaged over 20 independent trials.

### It Learns from Your Traffic, with Zero Labels

Starting from cold-start (no priors), the router delivers usable quality from the first request and improves steadily as it sees more traffic. With warmup priors generated from historical data, cold-start quality reaches 0.839–0.903 immediately. Within roughly **400 prompts**, the router surpasses static baselines — all without a single human label.

![BanditGPT's learning curve: quality improves with each routing decision, surpassing static baselines after ~400 prompts. The shaded region shows the 95% confidence interval across 20 trials.](images/learning_curve.png)

For high-volume use cases (support platforms, API gateways), that crossover happens within minutes of production traffic.

### We Show That It Can Scale from 2 Models to 10+ (with Caveats)

The real payoff comes with larger portfolios. We evaluated up to K=10 across six providers (Meta, Mistral, Google, Anthropic, OpenAI, DeepSeek) spanning a 600× cost range:

| Metric | K=2 | K=5 | K=10 |
|--------|-----|-----|------|
| Peak quality | 0.903 | 0.911 | 0.898 |
| Gap closure to oracle | 61.5% | 74.9% | 69.3% |
| Cost savings (moderate λ) | — | up to 91% | up to 98.7% |
| Max traffic to any single model | — | 32% | 17% |

At K=10, no single model receives more than 17% of traffic — the router is genuinely distributing prompts across the full portfolio based on context, not just funneling everything to one model.

**A note on portfolio size.** Nothing in the library prevents you from registering more than 10 models, but we've only validated performance up to K=10. There are principled reasons to be thoughtful about portfolio size: each additional model increases the exploration budget the bandit must spend before converging, dilutes per-model observations (slowing learning), and grows the covariance matrices that back the LinUCB updates. In our experiments, the exploration overhead was already visible at K=10 — peak quality dipped slightly relative to K=5, reflecting the cost of keeping more arms calibrated. If you're running larger portfolios, we'd recommend testing convergence on your own traffic and considering whether constraint filtering (Layer 1) can prune the candidate set to a manageable size before the bandit layer sees it.

![Traffic allocation across 10 models: the router distributes prompts based on per-prompt context, with no single model dominating. Different prompt types route to different models.](images/traffic_allocation_k10.png)

### It Finds Cost Savings You Can't Find Manually

By sweeping a single parameter (the cost penalty λ), BanditGPT traces a smooth cost-quality envelope. At K=10, quality degrades only 1% across a 41× cost range because the router discovers cheap-but-high-quality models automatically.

The most striking result: **up to 91% cost reduction with only 6% quality loss** compared to always routing to the most expensive model. This isn't a theoretical bound — it's measured on held-out prompts the router never saw during learning.

![BanditGPT's Pareto frontier: each point represents a different cost penalty setting. Green triangles show individual models' cost and quality when used exclusively. The router finds operating points that no single model can reach.](images/pareto_frontier.png)

### It Survives Distribution Shift Automatically

One of the most practical findings from our research: once online learning is enabled, **it doesn't matter where your priors came from.** Even with severe feature-space divergence between prior training data and deployment traffic, online learning overwrites the prior within a few hundred interactions. The practical hierarchy is clear:

1. **Enable online learning** — this is the dominant effect (+7.0 percentage points over frozen priors)
2. **Use Corralling** — adds a further +1.3 pp steady-state advantage
3. **Bootstrap from any available offline data** — prior source has no significant effect (p > 0.18)

### It Onboards New Models with Zero Downtime

When a new model drops, register it with a single call:

```python
router.add_model("google/gemini-2.0-flash", metadata={
    "capabilities": ["coding", "reasoning"],
    "speed": "fast",
})
```

No restart. No retraining. The router starts exploring the newcomer immediately and learns where it fits in the portfolio. Family parameter sharing means a new GPT variant inherits knowledge from its siblings — at K=5, the hybrid approach achieves **0.963 ± 0.006** versus 0.927 ± 0.028 for independent per-model learning, with **4.7× narrower confidence intervals**.

---

## Who Should Use This? Seven Real-World Use Cases

The theory is nice, but where does adaptive routing actually matter? Here are seven concrete deployment scenarios, ranked by how immediately the value shows up.

### Strong Fit — the value is immediate

**1. AI-Powered Customer Support Platforms**

This is probably the single strongest use case. Think Intercom, Zendesk, Freshdesk-style products routing millions of tickets per month. "Reset my password" and "your API returns a 500 when I POST with a nested JSON payload" are radically different prompts — and the traffic mix shifts after product launches, pricing changes, or seasonal spikes.

Today most of these companies either use one expensive model for everything (wasteful) or hand-write heuristic rules that break when traffic shifts. BanditGPT's cost constraints plus online adaptation are a direct answer. The roughly 65/35 easy-to-hard ratio in typical support traffic is exactly the distribution where adaptive routing shines — cheap models handle the bulk, expensive models handle the tail, and the boundary learns itself.

**2. LLM API Gateways and Proxy Services**

Companies like LiteLLM, Portkey, Helicone, and Martian proxy millions of requests, and their customers explicitly ask for cost optimization. The key insight: each customer's traffic is different, so a one-size-fits-all routing policy leaves money on the table.

BanditGPT can be instantiated per-customer, each instance learning its own optimal routing independently. The `MultiProviderClient` and `add_model()` APIs are already structured for this pattern — gateway operators add new models to the fleet constantly, and the bandit starts exploring them immediately without per-customer reconfiguration.

**3. Coding Assistant Backends**

Cursor, Cody, Continue, Windsurf-class products face a fundamental routing tension: autocomplete needs sub-200ms latency (local or fast cloud model), while complex refactoring needs frontier reasoning. The boundary between "easy" and "hard" is fuzzy, prompt-dependent, and shifts as models improve.

This is exactly the `max_latency` + contextual routing pattern. Hot model onboarding matters here because new models drop monthly and the product team needs to integrate them without re-tuning routing heuristics. The bandit handles the boundary discovery automatically.

**4. RAG-Based Enterprise Knowledge Systems**

Platforms like Glean, Guru, and internal corporate chatbots handle a bimodal distribution. Simple factual lookups ("What's our PTO policy?") versus complex multi-hop reasoning ("Summarize how our Q3 revenue guidance changed across the last three board decks"). Over-routing everything to GPT-4 wastes budget; under-routing complex queries to a cheap model produces hallucinations.

The prompt embedding naturally captures this complexity signal, and the cost penalty steers simple queries to cheap models without manual threshold tuning.

### Good Fit — real value, but you need a bit more volume

**5. Content Generation Platforms**

Jasper, Copy.ai, and marketing automation tools have a natural routing split: short social media copy goes to a cheap model; long-form SEO blog posts or brand-voice-sensitive content needs an expensive model. The router learns from editorial feedback (thumbs up/down or quality scores) what "good enough" means for each content type.

The catch: this only works well if you have enough volume to learn. A team generating 50 pieces per week would need a few weeks to converge; one generating 500 per day converges in hours.

**6. Multi-Tenant AI-as-a-Service**

Vertical SaaS companies wrapping LLMs for their SMB customers face a scaling problem: each tenant has different traffic patterns and budget constraints, and you can't manually tune routing per tenant. The pattern is straightforward — instantiate a `BanditRouter` per tenant, each learns independently. `add_model()` handles fleet-wide model additions without per-tenant reconfiguration.

This is operationally compelling but requires the platform to be large enough that per-tenant learning converges within a useful timeframe.

**7. Agentic Workflows and Tool-Calling Orchestrators**

Different steps in an agent pipeline have wildly different difficulty — high-level planning versus JSON formatting versus code generation versus summarization. Using GPT-4 for every step is wasteful. The router can be called per-step, with the step's prompt as context, and it learns which models handle which step types best.

This is a natural direction — the architecture supports it today — but we flag it as not yet validated experimentally. If you're building in this space, we'd love to see what you find.

---

## Try It Yourself: The Hands-On Tutorial

The fastest way to get a feel for BanditGPT is the **hands-on tutorial** — a single Python script that demonstrates the full learning loop using a synthetic reward oracle. No API keys needed, runs in under a minute.

```bash
pip install banditgpt
python examples/hands_on_tutorial.py
```

Here's what the tutorial walks through:

### Step 1: Define a 5-Model Portfolio

```python
model_registry = {
    "meta-llama/llama-3.1-8b":    {"cost_per_1k_input": 0.05},
    "mistralai/mixtral-8x7b":     {"cost_per_1k_input": 0.24},
    "meta-llama/llama-3.1-70b":   {"cost_per_1k_input": 0.52},
    "openai/gpt-4o":              {"cost_per_1k_input": 2.50},
    "anthropic/claude-sonnet-4":   {"cost_per_1k_input": 3.00},
}
```

A 600× cost range. The router's job is to figure out when the $0.05 model is good enough and when you genuinely need the $3.00 model — without being told.

### Step 2: The Adversarial Twist

The tutorial includes an "adversarial" prompt category where **expensive models perform worse** than cheap ones. This isn't hypothetical — it mirrors the 14.1% of real Arena prompts where the cheaper model wins. A static "always upgrade when uncertain" policy fails on these prompts. The bandit discovers them.

### Step 3: Watch the Learning Dynamics

The tutorial generates a four-panel visualization showing:

1. **Learning curve** — Rolling average reward converging toward the oracle, with the exploration phase highlighted
2. **Model selection frequency** — How the router transitions from exploration (trying everything) to exploitation (concentrating on winners)
3. **Per-category model preference** — Different prompt categories route to different models (coding ≠ chat ≠ reasoning)
4. **Reward vs. cost scatter** — Visual proof that "expensive ≠ always better"

### Step 4: Hot Model Onboarding

After 1,000 prompts, the tutorial registers a brand-new model to the running router:

```python
router.add_model("google/gemini-2.0-flash", metadata={
    "capabilities": ["coding", "reasoning"],
    "speed": "fast",
})
```

No restart. No retraining. The router starts exploring the newcomer immediately and learns where it fits in the portfolio. The tutorial tracks adoption over the next 300 prompts.

### Step 5: Suggested Experiments

The tutorial ends with seven experiments you can run by changing a single parameter — cold start vs. warm start, aggressive vs. conservative exploration, different portfolio sizes, different difficulty distributions. Each one reveals a different aspect of bandit behavior.

---

## Five Minutes to a Production Router

If you want to skip the tutorial and go straight to real models:

```bash
pip install banditgpt[full]
```

```python
from bandit_gpt import BanditRouter, MultiProviderClient, OpenAIClient, AnthropicClient

router = BanditRouter.create(model_registry)

client = MultiProviderClient({
    "openai":    OpenAIClient(api_key="sk-..."),
    "anthropic": AnthropicClient(api_key="sk-ant-..."),
})

# Route a prompt — the router selects the best model for this context
model_id, response, log = router.route_and_call(
    "Write a Python function to parse JSON", client
)

# After observing quality, update the router (microsecond operation)
router.process_feedback(log.request_id, reward=0.95)
```

No labels, no training pipeline, no GPU. The router runs locally and learns from its own routing outcomes.

### Business Constraints

Control cost, latency, and quality floors per request:

```python
model, log, mode = router.route(
    "Solve this calculus integral",
    max_cost=1.00,       # hard cost ceiling
    max_latency=2.0,     # speed limit (seconds)
    min_quality=70,       # quality floor
    cascade_rate=0.3,    # verify 30% of predictions
)
```

### Exploration Rate

Controls how much the router tries unproven models versus exploiting known winners:

| Setting | Behavior | Use Case |
|---------|----------|----------|
| `static` | Zero exploration | Production / fintech (no surprises) |
| `safe` | Minimal exploration | Default for most deployments |
| `balanced` | Standard bandit | General-purpose learning |
| `aggressive` | Heavy exploration | Day-1 calibration or shadow mode |

### Bring Your Own Embeddings

BanditGPT doesn't lock you into a specific embedding model. You can use the default sentence-transformer pipeline, plug in your own encoder (OpenAI, Cohere, a local ONNX model), or pass pre-computed embedding vectors directly. The library handles PCA compression, bias terms, and all the LinUCB math regardless of where your embeddings come from.

```python
from bandit_gpt import BanditRouter, FeatureService

# Use OpenAI embeddings instead of the default encoder
fs = FeatureService(
    custom_encoder=openai_embed,  # any str → np.ndarray callable
    embedding_dim=1536,
)

router = BanditRouter.create(model_registry, feature_service=fs)
```

This also means you can install just `pip install banditgpt` (no PyTorch, no Hugging Face) if you bring your own embeddings.

### Warmup Priors

BanditGPT works out of the box with no priors — the router learns from scratch using standard LinUCB cold-start. If you want faster convergence, the library provides utilities to generate warmup priors from your own historical data:

```python
from bandit_gpt import generate_warmup_priors

# Each entry: {"prompt": str, "rewards": {"model_id": float, ...}}
priors = generate_warmup_priors(
    rewards_data,
    encoder_model="sentence-transformers/all-MiniLM-L6-v2",
    output_path="my_priors.joblib",
)

router = BanditRouter.create(model_registry, priors="my_priors.joblib")
```

The priors encode learned feature correlations — which prompt types tend to prefer which models — and give the router a head start. In our experiments, warm-started routers achieved 0.839–0.903 quality from the very first request. But as we noted above, the source of offline priors has no significant effect once online learning kicks in, so even approximate priors from a related domain are useful.

---

## What We Got Wrong (Honest Limitations)

In the spirit of transparent research, here's what didn't work — and what the system can't do:

**Volume requirement.** The router needs traffic to learn from. If you're processing 10 prompts per day, convergence will be slow. The crossover point — where online learning starts outperforming a static baseline — is roughly 400 prompts for a 2-model portfolio. For high-volume use cases, this happens in minutes; for low-volume use cases, it may take weeks. At fewer than ~50 prompts per day, hand-tuned rules may be more practical.

**Semantic transfer is a null result.** We hypothesized that new models could be bootstrapped from semantically similar existing models (e.g., initialize GPT-5 from GPT-4's learned parameters). After extensive experiments, we found no statistically significant improvement over cold-start initialization (p > 0.07 across all configurations). We report this as a negative result. Cold-start model integration remains an open problem.

**Corralling has overhead.** The meta-learning safety net trades peak performance for worst-case protection. When priors are accurate, a single-expert warmup strategy outperforms Corralling. The insurance only pays off when prior quality is genuinely uncertain — so if you know your traffic matches the prior distribution well, you can skip it.

**Linear reward assumption.** LinUCB assumes reward is linear in the prompt features. While the PCA features capture meaningful signal (ρ = −0.370, p < 0.0001), highly nonlinear preference structures might require neural bandit architectures. This is listed as future work.

**Stationary experts.** Each expert bandit assumes stationary rewards. Non-stationarity (model quality drifting over time) is handled at the meta-level via Corralling, but gradual drift within a single expert requires meta-level reweighting, not expert-level adaptation.

**Auditability.** A bandit router's decisions are harder to audit than a hand-written rule set. If regulatory requirements demand you explain *why* each prompt went to a specific model, a simpler rule-based router may be more appropriate. BanditGPT logs all routing decisions (including scores per model), but the reasoning is a learned linear function rather than an interpretable rule.

---

## When BanditGPT Is the Right Choice (And When It Isn't)

**BanditGPT is a strong fit when:**
- Your prompt distribution shifts over time (customer support, coding assistants)
- You have no labeled routing data for your domain (enterprise RAG, vertical SaaS)
- You're routing across 3+ models at different price tiers (API gateways, multi-tenant platforms)
- New models appear regularly and need to be integrated without manual re-tuning
- You want your router to improve continuously without manual intervention

**A different approach may be better when:**
- Your traffic is stable and well-understood — if prompt types don't change, a static router's simplicity is a feature, not a bug
- You're routing between exactly two models on well-known benchmarks — the two-model case is where static routing is most competitive
- You process fewer than ~50 prompts per day — the bandit needs volume to learn; at very low volume, hand-tuned rules may be more practical
- You need fully auditable, deterministic routing decisions for compliance — a rule-based router gives you interpretable decisions by default

**The crossover point: ~400 prompts.** After that, online learning starts outperforming static baselines at moderate-to-high cost budgets. For high-volume use cases (support platforms, API gateways), that crossover happens within minutes of production traffic.

---

## The Bigger Picture

The LLM landscape is moving fast. Hundreds of models at different capability-cost points, new releases weekly, silent quality regressions, and deployment traffic that looks nothing like academic benchmarks. Static routing — whether rule-based, supervised, or semantic — is fundamentally mismatched to this reality.

Adaptive routing, where the system learns from its own outcomes and improves continuously, is the natural next step. BanditGPT is one implementation of this idea, using contextual bandits (a well-studied framework from the recommendation systems and online advertising literature) adapted for the LLM routing problem.

The key principles translate beyond any specific tool:

1. **Learn from deployment, not just benchmarks.** Your traffic is the ground truth.
2. **Adapt continuously, not episodically.** Weekly retraining cycles leave blind spots. Models regress, prices change, user behavior shifts — your router should keep up.
3. **Explore deliberately.** Trying new models on a small fraction of traffic is not waste — it's information acquisition with a quantifiable return.
4. **Fail safely.** Meta-learning provides insurance against bad priors. You don't have to be right on Day 1 to converge on the right answer by Day 30.

---

## Try It

BanditGPT is open-source under the Apache 2.0 license.

```bash
pip install banditgpt
```

- **GitHub**: [github.com/atabernermiller/banditgpt](https://github.com/atabernermiller/banditgpt)
- **Hands-on tutorial**: `examples/hands_on_tutorial.py` — full learning loop, no API keys, runs in under a minute
- **Paper**: *Density-Based Warm-Start for Adaptive LLM Routing* (2025)
- **API Reference**: [docs/API_REFERENCE.md](https://github.com/atabernermiller/banditgpt/blob/main/docs/API_REFERENCE.md) — complete reference for all public classes, methods, and configuration

The router learns from your first prompt. Every routing decision makes the next one smarter.

---

*If you found this useful, I'd appreciate a clap or follow. If you have questions, or want to share how you're using adaptive routing in production, drop a comment — I read every one.*

---

**Tags**: `#MachineLearning` `#LLM` `#AI` `#ContextualBandits` `#LLMRouting` `#OpenSource` `#MLOps` `#NLP`
