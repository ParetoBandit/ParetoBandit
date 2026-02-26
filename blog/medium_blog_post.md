# Your LLM Router Is Frozen in Time. Here's One That Learns While It Runs.

**How contextual bandits discover which model works best for each prompt — with zero labels, adapting continuously from your own traffic.**

![Static routers freeze at training time. Adaptive routers learn continuously from deployment traffic.](images/hero_dynamic_routing.png)

---

You shipped an LLM routing layer. It sends hard prompts to the expensive frontier model and easy ones to the cheap model. It worked great on your evaluation dataset. Then you deployed it.

Three weeks later, things look different:
- A new open-weights model dropped that offers flagship intelligence at a fraction of the cost — but your router doesn't know it exists.
- Your users shifted from simple queries to multi-step agentic workflows.
- Your primary expensive model caught a documented reasoning regression, silently degrading on the very tasks you explicitly routed to it.

Your router doesn't know any of this happened. It's still making the same decisions it memorized from a static dataset months ago.

This is the **frozen router problem**. Static routing — whether rule-based, semantic, or supervised — learns once and then stops. But the LLM landscape doesn't stop. Models change. Prices change. User behavior shifts. The routing decision that was optimal in January is wrong by March.

What if your router could learn *continuously*, directly from *your own* outcomes, and adapt automatically?

That's what BanditGPT does.

---

## The Insight: Expensive Models Aren't Always Better

Before we look at the system, let me share the finding that motivated it.

We analyzed 750 held-out prompts from the LMSYS Chatbot Arena and measured which model actually produced the better response. The conventional wisdom — "when in doubt, send it to the expensive model" — turned out to be wrong more often than you'd expect.

On **14.1% of prompts**, the cheaper model produced a strictly *better* response than the frontier model. Not "about the same for less money." Better. 

But here's the deeper finding: models that are strictly *worse on average* can still be the *best choice for specific prompts*. In our experiments, one model that was dominated in aggregate still received 32% of optimally-routed traffic because it outperformed on specific prompt types (like certain coding queries or JSON formatting). A static router that ranks models by average benchmark scores would never route to it — and would leave quality and money on the table.

![Dynamic routing engine distributing prompts.](images/routing_concept.png)
*Instead of assuming the expensive model is always best, an intelligent router discovers which prompt types match which models.*

This is the core insight behind BanditGPT: rather than applying a hardcoded decision boundary, let the router *discover* these preference structures online, from its own traffic.

---

## The Slot Machine Intuition

Imagine walking into a casino with 80 slot machines. Each has a different (unknown) payout rate. You have a limited budget. How do you maximize your total payout?

The naive approach: pull the first machine that pays well and never try anything else. But you might be leaving money on the table. The other extreme: try every machine equally. But that wastes pulls on machines you already know are bad.

The optimal strategy is a **bandit algorithm** — a mathematically principled way to balance **exploitation** (keep pulling what works) with **exploration** (try unknowns that might work better). 

BanditGPT treats each LLM as a slot machine. But unlike a simple bandit, it uses **context** — the content of your prompt — to make different decisions for different prompt types. A coding question gets routed differently than a creative writing prompt, even if both could technically go to any model.

This is a **contextual bandit**, the same algorithmic family behind Netflix recommendations and ad placement systems that serve billions of decisions per day. The key property: it converges on the optimal policy *while serving real traffic* — no separate training phase required.

---

## The Architecture: Three Layers of Intelligent Routing

Here's how BanditGPT works end to end. The system is a three-layer pipeline, each solving a distinct problem:

![Three layers of intelligent routing: Constraint filtering, Safety layer, and Smart routing.](images/architecture_diagram.png)

### Layer 1: Constraint Filtering
Before any learning happens, hard business constraints prune the candidate set. If you set `max_cost=1.00`, models exceeding $1 per 1K tokens are instantly masked out. If a model's known quality score falls below your floor, it's excluded. This guarantees your operational requirements are met per-request.

### Layer 2: The Safety Net (Corralling)
This is the insurance policy. BanditGPT maintains **two expert bandits** simultaneously:
- A **Warmup Expert** initialized with your historical data (to start strong).
- A **Tabula Rasa Expert** that starts completely from scratch.

A meta-learner tracks which expert is performing better on your *actual* traffic and dynamically shifts weight toward the winner. If your historical data is wildly out of date, the system safely falls back to learning from scratch. You are protected from bad priors.

### Layer 3: Cost-Aware Smart Routing
For the models that pass the filters, the router calculates a score:
`Quality Estimate + Exploration Bonus − Cost Penalty`

It learns what works (Quality Estimate), tries unproven models occasionally (Exploration Bonus), and favors cheaper models when quality is tied (Cost Penalty). When feedback arrives (a quality score from user feedback or an LLM-as-a-judge), the router updates in **microseconds**. Every routing decision makes the next one smarter.

---

## Seeing It In Action

We evaluated BanditGPT across portfolios ranging from 2 to 10 models (spanning a 600× cost range) on held-out LMSYS Arena prompts. Here is what we found:

### 1. It Learns from Your Traffic Rapidly
Starting completely from scratch, the router explores the space and surpasses static baselines within roughly **400 prompts**. 

![A sleek upward-trending line chart showing performance improving and stabilizing quickly.](images/learning_curve_blog.png)
*Quality improves with each routing decision, adapting to your specific user distribution.*

For high-volume use cases (like customer support platforms or API gateways), that crossover happens within minutes of hitting production.

### 2. It Distributes Traffic Intelligently
At 10 models, no single model receives more than 17% of traffic. The router genuinely distributes prompts across the full portfolio based on context, ensuring that specialized models handle their niches without wasting money on overkill frontier models.

![A sleek horizontal bar chart showing data traffic distributed across multiple pathways or models.](images/traffic_allocation.png)
*Traffic is naturally distributed according to prompt complexity and model strengths.*

### 3. Finding the "Magic" Cost-Quality Frontier
By sweeping a single parameter (the cost penalty), BanditGPT traces a smooth cost-quality envelope. 

The most striking result: we observed **up to 91% cost reduction with only a 6% quality loss** compared to always routing to the most expensive model. The router discovers these cheap-but-high-quality operating points automatically.

![A minimalist scatter plot showing the optimal tradeoff between cost and performance.](images/cost_quality_frontier.png)
*The router finds operating points on the Pareto frontier that no single static model can reach.*

---

## How to A/B Test the Bandit in Production

If you are considering adaptive routing for production, don't take our word for it—prove it on your own traffic. 

Here is a rigorous framework for setting up an online A/B test:

1. **Define Your Baselines:** Set up a Control track (your current static routing logic) and a Treatment track (BanditGPT).
2. **Isolate the Routing Layer:** Use a traffic splitter to randomly assign user prompts to either track. Log the selected model, the inference cost, the latency, and the resulting quality metric (e.g., explicit thumbs up/down, or an asynchronous LLM-as-a-judge score).
3. **Let It Converge:** Bandit algorithms need data to explore. Do not evaluate the router after 50 prompts. Wait for the "steady state" after ~500 prompts, where the algorithm transitions from exploration to exploitation.
4. **Analyze the Frontier:** Did average quality remain tied (or improve) while costs dropped? Does the traffic distribution show a healthy spread across your model portfolio?

---

## Five Minutes to a Production Router

You can onboard new models to BanditGPT with zero downtime. No restarts, no retraining.

```python
from bandit_gpt import BanditRouter, MultiProviderClient, OpenAIClient, AnthropicClient

# Initialize with your portfolio constraints
router = BanditRouter.create(model_registry)

client = MultiProviderClient({
    "openai":    OpenAIClient(api_key="sk-..."),
    "anthropic": AnthropicClient(api_key="sk-ant-..."),
})

# Route a prompt — the router selects the best model for this context
model_id, response, log = router.route_and_call(
    "Write a Python function to parse JSON", client,
    max_cost=1.00,       # hard cost ceiling
    max_latency=2.0,     # speed limit (seconds)
)

# After observing quality (e.g. from user thumbs up), update the router
router.process_feedback(log.request_id, reward=1.0)
```

No labels, no training pipeline, no GPU. The router runs locally and learns from its own routing outcomes in microseconds.

---

## When BanditGPT Is the Right Choice

**It's a strong fit when:**
- Your prompt distribution shifts over time (customer support, coding assistants).
- You have no labeled routing data for your specific domain (enterprise RAG, vertical SaaS).
- You're routing across 3+ models at different price tiers.
- New models appear regularly and need to be integrated without manual re-tuning.

**A different approach may be better when:**
- Your traffic is extremely low volume (<50 prompts per day) where bandits take too long to converge.
- You need fully auditable, static deterministic routing decisions for strict compliance reasons.

The LLM landscape is moving fast. Hundreds of models at different capability-cost points, silent quality regressions, and deployment traffic that looks nothing like academic benchmarks. Static routing is fundamentally mismatched to this reality. 

Adaptive routing is the natural next step. 

---

**Try it today:** BanditGPT is open-source under the Apache 2.0 license.

```bash
pip install banditgpt
```

- **GitHub**: [github.com/atabernermiller/banditgpt](https://github.com/atabernermiller/banditgpt)
- **Paper**: *banditGPT: Cost-Aware Online Learning for LLM Routing via Expert Corralling* (2026)

*If you found this useful, I'd appreciate a clap or follow. If you have questions, drop a comment — I read every one.*
