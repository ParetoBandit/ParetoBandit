# API Reference

Complete reference for the ParetoBandit public API.

```python
import pareto_bandit
print(pareto_bandit.__version__)  # "0.1.0"
```

---

## `BanditRouter`

The primary entry point for adaptive LLM routing. Maintains a contextual bandit over registered models and learns from routing outcomes.

### `BanditRouter.create()`

Factory method for creating a fully initialised router.

```python
@classmethod
def create(
    cls,
    model_registry: dict[str, Any] | None = None,
    context_model: str = "all-MiniLM-L6-v2",
    priors: str = "warmup",
    prior_n_effective: float = 1163.9,
    **kwargs,
) -> BanditRouter
```

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_registry` | `dict[str, Any] \| None` | `None` | Model configurations keyed by model ID. Each entry may include `input_cost_per_m`, `output_cost_per_m`, `time_to_first_token_seconds`, and capability metadata. When `None`, loads the shipped K=3 paper portfolio from `config/models.json`. |
| `context_model` | `str` | `"all-MiniLM-L6-v2"` | SentenceTransformer model for prompt embedding. Ignored when `feature_service` is provided. Custom ST models require matching PCA and warmup artifacts. |
| `priors` | `str` | `"warmup"` | Prior initialisation strategy: `"warmup"` (default — loads shipped K=3 offline priors; models not in the prior file receive heuristic initialisation), `"none"` (clean cold-start), or a path to a `.joblib` file generated via `generate_warmup_priors()`. |
| `prior_n_effective` | `float` | `1163.9` | Effective sample count attributed to loaded priors. Controls how strongly offline priors are trusted — higher values trust priors more (slower adaptation); lower values trust them less (faster override by online evidence). Default from `BEST_K3_HPARAMS` (derived from T_adapt-constrained Pareto knee-point selection). |
| `**kwargs` | | | Forwarded to `__init__`. Notable kwargs below. |

**Notable `**kwargs`**

| Kwarg | Type | Default | Description |
|-------|------|---------|-------------|
| `feature_service` | `FeatureService \| None` | `None` | Injected feature service for custom embedding pipelines. When provided, `context_model` is ignored. See [`FeatureService`](#featureservice). |
| `alpha` | `float` | `0.01` | Exploration coefficient for UCB. Default from `BEST_K3_HPARAMS`. |
| `init_lambda` | `float` | `1.0` | Regularisation for cold-start (A₀ = λI). |
| `forgetting_factor` | `float` | `0.997` | Temporal decay for non-stationarity (1.0 = stationary). Default from `BEST_K3_HPARAMS`. |
| `bandit_seed` | `int \| None` | `None` | Seed for tie-breaking in `select_arm()` and Thompson Sampling. Pass explicit seed for reproducible simulations. |
| `cost_penalty` | `float` | `0.3` | λ_c for UCB cost penalty (paper Eq. 4). `0.0` = quality-only, `0.3` = moderate cost awareness, `0.5+` = aggressive cost preference. |
| `budget_pacer` | `BudgetPacer \| None` | `None` | Optional budget pacer for online budget pacing (Primal-Dual CBwK). See [`BudgetPacer`](#budgetpacer). |
| `context_store` | `ContextStore \| None` | `None` | Storage backend for context vectors. Defaults to `EphemeralContextStore`. Pass `SqliteContextStore()` for delayed feedback / RLHF workflows. |
| `config` | `RouterConfig \| None` | `None` | Router configuration object. See [`RouterConfig`](#routerconfig). |
| `verbose_routing` | `bool` | `False` | Enable detailed breakdown logs for each routing decision. |
| `state_path` | `str \| Path \| None` | `None` | Path to load previously saved bandit state (extracted before `__init__`, not passed through). |
| `warmup_path` | `str \| Path \| None` | `None` | *Deprecated.* Pass the path directly via `priors` instead. |

**Returns**: Fully initialised `BanditRouter` instance.

**Example: Default usage (warm-start with shipped priors)**

```python
from pareto_bandit import BanditRouter

registry = {
    "openai/gpt-4o": {
        "model_id": "openai/gpt-4o",
        "input_cost_per_m": 2.50,
        "output_cost_per_m": 10.00,
        "time_to_first_token_seconds": 0.5,
    },
    "mistralai/mixtral-8x7b": {
        "model_id": "mistralai/mixtral-8x7b",
        "input_cost_per_m": 0.24,
        "output_cost_per_m": 0.24,
        "time_to_first_token_seconds": 0.3,
    },
}

# Default: warm-start with shipped priors (heuristic init for unknown models)
router = BanditRouter.create(registry)

# Cold-start: learns from scratch (recommended for custom portfolios)
router = BanditRouter.create(registry, priors="none")

# Or load custom priors generated from your own reward data
router = BanditRouter.create(registry, priors="path/to/my_priors.joblib")
```

**Example: Custom reward scale**

```python
from pareto_bandit import BanditRouter, RouterConfig

# Use preference-pair rewards in [-1, 1] instead of default [0, 1]
router = BanditRouter.create(
    registry,
    priors="none",
    config=RouterConfig(reward_min=-1.0, reward_max=1.0),
)
```

**Example: Custom encoder (no sentence-transformers needed)**

```python
from pareto_bandit import BanditRouter, FeatureService

fs = FeatureService(
    custom_encoder=my_encoder_fn,   # Callable[[str], np.ndarray]
    embedding_dim=768,              # must match your encoder's output
)

router = BanditRouter.create(registry, feature_service=fs, priors="none")
```

**Example: Pre-computed vectors**

```python
from pareto_bandit import BanditRouter, FeatureService

fs = FeatureService.for_precomputed(dimension=33)
router = BanditRouter.create(registry, feature_service=fs, priors="none")

# Pass numpy arrays instead of strings to route()
model_id, log = router.route(my_precomputed_vector)
```

---

### `BanditRouter.route()`

Route a prompt to the best model.

```python
def route(
    self,
    prompt: str | np.ndarray,
    *,
    max_cost: float | None = None,
    max_latency: float | None = None,
    quality_floor: dict[str, float | None] | None = None,
    input_tokens: int | None = None,
    output_tokens: int = 600,
) -> tuple[str, RoutingLog]
```

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | `str \| np.ndarray` | — | Input text or pre-computed feature vector. |
| `max_cost` | `float \| None` | `None` | Hard cost ceiling ($/1k tokens). Models exceeding this are filtered. |
| `max_latency` | `float \| None` | `None` | Hard latency ceiling (seconds). |
| `quality_floor` | `dict \| None` | `None` | Minimum quality scores per metric (e.g. `{"hle": 0.7}`). |
| `input_tokens` | `int \| None` | `None` | Input token count (auto-estimated from prompt if `None`). |
| `output_tokens` | `int` | `600` | Expected output tokens for cost estimation. |

**Returns**: `(model_id, routing_log)` — The selected model ID and a `RoutingLog` with decision metadata.

**Raises**:
- `NoEligibleModelsError` — No models pass the hard constraints.
- `ValueError` — Empty or whitespace-only prompt.
- `TypeError` — Prompt is neither `str` nor `np.ndarray`.

**Example: Basic routing**

```python
model_id, log = router.route("Write a Python function to parse JSON")

print(f"Selected: {model_id}")          # e.g. "mistralai/mixtral-8x7b"
print(f"Request ID: {log.request_id}")  # UUID for feedback
print(f"Cost: ${log.cost_usd:.6f}")     # Estimated cost
```

**Example: Route with constraints**

```python
model_id, log = router.route(
    "Explain the Riemann hypothesis",
    max_cost=5.0,         # Filter out models costing > $5/1k tokens
    output_tokens=200,    # Expected response length
)
```

---

### `BanditRouter.route_and_call()`

Route a prompt **and** call the selected model in a single step.

```python
def route_and_call(
    self,
    prompt: str | np.ndarray,
    client: LLMClient,
    *,
    messages: list[dict] | None = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
    **route_kwargs,
) -> tuple[str, str, RoutingLog]
```

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | `str \| np.ndarray` | — | Input text (used for routing features and, by default, as the user message). |
| `client` | `LLMClient` | — | Any object satisfying the `LLMClient` protocol (see [Providers](#providers)). |
| `messages` | `list[dict] \| None` | `None` | Chat messages to send.  When `None` and `prompt` is a string, a single `{"role": "user", "content": prompt}` message is used. |
| `max_tokens` | `int` | `512` | Passed to `client.complete()`. |
| `temperature` | `float` | `0.7` | Passed to `client.complete()`. |
| `**route_kwargs` | | | Forwarded to `route()` (e.g. `max_cost`, `max_latency`). |

**Returns**: `(model_id, response_text, routing_log)`

**Example**

```python
from pareto_bandit import BanditRouter, OpenRouterClient

router = BanditRouter.create(registry)
client = OpenRouterClient(api_key="sk-or-...")

model_id, response, log = router.route_and_call(
    "Explain quantum entanglement simply",
    client,
    max_cost=5.0,
)
print(f"{model_id}: {response[:80]}...")
router.process_feedback(log.request_id, reward=0.9)
```

---

### `BanditRouter.process_feedback()`

Process feedback for a previous routing decision.

```python
def process_feedback(self, request_id: str, reward: float) -> None
```

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `request_id` | `str` | The `RoutingLog.request_id` returned by `route()`. |
| `reward` | `float` | Observed quality signal. Values outside [`reward_min`, `reward_max`] (configurable via `RouterConfig`, default [0, 1]) are clamped with a warning. Typical sources: LLM-as-judge score, user thumbs-up/down (0 or 1), or normalised task metric. |

**Behaviour**:
- Looks up the stored context vector for the request.
- Rejects non-finite rewards (NaN/Inf) with a warning and no-op.
- Clamps reward to [`RouterConfig.reward_min`, `RouterConfig.reward_max`] (default [0, 1]).
- Updates the LinUCB policy.
- If a `BudgetPacer` is configured, observes the request cost.
- Supports **delayed feedback**: if the in-memory log has been evicted, falls back to the `SqliteContextStore`. Feedback can arrive hours or days later as long as the context has not expired (default TTL: 7 days).
- **No-op** if `request_id` is unknown (evicted from both stores). A warning is logged.

**Raises**: None. Designed to never crash the router.

**Example: Standard route-feedback loop**

```python
model_id, log = router.route("Explain quantum entanglement")
response = call_llm(model_id, "Explain quantum entanglement")
reward = evaluate_quality(response)  # returns 0.0–1.0
router.process_feedback(log.request_id, reward=reward)
```

**Example: Online learning loop**

```python
prompts = ["Write a haiku about AI", "Solve x^2 - 4 = 0", "Debug this Python code"]

for prompt in prompts:
    model_id, log = router.route(prompt)
    response = call_llm(model_id, prompt)
    reward = evaluate_quality(response)
    router.process_feedback(log.request_id, reward=reward)
```

---

### `BanditRouter.update()`

Direct bandit update (bypass the `process_feedback` flow).

```python
def update(
    self,
    model_id: str,
    context: str | np.ndarray,
    reward: float,
    weight: float = 1.0,
    advance_time: bool = True,
) -> None
```

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_id` | `str` | — | Model that was selected. |
| `context` | `str \| np.ndarray` | — | Prompt string or pre-computed feature vector. |
| `reward` | `float` | — | Quality signal in [`reward_min`, `reward_max`] (clamped). |
| `weight` | `float` | `1.0` | Importance weight for this observation. |
| `advance_time` | `bool` | `True` | Whether to increment the global time step `t`. Default `True` is correct for offline/batch learning. |

Use this for batch/offline learning where you already have `(model, context, reward)` triples. For the standard online workflow, prefer `route()` followed by `process_feedback()`.

**Raises**:
- `ValueError` — Feature vector has wrong dimension.
- `KeyError` — `model_id` is not registered.

**Example: Ingest historical data**

```python
historical_data = [
    ("openai/gpt-4o", "Write a Python quicksort", 0.95),
    ("mistralai/mixtral-8x7b", "Tell me a joke", 0.72),
    ("anthropic/claude-3.5-sonnet", "Explain relativity", 0.88),
]

for model, prompt, reward in historical_data:
    router.update(model, prompt, reward)
```

---

### `BanditRouter.get_probabilities()`

Estimate the probability each model is the best choice for a given context.

```python
def get_probabilities(
    self,
    context: str | np.ndarray,
    model_ids: list[str] | None = None,
) -> dict[str, float]
```

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `context` | `str \| np.ndarray` | — | Prompt string or pre-computed feature vector. |
| `model_ids` | `list[str] \| None` | `None` | Subset of models to evaluate. `None` means all registered models. |

**Returns**: Dictionary mapping model IDs to quality-best probabilities (sum to 1.0). These reflect the learned reward model and do **not** incorporate cost penalties — useful for dashboards, explainability, and posterior calibration.

**Example**

```python
probs = router.get_probabilities("Write a SQL query to find active users")

for model, prob in sorted(probs.items(), key=lambda x: -x[1]):
    print(f"  {model}: {prob:.1%}")
```

---

### `BanditRouter.explain_decision()`

Feature contribution analysis: decompose a model's **mean reward prediction** into per-feature contributions.

```python
def explain_decision(
    self,
    model_id: str,
    context_vector: np.ndarray,
    threshold: float = 0.01,
) -> dict[str, float]
```

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_id` | `str` | — | Model to explain. |
| `context_vector` | `np.ndarray` | — | Feature vector (from `FeatureService.extract_features()`). |
| `threshold` | `float` | `0.01` | Minimum absolute contribution to include. |

**Returns**: Dictionary mapping feature names (e.g., `"PCA_0"`, `"bias"`) to their contribution to the model's mean prediction `θ^T x`.

**Scope**: Decomposes the learned quality estimate only. Does **not** include the UCB exploration bonus or cost penalties. See `explain_selection()` for a convenience wrapper.

**Example**

```python
model_id, log = router.route("Write SQL to get active users")

explanation = router.explain_decision(model_id, log.context_vector)

print(f"Why {model_id} was selected:")
for feature, contribution in sorted(explanation.items(), key=lambda x: abs(x[1]), reverse=True):
    print(f"  {feature}: {contribution:+.4f}")
```

---

### `BanditRouter.explain_selection()`

Decompose the mean reward prediction for the top-k models.

```python
def explain_selection(
    self,
    prompt: str,
    top_k: int = 3,
    threshold: float = 0.01,
) -> dict[str, dict[str, float]]
```

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | `str` | — | Input prompt text. |
| `top_k` | `int` | `3` | Number of top models to explain. |
| `threshold` | `float` | `0.01` | Minimum absolute contribution to include. |

**Returns**: Dictionary mapping model_id to per-feature contribution dict, ranked by `θ^T x`.

**Example**

```python
explanations = router.explain_selection("Debug this Python code", top_k=2)
for model, features in explanations.items():
    print(f"{model}: {features}")
```

---

### `BanditRouter.register_model()`

Add a new model with progressive registration.

```python
def register_model(
    self,
    model_id: str,
    speed: SpeedProfile = "balanced",
    cost_usd: float | None = None,
    latency_s: float | None = None,
    blended_cost_per_m: float | None = None,
    input_cost_per_m: float | None = None,
    output_cost_per_m: float | None = None,
    initial_weights: dict[str, float] | None = None,
    strict_kwargs: bool | None = None,
    **kwargs,
) -> None
```

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_id` | `str` | — | Unique model identifier (e.g., `"openai/gpt-4o"`). |
| `speed` | `str` | `"balanced"` | T-shirt size: `"fast"`, `"balanced"`, or `"slow"`. Affects initial bias. |
| `input_cost_per_m` | `float \| None` | `None` | Input token cost in $/M tokens. Must be paired with `output_cost_per_m`. |
| `output_cost_per_m` | `float \| None` | `None` | Output token cost in $/M tokens. Must be paired with `input_cost_per_m`. |
| `blended_cost_per_m` | `float \| None` | `None` | Weighted average cost in $/M tokens. Used if input/output not provided. |
| `cost_usd` | `float \| None` | `None` | *Deprecated — prefer* `input_cost_per_m` + `output_cost_per_m`. Legacy input cost in $/M tokens. |
| `latency_s` | `float \| None` | `None` | Time-to-first-token in seconds. |
| `initial_weights` | `dict[str, float] \| None` | `None` | Power-user override for explicit theta vector entries. |
| `strict_kwargs` | `bool \| None` | `None` | Override for unknown-kwarg validation. If `None`, uses `RouterConfig.registration_strict_kwargs` (default `True`). |
| `**kwargs` | | | Accepted for backward compatibility (e.g. `capabilities`). Unknown keys raise `TypeError` in strict mode. |

**Cost specification** (in order of precedence):

1. `input_cost_per_m` + `output_cost_per_m` — exact per-token costs; blended cost is their average.
2. `blended_cost_per_m` — single blended rate.
3. `cost_usd` (legacy) — treated as input cost; output estimated at `cost_usd × 3`.
4. None — falls back to `RegistrationConfig.default_cost_per_1m`.

**Raises**:
- `ValueError` — Only one of `input_cost_per_m` / `output_cost_per_m` provided.
- `TypeError` — Unknown kwargs in strict mode.

**Example**

```python
# Exact pricing (preferred)
router.register_model(
    "google/gemini-2.0-flash",
    speed="fast",
    input_cost_per_m=0.10,
    output_cost_per_m=0.40,
)

# Single blended rate
router.register_model("local/llama-3-8b", speed="fast", blended_cost_per_m=0.2)

# Mystery model: no information (pessimistic defaults)
router.register_model("mystery/new-model")
```

---

### `BanditRouter.update_model_pricing()`

Update pricing for a model and recompute derived cost fields.

```python
def update_model_pricing(self, model_id: str, **pricing_fields: float) -> None
```

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_id` | `str` | Model identifier that must already exist in the registry. |
| `**pricing_fields` | `float` | Pricing fields to set (e.g., `input_cost_per_m=0.10`, `output_cost_per_m=0.10`). |

Use this when simulating or reacting to mid-stream price changes (e.g., a provider price drop).

**Raises**: `KeyError` — `model_id` is not in the registry.

**Example**

```python
# Simulate a Gemini price drop mid-experiment
router.update_model_pricing(
    "google/gemini-2.5-pro",
    input_cost_per_m=0.10,
    output_cost_per_m=0.10,
)
```

---

### `BanditRouter.exploit()`

Context manager for greedy exploitation (frozen policy evaluation).

```python
@contextmanager
def exploit(self) -> Generator[None, None, None]
```

Within this block, `route()` selects `argmax(θ^T x)` with no UCB exploration bonus (alpha=0). Thread-safe via `threading.local()`.

**Example**

```python
with router.exploit():
    model, log = router.route("Classify this document")
    # Pure exploitation — no exploration
```

---

### `BanditRouter.save_state()` / `BanditRouter.load_state()`

Persist and restore learned bandit parameters.

```python
def save_state(self, path: Path | str) -> None
def load_state(self, path: Path | str) -> None
```

**Known limitation**: Only the base `DisjointLinUCBPolicy` matrices (A, b) are persisted. Budget pacer state and meta-weights are not saved; they reset on reload.

**Example**

```python
router.save_state("checkpoints/router_state.npz")
router.load_state("checkpoints/router_state.npz")
```

---

### `BanditRouter.reference_model`

Property. Dynamically identifies the flagship model (highest `initial_quality` score) in the current registry.

```python
@property
def reference_model(self) -> dict[str, Any]
```

**Returns**: Dictionary containing flagship model metadata with keys `id`, `initial_quality`, `input_cost_per_m`, `output_cost_per_m`, etc.

**Example**

```python
ref = router.reference_model
print(f"Flagship: {ref['id']} (Quality: {ref.get('initial_quality', 0):.3f})")
```

---

## `RoutingLog`

Dataclass returned by `route()` containing decision metadata.

| Field | Type | Description |
|-------|------|-------------|
| `request_id` | `str` | Unique ID for this routing decision (use with `process_feedback()`). |
| `timestamp_s` | `float` | Unix timestamp of the routing decision. |
| `prompt` | `str` | The input prompt text. |
| `selected_model` | `str` | Model ID that was selected. |
| `predicted_utility` | `float` | LinUCB composite score for the selected model. |
| `cost_usd` | `float` | Estimated cost in USD. |
| `latency_s` | `float` | Estimated latency in seconds. |
| `context_vector` | `np.ndarray \| None` | Cached feature vector (used internally by `process_feedback()`). |
| `expected_reward` | `float` | Expected reward (default 0.0). |
| `total_priority_weight` | `float` | Sum of quality/cost/latency weights (default 1.0). |
| `pacer_lambda_t` | `float \| None` | Budget pacer dual variable at routing time (populated when `BudgetPacer` is active). |
| `pacer_cost_ema` | `float \| None` | Budget pacer cost EMA at routing time (populated when `BudgetPacer` is active). |

**Example: Inspecting the routing log**

```python
model_id, log = router.route("Solve x^2 + 2x + 1 = 0")

print(f"Model: {log.selected_model}")
print(f"Request ID: {log.request_id}")
print(f"Utility: {log.predicted_utility:.4f}")
print(f"Cost: ${log.cost_usd:.8f}")
print(f"Latency: {log.latency_s:.3f}s")
print(f"Context vector shape: {log.context_vector.shape}")  # (26,) with default PCA
```

---

## `FeatureService`

Handles prompt embedding and PCA compression independently from bandit math. Supports three embedding paths:

1. **Default SentenceTransformer** — `FeatureService()` (requires `pip install paretobandit[embeddings]`)
2. **Custom encoder callable** — `FeatureService(custom_encoder=fn, embedding_dim=N)` (no extra dependencies)
3. **Pre-computed vectors** — `FeatureService.for_precomputed(dim)` (no extra dependencies)

### Bundled PCA Artifact

A pre-trained PCA artifact (`pca_25.joblib`) ships inside the package and is loaded by default when no explicit `pca_path` is provided and no `custom_encoder` is set. It was trained on ~46K LMSYS Arena prompts (independent of ParetoBandit's experimental splits) using the default encoder (`all-MiniLM-L6-v2`). The 25 components compress 384-dimensional embeddings down to 26-dimensional feature vectors (25 PCA + 1 bias term).

To replace it with a domain-specific PCA, pass `pca_path` to the constructor or use `train_pca()` to generate one from your own prompts (see [Calibration API](#calibration-api)).

### Constructor

```python
FeatureService(
    encoder_model: str = "all-MiniLM-L6-v2",
    pca_path: Path | str | None = None,
    pca_components: int | None = None,
    target_variance: float = 0.60,
    whiten_pca: bool = True,
    allow_jit_training: bool = True,
    calibration_file: Path | str | None = None,
    custom_encoder: Callable[[str], np.ndarray] | None = None,
    embedding_dim: int | None = None,
    use_text_features: bool = False,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `encoder_model` | `str` | `"all-MiniLM-L6-v2"` | SentenceTransformer model name. Ignored when `custom_encoder` is provided. Custom ST models require explicit `pca_path`. |
| `pca_path` | `Path \| str \| None` | `None` | Path to a PCA artifact (`.joblib`). When `None` and using the default encoder, loads the bundled `pca_25.joblib`. When `None` and `custom_encoder` is set, **no PCA** is applied — raw embeddings are used directly. |
| `pca_components` | `int \| None` | `None` | Auto-detected from PCA file if not specified. |
| `target_variance` | `float` | `0.60` | Minimum explained variance threshold for PCA. If JIT-trained PCA falls below this, falls back to raw embeddings. |
| `whiten_pca` | `bool` | `True` | Scale PCA coordinates by `1/sqrt(explained_variance)` so each component has roughly unit variance. Makes the LinUCB isotropic prior (A₀=λI) better matched to feature scale. |
| `allow_jit_training` | `bool` | `True` | Allow JIT PCA retraining if the artifact is missing or corrupted. Set `False` in strict production to crash-fast instead of falling back to synthetic-data PCA. Automatically `False` when `custom_encoder` is provided. |
| `calibration_file` | `Path \| str \| None` | `None` | Line-delimited text file of real prompts for domain-specific JIT PCA training. Only used if the artifact is missing and `allow_jit_training=True`. |
| `custom_encoder` | `Callable[[str], np.ndarray] \| None` | `None` | A callable that maps a prompt string to a 1-D numpy embedding vector. When provided, `sentence-transformers` is **not** required. |
| `embedding_dim` | `int \| None` | `None` | Dimensionality of vectors returned by `custom_encoder`. **Required** when `custom_encoder` is provided; ignored otherwise. |
| `use_text_features` | `bool` | `False` | Append four lightweight regex-based text features (logical operator count, constraint keyword count, average word length, instruction×vague density) between PCA and bias. Increases the feature vector by 4 dimensions. Requires string prompts — incompatible with `for_precomputed()`. |

**Raises**:
- `ValueError` — Custom SentenceTransformer encoder without explicit `pca_path`.
- `ValueError` — `custom_encoder` provided without `embedding_dim`.
- `ValueError` — `pca_components` set without `pca_path` when using `custom_encoder`.

**Example: Default usage (bundled PCA)**

```python
from pareto_bandit import FeatureService

# Uses the default encoder and the bundled pca_25.joblib
# Requires: pip install paretobandit[embeddings]
fs = FeatureService()

vector = fs.extract_features("Explain the Pythagorean theorem")
print(f"Shape: {vector.shape}")    # (26,) — 25 PCA + 1 bias
print(f"Bias term: {vector[-1]}")  # 1.0
```

**Example: Custom PCA for your domain**

```python
from pareto_bandit import FeatureService

fs = FeatureService(pca_path="my_domain_pca.joblib")
```

**Example: Custom encoder (e.g., OpenAI embeddings) — no PCA**

```python
import numpy as np
from openai import OpenAI
from pareto_bandit import FeatureService, BanditRouter

client = OpenAI()

def openai_embed(prompt: str) -> np.ndarray:
    resp = client.embeddings.create(model="text-embedding-3-small", input=prompt)
    return np.array(resp.data[0].embedding)

# No sentence-transformers required, no PCA applied
fs = FeatureService(custom_encoder=openai_embed, embedding_dim=1536)

router = BanditRouter.create(model_registry=registry, feature_service=fs, priors="none")
model_id, log = router.route("Explain quantum computing")
```

The resulting feature vector has shape `(1537,)` — 1536 raw embedding dimensions plus the bias term.

**Example: Custom encoder with PCA compression**

```python
import joblib
from sklearn.decomposition import PCA
from pareto_bandit import FeatureService

embeddings = np.array([openai_embed(p) for p in representative_prompts])
pca = PCA(n_components=32).fit(embeddings)
joblib.dump(pca, "openai_pca_32.joblib")

fs = FeatureService(
    custom_encoder=openai_embed,
    embedding_dim=1536,
    pca_path="openai_pca_32.joblib",
)
```

**Example: With text features**

```python
from pareto_bandit import FeatureService

fs = FeatureService(use_text_features=True)
vector = fs.extract_features("If x > 5 and y < 3, find the minimum")
print(f"Shape: {vector.shape}")  # (30,) — 25 PCA + 4 text + 1 bias
```

### `FeatureService.for_precomputed()`

Create a lightweight service for pre-computed embedding vectors (no model loading, no PCA).

```python
@classmethod
def for_precomputed(cls, dimension: int) -> FeatureService
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `dimension` | `int` | Total feature-vector length (your embedding dimensions + 1 bias term). |

Passing a string prompt to a pre-computed service raises `RuntimeError`. Only `np.ndarray` inputs are accepted. Text features are disabled.

**Example: Testing without model downloads**

```python
import numpy as np
from pareto_bandit import FeatureService

fs = FeatureService.for_precomputed(dimension=26)

vector = np.random.randn(26)
vector[-1] = 1.0  # bias term
result = fs.extract_features(vector)  # passes through directly
```

### `FeatureService.extract_features()`

Convert a prompt to a feature vector.

```python
def extract_features(self, prompt: str | np.ndarray) -> np.ndarray
```

**Returns**: Feature vector of shape `(dimension,)`. The last element is a bias term (always 1.0). Default with bundled PCA: 26 dimensions (25 PCA + 1 bias). With text features: 30 dimensions (25 PCA + 4 text + 1 bias). With a custom encoder and no PCA: `embedding_dim + 1`.

**Raises**:
- `ValueError` — Empty prompt or dimension mismatch for pre-computed vectors.
- `TypeError` — `prompt` is neither `str` nor `np.ndarray`.

### `FeatureService.extract_features_batch()`

Batch feature extraction (more efficient than sequential calls).

```python
def extract_features_batch(self, prompts: list[str]) -> np.ndarray
```

**Returns**: Array of shape `(n_prompts, dimension)`.

**Raises**:
- `ValueError` — Empty prompt in the list.
- `TypeError` — Non-string in the list.

**Example**

```python
prompts = [
    "Write a Python quicksort",
    "Explain the Riemann hypothesis",
    "Tell me a joke about programmers",
]
vectors = fs.extract_features_batch(prompts)
print(f"Batch shape: {vectors.shape}")  # (3, 26) with default PCA
```

### `FeatureService.encode_prompt()`

Encode a single prompt to a raw embedding vector (before PCA).

```python
def encode_prompt(self, prompt: str) -> np.ndarray
```

Dispatches to the custom encoder when available, otherwise uses SentenceTransformer. Returns an L2-normalized 1-D array.

### `FeatureService.encode_prompts_batch()`

Encode multiple prompts to a 2-D embedding matrix (before PCA).

```python
def encode_prompts_batch(self, prompts: list[str]) -> np.ndarray
```

**Returns**: `np.ndarray` of shape `(len(prompts), embedding_dim)`.

### `FeatureService.get_feature_names()`

Human-readable feature names for interpretability.

```python
def get_feature_names(self) -> list[str]
```

**Returns**: List like `["PCA_0", "PCA_1", ..., "PCA_24", "bias"]`. With text features enabled: `["PCA_0", ..., "PCA_24", "n_logical_ops", "n_constraints", "avg_word_len", "instruction_x_vague_density", "bias"]`.

**Example**

```python
names = fs.get_feature_names()
print(names[:3])   # ['PCA_0', 'PCA_1', 'PCA_2']
print(names[-1])   # 'bias'
print(len(names))  # 26
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `dimension` | `int` | Total feature dimension (PCA [+ text features] + 1 bias). Default 26. |
| `bias_index` | `int` | Index of the bias term (always -1). |
| `using_pca` | `bool` | Whether PCA compression is active. |
| `has_encoder` | `bool` | Whether this service can encode string prompts (custom or SentenceTransformer). `False` for `for_precomputed()` services. |

---

## `RouterConfig`

Dataclass for all router hyperparameters. Pass to `BanditRouter.__init__()` or let defaults apply.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_log_size` | `int` | `10_000` | Ring buffer size for in-memory routing logs. |
| `init_lambda` | `float` | `1.0` | Regularisation for cold-start (A₀ = λI). |
| `stability_check_interval` | `int` | `1000` | Check numerical stability every N updates. |
| `stability_threshold` | `float` | `1e6` | Max trace(A\_inv) before reset. |
| `market_cost_floor` | `float` | `0.0001` | Cost normalisation floor ($/1k tokens). |
| `market_cost_ceiling` | `float` | `0.10` | Cost normalisation ceiling ($/1k tokens). Covers expensive reasoning models (o1-pro, Opus). Increase if your most expensive model exceeds this. |
| `reward_min` | `float` | `0.0` | Lower bound for reward clamping. Default 0.0 matches [0, 1] convention. |
| `reward_max` | `float` | `1.0` | Upper bound for reward clamping. Set to a different range (e.g., `-1.0` / `1.0`) for preference-pair rewards. |
| `default_missing_cost_per_m` | `float` | `10.00` | Pessimistic cost fallback for missing metadata. |
| `default_missing_latency` | `float` | `2.0` | Pessimistic latency fallback. |
| `registration` | `RegistrationConfig` | `RegistrationConfig()` | Sub-config for progressive registration priors. |
| `registration_strict_kwargs` | `bool` | `True` | Validate unknown kwargs in `register_model()`. Set `False` for backward-compatible integrations that pass extra keys. |

**`RegistrationConfig`** (nested dataclass):

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `fast_bias` | `float` | `0.0` | Bias for fast models (neutral). |
| `slow_bias` | `float` | `0.05` | Bias for slow/expensive models (positive = latent quality belief). |
| `balanced_bias` | `float` | `0.0` | Bias for balanced models (neutral). |
| `default_cost_per_1m` | `float` | `10.00` | Pessimistic fallback cost. |
| `default_latency_s` | `float` | `2.0` | Pessimistic fallback latency. |

**Example**

```python
from pareto_bandit import RouterConfig

config = RouterConfig(
    max_log_size=5_000,              # Smaller memory footprint
    init_lambda=2.0,                 # Stronger regularisation
    stability_check_interval=500,    # More frequent checks
    reward_min=-1.0, reward_max=1.0, # Preference-pair scale
)

router = BanditRouter(model_registry=registry, config=config)
```

---

## `ExplorationRate`

Named presets for the exploration parameter (alpha).

| Preset | Alpha | Use Case |
|--------|-------|----------|
| `ExplorationRate.STATIC` | `0.0` | Pure exploitation — production/fintech. |
| `ExplorationRate.SAFE` | `0.1` | Default. Minimal exploration. |
| `ExplorationRate.BALANCED` | `1.0` | Standard bandit behaviour. |
| `ExplorationRate.AGGRESSIVE` | `2.0` | Day-1 calibration or shadow mode. |

**Example**

```python
from pareto_bandit import ExplorationRate

router = BanditRouter.create(registry, alpha=ExplorationRate.SAFE)

# Resolve by name
alpha = ExplorationRate.get("balanced")  # returns 1.0
```

---

## `BudgetPacer`

Online budget pacer using a smoothed dual-ascent controller inspired by Primal-Dual CBwK (Agrawal & Devanur 2014). Enforces a per-request average cost target without requiring a known time horizon.

### `PacingMode`

| Mode | Description |
|------|-------------|
| `PacingMode.HARD` | Adaptive `max_cost` ceiling fed through constraint filtering. Safety mechanism. |
| `PacingMode.SOFT` | Dynamic per-request cost penalties in the UCB scoring function. Theoretically grounded. |
| `PacingMode.ADAPTIVE` | Both mechanisms active. Hard ceiling as safety net, soft penalty as optimizer. |

### Constructor

```python
BudgetPacer(
    target_avg_spend_usd: float,
    mode: PacingMode = PacingMode.ADAPTIVE,
    lr: float = 0.05,
    ema_alpha: float = 0.05,
    hard_ceiling_multiplier: float = 1.0,
    lambda_max: float = 5.0,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_avg_spend_usd` | `float` | — | Desired average cost per request in USD. Must be positive. |
| `mode` | `PacingMode` | `ADAPTIVE` | Enforcement mechanism. |
| `lr` | `float` | `0.05` | Learning rate for the dual variable (lambda). Operates on target-normalized costs. |
| `ema_alpha` | `float` | `0.05` | Smoothing factor for the cost EMA. Half-life of ~14 observations. |
| `hard_ceiling_multiplier` | `float` | `1.0` | Controls how aggressively the hard ceiling tightens. |
| `lambda_max` | `float` | `5.0` | Upper bound on the dual variable to prevent excluding all models. |

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `lambda_t` | `float` | Current Lagrange multiplier. Zero = no cost pressure. |
| `cost_ema` | `float` | Exponential moving average of per-request costs (USD). |
| `n_observations` | `int` | Total cost observations processed. |

### Methods

| Method | Description |
|--------|-------------|
| `observe(actual_cost_usd)` | Record one realized cost and update pacing state (EMA + dual update). |
| `get_cost_ceiling_per_1k(max_model_cost_per_1k)` | Return a cost-per-1k ceiling (HARD mode), or `None` if no constraint is active. |
| `get_extra_cost_penalties(model_costs)` | Compute per-model cost penalties scaled by the dual variable (SOFT mode). |
| `reset()` | Re-initialize pacing state, preserving configuration. |
| `get_state()` | Return a serializable snapshot for logging/checkpointing. |

**Example: Budget-constrained routing**

```python
from pareto_bandit import BanditRouter
from pareto_bandit.budget_pacer import BudgetPacer, PacingMode

pacer = BudgetPacer(
    target_avg_spend_usd=0.001,     # $0.001 per request target
    mode=PacingMode.ADAPTIVE,
)

router = BanditRouter.create(registry, budget_pacer=pacer)

# The pacer automatically:
# - Adjusts cost constraints in route()
# - Updates its state in process_feedback()
model_id, log = router.route("Explain relativity")
router.process_feedback(log.request_id, reward=0.85)

# Monitor pacing state
print(pacer.get_state())
```

---

## Calibration API

ParetoBandit ships with a pre-trained PCA artifact for the default encoder. The functions below let you **replace** it with a domain-specific projection or build one for a custom sentence transformer.

### `train_pca()`

Train a PCA artifact to replace the bundled default or to match a custom sentence transformer.

The bundled `pca_25.joblib` was trained on ~46K LMSYS Arena prompts (broad English: coding, math, reasoning, creative, chat). If your production traffic differs substantially from this distribution, training a domain-specific PCA on your own prompts will better capture the axes of variation that matter for your routing decisions.

```python
def train_pca(
    prompts: list[str],
    encoder_model: str,
    n_components: int = 32,
    output_path: Path | str | None = None,
    batch_size: int = 64,
) -> PCA
```

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompts` | `list[str]` | — | Representative corpus from your domain (200+ recommended for stable components). |
| `encoder_model` | `str` | — | HuggingFace SentenceTransformer model name. Must match the encoder used at routing time. |
| `n_components` | `int` | `32` | Number of PCA components to retain. Higher = richer signal but slower O(d²) bandit updates. |
| `output_path` | `Path \| str \| None` | `None` | Persist the PCA via joblib. |
| `batch_size` | `int` | `64` | Encoder batch size. |

**Returns**: Fitted `sklearn.decomposition.PCA` object.

**Raises**:
- `ValueError` — Empty prompts or fewer prompts than `n_components`.

**Example: Replace the bundled PCA with a domain-specific one**

```python
from pareto_bandit import train_pca, FeatureService, BanditRouter

prompts = [
    "Write a Python function to parse CSV files",
    "Explain the theory of relativity in simple terms",
    "Debug this SQL query that returns duplicate rows",
    # ... 200+ prompts recommended
]

pca = train_pca(
    prompts,
    encoder_model="all-MiniLM-L6-v2",
    n_components=25,
    output_path="my_pca.joblib",
)
print(f"Explained variance: {sum(pca.explained_variance_ratio_):.1%}")

fs = FeatureService(pca_path="my_pca.joblib")
router = BanditRouter.create(feature_service=fs)
```

---

### `generate_warmup_priors()`

Generate warmup priors (A, b matrices) for LinUCB from labelled data.

```python
def generate_warmup_priors(
    rewards_data: list[dict],
    encoder_model: str,
    pca: PCA | Path | str,
    plasticity: float = 0.1,
    whiten_pca: bool = True,
    output_path: Path | str | None = None,
    batch_size: int = 64,
    precomputed_raw_embeddings: dict[str, np.ndarray] | None = None,
    use_text_features: bool = False,
) -> dict
```

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rewards_data` | `list[dict]` | — | Each entry: `{"prompt": str, "rewards": {"model_id": float, ...}}`. |
| `encoder_model` | `str` | — | Must match the model used for `train_pca()`. |
| `pca` | `PCA \| Path \| str` | — | Fitted PCA object or path to joblib file. |
| `plasticity` | `float` | `0.1` | Scaling factor. Lower = softer priors, faster to override. |
| `whiten_pca` | `bool` | `True` | Whiten PCA coordinates by `1/sqrt(explained_variance)`. Must match the `FeatureService.whiten_pca` setting used at runtime. |
| `output_path` | `Path \| str \| None` | `None` | Persist the priors via joblib. |
| `batch_size` | `int` | `64` | Encoder batch size. |
| `precomputed_raw_embeddings` | `dict[str, np.ndarray] \| None` | `None` | Cache mapping `sha256(prompt)` to raw ST vectors (pre-PCA). The encoder is only loaded as fallback for cache misses. |
| `use_text_features` | `bool` | `False` | Append text features between PCA and bias. Produces priors compatible with `FeatureService(use_text_features=True)`. |

**Returns**: Dict with keys `A`, `b`, `models`, `n_prompts`, `context_dim`, `pca_components`, `pca_whitened`, `plasticity`, `reward_source`.

**Raises**:
- `ValueError` — Empty or malformed `rewards_data`.

**Example**

```python
from pareto_bandit import generate_warmup_priors

rewards_data = [
    {
        "prompt": "Write a Python quicksort",
        "rewards": {"openai/gpt-4o": 0.95, "mistralai/mixtral-8x7b": 0.70},
    },
    {
        "prompt": "Tell me a joke",
        "rewards": {"openai/gpt-4o": 0.80, "mistralai/mixtral-8x7b": 0.85},
    },
]

priors = generate_warmup_priors(
    rewards_data,
    encoder_model="all-MiniLM-L6-v2",
    pca="my_pca.joblib",
    plasticity=0.1,
    output_path="my_priors.joblib",
)
print(f"Built priors for {len(priors['models'])} models from {priors['n_prompts']} prompts")
```

---

### `calibrate_priors()`

Post-warmup calibration to prevent scale explosion in predictions.

```python
def calibrate_priors(
    bandit: DisjointLinUCBPolicy,
    target_max_pred: float = 0.9,
    calibration_contexts: list[np.ndarray] | None = None,
) -> None
```

Mutates the bandit's `b` vectors in place so that maximum predicted reward does not exceed `target_max_pred`. Called automatically by `BanditRouter.create()`.

---

## Storage

### `ContextStore` (ABC)

Abstract base class for context vector storage.

| Method | Signature | Description |
|--------|-----------|-------------|
| `save_context` | `(request_id: str, context: np.ndarray, model_id: str) -> None` | Store a context vector. |
| `get_context` | `(request_id: str) -> tuple[np.ndarray \| None, str \| None]` | Retrieve context and model ID. Returns `(None, None)` if expired/missing. |
| `prune` | `() -> int` | Remove expired entries. Returns count deleted. |

### `EphemeralContextStore`

RAM-based store with bounded deque. For testing and low-latency deployments where feedback arrives within seconds.

```python
EphemeralContextStore(max_size: int = 10_000)
```

### `SqliteContextStore`

Production-ready store using SQLite (WAL mode). Supports delayed feedback (RLHF) with configurable TTL.

```python
SqliteContextStore(
    db_path: str | Path = "data/router_context.db",
    ttl_seconds: int = 604800,  # 7 days
)
```

| Method | Description |
|--------|-------------|
| `stats() -> dict` | Returns `total_contexts`, `oldest_timestamp`, `newest_timestamp`, `db_size_mb`, `ttl_days`. |
| `prune(force=False) -> int` | Remove expired entries (or all if `force=True`). |

**Lazy initialisation**: The database file is not created until the first `save_context()` or `get_context()` call.

**Example: Custom context store**

```python
from pareto_bandit import BanditRouter
from pareto_bandit.storage import SqliteContextStore

store = SqliteContextStore(
    db_path="/var/app/bandit_router.db",
    ttl_seconds=86400 * 30,  # 30-day retention
)
router = BanditRouter.create(registry, context_store=store)

stats = store.stats()
print(f"Contexts: {stats['total_contexts']}, Size: {stats['db_size_mb']} MB")

pruned = store.prune()
print(f"Pruned {pruned} expired entries")
```

**Example: Ephemeral store for testing**

```python
from pareto_bandit.storage import EphemeralContextStore

store = EphemeralContextStore(max_size=100)
router = BanditRouter.create(registry, context_store=store)
```

---

## Providers

ParetoBandit ships with a `LLMClient` protocol and thin adapters for popular LLM providers. The router itself never calls an LLM — it only selects a model ID. The providers module bridges the gap, letting you route **and** call in one step via `route_and_call()`.

### `LLMClient` (Protocol)

```python
from pareto_bandit import LLMClient

class LLMClient(Protocol):
    def complete(
        self,
        model_id: str,
        messages: list[dict],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs,
    ) -> str: ...
```

Any object with a matching `complete` method satisfies this protocol — no subclassing required.

### Built-in Adapters

| Adapter | Provider | Install Extra | API Key Env Var |
|---------|----------|---------------|-----------------|
| `OpenRouterClient` | [OpenRouter](https://openrouter.ai) | `pip install paretobandit[openrouter]` | `OPENROUTER_API_KEY` |
| `OpenAIClient` | OpenAI (and any compatible endpoint) | `pip install paretobandit[openai]` | `OPENAI_API_KEY` |
| `AnthropicClient` | Anthropic | `pip install paretobandit[anthropic]` | `ANTHROPIC_API_KEY` |
| `GeminiClient` | Google Gemini | `pip install paretobandit[gemini]` | `GEMINI_API_KEY` |
| `OllamaClient` | Local Ollama | `pip install paretobandit[ollama]` | *(none)* |

**OpenAI-compatible providers** (DeepSeek, Grok, Together, etc.) work via `OpenAIClient` with a custom `base_url`:

```python
from pareto_bandit import OpenAIClient

client = OpenAIClient(api_key="sk-...", base_url="https://api.deepseek.com")
```

### Single-Provider Example

When all your models are reachable through one provider, pass a single client:

```python
from pareto_bandit import BanditRouter, OpenRouterClient

router = BanditRouter.create(registry)
client = OpenRouterClient(api_key="sk-or-...")

model_id, response, log = router.route_and_call("Solve x^2 = 4", client)
router.process_feedback(log.request_id, reward=0.9)
```

### Multi-Provider Example

When your model portfolio spans multiple providers, use `MultiProviderClient` to wire each provider prefix to the right client:

```python
from pareto_bandit import (
    BanditRouter, MultiProviderClient,
    OpenAIClient, AnthropicClient, OllamaClient,
)

registry = {
    "openai/gpt-4o": {
        "model_id": "openai/gpt-4o",
        "input_cost_per_m": 2.50,
        "output_cost_per_m": 10.00,
    },
    "anthropic/claude-3.5-sonnet": {
        "model_id": "anthropic/claude-3.5-sonnet",
        "input_cost_per_m": 3.00,
        "output_cost_per_m": 15.00,
    },
    "meta-llama/llama-3-8b": {
        "model_id": "meta-llama/llama-3-8b",
        "input_cost_per_m": 0.0,
        "output_cost_per_m": 0.0,
    },
}

router = BanditRouter.create(registry, priors="none")

client = MultiProviderClient({
    "openai":     OpenAIClient(api_key="sk-..."),
    "anthropic":  AnthropicClient(api_key="sk-ant-..."),
    "meta-llama": OllamaClient(),  # served locally
})

model_id, response, log = router.route_and_call("Solve x^2 = 4", client)
router.process_feedback(log.request_id, reward=0.9)
```

You can also add providers at runtime:

```python
from pareto_bandit import GeminiClient

client.register("google", GeminiClient(api_key="..."))
router.register_model("google/gemini-2.0-flash", speed="fast",
                       input_cost_per_m=0.10, output_cost_per_m=0.40)
```

### `MultiProviderClient`

```python
class MultiProviderClient:
    def __init__(
        self,
        providers: dict[str, LLMClient],
        *,
        default: LLMClient | None = None,
    ): ...
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `providers` | `dict[str, LLMClient]` | — | Maps provider prefix (the part before `/` in a model ID) to a client instance. |
| `default` | `LLMClient \| None` | `None` | Fallback client for model IDs whose prefix isn't in `providers`. If `None`, raises `KeyError`. |

`MultiProviderClient` satisfies `LLMClient`, so it works anywhere a single client does — including `route_and_call()`.

### Model ID Translation

The canonical model registry uses `provider/model-name` IDs (e.g. `openai/gpt-4o`). Each adapter automatically strips the `provider/` prefix when calling the native API, so you don't need to maintain separate ID mappings.

---

## Utility Functions

### `infer_model_family(model_id: str) -> str`

Infer model family from an ID string. Used for family-shared learning.

**Example**

```python
from pareto_bandit import infer_model_family

print(infer_model_family("openai/gpt-4o"))              # "openai/gpt-4o"
print(infer_model_family("anthropic/claude-3.5-sonnet")) # "anthropic/claude-3"
```

### `tetrachoric_corr(p_both: float, p_a: float, p_b: float) -> float`

Estimate tetrachoric correlation from binary agreement rates.

### `compute_correlation_families(data, models) -> dict`

Compute pairwise model-family correlation structure from binary preference data.

---

## CLI

```bash
paretobandit --version              # Show version
paretobandit "Your prompt here"     # Route a prompt
paretobandit --download-models      # Pre-download sentence transformer weights
paretobandit --max-cost 1.0 "..."   # Route with cost constraint
paretobandit --cost-penalty 0.5 "..." # Route with aggressive cost preference
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--version` | — | — | Show version and exit. |
| `--max-cost` | `float` | `None` | Hard cost ceiling ($/1k tokens). Models exceeding this are filtered out. |
| `--cost-penalty` | `float` | `0.3` | Static cost-quality trade-off weight λ_c (paper Eq. 4). `0.0` = quality-only routing, `0.3` = moderate cost awareness, `0.5+` = aggressive cost preference. |
| `--profile` | `str` | `best_value` | Optimization profile. |
| `--download-models` | — | — | Pre-download sentence transformer weights (for Docker/CI pre-warming). |
