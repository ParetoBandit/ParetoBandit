# API Reference

Complete reference for the BanditGPT public API.

```python
import bandit_gpt
print(bandit_gpt.__version__)
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
    context_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    priors: str = "warmup",
    **kwargs,
) -> BanditRouter
```

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_registry` | `dict[str, Any] \| None` | `None` | Model configurations keyed by model ID. Each entry may include `input_cost_per_m`, `output_cost_per_m`, `time_to_first_token_seconds`, and capability metadata. |
| `context_model` | `str` | `"sentence-transformers/all-MiniLM-L6-v2"` | SentenceTransformer model for prompt embedding. Custom models require matching PCA and warmup artifacts. |
| `priors` | `str` | `"warmup"` | Prior initialisation strategy: `"warmup"` (dense covariance from 80k battles), `"hle"` (benchmark-guided bias only), `"none"` (cold start), or a path to a `.joblib` file. |
| `exploration` | `str` | `"safe"` | Named exploration preset: `"static"` (0.0), `"safe"` (0.05), `"balanced"` (0.5), `"aggressive"` (1.0). |
| `alpha` | `float` | `0.05` | Explicit exploration rate (overrides `exploration`). |
| `prior_n_effective` | `float` | `10.0` | Controls how quickly online data overrides warm-start priors. Lower = softer priors. |
| `warmup_path` | `str \| Path \| None` | `None` | Explicit path to warmup priors `.joblib` file. Required when using a custom `context_model` with `priors="warmup"`. |
| `state_path` | `str \| Path \| None` | `None` | Path to load previously saved bandit state. |

**Returns**: Fully initialised `BanditRouter` instance.

**Raises**:
- `ValueError` — Custom `context_model` with `priors="warmup"` but no `warmup_path`.

**Example**

```python
from bandit_gpt import BanditRouter

router = BanditRouter.create(model_registry, priors="warmup")
```

---

### `BanditRouter.route()`

Route a prompt to the best model.

```python
def route(
    self,
    prompt: str | np.ndarray,
    *,
    profile: str | dict[str, float] = "auto",
    max_cost: float | None = None,
    max_latency: float | None = None,
    quality_floor: dict[str, float | None] = None,
    input_tokens: int | None = None,
    output_tokens: int = 600,
    total_steps: int = 1,
) -> tuple[str, RoutingLog]
```

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | `str \| np.ndarray` | — | Input text or pre-computed feature vector. |
| `profile` | `str \| dict` | `"auto"` | Optimisation profile. `"auto"` for default routing, or a dict of weights `{"w_q": ..., "w_c": ..., "w_l": ...}`. |
| `max_cost` | `float \| None` | `None` | Hard cost ceiling ($/1k tokens). Models exceeding this are filtered. |
| `max_latency` | `float \| None` | `None` | Hard latency ceiling (seconds). |
| `quality_floor` | `dict \| None` | `None` | Minimum quality scores per model. |
| `input_tokens` | `int \| None` | `None` | Input token count (auto-estimated from prompt if `None`). |
| `output_tokens` | `int` | `600` | Expected output tokens for cost estimation. |
| `total_steps` | `int` | `1` | Total training steps for alpha decay. Use `1` for production (stable exploitation). |

**Returns**: `(model_id, routing_log)` — The selected model ID and a `RoutingLog` with decision metadata.

**Raises**:
- `ValueError` — Empty or whitespace-only prompt.
- `TypeError` — Prompt is neither `str` nor `np.ndarray`.

**Example**

```python
model_id, log = router.route("Write a Python function to parse JSON")
print(f"Selected: {model_id}, Cost: ${log.cost_usd:.6f}")
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
| `reward` | `float` | Observed quality signal in [0, 1]. Values outside this range are clamped. Typical sources: LLM-as-judge score, user thumbs-up/down (0 or 1), or normalised task metric. |

**Behaviour**:
- Looks up the stored context vector for the request.
- Clamps reward to [0, 1] (required by the Exp4 importance-weighted loss estimator).
- Updates the Corralling meta-learner (or direct LinUCB if Corralling is disabled).
- Supports **delayed feedback**: if the in-memory log has been evicted, falls back to the `SqliteContextStore`. Feedback can arrive hours or days later as long as the context has not expired (default TTL: 7 days).
- **No-op** if `request_id` is unknown (evicted from both stores). A warning is logged.

**Raises**: None. Designed to never crash the router.

**Example**

```python
model_id, log = router.route("Explain quantum entanglement")
# ... call the LLM, evaluate quality ...
router.process_feedback(log.request_id, reward=0.92)
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
) -> None
```

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_id` | `str` | — | Model that was selected. |
| `context` | `str \| np.ndarray` | — | Prompt string or pre-computed feature vector. |
| `reward` | `float` | — | Quality signal in [0, 1] (clamped). |
| `weight` | `float` | `1.0` | Importance weight for this observation. |

Use this for batch/offline learning where you already have `(model, context, reward)` triples. For the standard online workflow, prefer `route()` followed by `process_feedback()`.

**Raises**:
- `ValueError` — Feature vector has wrong dimension.
- `KeyError` — `model_id` is not registered.

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

**Returns**: Dictionary mapping model IDs to selection probabilities (sum to 1.0).

---

### `BanditRouter.explain_decision()`

Feature contribution analysis: decompose a model's score into per-feature contributions.

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

**Returns**: Dictionary mapping feature names (e.g., `"PCA_0"`, `"bias"`) to their contribution to the model's score.

**Example**

```python
_, log = router.route("Write SQL to get active users")
explanation = router.explain_decision("gpt-4o", log.context_vector)
# {"PCA_0": +0.8, "PCA_5": +0.3, "bias": +0.2, ...}
```

---

### `BanditRouter.register_model()`

Add a new model with progressive registration.

```python
def register_model(
    self,
    model_id: str,
    capabilities: list[str] | None = None,
    speed: str = "balanced",
    cost_usd: float | None = None,
    latency_s: float | None = None,
    initial_weights: dict[str, float] | None = None,
) -> None
```

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_id` | `str` | — | Unique model identifier (e.g., `"openai/gpt-4o"`). |
| `capabilities` | `list[str] \| None` | `None` | Semantic capabilities: `"coding"`, `"math"`, `"creative"`, `"reasoning"`, `"general"`. |
| `speed` | `str` | `"balanced"` | T-shirt size: `"fast"`, `"balanced"`, or `"slow"`. Affects initial bias. |
| `cost_usd` | `float \| None` | `None` | Cost per 1k tokens (for constraint filtering). |
| `latency_s` | `float \| None` | `None` | Expected latency in seconds. |
| `initial_weights` | `dict[str, float] \| None` | `None` | Power-user override for explicit theta vector entries. |

**Three knowledge tiers**:
1. **Archetypes** — `capabilities=["coding", "math"]` applies semantic priors.
2. **T-shirt sizing** — `speed="fast"` sets appropriate bias.
3. **Agnostic** — Just `model_id`; initialises with neutral priors and high variance.

---

### `BanditRouter.save_state()` / `BanditRouter.load_state()`

Persist and restore learned bandit parameters.

```python
def save_state(self, path: Path | str) -> None
def load_state(self, path: Path | str) -> None
```

**Known limitation**: Only the base `DisjointLinUCBPolicy` matrices (A, b) are persisted. Corralling meta-weights and expert state are not saved; they reset to initial allocation on reload.

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
| `total_priority_weight` | `float` | Sum of quality/cost/latency weights. |

---

## `FeatureService`

Handles prompt embedding and PCA compression independently from bandit math.

### Constructor

```python
FeatureService(
    encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    pca_path: Path | str | None = None,
    pca_components: int | None = None,
    target_variance: float = 0.60,
    allow_jit_training: bool = True,
    calibration_file: Path | str | None = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `encoder_model` | `str` | Default MiniLM | SentenceTransformer model name. Custom models require explicit `pca_path`. |
| `pca_path` | `Path \| str \| None` | `None` | Path to pre-trained PCA artifact (`.joblib`). |
| `pca_components` | `int \| None` | `None` | Auto-detected from PCA file if not specified. |
| `target_variance` | `float` | `0.60` | Minimum explained variance threshold for PCA. |
| `allow_jit_training` | `bool` | `True` | Allow JIT PCA retraining if artifact is missing. Set `False` in strict production. |
| `calibration_file` | `Path \| str \| None` | `None` | Line-delimited text file of real prompts for domain-specific PCA training. |

**Raises**:
- `ValueError` — Custom encoder without explicit `pca_path`.

### `FeatureService.for_precomputed()`

Create a lightweight service for pre-computed embedding vectors (no model loading).

```python
@classmethod
def for_precomputed(cls, dimension: int) -> FeatureService
```

### `FeatureService.extract_features()`

Convert a prompt to a feature vector.

```python
def extract_features(self, prompt: str | np.ndarray) -> np.ndarray
```

**Returns**: Feature vector of shape `(pca_components + 1,)`. The last element is a bias term (always 1.0). Default: 33 dimensions (32 PCA + 1 bias).

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

### `FeatureService.get_feature_names()`

Human-readable feature names for interpretability.

```python
def get_feature_names(self) -> list[str]
```

**Returns**: List like `["PCA_0", "PCA_1", ..., "PCA_31", "bias"]`.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `dimension` | `int` | Total feature dimension (PCA components + 1 bias). |
| `bias_index` | `int` | Index of the bias term (always -1). |
| `using_pca` | `bool` | Whether PCA compression is active. |

---

## `RouterConfig`

Dataclass for all router hyperparameters. Pass to `BanditRouter.__init__()` or let defaults apply.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_log_size` | `int` | `10_000` | Ring buffer size for in-memory routing logs. |
| `init_lambda` | `float` | `1.0` | Regularisation for cold-start (A₀ = λI). |
| `update_lambda` | `float` | `0.0` | Runtime regularisation. `0.0` enables O(d²) Sherman-Morrison. |
| `stability_check_interval` | `int` | `1000` | Check numerical stability every N updates. |
| `stability_threshold` | `float` | `1e6` | Max trace(A\_inv) before reset. |
| `market_cost_floor` | `float` | `0.0001` | Cost normalisation floor ($/1k tokens). |
| `market_cost_ceiling` | `float` | `0.04` | Cost normalisation ceiling ($/1k tokens). |
| `default_missing_cost_per_m` | `float` | `10.00` | Pessimistic cost fallback for missing metadata. |
| `default_missing_latency` | `float` | `2.0` | Pessimistic latency fallback. |

---

## `ExplorationRate`

Named presets for the exploration parameter (alpha).

| Preset | Alpha | Use Case |
|--------|-------|----------|
| `ExplorationRate.STATIC` | `0.0` | Pure exploitation — production/fintech. |
| `ExplorationRate.SAFE` | `0.1` | Default. Minimal exploration. |
| `ExplorationRate.BALANCED` | `1.0` | Standard bandit behaviour. |
| `ExplorationRate.AGGRESSIVE` | `2.0` | Day-1 calibration or shadow mode. |

---

## `HybridLinUCBPolicy`

Hybrid LinUCB with family-shared and arm-specific ridge regression. Used internally by `BanditRouter` when model families are detected.

For arm *a* in family *F*: `E[r | x, a] = x^T beta_F + x^T theta_a`

This is an advanced internal class. Most users interact with it through `BanditRouter`.

---

## Calibration API

### `train_pca()`

Train a PCA artifact for a custom sentence transformer.

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
| `prompts` | `list[str]` | — | Representative corpus (100+ recommended). |
| `encoder_model` | `str` | — | HuggingFace SentenceTransformer model name. |
| `n_components` | `int` | `32` | Number of PCA components to retain. |
| `output_path` | `Path \| str \| None` | `None` | Persist the PCA via joblib. |
| `batch_size` | `int` | `64` | Encoder batch size. |

**Returns**: Fitted `sklearn.decomposition.PCA` object.

**Raises**:
- `ValueError` — Empty prompts or fewer prompts than `n_components`.

---

### `generate_warmup_priors()`

Generate warmup priors (A, b matrices) for LinUCB from labelled data.

```python
def generate_warmup_priors(
    rewards_data: list[dict],
    encoder_model: str,
    pca: PCA | Path | str,
    plasticity: float = 0.1,
    output_path: Path | str | None = None,
    batch_size: int = 64,
) -> dict
```

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rewards_data` | `list[dict]` | — | Each entry: `{"prompt": str, "rewards": {"model_id": float, ...}}`. |
| `encoder_model` | `str` | — | Must match the model used for `train_pca()`. |
| `pca` | `PCA \| Path \| str` | — | Fitted PCA object or path to joblib file. |
| `plasticity` | `float` | `0.1` | Scaling factor. Lower = softer priors, faster to override. |
| `output_path` | `Path \| str \| None` | `None` | Persist the priors via joblib. |

**Returns**: Dict with keys `A`, `b`, `models`, `n_prompts`, `context_dim`, `pca_components`, `plasticity`, `reward_source`.

**Raises**:
- `ValueError` — Empty or malformed `rewards_data`.

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

---

## Utility Functions

### `infer_model_family(model_id: str) -> str`

Infer model family from an ID string (e.g., `"openai/gpt-4o"` → `"gpt-4"`). Used for family-shared learning in `HybridLinUCBPolicy`.

### `tetrachoric_corr(p_both: float, p_a: float, p_b: float) -> float`

Estimate tetrachoric correlation from binary agreement rates.

### `compute_correlation_families(data, models) -> dict`

Compute pairwise model-family correlation structure from binary preference data.

---

## CLI

```bash
banditgpt --version              # Show version
banditgpt "Your prompt here"     # Route a prompt
banditgpt --download-models      # Pre-download sentence transformer weights
banditgpt --max-cost 1.0 "..."   # Route with cost constraint
```
