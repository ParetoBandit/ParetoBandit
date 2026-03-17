# Budget Pacing via Primal-Dual CBwK

> **Status**: V1 Plan (pending implementation)
> **Author**: Generated via code review of `src/pareto_bandit/router.py`
> **Theory**: Primal-Dual framework for Contextual Bandits with Knapsacks (Agrawal & Devanur 2014)

## Architecture

A standalone `BudgetPacer` class owns all pacing state. Three user-selectable modes share the same dual-variable tracking but differ in enforcement:

- **HARD** -- Adaptive `max_cost` ceiling fed through existing `_filter_by_constraints()`. Zero changes to expert/corralling internals.
- **SOFT** -- Dynamic per-request cost penalties injected into expert UCB scoring via a new `extra_cost_penalties` parameter. Requires small backward-compatible changes to 3 classes.
- **ADAPTIVE** -- Both mechanisms active. Hard ceiling as safety net, soft penalty as optimizer.

### User API

```python
from pareto_bandit.router import BanditRouter, BudgetPacer, PacingMode

# Hard-only: coarse-grained, zero internal changes
pacer = BudgetPacer(target_avg_spend_usd=0.01, mode=PacingMode.HARD)

# Soft-only: request-level token-aware Lagrangian
pacer = BudgetPacer(target_avg_spend_usd=0.01, mode=PacingMode.SOFT)

# Adaptive: hard ceiling + soft optimizer (recommended)
pacer = BudgetPacer(target_avg_spend_usd=0.01, mode=PacingMode.ADAPTIVE)

router = BanditRouter(model_registry, budget_pacer=pacer)
model, log = router.route(prompt)
router.process_feedback(log.request_id, reward)
```

### Data Flow

```
BanditRouter.route()
│
├─ BudgetPacer.get_cost_ceiling_per_1k()  ──► _filter_by_constraints(max_cost=ceiling)
│                                              [HARD path: exclude expensive models]
│
├─ BudgetPacer.lambda_t × _get_normalized_request_cost(m, in_tok, out_tok)
│   for each candidate m  ──────────────────► CorrallingRouter.select_model(extra_cost_penalties=...)
│                                              ├─ CostAwareLinUCBAdapter.select_model(extra_cost_penalties=...)
│                                              └─ CostAwareTabulaRasaRouter.select_model(extra_cost_penalties=...)
│                                              [SOFT path: penalize in UCB score]
│
└─ ... normal routing proceeds ...

BanditRouter.process_feedback()
│
└─ BudgetPacer.observe(actual_cost_usd)
   ├─ EMA update: cost_ema = (1 - α) * cost_ema + α * actual_cost
   └─ Dual update: lambda_t = max(0, lambda_t + lr * (actual_cost - target))
```

## Files to Change

**Primary**: `src/pareto_bandit/router.py`

## Implementation Steps

### Step 1: Add `PacingMode` enum and `BudgetPacer` class (~60 lines, new code)

Place near the top of `router.py` alongside other enums/dataclasses (after `RouterConfig`, ~line 641).

```python
class PacingMode(enum.Enum):
    HARD = "hard"
    SOFT = "soft"
    ADAPTIVE = "adaptive"

class BudgetPacer:
    def __init__(self, target_avg_spend_usd: float, mode: PacingMode = PacingMode.ADAPTIVE,
                 lr: float = 0.05, ema_alpha: float = 0.05):
        ...
        self.lambda_t: float = 0.0
        self.cost_ema: float = target_avg_spend_usd  # warm-start at target
        self._lock = threading.Lock()

    @property
    def uses_hard(self) -> bool:
        return self.mode in (PacingMode.HARD, PacingMode.ADAPTIVE)

    @property
    def uses_soft(self) -> bool:
        return self.mode in (PacingMode.SOFT, PacingMode.ADAPTIVE)

    def get_cost_ceiling_per_1k(self) -> float | None:
        """HARD mode: convert pacing state to a blended_cost_per_m ceiling."""
        ...

    def get_extra_cost_penalties(self, model_costs: Dict[str, float]) -> Dict[str, float]:
        """SOFT mode: lambda_t * normalized_request_cost for each model."""
        ...

    def observe(self, actual_cost_usd: float):
        """Dual update + EMA update. Called from process_feedback()."""
        ...
```

Key design points:

- `cost_ema` warm-starts at target (not 0) to avoid cold-start overshoot
- `lambda_t` starts at 0 (no penalty initially; ramps up only if overspending)
- Thread-safe via dedicated lock
- `get_cost_ceiling_per_1k()` returns `None` when underspending (no constraint)

### Step 2: Add `_get_normalized_request_cost()` to `BanditRouter` (~10 lines, new method)

Uses existing `_estimate_cost()` for raw USD, then normalizes through the same log-market-anchor transform used for static costs. This keeps the soft penalty commensurate with the [0, 1] UCB reward scale.

Place near `_estimate_cost()` (~line 4730):

```python
def _get_normalized_request_cost(self, model: str, in_tok: int, out_tok: int) -> float:
    cost_usd = self._estimate_cost(model, in_tok, out_tok)
    cost_per_1k = cost_usd / ((in_tok + out_tok) / 1000) if (in_tok + out_tok) > 0 else 0.0
    return self._calculate_absolute_cost_penalty(cost_per_1k)
```

### Step 3: Add `extra_cost_penalties` passthrough to expert `select_model()` (~5 lines each, 3 classes)

**`CostAwareLinUCBAdapter.select_model()`** (line 5871):

```python
def select_model(self, context, total_steps=0, candidates=None,
                 extra_cost_penalties: Dict[str, float] | None = None) -> str:
    # ... existing loop ...
    extra = extra_cost_penalties.get(model, 0.0) if extra_cost_penalties else 0.0
    score = (expected_reward + alpha * uncertainty) \
            - (self.cost_penalty * normalized_cost) \
            - extra
```

**`CostAwareTabulaRasaRouter.select_model()`** (line 6113): Same pattern.

**`CorrallingRouter.select_model()`** (line 5003): Accept param + pass through:

```python
def select_model(self, context, total_steps=0, candidates=None,
                 extra_cost_penalties: Dict[str, float] | None = None) -> Tuple[str, Dict]:
    recommendations = [
        expert.select_model(context, total_steps=total_steps,
                            candidates=candidates,
                            extra_cost_penalties=extra_cost_penalties)
        for expert in self.experts
    ]
```

All three changes are backward-compatible (`None` default = existing behavior).

### Step 4: Integrate `BudgetPacer` into `BanditRouter` (~25 lines)

**`__init__()`** (line 2661): Accept optional `budget_pacer` parameter.

**`route()`** (line 4000): Before existing constraint filtering:

```python
effective_max_cost = max_cost
extra_cost_penalties = None

if self.budget_pacer is not None:
    if self.budget_pacer.uses_hard:
        ceiling = self.budget_pacer.get_cost_ceiling_per_1k()
        if ceiling is not None:
            effective_max_cost = min(effective_max_cost, ceiling) if effective_max_cost else ceiling

    if self.budget_pacer.uses_soft:
        extra_cost_penalties = {
            m: self.budget_pacer.lambda_t * self._get_normalized_request_cost(m, in_tok, output_tokens)
            for m in candidates
        }
```

Then pass `effective_max_cost` to `_filter_by_constraints()` and `extra_cost_penalties` to the corralling/bandit selection call.

**`process_feedback()`** (line 4162): After existing update logic:

```python
if self.budget_pacer is not None:
    self.budget_pacer.observe(log.cost_usd)
```

### Step 5: Add `budget_pacer` to `RoutingLog` metadata (~3 lines)

Store `pacer_lambda_t` and `pacer_cost_ema` in `RoutingLog` for observability/debugging.

---

## Testing and Validation

### T1: Unit Tests -- `tests/test_budget_pacing.py`

Follows the conventions in `tests/test_correctness_invariants.py` and `tests/test_bandit_router.py`.

**Fixtures:**

- `sample_registry`: Reuse existing 2-model fixture (GPT-4o at $5/$15 per M, Gemma at $0.1/$0.1 per M) from `test_bandit_router.py`.
- `three_model_registry`: Add a mid-tier model (e.g., Sonnet at $3/$15 per M) for richer pacing tests.
- `make_router(budget_pacer=None)`: Helper that calls `BanditRouter.create(..., priors="none")` with optional pacer injection.

#### T1a: BudgetPacer isolation tests (no router dependency)

These test the `BudgetPacer` class in complete isolation -- just feed it cost observations and verify state evolution.

| Test | What it verifies |
|------|-----------------|
| `test_pacer_init_defaults` | `lambda_t == 0.0`, `cost_ema == target`, mode is ADAPTIVE |
| `test_pacer_lambda_increases_on_overspend` | Feed 50 observations above target; assert `lambda_t > 0` and monotonically increasing |
| `test_pacer_lambda_stays_zero_on_underspend` | Feed 50 observations below target; assert `lambda_t == 0.0` throughout |
| `test_pacer_lambda_recovers_after_burst` | Overspend for 20 steps, then underspend for 80; assert `lambda_t` rises then falls back toward 0 |
| `test_pacer_ema_tracks_mean` | Feed constant cost; assert EMA converges to that constant within tolerance |
| `test_pacer_ceiling_none_when_underspending` | In HARD mode, underspending returns `None` ceiling (no constraint) |
| `test_pacer_ceiling_tightens_on_overspend` | In HARD mode, overspending returns a finite ceiling that decreases as lambda grows |
| `test_pacer_extra_penalties_proportional_to_lambda` | In SOFT mode, verify penalties scale linearly with `lambda_t` and with model cost |
| `test_pacer_mode_properties` | `uses_hard` is True for HARD and ADAPTIVE; `uses_soft` is True for SOFT and ADAPTIVE |

#### T1b: Correctness invariants (no-regression)

These ensure the pacer does not break any existing behavior when inactive.

| Test | What it verifies |
|------|-----------------|
| `test_no_pacer_route_unchanged` | Router with `budget_pacer=None` produces identical routing decisions as before (compare model selections and log fields for 100 prompts with fixed seed) |
| `test_extra_cost_penalties_none_passthrough` | `CostAwareLinUCBAdapter.select_model(..., extra_cost_penalties=None)` produces identical scores as the original signature |
| `test_corralling_none_passthrough` | `CorrallingRouter.select_model(..., extra_cost_penalties=None)` produces identical selection tokens |
| `test_tabula_rasa_none_passthrough` | Same for `CostAwareTabulaRasaRouter` |

#### T1c: Integration tests (pacer + router)

| Test | What it verifies |
|------|-----------------|
| `test_hard_mode_excludes_expensive_model` | With tight target and HARD mode, after enough overspend observations, the expensive model stops appearing in selections |
| `test_soft_mode_shifts_preference` | With SOFT mode, expensive model is selected less frequently as lambda grows (compare selection frequency over 200 routes with feedback) |
| `test_adaptive_mode_both_active` | Verify that ADAPTIVE mode both filters (hard) and penalizes (soft) simultaneously |
| `test_pacing_converges_to_target` | Run 500 route+feedback cycles with oracle rewards; assert that the trailing-100 average cost is within 20% of target |
| `test_pacing_with_max_cost_takes_minimum` | User-supplied `max_cost=X` and pacer ceiling `Y` should use `min(X, Y)` |
| `test_pacer_observe_called_in_process_feedback` | Mock `pacer.observe` and verify it is called exactly once per `process_feedback()` invocation |
| `test_routing_log_contains_pacer_state` | Assert `log.pacer_lambda_t` and `log.pacer_cost_ema` are populated when pacer is active, and absent/None when not |

#### T1d: Thread safety

Follow the pattern from `tests/test_concurrency_bandit.py`.

| Test | What it verifies |
|------|-----------------|
| `test_concurrent_route_and_feedback_with_pacer` | 3 writer threads (calling `process_feedback`), 5 reader threads (calling `route`), pacer active in ADAPTIVE mode, run for 3 seconds. Assert: no crashes, no NaN in lambda or EMA, final lambda is finite |
| `test_pacer_lock_prevents_torn_reads` | Rapidly alternate `observe()` and `get_cost_ceiling_per_1k()` from separate threads; assert ceiling is always `None` or a valid positive float |

#### T1e: Edge cases

| Test | What it verifies |
|------|-----------------|
| `test_pacer_zero_target_raises` | `BudgetPacer(target_avg_spend_usd=0.0)` raises `ValueError` |
| `test_pacer_negative_target_raises` | Negative target raises `ValueError` |
| `test_pacer_with_single_model` | Pacing with only 1 model in registry doesn't crash (hard ceiling can't exclude the only model) |
| `test_pacer_all_models_filtered_falls_back` | If hard ceiling would exclude ALL models, behavior is graceful (either raise `NoEligibleModelsError` or relax ceiling) |

---

### T2: Offline Simulation Experiment -- `experiments/appendix/K_budget_pacing/run_budget_pacing.py`

Follows the pattern of `experiments/appendix/F_constraint_impact/run_constraint_experiment.py`. Uses the K=3 portfolio with pre-computed embeddings and oracle rewards.

**Protocol:**

1. Load K=3 portfolio (Llama-8B, Gemini Flash, GPT-4.1) with cost metadata
2. Load dev set (1,028 prompts) with pre-computed embeddings and oracle rewards
3. For each condition x N_SEEDS (5) seeds:
   a. Instantiate router via `create_experiment_router()` with warmup priors
   b. Inject `BudgetPacer(target, mode)`
   c. Online-learn: stream prompts, route, observe oracle reward + actual cost
   d. Track per-step: selected model, realized cost, reward, `lambda_t`, `cost_ema`
4. Output JSON results consumed by figure generator

**Conditions (rows in the results table):**

| Condition | Target | Mode | What it tests |
|-----------|--------|------|--------------|
| No pacing (baseline) | None | -- | Unconstrained quality-optimal routing |
| Hard, generous target | median model cost | HARD | Pacing barely activates; quality should match baseline |
| Hard, tight target | cheapest model cost | HARD | Forces near-exclusive use of cheap model; measures quality loss |
| Soft, generous target | median model cost | SOFT | Lambda stays near 0; quality matches baseline |
| Soft, tight target | cheapest model cost | SOFT | Lambda climbs; should still allow expensive model for short/easy prompts |
| Adaptive, tight target | cheapest model cost | ADAPTIVE | Best of both; ceiling prevents catastrophe, penalty optimizes within |
| Sweep: target from cheap to expensive | 5 log-spaced values | ADAPTIVE | Full Pareto frontier (quality vs realized cost) |

**Metrics to compute (per condition, averaged over seeds):**

- **Realized avg cost** vs target (convergence)
- **Cumulative budget violation**: sum of max(0, cost_t - target) over time
- **Mean reward** (quality under pacing)
- **Quality-cost Pareto efficiency**: does pacing achieve a better frontier than static `cost_penalty` sweep?
- **Convergence speed**: number of steps until trailing-100 avg cost is within 10% of target
- **Lambda trajectory**: plot `lambda_t` over time (should stabilize)
- **Model selection distribution**: how the mix shifts under different targets

**Output:** `experiments/appendix/K_budget_pacing/results/budget_pacing_results.json`

---

### T3: Figure Generator -- `experiments/appendix/K_budget_pacing/generate_figure.py`

Produces 3 publication-ready figures:

1. **Pacing convergence plot**: x-axis = request number, y-axis = trailing-100 avg cost. One line per mode (HARD, SOFT, ADAPTIVE) with the target shown as a horizontal dashed line. Shows how quickly each mode converges to the budget target.
2. **Quality-cost Pareto frontier**: Scatter plot with realized avg cost on x-axis, mean reward on y-axis. Points for each (target, mode) condition. Compare against static `cost_penalty` sweep baseline. The pacing frontier should Pareto-dominate or match the static sweep.
3. **Lambda and model-mix dynamics**: Dual-axis plot. Left y-axis = `lambda_t` over time. Right y-axis = stacked area showing model selection fractions. Demonstrates the interpretable feedback loop: lambda rises, expensive model share drops, cost falls, lambda stabilizes.

---

### T4: Robustness Stress Scenarios (within the simulation experiment)

These are additional conditions added to the simulation to test adversarial/non-stationary scenarios:

| Scenario | Description | What it validates |
|----------|-------------|------------------|
| **Cost spike** | After step 300, switch all prompts to 5x longer (simulating sudden increase in prompt length) | Lambda should spike and recover; pacing should not permanently break |
| **Distribution shift** | First 500 prompts are coding (cheap models adequate), next 500 are hard math (expensive model needed) | Pacing adapts to new cost regime without being stuck |
| **Tight-then-relaxed** | Target = $0.001 for first 500 steps, then $0.01 | Lambda should drop rapidly when target relaxes |

---

## What Is NOT in V1 (Deferred)

- **Lambda warm-start heuristic**: `lambda_0 = 0` with `cost_ema` warm-started at target is good enough.
- **Persistence in save/load_state**: Lambda resets on restart. Document this.
- **Replacing static cost_penalty**: Both coexist. Static penalty is model-level prior knowledge; dynamic penalty is budget enforcement. They are conceptually distinct.

## Background: Original Plan Review

The original proposal (Primal-Dual CBwK with single `target_avg_spend_usd` parameter) was reviewed and found to be theoretically sound but had 6 integration issues with the existing codebase:

1. **Critical**: The corralling path (default) was not covered -- cost penalties could not be injected from the `BanditRouter` level into expert-internal UCB scoring.
2. **Commensurability**: Raw USD penalties are not commensurate with the [0, 1] UCB reward scale.
3. **Cold-start overshoot**: `lambda_0 = 0` means no cost awareness for initial requests.
4. **Thread safety**: `dynamic_lambda` needs lock protection.
5. **Persistence**: Lambda not included in save/load_state.
6. **Dual mechanism confusion**: Static `cost_penalty` and dynamic `lambda_t` overlap.

The unified `BudgetPacer` design addresses issues 1-4 directly. Issues 5-6 are documented as deferred.
