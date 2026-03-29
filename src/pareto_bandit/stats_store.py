import json
import math


class StatsStore:
    def __init__(self, config_path: str):
        # Configuration: How strong is our prior belief?
        # A value of 20 means we treat the offline test as if it were 20 real user requests.
        self.PRIOR_STRENGTH = 20

        # Load static config
        with open(config_path) as f:
            data = json.load(f)
            # Support both direct list and nested "models" key
            self.models_config = data.get('models', data) if isinstance(data, dict) else data

        # Initialize in-memory stats (In prod, use Redis)
        # Structure: { model_id: { 'successes': 0, 'failures': 0 } }
        self.live_stats = {}
        for m in self.models_config:
            m_id = m.get('id') or m.get('model_id')
            if m_id:
                self.live_stats[m_id] = {'successes': 0, 'failures': 0}

    def get_model_stats(self, model_id: str) -> dict:
        """Returns the static config merged with dynamic stats."""
        static = next((m for m in self.models_config if m.get('id') == model_id or m.get('model_id') == model_id), None)
        stats = self.live_stats.get(model_id)

        if not static:
            raise ValueError(f"Model {model_id} not found in config")
        if stats is None:
            raise ValueError(f"Model {model_id} has no live stats")

        # Calculate Bayesian Smoothed Quality
        # We convert initial_quality % into virtual success/fail counts
        # Default to 0.5 if initial_quality is missing
        initial_quality = static.get('initial_quality', 0.5)
        prior_successes = initial_quality * self.PRIOR_STRENGTH
        prior_failures = self.PRIOR_STRENGTH - prior_successes

        total_successes = prior_successes + stats['successes']
        total_failures = prior_failures + stats['failures']
        total_events = total_successes + total_failures

        # Mean (Expected Quality)
        mean_quality = total_successes / total_events

        # Beta posterior std: sqrt(alpha*beta / ((alpha+beta)^2 * (alpha+beta+1)))
        alpha_param = total_successes
        beta_param = total_failures
        ab_sum = alpha_param + beta_param
        uncertainty = math.sqrt(
            (alpha_param * beta_param) / (ab_sum * ab_sum * (ab_sum + 1))
        )

        # Handle cost calculation if blended_cost_per_1k is missing
        cost = static.get('blended_cost_per_1k')
        if cost is None:
            # Estimate blended cost as (input + output) / 2 per 1k tokens
            # input_cost_per_m is per 1M tokens, so divide by 1000 for per 1k
            input_cost = static.get('input_cost_per_m', 0) / 1000.0
            output_cost = static.get('output_cost_per_m', 0) / 1000.0
            cost = (input_cost + output_cost) / 2.0

        return {
            "id": model_id,
            "cost": cost,
            "mean_quality": mean_quality,
            "uncertainty": uncertainty,
            "sample_count": stats['successes'] + stats['failures']
        }

    def update_stats(self, model_id: str, is_success: bool) -> None:
        if is_success:
            self.live_stats[model_id]['successes'] += 1
        else:
            self.live_stats[model_id]['failures'] += 1
