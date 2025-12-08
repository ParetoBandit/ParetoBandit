"""
Chebyshev Scorer for Multi-Objective Model Optimization.

This module implements the Chebyshev scalarization method for ranking models
based on quality, cost, and latency objectives.
"""

import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass

from llm_jury.optimization.total_cost_inference import calculate_tci, MIN_COST_FLOOR


@dataclass
class ModelScore:
    """Scoring result for a single model."""
    model_name: str
    quality_score: float
    cost_per_m: float
    latency_ms: float
    trustability_index: float
    chebyshev_distance: float
    quality_regret: float
    cost_regret: float
    latency_regret: float
    trustability_regret: float
    value_per_dollar: float


class ChebyshevScorer:
    """
    Multi-objective model scorer using Chebyshev scalarization.
    
    Minimizes the maximum regret across four dimensions relative to utopia:
    - Quality: 100% (perfect quality)
    - Cost: Minimum TCI (Total Cost of Inference)
    - Latency: 0ms (instant)
    - Trustability: ∞ (maximum trust, normalized to high positive value)
    
    Formula:
        Chebyshev Distance = max(w_q * q_regret, w_c * c_regret, w_l * l_regret, w_t * t_regret)
    
    Where regret is the normalized distance from the ideal value.
    
    Note on TCI (Total Cost of Inference):
        Open-source models with $0 API pricing are assigned shadow compute
        costs based on GPU infrastructure, preventing them from dominating
        cost-based optimization unfairly.
    """
    
    def __init__(
        self,
        baseline_quality: float,
        baseline_cost: float,
        baseline_latency: float,
        baseline_trustability: float = 2.0,  # High trustability baseline (2 std devs above mean)
        quality_weight: float = 0.3,
        cost_weight: float = 0.25,
        latency_weight: float = 0.25,
        trustability_weight: float = 0.2
    ):
        """
        Initialize the Chebyshev scorer with trustability.
        
        Args:
            baseline_quality: Reference quality score (e.g., from GPT-4o)
            baseline_cost: Reference cost per million tokens
            baseline_latency: Reference latency in milliseconds
            baseline_trustability: Reference trustability (normalized, mean=0, std=1)
            quality_weight: Weight for quality objective (0-1)
            cost_weight: Weight for cost objective (0-1)
            latency_weight: Weight for latency objective (0-1)
            trustability_weight: Weight for trustability objective (0-1)
        """
        self.baseline_quality = baseline_quality
        self.baseline_cost = baseline_cost
        self.baseline_latency = baseline_latency
        self.baseline_trustability = baseline_trustability
        
        # Normalize weights to sum to 1
        total = quality_weight + cost_weight + latency_weight + trustability_weight
        self.w_quality = quality_weight / total
        self.w_cost = cost_weight / total
        self.w_latency = latency_weight / total
        self.w_trustability = trustability_weight / total
    
    def calculate_regret(
        self,
        quality: float,
        cost: float,
        latency: float,
        trustability: float
    ) -> Dict[str, float]:
        """
        Calculate regret (distance from utopia) for each objective.
        
        Utopia point:
        - Quality: 100% (perfect)
        - Cost: $0 (free)
        - Latency: 0ms (instant)
        - Trustability: ∞ (maximum trust, using baseline as reference)
        
        Args:
            quality: Model quality score
            cost: Cost per million tokens
            latency: Latency in milliseconds
            trustability: Trustability index (normalized, mean=0, std=1)
            
        Returns:
            Dictionary with quality_regret, cost_regret, latency_regret, trustability_regret
        """
        # Quality regret: how much worse than baseline (0 = perfect, 1 = worst)
        quality_regret = max(0, 1.0 - (quality / self.baseline_quality)) if self.baseline_quality > 0 else 0
        
        # Cost regret: how much more expensive than baseline
        cost_regret = max(0, (cost - self.baseline_cost) / self.baseline_cost) if self.baseline_cost > 0 else 0
        
        # Latency regret: how much slower than baseline
        latency_regret = max(0, (latency - self.baseline_latency) / self.baseline_latency) if self.baseline_latency > 0 else 0
        
        # Trustability regret: how much less trustworthy than baseline
        # Since trustability is normalized (mean=0, std=1), higher is better
        # Regret = how much below the baseline
        trustability_regret = max(0, (self.baseline_trustability - trustability) / max(abs(self.baseline_trustability), 1.0))
        
        return {
            'quality_regret': quality_regret,
            'cost_regret': cost_regret,
            'latency_regret': latency_regret,
            'trustability_regret': trustability_regret
        }
    
    def calculate_chebyshev_distance(
        self,
        quality_regret: float,
        cost_regret: float,
        latency_regret: float,
        trustability_regret: float
    ) -> float:
        """
        Calculate Chebyshev distance (max weighted regret across 4 dimensions).
        
        Args:
            quality_regret: Quality regret [0, 1]
            cost_regret: Cost regret [0, ∞)
            latency_regret: Latency regret [0, ∞)
            trustability_regret: Trustability regret [0, ∞)
            
        Returns:
            Chebyshev distance (lower is better)
        """
        return max(
            self.w_quality * quality_regret,
            self.w_cost * cost_regret,
            self.w_latency * latency_regret,
            self.w_trustability * trustability_regret
        )
    
    def score_model(
        self,
        model_name: str,
        quality: float,
        cost: float,
        latency: float,
        trustability: float
    ) -> ModelScore:
        """
        Score a single model using 4-dimensional Chebyshev optimization.
        
        Args:
            model_name: Name of the model
            quality: Quality score
            cost: Cost per million tokens
            latency: Latency in milliseconds
            trustability: Trustability index (normalized)
            
        Returns:
            ModelScore object with all metrics
        """
        # Calculate regrets
        regrets = self.calculate_regret(quality, cost, latency, trustability)
        
        # Calculate Chebyshev distance
        chebyshev_dist = self.calculate_chebyshev_distance(
            regrets['quality_regret'],
            regrets['cost_regret'],
            regrets['latency_regret'],
            regrets['trustability_regret']
        )
        
        # Calculate value per dollar using TCI (Total Cost of Inference)
        # TCI handles $0 pricing by applying shadow compute costs
        # Use model_name for parameter estimation if cost is zero
        tci = calculate_tci(cost, cost, model_name=model_name) if cost <= MIN_COST_FLOOR else cost
        tci = max(tci, MIN_COST_FLOOR)  # Ensure non-zero denominator
        value_per_dollar = quality / tci
        
        return ModelScore(
            model_name=model_name,
            quality_score=quality,
            cost_per_m=cost,
            latency_ms=latency,
            trustability_index=trustability,
            chebyshev_distance=chebyshev_dist,
            quality_regret=regrets['quality_regret'],
            cost_regret=regrets['cost_regret'],
            latency_regret=regrets['latency_regret'],
            trustability_regret=regrets['trustability_regret'],
            value_per_dollar=value_per_dollar
        )
    
    def rank_models(
        self,
        models: List[Dict[str, any]],
        top_k: Optional[int] = None
    ) -> List[ModelScore]:
        """
        Rank multiple models by Chebyshev distance (4D optimization).
        
        Args:
            models: List of model dicts with 'name', 'quality', 'cost', 'latency', 'trustability'
            top_k: Optional number of top models to return
            
        Returns:
            List of ModelScore objects sorted by Chebyshev distance (best first)
        """
        scores = []
        
        for model in models:
            score = self.score_model(
                model_name=model['name'],
                quality=model['quality'],
                cost=model['cost'],
                latency=model['latency'],
                trustability=model.get('trustability', 0.0)  # Default to 0 (mean) if missing
            )
            scores.append(score)
        
        # Sort by Chebyshev distance (lower is better)
        scores.sort(key=lambda x: x.chebyshev_distance)
        
        if top_k:
            return scores[:top_k]
        return scores


def example_usage():
    """Example of how to use the ChebyshevScorer with trustability."""
    # Define baseline (e.g., GPT-4o)
    scorer = ChebyshevScorer(
        baseline_quality=88.7,
        baseline_cost=5.0,
        baseline_latency=500.0,
        baseline_trustability=2.0,  # 2 std devs above mean (top ~2.5%)
        quality_weight=0.3,
        cost_weight=0.25,
        latency_weight=0.25,
        trustability_weight=0.2
    )
    
    # Example models with trustability scores
    models = [
        {'name': 'GPT-4o', 'quality': 88.7, 'cost': 5.0, 'latency': 500, 'trustability': 2.1},
        {'name': 'Claude-3.5-Sonnet', 'quality': 88.3, 'cost': 3.0, 'latency': 450, 'trustability': 1.8},
        {'name': 'Llama-3.1-70B', 'quality': 82.0, 'cost': 0.88, 'latency': 400, 'trustability': 0.6},
        {'name': 'Llama-3.1-8B', 'quality': 68.4, 'cost': 0.15, 'latency': 200, 'trustability': 0.3},
    ]
    
    # Rank models
    ranked = scorer.rank_models(models, top_k=3)
    
    print("Top 3 Models by 4D Chebyshev Optimization:")
    print("(Quality, Cost, Latency, Trustability)")
    for i, score in enumerate(ranked, 1):
        print(f"\n{i}. {score.model_name}")
        print(f"   Chebyshev Distance: {score.chebyshev_distance:.4f}")
        print(f"   Quality: {score.quality_score:.1f}")
        print(f"   Cost: ${score.cost_per_m:.2f}/M")
        print(f"   Latency: {score.latency_ms:.0f}ms")
        print(f"   Trustability: {score.trustability_index:.2f}")
        print(f"   Value/Dollar: {score.value_per_dollar:.2f}")


if __name__ == "__main__":
    example_usage()
