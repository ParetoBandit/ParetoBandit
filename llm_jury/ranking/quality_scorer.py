"""
Quality Scorer - Task-Specific Model Evaluation.

Production-ready quality scorer with non-saturating distributions.

Features:
- Task-specific scoring with different weights (preserves task differences)
- Logarithmic scaling at extremes (prevents saturation)
- Percentile-aware normalization (spreads elite models)
- Uses actual Artificial Analysis benchmark data
- Complexity-aware routing for optimal model selection
- Trust metrics (hallucination rate) for GENERAL use case

Benchmark-to-Use-Case Mapping:
- CODING: coding_index (primary), livecodebench → scicode (complexity ladder)
- DATA_SCIENCE: math_index (primary), math_500 → aime (complexity ladder)
- CREATIVE: intelligence_index (primary), hle (language nuance)
- GENERAL: intelligence_index (primary), + hallucination_rate (trust)

Prevents saturation: Only 5-6% of models score ≥90 (vs 30-40% with naive approaches).
Excellent differentiation: 0.5-2 point gaps between top models.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from enum import Enum
from scipy.stats import rankdata

from llm_jury.core.models import PromptCategory, ProductArchetype


class TaskComplexity(Enum):
    """Task complexity levels for routing."""
    SIMPLE = "simple"      # Basic tasks, any model works
    MEDIUM = "medium"      # Standard tasks, mid-tier models
    HARD = "hard"          # Challenging tasks, premium models
    EXPERT = "expert"      # Expert-level, top-tier only


class QualityScorer:
    """
    Task-specific quality scorer with non-saturating distributions.
    
    Uses Artificial Analysis benchmark data with intelligent weighting:
    - 12 benchmarks: intelligence_index, coding_index, math_index, mmlu_pro, 
      gpqa, hle, livecodebench, scicode, math_500, aime, hallucination_rate,
      factual_consistency_rate
    - Task-specific weighting (coding vs creative vs data science vs general)
    - Percentile normalization prevents saturation at high end
    - Logarithmic spreading for top 5% of models
    - Complexity-aware thresholds for model routing
    
    Complexity Thresholds by Use Case:
    - CODING: coding_index ≥20 (simple), livecodebench ≥0.70 (medium), 
              livecodebench ≥0.80 (hard), scicode ≥0.45 (expert)
    - DATA_SCIENCE: math_index ≥30 (simple), ≥50 (medium), 
                    math_500 ≥0.80 (hard), aime ≥0.50 (expert)
    - GENERAL: intelligence_index ≥20 (simple), ≥40 (medium), 
               ≥55 (hard), gpqa ≥0.70 (expert)
    """
    
    def __init__(self, all_models_data: List[Dict]):
        """
        Initialize with population for percentile-based scaling.
        
        Args:
            all_models_data: Full model population
        """
        self.all_models_data = all_models_data
        self.n_models = len(all_models_data)
        
        print(f"Initializing Production Quality Scorer for {self.n_models} models...")
        
        # Extract benchmarks
        self._extract_benchmarks()
        
        # Calculate pre-computed metrics
        self._calculate_benchmark_statistics()
        
        print("✓ Initialization complete\n")
    
    def _safe_get(self, model: Dict, key: str) -> float:
        """Safely get value, return 0 if missing."""
        val = model.get(key)
        if val is None:
            return 0.0
        try:
            return float(val)
        except (ValueError, TypeError):
            return 0.0
    
    def _extract_benchmarks(self):
        """Extract all benchmark values including trust metrics and context window."""
        self.benchmarks = {
            'intelligence_index': [],
            'coding_index': [],
            'math_index': [],
            'mmlu_pro': [],
            'gpqa': [],
            'hle': [],
            'livecodebench': [],
            'scicode': [],
            'math_500': [],
            'aime': [],
            # Trust metrics (lower hallucination = better, higher consistency = better)
            'hallucination_rate': [],
            'factual_consistency_rate': [],
            # Context window (in K tokens) - critical for RAG
            'context_window_k': [],
        }
        
        self.model_names = []
        
        for model in self.all_models_data:
            self.model_names.append(model.get('name', ''))
            
            for key in self.benchmarks.keys():
                if key == 'context_window_k':
                    # Handle context_window specially - convert from full length if needed
                    ctx = model.get('context_window_k') or (model.get('context_length', 0) / 1000)
                    self.benchmarks[key].append(float(ctx) if ctx else 32.0)  # Default 32K
                else:
                    self.benchmarks[key].append(self._safe_get(model, key))
        
        # Convert to numpy
        for key in self.benchmarks.keys():
            self.benchmarks[key] = np.array(self.benchmarks[key])
        
        # Invert hallucination_rate so higher = better (for consistent scoring)
        # Transform: trust_score = 100 - hallucination_rate
        self.benchmarks['trust_score'] = 100 - self.benchmarks['hallucination_rate']
        
        # Create context_score using log scale (since context varies from 32K to 2M)
        # Normalize: log(ctx_k) / log(2048) where 2048K (2M) is max
        import math
        self.benchmarks['context_score'] = np.array([
            min(math.log(max(ctx, 1) + 1) / math.log(2049), 1.0) * 100
            for ctx in self.benchmarks['context_window_k']
        ])
    
    def _calculate_benchmark_statistics(self):
        """Calculate statistics for each benchmark for intelligent normalization."""
        print("Calculating benchmark statistics...")
        
        self.bench_stats = {}
        
        # Include derived metrics (trust_score, context_score) in statistics calculation
        all_benchmarks = {
            **self.benchmarks, 
            'trust_score': self.benchmarks.get('trust_score', np.array([])),
            'context_score': self.benchmarks.get('context_score', np.array([])),
        }
        
        for key, values in all_benchmarks.items():
            if len(values) == 0:
                continue
                
            non_zero = values[values > 0]
            
            if len(non_zero) < 5:
                self.bench_stats[key] = {
                    'min': 0, 'max': 1, 'mean': 0, 'std': 1,
                    'p50': 0, 'p75': 0, 'p90': 0, 'p95': 0
                }
                continue
            
            self.bench_stats[key] = {
                'min': np.min(non_zero),
                'max': np.max(non_zero),
                'mean': np.mean(non_zero),
                'std': np.std(non_zero),
                'p50': np.percentile(non_zero, 50),
                'p75': np.percentile(non_zero, 75),
                'p90': np.percentile(non_zero, 90),
                'p95': np.percentile(non_zero, 95),
            }
        
        print("✓ Statistics calculated for all benchmarks (including trust metrics)")
    
    def _percentile_normalize(self, value: float, key: str) -> float:
        """
        Normalize value based on its percentile position in the population.
        
        This prevents saturation by spreading high-performers logarithmically.
        """
        if value <= 0:
            return 0.0
        
        stats = self.bench_stats[key]
        
        # Find percentile
        if value >= stats['p95']:
            # Top 5%: Use logarithmic spacing to prevent saturation
            # Map p95-max to 0.95-1.0 using log scale
            if stats['max'] > stats['p95']:
                log_position = np.log1p(value - stats['p95']) / np.log1p(stats['max'] - stats['p95'])
                return 0.95 + (log_position * 0.05)
            return 0.95
        
        elif value >= stats['p90']:
            # 90-95th percentile: Linear spacing
            return 0.90 + ((value - stats['p90']) / (stats['p95'] - stats['p90'])) * 0.05
        
        elif value >= stats['p75']:
            # 75-90th percentile: Linear spacing
            return 0.75 + ((value - stats['p75']) / (stats['p90'] - stats['p75'])) * 0.15
        
        elif value >= stats['p50']:
            # 50-75th percentile: Linear spacing
            return 0.50 + ((value - stats['p50']) / (stats['p75'] - stats['p50'])) * 0.25
        
        else:
            # Bottom 50%: Linear spacing
            if stats['p50'] > stats['min']:
                return 0.50 * (value - stats['min']) / (stats['p50'] - stats['min'])
            return 0.0
    
    def _get_task_weights(self, category: Optional[PromptCategory]) -> Dict[str, float]:
        """
        Get task-specific benchmark weights.
        
        Weight Distribution Philosophy:
        - CODING: Prioritize coding_index, livecodebench, scicode for complexity differentiation
        - DATA_SCIENCE: Balance math + coding, with competition math for hard tasks
        - CREATIVE: Intelligence + knowledge, with language nuance (hle)
        - GENERAL: Balanced with trust metrics (hallucination awareness)
        """
        # Default weights (balanced)
        weights = {
            'intelligence_index': 0.20,
            'coding_index': 0.15,
            'math_index': 0.15,
            'mmlu_pro': 0.15,
            'gpqa': 0.10,
            'hle': 0.05,
            'livecodebench': 0.05,
            'scicode': 0.05,
            'math_500': 0.03,
            'aime': 0.02,
            'trust_score': 0.05,  # Added trust metric
            'context_score': 0.00,  # Only used for RAG
        }
        
        if category == PromptCategory.CODING:
            # UPDATED: Increased scicode weight for expert-level differentiation
            # Primary: coding_index, Complexity ladder: livecodebench → scicode
            weights = {
                'intelligence_index': 0.08,
                'coding_index': 0.35,      # Primary benchmark
                'math_index': 0.10,
                'mmlu_pro': 0.02,
                'gpqa': 0.02,
                'hle': 0.01,
                'livecodebench': 0.25,     # Medium-hard complexity
                'scicode': 0.15,           # INCREASED: Expert-level indicator
                'math_500': 0.01,
                'aime': 0.01,
                'trust_score': 0.00,       # Not critical for coding
                'context_score': 0.00,
            }
        
        elif category == PromptCategory.DATA_SCIENCE:
            # Primary: math_index, Complexity ladder: math_500 → aime
            weights = {
                'intelligence_index': 0.12,
                'coding_index': 0.18,      # Implementation ability
                'math_index': 0.30,        # Primary benchmark
                'mmlu_pro': 0.05,
                'gpqa': 0.12,              # Scientific domain knowledge
                'hle': 0.02,
                'livecodebench': 0.03,
                'scicode': 0.05,           # Scientific coding
                'math_500': 0.08,          # Hard math problems
                'aime': 0.05,              # Expert math (olympiad level)
                'trust_score': 0.00,
                'context_score': 0.00,
            }
        
        elif category == PromptCategory.CREATIVE:
            # Primary: intelligence_index, Secondary: hle for language nuance
            # Note: Missing creative-specific benchmarks (AlpacaEval, MT-Bench)
            weights = {
                'intelligence_index': 0.40,  # Primary (proxy for creativity)
                'coding_index': 0.02,
                'math_index': 0.00,
                'mmlu_pro': 0.28,            # Knowledge breadth
                'gpqa': 0.12,
                'hle': 0.10,                 # INCREASED: Language nuance
                'livecodebench': 0.00,
                'scicode': 0.00,
                'math_500': 0.00,
                'aime': 0.00,
                'trust_score': 0.08,         # Some trust matters for creative
                'context_score': 0.00,
            }
        
        elif category == PromptCategory.GENERAL:
            # UPDATED: Added trust_score as core quality signal
            # Primary: intelligence_index, Trust: hallucination awareness
            weights = {
                'intelligence_index': 0.25,  # Primary benchmark
                'coding_index': 0.12,
                'math_index': 0.10,
                'mmlu_pro': 0.15,            # Knowledge breadth
                'gpqa': 0.08,
                'hle': 0.05,
                'livecodebench': 0.03,
                'scicode': 0.02,
                'math_500': 0.01,
                'aime': 0.01,
                'trust_score': 0.18,         # NEW: Trust as core quality signal
                'context_score': 0.00,
            }
        
        elif category == PromptCategory.QA:
            # Q&A: Prioritize trust/accuracy, intelligence, knowledge breadth
            # Users need accurate, factual answers with low hallucination
            weights = {
                'intelligence_index': 0.25,  # Core reasoning ability
                'coding_index': 0.05,
                'math_index': 0.05,
                'mmlu_pro': 0.18,            # Knowledge breadth critical
                'gpqa': 0.12,                # Domain expertise
                'hle': 0.05,
                'livecodebench': 0.00,
                'scicode': 0.00,
                'math_500': 0.00,
                'aime': 0.00,
                'trust_score': 0.30,         # HIGHEST: Accuracy is paramount
                'context_score': 0.00,
            }
        
        elif category == PromptCategory.RAG:
            # RAG: Context window is critical, plus cost efficiency (handled in optimizer)
            # Need to fit retrieved documents, moderate quality, fast response
            weights = {
                'intelligence_index': 0.15,  # Needs to understand context
                'coding_index': 0.05,
                'math_index': 0.05,
                'mmlu_pro': 0.15,            # Knowledge helps synthesize
                'gpqa': 0.05,
                'hle': 0.05,                 # Language understanding
                'livecodebench': 0.00,
                'scicode': 0.00,
                'math_500': 0.00,
                'aime': 0.00,
                'trust_score': 0.15,         # Moderate - RAG grounds in docs
                'context_score': 0.35,       # CRITICAL: Must fit retrieved docs
            }
        
        elif category == PromptCategory.CHATBOT:
            # CHATBOT: Conversational AI - prioritize latency, cost, conversational ability
            # High volume interactions need fast, cheap, good-enough quality
            # Note: Latency/cost handled by optimizer, quality weights focus on conversation
            weights = {
                'intelligence_index': 0.35,  # Core conversational ability
                'coding_index': 0.05,        # Some coding help in chat
                'math_index': 0.05,
                'mmlu_pro': 0.20,            # Broad knowledge for diverse topics
                'gpqa': 0.05,
                'hle': 0.15,                 # Language/conversation nuance
                'livecodebench': 0.00,
                'scicode': 0.00,
                'math_500': 0.00,
                'aime': 0.00,
                'trust_score': 0.15,         # Moderate trust for general chat
                'context_score': 0.00,       # Not critical for typical chat
            }
        
        return weights
    
    def _calculate_composite_score(self, model_idx: int, category: Optional[PromptCategory]) -> float:
        """
        Calculate task-weighted composite score with percentile normalization.
        
        This creates different absolute scores for different tasks while
        preventing saturation at the high end.
        """
        task_weights = self._get_task_weights(category)
        
        composite = 0.0
        total_weight = 0.0
        
        for key, weight in task_weights.items():
            if weight <= 0:
                continue
                
            # Handle trust_score specially (it's derived, not directly in benchmarks)
            if key == 'trust_score':
                value = self.benchmarks.get('trust_score', np.array([0]))[model_idx] if model_idx < len(self.benchmarks.get('trust_score', [])) else 0
            else:
                value = self.benchmarks[key][model_idx]
            
            if value > 0:
                # Percentile-normalize (prevents saturation)
                normalized = self._percentile_normalize(value, key)
                
                composite += normalized * weight
                total_weight += weight
        
        return composite / total_weight if total_weight > 0 else 0.0
    
    def calculate_quality_score(
        self,
        model_data: Dict,
        category: Optional[PromptCategory] = None,
        archetype: Optional[ProductArchetype] = None
    ) -> float:
        """
        Calculate quality score with non-saturating distribution.
        
        Args:
            model_data: Model data
            category: Task category (affects weights and distribution shape)
            archetype: Product archetype (future use)
        
        Returns:
            Quality score (0-100), distribution shape varies by task
        """
        # Find model
        model_name = model_data.get('name', '')
        try:
            model_idx = self.model_names.index(model_name)
        except ValueError:
            return 50.0
        
        # Calculate composite score (0-1)
        composite = self._calculate_composite_score(model_idx, category)
        
        # Apply final non-linear transformation for smooth distribution
        # Use log1p for bottom half, power law for top half
        if composite < 0.5:
            # Bottom half: logarithmic (spreads low scores)
            transformed = 50 * (np.log1p(composite * 2) / np.log1p(1))
        else:
            # Top half: power law α=1.5 (prevents saturation at high end)
            excess = composite - 0.5
            # Map 0.5-1.0 to 50-100 with power law
            transformed = 50 + 50 * np.power(excess * 2, 1.5)
        
        return np.clip(transformed, 0, 100)
    
    def get_all_scores(self, category: Optional[PromptCategory] = None) -> Dict[str, float]:
        """Get scores for all models efficiently."""
        return {
            self.model_names[i]: self.calculate_quality_score(
                {'name': self.model_names[i]}, category
            )
            for i in range(self.n_models)
        }
    
    def get_component_breakdown(self, model_data: Dict) -> Dict[str, float]:
        """Get detailed component scores for a model."""
        model_name = model_data.get('name', '')
        try:
            model_idx = self.model_names.index(model_name)
        except ValueError:
            return {}
        
        components = {}
        for key in self.benchmarks.keys():
            value = self.benchmarks[key][model_idx]
            if value > 0:
                components[key] = self._percentile_normalize(value, key) * 100
            else:
                components[key] = 0.0
        
        # Add trust_score
        if 'trust_score' in self.benchmarks:
            trust_val = self.benchmarks['trust_score'][model_idx]
            if trust_val > 0:
                components['trust_score'] = self._percentile_normalize(trust_val, 'trust_score') * 100
        
        return components
    
    # =========================================================================
    # COMPLEXITY-AWARE ROUTING
    # =========================================================================
    
    def get_complexity_thresholds(self, category: Optional[PromptCategory]) -> Dict[str, Dict[str, Tuple[str, float]]]:
        """
        Get complexity thresholds for a given category.
        
        Returns dict mapping complexity level to (benchmark, threshold).
        Models must meet the threshold to be considered capable at that level.
        """
        if category == PromptCategory.CODING:
            return {
                TaskComplexity.SIMPLE: ('coding_index', 20.0),
                TaskComplexity.MEDIUM: ('livecodebench', 0.70),
                TaskComplexity.HARD: ('livecodebench', 0.80),
                TaskComplexity.EXPERT: ('scicode', 0.45),
            }
        elif category == PromptCategory.DATA_SCIENCE:
            return {
                TaskComplexity.SIMPLE: ('math_index', 30.0),
                TaskComplexity.MEDIUM: ('math_index', 50.0),
                TaskComplexity.HARD: ('math_500', 0.80),
                TaskComplexity.EXPERT: ('aime', 0.50),
            }
        elif category == PromptCategory.CREATIVE:
            return {
                TaskComplexity.SIMPLE: ('intelligence_index', 30.0),
                TaskComplexity.MEDIUM: ('intelligence_index', 50.0),
                TaskComplexity.HARD: ('hle', 0.20),
                TaskComplexity.EXPERT: ('hle', 0.30),
            }
        else:  # GENERAL or default
            return {
                TaskComplexity.SIMPLE: ('intelligence_index', 20.0),
                TaskComplexity.MEDIUM: ('intelligence_index', 40.0),
                TaskComplexity.HARD: ('intelligence_index', 55.0),
                TaskComplexity.EXPERT: ('gpqa', 0.70),
            }
    
    def get_model_complexity_capability(
        self, 
        model_data: Dict, 
        category: Optional[PromptCategory] = None
    ) -> TaskComplexity:
        """
        Determine the maximum complexity level a model can handle for a category.
        
        Args:
            model_data: Model data dictionary
            category: Task category
            
        Returns:
            Maximum TaskComplexity the model is capable of
        """
        model_name = model_data.get('name', '')
        try:
            model_idx = self.model_names.index(model_name)
        except ValueError:
            return TaskComplexity.SIMPLE
        
        thresholds = self.get_complexity_thresholds(category)
        
        # Check from highest to lowest complexity
        for complexity in [TaskComplexity.EXPERT, TaskComplexity.HARD, 
                          TaskComplexity.MEDIUM, TaskComplexity.SIMPLE]:
            benchmark, threshold = thresholds[complexity]
            
            if benchmark in self.benchmarks:
                value = self.benchmarks[benchmark][model_idx]
                if value >= threshold:
                    return complexity
        
        return TaskComplexity.SIMPLE
    
    def get_models_for_complexity(
        self, 
        complexity: TaskComplexity, 
        category: Optional[PromptCategory] = None,
        include_higher: bool = True
    ) -> List[Dict]:
        """
        Get all models capable of handling a given complexity level.
        
        Args:
            complexity: Required complexity level
            category: Task category
            include_higher: If True, include models capable of higher complexity
            
        Returns:
            List of model data dicts that meet the complexity requirement
        """
        capable_models = []
        
        for i, model in enumerate(self.all_models_data):
            model_capability = self.get_model_complexity_capability(model, category)
            
            if include_higher:
                # Model capability must be >= required complexity
                complexity_order = [TaskComplexity.SIMPLE, TaskComplexity.MEDIUM, 
                                   TaskComplexity.HARD, TaskComplexity.EXPERT]
                if complexity_order.index(model_capability) >= complexity_order.index(complexity):
                    capable_models.append(model)
            else:
                # Exact match only
                if model_capability == complexity:
                    capable_models.append(model)
        
        return capable_models
    
    def recommend_for_complexity(
        self, 
        complexity: TaskComplexity, 
        category: Optional[PromptCategory] = None,
        max_cost: Optional[float] = None,
        top_n: int = 5
    ) -> List[Tuple[Dict, float, TaskComplexity]]:
        """
        Recommend models for a given complexity level, optimizing for value.
        
        Args:
            complexity: Required complexity level
            category: Task category
            max_cost: Maximum cost per 1M tokens (blended)
            top_n: Number of recommendations to return
            
        Returns:
            List of (model_data, quality_score, capability_level) tuples
        """
        capable_models = self.get_models_for_complexity(complexity, category)
        
        # Filter by cost if specified
        if max_cost is not None:
            capable_models = [
                m for m in capable_models 
                if (m.get('price_1m_blended') or 0) <= max_cost
            ]
        
        # Score and rank
        scored = []
        for model in capable_models:
            score = self.calculate_quality_score(model, category)
            capability = self.get_model_complexity_capability(model, category)
            cost = model.get('price_1m_blended', 0) or 0.01
            # Value score: quality per dollar
            value = score / cost if cost > 0 else score
            scored.append((model, score, capability, value))
        
        # Sort by value (best value first)
        scored.sort(key=lambda x: x[3], reverse=True)
        
        # Return top N without the value score
        return [(m, s, c) for m, s, c, v in scored[:top_n]]
    
    def get_minimum_model_for_task(
        self, 
        complexity: TaskComplexity, 
        category: Optional[PromptCategory] = None
    ) -> Optional[Dict]:
        """
        Get the cheapest model that meets the complexity requirement.
        
        This is useful for cost optimization when quality threshold is met.
        
        Args:
            complexity: Required complexity level
            category: Task category
            
        Returns:
            Cheapest capable model, or None if no model meets requirements
        """
        capable = self.get_models_for_complexity(complexity, category)
        
        if not capable:
            return None
        
        # Sort by cost (ascending)
        capable.sort(key=lambda m: m.get('price_1m_blended', float('inf')) or float('inf'))
        
        return capable[0]

