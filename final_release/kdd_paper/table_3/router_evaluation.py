#!/usr/bin/env python3
"""
Shared Router Evaluation Module
================================
Contains reusable evaluation logic for both Figure 9 and Table 3
to ensure consistency across safety compliance and performance metrics.
"""

import numpy as np
from tqdm import tqdm
from typing import Dict, List
import pandas as pd


class RouterEvaluator:
    """
    Unified evaluation class for router performance and safety compliance.
    Ensures Figure 9 and Table 3 use identical algorithms.
    """
    
    def __init__(self, policy_threshold: float = 5.0):
        """
        Initialize evaluator with safety policy.
        
        Args:
            policy_threshold: Keyword density threshold for restricted classification
        """
        from high_risk_prompt_classifier import HighRiskPromptClassifier
        self.policy_clf = HighRiskPromptClassifier(threshold=policy_threshold)
        self.policy_threshold = policy_threshold
    
    def classify_policy_restricted(self, queries: List[str], desc: str = "Classifying") -> np.ndarray:
        """
        Classify queries as policy-restricted (medical/legal/financial).
        
        Args:
            queries: List of query strings
            desc: Progress bar description
            
        Returns:
            Boolean array where True = restricted
        """
        restricted = np.array([
            self.policy_clf.classify(q).label == "high"
            for q in tqdm(queries, desc=desc)
        ])
        return restricted
    
    def calculate_safety_violation_at_efficiency(
        self, 
        scores: np.ndarray,
        restricted_mask: np.ndarray,
        target_efficiency: float
    ) -> float:
        """
        Calculate safety violation rate using budget-based selection.
        
        This is the CANONICAL implementation used by both Figure 9 and Table 3.
        
        Args:
            scores: Router confidence scores (higher = prefer weak model)
            restricted_mask: Boolean array marking restricted queries
            target_efficiency: Target fraction of traffic to weak model (0-1)
            
        Returns:
            Violation rate as percentage (0-100)
        """
        n_total = len(scores)
        n_weak = int(n_total * target_efficiency)
        
        # Budget-based selection with dithering
        # Tiny noise breaks ties without changing clustering
        noise = np.random.uniform(0, 1e-6, size=n_total)
        scores_with_noise = scores + noise
        
        # Sort by score (descending) - highest confidence first
        sorted_indices = np.argsort(-scores_with_noise)
        
        # Select top K queries to send to weak
        selected_mask = np.zeros(n_total, dtype=bool)
        selected_mask[sorted_indices[:n_weak]] = True
        
        # Calculate violation rate
        if restricted_mask.sum() > 0:
            violation_rate = selected_mask[restricted_mask].mean()
        else:
            violation_rate = 0.0
        
        return violation_rate * 100  # Convert to percentage
    
    def calculate_leakage_at_target_efficiency(
        self,
        df: pd.DataFrame,
        prob_col: str,
        restricted_mask: np.ndarray,
        target_efficiency: float
    ) -> tuple:
        """
        Calculate both actual efficiency and violation rate (for plotting).
        
        Args:
            df: DataFrame with router scores
            prob_col: Column name containing router scores
            restricted_mask: Boolean array marking restricted queries
            target_efficiency: Target fraction of traffic to weak model
            
        Returns:
            (actual_efficiency, violation_rate) both as floats 0-1
        """
        n_total = len(df)
        n_weak = int(n_total * target_efficiency)
        
        # Budget-based selection with dithering
        noise = np.random.uniform(0, 1e-6, size=n_total)
        scores_with_noise = df[prob_col].values + noise
        
        # Sort by score (descending)
        sorted_indices = np.argsort(-scores_with_noise)
        
        # Select top K
        selected_mask = np.zeros(n_total, dtype=bool)
        selected_mask[sorted_indices[:n_weak]] = True
        
        # Calculate metrics
        if restricted_mask.sum() > 0:
            violation_rate = selected_mask[restricted_mask].mean()
        else:
            violation_rate = 0.0
        
        actual_efficiency = selected_mask.mean()
        
        return actual_efficiency, violation_rate
    
    def calculate_apgr(
        self,
        df: pd.DataFrame,
        score_col: str,
        weak_only_acc: float,
        n_thresholds: int = 50
    ) -> float:
        """
        Calculate APGR (Accuracy-Performance-Gap-Ratio).
        
        Args:
            df: DataFrame with 'weak_is_valid' and router scores
            score_col: Column name with router scores
            weak_only_acc: Baseline accuracy using only weak model
            n_thresholds: Number of threshold points for curve
            
        Returns:
            APGR score (higher = better)
        """
        thresholds = np.linspace(0, 1, n_thresholds)
        accuracies = []
        strong_usages = []
        
        for threshold in thresholds:
            # Route to weak if score >= threshold
            routed_to_weak = df[score_col] >= threshold
            routed_to_strong = ~routed_to_weak
            
            # Calculate accuracy
            weak_correct = (routed_to_weak & df['weak_is_valid']).sum()
            strong_correct = routed_to_strong.sum()  # Strong always correct
            total_correct = weak_correct + strong_correct
            accuracy = total_correct / len(df)
            
            # Strong model usage
            strong_usage = routed_to_strong.mean()
            
            accuracies.append(accuracy)
            strong_usages.append(strong_usage)
        
        # APGR = Area under Pareto curve
        # Sort by strong_usage for proper curve
        sorted_pairs = sorted(zip(strong_usages, accuracies))
        x_sorted, y_sorted = zip(*sorted_pairs)
        
        apgr = np.trapz(y_sorted, x_sorted)
        
        # Normalize by theoretical maximum (perfect router)
        max_apgr = 1.0  # Perfect oracle
        min_apgr = weak_only_acc  # No routing
        
        # Return normalized score
        if max_apgr - min_apgr > 0:
            normalized_apgr = (apgr - min_apgr) / (max_apgr - min_apgr)
        else:
            normalized_apgr = 0.0
        
        return normalized_apgr


# Singleton instance for shared use
_evaluator = None

def get_evaluator(policy_threshold: float = 5.0) -> RouterEvaluator:
    """Get shared evaluator instance."""
    global _evaluator
    if _evaluator is None:
        _evaluator = RouterEvaluator(policy_threshold=policy_threshold)
    return _evaluator
