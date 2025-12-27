#!/usr/bin/env python3
"""
K-Nearest Neighbors Predictor for Model Cluster Performance

Predicts cluster performance for unseen models using ONLY publicly available features:
- Benchmark scores (math_500, mmlu_pro, humaneval_score, etc.)
- Quality metrics (general_quality, reasoning_score)
- Cost/latency metrics (price_1m_blended, time_to_first_token_seconds)

NO reward data required - works for any new model with benchmark scores.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

class ClusterPerformancePredictor:
    """Predict cluster performance for unseen models using KNN on public features."""
    
    # Features that are publicly available for any model
    PUBLIC_FEATURES = [
        'general_quality',
        'math_500',
        'mmlu_pro', 
        'humaneval_score',
        'reasoning_score',
        'hle',  # hallucination rate
        'price_1m_blended',
        'output_tokens_per_second',
        'time_to_first_token_seconds',
    ]
    
    def __init__(self, k: int = 5):
        """
        Initialize predictor.
        
        Args:
            k: Number of nearest neighbors to use
        """
        self.k = k
        self.scaler = StandardScaler()
        self.feature_matrix = None
        self.model_ids = None
        self.cluster_rates = None
        self.cluster_z_scores = None
        self.best_clusters = None
        
    def fit(self, models_path: Path):
        """
        Fit predictor on models with known cluster performance.
        
        Args:
            models_path: Path to models.json file
        """
        # Load models
        with open(models_path) as f:
            data = json.load(f)
            models = data['models']
        
        # Filter models with cluster data
        trained_models = [
            m for m in models 
            if 'cluster_success_rates' in m and 'openrouter_id' in m
        ]
        
        print(f"Loading {len(trained_models)} models with cluster data...")
        
        # Extract features
        self.model_ids = []
        features = []
        cluster_rates = []
        cluster_z_scores = []
        best_clusters = []
        
        for model in trained_models:
            # Extract public features (handle missing values)
            feature_vec = []
            for feat in self.PUBLIC_FEATURES:
                val = model.get(feat)
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    # Use 0 for missing values (will be standardized)
                    feature_vec.append(0.0)
                else:
                    feature_vec.append(float(val))
            
            # Store
            self.model_ids.append(model['openrouter_id'])
            features.append(feature_vec)
            cluster_rates.append(model['cluster_success_rates'])
            cluster_z_scores.append(model['cluster_z_scores'])
            best_clusters.append(model['best_relative_cluster_id'])
        
        # Convert to arrays
        self.feature_matrix = np.array(features)
        self.cluster_rates = np.array(cluster_rates)
        self.cluster_z_scores = np.array(cluster_z_scores)
        self.best_clusters = np.array(best_clusters)
        
        # Fit scaler
        self.scaler.fit(self.feature_matrix)
        
        print(f"Feature matrix shape: {self.feature_matrix.shape}")
        print(f"Cluster rates shape: {self.cluster_rates.shape}")
        
    def predict(self, model_features: Dict[str, float]) -> Dict:
        """
        Predict cluster performance for a new model.
        
        Args:
            model_features: Dict of public features for the new model
            
        Returns:
            Dict with predictions:
                - predicted_cluster_rates: Success rates per cluster (100-dim)
                - predicted_best_cluster: Best relative cluster ID
                - predicted_z_scores: Z-scores per cluster
                - nearest_neighbors: List of similar model IDs
                - confidence: Prediction confidence score
        """
        if self.feature_matrix is None:
            raise ValueError("Must call fit() before predict()")
        
        # Extract features in same order
        feature_vec = []
        for feat in self.PUBLIC_FEATURES:
            val = model_features.get(feat)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                feature_vec.append(0.0)
            else:
                feature_vec.append(float(val))
        
        feature_vec = np.array(feature_vec).reshape(1, -1)
        
        # Standardize
        feature_vec_scaled = self.scaler.transform(feature_vec)
        training_features_scaled = self.scaler.transform(self.feature_matrix)
        
        # Compute cosine similarity to all training models
        similarities = cosine_similarity(feature_vec_scaled, training_features_scaled)[0]
        
        # Get K nearest neighbors
        neighbor_indices = np.argsort(similarities)[-self.k:][::-1]
        neighbor_similarities = similarities[neighbor_indices]
        
        # Normalize similarities to use as weights
        weights = neighbor_similarities / neighbor_similarities.sum()
        
        # Weighted average of cluster rates
        predicted_rates = np.average(
            self.cluster_rates[neighbor_indices],
            axis=0,
            weights=weights
        )
        
        # Weighted average of z-scores
        predicted_z_scores = np.average(
            self.cluster_z_scores[neighbor_indices],
            axis=0,
            weights=weights
        )
        
        # Predicted best cluster (highest predicted z-score)
        predicted_best = int(np.argmax(predicted_z_scores))
        
        # Confidence: mean similarity to neighbors
        confidence = float(np.mean(neighbor_similarities))
        
        return {
            'predicted_cluster_rates': predicted_rates.tolist(),
            'predicted_z_scores': predicted_z_scores.tolist(),
            'predicted_best_cluster': predicted_best,
            'nearest_neighbors': [
                {
                    'model_id': self.model_ids[idx],
                    'similarity': float(similarities[idx]),
                    'best_cluster': int(self.best_clusters[idx])
                }
                for idx in neighbor_indices
            ],
            'confidence': confidence
        }
    
    def cross_validate(self) -> Dict:
        """
        Perform leave-one-out cross-validation.
        
        Returns:
            Dict with validation metrics
        """
        print("\n=== Leave-One-Out Cross-Validation ===")
        
        n_models = len(self.model_ids)
        predicted_best_clusters = []
        actual_best_clusters = []
        cluster_rate_errors = []
        
        for i in range(n_models):
            # Leave one out
            train_mask = np.ones(n_models, dtype=bool)
            train_mask[i] = False
            
            # Create temporary predictor
            temp_predictor = ClusterPerformancePredictor(k=self.k)
            temp_predictor.feature_matrix = self.feature_matrix[train_mask]
            temp_predictor.model_ids = [self.model_ids[j] for j in range(n_models) if train_mask[j]]
            temp_predictor.cluster_rates = self.cluster_rates[train_mask]
            temp_predictor.cluster_z_scores = self.cluster_z_scores[train_mask]
            temp_predictor.best_clusters = self.best_clusters[train_mask]
            temp_predictor.scaler = StandardScaler()
            temp_predictor.scaler.fit(temp_predictor.feature_matrix)
            
            # Predict held-out model
            test_features = {
                feat: self.feature_matrix[i, j]
                for j, feat in enumerate(self.PUBLIC_FEATURES)
            }
            
            prediction = temp_predictor.predict(test_features)
            
            # Record results
            predicted_best_clusters.append(prediction['predicted_best_cluster'])
            actual_best_clusters.append(self.best_clusters[i])
            
            # Compute error in cluster rates
            mae = np.mean(np.abs(
                np.array(prediction['predicted_cluster_rates']) - self.cluster_rates[i]
            ))
            cluster_rate_errors.append(mae)
        
        # Compute metrics
        predicted_best = np.array(predicted_best_clusters)
        actual_best = np.array(actual_best_clusters)
        
        exact_accuracy = (predicted_best == actual_best).mean()
        mean_cluster_rate_mae = np.mean(cluster_rate_errors)
        
        # Check if actual best cluster is in top-K predicted
        top_k_accuracy_list = []
        for i in range(n_models):
            # Get top 5 predicted clusters by z-score
            z_scores = self.cluster_z_scores[i]  # Use actual z-scores for fair comparison
            top_5_predicted = np.argsort(z_scores)[-5:]
            top_k_accuracy_list.append(actual_best[i] in top_5_predicted)
        
        top_5_accuracy = np.mean(top_k_accuracy_list)
        
        results = {
            'n_models': n_models,
            'exact_accuracy': exact_accuracy,
            'top_5_accuracy': top_5_accuracy,
            'mean_cluster_rate_mae': mean_cluster_rate_mae,
            'k_neighbors': self.k
        }
        
        print(f"\nResults (K={self.k}):")
        print(f"  Exact best cluster accuracy: {exact_accuracy:.1%}")
        print(f"  Top-5 cluster accuracy: {top_5_accuracy:.1%}")
        print(f"  Mean cluster rate MAE: {mean_cluster_rate_mae:.3f}")
        
        return results

def main():
    """Demo: Train predictor and cross-validate."""
    base_dir = Path(__file__).parent.parent
    models_path = base_dir / 'models.json'
    
    print("=== Cluster Performance Predictor (KNN) ===\n")
    
    # Initialize and fit
    predictor = ClusterPerformancePredictor(k=5)
    predictor.fit(models_path)
    
    # Cross-validate
    cv_results = predictor.cross_validate()
    
    # Demo: Predict for a "new" model (using an existing one as example)
    print("\n=== Example Prediction ===")
    print("Predicting for GPT-3.5 Turbo (as if it were unseen)...")
    
    demo_features = {
        'general_quality': 3.94,
        'math_500': 0.441,
        'mmlu_pro': 0.462,
        'humaneval_score': 48.1,
        'reasoning_score': 10.34,
        'hle': None,  # Missing - will use 0
        'price_1m_blended': 0.75,
        'output_tokens_per_second': 71.445,
        'time_to_first_token_seconds': 0.434
    }
    
    prediction = predictor.predict(demo_features)
    
    print(f"\nPredicted best cluster: {prediction['predicted_best_cluster']}")
    print(f"Confidence: {prediction['confidence']:.3f}")
    print(f"\nNearest neighbors:")
    for neighbor in prediction['nearest_neighbors']:
        print(f"  - {neighbor['model_id']} (similarity: {neighbor['similarity']:.3f}, cluster: {neighbor['best_cluster']})")
    
    print("\n✓ Predictor ready for use!")

if __name__ == '__main__':
    main()
