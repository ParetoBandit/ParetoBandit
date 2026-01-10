"""
PCA Manager for BanditGPT Dimensionality Reduction

Handles PCA training, transformation, and prior covariance matrix generation
for the Hybrid feature architecture (PCA semantic + handcrafted + cluster features).
"""

import json
import numpy as np
import joblib
import re
from pathlib import Path
from typing import List, Tuple, Optional
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA


class FeatureExtractor:
    """Extracts handcrafted features from text prompts"""
    
    @staticmethod
    def count_syllables(word: str) -> int:
        """Estimate syllable count for Flesch-Kincaid calculation"""
        word = word.lower().strip(".:;?!")
        if not word: return 0
        if len(word) <= 3: return 1
        count = len(re.findall(r'[aeiouy]+', word))
        if word.endswith('e'): count -= 1
        return max(1, count)
    
    @staticmethod
    def extract_handcrafted_features(text: str) -> np.ndarray:
        """
        Extract 8 handcrafted complexity features
        
        Returns:
            Array of [is_code_heavy, requires_json, input_length_log, list_density,
                     instruction_density, fk_normalized, question_count, toxicity]
        """
        if not text:
            return np.zeros(8)
        
        total_len = len(text)
        words = re.findall(r'\b\w+\b', text.lower())
        n_words = len(words)
        lines = text.split('\n')
        n_lines = len(lines)
        
        # Code detection
        code_blocks = re.findall(r'`{1,3}(.*?)`{1,3}', text, re.DOTALL)
        code_len = sum(len(c) for c in code_blocks)
        is_code_heavy = (code_len / total_len) if total_len > 0 else 0.0
        
        # JSON requirement detection
        requires_json = 1.0 if any(k in text.lower() for k in ["json", "valid format", "schema"]) else 0.0
        
        # Token length (log scale)
        n_tokens = n_words * 1.3
        input_length_log = np.log(n_tokens + 1.0)
        
        # List density
        list_markers = [l for l in lines if l.strip().startswith(('-', '*', '1.', '2.'))]
        list_density = (len(list_markers) / n_lines) if n_lines > 0 else 0.0
        
        # Instruction density
        imperatives = {"create", "write", "solve", "analyze", "explain", "summarize", "find", "calculate", "implement", "design"}
        n_imperatives = sum(1 for w in words if w in imperatives)
        instruction_density = (n_imperatives / n_words) if n_words > 0 else 0.0
        
        # Flesch-Kincaid readability
        sentences = re.split(r'[.!?]+', text)
        n_sentences = max(1, len([s for s in sentences if s.strip()]))
        if n_words > 0:
            n_syllables = sum(FeatureExtractor.count_syllables(w) for w in words)
            fk_grade = 0.39 * (n_words / n_sentences) + 11.8 * (n_syllables / n_words) - 15.59
        else:
            fk_grade = 0.0
        fk_normalized = max(0.0, min(fk_grade, 20.0)) / 20.0
        
        # Question markers
        q_count = text.count('?')
        question_count = np.log(q_count + 1.0)
        
        # Toxicity (placeholder - would need external API)
        toxicity = 0.0
        
        return np.array([is_code_heavy, requires_json, input_length_log, list_density,
                        instruction_density, fk_normalized, question_count, toxicity])


class PCAManager:
    """Manages PCA dimensionality reduction for BanditGPT"""
    
    def __init__(self, n_components: int = 32, encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Args:
            n_components: Target PCA dimensions (default 32)
            encoder_model: SentenceTransformer model name
        """
        self.n_components = n_components
        self.encoder_model = encoder_model
        self.encoder = None
        self.pca = None
        self.feature_extractor = FeatureExtractor()
    
    def load_encoder(self):
        """Lazy load sentence transformer"""
        if self.encoder is None:
            print(f"Loading encoder: {self.encoder_model}")
            self.encoder = SentenceTransformer(self.encoder_model)
    
    def fit_pca(self, prompts: List[str], pca_path: Optional[Path] = None, 
                batch_size: int = 64) -> PCA:
        """
        Fit PCA on prompt embeddings
        
        Args:
            prompts: List of prompt strings
            pca_path: Optional path to save fitted PCA model
            batch_size: Embedding batch size
            
        Returns:
            Fitted PCA object
        """
        self.load_encoder()
        
        print(f"Embedding {len(prompts)} prompts...")
        embeddings = self.encoder.encode(
            prompts,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=batch_size
        )
        
        print(f"Fitting PCA ({embeddings.shape[1]} -> {self.n_components})...")
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(embeddings)
        
        explained_var = np.sum(self.pca.explained_variance_ratio_)
        print(f"Explained variance: {explained_var:.2%}")
        
        if pca_path:
            joblib.dump(self.pca, pca_path)
            print(f"Saved PCA to {pca_path}")
        
        return self.pca
    
    def load_pca(self, pca_path: Path):
        """Load pre-trained PCA model"""
        print(f"Loading PCA from {pca_path}")
        self.pca = joblib.load(pca_path)
        self.n_components = self.pca.n_components_
    
    def transform_prompts(self, prompts: List[str], batch_size: int = 64) -> np.ndarray:
        """
        Transform prompts to PCA space
        
        Args:
            prompts: List of prompt strings
            batch_size: Embedding batch size
            
        Returns:
            PCA-transformed embeddings
        """
        if self.pca is None:
            raise ValueError("PCA not fitted or loaded. Call fit_pca() or load_pca() first.")
        
        self.load_encoder()
        
        embeddings = self.encoder.encode(
            prompts,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=batch_size
        )
        
        return self.pca.transform(embeddings)
    
    def extract_hybrid_features(self, prompts: List[str], 
                                include_cluster_features: bool = False,
                                cluster_detector = None) -> np.ndarray:
        """
        Extract hybrid features: PCA + handcrafted + (optional) cluster distances
        
        Args:
            prompts: List of prompt strings
            include_cluster_features: Whether to include anchor cluster distances
            cluster_detector: ClusterDetector instance (required if include_cluster_features=True)
            
        Returns:
            Hybrid feature matrix (n_prompts, n_components + 8 + [5 if clusters])
        """
        # PCA features
        X_pca = self.transform_prompts(prompts)
        
        # Handcrafted features
        print("Extracting handcrafted features...")
        X_explicit = np.array([self.feature_extractor.extract_handcrafted_features(p) for p in prompts])
        
        # Cluster features (optional)
        if include_cluster_features:
            if cluster_detector is None:
                raise ValueError("cluster_detector required when include_cluster_features=True")
            
            print("Computing anchor cluster distances...")
            self.load_encoder()
            embeddings = self.encoder.encode(prompts, normalize_embeddings=True, batch_size=64)
            X_clusters = np.array([cluster_detector.get_anchor_distances(emb) for emb in embeddings])
        else:
            X_clusters = np.zeros((len(prompts), 5))
        
        # Concatenate all features
        X_hybrid = np.concatenate([X_pca, X_explicit, X_clusters], axis=1)
        
        print(f"Hybrid features shape: {X_hybrid.shape} (PCA:{X_pca.shape[1]} + Explicit:{X_explicit.shape[1]} + Clusters:{X_clusters.shape[1]})")
        
        return X_hybrid
    
    def generate_prior_covariance(self, prompts: List[str], 
                                  output_path: Path,
                                  include_cluster_features: bool = False,
                                  cluster_detector = None,
                                  n_clusters: int = 100):
        """
        Generate prior covariance matrix from prompt set
        
        Args:
            prompts: List of prompts for prior estimation
            output_path: Path to save .npz file
            include_cluster_features: Whether to include cluster features
            cluster_detector: ClusterDetector instance
            n_clusters: Number of clusters for cluster_sums
        """
        # Extract hybrid features
        X_hybrid = self.extract_hybrid_features(prompts, include_cluster_features, cluster_detector)
        
        # Compute statistics
        print("Computing covariance matrix...")
        sum_vec = np.sum(X_hybrid, axis=0)
        cov_matrix = X_hybrid.T @ X_hybrid
        
        # Cluster sums (zeros for now - would need cluster assignments)
        cluster_sums = np.zeros((n_clusters, X_hybrid.shape[1]))
        cluster_counts = np.zeros(n_clusters)
        
        # Save
        print(f"Saving to {output_path}")
        np.savez(
            output_path,
            cov_matrix=cov_matrix,
            sum_vec=sum_vec,
            cluster_sums=cluster_sums,
            cluster_counts=cluster_counts,
            global_sum=sum_vec
        )
        
        print(f"✓ Saved prior covariance ({X_hybrid.shape[1]}x{X_hybrid.shape[1]})")


# Convenience function
def train_pca_pipeline(source_prompts_path: Path,
                      exclusion_paths: List[Path],
                      output_dir: Path,
                      n_components: int = 32,
                      max_prompts: int = 25000):
    """
    Complete PCA training pipeline with leakage prevention
    
    Args:
        source_prompts_path: Path to source prompts JSONL
        exclusion_paths: Paths to train/test sets to exclude
        output_dir: Directory for outputs
        n_components: PCA dimensions
        max_prompts: Maximum prompts to use
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load exclusions
    exclusion_prompts = set()
    for path in exclusion_paths:
        if path.exists():
            print(f"Loading exclusions from {path.name}...")
            with open(path) as f:
                for line in f:
                    try:
                        exclusion_prompts.add(json.loads(line)["prompt"])
                    except:
                        pass
    
    print(f"Excluding {len(exclusion_prompts)} prompts (train/test leakage prevention)")
    
    # Load safe prompts
    prompts = []
    import gzip
    open_fn = gzip.open if str(source_prompts_path).endswith(".gz") else open
    with open_fn(source_prompts_path, "rt") as f:
        for line in f:
            data = json.loads(line)
            if data["prompt"] not in exclusion_prompts:
                prompts.append(data["prompt"])
            if len(prompts) >= max_prompts:
                break
    
    print(f"Using {len(prompts)} safe prompts")
    
    # Train PCA
    manager = PCAManager(n_components=n_components)
    pca_path = output_dir / f"pca_{n_components}.joblib"
    manager.fit_pca(prompts, pca_path=pca_path)
    
    # Generate priors
    priors_path = output_dir / "priors_meta_pca.npz"
    manager.generate_prior_covariance(prompts, priors_path)
    
    print(f"\n✓ Pipeline complete!")
    print(f"  PCA model: {pca_path}")
    print(f"  Prior covariance: {priors_path}")
