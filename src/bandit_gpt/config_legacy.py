"""
RouterConfig: Production-Ready Configuration for BanditRouter

This module provides Pydantic validation for the BanditRouter architecture,
enforcing:
- Valid Virtual Anchors (semantic neighborhoods)
- Safe Regex Features (structural skip connections)
- Consistent Intuition Weights (warm start priors)

Usage:
    from banditgpt.config import RouterConfig
    
    config = RouterConfig(
        anchors={"coding": "python java SQL algorithm...", ...},
        complexity_mean=-0.0037,  # Calibrated on N=1000 LMSYS
        complexity_std=0.095
    )
    router = BanditRouter.from_config(config, model_registry)
"""

from typing import Dict, List, Optional, Literal
from pydantic import BaseModel, Field, field_validator, model_validator
import re


# --- 1. Type Definitions ---

TransformType = Literal["binarize", "log1p", "sigmoid", "minmax"]
FeatureSource = Literal["regex_count", "token_count", "complexity_projection", "embedding_projection"]


# --- 2. Feature Definitions (Structural Skip Connections) ---

class StructuralFeature(BaseModel):
    """
    Defines a 'Structural Skip Connection' feature.
    
    These features bypass the embedding layer to capture syntax that embeddings miss:
    - Code blocks (```)
    - LaTeX markers ($, \\begin)
    - Question marks
    
    The 'transforms' field ensures LinUCB linearity assumption is satisfied.
    """
    name: str = Field(..., description="Feature name (e.g., 'latex_density')")
    source: FeatureSource = Field(..., description="How to extract the raw signal")
    pattern: Optional[str] = Field(None, description="Regex pattern (required for regex_count)")
    transforms: List[TransformType] = Field(
        default_factory=lambda: ["log1p"],
        description="Transforms to apply for linearity. 'binarize' creates step function, 'log1p' creates slope."
    )
    max_value: float = Field(5.0, description="Max expected log value for normalization to [0,1]")
    
    @field_validator('pattern')
    @classmethod
    def validate_regex(cls, v, info):
        """Ensure regex patterns are valid at config time, not runtime."""
        source = info.data.get('source')
        if source == 'regex_count' and not v:
            raise ValueError("regex_count source requires a 'pattern' string.")
        if v:
            try:
                re.compile(v)
            except re.error as e:
                raise ValueError(f"Invalid regex pattern '{v}': {e}")
        return v
    
    def expand_names(self) -> List[str]:
        """
        Returns the column names this feature will produce after transforms.
        
        Example: 
            StructuralFeature(name="latex", transforms=["binarize", "log1p"])
            -> ["has_latex", "latex_log"]
        """
        names = []
        for t in self.transforms:
            if t == "binarize":
                names.append(f"has_{self.name}")
            elif t == "log1p":
                names.append(f"{self.name}_log")
            elif t == "sigmoid":
                names.append(f"{self.name}_sigmoid")
            else:
                names.append(f"{self.name}_{t}")
        return names


# --- 3. Model Intuition (Warm Start Priors) ---

class ModelWeights(BaseModel):
    """
    Starting 'intuition' weights for a specific LLM arm.
    
    These define the initial theta vector for LinUCB before any user feedback.
    Positive weights increase selection probability for matching features.
    """
    bias: float = Field(0.0, description="Intercept term. Positive = default preference.")
    anchor_weights: Dict[str, float] = Field(
        default_factory=dict,
        description="Map of anchor_name -> weight. E.g., {'coding': 2.5, 'math': 2.5}"
    )
    feature_weights: Dict[str, float] = Field(
        default_factory=dict,
        description="Map of feature_name -> weight. E.g., {'has_latex': 2.0, 'latex_log': 0.5}"
    )
    complexity_weight: float = Field(0.0, description="Weight for complexity score feature")


class IntuitionConfig(BaseModel):
    """
    Collection of warm-start priors for all model archetypes.
    """
    archetypes: Dict[str, ModelWeights] = Field(
        default_factory=dict,
        description="Map of archetype_name -> weights. E.g., {'reasoning_model': {...}}"
    )
    hle_threshold: float = Field(
        0.15,
        description="HLE score threshold. Models above this use 'reasoning_model' archetype."
    )


# --- 4. Virtual Anchors ---

class AnchorConfig(BaseModel):
    """
    Defines a semantic neighborhood (Virtual Anchor).
    
    These are embedded at startup and used to compute 'Contrastive Anchor Distances'
    which tell the router how 'coding-like' or 'math-like' a prompt is.
    """
    name: str = Field(..., description="Anchor name (e.g., 'coding', 'math')")
    definition: str = Field(
        ..., 
        description="Representative text for this cluster. Will be embedded.",
        min_length=10
    )
    complexity_contribution: float = Field(
        0.0,
        description="How much this anchor contributes to 'hardness'. Positive = harder."
    )


# --- 5. Master Configuration ---

class LegacyRouterConfig(BaseModel):
    """
    ⚠️ DEPRECATED: Legacy configuration for older BanditRouter architecture.
    
    This config is for the virtual anchors + regex features architecture,
    which has been simplified in the production router.
    
    **MIGRATION**:
    Use the `RouterConfig` dataclass in `router.py` instead, which supports:
    - 24D embeddings (23 PCA + 1 bias)
    - HLE-based utility calibration
    - Production stability parameters
    
    This class is maintained for backward compatibility with `core.py` (BanditGPT),
    which is also deprecated. Will be removed in v2.0.
    
    ---
    
    Master configuration for BanditRouter (legacy architecture).
    
    This is the single source of truth for router architecture, ensuring:
    - Virtual Anchors are properly defined
    - Regex features have valid patterns
    - Intuition weights match defined features/anchors
    - Complexity calibration uses real data
    
    Example:
        config = LegacyRouterConfig(
            anchors=[
                AnchorConfig(name="coding", definition="python java algorithm..."),
                AnchorConfig(name="math", definition="calculus integral derivative..."),
            ],
            complexity_mean=-0.0037,  # From N=1000 LMSYS calibration
            complexity_std=0.095
        )
    """
    
    # A. Semantic Neighborhoods (Virtual Anchors)
    anchors: List[AnchorConfig] = Field(
        default_factory=lambda: [
            AnchorConfig(
                name="coding",
                definition="Write a Python function to solve this algorithm problem. Debug this code. Implement a class.",
                complexity_contribution=0.3
            ),
            AnchorConfig(
                name="math", 
                definition="Solve this calculus integral. Prove this theorem. Find the derivative of this expression.",
                complexity_contribution=0.4
            ),
            AnchorConfig(
                name="reasoning",
                definition="Explain the logical steps. What are the implications? Analyze this argument.",
                complexity_contribution=0.2
            ),
            AnchorConfig(
                name="creative",
                definition="Write a short story. Compose a poem. Create a fantasy world.",
                complexity_contribution=-0.1
            ),
            AnchorConfig(
                name="humor",
                definition="Tell me a joke. Something funny happened. Make me laugh.",
                complexity_contribution=-0.3
            ),
        ],
        description="Virtual Anchors defining semantic neighborhoods for contrastive routing."
    )
    
    # B. Structural Features (Skip Connections)
    structural_features: List[StructuralFeature] = Field(
        default_factory=lambda: [
            StructuralFeature(
                name="code_blocks",
                source="regex_count",
                pattern=r"```",
                transforms=["binarize", "log1p"]
            ),
            StructuralFeature(
                name="latex",
                source="regex_count",
                pattern=r"\$|\\\\|\^|_{}",
                transforms=["binarize", "log1p"]
            ),
            StructuralFeature(
                name="questions",
                source="regex_count",
                pattern=r"\?",
                transforms=["binarize", "log1p"]
            ),
            StructuralFeature(
                name="length",
                source="token_count",
                transforms=["binarize", "log1p"],
                max_value=10.0  # log(22000) ≈ 10
            ),
        ],
        description="Structural features that bypass embeddings (Syntactic Skip Connections)."
    )
    
    # C. Complexity Calibration (Sigmoid Normalization)
    complexity_mean: float = Field(
        -0.0037,
        description="Mean of complexity projection. Calibrated on N=1000 LMSYS train prompts."
    )
    complexity_std: float = Field(
        0.095,
        description="Std dev of complexity projection. Used for sigmoid normalization."
    )
    
    # D. Model Intuition (Warm Start)
    intuition: Optional[IntuitionConfig] = Field(
        None,
        description="Warm-start weights. If None, defaults based on HLE scores are used."
    )
    
    # E. Procedural Warmup
    procedural_warmup_samples: int = Field(
        15,
        ge=0,
        le=100,
        description="Synthetic samples for covariance shaping. 15 optimal (tested). 0 = disabled."
    )
    
    # F. Embedding Configuration
    embedding_model: str = Field(
        "all-MiniLM-L6-v2",
        description="SentenceTransformer model for semantic embeddings."
    )
    pca_dimensions: int = Field(
        32,
        ge=8,
        le=128,
        description="PCA reduction dimensions for embedding. 32 = good balance."
    )
    
    # G. LinUCB Parameters
    init_lambda: float = Field(
        1.0,
        gt=0,
        description="Initial regularization (initialization-only). Ensures cold-start stability."
    )
    ridge_lambda: float = Field(
        1.0,
        gt=0,
        description="Ridge regularization for LinUCB. Higher = more exploration."
    )
    exploration_rate: float = Field(
        0.05,
        ge=0.0,
        le=2.0,
        description="Alpha for UCB exploration. Higher = more exploration, lower = more exploitation."
    )
    forgetting_factor: float = Field(
        0.95,
        gt=0.0,
        le=1.0,
        description="Time decay gamma for non-stationary environments. 1.0 = no decay, <1.0 = exponential forgetting."
    )
    prior_n_effective: float = Field(
        100.0,
        ge=0,
        description="Effective sample count for priors. Higher = more trust in priors."
    )
    
    # H. Pruning Configuration
    pruning_min_samples: int = Field(
        50,
        ge=10,
        le=500,
        description="Minimum samples before arm is eligible for pruning (probationary period)."
    )
    pruning_enabled: bool = Field(
        True,
        description="Enable hybrid pruning (theoretical + empirical guardrail)."
    )
    
    @model_validator(mode='after')
    def validate_intuition_keys(self):
        """
        Ensures intuition weights reference valid anchors and features.
        Prevents silent failures from typos like 'codeing' instead of 'coding'.
        """
        if not self.intuition:
            return self
        
        # Build set of valid weight keys
        valid_anchor_keys = {f"anchor_{a.name}" for a in self.anchors}
        valid_feature_keys = set()
        for f in self.handcrafted_features:
            valid_feature_keys.update(f.expand_names())
        
        all_valid = valid_anchor_keys | valid_feature_keys | {"complexity_score", "bias"}
        
        # Validate each archetype's weights
        for arch_name, arch_weights in self.intuition.archetypes.items():
            # Check anchor weights
            for key in arch_weights.anchor_weights.keys():
                if key not in {a.name for a in self.anchors}:
                    raise ValueError(
                        f"Intuition archetype '{arch_name}' references unknown anchor '{key}'. "
                        f"Valid anchors: {[a.name for a in self.anchors]}"
                    )
            
            # Check feature weights
            for key in arch_weights.feature_weights.keys():
                if key not in valid_feature_keys:
                    raise ValueError(
                        f"Intuition archetype '{arch_name}' references unknown feature '{key}'. "
                        f"Valid features: {sorted(valid_feature_keys)}"
                    )
        
        return self
    
    def get_anchor_dict(self) -> Dict[str, str]:
        """Returns anchors as {name: definition} dict for backward compatibility."""
        return {a.name: a.definition for a in self.anchors}
    
    def get_complexity_vector_weights(self) -> Dict[str, float]:
        """Returns anchor contributions to complexity for building complexity vector."""
        return {a.name: a.complexity_contribution for a in self.anchors}
    
    def get_feature_names(self) -> List[str]:
        """Returns all expanded feature names (after transforms)."""
        names = []
        for f in self.handcrafted_features:
            names.extend(f.expand_names())
        return names
    
    class Config:
        extra = "forbid"  # Catch typos in config files


# --- 6. Default Configurations ---

def get_default_intuition() -> IntuitionConfig:
    """
    Returns default intuition weights based on KDD-validated archetypes.
    
    These weights were derived from analyzing HLE scores across model families
    and empirically validated on N=1000 LMSYS prompts.
    """
    return IntuitionConfig(
        archetypes={
            "reasoning_model": ModelWeights(
                bias=-0.5,
                anchor_weights={
                    "coding": 2.5,
                    "math": 2.5,
                    "reasoning": 2.0,
                    "creative": 0.5,
                    "humor": 0.0
                },
                feature_weights={
                    "has_code_block": 1.5,
                    "code_block_log": 0.3,
                    "has_latex": 2.0,
                    "latex_log": 0.5,
                    "has_question": 0.0,
                    "question_log": 0.0,
                    "has_length": 0.5,
                    "length_log": 0.2
                },
                complexity_weight=3.0
            ),
            "turbo_model": ModelWeights(
                bias=1.5,
                anchor_weights={
                    "coding": -1.0,
                    "math": -1.5,
                    "reasoning": -1.0,
                    "creative": 0.8,
                    "humor": 1.2
                },
                feature_weights={
                    "has_code_block": -1.5,
                    "code_block_log": -0.3,
                    "has_latex": -2.0,
                    "latex_log": -0.5,
                    "has_question": 0.2,
                    "question_log": 0.1,
                    "has_length": -0.3,
                    "length_log": -0.1
                },
                complexity_weight=-3.0
            )
        },
        hle_threshold=0.15
    )


# --- 7. Config Loader ---

def load_config(path: str) -> LegacyRouterConfig:
    """
    Load LegacyRouterConfig from a JSON or YAML file.
    
    ⚠️ DEPRECATED: Use RouterConfig dataclass from router.py instead.
    
    Args:
        path: Path to config file (*.json or *.yaml)
        
    Returns:
        Validated LegacyRouterConfig
        
    Raises:
        ValidationError: If config is invalid
    """
    import json
    from pathlib import Path as P
    
    path = P(path)
    
    if path.suffix == ".json":
        with open(path) as f:
            data = json.load(f)
    elif path.suffix in (".yaml", ".yml"):
        try:
            import yaml
            with open(path) as f:
                data = yaml.safe_load(f)
        except ImportError:
            raise ImportError("PyYAML required for YAML config files: pip install pyyaml")
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}. Use .json or .yaml")
    
    return LegacyRouterConfig(**data)
