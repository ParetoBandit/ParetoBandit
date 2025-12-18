"""
Neural IRT Router - Item Response Theory based model routing.

This module implements a Neural Item Response Theory (IRT) router that learns:
- Prompt difficulty vectors d(prompt) from text embeddings
- Model skill vectors s(model) as learned embeddings  
- Probability of success: P(correct | prompt, model) = sigmoid(s · d + bias)

Supports two training modes using REAL HUMAN-LABELED DATA ONLY:

1. Difficulty-Only Mode: Train on complexity_training_data.jsonl
   Data sources (all human-labeled):
   - HelpSteer2: Human-annotated complexity scores (0-4 scale)
   - GPQA Diamond: PhD-level expert-written questions
   - IFEval: Human-verified constraint annotations
   - GSM8K: Human-written math problems with step-by-step solutions
   - BBH: Human-curated tasks where LLMs historically failed

2. Full IRT Mode: Train on OpenCompass instance-level data
   Real (prompt, model, success) tuples from actual LLM evaluations.
   To download this data, run:
     python KDD/data/core_scripts/build_instance_level_training_data.py

Optimized for Apple Silicon (M1/M2/M3) with MPS backend.
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
from enum import Enum
import numpy as np

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(x, **kwargs):
        return x

# Optional imports - graceful degradation
try:
    import pytorch_lightning as pl
    HAS_LIGHTNING = True
except ImportError:
    HAS_LIGHTNING = False
    pl = None

try:
    from transformers import AutoModel, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


# =============================================================================
# Configuration & Data Classes
# =============================================================================

def get_device() -> torch.device:
    """Get optimal device for current hardware."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class TrainingMode(Enum):
    """Training mode for the IRT router."""
    DIFFICULTY_ONLY = "difficulty_only"  # Train difficulty encoder only
    FULL_IRT = "full_irt"  # Train full IRT with model embeddings


@dataclass
class IRTRouterConfig:
    """Configuration for the Neural IRT Router."""
    # Model architecture
    backbone: str = "sentence-transformers/all-MiniLM-L6-v2"
    latent_dim: int = 8  # Dimensions for difficulty/skill vectors
    hidden_dim: int = 128  # Hidden layer size
    
    # Training
    learning_rate: float = 1e-4
    head_learning_rate: float = 1e-3  # Higher LR for head when backbone is frozen
    batch_size: int = 32
    max_epochs: int = 10
    max_length: int = 256  # Max tokens per prompt
    max_samples: Optional[int] = None  # Limit samples for quick testing (None = use all)
    freeze_backbone: bool = True  # Freeze sentence transformer backbone (recommended for < 10K samples)
    
    # Regularization
    ortho_weight: float = 0.1  # Weight for orthogonality loss
    dropout: float = 0.1
    
    # Model registry (for full IRT mode)
    model_names: List[str] = field(default_factory=list)
    
    # Paths
    data_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent / "data")
    checkpoint_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent / "data" / "neural_irt")


@dataclass  
class IRTRoutingResult:
    """Result from the IRT router."""
    prompt: str
    difficulty_vector: np.ndarray  # [latent_dim] difficulty factors
    difficulty_score: float  # Scalar difficulty (0-1)
    
    # Per-model predictions (if model embeddings are trained)
    model_scores: Optional[Dict[str, float]] = None  # model_name -> P(success)
    recommended_model: Optional[str] = None
    confidence: float = 0.0


# =============================================================================
# Data Loading
# =============================================================================

def load_complexity_training_data(data_path: Optional[Path] = None) -> List[Dict]:
    """
    Load complexity training data from JSONL file.
    
    Returns list of dicts with keys: prompt, complexity_score, source, routing_class
    """
    if data_path is None:
        data_path = Path(__file__).parent.parent.parent / "data" / "complexity_training_data.jsonl"
    
    if not data_path.exists():
        raise FileNotFoundError(
            f"Complexity training data not found at {data_path}. "
            "Run `python -m llm_jury.prompt_complexity_classification.labled_data` first."
        )
    
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    print(f"Loaded {len(data):,} samples from {data_path.name}")
    return data


def load_instance_level_data(data_path: Optional[Path] = None) -> List[Dict]:
    """
    Load instance-level (prompt, model, success) data.
    
    Returns list of dicts with keys: prompt, model, success, intent
    """
    if data_path is None:
        # Check multiple possible locations
        possible_paths = [
            Path(__file__).parent.parent.parent / "KDD" / "data" / "core_scripts" / "instance_level_training_data" / "instance_level_training_data.json",
            Path(__file__).parent.parent.parent / "data" / "instance_level_training_data.json",
        ]
        
        for path in possible_paths:
            if path.exists():
                data_path = path
                break
    
    if data_path is None or not data_path.exists():
        print("⚠️  Instance-level data not found. Run build_instance_level_training_data.py first.")
        print("   Falling back to difficulty-only training mode.")
        return []
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data):,} instance-level samples")
    return data


# =============================================================================
# Dataset Classes  
# =============================================================================

class ComplexityDataset(Dataset):
    """Dataset for difficulty-only training on complexity scores."""
    
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        encoding = self.tokenizer(
            item['prompt'],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'difficulty': torch.tensor(item['complexity_score'], dtype=torch.float32),
            'routing_class': torch.tensor(item.get('routing_class', 0), dtype=torch.long),
        }


class IRTDataset(Dataset):
    """Dataset for full IRT training with (prompt, model, success) tuples."""
    
    def __init__(self, data: List[Dict], tokenizer, model_map: Dict[str, int], max_length: int = 256):
        self.data = data
        self.tokenizer = tokenizer
        self.model_map = model_map
        self.max_length = max_length
        
        # Filter to only include samples with known models
        self.data = [d for d in data if d.get('model') in model_map]
        print(f"IRTDataset: {len(self.data)} samples with known models")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        encoding = self.tokenizer(
            item['prompt'],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'model_id': torch.tensor(self.model_map[item['model']], dtype=torch.long),
            'success': torch.tensor(float(item['success']), dtype=torch.float32),
        }


# =============================================================================
# Neural IRT Model
# =============================================================================

class DifficultyEncoder(nn.Module):
    """Encodes prompts into difficulty vectors."""
    
    def __init__(self, backbone: str, latent_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers package required. Install with: pip install transformers")
        
        self.encoder = AutoModel.from_pretrained(backbone)
        encoder_dim = self.encoder.config.hidden_size  # 384 for MiniLM
        
        self.head = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.Sigmoid()  # Constrain difficulty to [0, 1]
        )
    
    def forward(self, input_ids, attention_mask):
        # Get encoder outputs
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Mean pooling
        token_embeddings = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        sentence_embedding = sum_embeddings / sum_mask
        
        # Project to difficulty vector
        difficulty = self.head(sentence_embedding)
        return difficulty


class NeuralIRTRouter(nn.Module):
    """
    Neural Item Response Theory Router.
    
    Learns to predict P(success | prompt, model) using:
    - Difficulty encoder: prompt → d (difficulty vector)
    - Model embeddings: model_id → s (skill vector)
    - IRT interaction: P = sigmoid(s · d + bias)
    
    Can also be used in difficulty-only mode for prompt complexity prediction.
    """
    
    def __init__(self, config: IRTRouterConfig):
        super().__init__()
        self.config = config
        self.device = get_device()
        
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers package required. Install with: pip install transformers")
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.backbone)
        
        # Difficulty encoder (prompt → difficulty vector)
        self.difficulty_encoder = DifficultyEncoder(
            backbone=config.backbone,
            latent_dim=config.latent_dim,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout
        )
        
        # Scalar difficulty predictor (for complexity score)
        self.difficulty_scalar = nn.Linear(config.latent_dim, 1)
        
        # Binary routing classifier
        self.routing_classifier = nn.Linear(config.latent_dim, 2)
        
        # Model embeddings (for full IRT mode)
        self.model_names = config.model_names
        self.model_map = {name: i for i, name in enumerate(config.model_names)}
        
        if config.model_names:
            self.model_embeddings = nn.Embedding(len(config.model_names), config.latent_dim)
            self.model_bias = nn.Embedding(len(config.model_names), 1)
        else:
            self.model_embeddings = None
            self.model_bias = None
    
    def encode_difficulty(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode prompts to difficulty vectors."""
        return self.difficulty_encoder(input_ids, attention_mask)
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor,
        model_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            input_ids: [batch, seq_len] tokenized prompts
            attention_mask: [batch, seq_len] attention mask
            model_ids: [batch] model indices (optional, for IRT mode)
        
        Returns:
            difficulty_vec: [batch, latent_dim] difficulty vectors
            difficulty_scalar: [batch] scalar difficulty scores
            success_prob: [batch] P(success) if model_ids provided, else None
        """
        # Encode difficulty
        difficulty_vec = self.encode_difficulty(input_ids, attention_mask)
        
        # Scalar difficulty
        difficulty_scalar = torch.sigmoid(self.difficulty_scalar(difficulty_vec)).squeeze(-1)
        
        # IRT prediction if model_ids provided
        success_prob = None
        if model_ids is not None and self.model_embeddings is not None:
            skill_vec = self.model_embeddings(model_ids)
            bias = self.model_bias(model_ids).squeeze(-1)
            
            # IRT interaction: P = sigmoid(skill · difficulty + bias)
            interaction = torch.sum(skill_vec * difficulty_vec, dim=-1) + bias
            success_prob = torch.sigmoid(interaction)
        
        return difficulty_vec, difficulty_scalar, success_prob
    
    def predict_routing_class(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Predict binary routing class (Easy vs Not Easy)."""
        difficulty_vec = self.encode_difficulty(input_ids, attention_mask)
        logits = self.routing_classifier(difficulty_vec)
        return logits
    
    @torch.no_grad()
    def route(self, prompt: str) -> IRTRoutingResult:
        """
        Route a prompt to optimal model(s).
        
        Args:
            prompt: User prompt text
            
        Returns:
            IRTRoutingResult with difficulty and model scores
        """
        self.eval()
        device = next(self.parameters()).device
        
        # Tokenize
        encoding = self.tokenizer(
            prompt,
            padding='max_length',
            truncation=True,
            max_length=self.config.max_length,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        # Get difficulty
        difficulty_vec, difficulty_scalar, _ = self.forward(input_ids, attention_mask)
        
        difficulty_np = difficulty_vec.cpu().numpy()[0]
        difficulty_score = difficulty_scalar.cpu().item()
        
        # Score all models if embeddings exist
        model_scores = None
        recommended_model = None
        
        if self.model_embeddings is not None and self.model_names:
            model_scores = {}
            
            for model_name in self.model_names:
                model_id = torch.tensor([self.model_map[model_name]], device=device)
                skill_vec = self.model_embeddings(model_id)
                bias = self.model_bias(model_id)
                
                score = torch.sigmoid(
                    torch.sum(difficulty_vec * skill_vec) + bias
                ).cpu().item()
                model_scores[model_name] = score
            
            # Recommend model with highest P(success)
            recommended_model = max(model_scores, key=model_scores.get)
        
        # Routing class prediction
        routing_logits = self.routing_classifier(difficulty_vec)
        routing_probs = F.softmax(routing_logits, dim=-1)
        confidence = routing_probs.max().cpu().item()
        
        return IRTRoutingResult(
            prompt=prompt,
            difficulty_vector=difficulty_np,
            difficulty_score=difficulty_score,
            model_scores=model_scores,
            recommended_model=recommended_model,
            confidence=confidence
        )
    
    def calibrate_new_model(
        self, 
        new_model_func, 
        calibration_prompts: List[str],
        steps: int = 50,
        lr: float = 0.1,
        verbose: bool = True
    ) -> torch.Tensor:
        """
        Estimates skill vector 's' for a new/black-box model using Adaptive Testing.
        
        This allows adding a new model to the router WITHOUT retraining the entire
        system. The method queries the model on strategically selected prompts
        (those with maximum uncertainty) to efficiently estimate its capabilities.
        
        Args:
            new_model_func: A function that takes a prompt string and returns 
                           a score (0.0-1.0) or boolean indicating success.
                           Example: lambda prompt: call_gpt5(prompt)
            calibration_prompts: List of prompt strings to use for calibration.
                                Should include diverse difficulty levels.
            steps: Number of adaptive queries to make (default 50).
            lr: Learning rate for skill vector optimization.
            verbose: Whether to print progress.
        
        Returns:
            Estimated skill vector [1, latent_dim] for the new model.
        
        Example:
            >>> def query_new_model(prompt):
            ...     response = openai.chat(model="gpt-5", messages=[{"role": "user", "content": prompt}])
            ...     return 1.0 if is_correct(response) else 0.0
            >>> 
            >>> # Use prompts from GPQA/HelpSteer as calibration set
            >>> calib_prompts = [d['prompt'] for d in load_complexity_training_data()[:500]]
            >>> skill_vec = router.calibrate_new_model(query_new_model, calib_prompts, steps=50)
            >>> 
            >>> # Add to model registry
            >>> router.add_model("gpt-5", skill_vec)
        """
        self.eval()
        device = next(self.parameters()).device
        
        if verbose:
            print(f"\n{'='*60}")
            print("Starting Adaptive Calibration")
            print(f"{'='*60}")
            print(f"Calibration prompts: {len(calibration_prompts)}")
            print(f"Adaptive steps: {steps}")
        
        # 1. Initialize a random skill vector (requires_grad=True)
        # We optimize ONLY this vector, freezing the rest of the network.
        s_est = torch.zeros(1, self.config.latent_dim, device=device, requires_grad=True)
        optimizer = torch.optim.Adam([s_est], lr=lr)
        
        # 2. Pre-compute difficulty vectors for all calibration items
        # (Optimization: Don't re-compute difficulty every step)
        items = []
        
        with torch.no_grad():
            for prompt in calibration_prompts:
                encoding = self.tokenizer(
                    prompt,
                    padding='max_length',
                    truncation=True,
                    max_length=self.config.max_length,
                    return_tensors='pt'
                )
                input_ids = encoding['input_ids'].to(device)
                attention_mask = encoding['attention_mask'].to(device)
                
                # Get difficulty vector from the trained encoder
                difficulty_vec = self.encode_difficulty(input_ids, attention_mask)
                
                items.append({
                    'd': difficulty_vec.squeeze(0),  # [latent_dim]
                    'text': prompt
                })
        
        if verbose:
            print(f"Pre-computed difficulty vectors for {len(items)} prompts")
        
        # 3. The Adaptive Loop
        used_indices = set()
        query_history = []
        
        for step in range(steps):
            # A. Active Selection: Find item with P(success) closest to 0.5
            # This maximizes information gain (maximum uncertainty = most informative)
            best_idx = -1
            best_uncertainty = 1.0
            
            current_skills = s_est.detach().squeeze(0)  # [latent_dim]
            
            for i, item in enumerate(items):
                if i in used_indices:
                    continue
                
                # Predict probability with current estimated skill
                # P = sigmoid(d · s)
                logit = torch.sum(item['d'] * current_skills)
                prob = torch.sigmoid(logit).item()
                
                # Distance from 0.5 (maximum uncertainty point)
                uncertainty = abs(prob - 0.5)
                
                if uncertainty < best_uncertainty:
                    best_uncertainty = uncertainty
                    best_idx = i
            
            if best_idx == -1:
                if verbose:
                    print(f"No more calibration items available at step {step+1}")
                break
            
            # B. Query the Black Box (The actual LLM call)
            target_item = items[best_idx]
            used_indices.add(best_idx)
            
            # This is the ONLY slow part (actual model inference)
            real_score = new_model_func(target_item['text'])
            
            # Normalize to 0-1 if boolean
            if isinstance(real_score, bool):
                real_score = 1.0 if real_score else 0.0
            real_score = float(real_score)
            real_score_tensor = torch.tensor(real_score, device=device, dtype=torch.float32)
            
            # C. Update Step (Backprop into s_est)
            optimizer.zero_grad()
            
            # Re-calculate prediction with gradient tracking
            predicted_logit = torch.sum(target_item['d'] * s_est.squeeze(0))
            loss = F.mse_loss(torch.sigmoid(predicted_logit), real_score_tensor)
            
            loss.backward()
            optimizer.step()
            
            # Record history
            query_history.append({
                'step': step + 1,
                'prompt': target_item['text'][:50],
                'score': real_score,
                'loss': loss.item()
            })
            
            if verbose:
                print(f"Step {step+1}/{steps}: "
                      f"Score={real_score:.2f} | "
                      f"Loss={loss.item():.4f} | "
                      f"Skill[0]={s_est[0][0].item():.3f}")
        
        if verbose:
            print(f"\n✓ Calibration complete. Used {len(used_indices)} queries.")
            print(f"  Final skill vector: {s_est.detach().cpu().numpy().round(3)}")
        
        return s_est.detach()
    
    def add_model(self, model_name: str, skill_vector: torch.Tensor, bias: float = 0.0):
        """
        Add a new model to the router with a pre-computed skill vector.
        
        Args:
            model_name: Name for the new model
            skill_vector: Skill vector [1, latent_dim] from calibrate_new_model()
            bias: Optional bias term (default 0.0)
        """
        device = next(self.parameters()).device
        
        # Add to model registry
        new_idx = len(self.model_names)
        self.model_names.append(model_name)
        self.model_map[model_name] = new_idx
        
        # Expand embeddings
        if self.model_embeddings is None:
            # First model being added
            self.model_embeddings = nn.Embedding(1, self.config.latent_dim).to(device)
            self.model_bias = nn.Embedding(1, 1).to(device)
        else:
            # Expand existing embeddings
            old_weights = self.model_embeddings.weight.data
            old_bias = self.model_bias.weight.data
            
            new_embeddings = nn.Embedding(len(self.model_names), self.config.latent_dim).to(device)
            new_bias_emb = nn.Embedding(len(self.model_names), 1).to(device)
            
            # Copy old weights
            new_embeddings.weight.data[:new_idx] = old_weights
            new_bias_emb.weight.data[:new_idx] = old_bias
            
            self.model_embeddings = new_embeddings
            self.model_bias = new_bias_emb
        
        # Set the new model's skill vector
        self.model_embeddings.weight.data[new_idx] = skill_vector.squeeze(0).to(device)
        self.model_bias.weight.data[new_idx] = torch.tensor([[bias]], device=device)
        
        print(f"✓ Added model '{model_name}' at index {new_idx}")
    
    def save(self, path: Path):
        """Save model checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'config': self.config,
            'state_dict': self.state_dict(),
            'model_names': self.model_names,
            'model_map': self.model_map,
        }, path)
        print(f"Saved model to {path}")
    
    @classmethod
    def load(cls, path: Path, device: Optional[torch.device] = None) -> 'NeuralIRTRouter':
        """Load model from checkpoint."""
        if device is None:
            device = get_device()
        
        checkpoint = torch.load(path, map_location=device)
        
        config = checkpoint['config']
        config.model_names = checkpoint.get('model_names', [])
        
        model = cls(config)
        model.load_state_dict(checkpoint['state_dict'])
        model.model_map = checkpoint.get('model_map', {})
        model.to(device)
        
        print(f"Loaded model from {path}")
        return model


# =============================================================================
# Training Functions
# =============================================================================

def train_difficulty_only(
    config: IRTRouterConfig,
    data: Optional[List[Dict]] = None,
    val_split: float = 0.1,
) -> NeuralIRTRouter:
    """
    Train the router in difficulty-only mode on complexity data.
    
    Args:
        config: Router configuration
        data: Training data (loads from default path if None)
        val_split: Validation split ratio
    
    Returns:
        Trained NeuralIRTRouter
    """
    device = get_device()
    print(f"Training on device: {device}")
    
    # Load data
    if data is None:
        data = load_complexity_training_data(config.data_dir / "complexity_training_data.jsonl")
    
    # Limit samples if specified (for quick testing)
    if config.max_samples is not None and config.max_samples < len(data):
        print(f"Limiting to {config.max_samples:,} samples (from {len(data):,})")
        data = data[:config.max_samples]
    
    # Split data
    np.random.seed(42)
    indices = np.random.permutation(len(data))
    val_size = int(len(data) * val_split)
    
    train_data = [data[i] for i in indices[val_size:]]
    val_data = [data[i] for i in indices[:val_size]]
    
    print(f"Train: {len(train_data):,} | Val: {len(val_data):,}")
    
    # Initialize model
    model = NeuralIRTRouter(config)
    model.to(device)
    
    # Freeze backbone if requested (recommended for small datasets)
    if config.freeze_backbone:
        for param in model.difficulty_encoder.encoder.parameters():
            param.requires_grad = False
        print("✓ Backbone frozen (training head only)")
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Trainable: {trainable_params:,} / {total_params:,} parameters ({100*trainable_params/total_params:.1f}%)")
    
    # Create datasets
    train_dataset = ComplexityDataset(train_data, model.tokenizer, config.max_length)
    val_dataset = ComplexityDataset(val_data, model.tokenizer, config.max_length)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=0,  # MPS doesn't like multiprocessing
        pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    # Optimizer - use higher LR for head when backbone is frozen
    if config.freeze_backbone:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=config.head_learning_rate
        )
        print(f"  Using head LR: {config.head_learning_rate}")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(config.max_epochs):
        # Train
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.max_epochs}", leave=True)
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            difficulty_target = batch['difficulty'].to(device)
            routing_target = batch['routing_class'].to(device)
            
            optimizer.zero_grad()
            
            difficulty_vec, difficulty_pred, _ = model(input_ids, attention_mask)
            routing_logits = model.predict_routing_class(input_ids, attention_mask)
            
            # Losses
            mse_loss = F.mse_loss(difficulty_pred, difficulty_target)
            ce_loss = F.cross_entropy(routing_logits, routing_target)
            
            # Orthogonality regularization (encourage diverse latent factors)
            if config.ortho_weight > 0:
                cov = torch.corrcoef(difficulty_vec.T)
                ortho_loss = torch.sum(torch.abs(cov - torch.eye(config.latent_dim, device=device)))
            else:
                ortho_loss = 0.0
            
            loss = mse_loss + ce_loss + config.ortho_weight * ortho_loss
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': f'{train_loss/num_batches:.4f}'})
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating", leave=False):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                difficulty_target = batch['difficulty'].to(device)
                routing_target = batch['routing_class'].to(device)
                
                difficulty_vec, difficulty_pred, _ = model(input_ids, attention_mask)
                routing_logits = model.predict_routing_class(input_ids, attention_mask)
                
                mse_loss = F.mse_loss(difficulty_pred, difficulty_target)
                ce_loss = F.cross_entropy(routing_logits, routing_target)
                
                val_loss += (mse_loss + ce_loss).item()
                
                preds = routing_logits.argmax(dim=-1)
                val_acc += (preds == routing_target).float().mean().item()
        
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{config.max_epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            model.save(config.checkpoint_dir / "best_model.pt")
    
    return model


def train_full_irt(
    config: IRTRouterConfig,
    data: Optional[List[Dict]] = None,
    val_split: float = 0.1,
) -> NeuralIRTRouter:
    """
    Train the router in full IRT mode with model embeddings.
    
    Requires instance-level (prompt, model, success) data.
    """
    device = get_device()
    print(f"Training on device: {device}")
    
    # Load data
    if data is None:
        data = load_instance_level_data()
    
    if not data:
        raise ValueError(
            "No instance-level data found. Run build_instance_level_training_data.py first, "
            "or use train_difficulty_only() for complexity-only training."
        )
    
    # Get unique models
    models = sorted(set(d['model'] for d in data if d.get('model')))
    print(f"Found {len(models)} unique models")
    config.model_names = models
    
    # Split data
    np.random.seed(42)
    indices = np.random.permutation(len(data))
    val_size = int(len(data) * val_split)
    
    train_data = [data[i] for i in indices[val_size:]]
    val_data = [data[i] for i in indices[:val_size]]
    
    print(f"Train: {len(train_data):,} | Val: {len(val_data):,}")
    
    # Initialize model
    model = NeuralIRTRouter(config)
    model.to(device)
    
    # Create datasets
    train_dataset = IRTDataset(train_data, model.tokenizer, model.model_map, config.max_length)
    val_dataset = IRTDataset(val_data, model.tokenizer, model.model_map, config.max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(config.max_epochs):
        # Train
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            model_ids = batch['model_id'].to(device)
            success_target = batch['success'].to(device)
            
            optimizer.zero_grad()
            
            difficulty_vec, _, success_pred = model(input_ids, attention_mask, model_ids)
            
            # Binary cross entropy for success prediction
            bce_loss = F.binary_cross_entropy(success_pred, success_target)
            
            # Orthogonality regularization
            if config.ortho_weight > 0:
                cov = torch.corrcoef(difficulty_vec.T)
                ortho_loss = torch.sum(torch.abs(cov - torch.eye(config.latent_dim, device=device)))
            else:
                ortho_loss = 0.0
            
            loss = bce_loss + config.ortho_weight * ortho_loss
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                model_ids = batch['model_id'].to(device)
                success_target = batch['success'].to(device)
                
                _, _, success_pred = model(input_ids, attention_mask, model_ids)
                
                bce_loss = F.binary_cross_entropy(success_pred, success_target)
                val_loss += bce_loss.item()
                
                preds = (success_pred > 0.5).float()
                val_acc += (preds == success_target).float().mean().item()
        
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{config.max_epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            model.save(config.checkpoint_dir / "best_irt_model.pt")
    
    return model


# =============================================================================
# Bradley-Terry Pairwise Ranking (Industry Standard)
# =============================================================================

class PairwiseDataset(Dataset):
    """
    Dataset for Bradley-Terry pairwise training with prompt complexity comparisons.
    """
    
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pairs = data
        print(f"Loaded {len(self.pairs):,} pairs")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        # For HelpSteer pairs: tokenize prompt + response together
        if 'response_a' in pair:
            # HelpSteer format: compare responses
            text_a = f"{pair['prompt']}\n\n{pair['response_a']}"
            text_b = f"{pair['prompt']}\n\n{pair['response_b']}"
        else:
            # Prompt-only format: compare prompts
            text_a = pair.get('prompt_a', pair.get('prompt', ''))
            text_b = pair.get('prompt_b', '')
        
        enc_a = self.tokenizer(
            text_a,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        enc_b = self.tokenizer(
            text_b,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids_a': enc_a['input_ids'].squeeze(0),
            'attention_mask_a': enc_a['attention_mask'].squeeze(0),
            'input_ids_b': enc_b['input_ids'].squeeze(0),
            'attention_mask_b': enc_b['attention_mask'].squeeze(0),
            'label': torch.tensor(pair['label'], dtype=torch.float32),
            'score_a': torch.tensor(pair.get('score_a', 0.5), dtype=torch.float32),
            'score_b': torch.tensor(pair.get('score_b', 0.5), dtype=torch.float32),
        }


def load_helpsteer_pairs(data_dir: Path = None) -> List[Dict]:
    """Load HelpSteer2 pairwise data."""
    if data_dir is None:
        data_dir = Path(__file__).parent.parent.parent / "data"
    
    pairs_path = data_dir / "helpsteer2_pairs.json"
    
    if not pairs_path.exists():
        print(f"HelpSteer pairs not found at {pairs_path}")
        print("Building from HelpSteer2...")
        from llm_jury.neural_routing.build_helpsteer_pairs import create_pairs_from_helpsteer2
        return create_pairs_from_helpsteer2(save_path=pairs_path)
    
    import json
    with open(pairs_path) as f:
        pairs = json.load(f)
    
    print(f"Loaded {len(pairs):,} HelpSteer pairs from {pairs_path}")
    return pairs


def train_pairwise_bradley_terry(
    config: IRTRouterConfig,
    data: Optional[List[Dict]] = None,
    val_split: float = 0.1,
    margin: float = 0.1,
    freeze_epochs: int = 1,  # Freeze encoder for first N epochs
    use_helpsteer: bool = True,  # Use HelpSteer2 real pairs
) -> NeuralIRTRouter:
    """
    Train the router using Bradley-Terry pairwise ranking.
    
    This is the industry standard approach used by top reward models.
    Instead of predicting absolute scores, we predict which of two items is "better".
    
    Key benefits:
    - Normalizes heterogeneous data automatically
    - More robust to noisy labels
    - Expected accuracy: >80% (vs ~55% for absolute regression)
    
    Args:
        config: Router configuration
        data: Training data (loads from default path if None)
        val_split: Validation split ratio
        margin: Minimum score difference (only used if not using HelpSteer)
        freeze_epochs: Number of epochs to keep encoder frozen (stabilizes training)
        use_helpsteer: Use real HelpSteer2 pairwise data (recommended)
    
    Returns:
        Trained NeuralIRTRouter
    """
    device = get_device()
    print(f"\n{'='*60}")
    print(f"Bradley-Terry Pairwise Training")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Data source: {'HelpSteer2 (real pairs)' if use_helpsteer else 'synthetic pairs'}")
    print(f"Freeze epochs: {freeze_epochs}")
    print(f"{'='*60}\n")
    
    # Load data
    if use_helpsteer:
        data = load_helpsteer_pairs(config.data_dir)
    elif data is None:
        # Fallback to synthetic pairs from complexity data
        data = load_complexity_training_data(config.data_dir / "complexity_training_data.jsonl")
    
    # Limit samples if specified
    if config.max_samples is not None and config.max_samples < len(data):
        print(f"Limiting to {config.max_samples:,} pairs (from {len(data):,})")
        data = data[:config.max_samples]
    
    # Split data
    np.random.seed(42)
    indices = np.random.permutation(len(data))
    val_size = int(len(data) * val_split)
    
    train_data = [data[i] for i in indices[val_size:]]
    val_data = [data[i] for i in indices[:val_size]]
    
    print(f"Pairs - Train: {len(train_data):,} | Val: {len(val_data):,}")
    
    # Initialize model
    model = NeuralIRTRouter(config)
    model.to(device)
    
    # Create pairwise datasets
    train_dataset = PairwiseDataset(train_data, model.tokenizer, config.max_length)
    val_dataset = PairwiseDataset(val_data, model.tokenizer, config.max_length)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    # Differential learning rates (key for transfer learning)
    encoder_params = list(model.difficulty_encoder.encoder.parameters())
    encoder_param_ids = {id(p) for p in encoder_params}
    head_params = [p for p in model.parameters() if id(p) not in encoder_param_ids]
    
    # Start with encoder frozen
    for param in encoder_params:
        param.requires_grad = False
    
    optimizer = torch.optim.AdamW([
        {'params': head_params, 'lr': config.head_learning_rate},  # 1e-3 for head
    ])
    
    print(f"✓ Encoder frozen for first {freeze_epochs} epoch(s)")
    print(f"  Head LR: {config.head_learning_rate}")
    print(f"  Encoder LR (after unfreeze): {config.learning_rate}")
    
    # Training loop
    best_val_acc = 0.0
    
    for epoch in range(config.max_epochs):
        # Unfreeze encoder after freeze_epochs
        if epoch == freeze_epochs:
            print(f"\n✓ Unfreezing encoder at epoch {epoch + 1}")
            for param in encoder_params:
                param.requires_grad = True
            
            # Add encoder params to optimizer with lower LR
            optimizer.add_param_group({
                'params': encoder_params, 
                'lr': config.learning_rate  # 1e-5 for encoder
            })
        
        # Train
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.max_epochs}", leave=True)
        for batch in pbar:
            # Get inputs
            input_ids_a = batch['input_ids_a'].to(device)
            attention_mask_a = batch['attention_mask_a'].to(device)
            input_ids_b = batch['input_ids_b'].to(device)
            attention_mask_b = batch['attention_mask_b'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            # Get difficulty vectors for both prompts
            d_vec_a = model.encode_difficulty(input_ids_a, attention_mask_a)
            d_vec_b = model.encode_difficulty(input_ids_b, attention_mask_b)
            
            # Compute scalar difficulty scores
            # Higher score = harder prompt
            score_a = model.difficulty_scalar(d_vec_a).squeeze(-1)
            score_b = model.difficulty_scalar(d_vec_b).squeeze(-1)
            
            # Bradley-Terry: P(A harder than B) = sigmoid(score_A - score_B)
            logits = score_a - score_b
            
            # Binary cross-entropy loss
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            preds = (logits > 0).float()
            acc = (preds == labels).float().mean().item()
            
            train_loss += loss.item()
            train_acc += acc
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f'{train_loss/num_batches:.4f}',
                'acc': f'{train_acc/num_batches:.4f}'
            })
        
        train_loss /= num_batches
        train_acc /= num_batches
        
        # Validate
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating", leave=False):
                input_ids_a = batch['input_ids_a'].to(device)
                attention_mask_a = batch['attention_mask_a'].to(device)
                input_ids_b = batch['input_ids_b'].to(device)
                attention_mask_b = batch['attention_mask_b'].to(device)
                labels = batch['label'].to(device)
                
                d_vec_a = model.encode_difficulty(input_ids_a, attention_mask_a)
                d_vec_b = model.encode_difficulty(input_ids_b, attention_mask_b)
                
                score_a = model.difficulty_scalar(d_vec_a).squeeze(-1)
                score_b = model.difficulty_scalar(d_vec_b).squeeze(-1)
                
                logits = score_a - score_b
                loss = F.binary_cross_entropy_with_logits(logits, labels)
                
                preds = (logits > 0).float()
                acc = (preds == labels).float().mean().item()
                
                val_loss += loss.item()
                val_acc += acc
                val_batches += 1
        
        val_loss /= val_batches
        val_acc /= val_batches
        
        print(f"Epoch {epoch+1}/{config.max_epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            model.save(config.checkpoint_dir / "best_pairwise_model.pt")
            print(f"  ✓ New best model saved (Val Acc: {val_acc:.4f})")
    
    print(f"\n{'='*60}")
    print(f"Training complete! Best Val Acc: {best_val_acc:.4f}")
    print(f"{'='*60}")
    
    return model


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Neural IRT Router")
    parser.add_argument("--mode", choices=["difficulty", "irt", "pairwise"], default="pairwise",
                       help="Training mode: 'difficulty' (regression), 'irt' (full IRT), 'pairwise' (Bradley-Terry ranking)")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for encoder")
    parser.add_argument("--latent-dim", type=int, default=8, help="Latent dimension size")
    parser.add_argument("--max-samples", type=int, default=None, 
                       help="Limit training samples (for quick testing)")
    parser.add_argument("--unfreeze-backbone", action="store_true",
                       help="Unfreeze backbone (requires more data, slower)")
    parser.add_argument("--head-lr", type=float, default=1e-3,
                       help="Learning rate for head")
    parser.add_argument("--margin", type=float, default=0.1,
                       help="Minimum score difference for pairwise pairs")
    parser.add_argument("--freeze-epochs", type=int, default=1,
                       help="Epochs to freeze encoder (pairwise mode)")
    
    args = parser.parse_args()
    
    config = IRTRouterConfig(
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        head_learning_rate=args.head_lr,
        latent_dim=args.latent_dim,
        max_samples=args.max_samples,
        freeze_backbone=not args.unfreeze_backbone,
    )
    
    print(f"\n{'='*60}")
    print(f"Neural IRT Router Training")
    print(f"{'='*60}")
    print(f"Mode: {args.mode}")
    print(f"Device: {get_device()}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Latent Dim: {args.latent_dim}")
    print(f"Max Samples: {args.max_samples or 'all'}")
    if args.mode == "pairwise":
        print(f"Margin: {args.margin}")
        print(f"Freeze Epochs: {args.freeze_epochs}")
    print(f"{'='*60}\n")
    
    if args.mode == "difficulty":
        model = train_difficulty_only(config)
    elif args.mode == "pairwise":
        model = train_pairwise_bradley_terry(config, margin=args.margin, freeze_epochs=args.freeze_epochs)
    else:
        model = train_full_irt(config)
    
    # Test inference
    print("\n--- Testing Inference ---")
    test_prompts = [
        "What is 2+2?",
        "Explain quantum entanglement and its implications for faster-than-light communication.",
        "Write a Python function to find the longest common subsequence of two strings using dynamic programming.",
    ]
    
    for prompt in test_prompts:
        result = model.route(prompt)
        print(f"\nPrompt: {prompt[:50]}...")
        print(f"  Difficulty Score: {result.difficulty_score:.3f}")
        print(f"  Difficulty Vector: {result.difficulty_vector.round(2)}")
        if result.model_scores:
            top_3 = sorted(result.model_scores.items(), key=lambda x: -x[1])[:3]
            print(f"  Top Models: {top_3}")
