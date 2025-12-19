#!/usr/bin/env python3
"""
Quality & Cost Predictor trained on HelpSteer2 + LMSYS Arena.

Data Sources:
- HelpSteer2: Human-rated correctness scores (0-4)
- LMSYS Arena: Human preference battles (winner=GOOD, loser=BAD)

The LMSYS augmentation broadens the distribution to include:
- Short correct answers (e.g., "A lynx jumps quick." winning over verbose wrong answer)
- Short wrong answers (e.g., "Dune" losing to detailed correct explanation)

Uses "Goldilocks Injection" for length debiasing:
- Training: Model learns semantic + length bias together
- Inference: We inject a standard length to neutralize bias

Key Features:
- Multi-source data augmentation (HelpSteer2 + LMSYS)
- Length-debiased predictions via Goldilocks Injection
- Weighted BCE loss for class imbalance
- GroupShuffleSplit to prevent data leakage
- F1 score optimization

Usage:
    python -m banditgpt.neural_routing.quality_cost_predictor --epochs 3
    python -m banditgpt.neural_routing.quality_cost_predictor --epochs 3 --no-lmsys
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Tuple, Any, List
from dataclasses import dataclass, field
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(x, **kwargs):
        return x

try:
    from transformers import AutoTokenizer, AutoModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class QualityCostConfig:
    """Configuration for Quality/Cost Predictor."""
    backbone: str = "sentence-transformers/all-MiniLM-L6-v2"  # 22M params, fast
    batch_size: int = 64
    learning_rate: float = 2e-4
    max_length: int = 256
    max_epochs: int = 3
    
    # Classification threshold: correctness <= this = BAD
    quality_threshold: float = 2.5

    # -------------------------------------------------------------------------
    # Production reward policy (for async bandit learning)
    #
    # IMPORTANT TERMINOLOGY:
    # This model is a *competence/quality grader* (Type B "competence risk"),
    # not a policy safety classifier (Type A).
    #
    # Therefore, do NOT clamp reward to 0.0 when competence looks low.
    # The bandit must learn "bad vs terrible", so we use the calibrated
    # probability of correctness/competence as the learning signal:
    #
    #   reward = P_correct  (optionally clipped for numerical stability)
    #
    # Routing thresholds belong in decision-time utility, not in reward.
    # -------------------------------------------------------------------------
    reward_clip_eps: float = 0.01  # clip reward into [eps, 1-eps] for stability/log longevity

    # "Safe logit" epsilon for the logit-stretched KPI / optional learning signal:
    # Use strict bounds to avoid +/-inf when p touches 0 or 1.
    # Recommended: 1e-3 or 1e-4.
    reward_logit_eps: float = 1e-4

    # -------------------------------------------------------------------------
    # Backward compatibility (deprecated)
    #
    # Older checkpoints stored a "safety_p_bad_threshold" derived from BAD-recall
    # calibration. We no longer interpret this as "unsafe" in production, but we
    # keep these fields so old checkpoints can still be loaded.
    # -------------------------------------------------------------------------
    safety_p_bad_threshold: float = 0.5  # DEPRECATED: do not use as policy safety
    target_safety_recall: float = 0.95   # DEPRECATED
    routing_p_bad_threshold: float = 0.5  # DEPRECATED: older routing used P(BAD) cutoff

    # Competence routing threshold on P_correct:
    # Route to a stronger model when predicted competence is below this.
    # Recommended calibration: "GPT-4 anchor" → set to the (1 - anchor_keep_rate)
    # quantile of GPT-4(o) scores (e.g., keep_rate=0.95 → 5th percentile).
    routing_p_correct_threshold: Optional[float] = None
    anchor_keep_rate: float = 0.95  # fraction of anchor answers considered "good enough"
    
    # Loss weights
    quality_weight: float = 1.0
    verbosity_weight: float = 0.5
    
    # Validation split
    val_split: float = 0.1
    
    # Debiasing with semantic head regularization
    # At inference, we use semantic path only (lengths=None)
    use_length_debiasing: bool = True
    semantic_reg_weight: float = 0.01  # Light regularization to keep semantic head centered

    # When debiasing is enabled, we must ensure the *semantic* head is directly
    # trained to solve the task (since inference uses semantic-only).
    semantic_only_loss_weight: float = 1.0
    # The combined (semantic + length-bias) logit can be useful during training,
    # but our production path (and evaluation) uses semantic-only. When this is
    # non-zero, the model can learn shortcuts that hurt semantic-only quality.
    combined_loss_weight: float = 0.0
    
    # Data augmentation: add LMSYS Arena battles
    use_lmsys_augmentation: bool = True
    lmsys_max_samples: int = 20000  # Max samples from LMSYS (each side of battle)

    # -------------------------------------------------------------------------
    # Deterministic edge-case augmentation (offline fine-tuning only)
    #
    # Goal: Improve correctness calibration on brittle STEM edge cases where many
    # models give plausible-but-wrong answers (e.g., dilute strong acid pH where
    # water autoionization matters).
    #
    # This does NOT require any online correctness checks; we just train the grader
    # offline with known-ground-truth synthetic pairs.
    # -------------------------------------------------------------------------
    # Enable by default: without deterministic "short correct vs short wrong"
    # anchors, the grader can regress to "short = bad" and misroute everything.
    use_deterministic_edgecases: bool = True
    deterministic_edgecases_n: int = 4000
    deterministic_edgecases_weight: float = 3.0  # upweight these samples in the loss (keep stable)
    
    checkpoint_dir: Path = field(
        default_factory=lambda: Path(__file__).parent.parent.parent / "data" / "quality_predictor"
    )


def get_device() -> torch.device:
    """Get optimal device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def clip01(x: float, eps: float = 0.01) -> float:
    """Clip a probability-like value into [eps, 1-eps]."""
    x = float(x)
    eps = float(eps)
    return float(min(max(x, eps), 1.0 - eps))

def clipped_quality_reward(quality_score: float, clip_eps: float = 0.01) -> float:
    """Production reward for bandit.update(): clipped P_correct in [eps, 1-eps]."""
    return clip01(float(quality_score), eps=clip_eps)

def logit_clipped_prob(p: float, eps: float = 0.01) -> float:
    """
    Monotonic "stretch" transform for reporting:
        logit(p) = log(p / (1 - p))

    Use clipped probabilities to avoid infinities and to keep this KPI stable.
    This is intended for monitoring / dashboards (absolute-ish scale), not for
    online normalization (which uses reward_z).
    """
    p = clip01(float(p), eps=float(eps))
    return float(np.log(p / (1.0 - p)))


class LogitReward:
    """
    Safe logit transform with strict clipping.

    Stretches probabilities p in (0, 1) to (-inf, +inf):
        logit(p) = log(p / (1 - p))

    This is useful as an absolute-ish "stretched KPI" (and can also be used as a
    learning signal if you want to avoid the non-stationarity of z-scores).
    """

    def __init__(self, epsilon: float = 1e-4):
        self.epsilon = float(epsilon)
        e = self.epsilon
        self.min_val = float(np.log(e / (1.0 - e)))
        self.max_val = float(np.log((1.0 - e) / e))

    def transform(self, p_correct: float) -> float:
        p = clip01(float(p_correct), eps=self.epsilon)
        return float(np.log(p / (1.0 - p)))

class RunningZScoreNormalizer:
    """
    Online reward normalization using exponential moving averages.

    Why:
      - Raw grader outputs are often score-compressed (e.g., 0.60-0.70).
      - A contextual bandit comparing arms with large cost differences can become
        a "cheapest-router" if reward deltas are tiny.

    This normalizer "stretches" the signal by converting raw rewards into a
    clamped z-score:
        z = (r - mean) / (std + eps)

    Notes:
      - Keep a single normalizer per deployment (persist its state).
      - This is competence-risk normalization (Type B), not policy safety.
    """

    def __init__(
        self,
        mean_init: float = 0.65,
        std_init: float = 0.05,
        alpha: float = 0.01,
        clamp: float = 3.0,
        eps: float = 1e-9,
        auto_init_from_first_sample: bool = False,
    ):
        self.mean = float(mean_init)
        # Track variance to derive std (more stable than EMA(|x-mean|))
        self.var = float(max(std_init, eps) ** 2)
        self.alpha = float(alpha)
        self.clamp = float(clamp)
        self.eps = float(eps)
        self.auto_init_from_first_sample = bool(auto_init_from_first_sample)
        self.n_seen = 0

    @property
    def std(self) -> float:
        return float(np.sqrt(max(self.var, self.eps)))

    def update(self, x: float) -> None:
        x = float(x)
        # EMA mean
        new_mean = (1.0 - self.alpha) * self.mean + self.alpha * x
        # EMA variance around the *new* mean (stable in practice)
        err = x - new_mean
        new_var = (1.0 - self.alpha) * self.var + self.alpha * (err * err)
        self.mean = float(new_mean)
        self.var = float(max(new_var, self.eps))

    def normalize(self, x: float, *, update: bool = True) -> float:
        x = float(x)
        # Optional bootstrap: initialize mean from the first observed sample.
        # This prevents misleading "everything is negative" behavior when the
        # provided mean_init is far from the current traffic distribution.
        if self.auto_init_from_first_sample and self.n_seen == 0:
            self.mean = float(x)
            # keep existing var as a reasonable prior scale
            self.n_seen = 1
            return 0.0

        # IMPORTANT:
        # Compute z using the *current* running stats, then (optionally) update.
        # This preserves the magnitude of rare events instead of immediately
        # pulling the mean toward the new value before scoring.
        z = (x - self.mean) / (self.std + self.eps)
        if self.clamp > 0:
            z = max(min(z, self.clamp), -self.clamp)
        if update:
            self.update(x)
        self.n_seen += 1
        return float(z)

    def state_dict(self) -> Dict[str, float]:
        return {
            "mean": float(self.mean),
            "var": float(self.var),
            "alpha": float(self.alpha),
            "clamp": float(self.clamp),
            "eps": float(self.eps),
        }

    @classmethod
    def from_state_dict(cls, d: Dict[str, Any]) -> "RunningZScoreNormalizer":
        obj = cls(
            mean_init=float(d.get("mean", 0.65)),
            std_init=float(np.sqrt(max(float(d.get("var", 0.05**2)), 1e-12))),
            alpha=float(d.get("alpha", 0.01)),
            clamp=float(d.get("clamp", 3.0)),
            eps=float(d.get("eps", 1e-9)),
        )
        obj.mean = float(d.get("mean", obj.mean))
        obj.var = float(d.get("var", obj.var))
        return obj


def calibrate_routing_threshold_from_anchor_scores(
    anchor_p_correct: List[float],
    *,
    keep_rate: float = 0.95,
) -> float:
    """
    "GPT-4 anchor" calibration:
      T = quantile(anchor_scores, q = 1 - keep_rate)
    such that ~keep_rate of anchor answers are above T.
    """
    scores = np.asarray(anchor_p_correct, dtype=np.float64)
    if scores.size == 0:
        return 0.5
    q = float(max(0.0, min(1.0, 1.0 - keep_rate)))
    return float(np.quantile(scores, q=q))


# =============================================================================
# Dataset
# =============================================================================

class QualityDataset(Dataset):
    """
    Dataset for quality/verbosity prediction.
    
    Supports two data sources:
    - HelpSteer2: Has correctness (0-4) and verbosity (0-4)
    - LMSYS Arena: Has is_bad (0/1), verbosity is estimated from response length
    """
    
    def __init__(self, df: pd.DataFrame, tokenizer, config: QualityCostConfig):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.config = config
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        text = f"User: {row['prompt']}\nAssistant: {row['response']}"
        
        # Quality target: is_bad (1 = BAD, 0 = GOOD)
        if 'is_bad' in self.df.columns:
            # LMSYS data: already has is_bad
            is_bad = float(row['is_bad'])
        else:
            # HelpSteer2 data: derive from correctness
            is_bad = 1.0 if row['correctness'] <= self.config.quality_threshold else 0.0
        
        # Verbosity target: 0-1
        if 'verbosity' in self.df.columns and pd.notna(row.get('verbosity')):
            # HelpSteer2: has explicit verbosity
            verbosity = row['verbosity'] / 4.0
        else:
            # LMSYS: estimate from response length (words)
            # Normalize: 0 words = 0.0, 500+ words = 1.0
            word_count = len(str(row['response']).split())
            verbosity = min(word_count / 500.0, 1.0)
        
        enc = self.tokenizer(
            text,
            max_length=self.config.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        sample_weight = float(row['sample_weight']) if 'sample_weight' in self.df.columns else 1.0
        
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'is_bad_target': torch.tensor(is_bad, dtype=torch.float32),
            'verbosity_target': torch.tensor(verbosity, dtype=torch.float32),
            'sample_weight': torch.tensor(sample_weight, dtype=torch.float32),
        }


def load_helpsteer_grouped(config: QualityCostConfig, seed: int = 42):
    """Load HelpSteer2 with grouped splitting (no data leakage)."""
    from datasets import load_dataset
    from sklearn.model_selection import GroupShuffleSplit
    
    print("Loading HelpSteer2...")
    dataset = load_dataset("nvidia/HelpSteer2", split="train")
    df = pd.DataFrame(dataset)
    print(f"  Raw: {len(df):,} samples")
    
    # Drop exact duplicates
    df = df.drop_duplicates(subset=['prompt', 'response'])
    print(f"  After dedup: {len(df):,}")
    
    # Class distribution
    n_bad = len(df[df['correctness'] <= config.quality_threshold])
    n_good = len(df) - n_bad
    pos_weight = n_good / n_bad if n_bad > 0 else 1.0
    
    print(f"\nClass distribution (threshold={config.quality_threshold}):")
    print(f"  GOOD: {n_good:,} ({100*n_good/len(df):.1f}%)")
    print(f"  BAD:  {n_bad:,} ({100*n_bad/len(df):.1f}%)")
    print(f"  pos_weight: {pos_weight:.2f}")
    
    # Group split by prompt
    splitter = GroupShuffleSplit(n_splits=1, test_size=config.val_split, random_state=seed)
    train_idx, val_idx = next(splitter.split(df, groups=df['prompt']))
    
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    
    # Verify no overlap
    assert len(set(train_df['prompt']) & set(val_df['prompt'])) == 0, "Data leakage!"
    
    print(f"\nSplit (no prompt overlap):")
    print(f"  Train: {len(train_df):,} | Val: {len(val_df):,}")
    
    return train_df, val_df, pos_weight


def load_lmsys_arena(config: QualityCostConfig, seed: int = 42):
    """
    Load LMSYS Arena human preference battles.
    
    Converts winner/loser to quality labels:
    - Winner response → GOOD (is_bad=0)
    - Loser response → BAD (is_bad=1)
    
    This broadens the distribution with:
    - Short correct answers (e.g., "A lynx jumps quick." winning)
    - Short wrong answers (e.g., "Dune" losing)
    """
    from datasets import load_dataset
    import numpy as np
    import json
    
    def parse_lmsys_field(val):
        """Parse LMSYS field which may be JSON-encoded string or list."""
        if val is None:
            return ""
        # If it's a string that looks like JSON list, parse it
        if isinstance(val, str):
            if val.startswith('[') and val.endswith(']'):
                try:
                    parsed = json.loads(val)
                    if isinstance(parsed, list) and len(parsed) > 0:
                        return str(parsed[0])
                except json.JSONDecodeError:
                    pass
            return val
        # If it's already a list
        if isinstance(val, list):
            return str(val[0]) if val else ""
        return str(val)
    
    print("\nLoading LMSYS Arena (human preference battles)...")
    ds = load_dataset('lmsys/lmsys-arena-human-preference-55k', split='train')
    df = ds.to_pandas()
    print(f"  Raw battles: {len(df):,}")
    
    # Determine winner from columns
    df['winner'] = df.apply(
        lambda r: 'A' if r['winner_model_a'] else ('B' if r['winner_model_b'] else 'tie'), 
        axis=1
    )
    
    # Filter out ties (ambiguous signal)
    df_clear = df[df['winner'] != 'tie'].copy()
    print(f"  Clear winners: {len(df_clear):,} (excluding ties)")
    
    # Convert battles to individual (prompt, response, is_bad) samples
    samples = []
    
    for _, row in df_clear.iterrows():
        prompt = parse_lmsys_field(row['prompt'])
        resp_a = parse_lmsys_field(row['response_a'])
        resp_b = parse_lmsys_field(row['response_b'])
        
        # Skip if any field is empty
        if not prompt or not resp_a or not resp_b:
            continue
        
        winner = row['winner']
        
        # Winner = GOOD, Loser = BAD
        if winner == 'A':
            samples.append({'prompt': prompt, 'response': resp_a, 'is_bad': 0, 'source': 'lmsys'})
            samples.append({'prompt': prompt, 'response': resp_b, 'is_bad': 1, 'source': 'lmsys'})
        else:  # winner == 'B'
            samples.append({'prompt': prompt, 'response': resp_b, 'is_bad': 0, 'source': 'lmsys'})
            samples.append({'prompt': prompt, 'response': resp_a, 'is_bad': 1, 'source': 'lmsys'})
    
    lmsys_df = pd.DataFrame(samples)
    print(f"  Total samples: {len(lmsys_df):,} (GOOD: {len(lmsys_df[lmsys_df['is_bad']==0]):,}, BAD: {len(lmsys_df[lmsys_df['is_bad']==1]):,})")
    
    # Sample if needed
    if config.lmsys_max_samples and len(lmsys_df) > config.lmsys_max_samples * 2:
        np.random.seed(seed)
        lmsys_df = lmsys_df.sample(n=config.lmsys_max_samples * 2, random_state=seed)
        print(f"  Sampled: {len(lmsys_df):,}")
    
    # Report short response stats
    lmsys_df['word_count'] = lmsys_df['response'].apply(lambda x: len(str(x).split()))
    short_good = len(lmsys_df[(lmsys_df['word_count'] < 30) & (lmsys_df['is_bad'] == 0)])
    short_bad = len(lmsys_df[(lmsys_df['word_count'] < 30) & (lmsys_df['is_bad'] == 1)])
    print(f"  Short (<30 words): {short_good:,} GOOD, {short_bad:,} BAD")
    
    return lmsys_df


def _format_sci(x: float) -> str:
    """Format concentrations like 1e-8 as 10^{-8} for prompts."""
    if x <= 0:
        return str(x)
    exp = int(round(math.log10(x)))
    if abs(x - (10 ** exp)) / x < 1e-9:
        return f"10^{{{exp}}}"
    # fallback
    return f"{x:.2e}"


def build_deterministic_edgecases(config: QualityCostConfig, seed: int = 42) -> pd.DataFrame:
    """
    Generate small, high-signal synthetic pairs with deterministic ground truth.

    These examples are specifically chosen to break common heuristics (e.g. "strong acid pH = -log(C)")
    and therefore improve the grader's calibration on brittle STEM correctness.
    """
    rng = np.random.default_rng(seed)

    rows: List[Dict[str, Any]] = []
    n = int(max(0, config.deterministic_edgecases_n))
    if n == 0:
        return pd.DataFrame([])

    # --- Edgecase family 0: trivial arithmetic (anchors basic correctness) ---
    # These are deterministic, unambiguous labels and should strongly calibrate the grader.
    n_arith = max(1, n // 2)
    for _ in range(n_arith):
        a = int(rng.integers(1, 200))
        b = int(rng.integers(1, 200))
        op = rng.choice(["+", "-", "*"])
        if op == "+":
            ans = a + b
        elif op == "-":
            ans = a - b
        else:
            ans = a * b

        prompt = f"What is {a} {op} {b}?"

        # GOOD responses: include both numeric-only and short natural language.
        good_templates = [
            "{ans}",
            "{ans}.",
            "The answer is {ans}.",
            "{ans} (exact).",
        ]
        good = rng.choice(good_templates).format(ans=ans)

        # wrong answer: off by a random delta (avoid accidental correctness)
        delta = int(rng.integers(1, 20))
        wrong = int(ans + delta if delta != 0 else ans + 1)
        bad_templates = [
            "{wrong}",
            "{wrong}.",
            "The answer is {wrong}.",
            "{wrong} (exact).",
        ]
        bad = rng.choice(bad_templates).format(wrong=wrong)

        rows.append({"prompt": prompt, "response": good, "is_bad": 0, "source": "deterministic_edgecases"})
        rows.append({"prompt": prompt, "response": bad, "is_bad": 1, "source": "deterministic_edgecases"})

    # --- Edgecase family 0.5: short factual QA (breaks "short = bad") ---
    # These are short, *correct* answers that many graders mistakenly penalize.
    # We include paired plausible-wrong answers as hard negatives.
    # Keep the set small but repeated across n to create strong anchors.
    fact_bank: List[Tuple[str, str, str]] = [
        ("Capital of France? Answer with just the city.", "Paris", "Lyon"),
        ("What is the speed of light in m/s? Answer with just the number.", "299792458", "300000000"),
        ("How many minutes are in an hour? Answer with just the number.", "60", "100"),
        ("What is the chemical symbol for water? Answer with just the symbol.", "H2O", "HO2"),
        ("What is 2 + 2? Answer with just the number.", "4", "5"),
        ("Write Python to sort a list named lst. Respond with just code.", "sorted(lst)", "lst.sorted()"),
        ("Write Python to sort a list named lst in-place. Respond with just code.", "lst.sort()", "sort(lst)"),
    ]
    n_fact = max(1, n // 6)
    for _ in range(n_fact):
        prompt, good, bad = fact_bank[int(rng.integers(0, len(fact_bank)))]
        rows.append({"prompt": prompt, "response": good, "is_bad": 0, "source": "deterministic_edgecases"})
        rows.append({"prompt": prompt, "response": bad, "is_bad": 1, "source": "deterministic_edgecases"})

    # --- Edgecase family 0.75: fixed "sanity prompts" (matches built-in smoke tests) ---
    # These are the exact kinds of cases that previously caused catastrophic misrouting.
    # We intentionally include both short and medium-length phrasing variants.
    sanity_pairs: List[Tuple[str, str, int]] = [
        ("What is 2+2?", "The answer is 4.", 0),
        ("What is 2+2?", "4", 0),
        ("What is 2+2?", "I think maybe fish?", 1),
        ("Write Python to sort a list", "sorted(lst)", 0),
        ("Write Python to sort a list", "lst.sorted()", 1),
        ("Explain quantum computing", "I don't know.", 1),
        ("Explain quantum computing", "Quantum computing uses qubits that can exist in superposition and can be entangled, enabling certain algorithms to outperform classical ones.", 0),
        ("What is the speed of light?", "Approximately 299,792,458 meters per second.", 0),
        ("What is the speed of light?", "Approximately 150,000,000 meters per second.", 1),
        ("Capital of France?", "Paris", 0),
        ("Capital of France?", "Lyon", 1),
    ]
    # Repeat these anchors so they actually move the decision boundary.
    n_sanity = max(50, n // 10)
    for _ in range(n_sanity):
        prompt, resp, is_bad = sanity_pairs[int(rng.integers(0, len(sanity_pairs)))]
        rows.append({"prompt": prompt, "response": resp, "is_bad": int(is_bad), "source": "deterministic_edgecases"})

    # --- Edgecase family 1: dilute strong acid pH (water autoionization matters) ---
    # For a strong acid of formal concentration Ca:
    #   [H+] = (Ca + sqrt(Ca^2 + 4Kw))/2  with Kw=1e-14
    # pH = -log10([H+])
    Kw = 1e-14

    # Reserve room for fact + sanity anchors already added above.
    n_ph = max(1, n - n_arith - n_fact - n_sanity)
    for _ in range(n_ph):
        # sample Ca in [1e-10, 1e-6] where the naive formula becomes noticeably wrong
        exp = float(rng.uniform(-10.0, -6.0))
        Ca = float(10 ** exp)

        H = (Ca + math.sqrt(Ca * Ca + 4.0 * Kw)) / 2.0
        ph = -math.log10(H)
        ph_naive = -math.log10(Ca)

        # prompt
        sci = _format_sci(Ca)
        prompt = f"Calculate the pH of a ${sci}$ M solution of HCl."

        # GOOD response: multiple phrasing variants (short + long) to prevent overfitting
        good_templates = [
            # short answer with key formula
            "At this dilution you must include water autoionization. "
            "Use [H+] = (C + sqrt(C^2 + 4Kw))/2 with Kw=1e-14. "
            "For C={C:.2e}, [H+]≈{H:.2e} so pH≈{pH:.2f}.",
            # explicit quadratic
            "Include water autoionization: [H+] = C + Kw/[H+]. "
            "So [H+]^2 - C[H+] - Kw = 0 and [H+] = (C + sqrt(C^2 + 4Kw))/2. "
            "With C={C:.2e}, pH≈{pH:.2f}.",
            # numeric-only style (models see these in LMSYS)
            "{pH:.2f}",
        ]
        good = rng.choice(good_templates).format(C=Ca, H=H, pH=ph)

        bad_templates = [
            "HCl is a strong acid so [H+] = C. pH = -log10(C) = {pHn:.2f}.",
            "{pHn:.2f}",
            "pH = {pHn:.2f}",
        ]
        bad = rng.choice(bad_templates).format(pHn=ph_naive)

        rows.append({"prompt": prompt, "response": good, "is_bad": 0, "source": "deterministic_edgecases"})
        rows.append({"prompt": prompt, "response": bad, "is_bad": 1, "source": "deterministic_edgecases"})

    df = pd.DataFrame(rows)
    # Attach per-sample weight so training can focus on these hard negatives.
    df["sample_weight"] = float(config.deterministic_edgecases_weight)
    return df


def load_combined_data(config: QualityCostConfig, seed: int = 42):
    """
    Load combined HelpSteer2 + LMSYS Arena data.
    
    Returns train_df, val_df, pos_weight with properly balanced classes.
    """
    from sklearn.model_selection import GroupShuffleSplit
    
    # Load HelpSteer2
    hs_train, hs_val, _ = load_helpsteer_grouped(config, seed)
    hs_train['source'] = 'helpsteer2'
    hs_val['source'] = 'helpsteer2'
    
    # Convert HelpSteer2 to unified format (add is_bad column)
    hs_train['is_bad'] = (hs_train['correctness'] <= config.quality_threshold).astype(int)
    hs_val['is_bad'] = (hs_val['correctness'] <= config.quality_threshold).astype(int)
    
    if config.use_lmsys_augmentation:
        # Load LMSYS
        lmsys_df = load_lmsys_arena(config, seed)
        
        # Split LMSYS by prompt (no leakage)
        splitter = GroupShuffleSplit(n_splits=1, test_size=config.val_split, random_state=seed)
        lmsys_train_idx, lmsys_val_idx = next(splitter.split(lmsys_df, groups=lmsys_df['prompt']))
        
        lmsys_train = lmsys_df.iloc[lmsys_train_idx]
        lmsys_val = lmsys_df.iloc[lmsys_val_idx]
        
        print(f"\nLMSYS split: Train {len(lmsys_train):,} | Val {len(lmsys_val):,}")
        
        # Combine
        train_df = pd.concat([hs_train, lmsys_train], ignore_index=True)
        val_df = pd.concat([hs_val, lmsys_val], ignore_index=True)
    else:
        train_df = hs_train
        val_df = hs_val
    
    # Shuffle
    train_df = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    val_df = val_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Default weights
    if "sample_weight" not in train_df.columns:
        train_df["sample_weight"] = 1.0
    if "sample_weight" not in val_df.columns:
        val_df["sample_weight"] = 1.0

    # Optional deterministic edge-case augmentation (train only)
    if config.use_deterministic_edgecases:
        edge_df = build_deterministic_edgecases(config, seed=seed)
        if len(edge_df) > 0:
            # Ensure required columns exist
            edge_df = edge_df[["prompt", "response", "is_bad", "source", "sample_weight"]].copy()
            train_df = pd.concat([train_df, edge_df], ignore_index=True)
            train_df = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)
            print(f"\nAdded deterministic edgecases: {len(edge_df):,} rows (weight={config.deterministic_edgecases_weight})")
    
    # Calculate pos_weight for combined data
    n_bad = train_df['is_bad'].sum()
    n_good = len(train_df) - n_bad
    pos_weight = n_good / n_bad if n_bad > 0 else 1.0
    
    print(f"\n{'='*60}")
    print("COMBINED DATA SUMMARY")
    print(f"{'='*60}")
    print(f"Train: {len(train_df):,} samples")
    print(f"  - HelpSteer2: {len(train_df[train_df['source']=='helpsteer2']):,}")
    if config.use_lmsys_augmentation:
        print(f"  - LMSYS Arena: {len(train_df[train_df['source']=='lmsys']):,}")
    print(f"Val: {len(val_df):,} samples")
    print(f"\nClass balance (train):")
    print(f"  GOOD: {n_good:,} ({100*n_good/len(train_df):.1f}%)")
    print(f"  BAD:  {int(n_bad):,} ({100*n_bad/len(train_df):.1f}%)")
    print(f"  pos_weight: {pos_weight:.2f}")
    
    return train_df, val_df, pos_weight


# =============================================================================
# Model with Length Debiasing
# =============================================================================

class QualityCostPredictor(nn.Module):
    """
    Debiased Quality Predictor using "Wide & Deep" architecture.
    
    Path A (Semantic): Encoder learns actual quality from content
    Path B (Bias): Linear layer learns length bias ("short = bad")
    
    Training: Logit = Path_A + Path_B
    Inference: Logit = Path_A only (length-independent)
    """
    
    def __init__(self, config: QualityCostConfig = None):
        super().__init__()
        
        if config is None:
            config = QualityCostConfig()
        self.config = config
        
        # Encoder
        self.encoder = AutoModel.from_pretrained(config.backbone)
        self.tokenizer = AutoTokenizer.from_pretrained(config.backbone)
        hidden_size = self.encoder.config.hidden_size  # 384 for MiniLM
        
        # Path A: Semantic Quality Head (the "Deep" path)
        self.quality_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
        
        # Path B: Length Bias Head (the "Wide" path)
        # Takes log(length) as input, learns the bias
        self.length_bias_head = nn.Linear(1, 1)
        
        # Verbosity Head: Regression (0-1)
        self.verbosity_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def encode(self, input_ids, attention_mask):
        """Mean pooling."""
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_emb = outputs.last_hidden_state
        
        mask = attention_mask.unsqueeze(-1).expand(token_emb.size()).float()
        sum_emb = torch.sum(token_emb * mask, dim=1)
        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
        
        return sum_emb / sum_mask
    
    def forward(self, input_ids, attention_mask, lengths=None, return_components=False):
        """
        Forward pass with optional length debiasing.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            lengths: Token counts (only used during training for debiasing)
            return_components: If True, return (semantic_logit, bias_logit, v_pred) separately
        
        Returns:
            If return_components=False:
                q_logits: Quality logits (semantic only at inference, semantic+bias during training)
                v_pred: Verbosity prediction
            If return_components=True:
                semantic_logit, bias_logit, v_pred (for regularization)
        """
        emb = self.encode(input_ids, attention_mask)
        
        # Path A: Semantic analysis
        semantic_logit = self.quality_head(emb)
        
        # Path B: Length bias (only during training)
        if lengths is not None and self.config.use_length_debiasing:
            # Use log(length) because impact diminishes with length
            log_len = torch.log(lengths.float().unsqueeze(1) + 1)
            bias_logit = self.length_bias_head(log_len)
        else:
            bias_logit = None
        
        # Verbosity prediction (unchanged)
        v_pred = self.verbosity_head(emb)
        
        # Return components for regularization during training
        if return_components:
            return semantic_logit, bias_logit, v_pred
        
        # Combine for standard forward pass
        if bias_logit is not None:
            q_logits = semantic_logit + bias_logit
        else:
            q_logits = semantic_logit
        
        return q_logits, v_pred
    
    def predict(self, prompt: str, response: str, use_length_bias: bool = False) -> Dict[str, float]:
        """
        Predict for a single (prompt, response) pair.
        
        By default, uses only the semantic path (lengths=None) for length-independent
        quality prediction. Set use_length_bias=True to include the learned bias.
        """
        device = next(self.parameters()).device
        
        text = f"User: {prompt}\nAssistant: {response}"
        enc = self.tokenizer(
            text,
            max_length=self.config.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            input_ids = enc['input_ids'].to(device)
            attention_mask = enc['attention_mask'].to(device)
            
            if use_length_bias:
                # Use actual token length
                lengths = attention_mask.sum(dim=1)
            else:
                # Semantic path only (length-independent)
                lengths = None
            
            q_logits, v_pred = self(input_ids, attention_mask, lengths=lengths)
            p_bad = torch.sigmoid(q_logits).item()
        
        return {
            'p_bad': p_bad,
            'is_bad': p_bad > 0.5,
            'quality': 1.0 - p_bad,  # Higher = better
            'verbosity': v_pred.item(),
        }

    def predict_production(
        self,
        prompt: str,
        response: str,
        *,
        clip_eps: Optional[float] = None,
        reward_normalizer: Optional[RunningZScoreNormalizer] = None,
    ) -> Dict[str, Any]:
        """
        Production-facing output for the async bandit learning loop.

        Returns a reward signal that is:
          - quality-only (no cost/latency baked in)
          - NOT clamped to 0 (bandit must learn bad vs terrible)
          - clipped for numerical stability/log longevity
        """
        # Hard failure mode: empty responses (timeouts / "thinking loop" stripped / crashes).
        # Treat as extremely low competence.
        if response is None or len(str(response).strip()) == 0:
            eps = float(self.config.reward_clip_eps if clip_eps is None else clip_eps)
            logit_eps = float(self.config.reward_logit_eps)
            lr = LogitReward(epsilon=logit_eps)
            p_correct_raw = 0.0
            p_correct_clipped = clip01(p_correct_raw, eps=eps)
            reward_raw = clipped_quality_reward(p_correct_raw, clip_eps=eps)
            # Hard penalty lower than the minimum achievable valid logit
            reward_logit = float(lr.min_val * 1.5)
            reward_z = -3.0  # immediate "terrible" signal (normalized domain)
            t_route = self.config.routing_p_correct_threshold
            route_to_strong = None if t_route is None else True
            return {
                "p_bad": 1.0,
                "verbosity": 0.0,
                "p_correct_raw": p_correct_raw,
                "p_correct_clipped": p_correct_clipped,
                "routing_p_correct_threshold": t_route,
                "route_to_strong": route_to_strong,
                "competence_risk": 1.0,
                "reward_raw": reward_raw,
                "reward_logit": reward_logit,
                "reward_z": reward_z,
                "is_empty_response": True,
            }

        pred = self.predict(prompt, response, use_length_bias=False)
        p_bad = float(pred["p_bad"])
        p_correct_raw = float(pred["quality"])
        verbosity = float(pred["verbosity"])

        eps = float(self.config.reward_clip_eps if clip_eps is None else clip_eps)
        logit_eps = float(self.config.reward_logit_eps)
        lr = LogitReward(epsilon=logit_eps)

        p_correct_clipped = clip01(p_correct_raw, eps=eps)
        reward_raw = clipped_quality_reward(p_correct_raw, clip_eps=eps)
        reward_logit = lr.transform(p_correct_raw)
        reward_z = None if reward_normalizer is None else reward_normalizer.normalize(reward_raw, update=True)

        # Competence routing decision (Type B): route if predicted competence is below threshold.
        t_route = self.config.routing_p_correct_threshold
        route_to_strong = None if t_route is None else bool(p_correct_raw < float(t_route))

        return {
            "p_bad": p_bad,
            "verbosity": verbosity,
            "p_correct_raw": p_correct_raw,
            "p_correct_clipped": p_correct_clipped,
            "routing_p_correct_threshold": t_route,
            "route_to_strong": route_to_strong,
            "competence_risk": float(1.0 - p_correct_raw),
            "reward_raw": reward_raw,
            "reward_logit": reward_logit,
            "reward_z": reward_z,
            "is_empty_response": False,
        }
    
    def get_bias_weight(self) -> float:
        """Get the learned length bias weight (for debugging)."""
        return self.length_bias_head.weight.item()
    
    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'config': {
                'backbone': self.config.backbone,
                'max_length': self.config.max_length,
                'quality_threshold': self.config.quality_threshold,
                'use_length_debiasing': self.config.use_length_debiasing,
                'semantic_reg_weight': self.config.semantic_reg_weight,
                'reward_clip_eps': self.config.reward_clip_eps,
                'reward_logit_eps': self.config.reward_logit_eps,
                'routing_p_bad_threshold': self.config.routing_p_bad_threshold,
                'routing_p_correct_threshold': self.config.routing_p_correct_threshold,
                'anchor_keep_rate': self.config.anchor_keep_rate,
            },
            'state_dict': self.state_dict(),
        }, path)
        print(f"Saved to {path}")
    
    @classmethod
    def load(cls, path: Path, device=None):
        if device is None:
            device = get_device()
        
        ckpt = torch.load(path, map_location=device, weights_only=False)
        config = QualityCostConfig(**ckpt['config'])
        model = cls(config)
        model.load_state_dict(ckpt['state_dict'])
        model.to(device)
        model.eval()
        print(f"Loaded from {path}")
        return model


# =============================================================================
# Training
# =============================================================================

def compute_f1(preds, targets):
    """Compute F1, Precision, Recall for binary classification."""
    preds = preds.bool()
    targets = targets.bool()
    
    tp = (preds & targets).sum().float()
    fp = (preds & ~targets).sum().float()
    fn = (~preds & targets).sum().float()
    
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    
    return f1.item(), prec.item(), rec.item()


def find_best_threshold(model, val_loader, config, device=None):
    """
    Calibration script to find optimal production threshold.
    
    Uses semantic path only (lengths=None) for length-independent evaluation.
    """
    if device is None:
        device = get_device()
    
    model.eval()
    all_probs = []
    all_targets = []
    
    # Collect all predictions using semantic path only
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Calibrating", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Semantic path only (lengths=None) for length-independent evaluation
            logits, _ = model(input_ids, attention_mask, lengths=None)
            probs = torch.sigmoid(logits.squeeze(-1))
            
            all_probs.append(probs.cpu())
            all_targets.append(batch['is_bad_target'])
    
    probs = torch.cat(all_probs)
    targets = torch.cat(all_targets)
    
    # Test thresholds
    print(f"\n{'='*75}")
    print("THRESHOLD CALIBRATION (semantic path only, length-independent)")
    print(f"{'='*75}")
    print(f"{'Threshold':<10} | {'Recall':<8} | {'Precision':<10} | {'F1':<8} | {'Traffic → GPT-4'}")
    print(f"{'':<10} | {'(Safety)':<8} | {'(Cost Eff.)':<10} | {'':<8} | {'(flagged BAD)'}")
    print("-" * 75)
    
    best_threshold = 0.5
    best_f1 = 0.0
    
    for t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        preds = (probs > t).float()
        
        tp = ((preds == 1) & (targets == 1)).sum().float()
        fp = ((preds == 1) & (targets == 0)).sum().float()
        fn = ((preds == 0) & (targets == 1)).sum().float()
        
        precision = (tp / (tp + fp + 1e-8)).item()
        recall = (tp / (tp + fn + 1e-8)).item()
        f1 = (2 * precision * recall / (precision + recall + 1e-8))
        traffic = preds.mean().item()
        
        marker = ""
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
            marker = " ← best F1"
        
        print(f"{t:<10.1f} | {recall:<8.3f} | {precision:<10.3f} | {f1:<8.3f} | {traffic*100:>5.1f}%{marker}")
    
    print("-" * 75)
    print(f"Recommended threshold: {best_threshold} (best F1={best_f1:.3f})")
    print(f"{'='*75}\n")
    
    return best_threshold


def find_safety_threshold_for_recall(*args, **kwargs) -> float:
    """
    DEPRECATED.

    This project now treats the grader as a competence (Type B) signal, not policy safety.
    Keep the function name for backward compatibility, but do not use it for production.
    """
    raise RuntimeError(
        "find_safety_threshold_for_recall() is deprecated. "
        "Use routing_p_correct_threshold calibrated via anchor scores instead."
    )


def train_quality_predictor(config: QualityCostConfig = None):
    """Train with Wide & Deep debiasing."""
    if config is None:
        config = QualityCostConfig()
    
    device = get_device()
    
    print(f"\n{'='*60}")
    print("Quality/Cost Predictor Training (Length-Debiased)")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Backbone: {config.backbone}")
    print(f"Batch size: {config.batch_size}")
    print(f"LR: {config.learning_rate}")
    print(f"Epochs: {config.max_epochs}")
    print(f"Threshold: {config.quality_threshold}")
    print(f"Length Debiasing: {config.use_length_debiasing}")
    print(f"LMSYS Augmentation: {config.use_lmsys_augmentation}")
    print(f"{'='*60}\n")
    
    # Load combined data (HelpSteer2 + LMSYS Arena)
    train_df, val_df, pos_weight = load_combined_data(config)
    
    # Model
    model = QualityCostPredictor(config)
    model.to(device)
    
    # Dataloaders
    train_ds = QualityDataset(train_df, model.tokenizer, config)
    val_ds = QualityDataset(val_df, model.tokenizer, config)
    
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=0)
    
    # Loss functions
    pos_weight_tensor = torch.tensor([pos_weight], device=device, dtype=torch.float32)
    # reduction='none' so we can apply per-sample weights (edge-case upweighting)
    q_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor, reduction="none")
    v_criterion = nn.MSELoss()
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    # Training
    best_f1 = 0.0
    best_state_dict = None
    
    for epoch in range(config.max_epochs):
        # Train
        model.train()
        train_preds, train_targets = [], []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.max_epochs}")
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            is_bad = batch['is_bad_target'].to(device)
            verbosity = batch['verbosity_target'].to(device)
            sample_w = batch.get('sample_weight', torch.ones_like(is_bad)).to(device)
            # Stabilize weighting so edge-case upweighting doesn't blow up training.
            sample_w = sample_w / (sample_w.mean() + 1e-8)
            
            optimizer.zero_grad()
            
            # Get lengths for bias learning (only if debiasing enabled)
            if config.use_length_debiasing:
                lengths = attention_mask.sum(dim=1)  # Count non-padding tokens
                
                # Use return_components=True to get semantic and bias separately
                semantic_logit, bias_logit, v_pred = model(
                    input_ids, attention_mask, lengths=lengths, return_components=True
                )
                
                # Combine for main loss
                q_logits = (semantic_logit + bias_logit).squeeze(-1)
                v_pred = v_pred.squeeze(-1)
                
                # Main losses
                # IMPORTANT: semantic-only loss trains the actual inference path.
                semantic_only = semantic_logit.squeeze(-1)
                loss_q_sem = (q_criterion(semantic_only, is_bad) * sample_w).mean()
                loss_q_comb = (q_criterion(q_logits, is_bad) * sample_w).mean()
                loss_q = (config.semantic_only_loss_weight * loss_q_sem) + (config.combined_loss_weight * loss_q_comb)
                loss_v = v_criterion(v_pred, verbosity)
                
                # REGULARIZATION: Force semantic head to stay centered around 0
                # This prevents it from drifting to compensate for the bias head
                semantic_mean = torch.mean(semantic_logit)
                loss_reg = semantic_mean ** 2
                
                loss = (config.quality_weight * loss_q + 
                        config.verbosity_weight * loss_v + 
                        config.semantic_reg_weight * loss_reg)
            else:
                q_logits, v_pred = model(input_ids, attention_mask, lengths=None)
                q_logits = q_logits.squeeze(-1)
                v_pred = v_pred.squeeze(-1)
                
                loss_q = (q_criterion(q_logits, is_bad) * sample_w).mean()
                loss_v = v_criterion(v_pred, verbosity)
                loss = config.quality_weight * loss_q + config.verbosity_weight * loss_v
            
            loss.backward()
            optimizer.step()
            
            preds = (torch.sigmoid(q_logits) > 0.5).float()
            train_preds.append(preds.cpu())
            train_targets.append(is_bad.cpu())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_preds = torch.cat(train_preds)
        train_targets = torch.cat(train_targets)
        train_f1, train_prec, train_rec = compute_f1(train_preds, train_targets)
        train_acc = (train_preds == train_targets).float().mean().item()
        
        # Validate with semantic path only (length-independent)
        model.eval()
        val_preds, val_targets = [], []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating", leave=False):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                is_bad = batch['is_bad_target'].to(device)
                
                # Semantic path only (lengths=None) for length-independent evaluation
                q_logits, _ = model(input_ids, attention_mask, lengths=None)
                preds = (torch.sigmoid(q_logits.squeeze(-1)) > 0.5).float()
                
                val_preds.append(preds.cpu())
                val_targets.append(is_bad.cpu())
        
        val_preds = torch.cat(val_preds)
        val_targets = torch.cat(val_targets)
        val_f1, val_prec, val_rec = compute_f1(val_preds, val_targets)
        val_acc = (val_preds == val_targets).float().mean().item()
        
        n_pred_bad = val_preds.sum().item()
        n_actual_bad = val_targets.sum().item()
        
        # Get learned bias weight
        bias_weight = model.get_bias_weight()
        
        print(f"\nEpoch {epoch+1}/{config.max_epochs}")
        print(f"  Train: F1={train_f1:.3f} | Acc={train_acc:.3f} | Prec={train_prec:.3f} | Rec={train_rec:.3f}")
        print(f"  Val (debiased): F1={val_f1:.3f} | Acc={val_acc:.3f} | Prec={val_prec:.3f} | Rec={val_rec:.3f}")
        print(f"  Predicted {int(n_pred_bad)} BAD / {len(val_preds)} (actual: {int(n_actual_bad)} BAD)")
        print(f"  Length Bias Weight: {bias_weight:.3f} (negative = 'short is bad')")
        
        # Save best
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            model.save(config.checkpoint_dir / "best_quality_predictor.pt")
            print(f"  ✓ New best model (F1={val_f1:.3f})")
    
    print(f"\n{'='*60}")
    print(f"Training complete! Best Val F1: {best_f1:.3f}")
    print(f"Learned Length Bias Weight: {model.get_bias_weight():.3f}")
    print(f"{'='*60}")
    
    # Restore best weights for calibration + final save
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # Run threshold calibration with Goldilocks Injection (semantic path only)
    # NOTE: This is a *routing* threshold helper on the training distribution, not a
    # production "competence anchor" threshold. Production should use
    # routing_p_correct_threshold calibrated from a strong-model anchor set.
    best_threshold = find_best_threshold(model, val_loader, config, device)
    model.config.routing_p_bad_threshold = float(best_threshold)

    # Persist calibrated production thresholds in the checkpoint.
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save(config.checkpoint_dir / "best_quality_predictor.pt")
    
    return model, val_loader, best_threshold


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    # Get defaults from config
    _defaults = QualityCostConfig()
    
    parser = argparse.ArgumentParser(description="Train Quality/Cost Predictor")
    parser.add_argument("--epochs", type=int, default=_defaults.max_epochs)
    parser.add_argument("--batch-size", type=int, default=_defaults.batch_size)
    parser.add_argument("--lr", type=float, default=_defaults.learning_rate)
    parser.add_argument("--threshold", type=float, default=_defaults.quality_threshold)
    parser.add_argument("--no-debias", action="store_true", help="Disable length debiasing")
    parser.add_argument("--no-lmsys", action="store_true", help="Disable LMSYS Arena augmentation")
    parser.add_argument("--lmsys-max", type=int, default=_defaults.lmsys_max_samples, 
                        help="Max samples from LMSYS per class")
    parser.add_argument("--det-edgecases", action="store_true", help="Add deterministic STEM edge-case augmentation")
    parser.add_argument("--det-edgecases-n", type=int, default=_defaults.deterministic_edgecases_n)
    parser.add_argument("--det-edgecases-weight", type=float, default=_defaults.deterministic_edgecases_weight)
    parser.add_argument("--semantic-only-w", type=float, default=_defaults.semantic_only_loss_weight)
    parser.add_argument("--combined-w", type=float, default=_defaults.combined_loss_weight)
    
    args = parser.parse_args()
    
    config = QualityCostConfig(
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        quality_threshold=args.threshold,
        use_length_debiasing=not args.no_debias,
        use_lmsys_augmentation=not args.no_lmsys,
        lmsys_max_samples=args.lmsys_max,
        use_deterministic_edgecases=bool(args.det_edgecases),
        deterministic_edgecases_n=int(args.det_edgecases_n),
        deterministic_edgecases_weight=float(args.det_edgecases_weight),
        semantic_only_loss_weight=float(args.semantic_only_w),
        combined_loss_weight=float(args.combined_w),
    )
    
    model, val_loader, best_threshold = train_quality_predictor(config)
    
    # Test with calibrated threshold
    print("\n" + "="*60)
    print(f"Test Predictions (debiased, threshold={best_threshold})")
    print("="*60 + "\n")
    
    test_cases = [
        ("What is 2+2?", "The answer is 4."),
        ("What is 2+2?", "I think maybe fish?"),
        ("Write Python to sort a list", "sorted(lst)"),
        ("Explain quantum computing", "I don't know."),
        ("Capital of France?", "Paris"),
        ("What is the speed of light?", "Approximately 299,792,458 meters per second."),
        # Deterministic STEM edge-case sanity checks (dilute strong acid pH)
        (r"Calculate the pH of a $10^{-8}$ M solution of HCl.", "pH = 8 (because pH = -log10(1e-8))."),
        (r"Calculate the pH of a $10^{-8}$ M solution of HCl.", "Include water autoionization: [H+]=(C+sqrt(C^2+4Kw))/2 ≈ 1.05e-7 so pH ≈ 6.98."),
    ]
    
    model.eval()
    print(f"Length Bias Weight: {model.get_bias_weight():.3f}")
    print("-" * 60)
    
    for prompt, response in test_cases:
        r = model.predict(prompt, response)
        is_bad = r['p_bad'] > best_threshold
        status = "✗ BAD → route to GPT-4" if is_bad else "✓ GOOD → use cheap model"
        print(f"Q: {prompt[:40]}")
        print(f"A: {response[:50]}")
        print(f"   P(BAD)={r['p_bad']:.3f} {status}")
        print()
