import json
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
from collections import Counter
from sklearn.model_selection import train_test_split

from bandit_gpt.rewards import extract_reward
# from src.bandit_gpt.router import BanditRouter # Removed to avoid circular import

class ExperimentBurnIn:
    """
    Facilitates the Conference-compliant burn-in process for experiments.
    
    This class centralizes the loading of splits, curriculum generation, 
    and router burn-in logic to ensure consistency across different evaluation runs.
    """
    
    def __init__(
        self, 
        registry: Dict[str, Dict], 
        oracle_rewards: Dict[str, Dict[str, float]] = None, 
        splits_path: Path = None,
        encoder = None
    ):
        self.registry = registry
        self.oracle_rewards = oracle_rewards or {}
        self.splits_path = Path(splits_path) if splits_path else None
        self.encoder = encoder

    @staticmethod
    def _get_stratification_key(
        prompt: str,
        rewards_map: Dict[str, float]
    ) -> str:
        """
        Calculates a coarse stratification key for a prompt.
        
        Axes:
        1. Category (STEM, CODE, GENERAL)
        2. Complexity (Low, Med, High)
        3. Difficulty/Signal (Stable-Easy, Stable-Hard, Contentious)
        """
        prompt_lower = prompt.lower()
        
        # 1. Category Heuristics
        category = "GENERAL"
        if any(kw in prompt_lower for kw in ["integral", "derivative", "theorem", "proof", "math", "calculus", "equation"]):
            category = "STEM"
        elif any(kw in prompt_lower for kw in ["code", "function", "class", "debug", "python", "javascript", "rust", "```"]):
            category = "CODE"
            
        # 2. Complexity Heuristics (Score 0-1)
        score = 0.0
        if len(prompt) > 500: score += 0.3
        if len(prompt) > 2000: score += 0.2
        if "```" in prompt: score += 0.2
        if any(kw in prompt_lower for kw in ["step-by-step", "explain", "why", "how"]): score += 0.1
        if any(kw in prompt_lower for kw in ["without", "avoid", "only", "constraint"]): score += 0.2
        
        complexity = "Low"
        if score >= 0.7: complexity = "High"
        elif score >= 0.3: complexity = "Med"
        
        # 3. Difficulty/Signal (from Oracle Rewards)
        rewards = list(rewards_map.values())
        if not rewards:
            signal = "Unknown"
        else:
            var = np.var(rewards)
            avg = np.mean(rewards)
            
            if var > 0.05:
                signal = "Contentious"
            elif avg < 0.8:
                signal = "Hard"
            else:
                signal = "Easy"
                
        return f"{category}_{complexity}_{signal}"

    @staticmethod
    def create_three_way_splits(
        oracle_rewards: Dict[str, Dict[str, float]],
        splits_path: Path,
        prior_ratio: float = 0.40,
        random_state: int = 42,
        min_models: int = 43,
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Generate stratified three-way split: prior-training / online-learning / holdout.

        The prior-training set is used to build warmup priors for K-model
        experiments.  The online-learning set feeds the bandit during
        evaluation.  The holdout set (from the existing canonical split)
        remains untouched.

        This method splits only the *dev* prompts that have full model
        coverage (>= min_models).  Holdout prompts are loaded from the
        existing canonical split and verified disjoint.

        Args:
            oracle_rewards: {prompt: {model: reward}} for dev prompts.
            splits_path:    Where to save the three-way split JSON.
            prior_ratio:    Fraction of dev prompts allocated to prior
                            training (default 0.40).
            random_state:   Reproducibility seed.
            min_models:     Minimum model coverage required per prompt.

        Returns:
            (prior_train_prompts, online_learn_prompts, holdout_prompts)
        """
        # Filter to full-coverage prompts
        full_cov = {
            p: rewards
            for p, rewards in oracle_rewards.items()
            if len(rewards) >= min_models
        }
        all_prompts = list(full_cov.keys())
        print(f"  {len(all_prompts)} prompts with >= {min_models} model coverage")

        # Stratify using the same axes as create_canonical_splits.
        # Merge any class with < 2 members into the largest existing
        # stratum so that sklearn's stratified split can proceed.
        strata = [
            ExperimentBurnIn._get_stratification_key(p, full_cov[p])
            for p in all_prompts
        ]
        final_strata = list(strata)
        counts = Counter(final_strata)
        largest = counts.most_common(1)[0][0]
        for i, s in enumerate(final_strata):
            if counts[s] < 2:
                final_strata[i] = largest

        prior_train, online_learn = train_test_split(
            all_prompts,
            test_size=1.0 - prior_ratio,
            random_state=random_state,
            stratify=final_strata,
        )

        # Pairwise disjointness
        sets = {
            "prior_train": set(prior_train),
            "online_learn": set(online_learn),
        }
        for a_name, a_set in sets.items():
            for b_name, b_set in sets.items():
                if a_name >= b_name:
                    continue
                overlap = a_set & b_set
                if overlap:
                    raise ValueError(
                        f"DATA LEAKAGE: {len(overlap)} prompts overlap "
                        f"between {a_name} and {b_name}"
                    )

        splits_path.parent.mkdir(parents=True, exist_ok=True)
        with open(splits_path, "w") as f:
            json.dump(
                {
                    "prior_train_pool": prior_train,
                    "online_learn_pool": online_learn,
                    "min_models": min_models,
                    "prior_ratio": prior_ratio,
                    "random_state": random_state,
                },
                f,
                indent=2,
            )

        print(f"  Created three-way split (seed={random_state}):")
        print(f"    Prior training : {len(prior_train)} prompts")
        print(f"    Online learning: {len(online_learn)} prompts")
        print(f"    Strata count   : {len(set(final_strata))}")
        print(f"    Saved to       : {splits_path}")

        return prior_train, online_learn

    def generate_curriculum(self, dev_prompts: List[str]) -> List[str]:
        """
        Generates a signal-aware curriculum by oversampling contentious prompts.
        
        Args:
            dev_prompts: List of prompts from the development pool.
            
        Returns:
            List[str]: Shuffled curriculum for burn-in.
        """
        hard_train = []
        easy_train = []
        
        for p in dev_prompts:
            rewards = list(self.oracle_rewards.get(p, {}).values())
            if not rewards: 
                continue
            
            # Variance > 0.05 means models disagree (High Signal)
            if np.var(rewards) > 0.05:
                hard_train.append(p)
            else:
                easy_train.append(p)
                
        # Oversample hard prompts with replacement to emphasize contention while
        # avoiding degenerate duplication patterns.
        # Strategy: Create 3x samples from hard pool (can include repeats),
        # then balance with easy prompts for 50/50 split
        burn_in_list = []
        
        if hard_train:
            # Sample with replacement to get diversity while emphasizing hard prompts
            hard_samples = np.random.choice(
                hard_train,
                size=len(hard_train) * 3,
                replace=True  # This creates variety in which prompts are repeated
            ).tolist()
            burn_in_list.extend(hard_samples)
        
        # Sample easy prompts to match the hard volume (50/50 split)
        target_len = len(burn_in_list) 
        if easy_train:
            selected_easy = np.random.choice(
                easy_train, 
                min(len(easy_train), target_len), 
                replace=True  # Allow repeats to reach target size
            ).tolist()
            burn_in_list.extend(selected_easy)
            
        random.shuffle(burn_in_list)
        return burn_in_list

    def perform_burn_in(self, router, burn_in_list: List[str]):
        """
        Executes the burn-in loop on the provided router.
        
        Args:
            router: The BanditRouter instance to warm up.
            burn_in_list: The curriculum prompts to learn from.
            
        Returns:
            BanditRouter: The burned-in router.
        """
        for prompt in tqdm(burn_in_list, desc="  Burn-in", leave=False):
            # 1. Select Arm (arbitrage profile for learning balance)
            model_id, _ = router.route(prompt, profile="arbitrage")
            
            # 2. Get Reward (Oracle)
            reward = self.oracle_rewards.get(prompt, {}).get(model_id, 0.0)
            
            # 3. Update Bandit
            router.update(model_id, prompt, reward)
            
        return router

    def create_burned_in_router(
        self,
        priors: str = "none",
        prior_n_effective: float = 1.0,
        alpha: float = 0.1,
    ) -> object:
        """
        Full workflow: load the val split, generate curriculum, and return a
        burned-in BanditRouter.

        Uses ``VAL_DATA_PATH_ALL_MODELS`` (1,543 prompts) for the online-learn
        curriculum and ``DEFAULT_PCA_PATH`` for the feature encoder.  Holdout
        evaluation should be performed separately using
        ``HOLDOUT_DATA_PATH_ALL_MODELS``.

        Args:
            priors: Prior initialization strategy — ``"none"`` for a cold
                start, or a path to a ``.joblib`` file with pre-computed
                warmup priors.
            prior_n_effective: Effective sample size for the priors.
            alpha: LinUCB exploration parameter.

        Returns:
            BanditRouter: The burned-in router, ready for holdout evaluation.

        Raises:
            FileNotFoundError: If ``VAL_DATA_PATH_ALL_MODELS`` does not exist.
        """
        import gzip as _gzip
        from bandit_gpt.router import BanditRouter
        from bandit_gpt.config import VAL_DATA_PATH_ALL_MODELS, DEFAULT_PCA_PATH
        from bandit_gpt.rewards import extract_reward

        if not VAL_DATA_PATH_ALL_MODELS.exists():
            raise FileNotFoundError(
                f"Val rewards not found at {VAL_DATA_PATH_ALL_MODELS}."
            )

        val_prompts: List[str] = []
        seen: set = set()
        with _gzip.open(VAL_DATA_PATH_ALL_MODELS, "rt") as f:
            for line in f:
                p = json.loads(line)["prompt"]
                if p not in seen:
                    seen.add(p)
                    val_prompts.append(p)

        curriculum = self.generate_curriculum(val_prompts)

        router = BanditRouter.create(
            self.registry,
            context_encoder=self.encoder,
            priors=priors,
            prior_n_effective=prior_n_effective,
            pca_path=DEFAULT_PCA_PATH,
        )
        router.bandit.alpha = alpha

        self.perform_burn_in(router, curriculum)
        return router
