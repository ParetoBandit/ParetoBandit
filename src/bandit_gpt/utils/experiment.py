import re
import json
import gzip
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
from collections import Counter
from sklearn.model_selection import train_test_split
# from src.bandit_gpt.router import BanditRouter # Removed to avoid circular import

class ExperimentBurnIn:
    """
    Facilitates the KDD-compliant burn-in process for experiments.
    
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
    def create_canonical_splits(
        oracle_rewards: Dict[str, Dict[str, float]],
        splits_path: Path,
        test_ratio: float = 0.4,
        random_state: int = 42
    ) -> Tuple[List[str], List[str]]:
        """
        Generate and save stratified canonical train/test splits.
        
        This ensures both pools have similar complexity and category distributions.
        """
        all_prompts = list(oracle_rewards.keys())
        
        # Calculate stratification keys
        print(f"🧬 Calculating stratification keys for {len(all_prompts)} prompts...")
        strata = []
        for p in all_prompts:
            key = ExperimentBurnIn._get_stratification_key(p, oracle_rewards[p])
            strata.append(key)
            
        # Handle sparse strata (train_test_split requires >= 2 members per stratum)
        counts = Counter(strata)
        final_strata = []
        for s in strata:
            if counts[s] < 2:
                # Merge into a fallback stratum
                parts = s.split("_")
                fallback = f"{parts[0]}_Mixed_Mixed"
                final_strata.append(fallback)
            else:
                final_strata.append(s)
                
        # Re-check sparse strata after merging
        final_counts = Counter(final_strata)
        for i, s in enumerate(final_strata):
            if final_counts[s] < 2:
                final_strata[i] = "GENERAL_Mixed_Mixed"

        # Generate strict stratified split
        dev_pool, holdout_pool = train_test_split(
            all_prompts, 
            test_size=test_ratio, 
            random_state=random_state,
            stratify=final_strata
        )
        
        # Verify disjointness
        dev_set = set(dev_pool)
        holdout_set = set(holdout_pool)
        overlap = dev_set.intersection(holdout_set)
        if overlap:
            raise ValueError(
                f"❌ CRITICAL: Data leakage detected! "
                f"Found {len(overlap)} overlapping prompts between dev and holdout."
            )
        
        # Save splits
        splits_path.parent.mkdir(parents=True, exist_ok=True)
        with open(splits_path, "w") as f:
            json.dump({
                "dev_pool": dev_pool,
                "holdout_pool": holdout_pool
            }, f, indent=2)
        
        print(f"✓ Created stratified canonical splits:")
        print(f"  - Dev: {len(dev_pool)} prompts")
        print(f"  - Holdout: {len(holdout_pool)} prompts")
        print(f"  - Strata count: {len(set(final_strata))}")
        print(f"  - Saved to: {splits_path}")
        
        return dev_pool, holdout_pool

    def get_splits(
        self, 
        load_rewards: bool = False
    ) -> Tuple[List[str], List[str]] | Tuple[Tuple[List[str], Dict], Tuple[List[str], Dict]]:
        """
        Loads pre-configured splits from 'splits.json'.
        
        Args:
            load_rewards: If True, also load split-specific rewards and return them
                         joined with the prompts. This uses dev_rewards.jsonl.gz and
                         holdout_rewards.jsonl.gz for automatic joining.
        
        Returns:
            If load_rewards=False (default):
                Tuple[List[str], List[str]]: (dev_prompts, holdout_prompts)
            
            If load_rewards=True:
                Tuple[
                    Tuple[List[str], Dict[str, Dict[str, float]]],  # (dev_prompts, dev_rewards)
                    Tuple[List[str], Dict[str, Dict[str, float]]]   # (holdout_prompts, holdout_rewards)
                ]
        
        Example:
            >>> # Without rewards (backward compatible)
            >>> dev_prompts, holdout_prompts = burner.get_splits()
            
            >>> # With rewards automatically joined
            >>> (dev_prompts, dev_rewards), (holdout_prompts, holdout_rewards) = burner.get_splits(load_rewards=True)
        """
        if not self.splits_path.exists():
            raise FileNotFoundError(
                f"❌ Critical Error: {self.splits_path} not found. "
                "Ensure canonical KDD dev/test splits are generated."
            )
            
        with open(self.splits_path) as f:
            splits_data = json.load(f)
            
        dev = splits_data["dev_pool"]
        test = splits_data["holdout_pool"]
        
        # Verify Disjointness
        overlap = set(dev).intersection(set(test))
        if overlap:
            raise ValueError(f"❌ DATA LEAKAGE DETECTED! Found {len(overlap)} overlapping prompts.")
        
        if not load_rewards:
            return dev, test
        
        # Load split-specific rewards
        import gzip
        from pathlib import Path
        
        # __file__ is in src/bandit_gpt/utils/experiment.py
        # Go up to src/bandit_gpt, then to data/offline_dataset
        data_dir = Path(__file__).parent.parent / "data" / "offline_dataset"
        dev_rewards_path = data_dir / "dev_rewards.jsonl.gz"
        holdout_rewards_path = data_dir / "holdout_rewards.jsonl.gz"
        
        registry_models = set(self.registry.keys())
        
        # Load dev rewards
        dev_rewards: Dict[str, Dict[str, float]] = {}
        if dev_rewards_path.exists():
            with gzip.open(dev_rewards_path, 'rt') as f:
                for line in f:
                    entry = json.loads(line)
                    if entry.get("ok"):
                        prompt = entry["prompt"]
                        model_id = entry["model_id"]
                        reward = entry["raw_score"]
                        
                        # Filter to registry models
                        if model_id not in registry_models:
                            continue
                            
                        if prompt not in dev_rewards:
                            dev_rewards[prompt] = {}
                        dev_rewards[prompt][model_id] = reward
        else:
            raise FileNotFoundError(
                f"❌ dev_rewards.jsonl.gz not found at {dev_rewards_path}\n"
                f"   Run: python3 scripts/create_split_rewards.py"
            )
        
        # Load holdout rewards
        holdout_rewards: Dict[str, Dict[str, float]] = {}
        if holdout_rewards_path.exists():
            with gzip.open(holdout_rewards_path, 'rt') as f:
                for line in f:
                    entry = json.loads(line)
                    if entry.get("ok"):
                        prompt = entry["prompt"]
                        model_id = entry["model_id"]
                        reward = entry["raw_score"]
                        
                        # Filter to registry models
                        if model_id not in registry_models:
                            continue
                            
                        if prompt not in holdout_rewards:
                            holdout_rewards[prompt] = {}
                        holdout_rewards[prompt][model_id] = reward
        else:
            raise FileNotFoundError(
                f"❌ holdout_rewards.jsonl.gz not found at {holdout_rewards_path}\n"
                f"   Run: python3 scripts/create_split_rewards.py"
            )
        
        self.oracle_rewards.update(dev_rewards)
        self.oracle_rewards.update(holdout_rewards)
        
        return (dev, dev_rewards), (test, holdout_rewards)


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
                
        # FIXED: Oversample hard prompts WITH REPLACEMENT to avoid exact duplicates
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
        priors: str = "warmup", 
        prior_n_effective: float = 1.0, 
        alpha: float = 0.1
    ) -> Tuple[object, List[str]]:
        """
        Full workflow: Load splits, generate curriculum, and create a hot router.
        
        Args:
            priors: Prior initialization strategy ("none", "hle", "warmup").
            prior_n_effective: Effective sample size for priors.
            alpha: Exploration parameter for the bandit.
            
        Returns:
            Tuple[BanditRouter, List[str]]: (burned_in_router, test_prompts)
        """
        from src.bandit_gpt.router import BanditRouter
        
        dev_prompts, test_prompts = self.get_splits()
        curriculum = self.generate_curriculum(dev_prompts)
        
        # Use existing PCA artifact (don't recreate)
        from pathlib import Path as P
        pca_path = P(__file__).parent.parent.parent / "artifacts" / "pca_23.joblib"
        
        router = BanditRouter.create(
            self.registry,
            context_encoder=self.encoder,
            priors=priors,
            prior_n_effective=prior_n_effective,
            update_lambda=1.0,  # Strong ongoing regularization for intensive burn-in
            pca_path=pca_path  # Use existing PCA data
        )
        router.bandit.alpha = alpha
        
        self.perform_burn_in(router, curriculum)
        
        return router, test_prompts
