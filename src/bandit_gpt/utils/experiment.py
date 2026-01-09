import json
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
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
        oracle_rewards: Dict[str, Dict[str, float]], 
        splits_path: Path,
        encoder = None
    ):
        self.registry = registry
        self.oracle_rewards = oracle_rewards
        self.splits_path = Path(splits_path)
        self.encoder = encoder

    def get_splits(self) -> Tuple[List[str], List[str]]:
        """
        Loads pre-configured splits from 'splits.json'.
        
        Returns:
            Tuple[List[str], List[str]]: (dev_pool, holdout_pool)
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
            
        return dev, test

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
                
        # Oversample Hard 3x to match the volume of Easy prompts
        burn_in_list = []
        burn_in_list.extend(hard_train * 3)
        
        # Sample easy prompts to match the hard volume (50/50 split)
        target_len = len(burn_in_list) 
        if easy_train:
            selected_easy = np.random.choice(
                easy_train, 
                min(len(easy_train), target_len), 
                replace=False
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
        
        router = BanditRouter.create(
            self.registry,
            context_encoder=self.encoder,
            priors=priors,
            prior_n_effective=prior_n_effective
        )
        router.bandit.alpha = alpha
        
        self.perform_burn_in(router, curriculum)
        
        return router, test_prompts
