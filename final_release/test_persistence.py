import os
import numpy as np
from pathlib import Path
try:
    from .bandit import BanditRouter
except (ImportError, ValueError):
    try:
        from final_release.bandit import BanditRouter
    except (ImportError, ValueError):
        from bandit import BanditRouter

def test_persistence():
    print("Testing Bandit Router Persistence...")
    
    # 1. Initialize router
    router1 = BanditRouter.create(priors="none") # Start cold for clear test
    model = list(router1.registry.keys())[0]
    prompt = "Hello, how are you?"
    
    # Get initial prediction
    x = router1.encoder.encode(prompt)
    x = x / np.linalg.norm(x)
    _, ucb1 = router1.bandit.select_arm(x, candidates=[model])
    print(f"Initial UCB for {model}: {ucb1:.6f}")
    
    # 2. Perform an update
    reward = 0.9
    router1.bandit.update(model, x, reward)
    _, ucb2 = router1.bandit.select_arm(x, candidates=[model])
    print(f"UCB after update: {ucb2:.6f}")
    assert ucb2 != ucb1, "UCB should change after update"
    
    # 3. Save state
    state_path = "test_state.npz"
    router1.save_state(state_path)
    print(f"State saved to {state_path}")
    
    # 4. Create new router from state
    router2 = BanditRouter.create(state_path=state_path)
    _, ucb3 = router2.bandit.select_arm(x, candidates=[model])
    print(f"UCB from loaded state: {ucb3:.6f}")
    
    # 5. Verify
    assert abs(ucb3 - ucb2) < 1e-6, f"UCB mismatch! Expected {ucb2}, got {ucb3}"
    print("✓ Persistence verified! State correctly loaded.")
    
    # Cleanup
    if os.path.exists(state_path):
        os.remove(state_path)

if __name__ == "__main__":
    test_persistence()
