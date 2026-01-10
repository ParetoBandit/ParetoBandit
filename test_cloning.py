
import copy
import threading
from src.bandit_gpt.router import BanditRouter

# Mock registry
registry = {"test-model": {"hle": 0.3}}

# Create router
router = BanditRouter.create(registry)
print("Router created.")

# Test deepcopy
try:
    router_copy = copy.deepcopy(router)
    print("Cloning successful!")
    
    # Verify lock exists in clone
    if hasattr(router_copy.bandit, "_lock"):
        print("Clone has its own lock.")
    else:
        print("Clone missing lock!")
        
    # Verify encoder is shared (same object)
    if router.encoder is router_copy.encoder:
        print("Encoder is correctly shared.")
    else:
        print("Encoder was unnecessarily copied!")
        
except Exception as e:
    print(f"Cloning failed: {e}")
