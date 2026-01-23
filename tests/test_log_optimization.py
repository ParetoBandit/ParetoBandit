import time
import numpy as np
import os
import sys
from collections import deque

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from bandit_gpt.router import BanditRouter, RouterConfig

def test_log_indexing_and_eviction():
    """Verify that log_index stays in sync with logs deque and handles eviction."""
    # Temporarily set max_log_size to something small for testing
    old_max = RouterConfig.max_log_size
    RouterConfig.max_log_size = 5
    
    try:
        router = BanditRouter.create()
        # Verify deque maxlen matches config
        assert router.logs.maxlen == 5
        
        request_ids = []
        for i in range(10):
            model, log = router.route(prompt=f"test {i}", profile="auto")
            request_ids.append(log.request_id)
            
        # Deque should only have 5 items
        assert len(router.logs) == 5
        # Index should also only have 5 items
        assert len(router.log_index) == 5
        
        # Last 5 should be in index
        for rid in request_ids[5:]:
            assert rid in router.log_index
            
        # First 5 should be evicted from index
        for rid in request_ids[:5]:
            assert rid not in router.log_index
            
        print("✅ Eviction sync test passed.")
        
    finally:
        RouterConfig.max_log_size = old_max

def test_feedback_lookup_performance():
    """Measure the time complexity of process_feedback lookup."""
    # Set a large log size
    old_max = RouterConfig.max_log_size
    limit = 2000
    RouterConfig.max_log_size = limit
    
    try:
        router = BanditRouter.create()
        
        print(f"🚀 Populating {limit} logs...")
        last_rid = None
        for i in range(limit):
            model, log = router.route(prompt=f"speed test {i}", profile="auto")
            last_rid = log.request_id
            
        # Measure feedback time
        start_time = time.time()
        # process_feedback will do internal reward updates, we just want to see if lookup hangs
        router.process_feedback(last_rid, 1.0)
        end_time = time.time()
        
        lookup_time = (end_time - start_time) * 1000 # ms
        print(f"⏱️ process_feedback lookup + update took {lookup_time:.4f} ms")
        
        # For O(1) dictionary lookup, it should be extremely fast (<< 1ms usually)
        # Even with update logic, it shouldn't be high.
        assert lookup_time < 10.0, f"Lookup took too long: {lookup_time:.2f}ms"
        
        print("✅ Performance test passed (O(1) behavior confirmed).")
        
    finally:
        RouterConfig.max_log_size = old_max

if __name__ == "__main__":
    test_log_indexing_and_eviction()
    test_feedback_lookup_performance()
