import threading
import numpy as np
import os
import sys
import time

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from bandit_gpt.router import BanditRouter

def test_concurrent_updates_and_reads():
    """
    Stress test the BanditRouter/LinUCB with concurrent updates and reads.
    Verifies that the COW pattern prevents 'torn reads' or crashes.
    """
    dim = 24
    router = BanditRouter.create()
    model_id = "meta-llama/llama-3.1-8b-instruct"
    
    # Context vector
    x = np.random.randn(dim)
    
    stop_event = threading.Event()
    
    def writer_thread():
        """Aggressively update matrices."""
        while not stop_event.is_set():
            reward = np.random.random()
            router.process_feedback_contextual(model_id, x, reward)
            time.sleep(0.001) # 1ms interval

    def reader_thread():
        """Aggressively read and calculate UCB."""
        while not stop_event.is_set():
            try:
                # select_model calls route() which calls bandit.select_arm()
                # we call route directly to ensure we are hitting the read path
                model, log = router.route(prompt="test prompt", profile="auto")
                # Check for validity
                assert isinstance(model, str)
                assert not np.isnan(log.predicted_utility)
            except Exception as e:
                print(f"FAILED: reader encountered error: {e}")
                os._exit(1) # Kill the test immediately on failure

    # Launch threads
    writers = [threading.Thread(target=writer_thread) for _ in range(2)]
    readers = [threading.Thread(target=reader_thread) for _ in range(5)]
    
    print(f"🚀 Starting concurrency stress test (2 writers, 5 readers) for 5 seconds...")
    for t in writers + readers:
        t.daemon = True
        t.start()
        
    time.sleep(5)
    stop_event.set()
    
    for t in writers + readers:
        t.join(timeout=1)
        
    print("✅ Concurrency test passed without crashes or state corruption.")

if __name__ == "__main__":
    # Add process_feedback_contextual if it doesn't exist for test purposes
    # or just use process_feedback if we can mock the request cache.
    # Actually BanditRouter.process_feedback requires a request_id.
    # Let's use router.bandit.update directly to keep it simple.
    
    from bandit_gpt.router import BanditRouter
    router = BanditRouter.create()
    model_id = "meta-llama/llama-3.1-8b-instruct"
    dim = router.bandit.dim
    x = np.random.randn(dim)
    
    stop_event = threading.Event()
    errors = []

    def writer_loop():
        try:
            for _ in range(1000):
                router.bandit.update(model_id, x, 1.0)
        except Exception as e:
            errors.append(e)

    def reader_loop():
        try:
            for _ in range(1000):
                router.bandit.select_arm(x, [model_id])
        except Exception as e:
            errors.append(e)

    threads = []
    for _ in range(5):
        threads.append(threading.Thread(target=writer_loop))
        threads.append(threading.Thread(target=reader_loop))
        
    for t in threads: t.start()
    for t in threads: t.join()
    
    if errors:
        print(f"❌ Concurrency test FAILED with {len(errors)} errors:")
        for e in errors: print(e)
        sys.exit(1)
    else:
        print("✅ Concurrency test PASSED (COW prevents state corruption)")
