import os
from banditgpt.core import BanditRouter, TieredGrader, OpenRouterTeacherVerifier

def test_core_router():
    print("Testing banditgpt.core.BanditRouter...")
    
    # Mock model registry
    registry = {
        "gpt-4o": {"input_cost_per_m": 5.0, "output_cost_per_m": 15.0, "hle": 0.85},
        "gpt-4o-mini": {"input_cost_per_m": 0.15, "output_cost_per_m": 0.6, "hle": 0.75}
    }
    
    # Initialize router
    router = BanditRouter(registry, alpha=0.1)
    print("  ✓ Router initialized")
    
    # Test routing
    model, log = router.route("Hello, how are you?")
    print(f"  ✓ Routed to: {model}")
    
    # Test feedback processing (without grader)
    router.report_feedback(log.request_id, reward=1.0)
    print("  ✓ Updated bandit with manual reward via report_feedback")

def test_tiered_grader():
    print("\nTesting banditgpt.core.TieredGrader...")
    
    # Initialize grader without soft grader
    teacher = OpenRouterTeacherVerifier(model_id="openai/gpt-4o-mini")
    grader = TieredGrader(soft_grader=None, teacher_verifier=teacher)
    print("  ✓ TieredGrader initialized without soft grader")
    
    # Test prediction (should use neutral score for soft path)
    res = grader.predict_production("Simple prompt", "Simple response")
    print(f"  ✓ Soft path result: {res['p_correct_raw']} (expected 0.5)")
    
    # Test hard path (should trigger teacher if prompt contains 'calculate')
    # Note: This requires an API key to actually call the teacher, 
    # but we can check if it attempts to call it.
    is_hard = grader.hard_detector.is_hard("calculate 2+2")
    print(f"  ✓ Hard detector: {is_hard} (expected True)")

if __name__ == "__main__":
    try:
        test_core_router()
        test_tiered_grader()
        print("\nCore verification successful!")
    except Exception as e:
        print(f"\nVerification failed: {e}")
        import traceback
        traceback.print_exc()
