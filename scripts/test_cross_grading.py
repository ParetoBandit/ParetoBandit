import os
import logging
from unittest.mock import MagicMock
from banditgpt.core.tiered_grader import TieredGrader, OpenRouterTeacherVerifier

# Setup logging to see the switch
logging.basicConfig(level=logging.INFO)

def test_cross_grading():
    print("Testing Cross-Grading Logic...")
    
    # 1. Mock the soft grader
    soft_grader = MagicMock()
    soft_grader.predict_production.return_value = {"reward_raw": 0.8, "p_correct_raw": 0.8}
    
    # 2. Setup the teacher verifier (GPT-4o)
    teacher = OpenRouterTeacherVerifier(model_id="openai/gpt-4o")
    # Mock the verify call to avoid actual API hits
    teacher.verify = MagicMock(return_value=(0.9, {"ok": True, "teacher_model": "openai/gpt-4o"}))
    
    # 3. Create the TieredGrader with a cross-judge
    cross_judge = "anthropic/claude-3-5-sonnet-20241022"
    grader = TieredGrader(
        soft_grader=soft_grader,
        teacher_verifier=teacher,
        cross_judge_model_id=cross_judge
    )
    
    # Test Case A: Grading a DIFFERENT model (e.g., Llama 3)
    print("\nCase A: Grading 'meta-llama/llama-3-70b' (Should use GPT-4o)")
    grader.predict_production("Solve 2+2", "4", model_id="meta-llama/llama-3-70b")
    # Check if teacher.model_id was gpt-4o during the call
    # Since we mocked verify, we can't easily check the internal state during the call 
    # unless we wrap it, but we can check if it's still gpt-4o after.
    print(f"Current teacher model: {teacher.model_id}")
    
    # Test Case B: Grading GPT-4o (Should switch to Claude)
    print("\nCase B: Grading 'openai/gpt-4o' (Should switch to Claude)")
    
    # We need to verify that teacher.verify was called with the cross_judge_model_id
    # Let's update the mock to record the model_id at the time of call
    call_model_ids = []
    def mock_verify(p, r):
        call_model_ids.append(teacher.model_id)
        return (0.9, {"ok": True})
    
    teacher.verify = mock_verify
    
    grader.predict_production("Solve 2+2", "4", model_id="openai/gpt-4o")
    
    print(f"Model IDs used during calls: {call_model_ids}")
    
    if cross_judge in call_model_ids:
        print("\nSUCCESS: Cross-grading detected! GPT-4o was graded by Claude.")
    else:
        print("\nFAILURE: GPT-4o was graded by itself.")

if __name__ == "__main__":
    test_cross_grading()
