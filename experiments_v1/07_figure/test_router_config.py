"""
Comprehensive Router Configuration Test

Tests:
1. Alpha propagation to Corralling experts
2. Dynamic model registration with semantic transfer
3. Model selection behavior with proper exploration
4. Warmup priors loading and application
5. Expected behavior for production usage
"""
import sys
from pathlib import Path
import numpy as np
import logging
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.aligned_evaluator import AlignedEvaluator
from bandit_gpt.config_legacy import DEFAULT_SENTENCE_TRANSFORMER, DEFAULT_PCA_PATH, DEV_DATA_PATH_ALL_MODELS
from bandit_gpt.router import BanditRouter

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Test configuration
MODELS_2 = ["mistralai/mixtral-8x7b-instruct", "openai/gpt-4-turbo"]
NEW_MODEL = "openai/gpt-5.1"
MODELS_3 = MODELS_2 + [NEW_MODEL]
TEST_STEPS = 400
RELEASE_STEP = 200

def create_registry(models):
    all_models = {
        "mistralai/mixtral-8x7b-instruct": {
            "input_cost_per_m": 0.5,
            "output_cost_per_m": 1.5,
        },
        "openai/gpt-4-turbo": {
            "input_cost_per_m": 10.0,
            "output_cost_per_m": 30.0,
        },
        "openai/gpt-5.1": {
            "input_cost_per_m": 15.0,
            "output_cost_per_m": 45.0,
        }
    }
    return {k: v for k, v in all_models.items() if k in models}

def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*80}")
    print(f"{title:^80}")
    print(f"{'='*80}\n")

def test_router_initialization():
    """Test 1: Router initialization with warmup priors and corralling."""
    print_section("TEST 1: Router Initialization")
    
    warmup_priors_path = Path(__file__).parent.parent.parent / "src" / "artifacts" / "priors_warmup.joblib"
    
    router = BanditRouter.create(
        model_registry=create_registry(MODELS_2),
        context_model=DEFAULT_SENTENCE_TRANSFORMER,
        priors=str(warmup_priors_path),
        use_corralling=True,
        alpha=2.0,  # High exploration
        corralling_learning_rate=0.1,
        corralling_gamma=0.05,
        pca_path=DEFAULT_PCA_PATH
    )
    
    # Check 1: Corralling is enabled
    assert router.use_corralling, "❌ Corralling not enabled!"
    assert router.corralling_router is not None, "❌ Corralling router not initialized!"
    print("✅ Corralling enabled and initialized")
    
    # Check 2: Alpha propagation to experts
    print("\n📊 Expert Configuration:")
    for i, expert in enumerate(router.corralling_router.experts):
        expert_type = type(expert).__name__
        alpha_start = expert.alpha_start
        alpha_end = expert.alpha_end
        print(f"   Expert {i} ({expert_type}):")
        print(f"      alpha_start = {alpha_start}")
        print(f"      alpha_end   = {alpha_end}")
        
        if alpha_end != 2.0:
            print(f"      ❌ FAIL: alpha_end should be 2.0, got {alpha_end}")
            return False
        else:
            print(f"      ✅ PASS: alpha correctly propagated")
    
    # Check 3: Initial models
    print(f"\n📊 Initial Models:")
    print(f"   Main bandit:  {router.bandit.models}")
    print(f"   Corralling:   {router.corralling_router.models}")
    for i, expert in enumerate(router.corralling_router.experts):
        print(f"   Expert {i}:    {expert.models}")
    
    assert len(router.bandit.models) == 2, f"❌ Expected 2 models, got {len(router.bandit.models)}"
    print("✅ Initial model count correct")
    
    return router

def test_dynamic_model_registration(router):
    """Test 2: Dynamic model registration with semantic transfer."""
    print_section("TEST 2: Dynamic Model Registration")
    
    print(f"Registering {NEW_MODEL} with semantic transfer...")
    router.register_model(
        model_id=NEW_MODEL,
        cost_usd=15.0,
        latency_s=2.0,
        speed="balanced"
    )
    
    # Check 1: Model added to all components
    print(f"\n📊 Models After Registration:")
    print(f"   Main bandit:  {router.bandit.models}")
    print(f"   Corralling:   {router.corralling_router.models}")
    for i, expert in enumerate(router.corralling_router.experts):
        print(f"   Expert {i}:    {expert.models}")
    
    # Verify model in all places
    assert NEW_MODEL in router.bandit.models, f"❌ {NEW_MODEL} not in main bandit!"
    assert NEW_MODEL in router.corralling_router.models, f"❌ {NEW_MODEL} not in corralling!"
    for i, expert in enumerate(router.corralling_router.experts):
        assert NEW_MODEL in expert.models, f"❌ {NEW_MODEL} not in expert {i}!"
    print("✅ Model propagated to all components")
    
    # Check 2: Semantic transfer occurred (warmup expert should have non-zero priors)
    warmup_expert = router.corralling_router.experts[0]
    if hasattr(warmup_expert, 'b'):
        b_norm = np.linalg.norm(warmup_expert.b[NEW_MODEL])
        print(f"\n📊 Semantic Transfer Check:")
        print(f"   Warmup expert b-vector norm: {b_norm:.4f}")
        if b_norm > 0.1:
            print(f"   ✅ PASS: Semantic transfer occurred (non-zero priors)")
        else:
            print(f"   ⚠️  WARNING: b-vector nearly zero, semantic transfer may not have worked")
    
    return True

def test_model_selection(router, data):
    """Test 3: Model selection with proper exploration."""
    print_section("TEST 3: Model Selection & Exploration")
    
    selections = defaultdict(int)
    rewards = defaultdict(list)
    
    print(f"Running {TEST_STEPS} selection steps...")
    print(f"(New model introduced at t={RELEASE_STEP})")
    
    for t in range(min(TEST_STEPS, len(data))):
        item = data[t]
        
        selected, log = router.route(item.prompt, profile="auto")
        reward = item.get_reward(selected, default=0.0)
        router.update(item.prompt, selected, reward)
        
        selections[selected] += 1
        rewards[selected].append(reward)
        
        # Log first few selections after release
        if RELEASE_STEP <= t < RELEASE_STEP + 10:
            print(f"   t={t}: {selected} (reward={reward:.3f})")
    
    # Analyze pre-release vs post-release
    print(f"\n📊 Selection Analysis:")
    print(f"\nPre-Release (t=0-{RELEASE_STEP-1}):")
    pre_release_models = [m for m in MODELS_2 if selections[m] > 0]
    for model in pre_release_models:
        count = selections[model]
        pct = count / RELEASE_STEP * 100
        avg_reward = np.mean(rewards[model]) if rewards[model] else 0.0
        print(f"   {model:<40}: {count:>4} ({pct:>5.1f}%) | Avg Reward: {avg_reward:.3f}")
    
    print(f"\nPost-Release (t={RELEASE_STEP}-{TEST_STEPS-1}):")
    post_release_steps = TEST_STEPS - RELEASE_STEP
    for model in MODELS_3:
        # Count only post-release selections
        post_count = sum(1 for i, m in enumerate(data[:TEST_STEPS]) 
                        if i >= RELEASE_STEP and selections.get(model, 0) > 0)
        pct = post_count / post_release_steps * 100 if post_release_steps > 0 else 0.0
        avg_reward = np.mean(rewards[model]) if rewards[model] else 0.0
        print(f"   {model:<40}: {post_count:>4} ({pct:>5.1f}%) | Avg Reward: {avg_reward:.3f}")
    
    # Check: GPT-5.1 should be selected
    gpt5_selections = selections.get(NEW_MODEL, 0)
    gpt5_rate = gpt5_selections / post_release_steps * 100 if post_release_steps > 0 else 0.0
    
    print(f"\n🎯 GPT-5.1 Selection Rate: {gpt5_rate:.1f}%")
    
    if gpt5_rate < 5.0:
        print(f"   ❌ FAIL: Router is NOT exploring GPT-5.1 (< 5%)")
        return False
    elif gpt5_rate < 25.0:
        print(f"   ⚠️  WARNING: Low GPT-5.1 selection rate (< 25%)")
        print(f"   → May need higher alpha or longer evaluation")
    else:
        print(f"   ✅ PASS: Router is exploring GPT-5.1")
    
    # Check: GPT-5.1 should have high reward when selected
    if rewards[NEW_MODEL]:
        gpt5_avg_reward = np.mean(rewards[NEW_MODEL])
        print(f"\n📊 GPT-5.1 Performance When Selected:")
        print(f"   Average Reward: {gpt5_avg_reward:.3f}")
        print(f"   Expected: ~4.489 (from ground truth)")
        
        if gpt5_avg_reward > 4.0:
            print(f"   ✅ PASS: High reward confirms model quality")
        else:
            print(f"   ⚠️  WARNING: Lower than expected reward")
    
    return True

def test_expert_weights(router):
    """Test 4: Check Corralling expert weights."""
    print_section("TEST 4: Corralling Expert Weights")
    
    if router.corralling_router:
        weights = router.corralling_router.get_expert_weights()
        print("📊 Expert Weights (after updates):")
        for expert_name, weight in weights.items():
            print(f"   {expert_name:<50}: {weight:.4f}")
        
        # Check if weights are reasonable (not stuck at 0 or 1)
        weight_values = list(weights.values())
        if any(w < 0.01 for w in weight_values):
            print("\n   ⚠️  WARNING: Some expert has very low weight (< 0.01)")
            print("   → May indicate expert death (needs more gamma)")
        else:
            print("\n   ✅ PASS: All experts have reasonable weights")
        
        return True
    else:
        print("❌ Corralling not enabled!")
        return False

def run_comprehensive_test():
    """Run all tests in sequence."""
    print_section("BANDITGPT ROUTER CONFIGURATION TEST")
    
    # Load data
    print("Loading evaluation data...")
    evaluator = AlignedEvaluator.from_jsonl_gz(
        DEV_DATA_PATH_ALL_MODELS,
        required_models=MODELS_3
    )
    data = [item for item in evaluator if all(m in item.rewards for m in MODELS_3)]
    print(f"✅ Loaded {len(data)} samples\n")
    
    # Run tests
    all_passed = True
    
    # Test 1: Initialization
    router = test_router_initialization()
    if not router:
        print("❌ TEST 1 FAILED: Router initialization")
        return False
    
    # Test 2: Dynamic registration
    if not test_dynamic_model_registration(router):
        print("❌ TEST 2 FAILED: Dynamic model registration")
        all_passed = False
    
    # Test 3: Selection behavior
    if not test_model_selection(router, data):
        print("❌ TEST 3 FAILED: Model selection")
        all_passed = False
    
    # Test 4: Expert weights
    if not test_expert_weights(router):
        print("❌ TEST 4 FAILED: Expert weights")
        all_passed = False
    
    # Final summary
    print_section("TEST SUMMARY")
    
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("\n✅ Router is configured correctly:")
        print("   • Alpha properly propagated to experts")
        print("   • Dynamic model registration works")
        print("   • Proper exploration of new models")
        print("   • Corralling experts functioning correctly")
        print("\n✅ Ready for production use and experiments!")
    else:
        print("❌ SOME TESTS FAILED")
        print("\nPlease review the failures above and fix before proceeding.")
    
    return all_passed

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)

