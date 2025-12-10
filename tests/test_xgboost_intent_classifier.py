"""
Unit tests for XGBoost Intent Classifier.

Tests cover:
- Feature extraction
- Classification with trained model
- Confidence threshold for GENERAL fallback
- Model save/load
- Batch classification
"""

import pytest
import numpy as np
import json
import tempfile
from pathlib import Path
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_jury.routing.xgboost_intent_classifier import (
    XGBoostIntentClassifier,
    FeatureExtractor,
    XGBoostIntentResult,
    XGBOOST_AVAILABLE,
)
from llm_jury.routing.intent_classifier import IntentCategory


# Skip all tests if XGBoost not available
pytestmark = pytest.mark.skipif(
    not XGBOOST_AVAILABLE,
    reason="XGBoost not installed"
)


class TestFeatureExtractor:
    """Tests for the FeatureExtractor class."""
    
    @pytest.fixture
    def extractor(self):
        return FeatureExtractor()
    
    def test_extract_features_returns_dict(self, extractor):
        """Feature extraction should return a dictionary."""
        features = extractor.extract_features("Write a Python function")
        assert isinstance(features, dict)
        assert len(features) > 0
    
    def test_extract_features_coding_signals(self, extractor):
        """Coding prompts should have coding-related features."""
        features = extractor.extract_features("Write a Python function to sort a list")
        
        assert features['has_programming_langs'] == 1.0
        assert features['coding_word_count'] > 0
    
    def test_extract_features_math_signals(self, extractor):
        """Math prompts should have math-related features."""
        features = extractor.extract_features("Calculate the integral of x^2 dx")
        
        assert features['has_math_words'] == 1.0 or features['math_word_count'] > 0
    
    def test_extract_features_question_signals(self, extractor):
        """Question prompts should have question-related features."""
        features = extractor.extract_features("What is the capital of France?")
        
        assert features['starts_with_question'] == 1.0
        assert features['has_question_mark'] == 1.0
    
    def test_extract_features_code_block(self, extractor):
        """Code blocks should be detected."""
        prompt = """Fix this code:
```python
def foo():
    return bar
```"""
        features = extractor.extract_features(prompt)
        assert features['has_code_block'] == 1.0
    
    def test_extract_features_length_features(self, extractor):
        """Length features should be calculated correctly."""
        prompt = "Hello world"
        features = extractor.extract_features(prompt)
        
        assert features['length_chars'] == len(prompt)
        assert features['length_words'] == 2
    
    def test_extract_batch(self, extractor):
        """Batch extraction should return numpy array."""
        prompts = [
            "Write Python code",
            "What is AI?",
            "Calculate 2+2",
        ]
        X, feature_names = extractor.extract_batch(prompts)
        
        assert isinstance(X, np.ndarray)
        assert X.shape[0] == 3  # 3 prompts
        assert len(feature_names) > 0
        assert X.shape[1] == len(feature_names)


class TestXGBoostIntentClassifier:
    """Tests for the XGBoostIntentClassifier class."""
    
    @pytest.fixture
    def trained_classifier(self):
        """Load the trained classifier if available."""
        model_path = Path(__file__).parent.parent / "models" / "xgboost_intent_classifier.json"
        if model_path.exists():
            return XGBoostIntentClassifier(model_path=str(model_path))
        pytest.skip("Trained model not found")
    
    @pytest.fixture
    def simple_classifier(self):
        """Create a simple trained classifier for testing."""
        classifier = XGBoostIntentClassifier()
        
        # Simple training data
        prompts = [
            "Write a Python function to sort",
            "Create JavaScript code for API",
            "Implement a class in Java",
            "Debug this Python error",
            "What is machine learning?",
            "Explain how neural networks work",
            "Who invented the telephone?",
            "Define artificial intelligence",
            "Solve this equation: 2x + 5 = 15",
            "Calculate the derivative of x^3",
            "Prove this theorem",
            "Analyze this statistical data",
            "Create a workflow to process data",
            "Plan a multi-step automation",
            "Execute this pipeline",
            "Use the API to fetch weather",
        ]
        labels = [
            'coding', 'coding', 'coding', 'coding',
            'factual_qa', 'factual_qa', 'factual_qa', 'factual_qa',
            'reasoning', 'reasoning', 'reasoning', 'reasoning',
            'agentic_execution', 'agentic_execution', 'agentic_execution', 'agentic_execution',
        ]
        
        # Extract features
        extractor = FeatureExtractor()
        X, feature_names = extractor.extract_batch(prompts)
        
        # Encode labels
        label_to_idx = {'reasoning': 0, 'coding': 1, 'factual_qa': 2, 'agentic_execution': 3}
        y = np.array([label_to_idx[l] for l in labels])
        
        # Train
        classifier.train(X, y, feature_names=feature_names, n_estimators=50)
        
        return classifier
    
    def test_classifier_init(self):
        """Classifier should initialize without errors."""
        classifier = XGBoostIntentClassifier()
        assert classifier.model is None
        assert classifier.confidence_threshold == 0.5
    
    def test_classifier_has_4_classes(self):
        """Classifier should have 4 classes (not 5)."""
        classifier = XGBoostIntentClassifier()
        assert len(classifier.label_encoder) == 4
        assert 'general' not in classifier.label_encoder
        assert 'reasoning' in classifier.label_encoder
        assert 'coding' in classifier.label_encoder
        assert 'factual_qa' in classifier.label_encoder
        assert 'agentic_execution' in classifier.label_encoder
    
    def test_classify_coding_prompt(self, trained_classifier):
        """Coding prompts should be classified as CODING."""
        result = trained_classifier.classify("Write a Python function to sort a list")
        
        assert isinstance(result, XGBoostIntentResult)
        assert result.category == IntentCategory.CODING
        assert result.confidence > 0.5
    
    def test_classify_reasoning_prompt(self, trained_classifier):
        """Reasoning prompts should be classified as REASONING."""
        result = trained_classifier.classify("Solve the equation 3x + 7 = 22")
        
        assert isinstance(result, XGBoostIntentResult)
        assert result.category == IntentCategory.REASONING, \
            f"Expected REASONING, got {result.category}. Probs: {result.probabilities}"
    
    def test_classify_factual_qa_prompt(self, trained_classifier):
        """Factual QA prompts should be classified as FACTUAL_QA."""
        result = trained_classifier.classify("What is the capital of Japan?")
        
        assert isinstance(result, XGBoostIntentResult)
        assert result.category == IntentCategory.FACTUAL_QA
    
    def test_classify_agentic_prompt(self, trained_classifier):
        """Agentic prompts should be classified as AGENTIC_EXECUTION."""
        result = trained_classifier.classify("Book a flight from NYC to London and schedule a meeting for next Tuesday")
        
        assert isinstance(result, XGBoostIntentResult)
        assert result.category == IntentCategory.AGENTIC_EXECUTION, \
            f"Expected AGENTIC_EXECUTION, got {result.category}. Probs: {result.probabilities}"
    
    def test_low_confidence_returns_general(self, simple_classifier):
        """Low confidence predictions should return GENERAL."""
        # An ambiguous prompt that doesn't clearly fit any category
        result = simple_classifier.classify("hello")
        
        assert isinstance(result, XGBoostIntentResult)
        # If confidence < 0.5, should be GENERAL
        if result.confidence < 0.5:
            assert result.category == IntentCategory.GENERAL
    
    def test_classify_returns_probabilities(self, trained_classifier):
        """Classification should return probability distribution."""
        result = trained_classifier.classify("Write Python code")
        
        assert 'reasoning' in result.probabilities
        assert 'coding' in result.probabilities
        assert 'factual_qa' in result.probabilities
        assert 'agentic_execution' in result.probabilities
        
        # Probabilities should be valid
        for prob in result.probabilities.values():
            assert 0.0 <= prob <= 1.0
    
    def test_classify_returns_latency(self, trained_classifier):
        """Classification should return latency measurement."""
        result = trained_classifier.classify("Test prompt")
        
        assert result.latency_ms > 0
        assert result.latency_ms < 1000  # Should be fast (<1 second)
    
    def test_classify_batch(self, trained_classifier):
        """Batch classification should work."""
        prompts = [
            "Write Python code",
            "What is AI?",
            "Solve 2+2",
        ]
        results = trained_classifier.classify_batch(prompts)
        
        assert len(results) == 3
        assert all(isinstance(r, XGBoostIntentResult) for r in results)
    
    def test_result_to_dict(self, trained_classifier):
        """Result should be serializable to dict."""
        result = trained_classifier.classify("Test prompt")
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert 'category' in result_dict
        assert 'confidence' in result_dict
        assert 'probabilities' in result_dict
        assert 'latency_ms' in result_dict


class TestModelPersistence:
    """Tests for model save/load functionality."""
    
    def test_save_and_load_model(self):
        """Model should be saveable and loadable."""
        # Create and train a simple model
        classifier = XGBoostIntentClassifier()
        
        prompts = ["code python", "what is", "solve math", "automate workflow"] * 4
        labels = ['coding', 'factual_qa', 'reasoning', 'agentic_execution'] * 4
        
        extractor = FeatureExtractor()
        X, feature_names = extractor.extract_batch(prompts)
        
        label_to_idx = {'reasoning': 0, 'coding': 1, 'factual_qa': 2, 'agentic_execution': 3}
        y = np.array([label_to_idx[l] for l in labels])
        
        classifier.train(X, y, feature_names=feature_names, n_estimators=10)
        
        # Save to temp file
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model.json"
            classifier.save(str(model_path))
            
            # Check files exist
            assert model_path.exists()
            assert model_path.with_suffix('.meta.json').exists()
            
            # Load and verify
            loaded = XGBoostIntentClassifier(model_path=str(model_path))
            assert loaded.model is not None
            assert loaded.feature_names == classifier.feature_names
            
            # Predictions should match
            result1 = classifier.classify("Write Python code")
            result2 = loaded.classify("Write Python code")
            assert result1.category == result2.category


class TestConfidenceThreshold:
    """Tests for confidence threshold behavior."""
    
    def test_default_threshold_is_0_5(self):
        """Default confidence threshold should be 0.5."""
        classifier = XGBoostIntentClassifier()
        assert classifier.confidence_threshold == 0.5
    
    def test_general_probability_in_output(self):
        """GENERAL probability should be in output when below threshold."""
        model_path = Path(__file__).parent.parent / "models" / "xgboost_intent_classifier.json"
        if not model_path.exists():
            pytest.skip("Trained model not found")
        
        classifier = XGBoostIntentClassifier(model_path=str(model_path))
        result = classifier.classify("hmm")
        
        # 'general' should be in probabilities dict
        assert 'general' in result.probabilities


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_prompt(self):
        """Empty prompt should not crash."""
        model_path = Path(__file__).parent.parent / "models" / "xgboost_intent_classifier.json"
        if not model_path.exists():
            pytest.skip("Trained model not found")
        
        classifier = XGBoostIntentClassifier(model_path=str(model_path))
        result = classifier.classify("")
        
        assert isinstance(result, XGBoostIntentResult)
    
    def test_very_long_prompt(self):
        """Very long prompt should not crash."""
        model_path = Path(__file__).parent.parent / "models" / "xgboost_intent_classifier.json"
        if not model_path.exists():
            pytest.skip("Trained model not found")
        
        classifier = XGBoostIntentClassifier(model_path=str(model_path))
        long_prompt = "Write code " * 1000
        result = classifier.classify(long_prompt)
        
        assert isinstance(result, XGBoostIntentResult)
    
    def test_special_characters(self):
        """Special characters should not crash."""
        model_path = Path(__file__).parent.parent / "models" / "xgboost_intent_classifier.json"
        if not model_path.exists():
            pytest.skip("Trained model not found")
        
        classifier = XGBoostIntentClassifier(model_path=str(model_path))
        result = classifier.classify("∫∑∏∂∇√ λ→∞")
        
        assert isinstance(result, XGBoostIntentResult)
    
    def test_unicode_prompt(self):
        """Unicode characters should not crash."""
        model_path = Path(__file__).parent.parent / "models" / "xgboost_intent_classifier.json"
        if not model_path.exists():
            pytest.skip("Trained model not found")
        
        classifier = XGBoostIntentClassifier(model_path=str(model_path))
        result = classifier.classify("日本語のテスト 🎉 émojis")
        
        assert isinstance(result, XGBoostIntentResult)
    
    def test_predict_without_training_raises(self):
        """Predicting without training should raise error."""
        classifier = XGBoostIntentClassifier()
        
        with pytest.raises(ValueError, match="Model not trained"):
            classifier.predict(np.array([[1, 2, 3]]))


# Quality tests - these should pass for a well-trained model
class TestClassificationQuality:
    """
    Quality tests for model accuracy.
    
    These tests verify that clear, unambiguous prompts are classified correctly.
    Failing tests indicate the model needs improvement (more/better training data).
    """
    
    @pytest.fixture
    def classifier(self):
        model_path = Path(__file__).parent.parent / "models" / "xgboost_intent_classifier.json"
        if not model_path.exists():
            pytest.skip("Trained model not found")
        return XGBoostIntentClassifier(model_path=str(model_path))
    
    def test_coding_prompts(self, classifier):
        """Clear coding prompts should be classified as CODING."""
        test_cases = [
            ("Write a Python function to reverse a string", IntentCategory.CODING),
            ("Implement a binary search algorithm in Java", IntentCategory.CODING),
            ("Debug this Python code that throws IndexError", IntentCategory.CODING),
            ("Create a REST API endpoint using Flask", IntentCategory.CODING),
            ("Refactor this JavaScript class to use async/await", IntentCategory.CODING),
        ]
        
        failures = []
        for prompt, expected in test_cases:
            result = classifier.classify(prompt)
            if result.category != expected:
                failures.append(f"  '{prompt}' -> {result.category.value} (expected {expected.value})")
        
        assert not failures, f"CODING classification failures:\n" + "\n".join(failures)
    
    def test_reasoning_prompts(self, classifier):
        """Clear reasoning prompts should be classified as REASONING."""
        test_cases = [
            ("Solve the equation 2x + 5 = 15", IntentCategory.REASONING),
            ("Calculate the derivative of sin(x) * cos(x)", IntentCategory.REASONING),
            ("Prove that the sum of angles in a triangle equals 180 degrees", IntentCategory.REASONING),
            ("What is the probability of rolling a 7 with two dice?", IntentCategory.REASONING),
            ("Analyze this data and compute the standard deviation", IntentCategory.REASONING),
        ]
        
        failures = []
        for prompt, expected in test_cases:
            result = classifier.classify(prompt)
            if result.category != expected:
                failures.append(f"  '{prompt}' -> {result.category.value} (expected {expected.value})")
        
        assert not failures, f"REASONING classification failures:\n" + "\n".join(failures)
    
    def test_factual_qa_prompts(self, classifier):
        """Clear factual QA prompts should be classified as FACTUAL_QA."""
        test_cases = [
            ("What is the capital of France?", IntentCategory.FACTUAL_QA),
            ("Who invented the telephone?", IntentCategory.FACTUAL_QA),
            ("Explain how photosynthesis works", IntentCategory.FACTUAL_QA),
            ("Define machine learning", IntentCategory.FACTUAL_QA),
            ("What are the symptoms of diabetes?", IntentCategory.FACTUAL_QA),
        ]
        
        failures = []
        for prompt, expected in test_cases:
            result = classifier.classify(prompt)
            if result.category != expected:
                failures.append(f"  '{prompt}' -> {result.category.value} (expected {expected.value})")
        
        assert not failures, f"FACTUAL_QA classification failures:\n" + "\n".join(failures)
    
    def test_agentic_prompts(self, classifier):
        """Clear agentic prompts should be classified as AGENTIC_EXECUTION."""
        test_cases = [
            ("Book a flight from NYC to London for next Tuesday", IntentCategory.AGENTIC_EXECUTION),
            ("Search the web for recent news about AI", IntentCategory.AGENTIC_EXECUTION),
            ("Send an email to John with the meeting notes", IntentCategory.AGENTIC_EXECUTION),
            ("Create a workflow to backup files daily", IntentCategory.AGENTIC_EXECUTION),
            ("Schedule a reminder for tomorrow at 9am", IntentCategory.AGENTIC_EXECUTION),
        ]
        
        failures = []
        for prompt, expected in test_cases:
            result = classifier.classify(prompt)
            if result.category != expected:
                failures.append(f"  '{prompt}' -> {result.category.value} (expected {expected.value})")
        
        assert not failures, f"AGENTIC_EXECUTION classification failures:\n" + "\n".join(failures)
    
    def test_overall_accuracy(self, classifier):
        """Overall accuracy should be above 70% on clear test cases."""
        test_cases = [
            # CODING
            ("Write a Python script to parse JSON", IntentCategory.CODING),
            ("Fix this bug in my JavaScript code", IntentCategory.CODING),
            # REASONING  
            ("Solve for x: 3x - 7 = 14", IntentCategory.REASONING),
            ("Calculate compound interest", IntentCategory.REASONING),
            # FACTUAL_QA
            ("What is the speed of light?", IntentCategory.FACTUAL_QA),
            ("Explain quantum entanglement", IntentCategory.FACTUAL_QA),
            # AGENTIC
            ("Book a restaurant reservation", IntentCategory.AGENTIC_EXECUTION),
            ("Find and download the latest report", IntentCategory.AGENTIC_EXECUTION),
        ]
        
        correct = 0
        for prompt, expected in test_cases:
            result = classifier.classify(prompt)
            if result.category == expected:
                correct += 1
        
        accuracy = correct / len(test_cases)
        assert accuracy >= 0.70, f"Overall accuracy {accuracy:.1%} is below 70% threshold"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

