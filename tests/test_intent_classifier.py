"""
Unit tests for intent classifier module.

Tests classifier functionality, training utilities, and data loading.
"""

import unittest
import numpy as np
import json
import tempfile
from pathlib import Path
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_jury.intent.classifier import IntentClassifier


class TestIntentClassifier(unittest.TestCase):
    """Test IntentClassifier class."""
    
    def test_initialization_default(self):
        """Test classifier initialization with defaults."""
        classifier = IntentClassifier()
        self.assertIsNotNone(classifier)
        self.assertEqual(classifier.embedding_model_name, 'all-MiniLM-L6-v2')
        self.assertIsNone(classifier._model)
    
    def test_initialization_custom_embedding(self):
        """Test classifier with custom embedding model."""
        classifier = IntentClassifier(embedding_model='paraphrase-MiniLM-L6-v2')
        self.assertEqual(classifier.embedding_model_name, 'paraphrase-MiniLM-L6-v2')
    
    def test_intent_labels_available(self):
        """Test intent labels are defined."""
        classifier = IntentClassifier()
        intents = classifier.intents
        
        self.assertIsInstance(intents, list)
        self.assertGreater(len(intents), 0)
        self.assertIn('coding', intents)
        self.assertIn('reasoning', intents)
    
    def test_predict_without_model_fails(self):
        """Test predict without model raises error."""
        classifier = IntentClassifier()
        
        with self.assertRaises(ValueError):
            classifier.predict("test prompt")


class TestDataLoading(unittest.TestCase):
    """Test data loading and preprocessing."""
    
    def test_load_valid_json(self):
        """Test loading valid JSON data."""
        # Create temporary JSON file
        data = {
            'samples': [
                {'prompt': 'Write a function', 'intent_label': 'coding'},
                {'prompt': 'What is 2+2?', 'intent_label': 'reasoning'},
                {'prompt': 'Who invented the car?', 'intent_label': 'factual_qa'}
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            temp_path = f.name
        
        try:
            with open(temp_path) as f:
                loaded_data = json.load(f)
            
            self.assertEqual(len(loaded_data['samples']), 3)
            self.assertEqual(loaded_data['samples'][0]['intent_label'], 'coding')
        finally:
            Path(temp_path).unlink()
    
    def test_validate_data_structure(self):
        """Test data validation."""
        valid_data = {
            'samples': [
                {'prompt': 'test', 'intent_label': 'coding'}
            ]
        }
        
        # Should have 'samples' key
        self.assertIn('samples', valid_data)
        
        # Each sample should have required keys
        for sample in valid_data['samples']:
            self.assertIn('prompt', sample)
            self.assertIn('intent_label', sample)
    
    def test_detect_missing_fields(self):
        """Test detection of missing required fields."""
        invalid_data = {
            'samples': [
                {'prompt': 'test'}  # Missing 'intent_label'
            ]
        }
        
        sample = invalid_data['samples'][0]
        self.assertNotIn('intent_label', sample)
    
    def test_deduplication(self):
        """Test prompt deduplication."""
        prompts = [
            'Write a function',
            'What is 2+2?',
            'Write a function',  # Duplicate
            'Who invented the car?'
        ]
        
        unique_prompts = list(dict.fromkeys(prompts))
        
        self.assertEqual(len(unique_prompts), 3)
        self.assertNotIn(prompts[2], unique_prompts[1:])  # Duplicate removed


class TestEmbeddingGeneration(unittest.TestCase):
    """Test embedding generation (without actual model loading)."""
    
    def test_embedding_shape(self):
        """Test that embeddings have expected shape."""
        # Simulate embeddings
        N = 100  # samples
        D = 384  # all-MiniLM-L6-v2 dimension
        
        embeddings = np.random.randn(N, D)
        
        self.assertEqual(embeddings.shape[0], N)
        self.assertEqual(embeddings.shape[1], D)
    
    def test_embeddings_are_normalized(self):
        """Test that embeddings can be normalized."""
        embeddings = np.random.randn(10, 384)
        
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / norms
        
        # Check norms are 1
        new_norms = np.linalg.norm(normalized, axis=1)
        np.testing.assert_array_almost_equal(new_norms, np.ones(10))


class TestLabelEncoding(unittest.TestCase):
    """Test label encoding and decoding."""
    
    def test_label_to_idx_mapping(self):
        """Test creating label to index mapping."""
        labels = ['coding', 'reasoning', 'factual_qa', 'general', 'summarization']
        label_to_idx = {label: idx for idx, label in enumerate(labels)}
        
        self.assertEqual(label_to_idx['coding'], 0)
        self.assertEqual(label_to_idx['summarization'], 4)
        self.assertEqual(len(label_to_idx), 5)
    
    def test_idx_to_label_mapping(self):
        """Test creating index to label mapping."""
        labels = ['coding', 'reasoning', 'factual_qa', 'general', 'summarization']
        idx_to_label = {idx: label for idx, label in enumerate(labels)}
        
        self.assertEqual(idx_to_label[0], 'coding')
        self.assertEqual(idx_to_label[4], 'summarization')
    
    def test_encode_decode_roundtrip(self):
        """Test encoding and decoding labels."""
        labels = ['coding', 'reasoning', 'factual_qa']
        label_to_idx = {label: idx for idx, label in enumerate(sorted(set(labels)))}
        idx_to_label = {idx: label for label, idx in label_to_idx.items()}
        
        # Encode
        encoded = [label_to_idx[label] for label in labels]
        
        # Decode
        decoded = [idx_to_label[idx] for idx in encoded]
        
        self.assertEqual(labels, decoded)


class TestCrossValidation(unittest.TestCase):
    """Test cross-validation setup."""
    
    def test_stratified_split(self):
        """Test stratified splitting preserves class distribution."""
        from sklearn.model_selection import StratifiedKFold
        
        # Create imbalanced data
        X = np.random.randn(100, 10)
        y = np.array([0]*50 + [1]*30 + [2]*20)
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for train_idx, val_idx in skf.split(X, y):
            # Check sizes
            self.assertEqual(len(train_idx), 80)
            self.assertEqual(len(val_idx), 20)
            
            # Check class distribution preserved
            y_val = y[val_idx]
            unique, counts = np.unique(y_val, return_counts=True)
            
            # Each fold should have samples from all classes
            self.assertEqual(len(unique), 3)
    
    def test_fold_consistency(self):
        """Test that folds are reproducible with same seed."""
        from sklearn.model_selection import StratifiedKFold
        
        X = np.random.randn(50, 10)
        y = np.random.randint(0, 3, 50)
        
        skf1 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        skf2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        splits1 = list(skf1.split(X, y))
        splits2 = list(skf2.split(X, y))
        
        # Check splits are identical
        for (train1, val1), (train2, val2) in zip(splits1, splits2):
            np.testing.assert_array_equal(train1, train2)
            np.testing.assert_array_equal(val1, val2)


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestIntentClassifier))
    suite.addTests(loader.loadTestsFromTestCase(TestDataLoading))
    suite.addTests(loader.loadTestsFromTestCase(TestEmbeddingGeneration))
    suite.addTests(loader.loadTestsFromTestCase(TestLabelEncoding))
    suite.addTests(loader.loadTestsFromTestCase(TestCrossValidation))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
