#!/usr/bin/env python3
"""
Unit tests for data_loader gzip decompression functionality.

Tests ensure consistent loading of both compressed (.gz) and uncompressed (.jsonl) files.
"""

import gzip
import json
import tempfile
import unittest
from pathlib import Path
import sys

# Add experiments to path
sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))

from utils.data_loader import load_oracle_rewards


class TestDataLoaderGzip(unittest.TestCase):
    """Test gzip decompression in data_loader."""
    
    def setUp(self):
        """Create temporary test data."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Sample test data (minimal valid entries)
        self.test_data = [
            {
                "model_id": "test/model-a",
                "prompt": "What is 2+2?",
                "response": "4",
                "ok": True,
                "raw_score": 0.95
            },
            {
                "model_id": "test/model-b",
                "prompt": "What is 2+2?",
                "response": "Four",
                "ok": True,
                "raw_score": 0.85
            },
            {
                "model_id": "test/model-a",
                "prompt": "What is Python?",
                "response": "A programming language",
                "ok": True,
                "raw_score": 0.90
            }
        ]
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_load_uncompressed_jsonl(self):
        """Test loading from standard uncompressed .jsonl file."""
        # Create uncompressed file
        test_file = self.temp_path / "test_data.jsonl"
        with open(test_file, 'w') as f:
            for entry in self.test_data:
                f.write(json.dumps(entry) + '\n')
        
        # Load using data_loader (with mocked DATA_DIR)
        import utils.data_loader as dl
        original_data_dir = dl.DATA_DIR
        try:
            # Mock DATA_DIR to point to temp directory
            dl.DATA_DIR = self.temp_path
            
            rewards = load_oracle_rewards("test_data.jsonl")
            
            # Validate structure
            self.assertEqual(len(rewards), 2)  # 2 unique prompts
            self.assertIn("What is 2+2?", rewards)
            self.assertIn("What is Python?", rewards)
            
            # Validate rewards
            self.assertAlmostEqual(rewards["What is 2+2?"]["test/model-a"], 0.95)
            self.assertAlmostEqual(rewards["What is 2+2?"]["test/model-b"], 0.85)
            self.assertAlmostEqual(rewards["What is Python?"]["test/model-a"], 0.90)
            
        finally:
            dl.DATA_DIR = original_data_dir
    
    def test_load_compressed_gzip(self):
        """Test loading from compressed .jsonl.gz file."""
        # Create compressed file
        test_file = self.temp_path / "test_data.jsonl.gz"
        with gzip.open(test_file, 'wt') as f:
            for entry in self.test_data:
                f.write(json.dumps(entry) + '\n')
        
        # Load using data_loader
        import utils.data_loader as dl
        original_data_dir = dl.DATA_DIR
        try:
            dl.DATA_DIR = self.temp_path
            
            rewards = load_oracle_rewards("test_data.jsonl")  # Note: no .gz extension
            
            # Should auto-detect and load .gz version
            self.assertEqual(len(rewards), 2)
            self.assertAlmostEqual(rewards["What is 2+2?"]["test/model-a"], 0.95)
            
        finally:
            dl.DATA_DIR = original_data_dir
    
    def test_prefer_uncompressed_when_both_exist(self):
        """Test that uncompressed file is loaded when both .jsonl and .jsonl.gz exist."""
        # Create both versions with DIFFERENT data to verify which is loaded
        uncompressed_data = [self.test_data[0]]  # Only first entry
        compressed_data = self.test_data  # All entries
        
        rewards_dir = self.temp_path / "rewards"
        rewards_dir.mkdir(exist_ok=True)

        # Uncompressed
        test_file_plain = rewards_dir / "test_data.jsonl"
        with open(test_file_plain, 'w') as f:
            for entry in uncompressed_data:
                f.write(json.dumps(entry) + '\n')

        # Compressed
        test_file_gz = rewards_dir / "test_data.jsonl.gz"
        with gzip.open(test_file_gz, 'wt') as f:
            for entry in compressed_data:
                f.write(json.dumps(entry) + '\n')

        # Load - should prefer uncompressed when both exist
        import utils.data_loader as dl
        original_rewards_dir = dl.OFFLINE_DATASET_DIR
        try:
            dl.OFFLINE_DATASET_DIR = rewards_dir
            
            rewards = load_oracle_rewards("test_data.jsonl")
            
            # Should have loaded uncompressed version (1 entry → 1 prompt)
            self.assertEqual(len(rewards), 1)
            self.assertIn("What is 2+2?", rewards)
            self.assertNotIn("What is Python?", rewards)  # Only in compressed
            
        finally:
            dl.OFFLINE_DATASET_DIR = original_rewards_dir
    
    def test_load_with_gz_extension_explicit(self):
        """Test loading when .gz extension is explicitly provided."""
        # Create compressed file
        test_file = self.temp_path / "test_data.jsonl.gz"
        with gzip.open(test_file, 'wt') as f:
            for entry in self.test_data:
                f.write(json.dumps(entry) + '\n')
        
        import utils.data_loader as dl
        original_data_dir = dl.DATA_DIR
        try:
            dl.DATA_DIR = self.temp_path
            
            # Explicitly pass .gz extension
            rewards = load_oracle_rewards("test_data.jsonl.gz")
            
            self.assertEqual(len(rewards), 2)
            self.assertAlmostEqual(rewards["What is 2+2?"]["test/model-a"], 0.95)
            
        finally:
            dl.DATA_DIR = original_data_dir
    
    def test_filter_failed_responses(self):
        """Test that entries with ok=False are filtered out."""
        test_data_with_failures = self.test_data + [
            {
                "model_id": "test/model-c",
                "prompt": "What is 2+2?",
                "response": "Error",
                "ok": False,  # Should be filtered
                "raw_score": 0.0
            }
        ]
        
        # Create compressed file
        test_file = self.temp_path / "test_data.jsonl.gz"
        with gzip.open(test_file, 'wt') as f:
            for entry in test_data_with_failures:
                f.write(json.dumps(entry) + '\n')
        
        import utils.data_loader as dl
        original_data_dir = dl.DATA_DIR
        try:
            dl.DATA_DIR = self.temp_path
            
            rewards = load_oracle_rewards("test_data.jsonl")
            
            # Should only have 2 models for "What is 2+2?" (not 3)
            self.assertEqual(len(rewards["What is 2+2?"]), 2)
            self.assertNotIn("test/model-c", rewards["What is 2+2?"])
            
        finally:
            dl.DATA_DIR = original_data_dir
    
    def test_rewards_directory(self):
        """Test that OFFLINE_DATASET_DIR (rewards directory) is checked first."""
        rewards_dir = self.temp_path / "rewards"
        rewards_dir.mkdir()

        test_file = rewards_dir / "test_data.jsonl.gz"
        with gzip.open(test_file, 'wt') as f:
            for entry in self.test_data:
                f.write(json.dumps(entry) + '\n')

        import utils.data_loader as dl
        original_rewards_dir = dl.OFFLINE_DATASET_DIR
        try:
            dl.OFFLINE_DATASET_DIR = rewards_dir

            rewards = load_oracle_rewards("test_data.jsonl")

            self.assertEqual(len(rewards), 2)

        finally:
            dl.OFFLINE_DATASET_DIR = original_rewards_dir
    
    def test_large_file_streaming(self):
        """Test that large files decompress efficiently without loading everything into memory."""
        # Create a moderately large compressed file (1000 entries)
        large_data = []
        for i in range(1000):
            large_data.append({
                "model_id": f"test/model-{i % 10}",
                "prompt": f"Test prompt {i}",
                "response": f"Response {i}",
                "ok": True,
                "raw_score": 0.5 + (i % 10) * 0.05
            })
        
        test_file = self.temp_path / "large_data.jsonl.gz"
        with gzip.open(test_file, 'wt') as f:
            for entry in large_data:
                f.write(json.dumps(entry) + '\n')
        
        import utils.data_loader as dl
        original_data_dir = dl.DATA_DIR
        try:
            dl.DATA_DIR = self.temp_path
            
            # Should load without memory issues
            rewards = load_oracle_rewards("large_data.jsonl")
            
            # Validate count (1000 unique prompts)
            self.assertEqual(len(rewards), 1000)
            
        finally:
            dl.DATA_DIR = original_data_dir


if __name__ == "__main__":
    unittest.main()
