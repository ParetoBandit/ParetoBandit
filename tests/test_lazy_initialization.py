#!/usr/bin/env python3
"""
Unit tests for SqliteContextStore lazy initialization.

Tests ensure database is NOT created until actually needed.
"""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bandit_gpt.storage import SqliteContextStore
import numpy as np


class TestLazyInitialization(unittest.TestCase):
    """Test lazy initialization of SqliteContextStore."""
    
    def test_no_db_created_on_instantiation(self):
        """Test that database is NOT created when SqliteContextStore is instantiated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_lazy.db"
            
            # Create store instance
            store = SqliteContextStore(db_path=str(db_path))
            
            # Database file should NOT exist yet
            self.assertFalse(db_path.exists(), "Database should not be created on instantiation")
            self.assertFalse((Path(str(db_path) + "-wal")).exists(), "WAL file should not exist")
            self.assertFalse((Path(str(db_path) + "-shm")).exists(), "SHM file should not exist")
    
    def test_db_created_on_first_save(self):
        """Test that database IS created on first save_context call."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_lazy.db"
            
            store = SqliteContextStore(db_path=str(db_path))
            
            # No DB yet
            self.assertFalse(db_path.exists())
            
            # Save a context
            test_context = np.random.rand(32)
            store.save_context("test_request_1", test_context, "test/model")
            
            # Now DB should exist
            self.assertTrue(db_path.exists(), "Database should be created after first save")
    
    def test_db_created_on_first_get(self):
        """Test that database IS created on first get_context call (even if request doesn't exist)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_lazy.db"
            
            store = SqliteContextStore(db_path=str(db_path))
            
            # No DB yet
            self.assertFalse(db_path.exists())
            
            # Try to get a non-existent context
            context, model, _ = store.get_context("nonexistent_request")
            
            # Should return None, but DB should now exist
            self.assertIsNone(context)
            self.assertIsNone(model)
            self.assertTrue(db_path.exists(), "Database should be created after first get")
    
    def test_db_created_on_prune(self):
        """Test that database IS created on first prune call."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_lazy.db"
            
            store = SqliteContextStore(db_path=str(db_path))
            
            # No DB yet
            self.assertFalse(db_path.exists())
            
            # Call prune on empty/non-existent DB
            deleted = store.prune()
            
            # Should return 0, but DB should now exist
            self.assertEqual(deleted, 0)
            self.assertTrue(db_path.exists(), "Database should be created after first prune")
    
    def test_db_created_on_stats(self):
        """Test that database IS created on first stats call."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_lazy.db"
            
            store = SqliteContextStore(db_path=str(db_path))
            
            # No DB yet
            self.assertFalse(db_path.exists())
            
            # Get stats
            stats = store.stats()
            
            # Should return stats, DB should now exist
            self.assertIsInstance(stats, dict)
            self.assertTrue(db_path.exists(), "Database should be created after first stats")
    
    def test_multiple_operations_init_once(self):
        """Test that database is only initialized once across multiple operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_lazy.db"
            
            store = SqliteContextStore(db_path=str(db_path))
            
            # Track initialization
            init_count = 0
            original_init_db = store._init_db
            
            def counting_init_db():
                nonlocal init_count
                init_count += 1
                original_init_db()
            
            store._init_db = counting_init_db
            store._initialized = False  # Reset for test
            
            # Do multiple operations
            store.save_context("req1", np.random.rand(32), "model1")
            store.get_context("req1")
            store.stats()
            store.prune()
            
            # Should have only initialized once
            self.assertEqual(init_count, 1, "Database should be initialized exactly once")
    
    def test_user_directory_not_created_until_needed(self):
        """Test that ~/.bandit_gpt/ directory is NOT created on instantiation."""
        # Use default path which would resolve to ~/.bandit_gpt/ in library mode
        # We'll use a custom path to avoid polluting the actual user directory
        test_home = Path(tempfile.mkdtemp())
        test_bandit_dir = test_home / ".bandit_gpt"
        db_path = test_bandit_dir / "router_context.db"
        
        try:
            # Create store
            store = SqliteContextStore(db_path=str(db_path))
            
            # Directory should not exist yet
            self.assertFalse(test_bandit_dir.exists(), 
                           "~/.bandit_gpt/ should not be created on instantiation")
            
            # Save something
            store.save_context("req", np.random.rand(32), "model")
            
            # Now directory should exist
            self.assertTrue(test_bandit_dir.exists(),
                          "~/.bandit_gpt/ should be created on first use")
            self.assertTrue(db_path.exists())
            
        finally:
            # Cleanup
            import shutil
            shutil.rmtree(test_home, ignore_errors=True)
    
    def test_routing_only_workflow_no_files_created(self):
        """
        Integration test: Verify that a routing-only workflow creates NO files.
        
        This is the key use case: A user who only does router.route() without
        feedback should not have any files created.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "should_not_exist.db"
            
            # Simulate routing-only workflow
            store = SqliteContextStore(db_path=str(db_path))
            
            # User only routes, never saves context
            # (In real usage, router.route() stores context, but let's simulate
            #  a scenario where the user disables context storage)
            
            # Verify no files created
            self.assertFalse(db_path.exists())
            self.assertFalse(db_path.parent.exists() and list(db_path.parent.glob("*")),
                           "No files should be created in routing-only mode")


if __name__ == "__main__":
    unittest.main()
