#!/usr/bin/env python3
"""
Unit Test: SQLite Context Persistence

Verifies that the SqliteContextStore correctly:
1. Writes context vectors to SQLite database
2. Reads context vectors back from database
3. Handles TTL expiration correctly
4. Prunes old entries

This validates the "Production RLHF" feedback mechanism where context vectors
are persisted for multi-request conversations.
"""

import sys
from pathlib import Path
import tempfile
import time
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from bandit_gpt.storage import SqliteContextStore, EphemeralContextStore


def test_sqlite_write_read():
    """Test basic write and read operations."""
    print("\n" + "=" * 70)
    print("TEST 1: SQLite Write/Read")
    print("=" * 70)
    
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    
    try:
        store = SqliteContextStore(db_path=db_path, ttl_seconds=3600)
        
        # Create test context
        request_id = "test_request_123"
        context = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        model_id = "gpt-4"
        
        # Write
        store.save_context(request_id, context, model_id)
        print(f"✓ Wrote context for request_id='{request_id}'")
        
        # Read back
        retrieved_context, retrieved_model, _ = store.get_context(request_id)
        
        # Verify
        assert retrieved_context is not None, "Context should not be None"
        assert retrieved_model == model_id, f"Model mismatch: {retrieved_model} != {model_id}"
        assert np.allclose(retrieved_context, context), "Context vectors don't match"
        
        print(f"✓ Read context back successfully")
        print(f"  Retrieved model: {retrieved_model}")
        print(f"  Context match: {np.allclose(retrieved_context, context)}")
        print("\n✅ PASS: SQLite Write/Read")
        
    finally:
        # Cleanup
        Path(db_path).unlink(missing_ok=True)


def test_sqlite_overwrite():
    """Test that re-saving the same request_id overwrites."""
    print("\n" + "=" * 70)
    print("TEST 2: SQLite Overwrite")
    print("=" * 70)
    
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    
    try:
        store = SqliteContextStore(db_path=db_path, ttl_seconds=3600)
        
        request_id = "duplicate_request"
        
        # First write
        context1 = np.array([1.0, 2.0, 3.0])
        store.save_context(request_id, context1, "model_a")
        
        # Second write (should overwrite)
        context2 = np.array([4.0, 5.0, 6.0])
        store.save_context(request_id, context2, "model_b")
        
        # Verify only the latest exists
        retrieved_context, retrieved_model, _ = store.get_context(request_id)
        
        assert retrieved_model == "model_b", "Should have latest model"
        assert np.allclose(retrieved_context, context2), "Should have latest context"
        print(f"✓ Overwrite successful: {retrieved_model}")
        print("\n✅ PASS: SQLite Overwrite")
        
    finally:
        Path(db_path).unlink(missing_ok=True)


def test_sqlite_ttl_pruning():
    """Test that TTL-based pruning removes old entries."""
    print("\n" + "=" * 70)
    print("TEST 3: SQLite TTL Pruning")
    print("=" * 70)
    
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    
    try:
        # Very short TTL for testing (1 second)
        store = SqliteContextStore(db_path=db_path, ttl_seconds=1)
        
        # Save a context
        request_id = "expire_me"
        context = np.array([1.0, 2.0, 3.0])
        store.save_context(request_id, context, "gpt-4")
        
        print(f"✓ Saved context with TTL=1s")
        
        # Verify it exists immediately
        retrieved, _, _ = store.get_context(request_id)
        assert retrieved is not None, "Context should exist immediately"
        print(f"✓ Context exists immediately after save")
        
        # Wait for TTL to expire
        print(f"  Waiting 1.5 seconds for TTL expiration...")
        time.sleep(1.5)
        
        # Prune expired entries
        pruned_count = store.prune()
        print(f"✓ Pruned {pruned_count} expired entries")
        
        # Verify it's gone
        retrieved_after_prune, _, _ = store.get_context(request_id)
        assert retrieved_after_prune is None, "Context should be None after prune"
        print(f"✓ Context successfully removed after TTL expiration")
        print("\n✅ PASS: SQLite TTL Pruning")
        
    finally:
        Path(db_path).unlink(missing_ok=True)


def test_ephemeral_store():
    """Test the RAM-based EphemeralContextStore."""
    print("\n" + "=" * 70)
    print("TEST 4: Ephemeral (RAM) Store")
    print("=" * 70)
    
    store = EphemeralContextStore(max_size=3)
    
    # Add 3 entries
    for i in range(3):
        request_id = f"request_{i}"
        context = np.array([float(i)] * 5)
        store.save_context(request_id, context, f"model_{i}")
    
    print(f"✓ Saved 3 contexts (max_size=3)")
    
    # Verify all 3 exist
    for i in range(3):
        retrieved, model, _ = store.get_context(f"request_{i}")
        assert retrieved is not None, f"Context {i} should exist"
    
    print(f"✓ All 3 contexts retrievable")
    
    # Add a 4th entry (should evict the oldest)
    store.save_context("request_3", np.array([3.0] * 5), "model_3")
    
    # Verify request_0 is gone (FIFO eviction)
    evicted, _, _ = store.get_context("request_0")
    assert evicted is None, "Oldest context should be evicted"
    
    # Verify request_3 exists
    newest, _, _ = store.get_context("request_3")
    assert newest is not None, "Newest context should exist"
    
    print(f"✓ FIFO eviction works correctly")
    print("\n✅ PASS: Ephemeral Store")


def test_missing_context():
    """Test that requesting a non-existent context returns None."""
    print("\n" + "=" * 70)
    print("TEST 5: Missing Context Handling")
    print("=" * 70)
    
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    
    try:
        store = SqliteContextStore(db_path=db_path)
        
        # Request non-existent context
        retrieved, model, _ = store.get_context("does_not_exist")
        
        assert retrieved is None, "Should return None for missing context"
        assert model is None, "Should return None for missing model"
        
        print(f"✓ Missing context returns (None, None)")
        print("\n✅ PASS: Missing Context Handling")
        
    finally:
        Path(db_path).unlink(missing_ok=True)


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("UNIT TESTS: SQLite Context Persistence")
    print("=" * 70)
    
    try:
        test_sqlite_write_read()
        test_sqlite_overwrite()
        test_sqlite_ttl_pruning()
        test_ephemeral_store()
        test_missing_context()
        
        print("\n" + "=" * 70)
        print("ALL TESTS PASSED ✅")
        print("=" * 70)
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
