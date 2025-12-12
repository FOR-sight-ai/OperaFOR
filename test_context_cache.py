"""
Test suite for context caching system.

Tests cache creation, retrieval, invalidation, and incremental summarization.
"""

import os
import sys
import json
import shutil
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from context_cache import (
    get_cache_path,
    ensure_cache_directory,
    compute_message_hash,
    load_cache_metadata,
    save_cache_metadata,
    load_stats,
    save_stats,
    get_cached_summary,
    save_summary_segment,
    invalidate_cache
)
from utils import clear_sandbox_cache, get_cache_stats


def cleanup_test_sandbox(sandbox_id):
    """Clean up test sandbox directory."""
    from utils import get_sandbox_path
    sandbox_path = Path(get_sandbox_path(sandbox_id))
    if sandbox_path.exists():
        shutil.rmtree(sandbox_path)


def test_cache_directory_creation():
    """Test that cache directory is created properly."""
    print("Test 1: Cache directory creation...")
    
    sandbox_id = "test_cache_dir_123"
    cleanup_test_sandbox(sandbox_id)
    
    try:
        cache_path = ensure_cache_directory(sandbox_id)
        
        assert cache_path.exists(), "Cache directory not created"
        assert (cache_path / "summaries").exists(), "Summaries subdirectory not created"
        
        print("✓ Cache directory creation test passed")
        return True
    except AssertionError as e:
        print(f"✗ Cache directory creation test failed: {e}")
        return False
    finally:
        cleanup_test_sandbox(sandbox_id)


def test_message_hash():
    """Test message hash computation."""
    print("\nTest 2: Message hash computation...")
    
    messages1 = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"}
    ]
    
    messages2 = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"}
    ]
    
    messages3 = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Different response"}
    ]
    
    try:
        hash1 = compute_message_hash(messages1)
        hash2 = compute_message_hash(messages2)
        hash3 = compute_message_hash(messages3)
        
        assert hash1 == hash2, "Same messages should have same hash"
        assert hash1 != hash3, "Different messages should have different hash"
        
        print("✓ Message hash computation test passed")
        return True
    except AssertionError as e:
        print(f"✗ Message hash computation test failed: {e}")
        return False


def test_metadata_operations():
    """Test metadata save and load."""
    print("\nTest 3: Metadata operations...")
    
    sandbox_id = "test_metadata_456"
    cleanup_test_sandbox(sandbox_id)
    
    try:
        # Load default metadata
        metadata = load_cache_metadata(sandbox_id)
        assert metadata["version"] == "1.0", "Default metadata version incorrect"
        assert metadata["total_messages_cached"] == 0, "Default cached count should be 0"
        
        # Modify and save
        metadata["total_messages_cached"] = 10
        metadata["segments"].append({
            "segment_id": "test_segment",
            "message_range": [0, 5]
        })
        save_cache_metadata(sandbox_id, metadata)
        
        # Load again and verify
        loaded = load_cache_metadata(sandbox_id)
        assert loaded["total_messages_cached"] == 10, "Metadata not saved correctly"
        assert len(loaded["segments"]) == 1, "Segments not saved correctly"
        
        print("✓ Metadata operations test passed")
        return True
    except AssertionError as e:
        print(f"✗ Metadata operations test failed: {e}")
        return False
    finally:
        cleanup_test_sandbox(sandbox_id)


def test_summary_save_and_retrieve():
    """Test saving and retrieving summaries."""
    print("\nTest 4: Summary save and retrieve...")
    
    sandbox_id = "test_summary_789"
    cleanup_test_sandbox(sandbox_id)
    
    try:
        # Create test messages
        messages = [
            {"role": "user", "content": "Test message 1"},
            {"role": "assistant", "content": "Response 1"}
        ]
        
        message_hash = compute_message_hash(messages)
        summary_text = "This is a test summary"
        
        # Save summary
        save_summary_segment(
            sandbox_id,
            "segment_0",
            summary_text,
            {
                "message_range": [0, 1],
                "token_count": 50,
                "summary_tokens": 10,
                "hash": message_hash
            }
        )
        
        # Retrieve summary
        retrieved = get_cached_summary(sandbox_id, (0, 1), message_hash)
        assert retrieved == summary_text, f"Retrieved summary doesn't match: {retrieved}"
        
        # Check stats
        stats = load_stats(sandbox_id)
        assert stats["cache_hits"] == 1, "Cache hit not recorded"
        assert stats["summaries_generated"] == 1, "Summary generation not recorded"
        
        print("✓ Summary save and retrieve test passed")
        return True
    except AssertionError as e:
        print(f"✗ Summary save and retrieve test failed: {e}")
        return False
    finally:
        cleanup_test_sandbox(sandbox_id)


def test_cache_invalidation():
    """Test cache invalidation with hash mismatch."""
    print("\nTest 5: Cache invalidation...")
    
    sandbox_id = "test_invalidation_abc"
    cleanup_test_sandbox(sandbox_id)
    
    try:
        # Save summary with one hash
        messages1 = [{"role": "user", "content": "Original"}]
        hash1 = compute_message_hash(messages1)
        
        save_summary_segment(
            sandbox_id,
            "segment_0",
            "Original summary",
            {
                "message_range": [0, 0],
                "token_count": 10,
                "summary_tokens": 5,
                "hash": hash1
            }
        )
        
        # Try to retrieve with different hash
        messages2 = [{"role": "user", "content": "Modified"}]
        hash2 = compute_message_hash(messages2)
        
        retrieved = get_cached_summary(sandbox_id, (0, 0), hash2)
        assert retrieved is None, "Cache should be invalidated with hash mismatch"
        
        # The first save_summary_segment doesn't count as a miss, only the get_cached_summary does
        # So we should have exactly 1 cache miss from the failed retrieval above
        stats = load_stats(sandbox_id)
        # Note: The initial save generates a summary (summaries_generated=1)
        # The failed retrieval should record a cache miss
        assert stats.get("cache_misses", 0) == 1, f"Expected 1 cache miss, got {stats.get('cache_misses', 0)}"
        
        print("✓ Cache invalidation test passed")
        return True
    except AssertionError as e:
        print(f"✗ Cache invalidation test failed: {e}")
        return False
    finally:
        cleanup_test_sandbox(sandbox_id)


def test_cache_clearing():
    """Test cache clearing utility."""
    print("\nTest 6: Cache clearing...")
    
    sandbox_id = "test_clear_def"
    cleanup_test_sandbox(sandbox_id)
    
    try:
        # Create cache with data
        save_summary_segment(
            sandbox_id,
            "segment_0",
            "Test summary",
            {
                "message_range": [0, 5],
                "token_count": 100,
                "summary_tokens": 20,
                "hash": "test_hash"
            }
        )
        
        cache_path = get_cache_path(sandbox_id)
        assert cache_path.exists(), "Cache should exist before clearing"
        
        # Clear cache
        result = clear_sandbox_cache(sandbox_id)
        assert result is True, "Cache clearing should succeed"
        assert not cache_path.exists(), "Cache should not exist after clearing"
        
        print("✓ Cache clearing test passed")
        return True
    except AssertionError as e:
        print(f"✗ Cache clearing test failed: {e}")
        return False
    finally:
        cleanup_test_sandbox(sandbox_id)


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("Context Cache System Test Suite")
    print("=" * 60)
    
    tests = [
        test_cache_directory_creation,
        test_message_hash,
        test_metadata_operations,
        test_summary_save_and_retrieve,
        test_cache_invalidation,
        test_cache_clearing
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 60)
    print(f"Test Results: {sum(results)}/{len(results)} passed")
    print("=" * 60)
    
    if all(results):
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
