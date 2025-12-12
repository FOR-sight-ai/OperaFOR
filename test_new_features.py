#!/usr/bin/env python3
"""
Test script for new features:
1. Ephemeral sandbox context injection
2. URL detection and processing
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from url_handler import extract_urls, process_urls_in_prompt
from agent import inject_sandbox_context
from utils import get_sandbox_path
import tempfile
import shutil


def test_url_extraction():
    """Test URL extraction from text."""
    print("Testing URL extraction...")
    
    test_text = """
    Check out https://example.com/page.html and also
    look at /tmp/test.txt and C:\\Users\\test\\file.txt
    Also see file:///home/user/document.pdf
    """
    
    urls = extract_urls(test_text)
    print(f"  Web URLs: {urls['web_urls']}")
    print(f"  File URLs: {urls['file_urls']}")
    
    assert 'https://example.com/page.html' in urls['web_urls'], "Web URL not detected"
    print("✓ URL extraction works!")


def test_sandbox_context_injection():
    """Test ephemeral sandbox context injection."""
    print("\nTesting sandbox context injection...")
    
    # Create a temporary sandbox with some files
    temp_dir = tempfile.mkdtemp()
    sandbox_id = "test_sandbox"
    
    try:
        # Create some test files
        sandbox_path = os.path.join(temp_dir, sandbox_id)
        os.makedirs(sandbox_path, exist_ok=True)
        with open(os.path.join(sandbox_path, "test1.txt"), "w") as f:
            f.write("test content 1")
        with open(os.path.join(sandbox_path, "test2.py"), "w") as f:
            f.write("print('hello')")
        
        # Mock get_sandbox_path in all modules
        def mock_get_path(sid):
            return os.path.join(temp_dir, sid)
        
        import utils
        import agent
        import tools
        
        original_utils = utils.get_sandbox_path
        original_agent = agent.get_sandbox_path
        original_tools = tools.get_sandbox_path
        
        utils.get_sandbox_path = mock_get_path
        agent.get_sandbox_path = mock_get_path
        tools.get_sandbox_path = mock_get_path
        
        # Test injection
        messages = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
        
        updated_messages = inject_sandbox_context(sandbox_id, messages)
        
        # Restore original functions
        utils.get_sandbox_path = original_utils
        agent.get_sandbox_path = original_agent
        tools.get_sandbox_path = original_tools
        
        # Verify context was injected
        assert len(updated_messages) == 2, f"Expected 2 messages, got {len(updated_messages)}"
        assert updated_messages[1]["role"] == "system", "Injected message should be system role"
        assert "test1.txt" in updated_messages[1]["content"], "File list should contain test1.txt"
        assert "test2.py" in updated_messages[1]["content"], "File list should contain test2.py"
        
        print("  Injected context:", updated_messages[1]["content"][:100] + "...")
        print("✓ Sandbox context injection works!")
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_empty_sandbox_no_injection():
    """Test that empty sandboxes don't get context injection."""
    print("\nTesting empty sandbox (no injection)...")
    
    temp_dir = tempfile.mkdtemp()
    sandbox_id = "empty_sandbox"
    
    try:
        # Create empty sandbox
        os.makedirs(os.path.join(temp_dir, sandbox_id), exist_ok=True)
        
        # Mock get_sandbox_path
        def mock_get_path(sid):
            return os.path.join(temp_dir, sid)
        
        import utils
        original_get_path = utils.get_sandbox_path
        utils.get_sandbox_path = mock_get_path
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
        
        updated_messages = inject_sandbox_context(sandbox_id, messages)
        
        utils.get_sandbox_path = original_get_path
        
        # Verify no injection for empty sandbox
        assert len(updated_messages) == 1, f"Empty sandbox should not inject context, got {len(updated_messages)} messages"
        
        print("✓ Empty sandbox correctly skips injection!")
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    print("=" * 60)
    print("Testing OperaFOR New Features")
    print("=" * 60)
    
    try:
        test_url_extraction()
        test_sandbox_context_injection()
        test_empty_sandbox_no_injection()
        
        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
