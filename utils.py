import os
import json
import sys
from datetime import datetime


def get_config_dir():
    """Returns a persistent application folder compatible with all OS."""
    if os.name == "nt":  # Windows
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
        return os.path.join(base, "operafor")
    else:  # macOS, Linux, others
        return os.path.join(os.path.expanduser("~"), ".operafor")


# Determine paths based on whether we're running as a packaged app
if hasattr(sys, '_MEIPASS'):
    DATA_DIR = get_config_dir()
    os.makedirs(DATA_DIR, exist_ok=True)
    CONFIG_PATH = os.path.join(DATA_DIR, "config.json")
    CONV_FILE = os.path.join(DATA_DIR, "sandboxes.json")
    SANDBOXES_DIR = os.path.join(DATA_DIR, "sandboxes")
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
    CONV_FILE = os.path.join(BASE_DIR, "sandboxes.json")
    SANDBOXES_DIR = os.path.join(BASE_DIR, "sandboxes")


# Default configuration
DEFAULT_CONFIG = {
    "llm": {
        "endpoint": "https://openrouter.ai/api/v1",
        "model": "deepseek/deepseek-chat",
        "apiKey": "your_api_key"
    },
    "context_management": {
        "enabled": True,
        "strategy": "hybrid",
        "max_tokens": 4000,
        "summarization_threshold": 3000,
        "preserve_recent_messages": 5,
        "preserve_system_prompt": True,
        "max_context_during_run": 100000
    }
}


# --- Git utilities ---

def init_or_get_repo(sandbox_path: str):
    """Initialize git repository in sandbox folder if it doesn't exist, or get existing repo."""
    from dulwich import porcelain
    from dulwich.repo import Repo

    if not os.path.exists(sandbox_path):    
        os.makedirs(sandbox_path)
    git_path = os.path.join(sandbox_path, '.git')
    if not os.path.exists(git_path):
        repo = porcelain.init(sandbox_path)
    else:
        repo = Repo(sandbox_path)
    return repo


def write_conversation_json(sandbox_path: str, messages: list) -> str:
    """Write the conversation messages to conversation.json in the sandbox folder."""
    conversation_path = os.path.join(sandbox_path, 'conversation.json')
    with open(conversation_path, 'w') as f:
        json.dump({"messages": messages}, f, indent=2)
    return conversation_path


def commit_sandbox_changes(sandbox_path: str, messages: list, commit_message: str) -> str:
    """Commit all changes in the sandbox folder including conversation.json."""
    from dulwich import porcelain
    repo = init_or_get_repo(sandbox_path)
    
    # Write conversation.json
    write_conversation_json(sandbox_path, messages)
    
    # Add all files in the sandbox folder
    for root, dirs, files in os.walk(sandbox_path):
        # Skip .git directory
        if '.git' in dirs:
            dirs.remove('.git')
        for file in files:
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, sandbox_path)
            try:
                porcelain.add(sandbox_path, rel_path)
            except Exception as e:
                print(f"Warning: Could not add file {rel_path}: {e}")
    
    # Commit changes
    try:
        commit_hash = porcelain.commit(sandbox_path, commit_message.encode('utf-8'))
        return commit_hash.decode('utf-8')
    except Exception as e:
        print(f"Error committing changes: {e}")
        return ""


def revert_sandbox_to_commit(sandbox_path: str, commit_hash: str) -> bool:
    """Revert the sandbox to a specific commit."""
    from dulwich import porcelain
    try:
        porcelain.reset(sandbox_path, "hard", commit_hash)

        # Update sandboxes.json
        with open(CONV_FILE, 'r') as f:
            sandboxes = json.load(f)
        sandbox_id = os.path.basename(sandbox_path)
        conv = sandboxes.get(sandbox_id)
        if conv is not None:
            commits = conv.get("commits", [])
            idx = next((i for i, c in enumerate(commits) if c["hash"] == commit_hash), None)
            if idx is not None:
                conv["commits"] = commits[:idx+1]
            sandboxes[sandbox_id] = conv
            with open(CONV_FILE, 'w') as f:
                json.dump(sandboxes, f, indent=2)
        return True
    except Exception as e:
        print(f"Error reverting to commit {commit_hash}: {e}")
        return False


# --- Sandbox management ---

def load_all_sandboxes():
    if not os.path.exists(CONV_FILE):
        return {}
    with open(CONV_FILE, 'r') as f:
        return json.load(f)


def save_all_sandboxes(convs):
    with open(CONV_FILE, 'w') as f:
        json.dump(convs, f, indent=2)


def get_sandbox_path(sandbox_id: str) -> str:
    convs = load_all_sandboxes()
    conv = convs.get(sandbox_id)
    if conv and conv.get("custom_path"):
        return conv["custom_path"]
    return os.path.join(SANDBOXES_DIR, sandbox_id)


# --- Model utilities ---

def is_vlm(model_name: str) -> bool:
    """Check if the model has vision capabilities."""
    if not model_name:
        return False
    model_lower = model_name.lower()
    vision_keywords = ["vision", "4o", "claude-3", "gemini", "sonnet", "opus", "pixtral", "llavanext"]
    return any(keyword in model_lower for keyword in vision_keywords)


# --- Context cache management ---

def clear_sandbox_cache(sandbox_id: str) -> bool:
    """
    Clear context cache for a specific sandbox.
    Useful for debugging or when cache becomes corrupted.
    
    Returns:
        True if cache was cleared successfully, False otherwise
    """
    try:
        from context_cache import invalidate_cache
        invalidate_cache(sandbox_id)
        return True
    except Exception as e:
        print(f"Error clearing cache for sandbox {sandbox_id}: {e}")
        return False


def get_cache_stats(sandbox_id: str) -> dict:
    """
    Get cache statistics for a sandbox.
    
    Returns:
        Dictionary with cache statistics or empty dict if unavailable
    """
    try:
        from context_cache import load_stats
        return load_stats(sandbox_id)
    except Exception as e:
        print(f"Error loading cache stats for sandbox {sandbox_id}: {e}")
        return {}

