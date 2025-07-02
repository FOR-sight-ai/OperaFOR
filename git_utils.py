import os
import json
from datetime import datetime
from dulwich import porcelain
from dulwich.repo import Repo
from dulwich.objects import Blob, Tree, Commit
from dulwich.index import build_index_from_tree

def init_or_get_repo(sandbox_path: str) -> Repo:
    """Initialize git repository in sandbox folder if it doesn't exist, or get existing repo.
    Args:
        sandbox_path (str): Path to the sandbox folder
    Returns:
        Repo: The dulwich repository object
    """
    git_path = os.path.join(sandbox_path, '.git')
    if not os.path.exists(git_path):
        repo = porcelain.init(sandbox_path)
    else:
        repo = Repo(sandbox_path)
    return repo

def write_conversation_json(sandbox_path: str, messages: list) -> str:
    """Write the conversation messages to conversation.json in the sandbox folder.
    Args:
        sandbox_path (str): Path to the sandbox folder
        messages (list): List of conversation messages
    Returns:
        str: Path to the conversation.json file
    """
    conversation_path = os.path.join(sandbox_path, 'conversation.json')
    with open(conversation_path, 'w') as f:
        json.dump({"messages": messages}, f, indent=2)
    return conversation_path

def commit_sandbox_changes(sandbox_path: str, messages: list, commit_message: str) -> str:
    """Commit all changes in the sandbox folder including conversation.json.
    Args:
        sandbox_path (str): Path to the sandbox folder
        messages (list): List of conversation messages
        commit_message (str): Commit message
    Returns:
        str: The commit hash
    """
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
    """Revert the sandbox to a specific commit.
    Args:
        sandbox_path (str): Path to the sandbox folder
        commit_hash (str): The commit hash to revert to
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Use porcelain reset which properly handles file checkout
        porcelain.reset(sandbox_path, "hard", commit_hash)
        return True
    except Exception as e:
        print(f"Error reverting to commit {commit_hash}: {e}")
        return False