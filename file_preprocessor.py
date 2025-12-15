import os
import json
import base64
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from utils import get_sandbox_path


logger = logging.getLogger(__name__)


def get_file_hash(file_path: str) -> str:
    """Calculate SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except Exception as e:
        logger.error(f"Error hashing file {file_path}: {e}")
        return ""


def check_git_diff(sandbox_path: str, file_path: str) -> bool:
    """
    Check if a file has changed since last git commit using dulwich.

    Returns:
        True if file has changed or is new, False if unchanged
    """
    try:
        from dulwich.repo import Repo
        from dulwich.errors import NotGitRepository

        # Check if git repo exists
        try:
            repo = Repo(sandbox_path)
        except (NotGitRepository, FileNotFoundError):
            return True  # No git repo, treat as new

        # Get relative path from sandbox
        rel_path = os.path.relpath(file_path, sandbox_path)
        # Convert to bytes for dulwich (uses bytes for paths)
        rel_path_bytes = rel_path.encode('utf-8')

        # Check if file is tracked in the index
        index = repo.open_index()
        if rel_path_bytes not in index:
            # File is not tracked, it's new
            return True

        # Check if file has uncommitted changes
        # Get HEAD tree
        try:
            head = repo[b'HEAD']
            tree = repo[head.tree]
        except KeyError:
            # No HEAD commit yet (empty repo)
            return True

        # Get the file content from HEAD
        try:
            # Navigate through tree structure to find the file
            parts = rel_path.split(os.sep)
            current_tree = tree

            for i, part in enumerate(parts[:-1]):
                # Navigate through directories
                part_bytes = part.encode('utf-8')
                mode, sha = current_tree[part_bytes]
                current_tree = repo[sha]

            # Get the file blob from the tree
            filename_bytes = parts[-1].encode('utf-8')
            mode, blob_sha = current_tree[filename_bytes]
            blob = repo[blob_sha]
            head_content = blob.data

        except (KeyError, IndexError):
            # File doesn't exist in HEAD, it's new
            return True

        # Compare with working tree file
        try:
            with open(file_path, 'rb') as f:
                working_content = f.read()

            # Compare content
            return head_content != working_content

        except Exception as e:
            logger.error(f"Error reading working file {file_path}: {e}")
            return True

    except Exception as e:
        logger.error(f"Error checking git diff for {file_path}: {e}")
        return True  # On error, assume file changed


def convert_pdf_to_text(file_path: str, model_name: str = None) -> Tuple[str, str]:
    """
    Convert PDF to text or images based on model capabilities.

    Args:
        file_path: Path to PDF file
        model_name: Name of the model (to check if it supports vision)

    Returns:
        Tuple of (converted_content, format_type)
        format_type is either 'text' or 'image'
    """
    try:
        import fitz  # PyMuPDF
        from utils import is_vlm

        doc = fitz.open(file_path)

        # Check if model supports vision
        if is_vlm(model_name):
            # Convert to images
            images = []
            for page in doc:
                pix = page.get_pixmap()
                img_data = pix.tobytes("png")
                b64_img = base64.b64encode(img_data).decode("utf-8")
                images.append(b64_img)
            doc.close()

            # Return as JSON for image format
            content = json.dumps({"__type__": "image", "images": images})
            return content, "image"
        else:
            # Convert to text
            text_content = []
            for i, page in enumerate(doc):
                text = page.get_text()
                text_content.append(f"--- Page {i+1} ---\n{text}")
            doc.close()

            content = "\n".join(text_content) if text_content else "PDF is empty or contains no extractable text."
            return content, "text"

    except Exception as e:
        error_msg = f"Error converting PDF {file_path}: {e}"
        logger.error(error_msg)
        return error_msg, "error"


def get_converted_file_path(original_path: str, format_type: str) -> str:
    """
    Get path for converted file.

    Args:
        original_path: Path to original file
        format_type: 'text' or 'image'

    Returns:
        Path where converted content should be stored
    """
    # Create .converted directory in same location as original
    dir_name = os.path.dirname(original_path)
    base_name = os.path.basename(original_path)
    name_without_ext = os.path.splitext(base_name)[0]

    converted_dir = os.path.join(dir_name, '.converted')
    os.makedirs(converted_dir, exist_ok=True)

    if format_type == "text":
        return os.path.join(converted_dir, f"{name_without_ext}.txt")
    elif format_type == "image":
        return os.path.join(converted_dir, f"{name_without_ext}.json")
    else:
        return os.path.join(converted_dir, f"{name_without_ext}.converted")


def should_convert_file(file_path: str, converted_path: str, sandbox_path: str) -> bool:
    """
    Determine if file needs conversion.

    Returns:
        True if conversion is needed, False if cached version is valid
    """
    # If converted file doesn't exist, convert
    if not os.path.exists(converted_path):
        return True

    # Check if original file has changed via git
    has_changed = check_git_diff(sandbox_path, file_path)
    if has_changed:
        return True

    # Check file modification times as fallback
    original_mtime = os.path.getmtime(file_path)
    converted_mtime = os.path.getmtime(converted_path)

    if original_mtime > converted_mtime:
        return True

    return False


def preprocess_file(file_path: str, sandbox_path: str, model_name: str = None) -> Optional[str]:
    """
    Preprocess a file by converting non-text formats to text.

    Args:
        file_path: Absolute path to file
        sandbox_path: Path to sandbox
        model_name: Model name for vision capability check

    Returns:
        Path to converted file if conversion happened, None if no conversion needed
    """
    # Check if file needs conversion
    file_ext = os.path.splitext(file_path)[1].lower()

    if file_ext == '.pdf':
        # Determine format type
        from utils import is_vlm
        format_type = "image" if is_vlm(model_name) else "text"

        # Get converted file path
        converted_path = get_converted_file_path(file_path, format_type)

        # Check if conversion is needed
        if not should_convert_file(file_path, converted_path, sandbox_path):
            logger.info(f"Using cached conversion for {file_path}")
            return converted_path

        # Convert PDF
        logger.info(f"Converting PDF {file_path} to {format_type}")
        content, actual_format = convert_pdf_to_text(file_path, model_name)

        if actual_format == "error":
            logger.error(f"Failed to convert {file_path}")
            return None

        # Save converted content
        try:
            with open(converted_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"Saved converted content to {converted_path}")
            return converted_path
        except Exception as e:
            logger.error(f"Error saving converted file {converted_path}: {e}")
            return None

    # Add more file type conversions here (e.g., .docx, .xlsx, etc.)
    # elif file_ext == '.docx':
    #     ...

    return None


def preprocess_sandbox_files(sandbox_id: str, model_name: str = None) -> Dict[str, str]:
    """
    Preprocess all files in sandbox that need conversion.

    Args:
        sandbox_id: ID of the sandbox
        model_name: Model name for vision capability check

    Returns:
        Dictionary mapping original file paths to converted file paths
    """
    sandbox_path = get_sandbox_path(sandbox_id)
    conversions = {}

    if not os.path.exists(sandbox_path):
        return conversions

    # Walk through sandbox files
    for root, dirs, files in os.walk(sandbox_path):
        # Skip .git and .converted directories
        dirs[:] = [d for d in dirs if d not in ['.git', '.converted']]

        for file in files:
            file_path = os.path.join(root, file)

            # Try to preprocess
            converted_path = preprocess_file(file_path, sandbox_path, model_name)

            if converted_path:
                # Store relative paths for easier handling
                rel_original = os.path.relpath(file_path, sandbox_path)
                rel_converted = os.path.relpath(converted_path, sandbox_path)
                conversions[rel_original] = rel_converted

    if conversions:
        logger.info(f"Preprocessed {len(conversions)} files in sandbox {sandbox_id}")

    return conversions
