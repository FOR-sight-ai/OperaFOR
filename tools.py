import os
import json
import base64
import difflib
import re
from typing import Any, Dict, List, Union
from datetime import datetime

from utils import get_sandbox_path, is_vlm


# --- Tool Functions ---

def list_sandbox_files(sandbox_id: str, max_depth: int = None) -> List[str]:
    """
    List all files in the sandbox directory.

    Args:
        sandbox_id: The sandbox ID
        max_depth: Maximum depth to traverse (None = unlimited, 1 = top-level only, etc.)
    """
    sandbox_path = get_sandbox_path(sandbox_id)
    output_files = []
    if not os.path.exists(sandbox_path):
         return ["Sandbox directory does not exist yet."]

    for root, dirs, files in os.walk(sandbox_path):
        # Calculate current depth relative to sandbox_path
        rel_root = os.path.relpath(root, sandbox_path)
        if rel_root == '.':
            current_depth = 0
        else:
            current_depth = rel_root.count(os.sep) + 1

        # Skip directories beyond max_depth
        if max_depth is not None and current_depth >= max_depth:
            dirs[:] = []  # Don't traverse deeper

        for file in files:
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, sandbox_path)
            if '.git' in rel_path or rel_path.startswith('.git/') or rel_path == 'conversation.json' or rel_path == '.context_cache':
                continue
            if os.path.isfile(file_path):
                output_files.append(rel_path)

    if len(output_files) == 0:
        return ["No files found in this sandbox."]
    return output_files


def read_file_sandbox(sandbox_id: str, file_name: str, model_name: str = None) -> str:
    """Read a file from the sandbox directory."""
    from file_preprocessor import get_converted_file_path

    sandbox_path = get_sandbox_path(sandbox_id)
    file_path = os.path.join(sandbox_path, file_name)
    if not os.path.exists(file_path):
        return f"Error: File {file_name} not found"

    # Check if this is a PDF that has been preprocessed
    if file_name.lower().endswith(".pdf"):
        # Determine expected format type
        format_type = "image" if is_vlm(model_name) else "text"
        converted_path = get_converted_file_path(file_path, format_type)

        # If converted file exists, read from it
        if os.path.exists(converted_path):
            try:
                with open(converted_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # If it's image format, the content is JSON
                if format_type == "image":
                    # Validate it's proper JSON
                    data = json.loads(content)
                    if data.get("__type__") == "image":
                        return content
                else:
                    return content
            except Exception as e:
                # Fall through to read original if converted file is corrupted
                pass

        # If no converted file, return error suggesting preprocessing should have happened
        return f"Error: PDF file {file_name} has not been preprocessed. Please ensure file preprocessing runs before agent loop."

    try:
        with open(file_path, 'r') as f:
            return f.read()
    except Exception as e:
         return f"Error reading file: {e}"


def save_file_sandbox(sandbox_id: str, file_name: str, content: str) -> str:
    """Write content to a file in the sandbox."""
    sandbox_path = get_sandbox_path(sandbox_id)
    file_path = os.path.join(sandbox_path, file_name)
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            f.write(content)
        return f"File {file_name} saved successfully."
    except Exception as e:
        return f"Error saving file: {e}"


def append_file_sandbox(sandbox_id: str, file_name: str, content: str) -> str:
    """Append content to a file in the sandbox."""
    sandbox_path = get_sandbox_path(sandbox_id)
    file_path = os.path.join(sandbox_path, file_name)
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'a') as f:
            f.write(content)
        return f"Content appended to {file_name}."
    except Exception as e:
        return f"Error appending to file: {e}"


def delete_this_file_sandbox(sandbox_id: str, file_name: str) -> str:
    """Delete a file in the sandbox."""
    sandbox_path = get_sandbox_path(sandbox_id)
    file_path = os.path.join(sandbox_path, file_name)
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            return f"File {file_name} deleted."
        return f"File {file_name} not found."
    except Exception as e:
        return f"Error deleting file: {e}"


def edit_file_sandbox(sandbox_id: str, file_path: str, edits: List[Dict[str, str]], dry_run: bool = False, options: Dict[str, Any] = None) -> Union[Dict[str, Any], str]:
    """Make selective edits to files while preserving formatting."""
    
    def normalize_line_endings(text: str) -> str:
        return text.replace("\r\n", "\n")

    def normalize_whitespace(text: str) -> str:
        result = re.sub(r"[ \t]+", " ", text)
        result = "\n".join(line.strip() for line in result.split("\n"))
        return result

    def get_line_indentation(line: str) -> str:
        match = re.match(r"^(\s*)", line)
        return match.group(1) if match else ""

    def preserve_indentation(old_text: str, new_text: str) -> str:
        if ("- " in new_text or "* " in new_text) and ("- " in old_text or "* " in old_text):
            return new_text
        old_lines = old_text.split("\n")
        new_lines = new_text.split("\n")
        if not old_lines or not new_lines:
            return new_text
        base_indent = get_line_indentation(old_lines[0]) if old_lines and old_lines[0].strip() else ""
        old_indents = {i: get_line_indentation(line) for i, line in enumerate(old_lines) if line.strip()}
        new_indents = {i: get_line_indentation(line) for i, line in enumerate(new_lines) if line.strip()}
        first_new_indent_len = len(new_indents.get(0, "")) if new_indents else 0
        result_lines = []
        for i, new_line in enumerate(new_lines):
            if not new_line.strip():
                result_lines.append("")
                continue
            new_indent = new_indents.get(i, "")
            if i < len(old_lines) and i in old_indents:
                target_indent = old_indents[i]
            elif i == 0:
                target_indent = base_indent
            elif first_new_indent_len > 0:
                curr_indent_len = len(new_indent)
                indent_diff = max(0, curr_indent_len - first_new_indent_len)
                target_indent = base_indent
                for prev_i in range(i - 1, -1, -1):
                    if prev_i in old_indents and prev_i in new_indents:
                        prev_old = old_indents[prev_i]
                        prev_new = new_indents[prev_i]
                        if len(prev_new) <= curr_indent_len:
                            relative_spaces = curr_indent_len - len(prev_new)
                            target_indent = prev_old + " " * relative_spaces
                            break
            else:
                target_indent = new_indent
            result_lines.append(target_indent + new_line.lstrip())
        return "\n".join(result_lines)

    def create_unified_diff(original: str, modified: str, file_path: str) -> str:
        original_lines = original.splitlines(True)
        modified_lines = modified.splitlines(True)
        diff_lines = difflib.unified_diff(original_lines, modified_lines, fromfile=f"a/{file_path}", tofile=f"b/{file_path}", lineterm="")
        return "".join(diff_lines)

    def find_exact_match(content: str, pattern: str):
        if pattern in content:
            lines_before = content[: content.find(pattern)].count("\n")
            line_count = pattern.count("\n") + 1
            return True, lines_before, line_count
        return False, -1, 0

    full_sandbox_path = get_sandbox_path(sandbox_id)
    full_file_path = os.path.join(full_sandbox_path, file_path)

    if not file_path or not isinstance(file_path, str):
        return json.dumps({"success": False, "error": f"File path must be a non-empty string, got {type(file_path)}"})
    if not isinstance(edits, list) or not edits:
        return json.dumps({"success": False, "error": "Edits must be a non-empty list"})
    if not os.path.isfile(full_file_path):
        return json.dumps({"success": False, "error": f"File not found: {file_path}"})

    normalized_edits = []
    for i, edit in enumerate(edits):
        if not isinstance(edit, dict):
            return json.dumps({"success": False, "error": f"Edit #{i} must be a dictionary"})
        if "old_text" not in edit or "new_text" not in edit:
            return json.dumps({"success": False, "error": f"Edit #{i} missing old_text or new_text"})
        normalized_edits.append({"old_text": edit["old_text"], "new_text": edit["new_text"]})

    preserve_indent = options.get("preserve_indentation", True) if options else True
    normalize_ws = options.get("normalize_whitespace", True) if options else True

    try:
        with open(full_file_path, "r", encoding="utf-8") as f:
            original_content = f.read()
    except Exception as e:
        return json.dumps({"success": False, "error": f"Error reading file: {str(e)}"})

    match_results = []
    changes_made = False
    modified_content = normalize_line_endings(original_content)
    
    for i, edit in enumerate(normalized_edits):
        old = normalize_line_endings(edit["old_text"])
        new = normalize_line_endings(edit["new_text"])
        
        if normalize_ws:
            old_search = normalize_whitespace(old)
            pass 

        if old in modified_content:
             modified_content = modified_content.replace(old, new, 1)
             changes_made = True
             match_results.append({"edit_index": i, "match_type": "exact"})
        else:
             match_results.append({"edit_index": i, "match_type": "failed"})

    if not changes_made:
         return json.dumps({"success": True, "message": "No changes made"})

    diff = create_unified_diff(original_content, modified_content, file_path)
    
    if not dry_run:
        try:
            with open(full_file_path, "w", encoding="utf-8") as f:
                f.write(modified_content)
        except Exception as e:
             return json.dumps({"success": False, "error": f"Error writing file: {str(e)}"})

    return json.dumps({"success": True, "diff": diff})


def get_folder_structure_sandbox(sandbox_id: str, max_depth: int = 2) -> str:
    """Returns a tree-like string representation of the folder structure."""
    sandbox_path = get_sandbox_path(sandbox_id)
    if not os.path.exists(sandbox_path):
        return "Sandbox directory does not exist."
    
    output = []
    
    def add_to_tree(path, prefix="", depth=0):
        if depth > max_depth:
            return
        
        try:
            entries = sorted(os.listdir(path))
        except PermissionError:
            return

        entries = [e for e in entries if e != '.git' and e != '.context_cache' and e != 'conversation.json']
        
        for i, entry in enumerate(entries):
            is_last = (i == len(entries) - 1)
            full_path = os.path.join(path, entry)
            is_dir = os.path.isdir(full_path)
            
            connector = "└── " if is_last else "├── "
            output.append(f"{prefix}{connector}{entry}{'/' if is_dir else ''}")
            
            if is_dir:
                extension = "    " if is_last else "│   "
                add_to_tree(full_path, prefix + extension, depth + 1)
                
    output.append(".")
    add_to_tree(sandbox_path)
    return "\n".join(output)


def search_files_sandbox(sandbox_id: str, pattern: str) -> List[str]:
    """Search for files in the sandbox matching a glob pattern."""
    import fnmatch
    sandbox_path = get_sandbox_path(sandbox_id)
    if not os.path.exists(sandbox_path):
        return ["Sandbox directory does not exist."]
    
    matches = []
    
    for root, dirs, files in os.walk(sandbox_path):
        # Skip git
        if '.git' in dirs:
            dirs.remove('.git')
            
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                rel_path = os.path.relpath(os.path.join(root, name), sandbox_path)
                matches.append(rel_path)
    
    if not matches:
        return [f"No files found matching pattern '{pattern}'"]
    
    return matches


def search_content_sandbox(sandbox_id: str, query: str, case_sensitive: bool = False) -> str:
    """Search for text content within files."""
    sandbox_path = get_sandbox_path(sandbox_id)
    if not os.path.exists(sandbox_path):
        return "Sandbox directory does not exist."
    
    results = []
    
    for root, dirs, files in os.walk(sandbox_path):
        if '.git' in dirs:
             dirs.remove('.git')
             
        for file in files:
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, sandbox_path)
            
            # Skip likely binary files or large files to prevent freezing
            skip_exts = ['.pyc', '.git', '.png', '.jpg', '.jpeg', '.pdf', '.zip', '.exe','conversation.json','.context_cache']
            if any(file.lower().endswith(ext) for ext in skip_exts):
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    
                for i, line in enumerate(lines):
                    if case_sensitive:
                        match = query in line
                    else:
                        match = query.lower() in line.lower()
                        
                    if match:
                        # Truncate line if too long
                        content = line.strip()
                        if len(content) > 200:
                            content = content[:200] + "..."
                        results.append(f"{rel_path}:{i+1}: {content}")
                        if len(results) > 100:  # Global limit
                             break
            except Exception:
                pass  # Skip files we can't read
            
            if len(results) > 100:
                results.append("... (results truncated)")
                break
                
    if not results:
        return f"No matches found for '{query}'"
        
    return "\n".join(results)


def get_file_info_sandbox(sandbox_id: str, file_path: str) -> str:
    """Get metadata about a specific file."""
    sandbox_path = get_sandbox_path(sandbox_id)
    full_path = os.path.join(sandbox_path, file_path)
    
    if not os.path.exists(full_path):
        return f"File not found: {file_path}"
        
    try:
        stat = os.stat(full_path)
        is_dir = os.path.isdir(full_path)
        
        info = {
            "name": os.path.basename(file_path),
            "type": "directory" if is_dir else "file",
            "size_bytes": stat.st_size,
            "created": datetime.fromtimestamp(stat.st_ctime).strftime('%Y-%m-%d %H:%M:%S'),
            "modified": datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
        }
        
        if not is_dir:
            # Count lines for text files
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    info["lines"] = sum(1 for _ in f)
            except:
                info["lines"] = "N/A (binary or non-utf8)"
                
        return json.dumps(info, indent=2)
    except Exception as e:
        return f"Error getting file info: {e}"


# --- Tool Registry ---

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "list_sandbox_files",
            "description": "List all files in the sandbox directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sandbox_id": {"type": "string", "description": "The ID of the sandbox."}
                },
                "required": ["sandbox_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_folder_structure_sandbox",
            "description": "Get a tree-like representation of the folders.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sandbox_id": {"type": "string", "description": "The ID of the sandbox."},
                    "max_depth": {"type": "integer", "description": "Max depth to traverse (default 2)."}
                },
                "required": ["sandbox_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_files_sandbox",
            "description": "Search for files matching a pattern (e.g. *.py).",
            "parameters": {
                "type": "object",
                "properties": {
                    "sandbox_id": {"type": "string", "description": "The ID of the sandbox."},
                    "pattern": {"type": "string", "description": "Glob pattern to search for."}
                },
                "required": ["sandbox_id", "pattern"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_content_sandbox",
            "description": "Search for text content within files (grep-like).",
            "parameters": {
                "type": "object",
                "properties": {
                    "sandbox_id": {"type": "string", "description": "The ID of the sandbox."},
                    "query": {"type": "string", "description": "Text to search for."},
                    "case_sensitive": {"type": "boolean", "description": "Case sensitive match."}
                },
                "required": ["sandbox_id", "query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_file_info_sandbox",
            "description": "Get metadata (size, lines, time) for a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sandbox_id": {"type": "string", "description": "The ID of the sandbox."},
                    "file_path": {"type": "string", "description": "Relative path to the file."}
                },
                "required": ["sandbox_id", "file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file_sandbox",
            "description": "Read a file from the sandbox directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sandbox_id": {"type": "string", "description": "The ID of the sandbox."},
                    "file_name": {"type": "string", "description": "The name of the file to read."}
                },
                "required": ["sandbox_id", "file_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_file_sandbox",
            "description": "Write content to a file in the sandbox (overwrites).",
            "parameters": {
                "type": "object",
                "properties": {
                    "sandbox_id": {"type": "string", "description": "The ID of the sandbox."},
                    "file_name": {"type": "string", "description": "The name of the file to write to."},
                    "content": {"type": "string", "description": "Content to write."}
                },
                "required": ["sandbox_id", "file_name", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "append_file_sandbox",
            "description": "Append content to a file in the sandbox.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sandbox_id": {"type": "string", "description": "The ID of the sandbox."},
                    "file_name": {"type": "string", "description": "The name of the file to append to."},
                    "content": {"type": "string", "description": "Content to append."}
                },
                "required": ["sandbox_id", "file_name", "content"]
            }
        }
    },
     {
        "type": "function",
        "function": {
            "name": "delete_this_file_sandbox",
            "description": "Delete a file in the sandbox.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sandbox_id": {"type": "string", "description": "The ID of the sandbox."},
                    "file_name": {"type": "string", "description": "The name of the file to delete."}
                },
                "required": ["sandbox_id", "file_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file_sandbox",
            "description": "Edit a file using search and replace blocks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sandbox_id": {"type": "string", "description": "The ID of the sandbox."},
                    "file_path": {"type": "string", "description": "Path to the file."},
                    "edits": {
                        "type": "array", 
                        "items": {
                            "type": "object",
                            "properties": {
                                "old_text": {"type": "string"},
                                "new_text": {"type": "string"}
                            },
                            "required": ["old_text", "new_text"]
                        }
                    }
                },
                "required": ["sandbox_id", "file_path", "edits"]
            }
        }
    }
]


# --- Tool Execution ---

def execute_tool(name: str, args: Dict[str, Any]) -> str:
    """Execute a tool by name with arguments."""
    try:
        if name == "list_sandbox_files":
            return str(list_sandbox_files(args.get("sandbox_id")))
        elif name == "get_folder_structure_sandbox":
            return str(get_folder_structure_sandbox(args.get("sandbox_id"), args.get("max_depth", 2)))
        elif name == "search_files_sandbox":
            return str(search_files_sandbox(args.get("sandbox_id"), args.get("pattern")))
        elif name == "search_content_sandbox":
             return str(search_content_sandbox(args.get("sandbox_id"), args.get("query"), args.get("case_sensitive", False)))
        elif name == "get_file_info_sandbox":
             return str(get_file_info_sandbox(args.get("sandbox_id"), args.get("file_path")))
        elif name == "read_file_sandbox":
            return str(read_file_sandbox(args.get("sandbox_id"), args.get("file_name"), args.get("model_name")))
        elif name == "save_file_sandbox":
            return str(save_file_sandbox(args.get("sandbox_id"), args.get("file_name"), args.get("content")))
        elif name == "append_file_sandbox":
            return str(append_file_sandbox(args.get("sandbox_id"), args.get("file_name"), args.get("content")))
        elif name == "delete_this_file_sandbox":
            return str(delete_this_file_sandbox(args.get("sandbox_id"), args.get("file_name")))
        elif name == "edit_file_sandbox":
            return str(edit_file_sandbox(
                args.get("sandbox_id"), 
                args.get("file_path"), 
                args.get("edits"), 
                args.get("dry_run", False), 
                args.get("options")
            ))
        else:
            return f"Error: Tool {name} not found."
    except Exception as e:
        return f"Error executing tool {name}: {e}"
