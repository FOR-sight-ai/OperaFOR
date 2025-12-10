import os
import threading
import traceback
from typing import Any, Dict, List, Optional, Union
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
import webview
import logging
import uuid
import json
import sys
from datetime import datetime
import subprocess
import asyncio
import http.client
import urllib.parse
import io
import base64
# import fitz # Lazy loaded


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s",
    handlers=[
        logging.FileHandler("app.log"), 
        logging.StreamHandler()
    ]
)

# Set up file handler to capture all debug info and errors
file_handler = logging.FileHandler("app.log")
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] [%(threadName)s] %(name)s: %(message)s")
file_handler.setFormatter(file_formatter)

def get_config_dir():
    """Returns a persistent application folder compatible with all OS."""
    if os.name == "nt":  # Windows
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
        return os.path.join(base, "operafor")
    else:  # macOS, Linux, others
        return os.path.join(os.path.expanduser("~"), ".operafor")

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


# --- git utils  ---

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


# --- Helper function for sandbox paths ---
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

def is_vlm(model_name: str) -> bool:
    """Check if the model has vision capabilities."""
    if not model_name:
        return False
    model_lower = model_name.lower()
    vision_keywords = ["vision", "4o", "claude-3", "gemini", "sonnet", "opus", "pixtral", "llavanext"]
    return any(keyword in model_lower for keyword in vision_keywords)



# --- Tool Definitions ---

def list_sandbox_files(sandbox_id: str) -> List[str]:
    """ List all files in the sandbox directory."""
    sandbox_path = get_sandbox_path(sandbox_id)
    output_files = []
    if not os.path.exists(sandbox_path):
         return ["Sandbox directory does not exist yet."]
    for root, _, files in os.walk(sandbox_path):
        for file in files:
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, sandbox_path)
            if '.git' in rel_path or rel_path.startswith('.git/') or rel_path == 'conversation.json':
                continue
            if os.path.isfile(file_path):
                output_files.append(rel_path)
    if len(output_files) == 0:
        return ["No files found in this sandbox."]
    return output_files

def read_file_sandbox(sandbox_id: str, file_name:str, model_name: str = None) -> str:
    """ Read a file from the sandbox directory."""
    sandbox_path = get_sandbox_path(sandbox_id)
    file_path = os.path.join(sandbox_path, file_name)
    if not os.path.exists(file_path):
        return f"Error: File {file_name} not found"
    
    if file_name.lower().endswith(".pdf"):
        import fitz # PyMuPDF
        if is_vlm(model_name):
            try:
                doc = fitz.open(file_path)
                images = []
                for db in doc: # iterate through pages
                    pix = db.get_pixmap()
                    img_data = pix.tobytes("png")
                    b64_img = base64.b64encode(img_data).decode("utf-8")
                    images.append(b64_img)
                doc.close()
                return json.dumps({"__type__": "image", "images": images})
            except Exception as e:
                 return f"Error reading PDF file: {e}"
        else:
            try:
                doc = fitz.open(file_path)
                text_content = []
                for i, page in enumerate(doc):
                    text = page.get_text()
                    text_content.append(f"--- Page {i+1} ---\n{text}")
                doc.close()
                return "\n".join(text_content) if text_content else "PDF is empty or contains no extractable text."
            except Exception as e:
                return f"Error reading PDF file: {e}"

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
    # ... logic copied from original ...
    import difflib
    import re

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
            # We don't normalize new_text for insertion, but maybe for comparison
            # Actually original code normalized both for exact match check
            # For simplicity let's stick to simple find if normalize_ws is bad
            pass 

        # Simplified match logic compared to original to fit in one file cleaner
        # Original logic was quite complex about whitespace. 
        # Let's try basic replace first.
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

# Tool Registry
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

def execute_tool(name: str, args: Dict[str, Any]) -> str:
    """Execute a tool by name with arguments."""
    try:
        if name == "list_sandbox_files":
            return str(list_sandbox_files(args.get("sandbox_id")))
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

# --- FastAPI App ---
app = FastAPI()
app.mount("/static", StaticFiles(directory=os.path.dirname(os.path.abspath(__file__))), name="static")

DEFAULT_CONFIG = {
    "llm": {
        "endpoint": "https://openrouter.ai/api/v1",
        "model": "deepseek/deepseek-chat",
        "apiKey": "your_api_key"
    }
}

async def simple_llm_call(messages: List[Dict], tools: List[Dict], config: Dict) -> Dict:
    """Make a call to the LLM."""
    endpoint = config.get("llm", {}).get("endpoint", "https://openrouter.ai/api/v1")
    api_key = config.get("llm", {}).get("apiKey")
    model = config.get("llm", {}).get("model", "deepseek/deepseek-chat")
    
    # Parse URL
    parsed = urllib.parse.urlparse(endpoint)
    host = parsed.netloc
    path = parsed.path.rstrip('/') + "/chat/completions"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "stream": True # We will handle streaming support or just simple wait?
                       # Task said "make a simple succesion of calls to an LLM"
                       # But existing code used streaming response. 
                       # Let's keep stream=True for UX but we need to handle it.
                       # For the simple initial implementation, let's use stream=False in the internal call logic 
                       # OR handle SSE. Since I need to return an iterator for the runAgent, handling SSE is better.
        # However, to minimize complexity for the logic loop (call -> tool -> call), 
        # it's easier to use stream=False for the "thinking" part or just handle full response.
        # But user wants to see progress.
        # Let's use stream=True and yield chunks.
    }
    
    # Actually, using standard library for SSE is painful. 
    # I'll use stream=False and simulated streaming or just wait.
    # The requirement is "simple succession of calls".
    # User said "minimize footprint".
    # I'll implement a non-streaming generator for simplicity of implementation first, 
    # but yield the chunks if I can.
    
    # Let's try non-streaming first for robustness of the loop, 
    # and maybe just yield the final text. 
    # BUT the runAgent yield logic is expected by frontend.
    
    # Let's do stream=False for tool calls and stream=True for final answer? 
    # No, we don't know if it's final.
    
    # Let's use `requests` library since I added it to pyproject.toml?
    # No I added `requests` in the previous step? 
    # Wait, I added `requests` in the `replace_file_content` call. Yes.
    # So I can use `requests`.
    
    pass 

# Retrying `simple_llm_call` with `requests`

def call_llm(messages, tools, config):
    import time
    import requests
    endpoint = config.get("llm", {}).get("endpoint", "https://openrouter.ai/api/v1")
    if not endpoint.endswith("/chat/completions"):
        endpoint = endpoint.rstrip("/") + "/chat/completions"
    
    api_key = config.get("llm", {}).get("apiKey")
    model = config.get("llm", {}).get("model")
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # Some providers 400 if tools is empty list
    payload_tools = tools if tools else None
    
    data = {
        "model": model,
        "messages": messages,
        "tools": payload_tools,
        "stream": False 
    }
    if not payload_tools:
        del data["tools"]
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"LLM Call Attempt {attempt+1}/{max_retries}...")
            response = requests.post(endpoint, json=data, headers=headers, timeout=60)
            
            if response.status_code == 400:
                print(f"400 Bad Request Details: {response.text}")
                return {"error": f"400 Bad Request: {response.text}"}
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"LLM Call Error (Attempt {attempt+1}): {e}")
            if getattr(e, 'response', None):
                 print(f"Error Response Body: {e.response.text}")

            if attempt < max_retries - 1:
                sleep_time = 2 ** attempt
                print(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                 return {"error": str(e)}
    
    return {"error": "Max retries exceeded"}

async def runAgent(sandbox_id):
    """Run the agent loop."""
    # Load Config
    if not os.path.exists(CONFIG_PATH):
        config = DEFAULT_CONFIG
    else:
        with open(CONFIG_PATH, "r") as f:
            config = json.load(f)

    # Load Conversation
    convs = load_all_sandboxes()
    conv = convs.get(sandbox_id)
    if not conv:
        yield "Error: Sandbox not found"
        return

    messages = conv.get("messages", [])
    
    # --- PERSISTENCE: Save a pending assistant message ---
    # This ensures "Working..." is shown even after reload
    pending_msg = {"role": "assistant", "content": "", "status": "pending"}
    # We append it to the stored conversation
    messages.append(pending_msg)
    conv["messages"] = messages
    convs[sandbox_id] = conv
    save_all_sandboxes(convs)
    
    # Ensure messages are in OpenAI format
    openai_messages = []
    # Note: We iterate over messages but SKIP the last one (which is our pending placeholder)
    # for the LLM context, because we don't want to feed an empty assistant message to the LLM.
    for m in messages[:-1]:
        role = m.get("role")
        content = m.get("content")
        # Map roles if needed, but they should be standard
        if role not in ["user", "assistant", "system", "tool"]:
             role = "user" # default
        
        msg_obj = {"role": role, "content": content}
        if "tool_calls" in m:
            msg_obj["tool_calls"] = m["tool_calls"]
        if "tool_call_id" in m:
            msg_obj["tool_call_id"] = m["tool_call_id"]
        if "name" in m:
             msg_obj["name"] = m["name"]
            
        openai_messages.append(msg_obj)
        
    # We want to respond to the last message
    # Determine available tools and system prompt based on read-only status
    read_only = conv.get("read_only", False)
    
    current_tools = TOOL_DEFINITIONS
    if read_only:
        # Filter out writing tools
        read_only_tool_names = ["list_sandbox_files", "read_file_sandbox"]
        current_tools = [t for t in TOOL_DEFINITIONS if t["function"]["name"] in read_only_tool_names]
        system_prompt = f"You are a coding assistant. You have access to a sandbox environment with ID {sandbox_id}. This sandbox is pending READ-ONLY mode. You can ONLY read files. You CANNOT write, edit, or delete files. Use the provided tools."
    else:
        system_prompt = f"You are a coding assistant. You have access to a sandbox environment with ID {sandbox_id}. You can read, write, edit files. Prefer editing files over overwriting them. Use the provided tools."
    
    openai_messages.insert(0, {"role": "system", "content": system_prompt})
    
    # Agent Loop
    max_turns = 10
    current_turn = 0
    
    # We need to track new interactions to save them later
    # Initial openai_messages has system + history
    initial_openai_count = len(openai_messages)
    
    agent_success = False
    
    try:
        while current_turn < max_turns:
            current_turn += 1
            
            # Notify user
            # yield json.dumps({"type": "status", "data": f"Thinking (Turn {current_turn})..."}) + "\n"
            
            response_data = call_llm(openai_messages, current_tools, config)
            
            if "error" in response_data:
                yield json.dumps({"type": "error", "data": f"Error from LLM: {response_data['error']}"}) + "\n"
                # Add to history as error
                openai_messages.append({"role": "assistant", "content": f"Error from LLM: {response_data['error']}"})
                break
                
            choice = response_data.get("choices", [{}])[0]
            message = choice.get("message", {})
            content = message.get("content")
            tool_calls = message.get("tool_calls")
            
            # Append assistant message to history used for next turn
            openai_messages.append(message)
            
            # Yield content if any
            if content:
                # If we have thoughts mixed with content (e.g. CoT), we might want to separate them later. 
                # For now, let's assume content is the "thought" or "answer".
                # Unless we have tool calls, everything is "content" or "thought".
                # But typically modern models put CoT in content.
                # We'll emit it as "content" for now, frontend can render it.
                # Actually, if there are tool calls, the content is often the "thought" explaining why.
                # Let's just send "content" and let frontend decide or just display it.
                # IMPROVEMENT: send as "thought" if tool_calls is not empty?
                msg_type = "thought" if tool_calls else "content"
                yield json.dumps({"type": msg_type, "data": content}) + "\n"
                
            if tool_calls:
                # yield json.dumps({"type": "status", "data": f"Executing {len(tool_calls)} tools..."}) + "\n"
                for tc in tool_calls:
                    func_name = tc["function"]["name"]
                    args_str = tc["function"]["arguments"]
                    call_id = tc["id"]
                    
                    # Emit tool call event
                    yield json.dumps({
                        "type": "tool_call", 
                        "data": {
                            "name": func_name, 
                            "arguments": args_str,
                            "id": call_id
                        }
                    }) + "\n"
                    
                    try:
                        args = json.loads(args_str)
                        args["sandbox_id"] = sandbox_id # Inject sandbox_id
                        args["model_name"] = config.get("llm", {}).get("model") # Inject model_name
                        
                        result = execute_tool(func_name, args)
                        
                        # Emit tool result event
                        # Check for images in result
                        content_payload = result
                        is_image = False
                        try:
                            if result.strip().startswith('{"__type__": "image"'):
                                data = json.loads(result)
                                if data.get("__type__") == "image":
                                    is_image = True
                                    images = data.get("images", [])
                                    content_payload = [{"type": "text", "text": "PDF Content (rendered as images):"}]
                                    for img in images:
                                        content_payload.append({
                                            "type": "image_url",
                                            "image_url": {"url": f"data:image/png;base64,{img}"}
                                        })
                        except Exception as e:
                            print(f"Error parsing image result: {e}")
                            pass
                        
                        yield json.dumps({
                            "type": "tool_result",
                            "data": {
                                "id": call_id,
                                "name": func_name,
                                "result": "PDF Images (hidden)" if is_image else result
                            }
                        }) + "\n"

                    except Exception as e:
                        result = f"Error processing arguments: {e}"
                        content_payload = result
                        yield json.dumps({
                            "type": "tool_result",
                            "data": {
                                "id": call_id,
                                "name": func_name,
                                "result": result,
                                "is_error": True
                            }
                        }) + "\n"
                    
                    # Append tool result
                    openai_messages.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": content_payload
                    })
            else:
                 # No tool calls, we are done
                 agent_success = True
                 break
        
    except Exception as e:
        # Yield error to user
        yield json.dumps({"type": "error", "data": f"Error during execution: {str(e)}"}) + "\n"
        # We will handle persistence in finally
        
    finally:
        # --- FINALLY: Update the pending message ---
        # This runs whether success, error, or cancelled (client disconnect)
        try:
            convs = load_all_sandboxes()
            conv = convs.get(sandbox_id)
            if conv:
                messages = conv.get("messages", [])
                # Remove the pending message we added at the start
                if messages and messages[-1].get("status") == "pending":
                    messages.pop()
                
                # Use agent_success flag to determine status
                final_status = "done" if agent_success else "error"
                
                # If we failed (or cancelled), we might want to also append what we have so far?
                # For simplicity, if success: append new messages with "done"
                # If failed: append new messages with "done" (for partials) and maybe an error message at end?
                # Or just mark all new interactions as "done" and append error if exception caught?
                
                # Let's simplify:
                # 1. Recover any new messages generated (partially or fully)
                # 2. Append them
                # 3. If failed/cancelled, append an extra error message
                
                new_msgs = openai_messages[initial_openai_count:]
                for msg in new_msgs:
                     msg["status"] = "done"
                     messages.append(msg)
                
                if not agent_success:
                     # Check if we already have an error message?
                     # The try/except block might have yielded one but maybe not added to openai_messages?
                     # If it was a disconnect, we won't have an error message in openai_messages.
                     messages.append({"role": "assistant", "content": "Generation interrupted or failed.", "status": "error"})
                
                # --- FIX: Update the User message status ---
                # The frontend sets the triggering user message to 'pending'. We must mark it as done/error.
                # We look for the *last* user message.
                for i in range(len(messages) - 1, -1, -1):
                    if messages[i].get("role") == "user":
                        # If it is pending, close it.
                        if messages[i].get("status") == "pending":
                             messages[i]["status"] = "done" if agent_success else "error"
                        # We only need to update the last one that triggered this run
                        break

                conv["messages"] = messages
                convs[sandbox_id] = conv
                save_all_sandboxes(convs)
                
                # Git Commit
                last_user_msg = "Update"
                for m in reversed(messages):
                    if m.get("role") == "user":
                        last_user_msg = m.get("content", "Update")
                        break
                
                commit_msg = f"Agent update: {last_user_msg[:30]}..."
                
                sandbox_path = get_sandbox_path(sandbox_id)
                commit_sandbox_changes(sandbox_path, conv["messages"], commit_msg)
        except Exception as e:
            print(f"Critical error saving conversation state: {e}")
            
    yield "" # Close stream

@app.post("/agent")
async def run_agent(request: Request):
    data = await request.json()
    return StreamingResponse(runAgent(data.get("sandbox_id")), media_type="text/plain")

@app.get("/")
async def serve_index():
    try:
        if hasattr(sys, '_MEIPASS'):
            index_path = os.path.join(sys._MEIPASS, "index.html")
        else:
            index_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
        with open(index_path, "rb") as f:
            content = f.read()
        return Response(content, media_type="text/html")
    except Exception as e:
        return Response(f"Error loading interface: {e}", status_code=500)

@app.get("/config.json")
async def get_config():
    if not os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "w") as f:
            json.dump(DEFAULT_CONFIG, f, indent=2)
        return JSONResponse(content=DEFAULT_CONFIG)
    with open(CONFIG_PATH, "r") as f:
        return JSONResponse(content=json.load(f))

@app.post("/config.json")
async def set_config(request: Request):
    data = await request.json()
    with open(CONFIG_PATH, "w") as f:
        json.dump(data, f, indent=2)
    return {"status": "ok"}

@app.get("/sandboxes")
async def api_list_sandboxes():
    convs = load_all_sandboxes()
    return [ {"id": cid, "title": c.get("title", f"Sandbox {cid}"), "read_only": c.get("read_only", False)} for cid, c in convs.items() ]

@app.get("/sandboxes/{conv_id}")
async def api_get_sandbox(conv_id: str):
    convs = load_all_sandboxes()
    conv = convs.get(conv_id)
    if conv is None:
        return JSONResponse(status_code=404, content={"error": "Not found"})
    return conv

@app.post("/sandboxes")
async def api_create_sandbox(request: Request):
    data = await request.json()
    conv_id = str(uuid.uuid4())
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    title = data.get("title") or f"{now}"
    read_only = data.get("read_only", False)
    messages = data.get("messages")
    source_id = data.get("source_id")

    if not isinstance(messages, list):
        messages = []

    # Handle file copying for forks
    if source_id:
        convs = load_all_sandboxes()
        if source_id in convs:
            source_path = get_sandbox_path(source_id)
            if os.path.exists(source_path):
                import shutil
                target_path = os.path.join(SANDBOXES_DIR, conv_id)
                try:
                    # Create target directory and copy files
                    shutil.copytree(source_path, target_path, ignore=shutil.ignore_patterns('.git', 'conversation.json'))
                except Exception as e:
                    print(f"Error copying files during fork: {e}")

    conv = { "id": conv_id, "title": title, "read_only": read_only, "messages": messages }
    convs = load_all_sandboxes()
    convs[conv_id] = conv
    save_all_sandboxes(convs)
    sandbox_path = get_sandbox_path(conv_id)
    init_or_get_repo(sandbox_path)
    return conv

@app.post("/sandboxes/{conv_id}")
async def api_add_message(conv_id: str, request: Request):
    data = await request.json()
    convs = load_all_sandboxes()
    conv = convs.get(conv_id)
    if conv is None:
        return JSONResponse(status_code=404, content={"error": "Not found"})
    conv.setdefault("messages", []).append(data)
    convs[conv_id] = conv
    save_all_sandboxes(convs)
    return {"status": "ok"}

@app.delete("/sandboxes/{conv_id}")
async def api_delete_sandbox(conv_id: str, delete_folder: bool = True):
    convs = load_all_sandboxes()
    if conv_id not in convs:
        return JSONResponse(status_code=404, content={"error": "Not found"})
    sandbox_path = get_sandbox_path(conv_id)
    del convs[conv_id]
    save_all_sandboxes(convs)
    if delete_folder:
        import shutil
        if os.path.exists(sandbox_path):
            shutil.rmtree(sandbox_path)
    return {"status": "deleted", "folder_deleted": delete_folder}

@app.patch("/sandboxes/{conv_id}")
async def api_patch_sandbox(conv_id: str, request: Request):
    data = await request.json()
    convs = load_all_sandboxes()
    conv = convs.get(conv_id)
    if conv is None:
        return JSONResponse(status_code=404, content={"error": "Not found"})
    if "title" in data:
        conv["title"] = data["title"]
    if "read_only" in data:
        conv["read_only"] = data["read_only"]
    update_commit = False
    if "messages" in data:
        old_messages = conv.get("messages", [])
        new_messages = data["messages"]
        conv["messages"] = new_messages
        if len(new_messages) < len(old_messages):
            update_commit = True
    convs[conv_id] = conv
    save_all_sandboxes(convs)
    if update_commit:
        commits = conv.get("commits", [])
        target_step = len(new_messages) - 1
        target_commit = next((c for c in commits if c["step"] == target_step), None)
        if target_commit:
            sandbox_path = get_sandbox_path(conv_id)
            revert_sandbox_to_commit(sandbox_path, target_commit["hash"])
    return {"status": "ok"}

@app.get("/sandboxes/{conv_id}/commits")
async def api_get_sandbox_commits(conv_id: str):
    convs = load_all_sandboxes()
    conv = convs.get(conv_id)
    if conv is None:
        return JSONResponse(status_code=404, content={"error": "Sandbox not found"})
    return {"commits": conv.get("commits", [])}

@app.post("/sandboxes/{conv_id}/revert")
async def api_revert_sandbox(conv_id: str, request: Request):
    data = await request.json()
    commit_hash = data.get("commit_hash")
    step = data.get("step")
    if not commit_hash and step is None:
        return JSONResponse(status_code=400, content={"error": "Either commit_hash or step must be provided"})
    convs = load_all_sandboxes()
    conv = convs.get(conv_id)
    if conv is None:
        return JSONResponse(status_code=404, content={"error": "Sandbox not found"})
    if step is not None and commit_hash is None:
        commits = conv.get("commits", [])
        target_commit = next((c for c in commits if c["step"] == step), None)
        if not target_commit:
            return JSONResponse(status_code=404, content={"error": "Commit for step not found"})
        commit_hash = target_commit["hash"]
    sandbox_path = get_sandbox_path(conv_id)
    if not os.path.exists(sandbox_path):
        return JSONResponse(status_code=404, content={"error": "Sandbox folder not found"})
    success = revert_sandbox_to_commit(sandbox_path, commit_hash)
    if success:
        if step is not None:
            conv["messages"] = conv["messages"][:step+1]
            convs[conv_id] = conv
            save_all_sandboxes(convs)
        return {"status": "reverted", "commit_hash": commit_hash}
    else:
        return JSONResponse(status_code=500, content={"error": "Failed to revert sandbox"})

@app.post("/sandboxes/{conv_id}/change_folder")
async def api_change_sandbox_folder(conv_id: str, request: Request):
    data = await request.json()
    new_path = data.get("path", "").strip()
    if not new_path:
        return JSONResponse(status_code=400, content={"error": "Path is required"})
    if not os.path.exists(new_path):
        try:
            os.makedirs(new_path, exist_ok=True)
        except Exception as e:
            return JSONResponse(status_code=400, content={"error": f"Cannot create directory: {str(e)}"})
    if not os.path.isdir(new_path):
        return JSONResponse(status_code=400, content={"error": "Path must be a directory"})
    convs = load_all_sandboxes()
    conv = convs.get(conv_id)
    if conv is None:
        return JSONResponse(status_code=404, content={"error": "Sandbox not found"})
    old_path = get_sandbox_path(conv_id)
    if old_path != new_path and os.path.exists(old_path) and os.listdir(old_path):
        copy_files = data.get("copy_files", True)
        if copy_files:
            import shutil
            try:
                for item in os.listdir(old_path):
                    s = os.path.join(old_path, item)
                    d = os.path.join(new_path, item)
                    if os.path.isdir(s):
                        if os.path.exists(d): shutil.rmtree(d)
                        shutil.copytree(s, d)
                    else:
                        shutil.copy2(s, d)
            except Exception as e:
                return JSONResponse(status_code=500, content={"error": f"Failed to copy files: {str(e)}"})
    conv["custom_path"] = new_path
    convs[conv_id] = conv
    save_all_sandboxes(convs)
    init_or_get_repo(new_path)
    return {"status": "ok", "new_path": new_path}

@app.post("/open_sandbox_folder/{sandbox_id}")
async def open_sandbox_folder(sandbox_id: str):
    sandbox_path = get_sandbox_path(sandbox_id)
    if not os.path.exists(sandbox_path):
        os.makedirs(sandbox_path)
    try:
        if sys.platform.startswith("darwin"):
            subprocess.Popen(["open", sandbox_path])
        elif sys.platform.startswith("win"):
            os.startfile(sandbox_path)
        else:
            subprocess.Popen(["xdg-open", sandbox_path])
        return {"status": "ok"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

def run_fastapi():
    port = int(os.getenv("PORT", "9001"))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info", reload=False)

def run_webview():
    webview.create_window("OperaFOR", url=f"http://localhost:{os.getenv('PORT', '9001')}")
    webview.start()

if __name__ == "__main__":
    fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)
    fastapi_thread.start()
    run_webview()
