import os
import threading
import traceback
from typing import Any, Dict, List
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from smolagents import MCPClient, CodeAgent
from mcp import StdioServerParameters
from fastmcp import FastMCP
import webview
import logging
import uuid
import json
import sys
from datetime import datetime
import subprocess
import asyncio
import threading
import logging
import http.client

# HTTP debugging disabled by default (can be enabled for troubleshooting)
# http.client.HTTPConnection.debuglevel = 1

# Configure logging - INFO level for console, DEBUG for file to capture all errors
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s",
    handlers=[
        logging.FileHandler("app.log"), 
        logging.StreamHandler()          # Show INFO and above in console
    ]
)

# Set up file handler to capture all debug info and errors
file_handler = logging.FileHandler("app.log")
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] [%(threadName)s] %(name)s: %(message)s")
file_handler.setFormatter(file_formatter)

# Disable verbose network logging by default but ensure errors are captured
requests_log = logging.getLogger("requests.packages.urllib3")
requests_log.setLevel(logging.WARNING)  # Only show warnings and errors
requests_log.addHandler(file_handler)
requests_log.propagate = True

# --- FastMCP Server Definition ---
mcp = FastMCP("OperaFOR")


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

import os
import json
from datetime import datetime
from dulwich import porcelain
from dulwich.repo import Repo

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
    """Revert the sandbox to a specific commit and update conversation.json and sandboxes.json accordingly.
    Args:
        sandbox_path (str): Path to the sandbox folder
        commit_hash (str): The commit hash to revert to
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Use porcelain reset which properly handles file checkout
        porcelain.reset(sandbox_path, "hard", commit_hash)

        # Update sandboxes.json: update commits for this sandbox
        from pathlib import Path
        with open(CONV_FILE, 'r') as f:
            sandboxes = json.load(f)
        # Find the sandbox id from the path
        sandbox_id = os.path.basename(sandbox_path)
        conv = sandboxes.get(sandbox_id)
        if conv is not None:
            # Truncate commits up to and including the reverted commit
            commits = conv.get("commits", [])
            idx = next((i for i, c in enumerate(commits) if c["hash"] == commit_hash), None)
            print(f"Reverting to commit {commit_hash} for sandbox {sandbox_id}, found at index {idx}")
            if idx is not None:
                conv["commits"] = commits[:idx+1]
            sandboxes[sandbox_id] = conv
            with open(CONV_FILE, 'w') as f:
                json.dump(sandboxes, f, indent=2)
        return True
    except Exception as e:
        print(f"Error reverting to commit {commit_hash}: {e}")
        return False



@mcp.tool()
def list_sandbox_files(sandbox_id: str) -> List[str]:
    """ List all files in the sandbox directory.
    Args:
        sandbox_id (str): The ID of the sandbox.
    Returns:
        List[str]: A list containing the paths of files in the sandbox.
    """
    # return the content of the output folder
    sandbox_path = os.path.join(SANDBOXES_DIR, sandbox_id)
    output_files = []
    for root, _, files in os.walk(sandbox_path):
        for file in files:
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, sandbox_path)

            # ignore path containing .git directory and conversation.json
            if '.git' in rel_path or rel_path.startswith('.git/') or rel_path == 'conversation.json':
                continue

            # Ensure we only return files, not directories
            if os.path.isfile(file_path):
                output_files.append(rel_path)
    if len(output_files) == 0:
        output_files.append("No files found in this sandbox.")
    print(f"Files in sandbox {sandbox_id}: {output_files}")
    return output_files


@mcp.tool()
def read_file_sandbox(sandbox_id: str, file_name:str) -> str:
    """ Read a file from the sandbox directory.
    Args:
        sandbox_id (str): The ID of the sandbox.
        file_name (str): The name of the file to read.
    Returns:
        str: The content of the file.
    """
    sandbox_path = os.path.join(SANDBOXES_DIR, sandbox_id)
    file_path = os.path.join(sandbox_path, file_name)
    
    if not os.path.exists(file_path):
        return {"error": "File not found"}

    with open(file_path, 'r') as f:
        content = f.read()
    
    return content


@mcp.tool()
def save_file_sandbox(sandbox_id: str, file_name: str, content: str) -> bool:
    """Write content to a file in the sandbox.

    Args:
        sandbox_id: The ID of the sandbox.
        file_name: The name of the file to write to.
        content: Content to write to the file

    Returns:
        True if the file was written successfully
    """
    sandbox_path = os.path.join(SANDBOXES_DIR, sandbox_id)
    file_path = os.path.join(sandbox_path, file_name)

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        f.write(content)

    return True


@mcp.tool()
def append_file_sandbox(sandbox_id: str, file_name: str, content: str) -> bool:
    """Append content to a file in the sandbox.

    Args:
        sandbox_id: The ID of the sandbox.
        file_name: The name of the file to append to.
        content: Content to append to the file.

    Returns:
        True if the file was written successfully
    """
    sandbox_path = os.path.join(SANDBOXES_DIR, sandbox_id)
    file_path = os.path.join(sandbox_path, file_name)

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'a') as f:
        f.write(content)

    return True


@mcp.tool()
def delete_this_file_sandbox(sandbox_id: str, file_name: str) -> bool:
    """Delete a file in the sandbox.

    Args:
        sandbox_id: The ID of the sandbox.
        file_name: The name of the file to delete.

    Returns:
        True if the file was deleted successfully
    """
    sandbox_path = os.path.join(SANDBOXES_DIR, sandbox_id)
    file_path = os.path.join(sandbox_path, file_name)

    try:
        os.remove(file_path)
        return True
    except Exception as e:
        print(f"Error deleting file {file_path}: {e}")
        return False

@mcp.tool()
def edit_file_sandbox(
    sandbox_id: str,
    file_path: str,
    edits: List[Dict[str, str]],
    dry_run: bool = False,
    options: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Make selective edits to files while preserving formatting.

    Features:
        - Line-based and multi-line content matching
        - Whitespace normalization with indentation preservation
        - Multiple simultaneous edits with correct positioning
        - Smart detection of already-applied edits
        - Git-style diff output with context
        - Preview changes with dry run mode

    Args:
        sandbox_id: ID of the sandbox to edit
        file_path: Path to the file to edit (relative to project directory)
        edits: List of edit operations (each containing old_text and new_text)
        dry_run: Preview changes without applying (default: False)
        options: Optional formatting settings
                    - preserve_indentation: Keep existing indentation (default: True)
                    - normalize_whitespace: Normalize spaces (default: True)

    Returns:
        Detailed diff and match information including success status
    """
    import difflib
    import re

    # --- Utilitaires internes ---
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
        diff_lines = difflib.unified_diff(
            original_lines,
            modified_lines,
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
            lineterm="",
        )
        return "".join(diff_lines)

    def find_exact_match(content: str, pattern: str):
        if pattern in content:
            lines_before = content[: content.find(pattern)].count("\n")
            line_count = pattern.count("\n") + 1
            return True, lines_before, line_count
        return False, -1, 0

    # --- Début de la logique principale ---
    import os
    sandbox_path = os.path.join(SANDBOXES_DIR, sandbox_id)
    full_file_path = os.path.join(sandbox_path, file_path)

    # Validation des paramètres
    if not file_path or not isinstance(file_path, str):
        return {"success": False, "error": f"File path must be a non-empty string, got {type(file_path)}"}
    if not isinstance(edits, list) or not edits:
        return {"success": False, "error": "Edits must be a non-empty list"}
    if not os.path.isfile(full_file_path):
        return {"success": False, "error": f"File not found: {file_path}"}

    # Normalisation des edits
    normalized_edits = []
    for i, edit in enumerate(edits):
        if not isinstance(edit, dict):
            return {"success": False, "error": f"Edit #{i} must be a dictionary, got {type(edit)}"}
        if "old_text" not in edit or "new_text" not in edit:
            missing = ", ".join([f for f in ["old_text", "new_text"] if f not in edit])
            return {"success": False, "error": f"Edit #{i} is missing required field(s): {missing}"}
        normalized_edits.append({"old_text": edit["old_text"], "new_text": edit["new_text"]})

    # Options
    preserve_indent = options.get("preserve_indentation", True) if options else True
    normalize_ws = options.get("normalize_whitespace", True) if options else True

    # Lecture du contenu original
    try:
        with open(full_file_path, "r", encoding="utf-8") as f:
            original_content = f.read()
    except Exception as e:
        return {"success": False, "error": f"Error reading file: {str(e)}"}

    # Application des edits
    match_results = []
    changes_made = False
    modified_content = normalize_line_endings(original_content)
    for i, edit in enumerate(normalized_edits):
        old = normalize_line_endings(edit["old_text"])
        new = normalize_line_endings(edit["new_text"])
        if normalize_ws:
            old = normalize_whitespace(old)
            new = normalize_whitespace(new)
            mod_content_ws = normalize_whitespace(modified_content)
        else:
            mod_content_ws = modified_content
        # Si déjà appliqué
        if new in mod_content_ws and old not in mod_content_ws:
            match_results.append({
                "edit_index": i,
                "match_type": "skipped",
                "details": "Edit already applied - content already in desired state",
            })
            continue
        # Recherche exacte
        found, line_index, line_count = find_exact_match(modified_content, old)
        if found:
            # Préservation indentation
            if preserve_indent:
                new = preserve_indentation(old, new)
            start_pos = modified_content.find(old)
            end_pos = start_pos + len(old)
            modified_content = modified_content[:start_pos] + new + modified_content[end_pos:]
            changes_made = True
            match_results.append({
                "edit_index": i,
                "match_type": "exact",
                "line_index": line_index,
                "line_count": line_count,
            })
        else:
            match_results.append({
                "edit_index": i,
                "match_type": "failed",
                "details": "No exact match found",
            })
    failed_matches = [r for r in match_results if r.get("match_type") == "failed"]
    already_applied = [r for r in match_results if r.get("match_type") == "skipped" and "already applied" in r.get("details", "")]
    result = {
        "match_results": match_results,
        "file": file_path,
        "dry_run": dry_run,
    }
    if failed_matches:
        result.update({"success": False, "error": "Failed to find exact match for one or more edits"})
        return result
    if not changes_made or (already_applied and len(already_applied) == len(normalized_edits)):
        result.update({
            "success": True,
            "diff": "",
            "message": "No changes needed - content already in desired state",
        })
        return result
    diff = create_unified_diff(original_content, modified_content, file_path)
    result.update({"diff": diff, "success": True})
    if not dry_run and changes_made:
        try:
            with open(full_file_path, "w", encoding="utf-8") as f:
                f.write(modified_content)
        except Exception as e:
            result.update({"success": False, "error": f"Error writing to file: {str(e)}"})
            return result
    return result


# --- FastAPI App Definition ---
app = FastAPI()

# Add mount for static files
app.mount("/static", StaticFiles(directory=os.path.dirname(os.path.abspath(__file__))), name="static")

def load_all_sandboxes():
    if not os.path.exists(CONV_FILE):
        with open(CONV_FILE, 'w') as f:           
            first_sandbox = {
                "id": str(uuid.uuid4()),
                "title": "Welcome",
                "messages": [
                    {"role": "assistant", "content": "Welcome ! What can I do for you today? Don't forget to configure your API key in the configuration panel."}
                ]
            }
            json.dump({first_sandbox["id"]: first_sandbox}, f)
    with open(CONV_FILE, 'r') as f:
        return json.load(f)

def save_all_sandboxes(convs):
    with open(CONV_FILE, 'w') as f:
        json.dump(convs, f, indent=2)

async def runAgent(sandbox_id):
    """Run the agent with the provided data and stream results."""
    loop = asyncio.get_event_loop()
    queue = asyncio.Queue()

    def agent_worker():
        # load from config file
        with open(CONFIG_PATH, "r") as f:
            config = json.load(f)
        servers = config.get("servers", [])

        server_params = []
        for s in servers:
            if isinstance(s, dict) and s.get("type") == "http":
                if "url" in s:
                    server_params.append({"url": s["url"], "transport": "streamable-http"})
                elif "config" in s and "url" in s["config"]:
                    server_params.append({"url": s["config"]["url"], "transport": "streamable-http"})
            elif isinstance(s, dict) and s.get("type") == "stdio":
                server_params.append(StdioServerParameters(
                    command=s.get("command"),
                    args=s.get("args", []),
                    env=s.get("env", os.environ)
                ))
        if not server_params:
            server_params = [{"url": "http://localhost:9000/mcp", "transport": "streamable-http"}]
        model_name = config.get("llm", {}).get("model")
        api_key = config.get("llm", {}).get("apiKey") or config.get("llm", {}).get("api_key")
        endpoint = config.get("llm", {}).get("endpoint") or config.get("llm", {}).get("base_url")
        from smolagents import OpenAIServerModel, MessageRole
        model = OpenAIServerModel(
            model_id=model_name,
            api_base=endpoint,
            api_key=api_key,
            custom_role_conversions={
                MessageRole.ASSISTANT: "assistant",
                MessageRole.USER: "user",
                MessageRole.SYSTEM: "system",
            }
        )
        # find corresponding sandbox from sandbox json file
        convs = load_all_sandboxes()
        conv = convs.get(sandbox_id)

        messages = conv.get("messages", [])
        # find the last user message
        prompt = (messages[-1].get("content") if messages else "").strip()
        instructions = f"Whenever creating or editing files prefers to do it in the sandbox using the tools"
        try:
            response = ""
            with MCPClient(server_params) as tools:
                agent = CodeAgent(tools=tools, 
                                  instructions=instructions,
                                  model=model, 
                                  planning_interval = 3,
                                  add_base_tools=False)
                response = agent.run(prompt, reset = False, additional_args = {
                    "sandbox_id": sandbox_id,
                    "sandbox_path": os.path.join(SANDBOXES_DIR, sandbox_id),
                    "past_conversation_file": os.path.join(SANDBOXES_DIR, sandbox_id, "conversation.json"),})
                queue.put_nowait(response)
        except Exception as e:
            tb_str = traceback.format_exc()
            queue.put_nowait(f"\nError during agent run: {e}\n{tb_str}\n")
            response = ""

        # --- Sauvegarde de la conversation et commit git  ---
        new_message = {"role": "assistant", "content": response, "status": "done"}
        convs = load_all_sandboxes()
        conv = convs.get(sandbox_id)
        if conv is None:
            conv = {"id": sandbox_id, "messages": []}
        conv.setdefault("messages", []).append(new_message)
        sandbox_path = os.path.join(SANDBOXES_DIR, sandbox_id)
        os.makedirs(sandbox_path, exist_ok=True)
        all_messages = conv.get("messages", [])
        user_prompt = prompt if prompt else "Agent interaction"
        commit_message = f"Agent response to: {user_prompt[:50]}..."
        conversation_file = os.path.join(sandbox_path, "conversation.json")
        with open(conversation_file, "w") as f:
            json.dump(all_messages, f, indent=2)
        commit_hash = commit_sandbox_changes(sandbox_path, all_messages, commit_message)
        if commit_hash:
            conv.setdefault("commits", []).append({
                "step": len(all_messages) - 1,
                "hash": commit_hash,
                "message": commit_message,
                "timestamp": datetime.now().isoformat()
            })
        convs[sandbox_id] = conv
        save_all_sandboxes(convs)
        queue.put_nowait(None)  # Signal de fin

    # Lance l'agent dans un thread
    threading.Thread(target=agent_worker, daemon=True).start()

    # Stream les résultats au fur et à mesure
    while True:
        item = await queue.get()
        if item is None:
            break
        yield item

@app.post("/agent")
async def run_agent(request: Request):
    """Handle a streaming prompt with Agent.run()."""
    data = await request.json()
    return StreamingResponse(runAgent(data.get("sandbox_id")), media_type="text/plain")

@app.get("/")
async def serve_index():
    """Serve index.html at the root, compatible with PyInstaller."""
    try:
        # If executed via PyInstaller, __file__ is in a bundle
        if hasattr(sys, '_MEIPASS'):
            index_path = os.path.join(sys._MEIPASS, "index.html")
        else:
            index_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
        with open(index_path, "rb") as f:
            content = f.read()
        return Response(content, media_type="text/html")
    except Exception as e:
        return Response(f"Error loading interface: {e}", status_code=500)


DEFAULT_CONFIG = {
    "llm": {
        "endpoint": "https://openrouter.ai/api/v1",
        "model": "deepseek/deepseek-chat-v3-0324:free",
        "apiKey": "your_api_key"
    },
    "servers": [
        {
            "type": "http",
            "url": "http://localhost:9000/mcp"
        }
    ]
}

@app.get("/config.json")
async def get_config():
    if not os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "w") as f:
            import json
            json.dump(DEFAULT_CONFIG, f, indent=2)
        return JSONResponse(content=DEFAULT_CONFIG)
    with open(CONFIG_PATH, "r") as f:
        import json
        return JSONResponse(content=json.load(f))

@app.post("/config.json")
async def set_config(request: Request):
    data = await request.json()
    with open(CONFIG_PATH, "w") as f:
        import json
        json.dump(data, f, indent=2)
    return {"status": "ok"}

@app.get("/sandboxes")
async def api_list_sandboxes():
    convs = load_all_sandboxes()
    return [
        {"id": cid, "title": c.get("title", f"Sandbox {cid}")}
        for cid, c in convs.items()
    ]

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
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    title = data.get("title") or f"{now}"
    messages = data.get("messages")
    if not isinstance(messages, list):
        messages = []
    conv = {
        "id": conv_id,
        "title": title,
        "messages": messages
    }
    convs = load_all_sandboxes()
    convs[conv_id] = conv
    save_all_sandboxes(convs)
    sandbox_path = os.path.join(SANDBOXES_DIR, conv_id)
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
async def api_delete_sandbox(conv_id: str):
    convs = load_all_sandboxes()
    if conv_id not in convs:
        return JSONResponse(status_code=404, content={"error": "Not found"})
    del convs[conv_id]
    save_all_sandboxes(convs)
    # Suppression du dossier sandbox correspondant
    import shutil
    sandbox_path = os.path.join(SANDBOXES_DIR, conv_id)
    if os.path.exists(sandbox_path):
        shutil.rmtree(sandbox_path)
    return {"status": "deleted"}

@app.patch("/sandboxes/{conv_id}")
async def api_patch_sandbox(conv_id: str, request: Request):
    data = await request.json()
    convs = load_all_sandboxes()
    conv = convs.get(conv_id)
    if conv is None:
        return JSONResponse(status_code=404, content={"error": "Not found"})
    # Allows modification of title and messages
    if "title" in data:
        conv["title"] = data["title"]
    update_commit = False
    if "messages" in data:
        old_messages = conv.get("messages", [])
        new_messages = data["messages"]
        conv["messages"] = new_messages
        # Si des messages ont été supprimés, revert le sandbox au commit correspondant
        if len(new_messages) < len(old_messages):
            update_commit = True

    convs[conv_id] = conv
    save_all_sandboxes(convs)

    if update_commit:
        commits = conv.get("commits", [])
        # On cherche le commit dont le step correspond au dernier message restant
        target_step = len(new_messages) - 1
        target_commit = next((c for c in commits if c["step"] == target_step), None)
        if target_commit:
            sandbox_path = os.path.join(SANDBOXES_DIR, conv_id)
            revert_sandbox_to_commit(sandbox_path, target_commit["hash"])
    return {"status": "ok"}

@app.get("/sandboxes/{conv_id}/commits")
async def api_get_sandbox_commits(conv_id: str):
    """Get the commit history for a sandbox."""
    convs = load_all_sandboxes()
    conv = convs.get(conv_id)
    if conv is None:
        return JSONResponse(status_code=404, content={"error": "Sandbox not found"})
    
    commits = conv.get("commits", [])
    return {"commits": commits}

@app.post("/sandboxes/{conv_id}/revert")
async def api_revert_sandbox(conv_id: str, request: Request):
    """Revert a sandbox to a specific commit."""
    data = await request.json()
    commit_hash = data.get("commit_hash")
    step = data.get("step")
    
    if not commit_hash and step is None:
        return JSONResponse(status_code=400, content={"error": "Either commit_hash or step must be provided"})
    
    convs = load_all_sandboxes()
    conv = convs.get(conv_id)
    if conv is None:
        return JSONResponse(status_code=404, content={"error": "Sandbox not found"})
    
    # If step is provided, find the corresponding commit hash
    if step is not None and commit_hash is None:
        commits = conv.get("commits", [])
        target_commit = next((c for c in commits if c["step"] == step), None)
        if not target_commit:
            return JSONResponse(status_code=404, content={"error": "Commit for step not found"})
        commit_hash = target_commit["hash"]
    
    # Revert the sandbox
    sandbox_path = os.path.join(SANDBOXES_DIR, conv_id)
    if not os.path.exists(sandbox_path):
        return JSONResponse(status_code=404, content={"error": "Sandbox folder not found"})
    
    success = revert_sandbox_to_commit(sandbox_path, commit_hash)
    if success:
        # Also update the messages to match the reverted state
        if step is not None:
            conv["messages"] = conv["messages"][:step+1]
            convs[conv_id] = conv
            save_all_sandboxes(convs)
        return {"status": "reverted", "commit_hash": commit_hash}
    else:
        return JSONResponse(status_code=500, content={"error": "Failed to revert sandbox"})

@app.post("/open_sandbox_folder/{sandbox_id}")
async def open_sandbox_folder(sandbox_id: str):
    """Ouvre le dossier du sandbox dans l'explorateur natif."""
    sandbox_path = os.path.join(SANDBOXES_DIR, sandbox_id)
    if not os.path.exists(sandbox_path):
        return JSONResponse(status_code=404, content={"error": "Sandbox folder not found"})
    try:
        if sys.platform.startswith("darwin"):  # macOS
            subprocess.Popen(["open", sandbox_path])
        elif sys.platform.startswith("win"):  # Windows
            os.startfile(sandbox_path)
        else:  # Linux et autres
            subprocess.Popen(["xdg-open", sandbox_path])
        return {"status": "ok"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# --- Server Launchers ---
def run_mcp():
    mcp.run(transport="streamable-http", port=9000, log_level="DEBUG")

def run_fastapi():
    import uvicorn
    port = int(os.getenv("PORT", "9001"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info", reload=False)

def run_webview():
    """Launch pywebview on the FastAPI URL."""
    webview.create_window("OperaFOR", url=f"http://localhost:{os.getenv('PORT', '9001')}")
    webview.start()

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s",
)
logger = logging.getLogger("mcp_app")

# Ensure errors and exceptions are written to file with full stack traces
def log_exception(exc_type, exc_value, exc_traceback):
    """Custom exception handler to log uncaught exceptions to file."""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    
    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

# Set custom exception handler
sys.excepthook = log_exception


if __name__ == "__main__":
    # Launch MCP in a separate thread
    mcp_thread = threading.Thread(target=run_mcp, daemon=True)
    mcp_thread.start()

    # Launch FastAPI (uvicorn) in a separate thread
    fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)
    fastapi_thread.start()
    
    # Launch pywebview in the main (blocking) thread
    run_webview()
