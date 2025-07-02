import os
import threading
import traceback
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from huggingface_hub import Agent  # Agent wraps MCPClient and handles tools
from fastmcp import FastMCP
import webview
import logging
import uuid
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

        # Update sandboxes.json: update messages and commits for this sandbox
        from pathlib import Path
        if CONV_FILE.exists():
            with open(CONV_FILE, 'r') as f:
                sandboxes = json.load(f)
            # Find the sandbox id from the path
            sandbox_id = os.path.basename(sandbox_path)
            conv = sandboxes.get(sandbox_id)
            if conv is not None:
                # Truncate commits up to and including the reverted commit
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

# --- MCP Server Definition ---
mcp = FastMCP("OperaFOR")



@mcp.tool()
def list_sandbox_files(sandbox_id: str) -> dict:
    """ List all output files in the sandbox directory.
    Args:
        folder_out (str): The path to the output folder.
    Returns:
        dict: A dictionary containing the list of output files.
    """
    # return the content of the output folder
    sandbox_path = os.path.join(SANDBOXES_DIR, sandbox_id)
    output_files = []
    for root, _, files in os.walk(sandbox_path):
        for file in files:
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, sandbox_path)

            # ignore .git directory and conversation.json
            if rel_path.startswith('.git/') or rel_path == 'conversation.json':
                continue

            # Ensure we only return files, not directories
            if os.path.isfile(file_path):
                output_files.append(rel_path)
    return {"files in the sandbox": output_files}


@mcp.tool()
def read_file_sandbox(sandbox_id: str, file_name:str) -> dict:
    """ Read a file from the sandbox directory.
    Args:
        sandbox_id (str): The ID of the sandbox.
        file_name (str): The name of the file to read.
    Returns:
        dict: A dictionary containing the file content.
    """
    sandbox_path = os.path.join(SANDBOXES_DIR, sandbox_id)
    file_path = os.path.join(sandbox_path, file_name)
    
    if not os.path.exists(file_path):
        return {"error": "File not found"}

    with open(file_path, 'r') as f:
        content = f.read()
    
    return {"file_content": content}


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

async def runAgent(data):
    # Retrieve the full history if provided
    messages = data.get("messages")
    prompt = (data.get("prompt") or "").strip()


    sandbox_id = (data.get("sandbox_id") or "0")
    print(f"Config : {data}")

    async with Agent(
        model=data.get("model"),
        base_url=data.get("base_url"),
        api_key=data.get("api_key"),
        servers=data.get("servers")
    ) as agent:
        await agent.load_tools()

        response_acum = ""
        try:
            agent.messages.append(
                    {"role": "system", "content": f"Sandbox ID : {sandbox_id}"})
                    #"Always structure your response to the user with html code to be displayed directly in an existing <div>, styled using tailwindcss and fontawesome"})
            if messages:
                agent.messages.extend(messages[:-1])
            async for chunk in agent.run(prompt):
                if hasattr(chunk, "choices"):
                    delta = chunk.choices[0].delta
                    if delta.content:
                        yield delta.content
                        response_acum += delta.content
                    if delta.tool_calls:
                        for call in delta.tool_calls:
                            if call.function.name:
                                print(f"Calling Tool : {call.function.name} {call.function.arguments}\n")
        except Exception as e:
            tb_str = traceback.format_exc()
            print(f"\nError during agent run: {e}\n{tb_str}", flush=True)
    
    # saving in conversation
    sandbox_id = (data.get("sandbox_id") or "0")
    new_message = {"role": "assistant", "content": response_acum}
    convs = load_all_sandboxes()
    conv = convs.get(sandbox_id)
    if conv is None:
        conv = {"id": sandbox_id, "messages": []}
    conv.setdefault("messages", []).append(new_message)
    
    # Git commit functionality
    sandbox_path = os.path.join(SANDBOXES_DIR, sandbox_id)
    os.makedirs(sandbox_path, exist_ok=True)
    
    # Get all messages for this conversation
    all_messages = conv.get("messages", [])
    
    # Create commit message based on the last user message and response
    user_prompt = prompt if prompt else "Agent interaction"
    commit_message = f"Agent response to: {user_prompt[:50]}..."

    # Write the conversation to a JSON file in the sandbox
    conversation_file = os.path.join(sandbox_path, "conversation.json")
    with open(conversation_file, "w") as f:
        json.dump(all_messages, f, indent=2)

    # Commit changes and get hash
    commit_hash = commit_sandbox_changes(sandbox_path, all_messages, commit_message)
    
    # Store commit hash in sandbox data
    if commit_hash:
        conv.setdefault("commits", []).append({
            "step": len(all_messages) - 1,  # Index of the assistant message
            "hash": commit_hash,
            "message": commit_message,
            "timestamp": datetime.now().isoformat()
        })
    
    convs[sandbox_id] = conv
    save_all_sandboxes(convs)

@app.post("/agent")
async def run_agent(request: Request):
    """Handle a streaming prompt with Agent.run()."""
    data = await request.json()
    logger.info(f"/agent endpoint called. Payload: {data}")
    return StreamingResponse(runAgent(data), media_type="text/plain")

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
        "endpoint": "https://openrouter.ai/api/v1/chat/completions",
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
    if "messages" in data:
        old_messages = conv.get("messages", [])
        new_messages = data["messages"]
        conv["messages"] = new_messages
        # Si des messages ont été supprimés, revert le sandbox au commit correspondant
        if len(new_messages) < len(old_messages):
            commits = conv.get("commits", [])
            # On cherche le commit dont le step correspond au dernier message restant
            target_step = len(new_messages) - 1
            target_commit = next((c for c in commits if c["step"] == target_step), None)
            if target_commit:
                sandbox_path = os.path.join(SANDBOXES_DIR, conv_id)
                revert_sandbox_to_commit(sandbox_path, target_commit["hash"])
    convs[conv_id] = conv
    save_all_sandboxes(convs)
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


if __name__ == "__main__":
    # Launch MCP in a separate thread
    mcp_thread = threading.Thread(target=run_mcp, daemon=True)
    mcp_thread.start()

    # Launch FastAPI (uvicorn) in a separate thread
    fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)
    fastapi_thread.start()
    
    # Launch pywebview in the main (blocking) thread
    run_webview()
