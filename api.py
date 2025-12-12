import os
import sys
import json
import uuid
import subprocess
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from url_handler import process_urls_in_prompt
from utils import (
    load_all_sandboxes,
    save_all_sandboxes,
    get_sandbox_path,
    init_or_get_repo,
    revert_sandbox_to_commit,
    DEFAULT_CONFIG,
    CONFIG_PATH,
    SANDBOXES_DIR
)
from agent import runAgent


# --- FastAPI App ---
app = FastAPI()
app.mount("/static", StaticFiles(directory=os.path.dirname(os.path.abspath(__file__))), name="static")


# --- Agent Endpoint ---

@app.post("/agent")
async def run_agent(request: Request):
    data = await request.json()
    return StreamingResponse(runAgent(data.get("sandbox_id")), media_type="text/plain")


# --- Static Files ---

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


# --- Configuration Endpoints ---

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


# --- Sandbox CRUD Endpoints ---

@app.get("/sandboxes")
async def api_list_sandboxes():
    convs = load_all_sandboxes()
    return [{"id": cid, "title": c.get("title", f"Sandbox {cid}"), "read_only": c.get("read_only", False)} for cid, c in convs.items()]


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

    conv = {"id": conv_id, "title": title, "read_only": read_only, "messages": messages}
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
    
    # Process URLs in user messages
    if data.get("role") == "user" and "content" in data:
        try:
            updated_content, url_results = process_urls_in_prompt(data["content"], conv_id)
            if url_results:
                # Update message content with download results
                data["content"] = updated_content
        except Exception as e:
            # Log error but don't fail the message addition
            print(f"Error processing URLs in message: {e}")
    
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


# --- Git Endpoints ---

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


# --- Utility Endpoints ---

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
