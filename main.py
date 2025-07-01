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
import importlib.resources

# --- MCP Server Definition ---
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


def find_numbered_folders(sandbox_id: str) -> str:
    """Find the highest numbered folder in a given sandbox.
    Args:
        sandbox_id (str): The ID of the sandbox where numbered folders are searched.
    Returns:
        tuple: A tuple containing the path to the input folder and the next output folder.
        If no folders are found, returns None and "step_0".

    """

    folder_out = os.path.join(SANDBOXES_DIR, sandbox_id)
    if not os.path.exists(folder_out):
        os.makedirs(folder_out)
    # In the sandbox id folder, find the highest numbered folder
    existing_folders = [d for d in os.listdir(folder_out) if os.path.isdir(os.path.join(folder_out, d))]
    if existing_folders:
        highest_folder = max(existing_folders, key=lambda x: int(x.split('_')[-1]) if '_' in x else 0)
        folder_in = os.path.join(folder_out, highest_folder)
        folder_next = os.path.join(folder_out,  f"step_{int(highest_folder.split('_')[-1]) + 1}")

    else:
        return folder_out,  os.path.join(folder_out, "step_0")

    return folder_in, folder_next

def list_folder_files(folder_out: str) -> dict:
    """ List all output files in the sandbox directory.
    Args:
        folder_out (str): The path to the output folder.
    Returns:
        dict: A dictionary containing the list of output files.
    """
    # return the content of the output folder
    output_files = []
    for root, _, files in os.walk(folder_out):
        for file in files:
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, folder_out)
            output_files.append(rel_path)
    return {"output_files": output_files}

@mcp.tool()
def convert_urls_to_markdown(query: str, sandbox_id: str):
    """
    MCP Tool: Downloads URLs from a query and converts them to markdown files.
    Args:
        query (str): The search query to find URLs.
        sandbox_id (str): The ID of the sandbox where the markdown files will be saved.
    Returns:

    """
    from forcolate import convert_URLS_to_markdown

    folder_in, folder_out = find_numbered_folders(sandbox_id)
    convert_URLS_to_markdown(query, folder_in, folder_out)
    return list_folder_files(folder_out)

@mcp.tool()
def search_folder(query: str, sandbox_id: str):
    from forcolate import search_folder
    folder_in, folder_out = find_numbered_folders(sandbox_id)
    file_paths = search_folder(query, folder_in, folder_out)
    print(file_paths)
    return list_folder_files(folder_out)


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
    convId = (data.get("convId") or "0")
    new_message = {"role": "assistant", "content": response_acum}
    convs = load_all_sandboxes()
    conv = convs.get(convId)
    if conv is None:
        conv = {"id": convId, "messages": []}
    conv.setdefault("messages", []).append(new_message)
    convs[convId] = conv
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
    }
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
        conv["messages"] = data["messages"]
    convs[conv_id] = conv
    save_all_sandboxes(convs)
    return {"status": "ok"}

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
