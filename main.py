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
mcp = FastMCP("Demo")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    logger.info(f"[TOOL] add utilisé avec a={a}, b={b}")
    return a+b

@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting"""
    logger.info(f"[RESOURCE] get_greeting utilisé avec name={name}")
    return f"Hello, {name}!"

# --- FastAPI App Definition ---
app = FastAPI()

# Ajout du montage pour les fichiers statiques
app.mount("/static", StaticFiles(directory=os.path.dirname(os.path.abspath(__file__))), name="static")

CONV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sandboxes.json")

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
    # On récupère l'historique complet si fourni
    messages = data.get("messages")
    prompt = (data.get("prompt") or "").strip()

    async with Agent(
        model=data.get("model"),
        base_url=data.get("base_url"),
        api_key=data.get("api_key"),
        servers=data.get("servers")
    ) as agent:
        await agent.load_tools()

        response_acum = ""
        try:
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
    """Gérer un prompt en streaming avec Agent.run()."""
    data = await request.json()
    logger.info(f"/agent endpoint called. Payload: {data}")
    return StreamingResponse(runAgent(data), media_type="text/plain")

@app.get("/")
async def serve_index():
    """Servir index.html à la racine, compatible PyInstaller."""
    try:
        # Si exécuté via PyInstaller, __file__ est dans un bundle
        if hasattr(sys, '_MEIPASS'):
            index_path = os.path.join(sys._MEIPASS, "index.html")
        else:
            index_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
        with open(index_path, "rb") as f:
            content = f.read()
        return Response(content, media_type="text/html")
    except Exception as e:
        return Response(f"Erreur lors du chargement de l'interface : {e}", status_code=500)

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
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
    conv = {
        "id": conv_id,
        "title": title,
        "messages": []
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
    # Autorise la modification du titre uniquement
    if "title" in data:
        conv["title"] = data["title"]
    convs[conv_id] = conv
    save_all_sandboxes(convs)
    return {"status": "ok"}

# --- Server Launchers ---
def run_mcp():
    mcp.run(transport="streamable-http", port=9000, log_level="DEBUG")

def run_fastapi():
    import uvicorn
    port = int(os.getenv("PORT", "5000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info", reload=False)

def run_webview():
    """Lancer pywebview sur l'URL FastAPI."""
    webview.create_window("OperaFOR", url=f"http://localhost:{os.getenv('PORT', '5000')}")
    webview.start()

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s",
)
logger = logging.getLogger("mcp_app")


if __name__ == "__main__":
    # Lancer MCP dans un thread séparé
    mcp_thread = threading.Thread(target=run_mcp, daemon=True)
    mcp_thread.start()

    # Lancer FastAPI (uvicorn) dans un thread séparé
    fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)
    fastapi_thread.start()
    
    # Lancer pywebview dans le thread principal (bloquant)
    run_webview()
