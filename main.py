import os
import threading
import traceback
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from huggingface_hub import Agent  # Agent wraps MCPClient and handles tools
from fastmcp import FastMCP
import webview
import logging

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


async def runAgent(data):
    prompt = (data.get("prompt") or "").strip()

    async with Agent(
        model=data.get("model"),
        base_url=data.get("base_url"),
        api_key=data.get("api_key"),
        servers=data.get("servers")
    ) as agent:
        await agent.load_tools()

        try:
            async for chunk in agent.run(prompt):
                if hasattr(chunk, "choices"):
                    delta = chunk.choices[0].delta
                    if delta.content:
                        yield delta.content
                    if delta.tool_calls:
                        for call in delta.tool_calls:
                            if call.function.name:
                                print(f"Calling Tool : {call.function.name} {call.function.arguments}\n")
        except Exception as e:
            tb_str = traceback.format_exc()
            print(f"\nError during agent run: {e}\n{tb_str}", flush=True)


@app.post("/agent")
async def run_agent(request: Request):
    """Gérer un prompt en streaming avec Agent.run()."""
    data = await request.json()
    logger.info(f"/agent endpoint called. Payload: {data}")
    return StreamingResponse(runAgent(data), media_type="text/plain")

@app.get("/")
async def serve_index():
    """Servir index.html à la racine."""
    index_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    return FileResponse(index_path, media_type="text/html")

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

# --- Server Launchers ---
def run_mcp():
    mcp.run(transport="streamable-http", port=9000, log_level="DEBUG")

def run_fastapi():
    import uvicorn
    port = int(os.getenv("PORT", "5000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info", reload=False)

def run_webview():
    """Lancer pywebview sur l'URL FastAPI."""
    webview.create_window("Demo MCP", url=f"http://localhost:{os.getenv('PORT', '5000')}")
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
