import os
import asyncio
import threading
from flask import Flask, request, jsonify, send_from_directory, stream_with_context
from huggingface_hub import Agent  # Agent wraps MCPClient and handles tools
from fastmcp import FastMCP
import webview
import logging

# --- MCP Server Definition ---
mcp = FastMCP("Demo")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting"""
    return f"Hello, {name}!"

# --- Flask App Definition ---
app = Flask(__name__)

@app.route("/agent", methods=["POST"])
def run_agent():
    """Gérer un prompt en streaming avec Agent.run()."""
    data = request.get_json()
    logger.info(f"/agent endpoint called. Payload: {data}")
    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        logger.warning("No prompt provided in request.")
        return jsonify({"error": "No prompt provided"}), 400

    # Récupérer dynamiquement la configuration de l'agent depuis le payload
    model = data.get("model")
    base_url = data.get("base_url")
    api_key = data.get("api_key")
    servers = data.get("servers")
    logger.info(f"Agent config received: model={model}, base_url={base_url}, api_key={'***' if api_key else None}, servers={servers}")
    if not all([model, base_url, api_key, servers]):
        logger.error("Missing agent configuration parameters (model, base_url, api_key, servers)")
        return jsonify({"error": "Missing agent configuration parameters (model, base_url, api_key, servers)"}), 400

    # Instancier dynamiquement l'agent
    try:
        agent = Agent(
            model=model,
            base_url=base_url,
            api_key=api_key,
            servers=servers
        )
        logger.info("Agent instance created successfully.")
    except Exception as e:
        logger.exception("Failed to instantiate Agent.")
        return jsonify({"error": f"Agent instantiation failed: {e}"}), 500

    def stream_response():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        async def _run():
            try:
                async for chunk in agent.run(prompt):
                    logger.debug(f"Agent chunk: {chunk}")
                    # On extrait le texte du chunk si présent
                    try:
                        content = chunk.choices[0].delta.content
                    except Exception:
                        content = None
                    if content:
                        yield content
            except Exception as e:
                logger.exception("Error during agent.run() execution.")
                yield f"[ERROR] {str(e)}"
        # Utiliser run_until_complete pour exécuter l'async generator
        for part in loop.run_until_complete(_collect()):
            yield part
        loop.close()

    async def _collect():
        result = []
        async for part in agent.run(prompt):
            try:
                content = part.choices[0].delta.content
            except Exception:
                content = None
            if content:
                result.append(content)
                yield content
        return result

    logger.info("Streaming agent response...")
    return app.response_class(stream_with_context(stream_response()), mimetype="text/plain")

@app.route("/")
def serve_index():
    """Servir index.html à la racine."""
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "index.html")

# --- Server Launchers ---
def run_mcp():
    mcp.run(transport="streamable-http", port=9000)

def run_flask():
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))

def run_webview():
    """Lancer pywebview sur l'URL Flask."""
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
    # Lancer Flask dans un thread séparé
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    # Lancer pywebview dans le thread principal (bloquant)
    run_webview()
