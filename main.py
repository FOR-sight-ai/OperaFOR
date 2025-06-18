import json
import os
import asyncio
import threading
import traceback
from typing import AsyncGenerator, AsyncIterable, Dict, List, Optional, Union
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
    logger.info(f"[TOOL] add utilisé avec a={a}, b={b}")
    return a+b

@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting"""
    logger.info(f"[RESOURCE] get_greeting utilisé avec name={name}")
    return f"Hello, {name}!"

# --- Flask App Definition ---
app = Flask(__name__)

@app.route("/agent", methods=["POST"])
def run_agent():
    """Gérer un prompt en streaming avec Agent.run()."""
    data = request.get_json()
    logger.info(f"/agent endpoint called. Payload: {data}")

    agent = Agent(
        model=data.get("model"),
        base_url=data.get("base_url"),
        api_key=data.get("api_key"),
        servers=data.get("servers")
    )

    def stream_response():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        async def _run():
            result = []
            try:
                async for chunk in agent.run((data.get("prompt") or "").strip()):
                    logger.debug(f"Agent chunk: {chunk}")
                    try:
                        content = chunk.choices[0].delta.content
                    except Exception:
                        content = None
                    if content:
                        result.append(content)
            except Exception as e:
                logger.exception("Error during agent.run() execution.")
                result.append(f"[ERROR] {str(e)}")
            # Tentative de fermeture explicite de la session asynchrone
            try:
                await agent.aclose()
            except AttributeError:
                pass  # Si la méthode n'existe pas, on ignore
            return result
        # On exécute la coroutine _run() qui retourne une liste de chunks
        for part in loop.run_until_complete(_run()):
            yield part
        loop.close()

    logger.info("Streaming agent response...")
    return app.response_class(stream_with_context(stream_response()), mimetype="text/plain")

@app.route("/")
def serve_index():
    """Servir index.html à la racine."""
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "index.html")

# --- Server Launchers ---
def run_mcp():
    mcp.run(transport="streamable-http", port=9000, log_level="DEBUG")

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


async def runAgent(prompt):

    servers = [{"type":"http","config":{"url": "http://localhost:9000/mcp"}}]

    async with Agent(
        model="deepseek/deepseek-chat-v3-0324:free",
        base_url="https://openrouter.ai/api/v1/chat/completions",
        api_key="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",
        servers=servers
    ) as agent:
        await agent.load_tools()
        print(f"Agent loaded with {len(agent.available_tools)} tools:")
        for t in agent.available_tools:
            print(f" • {t.function.name}")

        try:
            async for chunk in agent.run(prompt):
                if hasattr(chunk, "choices"):
                    delta = chunk.choices[0].delta
                    if delta.content:
                        print(delta.content, end="", flush=True)
                    if delta.tool_calls:
                        for call in delta.tool_calls:
                            if call.id:
                                print(f"<Tool {call.id}>", end="")
                            if call.function.name:
                                print(f"{call.function.name}", end=" ")
                            if call.function.arguments:
                                print(f"{call.function.arguments}", end="")


        except Exception as e:
            tb_str = traceback.format_exc()
            print(f"\nError during agent run: {e}\n{tb_str}", flush=True)
    return


if __name__ == "__main__":


    # Lancer MCP dans un thread séparé
    mcp_thread = threading.Thread(target=run_mcp, daemon=True)
    mcp_thread.start()

    loop = asyncio.get_event_loop()
    balances = loop.run_until_complete(runAgent("how much is 25 + 325 ?"))


def runother():

    # Lancer Flask dans un thread séparé
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    # Lancer pywebview dans le thread principal (bloquant)
    run_webview()
