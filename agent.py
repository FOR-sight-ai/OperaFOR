import json
import logging
from typing import Dict

from utils import (
    load_all_sandboxes,
    save_all_sandboxes,
    get_sandbox_path,
    commit_sandbox_changes,
    DEFAULT_CONFIG,
    CONFIG_PATH
)
from tools import TOOL_DEFINITIONS, execute_tool
from context_manager import apply_context_strategy


logger = logging.getLogger(__name__)


def inject_sandbox_context(sandbox_id: str, openai_messages: list) -> list:
    """
    Inject sandbox file list into context if sandbox is not empty.
    This is ephemeral and not stored in conversation history.
    """
    from tools import list_sandbox_files
    import os
    
    sandbox_path = get_sandbox_path(sandbox_id)
    if not os.path.exists(sandbox_path):
        return openai_messages
    
    # Check if sandbox has any files
    files = list_sandbox_files(sandbox_id)
    if not files or files == ["No files found in this sandbox."] or files == ["Sandbox directory does not exist yet."]:
        return openai_messages
    
    # Create ephemeral context message
    file_list_str = "\n".join(files)
    context_msg = {
        "role": "system",
        "content": f"📁 Current files in sandbox:\n{file_list_str}\n\nYou can use these files in your work."
    }
    
    # Insert after main system prompt (index 0)
    if len(openai_messages) > 0:
        openai_messages.insert(1, context_msg)
    
    return openai_messages


def call_llm(messages, tools, config):
    """Call LLM with retry logic."""
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
    import os
    
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
             role = "user"  # default
        
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
        read_only_tool_names = ["list_sandbox_files", "read_file_sandbox", "get_folder_structure_sandbox", "search_files_sandbox", "search_content_sandbox", "get_file_info_sandbox"]
        current_tools = [t for t in TOOL_DEFINITIONS if t["function"]["name"] in read_only_tool_names]
        system_prompt = f"You are a coding assistant. You have access to a sandbox environment with ID {sandbox_id}. This sandbox is pending READ-ONLY mode. You can ONLY read files. You CANNOT write, edit, or delete files. Use the provided tools."
    else:
        system_prompt = f"You are a coding assistant. You have access to a sandbox environment with ID {sandbox_id}. You can read, write, edit files. Prefer editing files over overwriting them. Use the provided tools."
    
    openai_messages.insert(0, {"role": "system", "content": system_prompt})
    
    # Inject ephemeral sandbox file list context (not stored in conversation)
    openai_messages = inject_sandbox_context(sandbox_id, openai_messages)
    
    # Apply context management before agent loop
    context_config = config.get("context_management", {})
    
    if context_config.get("enabled", True):
        try:
            openai_messages, context_stats = apply_context_strategy(
                openai_messages,
                context_config,
                config.get("llm", {}),
                sandbox_id=sandbox_id  # Pass sandbox_id for caching
            )
            
            # Log context reduction silently
            if context_stats and context_stats.get("strategy") not in ["none", "disabled"]:
                logger.info(f"Context reduction applied: {context_stats}")
        except Exception as e:
            logger.error(f"Error applying context management: {e}")
            # Continue with original messages on error
    
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
                # Determine message type: if tools are called, content is usually "thought"
                msg_type = "thought" if tool_calls else "content"
                yield json.dumps({"type": msg_type, "data": content}) + "\n"
                
            if tool_calls:
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
                        args["sandbox_id"] = sandbox_id  # Inject sandbox_id
                        args["model_name"] = config.get("llm", {}).get("model")  # Inject model_name
                        
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
                
                # Recover any new messages and append them
                new_msgs = openai_messages[initial_openai_count:]
                for msg in new_msgs:
                     msg["status"] = "done"
                     messages.append(msg)
                
                if not agent_success:
                     messages.append({"role": "assistant", "content": "Generation interrupted or failed.", "status": "error"})
                
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
            
    yield ""  # Close stream
