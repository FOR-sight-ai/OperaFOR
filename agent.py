import json
import logging
from typing import Dict

from utils import (
    load_all_sandboxes,
    save_all_sandboxes,
    get_sandbox_path,
    commit_sandbox_changes,
    DEFAULT_CONFIG,
    CONFIG_PATH,
    get_proxies,
    call_llm
)
from tools import TOOL_DEFINITIONS, execute_tool
from context_manager import apply_context_strategy, count_messages_tokens
from file_preprocessor import preprocess_sandbox_files
from url_handler import process_urls_in_prompt


logger = logging.getLogger(__name__)


def inject_sandbox_context(sandbox_id: str, openai_messages: list) -> list:
    """
    Inject sandbox file list into context if sandbox is not empty.
    This is ephemeral and not stored in conversation history.
    The agent receives only the list of file paths, not their content.
    This allows the agent to know what files exist (including URL downloads)
    without consuming context with file contents.
    """
    from tools import list_files
    import os

    sandbox_path = get_sandbox_path(sandbox_id)
    if not os.path.exists(sandbox_path):
        return openai_messages

    # Check if sandbox has any files
    # Use unlimited depth (None) to show all files including nested URL downloads
    files = list_files(sandbox_id, max_depth=None)
    if not files or files == ["No files found in this sandbox."] or files == ["Sandbox directory does not exist yet."]:
        return openai_messages

    # Create ephemeral context message with file list only (no content)
    file_list_str = "\n".join(files)
    context_msg = {
        "role": "system",
        "content": f"📁 Current files in sandbox:\n{file_list_str}\n\nThese files have been imported and are available for you to read and work with using the provided tools."
    }

    # Insert after main system prompt (index 0)
    if len(openai_messages) > 0:
        openai_messages.insert(1, context_msg)

    return openai_messages




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

    # Find last user message step for commit association
    last_user_msg_idx = len(messages) - 1
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            last_user_msg_idx = i
            break
    
    # Startup commit
    commit_sandbox_changes(sandbox_id, last_user_msg_idx, "Agent startup")

    # --- URL PROCESSING: Process URLs in the last user message before building context ---
    artifact_change = False
    if messages:
        if last_user_msg_idx != -1:
            last_user_msg = messages[last_user_msg_idx]
            content = last_user_msg.get("content", "")

            if content:
                try:
                    updated_content, url_results = process_urls_in_prompt(content, sandbox_id)
                    if url_results:
                        artifact_change = True
                        logger.info(f"Processed URLs in user message: {len(url_results)} files imported")
                except Exception as e:
                    logger.error(f"Error processing URLs in message: {e}")

    # --- PREPROCESSING: Convert non-text files before agent loop ---
    try:
        model_name = config.get("llm", {}).get("model")
        conversions = preprocess_sandbox_files(sandbox_id, model_name)
        if conversions:
            artifact_change = True
            logger.info(f"Preprocessed {len(conversions)} files: {conversions}")
    except Exception as e:
        logger.error(f"Error during file preprocessing: {e}")
    
    # Commit after artifacts if any
    if artifact_change:
        commit_sandbox_changes(sandbox_id, last_user_msg_idx, "Artifacts downloaded and converted")

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
    for m in messages[:-1]:
        role = m.get("role")
        content = m.get("content")
        if role not in ["user", "assistant", "system", "tool"]:
             role = "user"
        
        msg_obj = {"role": role, "content": content}
        if "tool_calls" in m and m["tool_calls"]:
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
        read_only_tool_names = ["list_files", "read_file", "get_folder_structure", "search_files", "search_content", "get_file_info"]
        current_tools = [t for t in TOOL_DEFINITIONS if t["function"]["name"] in read_only_tool_names]
        system_prompt = f"You are a coding assistant. You have access to a sandbox environment with ID {sandbox_id}. This sandbox is pending READ-ONLY mode. You can ONLY read files (including .docx and .pptx). You CANNOT write, edit, or delete files. Use the provided tools."
    else:
        system_prompt = f"You are a coding assistant. You have access to a sandbox environment with ID {sandbox_id}. You can read, write, edit files (including .docx and .pptx). Prefer editing files over overwriting them. Use the provided tools."
    
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

    # Context monitoring during agent run
    max_context_tokens = context_config.get("max_context_during_run", 100000)

    try:
        while current_turn < max_turns:
            current_turn += 1

            # Check context size before each LLM call
            current_context_tokens = count_messages_tokens(openai_messages)

            if current_context_tokens > max_context_tokens:
                logger.info(f"Context size ({current_context_tokens} tokens) exceeds threshold ({max_context_tokens} tokens). Applying compression...")

                # Apply context compression during the run
                try:
                    openai_messages, compression_stats = apply_context_strategy(
                        openai_messages,
                        context_config,
                        config.get("llm", {}),
                        sandbox_id=sandbox_id
                    )

                    new_context_tokens = count_messages_tokens(openai_messages)
                    logger.info(f"Context compressed from {current_context_tokens} to {new_context_tokens} tokens")

                    # Yield a status message to the user
                    yield json.dumps({
                        "type": "system",
                        "data": f"[Context compressed: {current_context_tokens} → {new_context_tokens} tokens]"
                    }) + "\n"

                except Exception as e:
                    logger.error(f"Error during context compression: {e}")
                    # Continue with original messages on error

            # Notify user that we're calling the LLM
            yield json.dumps({
                "type": "status",
                "data": f"Calling LLM (turn {current_turn}/{max_turns})..."
            }) + "\n"

            response_data = call_llm(openai_messages, current_tools, config)

            if "error" in response_data:
                yield json.dumps({"type": "error", "data": f"Error from LLM: {response_data['error']}"}) + "\n"
                # Add to history as error
                openai_messages.append({"role": "assistant", "content": f"Error from LLM: {response_data['error']}"})
                break

            # Emit token usage if available
            usage = response_data.get("usage", {})
            if usage:
                yield json.dumps({
                    "type": "usage",
                    "data": {
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0)
                    }
                }) + "\n"

            choice = response_data.get("choices", [{}])[0]
            message = choice.get("message", {})
            content = message.get("content")
            tool_calls = message.get("tool_calls")
            
            # Save assistant message to history (incremental)
            msg_to_save = message.copy()
            msg_to_save["status"] = "done"
            # Remove pending placeholder before adding new Turn
            if messages and messages[-1].get("status") == "pending":
                messages.pop()
            messages.append(msg_to_save)
            
            # Re-add pending placeholder for the next assistant or tool result
            messages.append({"role": "assistant", "content": "", "status": "pending"})
            conv["messages"] = messages
            convs[sandbox_id] = conv
            save_all_sandboxes(convs)
            
            # Yield content if any
            if content:
                msg_type = "thought" if tool_calls else "content"
                yield json.dumps({"type": msg_type, "data": content}) + "\n"
                
            if tool_calls:
                yield json.dumps({
                    "type": "status",
                    "data": f"Executing {len(tool_calls)} tool(s)..."
                }) + "\n"

                modifying_tools = ["write_to_file", "append_to_file", "delete_file", "edit_file", "import_outlook_emails"]

                for tc in tool_calls:
                    func_name = tc["function"]["name"]
                    args_str = tc["function"]["arguments"]
                    call_id = tc["id"]

                    yield json.dumps({
                        "type": "tool_call",
                        "data": {"name": func_name, "arguments": args_str, "id": call_id}
                    }) + "\n"

                    try:
                        args = json.loads(args_str)
                        args["sandbox_id"] = sandbox_id
                        args["model_name"] = config.get("llm", {}).get("model")
                        
                        result = execute_tool(func_name, args)
                        
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
                    
                    # Append tool result to history (incremental)
                    tool_msg = {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": func_name,
                        "content": content_payload,
                        "status": "done"
                    }
                    openai_messages.append(tool_msg)
                    
                    # Update saved state
                    if messages and messages[-1].get("status") == "pending":
                        messages.pop()
                    messages.append(tool_msg)
                    messages.append({"role": "assistant", "content": "", "status": "pending"})
                    conv["messages"] = messages
                    convs[sandbox_id] = conv
                    save_all_sandboxes(convs)

                    # Commit if it's a modifying tool
                    if func_name in modifying_tools:
                        commit_sandbox_changes(sandbox_id, len(messages)-2, f"Tool: {func_name}")

            else:
                 # No tool calls, we are done
                 agent_success = True
                 # Final commit after last response
                 commit_sandbox_changes(sandbox_id, len(messages)-2, "Agent final response")
                 break
        
    except Exception as e:
        # Yield error to user
        yield json.dumps({"type": "error", "data": f"Error during execution: {str(e)}"}) + "\n"
        # We will handle persistence in finally
        
    finally:
        # --- FINALLY: Update the pending message ---
        try:
            convs = load_all_sandboxes()
            conv = convs.get(sandbox_id)
            if conv:
                messages = conv.get("messages", [])
                # Remove the pending message
                if messages and messages[-1].get("status") == "pending":
                    messages.pop()
                
                if not agent_success:
                     messages.append({"role": "assistant", "content": "Generation interrupted or failed.", "status": "error"})
                
                # Ensure last user message is marked as done
                for i in range(len(messages) - 1, -1, -1):
                    if messages[i].get("role") == "user":
                        if messages[i].get("status") == "pending":
                             messages[i]["status"] = "done" if agent_success else "error"
                        break

                conv["messages"] = messages
                convs[sandbox_id] = conv
                save_all_sandboxes(convs)
                
        except Exception as e:
            print(f"Critical error saving conversation state: {e}")
            
    yield ""  # Close stream
