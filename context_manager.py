"""
Context Management Module for OperaFOR

Implements aggressive strategies for reducing LLM context window usage:
- Accurate token counting and estimation
- Tool result truncation and compression
- Rolling window summarization with lower thresholds
- Context scoping for tasks and subtasks
- Pragmatic truncation strategies
"""

import json
import logging
import hashlib
from typing import List, Dict, Any, Optional, Tuple

try:
    from context_cache import get_incremental_summary
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False
    logging.warning("context_cache module not available, caching disabled")

logger = logging.getLogger(__name__)


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for a given text.
    Uses improved heuristic: ~3.5 characters per token (more accurate for English).
    """
    if not text:
        return 0
    # More accurate heuristic: 3.5 chars per token
    return int(len(text) / 3.5) + 1


def count_message_tokens(message: Dict[str, Any]) -> int:
    """Count tokens in a single message."""
    tokens = 0

    # Count content
    content = message.get("content", "")
    if isinstance(content, str):
        tokens += estimate_tokens(content)
    elif isinstance(content, list):
        # Handle multimodal content (images, etc.)
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    tokens += estimate_tokens(item.get("text", ""))
                elif item.get("type") == "image_url":
                    # Images are expensive - rough estimate
                    tokens += 765  # Approximate for vision models
            elif isinstance(item, str):
                tokens += estimate_tokens(item)

    # Count tool calls
    if "tool_calls" in message:
        for tc in message.get("tool_calls", []):
            tokens += estimate_tokens(json.dumps(tc))

    # Count tool results
    if message.get("role") == "tool":
        tool_content = message.get("content", "")
        if isinstance(tool_content, str):
            tokens += estimate_tokens(tool_content)
        elif isinstance(tool_content, list):
            tokens += estimate_tokens(json.dumps(tool_content))

    # Add overhead for message structure (role, metadata, etc.)
    tokens += 4

    return tokens


def count_messages_tokens(messages: List[Dict[str, Any]]) -> int:
    """Count total tokens in a list of messages."""
    return sum(count_message_tokens(msg) for msg in messages)


def truncate_tool_result(content: str, max_length: int = 2000, tool_name: str = None) -> str:
    """
    Intelligently truncate tool result content.

    Args:
        content: The tool result content
        max_length: Maximum character length (not tokens, for simplicity)
        tool_name: Name of the tool (for context-aware truncation)

    Returns:
        Truncated content with summary
    """
    if len(content) <= max_length:
        return content

    # Special handling for different tool types
    if tool_name in ["read_file", "Read"]:
        # For file reads, keep beginning and end
        head_size = int(max_length * 0.7)
        tail_size = int(max_length * 0.2)
        return (
            content[:head_size] +
            f"\n\n[... {len(content) - head_size - tail_size} characters truncated ...]\n\n" +
            content[-tail_size:]
        )

    elif tool_name in ["search", "grep", "Grep"]:
        # For search results, keep as many complete lines as possible
        lines = content.split('\n')
        kept_lines = []
        current_length = 0

        for line in lines:
            if current_length + len(line) + 1 <= max_length:
                kept_lines.append(line)
                current_length += len(line) + 1
            else:
                break

        truncated_count = len(lines) - len(kept_lines)
        result = '\n'.join(kept_lines)
        if truncated_count > 0:
            result += f"\n[... {truncated_count} more lines truncated ...]"
        return result

    elif tool_name in ["list_files", "Glob", "glob"]:
        # For file lists, keep as many complete entries as possible
        lines = content.split('\n')
        kept_lines = []
        current_length = 0

        for line in lines[:200]:  # Max 200 files shown
            if current_length + len(line) + 1 <= max_length:
                kept_lines.append(line)
                current_length += len(line) + 1
            else:
                break

        result = '\n'.join(kept_lines)
        if len(lines) > len(kept_lines):
            result += f"\n[... {len(lines) - len(kept_lines)} more files truncated ...]"
        return result

    else:
        # Generic truncation: keep beginning
        return content[:max_length] + f"\n\n[... {len(content) - max_length} characters truncated ...]"


def compress_tool_messages(messages: List[Dict[str, Any]], max_tool_result_chars: int = 2000) -> List[Dict[str, Any]]:
    """
    Compress tool messages by truncating large results.
    This is the PRIMARY compression mechanism for reducing context.

    Args:
        messages: List of messages
        max_tool_result_chars: Maximum characters per tool result

    Returns:
        Messages with compressed tool results
    """
    compressed = []

    for msg in messages:
        if msg.get("role") == "tool":
            # Get tool name from the message
            tool_name = msg.get("name", "unknown")
            content = msg.get("content", "")

            if isinstance(content, str) and len(content) > max_tool_result_chars:
                # Truncate the content
                truncated = truncate_tool_result(content, max_tool_result_chars, tool_name)

                # Create new message with truncated content
                compressed_msg = msg.copy()
                compressed_msg["content"] = truncated
                compressed.append(compressed_msg)
            else:
                compressed.append(msg)
        else:
            compressed.append(msg)

    return compressed


def group_messages(messages: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """
    Group messages into atomic units that must stay together.

    An atomic unit is:
    - An assistant message with tool_calls + all following tool messages
    - A single message without tool dependencies

    Returns:
        List of message groups (each group is a list of related messages)
    """
    groups = []
    current_group = []
    waiting_for_tools = False
    expected_tool_ids = set()

    for msg in messages:
        role = msg.get("role")

        if role == "assistant" and "tool_calls" in msg:
            # Start a new group for this assistant message and its tool results
            if current_group:
                groups.append(current_group)
            current_group = [msg]
            waiting_for_tools = True
            # Track which tool call IDs we're expecting results for
            expected_tool_ids = {tc["id"] for tc in msg.get("tool_calls", [])}
        elif role == "tool" and waiting_for_tools:
            # This tool message belongs to the current group
            current_group.append(msg)
            # Remove this tool_call_id from expected set
            tool_call_id = msg.get("tool_call_id")
            if tool_call_id in expected_tool_ids:
                expected_tool_ids.remove(tool_call_id)
            # If we've received all expected tool results, close the group
            if not expected_tool_ids:
                groups.append(current_group)
                current_group = []
                waiting_for_tools = False
        else:
            # This is a standalone message (user, system, or assistant without tools)
            if current_group:
                # Close any pending group first
                groups.append(current_group)
                current_group = []
                waiting_for_tools = False
            groups.append([msg])

    # Don't forget the last group if any
    if current_group:
        groups.append(current_group)

    return groups


def validate_message_structure(messages: List[Dict[str, Any]]) -> bool:
    """
    Validate that message structure follows OpenAI API requirements.

    Specifically checks that every tool message has a preceding assistant
    message with a matching tool_call_id.

    Returns:
        True if structure is valid, False otherwise
    """
    # Track tool_call_ids from assistant messages
    available_tool_call_ids = set()

    for msg in messages:
        role = msg.get("role")

        if role == "assistant" and "tool_calls" in msg:
            # Add all tool_call_ids from this message
            for tc in msg.get("tool_calls", []):
                available_tool_call_ids.add(tc["id"])
        elif role == "tool":
            # Check if this tool message has a valid parent
            tool_call_id = msg.get("tool_call_id")
            if tool_call_id not in available_tool_call_ids:
                logger.error(
                    f"Invalid message structure: tool message with tool_call_id={tool_call_id} "
                    f"has no preceding assistant message with matching tool_call"
                )
                return False

    return True


def summarize_messages_with_llm(
    messages: List[Dict[str, Any]],
    llm_config: Dict[str, Any],
    sandbox_id: Optional[str] = None
) -> str:
    """
    Use the LLM to create a summary of the given messages.
    Returns a concise summary string.

    If sandbox_id is provided and caching is available, will use incremental
    summarization with cache.
    """
    # Try to use cache if available
    if sandbox_id and CACHE_AVAILABLE:
        try:
            summary = get_incremental_summary(sandbox_id, messages, llm_config)
            if summary:
                logger.info(f"Using cached/incremental summary for {len(messages)} messages")
                return summary
        except Exception as e:
            logger.warning(f"Cache lookup failed, falling back to direct LLM call: {e}")

    import requests

    # Build a prompt for summarization
    conversation_text = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")

        # Handle multimodal content
        if isinstance(content, list):
            text_parts = [
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            ]
            content = " ".join(text_parts)

        # Skip tool messages for summary (too verbose)
        if role == "tool":
            tool_name = msg.get('name', 'unknown')
            content_preview = content[:100] if isinstance(content, str) else str(content)[:100]
            conversation_text.append(f"[Tool: {tool_name} returned {len(str(content))} chars]")
        elif role == "assistant" and "tool_calls" in msg:
            tool_names = [tc["function"]["name"] for tc in msg.get("tool_calls", [])]
            conversation_text.append(f"Assistant: [Called tools: {', '.join(tool_names)}]")
            if content:
                conversation_text.append(f"  Thought: {content}")
        else:
            conversation_text.append(f"{role.capitalize()}: {content}")

    conversation_str = "\n".join(conversation_text)

    summarization_prompt = f"""Summarize the following conversation in 1-2 concise sentences, focusing on:
- Main task or goal
- Key decisions or findings
- Current state

Conversation:
{conversation_str}

Summary (1-2 sentences):"""

    # Call LLM for summarization
    endpoint = llm_config.get("endpoint", "https://openrouter.ai/api/v1")
    if not endpoint.endswith("/chat/completions"):
        endpoint = endpoint.rstrip("/") + "/chat/completions"

    api_key = llm_config.get("apiKey")
    model = llm_config.get("model")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    data = {
        "model": model,
        "messages": [{"role": "user", "content": summarization_prompt}],
        "temperature": 0.3,  # Lower temperature for more focused summaries
        "max_tokens": 150  # Reduced from 300 to force more concise summaries
    }

    try:
        response = requests.post(endpoint, json=data, headers=headers, timeout=30)
        response.raise_for_status()
        result = response.json()
        summary = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        return summary.strip()
    except Exception as e:
        logger.error(f"Error during summarization: {e}")
        # Fallback: create a simple summary
        return f"[Previous {len(messages)} messages]"


def apply_rolling_window_strategy(
    messages: List[Dict[str, Any]],
    config: Dict[str, Any],
    llm_config: Dict[str, Any],
    sandbox_id: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Apply rolling window summarization strategy with aggressive compression.

    Returns:
        Tuple of (reduced_messages, stats)
    """
    max_tokens = config.get("max_tokens", 4000)
    threshold = config.get("summarization_threshold", 1500)  # Reduced from 3000
    preserve_recent = config.get("preserve_recent_messages", 3)  # Reduced from 5
    preserve_system = config.get("preserve_system_prompt", True)
    max_tool_result_chars = config.get("max_tool_result_chars", 2000)

    # Step 1: Compress tool results FIRST (most effective compression)
    messages = compress_tool_messages(messages, max_tool_result_chars)

    total_tokens = count_messages_tokens(messages)

    # If under threshold, no reduction needed
    if total_tokens <= threshold:
        return messages, {
            "strategy": "tool_compression_only",
            "original_tokens": total_tokens,
            "reduced_tokens": total_tokens,
            "messages_summarized": 0
        }

    # Group messages into atomic units
    message_groups = group_messages(messages)

    # Separate system messages, old groups, and recent groups
    system_messages = []
    middle_groups = []
    recent_groups = []

    # Count how many individual messages we want to preserve
    # We'll work backwards from the end, preserving whole groups
    messages_from_end = 0
    for i in range(len(message_groups) - 1, -1, -1):
        group = message_groups[i]

        # Check if this is a system message group
        if len(group) == 1 and group[0].get("role") == "system":
            if preserve_system:
                system_messages.insert(0, group)
            continue

        # Count messages in this group
        group_msg_count = len(group)

        if messages_from_end < preserve_recent:
            # We still need to preserve more recent messages
            recent_groups.insert(0, group)
            messages_from_end += group_msg_count
        else:
            # This group goes into the middle (to be summarized)
            middle_groups.insert(0, group)

    # Flatten groups back to messages for summarization
    middle_messages = [msg for group in middle_groups for msg in group]

    # If we still need to reduce, summarize the middle messages
    if middle_messages:
        logger.info(f"Summarizing {len(middle_messages)} messages to reduce context")
        summary_text = summarize_messages_with_llm(middle_messages, llm_config, sandbox_id)

        # Create a summary message
        summary_message = {
            "role": "system",
            "content": f"[Summary of {len(middle_messages)} messages]: {summary_text}"
        }

        # Flatten all groups back to messages
        system_msgs_flat = [msg for group in system_messages for msg in group]
        recent_msgs_flat = [msg for group in recent_groups for msg in group]

        # Reconstruct message list
        reduced_messages = system_msgs_flat + [summary_message] + recent_msgs_flat
    else:
        # No middle messages to summarize
        system_msgs_flat = [msg for group in system_messages for msg in group]
        recent_msgs_flat = [msg for group in recent_groups for msg in group]
        reduced_messages = system_msgs_flat + recent_msgs_flat

    # Validate the structure before returning
    if not validate_message_structure(reduced_messages):
        logger.warning("Message structure validation failed after rolling window reduction, returning original messages")
        return messages, {
            "strategy": "rolling_window_failed",
            "original_tokens": total_tokens,
            "reduced_tokens": total_tokens,
            "error": "Structure validation failed"
        }

    reduced_tokens = count_messages_tokens(reduced_messages)

    return reduced_messages, {
        "strategy": "rolling_window",
        "original_tokens": total_tokens,
        "reduced_tokens": reduced_tokens,
        "messages_summarized": len(middle_messages),
        "compression_ratio": round(reduced_tokens / total_tokens, 2) if total_tokens > 0 else 1.0
    }



def apply_truncation_strategy(
    messages: List[Dict[str, Any]],
    config: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Apply simple truncation strategy - keep only recent messages.
    Also compresses tool results.

    Returns:
        Tuple of (reduced_messages, stats)
    """
    max_tokens = config.get("max_tokens", 4000)
    preserve_system = config.get("preserve_system_prompt", True)
    max_tool_result_chars = config.get("max_tool_result_chars", 2000)

    # Step 1: Compress tool results FIRST
    messages = compress_tool_messages(messages, max_tool_result_chars)

    total_tokens = count_messages_tokens(messages)

    # Group messages into atomic units
    message_groups = group_messages(messages)

    # Separate system messages and non-system groups
    system_groups = []
    non_system_groups = []

    for group in message_groups:
        if len(group) == 1 and group[0].get("role") == "system":
            if preserve_system:
                system_groups.append(group)
        else:
            non_system_groups.append(group)

    # Calculate budget for non-system messages
    system_msgs_flat = [msg for group in system_groups for msg in group]
    system_tokens = count_messages_tokens(system_msgs_flat)
    remaining_budget = max_tokens - system_tokens

    # Take groups from the end until we hit the budget
    reduced_non_system_groups = []
    current_tokens = 0

    for group in reversed(non_system_groups):
        group_tokens = sum(count_message_tokens(msg) for msg in group)
        if current_tokens + group_tokens <= remaining_budget:
            reduced_non_system_groups.insert(0, group)
            current_tokens += group_tokens
        else:
            break

    # Flatten groups back to messages
    reduced_non_system = [msg for group in reduced_non_system_groups for msg in group]
    reduced_messages = system_msgs_flat + reduced_non_system

    # Validate the structure before returning
    if not validate_message_structure(reduced_messages):
        logger.warning("Message structure validation failed after truncation, returning original messages")
        return messages, {
            "strategy": "truncation_failed",
            "original_tokens": total_tokens,
            "reduced_tokens": total_tokens,
            "error": "Structure validation failed"
        }

    reduced_tokens = count_messages_tokens(reduced_messages)

    total_non_system_msgs = sum(len(group) for group in non_system_groups)
    reduced_non_system_msgs = sum(len(group) for group in reduced_non_system_groups)

    return reduced_messages, {
        "strategy": "truncation",
        "original_tokens": total_tokens,
        "reduced_tokens": reduced_tokens,
        "messages_removed": total_non_system_msgs - reduced_non_system_msgs
    }



def apply_hybrid_strategy(
    messages: List[Dict[str, Any]],
    config: Dict[str, Any],
    llm_config: Dict[str, Any],
    sandbox_id: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Apply hybrid strategy - preserve pinned messages + summarize middle + keep recent.
    Includes aggressive tool result compression.

    Returns:
        Tuple of (reduced_messages, stats)
    """
    max_tokens = config.get("max_tokens", 4000)
    threshold = config.get("summarization_threshold", 1500)  # Reduced from 3000
    preserve_recent = config.get("preserve_recent_messages", 3)  # Reduced from 5
    max_tool_result_chars = config.get("max_tool_result_chars", 2000)

    # Step 1: Compress tool results FIRST (most impactful)
    original_token_count = count_messages_tokens(messages)
    messages = compress_tool_messages(messages, max_tool_result_chars)
    total_tokens = count_messages_tokens(messages)

    tool_compression_saved = original_token_count - total_tokens

    if total_tokens <= threshold:
        return messages, {
            "strategy": "tool_compression_only",
            "original_tokens": original_token_count,
            "reduced_tokens": total_tokens,
            "tool_compression_saved": tool_compression_saved
        }

    # Group messages into atomic units
    message_groups = group_messages(messages)

    # Identify pinned groups (system + first user message)
    pinned_groups = []
    middle_groups = []
    recent_groups = []

    first_user_seen = False
    messages_from_end = 0

    # First pass: identify pinned messages (system and first user)
    for i, group in enumerate(message_groups):
        if len(group) == 1:
            msg = group[0]
            role = msg.get("role")

            # Pin system messages
            if role == "system":
                pinned_groups.append(group)
                continue

            # Pin first user message
            if role == "user" and not first_user_seen:
                pinned_groups.append(group)
                first_user_seen = True
                continue

    # Second pass: separate middle and recent (working backwards)
    for i in range(len(message_groups) - 1, -1, -1):
        group = message_groups[i]

        # Skip if already pinned
        if group in pinned_groups:
            continue

        group_msg_count = len(group)

        if messages_from_end < preserve_recent:
            recent_groups.insert(0, group)
            messages_from_end += group_msg_count
        else:
            middle_groups.insert(0, group)

    # Flatten groups back to messages
    middle_messages = [msg for group in middle_groups for msg in group]

    # Summarize middle messages
    if middle_messages:
        summary_text = summarize_messages_with_llm(middle_messages, llm_config, sandbox_id)
        summary_message = {
            "role": "system",
            "content": f"[Summary]: {summary_text}"
        }

        pinned_msgs_flat = [msg for group in pinned_groups for msg in group]
        recent_msgs_flat = [msg for group in recent_groups for msg in group]
        reduced_messages = pinned_msgs_flat + [summary_message] + recent_msgs_flat
    else:
        pinned_msgs_flat = [msg for group in pinned_groups for msg in group]
        recent_msgs_flat = [msg for group in recent_groups for msg in group]
        reduced_messages = pinned_msgs_flat + recent_msgs_flat

    # Validate the structure before returning
    if not validate_message_structure(reduced_messages):
        logger.warning("Message structure validation failed after hybrid reduction, returning original messages")
        return messages, {
            "strategy": "hybrid_failed",
            "original_tokens": original_token_count,
            "reduced_tokens": total_tokens,
            "error": "Structure validation failed"
        }

    reduced_tokens = count_messages_tokens(reduced_messages)

    return reduced_messages, {
        "strategy": "hybrid",
        "original_tokens": original_token_count,
        "reduced_tokens": reduced_tokens,
        "messages_summarized": len(middle_messages),
        "tool_compression_saved": tool_compression_saved,
        "compression_ratio": round(reduced_tokens / original_token_count, 2) if original_token_count > 0 else 1.0
    }



def apply_context_strategy(
    messages: List[Dict[str, Any]],
    context_config: Dict[str, Any],
    llm_config: Dict[str, Any],
    sandbox_id: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Main entry point for context reduction.

    Args:
        messages: List of conversation messages
        context_config: Context management configuration
        llm_config: LLM configuration for summarization
        sandbox_id: Optional sandbox ID for caching

    Returns:
        Tuple of (reduced_messages, stats_dict)
    """
    if not context_config.get("enabled", True):
        total_tokens = count_messages_tokens(messages)
        return messages, {
            "strategy": "disabled",
            "original_tokens": total_tokens,
            "reduced_tokens": total_tokens
        }

    strategy = context_config.get("strategy", "hybrid")

    try:
        if strategy == "rolling_window":
            return apply_rolling_window_strategy(messages, context_config, llm_config, sandbox_id)
        elif strategy == "truncation":
            return apply_truncation_strategy(messages, context_config)
        elif strategy == "hybrid":
            return apply_hybrid_strategy(messages, context_config, llm_config, sandbox_id)
        else:
            # Unknown strategy, return original
            logger.warning(f"Unknown context strategy: {strategy}")
            total_tokens = count_messages_tokens(messages)
            return messages, {
                "strategy": "none",
                "original_tokens": total_tokens,
                "reduced_tokens": total_tokens
            }
    except Exception as e:
        logger.error(f"Error applying context strategy: {e}")
        # On error, return original messages
        total_tokens = count_messages_tokens(messages)
        return messages, {
            "strategy": "error",
            "original_tokens": total_tokens,
            "reduced_tokens": total_tokens,
            "error": str(e)
        }


# Context Scoping - NEW FEATURE
class ContextScope:
    """
    Manages scoped contexts for tasks and subtasks.
    Allows limiting context to specific task boundaries.
    """

    def __init__(self):
        self.scopes = {}  # scope_id -> scope_data
        self.active_scope = None

    def create_scope(self, scope_id: str, initial_messages: List[Dict[str, Any]], description: str = ""):
        """Create a new context scope."""
        self.scopes[scope_id] = {
            "id": scope_id,
            "description": description,
            "messages": initial_messages.copy(),
            "created_at": None,  # Could add timestamp
        }
        logger.info(f"Created context scope '{scope_id}': {description}")

    def enter_scope(self, scope_id: str):
        """Enter a specific scope."""
        if scope_id not in self.scopes:
            logger.warning(f"Scope '{scope_id}' does not exist")
            return False
        self.active_scope = scope_id
        logger.info(f"Entered scope '{scope_id}'")
        return True

    def exit_scope(self):
        """Exit current scope."""
        old_scope = self.active_scope
        self.active_scope = None
        logger.info(f"Exited scope '{old_scope}'")

    def get_scoped_messages(self, full_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Get messages limited to the active scope.
        If no active scope, return all messages.
        """
        if not self.active_scope or self.active_scope not in self.scopes:
            return full_messages

        scope_data = self.scopes[self.active_scope]
        scope_start_messages = scope_data["messages"]

        # Find where scope starts in full message list
        # Return only messages from that point onward
        # This is a simple implementation - could be enhanced
        return scope_start_messages + full_messages[len(scope_start_messages):]

    def update_scope_messages(self, scope_id: str, messages: List[Dict[str, Any]]):
        """Update messages for a specific scope."""
        if scope_id in self.scopes:
            self.scopes[scope_id]["messages"] = messages

    def delete_scope(self, scope_id: str):
        """Delete a scope."""
        if scope_id in self.scopes:
            del self.scopes[scope_id]
            if self.active_scope == scope_id:
                self.active_scope = None
            logger.info(f"Deleted scope '{scope_id}'")


# Global context scope manager (can be instantiated per agent if needed)
global_context_scope = ContextScope()
