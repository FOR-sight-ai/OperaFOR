"""
Context Management Module for OperaFOR

Implements various strategies for reducing LLM context window usage:
- Token counting and estimation
- Rolling window summarization
- Hybrid memory (preserve important messages)
- Truncation strategies
- Semantic filtering
"""

import json
import logging
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for a given text.
    Uses a simple heuristic: ~4 characters per token (conservative estimate).
    For more accuracy, could integrate tiktoken library.
    """
    if not text:
        return 0
    # Simple heuristic: average of 4 chars per token
    # This is conservative (actual is often 3-3.5 for English)
    return len(text) // 4 + 1


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


def summarize_messages_with_llm(
    messages: List[Dict[str, Any]], 
    llm_config: Dict[str, Any]
) -> str:
    """
    Use the LLM to create a summary of the given messages.
    Returns a concise summary string.
    """
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
            conversation_text.append(f"[Tool executed: {msg.get('name', 'unknown')}]")
        elif role == "assistant" and "tool_calls" in msg:
            tool_names = [tc["function"]["name"] for tc in msg.get("tool_calls", [])]
            conversation_text.append(f"Assistant: [Called tools: {', '.join(tool_names)}]")
            if content:
                conversation_text.append(f"  Thought: {content}")
        else:
            conversation_text.append(f"{role.capitalize()}: {content}")
    
    conversation_str = "\n".join(conversation_text)
    
    summarization_prompt = f"""Summarize the following conversation concisely, preserving key information, decisions, and context. Focus on:
- Main topics discussed
- Important decisions or conclusions
- Key facts or data mentioned
- Current state or progress

Conversation:
{conversation_str}

Provide a concise summary in 2-4 sentences:"""
    
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
        "max_tokens": 300
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
        return f"[Previous conversation with {len(messages)} messages]"


def apply_rolling_window_strategy(
    messages: List[Dict[str, Any]],
    config: Dict[str, Any],
    llm_config: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Apply rolling window summarization strategy.
    
    Returns:
        Tuple of (reduced_messages, stats)
    """
    max_tokens = config.get("max_tokens", 4000)
    threshold = config.get("summarization_threshold", 3000)
    preserve_recent = config.get("preserve_recent_messages", 5)
    preserve_system = config.get("preserve_system_prompt", True)
    
    total_tokens = count_messages_tokens(messages)
    
    # If under threshold, no reduction needed
    if total_tokens <= threshold:
        return messages, {
            "strategy": "none",
            "original_tokens": total_tokens,
            "reduced_tokens": total_tokens,
            "messages_summarized": 0
        }
    
    # Separate system messages, old messages, and recent messages
    system_messages = []
    middle_messages = []
    recent_messages = []
    
    for i, msg in enumerate(messages):
        if preserve_system and msg.get("role") == "system":
            system_messages.append(msg)
        elif i >= len(messages) - preserve_recent:
            recent_messages.append(msg)
        else:
            if msg.get("role") != "system":  # Don't duplicate system messages
                middle_messages.append(msg)
    
    # If we still need to reduce, summarize the middle messages
    if middle_messages:
        logger.info(f"Summarizing {len(middle_messages)} messages to reduce context")
        summary_text = summarize_messages_with_llm(middle_messages, llm_config)
        
        # Create a summary message
        summary_message = {
            "role": "system",
            "content": f"[Conversation Summary - {len(middle_messages)} messages compressed]: {summary_text}"
        }
        
        # Reconstruct message list
        reduced_messages = system_messages + [summary_message] + recent_messages
    else:
        reduced_messages = system_messages + recent_messages
    
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
    
    Returns:
        Tuple of (reduced_messages, stats)
    """
    max_tokens = config.get("max_tokens", 4000)
    preserve_system = config.get("preserve_system_prompt", True)
    
    total_tokens = count_messages_tokens(messages)
    
    # Separate system messages
    system_messages = [msg for msg in messages if msg.get("role") == "system"] if preserve_system else []
    non_system_messages = [msg for msg in messages if msg.get("role") != "system"]
    
    # Calculate budget for non-system messages
    system_tokens = count_messages_tokens(system_messages)
    remaining_budget = max_tokens - system_tokens
    
    # Take messages from the end until we hit the budget
    reduced_non_system = []
    current_tokens = 0
    
    for msg in reversed(non_system_messages):
        msg_tokens = count_message_tokens(msg)
        if current_tokens + msg_tokens <= remaining_budget:
            reduced_non_system.insert(0, msg)
            current_tokens += msg_tokens
        else:
            break
    
    reduced_messages = system_messages + reduced_non_system
    reduced_tokens = count_messages_tokens(reduced_messages)
    
    return reduced_messages, {
        "strategy": "truncation",
        "original_tokens": total_tokens,
        "reduced_tokens": reduced_tokens,
        "messages_removed": len(non_system_messages) - len(reduced_non_system)
    }


def apply_hybrid_strategy(
    messages: List[Dict[str, Any]],
    config: Dict[str, Any],
    llm_config: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Apply hybrid strategy - preserve pinned messages + summarize middle + keep recent.
    
    Returns:
        Tuple of (reduced_messages, stats)
    """
    max_tokens = config.get("max_tokens", 4000)
    threshold = config.get("summarization_threshold", 3000)
    preserve_recent = config.get("preserve_recent_messages", 5)
    
    total_tokens = count_messages_tokens(messages)
    
    if total_tokens <= threshold:
        return messages, {
            "strategy": "none",
            "original_tokens": total_tokens,
            "reduced_tokens": total_tokens
        }
    
    # Identify pinned messages (system + first user message)
    pinned_messages = []
    middle_messages = []
    recent_messages = []
    
    for i, msg in enumerate(messages):
        role = msg.get("role")
        
        # Pin system messages and first user message
        if role == "system" or (role == "user" and i == 0) or (role == "user" and i == 1):
            pinned_messages.append(msg)
        elif i >= len(messages) - preserve_recent:
            recent_messages.append(msg)
        else:
            middle_messages.append(msg)
    
    # Summarize middle messages
    if middle_messages:
        summary_text = summarize_messages_with_llm(middle_messages, llm_config)
        summary_message = {
            "role": "system",
            "content": f"[Context Summary]: {summary_text}"
        }
        reduced_messages = pinned_messages + [summary_message] + recent_messages
    else:
        reduced_messages = pinned_messages + recent_messages
    
    reduced_tokens = count_messages_tokens(reduced_messages)
    
    return reduced_messages, {
        "strategy": "hybrid",
        "original_tokens": total_tokens,
        "reduced_tokens": reduced_tokens,
        "messages_summarized": len(middle_messages),
        "compression_ratio": round(reduced_tokens / total_tokens, 2) if total_tokens > 0 else 1.0
    }


def apply_context_strategy(
    messages: List[Dict[str, Any]],
    context_config: Dict[str, Any],
    llm_config: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Main entry point for context reduction.
    
    Args:
        messages: List of conversation messages
        context_config: Context management configuration
        llm_config: LLM configuration for summarization
    
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
    
    strategy = context_config.get("strategy", "rolling_window")
    
    try:
        if strategy == "rolling_window":
            return apply_rolling_window_strategy(messages, context_config, llm_config)
        elif strategy == "truncation":
            return apply_truncation_strategy(messages, context_config)
        elif strategy == "hybrid":
            return apply_hybrid_strategy(messages, context_config, llm_config)
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
