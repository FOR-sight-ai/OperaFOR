"""
Context Cache Module for OperaFOR

Implements file-based caching for conversation summaries to reduce LLM calls.
Each sandbox has a .context_cache/ directory containing:
- metadata.json: Cache index and metadata
- summaries/: Individual summary segment files
- stats.json: Usage statistics
"""

import json
import logging
import hashlib
import os
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


def get_cache_path(sandbox_id: str) -> Path:
    """Get the cache directory path for a sandbox."""
    from utils import get_sandbox_path
    sandbox_path = Path(get_sandbox_path(sandbox_id))
    cache_path = sandbox_path / ".context_cache"
    return cache_path


def ensure_cache_directory(sandbox_id: str) -> Path:
    """Ensure cache directory exists and return its path."""
    cache_path = get_cache_path(sandbox_id)
    cache_path.mkdir(parents=True, exist_ok=True)
    (cache_path / "summaries").mkdir(exist_ok=True)
    return cache_path


def compute_message_hash(messages: List[Dict[str, Any]]) -> str:
    """
    Compute a hash of message content to detect changes.
    Only hashes role and content to detect meaningful changes.
    """
    # Create a stable representation of messages
    stable_repr = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        
        # Handle multimodal content
        if isinstance(content, list):
            content_str = json.dumps(content, sort_keys=True)
        else:
            content_str = str(content)
        
        stable_repr.append(f"{role}:{content_str}")
    
    # Hash the representation
    combined = "|".join(stable_repr)
    return hashlib.sha256(combined.encode()).hexdigest()


def load_cache_metadata(sandbox_id: str) -> Dict[str, Any]:
    """Load cache metadata for a sandbox."""
    cache_path = get_cache_path(sandbox_id)
    metadata_file = cache_path / "metadata.json"
    
    if not metadata_file.exists():
        # Return default metadata
        return {
            "version": "1.0",
            "last_updated": datetime.utcnow().isoformat(),
            "total_messages_cached": 0,
            "segments": []
        }
    
    try:
        with open(metadata_file, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading cache metadata: {e}")
        return {
            "version": "1.0",
            "last_updated": datetime.utcnow().isoformat(),
            "total_messages_cached": 0,
            "segments": []
        }


def save_cache_metadata(sandbox_id: str, metadata: Dict[str, Any]) -> None:
    """Save cache metadata for a sandbox."""
    try:
        cache_path = ensure_cache_directory(sandbox_id)
        metadata_file = cache_path / "metadata.json"
        
        metadata["last_updated"] = datetime.utcnow().isoformat()
        
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving cache metadata: {e}")


def load_stats(sandbox_id: str) -> Dict[str, Any]:
    """Load cache usage statistics."""
    cache_path = get_cache_path(sandbox_id)
    stats_file = cache_path / "stats.json"
    
    if not stats_file.exists():
        return {
            "cache_hits": 0,
            "cache_misses": 0,
            "summaries_generated": 0,
            "summaries_reused": 0,
            "total_tokens_saved": 0
        }
    
    try:
        with open(stats_file, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading stats: {e}")
        return {
            "cache_hits": 0,
            "cache_misses": 0,
            "summaries_generated": 0,
            "summaries_reused": 0,
            "total_tokens_saved": 0
        }


def save_stats(sandbox_id: str, stats: Dict[str, Any]) -> None:
    """Save cache usage statistics."""
    try:
        cache_path = ensure_cache_directory(sandbox_id)
        stats_file = cache_path / "stats.json"
        
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving stats: {e}")


def get_cached_summary(
    sandbox_id: str, 
    message_range: Tuple[int, int],
    message_hash: Optional[str] = None
) -> Optional[str]:
    """
    Retrieve cached summary for a message range.
    
    Args:
        sandbox_id: Sandbox identifier
        message_range: Tuple of (start_index, end_index)
        message_hash: Optional hash to validate cache freshness
    
    Returns:
        Cached summary string or None if not found/invalid
    """
    try:
        metadata = load_cache_metadata(sandbox_id)
        
        # Find matching segment
        for segment in metadata.get("segments", []):
            seg_range = tuple(segment.get("message_range", []))
            if seg_range == message_range:
                # Check hash if provided
                if message_hash and segment.get("hash") != message_hash:
                    logger.info(f"Cache invalidated for range {message_range} (hash mismatch)")
                    # Record cache miss for invalidated cache
                    stats = load_stats(sandbox_id)
                    stats["cache_misses"] = stats.get("cache_misses", 0) + 1
                    save_stats(sandbox_id, stats)
                    return None
                
                # Load summary file
                cache_path = get_cache_path(sandbox_id)
                segment_id = segment.get("segment_id")
                summary_file = cache_path / "summaries" / f"{segment_id}.json"
                
                if summary_file.exists():
                    with open(summary_file, "r") as f:
                        data = json.load(f)
                        
                        # Update stats
                        stats = load_stats(sandbox_id)
                        stats["cache_hits"] = stats.get("cache_hits", 0) + 1
                        stats["summaries_reused"] = stats.get("summaries_reused", 0) + 1
                        save_stats(sandbox_id, stats)
                        
                        logger.info(f"Cache hit for range {message_range}")
                        return data.get("summary")
        
        # Update stats for cache miss
        stats = load_stats(sandbox_id)
        stats["cache_misses"] = stats.get("cache_misses", 0) + 1
        save_stats(sandbox_id, stats)
        
        return None
    except Exception as e:
        logger.error(f"Error retrieving cached summary: {e}")
        return None


def save_summary_segment(
    sandbox_id: str,
    segment_id: str,
    summary: str,
    segment_metadata: Dict[str, Any]
) -> None:
    """
    Save a summary segment to cache.
    
    Args:
        sandbox_id: Sandbox identifier
        segment_id: Unique segment identifier (e.g., "segment_0")
        summary: Summary text
        segment_metadata: Metadata including message_range, token_count, hash, etc.
    """
    try:
        cache_path = ensure_cache_directory(sandbox_id)
        
        # Save summary file
        summary_file = cache_path / "summaries" / f"{segment_id}.json"
        summary_data = {
            "summary": summary,
            "created_at": datetime.utcnow().isoformat(),
            **segment_metadata
        }
        
        with open(summary_file, "w") as f:
            json.dump(summary_data, f, indent=2)
        
        # Update metadata
        metadata = load_cache_metadata(sandbox_id)
        
        # Remove old segment with same ID if exists
        metadata["segments"] = [
            seg for seg in metadata.get("segments", [])
            if seg.get("segment_id") != segment_id
        ]
        
        # Add new segment
        metadata["segments"].append({
            "segment_id": segment_id,
            "created_at": datetime.utcnow().isoformat(),
            **segment_metadata
        })
        
        # Update total messages cached
        message_range = segment_metadata.get("message_range", [0, 0])
        metadata["total_messages_cached"] = max(
            metadata.get("total_messages_cached", 0),
            message_range[1] + 1
        )
        
        save_cache_metadata(sandbox_id, metadata)
        
        # Update stats
        stats = load_stats(sandbox_id)
        stats["summaries_generated"] = stats.get("summaries_generated", 0) + 1
        save_stats(sandbox_id, stats)
        
        logger.info(f"Saved summary segment {segment_id} for range {message_range}")
    except Exception as e:
        logger.error(f"Error saving summary segment: {e}")


def invalidate_cache(sandbox_id: str) -> None:
    """Clear all cache for a sandbox."""
    try:
        cache_path = get_cache_path(sandbox_id)
        
        if cache_path.exists():
            import shutil
            shutil.rmtree(cache_path)
            logger.info(f"Cache invalidated for sandbox {sandbox_id}")
    except Exception as e:
        logger.error(f"Error invalidating cache: {e}")


def get_incremental_summary(
    sandbox_id: str,
    messages: List[Dict[str, Any]],
    llm_config: Dict[str, Any]
) -> str:
    """
    Get summary using incremental approach - reuse cached summaries and only
    summarize new content.
    
    Args:
        sandbox_id: Sandbox identifier
        messages: Messages to summarize
        llm_config: LLM configuration for generating new summaries
    
    Returns:
        Combined summary string
    """
    from context_manager import summarize_messages_with_llm, count_messages_tokens
    
    if not messages:
        return ""
    
    try:
        metadata = load_cache_metadata(sandbox_id)
        segments = metadata.get("segments", [])
        
        # Sort segments by message range
        segments.sort(key=lambda s: s.get("message_range", [0, 0])[0])
        
        # Determine which messages are already cached
        message_count = len(messages)
        cached_summaries = []
        last_cached_index = -1
        
        for segment in segments:
            seg_range = segment.get("message_range", [0, 0])
            seg_hash = segment.get("hash")
            
            # Check if this segment is within our message range
            if seg_range[1] < message_count:
                # Compute hash for this segment in current messages
                segment_messages = messages[seg_range[0]:seg_range[1]+1]
                current_hash = compute_message_hash(segment_messages)
                
                # If hash matches, we can reuse this summary
                if current_hash == seg_hash:
                    cached_summary = get_cached_summary(sandbox_id, tuple(seg_range), current_hash)
                    if cached_summary:
                        cached_summaries.append(cached_summary)
                        last_cached_index = seg_range[1]
        
        # Determine if we need to summarize new messages
        new_messages = messages[last_cached_index + 1:] if last_cached_index >= 0 else messages
        
        if new_messages:
            # Generate summary for new messages
            new_summary = summarize_messages_with_llm(new_messages, llm_config)
            
            # Save this new segment to cache
            segment_id = f"segment_{len(segments)}"
            message_range = [last_cached_index + 1, message_count - 1]
            message_hash = compute_message_hash(new_messages)
            token_count = count_messages_tokens(new_messages)
            
            from context_manager import estimate_tokens
            summary_tokens = estimate_tokens(new_summary)
            
            save_summary_segment(
                sandbox_id,
                segment_id,
                new_summary,
                {
                    "message_range": message_range,
                    "token_count": token_count,
                    "summary_tokens": summary_tokens,
                    "hash": message_hash
                }
            )
            
            cached_summaries.append(new_summary)
            
            # Update stats with tokens saved
            stats = load_stats(sandbox_id)
            tokens_saved = token_count - summary_tokens
            stats["total_tokens_saved"] = stats.get("total_tokens_saved", 0) + tokens_saved
            save_stats(sandbox_id, stats)
        
        # Combine all summaries
        if len(cached_summaries) > 1:
            combined = " | ".join(cached_summaries)
            return f"[Multi-segment summary]: {combined}"
        elif cached_summaries:
            return cached_summaries[0]
        else:
            return ""
            
    except Exception as e:
        logger.error(f"Error in incremental summarization: {e}")
        # Fallback to regular summarization
        from context_manager import summarize_messages_with_llm
        return summarize_messages_with_llm(messages, llm_config)
