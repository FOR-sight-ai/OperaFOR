# Git Integration Implementation Summary

## Overview

This implementation adds comprehensive git version control functionality to the OperaFOR sandbox system using the `dulwich` Python library. Every agent interaction and tool usage is now automatically tracked in git commits, providing full history and rollback capabilities.

## Key Features Implemented

### 1. **Automatic Git Repository Initialization**
- Each sandbox automatically gets a git repository when first used
- Located in `SANDBOXES_DIR/sandbox_id/.git`
- No manual setup required

### 2. **Automatic Commits After Agent Interactions**
- Every agent response triggers a git commit
- Includes all files in the sandbox folder
- Automatically creates and commits `conversation.json` with full message history
- Commit messages include truncated user prompt for context

### 3. **Tool Integration**
- MCP tools (`convert_urls_to_markdown`, `search_folder`) automatically commit their changes
- Each tool execution gets its own commit with descriptive message

### 4. **Commit History Tracking**
- Each commit hash is stored in the sandbox JSON data structure
- Includes metadata: step number, hash, message, timestamp
- Available via REST API endpoint: `GET /sandboxes/{conv_id}/commits`

### 5. **Sandbox Revert Functionality** 
- Revert any sandbox to a previous commit state
- Available via REST API: `POST /sandboxes/{conv_id}/revert`
- Can revert by commit hash or step number
- Automatically updates conversation messages to match reverted state

## Technical Implementation

### Files Added/Modified

1. **`pyproject.toml`** - Added `dulwich>=0.21.7` dependency
2. **`git_utils.py`** - New module containing core git functions:
   - `init_or_get_repo()` - Initialize or get existing repository
   - `write_conversation_json()` - Write messages to conversation.json
   - `commit_sandbox_changes()` - Commit all sandbox changes
   - `revert_sandbox_to_commit()` - Revert to specific commit
3. **`main.py`** - Updated with git integration:
   - Import git utilities
   - Modified `runAgent()` to commit after each interaction
   - Updated MCP tools to commit changes
   - Added new API endpoints for git operations

### API Endpoints Added

- `GET /sandboxes/{conv_id}/commits` - Get commit history
- `POST /sandboxes/{conv_id}/revert` - Revert to previous commit
  - Body: `{"commit_hash": "hash"}` or `{"step": number}`

### Data Structure Changes

Sandbox objects now include a `commits` array:
```json
{
  "id": "sandbox_id",
  "title": "Sandbox Title", 
  "messages": [...],
  "commits": [
    {
      "step": 0,
      "hash": "commit_hash",
      "message": "commit message",
      "timestamp": "2024-01-01T12:00:00"
    }
  ]
}
```

## Usage Examples

### Automatic Behavior
- No changes needed - all agent interactions automatically create commits
- Every sandbox response creates a new commit with conversation.json
- Tool usage (URL conversion, folder search) creates dedicated commits

### Manual Revert via API
```bash
# Revert to specific commit
curl -X POST "http://localhost:9001/sandboxes/sandbox_id/revert" \
  -H "Content-Type: application/json" \
  -d '{"commit_hash": "a3b0d3948a2041f11e361b22dcb7527b3b5ed9ca"}'

# Revert to specific step
curl -X POST "http://localhost:9001/sandboxes/sandbox_id/revert" \
  -H "Content-Type: application/json" \
  -d '{"step": 2}'
```

### Get Commit History
```bash
curl "http://localhost:9001/sandboxes/sandbox_id/commits"
```

## Benefits

1. **Full Audit Trail** - Every change is tracked with timestamps and descriptions
2. **Easy Rollback** - Instantly revert to any previous state
3. **Conversation Persistence** - Full conversation history stored in each commit
4. **Tool Integration** - All tool outputs are versioned alongside conversations  
5. **No Manual Work** - Everything happens automatically

## Testing

The implementation has been thoroughly tested with:
- Repository initialization
- File creation and modification
- Commit creation and hash generation
- Revert functionality verification
- Conversation.json persistence

All tests pass successfully, confirming the git integration works as designed.