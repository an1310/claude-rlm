#!/usr/bin/env python3
"""Session stop hook for Claude Memory.

Runs at session end to capture learnings from the conversation.
This script is configured as a Stop hook in .claude/settings.json.

The conversation transcript is passed via stdin.

Features:
- Pattern-based capture for explicit "remember" phrases
- Extracts preferences, decisions, and facts
- Stores in memory database with appropriate categorization
- Optionally generates embeddings for semantic search
- Gracefully handles missing dependencies

Capture Patterns:
- "remember that..." / "remember this..." -> explicit memory
- "I prefer..." / "I like..." -> preference
- "always..." / "never..." -> instruction
- "we decided..." / "the decision is..." -> decision
"""

import json
import re
import sys
from pathlib import Path
from typing import Any

# Add scripts directory to path for imports
SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPTS_DIR))


# =============================================================================
# Pattern Definitions for Memory Extraction
# =============================================================================

# Patterns for explicit "remember" requests
REMEMBER_PATTERNS = [
    re.compile(r'remember\s+that\s+(.+?)(?:\.|$)', re.IGNORECASE),
    re.compile(r'remember\s+this[:\s]+(.+?)(?:\.|$)', re.IGNORECASE),
    re.compile(r'don\'t\s+forget\s+(?:that\s+)?(.+?)(?:\.|$)', re.IGNORECASE),
    re.compile(r'keep\s+in\s+mind\s+(?:that\s+)?(.+?)(?:\.|$)', re.IGNORECASE),
    re.compile(r'note\s+that\s+(.+?)(?:\.|$)', re.IGNORECASE),
]

# Patterns for preferences
PREFERENCE_PATTERNS = [
    re.compile(r'I\s+prefer\s+(.+?)(?:\s+over\s+.+?)?(?:\.|$)', re.IGNORECASE),
    re.compile(r'I\s+(?:really\s+)?like\s+(.+?)(?:\.|$)', re.IGNORECASE),
    re.compile(r'I\s+(?:really\s+)?don\'t\s+like\s+(.+?)(?:\.|$)', re.IGNORECASE),
    re.compile(r'my\s+preference\s+is\s+(.+?)(?:\.|$)', re.IGNORECASE),
    re.compile(r'I\s+want\s+you\s+to\s+(.+?)(?:\.|$)', re.IGNORECASE),
]

# Patterns for standing instructions
INSTRUCTION_PATTERNS = [
    re.compile(r'always\s+(.+?)(?:\.|$)', re.IGNORECASE),
    re.compile(r'never\s+(.+?)(?:\.|$)', re.IGNORECASE),
    re.compile(r'make\s+sure\s+(?:to\s+)?(.+?)(?:\.|$)', re.IGNORECASE),
    re.compile(r'from\s+now\s+on[,\s]+(.+?)(?:\.|$)', re.IGNORECASE),
]

# Patterns for decisions
DECISION_PATTERNS = [
    re.compile(r'we\s+(?:have\s+)?decided\s+(?:to\s+)?(.+?)(?:\.|$)', re.IGNORECASE),
    re.compile(r'the\s+decision\s+is\s+(.+?)(?:\.|$)', re.IGNORECASE),
    re.compile(r'let\'s\s+go\s+with\s+(.+?)(?:\.|$)', re.IGNORECASE),
    re.compile(r'we\'ll\s+use\s+(.+?)(?:\s+for\s+.+?)?(?:\.|$)', re.IGNORECASE),
]


# =============================================================================
# Memory Extraction
# =============================================================================

def extract_memories_from_text(text: str) -> list[dict[str, Any]]:
    """Extract memories from conversation text using pattern matching.

    Returns list of dicts with keys: content, memory_type, importance, source
    """
    memories = []
    seen_contents = set()

    def add_memory(content: str, memory_type: str, importance: float = 0.5,
                   raw_content: str = None):
        """Add a memory if not already seen.

        Args:
            content: Final content to store
            memory_type: Type of memory
            importance: Importance score
            raw_content: Raw extracted content for length checking (uses content if not provided)
        """
        content = content.strip()
        check_content = (raw_content or content).strip()
        # Skip very short or very long matches (check raw content length)
        if len(check_content) < 10 or len(check_content) > 500:
            return
        # Skip if already captured
        content_lower = content.lower()
        if content_lower in seen_contents:
            return
        seen_contents.add(content_lower)

        memories.append({
            'content': content,
            'memory_type': memory_type,
            'importance': importance,
            'source': 'extracted',
        })

    # Extract explicit "remember" requests (highest priority)
    for pattern in REMEMBER_PATTERNS:
        for match in pattern.finditer(text):
            content = match.group(1)
            add_memory(content, 'fact', importance=0.9)

    # Extract preferences
    for pattern in PREFERENCE_PATTERNS:
        for match in pattern.finditer(text):
            content = match.group(1)
            add_memory(f"User prefers: {content}", 'preference', importance=0.7,
                      raw_content=content)

    # Extract standing instructions
    for pattern in INSTRUCTION_PATTERNS:
        for match in pattern.finditer(text):
            content = match.group(1)
            # Skip common false positives
            if content.lower().startswith(('the ', 'a ', 'an ', 'this ')):
                continue
            add_memory(content, 'instruction', importance=0.8)

    # Extract decisions
    for pattern in DECISION_PATTERNS:
        for match in pattern.finditer(text):
            content = match.group(1)
            add_memory(f"Decision: {content}", 'decision', importance=0.7,
                      raw_content=content)

    return memories


def parse_conversation_transcript(transcript: str) -> tuple[list[str], list[str]]:
    """Parse conversation transcript into user and assistant messages.

    Returns (user_messages, assistant_messages)
    """
    user_messages = []
    assistant_messages = []

    # Try to detect common transcript formats
    # Format 1: "User: ..." / "Assistant: ..."
    user_pattern = re.compile(r'^(?:User|Human|You):\s*(.+?)(?=^(?:Assistant|Claude|AI):|$)', re.MULTILINE | re.DOTALL | re.IGNORECASE)
    assistant_pattern = re.compile(r'^(?:Assistant|Claude|AI):\s*(.+?)(?=^(?:User|Human|You):|$)', re.MULTILINE | re.DOTALL | re.IGNORECASE)

    for match in user_pattern.finditer(transcript):
        user_messages.append(match.group(1).strip())

    for match in assistant_pattern.finditer(transcript):
        assistant_messages.append(match.group(1).strip())

    # If no structured format found, treat entire transcript as conversation
    if not user_messages and not assistant_messages:
        # Just return the whole transcript in both
        user_messages = [transcript]

    return user_messages, assistant_messages


def extract_from_user_messages(user_messages: list[str]) -> list[dict]:
    """Extract memories from user messages only."""
    combined = '\n'.join(user_messages)
    return extract_memories_from_text(combined)


# =============================================================================
# Storage
# =============================================================================

def store_memories(memories: list[dict]) -> int:
    """Store extracted memories in the database.

    Returns number of memories stored.
    """
    if not memories:
        return 0

    try:
        from memory_db import MemoryDatabase, MemoryType, MemorySource
    except ImportError:
        return 0

    try:
        db = MemoryDatabase()

        stored = 0
        for mem in memories:
            # Map string type to enum
            memory_type = MemoryType(mem['memory_type'])
            source = MemorySource(mem.get('source', 'extracted'))

            db.add_memory(
                content=mem['content'],
                memory_type=memory_type,
                source=source,
                importance=mem.get('importance', 0.5),
            )
            stored += 1

        db.close()
        return stored

    except Exception:
        return 0


def add_embeddings_for_memories(memories: list[dict]) -> int:
    """Add embeddings for memories if available.

    Returns number of embeddings added.
    """
    try:
        from embeddings import embeddings_available, EmbeddingIndex
        from memory_db import MemoryDatabase
    except ImportError:
        return 0

    if not embeddings_available():
        return 0

    try:
        # Get memories that need embeddings
        db = MemoryDatabase()
        recent = db.get_recent_memories(limit=len(memories) + 5)
        needs_embedding = [m for m in recent if m.embedding_id is None]

        if not needs_embedding:
            db.close()
            return 0

        # Add embeddings
        index = EmbeddingIndex()
        added = 0

        for mem in needs_embedding:
            eid = index.add_text(
                mem.content,
                metadata={'memory_id': mem.memory_id}
            )
            db.update_memory(mem.memory_id, metadata={'embedding_id': eid})
            # Also update embedding_id directly
            db.execute(
                "UPDATE memories SET embedding_id = ? WHERE memory_id = ?",
                (eid, mem.memory_id)
            )
            added += 1

        db.close()
        return added

    except Exception:
        return 0


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point for session stop hook.

    Reads conversation transcript from stdin and extracts memories.
    """
    # Read transcript from stdin
    try:
        transcript = sys.stdin.read()
    except Exception:
        return 0

    if not transcript or len(transcript) < 50:
        return 0

    # Parse into messages
    user_messages, _ = parse_conversation_transcript(transcript)

    # Extract memories from user messages
    memories = extract_from_user_messages(user_messages)

    if not memories:
        return 0

    # Store memories
    stored = store_memories(memories)

    # Add embeddings if available
    embedded = add_embeddings_for_memories(memories)

    # Output summary (visible in logs)
    if stored > 0:
        print(f"Captured {stored} memories from session", file=sys.stderr)
        if embedded > 0:
            print(f"Added {embedded} embeddings", file=sys.stderr)

    return 0


if __name__ == '__main__':
    sys.exit(main())
