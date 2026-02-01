#!/usr/bin/env python3
"""Session start hook for Claude Memory.

Runs at session start to inject relevant context from the memory system.
This script is configured as a SessionStart hook in .claude/settings.json.

Output is printed to stdout and becomes part of the system context.

Features:
- Injects user preferences
- Injects standing instructions
- Injects high-importance memories
- Injects recent context
- Respects character budget (~2000 chars to avoid prompt bloat)
- Gracefully handles missing dependencies
"""

import sys
from pathlib import Path

# Add scripts directory to path for imports
SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPTS_DIR))


def get_context_injection(max_chars: int = 2000) -> str:
    """Get context to inject at session start.

    Returns formatted markdown string within character budget.
    """
    try:
        from memory_db import MemoryDatabase
    except ImportError:
        return ""

    # Check if database exists
    db_path = Path(".claude/rlm_state/memory.db")
    if not db_path.exists():
        return ""

    try:
        db = MemoryDatabase(db_path)
        context = db.get_context_for_session(max_chars=max_chars)
        db.close()
        return context
    except Exception:
        return ""


def inject_vector_context(query: str | None = None, max_chars: int = 500) -> str:
    """Optionally inject vector-similar context if available.

    Args:
        query: Optional query to search for relevant context
        max_chars: Character budget for this section

    Returns:
        Formatted context string
    """
    try:
        from embeddings import embeddings_available, EmbeddingIndex
        from memory_db import MemoryDatabase
    except ImportError:
        return ""

    if not embeddings_available():
        return ""

    db_path = Path(".claude/rlm_state/memory.db")
    if not db_path.exists():
        return ""

    try:
        # If we have a query, search for relevant memories
        if query:
            index = EmbeddingIndex()
            results = index.search(query, k=3)

            if results:
                parts = ["## Relevant Context"]
                char_count = len(parts[0])

                for r in results:
                    entry = f"- {r.get('text', '')[:100]}"
                    if char_count + len(entry) < max_chars:
                        parts.append(entry)
                        char_count += len(entry)

                return '\n'.join(parts)
    except Exception:
        pass

    return ""


def main():
    """Main entry point for session start hook."""
    # Get main context
    context = get_context_injection(max_chars=2000)

    if context:
        # Output as system context
        print("<rlm-memory-context>")
        print(context)
        print("</rlm-memory-context>")

    return 0


if __name__ == '__main__':
    sys.exit(main())
