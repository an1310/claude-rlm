#!/usr/bin/env python3
"""Memory database module for Claude Memory.

Provides persistent memory storage with:
- Memories table for learned facts, preferences, patterns
- Profile table for user preferences and settings
- FTS5 full-text search for keyword queries
- Hybrid search combining FTS5 with vector similarity (when embeddings available)

Separate from index.db to maintain clean separation of concerns.
"""

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class MemoryType(Enum):
    """Categories of memories."""
    FACT = "fact"              # Learned facts about codebase/project
    PREFERENCE = "preference"  # User preferences
    PATTERN = "pattern"        # Code patterns to follow
    DECISION = "decision"      # Design decisions made
    CONTEXT = "context"        # Project context
    INSTRUCTION = "instruction"  # Standing instructions


class MemorySource(Enum):
    """How the memory was captured."""
    EXPLICIT = "explicit"      # User explicitly said "remember this"
    EXTRACTED = "extracted"    # Extracted from conversation
    INFERRED = "inferred"      # Inferred from patterns


@dataclass
class Memory:
    """Represents a stored memory."""
    memory_id: int | None
    content: str
    memory_type: MemoryType
    source: MemorySource
    tags: list[str]
    importance: float  # 0.0 to 1.0
    created_at: str
    last_accessed: str
    access_count: int
    embedding_id: int | None = None
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'memory_id': self.memory_id,
            'content': self.content,
            'memory_type': self.memory_type.value,
            'source': self.source.value,
            'tags': self.tags,
            'importance': self.importance,
            'created_at': self.created_at,
            'last_accessed': self.last_accessed,
            'access_count': self.access_count,
            'embedding_id': self.embedding_id,
            'metadata': self.metadata,
        }

    @classmethod
    def from_row(cls, row: dict) -> 'Memory':
        """Create from database row."""
        return cls(
            memory_id=row['memory_id'],
            content=row['content'],
            memory_type=MemoryType(row['memory_type']),
            source=MemorySource(row['source']),
            tags=json.loads(row['tags']) if row['tags'] else [],
            importance=row['importance'],
            created_at=row['created_at'],
            last_accessed=row['last_accessed'],
            access_count=row['access_count'],
            embedding_id=row.get('embedding_id'),
            metadata=json.loads(row['metadata']) if row.get('metadata') else None,
        )


@dataclass
class UserProfile:
    """User profile with preferences."""
    profile_id: int
    name: str | None
    preferences: dict[str, Any]
    created_at: str
    updated_at: str

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'profile_id': self.profile_id,
            'name': self.name,
            'preferences': self.preferences,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
        }


class MemoryDatabase:
    """SQLite database for memory storage with FTS5."""

    DEFAULT_PATH = Path(".claude/rlm_state/memory.db")

    SCHEMA = """
    -- Memory storage
    CREATE TABLE IF NOT EXISTS memories (
        memory_id INTEGER PRIMARY KEY AUTOINCREMENT,
        content TEXT NOT NULL,
        memory_type TEXT NOT NULL,
        source TEXT NOT NULL,
        tags TEXT,  -- JSON array
        importance REAL DEFAULT 0.5,
        created_at TEXT DEFAULT (datetime('now')),
        last_accessed TEXT DEFAULT (datetime('now')),
        access_count INTEGER DEFAULT 0,
        embedding_id INTEGER,  -- Reference to vector index
        metadata TEXT  -- JSON object for extensibility
    );

    -- FTS5 for fast text search on memories
    CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
        content,
        tags,
        memory_id UNINDEXED,
        content='memories',
        content_rowid='memory_id'
    );

    -- Triggers to sync FTS5 with memories table
    CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
        INSERT INTO memories_fts(rowid, content, tags, memory_id)
        VALUES (new.memory_id, new.content, new.tags, new.memory_id);
    END;

    CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
        INSERT INTO memories_fts(memories_fts, rowid, content, tags, memory_id)
        VALUES ('delete', old.memory_id, old.content, old.tags, old.memory_id);
    END;

    CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
        INSERT INTO memories_fts(memories_fts, rowid, content, tags, memory_id)
        VALUES ('delete', old.memory_id, old.content, old.tags, old.memory_id);
        INSERT INTO memories_fts(rowid, content, tags, memory_id)
        VALUES (new.memory_id, new.content, new.tags, new.memory_id);
    END;

    -- User profile storage
    CREATE TABLE IF NOT EXISTS profile (
        profile_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        preferences TEXT,  -- JSON object
        created_at TEXT DEFAULT (datetime('now')),
        updated_at TEXT DEFAULT (datetime('now'))
    );

    -- Session history for context
    CREATE TABLE IF NOT EXISTS sessions (
        session_id INTEGER PRIMARY KEY AUTOINCREMENT,
        started_at TEXT DEFAULT (datetime('now')),
        ended_at TEXT,
        summary TEXT,
        memories_created INTEGER DEFAULT 0,
        metadata TEXT  -- JSON object
    );

    -- Indexes for common queries
    CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);
    CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance DESC);
    CREATE INDEX IF NOT EXISTS idx_memories_accessed ON memories(last_accessed DESC);
    CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at DESC);
    CREATE INDEX IF NOT EXISTS idx_memories_embedding ON memories(embedding_id);
    """

    def __init__(self, db_path: Path | str | None = None):
        """Initialize the memory database."""
        self.db_path = Path(db_path) if db_path else self.DEFAULT_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

        # Enable WAL mode and foreign keys
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA synchronous=NORMAL")

        # Create schema
        self._conn.executescript(self.SCHEMA)
        self._conn.commit()

        # Ensure default profile exists
        self._ensure_default_profile()

    def _ensure_default_profile(self):
        """Create default profile if none exists."""
        result = self._conn.execute("SELECT COUNT(*) FROM profile").fetchone()
        if result[0] == 0:
            self._conn.execute(
                "INSERT INTO profile (name, preferences) VALUES (?, ?)",
                (None, json.dumps({}))
            )
            self._conn.commit()

    # ==========================================================================
    # Memory Operations
    # ==========================================================================

    def add_memory(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.FACT,
        source: MemorySource = MemorySource.EXTRACTED,
        tags: list[str] | None = None,
        importance: float = 0.5,
        embedding_id: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Add a new memory. Returns the memory_id."""
        cursor = self._conn.execute(
            """INSERT INTO memories
               (content, memory_type, source, tags, importance, embedding_id, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                content,
                memory_type.value,
                source.value,
                json.dumps(tags or []),
                importance,
                embedding_id,
                json.dumps(metadata) if metadata else None,
            )
        )
        self._conn.commit()
        return cursor.lastrowid

    def get_memory(self, memory_id: int) -> Memory | None:
        """Get a memory by ID."""
        cursor = self._conn.execute(
            "SELECT * FROM memories WHERE memory_id = ?",
            (memory_id,)
        )
        row = cursor.fetchone()
        if row:
            return Memory.from_row(dict(row))
        return None

    def update_memory(
        self,
        memory_id: int,
        content: str | None = None,
        importance: float | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Update an existing memory."""
        updates = []
        params = []

        if content is not None:
            updates.append("content = ?")
            params.append(content)
        if importance is not None:
            updates.append("importance = ?")
            params.append(importance)
        if tags is not None:
            updates.append("tags = ?")
            params.append(json.dumps(tags))
        if metadata is not None:
            updates.append("metadata = ?")
            params.append(json.dumps(metadata))

        if not updates:
            return False

        params.append(memory_id)
        cursor = self._conn.execute(
            f"UPDATE memories SET {', '.join(updates)} WHERE memory_id = ?",
            tuple(params)
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def delete_memory(self, memory_id: int) -> bool:
        """Delete a memory."""
        cursor = self._conn.execute(
            "DELETE FROM memories WHERE memory_id = ?",
            (memory_id,)
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def touch_memory(self, memory_id: int):
        """Update access time and count for a memory."""
        self._conn.execute(
            """UPDATE memories
               SET last_accessed = datetime('now'), access_count = access_count + 1
               WHERE memory_id = ?""",
            (memory_id,)
        )
        self._conn.commit()

    def search_memories_fts(
        self,
        query: str,
        limit: int = 20,
        memory_type: MemoryType | None = None,
    ) -> list[Memory]:
        """Search memories using FTS5."""
        safe_query = query.replace('"', '""')

        sql = """
            SELECT m.* FROM memories_fts fts
            JOIN memories m ON m.memory_id = fts.memory_id
            WHERE memories_fts MATCH ?
        """
        params: list[Any] = [f'"{safe_query}"']

        if memory_type:
            sql += " AND m.memory_type = ?"
            params.append(memory_type.value)

        sql += " ORDER BY m.importance DESC, m.last_accessed DESC LIMIT ?"
        params.append(limit)

        cursor = self._conn.execute(sql, tuple(params))
        return [Memory.from_row(dict(row)) for row in cursor.fetchall()]

    def get_recent_memories(
        self,
        limit: int = 10,
        memory_type: MemoryType | None = None,
    ) -> list[Memory]:
        """Get most recently accessed memories."""
        sql = "SELECT * FROM memories"
        params: list[Any] = []

        if memory_type:
            sql += " WHERE memory_type = ?"
            params.append(memory_type.value)

        sql += " ORDER BY last_accessed DESC LIMIT ?"
        params.append(limit)

        cursor = self._conn.execute(sql, tuple(params))
        return [Memory.from_row(dict(row)) for row in cursor.fetchall()]

    def get_important_memories(
        self,
        min_importance: float = 0.7,
        limit: int = 20,
    ) -> list[Memory]:
        """Get high-importance memories."""
        cursor = self._conn.execute(
            """SELECT * FROM memories
               WHERE importance >= ?
               ORDER BY importance DESC, last_accessed DESC
               LIMIT ?""",
            (min_importance, limit)
        )
        return [Memory.from_row(dict(row)) for row in cursor.fetchall()]

    def get_memories_by_type(
        self,
        memory_type: MemoryType,
        limit: int = 50,
    ) -> list[Memory]:
        """Get all memories of a specific type."""
        cursor = self._conn.execute(
            """SELECT * FROM memories
               WHERE memory_type = ?
               ORDER BY importance DESC, last_accessed DESC
               LIMIT ?""",
            (memory_type.value, limit)
        )
        return [Memory.from_row(dict(row)) for row in cursor.fetchall()]

    def get_memories_by_tag(self, tag: str, limit: int = 50) -> list[Memory]:
        """Get memories with a specific tag."""
        cursor = self._conn.execute(
            """SELECT * FROM memories
               WHERE tags LIKE ?
               ORDER BY importance DESC, last_accessed DESC
               LIMIT ?""",
            (f'%"{tag}"%', limit)
        )
        return [Memory.from_row(dict(row)) for row in cursor.fetchall()]

    def get_all_memories(self, limit: int = 100) -> list[Memory]:
        """Get all memories, ordered by importance."""
        cursor = self._conn.execute(
            """SELECT * FROM memories
               ORDER BY importance DESC, last_accessed DESC
               LIMIT ?""",
            (limit,)
        )
        return [Memory.from_row(dict(row)) for row in cursor.fetchall()]

    def get_context_for_session(
        self,
        max_chars: int = 2000,
        include_types: list[MemoryType] | None = None,
    ) -> str:
        """Get relevant context for session start injection.

        Prioritizes:
        1. High-importance memories
        2. Recent memories
        3. User preferences

        Returns formatted text within the character budget.
        """
        parts = []
        char_count = 0

        # Get profile preferences first
        profile = self.get_profile()
        if profile and profile.preferences:
            prefs_str = self._format_preferences(profile.preferences)
            if prefs_str:
                parts.append(f"## User Preferences\n{prefs_str}")
                char_count += len(parts[-1])

        # Get standing instructions
        instructions = self.get_memories_by_type(MemoryType.INSTRUCTION, limit=5)
        if instructions:
            instr_str = "\n".join(f"- {m.content}" for m in instructions)
            if char_count + len(instr_str) + 30 < max_chars:
                parts.append(f"## Standing Instructions\n{instr_str}")
                char_count += len(parts[-1])

        # Get high-importance memories
        important = self.get_important_memories(min_importance=0.7, limit=10)
        if important:
            for mem in important:
                if include_types and mem.memory_type not in include_types:
                    continue
                entry = f"- [{mem.memory_type.value}] {mem.content}"
                if char_count + len(entry) + 30 < max_chars:
                    if "## Key Knowledge" not in '\n'.join(parts):
                        parts.append("## Key Knowledge")
                        char_count += len(parts[-1])
                    parts.append(entry)
                    char_count += len(entry)

        # Get recent context
        recent = self.get_recent_memories(limit=5, memory_type=MemoryType.CONTEXT)
        if recent:
            for mem in recent:
                entry = f"- {mem.content}"
                if char_count + len(entry) + 30 < max_chars:
                    if "## Recent Context" not in '\n'.join(parts):
                        parts.append("## Recent Context")
                        char_count += len(parts[-1])
                    parts.append(entry)
                    char_count += len(entry)

        return '\n'.join(parts) if parts else ""

    def _format_preferences(self, prefs: dict[str, Any]) -> str:
        """Format preferences for display."""
        if not prefs:
            return ""
        lines = []
        for key, value in prefs.items():
            if isinstance(value, (list, dict)):
                value = json.dumps(value)
            lines.append(f"- {key}: {value}")
        return '\n'.join(lines)

    # ==========================================================================
    # Profile Operations
    # ==========================================================================

    def get_profile(self) -> UserProfile | None:
        """Get the user profile."""
        cursor = self._conn.execute("SELECT * FROM profile LIMIT 1")
        row = cursor.fetchone()
        if row:
            return UserProfile(
                profile_id=row['profile_id'],
                name=row['name'],
                preferences=json.loads(row['preferences']) if row['preferences'] else {},
                created_at=row['created_at'],
                updated_at=row['updated_at'],
            )
        return None

    def update_profile(
        self,
        name: str | None = None,
        preferences: dict[str, Any] | None = None,
    ) -> bool:
        """Update the user profile."""
        updates = ["updated_at = datetime('now')"]
        params: list[Any] = []

        if name is not None:
            updates.append("name = ?")
            params.append(name)
        if preferences is not None:
            updates.append("preferences = ?")
            params.append(json.dumps(preferences))

        cursor = self._conn.execute(
            f"UPDATE profile SET {', '.join(updates)} WHERE profile_id = 1",
            tuple(params)
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def set_preference(self, key: str, value: Any) -> bool:
        """Set a single preference."""
        profile = self.get_profile()
        if not profile:
            return False

        prefs = profile.preferences.copy()
        prefs[key] = value
        return self.update_profile(preferences=prefs)

    def get_preference(self, key: str, default: Any = None) -> Any:
        """Get a single preference."""
        profile = self.get_profile()
        if not profile:
            return default
        return profile.preferences.get(key, default)

    # ==========================================================================
    # Session Operations
    # ==========================================================================

    def start_session(self, metadata: dict[str, Any] | None = None) -> int:
        """Start a new session. Returns session_id."""
        cursor = self._conn.execute(
            "INSERT INTO sessions (metadata) VALUES (?)",
            (json.dumps(metadata) if metadata else None,)
        )
        self._conn.commit()
        return cursor.lastrowid

    def end_session(
        self,
        session_id: int,
        summary: str | None = None,
        memories_created: int = 0,
    ):
        """End a session."""
        self._conn.execute(
            """UPDATE sessions
               SET ended_at = datetime('now'), summary = ?, memories_created = ?
               WHERE session_id = ?""",
            (summary, memories_created, session_id)
        )
        self._conn.commit()

    def get_session(self, session_id: int) -> dict | None:
        """Get session info."""
        cursor = self._conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?",
            (session_id,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_recent_sessions(self, limit: int = 10) -> list[dict]:
        """Get recent sessions."""
        cursor = self._conn.execute(
            "SELECT * FROM sessions ORDER BY started_at DESC LIMIT ?",
            (limit,)
        )
        return [dict(row) for row in cursor.fetchall()]

    # ==========================================================================
    # Statistics and Maintenance
    # ==========================================================================

    def get_stats(self) -> dict:
        """Get memory database statistics."""
        stats = {}

        # Total memories
        result = self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()
        stats['total_memories'] = result[0]

        # By type
        cursor = self._conn.execute(
            """SELECT memory_type, COUNT(*) as count
               FROM memories GROUP BY memory_type"""
        )
        stats['by_type'] = {row['memory_type']: row['count'] for row in cursor}

        # By source
        cursor = self._conn.execute(
            """SELECT source, COUNT(*) as count
               FROM memories GROUP BY source"""
        )
        stats['by_source'] = {row['source']: row['count'] for row in cursor}

        # Session count
        result = self._conn.execute("SELECT COUNT(*) FROM sessions").fetchone()
        stats['total_sessions'] = result[0]

        # Average importance
        result = self._conn.execute(
            "SELECT AVG(importance) FROM memories"
        ).fetchone()
        stats['avg_importance'] = round(result[0] or 0, 2)

        return stats

    def vacuum(self):
        """Vacuum database and optimize FTS5."""
        self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self._conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('optimize')")
        self._conn.execute("VACUUM")
        self._conn.commit()

    def close(self):
        """Close the database connection."""
        self._conn.close()

    # ==========================================================================
    # Utility Methods
    # ==========================================================================

    def query(self, sql: str, params: tuple = ()) -> list[dict]:
        """Execute a raw query and return results."""
        cursor = self._conn.execute(sql, params)
        columns = [d[0] for d in cursor.description] if cursor.description else []
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        """Execute a SQL statement."""
        cursor = self._conn.execute(sql, params)
        self._conn.commit()
        return cursor


# =============================================================================
# CLI for testing
# =============================================================================

def main():
    """Simple CLI for testing the memory database."""
    import argparse

    parser = argparse.ArgumentParser(description="Memory Database CLI")
    parser.add_argument('--db', default=None, help='Database path')

    sub = parser.add_subparsers(dest='cmd', required=True)

    # add
    p_add = sub.add_parser('add', help='Add a memory')
    p_add.add_argument('content', help='Memory content')
    p_add.add_argument('--type', default='fact', choices=[t.value for t in MemoryType])
    p_add.add_argument('--importance', type=float, default=0.5)
    p_add.add_argument('--tags', help='Comma-separated tags')

    # search
    p_search = sub.add_parser('search', help='Search memories')
    p_search.add_argument('query', help='Search query')
    p_search.add_argument('--limit', type=int, default=10)

    # list
    p_list = sub.add_parser('list', help='List memories')
    p_list.add_argument('--type', choices=[t.value for t in MemoryType])
    p_list.add_argument('--limit', type=int, default=20)

    # stats
    sub.add_parser('stats', help='Show statistics')

    # context
    p_ctx = sub.add_parser('context', help='Get session context')
    p_ctx.add_argument('--max-chars', type=int, default=2000)

    args = parser.parse_args()

    db = MemoryDatabase(args.db)

    try:
        if args.cmd == 'add':
            tags = args.tags.split(',') if args.tags else []
            memory_id = db.add_memory(
                args.content,
                memory_type=MemoryType(args.type),
                importance=args.importance,
                tags=tags,
            )
            print(f"Added memory {memory_id}")

        elif args.cmd == 'search':
            results = db.search_memories_fts(args.query, limit=args.limit)
            if results:
                print(f"Found {len(results)} memories:")
                for m in results:
                    print(f"  [{m.memory_id}] ({m.memory_type.value}) {m.content[:60]}...")
            else:
                print("No memories found")

        elif args.cmd == 'list':
            if args.type:
                results = db.get_memories_by_type(MemoryType(args.type), limit=args.limit)
            else:
                results = db.get_all_memories(limit=args.limit)

            if results:
                print(f"Memories ({len(results)}):")
                for m in results:
                    print(f"  [{m.memory_id}] ({m.memory_type.value}, {m.importance:.1f}) {m.content[:60]}")
            else:
                print("No memories")

        elif args.cmd == 'stats':
            stats = db.get_stats()
            print(f"Memory Database Statistics:")
            print(f"  Total memories: {stats['total_memories']}")
            print(f"  By type: {stats['by_type']}")
            print(f"  By source: {stats['by_source']}")
            print(f"  Total sessions: {stats['total_sessions']}")
            print(f"  Avg importance: {stats['avg_importance']}")

        elif args.cmd == 'context':
            context = db.get_context_for_session(max_chars=args.max_chars)
            if context:
                print(context)
            else:
                print("No context available")

    finally:
        db.close()


if __name__ == '__main__':
    main()
