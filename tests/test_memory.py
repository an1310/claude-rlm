"""Tests for memory system."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / ".claude" / "skills" / "rlm" / "scripts"))

from memory_db import MemoryDatabase, MemoryType, MemorySource, Memory


def test_memory_crud():
    """Test memory create, read, update, delete."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "memory.db"
        db = MemoryDatabase(db_path)

        # Create
        mem_id = db.add_memory(
            "Test memory content",
            memory_type=MemoryType.FACT,
            source=MemorySource.EXPLICIT,
            importance=0.7,
            tags=["test", "example"],
        )
        assert mem_id > 0

        # Read
        mem = db.get_memory(mem_id)
        assert mem is not None
        assert mem.content == "Test memory content"
        assert mem.memory_type == MemoryType.FACT
        assert mem.importance == 0.7
        assert "test" in mem.tags

        # Update
        db.update_memory(mem_id, importance=0.9)
        mem = db.get_memory(mem_id)
        assert mem.importance == 0.9

        # Delete
        success = db.delete_memory(mem_id)
        assert success
        mem = db.get_memory(mem_id)
        assert mem is None

        db.close()


def test_memory_types():
    """Test different memory types."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "memory.db"
        db = MemoryDatabase(db_path)

        # Add different types
        db.add_memory("A fact", memory_type=MemoryType.FACT)
        db.add_memory("A preference", memory_type=MemoryType.PREFERENCE)
        db.add_memory("An instruction", memory_type=MemoryType.INSTRUCTION)
        db.add_memory("A decision", memory_type=MemoryType.DECISION)
        db.add_memory("Some context", memory_type=MemoryType.CONTEXT)
        db.add_memory("A pattern", memory_type=MemoryType.PATTERN)

        # Get by type
        facts = db.get_memories_by_type(MemoryType.FACT)
        assert len(facts) == 1
        assert facts[0].content == "A fact"

        prefs = db.get_memories_by_type(MemoryType.PREFERENCE)
        assert len(prefs) == 1

        # Stats should show all types
        stats = db.get_stats()
        assert stats['total_memories'] == 6
        assert len(stats['by_type']) == 6

        db.close()


def test_memory_fts_search():
    """Test FTS5 search on memories."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "memory.db"
        db = MemoryDatabase(db_path)

        db.add_memory("User prefers TypeScript over JavaScript")
        db.add_memory("Project uses PostgreSQL database")
        db.add_memory("Always run tests before committing")

        # Search
        results = db.search_memories_fts("TypeScript")
        assert len(results) == 1
        assert "TypeScript" in results[0].content

        results = db.search_memories_fts("PostgreSQL")
        assert len(results) == 1

        results = db.search_memories_fts("tests")
        assert len(results) == 1

        results = db.search_memories_fts("nonexistent")
        assert len(results) == 0

        db.close()


def test_memory_importance():
    """Test importance-based filtering."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "memory.db"
        db = MemoryDatabase(db_path)

        db.add_memory("Low importance", importance=0.3)
        db.add_memory("Medium importance", importance=0.5)
        db.add_memory("High importance", importance=0.8)
        db.add_memory("Critical importance", importance=0.95)

        # Get high importance
        important = db.get_important_memories(min_importance=0.7)
        assert len(important) == 2

        important = db.get_important_memories(min_importance=0.9)
        assert len(important) == 1
        assert "Critical" in important[0].content

        db.close()


def test_memory_tags():
    """Test tag-based filtering."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "memory.db"
        db = MemoryDatabase(db_path)

        db.add_memory("Database info", tags=["database", "tech"])
        db.add_memory("API info", tags=["api", "tech"])
        db.add_memory("UI preference", tags=["ui", "preference"])

        # Search by tag
        tech = db.get_memories_by_tag("tech")
        assert len(tech) == 2

        db_mems = db.get_memories_by_tag("database")
        assert len(db_mems) == 1

        db.close()


def test_context_generation():
    """Test session context generation."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "memory.db"
        db = MemoryDatabase(db_path)

        # Add various memories
        db.add_memory(
            "Always validate input",
            memory_type=MemoryType.INSTRUCTION,
            importance=0.9
        )
        db.add_memory(
            "Project uses React",
            memory_type=MemoryType.CONTEXT,
            importance=0.6
        )
        db.add_memory(
            "Prefers functional style",
            memory_type=MemoryType.PREFERENCE,
            importance=0.7
        )

        # Generate context
        context = db.get_context_for_session(max_chars=2000)

        assert len(context) > 0
        assert len(context) <= 2000
        # Should include standing instructions
        assert "Standing Instructions" in context or "validate" in context.lower()

        db.close()


def test_profile():
    """Test user profile management."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "memory.db"
        db = MemoryDatabase(db_path)

        # Default profile should exist
        profile = db.get_profile()
        assert profile is not None

        # Set preferences
        db.set_preference("editor", "vscode")
        db.set_preference("language", "python")

        profile = db.get_profile()
        assert profile.preferences["editor"] == "vscode"
        assert profile.preferences["language"] == "python"

        # Get single preference
        editor = db.get_preference("editor")
        assert editor == "vscode"

        # Get with default
        missing = db.get_preference("nonexistent", "default")
        assert missing == "default"

        # Update profile
        db.update_profile(name="Test User")
        profile = db.get_profile()
        assert profile.name == "Test User"

        db.close()


def test_sessions():
    """Test session tracking."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "memory.db"
        db = MemoryDatabase(db_path)

        # Start session
        session_id = db.start_session(metadata={"test": True})
        assert session_id > 0

        # Get session
        session = db.get_session(session_id)
        assert session is not None
        assert session['ended_at'] is None

        # End session
        db.end_session(session_id, summary="Test completed", memories_created=5)

        session = db.get_session(session_id)
        assert session['ended_at'] is not None
        assert session['summary'] == "Test completed"
        assert session['memories_created'] == 5

        # Recent sessions
        recent = db.get_recent_sessions(limit=10)
        assert len(recent) >= 1

        db.close()


def test_memory_access_tracking():
    """Test access time and count tracking."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "memory.db"
        db = MemoryDatabase(db_path)

        mem_id = db.add_memory("Test memory")

        mem = db.get_memory(mem_id)
        initial_count = mem.access_count

        # Touch memory
        db.touch_memory(mem_id)
        db.touch_memory(mem_id)

        mem = db.get_memory(mem_id)
        assert mem.access_count == initial_count + 2

        db.close()


def test_memory_extraction_patterns():
    """Test pattern-based memory extraction from text."""
    sys.path.insert(0, str(Path(__file__).parent.parent / ".claude" / "skills" / "rlm" / "scripts"))

    try:
        from session_stop import extract_memories_from_text
    except ImportError:
        print("  Skipping extraction test - module not available")
        return

    text = """
    Remember that the API uses JWT tokens for authentication.
    I prefer using async/await over callbacks.
    Always validate user input before processing.
    We decided to use PostgreSQL instead of MySQL.
    Don't forget that the rate limit is 100 requests per minute.
    """

    memories = extract_memories_from_text(text)

    assert len(memories) >= 4, f"Expected at least 4 memories, got {len(memories)}"

    types = [m['memory_type'] for m in memories]
    assert 'fact' in types, "Should extract facts"
    assert 'preference' in types, "Should extract preferences"
    assert 'instruction' in types, "Should extract instructions"


def test_memory_extraction_edge_cases():
    """Test edge cases in memory extraction."""
    try:
        from session_stop import extract_memories_from_text
    except ImportError:
        print("  Skipping extraction test - module not available")
        return

    # Very short text
    memories = extract_memories_from_text("hi")
    assert len(memories) == 0, f"Short text should produce no memories, got {len(memories)}"

    # No patterns
    memories = extract_memories_from_text("The quick brown fox jumps over the lazy dog.")
    assert len(memories) == 0, f"No pattern text should produce no memories, got {len(memories)}"

    # Multiple matches - need longer content to pass the min length filter
    text = """
    Remember that the API uses versioning for backwards compatibility.
    Remember that the database schema is managed with migrations.
    Remember that all endpoints require authentication tokens.
    """
    memories = extract_memories_from_text(text)
    assert len(memories) >= 3, f"Expected at least 3 memories from multiple patterns, got {len(memories)}: {[m['content'] for m in memories]}"


def run_tests():
    """Run all memory tests."""
    print("=" * 60)
    print("Testing Memory System")
    print("=" * 60)

    tests = [
        ("Memory CRUD", test_memory_crud),
        ("Memory types", test_memory_types),
        ("FTS5 search", test_memory_fts_search),
        ("Importance filtering", test_memory_importance),
        ("Tag filtering", test_memory_tags),
        ("Context generation", test_context_generation),
        ("Profile management", test_profile),
        ("Session tracking", test_sessions),
        ("Access tracking", test_memory_access_tracking),
        ("Extraction patterns", test_memory_extraction_patterns),
        ("Extraction edge cases", test_memory_extraction_edge_cases),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            test_fn()
            print(f"  ✓ {name}")
            passed += 1
        except AssertionError as e:
            print(f"  ✗ {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ {name}: {type(e).__name__}: {e}")
            failed += 1

    print(f"\nMemory tests: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
