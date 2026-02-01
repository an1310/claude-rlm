"""Tests for session hooks and extraction patterns."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / ".claude" / "skills" / "rlm" / "scripts"))


def test_session_start_no_db():
    """Test session_start when no database exists."""
    from session_start import get_context_injection

    with tempfile.TemporaryDirectory() as tmp:
        import os
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp)
            # No database exists
            context = get_context_injection()
            assert context == ""  # Should return empty, not error
        finally:
            os.chdir(old_cwd)


def test_session_start_with_memories():
    """Test session_start with populated database."""
    from session_start import get_context_injection
    from memory_db import MemoryDatabase, MemoryType, MemorySource

    with tempfile.TemporaryDirectory() as tmp:
        import os
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp)

            # Create and populate database
            db = MemoryDatabase()
            db.add_memory(
                "Always use TypeScript",
                memory_type=MemoryType.INSTRUCTION,
                source=MemorySource.EXPLICIT,
                importance=0.9
            )
            db.add_memory(
                "Project uses PostgreSQL",
                memory_type=MemoryType.FACT,
                importance=0.8
            )
            db.close()

            # Get context
            context = get_context_injection()
            assert len(context) > 0
            assert "TypeScript" in context or "PostgreSQL" in context

        finally:
            os.chdir(old_cwd)


def test_session_start_respects_budget():
    """Test that context respects character budget."""
    from session_start import get_context_injection
    from memory_db import MemoryDatabase, MemoryType

    with tempfile.TemporaryDirectory() as tmp:
        import os
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp)

            db = MemoryDatabase()
            # Add many memories
            for i in range(50):
                db.add_memory(
                    f"Memory number {i} with some content that takes up space",
                    memory_type=MemoryType.FACT,
                    importance=0.5
                )
            db.close()

            # Small budget
            context = get_context_injection(max_chars=500)
            assert len(context) <= 500

            # Larger budget
            context = get_context_injection(max_chars=2000)
            assert len(context) <= 2000

        finally:
            os.chdir(old_cwd)


def test_extraction_remember_patterns():
    """Test extraction of 'remember that' patterns."""
    from session_stop import extract_memories_from_text

    text = """
    Remember that the API rate limit is 100 requests per minute.
    Also remember that we use UTC for all timestamps.
    Don't forget that the cache expires after 1 hour.
    """

    memories = extract_memories_from_text(text)
    contents = [m['content'].lower() for m in memories]

    assert any('rate limit' in c for c in contents)
    assert any('utc' in c or 'timestamps' in c for c in contents)


def test_extraction_preference_patterns():
    """Test extraction of preference patterns."""
    from session_stop import extract_memories_from_text

    text = """
    I prefer using async/await over callbacks.
    I really like the functional programming approach.
    I don't like using global state.
    My preference is to use TypeScript.
    """

    memories = extract_memories_from_text(text)
    types = [m['memory_type'] for m in memories]

    assert 'preference' in types
    assert len([t for t in types if t == 'preference']) >= 2


def test_extraction_instruction_patterns():
    """Test extraction of instruction patterns."""
    from session_stop import extract_memories_from_text

    text = """
    Always run tests before committing code.
    Never commit directly to main branch.
    Make sure to update the changelog for every release.
    From now on, use semantic versioning for all packages.
    """

    memories = extract_memories_from_text(text)
    types = [m['memory_type'] for m in memories]

    assert 'instruction' in types
    assert len([t for t in types if t == 'instruction']) >= 2


def test_extraction_decision_patterns():
    """Test extraction of decision patterns."""
    from session_stop import extract_memories_from_text

    text = """
    We decided to use PostgreSQL for the database.
    The decision is to implement caching at the API layer.
    Let's go with the microservices architecture.
    We'll use Redis for session storage.
    """

    memories = extract_memories_from_text(text)
    types = [m['memory_type'] for m in memories]

    assert 'decision' in types
    assert len([t for t in types if t == 'decision']) >= 2


def test_extraction_filters_short():
    """Test that very short matches are filtered out."""
    from session_stop import extract_memories_from_text

    text = """
    Remember that x.
    I prefer y.
    Always do z.
    """

    memories = extract_memories_from_text(text)
    # Single letters should be filtered (< 10 chars)
    assert len(memories) == 0


def test_extraction_deduplication():
    """Test that duplicate content is deduplicated."""
    from session_stop import extract_memories_from_text

    text = """
    Remember that the API uses JWT tokens.
    Remember that the API uses JWT tokens.
    Remember that the API uses JWT tokens.
    """

    memories = extract_memories_from_text(text)
    # Should only have one memory
    assert len(memories) == 1


def test_extraction_importance_levels():
    """Test that different patterns get appropriate importance."""
    from session_stop import extract_memories_from_text

    text = """
    Remember that the project uses React for the frontend.
    I prefer to use functional components.
    Always run linting before commits.
    We decided to use GraphQL.
    """

    memories = extract_memories_from_text(text)

    # Check importance levels
    for m in memories:
        if m['memory_type'] == 'fact':
            assert m['importance'] >= 0.8  # Remember patterns are high
        elif m['memory_type'] == 'instruction':
            assert m['importance'] >= 0.7
        elif m['memory_type'] == 'preference':
            assert m['importance'] >= 0.6


def test_parse_conversation_transcript():
    """Test parsing conversation transcripts."""
    from session_stop import parse_conversation_transcript

    transcript = """
User: Hello, I'd like to set up a new project.
Assistant: Sure! What kind of project are you working on?
User: It's a React app. Remember that I prefer TypeScript.
Assistant: Got it, I'll use TypeScript for the React project.
User: Also, always use functional components.
Assistant: Understood, functional components it is.
"""

    user_msgs, assistant_msgs = parse_conversation_transcript(transcript)

    assert len(user_msgs) >= 1
    assert any('prefer' in msg.lower() for msg in user_msgs)


def test_store_memories():
    """Test storing extracted memories."""
    from session_stop import store_memories
    from memory_db import MemoryDatabase

    with tempfile.TemporaryDirectory() as tmp:
        import os
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp)

            memories = [
                {'content': 'Test fact memory', 'memory_type': 'fact', 'importance': 0.7, 'source': 'extracted'},
                {'content': 'Test preference memory', 'memory_type': 'preference', 'importance': 0.6, 'source': 'extracted'},
            ]

            stored = store_memories(memories)
            assert stored == 2

            # Verify in database
            db = MemoryDatabase()
            all_mems = db.get_all_memories()
            assert len(all_mems) >= 2
            db.close()

        finally:
            os.chdir(old_cwd)


def test_full_extraction_pipeline():
    """Test full extraction from text to storage."""
    from session_stop import extract_memories_from_text, store_memories
    from memory_db import MemoryDatabase

    with tempfile.TemporaryDirectory() as tmp:
        import os
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp)

            text = """
            Remember that we're building a REST API with Express.
            I prefer to use async/await for all asynchronous code.
            Always validate request bodies before processing.
            We decided to use MongoDB for the database.
            """

            # Extract
            memories = extract_memories_from_text(text)
            assert len(memories) >= 3

            # Store
            stored = store_memories(memories)
            assert stored >= 3

            # Verify
            db = MemoryDatabase()
            all_mems = db.get_all_memories()
            contents = [m.content.lower() for m in all_mems]

            assert any('rest api' in c or 'express' in c for c in contents)
            assert any('async' in c or 'await' in c for c in contents)
            db.close()

        finally:
            os.chdir(old_cwd)


def test_context_injection_format():
    """Test that context injection is properly formatted."""
    from session_start import get_context_injection
    from memory_db import MemoryDatabase, MemoryType, MemorySource

    with tempfile.TemporaryDirectory() as tmp:
        import os
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp)

            db = MemoryDatabase()
            db.add_memory(
                "Always use type hints in Python code",
                memory_type=MemoryType.INSTRUCTION,
                importance=0.95
            )
            db.set_preference("language", "Python")
            db.set_preference("style", "functional")
            db.close()

            context = get_context_injection()

            # Should have markdown headers
            assert "##" in context or context == ""
            # Should be readable text
            if context:
                assert not context.startswith("{")  # Not JSON

        finally:
            os.chdir(old_cwd)


def run_tests():
    """Run all hook tests."""
    print("=" * 60)
    print("Testing Session Hooks")
    print("=" * 60)

    tests = [
        ("Session start no DB", test_session_start_no_db),
        ("Session start with memories", test_session_start_with_memories),
        ("Session start respects budget", test_session_start_respects_budget),
        ("Extraction remember patterns", test_extraction_remember_patterns),
        ("Extraction preference patterns", test_extraction_preference_patterns),
        ("Extraction instruction patterns", test_extraction_instruction_patterns),
        ("Extraction decision patterns", test_extraction_decision_patterns),
        ("Extraction filters short", test_extraction_filters_short),
        ("Extraction deduplication", test_extraction_deduplication),
        ("Extraction importance levels", test_extraction_importance_levels),
        ("Parse conversation transcript", test_parse_conversation_transcript),
        ("Store memories", test_store_memories),
        ("Full extraction pipeline", test_full_extraction_pipeline),
        ("Context injection format", test_context_injection_format),
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

    print(f"\nHook tests: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
