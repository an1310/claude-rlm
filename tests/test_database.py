"""Tests for database features."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / ".claude" / "skills" / "rlm" / "scripts"))

from rlm_repl import Database


def test_wal_mode():
    """Test WAL mode is enabled."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        db = Database(db_path)

        result = db.query("PRAGMA journal_mode")
        assert result[0]['journal_mode'] == 'wal', "WAL mode should be enabled"

        db.close()


def test_foreign_keys():
    """Test foreign keys are enabled."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        db = Database(db_path)

        result = db.query("PRAGMA foreign_keys")
        assert result[0]['foreign_keys'] == 1, "Foreign keys should be enabled"

        db.close()


def test_fts5_table():
    """Test FTS5 virtual table is created."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        db = Database(db_path)

        result = db.query(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='chunks_fts'"
        )
        assert len(result) > 0, "FTS5 table should exist"

        db.close()


def test_transaction_batching():
    """Test transaction batching for bulk operations."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        db = Database(db_path)

        # Insert many rows in a single transaction
        db.begin_transaction()
        for i in range(100):
            db._conn.execute(
                "INSERT INTO files (filepath, language) VALUES (?, ?)",
                (f"test_{i}.py", "python")
            )
        db.commit()

        count = db.query("SELECT COUNT(*) as c FROM files")[0]['c']
        assert count == 100, f"Expected 100 files, got {count}"

        # Clean up
        db.execute("DELETE FROM files")
        db.close()


def test_repo_operations():
    """Test repository CRUD operations."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        db = Database(db_path)

        # Create repo
        repo_id = db.get_or_create_repo("test-repo", "/path/to/repo")
        assert repo_id > 0

        # Get same repo
        repo_id_2 = db.get_or_create_repo("test-repo", "/path/to/repo")
        assert repo_id == repo_id_2, "Should return same repo ID"

        # List repos
        repos = db.list_repos()
        assert len(repos) == 1
        assert repos[0]['repo_name'] == "test-repo"

        # Delete repo
        success = db.delete_repo("test-repo")
        assert success

        repos = db.list_repos()
        assert len(repos) == 0

        db.close()


def test_fts5_search():
    """Test FTS5 content search."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        db = Database(db_path)

        # Create repo and file
        repo_id = db.get_or_create_repo("test", "/test")
        cursor = db.execute(
            "INSERT INTO files (repo_id, filepath, language) VALUES (?, ?, ?)",
            (repo_id, "test.py", "python")
        )
        file_id = cursor.lastrowid

        # Insert chunks
        db.execute(
            """INSERT INTO chunks (file_id, chunk_type, symbol_name, start_line, end_line, content)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (file_id, "function", "authenticate", 1, 10, "def authenticate(user, password): pass")
        )
        db.execute(
            """INSERT INTO chunks (file_id, chunk_type, symbol_name, start_line, end_line, content)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (file_id, "function", "validate", 11, 20, "def validate(data): return True")
        )
        db._conn.commit()

        # Search
        results = db.search_content_fts("authenticate")
        assert len(results) >= 1, "Should find authenticate"
        assert results[0]['symbol_name'] == "authenticate"

        results = db.search_content_fts("validate")
        assert len(results) >= 1, "Should find validate"

        results = db.search_content_fts("nonexistent")
        assert len(results) == 0, "Should not find nonexistent"

        db.close()


def test_stats():
    """Test database statistics."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        db = Database(db_path)

        # Create some data
        repo_id = db.get_or_create_repo("test", "/test")
        cursor = db.execute(
            "INSERT INTO files (repo_id, filepath, language) VALUES (?, ?, ?)",
            (repo_id, "test.py", "python")
        )
        file_id = cursor.lastrowid

        db.execute(
            """INSERT INTO chunks (file_id, chunk_type, symbol_name, start_line, end_line, content)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (file_id, "function", "test", 1, 10, "def test(): pass")
        )
        db._conn.commit()

        stats = db.get_stats()
        assert stats['files'] >= 1
        assert stats['chunks'] >= 1
        assert stats['repos'] >= 1
        assert 'languages' in stats
        assert 'chunk_types' in stats

        db.close()


def test_vacuum():
    """Test vacuum operation."""
    import sqlite3

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        db = Database(db_path)

        # Add and delete data to create fragmentation
        repo_id = db.get_or_create_repo("test", "/test")
        for i in range(50):
            db.execute(
                "INSERT INTO files (repo_id, filepath, language) VALUES (?, ?, ?)",
                (repo_id, f"test_{i}.py", "python")
            )
        db._conn.commit()

        db.execute("DELETE FROM files")
        db._conn.commit()
        db.close()

        # Use a fresh connection with isolation_level=None for VACUUM
        # This is how vacuum should work outside the ORM
        conn = sqlite3.connect(str(db_path), isolation_level=None)
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.execute("VACUUM")
        conn.close()

        # Verify database still works
        db = Database(db_path)
        stats = db.get_stats()
        assert stats['files'] == 0  # All deleted
        db.close()


def run_tests():
    """Run all database tests."""
    print("=" * 60)
    print("Testing Database Features")
    print("=" * 60)

    tests = [
        ("WAL mode", test_wal_mode),
        ("Foreign keys", test_foreign_keys),
        ("FTS5 table", test_fts5_table),
        ("Transaction batching", test_transaction_batching),
        ("Repo operations", test_repo_operations),
        ("FTS5 search", test_fts5_search),
        ("Statistics", test_stats),
        ("Vacuum", test_vacuum),
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

    print(f"\nDatabase tests: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
