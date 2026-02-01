"""Tests for codebase indexing."""

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / ".claude" / "skills" / "rlm" / "scripts"))

from rlm_repl import Database, CodebaseIndexer, _make_helpers
from conftest import create_test_files


def test_full_indexing():
    """Test full codebase indexing workflow."""
    with tempfile.TemporaryDirectory() as tmp:
        test_dir = Path(tmp)
        db_path = test_dir / "state" / "index.db"

        # Create test files
        create_test_files(test_dir)

        # Index
        db = Database(db_path)
        indexer = CodebaseIndexer(db, test_dir, repo_name="test-project")
        stats = indexer.index_codebase()

        assert stats['indexed'] >= 4, f"Expected at least 4 files indexed, got {stats['indexed']}"
        assert stats['errors'] == 0, f"Expected 0 errors, got {stats['errors']}"

        # Verify database contents
        db_stats = db.get_stats()
        assert db_stats['files'] >= 4
        assert db_stats['chunks'] >= 10
        assert db_stats['repos'] == 1

        # Verify language detection
        assert 'python' in db_stats['languages']
        assert 'java' in db_stats['languages']
        assert 'javascript' in db_stats['languages']
        assert 'typescript' in db_stats['languages']

        db.close()


def test_incremental_indexing():
    """Test incremental indexing skips unchanged files."""
    with tempfile.TemporaryDirectory() as tmp:
        test_dir = Path(tmp)
        db_path = test_dir / "state" / "index.db"

        create_test_files(test_dir)

        db = Database(db_path)
        indexer = CodebaseIndexer(db, test_dir, repo_name="test-project")

        # First index
        stats1 = indexer.index_codebase()
        indexed_first = stats1['indexed']

        # Second index - should skip all
        stats2 = indexer.index_codebase()
        assert stats2['skipped'] == indexed_first, "All files should be skipped"
        assert stats2['indexed'] == 0, "No files should be re-indexed"

        # Modify a file
        auth_file = test_dir / "python" / "auth.py"
        original = auth_file.read_text()
        auth_file.write_text(original + "\n\ndef new_function():\n    pass\n")

        # Third index - should only index modified file
        stats3 = indexer.index_codebase()
        assert stats3['indexed'] == 1, f"Only 1 file should be indexed, got {stats3['indexed']}"
        assert stats3['skipped'] >= indexed_first - 1

        db.close()


def test_full_reindex():
    """Test forced full re-index."""
    with tempfile.TemporaryDirectory() as tmp:
        test_dir = Path(tmp)
        db_path = test_dir / "state" / "index.db"

        create_test_files(test_dir)

        db = Database(db_path)
        indexer = CodebaseIndexer(db, test_dir, repo_name="test-project")

        # First index
        stats1 = indexer.index_codebase()
        indexed_first = stats1['indexed']

        # Force full re-index
        stats2 = indexer.index_codebase(full=True)
        assert stats2['indexed'] == indexed_first, "All files should be re-indexed"
        assert stats2['skipped'] == 0, "No files should be skipped"

        db.close()


def test_multi_repo():
    """Test multi-repository support."""
    with tempfile.TemporaryDirectory() as tmp:
        test_dir = Path(tmp)
        db_path = test_dir / "state" / "index.db"

        # Create first repo
        create_test_files(test_dir)

        # Create second repo
        repo2_dir = test_dir / "second-repo"
        repo2_dir.mkdir(parents=True)
        (repo2_dir / "shared.py").write_text('''
"""Shared utilities."""

class SharedService:
    def process(self, data):
        return data

def helper_function():
    return "shared"
''')

        db = Database(db_path)

        # Index both repos
        indexer1 = CodebaseIndexer(db, test_dir, repo_name="main-project")
        indexer1.index_codebase()

        indexer2 = CodebaseIndexer(db, repo2_dir, repo_name="shared-lib")
        indexer2.index_codebase()

        # Verify both repos exist
        repos = db.list_repos()
        repo_names = [r['repo_name'] for r in repos]
        assert "main-project" in repo_names
        assert "shared-lib" in repo_names

        # Test helpers with repo filtering
        chunks_dir = test_dir / "chunks"
        helpers = _make_helpers(db, chunks_dir)

        # Search across all repos
        all_services = helpers['find_symbol']('Service', 'class')
        assert len(all_services) >= 2, "Should find services in multiple repos"

        # Search in specific repo
        shared_only = helpers['find_symbol']('Service', 'class', repo='shared-lib')
        assert len(shared_only) == 1, "Should find 1 service in shared-lib"
        assert shared_only[0]['repo_name'] == 'shared-lib'

        # Test repo removal
        db.delete_repo('shared-lib')
        repos_after = db.list_repos()
        assert len(repos_after) == 1
        assert repos_after[0]['repo_name'] == 'main-project'

        db.close()


def test_extension_filtering():
    """Test extension-based file filtering."""
    with tempfile.TemporaryDirectory() as tmp:
        test_dir = Path(tmp)
        db_path = test_dir / "state" / "index.db"

        create_test_files(test_dir)

        db = Database(db_path)

        # Index only Python files
        indexer = CodebaseIndexer(
            db, test_dir,
            repo_name="python-only",
            extensions=[".py"]
        )
        stats = indexer.index_codebase()

        # Verify only Python files indexed
        files = db.query("SELECT * FROM files")
        for f in files:
            assert f['language'] == 'python', f"Expected python, got {f['language']}"

        db.close()


def test_helper_functions():
    """Test helper functions in exec mode."""
    with tempfile.TemporaryDirectory() as tmp:
        test_dir = Path(tmp)
        db_path = test_dir / "state" / "index.db"
        chunks_dir = test_dir / "chunks"

        create_test_files(test_dir)

        db = Database(db_path)
        indexer = CodebaseIndexer(db, test_dir, repo_name="test-project")
        indexer.index_codebase()

        helpers = _make_helpers(db, chunks_dir)

        # find_symbol
        results = helpers['find_symbol']('auth')
        assert len(results) > 0, "Should find auth-related symbols"

        # find_symbol_exact
        results = helpers['find_symbol_exact']('AuthService')
        assert len(results) >= 1, "Should find AuthService exactly"
        assert results[0]['symbol_name'] == 'AuthService'

        # get_class_methods
        methods = helpers['get_class_methods']('AuthService')
        assert len(methods) >= 2, "AuthService should have multiple methods"

        # search_content (FTS5)
        results = helpers['search_content']('password')
        assert len(results) >= 1, "Should find password in content"

        # get_files_by_language
        python_files = helpers['get_files_by_language']('python')
        assert len(python_files) >= 1

        # stats
        stats = helpers['stats']()
        assert stats['files'] >= 4
        assert stats['chunks'] >= 10

        db.close()


def test_chunk_materialization():
    """Test writing chunks to files."""
    with tempfile.TemporaryDirectory() as tmp:
        test_dir = Path(tmp)
        db_path = test_dir / "state" / "index.db"
        chunks_dir = test_dir / "chunks"

        create_test_files(test_dir)

        db = Database(db_path)
        indexer = CodebaseIndexer(db, test_dir, repo_name="test-project")
        indexer.index_codebase()

        helpers = _make_helpers(db, chunks_dir)

        # Get a chunk
        chunks = db.query("SELECT chunk_id FROM chunks LIMIT 1")
        assert len(chunks) > 0

        chunk_id = chunks[0]['chunk_id']

        # Write chunk to file
        filepath = helpers['write_chunk_to_file'](chunk_id)
        assert filepath is not None
        assert Path(filepath).exists()

        # Verify content
        content = Path(filepath).read_text()
        assert "# Chunk:" in content
        assert "# Type:" in content

        db.close()


def run_tests():
    """Run all indexing tests."""
    print("=" * 60)
    print("Testing Codebase Indexing")
    print("=" * 60)

    tests = [
        ("Full indexing", test_full_indexing),
        ("Incremental indexing", test_incremental_indexing),
        ("Full re-index", test_full_reindex),
        ("Multi-repo support", test_multi_repo),
        ("Extension filtering", test_extension_filtering),
        ("Helper functions", test_helper_functions),
        ("Chunk materialization", test_chunk_materialization),
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

    print(f"\nIndexing tests: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
