"""Edge case and error handling tests."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / ".claude" / "skills" / "rlm" / "scripts"))

from rlm_repl import (
    Database, CodebaseIndexer, PythonAnalyzer, JavaScriptAnalyzer, JavaAnalyzer,
    _make_helpers
)


def test_empty_file():
    """Test handling of empty files."""
    with tempfile.TemporaryDirectory() as tmp:
        test_dir = Path(tmp)
        db_path = test_dir / "index.db"

        # Create empty file
        (test_dir / "empty.py").write_text("")

        db = Database(db_path)
        indexer = CodebaseIndexer(db, test_dir, repo_name="test")
        stats = indexer.index_codebase()

        # Should index without error
        assert stats['errors'] == 0
        db.close()


def test_syntax_error_file():
    """Test handling of files with syntax errors."""
    with tempfile.TemporaryDirectory() as tmp:
        test_dir = Path(tmp)
        db_path = test_dir / "index.db"

        # Create Python file with syntax error
        (test_dir / "broken.py").write_text("""
def broken_function(
    # Missing closing paren and colon
    pass
""")

        db = Database(db_path)
        indexer = CodebaseIndexer(db, test_dir, repo_name="test")
        stats = indexer.index_codebase()

        # Should handle gracefully (file indexed but no chunks extracted)
        assert stats['errors'] == 0
        db.close()


def test_unicode_content():
    """Test handling of unicode in code."""
    with tempfile.TemporaryDirectory() as tmp:
        test_dir = Path(tmp)
        db_path = test_dir / "index.db"

        (test_dir / "unicode.py").write_text('''
# -*- coding: utf-8 -*-
"""Module with unicode: 日本語, émojis 🎉, and symbols ∑∏∫"""

def greet(name: str) -> str:
    """Greet with unicode: Héllo Wörld 你好"""
    return f"Hello, {name}! 👋"

class Café:
    """A café class with accents."""
    menu = ["Espresso ☕", "Croissant 🥐"]
''', encoding='utf-8')

        db = Database(db_path)
        indexer = CodebaseIndexer(db, test_dir, repo_name="test")
        stats = indexer.index_codebase()

        assert stats['indexed'] == 1
        assert stats['errors'] == 0

        # Verify content preserved
        chunks = db.query("SELECT * FROM chunks WHERE symbol_name = 'Café'")
        assert len(chunks) == 1
        assert "café" in chunks[0]['content'].lower() or "Café" in chunks[0]['content']

        db.close()


def test_very_long_function():
    """Test handling of very long functions."""
    with tempfile.TemporaryDirectory() as tmp:
        test_dir = Path(tmp)
        db_path = test_dir / "index.db"

        # Create file with very long function
        long_body = "\n".join([f"    x = {i}" for i in range(1000)])
        (test_dir / "long.py").write_text(f'''
def very_long_function():
    """A function with 1000 lines."""
{long_body}
    return x
''')

        db = Database(db_path)
        indexer = CodebaseIndexer(db, test_dir, repo_name="test")
        stats = indexer.index_codebase()

        assert stats['indexed'] == 1

        chunks = db.query("SELECT * FROM chunks WHERE symbol_name = 'very_long_function'")
        assert len(chunks) == 1
        assert chunks[0]['end_line'] - chunks[0]['start_line'] > 900

        db.close()


def test_deeply_nested_classes():
    """Test handling of nested class definitions."""
    with tempfile.TemporaryDirectory() as tmp:
        test_dir = Path(tmp)
        db_path = test_dir / "index.db"

        (test_dir / "nested.py").write_text('''
class Outer:
    """Outer class."""

    class Inner:
        """Inner class."""

        class DeepInner:
            """Deeply nested class."""

            def deep_method(self):
                pass

        def inner_method(self):
            pass

    def outer_method(self):
        pass
''')

        db = Database(db_path)
        indexer = CodebaseIndexer(db, test_dir, repo_name="test")
        indexer.index_codebase()

        # Should find all classes
        classes = db.query("SELECT * FROM symbols WHERE symbol_type = 'class'")
        class_names = [c['symbol_name'] for c in classes]
        assert "Outer" in class_names
        assert "Inner" in class_names
        assert "DeepInner" in class_names

        db.close()


def test_special_characters_in_strings():
    """Test handling of special regex characters in code."""
    with tempfile.TemporaryDirectory() as tmp:
        test_dir = Path(tmp)
        db_path = test_dir / "index.db"

        (test_dir / "special.js").write_text('''
const regex = /[a-z]+\\.(js|ts)$/;
const template = `Hello ${name}!`;
const escape = "Line1\\nLine2\\tTabbed";

function matchPattern(str) {
    return str.match(/^[A-Z][a-z]*$/);
}
''')

        db = Database(db_path)
        indexer = CodebaseIndexer(db, test_dir, repo_name="test")
        stats = indexer.index_codebase()

        assert stats['errors'] == 0
        db.close()


def test_file_with_only_imports():
    """Test file that only has imports."""
    with tempfile.TemporaryDirectory() as tmp:
        test_dir = Path(tmp)
        db_path = test_dir / "index.db"

        (test_dir / "imports_only.py").write_text('''
import os
import sys
from typing import List, Dict, Optional
from pathlib import Path
from collections import defaultdict
''')

        db = Database(db_path)
        indexer = CodebaseIndexer(db, test_dir, repo_name="test")
        stats = indexer.index_codebase()

        assert stats['errors'] == 0

        # Should have imports but no chunks
        imports = db.query("SELECT * FROM imports")
        assert len(imports) >= 4

        db.close()


def test_database_reopening():
    """Test database can be closed and reopened."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"

        # Create and populate
        db1 = Database(db_path)
        db1.get_or_create_repo("test", "/test")
        db1.execute("INSERT INTO files (repo_id, filepath) VALUES (1, 'test.py')")
        db1._conn.commit()
        db1.close()

        # Reopen and verify
        db2 = Database(db_path)
        files = db2.query("SELECT * FROM files")
        assert len(files) == 1
        assert files[0]['filepath'] == 'test.py'
        db2.close()


def test_concurrent_file_modifications():
    """Test incremental indexing with file modifications."""
    with tempfile.TemporaryDirectory() as tmp:
        test_dir = Path(tmp)
        db_path = test_dir / "index.db"

        # Initial file
        test_file = test_dir / "changing.py"
        test_file.write_text("def v1(): pass")

        db = Database(db_path)
        indexer = CodebaseIndexer(db, test_dir, repo_name="test")

        # First index
        stats1 = indexer.index_codebase()
        assert stats1['indexed'] == 1

        chunks1 = db.query("SELECT * FROM chunks")
        assert len(chunks1) == 1
        assert chunks1[0]['symbol_name'] == 'v1'

        # Modify file
        test_file.write_text("def v2(): pass\ndef v2b(): pass")

        # Re-index
        stats2 = indexer.index_codebase()
        assert stats2['indexed'] == 1
        assert stats2['skipped'] == 0

        chunks2 = db.query("SELECT * FROM chunks ORDER BY symbol_name")
        assert len(chunks2) == 2
        assert chunks2[0]['symbol_name'] == 'v2'
        assert chunks2[1]['symbol_name'] == 'v2b'

        db.close()


def test_fts_special_characters():
    """Test FTS5 search with special characters."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        db = Database(db_path)

        repo_id = db.get_or_create_repo("test", "/test")
        cursor = db.execute(
            "INSERT INTO files (repo_id, filepath, language) VALUES (?, ?, ?)",
            (repo_id, "test.py", "python")
        )
        file_id = cursor.lastrowid

        # Insert content with special characters
        db.execute(
            """INSERT INTO chunks (file_id, chunk_type, symbol_name, start_line, end_line, content)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (file_id, "function", "test_func", 1, 10,
             'def test_func(): return "Hello, World!" + str(123)')
        )
        db._conn.commit()

        # Search should work
        results = db.search_content_fts("Hello")
        assert len(results) >= 1

        # Search with quotes
        results = db.search_content_fts("World")
        assert len(results) >= 1

        db.close()


def test_empty_search_results():
    """Test search with no results."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        db = Database(db_path)

        results = db.search_content_fts("nonexistent_term_xyz")
        assert len(results) == 0

        results = db.query(
            "SELECT * FROM symbols WHERE symbol_name = ?",
            ("nonexistent",)
        )
        assert len(results) == 0

        db.close()


def test_duplicate_symbol_names():
    """Test handling of duplicate symbol names across files."""
    with tempfile.TemporaryDirectory() as tmp:
        test_dir = Path(tmp)
        db_path = test_dir / "index.db"

        # Create multiple files with same function name
        (test_dir / "file1.py").write_text("def process(): pass")
        (test_dir / "file2.py").write_text("def process(): pass")
        (test_dir / "file3.py").write_text("def process(): pass")

        db = Database(db_path)
        indexer = CodebaseIndexer(db, test_dir, repo_name="test")
        indexer.index_codebase()

        # All should be indexed
        chunks = db.query("SELECT * FROM chunks WHERE symbol_name = 'process'")
        assert len(chunks) == 3

        # All in different files
        file_ids = set(c['file_id'] for c in chunks)
        assert len(file_ids) == 3

        db.close()


def test_binary_file_handling():
    """Test that binary files are skipped gracefully."""
    with tempfile.TemporaryDirectory() as tmp:
        test_dir = Path(tmp)
        db_path = test_dir / "index.db"

        # Create a Python file
        (test_dir / "good.py").write_text("def good(): pass")

        # Create a binary-ish file with .py extension (unusual but possible)
        (test_dir / "binary.py").write_bytes(b'\x00\x01\x02\x03\xff\xfe')

        db = Database(db_path)
        indexer = CodebaseIndexer(db, test_dir, repo_name="test")
        stats = indexer.index_codebase()

        # Should handle gracefully
        assert stats['indexed'] >= 1  # At least the good file
        db.close()


def test_symlink_handling():
    """Test that symlinks don't cause infinite loops."""
    with tempfile.TemporaryDirectory() as tmp:
        test_dir = Path(tmp)
        db_path = test_dir / "index.db"

        # Create actual file
        (test_dir / "real.py").write_text("def real(): pass")

        # Create symlink (may fail on some systems)
        try:
            (test_dir / "link.py").symlink_to(test_dir / "real.py")
        except OSError:
            # Symlinks not supported, skip this part
            pass

        db = Database(db_path)
        indexer = CodebaseIndexer(db, test_dir, repo_name="test")
        stats = indexer.index_codebase()

        # Should complete without hanging
        assert stats['indexed'] >= 1
        db.close()


def run_tests():
    """Run all edge case tests."""
    print("=" * 60)
    print("Testing Edge Cases")
    print("=" * 60)

    tests = [
        ("Empty file", test_empty_file),
        ("Syntax error file", test_syntax_error_file),
        ("Unicode content", test_unicode_content),
        ("Very long function", test_very_long_function),
        ("Deeply nested classes", test_deeply_nested_classes),
        ("Special characters in strings", test_special_characters_in_strings),
        ("File with only imports", test_file_with_only_imports),
        ("Database reopening", test_database_reopening),
        ("Concurrent file modifications", test_concurrent_file_modifications),
        ("FTS special characters", test_fts_special_characters),
        ("Empty search results", test_empty_search_results),
        ("Duplicate symbol names", test_duplicate_symbol_names),
        ("Binary file handling", test_binary_file_handling),
        ("Symlink handling", test_symlink_handling),
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

    print(f"\nEdge case tests: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
