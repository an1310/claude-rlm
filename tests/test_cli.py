"""CLI command tests."""

import subprocess
import sys
import tempfile
from pathlib import Path

# Path to the CLI script
CLI_PATH = Path(__file__).parent.parent / ".claude" / "skills" / "rlm" / "scripts" / "rlm_repl.py"


def run_cli(*args, input_text=None, cwd=None):
    """Run CLI command and return (returncode, stdout, stderr)."""
    cmd = [sys.executable, str(CLI_PATH)] + list(args)
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        input=input_text,
        cwd=cwd,
        timeout=30,
    )
    return result.returncode, result.stdout, result.stderr


def test_help():
    """Test help command."""
    code, stdout, stderr = run_cli("--help")
    assert code == 0
    assert "RLM" in stdout or "codebase" in stdout.lower()
    assert "init" in stdout
    assert "search" in stdout
    assert "memory" in stdout


def test_init_help():
    """Test init command help."""
    code, stdout, stderr = run_cli("init", "--help")
    assert code == 0
    assert "path" in stdout.lower()
    assert "--name" in stdout


def test_search_help():
    """Test search command help."""
    code, stdout, stderr = run_cli("search", "--help")
    assert code == 0
    assert "--symbol" in stdout
    assert "--pattern" in stdout
    assert "--fts" in stdout


def test_memory_help():
    """Test memory command help."""
    code, stdout, stderr = run_cli("memory", "--help")
    assert code == 0
    assert "add" in stdout
    assert "search" in stdout
    assert "list" in stdout


def test_init_and_status():
    """Test init followed by status."""
    with tempfile.TemporaryDirectory() as tmp:
        test_dir = Path(tmp)
        db_path = test_dir / "state" / "index.db"

        # Create test file
        (test_dir / "test.py").write_text("def hello(): pass")

        # Init
        code, stdout, stderr = run_cli(
            "--db", str(db_path),
            "init", str(test_dir),
            "--name", "test-repo"
        )
        assert code == 0, f"Init failed: {stderr}"
        assert "Indexing" in stdout or "indexed" in stdout.lower()

        # Status
        code, stdout, stderr = run_cli(
            "--db", str(db_path),
            "status"
        )
        assert code == 0, f"Status failed: {stderr}"
        assert "Files:" in stdout or "files" in stdout.lower()


def test_search_symbol():
    """Test symbol search."""
    with tempfile.TemporaryDirectory() as tmp:
        test_dir = Path(tmp)
        db_path = test_dir / "state" / "index.db"

        (test_dir / "auth.py").write_text("""
class AuthService:
    def authenticate(self):
        pass
""")

        # Init
        run_cli("--db", str(db_path), "init", str(test_dir))

        # Search
        code, stdout, stderr = run_cli(
            "--db", str(db_path),
            "search", "--symbol", "Auth"
        )
        assert code == 0, f"Search failed: {stderr}"
        assert "AuthService" in stdout or "authenticate" in stdout


def test_search_pattern_fts():
    """Test FTS pattern search."""
    with tempfile.TemporaryDirectory() as tmp:
        test_dir = Path(tmp)
        db_path = test_dir / "state" / "index.db"

        (test_dir / "handler.py").write_text("""
def handle_request():
    '''Process incoming HTTP request'''
    pass
""")

        run_cli("--db", str(db_path), "init", str(test_dir))

        code, stdout, stderr = run_cli(
            "--db", str(db_path),
            "search", "--pattern", "request", "--fts"
        )
        assert code == 0, f"Search failed: {stderr}"
        assert "handle_request" in stdout or "request" in stdout.lower()


def test_repos_command():
    """Test repos list command."""
    with tempfile.TemporaryDirectory() as tmp:
        test_dir = Path(tmp)
        db_path = test_dir / "state" / "index.db"

        (test_dir / "test.py").write_text("x = 1")

        run_cli("--db", str(db_path), "init", str(test_dir), "--name", "my-repo")

        code, stdout, stderr = run_cli("--db", str(db_path), "repos")
        assert code == 0, f"Repos failed: {stderr}"
        assert "my-repo" in stdout


def test_exec_command():
    """Test exec command."""
    with tempfile.TemporaryDirectory() as tmp:
        test_dir = Path(tmp)
        db_path = test_dir / "state" / "index.db"

        (test_dir / "test.py").write_text("def foo(): pass")

        run_cli("--db", str(db_path), "init", str(test_dir))

        code, stdout, stderr = run_cli(
            "--db", str(db_path),
            "exec", "-c", "print(stats())"
        )
        assert code == 0, f"Exec failed: {stderr}"
        assert "files" in stdout.lower() or "chunks" in stdout.lower()


def test_memory_add_and_list():
    """Test memory add and list commands."""
    with tempfile.TemporaryDirectory() as tmp:
        test_dir = Path(tmp)
        # Memory uses its own default path, so we work in the temp dir
        import os
        old_cwd = os.getcwd()

        try:
            os.chdir(tmp)

            # Add memory
            code, stdout, stderr = run_cli(
                "memory", "add", "Test memory content",
                "--type", "fact",
                "--importance", "0.8"
            )
            assert code == 0, f"Memory add failed: {stderr}"
            assert "Added" in stdout or "memory" in stdout.lower()

            # List memories
            code, stdout, stderr = run_cli("memory", "list")
            assert code == 0, f"Memory list failed: {stderr}"
            assert "Test memory content" in stdout or "fact" in stdout

        finally:
            os.chdir(old_cwd)


def test_memory_search():
    """Test memory search command."""
    with tempfile.TemporaryDirectory() as tmp:
        import os
        old_cwd = os.getcwd()

        try:
            os.chdir(tmp)

            # Add memories
            run_cli("memory", "add", "PostgreSQL is the database")
            run_cli("memory", "add", "React is the frontend framework")

            # Search
            code, stdout, stderr = run_cli("memory", "search", "database")
            assert code == 0, f"Memory search failed: {stderr}"
            assert "PostgreSQL" in stdout or "database" in stdout.lower()

        finally:
            os.chdir(old_cwd)


def test_remember_shortcut():
    """Test remember quick command."""
    with tempfile.TemporaryDirectory() as tmp:
        import os
        old_cwd = os.getcwd()

        try:
            os.chdir(tmp)

            code, stdout, stderr = run_cli(
                "remember", "User prefers dark mode"
            )
            assert code == 0, f"Remember failed: {stderr}"
            assert "Remembered" in stdout or "dark mode" in stdout.lower()

        finally:
            os.chdir(old_cwd)


def test_memory_stats():
    """Test memory stats command."""
    with tempfile.TemporaryDirectory() as tmp:
        import os
        old_cwd = os.getcwd()

        try:
            os.chdir(tmp)

            # Add some memories
            run_cli("memory", "add", "Fact 1", "--type", "fact")
            run_cli("memory", "add", "Preference 1", "--type", "preference")

            code, stdout, stderr = run_cli("memory", "stats")
            assert code == 0, f"Memory stats failed: {stderr}"
            assert "total" in stdout.lower() or "memories" in stdout.lower()

        finally:
            os.chdir(old_cwd)


def test_memory_context():
    """Test memory context command."""
    with tempfile.TemporaryDirectory() as tmp:
        import os
        old_cwd = os.getcwd()

        try:
            os.chdir(tmp)

            # Add instruction (high priority for context)
            run_cli("memory", "add", "Always write tests first",
                    "--type", "instruction", "--importance", "0.9")

            code, stdout, stderr = run_cli("memory", "context")
            assert code == 0, f"Memory context failed: {stderr}"
            # May or may not have content depending on thresholds

        finally:
            os.chdir(old_cwd)


def test_invalid_command():
    """Test invalid command handling."""
    code, stdout, stderr = run_cli("nonexistent_command")
    assert code != 0
    assert "error" in stderr.lower() or "invalid" in stderr.lower() or "argument" in stderr.lower()


def test_missing_database():
    """Test graceful handling of missing database."""
    with tempfile.TemporaryDirectory() as tmp:
        nonexistent_db = Path(tmp) / "nonexistent" / "db.db"

        code, stdout, stderr = run_cli(
            "--db", str(nonexistent_db),
            "status"
        )
        # Should fail gracefully
        assert code != 0 or "No database" in stdout or "not found" in stdout.lower()


def test_init_nonexistent_path():
    """Test init with nonexistent path."""
    code, stdout, stderr = run_cli(
        "init", "/nonexistent/path/that/does/not/exist"
    )
    assert code != 0
    assert "Error" in stderr or "not exist" in stderr.lower() or "error" in stdout.lower()


def run_tests():
    """Run all CLI tests."""
    print("=" * 60)
    print("Testing CLI Commands")
    print("=" * 60)

    tests = [
        ("Help", test_help),
        ("Init help", test_init_help),
        ("Search help", test_search_help),
        ("Memory help", test_memory_help),
        ("Init and status", test_init_and_status),
        ("Search symbol", test_search_symbol),
        ("Search pattern FTS", test_search_pattern_fts),
        ("Repos command", test_repos_command),
        ("Exec command", test_exec_command),
        ("Memory add and list", test_memory_add_and_list),
        ("Memory search", test_memory_search),
        ("Remember shortcut", test_remember_shortcut),
        ("Memory stats", test_memory_stats),
        ("Memory context", test_memory_context),
        ("Invalid command", test_invalid_command),
        ("Missing database", test_missing_database),
        ("Init nonexistent path", test_init_nonexistent_path),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            test_fn()
            print(f"  ✓ {name}")
            passed += 1
        except subprocess.TimeoutExpired:
            print(f"  ✗ {name}: Timeout")
            failed += 1
        except AssertionError as e:
            print(f"  ✗ {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ {name}: {type(e).__name__}: {e}")
            failed += 1

    print(f"\nCLI tests: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
