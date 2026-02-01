#!/usr/bin/env python3
"""Run all RLM tests.

Can be run with or without pytest:
  python tests/run_all.py        # Uses built-in runner
  pytest tests/                  # Uses pytest (if installed)
"""

import sys
from pathlib import Path

# Ensure we can import test modules
sys.path.insert(0, str(Path(__file__).parent))


def run_all_tests():
    """Run all test modules."""
    print("=" * 60)
    print("Claude Memory Test Suite")
    print("=" * 60)

    from test_analyzers import run_tests as run_analyzer_tests
    from test_database import run_tests as run_database_tests
    from test_indexing import run_tests as run_indexing_tests
    from test_memory import run_tests as run_memory_tests
    from test_edge_cases import run_tests as run_edge_case_tests
    from test_hooks import run_tests as run_hook_tests
    from test_embeddings import run_tests as run_embedding_tests
    from test_cli import run_tests as run_cli_tests

    results = []

    # Run each test module
    results.append(("Analyzers", run_analyzer_tests()))
    results.append(("Database", run_database_tests()))
    results.append(("Indexing", run_indexing_tests()))
    results.append(("Memory", run_memory_tests()))
    results.append(("Edge Cases", run_edge_case_tests()))
    results.append(("Session Hooks", run_hook_tests()))
    results.append(("Embeddings", run_embedding_tests()))
    results.append(("CLI Commands", run_cli_tests()))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
