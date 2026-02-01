#!/usr/bin/env python3
"""RLM Test Suite Runner.

This is a convenience wrapper that runs all tests from the tests/ directory.

Usage:
    python test_rlm.py              # Run all tests with built-in runner
    python -m pytest tests/         # Run with pytest (if installed)
    python tests/run_all.py         # Run directly from tests directory
"""

import sys
from pathlib import Path

# Add tests directory to path
sys.path.insert(0, str(Path(__file__).parent / "tests"))

from run_all import run_all_tests

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
