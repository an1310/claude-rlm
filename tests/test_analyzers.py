"""Tests for language analyzers."""

import sys
from pathlib import Path

# Ensure imports work
sys.path.insert(0, str(Path(__file__).parent.parent / ".claude" / "skills" / "rlm" / "scripts"))

from rlm_repl import PythonAnalyzer, JavaScriptAnalyzer, JavaAnalyzer


def test_python_analyzer():
    """Test Python AST-based analyzer."""
    code = '''
class MyClass:
    def method_one(self):
        pass

    async def method_two(self):
        pass

def standalone_function():
    pass
'''
    analyzer = PythonAnalyzer("test.py", code)
    chunks = analyzer.extract_chunks()
    symbols = analyzer.extract_symbols()

    # Check chunks
    assert len(chunks) == 4, f"Expected 4 chunks, got {len(chunks)}"

    chunk_names = [c.symbol_name for c in chunks]
    assert "MyClass" in chunk_names
    assert "standalone_function" in chunk_names
    assert "method_one" in chunk_names
    assert "method_two" in chunk_names

    # Check parent tracking
    method_chunks = [c for c in chunks if c.chunk_type == "method"]
    assert len(method_chunks) == 2
    for mc in method_chunks:
        assert mc.parent_symbol == "MyClass", f"Method {mc.symbol_name} should have parent MyClass"

    # Check symbols
    assert len(symbols) == 4
    symbol_types = {s[0]: s[1] for s in symbols}
    assert symbol_types["MyClass"] == "class"
    assert symbol_types["standalone_function"] == "function"


def test_python_analyzer_imports():
    """Test Python import extraction."""
    code = '''
import os
import sys
from typing import Optional, List
from .relative import something
from ..parent import other
'''
    analyzer = PythonAnalyzer("test.py", code)
    imports = analyzer.extract_imports()

    assert len(imports) >= 4
    module_names = [i.module_name for i in imports]
    assert "os" in module_names
    assert "sys" in module_names
    assert "typing" in module_names

    # Check relative imports
    relative_imports = [i for i in imports if i.is_relative]
    assert len(relative_imports) >= 2


def test_javascript_analyzer():
    """Test JavaScript regex-based analyzer."""
    code = '''
import { something } from 'module';

export class ApiController {
    async handleRequest(req, res) {
        return res.json({});
    }
}

export async function processData(data) {
    return data;
}

const helper = async () => {
    return true;
};
'''
    analyzer = JavaScriptAnalyzer("test.js", code)
    chunks = analyzer.extract_chunks()
    symbols = analyzer.extract_symbols()
    imports = analyzer.extract_imports()

    # Check chunks
    assert len(chunks) >= 3, f"Expected at least 3 chunks, got {len(chunks)}"

    chunk_types = {c.symbol_name: c.chunk_type for c in chunks}
    assert chunk_types.get("ApiController") == "class"
    assert chunk_types.get("processData") == "function"
    assert chunk_types.get("helper") == "arrow_function"

    # Check imports
    assert len(imports) >= 1
    assert imports[0].module_name == "module"


def test_javascript_analyzer_commonjs():
    """Test CommonJS require extraction."""
    code = '''
const express = require('express');
const { Router } = require('express');
const path = require('path');
'''
    analyzer = JavaScriptAnalyzer("test.js", code)
    imports = analyzer.extract_imports()

    assert len(imports) >= 2
    module_names = [i.module_name for i in imports]
    assert "express" in module_names
    assert "path" in module_names


def test_java_analyzer():
    """Test Java regex-based analyzer."""
    code = '''
package com.example;

import java.util.List;
import org.springframework.stereotype.Service;

@Service
public class UserService {

    private final UserRepository repository;

    public User findById(Long id) {
        return repository.findById(id).orElse(null);
    }

    public List<User> findAll() {
        return repository.findAll();
    }
}

interface UserRepository {
    Optional<User> findById(Long id);
}
'''
    analyzer = JavaAnalyzer("UserService.java", code)
    chunks = analyzer.extract_chunks()
    symbols = analyzer.extract_symbols()
    imports = analyzer.extract_imports()

    # Check chunks - should have class, interface, and methods
    assert len(chunks) >= 2, f"Expected at least 2 chunks, got {len(chunks)}"

    # Find specific chunk types
    class_chunks = [c for c in chunks if c.chunk_type == "class"]
    interface_chunks = [c for c in chunks if c.chunk_type == "interface"]
    method_chunks = [c for c in chunks if c.chunk_type == "method"]

    assert len(class_chunks) >= 1, f"Expected at least 1 class, got {len(class_chunks)}"
    assert len(interface_chunks) >= 1, f"Expected at least 1 interface, got {len(interface_chunks)}"

    # Verify UserRepository is found as interface
    interface_names = [c.symbol_name for c in interface_chunks]
    assert "UserRepository" in interface_names, f"UserRepository not found in interfaces: {interface_names}"

    # Check imports
    assert len(imports) >= 2, f"Expected at least 2 imports, got {len(imports)}"
    module_names = [i.module_name for i in imports]
    assert "java.util.List" in module_names, f"java.util.List not in {module_names}"


def test_java_analyzer_annotations():
    """Test Java annotation handling."""
    code = '''
@RestController
@RequestMapping("/api")
public class ApiController {

    @Autowired
    private Service service;

    @GetMapping("/{id}")
    public Response get(@PathVariable Long id) {
        return service.get(id);
    }
}
'''
    analyzer = JavaAnalyzer("ApiController.java", code)
    chunks = analyzer.extract_chunks()

    # Class chunk should include annotations
    class_chunk = next((c for c in chunks if c.symbol_name == "ApiController"), None)
    assert class_chunk is not None
    assert "@RestController" in class_chunk.content or class_chunk.start_line <= 2


def run_tests():
    """Run all analyzer tests."""
    print("=" * 60)
    print("Testing Language Analyzers")
    print("=" * 60)

    tests = [
        ("Python analyzer", test_python_analyzer),
        ("Python imports", test_python_analyzer_imports),
        ("JavaScript analyzer", test_javascript_analyzer),
        ("JavaScript CommonJS", test_javascript_analyzer_commonjs),
        ("Java analyzer", test_java_analyzer),
        ("Java annotations", test_java_analyzer_annotations),
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

    print(f"\nAnalyzer tests: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
