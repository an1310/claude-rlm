#!/usr/bin/env python3
"""RLM: SQLite-backed codebase indexer for Recursive Language Model workflows.

This script provides a persistent, indexed view of codebases using:
- SQLite with WAL mode for concurrent access and crash recovery
- FTS5 full-text search for instant content queries
- AST-based (Python) and regex-based (JS/TS/Java) code analysis
- Incremental indexing that only re-processes changed files
- Multi-repository support with named repos

Typical flow:
  1) Index codebase(s):
       python rlm_repl.py init /path/to/repo --name my-repo
       python rlm_repl.py init /path/to/other --name other-repo

  2) Search and analyze:
       python rlm_repl.py search --symbol UserAuth
       python rlm_repl.py search --pattern "authenticate" --fts

  3) Execute code with helpers:
       python rlm_repl.py exec -c "print(stats())"

Based on the RLM paper: arXiv:2512.24601
"""

import argparse
import ast
import hashlib
import io
import re
import sqlite3
import sys
import textwrap
import traceback
from abc import ABC, abstractmethod
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_STATE_DIR = Path(".claude/rlm_state")
DEFAULT_DB_PATH = DEFAULT_STATE_DIR / "index.db"
DEFAULT_CHUNKS_DIR = DEFAULT_STATE_DIR / "chunks"
DEFAULT_MAX_OUTPUT_CHARS = 8000

# Language extensions mapping
LANGUAGE_EXTENSIONS = {
    ".py": "python",
    ".pyw": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".mts": "typescript",
    ".cts": "typescript",
    ".java": "java",
}

DEFAULT_EXTENSIONS = list(LANGUAGE_EXTENSIONS.keys())


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class Chunk:
    """Represents a semantic unit of code (function, class, method, etc.)."""
    chunk_type: str      # 'function', 'class', 'method', 'interface', etc.
    symbol_name: str
    start_line: int
    end_line: int
    content: str
    parent_symbol: str | None = None


@dataclass
class Import:
    """Represents an import statement."""
    module_name: str
    imported_names: list[str]
    is_relative: bool
    line_number: int


# =============================================================================
# Database
# =============================================================================

class Database:
    """SQLite database with WAL mode, FTS5, and transaction batching."""

    SCHEMA = """
    -- Repository metadata
    CREATE TABLE IF NOT EXISTS repos (
        repo_id INTEGER PRIMARY KEY AUTOINCREMENT,
        repo_name TEXT UNIQUE NOT NULL,
        root_path TEXT NOT NULL,
        file_count INTEGER DEFAULT 0,
        indexed_at TEXT DEFAULT (datetime('now'))
    );

    -- File metadata
    CREATE TABLE IF NOT EXISTS files (
        file_id INTEGER PRIMARY KEY AUTOINCREMENT,
        repo_id INTEGER REFERENCES repos(repo_id) ON DELETE CASCADE,
        filepath TEXT NOT NULL,
        language TEXT,
        hash TEXT,
        size_bytes INTEGER,
        indexed_at TEXT DEFAULT (datetime('now')),
        UNIQUE(repo_id, filepath)
    );

    -- Code chunks
    CREATE TABLE IF NOT EXISTS chunks (
        chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_id INTEGER NOT NULL REFERENCES files(file_id) ON DELETE CASCADE,
        chunk_type TEXT NOT NULL,
        symbol_name TEXT NOT NULL,
        parent_symbol TEXT,
        start_line INTEGER NOT NULL,
        end_line INTEGER NOT NULL,
        content TEXT NOT NULL
    );

    -- FTS5 virtual table for fast content search
    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
        content,
        symbol_name,
        chunk_id UNINDEXED,
        content='chunks',
        content_rowid='chunk_id'
    );

    -- Triggers to keep FTS in sync
    CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
        INSERT INTO chunks_fts(rowid, content, symbol_name, chunk_id)
        VALUES (new.chunk_id, new.content, new.symbol_name, new.chunk_id);
    END;

    CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
        INSERT INTO chunks_fts(chunks_fts, rowid, content, symbol_name, chunk_id)
        VALUES ('delete', old.chunk_id, old.content, old.symbol_name, old.chunk_id);
    END;

    CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
        INSERT INTO chunks_fts(chunks_fts, rowid, content, symbol_name, chunk_id)
        VALUES ('delete', old.chunk_id, old.content, old.symbol_name, old.chunk_id);
        INSERT INTO chunks_fts(rowid, content, symbol_name, chunk_id)
        VALUES (new.chunk_id, new.content, new.symbol_name, new.chunk_id);
    END;

    -- Symbol definitions
    CREATE TABLE IF NOT EXISTS symbols (
        symbol_id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_id INTEGER NOT NULL REFERENCES files(file_id) ON DELETE CASCADE,
        symbol_name TEXT NOT NULL,
        symbol_type TEXT NOT NULL,
        parent_symbol TEXT,
        definition_line INTEGER NOT NULL,
        chunk_id INTEGER REFERENCES chunks(chunk_id) ON DELETE SET NULL
    );

    -- Import relationships
    CREATE TABLE IF NOT EXISTS imports (
        import_id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_id INTEGER NOT NULL REFERENCES files(file_id) ON DELETE CASCADE,
        module_name TEXT NOT NULL,
        imported_names TEXT,
        is_relative INTEGER DEFAULT 0,
        line_number INTEGER
    );

    -- Indexes for common queries
    CREATE INDEX IF NOT EXISTS idx_files_repo ON files(repo_id);
    CREATE INDEX IF NOT EXISTS idx_files_language ON files(language);
    CREATE INDEX IF NOT EXISTS idx_files_hash ON files(hash);
    CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_id);
    CREATE INDEX IF NOT EXISTS idx_chunks_type ON chunks(chunk_type);
    CREATE INDEX IF NOT EXISTS idx_chunks_symbol ON chunks(symbol_name);
    CREATE INDEX IF NOT EXISTS idx_symbols_file ON symbols(file_id);
    CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(symbol_name);
    CREATE INDEX IF NOT EXISTS idx_symbols_type ON symbols(symbol_type);
    CREATE INDEX IF NOT EXISTS idx_imports_file ON imports(file_id);
    CREATE INDEX IF NOT EXISTS idx_imports_module ON imports(module_name);
    """

    def __init__(self, db_path: Path | str):
        """Initialize database with WAL mode."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

        # Enable WAL mode for better concurrency and crash recovery
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA synchronous=NORMAL")

        # Create schema
        self._conn.executescript(self.SCHEMA)
        self._conn.commit()

        self._in_transaction = False

    def begin_transaction(self):
        """Start a transaction for batching operations."""
        if not self._in_transaction:
            # Check if we're already in a transaction (autocommit mode)
            if self._conn.in_transaction:
                self._in_transaction = True
            else:
                self._conn.execute("BEGIN IMMEDIATE")
                self._in_transaction = True

    def commit(self):
        """Commit the current transaction."""
        if self._in_transaction:
            self._conn.commit()
            self._in_transaction = False

    def rollback(self):
        """Rollback the current transaction."""
        if self._in_transaction:
            self._conn.rollback()
            self._in_transaction = False

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        """Execute a SQL statement."""
        return self._conn.execute(sql, params)

    def executemany(self, sql: str, params_list: list[tuple]) -> sqlite3.Cursor:
        """Execute a SQL statement with multiple parameter sets."""
        return self._conn.executemany(sql, params_list)

    def query(self, sql: str, params: tuple = ()) -> list[dict]:
        """Execute a query and return results as list of dicts."""
        cursor = self._conn.execute(sql, params)
        columns = [d[0] for d in cursor.description] if cursor.description else []
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def search_content_fts(self, query: str, limit: int = 100) -> list[dict]:
        """Search chunk content using FTS5."""
        # Escape FTS5 special characters
        safe_query = query.replace('"', '""')
        sql = """
            SELECT c.*, f.filepath, r.repo_name
            FROM chunks_fts fts
            JOIN chunks c ON c.chunk_id = fts.chunk_id
            JOIN files f ON c.file_id = f.file_id
            LEFT JOIN repos r ON f.repo_id = r.repo_id
            WHERE chunks_fts MATCH ?
            LIMIT ?
        """
        return self.query(sql, (f'"{safe_query}"', limit))

    def get_or_create_repo(self, repo_name: str, root_path: str) -> int:
        """Get or create a repository, returning its ID."""
        result = self.query(
            "SELECT repo_id FROM repos WHERE repo_name = ?",
            (repo_name,)
        )
        if result:
            # Update root path if it changed
            self.execute(
                "UPDATE repos SET root_path = ? WHERE repo_id = ?",
                (root_path, result[0]['repo_id'])
            )
            return result[0]['repo_id']

        cursor = self.execute(
            "INSERT INTO repos (repo_name, root_path) VALUES (?, ?)",
            (repo_name, root_path)
        )
        return cursor.lastrowid

    def list_repos(self) -> list[dict]:
        """List all indexed repositories."""
        return self.query("""
            SELECT r.*, COUNT(f.file_id) as actual_file_count
            FROM repos r
            LEFT JOIN files f ON r.repo_id = f.repo_id
            GROUP BY r.repo_id
            ORDER BY r.repo_name
        """)

    def delete_repo(self, repo_name: str) -> bool:
        """Delete a repository and all its data."""
        result = self.query(
            "SELECT repo_id FROM repos WHERE repo_name = ?",
            (repo_name,)
        )
        if not result:
            return False

        repo_id = result[0]['repo_id']
        self.execute("DELETE FROM repos WHERE repo_id = ?", (repo_id,))
        self._conn.commit()
        return True

    def get_stats(self) -> dict:
        """Get database statistics."""
        stats = {}

        # File counts
        result = self.query("SELECT COUNT(*) as c FROM files")
        stats['files'] = result[0]['c']

        result = self.query("SELECT COUNT(*) as c FROM chunks")
        stats['chunks'] = result[0]['c']

        result = self.query("SELECT COUNT(*) as c FROM symbols")
        stats['symbols'] = result[0]['c']

        result = self.query("SELECT COUNT(*) as c FROM imports")
        stats['imports'] = result[0]['c']

        result = self.query("SELECT COUNT(*) as c FROM repos")
        stats['repos'] = result[0]['c']

        # Language breakdown
        result = self.query("""
            SELECT language, COUNT(*) as count
            FROM files
            WHERE language IS NOT NULL
            GROUP BY language
            ORDER BY count DESC
        """)
        stats['languages'] = {r['language']: r['count'] for r in result}

        # Chunk type breakdown
        result = self.query("""
            SELECT chunk_type, COUNT(*) as count
            FROM chunks
            GROUP BY chunk_type
            ORDER BY count DESC
        """)
        stats['chunk_types'] = {r['chunk_type']: r['count'] for r in result}

        return stats

    def vacuum(self):
        """Vacuum the database and optimize FTS5."""
        self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self._conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('optimize')")
        self._conn.execute("VACUUM")
        self._conn.commit()

    def close(self):
        """Close the database connection."""
        if self._in_transaction:
            self.commit()
        self._conn.close()


# =============================================================================
# Language Analyzers
# =============================================================================

class BaseAnalyzer(ABC):
    """Base class for language analyzers."""

    def __init__(self, filepath: str, content: str):
        self.filepath = filepath
        self.content = content
        self.lines = content.split('\n')

    @abstractmethod
    def extract_chunks(self) -> list[Chunk]:
        """Extract code chunks from the content."""
        pass

    @abstractmethod
    def extract_symbols(self) -> list[tuple]:
        """Extract symbols as (name, type, line, parent) tuples."""
        pass

    @abstractmethod
    def extract_imports(self) -> list[Import]:
        """Extract import statements."""
        pass

    def _get_content_for_lines(self, start: int, end: int) -> str:
        """Get content for a line range (1-indexed)."""
        return '\n'.join(self.lines[start - 1:end])


class PythonAnalyzer(BaseAnalyzer):
    """AST-based Python analyzer."""

    def extract_chunks(self) -> list[Chunk]:
        """Extract functions, classes, and methods using AST."""
        chunks = []
        try:
            tree = ast.parse(self.content)
        except SyntaxError:
            return chunks

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                parent = self._find_parent_class(tree, node)
                chunk_type = 'method' if parent else ('async_function' if isinstance(node, ast.AsyncFunctionDef) else 'function')
                chunks.append(Chunk(
                    chunk_type=chunk_type,
                    symbol_name=node.name,
                    start_line=node.lineno,
                    end_line=node.end_lineno or node.lineno,
                    content=self._get_content_for_lines(node.lineno, node.end_lineno or node.lineno),
                    parent_symbol=parent
                ))
            elif isinstance(node, ast.ClassDef):
                chunks.append(Chunk(
                    chunk_type='class',
                    symbol_name=node.name,
                    start_line=node.lineno,
                    end_line=node.end_lineno or node.lineno,
                    content=self._get_content_for_lines(node.lineno, node.end_lineno or node.lineno),
                    parent_symbol=None
                ))

        return chunks

    def _find_parent_class(self, tree: ast.Module, target: ast.AST) -> str | None:
        """Find the parent class of a function/method."""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for child in ast.iter_child_nodes(node):
                    if child is target:
                        return node.name
        return None

    def extract_symbols(self) -> list[tuple]:
        """Extract symbol definitions."""
        symbols = []
        try:
            tree = ast.parse(self.content)
        except SyntaxError:
            return symbols

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                parent = self._find_parent_class(tree, node)
                sym_type = 'method' if parent else ('async_function' if isinstance(node, ast.AsyncFunctionDef) else 'function')
                symbols.append((node.name, sym_type, node.lineno, parent))
            elif isinstance(node, ast.ClassDef):
                symbols.append((node.name, 'class', node.lineno, None))
            elif isinstance(node, ast.Assign):
                # Module-level variable assignments
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        symbols.append((target.id, 'variable', node.lineno, None))

        return symbols

    def extract_imports(self) -> list[Import]:
        """Extract import statements."""
        imports = []
        try:
            tree = ast.parse(self.content)
        except SyntaxError:
            return imports

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(Import(
                        module_name=alias.name,
                        imported_names=[alias.asname or alias.name],
                        is_relative=False,
                        line_number=node.lineno
                    ))
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                names = [alias.name for alias in node.names]
                imports.append(Import(
                    module_name=module,
                    imported_names=names,
                    is_relative=node.level > 0,
                    line_number=node.lineno
                ))

        return imports


class JavaScriptAnalyzer(BaseAnalyzer):
    """Regex-based JavaScript/TypeScript analyzer."""

    # Patterns for JavaScript/TypeScript
    PATTERNS = {
        'class': re.compile(
            r'(?:export\s+)?(?:default\s+)?class\s+(\w+)',
            re.MULTILINE
        ),
        'function': re.compile(
            r'(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(',
            re.MULTILINE
        ),
        'arrow_function': re.compile(
            r'(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?(?:\([^)]*\)|[\w]+)\s*=>\s*',
            re.MULTILINE
        ),
        'interface': re.compile(
            r'(?:export\s+)?interface\s+(\w+)',
            re.MULTILINE
        ),
        'type': re.compile(
            r'(?:export\s+)?type\s+(\w+)\s*=',
            re.MULTILINE
        ),
        'import_es6': re.compile(
            r'import\s+(?:{[^}]+}|\*\s+as\s+\w+|\w+)\s+from\s+[\'"]([^\'"]+)[\'"]',
            re.MULTILINE
        ),
        'require': re.compile(
            r'(?:const|let|var)\s+(?:{[^}]+}|\w+)\s*=\s*require\s*\(\s*[\'"]([^\'"]+)[\'"]',
            re.MULTILINE
        ),
    }

    def extract_chunks(self) -> list[Chunk]:
        """Extract code chunks using regex patterns."""
        chunks = []

        # Classes
        for match in self.PATTERNS['class'].finditer(self.content):
            name = match.group(1)
            start_line = self.content[:match.start()].count('\n') + 1
            end_line = self._find_block_end(match.start())
            chunks.append(Chunk(
                chunk_type='class',
                symbol_name=name,
                start_line=start_line,
                end_line=end_line,
                content=self._get_content_for_lines(start_line, end_line),
                parent_symbol=None
            ))

        # Functions
        for match in self.PATTERNS['function'].finditer(self.content):
            name = match.group(1)
            start_line = self.content[:match.start()].count('\n') + 1
            end_line = self._find_block_end(match.start())
            chunks.append(Chunk(
                chunk_type='function',
                symbol_name=name,
                start_line=start_line,
                end_line=end_line,
                content=self._get_content_for_lines(start_line, end_line),
                parent_symbol=None
            ))

        # Arrow functions
        for match in self.PATTERNS['arrow_function'].finditer(self.content):
            name = match.group(1)
            start_line = self.content[:match.start()].count('\n') + 1
            end_line = self._find_arrow_function_end(match.start())
            chunks.append(Chunk(
                chunk_type='arrow_function',
                symbol_name=name,
                start_line=start_line,
                end_line=end_line,
                content=self._get_content_for_lines(start_line, end_line),
                parent_symbol=None
            ))

        # Interfaces (TypeScript)
        for match in self.PATTERNS['interface'].finditer(self.content):
            name = match.group(1)
            start_line = self.content[:match.start()].count('\n') + 1
            end_line = self._find_block_end(match.start())
            chunks.append(Chunk(
                chunk_type='interface',
                symbol_name=name,
                start_line=start_line,
                end_line=end_line,
                content=self._get_content_for_lines(start_line, end_line),
                parent_symbol=None
            ))

        # Type aliases (TypeScript)
        for match in self.PATTERNS['type'].finditer(self.content):
            name = match.group(1)
            start_line = self.content[:match.start()].count('\n') + 1
            # Types typically end at semicolon or newline
            end_pos = self.content.find(';', match.end())
            if end_pos == -1:
                end_pos = self.content.find('\n', match.end())
            end_line = self.content[:end_pos].count('\n') + 1 if end_pos != -1 else start_line
            chunks.append(Chunk(
                chunk_type='type',
                symbol_name=name,
                start_line=start_line,
                end_line=end_line,
                content=self._get_content_for_lines(start_line, end_line),
                parent_symbol=None
            ))

        return chunks

    def _find_block_end(self, start_pos: int) -> int:
        """Find the end of a brace-delimited block."""
        brace_start = self.content.find('{', start_pos)
        if brace_start == -1:
            return self.content[:start_pos].count('\n') + 1

        depth = 0
        pos = brace_start
        while pos < len(self.content):
            char = self.content[pos]
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    return self.content[:pos + 1].count('\n') + 1
            pos += 1

        return len(self.lines)

    def _find_arrow_function_end(self, start_pos: int) -> int:
        """Find the end of an arrow function."""
        arrow_pos = self.content.find('=>', start_pos)
        if arrow_pos == -1:
            return self.content[:start_pos].count('\n') + 1

        # Check if it's a block body or expression
        after_arrow = self.content[arrow_pos + 2:].lstrip()
        if after_arrow.startswith('{'):
            return self._find_block_end(arrow_pos)
        else:
            # Expression body - find semicolon or newline
            end_pos = self.content.find(';', arrow_pos)
            if end_pos == -1:
                end_pos = self.content.find('\n', arrow_pos)
            return self.content[:end_pos].count('\n') + 1 if end_pos != -1 else len(self.lines)

    def extract_symbols(self) -> list[tuple]:
        """Extract symbol definitions."""
        symbols = []

        for pattern_name, pattern in self.PATTERNS.items():
            if pattern_name.startswith('import') or pattern_name == 'require':
                continue

            sym_type = pattern_name
            for match in pattern.finditer(self.content):
                name = match.group(1)
                line = self.content[:match.start()].count('\n') + 1
                symbols.append((name, sym_type, line, None))

        return symbols

    def extract_imports(self) -> list[Import]:
        """Extract import statements."""
        imports = []

        # ES6 imports
        for match in self.PATTERNS['import_es6'].finditer(self.content):
            module = match.group(1)
            line = self.content[:match.start()].count('\n') + 1
            imports.append(Import(
                module_name=module,
                imported_names=[],
                is_relative=module.startswith('.'),
                line_number=line
            ))

        # CommonJS require
        for match in self.PATTERNS['require'].finditer(self.content):
            module = match.group(1)
            line = self.content[:match.start()].count('\n') + 1
            imports.append(Import(
                module_name=module,
                imported_names=[],
                is_relative=module.startswith('.'),
                line_number=line
            ))

        return imports


class TypeScriptAnalyzer(JavaScriptAnalyzer):
    """TypeScript analyzer - extends JavaScript with additional patterns."""
    pass  # JavaScript analyzer already handles interfaces and types


class JavaAnalyzer(BaseAnalyzer):
    """Regex-based Java analyzer."""

    PATTERNS = {
        'class': re.compile(
            r'(?:(?:public|private|protected|abstract|final|static)\s+)*class\s+(\w+)',
            re.MULTILINE
        ),
        'interface': re.compile(
            r'(?:(?:public|private|protected)\s+)?interface\s+(\w+)',
            re.MULTILINE
        ),
        'enum': re.compile(
            r'(?:(?:public|private|protected)\s+)?enum\s+(\w+)',
            re.MULTILINE
        ),
        'method': re.compile(
            r'(?:(?:public|private|protected|abstract|static|final|synchronized|native)\s+)*(?:<[\w\s,<>?]+>\s+)?(\w+)\s+(\w+)\s*\([^)]*\)\s*(?:throws\s+[\w\s,]+)?\s*\{',
            re.MULTILINE
        ),
        'constructor': re.compile(
            r'(?:(?:public|private|protected)\s+)?(\w+)\s*\([^)]*\)\s*(?:throws\s+[\w\s,]+)?\s*\{',
            re.MULTILINE
        ),
        'import': re.compile(
            r'import\s+(?:static\s+)?([\w.]+(?:\.\*)?)\s*;',
            re.MULTILINE
        ),
        'package': re.compile(
            r'package\s+([\w.]+)\s*;',
            re.MULTILINE
        ),
    }

    def extract_chunks(self) -> list[Chunk]:
        """Extract classes, interfaces, enums, and methods."""
        chunks = []
        class_ranges = []  # Track class boundaries for parent detection

        # Classes
        for match in self.PATTERNS['class'].finditer(self.content):
            name = match.group(1)
            # Look back for annotations
            actual_start = self._find_annotation_start(match.start())
            actual_start_line = self.content[:actual_start].count('\n') + 1
            end_line = self._find_block_end(match.start())
            class_ranges.append((name, actual_start_line, end_line))
            chunks.append(Chunk(
                chunk_type='class',
                symbol_name=name,
                start_line=actual_start_line,
                end_line=end_line,
                content=self._get_content_for_lines(actual_start_line, end_line),
                parent_symbol=None
            ))

        # Interfaces
        for match in self.PATTERNS['interface'].finditer(self.content):
            name = match.group(1)
            actual_start = self._find_annotation_start(match.start())
            actual_start_line = self.content[:actual_start].count('\n') + 1
            end_line = self._find_block_end(match.start())
            chunks.append(Chunk(
                chunk_type='interface',
                symbol_name=name,
                start_line=actual_start_line,
                end_line=end_line,
                content=self._get_content_for_lines(actual_start_line, end_line),
                parent_symbol=None
            ))

        # Enums
        for match in self.PATTERNS['enum'].finditer(self.content):
            name = match.group(1)
            actual_start = self._find_annotation_start(match.start())
            actual_start_line = self.content[:actual_start].count('\n') + 1
            end_line = self._find_block_end(match.start())
            chunks.append(Chunk(
                chunk_type='enum',
                symbol_name=name,
                start_line=actual_start_line,
                end_line=end_line,
                content=self._get_content_for_lines(actual_start_line, end_line),
                parent_symbol=None
            ))

        # Methods
        for match in self.PATTERNS['method'].finditer(self.content):
            return_type = match.group(1)
            name = match.group(2)
            # Skip if it looks like a constructor (return type matches a known class)
            if return_type in [c[0] for c in class_ranges]:
                continue
            actual_start = self._find_annotation_start(match.start())
            actual_start_line = self.content[:actual_start].count('\n') + 1
            end_line = self._find_block_end(match.start())

            # Find parent class
            parent = None
            for class_name, class_start, class_end in class_ranges:
                if class_start < actual_start_line <= class_end:
                    parent = class_name
                    break

            chunks.append(Chunk(
                chunk_type='method',
                symbol_name=name,
                start_line=actual_start_line,
                end_line=end_line,
                content=self._get_content_for_lines(actual_start_line, end_line),
                parent_symbol=parent
            ))

        return chunks

    def _find_annotation_start(self, pos: int) -> int:
        """Look backwards for annotation(s) before a declaration."""
        # Find the start of the current line
        line_start = self.content.rfind('\n', 0, pos)
        if line_start == -1:
            line_start = 0
        else:
            line_start += 1

        # Look backwards for @annotation lines
        while line_start > 0:
            prev_line_end = line_start - 1
            prev_line_start = self.content.rfind('\n', 0, prev_line_end)
            if prev_line_start == -1:
                prev_line_start = 0
            else:
                prev_line_start += 1

            prev_line = self.content[prev_line_start:prev_line_end].strip()
            if prev_line.startswith('@'):
                line_start = prev_line_start
            else:
                break

        return line_start

    def _find_block_end(self, start_pos: int) -> int:
        """Find the end of a brace-delimited block."""
        brace_start = self.content.find('{', start_pos)
        if brace_start == -1:
            return self.content[:start_pos].count('\n') + 1

        depth = 0
        pos = brace_start
        while pos < len(self.content):
            char = self.content[pos]
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    return self.content[:pos + 1].count('\n') + 1
            pos += 1

        return len(self.lines)

    def extract_symbols(self) -> list[tuple]:
        """Extract symbol definitions."""
        symbols = []
        class_ranges = []

        # Classes
        for match in self.PATTERNS['class'].finditer(self.content):
            name = match.group(1)
            line = self.content[:match.start()].count('\n') + 1
            end_line = self._find_block_end(match.start())
            class_ranges.append((name, line, end_line))
            symbols.append((name, 'class', line, None))

        # Interfaces
        for match in self.PATTERNS['interface'].finditer(self.content):
            name = match.group(1)
            line = self.content[:match.start()].count('\n') + 1
            symbols.append((name, 'interface', line, None))

        # Enums
        for match in self.PATTERNS['enum'].finditer(self.content):
            name = match.group(1)
            line = self.content[:match.start()].count('\n') + 1
            symbols.append((name, 'enum', line, None))

        # Methods
        for match in self.PATTERNS['method'].finditer(self.content):
            return_type = match.group(1)
            name = match.group(2)
            if return_type in [c[0] for c in class_ranges]:
                continue
            line = self.content[:match.start()].count('\n') + 1

            parent = None
            for class_name, class_start, class_end in class_ranges:
                if class_start < line <= class_end:
                    parent = class_name
                    break

            symbols.append((name, 'method', line, parent))

        return symbols

    def extract_imports(self) -> list[Import]:
        """Extract import statements."""
        imports = []

        for match in self.PATTERNS['import'].finditer(self.content):
            module = match.group(1)
            line = self.content[:match.start()].count('\n') + 1
            imports.append(Import(
                module_name=module,
                imported_names=[],
                is_relative=False,
                line_number=line
            ))

        return imports


# =============================================================================
# Codebase Indexer
# =============================================================================

class CodebaseIndexer:
    """Indexes a codebase into the database."""

    def __init__(self, db: Database, root_path: Path | str, repo_name: str | None = None,
                 extensions: list[str] | None = None):
        self.db = db
        self.root_path = Path(root_path).resolve()
        self.repo_name = repo_name or self.root_path.name
        self.extensions = set(extensions or DEFAULT_EXTENSIONS)

    def index_codebase(self, full: bool = False) -> dict:
        """Index the codebase, returns statistics."""
        stats = {'indexed': 0, 'skipped': 0, 'errors': 0, 'total': 0}

        # Get or create repo
        repo_id = self.db.get_or_create_repo(self.repo_name, str(self.root_path))

        # If full reindex, delete existing data
        if full:
            self.db.execute("DELETE FROM files WHERE repo_id = ?", (repo_id,))
            self.db.commit()

        # Get existing file hashes
        existing_files = {}
        for row in self.db.query(
            "SELECT filepath, hash FROM files WHERE repo_id = ?",
            (repo_id,)
        ):
            existing_files[row['filepath']] = row['hash']

        # Scan files
        files_to_index = []
        for filepath in self._scan_files():
            stats['total'] += 1
            rel_path = str(filepath.relative_to(self.root_path))

            try:
                content = filepath.read_text(encoding='utf-8', errors='replace')
                file_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

                if rel_path in existing_files and existing_files[rel_path] == file_hash:
                    stats['skipped'] += 1
                    continue

                files_to_index.append((filepath, rel_path, content, file_hash))
            except Exception:
                stats['errors'] += 1
                continue

        # Index files with transaction batching
        if files_to_index:
            self.db.begin_transaction()
            try:
                for filepath, rel_path, content, file_hash in files_to_index:
                    self._index_file(repo_id, filepath, rel_path, content, file_hash)
                    stats['indexed'] += 1

                # Update repo file count
                self.db.execute(
                    "UPDATE repos SET file_count = (SELECT COUNT(*) FROM files WHERE repo_id = ?), indexed_at = datetime('now') WHERE repo_id = ?",
                    (repo_id, repo_id)
                )
                self.db.commit()
            except Exception:
                self.db.rollback()
                raise

        return stats

    def _scan_files(self):
        """Scan for source files."""
        for filepath in self.root_path.rglob('*'):
            if filepath.is_file() and filepath.suffix in self.extensions:
                # Skip common non-source directories
                parts = filepath.parts
                if any(part.startswith('.') or part in ('node_modules', 'venv', '__pycache__', 'dist', 'build', 'target')
                       for part in parts):
                    continue
                yield filepath

    def _index_file(self, repo_id: int, filepath: Path, rel_path: str, content: str, file_hash: str):
        """Index a single file."""
        # Determine language
        language = LANGUAGE_EXTENSIONS.get(filepath.suffix)

        # Delete existing file data if it exists
        existing = self.db.query(
            "SELECT file_id FROM files WHERE repo_id = ? AND filepath = ?",
            (repo_id, rel_path)
        )
        if existing:
            self.db.execute("DELETE FROM files WHERE file_id = ?", (existing[0]['file_id'],))

        # Insert file
        cursor = self.db.execute(
            "INSERT INTO files (repo_id, filepath, language, hash, size_bytes) VALUES (?, ?, ?, ?, ?)",
            (repo_id, rel_path, language, file_hash, len(content.encode()))
        )
        file_id = cursor.lastrowid

        # Get analyzer
        analyzer = self._get_analyzer(filepath, content)
        if not analyzer:
            return

        # Extract and insert chunks
        chunks = analyzer.extract_chunks()
        chunk_map = {}  # Map (symbol_name, start_line) -> chunk_id

        for chunk in chunks:
            cursor = self.db.execute(
                """INSERT INTO chunks (file_id, chunk_type, symbol_name, parent_symbol, start_line, end_line, content)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (file_id, chunk.chunk_type, chunk.symbol_name, chunk.parent_symbol,
                 chunk.start_line, chunk.end_line, chunk.content)
            )
            chunk_map[(chunk.symbol_name, chunk.start_line)] = cursor.lastrowid

        # Extract and insert symbols
        symbols = analyzer.extract_symbols()
        for name, sym_type, line, parent in symbols:
            chunk_id = chunk_map.get((name, line))
            self.db.execute(
                """INSERT INTO symbols (file_id, symbol_name, symbol_type, parent_symbol, definition_line, chunk_id)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (file_id, name, sym_type, parent, line, chunk_id)
            )

        # Extract and insert imports
        imports = analyzer.extract_imports()
        for imp in imports:
            self.db.execute(
                """INSERT INTO imports (file_id, module_name, imported_names, is_relative, line_number)
                   VALUES (?, ?, ?, ?, ?)""",
                (file_id, imp.module_name, ','.join(imp.imported_names), imp.is_relative, imp.line_number)
            )

    def _get_analyzer(self, filepath: Path, content: str) -> BaseAnalyzer | None:
        """Get the appropriate analyzer for a file."""
        suffix = filepath.suffix
        if suffix in ('.py', '.pyw'):
            return PythonAnalyzer(str(filepath), content)
        elif suffix in ('.js', '.mjs', '.cjs', '.jsx'):
            return JavaScriptAnalyzer(str(filepath), content)
        elif suffix in ('.ts', '.tsx', '.mts', '.cts'):
            return TypeScriptAnalyzer(str(filepath), content)
        elif suffix == '.java':
            return JavaAnalyzer(str(filepath), content)
        return None


# =============================================================================
# Helper Functions for exec mode
# =============================================================================

def _make_helpers(db: Database, chunks_dir: Path) -> dict:
    """Create helper functions for exec mode."""
    chunks_dir.mkdir(parents=True, exist_ok=True)

    def find_symbol(name: str, symbol_type: str | None = None, repo: str | None = None) -> list[dict]:
        """Find symbols matching name (LIKE match), optionally filtered by type and repo."""
        sql = """
            SELECT s.*, f.filepath, f.language, r.repo_name
            FROM symbols s
            JOIN files f ON s.file_id = f.file_id
            LEFT JOIN repos r ON f.repo_id = r.repo_id
            WHERE s.symbol_name LIKE ?
        """
        params = [f'%{name}%']
        if symbol_type:
            sql += " AND s.symbol_type = ?"
            params.append(symbol_type)
        if repo:
            sql += " AND r.repo_name = ?"
            params.append(repo)
        sql += " ORDER BY f.filepath, s.definition_line"
        return db.query(sql, tuple(params))

    def find_symbol_exact(name: str, symbol_type: str | None = None, repo: str | None = None) -> list[dict]:
        """Find symbols with exact name match."""
        sql = """
            SELECT s.*, f.filepath, f.language, r.repo_name
            FROM symbols s
            JOIN files f ON s.file_id = f.file_id
            LEFT JOIN repos r ON f.repo_id = r.repo_id
            WHERE s.symbol_name = ?
        """
        params = [name]
        if symbol_type:
            sql += " AND s.symbol_type = ?"
            params.append(symbol_type)
        if repo:
            sql += " AND r.repo_name = ?"
            params.append(repo)
        sql += " ORDER BY f.filepath, s.definition_line"
        return db.query(sql, tuple(params))

    def get_class_methods(class_name: str) -> list[dict]:
        """Get all methods of a class."""
        return db.query("""
            SELECT s.*, f.filepath, r.repo_name
            FROM symbols s
            JOIN files f ON s.file_id = f.file_id
            LEFT JOIN repos r ON f.repo_id = r.repo_id
            WHERE s.parent_symbol = ? AND s.symbol_type = 'method'
            ORDER BY s.definition_line
        """, (class_name,))

    def list_repos() -> list[dict]:
        """List all indexed repositories."""
        return db.list_repos()

    def get_repo(repo_name: str) -> dict | None:
        """Get repository info."""
        repos = db.query("SELECT * FROM repos WHERE repo_name = ?", (repo_name,))
        return repos[0] if repos else None

    def get_files_in_repo(repo_name: str) -> list[dict]:
        """Get all files in a repository."""
        return db.query("""
            SELECT f.* FROM files f
            JOIN repos r ON f.repo_id = r.repo_id
            WHERE r.repo_name = ?
            ORDER BY f.filepath
        """, (repo_name,))

    def cross_repo_imports(from_repo: str, to_repo: str) -> list[dict]:
        """Find potential cross-repo dependencies."""
        return db.query("""
            SELECT f.filepath, i.module_name, i.line_number
            FROM imports i
            JOIN files f ON i.file_id = f.file_id
            JOIN repos r ON f.repo_id = r.repo_id
            WHERE r.repo_name = ?
            AND EXISTS (
                SELECT 1 FROM files f2
                JOIN repos r2 ON f2.repo_id = r2.repo_id
                WHERE r2.repo_name = ?
                AND i.module_name LIKE '%' || REPLACE(f2.filepath, '/', '.') || '%'
            )
        """, (from_repo, to_repo))

    def search_content(query: str, limit: int = 100) -> list[dict]:
        """Search content using FTS5."""
        return db.search_content_fts(query, limit)

    def search_chunks(pattern: str, chunk_type: str | None = None, limit: int = 100) -> list[dict]:
        """Search chunks using regex."""
        results = []
        sql = "SELECT c.*, f.filepath, r.repo_name FROM chunks c JOIN files f ON c.file_id = f.file_id LEFT JOIN repos r ON f.repo_id = r.repo_id"
        if chunk_type:
            sql += " WHERE c.chunk_type = ?"
            rows = db.query(sql, (chunk_type,))
        else:
            rows = db.query(sql)

        try:
            regex = re.compile(pattern)
            for row in rows:
                if regex.search(row['content']):
                    results.append(row)
                    if len(results) >= limit:
                        break
        except re.error:
            pass
        return results

    def get_chunk(chunk_id: int) -> dict | None:
        """Get chunk by ID."""
        results = db.query("""
            SELECT c.*, f.filepath, r.repo_name
            FROM chunks c
            JOIN files f ON c.file_id = f.file_id
            LEFT JOIN repos r ON f.repo_id = r.repo_id
            WHERE c.chunk_id = ?
        """, (chunk_id,))
        return results[0] if results else None

    def get_file_chunks(filepath: str) -> list[dict]:
        """Get all chunks in a file."""
        return db.query("""
            SELECT c.*, f.filepath, r.repo_name
            FROM chunks c
            JOIN files f ON c.file_id = f.file_id
            LEFT JOIN repos r ON f.repo_id = r.repo_id
            WHERE f.filepath = ?
            ORDER BY c.start_line
        """, (filepath,))

    def write_chunk_to_file(chunk_id: int) -> str | None:
        """Write a chunk to a file with metadata header."""
        chunk = get_chunk(chunk_id)
        if not chunk:
            return None

        filename = f"chunk_{chunk_id:06d}_{chunk['symbol_name']}.txt"
        filepath = chunks_dir / filename

        header = f"""# Chunk: {chunk['symbol_name']}
# Type: {chunk['chunk_type']}
# File: {chunk['filepath']}
# Lines: {chunk['start_line']}-{chunk['end_line']}
# Repo: {chunk.get('repo_name', 'unknown')}
# Parent: {chunk['parent_symbol'] or 'none'}
# ---

"""
        filepath.write_text(header + chunk['content'], encoding='utf-8')
        return str(filepath)

    def write_file_chunks(filepath: str) -> list[str]:
        """Write all chunks from a file."""
        chunks = get_file_chunks(filepath)
        return [write_chunk_to_file(c['chunk_id']) for c in chunks if c['chunk_id']]

    def write_chunks_combined(chunk_ids: list[int], output_path: str) -> str:
        """Combine multiple chunks into a single file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        content_parts = []
        for chunk_id in chunk_ids:
            chunk = get_chunk(chunk_id)
            if chunk:
                header = f"# === Chunk {chunk_id}: {chunk['symbol_name']} ({chunk['chunk_type']}) ===\n"
                header += f"# File: {chunk['filepath']} lines {chunk['start_line']}-{chunk['end_line']}\n"
                header += f"# Repo: {chunk.get('repo_name', 'unknown')}\n\n"
                content_parts.append(header + chunk['content'] + "\n")

        path.write_text('\n'.join(content_parts), encoding='utf-8')
        return str(path)

    def get_files_by_language(language: str) -> list[dict]:
        """Get all files of a specific language."""
        return db.query("""
            SELECT f.*, r.repo_name
            FROM files f
            LEFT JOIN repos r ON f.repo_id = r.repo_id
            WHERE f.language = ?
            ORDER BY f.filepath
        """, (language,))

    def get_imports_for_file(filepath: str) -> list[dict]:
        """Get all imports for a file."""
        return db.query("""
            SELECT i.* FROM imports i
            JOIN files f ON i.file_id = f.file_id
            WHERE f.filepath = ?
            ORDER BY i.line_number
        """, (filepath,))

    def get_files_importing(module_name: str) -> list[dict]:
        """Find files that import a module."""
        return db.query("""
            SELECT DISTINCT f.*, r.repo_name
            FROM files f
            JOIN imports i ON f.file_id = i.file_id
            LEFT JOIN repos r ON f.repo_id = r.repo_id
            WHERE i.module_name LIKE ?
            ORDER BY f.filepath
        """, (f'%{module_name}%',))

    def analyze_dependencies(filepath: str) -> dict:
        """Analyze dependencies for a file."""
        imports = get_imports_for_file(filepath)
        imported_by = db.query("""
            SELECT DISTINCT f.filepath, r.repo_name
            FROM files f
            JOIN imports i ON f.file_id = i.file_id
            LEFT JOIN repos r ON f.repo_id = r.repo_id
            WHERE i.module_name LIKE ?
            ORDER BY f.filepath
        """, (f'%{Path(filepath).stem}%',))

        return {
            'filepath': filepath,
            'imports': imports,
            'imported_by': imported_by
        }

    def stats() -> dict:
        """Get full statistics."""
        return db.get_stats()

    # =========================================================================
    # Memory helpers
    # =========================================================================

    def memory_add(content: str, memory_type: str = 'fact', importance: float = 0.5, tags: list[str] | None = None) -> int:
        """Add a memory. Returns memory_id."""
        try:
            from memory_db import MemoryDatabase, MemoryType, MemorySource
            mdb = MemoryDatabase()
            memory_id = mdb.add_memory(
                content,
                memory_type=MemoryType(memory_type),
                source=MemorySource.EXPLICIT,
                importance=importance,
                tags=tags or [],
            )
            mdb.close()
            return memory_id
        except ImportError:
            print("Memory module not available")
            return -1

    def memory_search(query: str, limit: int = 10) -> list[dict]:
        """Search memories using FTS5."""
        try:
            from memory_db import MemoryDatabase
            mdb = MemoryDatabase()
            results = mdb.search_memories_fts(query, limit=limit)
            mdb.close()
            return [m.to_dict() for m in results]
        except ImportError:
            return []

    def memory_list(memory_type: str | None = None, limit: int = 20) -> list[dict]:
        """List memories, optionally filtered by type."""
        try:
            from memory_db import MemoryDatabase, MemoryType
            mdb = MemoryDatabase()
            if memory_type:
                results = mdb.get_memories_by_type(MemoryType(memory_type), limit=limit)
            else:
                results = mdb.get_all_memories(limit=limit)
            mdb.close()
            return [m.to_dict() for m in results]
        except ImportError:
            return []

    def memory_context(max_chars: int = 2000) -> str:
        """Get session context from memories."""
        try:
            from memory_db import MemoryDatabase
            mdb = MemoryDatabase()
            context = mdb.get_context_for_session(max_chars=max_chars)
            mdb.close()
            return context
        except ImportError:
            return ""

    def memory_stats() -> dict:
        """Get memory database statistics."""
        try:
            from memory_db import MemoryDatabase
            mdb = MemoryDatabase()
            stats = mdb.get_stats()
            mdb.close()
            return stats
        except ImportError:
            return {}

    def set_preference(key: str, value) -> bool:
        """Set a user preference."""
        try:
            from memory_db import MemoryDatabase
            mdb = MemoryDatabase()
            result = mdb.set_preference(key, value)
            mdb.close()
            return result
        except ImportError:
            return False

    def get_preference(key: str, default=None):
        """Get a user preference."""
        try:
            from memory_db import MemoryDatabase
            mdb = MemoryDatabase()
            result = mdb.get_preference(key, default)
            mdb.close()
            return result
        except ImportError:
            return default

    return {
        'find_symbol': find_symbol,
        'find_symbol_exact': find_symbol_exact,
        'get_class_methods': get_class_methods,
        'list_repos': list_repos,
        'get_repo': get_repo,
        'get_files_in_repo': get_files_in_repo,
        'cross_repo_imports': cross_repo_imports,
        'search_content': search_content,
        'search_chunks': search_chunks,
        'get_chunk': get_chunk,
        'get_file_chunks': get_file_chunks,
        'write_chunk_to_file': write_chunk_to_file,
        'write_file_chunks': write_file_chunks,
        'write_chunks_combined': write_chunks_combined,
        'get_files_by_language': get_files_by_language,
        'get_imports_for_file': get_imports_for_file,
        'get_files_importing': get_files_importing,
        'analyze_dependencies': analyze_dependencies,
        'stats': stats,
        # Memory helpers
        'memory_add': memory_add,
        'memory_search': memory_search,
        'memory_list': memory_list,
        'memory_context': memory_context,
        'memory_stats': memory_stats,
        'set_preference': set_preference,
        'get_preference': get_preference,
    }


# =============================================================================
# CLI Commands
# =============================================================================

def cmd_init(args: argparse.Namespace) -> int:
    """Initialize and index a codebase."""
    db_path = Path(args.db) if args.db else DEFAULT_DB_PATH
    codebase_path = Path(args.path).resolve()

    if not codebase_path.exists():
        print(f"Error: Path does not exist: {codebase_path}", file=sys.stderr)
        return 1

    extensions = None
    if args.extensions:
        extensions = [ext.strip() if ext.startswith('.') else f'.{ext.strip()}'
                      for ext in args.extensions.split(',')]

    print(f"Indexing: {codebase_path}")
    print(f"Repo name: {args.name or codebase_path.name}")
    if extensions:
        print(f"Extensions: {', '.join(extensions)}")

    db = Database(db_path)
    try:
        indexer = CodebaseIndexer(
            db, codebase_path,
            repo_name=args.name,
            extensions=extensions
        )
        stats = indexer.index_codebase(full=args.full)

        print(f"\nIndexing complete:")
        print(f"  Total files scanned: {stats['total']}")
        print(f"  Files indexed: {stats['indexed']}")
        print(f"  Files skipped (unchanged): {stats['skipped']}")
        print(f"  Errors: {stats['errors']}")

        return 0
    finally:
        db.close()


def cmd_repos(args: argparse.Namespace) -> int:
    """List or manage repositories."""
    db_path = Path(args.db) if args.db else DEFAULT_DB_PATH

    if not db_path.exists():
        print("No database found. Run 'init' first.")
        return 1

    db = Database(db_path)
    try:
        if args.remove:
            if not args.yes:
                confirm = input(f"Delete repo '{args.remove}' and all its data? [y/N] ")
                if confirm.lower() != 'y':
                    print("Cancelled.")
                    return 0

            if db.delete_repo(args.remove):
                print(f"Deleted repo: {args.remove}")
            else:
                print(f"Repo not found: {args.remove}")
                return 1
        else:
            repos = db.list_repos()
            if not repos:
                print("No repositories indexed.")
            else:
                print(f"Repositories ({len(repos)}):")
                for r in repos:
                    print(f"  {r['repo_name']}: {r['actual_file_count']} files")
                    print(f"    Path: {r['root_path']}")
                    print(f"    Indexed: {r['indexed_at']}")

        return 0
    finally:
        db.close()


def cmd_status(args: argparse.Namespace) -> int:
    """Show database status."""
    db_path = Path(args.db) if args.db else DEFAULT_DB_PATH

    if not db_path.exists():
        print("No database found. Run 'init' first.")
        return 1

    db = Database(db_path)
    try:
        stats = db.get_stats()
        size_mb = db_path.stat().st_size / (1024 * 1024)

        print("RLM REPL Status")
        print(f"  Database: {db_path}")
        print(f"  Size: {size_mb:.2f} MB")
        print(f"  Repos: {stats['repos']}")
        print(f"  Files: {stats['files']}")
        print(f"  Chunks: {stats['chunks']}")
        print(f"  Symbols: {stats['symbols']}")
        print(f"  Imports: {stats['imports']}")

        if args.languages and stats['languages']:
            print("\nLanguages:")
            for lang, count in stats['languages'].items():
                print(f"  {lang}: {count} files")

        if args.chunks and stats['chunk_types']:
            print("\nChunk types:")
            for ctype, count in stats['chunk_types'].items():
                print(f"  {ctype}: {count}")

        return 0
    finally:
        db.close()


def cmd_search(args: argparse.Namespace) -> int:
    """Search the index."""
    db_path = Path(args.db) if args.db else DEFAULT_DB_PATH

    if not db_path.exists():
        print("No database found. Run 'init' first.")
        return 1

    db = Database(db_path)
    try:
        if args.symbol:
            results = db.query("""
                SELECT s.*, f.filepath, r.repo_name
                FROM symbols s
                JOIN files f ON s.file_id = f.file_id
                LEFT JOIN repos r ON f.repo_id = r.repo_id
                WHERE s.symbol_name LIKE ?
                ORDER BY f.filepath, s.definition_line
                LIMIT 50
            """, (f'%{args.symbol}%',))

            if results:
                print(f"Found {len(results)} symbols matching '{args.symbol}':")
                for r in results:
                    parent = f" in {r['parent_symbol']}" if r['parent_symbol'] else ""
                    repo = f"[{r['repo_name']}] " if r['repo_name'] else ""
                    print(f"  {repo}{r['filepath']}:{r['definition_line']} - {r['symbol_type']} {r['symbol_name']}{parent}")
            else:
                print(f"No symbols found matching '{args.symbol}'")

        elif args.pattern:
            if args.fts:
                results = db.search_content_fts(args.pattern, limit=50)
                if results:
                    print(f"Found {len(results)} chunks matching '{args.pattern}' (FTS5):")
                    for r in results:
                        repo = f"[{r['repo_name']}] " if r['repo_name'] else ""
                        print(f"  {repo}{r['filepath']}:{r['start_line']}-{r['end_line']} - {r['chunk_type']} {r['symbol_name']}")
                else:
                    print(f"No content found matching '{args.pattern}'")
            else:
                # Regex search
                rows = db.query("""
                    SELECT c.*, f.filepath, r.repo_name
                    FROM chunks c
                    JOIN files f ON c.file_id = f.file_id
                    LEFT JOIN repos r ON f.repo_id = r.repo_id
                """)
                results = []
                try:
                    regex = re.compile(args.pattern)
                    for row in rows:
                        if regex.search(row['content']):
                            results.append(row)
                            if len(results) >= 50:
                                break
                except re.error as e:
                    print(f"Invalid regex: {e}", file=sys.stderr)
                    return 1

                if results:
                    print(f"Found {len(results)} chunks matching pattern (regex):")
                    for r in results:
                        repo = f"[{r['repo_name']}] " if r['repo_name'] else ""
                        print(f"  {repo}{r['filepath']}:{r['start_line']}-{r['end_line']} - {r['chunk_type']} {r['symbol_name']}")
                else:
                    print(f"No content found matching pattern")

        elif args.imports:
            results = db.query("""
                SELECT DISTINCT f.filepath, i.module_name, i.line_number, r.repo_name
                FROM imports i
                JOIN files f ON i.file_id = f.file_id
                LEFT JOIN repos r ON f.repo_id = r.repo_id
                WHERE i.module_name LIKE ?
                ORDER BY f.filepath
                LIMIT 50
            """, (f'%{args.imports}%',))

            if results:
                print(f"Found {len(results)} imports matching '{args.imports}':")
                for r in results:
                    repo = f"[{r['repo_name']}] " if r['repo_name'] else ""
                    print(f"  {repo}{r['filepath']}:{r['line_number']} - {r['module_name']}")
            else:
                print(f"No imports found matching '{args.imports}'")
        else:
            print("Specify --symbol, --pattern, or --imports")
            return 1

        return 0
    finally:
        db.close()


def cmd_exec(args: argparse.Namespace) -> int:
    """Execute Python code with helpers."""
    db_path = Path(args.db) if args.db else DEFAULT_DB_PATH
    chunks_dir = Path(args.chunks_dir) if args.chunks_dir else DEFAULT_CHUNKS_DIR

    if not db_path.exists():
        print("No database found. Run 'init' first.", file=sys.stderr)
        return 1

    code = args.code
    if code is None:
        code = sys.stdin.read()

    db = Database(db_path)
    try:
        helpers = _make_helpers(db, chunks_dir)

        env: dict[str, Any] = {}
        env['db'] = db
        env.update(helpers)

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        try:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                exec(code, env, env)
        except Exception:
            traceback.print_exc(file=stderr_buf)

        out = stdout_buf.getvalue()
        err = stderr_buf.getvalue()

        max_chars = args.max_output_chars

        if out:
            if len(out) > max_chars:
                out = out[:max_chars] + f"\n... [truncated to {max_chars} chars] ...\n"
            sys.stdout.write(out)

        if err:
            if len(err) > max_chars:
                err = err[:max_chars] + f"\n... [truncated to {max_chars} chars] ...\n"
            sys.stderr.write(err)

        return 0
    finally:
        db.close()


def cmd_vacuum(args: argparse.Namespace) -> int:
    """Vacuum and optimize the database."""
    db_path = Path(args.db) if args.db else DEFAULT_DB_PATH

    if not db_path.exists():
        print("No database found.")
        return 1

    print(f"Optimizing database: {db_path}")
    db = Database(db_path)
    try:
        size_before = db_path.stat().st_size
        db.vacuum()
        size_after = db_path.stat().st_size

        saved = size_before - size_after
        print(f"Done. Saved {saved / 1024:.1f} KB")
        return 0
    finally:
        db.close()


def cmd_reset(args: argparse.Namespace) -> int:
    """Reset the database."""
    db_path = Path(args.db) if args.db else DEFAULT_DB_PATH

    if not db_path.exists():
        print("No database to reset.")
        return 0

    if not args.yes:
        confirm = input("Delete all indexed data? [y/N] ")
        if confirm.lower() != 'y':
            print("Cancelled.")
            return 0

    # Remove database and WAL files
    for suffix in ['', '-wal', '-shm']:
        p = Path(str(db_path) + suffix)
        if p.exists():
            p.unlink()

    print(f"Reset complete. Database deleted: {db_path}")
    return 0


# =============================================================================
# Memory Commands (v2)
# =============================================================================

def cmd_memory(args: argparse.Namespace) -> int:
    """Memory management command."""
    try:
        from memory_db import MemoryDatabase, MemoryType, MemorySource
    except ImportError:
        print("Memory module not available. Check memory_db.py exists.", file=sys.stderr)
        return 1

    db = MemoryDatabase()

    try:
        if args.action == 'add':
            memory_type = MemoryType(args.type) if args.type else MemoryType.FACT
            tags = args.tags.split(',') if args.tags else []
            memory_id = db.add_memory(
                args.content,
                memory_type=memory_type,
                source=MemorySource.EXPLICIT,
                importance=args.importance,
                tags=tags,
            )
            print(f"Added memory {memory_id}")

        elif args.action == 'search':
            results = db.search_memories_fts(args.query, limit=args.limit)
            if results:
                print(f"Found {len(results)} memories:")
                for m in results:
                    print(f"  [{m.memory_id}] ({m.memory_type.value}, {m.importance:.1f}) {m.content[:60]}")
            else:
                print("No memories found")

        elif args.action == 'list':
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

        elif args.action == 'delete':
            if db.delete_memory(args.id):
                print(f"Deleted memory {args.id}")
            else:
                print(f"Memory {args.id} not found")

        elif args.action == 'stats':
            stats = db.get_stats()
            print("Memory Database Statistics:")
            print(f"  Total memories: {stats['total_memories']}")
            print(f"  By type: {stats['by_type']}")
            print(f"  By source: {stats['by_source']}")
            print(f"  Total sessions: {stats['total_sessions']}")
            print(f"  Avg importance: {stats['avg_importance']}")

        elif args.action == 'context':
            context = db.get_context_for_session(max_chars=args.max_chars)
            if context:
                print(context)
            else:
                print("No context available")

        elif args.action == 'profile':
            profile = db.get_profile()
            if profile:
                print(f"Profile: {profile.name or '(unnamed)'}")
                print(f"  Created: {profile.created_at}")
                print(f"  Updated: {profile.updated_at}")
                if profile.preferences:
                    print(f"  Preferences:")
                    for k, v in profile.preferences.items():
                        print(f"    {k}: {v}")
            else:
                print("No profile found")

        elif args.action == 'set-pref':
            if db.set_preference(args.key, args.value):
                print(f"Set preference: {args.key} = {args.value}")
            else:
                print("Failed to set preference")

        return 0
    finally:
        db.close()


def cmd_remember(args: argparse.Namespace) -> int:
    """Quick command to add a memory."""
    try:
        from memory_db import MemoryDatabase, MemoryType, MemorySource
    except ImportError:
        print("Memory module not available.", file=sys.stderr)
        return 1

    db = MemoryDatabase()
    try:
        memory_id = db.add_memory(
            args.content,
            memory_type=MemoryType.FACT,
            source=MemorySource.EXPLICIT,
            importance=0.8,  # Explicit memories are important
        )
        print(f"Remembered: {args.content[:50]}... (id: {memory_id})")
        return 0
    finally:
        db.close()


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    p = argparse.ArgumentParser(
        prog='rlm_repl',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            RLM: SQLite-backed codebase indexer for Recursive Language Model workflows.

            Examples:
              python rlm_repl.py init /path/to/repo --name my-repo
              python rlm_repl.py status --languages
              python rlm_repl.py search --symbol UserAuth
              python rlm_repl.py search --pattern "authenticate" --fts
              python rlm_repl.py exec -c "print(stats())"
        """)
    )
    p.add_argument('--db', help=f'Database path (default: {DEFAULT_DB_PATH})')
    p.add_argument('-v', '--verbose', action='store_true', help='Verbose output')

    sub = p.add_subparsers(dest='cmd', required=True)

    # init
    p_init = sub.add_parser('init', help='Index a codebase')
    p_init.add_argument('path', help='Path to codebase')
    p_init.add_argument('--name', help='Repository name (default: directory name)')
    p_init.add_argument('--extensions', help='Comma-separated extensions (e.g., .py,.java)')
    p_init.add_argument('--full', action='store_true', help='Force full re-index')
    p_init.set_defaults(func=cmd_init)

    # repos
    p_repos = sub.add_parser('repos', help='List/manage repositories')
    p_repos.add_argument('--remove', help='Remove a repository')
    p_repos.add_argument('-y', '--yes', action='store_true', help='Skip confirmation')
    p_repos.set_defaults(func=cmd_repos)

    # status
    p_status = sub.add_parser('status', help='Show database status')
    p_status.add_argument('--languages', action='store_true', help='Show language breakdown')
    p_status.add_argument('--chunks', action='store_true', help='Show chunk type breakdown')
    p_status.set_defaults(func=cmd_status)

    # search
    p_search = sub.add_parser('search', help='Search the index')
    p_search.add_argument('--symbol', help='Search for symbol names')
    p_search.add_argument('--pattern', help='Search content (regex or FTS5)')
    p_search.add_argument('--fts', action='store_true', help='Use FTS5 for pattern search')
    p_search.add_argument('--imports', help='Search for imports')
    p_search.set_defaults(func=cmd_search)

    # exec
    p_exec = sub.add_parser('exec', help='Execute Python code with helpers')
    p_exec.add_argument('-c', '--code', help='Code to execute (reads stdin if omitted)')
    p_exec.add_argument('--chunks-dir', help=f'Chunks output directory (default: {DEFAULT_CHUNKS_DIR})')
    p_exec.add_argument('--max-output-chars', type=int, default=DEFAULT_MAX_OUTPUT_CHARS,
                        help=f'Max output chars (default: {DEFAULT_MAX_OUTPUT_CHARS})')
    p_exec.set_defaults(func=cmd_exec)

    # vacuum
    p_vacuum = sub.add_parser('vacuum', help='Optimize the database')
    p_vacuum.set_defaults(func=cmd_vacuum)

    # reset
    p_reset = sub.add_parser('reset', help='Delete all indexed data')
    p_reset.add_argument('-y', '--yes', action='store_true', help='Skip confirmation')
    p_reset.set_defaults(func=cmd_reset)

    # ==========================================================================
    # Memory commands (v2)
    # ==========================================================================

    # memory (main command with subcommands)
    p_memory = sub.add_parser('memory', help='Memory management')
    memory_sub = p_memory.add_subparsers(dest='action', required=True)

    # memory add
    p_mem_add = memory_sub.add_parser('add', help='Add a memory')
    p_mem_add.add_argument('content', help='Memory content')
    p_mem_add.add_argument('--type', default='fact',
                           choices=['fact', 'preference', 'pattern', 'decision', 'context', 'instruction'],
                           help='Memory type')
    p_mem_add.add_argument('--importance', type=float, default=0.5, help='Importance (0-1)')
    p_mem_add.add_argument('--tags', help='Comma-separated tags')

    # memory search
    p_mem_search = memory_sub.add_parser('search', help='Search memories')
    p_mem_search.add_argument('query', help='Search query')
    p_mem_search.add_argument('--limit', type=int, default=10)

    # memory list
    p_mem_list = memory_sub.add_parser('list', help='List memories')
    p_mem_list.add_argument('--type', choices=['fact', 'preference', 'pattern', 'decision', 'context', 'instruction'])
    p_mem_list.add_argument('--limit', type=int, default=20)

    # memory delete
    p_mem_del = memory_sub.add_parser('delete', help='Delete a memory')
    p_mem_del.add_argument('id', type=int, help='Memory ID to delete')

    # memory stats
    memory_sub.add_parser('stats', help='Show memory statistics')

    # memory context
    p_mem_ctx = memory_sub.add_parser('context', help='Get session context')
    p_mem_ctx.add_argument('--max-chars', type=int, default=2000)

    # memory profile
    memory_sub.add_parser('profile', help='Show user profile')

    # memory set-pref
    p_mem_pref = memory_sub.add_parser('set-pref', help='Set a preference')
    p_mem_pref.add_argument('key', help='Preference key')
    p_mem_pref.add_argument('value', help='Preference value')

    p_memory.set_defaults(func=cmd_memory)

    # remember (quick add)
    p_remember = sub.add_parser('remember', help='Quick add a memory')
    p_remember.add_argument('content', help='What to remember')
    p_remember.set_defaults(func=cmd_remember)

    return p


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
