#!/usr/bin/env python3
"""
Test script for RLM v2 improvements.

Creates sample files in different languages, indexes them, and verifies:
1. Persistent connection / WAL mode
2. FTS5 search
3. Incremental indexing
4. JavaScript/TypeScript support
5. Java support
6. Parent symbol tracking
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from rlm_repl_v2 import (
    Database, CodebaseIndexer, PythonAnalyzer, JavaScriptAnalyzer, JavaAnalyzer,
    _make_helpers, DEFAULT_STATE_DIR
)


def create_test_files(test_dir: Path) -> None:
    """Create sample source files for testing."""
    
    # Python file
    (test_dir / "python").mkdir(parents=True, exist_ok=True)
    (test_dir / "python" / "auth.py").write_text('''
"""Authentication module."""
from typing import Optional
from dataclasses import dataclass

@dataclass
class User:
    """User model."""
    id: int
    email: str
    name: str

class AuthService:
    """Handles authentication logic."""
    
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
    
    async def authenticate(self, email: str, password: str) -> Optional[User]:
        """Authenticate a user."""
        # Check credentials
        if self._verify_password(password):
            return User(id=1, email=email, name="Test")
        return None
    
    def _verify_password(self, password: str) -> bool:
        """Verify password hash."""
        return len(password) > 8

def get_current_user() -> Optional[User]:
    """Get the current authenticated user."""
    return None
''')

    # JavaScript file
    (test_dir / "javascript").mkdir(parents=True, exist_ok=True)
    (test_dir / "javascript" / "api.js").write_text('''
import { Router } from 'express';
import { authenticate } from './auth';

const router = Router();

export class ApiController {
    constructor(authService) {
        this.authService = authService;
    }
    
    async handleLogin(req, res) {
        const { email, password } = req.body;
        const user = await this.authService.authenticate(email, password);
        if (user) {
            res.json({ success: true, user });
        } else {
            res.status(401).json({ error: 'Invalid credentials' });
        }
    }
}

export async function setupRoutes(app) {
    const controller = new ApiController();
    router.post('/login', (req, res) => controller.handleLogin(req, res));
    app.use('/api', router);
}

const processRequest = async (data) => {
    return data;
};
''')

    # TypeScript file
    (test_dir / "typescript").mkdir(parents=True, exist_ok=True)
    (test_dir / "typescript" / "types.ts").write_text('''
import { User } from './models';

export interface AuthConfig {
    secretKey: string;
    tokenExpiry: number;
}

export interface AuthResult {
    success: boolean;
    token?: string;
    user?: User;
}

export type AuthCallback = (result: AuthResult) => void;

export class TokenService {
    private config: AuthConfig;
    
    constructor(config: AuthConfig) {
        this.config = config;
    }
    
    generateToken(user: User): string {
        return `token_${user.id}_${Date.now()}`;
    }
    
    validateToken(token: string): boolean {
        return token.startsWith('token_');
    }
}
''')

    # Java file
    (test_dir / "java").mkdir(parents=True, exist_ok=True)
    (test_dir / "java" / "UserController.java").write_text('''
package com.example.auth;

import org.springframework.web.bind.annotation.*;
import org.springframework.beans.factory.annotation.Autowired;
import java.util.Optional;

@RestController
@RequestMapping("/api/users")
public class UserController {
    
    @Autowired
    private UserService userService;
    
    @Autowired
    private AuthService authService;
    
    @GetMapping("/{id}")
    public ResponseEntity<User> getUser(@PathVariable Long id) {
        Optional<User> user = userService.findById(id);
        return user.map(ResponseEntity::ok)
                   .orElse(ResponseEntity.notFound().build());
    }
    
    @PostMapping("/login")
    public ResponseEntity<AuthResult> login(@RequestBody LoginRequest request) {
        AuthResult result = authService.authenticate(
            request.getEmail(), 
            request.getPassword()
        );
        if (result.isSuccess()) {
            return ResponseEntity.ok(result);
        }
        return ResponseEntity.status(401).body(result);
    }
    
    @PostMapping("/register")
    public ResponseEntity<User> register(@RequestBody RegisterRequest request) {
        User user = userService.createUser(request);
        return ResponseEntity.ok(user);
    }
}

interface UserRepository {
    Optional<User> findById(Long id);
    User save(User user);
}
''')

    print(f"Created test files in {test_dir}")


def test_analyzers():
    """Test individual language analyzers."""
    print("\n" + "="*60)
    print("Testing Language Analyzers")
    print("="*60)
    
    # Test Python analyzer
    python_code = '''
class MyClass:
    def method_one(self):
        pass
    
    async def method_two(self):
        pass

def standalone_function():
    pass
'''
    analyzer = PythonAnalyzer("test.py", python_code)
    chunks = analyzer.extract_chunks()
    symbols = analyzer.extract_symbols()
    
    print(f"\nPython Analyzer:")
    print(f"  Chunks: {len(chunks)}")
    for c in chunks:
        print(f"    - {c.chunk_type}: {c.symbol_name} (parent: {c.parent_symbol})")
    print(f"  Symbols: {len(symbols)}")
    for s in symbols:
        print(f"    - {s[1]}: {s[0]} (parent: {s[3]})")
    
    # Test JavaScript analyzer
    js_code = '''
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
    analyzer = JavaScriptAnalyzer("test.js", js_code)
    chunks = analyzer.extract_chunks()
    symbols = analyzer.extract_symbols()
    imports = analyzer.extract_imports()
    
    print(f"\nJavaScript Analyzer:")
    print(f"  Chunks: {len(chunks)}")
    for c in chunks:
        print(f"    - {c.chunk_type}: {c.symbol_name}")
    print(f"  Symbols: {len(symbols)}")
    print(f"  Imports: {len(imports)}")
    
    # Test Java analyzer
    java_code = '''
package com.example;

import java.util.List;
import org.springframework.stereotype.Service;

@Service
public class UserService {
    
    private final UserRepository repository;
    
    public UserService(UserRepository repository) {
        this.repository = repository;
    }
    
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
    analyzer = JavaAnalyzer("UserService.java", java_code)
    chunks = analyzer.extract_chunks()
    symbols = analyzer.extract_symbols()
    imports = analyzer.extract_imports()
    
    print(f"\nJava Analyzer:")
    print(f"  Chunks: {len(chunks)}")
    for c in chunks:
        print(f"    - {c.chunk_type}: {c.symbol_name} (parent: {c.parent_symbol})")
    print(f"  Symbols: {len(symbols)}")
    for s in symbols:
        print(f"    - {s[1]}: {s[0]} (parent: {s[3]})")
    print(f"  Imports: {len(imports)}")
    for i in imports:
        print(f"    - {i.module_name}")


def test_database_features(db_path: Path):
    """Test database features."""
    print("\n" + "="*60)
    print("Testing Database Features")
    print("="*60)
    
    # Clean up existing
    if db_path.exists():
        db_path.unlink()
    for suffix in ['-wal', '-shm']:
        p = Path(str(db_path) + suffix)
        if p.exists():
            p.unlink()
    
    db = Database(db_path)
    
    # Check WAL mode
    result = db.query("PRAGMA journal_mode")
    print(f"\nJournal mode: {result[0]['journal_mode']}")
    assert result[0]['journal_mode'] == 'wal', "WAL mode not enabled"
    
    # Check foreign keys
    result = db.query("PRAGMA foreign_keys")
    print(f"Foreign keys: {result[0]['foreign_keys']}")
    
    # Check FTS5 table exists
    result = db.query("SELECT name FROM sqlite_master WHERE type='table' AND name='chunks_fts'")
    print(f"FTS5 table exists: {len(result) > 0}")
    assert len(result) > 0, "FTS5 table not created"
    
    # Test transaction batching
    print("\nTesting transaction batching...")
    db.begin_transaction()
    for i in range(100):
        db._conn.execute(
            "INSERT INTO files (filepath, language) VALUES (?, ?)",
            (f"test_{i}.py", "python")
        )
    db.commit()
    
    count = db.query("SELECT COUNT(*) as c FROM files")[0]['c']
    print(f"Inserted {count} files in single transaction")
    
    # Clean up test data
    db.execute("DELETE FROM files")
    
    db.close()
    print("\n✓ Database features working correctly")


def test_full_indexing(test_dir: Path, db_path: Path):
    """Test full indexing workflow."""
    print("\n" + "="*60)
    print("Testing Full Indexing Workflow")
    print("="*60)
    
    # Create test files
    create_test_files(test_dir)
    
    # Initialize database
    db = Database(db_path)
    
    # Index codebase with explicit repo name
    indexer = CodebaseIndexer(db, test_dir, repo_name="test-project")
    stats = indexer.index_codebase()
    
    print(f"\nIndexing stats: {stats}")
    
    # Get database stats
    db_stats = db.get_stats()
    print(f"Database stats: {db_stats}")
    
    # Verify repos were created
    repos = db.list_repos()
    print(f"\nIndexed repos:")
    for r in repos:
        print(f"  {r['repo_name']}: {r['file_count']} files")
    
    assert len(repos) == 1, f"Expected 1 repo, got {len(repos)}"
    assert repos[0]['repo_name'] == 'test-project', f"Wrong repo name: {repos[0]['repo_name']}"
    
    # Verify files indexed
    files = db.query("SELECT * FROM files ORDER BY filepath")
    print(f"\nIndexed files:")
    for f in files:
        print(f"  {f['language']:12} {f['filepath']}")
    
    assert db_stats['files'] >= 4, f"Expected at least 4 files, got {db_stats['files']}"
    
    # Verify chunks extracted
    chunks = db.query("""
        SELECT c.chunk_type, c.symbol_name, c.parent_symbol, f.language
        FROM chunks c
        JOIN files f ON c.file_id = f.file_id
        ORDER BY f.language, c.start_line
    """)
    print(f"\nExtracted chunks ({len(chunks)}):")
    for c in chunks:
        parent = f" in {c['parent_symbol']}" if c['parent_symbol'] else ""
        print(f"  {c['language']:12} {c['chunk_type']:10} {c['symbol_name']}{parent}")
    
    # Verify symbols
    symbols = db.query("""
        SELECT s.symbol_type, s.symbol_name, s.parent_symbol, f.language
        FROM symbols s
        JOIN files f ON s.file_id = f.file_id
        ORDER BY f.language, s.definition_line
    """)
    print(f"\nExtracted symbols ({len(symbols)}):")
    
    # Test FTS5 search
    print("\nTesting FTS5 search...")
    fts_results = db.search_content_fts("authenticate", limit=10)
    print(f"FTS5 search for 'authenticate': {len(fts_results)} results")
    for r in fts_results[:3]:
        print(f"  {r['filepath']} - {r['symbol_name']}")
    
    # Test helpers
    print("\nTesting helper functions...")
    chunks_dir = test_dir / "chunks"
    helpers = _make_helpers(db, chunks_dir)
    
    # find_symbol
    auth_symbols = helpers['find_symbol']('auth')
    print(f"find_symbol('auth'): {len(auth_symbols)} results")
    
    # Verify repo_name is in results
    if auth_symbols:
        assert 'repo_name' in auth_symbols[0], "repo_name not in find_symbol results"
        print(f"  First result repo: {auth_symbols[0]['repo_name']}")
    
    # get_class_methods
    methods = helpers['get_class_methods']('AuthService')
    print(f"get_class_methods('AuthService'): {len(methods)} methods")
    
    # search_content (FTS5)
    content_results = helpers['search_content']('password')
    print(f"search_content('password'): {len(content_results)} results")
    
    # list_repos
    repos = helpers['list_repos']()
    print(f"list_repos(): {len(repos)} repos")
    
    # stats
    full_stats = helpers['stats']()
    print(f"stats(): {full_stats['languages']}")
    
    db.close()
    print("\n✓ Full indexing workflow working correctly")


def test_incremental_indexing(test_dir: Path, db_path: Path):
    """Test incremental indexing."""
    print("\n" + "="*60)
    print("Testing Incremental Indexing")
    print("="*60)
    
    db = Database(db_path)
    indexer = CodebaseIndexer(db, test_dir, repo_name="test-project")
    
    # First pass should skip all (already indexed)
    print("\nRe-running indexer (should skip unchanged files)...")
    stats = indexer.index_codebase()
    print(f"Stats: {stats}")
    assert stats['skipped'] > 0, "Expected some files to be skipped"
    
    # Modify a file
    test_file = test_dir / "python" / "auth.py"
    original = test_file.read_text()
    test_file.write_text(original + "\n\ndef new_function():\n    pass\n")
    
    # Re-index - should only update modified file
    print("\nAfter modifying auth.py...")
    stats = indexer.index_codebase()
    print(f"Stats: {stats}")
    assert stats['indexed'] == 1, f"Expected 1 file indexed, got {stats['indexed']}"
    
    # Restore original
    test_file.write_text(original)
    
    db.close()
    print("\n✓ Incremental indexing working correctly")


def test_multi_repo(test_dir: Path, db_path: Path):
    """Test multi-repo functionality."""
    print("\n" + "="*60)
    print("Testing Multi-Repo Support")
    print("="*60)
    
    # Create a second repo directory
    repo2_dir = test_dir / "second-repo"
    repo2_dir.mkdir(parents=True, exist_ok=True)
    
    (repo2_dir / "shared.py").write_text('''
"""Shared utilities used by other repos."""

class SharedService:
    """A service that other repos depend on."""
    
    def process(self, data):
        return data
    
    def validate(self, input):
        return bool(input)

def helper_function():
    """A helper that frontend and backend both use."""
    return "shared"
''')
    
    db = Database(db_path)
    
    try:
        # Index second repo
        print("\nIndexing second repo...")
        indexer2 = CodebaseIndexer(db, repo2_dir, repo_name="shared-lib")
        stats2 = indexer2.index_codebase()
        print(f"Second repo stats: {stats2}")
        
        # Verify both repos exist
        repos = db.list_repos()
        print(f"\nAll repos ({len(repos)}):")
        for r in repos:
            print(f"  {r['repo_name']}: {r['file_count']} files")
        
        assert len(repos) == 2, f"Expected 2 repos, got {len(repos)}"
        
        # Test helpers with repo filtering
        chunks_dir = test_dir / "chunks"
        helpers = _make_helpers(db, chunks_dir)
        
        # find_symbol across all repos
        all_services = helpers['find_symbol']('Service', 'class')
        print(f"\nfind_symbol('Service', 'class') across all repos: {len(all_services)} results")
        for s in all_services:
            print(f"  {s['repo_name']}/{s['filepath']}: {s['symbol_name']}")
        
        # find_symbol in specific repo
        shared_only = helpers['find_symbol']('Service', 'class', repo='shared-lib')
        print(f"\nfind_symbol in 'shared-lib' only: {len(shared_only)} results")
        for s in shared_only:
            print(f"  {s['repo_name']}/{s['filepath']}: {s['symbol_name']}")
        
        assert len(shared_only) < len(all_services), "Repo filter should reduce results"
        
        # Test list_repos helper
        repos_via_helper = helpers['list_repos']()
        print(f"\nlist_repos(): {len(repos_via_helper)} repos")
        
        # Test get_files_in_repo
        shared_files = helpers['get_files_in_repo']('shared-lib')
        print(f"get_files_in_repo('shared-lib'): {len(shared_files)} files")
        
        # Test cross_repo_imports (this tests the concept even if no actual imports exist)
        cross_imports = helpers['cross_repo_imports']('test-project', 'shared-lib')
        print(f"cross_repo_imports('test-project', 'shared-lib'): {len(cross_imports)} potential")
        
        # Test removing a repo
        print("\nTesting repo removal...")
        db.delete_repo('shared-lib')
        repos_after = db.list_repos()
        print(f"Repos after removal: {len(repos_after)}")
        assert len(repos_after) == 1, f"Expected 1 repo after removal, got {len(repos_after)}"
        
        print("\n✓ Multi-repo support working correctly")
        
    finally:
        db.close()


def main():
    """Run all tests."""
    print("="*60)
    print("RLM v2 Test Suite")
    print("="*60)
    
    # Create temp directory for tests
    test_dir = Path(tempfile.mkdtemp(prefix="rlm_test_"))
    db_path = test_dir / "state" / "index.db"
    
    print(f"\nTest directory: {test_dir}")
    
    try:
        # Run tests
        test_analyzers()
        test_database_features(db_path)
        test_full_indexing(test_dir, db_path)
        test_incremental_indexing(test_dir, db_path)
        test_multi_repo(test_dir, db_path)
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60)
        
        return 0
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 2
    finally:
        # Cleanup
        print(f"\nCleaning up {test_dir}...")
        shutil.rmtree(test_dir, ignore_errors=True)


if __name__ == '__main__':
    sys.exit(main())
