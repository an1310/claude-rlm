"""Shared fixtures and utilities for RLM tests."""

import shutil
import sys
import tempfile
from pathlib import Path

# Add scripts directory to path for imports
SCRIPTS_DIR = Path(__file__).parent.parent / ".claude" / "skills" / "rlm" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

# Try pytest import, fall back to simple fixture system
try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False


class TestContext:
    """Context manager for test fixtures when pytest isn't available."""

    def __init__(self):
        self.test_dir = None
        self.db_path = None
        self.memory_db_path = None
        self.chunks_dir = None

    def __enter__(self):
        self.test_dir = Path(tempfile.mkdtemp(prefix="rlm_test_"))
        self.db_path = self.test_dir / "state" / "index.db"
        self.memory_db_path = self.test_dir / "state" / "memory.db"
        self.chunks_dir = self.test_dir / "chunks"
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.test_dir and self.test_dir.exists():
            shutil.rmtree(self.test_dir, ignore_errors=True)


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


# Pytest fixtures (only defined if pytest is available)
if PYTEST_AVAILABLE:
    @pytest.fixture
    def test_dir(tmp_path):
        """Provide a temporary test directory."""
        return tmp_path

    @pytest.fixture
    def db_path(test_dir):
        """Provide a database path."""
        return test_dir / "state" / "index.db"

    @pytest.fixture
    def memory_db_path(test_dir):
        """Provide a memory database path."""
        return test_dir / "state" / "memory.db"

    @pytest.fixture
    def chunks_dir(test_dir):
        """Provide a chunks directory."""
        return test_dir / "chunks"

    @pytest.fixture
    def test_files(test_dir):
        """Create test source files."""
        create_test_files(test_dir)
        return test_dir
