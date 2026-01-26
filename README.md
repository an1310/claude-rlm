# Claude Code RLM

An implementation of Recursive Language Models (RLM) using Claude Code with SQLite-backed indexing for large-scale codebase analysis.

## About

This repository provides an enhanced RLM setup designed for analyzing codebases at scale. It implements the core RLM pattern with significant improvements:

- **Multi-repository support** - Index multiple repos into a single database
- **SQLite with WAL mode** for concurrent access and crash recovery
- **FTS5 full-text search** for instant content queries without memory loading
- **Incremental indexing** that only re-processes changed files
- **Multi-language support** with AST (Python) and regex (JS/TS/Java) analysis
- **Parent symbol tracking** linking methods to their containing classes
- **Persistent connections** with transaction batching for fast bulk operations

**Based on the RLM paper**:
> **Recursive Language Models**  
> Alex L. Zhang, Tim Kraska, Omar Khattab  
> MIT CSAIL  
> [arXiv:2512.24601](https://arxiv.org/abs/2512.24601)

## Multi-Repository Support

A key feature is the ability to index **multiple repositories** into a single database. This is essential for projects that span multiple repos:

```bash
# Index frontend repo
python3 rlm_repl.py init /path/to/frontend --name frontend

# Index backend repo (same database)
python3 rlm_repl.py init /path/to/backend --name backend-api

# Index shared libraries
python3 rlm_repl.py init /path/to/shared-libs --name shared

# List all repos
python3 rlm_repl.py repos

# Search across ALL repos
python3 rlm_repl.py search --symbol UserService

# Search in specific repo (via exec)
python3 rlm_repl.py exec -c "print(find_symbol('User', repo='backend-api'))"

# Find cross-repo dependencies
python3 rlm_repl.py exec -c "print(cross_repo_imports('frontend', 'shared'))"

# Remove a repo
python3 rlm_repl.py repos --remove old-repo
```

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Main Conversation                         │
│                    (Claude Opus 4.5)                         │
└─────────────────────┬───────────────────────────────────────┘
                      │
        ┌─────────────┴─────────────┐
        ▼                           ▼
┌───────────────────┐     ┌─────────────────────┐
│  SQLite Database  │     │  Subagent rlm-subcall│
│  ┌─────────────┐  │     │  (Claude Haiku)      │
│  │ repos       │  │     │                      │
│  │ files       │  │     │  Analyzes chunk      │
│  │ chunks      │◄─┼─────│  files on disk       │
│  │ chunks_fts  │  │     │                      │
│  │ symbols     │  │     └─────────────────────┘
│  │ imports     │  │
│  └─────────────┘  │
└───────────────────┘
```

### Database Schema

```sql
repos           -- Repository metadata (name, root_path, file_count)
files           -- File metadata (path, language, hash, repo_id)
chunks          -- Code chunks (functions, classes, methods)
chunks_fts      -- FTS5 virtual table for fast content search
symbols         -- Symbol definitions with parent tracking
imports         -- Import relationships
"references"    -- Cross-references (future)
```

## Supported Languages

| Language | Analysis | Chunks | Symbols | Imports | Parent Tracking |
|----------|----------|--------|---------|---------|-----------------|
| Python | AST | ✓ | ✓ | ✓ | ✓ |
| JavaScript | Regex | ✓ | ✓ | ✓ | - |
| TypeScript | Regex | ✓ | ✓ | ✓ | - |
| Java | Regex | ✓ | ✓ | ✓ | ✓ |
| Others | File-level | - | - | - | - |

## Quick Start

### 1. Index Your Repositories

```bash
# Index first repo
python3 rlm_repl.py init /path/to/frontend --name frontend

# Index second repo into same database
python3 rlm_repl.py init /path/to/backend --name backend

# Index third repo
python3 rlm_repl.py init /path/to/shared --name shared-lib
```

### 2. Check What Was Indexed

```bash
python3 rlm_repl.py status --languages
```

Output:
```
RLM REPL Status
  Database: .claude/rlm_state/index.db
  Size: 45.23 MB
  Repos: 3
  Files: 1,234
  Chunks: 5,678
  Symbols: 12,345

Repos (3):
  frontend: 456 files
    Path: /path/to/frontend
  backend: 512 files
    Path: /path/to/backend
  shared-lib: 266 files
    Path: /path/to/shared

Languages:
  python: 456 files
  java: 389 files
  typescript: 389 files
```

### 3. Search Across All Repos

```bash
# Find symbol in all repos
python3 rlm_repl.py search --symbol UserService

# Find with FTS5 (fast)
python3 rlm_repl.py search --pattern "authenticate" --fts
```

### 4. Search Within Specific Repo

```python
# Via exec command
python3 rlm_repl.py exec -c "
results = find_symbol('User', 'class', repo='backend')
for r in results:
    print(f\"{r['repo_name']}/{r['filepath']}: {r['symbol_name']}\")
"
```

## CLI Reference

### Commands

```bash
# Index a repo (can call multiple times for different repos)
rlm_repl.py init <path> [--name NAME] [--extensions .py,.java] [--full]

# List/manage repos
rlm_repl.py repos [--remove NAME] [-y]

# Show status
rlm_repl.py status [--languages] [--chunks]

# Search
rlm_repl.py search --symbol <name>
rlm_repl.py search --pattern <query> [--fts]
rlm_repl.py search --imports <module>

# Execute code with helpers
rlm_repl.py exec -c "code"

# Maintenance
rlm_repl.py vacuum
rlm_repl.py reset [-y]
```

### Helper Functions (in `exec` mode)

```python
# Symbol search (with optional repo filter)
find_symbol(name, symbol_type=None, repo=None)
find_symbol_exact(name, symbol_type=None, repo=None)
get_class_methods(class_name)

# Content search
search_content(query, limit=100)          # FTS5 (FAST)
search_chunks(pattern, chunk_type, limit) # Regex

# Repo management
list_repos()
get_repo(repo_name)
get_files_in_repo(repo_name)
cross_repo_imports(from_repo, to_repo)

# Chunk operations
get_chunk(chunk_id)
get_file_chunks(filepath)
write_chunk_to_file(chunk_id)
write_chunks_combined(chunk_ids, path)

# File and import queries
get_files_by_language(language)
get_imports_for_file(filepath)
get_files_importing(module_name)
analyze_dependencies(filepath)

# Statistics
stats()

# Direct SQL
db.query(sql, params)
```

## Usage Patterns

### Search Across Multiple Repos

```python
# Find all Service classes across all repos
services = find_symbol('Service', 'class')
print(f"Found {len(services)} services across all repos:")
for s in services:
    print(f"  {s['repo_name']}/{s['filepath']}: {s['symbol_name']}")
```

### Search in Specific Repo

```python
# Find User classes only in backend repo
backend_users = find_symbol('User', 'class', repo='backend')
for u in backend_users:
    print(f"{u['filepath']}:{u['definition_line']}")
```

### Analyze Cross-Repo Dependencies

```python
# What does frontend import from shared-lib?
cross_deps = cross_repo_imports('frontend', 'shared-lib')
print(f"Found {len(cross_deps)} cross-repo imports")
for dep in cross_deps:
    print(f"  {dep['filepath']} imports {dep['module_name']}")
```

### Get All Files in a Repo

```python
# List all files in backend repo
backend_files = get_files_in_repo('backend')
print(f"Backend has {len(backend_files)} files")
for f in backend_files:
    print(f"  {f['filepath']} ({f['language']})")
```

## Performance

### Benchmarks (1M LOC across 3 repos)

| Operation | Time |
|-----------|------|
| Initial indexing (all repos) | 5-15 minutes |
| Incremental re-index | 10-60 seconds |
| Symbol lookup (all repos) | <100ms |
| Symbol lookup (single repo) | <50ms |
| FTS5 content search | <500ms |

## Troubleshooting

### Remove a Repo

```bash
# Interactive
python3 rlm_repl.py repos --remove old-repo

# Non-interactive
python3 rlm_repl.py repos --remove old-repo -y
```

### Re-index a Single Repo

```bash
# Just run init again with --full flag
python3 rlm_repl.py init /path/to/repo --name my-repo --full
```

### Database Size Growing Too Large

```bash
# Vacuum to reclaim space
python3 rlm_repl.py vacuum
```

## File Structure

```
.claude/
├── skills/rlm/
│   ├── SKILL.md
│   └── scripts/
│       └── rlm_repl.py
├── agents/
│   └── rlm-subcall.md
├── rlm_state/
│   ├── index.db          # Single database for all repos
│   ├── index.db-wal
│   └── chunks/
└── CLAUDE.md
```

## Testing

```bash
python3 test_rlm.py
```

Expected output includes multi-repo test:
```
Testing Multi-Repo Support
==========================
Indexing second repo...
All repos (2):
  shared-lib: 1 files
  test-project: 4 files

find_symbol('Service', 'class') across all repos: 3 results
find_symbol in 'shared-lib' only: 1 results
✓ Multi-repo support working correctly
```
