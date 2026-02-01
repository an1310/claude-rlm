---
name: rlm
description: Persistent memory and codebase indexing for Claude Code. SQLite-backed code analysis with FTS5 search, AST/regex chunking, and a memory system that learns across sessions. Supports Python, JavaScript, TypeScript, and Java.
allowed-tools:
  - Read
  - Write
  - Edit
  - Grep
  - Glob
  - Bash
---

# Claude Memory - Codebase Indexing & Persistent Memory

Use this Skill when:
- Analyzing large codebases (100K+ LOC) that exceed typical context limits
- Finding definitions, usages, or cross-references across many files
- Performing architectural analysis or refactoring planning
- Understanding symbol relationships across Python, JavaScript, TypeScript, or Java
- Working with contexts that exhibit "middle child syndrome"

## Key Features

- **Multi-repository support** - Index multiple repos into single database
- **Persistent connection** - No connection churn during indexing
- **WAL mode** - Better concurrency, crash recovery
- **FTS5 full-text search** - O(log n) content search without memory loading
- **Incremental indexing** - Only re-index changed files (checks hash)
- **Multi-language support** - Python (AST), JavaScript/TypeScript (regex), Java (regex)
- **Parent tracking** - Methods linked to their containing classes
- **Transaction batching** - Fast bulk indexing

## Mental Model

```
Main conversation (Claude Opus)
    │
    ├── SQLite database (indexed codebase)
    │   ├── repos table (multiple repositories)
    │   ├── files table (linked to repos)
    │   ├── chunks table + FTS5 index
    │   ├── symbols table (with parent tracking)
    │   └── imports table
    │
    └── Subagent `rlm-subcall` (Claude Haiku)
        └── Analyzes materialized chunk files
```

## Workflow Types

### 1. Initial Indexing

First time with a codebase (or multiple repos):

```bash
# Index a single repo
python3 .claude/skills/rlm/scripts/rlm_repl.py init /path/to/codebase

# Index multiple repos into the same database
python3 .claude/skills/rlm/scripts/rlm_repl.py init /path/to/frontend --name frontend
python3 .claude/skills/rlm/scripts/rlm_repl.py init /path/to/backend --name backend
python3 .claude/skills/rlm/scripts/rlm_repl.py init /path/to/shared --name shared-lib

# Index specific languages only
python3 .claude/skills/rlm/scripts/rlm_repl.py init /path/to/codebase --extensions .py,.java

# List all repos
python3 .claude/skills/rlm/scripts/rlm_repl.py repos

# Check results
python3 .claude/skills/rlm/scripts/rlm_repl.py status --languages --chunks
```

### 2. Incremental Re-indexing

After code changes:

```bash
# Just run init again - unchanged files are skipped automatically
python3 .claude/skills/rlm/scripts/rlm_repl.py init /path/to/codebase

# Force full re-index if needed
python3 .claude/skills/rlm/scripts/rlm_repl.py init /path/to/codebase --full
```

### 3. Query-Driven Workflow

For specific questions:

**Step 1: Classify query intent**
- Definition search: "Where is class X defined?"
- Usage search: "Where is function Y called?"
- Relationship query: "What depends on module Z?"
- Architectural question: "How do these components interact?"

**Step 2: Use appropriate search method**

```bash
# Symbol search (any language)
python3 .claude/skills/rlm/scripts/rlm_repl.py search --symbol "UserAuth"

# FTS5 content search (FAST - use for keywords)
python3 .claude/skills/rlm/scripts/rlm_repl.py search --pattern "authenticate" --fts

# Regex content search (for complex patterns)
python3 .claude/skills/rlm/scripts/rlm_repl.py search --pattern "async def.*process"

# Import search
python3 .claude/skills/rlm/scripts/rlm_repl.py search --imports "springframework"
```

**Step 3: Materialize relevant chunks**

```bash
python3 .claude/skills/rlm/scripts/rlm_repl.py exec <<'PY'
# Find and write chunks
results = find_symbol('UserService', 'class')
for r in results[:5]:
    if r['chunk_id']:
        path = write_chunk_to_file(r['chunk_id'])
        print(f"Wrote: {path}")
PY
```

**Step 4: Delegate to subagent**

```
Use rlm-subcall on .claude/rlm_state/chunks/chunk_000123_UserService.txt 
with query: "How does authentication work?"
```

**Step 5: Synthesize results**

### 4. Comprehensive Analysis Workflow

For broad architectural questions:

```bash
python3 .claude/skills/rlm/scripts/rlm_repl.py exec <<'PY'
# Get overview
s = stats()
print(f"Files: {s['files']}, Chunks: {s['chunks']}, Symbols: {s['symbols']}")
print(f"Languages: {s['languages']}")
print(f"Chunk types: {s['chunk_types']}")

# Find scope
auth_symbols = find_symbol('auth')
print(f"\nAuth-related symbols: {len(auth_symbols)}")

# Group by file
from collections import defaultdict
by_file = defaultdict(list)
for sym in auth_symbols:
    by_file[sym['filepath']].append(sym)

print(f"Across {len(by_file)} files")

# Write combined chunks for efficient subagent processing
chunk_ids = [s['chunk_id'] for s in auth_symbols if s['chunk_id']][:15]
if chunk_ids:
    path = write_chunks_combined(chunk_ids, '.claude/rlm_state/chunks/auth_analysis.txt')
    print(f"\nCombined chunks: {path}")
PY
```

## Available Helpers

### Symbol Search
```python
find_symbol(name, symbol_type=None, repo=None)      # LIKE match, optional repo filter
find_symbol_exact(name, symbol_type=None, repo=None) # Exact match
get_class_methods(class_name)                        # Methods of a class
```

### Repo Management
```python
list_repos()                              # List all indexed repos
get_repo(repo_name)                       # Get repo info
get_files_in_repo(repo_name)              # All files in a repo
cross_repo_imports(from_repo, to_repo)    # Cross-repo dependencies
```

### Content Search
```python
search_content(query, limit=100)          # FTS5 (FAST)
search_chunks(pattern, chunk_type, limit) # Regex (flexible)
```

### Chunk Operations
```python
get_chunk(chunk_id)
get_file_chunks(filepath)
write_chunk_to_file(chunk_id)
write_file_chunks(filepath)
write_chunks_combined(chunk_ids, path)   # Batch for subagent
```

### File & Import Queries
```python
get_files_by_language(language)
get_imports_for_file(filepath)
get_files_importing(module_name)
analyze_dependencies(filepath)
```

### Statistics
```python
stats()  # Returns files, chunks, symbols, languages, chunk_types
```

### Direct SQL
```python
db.query(sql, params)
db.execute(sql, params)
db.search_content_fts(query)
```

## Language-Specific Notes

### Python
- Full AST analysis
- Extracts: functions, async functions, classes, methods, variables
- Tracks: imports (including relative), decorators
- Parent tracking: methods → classes

### JavaScript/TypeScript
- Regex-based analysis
- Extracts: functions, arrow functions, classes, interfaces, types
- Tracks: imports (ES6 and CommonJS require)
- Handles: async functions, export declarations

### Java
- Regex-based analysis  
- Extracts: classes, interfaces, enums, methods, constructors
- Tracks: imports, package declarations
- Parent tracking: methods → classes
- Handles: annotations, access modifiers

## Optimization Strategies

### 1. Use FTS5 for Keyword Searches
```python
# FAST - O(log n)
results = search_content("authentication")

# SLOW - O(n), loads all chunks
results = search_chunks(r"authentication")
```

### 2. Early Termination
```python
results = find_symbol('authenticate', 'function')
if len(results) <= 3:
    # Small result set - process all
    for r in results:
        chunk = get_chunk(r['chunk_id'])
        # analyze...
else:
    # Large result set - sample and verify
    for r in results[:5]:
        # analyze, stop if confidence high
```

### 3. Batch Subagent Calls
```python
# BAD: One subagent call per chunk
for chunk_id in chunk_ids:
    write_chunk_to_file(chunk_id)
    # call subagent...  (N calls)

# GOOD: Combined chunks, fewer calls
write_chunks_combined(chunk_ids[:10], 'batch1.txt')
# call subagent once with batch1.txt
```

### 4. Filter by Language
```python
# If you only need Java files
java_files = get_files_by_language('java')
for f in java_files:
    chunks = get_file_chunks(f['filepath'])
```

### 5. Leverage Parent Tracking
```python
# Find all methods of a specific class
methods = get_class_methods('UserController')
for m in methods:
    print(f"{m['symbol_name']} at line {m['definition_line']}")
```

## Query Classification

| Query Type | Method | Speed | Subagent? |
|------------|--------|-------|-----------|
| "Where is X defined?" | `find_symbol_exact` | <100ms | Usually no |
| "Where is X called?" | `search_content` (FTS5) | <500ms | Maybe |
| "What imports X?" | `get_files_importing` | <100ms | No |
| "How does X work?" | Multiple searches + chunks | 5-30s | Yes |
| "Explain architecture" | Comprehensive analysis | 30-120s | Yes, batched |

## Guardrails

1. **Always use FTS5** for keyword searches - 100x faster
2. **Check index first** - don't blindly chunk everything
3. **Batch chunks** - combine related code for fewer subagent calls
4. **Keep subagent outputs structured** - JSON preferred
5. **Use early termination** - stop when confidence is high
6. **Incremental is default** - no need for `--full` normally
7. **Monitor chunk count** - if >50 chunks needed, refine query
8. **Respect boundaries** - AST/regex chunking preserves function/class integrity

## File Organization

```
.claude/rlm_state/
├── index.db              # SQLite database
├── index.db-wal          # Write-ahead log (auto-managed)
├── index.db-shm          # Shared memory (auto-managed)
└── chunks/               # Materialized chunks
    ├── chunk_000001_MyClass.txt
    ├── chunk_000002_process.txt
    └── auth_analysis.txt  # Combined chunks
```

## Performance Expectations

For 1M LOC mixed codebase:
- Initial indexing: 3-8 minutes (depends on language mix)
- Incremental re-index: 10-60 seconds (only changed files)
- Symbol lookup: <100ms
- FTS5 search: <500ms
- Regex search: 2-10s (depending on result count)
- Chunk materialization: <100ms per chunk

## Troubleshooting

**Database locked**:
```bash
# Checkpoint WAL file
sqlite3 .claude/rlm_state/index.db "PRAGMA wal_checkpoint(TRUNCATE)"
```

**Slow FTS5 searches**:
```bash
# Rebuild FTS index
python3 .claude/skills/rlm/scripts/rlm_repl.py vacuum
```

**Missing chunks for a file**:
```bash
# Check if file was indexed
python3 .claude/skills/rlm/scripts/rlm_repl.py exec -c "
results = db.query('SELECT * FROM files WHERE filepath LIKE ?', ('%main.java%',))
print(results)
"

# Force re-index
python3 .claude/skills/rlm/scripts/rlm_repl.py exec -c "
db.execute('DELETE FROM files WHERE filepath LIKE ?', ('%main.java%',))
"
python3 .claude/skills/rlm/scripts/rlm_repl.py init /path/to/codebase
```

**Java annotations not captured**:
The analyzer looks backwards for `@` lines before class/method declarations.
If issues persist, the regex patterns in `JavaAnalyzer` may need adjustment.

---

## Memory System

Claude Memory includes a persistent memory system for storing learned facts, preferences, and context across sessions.

### Key Features

- **Automatic context injection** - Relevant memories injected at session start
- **Automatic capture** - Pattern-based extraction of memories from conversations
- **Local semantic search** - HNSW vector index with fastembed (optional)
- **User profile** - Persistent preferences
- **Zero cloud dependencies** - Fully air-gapped operation

### Memory Types

| Type | Description | Example |
|------|-------------|---------|
| `fact` | Learned information about project/domain | "Project uses PostgreSQL" |
| `preference` | User preferences for style/tools | "Prefers functional style" |
| `instruction` | Standing instructions to follow | "Always run tests" |
| `decision` | Design decisions made | "Using JWT for auth" |
| `context` | Project context | "Migrating from v1 API" |
| `pattern` | Code patterns to follow | "Error handling via Result type" |

### Memory Commands

```bash
# Add a memory
python3 rlm_repl.py memory add "User prefers TypeScript" --type preference --importance 0.8

# Quick remember
python3 rlm_repl.py remember "Project uses Prisma for database"

# Search memories
python3 rlm_repl.py memory search "database"

# List all memories
python3 rlm_repl.py memory list
python3 rlm_repl.py memory list --type preference

# Get session context
python3 rlm_repl.py memory context

# View statistics
python3 rlm_repl.py memory stats

# View/set preferences
python3 rlm_repl.py memory profile
python3 rlm_repl.py memory set-pref code_style functional
```

### Memory Helpers (in exec mode)

```python
# Add a memory
memory_add("User prefers tabs over spaces", memory_type='preference', importance=0.7)

# Search memories
results = memory_search("database")
for m in results:
    print(f"{m['content']} ({m['importance']})")

# List memories by type
prefs = memory_list(memory_type='preference')

# Get session context
context = memory_context(max_chars=2000)
print(context)

# Get/set preferences
set_preference('test_framework', 'pytest')
framework = get_preference('test_framework', 'unittest')

# Memory statistics
stats = memory_stats()
print(f"Total memories: {stats['total_memories']}")
```

### Automatic Capture Patterns

The system automatically captures memories from phrases like:
- "remember that..." / "remember this..."
- "I prefer..." / "I like..."
- "always..." / "never..."
- "we decided..." / "let's go with..."

### Semantic Search (Optional)

For semantic search, install optional dependencies:

```bash
pip install fastembed>=0.3.0 hnswlib>=0.8.0 numpy>=1.24.0
```

First embedding generation downloads ~500MB model (cached thereafter).

The system gracefully falls back to FTS5-only search if dependencies aren't available.

### Context Budget

Session start injection is limited to ~2000 characters to avoid prompt bloat. The system prioritizes:
1. User preferences
2. Standing instructions
3. High-importance memories
4. Recent context

### File Organization

```
.claude/rlm_state/
├── index.db              # Code index database
├── memory.db             # Memory database
├── embeddings.hnsw       # Vector index (if enabled)
├── embeddings_meta.json  # Embedding metadata
└── chunks/               # Materialized chunks
```
