# Claude RLM Quick Reference

One-page reference for common operations.

## What's New in This Version

- **Multi-repo support** - Index multiple repos into single database
- **Persistent connection** - No more connection churn
- **WAL mode** - Better concurrency and crash recovery
- **FTS5 search** - Fast full-text search without loading all chunks
- **Incremental indexing** - Only re-index changed files
- **JavaScript/TypeScript support** - Regex-based chunking
- **Java support** - Classes, interfaces, methods, imports
- **Parent tracking** - Methods linked to their classes

## Multi-Repo Workflow

```bash
REPL=".claude/skills/rlm/scripts/rlm_repl.py"

# Index multiple repos into same database
python3 $REPL init /path/to/frontend --name frontend
python3 $REPL init /path/to/backend --name backend-api
python3 $REPL init /path/to/shared --name shared-lib

# List all repos
python3 $REPL repos

# Remove a repo
python3 $REPL repos --remove old-repo -y

# Search across ALL repos
python3 $REPL search --symbol UserService

# Search in specific repo
python3 $REPL exec -c "print(find_symbol('User', repo='backend-api'))"
```

## Essential Commands

```bash
REPL=".claude/skills/rlm/scripts/rlm_repl.py"

# Initialize and index codebase
python3 $REPL init /path/to/codebase --name my-repo

# Force full re-index
python3 $REPL init /path/to/codebase --name my-repo --full

# Check status with breakdowns
python3 $REPL status --languages --chunks

# List repos
python3 $REPL repos

# Search symbols
python3 $REPL search --symbol UserAuth

# Search with FTS5 (FAST - preferred)
python3 $REPL search --pattern "authenticate" --fts

# Vacuum and optimize database
python3 $REPL vacuum

# Reset everything
python3 $REPL reset -y
```

## Helper Functions (in `exec` mode)

### Symbol Search
```python
find_symbol('UserAuth')                         # Find in ALL repos
find_symbol('process', 'function')              # Find functions only
find_symbol('User', 'class', repo='backend')    # Find in specific repo
find_symbol_exact('main', 'function')           # Exact name match
get_class_methods('UserAuth')                   # Get all methods of a class
```

### Repo Management
```python
list_repos()                                    # List all indexed repos
get_repo('backend')                             # Get repo info
get_files_in_repo('backend')                    # All files in a repo
cross_repo_imports('frontend', 'shared')        # Cross-repo dependencies
```

### Chunk Operations
```python
get_chunk(123)                                  # Get chunk by ID
get_file_chunks('src/main.py')                  # Get all chunks in file
search_content('authenticate')                  # FTS5 search (FAST)
search_chunks(r'async def.*', 'function')       # Regex search (flexible)
```

### File Queries
```python
get_files_by_language('python')            # All Python files
get_files_by_language('java')              # All Java files
get_imports_for_file('src/main.py')        # What this file imports
get_files_importing('fastapi')             # Files that import FastAPI
analyze_dependencies('src/auth.py')        # Full dependency analysis
```

### Materialization
```python
write_chunk_to_file(123)                   # Write chunk with metadata header
write_file_chunks('src/main.py')           # Write all chunks from file
write_chunks_combined([1,2,3], 'out.txt')  # Combine chunks into single file
```

### Stats & SQL
```python
stats()                                    # Full statistics with breakdowns
db.query("SELECT * FROM symbols WHERE ...") # Direct SQL
db.search_content_fts("keyword")           # Direct FTS5 search
```

## Search: FTS5 vs Regex

| Feature | FTS5 (`--fts`) | Regex |
|---------|---------------|-------|
| Speed | O(log n) - instant | O(n) - scans all |
| Memory | Minimal | Loads chunks |
| Syntax | Words, phrases | Full regex |
| Use case | Keyword search | Pattern matching |

```bash
# FTS5 - for keywords (FAST)
python3 $REPL search --pattern "authentication login" --fts

# Regex - for patterns (flexible)
python3 $REPL search --pattern "def\s+test_.*\("
```

## Supported Languages

| Language | Chunking | Symbols | Imports |
|----------|----------|---------|---------|
| Python | AST ✓ | ✓ | ✓ |
| JavaScript | Regex ✓ | ✓ | ✓ |
| TypeScript | Regex ✓ | ✓ | ✓ |
| Java | Regex ✓ | ✓ | ✓ |
| Others | File-level | - | - |

## Common Patterns

### Find Definition with Context
```python
results = find_symbol_exact('UserService', 'class')
if results:
    r = results[0]
    print(f"Found: {r['filepath']}:{r['definition_line']}")
    if r['chunk_id']:
        chunk = get_chunk(r['chunk_id'])
        print(chunk['content'][:500])
```

### Find All Usages
```python
# Fast FTS5 search
matches = search_content('send_email')
for m in matches:
    print(f"{m['filepath']}:{m['start_line']} - {m['symbol_name']}")
```

### Analyze Java Class
```python
# Find the class
cls = find_symbol_exact('UserController', 'class')[0]
print(f"Class at {cls['filepath']}:{cls['definition_line']}")

# Get its methods
methods = get_class_methods('UserController')
for m in methods:
    print(f"  {m['symbol_name']}() line {m['definition_line']}")
```

### Dependency Analysis
```python
deps = analyze_dependencies('src/services/auth.py')

print("Imports:")
for imp in deps['imports']:
    print(f"  {imp['module_name']}")

print("Imported by:")
for f in deps['imported_by']:
    print(f"  {f['filepath']}")
```

### Generate Chunks for Subagent
```python
# Find relevant symbols
auth_symbols = find_symbol('auth')
chunk_ids = [s['chunk_id'] for s in auth_symbols if s['chunk_id']]

# Write combined file for subagent (reduces calls)
if chunk_ids:
    path = write_chunks_combined(chunk_ids[:10], '.claude/rlm_state/chunks/auth_combined.txt')
    print(f"Wrote {path}")
```

## SQL Quick Reference

```sql
-- Find all classes across languages
SELECT s.symbol_name, f.filepath, f.language
FROM symbols s
JOIN files f ON s.file_id = f.file_id
WHERE s.symbol_type = 'class'

-- Find methods with their parent class
SELECT s.symbol_name, s.parent_symbol, f.filepath
FROM symbols s
JOIN files f ON s.file_id = f.file_id
WHERE s.symbol_type = 'method'
  AND s.parent_symbol IS NOT NULL

-- Most imported modules
SELECT module_name, COUNT(*) as count
FROM imports
GROUP BY module_name
ORDER BY count DESC
LIMIT 20

-- Large files by chunk count
SELECT f.filepath, COUNT(c.chunk_id) as chunks
FROM files f
JOIN chunks c ON f.file_id = c.file_id
GROUP BY f.filepath
ORDER BY chunks DESC
LIMIT 20

-- Java interfaces
SELECT s.symbol_name, f.filepath
FROM symbols s
JOIN files f ON s.file_id = f.file_id
WHERE s.symbol_type = 'interface'
```

## Performance Tips

1. **Use FTS5** for keyword searches - 100x faster than regex
2. **Incremental indexing** is automatic - just run `init` again
3. **Batch chunks** - use `write_chunks_combined()` for subagent calls
4. **Early termination** - stop at 3-5 high-confidence results
5. **Run vacuum** periodically to reclaim space
6. **Filter by language** if you only need one

## Troubleshooting

```bash
# Database locked
python3 -c "import sqlite3; c=sqlite3.connect('.claude/rlm_state/index.db'); c.execute('PRAGMA wal_checkpoint(TRUNCATE)'); c.close()"

# Rebuild FTS index
python3 $REPL vacuum

# Check WAL mode is enabled
python3 $REPL exec -c "print(db.query('PRAGMA journal_mode'))"

# Force re-index a file
python3 $REPL exec -c "
db.execute('DELETE FROM files WHERE filepath = ?', ('src/main.py',))
db.commit()
"
python3 $REPL init /path/to/codebase

# Verbose logging
python3 $REPL -v init /path/to/codebase
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
└── rlm_state/
    ├── index.db          # SQLite database
    ├── index.db-wal      # WAL file (auto-managed)
    ├── index.db-shm      # Shared memory (auto-managed)
    └── chunks/           # Materialized chunks
```

## Database Schema

```
files
  - file_id, filepath, size_bytes, language, hash, indexed_at

chunks
  - chunk_id, file_id, chunk_type, symbol_name, parent_symbol
  - start_line, end_line, content

chunks_fts (FTS5 virtual table)
  - content, symbol_name, chunk_id

symbols
  - symbol_id, symbol_name, symbol_type, parent_symbol
  - chunk_id, file_id, definition_line

imports
  - import_id, file_id, module_name
  - imported_names, is_relative, line_number

references (future)
  - ref_id, symbol_id, file_id, line_number, ref_type
```

## Environment Variables

```bash
export RLM_DB_PATH=.claude/rlm_state/index.db
export RLM_CHUNKS_DIR=.claude/rlm_state/chunks
```

## Links

- Full documentation: README.md
- Skill documentation: .claude/skills/rlm/SKILL.md
- RLM paper: https://arxiv.org/abs/2512.24601
