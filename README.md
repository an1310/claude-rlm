# Claude Memory

A persistent memory and codebase indexing plugin for Claude Code. Combines SQLite-backed code analysis with a semantic memory system that learns and remembers across sessions.

## What It Does

**Claude Memory** solves two problems:

1. **Large Codebase Analysis** - When codebases exceed Claude's context window, this plugin provides an indexed, searchable representation using SQLite with FTS5 full-text search. It extracts functions, classes, methods, and imports from Python, JavaScript, TypeScript, and Java.

2. **Session Persistence** - Claude Code sessions are ephemeral. This plugin maintains a persistent memory database that captures user preferences, project decisions, and learned context. Relevant memories are automatically injected at session start.

## Key Features

### Codebase Indexing
- **Multi-repository support** - Index multiple repos into a single searchable database
- **Incremental indexing** - Only re-processes changed files (hash-based)
- **AST analysis** (Python) and **regex analysis** (JS/TS/Java)
- **FTS5 full-text search** - O(log n) content queries
- **Parent tracking** - Methods linked to their containing classes
- **Chunk materialization** - Export code chunks for subagent analysis

### Memory System
- **Automatic context injection** - Relevant memories injected at session start via hooks
- **Automatic capture** - Pattern-based extraction of learnings from conversations
- **Memory types** - Facts, preferences, instructions, decisions, context, patterns
- **User profile** - Persistent preferences across sessions
- **Local semantic search** - Optional HNSW vector index with fastembed (air-gapped)

## Quick Start

### Install

```bash
# Clone the repository
git clone https://github.com/an1310/claude-memory.git
cd claude-memory

# Optional: Install semantic search dependencies
pip install -r requirements.txt
```

### Index a Codebase

```bash
# Index a repository
python3 .claude/skills/rlm/scripts/rlm_repl.py init /path/to/your/project

# Index multiple repos into the same database
python3 .claude/skills/rlm/scripts/rlm_repl.py init /path/to/frontend --name frontend
python3 .claude/skills/rlm/scripts/rlm_repl.py init /path/to/backend --name backend

# Check what was indexed
python3 .claude/skills/rlm/scripts/rlm_repl.py status --languages
```

### Search the Index

```bash
# Find symbol definitions
python3 .claude/skills/rlm/scripts/rlm_repl.py search --symbol UserService

# Full-text search (fast)
python3 .claude/skills/rlm/scripts/rlm_repl.py search --pattern "authenticate" --fts

# Find imports
python3 .claude/skills/rlm/scripts/rlm_repl.py search --imports "express"
```

### Use the Memory System

```bash
# Remember something
python3 .claude/skills/rlm/scripts/rlm_repl.py remember "Project uses PostgreSQL 15"

# Search memories
python3 .claude/skills/rlm/scripts/rlm_repl.py memory search "database"

# View session context
python3 .claude/skills/rlm/scripts/rlm_repl.py memory context

# Set a preference
python3 .claude/skills/rlm/scripts/rlm_repl.py memory set-pref test_framework pytest
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Claude Code Session                       │
│                    (Claude Opus/Sonnet)                      │
└─────────────────────┬───────────────────────────────────────┘
                      │
        ┌─────────────┴─────────────┐
        ▼                           ▼
┌───────────────────┐     ┌─────────────────────┐
│  Code Index DB    │     │    Memory DB        │
│  (index.db)       │     │    (memory.db)      │
│  ├── repos        │     │    ├── memories     │
│  ├── files        │     │    ├── profile      │
│  ├── chunks       │     │    └── sessions     │
│  ├── chunks_fts   │     │                     │
│  ├── symbols      │     │  ┌─────────────────┐│
│  └── imports      │     │  │ Vector Index    ││
│                   │     │  │ (optional)      ││
└───────────────────┘     │  └─────────────────┘│
                          └─────────────────────┘
        ▲                           ▲
        │                           │
        └───────────────────────────┘
                      │
              ┌───────┴───────┐
              │   Subagents   │
              │  (Haiku)      │
              │  rlm-subcall  │
              │  memory-ext   │
              └───────────────┘
```

## Session Hooks

The memory system uses Claude Code hooks for automatic operation:

- **SessionStart** - Injects relevant context from memories into the session
- **Stop** - Captures learnings from the conversation transcript

These are configured in `.claude/settings.json`:

```json
{
  "hooks": {
    "SessionStart": [
      {"type": "command", "command": "python3 .claude/skills/rlm/scripts/session_start.py"}
    ],
    "Stop": [
      {"type": "command", "command": "python3 .claude/skills/rlm/scripts/session_stop.py"}
    ]
  }
}
```

## Memory Types

| Type | Description | Example |
|------|-------------|---------|
| `fact` | Learned information about the project | "Project uses PostgreSQL 15" |
| `preference` | User preferences for style/tools | "Prefers functional style" |
| `instruction` | Standing instructions to follow | "Always run tests before commit" |
| `decision` | Design decisions made | "Using JWT for authentication" |
| `context` | Project context | "Migrating from REST to GraphQL" |
| `pattern` | Code patterns to follow | "Error handling via Result type" |

## Automatic Capture

The session stop hook automatically captures memories from phrases like:
- "remember that..." / "remember this..."
- "I prefer..." / "I like..."
- "always..." / "never..."
- "we decided..." / "let's go with..."

## CLI Reference

### Codebase Commands

```bash
# Index
rlm_repl.py init <path> [--name NAME] [--extensions .py,.java] [--full]

# Manage repos
rlm_repl.py repos [--remove NAME] [-y]

# Status
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

### Memory Commands

```bash
# Add memory
rlm_repl.py memory add "content" [--type TYPE] [--importance 0.5] [--tags a,b]
rlm_repl.py remember "content"  # Quick add

# Search/list
rlm_repl.py memory search "query"
rlm_repl.py memory list [--type TYPE] [--limit N]

# Context
rlm_repl.py memory context [--max-chars 2000]

# Profile
rlm_repl.py memory profile
rlm_repl.py memory set-pref <key> <value>

# Stats
rlm_repl.py memory stats
```

## Helper Functions (exec mode)

```python
# Codebase helpers
find_symbol(name, symbol_type=None, repo=None)
find_symbol_exact(name, symbol_type=None, repo=None)
get_class_methods(class_name)
search_content(query, limit=100)  # FTS5
get_files_by_language(language)
get_files_importing(module_name)
write_chunk_to_file(chunk_id)
stats()

# Memory helpers
memory_add(content, memory_type='fact', importance=0.5, tags=None)
memory_search(query, limit=10)
memory_list(memory_type=None, limit=20)
memory_context(max_chars=2000)
set_preference(key, value)
get_preference(key, default=None)
memory_stats()
```

## Supported Languages

| Language | Analysis | Chunks | Symbols | Imports | Parent Tracking |
|----------|----------|--------|---------|---------|-----------------|
| Python | AST | ✓ | ✓ | ✓ | ✓ |
| JavaScript | Regex | ✓ | ✓ | ✓ | - |
| TypeScript | Regex | ✓ | ✓ | ✓ | - |
| Java | Regex | ✓ | ✓ | ✓ | ✓ |

## Semantic Search (Optional)

For vector-based semantic search, install the optional dependencies:

```bash
pip install fastembed>=0.3.0 hnswlib>=0.8.0 numpy>=1.24.0
```

First embedding generation downloads ~500MB model (cached thereafter). The system works without these dependencies, falling back to FTS5-only search.

## File Structure

```
.claude/
├── settings.json             # Hook configuration
├── skills/rlm/
│   ├── SKILL.md              # Skill documentation
│   └── scripts/
│       ├── rlm_repl.py       # Main CLI
│       ├── memory_db.py      # Memory database
│       ├── embeddings.py     # Vector search
│       ├── session_start.py  # Context injection hook
│       └── session_stop.py   # Memory capture hook
├── agents/
│   ├── rlm-subcall.md        # Code chunk analyzer
│   └── memory-extractor.md   # Memory extraction agent
└── rlm_state/
    ├── index.db              # Code index
    ├── memory.db             # Memory store
    ├── embeddings.hnsw       # Vector index (optional)
    └── chunks/               # Materialized chunks
```

## Testing

```bash
python3 test_rlm.py

# Or with pytest
python -m pytest tests/
```

## Background

This plugin implements techniques from the Recursive Language Model (RLM) paper for handling large contexts:

> **Recursive Language Models**
> Alex L. Zhang, Tim Kraska, Omar Khattab
> MIT CSAIL
> [arXiv:2512.24601](https://arxiv.org/abs/2512.24601)

The memory system extends this with persistent storage across sessions, enabling Claude to learn and remember user preferences and project context over time.

## License

MIT
