# jb-recall

Semantic memory layer for workspace files. Index your notes, code, and documents for natural language search.

Built with [jumpboot](https://github.com/richinsley/jumpboot) - embeds Python + ML directly in a Go binary.

## Features

- **Semantic search** - Find content by meaning, not just keywords
- **Automatic chunking** - Splits large files for better retrieval
- **Change detection** - Only re-indexes modified files
- **Local-first** - All data stays on your machine (ChromaDB)

## Installation

```bash
go build -o jb-recall .
mv jb-recall ~/bin/  # or wherever you keep binaries
```

First run will download the embedding model (~90MB) and create a Python environment.

## Usage

```bash
# Index a file or directory
jb-recall index ~/notes
jb-recall index ./README.md

# Search
jb-recall search "how to configure the API"
jb-recall q migration steps      # shorthand

# JSON output (for scripts/integrations)
jb-recall json "database schema"

# Stats and maintenance
jb-recall stats
jb-recall clear
```

## How it works

1. **Go wrapper** manages the CLI and spawns a Python subprocess via jumpboot
2. **Python backend** uses sentence-transformers (`all-MiniLM-L6-v2`) for embeddings
3. **ChromaDB** stores vectors locally in `~/.jb-recall/db`

Files are chunked into ~500 character segments with overlap, embedded, and stored with metadata for retrieval.

## Supported file types

`.md`, `.txt`, `.py`, `.go`, `.js`, `.ts`, `.json`, `.yaml`, `.yml`

Skips hidden files, `node_modules`, `__pycache__`, etc.

## Requirements

- Go 1.21+
- Internet connection (first run only, for model download)

## License

MIT
