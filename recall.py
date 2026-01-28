"""
jb-recall: Semantic memory layer for workspace files.
Uses ChromaDB for vector storage and sentence-transformers for embeddings.
"""

import jumpboot
import json
import os
import hashlib
from pathlib import Path

# Lazy load heavy imports
_chroma_client = None
_collection = None
_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer('all-MiniLM-L6-v2')
    return _embedder

def get_collection(db_path):
    global _chroma_client, _collection
    if _collection is None:
        import chromadb
        from chromadb.config import Settings
        _chroma_client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        _collection = _chroma_client.get_or_create_collection(
            name="memory",
            metadata={"hnsw:space": "cosine"}
        )
    return _collection

def file_hash(path):
    """Quick hash to detect file changes."""
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start = end - overlap
    return chunks

def index_file(collection, embedder, file_path, force=False):
    """Index a single file, skipping if unchanged."""
    path = Path(file_path)
    if not path.exists() or not path.is_file():
        return {"status": "skipped", "reason": "not a file"}
    
    # Skip binary files
    try:
        text = path.read_text(encoding='utf-8')
    except:
        return {"status": "skipped", "reason": "not text"}
    
    # Check if already indexed with same hash
    current_hash = file_hash(file_path)
    doc_id_prefix = str(path.absolute())
    
    # Check existing
    existing = collection.get(where={"path": str(path.absolute())})
    if existing['ids'] and not force:
        if existing['metadatas'] and existing['metadatas'][0].get('hash') == current_hash:
            return {"status": "skipped", "reason": "unchanged"}
        # Delete old entries
        collection.delete(ids=existing['ids'])
    
    # Chunk and embed
    chunks = chunk_text(text)
    if not chunks:
        return {"status": "skipped", "reason": "empty"}
    
    embeddings = embedder.encode(chunks).tolist()
    
    # Store
    ids = [f"{doc_id_prefix}::{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "path": str(path.absolute()),
            "filename": path.name,
            "chunk_idx": i,
            "hash": current_hash
        }
        for i in range(len(chunks))
    ]
    
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas
    )
    
    return {"status": "indexed", "chunks": len(chunks), "path": str(path)}

def index_directory(collection, embedder, dir_path, extensions=None, force=False):
    """Recursively index a directory."""
    if extensions is None:
        extensions = ['.md', '.txt', '.py', '.go', '.js', '.ts', '.json', '.yaml', '.yml']
    
    results = {"indexed": 0, "skipped": 0, "files": []}
    dir_path = Path(dir_path)
    
    for path in dir_path.rglob('*'):
        if path.is_file() and path.suffix.lower() in extensions:
            # Skip hidden and common ignore patterns
            if any(part.startswith('.') for part in path.parts):
                continue
            if 'node_modules' in path.parts or '__pycache__' in path.parts:
                continue
            
            result = index_file(collection, embedder, str(path), force)
            if result['status'] == 'indexed':
                results['indexed'] += 1
            else:
                results['skipped'] += 1
            results['files'].append(result)
    
    return results

def search(collection, embedder, query, limit=5):
    """Semantic search over indexed content."""
    query_embedding = embedder.encode([query]).tolist()
    
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=limit,
        include=["documents", "metadatas", "distances"]
    )
    
    # Format results
    formatted = []
    if results['ids'] and results['ids'][0]:
        for i in range(len(results['ids'][0])):
            formatted.append({
                "id": results['ids'][0][i],
                "score": 1 - results['distances'][0][i],  # Convert distance to similarity
                "text": results['documents'][0][i],
                "path": results['metadatas'][0][i]['path'],
                "filename": results['metadatas'][0][i]['filename'],
                "chunk_idx": results['metadatas'][0][i]['chunk_idx']
            })
    
    return formatted

def handle_command(cmd: dict) -> dict:
    """Handle incoming commands."""
    global _collection, _embedder
    
    action = cmd.get('cmd', '')
    
    if action == 'init':
        db_path = cmd.get('db_path', os.path.expanduser('~/.jb-recall/db'))
        os.makedirs(db_path, exist_ok=True)
        _embedder = get_embedder()
        _collection = get_collection(db_path)
        stats = _collection.count()
        return {"status": "ok", "db_path": db_path, "count": stats}
    
    elif action == 'index_file':
        if not _collection:
            return {"status": "error", "error": "not initialized"}
        return index_file(_collection, _embedder, cmd['path'], cmd.get('force', False))
    
    elif action == 'index_dir':
        if not _collection:
            return {"status": "error", "error": "not initialized"}
        return index_directory(
            _collection, _embedder, 
            cmd['path'], 
            cmd.get('extensions'),
            cmd.get('force', False)
        )
    
    elif action == 'search':
        if not _collection:
            return {"status": "error", "error": "not initialized"}
        results = search(_collection, _embedder, cmd['query'], cmd.get('limit', 5))
        return {"status": "ok", "results": results}
    
    elif action == 'stats':
        if not _collection:
            return {"status": "error", "error": "not initialized"}
        return {"status": "ok", "count": _collection.count()}
    
    elif action == 'clear':
        if _collection:
            all_ids = _collection.get()['ids']
            if all_ids:
                _collection.delete(ids=all_ids)
        return {"status": "ok"}
    
    elif action == 'quit':
        return {"status": "bye"}
    
    return {"status": "error", "error": f"unknown command: {action}"}

def main():
    """Main loop using jumpboot's JSONQueue."""
    queue = jumpboot.JSONQueue(jumpboot.Pipe_in, jumpboot.Pipe_out)
    
    # Signal ready
    queue.put({"status": "ready"})
    
    while True:
        try:
            cmd = queue.get(block=True, timeout=1)
        except TimeoutError:
            continue
        except EOFError:
            break
        except Exception:
            continue
        
        if cmd is None:
            continue
        
        try:
            result = handle_command(cmd)
            queue.put(result)
            if cmd.get('cmd') == 'quit':
                break
        except Exception as e:
            queue.put({"status": "error", "error": str(e)})

if __name__ == "__main__":
    main()
