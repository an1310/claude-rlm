#!/usr/bin/env python3
"""Local embeddings module for Claude Memory.

Provides vector-based semantic search using:
- fastembed for local ONNX-based embeddings (no cloud dependencies)
- hnswlib for efficient approximate nearest neighbor search

Designed for air-gapped operation with graceful degradation:
- If dependencies are missing, falls back to FTS5-only search
- First embedding generation downloads ~500MB model (cached thereafter)

Usage:
    embeddings = EmbeddingIndex()
    embedding_id = embeddings.add_text("User prefers TypeScript over JavaScript")
    results = embeddings.search("programming language preference", k=5)
"""

import json
import struct
from pathlib import Path
from typing import Any

# Graceful import with fallback
_EMBEDDINGS_AVAILABLE = False
_IMPORT_ERROR = None

try:
    import numpy as np
    from fastembed import TextEmbedding
    import hnswlib
    _EMBEDDINGS_AVAILABLE = True
except ImportError as e:
    _IMPORT_ERROR = str(e)


def embeddings_available() -> bool:
    """Check if embedding dependencies are available."""
    return _EMBEDDINGS_AVAILABLE


def get_import_error() -> str | None:
    """Get the import error message if dependencies failed to load."""
    return _IMPORT_ERROR


class EmbeddingIndex:
    """HNSW-based vector index for semantic search.

    Uses fastembed for local embedding generation and hnswlib for
    efficient approximate nearest neighbor search.

    The index is persisted to disk and can be incrementally updated.
    """

    # Default embedding model - small, fast, good quality
    DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
    DEFAULT_DIM = 384  # Dimension for bge-small

    DEFAULT_INDEX_PATH = Path(".claude/rlm_state/embeddings.hnsw")
    DEFAULT_METADATA_PATH = Path(".claude/rlm_state/embeddings_meta.json")

    def __init__(
        self,
        index_path: Path | str | None = None,
        metadata_path: Path | str | None = None,
        model_name: str | None = None,
        ef_construction: int = 200,
        M: int = 16,
    ):
        """Initialize the embedding index.

        Args:
            index_path: Path to the HNSW index file
            metadata_path: Path to the metadata JSON file
            model_name: Fastembed model name (default: bge-small-en-v1.5)
            ef_construction: HNSW construction parameter (higher = better quality, slower build)
            M: HNSW M parameter (connections per node, higher = better quality, more memory)
        """
        if not _EMBEDDINGS_AVAILABLE:
            raise RuntimeError(
                f"Embedding dependencies not available: {_IMPORT_ERROR}. "
                "Install with: pip install fastembed hnswlib numpy"
            )

        self.index_path = Path(index_path) if index_path else self.DEFAULT_INDEX_PATH
        self.metadata_path = Path(metadata_path) if metadata_path else self.DEFAULT_METADATA_PATH
        self.model_name = model_name or self.DEFAULT_MODEL
        self.ef_construction = ef_construction
        self.M = M

        # Ensure directories exist
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize embedding model (lazy load on first use)
        self._model: TextEmbedding | None = None
        self._dim: int = self.DEFAULT_DIM

        # Initialize or load index
        self._index: hnswlib.Index | None = None
        self._metadata: dict[int, dict] = {}  # embedding_id -> metadata
        self._next_id: int = 0

        self._load_or_create_index()

    def _get_model(self) -> TextEmbedding:
        """Get or create the embedding model (lazy initialization)."""
        if self._model is None:
            self._model = TextEmbedding(model_name=self.model_name)
            # Update dimension based on model
            test_embedding = list(self._model.embed(["test"]))[0]
            self._dim = len(test_embedding)
        return self._model

    def _load_or_create_index(self):
        """Load existing index or create a new one."""
        if self.index_path.exists() and self.metadata_path.exists():
            try:
                self._load_index()
                return
            except Exception:
                # Corrupted index, recreate
                pass

        self._create_new_index()

    def _create_new_index(self):
        """Create a new HNSW index."""
        self._index = hnswlib.Index(space='cosine', dim=self._dim)
        # Initialize with reasonable starting capacity
        self._index.init_index(
            max_elements=10000,
            ef_construction=self.ef_construction,
            M=self.M
        )
        self._index.set_ef(50)  # Search-time parameter
        self._metadata = {}
        self._next_id = 0
        self._save_index()

    def _load_index(self):
        """Load index from disk."""
        # Load metadata first to get dimension
        with open(self.metadata_path, 'r') as f:
            data = json.load(f)
            self._metadata = {int(k): v for k, v in data['metadata'].items()}
            self._next_id = data['next_id']
            self._dim = data.get('dim', self.DEFAULT_DIM)

        # Load HNSW index
        self._index = hnswlib.Index(space='cosine', dim=self._dim)
        self._index.load_index(str(self.index_path))
        self._index.set_ef(50)

    def _save_index(self):
        """Save index to disk."""
        if self._index is None:
            return

        # Save HNSW index
        self._index.save_index(str(self.index_path))

        # Save metadata
        with open(self.metadata_path, 'w') as f:
            json.dump({
                'metadata': self._metadata,
                'next_id': self._next_id,
                'dim': self._dim,
                'model': self.model_name,
            }, f)

    def _ensure_capacity(self, needed: int = 1):
        """Ensure index has capacity for new elements."""
        if self._index is None:
            self._create_new_index()
            return

        current_count = self._index.get_current_count()
        max_elements = self._index.get_max_elements()

        if current_count + needed >= max_elements:
            # Need to resize - double capacity
            new_max = max(max_elements * 2, current_count + needed + 1000)
            self._index.resize_index(new_max)

    # ==========================================================================
    # Core Operations
    # ==========================================================================

    def add_text(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Add a text to the index and return its embedding_id."""
        model = self._get_model()

        # Generate embedding
        embedding = list(model.embed([text]))[0]

        # Ensure capacity and add
        self._ensure_capacity(1)

        embedding_id = self._next_id
        self._next_id += 1

        self._index.add_items(
            data=np.array([embedding]),
            ids=np.array([embedding_id])
        )

        # Store metadata
        self._metadata[embedding_id] = {
            'text': text[:500],  # Store truncated text for reference
            **(metadata or {})
        }

        self._save_index()
        return embedding_id

    def add_texts(
        self,
        texts: list[str],
        metadata_list: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        """Add multiple texts at once (more efficient). Returns list of embedding_ids."""
        if not texts:
            return []

        model = self._get_model()

        # Generate embeddings in batch
        embeddings = list(model.embed(texts))

        # Ensure capacity
        self._ensure_capacity(len(texts))

        # Generate IDs
        embedding_ids = list(range(self._next_id, self._next_id + len(texts)))
        self._next_id += len(texts)

        # Add to index
        self._index.add_items(
            data=np.array(embeddings),
            ids=np.array(embedding_ids)
        )

        # Store metadata
        for i, eid in enumerate(embedding_ids):
            meta = metadata_list[i] if metadata_list and i < len(metadata_list) else {}
            self._metadata[eid] = {
                'text': texts[i][:500],
                **meta
            }

        self._save_index()
        return embedding_ids

    def search(
        self,
        query: str,
        k: int = 10,
        filter_fn: callable | None = None,
    ) -> list[dict]:
        """Search for similar texts.

        Args:
            query: The search query
            k: Number of results to return
            filter_fn: Optional function to filter results (receives metadata, returns bool)

        Returns:
            List of dicts with 'embedding_id', 'distance', 'text', and other metadata
        """
        if self._index is None or self._index.get_current_count() == 0:
            return []

        model = self._get_model()

        # Generate query embedding
        query_embedding = list(model.embed([query]))[0]

        # Search (get more results if filtering)
        search_k = k * 3 if filter_fn else k
        labels, distances = self._index.knn_query(
            data=np.array([query_embedding]),
            k=min(search_k, self._index.get_current_count())
        )

        # Build results
        results = []
        for label, distance in zip(labels[0], distances[0]):
            if label not in self._metadata:
                continue

            meta = self._metadata[label]

            # Apply filter if provided
            if filter_fn and not filter_fn(meta):
                continue

            results.append({
                'embedding_id': int(label),
                'distance': float(distance),
                'similarity': 1.0 - float(distance),  # Cosine similarity
                **meta
            })

            if len(results) >= k:
                break

        return results

    def search_by_embedding(
        self,
        embedding: list[float] | Any,  # np.ndarray
        k: int = 10,
    ) -> list[dict]:
        """Search using a pre-computed embedding."""
        if self._index is None or self._index.get_current_count() == 0:
            return []

        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding)

        labels, distances = self._index.knn_query(
            data=np.array([embedding]),
            k=min(k, self._index.get_current_count())
        )

        results = []
        for label, distance in zip(labels[0], distances[0]):
            if label not in self._metadata:
                continue
            results.append({
                'embedding_id': int(label),
                'distance': float(distance),
                'similarity': 1.0 - float(distance),
                **self._metadata[label]
            })

        return results

    def get_embedding(self, text: str) -> list[float]:
        """Get the embedding vector for a text."""
        model = self._get_model()
        return list(model.embed([text]))[0].tolist()

    def delete(self, embedding_id: int) -> bool:
        """Delete an embedding by ID.

        Note: hnswlib doesn't support true deletion, so we just remove metadata.
        The index will be rebuilt on save/load if many deletions occur.
        """
        if embedding_id in self._metadata:
            del self._metadata[embedding_id]
            self._save_index()
            return True
        return False

    def get_metadata(self, embedding_id: int) -> dict | None:
        """Get metadata for an embedding."""
        return self._metadata.get(embedding_id)

    def update_metadata(self, embedding_id: int, metadata: dict) -> bool:
        """Update metadata for an embedding."""
        if embedding_id not in self._metadata:
            return False

        self._metadata[embedding_id].update(metadata)
        self._save_index()
        return True

    # ==========================================================================
    # Statistics and Maintenance
    # ==========================================================================

    def get_stats(self) -> dict:
        """Get index statistics."""
        return {
            'total_embeddings': self._index.get_current_count() if self._index else 0,
            'max_capacity': self._index.get_max_elements() if self._index else 0,
            'dimension': self._dim,
            'model': self.model_name,
            'index_path': str(self.index_path),
            'index_size_mb': self.index_path.stat().st_size / (1024 * 1024) if self.index_path.exists() else 0,
        }

    def rebuild_index(self):
        """Rebuild the index from metadata (useful after many deletions)."""
        if not self._metadata:
            self._create_new_index()
            return

        model = self._get_model()

        # Get texts and IDs
        valid_ids = list(self._metadata.keys())
        texts = [self._metadata[eid].get('text', '') for eid in valid_ids]

        # Regenerate embeddings
        embeddings = list(model.embed(texts))

        # Create new index
        self._index = hnswlib.Index(space='cosine', dim=self._dim)
        self._index.init_index(
            max_elements=max(len(valid_ids) * 2, 1000),
            ef_construction=self.ef_construction,
            M=self.M
        )
        self._index.set_ef(50)

        # Add all embeddings
        if embeddings:
            self._index.add_items(
                data=np.array(embeddings),
                ids=np.array(valid_ids)
            )

        self._save_index()


class HybridSearch:
    """Combines FTS5 text search with vector similarity search.

    Uses a weighted combination of:
    - FTS5 keyword matching (for exact terms)
    - Vector similarity (for semantic matching)
    """

    def __init__(
        self,
        memory_db,  # MemoryDatabase instance
        embedding_index: EmbeddingIndex | None = None,
        fts_weight: float = 0.3,
        vector_weight: float = 0.7,
    ):
        """Initialize hybrid search.

        Args:
            memory_db: MemoryDatabase instance for FTS5 search
            embedding_index: EmbeddingIndex for vector search (optional)
            fts_weight: Weight for FTS5 results (0-1)
            vector_weight: Weight for vector results (0-1)
        """
        self.memory_db = memory_db
        self.embedding_index = embedding_index
        self.fts_weight = fts_weight
        self.vector_weight = vector_weight

    def search(
        self,
        query: str,
        k: int = 10,
        use_fts: bool = True,
        use_vectors: bool = True,
    ) -> list[dict]:
        """Perform hybrid search.

        Returns results scored by weighted combination of FTS5 rank and vector similarity.
        """
        results_map: dict[int, dict] = {}  # memory_id -> result

        # FTS5 search
        if use_fts:
            fts_results = self.memory_db.search_memories_fts(query, limit=k * 2)
            for i, mem in enumerate(fts_results):
                # Approximate FTS5 score based on position
                fts_score = 1.0 - (i / (len(fts_results) or 1))
                results_map[mem.memory_id] = {
                    'memory': mem,
                    'fts_score': fts_score,
                    'vector_score': 0.0,
                }

        # Vector search
        if use_vectors and self.embedding_index is not None:
            try:
                vector_results = self.embedding_index.search(query, k=k * 2)
                for r in vector_results:
                    memory_id = r.get('memory_id')
                    if memory_id is None:
                        continue

                    if memory_id in results_map:
                        results_map[memory_id]['vector_score'] = r['similarity']
                    else:
                        # Need to fetch the memory
                        mem = self.memory_db.get_memory(memory_id)
                        if mem:
                            results_map[memory_id] = {
                                'memory': mem,
                                'fts_score': 0.0,
                                'vector_score': r['similarity'],
                            }
            except Exception:
                # Vector search failed, continue with FTS only
                pass

        # Compute combined scores
        scored_results = []
        for memory_id, data in results_map.items():
            combined_score = (
                self.fts_weight * data['fts_score'] +
                self.vector_weight * data['vector_score']
            )
            scored_results.append({
                'memory': data['memory'],
                'score': combined_score,
                'fts_score': data['fts_score'],
                'vector_score': data['vector_score'],
            })

        # Sort by combined score
        scored_results.sort(key=lambda x: x['score'], reverse=True)

        return scored_results[:k]


# =============================================================================
# CLI for testing
# =============================================================================

def main():
    """CLI for testing embeddings."""
    import argparse

    parser = argparse.ArgumentParser(description="Embedding Index CLI")
    sub = parser.add_subparsers(dest='cmd', required=True)

    # check
    sub.add_parser('check', help='Check if dependencies are available')

    # add
    p_add = sub.add_parser('add', help='Add text to index')
    p_add.add_argument('text', help='Text to add')

    # search
    p_search = sub.add_parser('search', help='Search for similar texts')
    p_search.add_argument('query', help='Search query')
    p_search.add_argument('-k', type=int, default=5, help='Number of results')

    # stats
    sub.add_parser('stats', help='Show index statistics')

    args = parser.parse_args()

    if args.cmd == 'check':
        if embeddings_available():
            print("Embedding dependencies are available")
        else:
            print(f"Embedding dependencies NOT available: {get_import_error()}")
            print("\nInstall with: pip install fastembed hnswlib numpy")
        return

    if not embeddings_available():
        print(f"Error: {get_import_error()}")
        print("Install with: pip install fastembed hnswlib numpy")
        return

    index = EmbeddingIndex()

    if args.cmd == 'add':
        eid = index.add_text(args.text)
        print(f"Added embedding {eid}")

    elif args.cmd == 'search':
        results = index.search(args.query, k=args.k)
        if results:
            print(f"Found {len(results)} results:")
            for r in results:
                print(f"  [{r['embedding_id']}] ({r['similarity']:.3f}) {r['text'][:60]}...")
        else:
            print("No results found")

    elif args.cmd == 'stats':
        stats = index.get_stats()
        print("Embedding Index Statistics:")
        for k, v in stats.items():
            print(f"  {k}: {v}")


if __name__ == '__main__':
    main()
