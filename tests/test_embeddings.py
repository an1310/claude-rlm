"""Tests for embeddings module.

These tests work whether or not the embedding dependencies are installed.
When not installed, tests verify graceful fallback behavior.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / ".claude" / "skills" / "rlm" / "scripts"))

from embeddings import embeddings_available, get_import_error


def test_availability_check():
    """Test that availability check works."""
    available = embeddings_available()
    assert isinstance(available, bool)

    if not available:
        error = get_import_error()
        assert error is not None
        assert isinstance(error, str)


def test_embedding_index_import_error():
    """Test EmbeddingIndex raises helpful error when deps missing."""
    if embeddings_available():
        # Dependencies available, skip this test
        return

    from embeddings import EmbeddingIndex

    try:
        index = EmbeddingIndex()
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "not available" in str(e).lower()
        assert "pip install" in str(e).lower()


def test_embedding_index_basic():
    """Test basic EmbeddingIndex operations."""
    if not embeddings_available():
        print("  (skipped - dependencies not installed)")
        return

    from embeddings import EmbeddingIndex

    with tempfile.TemporaryDirectory() as tmp:
        index_path = Path(tmp) / "test.hnsw"
        meta_path = Path(tmp) / "test_meta.json"

        index = EmbeddingIndex(index_path=index_path, metadata_path=meta_path)

        # Add text
        eid = index.add_text("Hello, this is a test document about Python programming.")
        assert eid >= 0

        # Search
        results = index.search("Python code", k=5)
        assert len(results) >= 1
        assert results[0]['embedding_id'] == eid

        # Stats
        stats = index.get_stats()
        assert stats['total_embeddings'] >= 1


def test_embedding_index_batch():
    """Test batch operations."""
    if not embeddings_available():
        print("  (skipped - dependencies not installed)")
        return

    from embeddings import EmbeddingIndex

    with tempfile.TemporaryDirectory() as tmp:
        index_path = Path(tmp) / "test.hnsw"
        meta_path = Path(tmp) / "test_meta.json"

        index = EmbeddingIndex(index_path=index_path, metadata_path=meta_path)

        # Add batch
        texts = [
            "Python is a programming language",
            "JavaScript runs in browsers",
            "Rust is memory safe",
            "Go has goroutines",
        ]
        eids = index.add_texts(texts)
        assert len(eids) == 4

        # Search should find relevant
        results = index.search("memory safety", k=2)
        assert len(results) >= 1
        # Rust should be most relevant
        assert any("Rust" in r.get('text', '') for r in results)


def test_embedding_index_persistence():
    """Test that index persists across instances."""
    if not embeddings_available():
        print("  (skipped - dependencies not installed)")
        return

    from embeddings import EmbeddingIndex

    with tempfile.TemporaryDirectory() as tmp:
        index_path = Path(tmp) / "test.hnsw"
        meta_path = Path(tmp) / "test_meta.json"

        # Create and add
        index1 = EmbeddingIndex(index_path=index_path, metadata_path=meta_path)
        eid = index1.add_text("Persistent test data")
        del index1

        # Reload
        index2 = EmbeddingIndex(index_path=index_path, metadata_path=meta_path)
        stats = index2.get_stats()
        assert stats['total_embeddings'] >= 1

        # Search should still work
        results = index2.search("persistent", k=5)
        assert len(results) >= 1


def test_embedding_index_metadata():
    """Test metadata storage and retrieval."""
    if not embeddings_available():
        print("  (skipped - dependencies not installed)")
        return

    from embeddings import EmbeddingIndex

    with tempfile.TemporaryDirectory() as tmp:
        index_path = Path(tmp) / "test.hnsw"
        meta_path = Path(tmp) / "test_meta.json"

        index = EmbeddingIndex(index_path=index_path, metadata_path=meta_path)

        # Add with metadata
        eid = index.add_text(
            "Test with metadata",
            metadata={'source': 'test', 'importance': 0.8}
        )

        # Retrieve metadata
        meta = index.get_metadata(eid)
        assert meta is not None
        assert meta['source'] == 'test'
        assert meta['importance'] == 0.8

        # Update metadata
        index.update_metadata(eid, {'importance': 0.9})
        meta = index.get_metadata(eid)
        assert meta['importance'] == 0.9


def test_embedding_index_delete():
    """Test deletion."""
    if not embeddings_available():
        print("  (skipped - dependencies not installed)")
        return

    from embeddings import EmbeddingIndex

    with tempfile.TemporaryDirectory() as tmp:
        index_path = Path(tmp) / "test.hnsw"
        meta_path = Path(tmp) / "test_meta.json"

        index = EmbeddingIndex(index_path=index_path, metadata_path=meta_path)

        eid = index.add_text("To be deleted")
        assert index.get_metadata(eid) is not None

        # Delete
        success = index.delete(eid)
        assert success

        # Metadata should be gone
        assert index.get_metadata(eid) is None


def test_hybrid_search_fallback():
    """Test HybridSearch falls back gracefully."""
    from embeddings import HybridSearch
    from memory_db import MemoryDatabase, MemoryType

    with tempfile.TemporaryDirectory() as tmp:
        import os
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp)

            db = MemoryDatabase()
            db.add_memory("Test memory about databases", memory_type=MemoryType.FACT)
            db.add_memory("Test memory about Python", memory_type=MemoryType.FACT)

            # Create hybrid search without embeddings
            hybrid = HybridSearch(db, embedding_index=None)

            # Should work with FTS only
            results = hybrid.search("databases", k=5, use_vectors=False)
            assert len(results) >= 1
            assert "database" in results[0]['memory'].content.lower()

            db.close()

        finally:
            os.chdir(old_cwd)


def test_hybrid_search_with_embeddings():
    """Test HybridSearch with embeddings."""
    if not embeddings_available():
        print("  (skipped - dependencies not installed)")
        return

    from embeddings import HybridSearch, EmbeddingIndex
    from memory_db import MemoryDatabase, MemoryType

    with tempfile.TemporaryDirectory() as tmp:
        import os
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp)

            db = MemoryDatabase()
            index = EmbeddingIndex()

            # Add memories and embeddings
            mem1_id = db.add_memory("PostgreSQL is used for data storage", memory_type=MemoryType.FACT)
            mem2_id = db.add_memory("React handles the user interface", memory_type=MemoryType.FACT)

            index.add_text("PostgreSQL is used for data storage", metadata={'memory_id': mem1_id})
            index.add_text("React handles the user interface", metadata={'memory_id': mem2_id})

            # Hybrid search
            hybrid = HybridSearch(db, embedding_index=index)
            results = hybrid.search("database storage")

            assert len(results) >= 1
            # Should find PostgreSQL memory
            contents = [r['memory'].content.lower() for r in results]
            assert any('postgresql' in c or 'database' in c for c in contents)

            db.close()

        finally:
            os.chdir(old_cwd)


def test_get_embedding():
    """Test getting raw embedding vector."""
    if not embeddings_available():
        print("  (skipped - dependencies not installed)")
        return

    from embeddings import EmbeddingIndex

    with tempfile.TemporaryDirectory() as tmp:
        index = EmbeddingIndex(
            index_path=Path(tmp) / "test.hnsw",
            metadata_path=Path(tmp) / "meta.json"
        )

        embedding = index.get_embedding("Test text")

        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)


def run_tests():
    """Run all embedding tests."""
    print("=" * 60)
    print("Testing Embeddings")
    print("=" * 60)

    if embeddings_available():
        print("  (Dependencies installed - running full tests)")
    else:
        print(f"  (Dependencies NOT installed: {get_import_error()})")
        print("  (Running fallback tests only)")

    tests = [
        ("Availability check", test_availability_check),
        ("Import error handling", test_embedding_index_import_error),
        ("Basic operations", test_embedding_index_basic),
        ("Batch operations", test_embedding_index_batch),
        ("Persistence", test_embedding_index_persistence),
        ("Metadata", test_embedding_index_metadata),
        ("Delete", test_embedding_index_delete),
        ("Hybrid search fallback", test_hybrid_search_fallback),
        ("Hybrid search with embeddings", test_hybrid_search_with_embeddings),
        ("Get embedding", test_get_embedding),
    ]

    passed = 0
    failed = 0
    skipped = 0

    for name, test_fn in tests:
        try:
            test_fn()
            print(f"  ✓ {name}")
            passed += 1
        except AssertionError as e:
            if "skipped" in str(e).lower():
                skipped += 1
            else:
                print(f"  ✗ {name}: {e}")
                failed += 1
        except Exception as e:
            print(f"  ✗ {name}: {type(e).__name__}: {e}")
            failed += 1

    print(f"\nEmbedding tests: {passed} passed, {failed} failed, {skipped} skipped")
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
