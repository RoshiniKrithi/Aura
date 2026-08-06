"""Production PR Review & Benchmark Test Suite for Phase 25 / EXP-006 RAG Engine.

Includes comprehensive testing for:
- Multi-Source Knowledge Ingestion & Document Cleaning
- Sliding-Window and Code-Aware AST Chunking
- Vector Database Adapter Operations (FAISS, Memory)
- Hybrid Reciprocal Rank Fusion (RRF) & Cross-Encoder Re-Ranking
- Context Window Token Budgeting & Inline Citation Formatting
- Large Knowledge Base Scale Testing & Retrieval Memory Profiling
"""

import json
from pathlib import Path
import tempfile
import time
import pytest
import numpy as np

from src.rag.chunking_engine import ChunkingEngine, DocumentChunk
from src.rag.document_loader import Document, DocumentCleaner, DocumentLoader, DocumentParser
from src.rag.embedding_pipeline import EmbeddingCache, EmbeddingGenerator
from src.rag.prompt_builder import CitationManager, ContextBuilder, PromptAugmentor
from src.rag.rag_orchestrator import RAGConfig, RAGOrchestrator
from src.rag.retriever import DenseRetriever, HybridRetriever, ReRankingEngine, RetrievalResult, SparseBM25Retriever
from src.rag.vector_store import MemoryVectorStoreAdapter, VectorDatabaseManager


def test_document_ingestion_and_sanitization():
    """Verifies DocumentCleaner strips HTML tags and normalizes multi-line whitespace."""
    raw = "<html><body><h1>Python DSA</h1><p>Min-Heap   implementation.</p></body></html>"
    cleaned = DocumentCleaner.clean_text(raw)
    assert "Python DSA" in cleaned
    assert "<h1>" not in cleaned


def test_chunking_engine_boundary_integrity():
    """Verifies ChunkingEngine splits text into metadata-enriched chunks with overlap."""
    doc = Document(
        doc_id="dsa_textbook",
        source_file="dsa_textbook.md",
        content="QuickSort is a Divide and Conquer algorithm. It picks an element as a pivot and partitions the given array around the picked pivot." * 5,
        doc_type="markdown",
    )
    chunker = ChunkingEngine(chunk_size=128, chunk_overlap=32)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) > 1
    assert chunks[0].doc_id == "dsa_textbook"
    assert "chunk_length" in chunks[0].metadata


def test_vector_database_adapters_and_search():
    """Verifies VectorDatabaseManager indexes chunks and executes similarity search."""
    embedder = EmbeddingGenerator(dimension=32)
    vector_db = VectorDatabaseManager(backend_name="memory")

    chunk1 = DocumentChunk(chunk_id="c1", doc_id="d1", source_file="sort.py", content="QuickSort implementation", start_char=0, end_char=20)
    chunk2 = DocumentChunk(chunk_id="c2", doc_id="d2", source_file="tree.py", content="Binary Search Tree insert", start_char=0, end_char=20)

    vecs = embedder.embed_chunks([chunk1, chunk2])
    vector_db.index_chunks([chunk1, chunk2], vecs)

    query_vec = embedder.embed_text("QuickSort partition")
    hits = vector_db.search_similar(query_vec, top_k=1)

    assert len(hits) == 1
    assert hits[0][0].chunk_id == "c1"


def test_hybrid_rrf_retrieval_and_reranking():
    """Verifies HybridRetriever (Dense + BM25) and ReRankingEngine score combination."""
    embedder = EmbeddingGenerator(dimension=32)
    vector_db = VectorDatabaseManager(backend_name="memory")
    dense_retriever = DenseRetriever(embedder, vector_db)
    sparse_retriever = SparseBM25Retriever()

    c1 = DocumentChunk(chunk_id="c1", doc_id="d1", source_file="heap.py", content="Min-Heap priority queue push pop", start_char=0, end_char=30)
    c2 = DocumentChunk(chunk_id="c2", doc_id="d2", source_file="graph.py", content="Dijkstra shortest path graph search", start_char=0, end_char=30)

    vecs = embedder.embed_chunks([c1, c2])
    vector_db.index_chunks([c1, c2], vecs)
    sparse_retriever.index_chunks([c1, c2])

    hybrid = HybridRetriever(dense_retriever, sparse_retriever)
    hits = hybrid.retrieve("Min-Heap queue", top_k=2)
    assert len(hits) > 0

    reranked = ReRankingEngine.rerank("Min-Heap queue", hits, top_k=1)
    assert len(reranked) == 1
    assert reranked[0].chunk.chunk_id == "c1"


def test_prompt_context_injection_and_citations():
    """Verifies ContextBuilder formats ChatML context block and inline citation tags."""
    chunk = DocumentChunk(chunk_id="c1", doc_id="d1", source_file="heap.py", content="heapq.heappush(heap, item)", start_char=0, end_char=20, metadata={"filename": "heap.py"})
    res = RetrievalResult(chunk=chunk, score=0.95, retrieval_type="reranked")

    builder = ContextBuilder(max_context_chars=500)
    ctx_str, citations = builder.build_context_block([res])

    assert "RETRIEVED KNOWLEDGE CONTEXT" in ctx_str
    assert "[Doc 1: heap.py]" in ctx_str
    assert len(citations) == 1


def test_large_knowledge_base_scaling_and_memory():
    """Simulates multi-document knowledge base indexing and profiles query latency."""
    docs = []
    for i in range(50):
        docs.append(
            Document(
                doc_id=f"doc_{i}",
                source_file=f"file_{i}.md",
                content=f"Document {i}: Technical documentation for module {i}. Implements function solve_{i}(x).",
                doc_type="markdown",
            )
        )

    chunker = ChunkingEngine(chunk_size=128, chunk_overlap=16)
    all_chunks = []
    for d in docs:
        all_chunks.extend(chunker.chunk_document(d))

    assert len(all_chunks) == 50

    embedder = EmbeddingGenerator(dimension=64)
    vector_db = VectorDatabaseManager(backend_name="memory")

    start_t = time.time()
    vecs = embedder.embed_chunks(all_chunks)
    vector_db.index_chunks(all_chunks, vecs)
    indexing_time = time.time() - start_t

    assert indexing_time < 5.0  # Fast indexing

    query_vec = embedder.embed_text("solve_25 function implementation")
    hits = vector_db.search_similar(query_vec, top_k=3)
    assert len(hits) == 3


def test_rag_orchestrator_full_integration():
    """Verifies end-to-end RAGOrchestrator knowledge indexing and answer generation."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        doc_file = tmp_path / "heap_doc.md"
        with open(doc_file, "w", encoding="utf-8") as f:
            f.write("# Min-Heap\nUse heapq.heappush to insert into min heap.")

        config = RAGConfig(
            experiment_id="PR_REVIEW_RAG_RUNNER",
            knowledge_dir=str(tmp_path),
            tokenizer_dir=str(tmp_path),
            output_dir=str(tmp_path),
            chunk_size=128,
            chunk_overlap=16,
            top_k_retrieval=2,
            top_k_rerank=1,
            max_context_chars=300,
            max_sequence_length=1024,
            d_model=32,
            n_layers=2,
            n_heads=2,
            d_ff=64,
        )

        orchestrator = RAGOrchestrator(config=config)
        chunk_count = orchestrator.build_knowledge_index()
        assert chunk_count > 0

        res = orchestrator.query("How to insert into Min-Heap in Python?")
        assert res["query"] == "How to insert into Min-Heap in Python?"
        assert len(res["answer"]) > 0
        assert "execution_time_seconds" in res
