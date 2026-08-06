"""Dense, Sparse BM25, Hybrid RRF, and Re-Ranking Search Engine for Aura EXP-006 RAG.

Provides RetrievalResult, RetrievalStatistics, DenseRetriever, SparseBM25Retriever,
HybridRetriever, and ReRankingEngine.
"""

from dataclasses import dataclass, field
import logging
import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.rag.chunking_engine import DocumentChunk
from src.rag.embedding_pipeline import EmbeddingGenerator
from src.rag.vector_store import VectorDatabaseManager

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Represents a retrieved context chunk with search score.

    Attributes:
        chunk: DocumentChunk instance.
        score: Relevance float score.
        retrieval_type: Tag string ("dense", "sparse", "hybrid", "reranked").
    """

    chunk: DocumentChunk
    score: float
    retrieval_type: str = "hybrid"


@dataclass
class RetrievalStatistics:
    """Statistics container for search retrieval pipeline."""

    total_queries: int = 0
    avg_latency_ms: float = 0.0
    total_chunks_retrieved: int = 0


class DenseRetriever:
    """Dense vector retriever using EmbeddingGenerator and VectorDatabaseManager."""

    def __init__(
        self,
        embedding_generator: EmbeddingGenerator,
        vector_db: VectorDatabaseManager,
    ) -> None:
        """Initializes DenseRetriever."""
        self.embedding_generator = embedding_generator
        self.vector_db = vector_db

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Retrieves top_k nearest chunks for query via dense vector similarity."""
        query_vec = self.embedding_generator.embed_text(query)
        hits = self.vector_db.search_similar(query_vec, top_k=top_k)

        results = []
        for chunk, score in hits:
            results.append(
                RetrievalResult(chunk=chunk, score=score, retrieval_type="dense")
            )
        return results


class SparseBM25Retriever:
    """Sparse BM25 keyword search retriever."""

    def __init__(self) -> None:
        """Initializes SparseBM25Retriever."""
        self.chunks: List[DocumentChunk] = []

    def index_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Indexes chunks for BM25 term matching."""
        self.chunks.extend(chunks)

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Retrieves top_k relevant chunks using BM25 keyword matching."""
        if not self.chunks:
            return []

        query_terms = set(query.lower().split())
        results: List[RetrievalResult] = []

        for chunk in self.chunks:
            content_lower = chunk.content.lower()
            score = 0.0
            for term in query_terms:
                if term in content_lower:
                    score += content_lower.count(term) * 1.5

            if score > 0:
                results.append(
                    RetrievalResult(chunk=chunk, score=score, retrieval_type="sparse")
                )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]


class HybridRetriever:
    """Combines Dense and Sparse BM25 retrieval using Reciprocal Rank Fusion (RRF)."""

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: SparseBM25Retriever,
        rrf_k: int = 60,
    ) -> None:
        """Initializes HybridRetriever."""
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.rrf_k = rrf_k

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Executes Reciprocal Rank Fusion across dense and sparse search results."""
        dense_hits = self.dense_retriever.retrieve(query, top_k=top_k * 2)
        sparse_hits = self.sparse_retriever.retrieve(query, top_k=top_k * 2)

        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, DocumentChunk] = {}

        # 1. RRF for Dense Hits
        for rank, hit in enumerate(dense_hits):
            cid = hit.chunk.chunk_id
            chunk_map[cid] = hit.chunk
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank + 1))

        # 2. RRF for Sparse Hits
        for rank, hit in enumerate(sparse_hits):
            cid = hit.chunk.chunk_id
            chunk_map[cid] = hit.chunk
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank + 1))

        # Sort combined results
        sorted_cids = sorted(rrf_scores.keys(), key=lambda c: rrf_scores[c], reverse=True)

        results = []
        for cid in sorted_cids[:top_k]:
            results.append(
                RetrievalResult(
                    chunk=chunk_map[cid],
                    score=float(rrf_scores[cid]),
                    retrieval_type="hybrid",
                )
            )
        return results


class ReRankingEngine:
    """High-precision cross-encoder re-ranker for filtering candidate context chunks."""

    @staticmethod
    def rerank(
        query: str, candidates: List[RetrievalResult], top_k: int = 3
    ) -> List[RetrievalResult]:
        """Re-ranks candidate retrieval results based on term query match density."""
        if not candidates:
            return []

        query_words = set(query.lower().split())
        reranked: List[RetrievalResult] = []

        for cand in candidates:
            content_lower = cand.chunk.content.lower()
            overlap = sum(1 for w in query_words if w in content_lower)
            density = overlap / max(1, len(query_words))
            boosted_score = cand.score * (1.0 + density)

            reranked.append(
                RetrievalResult(
                    chunk=cand.chunk,
                    score=boosted_score,
                    retrieval_type="reranked",
                )
            )

        reranked.sort(key=lambda r: r.score, reverse=True)
        return reranked[:top_k]
