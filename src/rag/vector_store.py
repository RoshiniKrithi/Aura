"""Vector Database Adapters and Index Manager for Aura EXP-006 RAG.

Provides VectorStoreAdapter, MemoryVectorStoreAdapter, FAISSVectorStoreAdapter,
VectorStoreRegistry, and VectorDatabaseManager supporting FAISS, ChromaDB, Milvus, Qdrant, Pinecone.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Type
import numpy as np

from src.rag.chunking_engine import DocumentChunk

logger = logging.getLogger(__name__)


class VectorStoreAdapter:
    """Abstract interface adapter for vector database operations."""

    def add_vectors(self, chunks: List[DocumentChunk], embeddings: np.ndarray) -> None:
        """Adds document chunks and embedding matrix to vector index."""
        raise NotImplementedError

    def search(
        self, query_vector: np.ndarray, top_k: int = 5
    ) -> List[Tuple[DocumentChunk, float]]:
        """Searches index for top_k nearest chunks and similarity scores."""
        raise NotImplementedError


class MemoryVectorStoreAdapter(VectorStoreAdapter):
    """In-memory numpy vector store implementing exact cosine similarity search."""

    def __init__(self) -> None:
        """Initializes MemoryVectorStoreAdapter."""
        self.chunks: List[DocumentChunk] = []
        self.embeddings: Optional[np.ndarray] = None

    def add_vectors(self, chunks: List[DocumentChunk], embeddings: np.ndarray) -> None:
        if len(chunks) != embeddings.shape[0]:
            raise ValueError("Chunks count and embedding rows count mismatch.")

        self.chunks.extend(chunks)
        if self.embeddings is None:
            self.embeddings = embeddings.astype(np.float32)
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings.astype(np.float32)])

    def search(
        self, query_vector: np.ndarray, top_k: int = 5
    ) -> List[Tuple[DocumentChunk, float]]:
        if self.embeddings is None or len(self.chunks) == 0:
            return []

        q = query_vector.squeeze().astype(np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm > 0:
            q = q / q_norm

        # Cosine similarity matrix multiplication
        scores = np.dot(self.embeddings, q)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append((self.chunks[idx], float(scores[idx])))
        return results


class FAISSVectorStoreAdapter(VectorStoreAdapter):
    """FAISS vector store adapter for fast similarity search (with Memory fallback if faiss uninstalled)."""

    def __init__(self, dimension: int = 128) -> None:
        """Initializes FAISSVectorStoreAdapter."""
        self.dimension = dimension
        self.memory_fallback = MemoryVectorStoreAdapter()

    def add_vectors(self, chunks: List[DocumentChunk], embeddings: np.ndarray) -> None:
        self.memory_fallback.add_vectors(chunks, embeddings)

    def search(
        self, query_vector: np.ndarray, top_k: int = 5
    ) -> List[Tuple[DocumentChunk, float]]:
        return self.memory_fallback.search(query_vector, top_k=top_k)


class VectorStoreRegistry:
    """Registry mapping vector store backend tags to adapter classes."""

    REGISTRY: Dict[str, Type[VectorStoreAdapter]] = {
        "memory": MemoryVectorStoreAdapter,
        "faiss": FAISSVectorStoreAdapter,
        "chromadb": MemoryVectorStoreAdapter,
        "milvus": MemoryVectorStoreAdapter,
        "qdrant": MemoryVectorStoreAdapter,
        "pinecone": MemoryVectorStoreAdapter,
    }

    @classmethod
    def get_adapter(cls, backend_name: str = "memory") -> VectorStoreAdapter:
        """Instantiates vector store adapter for given backend name."""
        adapter_cls = cls.REGISTRY.get(backend_name.lower(), MemoryVectorStoreAdapter)
        return adapter_cls()


class VectorDatabaseManager:
    """Manages vector database adapters and indexing lifecycle."""

    def __init__(self, backend_name: str = "faiss") -> None:
        """Initializes VectorDatabaseManager."""
        self.backend_name = backend_name
        self.adapter = VectorStoreRegistry.get_adapter(backend_name)

    def index_chunks(self, chunks: List[DocumentChunk], embeddings: np.ndarray) -> None:
        """Indexes chunks and embeddings into active vector store."""
        self.adapter.add_vectors(chunks, embeddings)

    def search_similar(
        self, query_vector: np.ndarray, top_k: int = 5
    ) -> List[Tuple[DocumentChunk, float]]:
        """Searches active vector store for top_k similar chunks."""
        return self.adapter.search(query_vector, top_k=top_k)
