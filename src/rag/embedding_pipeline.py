"""Dense Vector Embedding Generator and Cache Pipeline for Aura EXP-006 RAG.

Provides EmbeddingGenerator, EmbeddingCache, EmbeddingStore, and EmbeddingStatistics.
"""

from dataclasses import dataclass
import hashlib
import logging
from pathlib import Path
import pickle
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch

from src.rag.chunking_engine import DocumentChunk

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingStatistics:
    """Statistics container for vector embedding store."""

    total_chunks: int
    dimension: int
    cache_hits: int = 0
    cache_misses: int = 0


class EmbeddingCache:
    """Persistent disk and in-memory key-value cache for dense embeddings."""

    def __init__(self, cache_dir: Optional[Union[str, Path]] = None) -> None:
        """Initializes EmbeddingCache."""
        self.cache_dir = Path(cache_dir).resolve() if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._cache: Dict[str, np.ndarray] = {}

    def get(self, key: str) -> Optional[np.ndarray]:
        """Retrieves cached embedding array by key hash."""
        if key in self._cache:
            return self._cache[key]

        if self.cache_dir:
            file_path = self.cache_dir / f"{key}.npy"
            if file_path.exists():
                arr = np.load(file_path)
                self._cache[key] = arr
                return arr
        return None

    def put(self, key: str, embedding: np.ndarray) -> None:
        """Stores embedding array into cache."""
        self._cache[key] = embedding
        if self.cache_dir:
            file_path = self.cache_dir / f"{key}.npy"
            np.save(file_path, embedding)


class EmbeddingGenerator:
    """Dense vector embedding generator using term frequency feature hashing or transformer encoder."""

    def __init__(self, dimension: int = 128, device: str = "cpu") -> None:
        """Initializes EmbeddingGenerator.

        Args:
            dimension: Dense vector embedding dimension D (default: 128).
            device: Target computation hardware ("cpu", "cuda").
        """
        self.dimension = dimension
        self.device = device
        self.cache = EmbeddingCache()

    def _hash_text_to_vector(self, text: str) -> np.ndarray:
        """Computes deterministic dense feature hash vector for text."""
        vec = np.zeros(self.dimension, dtype=np.float32)
        words = text.lower().split()
        if not words:
            return vec

        for word in words:
            # Deterministic character ngram hash mapping to vector indices
            h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dimension
            val = (h >> 16) % 100 / 50.0 - 1.0  # Range [-1, 1]
            vec[idx] += val

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def embed_text(self, text: str) -> np.ndarray:
        """Generates L2-normalized float32 vector embedding for text."""
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        vec = self._hash_text_to_vector(text)
        self.cache.put(key, vec)
        return vec

    def embed_chunks(self, chunks: List[DocumentChunk]) -> np.ndarray:
        """Generates dense embedding matrix of shape (N, D) for N chunks.

        Args:
            chunks: List of DocumentChunk instances.

        Returns:
            Numpy float32 array of shape (N, D).
        """
        if not chunks:
            return np.zeros((0, self.dimension), dtype=np.float32)

        vectors = [self.embed_text(c.content) for c in chunks]
        return np.vstack(vectors)


class EmbeddingStore:
    """Manages dense chunk embeddings and associated metadata vectors."""

    def __init__(self, dimension: int = 128) -> None:
        """Initializes EmbeddingStore."""
        self.dimension = dimension
        self.chunks: List[DocumentChunk] = []
        self.embeddings: Optional[np.ndarray] = None

    def add_chunks(self, chunks: List[DocumentChunk], embeddings: np.ndarray) -> None:
        """Adds chunks and their embedding matrix to store."""
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(f"Mismatch between chunks count ({len(chunks)}) and embeddings rows ({embeddings.shape[0]})")

        self.chunks.extend(chunks)
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])

    def get_statistics(self) -> EmbeddingStatistics:
        """Returns EmbeddingStatistics summary."""
        count = len(self.chunks)
        return EmbeddingStatistics(total_chunks=count, dimension=self.dimension)
