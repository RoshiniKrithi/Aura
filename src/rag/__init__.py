"""Retrieval-Augmented Generation (RAG) Module for Aura LLM Architecture.

Provides Document, DocumentCleaner, DocumentParser, DocumentLoader, KnowledgeBaseManager,
DocumentChunk, ChunkMetadataManager, ChunkingEngine, EmbeddingGenerator, EmbeddingCache,
EmbeddingStore, VectorStoreAdapter, MemoryVectorStoreAdapter, FAISSVectorStoreAdapter,
VectorDatabaseManager, RetrievalResult, DenseRetriever, SparseBM25Retriever, HybridRetriever,
ReRankingEngine, CitationTracker, CitationManager, ContextBuilder, PromptAugmentor,
RAGConfig, RAGInferenceEngine, and RAGOrchestrator.
"""

from src.rag.chunking_engine import (
    ChunkMetadataManager,
    ChunkingEngine,
    DocumentChunk,
)
from src.rag.document_loader import (
    Document,
    DocumentCleaner,
    DocumentLoader,
    DocumentParser,
    KnowledgeBaseManager,
)
from src.rag.embedding_pipeline import (
    EmbeddingCache,
    EmbeddingGenerator,
    EmbeddingStatistics,
    EmbeddingStore,
)
from src.rag.prompt_builder import (
    CitationManager,
    CitationTracker,
    ContextBuilder,
    PromptAugmentor,
)
from src.rag.rag_orchestrator import (
    RAGConfig,
    RAGInferenceEngine,
    RAGOrchestrator,
)
from src.rag.retriever import (
    DenseRetriever,
    HybridRetriever,
    ReRankingEngine,
    RetrievalResult,
    RetrievalStatistics,
    SparseBM25Retriever,
)
from src.rag.vector_store import (
    FAISSVectorStoreAdapter,
    MemoryVectorStoreAdapter,
    VectorDatabaseManager,
    VectorStoreAdapter,
    VectorStoreRegistry,
)

__all__ = [
    "Document",
    "DocumentCleaner",
    "DocumentParser",
    "DocumentLoader",
    "KnowledgeBaseManager",
    "DocumentChunk",
    "ChunkMetadataManager",
    "ChunkingEngine",
    "EmbeddingGenerator",
    "EmbeddingCache",
    "EmbeddingStore",
    "EmbeddingStatistics",
    "VectorStoreAdapter",
    "MemoryVectorStoreAdapter",
    "FAISSVectorStoreAdapter",
    "VectorStoreRegistry",
    "VectorDatabaseManager",
    "RetrievalResult",
    "RetrievalStatistics",
    "DenseRetriever",
    "SparseBM25Retriever",
    "HybridRetriever",
    "ReRankingEngine",
    "CitationTracker",
    "CitationManager",
    "ContextBuilder",
    "PromptAugmentor",
    "RAGConfig",
    "RAGInferenceEngine",
    "RAGOrchestrator",
]
