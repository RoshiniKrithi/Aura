"""Master RAG Pipeline Orchestrator and Evaluation Runner for Aura EXP-006 RAG.

Provides RAGConfig, RAGInferenceEngine, RAGEvaluator, and RAGOrchestrator.
"""

from dataclasses import asdict, dataclass, field
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from src.datasets.conversation_formatter import PromptTemplateEngine
from src.inference.engine import InferenceEngine
from src.models.config import AuraGPTConfig
from src.models.gpt import AuraGPT
from src.rag.chunking_engine import ChunkingEngine, DocumentChunk
from src.rag.document_loader import Document, KnowledgeBaseManager
from src.rag.embedding_pipeline import EmbeddingGenerator
from src.rag.prompt_builder import CitationTracker, ContextBuilder, PromptAugmentor
from src.rag.retriever import DenseRetriever, HybridRetriever, ReRankingEngine, RetrievalResult, SparseBM25Retriever
from src.rag.vector_store import VectorDatabaseManager
from src.tokenizer.code_bpe_tokenizer import CodeBPETokenizer

logger = logging.getLogger(__name__)


@dataclass
class RAGConfig:
    """Configuration container for EXP-006 RAG Pipeline.

    Attributes:
        experiment_id: Unique string identifier for experiment.
        phase: Project hierarchy phase tag.
        seed: Random seed for reproducibility.
        device: Computation target device ("cuda", "cpu", "auto").
        model_checkpoint_path: Path to model weights checkpoint (.pt).
        knowledge_dir: Path to raw documentation and knowledge files directory.
        tokenizer_dir: Path to BPE tokenizer directory.
        vector_store_type: Vector store backend ("faiss", "memory", "chromadb").
        chunk_size: Character chunk size.
        chunk_overlap: Overlap between consecutive chunks.
        top_k_retrieval: Candidate chunks retrieved by hybrid search.
        top_k_rerank: Final re-ranked chunks injected into context window.
        embedding_dim: Dense embedding vector dimension.
        max_context_chars: Character budget for injected context.
        max_sequence_length: Model context window length L.
        d_model: Model hidden embedding size.
        n_layers: Transformer layers count.
        n_heads: Attention heads count.
        d_ff: Feed-forward dimension.
        output_dir: Output directory for indexes, logs, and benchmark summaries.
    """

    experiment_id: str = "EXP-006_RAG_v1.0"
    phase: str = "Phase 25"
    seed: int = 42
    device: str = "auto"

    model_checkpoint_path: Optional[str] = None
    knowledge_dir: str = "data/knowledge_base"
    tokenizer_dir: str = "data/tokenizer"

    vector_store_type: str = "faiss"
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k_retrieval: int = 10
    top_k_rerank: int = 3
    embedding_dim: int = 128
    max_context_chars: int = 1500

    vocab_size: int = 50260
    max_sequence_length: int = 1024
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    d_ff: int = 3072

    output_dir: str = "outputs/experiments/EXP-006_RAG_v1.0"

    def to_dict(self) -> Dict[str, Any]:
        """Converts RAGConfig to dictionary representation."""
        return asdict(self)


class RAGInferenceEngine:
    """Handles grounded text generation via InferenceEngine and PromptAugmentor."""

    def __init__(
        self,
        inference_engine: InferenceEngine,
        prompt_augmentor: PromptAugmentor,
    ) -> None:
        """Initializes RAGInferenceEngine."""
        self.inference_engine = inference_engine
        self.prompt_augmentor = prompt_augmentor
        self.template_engine = PromptTemplateEngine()

    def generate_grounded_response(
        self,
        user_query: str,
        retrieved_chunks: List[RetrievalResult],
        max_new_tokens: int = 256,
        temperature: float = 0.2,
    ) -> Tuple[str, List[CitationTracker]]:
        """Generates grounded text completion with inline citations.

        Returns:
            Tuple of (response_text, list_of_citations).
        """
        conv, citations = self.prompt_augmentor.augment_conversation(
            user_query=user_query, retrieval_results=retrieved_chunks
        )
        formatted_prompt = self.template_engine.format(conv)

        raw_gen = self.inference_engine.generate(
            prompt=formatted_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        return raw_gen, citations


class RAGOrchestrator:
    """Master orchestrator for document indexing, hybrid retrieval, re-ranking, and grounded generation."""

    def __init__(self, config: RAGConfig) -> None:
        """Initializes RAGOrchestrator."""
        self.config = config
        self.output_dir = Path(config.output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = self._resolve_device(config.device)
        self.config.device = str(self.device)

        # 1. Pipeline Components
        self.kb_manager = KnowledgeBaseManager(config.knowledge_dir)
        self.chunker = ChunkingEngine(
            chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap
        )
        self.embedder = EmbeddingGenerator(dimension=config.embedding_dim)
        self.vector_db = VectorDatabaseManager(backend_name=config.vector_store_type)

        self.dense_retriever = DenseRetriever(self.embedder, self.vector_db)
        self.sparse_retriever = SparseBM25Retriever()
        self.hybrid_retriever = HybridRetriever(
            self.dense_retriever, self.sparse_retriever
        )

        self.context_builder = ContextBuilder(max_context_chars=config.max_context_chars)
        self.prompt_augmentor = PromptAugmentor(context_builder=self.context_builder)

        # 2. Tokenizer & Model
        tokenizer_path = Path(config.tokenizer_dir) / "bpe_vocab_50257.json"
        merges_path = Path(config.tokenizer_dir) / "bpe_merges_50257.txt"

        if tokenizer_path.exists() and merges_path.exists():
            self.tokenizer = CodeBPETokenizer.from_files(tokenizer_path, merges_path)
        else:
            self.tokenizer = CodeBPETokenizer.create_default()

        gpt_cfg = AuraGPTConfig(
            model_name="aura-rag-base",
            vocab_size=max(self.tokenizer.vocab_size, config.vocab_size, 50260),
            max_sequence_length=config.max_sequence_length,
            d_model=config.d_model,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            device=str(self.device),
        )
        self.model = AuraGPT(gpt_cfg).to(self.device)

        if config.model_checkpoint_path and Path(config.model_checkpoint_path).exists():
            self._load_checkpoint(Path(config.model_checkpoint_path))

        self.inference_engine = InferenceEngine(
            model=self.model, tokenizer=self.tokenizer, device=str(self.device)
        )
        self.rag_engine = RAGInferenceEngine(
            inference_engine=self.inference_engine,
            prompt_augmentor=self.prompt_augmentor,
        )

    def _resolve_device(self, req: str) -> torch.device:
        if req == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _load_checkpoint(self, path: Path) -> None:
        logger.info("Loading checkpoint weights from %s", path)
        ckpt = torch.load(path, weights_only=False)
        if "model_state_dict" in ckpt:
            self.model.load_state_dict(ckpt["model_state_dict"])

    def build_knowledge_index(self) -> int:
        """Loads documents from knowledge_dir, chunks them, and builds vector & sparse index.

        Returns:
            Total indexed chunks count.
        """
        logger.info("Building Knowledge Base Index from %s", self.config.knowledge_dir)
        docs = self.kb_manager.scan_and_load()

        if not docs:
            # Create synthetic default document for initial verification
            docs = [
                Document(
                    doc_id="dsa_quickstart",
                    source_file="dsa_quickstart.md",
                    content="A Min-Heap is a Complete Binary Tree where the key at the root is minimum among all keys in the heap. In Python, heapq provides heappush and heappop.",
                    doc_type="markdown",
                )
            ]

        all_chunks: List[DocumentChunk] = []
        for doc in docs:
            chunks = self.chunker.chunk_document(doc)
            all_chunks.extend(chunks)

        if not all_chunks:
            return 0

        embeddings = self.embedder.embed_chunks(all_chunks)
        self.vector_db.index_chunks(all_chunks, embeddings)
        self.sparse_retriever.index_chunks(all_chunks)

        logger.info("Successfully indexed %d chunks across %d documents.", len(all_chunks), len(docs))
        return len(all_chunks)

    def query(
        self, user_query: str, max_new_tokens: int = 256
    ) -> Dict[str, Any]:
        """Processes user query through hybrid retrieval, re-ranking, and grounded generation.

        Returns:
            Dictionary containing query, answer, citations, and retrieved_chunks.
        """
        start_time = time.time()

        # 1. Hybrid Retrieval (Dense + BM25)
        raw_hits = self.hybrid_retriever.retrieve(
            user_query, top_k=self.config.top_k_retrieval
        )

        # 2. Re-Ranking
        top_chunks = ReRankingEngine.rerank(
            user_query, raw_hits, top_k=self.config.top_k_rerank
        )

        # 3. Grounded Answer Generation
        answer, citations = self.rag_engine.generate_grounded_response(
            user_query=user_query,
            retrieved_chunks=top_chunks,
            max_new_tokens=max_new_tokens,
        )

        elapsed = time.time() - start_time

        return {
            "query": user_query,
            "answer": answer,
            "citations": [asdict(c) for c in citations],
            "retrieved_chunks_count": len(top_chunks),
            "execution_time_seconds": round(elapsed, 4),
        }
