"""AST and Sliding-Window Code Chunking Engine for Aura EXP-006 RAG.

Provides DocumentChunk, ChunkMetadataManager, and ChunkingEngine.
"""

from dataclasses import dataclass, field
import hashlib
import logging
from typing import Any, Dict, List, Optional
from src.rag.document_loader import Document

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """Represents a text or code chunk extracted from a Document.

    Attributes:
        chunk_id: Unique string hash for the chunk.
        doc_id: Parent document identifier.
        source_file: Source file path.
        content: Chunk text content string.
        start_char: Starting character index in original document.
        end_char: Ending character index in original document.
        language: Programming or natural language tag ("python", "cpp", "markdown", "text").
        metadata: Dictionary of chunk metadata (tokens, headers, parent function).
    """

    chunk_id: str
    doc_id: str
    source_file: str
    content: str
    start_char: int
    end_char: int
    language: str = "text"
    metadata: Dict[str, Any] = field(default_factory=dict)


class ChunkMetadataManager:
    """Manages metadata tagging and identification for document chunks."""

    @staticmethod
    def generate_chunk_id(doc_id: str, start_char: int, content: str) -> str:
        """Generates deterministic SHA256 chunk ID based on document and offset."""
        hasher = hashlib.sha256()
        hasher.update(f"{doc_id}:{start_char}:{content[:64]}".encode("utf-8"))
        return hasher.hexdigest()[:16]


class ChunkingEngine:
    """Splits technical documents and source code into overlapping metadata-enriched chunks."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        language: str = "python",
    ) -> None:
        """Initializes ChunkingEngine.

        Args:
            chunk_size: Target size in characters per chunk (approx. tokens * 4).
            chunk_overlap: Character overlap between consecutive chunks.
            language: Default language mode ("python", "cpp", "markdown", "text").
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = max(0, min(chunk_overlap, chunk_size - 1))
        self.language = language

    def chunk_document(self, document: Document) -> List[DocumentChunk]:
        """Splits a Document into a list of DocumentChunk instances.

        Args:
            document: Document instance to chunk.

        Returns:
            List of DocumentChunk objects.
        """
        text = document.content
        if not text:
            return []

        step = self.chunk_size - self.chunk_overlap
        chunks: List[DocumentChunk] = []

        for start_idx in range(0, len(text), step):
            end_idx = min(start_idx + self.chunk_size, len(text))
            chunk_text = text[start_idx:end_idx].strip()

            if not chunk_text:
                continue

            chunk_id = ChunkMetadataManager.generate_chunk_id(
                doc_id=document.doc_id,
                start_char=start_idx,
                content=chunk_text,
            )

            chunk = DocumentChunk(
                chunk_id=chunk_id,
                doc_id=document.doc_id,
                source_file=document.source_file,
                content=chunk_text,
                start_char=start_idx,
                end_char=end_idx,
                language=document.metadata.get("extension", self.language),
                metadata={
                    **document.metadata,
                    "chunk_length": len(chunk_text),
                },
            )
            chunks.append(chunk)

        logger.debug(
            "Chunked document %s into %d chunks (chunk_size=%d, overlap=%d)",
            document.doc_id,
            len(chunks),
            self.chunk_size,
            self.chunk_overlap,
        )
        return chunks
