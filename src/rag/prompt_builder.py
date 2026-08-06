"""RAG Context Injection, Citation Manager, and Prompt Augmentor for Aura EXP-006 RAG.

Provides CitationTracker, CitationManager, ContextBuilder, and PromptAugmentor.
"""

from dataclasses import dataclass, field
import logging
from typing import List, Optional, Tuple

from src.datasets.conversation_formatter import Conversation, Message
from src.rag.retriever import RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class CitationTracker:
    """Tracks inline document citation mappings."""

    doc_index: int
    doc_title: str
    source_file: str
    chunk_id: str


class CitationManager:
    """Manages citation formatting and ground-truth link creation."""

    @staticmethod
    def format_citation(index: int, filename: str) -> str:
        """Formats inline citation tag (e.g. [Doc 1: filename.py])."""
        return f"[Doc {index}: {filename}]"


class ContextBuilder:
    """Builds structured context blocks from retrieved chunks within token budget limits."""

    def __init__(self, max_context_chars: int = 1500) -> None:
        """Initializes ContextBuilder.

        Args:
            max_context_chars: Maximum total characters allocated for context injection.
        """
        self.max_context_chars = max_context_chars

    def build_context_block(
        self, retrieval_results: List[RetrievalResult]
    ) -> Tuple[str, List[CitationTracker]]:
        """Formats retrieved chunks into a single context string with citation markers.

        Returns:
            Tuple of (formatted_context_text, list_of_CitationTrackers).
        """
        if not retrieval_results:
            return "", []

        context_lines: List[str] = ["=== RETRIEVED KNOWLEDGE CONTEXT ==="]
        citations: List[CitationTracker] = []
        total_chars = len(context_lines[0])

        for idx, res in enumerate(retrieval_results, start=1):
            filename = res.chunk.metadata.get("filename", res.chunk.doc_id)
            cite_tag = CitationManager.format_citation(idx, filename)
            header = f"\n--- {cite_tag} (Source: {res.chunk.source_file}) ---"

            chunk_text = res.chunk.content
            entry = f"{header}\n{chunk_text}\n"

            if total_chars + len(entry) > self.max_context_chars:
                # Truncate context if budget exceeded
                budget_rem = self.max_context_chars - total_chars - len(header) - 10
                if budget_rem > 50:
                    entry = f"{header}\n{chunk_text[:budget_rem]}...\n"
                    context_lines.append(entry)
                    citations.append(
                        CitationTracker(
                            doc_index=idx,
                            doc_title=filename,
                            source_file=res.chunk.source_file,
                            chunk_id=res.chunk.chunk_id,
                        )
                    )
                break

            context_lines.append(entry)
            citations.append(
                CitationTracker(
                    doc_index=idx,
                    doc_title=filename,
                    source_file=res.chunk.source_file,
                    chunk_id=res.chunk.chunk_id,
                )
            )
            total_chars += len(entry)

        context_lines.append("=== END KNOWLEDGE CONTEXT ===\n")
        return "\n".join(context_lines), citations


class PromptAugmentor:
    """Augments user queries with retrieved context into ChatML Conversation templates."""

    def __init__(self, context_builder: Optional[ContextBuilder] = None) -> None:
        """Initializes PromptAugmentor."""
        self.context_builder = context_builder or ContextBuilder()

    def augment_conversation(
        self,
        user_query: str,
        retrieval_results: List[RetrievalResult],
        system_prompt: Optional[str] = None,
    ) -> Tuple[Conversation, List[CitationTracker]]:
        """Constructs a context-injected Conversation object.

        Returns:
            Tuple of (augmented_Conversation, citations_list).
        """
        context_str, citations = self.context_builder.build_context_block(retrieval_results)

        if context_str:
            augmented_query = (
                f"{context_str}\n"
                f"Question: {user_query}\n\n"
                f"Instructions: Answer the question using the retrieved knowledge context above. "
                f"Cite sources using [Doc N] format."
            )
        else:
            augmented_query = user_query

        conv = Conversation(
            messages=[Message(role="user", content=augmented_query)],
            system_prompt=system_prompt,
        )
        return conv, citations
