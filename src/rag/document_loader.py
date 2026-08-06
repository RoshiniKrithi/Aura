"""Document Loader and Cleaning Pipeline for Aura EXP-006 RAG.

Provides Document, DocumentCleaner, DocumentParser, DocumentLoader, and KnowledgeBaseManager.
"""

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Represents a raw or parsed technical document.

    Attributes:
        doc_id: Unique string identifier for the document.
        source_file: Absolute or relative file path.
        content: Raw or cleaned text content string.
        doc_type: Document format string ("markdown", "txt", "code", "json", "html").
        metadata: Dictionary metadata (file size, language, author, title).
    """

    doc_id: str
    source_file: str
    content: str
    doc_type: str = "txt"
    metadata: Dict[str, Any] = field(default_factory=dict)


class DocumentCleaner:
    """Sanitizes text and normalizes whitespace in technical documentation."""

    @staticmethod
    def clean_text(raw_text: str) -> str:
        """Strips HTML tags, converts non-standard spaces, and normalizes newlines."""
        if not raw_text:
            return ""

        # Remove HTML tags if present
        cleaned = re.sub(r"<[^>]+>", " ", raw_text)
        # Normalize non-breaking spaces
        cleaned = cleaned.replace("\xa0", " ")
        # Collapse multiple blank lines
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()


class DocumentParser:
    """Parses various file types (Markdown, TXT, JSON, Code, HTML) into Document objects."""

    @classmethod
    def parse_file(cls, file_path: Union[str, Path]) -> Optional[Document]:
        """Parses a file into a Document container object."""
        path = Path(file_path).resolve()
        if not path.exists() or not path.is_file():
            logger.warning("File does not exist or is not a valid file: %s", path)
            return None

        doc_id = path.stem
        ext = path.suffix.lower()
        doc_type = "txt"

        if ext in [".md", ".markdown"]:
            doc_type = "markdown"
        elif ext in [".py", ".cpp", ".java", ".rs", ".go", ".js", ".ts", ".c", ".cs", ".sql"]:
            doc_type = "code"
        elif ext == ".json":
            doc_type = "json"
        elif ext in [".html", ".htm"]:
            doc_type = "html"

        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                raw_content = f.read()

            cleaned_content = DocumentCleaner.clean_text(raw_content)

            return Document(
                doc_id=doc_id,
                source_file=str(path),
                content=cleaned_content,
                doc_type=doc_type,
                metadata={
                    "filename": path.name,
                    "extension": ext,
                    "file_size": path.stat().st_size,
                },
            )
        except Exception as e:
            logger.error("Failed to parse file %s: %s", path, e)
            return None


class DocumentLoader:
    """Loads documents from files or directories recursively."""

    @staticmethod
    def load_directory(
        dir_path: Union[str, Path],
        extensions: Optional[List[str]] = None,
    ) -> List[Document]:
        """Recursively loads all documents from a directory matching extensions."""
        path = Path(dir_path).resolve()
        if not path.exists() or not path.is_dir():
            logger.warning("Directory path does not exist: %s", path)
            return []

        ext_set = set(e.lower() for e in extensions) if extensions else None
        documents: List[Document] = []

        for p in path.rglob("*"):
            if p.is_file():
                if ext_set is None or p.suffix.lower() in ext_set:
                    doc = DocumentParser.parse_file(p)
                    if doc and doc.content:
                        documents.append(doc)

        logger.info("Loaded %d documents from directory %s", len(documents), path.name)
        return documents


class KnowledgeBaseManager:
    """Manages knowledge base document collections."""

    def __init__(self, knowledge_dir: Union[str, Path]) -> None:
        """Initializes KnowledgeBaseManager."""
        self.knowledge_dir = Path(knowledge_dir).resolve()
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)
        self.documents: Dict[str, Document] = {}

    def load_knowledge_base() -> List[Document]:
        pass

    def scan_and_load(self) -> List[Document]:
        """Scans knowledge directory and loads all supported documents."""
        docs = DocumentLoader.load_directory(self.knowledge_dir)
        for doc in docs:
            self.documents[doc.doc_id] = doc
        return list(self.documents.values())
