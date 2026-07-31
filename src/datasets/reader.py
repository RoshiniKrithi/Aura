"""Dataset Readers for Aura LLM Pipeline.

Provides memory-efficient readers for single plain text files, folder-based
multi-file datasets, and streaming chunk generators for multi-gigabyte files.
"""

from abc import ABC, abstractmethod
import io
import logging
from pathlib import Path
from typing import Generator, List, Union

from src.datasets.exceptions import DatasetReadError

logger = logging.getLogger(__name__)


class DatasetReader(ABC):
    """Abstract Base Class defining reader interfaces for all dataset types."""

    @abstractmethod
    def read_all(self) -> str:
        """Reads and concatenates entire raw text corpus into a single string.

        Returns:
            Concatenated raw corpus text.
        """
        pass

    @abstractmethod
    def read_chunks(
        self, chunk_size: int = 1024 * 1024
    ) -> Generator[str, None, None]:
        """Yields text chunks lazily for memory-efficient streaming.

        Args:
            chunk_size: Target size of each read chunk in characters.

        Yields:
            Text strings of up to chunk_size length.
        """
        pass


class TextFileReader(DatasetReader):
    """Reads a single text file into memory or streams chunks lazily.

    Time Complexity:
        O(N) character read ops where N is file character length.

    Space Complexity:
        O(N) for read_all(), O(chunk_size) for read_chunks().
    """

    def __init__(
        self, file_path: Union[Path, str], encoding: str = "utf-8"
    ) -> None:
        """Initializes single file reader.

        Args:
            file_path: Path to target text file.
            encoding: Text encoding standard (default: utf-8).
        """
        self.file_path = Path(file_path).resolve()
        self.encoding = encoding

        if not self.file_path.is_file():
            raise DatasetReadError(
                f"File not found or invalid: {self.file_path}"
            )

    def read_all(self) -> str:
        """Reads entire file into a single string."""
        try:
            with open(self.file_path, "r", encoding=self.encoding) as f:
                return f.read()
        except Exception as e:
            raise DatasetReadError(
                f"Failed to read file {self.file_path}: {str(e)}"
            ) from e

    def read_chunks(
        self, chunk_size: int = 1024 * 1024
    ) -> Generator[str, None, None]:
        """Yields text chunks of fixed character length."""
        try:
            with open(self.file_path, "r", encoding=self.encoding) as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk
        except Exception as e:
            raise DatasetReadError(
                f"Streaming read failed on file {self.file_path}: {str(e)}"
            ) from e


class FolderDatasetReader(DatasetReader):
    """Reads and aggregates text files from a directory matching glob patterns.

    Time Complexity:
        O(F * N) where F is file count and N is average file size.

    Space Complexity:
        O(F * N) for read_all(), O(chunk_size) for read_chunks().
    """

    def __init__(
        self,
        folder_path: Union[Path, str],
        pattern: str = "*.txt",
        encoding: str = "utf-8",
        separator: str = "\n\n",
    ) -> None:
        """Initializes folder dataset reader.

        Args:
            folder_path: Target directory path.
            pattern: Glob matching pattern for target files (e.g. '*.txt', '*.py').
            encoding: File encoding standard.
            separator: String separator appended between file boundaries.
        """
        self.folder_path = Path(folder_path).resolve()
        self.pattern = pattern
        self.encoding = encoding
        self.separator = separator

        if not self.folder_path.is_dir():
            raise DatasetReadError(
                f"Folder path not found or invalid: {self.folder_path}"
            )

        self.file_paths: List[Path] = sorted(
            list(self.folder_path.glob(self.pattern))
        )
        if not self.file_paths:
            raise DatasetReadError(
                f"No files matching pattern '{self.pattern}' in directory {self.folder_path}"
            )

    def read_all(self) -> str:
        """Reads and concatenates all matching files with separator."""
        contents: List[str] = []
        for fp in self.file_paths:
            try:
                with open(fp, "r", encoding=self.encoding) as f:
                    contents.append(f.read())
            except Exception as e:
                raise DatasetReadError(
                    f"Failed reading file {fp} in directory: {str(e)}"
                ) from e
        return self.separator.join(contents)

    def read_chunks(
        self, chunk_size: int = 1024 * 1024
    ) -> Generator[str, None, None]:
        """Streams chunks across all files in directory sequence."""
        for fp in self.file_paths:
            reader = TextFileReader(fp, encoding=self.encoding)
            yield from reader.read_chunks(chunk_size=chunk_size)


class StreamingReader(DatasetReader):
    """Memory-efficient streaming generator wrapping single or multiple files.

    Yields text line-by-line or chunk-by-chunk without building complete corpus in RAM.
    """

    def __init__(
        self,
        file_paths: List[Union[Path, str]],
        encoding: str = "utf-8",
    ) -> None:
        """Initializes streaming reader.

        Args:
            file_paths: List of file paths to stream sequentially.
            encoding: File encoding.
        """
        self.file_paths = [Path(fp).resolve() for fp in file_paths]
        self.encoding = encoding

    def read_all(self) -> str:
        """Caution: Loads all streamed files into memory."""
        chunks = list(self.read_chunks())
        return "".join(chunks)

    def read_chunks(
        self, chunk_size: int = 1024 * 1024
    ) -> Generator[str, None, None]:
        """Yields chunks across all registered file paths sequentially."""
        for fp in self.file_paths:
            reader = TextFileReader(fp, encoding=self.encoding)
            yield from reader.read_chunks(chunk_size=chunk_size)

    def read_lines(self) -> Generator[str, None, None]:
        """Streams corpus line-by-line across all files."""
        for fp in self.file_paths:
            try:
                with open(fp, "r", encoding=self.encoding) as f:
                    for line in f:
                        yield line
            except Exception as e:
                raise DatasetReadError(
                    f"Error streaming lines from {fp}: {str(e)}"
                ) from e
