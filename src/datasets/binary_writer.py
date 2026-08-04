"""High-Performance Binary Shard Writer for Aura Data Pipeline.

Writes contiguous uint16 / uint32 token ID streams directly to memory-mapped binary (.bin) files
for zero-copy PyTorch DataLoader ingestion.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np

logger = logging.getLogger(__name__)


class BinaryDatasetWriter:
    """Writes token ID integer sequences into contiguous uint16 / uint32 binary disk shards.

    Design Decisions:
        - Memory footprint optimization: uint16 array representation reduces disk size by 75%
          compared to 64-bit PyTorch LongTensor disk dumps.
        - Automatic shard splitting when shard size exceeds `max_shard_size_bytes` (default 1GB).
        - Atomic shard metadata JSON logging for deterministic experiment tracking and crash recovery.
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
        shard_prefix: str = "train",
        max_shard_size_bytes: int = 1073741824,  # 1 GB
        vocab_size: int = 50257,
        dtype: str = "uint16",
    ) -> None:
        """Initializes BinaryDatasetWriter.

        Args:
            output_dir: Directory path for binary shard files.
            shard_prefix: Shard filename prefix ("train", "val", "test").
            max_shard_size_bytes: Maximum byte threshold before creating a new shard file.
            vocab_size: Target vocabulary size.
            dtype: Data type string ("uint16" or "uint32").
        """
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shard_prefix = shard_prefix
        self.max_shard_size_bytes = max_shard_size_bytes
        self.vocab_size = vocab_size

        if dtype == "uint16":
            if vocab_size > 65535:
                raise ValueError(
                    f"vocab_size ({vocab_size}) exceeds uint16 max capacity (65,535). Use dtype='uint32'."
                )
            self.np_dtype = np.uint16
            self.bytes_per_token = 2
        elif dtype == "uint32":
            self.np_dtype = np.uint32
            self.bytes_per_token = 4
        else:
            raise ValueError(f"Unsupported dtype: '{dtype}'. Must be 'uint16' or 'uint32'.")

        self.max_tokens_per_shard = self.max_shard_size_bytes // self.bytes_per_token

        self.current_shard_idx = 0
        self.total_tokens_written = 0
        self.shard_metadata: List[Dict[str, Any]] = []

        self._active_buffer: List[int] = []
        self._current_shard_tokens = 0

    def write_tokens(self, token_ids: List[int]) -> int:
        """Appends a list of token IDs to active binary buffer and flushes to disk when full.

        Args:
            token_ids: List of integer token IDs.

        Returns:
            Number of tokens appended.
        """
        if not token_ids:
            return 0

        self._active_buffer.extend(token_ids)
        self._current_shard_tokens += len(token_ids)
        self.total_tokens_written += len(token_ids)

        if self._current_shard_tokens >= self.max_tokens_per_shard:
            self.flush()

        return len(token_ids)

    def flush(self) -> Optional[Path]:
        """Flushes buffered token IDs to a binary .bin shard file on disk."""
        if not self._active_buffer:
            return None

        shard_filename = f"{self.shard_prefix}_{self.current_shard_idx:04d}.bin"
        shard_path = self.output_dir / shard_filename

        arr = np.array(self._active_buffer, dtype=self.np_dtype)
        with open(shard_path, "wb") as f:
            f.write(arr.tobytes())

        num_tokens = len(arr)
        file_bytes = shard_path.stat().st_size

        meta = {
            "shard_idx": self.current_shard_idx,
            "filename": shard_filename,
            "path": str(shard_path),
            "num_tokens": num_tokens,
            "file_size_bytes": file_bytes,
            "dtype": str(self.np_dtype.__name__),
        }
        self.shard_metadata.append(meta)

        logger.info(
            "Flushed binary shard '%s': %d tokens (%.2f MB)",
            shard_filename,
            num_tokens,
            file_bytes / (1024 * 1024),
        )

        self._active_buffer = []
        self._current_shard_tokens = 0
        self.current_shard_idx += 1

        return shard_path

    def close(self) -> Dict[str, Any]:
        """Flushes remaining buffered tokens and writes metadata JSON manifest file.

        Returns:
            Summary metadata dictionary.
        """
        self.flush()

        manifest_path = self.output_dir / f"{self.shard_prefix}_metadata.json"
        summary = {
            "shard_prefix": self.shard_prefix,
            "vocab_size": self.vocab_size,
            "dtype": str(self.np_dtype.__name__),
            "total_shards": len(self.shard_metadata),
            "total_tokens_written": self.total_tokens_written,
            "shards": self.shard_metadata,
        }

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        logger.info(
            "Closed BinaryDatasetWriter '%s': %d shards, %d total tokens written.",
            self.shard_prefix,
            len(self.shard_metadata),
            self.total_tokens_written,
        )

        return summary
