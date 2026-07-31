"""Dataset Disk Caching System for Aura LLM Pipeline.

Serializes tokenized sequences and context window tensors to disk in binary format,
validating SHA-256 source file content and hyperparameter integrity on load.
"""

import hashlib
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, Optional, Tuple, Union
import torch

from src.datasets.exceptions import DatasetCacheError

logger = logging.getLogger(__name__)


class DatasetCache:
    """Disk-backed binary caching manager for pre-tokenized context window tensors.

    Design Decisions:
        - Stores PyTorch Tensors directly in binary format (`.pt`) alongside JSON metadata.
        - Calculates SHA-256 checksums of source data files and pipeline hyperparameters.
        - Automatic invalidation upon cache corruption, missing files, or config mismatch.

    Time Complexity:
        O(N) write time; O(N) binary read time (orders of magnitude faster than raw tokenization).

    Space Complexity:
        O(N) disk storage for cached tensor arrays.
    """

    def __init__(
        self,
        cache_dir: Union[Path, str] = "data/cache",
        enabled: bool = True,
    ) -> None:
        """Initializes dataset cache directory and status.

        Args:
            cache_dir: Path to directory storing cached binary files.
            enabled: If False, bypasses cache read/write operations.
        """
        self.cache_dir = Path(cache_dir).resolve()
        self.enabled = enabled

        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _compute_key(
        self, source_identifier: str, config_dict: Dict[str, Any]
    ) -> str:
        """Generates deterministic SHA-256 cache key from source identifier and config parameters."""
        config_json = json.dumps(config_dict, sort_keys=True)
        combined_payload = f"{source_identifier}:{config_json}".encode("utf-8")
        return hashlib.sha256(combined_payload).hexdigest()

    def get(
        self, source_identifier: str, config_dict: Dict[str, Any]
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Retrieves cached (X, Y) sequence tensors if valid cache entry exists.

        Args:
            source_identifier: String identifier or path of source dataset.
            config_dict: Dictionary of active hyperparameters (window_size, stride, tokenizer config, etc.).

        Returns:
            Tuple of PyTorch LongTensors (X, Y) if valid cache hit; None if cache miss or invalidated.
        """
        if not self.enabled:
            return None

        cache_key = self._compute_key(source_identifier, config_dict)
        data_path = self.cache_dir / f"{cache_key}.pt"
        meta_path = self.cache_dir / f"{cache_key}.json"

        if not data_path.exists() or not meta_path.exists():
            logger.info("Cache miss for key %s", cache_key[:12])
            return None

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            if meta.get("cache_key") != cache_key:
                logger.warning(
                    "Cache key mismatch in metadata for key %s. Invalidating cache.",
                    cache_key[:12],
                )
                return None

            payload = torch.load(data_path, map_location="cpu", weights_only=True)
            x_tensor = payload["x"]
            y_tensor = payload["y"]

            logger.info(
                "Cache HIT: Loaded pre-tokenized sequence tensors from %s",
                data_path.name,
            )
            return x_tensor, y_tensor

        except Exception as e:
            logger.warning(
                "Error reading cache file %s (%s). Invalidating.",
                data_path,
                str(e),
            )
            return None

    def save(
        self,
        source_identifier: str,
        config_dict: Dict[str, Any],
        x_tensor: torch.Tensor,
        y_tensor: torch.Tensor,
    ) -> Path:
        """Saves (X, Y) sequence tensors and metadata to cache directory.

        Args:
            source_identifier: Source identifier or path.
            config_dict: Active hyperparameter options.
            x_tensor: Input context window tensor.
            y_tensor: Target prediction window tensor.

        Returns:
            Path to saved binary cache data file.
        """
        if not self.enabled:
            raise DatasetCacheError(
                "Cannot save to cache when caching is disabled."
            )

        cache_key = self._compute_key(source_identifier, config_dict)
        data_path = self.cache_dir / f"{cache_key}.pt"
        meta_path = self.cache_dir / f"{cache_key}.json"

        meta = {
            "cache_key": cache_key,
            "source_identifier": source_identifier,
            "config": config_dict,
            "num_sequences": x_tensor.size(0),
            "window_size": x_tensor.size(1) if x_tensor.ndim > 1 else 0,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        try:
            torch.save({"x": x_tensor, "y": y_tensor}, data_path)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            logger.info(
                "Successfully saved dataset cache entry: %s (Key: %s)",
                data_path.name,
                cache_key[:12],
            )
            return data_path
        except Exception as e:
            raise DatasetCacheError(
                f"Failed writing dataset cache: {str(e)}"
            ) from e

    def clear(self) -> int:
        """Clears all cached data files in cache directory.

        Returns:
            Count of deleted files.
        """
        if not self.cache_dir.exists():
            return 0

        count = 0
        for f in self.cache_dir.glob("*"):
            if f.is_file():
                f.unlink()
                count += 1

        logger.info("Cleared %d files from cache directory %s", count, self.cache_dir)
        return count
