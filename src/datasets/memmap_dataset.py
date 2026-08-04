"""Zero-Copy Memory-Mapped PyTorch Dataset Implementation for Aura LLM.

Wraps uint16 / uint32 binary disk shards using np.memmap for O(1) random-access
sequence window slicing without memory bloat.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset

from src.datasets.base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class MemmapCodeDataset(Dataset):
    """Zero-copy PyTorch Dataset interfacing directly with binary disk shards using np.memmap.

    Design Decisions:
        - Memory Efficiency: Leverages OS memory-mapped page cache (`np.memmap`), enabling
          seamless data loading from multi-gigabyte files with RAM consumption under 200 MB.
        - Virtual Concatenation: Stitches multiple binary shards into a continuous virtual sequence index.
        - Provides PyTorch LongTensor inputs and targets $(X, Y)$ shifted by 1 token for causal LM training.
    """

    def __init__(
        self,
        shard_paths: List[Union[str, Path]],
        sequence_length: int = 1024,
        stride: Optional[int] = None,
        dtype: str = "uint16",
        name: str = "MemmapCodeDataset",
    ) -> None:
        """Initializes MemmapCodeDataset.

        Args:
            shard_paths: List of binary .bin file paths.
            sequence_length: Context window length L.
            stride: Window stride size S (defaults to sequence_length).
            dtype: Data type string ("uint16" or "uint32").
            name: Dataset name tag.
        """
        self.name = name
        self.sequence_length = sequence_length
        self.stride = stride or sequence_length
        self.np_dtype = np.uint16 if dtype == "uint16" else np.uint32

        self.shard_paths = [Path(p).resolve() for p in shard_paths if Path(p).exists()]
        if not self.shard_paths:
            raise FileNotFoundError("No valid binary shard paths provided to MemmapCodeDataset.")

        # Open memmap views for each shard
        self.memmaps: List[np.ndarray] = []
        self.shard_lengths: List[int] = []
        self.cum_lengths: List[int] = [0]

        total_tokens = 0
        for p in self.shard_paths:
            mmap = np.memmap(p, dtype=self.np_dtype, mode="r")
            self.memmaps.append(mmap)
            n_tok = len(mmap)
            self.shard_lengths.append(n_tok)
            total_tokens += n_tok
            self.cum_lengths.append(total_tokens)

        self.total_tokens = total_tokens
        # Number of valid sequence windows (each window needs sequence_length + 1 tokens for X and Y)
        usable = self.total_tokens - self.sequence_length
        self.num_sequences = max(0, usable // self.stride) if usable > 0 else 0

        logger.info(
            "Instantiated %s: shards=%d, total_tokens=%d, num_sequences=%d (L=%d, stride=%d)",
            self.name,
            len(self.shard_paths),
            self.total_tokens,
            self.num_sequences,
            self.sequence_length,
            self.stride,
        )

    def close(self) -> None:
        """Flushes and closes underlying memory-mapped file handles."""
        for mmap in self.memmaps:
            if hasattr(mmap, "_mmap") and mmap._mmap is not None:
                mmap._mmap.close()
        self.memmaps = []

    def __len__(self) -> int:
        """Returns total number of sliding window sequence pairs in dataset."""
        return self.num_sequences

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieves LongTensor sequence pair (X_i, Y_i) at sequence index idx.

        Args:
            idx: Sequence window index integer.

        Returns:
            Tuple of LongTensors (X_i, Y_i) of shape (sequence_length,).
        """
        if idx < 0 or idx >= self.num_sequences:
            raise IndexError(f"Index {idx} out of bounds for dataset of size {self.num_sequences}.")

        start_pos = idx * self.stride
        end_pos = start_pos + self.sequence_length + 1

        # Fetch contiguous slice across virtual memory mapped shards
        raw_slice = self._read_virtual_slice(start_pos, end_pos)
        long_slice = torch.from_numpy(raw_slice.astype(np.int64))

        x_tensor = long_slice[: self.sequence_length]
        y_tensor = long_slice[1 : self.sequence_length + 1]

        return x_tensor, y_tensor

    def _read_virtual_slice(self, start_pos: int, end_pos: int) -> np.ndarray:
        """Reads virtual slice spanning one or across multiple binary memmap shards."""
        needed = end_pos - start_pos
        result = np.empty(needed, dtype=self.np_dtype)
        written = 0

        curr_pos = start_pos
        while written < needed:
            # Find shard containing curr_pos
            shard_idx = self._find_shard_index(curr_pos)
            shard_start_pos = self.cum_lengths[shard_idx]
            offset_in_shard = curr_pos - shard_start_pos
            shard_len = self.shard_lengths[shard_idx]

            avail_in_shard = shard_len - offset_in_shard
            to_read = min(needed - written, avail_in_shard)

            mmap = self.memmaps[shard_idx]
            result[written : written + to_read] = mmap[offset_in_shard : offset_in_shard + to_read]

            written += to_read
            curr_pos += to_read

        return result

    def _find_shard_index(self, pos: int) -> int:
        """Binary search to locate shard index containing target pos."""
        low = 0
        high = len(self.shard_lengths) - 1
        while low <= high:
            mid = (low + high) // 2
            if self.cum_lengths[mid] <= pos < self.cum_lengths[mid + 1]:
                return mid
            elif pos >= self.cum_lengths[mid + 1]:
                low = mid + 1
            else:
                high = mid - 1
        return len(self.shard_lengths) - 1
