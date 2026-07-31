"""PyTorch Streaming Iterable Dataset for Aura LLM Pipeline.

Provides memory-efficient tokenization and sliding window iteration over multi-gigabyte
corpora (e.g. CodeSearchNet, The Stack) without loading datasets into RAM.
"""

import logging
from typing import Any, Generator, List, Optional, Tuple
import torch
from torch.utils.data import IterableDataset, get_worker_info

from src.datasets.reader import StreamingReader
from src.datasets.text_cleaner import TextCleaner

logger = logging.getLogger(__name__)


class AuraStreamingDataset(IterableDataset):
    """PyTorch IterableDataset for streaming massive datasets directly from disk.

    Design Decisions:
        - Implements worker-aware partition logic via get_worker_info() for PyTorch DataLoader parallelization.
        - On-the-fly text cleaning, tokenization, and window sliding with minimal memory footprint.

    Time Complexity:
        O(1) memory overhead per sequence yielded.

    Space Complexity:
        O(window_size) sliding token buffer in RAM.
    """

    def __init__(
        self,
        streaming_reader: StreamingReader,
        tokenizer: Any,
        window_size: int = 64,
        stride: Optional[int] = None,
        cleaner: Optional[TextCleaner] = None,
    ) -> None:
        """Initializes AuraStreamingDataset.

        Args:
            streaming_reader: Instantiated StreamingReader containing source file paths.
            tokenizer: Tokenizer instance (BPETokenizer, CharacterTokenizer, etc.).
            window_size: Context window size L.
            stride: Window step size S. Defaults to window_size.
            cleaner: Optional TextCleaner instance.
        """
        self.reader = streaming_reader
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.stride = stride if stride is not None else window_size
        self.cleaner = cleaner or TextCleaner()

    def _get_worker_files(self) -> List[Any]:
        """Partitions input files across PyTorch DataLoader multi-process workers."""
        worker_info = get_worker_info()
        all_files = self.reader.file_paths

        if worker_info is None:
            return all_files
        else:
            # Distribute files evenly among parallel worker processes
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            return all_files[worker_id::num_workers]

    def __iter__(self) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
        """Iterates over partitioned files, tokenizing and yielding sequence pairs (X, Y)."""
        worker_files = self._get_worker_files()
        token_buffer: List[int] = []
        req_len = self.window_size + 1

        for file_path in worker_files:
            try:
                with open(file_path, "r", encoding=self.reader.encoding) as f:
                    for line in f:
                        cleaned_line = self.cleaner.clean(line)
                        if not cleaned_line:
                            continue

                        # Tokenize line
                        line_token_ids = self.tokenizer.encode(cleaned_line)
                        token_buffer.extend(line_token_ids)

                        # Drain token buffer into window pairs
                        while len(token_buffer) >= req_len:
                            x_slice = token_buffer[: self.window_size]
                            y_slice = token_buffer[1:req_len]

                            x_tensor = torch.tensor(x_slice, dtype=torch.long)
                            y_tensor = torch.tensor(y_slice, dtype=torch.long)

                            yield x_tensor, y_tensor

                            # Advance buffer by stride S
                            token_buffer = token_buffer[self.stride :]

            except Exception as e:
                logger.error("Streaming error reading file %s: %s", file_path, str(e))
                continue
