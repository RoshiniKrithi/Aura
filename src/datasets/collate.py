"""Custom Collate Function and Dynamic Padding for Aura LLM Pipeline.

Collates sequence samples into PyTorch batch tensors, performing dynamic padding
of variable-length sequences and tracking batch metadata statistics.
"""

from dataclasses import dataclass
import logging
from typing import List, Tuple
import torch

logger = logging.getLogger(__name__)


@dataclass
class BatchStatistics:
    """Summary metadata for a collated mini-batch."""

    batch_size: int
    max_sequence_length: int
    total_tokens: int
    padding_tokens: int
    padding_ratio: float


class CollateFunction:
    """Custom PyTorch DataLoader collate callable.

    Design Decisions:
        - Dynamically pads variable-length sequences within a batch to max sequence length in that batch.
        - Returns PyTorch LongTensors (input_ids, target_ids) ready for transformer embedding lookup.
        - Calculates zero-overhead batch padding statistics.

    Time Complexity:
        O(B * L) where B is batch size and L is max sequence length.

    Space Complexity:
        O(B * L) tensor allocation.
    """

    def __init__(self, pad_token_id: int = 0) -> None:
        """Initializes collate function.

        Args:
            pad_token_id: Integer token ID used for dynamic sequence padding.
        """
        self.pad_token_id = pad_token_id

    def __call__(
        self, batch: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Collates a list of (X, Y) sample tuples into stacked batch tensors.

        Args:
            batch: List of sample pairs [(x_1, y_1), (x_2, y_2), ...].

        Returns:
            Tuple of PyTorch LongTensors (X_batch, Y_batch) of shape (batch_size, max_seq_len).
        """
        if not batch:
            raise ValueError("Cannot collate an empty mini-batch.")

        x_samples = [sample[0] for sample in batch]
        y_samples = [sample[1] for sample in batch]

        max_len_x = max(x.size(0) for x in x_samples)
        max_len_y = max(y.size(0) for y in y_samples)
        max_len = max(max_len_x, max_len_y)

        padded_x: List[torch.Tensor] = []
        padded_y: List[torch.Tensor] = []

        for x, y in zip(x_samples, y_samples):
            pad_x = max_len - x.size(0)
            if pad_x > 0:
                x_pad = torch.full((pad_x,), self.pad_token_id, dtype=torch.long)
                x_curr = torch.cat([x, x_pad])
            else:
                x_curr = x

            pad_y = max_len - y.size(0)
            if pad_y > 0:
                y_pad = torch.full((pad_y,), self.pad_token_id, dtype=torch.long)
                y_curr = torch.cat([y, y_pad])
            else:
                y_curr = y

            padded_x.append(x_curr)
            padded_y.append(y_curr)

        x_batch = torch.stack(padded_x)
        y_batch = torch.stack(padded_y)

        return x_batch, y_batch

    def compute_batch_stats(
        self, x_batch: torch.Tensor, y_batch: torch.Tensor
    ) -> BatchStatistics:
        """Computes metadata statistics for a collated mini-batch."""
        b_size, seq_len = x_batch.shape
        total_tokens = b_size * seq_len
        pad_x_count = (x_batch == self.pad_token_id).sum().item()
        pad_ratio = pad_x_count / max(1, total_tokens)

        return BatchStatistics(
            batch_size=b_size,
            max_sequence_length=seq_len,
            total_tokens=total_tokens,
            padding_tokens=int(pad_x_count),
            padding_ratio=round(pad_ratio, 4),
        )
