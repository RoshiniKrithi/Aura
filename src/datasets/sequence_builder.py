"""Sliding Context Window and Target Sequence Generator for Aura LLM Pipeline.

Generates next-token prediction input-target pairs (X, Y) with configurable
window sizes, strides, BOS/EOS token insertion, and padding options.
"""

import logging
from typing import List, Sequence, Tuple
import torch

from src.datasets.exceptions import SequenceGenerationError

logger = logging.getLogger(__name__)


class SequenceBuilder:
    """Generates sliding context windows and target sequence pairs for transformer training.

    Design Decisions:
        - Strict next-token alignment: target tensor Y is input X shifted right by 1 position.
        - Vectorized slicing via PyTorch tensors for zero-copy efficiency.
        - Configurable stride S allows overlapping windows (e.g. S=1) or non-overlapping windows (e.g. S=L).

    Time Complexity:
        O(N / S * L) for sequence extraction and tensor creation.

    Space Complexity:
        O(N) space to hold window indices and PyTorch tensors.
    """

    def __init__(
        self,
        window_size: int = 64,
        stride: int | None = None,
        drop_last: bool = True,
        pad_token_id: int | None = None,
    ) -> None:
        """Initializes SequenceBuilder.

        Args:
            window_size: Length L of each context window.
            stride: Step size S between consecutive windows. Defaults to window_size (non-overlapping).
            drop_last: If True, discards trailing tokens that do not fill a full window_size + 1 block.
            pad_token_id: Optional token ID used to pad trailing partial windows if drop_last is False.
        """
        if window_size <= 0:
            raise SequenceGenerationError(
                f"window_size must be positive, got {window_size}"
            )

        self.window_size = window_size
        self.stride = stride if stride is not None else window_size
        if self.stride <= 0:
            raise SequenceGenerationError(
                f"stride must be positive, got {self.stride}"
            )

        self.drop_last = drop_last
        self.pad_token_id = pad_token_id

    def build_sequences(
        self, token_ids: Sequence[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Slices token ID array into input X and target Y PyTorch tensors.

        Args:
            token_ids: Flat sequence of token IDs.

        Returns:
            Tuple of PyTorch LongTensors (X, Y) each of shape (num_sequences, window_size).
        """
        n_tokens = len(token_ids)
        required_len = self.window_size + 1

        if n_tokens < required_len:
            if self.drop_last or self.pad_token_id is None:
                logger.warning(
                    "Token sequence length (%d) is smaller than minimum required window length (%d). Returning empty sequence tensors.",
                    n_tokens,
                    required_len,
                )
                empty_tensor = torch.empty(
                    (0, self.window_size), dtype=torch.long
                )
                return empty_tensor, empty_tensor
            else:
                # Pad single partial sequence
                pad_needed = required_len - n_tokens
                padded_tokens = list(token_ids) + [self.pad_token_id] * pad_needed
                x_seq = torch.tensor(
                    padded_tokens[: self.window_size], dtype=torch.long
                ).unsqueeze(0)
                y_seq = torch.tensor(
                    padded_tokens[1:required_len], dtype=torch.long
                ).unsqueeze(0)
                return x_seq, y_seq

        x_list: List[torch.Tensor] = []
        y_list: List[torch.Tensor] = []

        tokens_tensor = (
            token_ids
            if isinstance(token_ids, torch.Tensor)
            else torch.tensor(token_ids, dtype=torch.long)
        )

        i = 0
        while i + required_len <= n_tokens:
            x_win = tokens_tensor[i : i + self.window_size]
            y_win = tokens_tensor[i + 1 : i + required_len]
            x_list.append(x_win)
            y_list.append(y_win)
            i += self.stride

        # Handle remaining trailing partial tokens if not drop_last
        if not self.drop_last and i + 1 < n_tokens and self.pad_token_id is None:
            pass  # Without pad_token_id, we cannot build incomplete sequence
        elif (
            not self.drop_last
            and i + 1 < n_tokens
            and self.pad_token_id is not None
        ):
            remaining = tokens_tensor[i:]
            rem_len = len(remaining)
            pad_needed = required_len - rem_len
            pad_tensor = torch.full(
                (pad_needed,), self.pad_token_id, dtype=torch.long
            )
            full_block = torch.cat([remaining, pad_tensor])
            x_list.append(full_block[: self.window_size])
            y_list.append(full_block[1:required_len])

        if not x_list:
            empty_tensor = torch.empty((0, self.window_size), dtype=torch.long)
            return empty_tensor, empty_tensor

        x_tensors = torch.stack(x_list)
        y_tensors = torch.stack(y_list)

        logger.info(
            "Created %d context sequences of window_size=%d (stride=%d). Tensor shape: %s",
            x_tensors.size(0),
            self.window_size,
            self.stride,
            tuple(x_tensors.shape),
        )

        return x_tensors, y_tensors
