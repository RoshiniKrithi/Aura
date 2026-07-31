"""PyTorch In-Memory Text Dataset implementation for Aura LLM Pipeline.

Stores pre-sliced input-target context window tensors (X, Y) for fast random-access
indexing during transformer training iterations.
"""

import logging
from typing import Dict, Optional, Sequence, Tuple, Union
import torch

from src.datasets.base_dataset import BaseDataset
from src.datasets.sequence_builder import SequenceBuilder

logger = logging.getLogger(__name__)


class AuraTextDataset(BaseDataset):
    """PyTorch Dataset holding pre-sliced context window sequence pairs (X, Y).

    Design Decisions:
        - Stores ready-to-use PyTorch LongTensors (X, Y) in contiguous memory.
        - Provides O(1) random access indexing for PyTorch DataLoaders and parallel worker processes.

    Time Complexity:
        O(1) item retrieval via __getitem__.

    Space Complexity:
        O(S * L) space where S is number of sequences and L is window size.
    """

    def __init__(
        self,
        x_tensor: Optional[torch.Tensor] = None,
        y_tensor: Optional[torch.Tensor] = None,
        token_ids: Optional[Sequence[int]] = None,
        window_size: int = 64,
        stride: Optional[int] = None,
        name: str = "AuraTextDataset",
    ) -> None:
        """Initializes AuraTextDataset using pre-built tensors OR raw token IDs.

        Args:
            x_tensor: Input sequences PyTorch LongTensor of shape (N, window_size).
            y_tensor: Target sequences PyTorch LongTensor of shape (N, window_size).
            token_ids: Optional raw token sequence integer array (used if x_tensor is None).
            window_size: Sliding window size L.
            stride: Step size S between consecutive windows.
            name: Dataset label or name.
        """
        self.name = name

        if x_tensor is not None and y_tensor is not None:
            if x_tensor.shape != y_tensor.shape:
                raise ValueError(
                    f"X and Y tensor shapes must match. Got X: {x_tensor.shape}, Y: {y_tensor.shape}"
                )
            self.x_tensor = (
                x_tensor
                if x_tensor.dtype == torch.long
                else x_tensor.to(torch.long)
            )
            self.y_tensor = (
                y_tensor
                if y_tensor.dtype == torch.long
                else y_tensor.to(torch.long)
            )
            self.window_size = (
                x_tensor.size(1) if x_tensor.ndim > 1 else window_size
            )
        elif token_ids is not None:
            builder = SequenceBuilder(
                window_size=window_size, stride=stride, drop_last=True
            )
            self.x_tensor, self.y_tensor = builder.build_sequences(token_ids)
            self.window_size = window_size
        else:
            raise ValueError(
                "Either (x_tensor, y_tensor) pair or token_ids must be provided to AuraTextDataset."
            )

    def __len__(self) -> int:
        """Returns total number of sequence pairs in dataset."""
        return self.x_tensor.size(0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieves sequence pair (X_i, Y_i) at index idx.

        Args:
            idx: Sequence index.

        Returns:
            Tuple of LongTensors (X_i, Y_i).
        """
        return self.x_tensor[idx], self.y_tensor[idx]

    def get_metadata(self) -> Dict[str, Union[int, float, str]]:
        """Returns metadata summary."""
        return {
            "name": self.name,
            "num_sequences": len(self),
            "window_size": self.window_size,
            "tensor_dtype": str(self.x_tensor.dtype),
        }
