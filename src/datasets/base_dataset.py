"""Base Dataset Abstraction for Aura LLM Pipeline.

Defines standard interface for all PyTorch-compatible dataset variants in Aura.
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Union
import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    """Abstract Base Dataset class extending PyTorch Dataset.

    Requires implementation of standard dataset access hooks __len__ and __getitem__.
    """

    @abstractmethod
    def __len__(self) -> int:
        """Returns the total number of context window sequence pairs in dataset."""
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieves input-target sequence pair (X, Y) at given index.

        Args:
            idx: Sequence index.

        Returns:
            Tuple of PyTorch LongTensors (X_i, Y_i).
        """
        pass

    @abstractmethod
    def get_metadata(self) -> Dict[str, Union[int, float, str]]:
        """Returns structured metadata describing dataset dimensions and properties."""
        pass
