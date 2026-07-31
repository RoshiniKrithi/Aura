"""Position Embedding Utilities for Aura LLM Architecture.

Provides position ID sequence generation, relative position distance metrics,
and positional representation analysis.
"""

import logging
from typing import Optional, Union
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class PositionEmbeddingUtilities:
    """Utility functions for positional embedding calculations and ID generation.

    Design Decisions:
        - Pure PyTorch operations for fast position index vector generation.
    """

    @staticmethod
    def generate_position_ids(
        input_tensor_or_seq_len: Union[torch.Tensor, int],
        device: Optional[Union[str, torch.device]] = None,
    ) -> torch.Tensor:
        """Generates sequential position IDs [0, 1, ..., T-1] for an input tensor or sequence length integer.

        Args:
            input_tensor_or_seq_len: PyTorch Tensor of shape (B, T) or (T,) OR scalar integer seq_len.
            device: Target device for generated tensor.

        Returns:
            Position IDs LongTensor of shape (B, T) or (T,).
        """
        if isinstance(input_tensor_or_seq_len, torch.Tensor):
            target_device = device or input_tensor_or_seq_len.device
            if input_tensor_or_seq_len.ndim == 2:
                b_size, seq_len = input_tensor_or_seq_len.shape
                pos_1d = torch.arange(0, seq_len, dtype=torch.long, device=target_device)
                return pos_1d.unsqueeze(0).expand(b_size, -1)
            else:
                seq_len = input_tensor_or_seq_len.size(0)
                return torch.arange(0, seq_len, dtype=torch.long, device=target_device)
        else:
            seq_len = int(input_tensor_or_seq_len)
            target_device = device or "cpu"
            return torch.arange(0, seq_len, dtype=torch.long, device=target_device)

    @staticmethod
    def compute_positional_cosine_similarity(
        pos_embedding_module: torch.nn.Module, pos_a: int, pos_b: int
    ) -> float:
        """Computes cosine similarity between two position vectors pos_a and pos_b.

        Args:
            pos_embedding_module: Instantiated position embedding module (Learnable or Sinusoidal).
            pos_a: First position index.
            pos_b: Second position index.

        Returns:
            Cosine similarity float value in range [-1.0, 1.0].
        """
        vec_a = pos_embedding_module(sequence_length=pos_a + 1)[pos_a]
        vec_b = pos_embedding_module(sequence_length=pos_b + 1)[pos_b]
        sim = F.cosine_similarity(vec_a.unsqueeze(0), vec_b.unsqueeze(0))
        return float(sim.item())
