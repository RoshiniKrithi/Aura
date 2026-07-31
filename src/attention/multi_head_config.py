"""Multi-Head Attention Configuration Dataclass for Aura LLM Architecture.

Provides parameters for model dimension, number of attention heads, head dimension calculation,
dropout probability, causal masking, bias terms, and target device placement.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class MultiHeadAttentionConfig:
    """Hyperparameter configuration container for Multi-Head Attention.

    Attributes:
        d_model: Total embedding/model feature dimension d_model.
        n_heads: Number of parallel attention heads H.
        dropout: Attention weights and output projection dropout probability p.
        causal: If True, applies lower-triangular 4D causal mask.
        scale: Custom scaling factor (defaults to 1 / sqrt(d_head)).
        bias: If True, enables bias vectors in linear projection layers.
        device: Execution target device ("auto", "cpu", "cuda", "mps").
    """

    d_model: int = 768
    n_heads: int = 12
    dropout: float = 0.1
    causal: bool = True
    scale: Optional[float] = None
    bias: bool = True
    device: str = "auto"

    @property
    def head_dim(self) -> int:
        """Calculates dimension per attention head: d_head = d_model / n_heads.

        Returns:
            Integer dimension per head.

        Raises:
            ValueError: If d_model is not evenly divisible by n_heads.
        """
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})."
            )
        return self.d_model // self.n_heads
