"""Attention Configuration Dataclass for Aura LLM Architecture.

Provides parameters for model dimension, attention projection dimension,
dropout probability, causal triangular masking, scaling factor, and device placement.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class AttentionConfig:
    """Hyperparameter configuration container for Self-Attention.

    Attributes:
        d_model: Model input/output feature dimension d_model.
        d_attn: Query, Key, Value projection dimension d_k (defaults to d_model).
        dropout: Attention weights dropout probability p.
        causal: If True, applies lower-triangular causal mask to prevent attending to future tokens.
        scale: Optional custom scaling factor (defaults to 1 / sqrt(d_k)).
        bias: If True, enables learnable bias vectors in Q, K, V, and Output projections.
        device: Execution target device ("auto", "cpu", "cuda", "mps").
    """

    d_model: int = 768
    n_heads: int = 12
    d_attn: int = 768
    dropout: float = 0.1
    causal: bool = True
    scale: Optional[float] = None
    bias: bool = True
    device: str = "auto"

    @property
    def head_dim(self) -> int:
        """Calculates dimension per attention head."""
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        return self.d_model // self.n_heads
