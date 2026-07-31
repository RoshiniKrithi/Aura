"""Token Embedding Layer Configuration Dataclass for Aura LLM Architecture.

Provides strongly-typed parameters for vocabulary size, embedding dimension,
weight initialization schemes, scaling options, and device placement.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class EmbeddingConfig:
    """Hyperparameter configuration container for Token Embedding Layer.

    Attributes:
        vocab_size: Total vocabulary size V.
        d_model: Dense embedding vector dimension d.
        initializer: Weight initialization strategy ("normal", "uniform", "xavier_uniform", "xavier_normal", "kaiming_uniform", "truncated_normal").
        init_range: Standard range parameter for uniform initializations.
        init_std: Standard deviation parameter for normal initializations.
        scale_by_sqrt_d_model: If True, multiplies embedding vectors by sqrt(d_model) as in Vaswani et al.
        pad_idx: Optional token ID index to zero out during embedding lookup and gradients.
        device: Execution target device ("auto", "cpu", "cuda", "mps").
    """

    vocab_size: int = 50257
    d_model: int = 768
    initializer: str = "normal"
    init_range: float = 0.02
    init_std: float = 0.02
    scale_by_sqrt_d_model: bool = False
    pad_idx: Optional[int] = None
    device: str = "auto"
