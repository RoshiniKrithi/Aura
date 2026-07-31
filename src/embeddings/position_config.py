"""Positional Embedding Configuration Dataclass for Aura LLM Architecture.

Provides parameters for sequence length limits, embedding dimensions,
positional encoding types (learnable, sinusoidal), initializers, and dropout.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PositionEmbeddingConfig:
    """Hyperparameter configuration container for Positional Embeddings.

    Attributes:
        max_sequence_length: Maximum sequence length capacity M.
        d_model: Dense embedding vector dimension d.
        embedding_type: Positional encoding method ("learnable", "sinusoidal").
        initializer: Weight initialization strategy for learnable embeddings.
        init_range: Range parameter for uniform initializations.
        init_std: Standard deviation parameter for normal initializations.
        dropout: Dropout probability applied to combined input embeddings.
        device: Execution target device ("auto", "cpu", "cuda", "mps").
    """

    max_sequence_length: int = 1024
    d_model: int = 768
    embedding_type: str = "learnable"
    initializer: str = "normal"
    init_range: float = 0.02
    init_std: float = 0.02
    dropout: float = 0.1
    device: str = "auto"
