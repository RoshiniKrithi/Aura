"""Transformer Decoder Block Configuration Dataclass for Aura LLM Architecture.

Provides structured parameters for embedding dimension d_model, attention heads n_heads,
feed-forward hidden dimension d_ff, dropout rate, activation type, and norm layer choices.
"""

from dataclasses import dataclass, field
from typing import Optional

from src.attention.multi_head_config import MultiHeadAttentionConfig
from src.ffn.config import FeedForwardConfig
from src.normalization.config import LayerNormConfig
from src.transformer.config import ResidualConfig


@dataclass(frozen=True)
class TransformerBlockConfig:
    """Hyperparameter configuration container for TransformerBlock.

    Attributes:
        d_model: Feature dimension d_model across which block operates (default: 768).
        n_heads: Number of parallel attention heads (default: 12).
        d_ff: Hidden dimension inside Feed-Forward MLP (default: 3072).
        dropout: Probability of dropout applied to residual connections (default: 0.1).
        activation: FFN activation function ("swiglu" or "gelu", default: "swiglu").
        norm_type: Normalization layer type ("layer_norm" or "rms_norm", default: "layer_norm").
        eps: Numerical stability constant epsilon (default: 1e-5).
        bias: If True, linear projection layers include bias terms (default: False).
        device: Target execution compute device ("auto", "cpu", "cuda", "mps").
        attn_config: Sub-module Attention configuration override.
        ffn_config: Sub-module FFN configuration override.
        norm_config: Sub-module Normalization configuration override.
        residual_config: Sub-module Residual configuration override.
    """

    d_model: int = 768
    n_heads: int = 12
    d_ff: int = 3072
    dropout: float = 0.1
    activation: str = "swiglu"
    norm_type: str = "layer_norm"
    eps: float = 1e-5
    bias: bool = False
    device: str = "auto"
    attn_config: Optional[MultiHeadAttentionConfig] = None
    ffn_config: Optional[FeedForwardConfig] = None
    norm_config: Optional[LayerNormConfig] = None
    residual_config: Optional[ResidualConfig] = None
