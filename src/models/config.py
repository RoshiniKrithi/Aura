"""GPT Decoder Configuration Schema & Scaling Presets for Aura LLM Architecture.

Provides parameters for vocabulary size, max sequence length, embedding dimension d_model,
layer count n_layers, head count n_heads, FFN hidden dimension d_ff, dropout rate, and scaling presets.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class GPTConfig:
    """Hyperparameter configuration container for GPTModel decoder trunk.

    Attributes:
        model_name: Architectural model identifier string (default: "aura-gpt-125m").
        vocab_size: Vocabulary size (default: 50257).
        max_sequence_length: Maximum context sequence length T (default: 2048).
        d_model: Feature dimension d_model (default: 768).
        n_layers: Number of stacked Transformer blocks N (default: 12).
        n_heads: Number of parallel attention heads H (default: 12).
        d_ff: Hidden dimension inside Feed-Forward MLP (default: 3072).
        dropout: Global dropout probability (default: 0.1).
        activation: FFN activation function ("swiglu" or "gelu", default: "swiglu").
        norm_type: Normalization type ("layer_norm" or "rms_norm", default: "layer_norm").
        eps: Epsilon constant for normalization numerical stability (default: 1e-5).
        bias: If True, enables bias terms in linear projection layers (default: False).
        initializer_range: Base standard deviation for weight initialization (default: 0.02).
        device: Target compute device placement ("auto", "cpu", "cuda", "mps").
    """

    model_name: str = "aura-gpt-125m"
    vocab_size: int = 50257
    max_sequence_length: int = 2048
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    d_ff: int = 3072
    dropout: float = 0.1
    activation: str = "swiglu"
    norm_type: str = "layer_norm"
    eps: float = 1e-5
    tie_weights: bool = True
    bias: bool = False
    initializer_range: float = 0.02
    device: str = "auto"

    @property
    def head_dim(self) -> int:
        """Calculates feature dimension per attention head: head_dim = d_model / n_heads."""
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be evenly divisible by n_heads ({self.n_heads})"
            )
        return self.d_model // self.n_heads

    @classmethod
    def get_125m_config(cls, **kwargs: Any) -> "GPTConfig":
        """Returns 125M parameter base model configuration."""
        defaults = dict(
            model_name="aura-gpt-125m",
            d_model=768,
            n_layers=12,
            n_heads=12,
            d_ff=3072,
            max_sequence_length=2048,
        )
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def get_350m_config(cls, **kwargs: Any) -> "GPTConfig":
        """Returns 350M parameter medium model configuration."""
        defaults = dict(
            model_name="aura-gpt-350m",
            d_model=1024,
            n_layers=24,
            n_heads=16,
            d_ff=4096,
            max_sequence_length=2048,
        )
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def get_1_3b_config(cls, **kwargs: Any) -> "GPTConfig":
        """Returns 1.3B parameter large model configuration."""
        defaults = dict(
            model_name="aura-gpt-1.3b",
            d_model=2048,
            n_layers=24,
            n_heads=32,
            d_ff=8192,
            max_sequence_length=4096,
        )
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def get_7b_config(cls, **kwargs: Any) -> "GPTConfig":
        """Returns 7B parameter XL model configuration."""
        defaults = dict(
            model_name="aura-gpt-7b",
            d_model=4096,
            n_layers=32,
            n_heads=32,
            d_ff=11008,
            max_sequence_length=4096,
        )
        defaults.update(kwargs)
        return cls(**defaults)


# Alias for backward compatibility
AuraGPTConfig = GPTConfig
