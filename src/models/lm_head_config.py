"""LM Head Configuration Schema for Aura LLM Architecture.

Provides parameters for embedding dimension d_model, vocabulary size vocab_size,
weight tying enablement, bias vectors, and target execution devices.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class LMHeadConfig:
    """Hyperparameter configuration container for LanguageModelingHead module.

    Attributes:
        d_model: Input hidden feature dimension (default: 768).
        vocab_size: Vocabulary size V (default: 50257).
        tie_weights: If True, uses weight sharing with embedding layer (default: True).
        bias: If True, enables bias vector in linear projection (default: False).
        device: Target execution device placement ("auto", "cpu", "cuda", "mps").
    """

    d_model: int = 768
    vocab_size: int = 50257
    tie_weights: bool = True
    bias: bool = False
    device: str = "auto"
