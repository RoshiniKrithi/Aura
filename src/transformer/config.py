"""Residual Connection Configuration Dataclass for Aura LLM Architecture.

Provides parameters for dropout rate, norm positioning (pre_norm/post_norm),
model feature dimension d_model, and target device placement.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ResidualConfig:
    """Hyperparameter configuration container for Residual Connection.

    Attributes:
        dropout: Probability of dropout applied to sub-layer output before residual add.
        norm_position: Normalization layout strategy ("pre_norm" or "post_norm").
        d_model: Feature dimension d_model across which residual connection operates.
        device: Execution target device ("auto", "cpu", "cuda", "mps").
    """

    dropout: float = 0.1
    norm_position: str = "pre_norm"
    d_model: int = 768
    device: str = "auto"
