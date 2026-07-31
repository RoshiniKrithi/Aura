"""Layer Normalization Configuration Dataclass for Aura LLM Architecture.

Provides parameters for model feature dimension d_model, numerical stability constant epsilon,
learnable scale/shift affine parameters, and target device placement.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class LayerNormConfig:
    """Hyperparameter configuration container for Layer Normalization.

    Attributes:
        d_model: Feature dimension d_model across which normalization is computed.
        eps: Small numerical stability epsilon constant added to variance (defaults to 1e-5).
        elementwise_affine: If True, includes learnable gamma (scale) and beta (shift) parameters.
        bias: If True, includes learnable beta (shift) bias parameter when elementwise_affine=True.
        device: Execution target device ("auto", "cpu", "cuda", "mps").
    """

    d_model: int = 768
    eps: float = 1e-5
    elementwise_affine: bool = True
    bias: bool = True
    device: str = "auto"
