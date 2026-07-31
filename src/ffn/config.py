"""Feed Forward Network (FFN / MLP) Configuration Dataclass for Aura LLM Architecture.

Provides parameters for model dimension d_model, hidden expansion dimension d_ff (4x expansion),
activation function choice (GELU, ReLU, SiLU, SwiGLU), dropout probability, weight initializer, and target device placement.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class FeedForwardConfig:
    """Hyperparameter configuration container for Transformer Feed Forward Network.

    Attributes:
        d_model: Input and output feature dimension d_model.
        expansion_factor: Expansion factor multiplier (defaults to 4x, yielding d_ff = 4 * d_model).
        d_ff: Optional custom hidden dimension (overrides expansion_factor if specified).
        activation: Activation function choice ("gelu", "relu", "silu", "swiglu").
        dropout: Residual dropout probability p applied after activation.
        bias: If True, enables learnable bias terms in linear projections.
        initializer: Weight initialization strategy ("normal", "uniform", "xavier_uniform", "xavier_normal", "kaiming_uniform", "truncated_normal").
        init_range: Uniform initialization bound range [-init_range, init_range].
        init_std: Normal initialization standard deviation.
        device: Target execution device ("auto", "cpu", "cuda", "mps").
    """

    d_model: int = 768
    expansion_factor: int = 4
    d_ff: Optional[int] = None
    activation: str = "gelu"
    dropout: float = 0.1
    bias: bool = True
    initializer: str = "normal"
    init_range: float = 0.02
    init_std: float = 0.02
    device: str = "auto"

    @property
    def hidden_dim(self) -> int:
        """Calculates hidden dimension d_ff.

        Returns:
            Integer hidden dimension size.
        """
        if self.d_ff is not None and self.d_ff > 0:
            return self.d_ff
        return self.d_model * self.expansion_factor
