"""Feed Forward Network Utilities and Activation Registry for Aura LLM Architecture.

Provides activation function resolution, SwiGLU calculation helpers, and functional activation callables.
"""

import logging
from typing import Callable, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.ffn.exceptions import FeedForwardConfigError

logger = logging.getLogger(__name__)


class FeedForwardUtilities:
    """Activation registry and helper functions for FeedForwardNetwork."""

    _ACTIVATION_REGISTRY: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
        "gelu": F.gelu,
        "relu": F.relu,
        "silu": F.silu,
        "swish": F.silu,
    }

    @classmethod
    def get_activation_fn(cls, name: str) -> Callable[[torch.Tensor], torch.Tensor]:
        """Resolves activation function by string name.

        Args:
            name: Activation function name ("gelu", "relu", "silu", "swiglu").

        Returns:
            Callable activation function taking FloatTensor -> FloatTensor.

        Raises:
            FeedForwardConfigError: If activation function is not supported.
        """
        act_name = name.lower()
        if act_name == "swiglu":
            return F.silu  # SwiGLU handles gating explicitly in FeedForwardNetwork
        if act_name not in cls._ACTIVATION_REGISTRY:
            raise FeedForwardConfigError(
                f"Unsupported activation function: '{name}'. Supported: {list(cls._ACTIVATION_REGISTRY.keys()) + ['swiglu']}"
            )
        return cls._ACTIVATION_REGISTRY[act_name]

    @staticmethod
    def compute_swiglu(
        x: torch.Tensor,
        w_gate: nn.Linear,
        w_up: nn.Linear,
        w_down: nn.Linear,
        dropout: Optional[nn.Dropout] = None,
    ) -> torch.Tensor:
        """Calculates SwiGLU activation: SwiGLU(X) = Dropout(SiLU(X W_gate) * (X W_up)) W_down.

        Args:
            x: Input FloatTensor of shape (B, T, d_model).
            w_gate: Linear gate projection layer.
            w_up: Linear up projection layer.
            w_down: Linear down projection layer.
            dropout: Optional dropout layer.

        Returns:
            Output FloatTensor of shape (B, T, d_model).
        """
        gate = F.silu(w_gate(x))
        up = w_up(x)
        gated_act = gate * up
        if dropout is not None:
            gated_act = dropout(gated_act)
        return w_down(gated_act)
