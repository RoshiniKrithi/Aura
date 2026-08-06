"""Low-Rank Adaptation (LoRA) Linear Layer Wrapper for Aura EXP-007.

Provides LoRALinear module executing h = W_0(x) + (alpha / r) * B(A(dropout(x))).
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple


class LoRALinear(nn.Module):
    """Low-Rank Adaptation wrapper around base linear layer."""

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.05,
    ) -> None:
        """Initializes LoRALinear.

        Args:
            base_layer: Original nn.Linear layer to wrap and adapt.
            r: LoRA rank dimension.
            alpha: Constant scaling hyperparameter.
            dropout: Adapter dropout probability.
        """
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r if r > 0 else 1.0
        self.merged = False

        # Freeze base layer parameters
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        if r > 0:
            # Down-projection matrix A ~ N(0, 1/r)
            self.lora_A = nn.Parameter(torch.zeros(r, in_features))
            # Up-projection matrix B = 0 (ensures delta_W = 0 at step 0)
            self.lora_B = nn.Parameter(torch.zeros(out_features, r))
            self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

            self._reset_parameters()
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)
            self.lora_dropout = nn.Identity()

    def _reset_parameters(self) -> None:
        """Initializes A with Kaiming uniform distribution and B to zero."""
        if self.r > 0:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: computes base projection + low-rank adapter update."""
        if self.merged or self.r == 0:
            return self.base_layer(x)

        base_out = self.base_layer(x)
        # Compute adapter branch: (alpha / r) * (x @ A^T @ B^T)
        dropped_x = self.lora_dropout(x)
        lora_out = (dropped_x @ self.lora_A.T) @ self.lora_B.T
        return base_out + (lora_out * self.scaling)

    def merge_weights(self) -> None:
        """Merges (alpha / r) * B * A directly into base_layer weight matrix in-place."""
        if self.merged or self.r == 0:
            return

        with torch.no_grad():
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            self.base_layer.weight.data += delta_w.to(self.base_layer.weight.dtype)
            self.merged = True

    def unmerge_weights(self) -> None:
        """Subtracts adapter weights from base_layer weight matrix to restore original weights."""
        if not self.merged or self.r == 0:
            return

        with torch.no_grad():
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            self.base_layer.weight.data -= delta_w.to(self.base_layer.weight.dtype)
            self.merged = False
