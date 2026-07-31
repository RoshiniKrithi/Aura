"""Attention Statistics Extractor for Aura LLM Pipeline.

Computes attention distribution entropy, attention map sparsity, max weight distribution,
and gradient norms of query, key, value, and output projection matrices.
"""

from dataclasses import dataclass, field
import logging
import math
from typing import Dict, Union
import torch

from src.attention.multi_head import MultiHeadAttention
from src.attention.single_head import SelfAttention

logger = logging.getLogger(__name__)


@dataclass
class AttentionStats:
    """Summary container holding quantitative attention statistics."""

    d_model: int
    d_attn: int
    mean_entropy: float
    max_attention_weight: float
    sparsity_ratio: float
    q_grad_norm: float = 0.0
    k_grad_norm: float = 0.0
    v_grad_norm: float = 0.0
    out_grad_norm: float = 0.0


class AttentionStatistics:
    """Computes statistical metrics over SelfAttention and MultiHeadAttention outputs and weights.

    Time Complexity:
        O(B * H * T^2) entropy and sparsity scan over attention weight matrix.

    Space Complexity:
        O(1) scalar memory overhead.
    """

    @staticmethod
    def compute_stats(layer: Union[SelfAttention, MultiHeadAttention]) -> AttentionStats:
        """Calculates comprehensive metrics for given attention module.

        Args:
            layer: Instantiated SelfAttention or MultiHeadAttention module.

        Returns:
            AttentionStats summary object.

        Raises:
            ValueError: If layer has not executed forward pass yet (last_attention_weights is None).
        """
        if layer.last_attention_weights is None:
            raise ValueError(
                "Attention layer has no cached attention weights. Run forward pass first."
            )

        attn_w = layer.last_attention_weights  # Shape: (B, T, T) or (B, H, T, T)

        # 1. Entropy Calculation: H = -sum(p * log(p))
        eps = 1e-12
        entropy_tensor = -torch.sum(attn_w * torch.log(attn_w + eps), dim=-1)
        mean_entropy = entropy_tensor.mean().item()

        # 2. Max Attention Weight
        max_w = attn_w.max().item()

        # 3. Sparsity Ratio (% of non-zero entries below threshold 1e-4)
        sparsity = (attn_w < 1e-4).float().mean().item()

        # 4. Projection Gradient Norms
        if isinstance(layer, SelfAttention):
            q_grad = (
                torch.norm(layer.q_proj.weight.grad.detach().cpu(), p=2).item()
                if layer.q_proj.weight.grad is not None
                else 0.0
            )
            k_grad = (
                torch.norm(layer.k_proj.weight.grad.detach().cpu(), p=2).item()
                if layer.k_proj.weight.grad is not None
                else 0.0
            )
            v_grad = (
                torch.norm(layer.v_proj.weight.grad.detach().cpu(), p=2).item()
                if layer.v_proj.weight.grad is not None
                else 0.0
            )
            out_grad = (
                torch.norm(layer.out_proj.weight.grad.detach().cpu(), p=2).item()
                if layer.out_proj.weight.grad is not None
                else 0.0
            )
            d_attn_val = layer.d_attn
        else:
            # MultiHeadAttention
            c_attn_grad = (
                torch.norm(layer.c_attn.weight.grad.detach().cpu(), p=2).item()
                if layer.c_attn.weight.grad is not None
                else 0.0
            )
            q_grad = k_grad = v_grad = c_attn_grad / math.sqrt(3.0)
            out_grad = (
                torch.norm(layer.c_proj.weight.grad.detach().cpu(), p=2).item()
                if layer.c_proj.weight.grad is not None
                else 0.0
            )
            d_attn_val = layer.head_dim

        return AttentionStats(
            d_model=layer.d_model,
            d_attn=d_attn_val,
            mean_entropy=round(mean_entropy, 4),
            max_attention_weight=round(max_w, 4),
            sparsity_ratio=round(sparsity, 4),
            q_grad_norm=round(q_grad, 6),
            k_grad_norm=round(k_grad, 6),
            v_grad_norm=round(v_grad, 6),
            out_grad_norm=round(out_grad, 6),
        )
