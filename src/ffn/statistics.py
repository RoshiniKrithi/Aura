"""Feed Forward Network Statistics Extractor for Aura LLM Pipeline.

Computes hidden activation distribution statistics, activation sparsity ratios,
max activation values, and weight matrix gradient norms.
"""

from dataclasses import dataclass, field
import logging
import math
import torch

from src.ffn.network import FeedForwardNetwork

logger = logging.getLogger(__name__)


@dataclass
class FeedForwardStats:
    """Summary container holding quantitative Feed Forward Network statistics."""

    d_model: int
    hidden_dim: int
    mean_activation: float
    max_activation: float
    sparsity_ratio: float
    w1_grad_norm: float = 0.0
    w2_grad_norm: float = 0.0


class FeedForwardStatistics:
    """Computes statistical metrics over FeedForwardNetwork activations and weights.

    Time Complexity:
        O(B * T * d_ff) scan over cached hidden activations.

    Space Complexity:
        O(1) scalar memory overhead.
    """

    @staticmethod
    def compute_stats(layer: FeedForwardNetwork) -> FeedForwardStats:
        """Calculates comprehensive metrics for given FeedForwardNetwork module.

        Args:
            layer: Instantiated FeedForwardNetwork module.

        Returns:
            FeedForwardStats summary object.

        Raises:
            ValueError: If layer has not executed forward pass yet.
        """
        if layer.last_hidden_activations is None:
            raise ValueError("FeedForwardNetwork layer has no cached activations. Run forward pass first.")

        acts = layer.last_hidden_activations

        # 1. Mean Activation
        mean_act = acts.mean().item()

        # 2. Max Activation
        max_act = acts.abs().max().item()

        # 3. Sparsity Ratio (% of activations <= 1e-4)
        sparsity = (acts.abs() < 1e-4).float().mean().item()

        # 4. Projection Gradient Norms
        if layer.is_swiglu:
            w1_grad = (
                torch.norm(layer.w_gate.weight.grad.detach().cpu(), p=2).item()
                if layer.w_gate.weight.grad is not None
                else 0.0
            )
            w2_grad = (
                torch.norm(layer.w_down.weight.grad.detach().cpu(), p=2).item()
                if layer.w_down.weight.grad is not None
                else 0.0
            )
        else:
            w1_grad = (
                torch.norm(layer.w_1.weight.grad.detach().cpu(), p=2).item()
                if layer.w_1.weight.grad is not None
                else 0.0
            )
            w2_grad = (
                torch.norm(layer.w_2.weight.grad.detach().cpu(), p=2).item()
                if layer.w_2.weight.grad is not None
                else 0.0
            )

        return FeedForwardStats(
            d_model=layer.d_model,
            hidden_dim=layer.hidden_dim,
            mean_activation=round(mean_act, 4),
            max_activation=round(max_act, 4),
            sparsity_ratio=round(sparsity, 4),
            w1_grad_norm=round(w1_grad, 6),
            w2_grad_norm=round(w2_grad, 6),
        )
