"""Layer Normalization Statistics Extractor for Aura LLM Pipeline.

Computes activation mean, variance distribution, gamma/beta parameter L2 norms,
and parameter gradient norms.
"""

from dataclasses import dataclass, field
import logging
import torch

from src.normalization.layer_norm import LayerNormalization

logger = logging.getLogger(__name__)


@dataclass
class LayerNormStats:
    """Summary container holding quantitative Layer Normalization statistics."""

    d_model: int
    eps: float
    mean_value: float
    mean_variance: float
    gamma_norm: float = 0.0
    beta_norm: float = 0.0
    gamma_grad_norm: float = 0.0
    beta_grad_norm: float = 0.0


class LayerNormStatistics:
    """Computes statistical metrics over LayerNormalization outputs and weights.

    Time Complexity:
        O(B * T) mean scan over cached mean and variance tensors.

    Space Complexity:
        O(1) scalar memory overhead.
    """

    @staticmethod
    def compute_stats(layer: LayerNormalization) -> LayerNormStats:
        """Calculates comprehensive metrics for given LayerNormalization module.

        Args:
            layer: Instantiated LayerNormalization module.

        Returns:
            LayerNormStats summary object.

        Raises:
            ValueError: If layer has not executed forward pass yet.
        """
        if layer.last_mean is None or layer.last_variance is None:
            raise ValueError("LayerNormalization layer has no cached statistics. Run forward pass first.")

        mean_val = layer.last_mean.mean().item()
        var_val = layer.last_variance.mean().item()

        gamma_n = (
            torch.norm(layer.gamma.detach().cpu(), p=2).item()
            if layer.gamma is not None
            else 0.0
        )
        beta_n = (
            torch.norm(layer.beta.detach().cpu(), p=2).item()
            if layer.beta is not None
            else 0.0
        )

        gamma_g = (
            torch.norm(layer.gamma.grad.detach().cpu(), p=2).item()
            if layer.gamma is not None and layer.gamma.grad is not None
            else 0.0
        )
        beta_g = (
            torch.norm(layer.beta.grad.detach().cpu(), p=2).item()
            if layer.beta is not None and layer.beta.grad is not None
            else 0.0
        )

        return LayerNormStats(
            d_model=layer.d_model,
            eps=layer.eps,
            mean_value=round(mean_val, 6),
            mean_variance=round(var_val, 6),
            gamma_norm=round(gamma_n, 4),
            beta_norm=round(beta_n, 4),
            gamma_grad_norm=round(gamma_g, 6),
            beta_grad_norm=round(beta_g, 6),
        )
