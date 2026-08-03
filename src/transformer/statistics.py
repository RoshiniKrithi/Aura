"""Residual Connection Statistics Extractor for Aura LLM Pipeline.

Computes activation L2 norms for identity input, sub-layer output, and combined residual output,
as well as residual ratio metrics.
"""

from dataclasses import dataclass
import logging

from src.transformer.residual import ResidualConnection

logger = logging.getLogger(__name__)


@dataclass
class ResidualStats:
    """Summary container holding quantitative Residual Connection statistics."""

    d_model: int
    dropout_rate: float
    input_norm: float
    sub_layer_norm: float
    output_norm: float
    residual_ratio: float


class ResidualStatistics:
    """Computes statistical metrics over ResidualConnection activations.

    Time Complexity:
        O(1) scalar operations on cached L2 norms.

    Space Complexity:
        O(1) scalar memory overhead.
    """

    @staticmethod
    def compute_stats(layer: ResidualConnection) -> ResidualStats:
        """Calculates quantitative metrics for given ResidualConnection module.

        Args:
            layer: Instantiated ResidualConnection module.

        Returns:
            ResidualStats summary object.

        Raises:
            ValueError: If layer has not executed forward pass yet.
        """
        if (
            layer.last_x_norm is None
            or layer.last_sub_norm is None
            or layer.last_out_norm is None
        ):
            raise ValueError(
                "ResidualConnection layer has no cached statistics. Run forward pass first."
            )

        ratio = (
            layer.last_sub_norm / (layer.last_x_norm + 1e-8)
            if layer.last_x_norm is not None
            else 0.0
        )

        return ResidualStats(
            d_model=layer.d_model,
            dropout_rate=layer.dropout_rate,
            input_norm=round(layer.last_x_norm, 4),
            sub_layer_norm=round(layer.last_sub_norm, 4),
            output_norm=round(layer.last_out_norm, 4),
            residual_ratio=round(ratio, 4),
        )
