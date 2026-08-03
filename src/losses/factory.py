"""Loss Subsystem Factory API for Aura Architecture.

Provides factory builder methods for constructing CrossEntropyLoss instances
from CrossEntropyLossConfig or AppConfig.
"""

import logging
from typing import Any, Optional, Union

from src.losses.config import CrossEntropyLossConfig
from src.losses.cross_entropy import CrossEntropyLoss

logger = logging.getLogger(__name__)


class LossFactory:
    """Central factory for constructing and configuring loss modules."""

    @classmethod
    def create_loss(
        cls,
        config: Optional[Union[CrossEntropyLossConfig, Any]] = None,
        ignore_index: Optional[int] = None,
        label_smoothing: Optional[float] = None,
    ) -> CrossEntropyLoss:
        """Instantiates and configures a CrossEntropyLoss module.

        Args:
            config: Optional CrossEntropyLossConfig or AppConfig object.
            ignore_index: Optional ignore_index override.
            label_smoothing: Optional label_smoothing override.

        Returns:
            Instantiated CrossEntropyLoss module.
        """
        from src.utils.config import AppConfig

        if isinstance(config, AppConfig):
            cfg = CrossEntropyLossConfig(
                ignore_index=ignore_index if ignore_index is not None else -1,
                label_smoothing=label_smoothing or 0.0,
            )
        elif isinstance(config, CrossEntropyLossConfig):
            cfg = config
        else:
            cfg = CrossEntropyLossConfig(
                ignore_index=ignore_index if ignore_index is not None else -1,
                label_smoothing=label_smoothing or 0.0,
            )

        return CrossEntropyLoss(config=cfg)
