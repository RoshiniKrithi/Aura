"""Feed Forward Network Weight Initializer for Aura LLM Architecture.

Provides configurable parameter weight initializations for linear layers matching
GPT-2, GPT-3, LLaMA, and PyTorch production standards.
"""

import logging
import math
import torch
import torch.nn as nn

from src.ffn.config import FeedForwardConfig
from src.ffn.exceptions import FeedForwardConfigError

logger = logging.getLogger(__name__)


class FeedForwardInitializer:
    """Weight initializer engine for FeedForwardNetwork linear layers.

    Design Decisions:
        - Supports Normal, Uniform, Xavier (Glorot), Kaiming (He), and Truncated Normal initializers.
        - Zeroes bias vectors by default matching transformer conventions.
    """

    @classmethod
    def initialize_weights(
        cls,
        module: nn.Module,
        config: FeedForwardConfig,
    ) -> None:
        """Initializes weight and bias parameters of a FeedForwardNetwork module.

        Args:
            module: Target PyTorch nn.Module container.
            config: FeedForwardConfig dataclass specifying strategy and bounds.
        """
        init_type = config.initializer.lower()

        for name, param in module.named_parameters():
            if "weight" in name and param.dim() >= 2:
                if init_type == "normal":
                    nn.init.normal_(param, mean=0.0, std=config.init_std)
                elif init_type == "uniform":
                    nn.init.uniform_(param, a=-config.init_range, b=config.init_range)
                elif init_type == "xavier_uniform":
                    nn.init.xavier_uniform_(param)
                elif init_type == "xavier_normal":
                    nn.init.xavier_normal_(param)
                elif init_type == "kaiming_uniform":
                    nn.init.kaiming_uniform_(param, a=math.sqrt(5))
                elif init_type == "kaiming_normal":
                    nn.init.kaiming_normal_(param)
                elif init_type == "truncated_normal":
                    nn.init.trunc_normal_(param, mean=0.0, std=config.init_std, a=-2 * config.init_std, b=2 * config.init_std)
                else:
                    raise FeedForwardConfigError(f"Unsupported initializer type: {config.initializer}")
            elif "bias" in name and param is not None:
                nn.init.zeros_(param)

        logger.info("Initialized FeedForwardNetwork parameters using '%s' strategy", init_type)
