"""Optimizer Factory API for Aura LLM Architecture.

Provides central factory methods for constructing PyTorch optimizers (AdamW, SGD, Adam)
with intelligent parameter group filtering separating 2D weights from 1D biases and LayerNorms.
"""

import logging
from typing import Any, List, Optional, Union
import torch
import torch.nn as nn
from torch.optim import SGD, Adam, AdamW, Optimizer

from src.optimizers.config import OptimizerConfig

logger = logging.getLogger(__name__)


class OptimizerFactory:
    """Central factory for constructing and configuring PyTorch Optimizers."""

    @staticmethod
    def prepare_parameter_groups(
        model: nn.Module, weight_decay: float = 0.1, filter_weight_decay: bool = True
    ) -> List[dict]:
        """Separates model parameters into 2D decay groups and 1D non-decay groups.

        Design Decision:
            - 2D weight matrices (linear projections, embeddings) receive weight_decay.
            - 1D parameter vectors (biases, LayerNorm gamma/beta) receive weight_decay = 0.0.

        Args:
            model: PyTorch nn.Module instance.
            weight_decay: Decoupled weight decay coefficient.
            filter_weight_decay: If True, filters 1D parameters out of weight decay.

        Returns:
            List of parameter group dictionaries for PyTorch Optimizer.
        """
        if not filter_weight_decay or weight_decay == 0.0:
            return [{"params": [p for p in model.parameters() if p.requires_grad], "weight_decay": weight_decay}]

        decay_params = []
        no_decay_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            # 1D parameters (biases, norm scale/shift parameters) or scalar params do not receive weight decay
            if param.ndim < 2 or "bias" in name or "ln_" in name or "norm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        logger.info(
            "Prepared parameter groups for weight decay: decay_group_params=%d, no_decay_group_params=%d",
            sum(p.numel() for p in decay_params),
            sum(p.numel() for p in no_decay_params),
        )

        return param_groups

    @classmethod
    def create_optimizer(
        cls,
        model: nn.Module,
        config: Optional[Union[OptimizerConfig, Any]] = None,
        name: Optional[str] = None,
        lr: Optional[float] = None,
        weight_decay: Optional[float] = None,
    ) -> Optimizer:
        """Instantiates and configures a PyTorch Optimizer for given model.

        Args:
            model: Target PyTorch nn.Module.
            config: Optional OptimizerConfig or AppConfig object.
            name: Optional optimizer name override ("adamw", "sgd", "adam").
            lr: Optional learning rate override.
            weight_decay: Optional weight decay override.

        Returns:
            Instantiated PyTorch Optimizer instance.
        """
        from src.utils.config import AppConfig

        if isinstance(config, AppConfig):
            cfg = OptimizerConfig(
                name=name or config.optimizer.name,
                lr=lr or config.optimizer.learning_rate,
                weight_decay=weight_decay if weight_decay is not None else config.optimizer.weight_decay,
                beta1=config.optimizer.adam_beta1,
                beta2=config.optimizer.adam_beta2,
                eps=config.optimizer.adam_eps,
            )
        elif isinstance(config, OptimizerConfig):
            cfg = config
        else:
            cfg = OptimizerConfig(
                name=name or "adamw",
                lr=lr or 3e-4,
                weight_decay=weight_decay if weight_decay is not None else 0.1,
            )

        param_groups = cls.prepare_parameter_groups(
            model=model,
            weight_decay=cfg.weight_decay,
            filter_weight_decay=cfg.filter_weight_decay,
        )

        opt_name = cfg.name.lower().strip()

        if "adamw" in opt_name:
            optimizer = AdamW(
                param_groups,
                lr=cfg.lr,
                betas=(cfg.beta1, cfg.beta2),
                eps=cfg.eps,
            )
        elif "sgd" in opt_name:
            optimizer = SGD(param_groups, lr=cfg.lr, momentum=cfg.beta1)
        elif "adam" in opt_name:
            optimizer = Adam(
                param_groups,
                lr=cfg.lr,
                betas=(cfg.beta1, cfg.beta2),
                eps=cfg.eps,
            )
        else:
            optimizer = AdamW(
                param_groups,
                lr=cfg.lr,
                betas=(cfg.beta1, cfg.beta2),
                eps=cfg.eps,
            )

        logger.info("Instantiated Optimizer '%s' with lr=%.6f, weight_decay=%.4f", opt_name, cfg.lr, cfg.weight_decay)
        return optimizer
