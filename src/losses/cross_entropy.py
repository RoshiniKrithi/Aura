"""Production-Grade Cross-Entropy Loss Implementation for Aura LLM Architecture.

Computes sequence-shifted Cross-Entropy Loss, ignores padding tokens (ignore_index = -1),
and extracts Next-Token Accuracy and Perplexity (PPL = exp(loss)) metrics.
"""

import math
import logging
from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.config import CrossEntropyLossConfig
from src.losses.exceptions import LossValidationError
from src.losses.validator import LossValidator

logger = logging.getLogger(__name__)


class CrossEntropyLoss(nn.Module):
    """Production-grade Autoregressive Cross-Entropy Loss Module.

    Design Decisions:
        - Performs sequence label shifting: logits[:, :-1, :] vs targets[:, 1:].
        - Masking padding tokens using ignore_index (default: -1).
        - Computes Next-Token Accuracy and Perplexity (PPL = exp(loss)).

    Time Complexity:
        O(B * (T - 1) * V) cross-entropy evaluation.

    Space Complexity:
        O(1) memory overhead beyond PyTorch autograd loss graph.
    """

    def __init__(self, config: Optional[CrossEntropyLossConfig] = None) -> None:
        """Initializes CrossEntropyLoss module.

        Args:
            config: Optional CrossEntropyLossConfig hyperparameter object.
        """
        super().__init__()

        cfg = config or CrossEntropyLossConfig()

        self.config = cfg
        self.ignore_index = cfg.ignore_index
        self.label_smoothing = cfg.label_smoothing
        self.reduction = cfg.reduction
        self.compute_accuracy = cfg.compute_accuracy
        self.compute_perplexity = cfg.compute_perplexity

        self.validator = LossValidator()

        logger.info(
            "Instantiated CrossEntropyLoss: ignore_index=%d, label_smoothing=%.4f, reduction=%s",
            self.ignore_index,
            self.label_smoothing,
            self.reduction,
        )

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Forward pass computing sequence-shifted Cross-Entropy Loss and metrics.

        Args:
            logits: Unnormalized Logits FloatTensor of shape (B, T, V) or (T, V).
            targets: Ground-truth target token ID LongTensor of shape (B, T) or (T,).

        Returns:
            Tuple of (scalar_loss_tensor, metrics_dict).

        Raises:
            LossValidationError: If input shape or tensor validation checks fail.
        """
        val_res = self.validator.validate_inputs(logits, targets)
        if not val_res.is_valid:
            raise LossValidationError(f"Loss input validation failed: {val_res.errors}")

        # 1. Sequence Label Shifting
        # Logits at position t predict target at position t+1
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = targets[..., 1:].contiguous()

        vocab_size = shift_logits.size(-1)

        # 2. Flatten Tensors for Cross Entropy Evaluation
        flat_logits = shift_logits.view(-1, vocab_size)
        flat_labels = shift_labels.view(-1)

        # 3. Compute Differentiable PyTorch Cross Entropy Loss
        loss = F.cross_entropy(
            flat_logits,
            flat_labels,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
            reduction=self.reduction,
        )

        # 4. Extract Diagnostics & Evaluation Metrics
        loss_val = loss.item()
        metrics: Dict[str, float] = {"loss": round(loss_val, 6)}

        if self.compute_accuracy and flat_labels.numel() > 0:
            with torch.no_grad():
                valid_mask = flat_labels != self.ignore_index
                if valid_mask.any():
                    preds = flat_logits.argmax(dim=-1)
                    correct = (preds[valid_mask] == flat_labels[valid_mask]).float()
                    acc = correct.mean().item()
                    metrics["accuracy"] = round(acc * 100.0, 2)
                else:
                    metrics["accuracy"] = 0.0

        if self.compute_perplexity:
            with torch.no_grad():
                try:
                    # Clamp loss to prevent exp overflow
                    clamped_loss = min(loss_val, 100.0)
                    ppl = math.exp(clamped_loss)
                    metrics["perplexity"] = round(ppl, 4)
                except OverflowError:
                    metrics["perplexity"] = float("inf")

        return loss, metrics

    def extra_repr(self) -> str:
        """Provides readable PyTorch module representation."""
        return (
            f"ignore_index={self.ignore_index}, label_smoothing={self.label_smoothing}, "
            f"reduction='{self.reduction}'"
        )
