"""Loss Subsystem Configuration Schema for Aura LLM Architecture.

Provides structured parameters for ignore_index padding masks, label smoothing,
loss reduction methods, and metrics calculation preferences.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class CrossEntropyLossConfig:
    """Hyperparameter configuration container for CrossEntropyLoss module.

    Attributes:
        ignore_index: Target token ID to ignore during loss computation (default: -1).
        label_smoothing: Float label smoothing epsilon in range [0.0, 1.0) (default: 0.0).
        reduction: Reduction method applied to final loss ("mean", "sum", "none", default: "mean").
        compute_accuracy: If True, computes next-token prediction accuracy metric (default: True).
        compute_perplexity: If True, computes exponentiated perplexity metric (default: True).
    """

    ignore_index: int = -1
    label_smoothing: float = 0.0
    reduction: str = "mean"
    compute_accuracy: bool = True
    compute_perplexity: bool = True
