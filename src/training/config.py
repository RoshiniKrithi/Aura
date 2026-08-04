"""Training Engine Configuration Schema for Aura LLM Architecture.

Provides parameters for epoch counts, evaluation intervals, checkpoint frequencies,
Automatic Mixed Precision (AMP) settings, seed reproducibility, and device placement.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class TrainingEngineConfig:
    """Hyperparameter configuration container for TrainingEngine.

    Attributes:
        epochs: Total number of training epochs (default: 10).
        eval_interval: Epoch frequency for validation evaluation (default: 1).
        checkpoint_interval: Epoch frequency for saving checkpoints (default: 1).
        amp_enabled: If True, enables Automatic Mixed Precision (default: False).
        amp_dtype: Mixed precision float type ("bfloat16" or "float16", default: "bfloat16").
        max_steps: Maximum total training micro-steps override (optional).
        device: Target execution device ("auto", "cpu", "cuda", "mps").
        seed: Random seed for reproducibility (default: 42).
        checkpoint_dir: Directory path for saving checkpoints (default: "checkpoints").
    """

    epochs: int = 10
    eval_interval: int = 1
    checkpoint_interval: int = 1
    amp_enabled: bool = False
    amp_dtype: str = "bfloat16"
    max_steps: Optional[int] = None
    device: str = "auto"
    seed: int = 42
    checkpoint_dir: str = "checkpoints"
