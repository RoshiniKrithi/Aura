"""Automatic Mixed Precision (AMP FP16/BF16) Manager for Aura EXP-008.

Provides MixedPrecisionManager for high-throughput FP16/BF16 autocast execution.
"""

import contextlib
import logging
from typing import Any, Generator, Optional
import torch

logger = logging.getLogger(__name__)


class MixedPrecisionManager:
    """Manages PyTorch Automatic Mixed Precision (AMP FP16/BF16) autocast and GradScaler."""

    @staticmethod
    def resolve_dtype(dtype_str: str) -> torch.dtype:
        """Resolves precision string ("float16", "bfloat16", "float32") to torch.dtype."""
        dt = dtype_str.lower().strip()
        if dt in ("float16", "fp16"):
            return torch.float16
        if dt in ("bfloat16", "bf16"):
            return torch.bfloat16
        return torch.float32

    @classmethod
    def get_autocast_context(
        cls,
        enabled: bool = True,
        device_type: str = "cuda",
        dtype_str: str = "float16",
    ) -> Generator[None, None, None]:
        """Returns PyTorch autocast context manager for mixed precision execution.

        Args:
            enabled: If True, enables autocast.
            device_type: Device target ("cuda" or "cpu").
            dtype_str: Precision string tag ("float16" or "bfloat16").

        Returns:
            Context manager context.
        """
        if not enabled:
            return contextlib.nullcontext()

        target_dtype = cls.resolve_dtype(dtype_str)
        dev = "cuda" if "cuda" in device_type.lower() and torch.cuda.is_available() else "cpu"

        logger.debug("Entering AMP autocast context: device=%s, dtype=%s", dev, target_dtype)
        return torch.amp.autocast(device_type=dev, dtype=target_dtype)

    @staticmethod
    def get_scaler(enabled: bool = True, device_type: str = "cuda") -> Optional[torch.amp.GradScaler]:
        """Returns PyTorch GradScaler for loss scaling during FP16 training."""
        if enabled and "cuda" in device_type.lower() and torch.cuda.is_available():
            return torch.amp.GradScaler("cuda")
        return None
