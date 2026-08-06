"""TorchDynamo JIT Graph Compiler Wrapper (torch.compile) for Aura EXP-008.

Provides TorchCompileManager for JIT kernel fusion and launch overhead reduction.
"""

import logging
from typing import Optional
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class TorchCompileManager:
    """Manages JIT graph compilation via PyTorch torch.compile (TorchDynamo / Inductor)."""

    @staticmethod
    def is_compile_supported() -> bool:
        """Checks if PyTorch version supports torch.compile."""
        return hasattr(torch, "compile")

    @classmethod
    def compile_model(
        cls,
        model: nn.Module,
        mode: str = "reduce-overhead",
        fullgraph: bool = False,
        disable: bool = False,
    ) -> nn.Module:
        """Compiles PyTorch model into optimized JIT execution graph.

        Args:
            model: Input PyTorch nn.Module.
            mode: Compilation mode ("default", "reduce-overhead", "max-autotune").
            fullgraph: If True, requires complete graph capture without Python fallbacks.
            disable: If True, bypasses compilation.

        Returns:
            Optimized compiled model (or uncompiled model fallback).
        """
        if disable or not cls.is_compile_supported():
            logger.info("torch.compile disabled or unsupported. Returning eager-mode model.")
            return model

        try:
            logger.info("Compiling model graph using torch.compile (mode=%s)...", mode)
            compiled_model = torch.compile(model, mode=mode, fullgraph=fullgraph)
            return compiled_model
        except Exception as e:
            logger.warning("torch.compile graph capture failed: %s. Falling back to eager mode.", e)
            return model
