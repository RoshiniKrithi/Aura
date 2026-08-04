"""Inference Engine Validator for Aura LLM Architecture.

Sanity checks model, tokenizer, and generation prompt parameters before inference.
"""

from dataclasses import dataclass, field
import logging
from typing import Any, List
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class InferenceValidationResult:
    """Summary diagnostic report from InferenceEngine validation checks."""

    is_valid: bool
    errors: List[str] = field(default_factory=list)


class InferenceValidator:
    """Validates setup and input prompts for InferenceEngine."""

    @staticmethod
    def validate_setup(model: nn.Module, tokenizer: Any) -> InferenceValidationResult:
        """Validates model and tokenizer instances.

        Args:
            model: PyTorch nn.Module instance.
            tokenizer: Tokenizer object instance with encode/decode methods.

        Returns:
            InferenceValidationResult object.
        """
        result = InferenceValidationResult(is_valid=True)

        if not isinstance(model, nn.Module):
            result.is_valid = False
            result.errors.append(f"model must be a PyTorch nn.Module, got {type(model).__name__}")

        if not hasattr(tokenizer, "encode") or not hasattr(tokenizer, "decode"):
            result.is_valid = False
            result.errors.append(
                f"tokenizer must implement encode() and decode() methods, got {type(tokenizer).__name__}"
            )

        return result
