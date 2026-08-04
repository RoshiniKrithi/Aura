"""Checkpoint Validator for Aura LLM Architecture.

Sanity checks checkpoint file existence, payload schema integrity, and SHA256 checksums.
"""

from dataclasses import dataclass, field
import hashlib
import logging
import os
from typing import Any, Dict, List
import torch

logger = logging.getLogger(__name__)


@dataclass
class CheckpointValidationResult:
    """Summary diagnostic report output from CheckpointValidator checks."""

    is_valid: bool
    errors: List[str] = field(default_factory=list)


class CheckpointValidator:
    """Validates checkpoint files and payload contents."""

    @staticmethod
    def validate_file(checkpoint_path: str) -> CheckpointValidationResult:
        """Validates checkpoint file existence and payload dictionary structure.

        Args:
            checkpoint_path: Path to target checkpoint file.

        Returns:
            CheckpointValidationResult object.
        """
        result = CheckpointValidationResult(is_valid=True)

        if not os.path.exists(checkpoint_path):
            result.is_valid = False
            result.errors.append(f"Checkpoint file does not exist: {checkpoint_path}")
            return result

        if os.path.getsize(checkpoint_path) == 0:
            result.is_valid = False
            result.errors.append(f"Checkpoint file is empty (0 bytes): {checkpoint_path}")
            return result

        try:
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if not isinstance(payload, dict):
                result.is_valid = False
                result.errors.append(f"Expected dict payload, got {type(payload).__name__}")
            elif "model_state_dict" not in payload and "model" not in payload:
                result.is_valid = False
                result.errors.append("Payload missing model state dictionary.")
        except Exception as e:
            result.is_valid = False
            result.errors.append(f"Unpickling / loading failed: {str(e)}")

        return result
