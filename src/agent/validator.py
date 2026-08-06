"""Validation Suite for Aura Agentic Programming Assistant (EXP-009).

Validates safety constraints, tool execution permissions, path traversal bounds,
and agent parameter schemas.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional
import os

from src.agent.config import AgentConfig


@dataclass
class AgentValidationResult:
    """Encapsulates validation output status."""

    is_valid: bool
    errors: List[str] = field(default_factory=list)


class AgentValidator:
    """Safety and permission validator for Agent operations."""

    @staticmethod
    def validate_config(config: AgentConfig) -> AgentValidationResult:
        """Validates AgentConfig parameter integrity.

        Returns:
            AgentValidationResult object.
        """
        errors = []
        if config.max_reasoning_steps <= 0:
            errors.append(f"max_reasoning_steps must be > 0, got {config.max_reasoning_steps}")
        if config.execution_timeout_sec <= 0:
            errors.append(f"execution_timeout_sec must be > 0, got {config.execution_timeout_sec}")
        if config.max_memory_tokens <= 0:
            errors.append(f"max_memory_tokens must be > 0, got {config.max_memory_tokens}")

        return AgentValidationResult(is_valid=len(errors) == 0, errors=errors)

    @staticmethod
    def validate_file_path(path_str: str, workspace_root: str = ".") -> AgentValidationResult:
        """Validates file path to prevent directory traversal outside workspace_root.

        Returns:
            AgentValidationResult object.
        """
        errors = []
        try:
            root = Path(workspace_root).resolve()
            target = Path(path_str).resolve()

            # Path safety check
            if not str(target).startswith(str(root)):
                errors.append(f"Path '{path_str}' escapes sandboxed workspace_root '{root}'")
        except Exception as e:
            errors.append(f"Invalid path syntax '{path_str}': {str(e)}")

        return AgentValidationResult(is_valid=len(errors) == 0, errors=errors)

    @staticmethod
    def validate_tool_permission(tool_name: str, allowed_tools: List[str]) -> AgentValidationResult:
        """Validates if tool_name is registered in allowed_tools.

        Returns:
            AgentValidationResult object.
        """
        errors = []
        if tool_name not in allowed_tools:
            errors.append(f"Tool '{tool_name}' is not in allowed_tools list {allowed_tools}")

        return AgentValidationResult(is_valid=len(errors) == 0, errors=errors)
