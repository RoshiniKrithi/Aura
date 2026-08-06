"""Reflection and Bug Diagnosis Engine for Aura Agent (EXP-009).

Parses execution stderr output, stack traces, and PyTest failures,
formulating self-correction guidance prompts for subsequent reasoning steps.
"""

from typing import Dict, Optional
import re
import logging

logger = logging.getLogger(__name__)


class ReflectionEngine:
    """Engine diagnosing runtime errors and formulating self-corrections."""

    @staticmethod
    def diagnose_failure(tool_name: str, error_msg: str, output: str) -> str:
        """Diagnoses tool failure and generates corrective recommendation.

        Args:
            tool_name: Name of tool that failed.
            error_msg: Error message or stderr string.
            output: Combined tool output stdout/stderr string.

        Returns:
            String containing diagnostic summary and suggested fix.
        """
        combined = f"{error_msg}\n{output}"

        if "ModuleNotFoundError" in combined or "ImportError" in combined or "No module named" in combined:
            module = re.search(r"No module named '([^']+)'", combined)
            mod_name = module.group(1) if module else "missing_module"
            return (
                f"Reflection: ModuleNotFoundError detected for module '{mod_name}'. "
                f"Verify Python import paths or use standard library dependencies."
            )

        if "SyntaxError" in combined or "IndentationError" in combined:
            return (
                "Reflection: Syntax or Indentation error detected in generated code. "
                "Inspect line numbers and ensure correct block formatting."
            )

        if "AssertionError" in combined or "FAILED" in combined:
            return (
                "Reflection: PyTest test assertion failure detected. "
                "Analyze expected vs actual test outputs and adjust implementation logic."
            )

        if "FileNotFoundError" in combined:
            return (
                "Reflection: FileNotFoundError detected. Use workspace_search or file_reader "
                "to verify exact file paths in directory."
            )

        if "Timed out" in combined:
            return (
                "Reflection: Execution timed out. Optimize algorithm complexity or "
                "break code execution into smaller benchmark chunks."
            )

        return f"Reflection: Execution failed for tool '{tool_name}'. Error details: {error_msg[:150]}"
