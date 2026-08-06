"""Agentic Programming Assistant Configuration for Aura EXP-009.

Defines AgentConfig dataclass containing execution limits, reasoning parameters,
sandboxing bounds, and tool permission settings.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class AgentConfig:
    """Configuration container for Aura Agentic Programming Assistant.

    Attributes:
        max_reasoning_steps: Maximum ReAct reasoning loop iterations.
        execution_timeout_sec: Maximum timeout in seconds per tool execution.
        planning_depth: Maximum sub-task decomposition depth.
        max_memory_tokens: Token ceiling for short-term working context.
        enable_reflection: If True, executes self-correction on execution failures.
        enable_git_tools: If True, enables Git status and diff tool actions.
        workspace_root: Absolute or relative root directory for file sandboxing.
        allowed_tools: List of active tool identifier names.
        temperature: Sampling temperature for agent reasoning generation.
        output_dir: Directory path for exporting agent execution session logs.
    """

    max_reasoning_steps: int = 20
    execution_timeout_sec: float = 30.0
    planning_depth: int = 5
    max_memory_tokens: int = 4096
    enable_reflection: bool = True
    enable_git_tools: bool = True
    workspace_root: str = "."
    temperature: float = 0.2
    output_dir: str = "outputs/experiments/EXP-009_Agent_v1.0"
    allowed_tools: List[str] = field(
        default_factory=lambda: [
            "python_repl",
            "file_reader",
            "file_writer",
            "workspace_search",
            "ast_parser",
            "unit_test_runner",
            "git_ops",
            "rag_search",
        ]
    )
