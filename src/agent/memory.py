"""Multi-Tiered Memory Management Engine for Aura Agent (EXP-009).

Manages:
- Short-Term Memory: Active ReAct prompt, recent thoughts, actions, and observations
- Long-Term Memory: Persistent project knowledge and workspace summaries
- Tool Execution History: Record of tool invocation names, arguments, and outcomes
- Reflection History: Stack trace diagnoses and bug-fix lessons learned
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class MemoryRecord:
    """Represents a single step in agent working memory."""

    step_index: int
    thought: str
    action: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None
    reflection: Optional[str] = None


class MemoryManager:
    """Manages multi-tiered agent short and long-term working memory."""

    def __init__(self, max_memory_tokens: int = 4096) -> None:
        """Initializes MemoryManager.

        Args:
            max_memory_tokens: Maximum token ceiling for working context.
        """
        self.max_memory_tokens = max_memory_tokens
        self.records: List[MemoryRecord] = []
        self.long_term_knowledge: Dict[str, str] = {}
        self.reflection_history: List[str] = []

    def add_step(
        self,
        step_index: int,
        thought: str,
        action: Optional[str] = None,
        tool_args: Optional[Dict[str, Any]] = None,
        observation: Optional[str] = None,
        reflection: Optional[str] = None,
    ) -> MemoryRecord:
        """Appends a new step record to short-term memory."""
        rec = MemoryRecord(
            step_index=step_index,
            thought=thought,
            action=action,
            tool_args=tool_args,
            observation=observation,
            reflection=reflection,
        )
        self.records.append(rec)
        logger.info("Recorded memory step %d.", step_index)
        return rec

    def add_reflection(self, reflection_str: str) -> None:
        """Appends a lesson learned to reflection history."""
        self.reflection_history.append(reflection_str)
        logger.info("Recorded reflection insight: %s", reflection_str[:60])

    def get_working_context_prompt(self, max_steps: int = 5) -> str:
        """Formats recent step records into a prompt string for LLM ReAct steps."""
        recent = self.records[-max_steps:] if len(self.records) > max_steps else self.records
        context_parts = []

        if self.reflection_history:
            context_parts.append("=== Lessons Learned / Reflection History ===")
            for r in self.reflection_history[-3:]:
                context_parts.append(f"- {r}")

        context_parts.append("\n=== Execution Step History ===")
        for rec in recent:
            context_parts.append(f"Step {rec.step_index}:")
            context_parts.append(f"Thought: {rec.thought}")
            if rec.action:
                context_parts.append(f"Action: {rec.action}({rec.tool_args})")
            if rec.observation:
                context_parts.append(f"Observation: {rec.observation[:300]}")
            if rec.reflection:
                context_parts.append(f"Reflection: {rec.reflection}")
            context_parts.append("")

        return "\n".join(context_parts)

    def clear(self) -> None:
        """Clears working short-term memory."""
        self.records.clear()
        self.reflection_history.clear()
        logger.info("Cleared Agent memory.")
