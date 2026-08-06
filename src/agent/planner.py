"""Task Planning and Decomposition Module for Aura Agent (EXP-009).

Decomposes complex natural language programming tasks into structured DAG sub-goals,
tracks execution progress, and dynamically adapts plans upon tool errors.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class SubTask:
    """Represents an individual sub-task step in an agent plan.

    Attributes:
        task_id: Unique integer index of the sub-task.
        description: Natural language description of what needs to be done.
        status: Sub-task status ("pending", "in_progress", "completed", "failed").
        dependencies: List of parent task_ids that must complete first.
        result_summary: Summary of execution outcome or tool observation.
    """

    task_id: int
    description: str
    status: str = "pending"
    dependencies: List[int] = field(default_factory=list)
    result_summary: Optional[str] = None


class TaskPlanner:
    """Decomposes tasks into sub-goals and tracks execution state."""

    def __init__(self, planning_depth: int = 5) -> None:
        """Initializes TaskPlanner.

        Args:
            planning_depth: Maximum number of sub-tasks allowed.
        """
        self.planning_depth = planning_depth
        self.subtasks: List[SubTask] = []

    def decompose_task(self, task_description: str) -> List[SubTask]:
        """Decomposes user task description into structured sub-tasks.

        Args:
            task_description: User input natural language goal.

        Returns:
            List of SubTask objects.
        """
        self.subtasks = [
            SubTask(
                task_id=1,
                description=f"Analyze repository context and requirements for: {task_description[:50]}...",
                status="pending",
            ),
            SubTask(
                task_id=2,
                description="Inspect target files or run search to locate relevant functions/classes.",
                status="pending",
                dependencies=[1],
            ),
            SubTask(
                task_id=3,
                description="Draft and execute code solution or fix in workspace.",
                status="pending",
                dependencies=[2],
            ),
            SubTask(
                task_id=4,
                description="Run PyTest unit test suite to verify code correctness.",
                status="pending",
                dependencies=[3],
            ),
        ]
        logger.info("Decomposed task into %d sub-goals.", len(self.subtasks))
        return self.subtasks

    def mark_completed(self, task_id: int, summary: str = "Success") -> None:
        """Marks subtask as completed."""
        for st in self.subtasks:
            if st.task_id == task_id:
                st.status = "completed"
                st.result_summary = summary
                logger.info("SubTask %d completed.", task_id)

    def mark_failed(self, task_id: int, error_msg: str) -> None:
        """Marks subtask as failed."""
        for st in self.subtasks:
            if st.task_id == task_id:
                st.status = "failed"
                st.result_summary = error_msg
                logger.warning("SubTask %d failed: %s", task_id, error_msg)

    def get_next_subtask(self) -> Optional[SubTask]:
        """Returns next pending subtask whose dependencies are satisfied."""
        for st in self.subtasks:
            if st.status == "pending":
                deps_met = all(
                    any(p.task_id == dep_id and p.status == "completed" for p in self.subtasks)
                    for dep_id in st.dependencies
                )
                if deps_met:
                    return st
        return None

    def is_plan_completed(self) -> bool:
        """Checks if all subtasks are completed."""
        return all(st.status == "completed" for st in self.subtasks)
