"""Exception Hierarchy for Aura Agentic Programming Assistant (EXP-009)."""


class AgentException(Exception):
    """Base exception class for all Agentic Programming Assistant failures."""
    pass


class ToolExecutionError(AgentException):
    """Raised when a tool execution fails or times out."""
    pass


class AgentPlanningError(AgentException):
    """Raised when task decomposition or DAG planning fails."""
    pass


class MaxStepsExceededError(AgentException):
    """Raised when reasoning steps exceed max_reasoning_steps limit."""
    pass


class AgentValidationError(AgentException):
    """Raised when safety, permission, or input validation fails."""
    pass
