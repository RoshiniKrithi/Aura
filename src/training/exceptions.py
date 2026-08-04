"""Custom Domain Exceptions for Training Engine Subsystem.

Provides structured error handling for engine configuration errors, invalid input data,
and execution failures during training or validation loops.
"""


class EngineError(Exception):
    """Base exception for all training engine subsystem errors."""

    pass


class EngineValidationError(EngineError):
    """Raised when engine inputs or configurations fail validation checks."""

    pass


class EngineConfigError(EngineError):
    """Raised when engine hyperparameter configuration parameters are invalid."""

    pass
