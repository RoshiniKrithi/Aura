"""Custom Domain Exceptions for Optimization Subsystem.

Provides structured error handling for optimizer configuration errors,
invalid gradient updates, and numerical instabilities.
"""


class OptimizerError(Exception):
    """Base exception for all optimization subsystem errors."""

    pass


class OptimizerValidationError(OptimizerError):
    """Raised when optimizer or gradient validation checks fail."""

    pass


class OptimizerConfigError(OptimizerError):
    """Raised when optimizer hyperparameter configuration parameters are invalid."""

    pass
