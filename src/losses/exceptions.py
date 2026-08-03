"""Custom Domain Exceptions for Loss Subsystem.

Provides structured error handling for input tensor shape mismatches, NaN/Inf loss values,
and invalid loss configuration parameters.
"""


class LossError(Exception):
    """Base exception for all loss subsystem errors."""

    pass


class LossValidationError(LossError):
    """Raised when loss input logits or targets fail validation checks."""

    pass


class LossConfigError(LossError):
    """Raised when loss configuration parameters are invalid."""

    pass
