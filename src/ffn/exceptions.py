"""Custom Exceptions for Aura Feed Forward Network (FFN) Pipeline.

Provides structured error handling for FFN linear layers, activation choices,
dimension validation, and numerical stability errors.
"""


class FeedForwardError(Exception):
    """Base exception for all Feed Forward Network errors."""

    pass


class FeedForwardValidationError(FeedForwardError):
    """Raised when tensor shape, feature dimension, or value validation fails (e.g. NaN/Inf)."""

    pass


class FeedForwardConfigError(FeedForwardError):
    """Raised when FFN configuration options (e.g. unknown activation) are invalid."""

    pass
