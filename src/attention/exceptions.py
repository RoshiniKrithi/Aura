"""Custom Exceptions for Aura Attention Pipeline.

Provides structured error handling for attention score calculation, tensor shape mismatches,
numerical instabilities (NaN/Inf), and configuration errors.
"""


class AttentionError(Exception):
    """Base exception for all attention module errors."""

    pass


class AttentionValidationError(AttentionError):
    """Raised when input tensor shape, dtype, or value validation fails (e.g. NaN/Inf, non-3D tensor)."""

    pass


class AttentionConfigError(AttentionError):
    """Raised when attention configuration options are invalid."""

    pass
