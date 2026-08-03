"""Custom Exceptions for Aura Transformer Subsystem.

Provides structured error handling for residual calculations, transformer blocks,
tensor shape mismatches, configuration errors, and numerical instabilities (NaN/Inf).
"""


class ResidualError(Exception):
    """Base exception for all Residual Connection errors."""

    pass


class ResidualValidationError(ResidualError):
    """Raised when input tensor shape, feature dimension, or value validation fails (e.g. NaN/Inf)."""

    pass


class ResidualConfigError(ResidualError):
    """Raised when Residual Connection configuration parameters are invalid."""

    pass


class TransformerBlockError(Exception):
    """Base exception for all Transformer Block errors."""

    pass


class TransformerBlockValidationError(TransformerBlockError):
    """Raised when Transformer Block input tensor shape or value validation fails."""

    pass


class TransformerBlockConfigError(TransformerBlockError):
    """Raised when Transformer Block configuration parameters are invalid."""

    pass
