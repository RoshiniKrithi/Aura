"""Custom Exceptions for Aura Layer Normalization Pipeline.

Provides structured error handling for normalization calculations, tensor shape mismatches,
numerical instabilities (NaN/Inf), and configuration errors.
"""


class LayerNormError(Exception):
    """Base exception for all Layer Normalization errors."""

    pass


class LayerNormValidationError(LayerNormError):
    """Raised when input tensor shape, feature dimension, or value validation fails (e.g. NaN/Inf)."""

    pass


class LayerNormConfigError(LayerNormError):
    """Raised when Layer Normalization configuration parameters are invalid."""

    pass
