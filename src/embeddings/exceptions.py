"""Custom Exceptions for Aura Embedding Pipeline.

Provides structured error handling for embedding initialization, validation,
tensor shape mismatches, out-of-bounds token indices, and config errors.
"""


class EmbeddingError(Exception):
    """Base exception for all embedding module errors."""

    pass


class EmbeddingValidationError(EmbeddingError):
    """Raised when token ID tensor validation fails (e.g. out-of-bounds, negative IDs, empty batches)."""

    pass


class EmbeddingInitializationError(EmbeddingError):
    """Raised when weight initialization strategy or distribution parameters are invalid."""

    pass


class EmbeddingConfigError(EmbeddingError):
    """Raised when embedding configuration parameters are malformed."""

    pass
