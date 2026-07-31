"""Custom exceptions for the Aura dataset pipeline.

Provides structured error handling for dataset validation, reading, caching,
sequence building, and configuration errors.
"""


class DatasetError(Exception):
    """Base exception class for all dataset pipeline errors."""

    pass


class DatasetValidationError(DatasetError):
    """Raised when raw dataset validation fails (e.g. empty, corrupt, invalid UTF-8)."""

    pass


class DatasetReadError(DatasetError):
    """Raised when reading text or folder datasets encounters an I/O failure."""

    pass


class DatasetCacheError(DatasetError):
    """Raised when cache loading, saving, or hash integrity check fails."""

    pass


class DatasetConfigError(DatasetError):
    """Raised when dataset configuration parameters are invalid."""

    pass


class SequenceGenerationError(DatasetError):
    """Raised when sequence generation or sliding context window slicing fails."""

    pass
