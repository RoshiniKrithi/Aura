"""Custom Domain Exceptions for Inference & Text Generation Subsystem.

Provides structured error handling for invalid generation configurations,
context window overflow errors, and model generation failures.
"""


class InferenceError(Exception):
    """Base exception for all inference subsystem errors."""

    pass


class InferenceValidationError(InferenceError):
    """Raised when inference prompts or setup parameters fail validation checks."""

    pass


class InferenceConfigError(InferenceError):
    """Raised when inference hyperparameter configuration parameters are invalid."""

    pass
