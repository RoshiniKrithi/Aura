"""Custom Domain Exceptions for AuraGPT Architecture Subsystem.

Provides structured error handling for model initialization, forward pass errors,
tensor shape mismatches, configuration invalidity, and numerical instabilities (NaN/Inf).
"""


class ModelError(Exception):
    """Base exception for all Aura model errors."""

    pass


class ModelValidationError(ModelError):
    """Raised when input token IDs, shapes, or tensor values fail validation checks."""

    pass


class ModelConfigError(ModelError):
    """Raised when model hyperparameter configuration parameters are invalid."""

    pass


class LMHeadError(ModelError):
    """Base exception for Language Modeling Head errors."""

    pass


class LMHeadValidationError(LMHeadError):
    """Raised when LM Head tensor input validation fails."""

    pass


class LMHeadConfigError(LMHeadError):
    """Raised when LM Head configuration parameters are invalid."""

    pass

