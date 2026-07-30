"""Custom exception hierarchy for the Aura tokenizer system.

WHY THIS FILE EXISTS:
    To provide explicit domain-specific exception types for error handling during tokenization,
    vocabulary building, sequence encoding, and deserialization.

WHY THIS IMPLEMENTATION WAS CHOSEN:
    Subclassing Python's built-in `Exception` allows upstream callers (e.g. training loops or API endpoints)
    to catch tokenizer-specific errors cleanly without swallowing unrelated system exceptions.

TIME COMPLEXITY:
    O(1) for instantiation and representation.

SPACE COMPLEXITY:
    O(1) auxiliary space.

POSSIBLE IMPROVEMENTS:
    - Add structured error codes (e.g. `ERR_VOCAB_MISSING_TOKEN`) for API integration.
"""


class TokenizerError(Exception):
    """Base exception class for all tokenization failures."""

    pass


class VocabularyError(TokenizerError):
    """Exception raised for errors during vocabulary building, loading, or lookup operations."""

    pass
