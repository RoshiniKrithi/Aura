"""Aura Tokenizer Module.

WHY THIS FILE EXISTS:
    Root entry point for tokenization engines. Exports CharacterTokenizer, Vocabulary,
    SpecialTokens, and custom tokenizer exceptions.

WHY THIS IMPLEMENTATION WAS CHOSEN:
    Centralized module exports enable downstream datasets, models, and trainers to import
    tokenization primitives directly (`from src.tokenizer import CharacterTokenizer, Vocabulary`).

TIME COMPLEXITY:
    O(1) initialization.

SPACE COMPLEXITY:
    O(1) space.

POSSIBLE IMPROVEMENTS:
    - Will seamlessly export `BPETokenizer` in future Phase 2.2 update.
"""

from src.tokenizer.char_tokenizer import CharacterTokenizer
from src.tokenizer.exceptions import TokenizerError, VocabularyError
from src.tokenizer.vocab import SpecialTokens, Vocabulary

__all__ = [
    "CharacterTokenizer",
    "Vocabulary",
    "SpecialTokens",
    "TokenizerError",
    "VocabularyError",
]
