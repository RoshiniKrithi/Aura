"""Aura Tokenizer Module.

WHY THIS FILE EXISTS:
    Root entry point for tokenization engines. Exports CharacterTokenizer, BPETokenizer,
    BPETrainer, Vocabulary, BPEVocab, and custom tokenizer exceptions.

WHY THIS IMPLEMENTATION WAS CHOSEN:
    Centralized module exports enable downstream datasets, models, and trainers to import
    tokenization primitives directly (`from src.tokenizer import BPETokenizer, CharacterTokenizer`).

TIME COMPLEXITY:
    O(1) initialization.

SPACE COMPLEXITY:
    O(1) space.
"""

from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.tokenizer.bpe_trainer import BPETrainer
from src.tokenizer.bpe_vocab import BPESpecialTokens, BPEVocab
from src.tokenizer.char_tokenizer import CharacterTokenizer
from src.tokenizer.exceptions import TokenizerError, VocabularyError
from src.tokenizer.vocab import SpecialTokens, Vocabulary

__all__ = [
    "CharacterTokenizer",
    "BPETokenizer",
    "BPETrainer",
    "Vocabulary",
    "BPEVocab",
    "SpecialTokens",
    "BPESpecialTokens",
    "TokenizerError",
    "VocabularyError",
]
