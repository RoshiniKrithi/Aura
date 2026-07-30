"""Vocabulary management and character-to-ID mapping system for Aura.

WHY THIS FILE EXISTS:
    Encapsulates character mapping logic, special token reservations (<|pad|>, <|unk|>,
    <|startoftext|>, <|endoftext|>), character frequency distributions, and JSON persistence.

WHY THIS IMPLEMENTATION WAS CHOSEN:
    Dual hash maps (`char2id` and `id2char`) provide O(1) constant-time lookup speed for both
    character encoding and integer decoding operations.

TIME COMPLEXITY:
    - Vocabulary Construction: O(N) where N is total character count in the training corpus.
    - Token Lookup (get_id / get_char): O(1) average time complexity.
    - JSON Save / Load: O(V) where V is the vocabulary size.

SPACE COMPLEXITY:
    O(V) memory to store bidirectional mapping dictionaries and character frequency counts.

POSSIBLE IMPROVEMENTS:
    - Add byte-fallback handling for arbitrary binary streams.
"""

from collections import Counter
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from src.tokenizer.exceptions import VocabularyError
from src.utils.paths import project_paths

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpecialTokens:
    """Dataclass holding special control token definitions."""

    pad_token: str = "<|pad|>"
    unk_token: str = "<|unk|>"
    bos_token: str = "<|startoftext|>"
    eos_token: str = "<|endoftext|>"

    def to_list(self) -> List[str]:
        """Returns ordered list of special token strings."""
        return [self.pad_token, self.unk_token, self.bos_token, self.eos_token]


class Vocabulary:
    """Manages character vocabulary maps, frequency statistics, and persistence."""

    def __init__(
        self,
        char2id: Optional[Dict[str, int]] = None,
        id2char: Optional[Dict[int, str]] = None,
        counts: Optional[Dict[str, int]] = None,
        special_tokens: Optional[SpecialTokens] = None,
    ) -> None:
        """Initialize vocabulary instance.

        Args:
            char2id: Map from character string to integer token ID.
            id2char: Map from integer token ID to character string.
            counts: Map from character string to frequency count.
            special_tokens: SpecialTokens definition instance.
        """
        self.special_tokens = special_tokens or SpecialTokens()
        self.char2id: Dict[str, int] = char2id or {}
        self.id2char: Dict[int, str] = id2char or {}
        self.counts: Dict[str, int] = counts or {}

        # Ensure special tokens are registered upon instantiation if empty
        if not self.char2id:
            self._register_special_tokens()

    def _register_special_tokens(self) -> None:
        """Assign fixed IDs to special control tokens starting from 0."""
        for token in self.special_tokens.to_list():
            if token not in self.char2id:
                token_id = len(self.char2id)
                self.char2id[token] = token_id
                self.id2char[token_id] = token
                self.counts[token] = 0

    @property
    def pad_id(self) -> int:
        """Returns integer token ID for padding token."""
        return self.char2id[self.special_tokens.pad_token]

    @property
    def unk_id(self) -> int:
        """Returns integer token ID for unknown token."""
        return self.char2id[self.special_tokens.unk_token]

    @property
    def bos_id(self) -> int:
        """Returns integer token ID for beginning-of-sequence token."""
        return self.char2id[self.special_tokens.bos_token]

    @property
    def eos_id(self) -> int:
        """Returns integer token ID for end-of-sequence token."""
        return self.char2id[self.special_tokens.eos_token]

    def __len__(self) -> int:
        """Returns total vocabulary size."""
        return len(self.char2id)

    def __contains__(self, char: str) -> bool:
        """Checks if character or token is in vocabulary."""
        return char in self.char2id

    def get_id(self, char: str) -> int:
        """Lookup integer ID for a character, returning UNK ID if missing.

        Args:
            char: Character string to lookup.

        Returns:
            Integer token ID.
        """
        return self.char2id.get(char, self.unk_id)

    def get_char(self, token_id: int) -> str:
        """Lookup character string for a token ID.

        Args:
            token_id: Integer token ID to lookup.

        Returns:
            Character string.

        Raises:
            VocabularyError: If token_id is invalid.
        """
        if token_id not in self.id2char:
            raise VocabularyError(f"Token ID '{token_id}' not found in vocabulary.")
        return self.id2char[token_id]

    @classmethod
    def build_from_corpus(
        cls,
        corpus: str | Iterable[str],
        special_tokens: Optional[SpecialTokens] = None,
    ) -> "Vocabulary":
        """Builds a new vocabulary from a text string or iterable of text lines.

        Args:
            corpus: Raw string or iterable of strings to extract characters from.
            special_tokens: Optional custom SpecialTokens.

        Returns:
            Populated Vocabulary instance.
        """
        vocab = cls(special_tokens=special_tokens)
        counter: Counter[str] = Counter()

        if isinstance(corpus, str):
            counter.update(corpus)
        else:
            for text in corpus:
                counter.update(text)

        # Sort characters deterministically by frequency descending, then lexicographically
        sorted_chars = sorted(counter.keys(), key=lambda c: (-counter[c], c))

        for char in sorted_chars:
            if char not in vocab.char2id:
                token_id = len(vocab.char2id)
                vocab.char2id[char] = token_id
                vocab.id2char[token_id] = char
                vocab.counts[char] = counter[char]

        logger.info(
            "Built vocabulary with size %d from corpus (%d total characters).",
            len(vocab),
            sum(counter.values()),
        )
        return vocab

    def get_stats(self) -> Dict[str, Any]:
        """Returns dictionary of vocabulary diagnostics and frequency statistics."""
        total_occurrences = sum(self.counts.values())
        top_10 = sorted(
            [(c, freq) for c, freq in self.counts.items() if c not in self.special_tokens.to_list()],
            key=lambda x: x[1],
            reverse=True,
        )[:10]

        return {
            "vocab_size": len(self),
            "total_character_count": total_occurrences,
            "special_tokens": self.special_tokens.to_list(),
            "top_10_characters": top_10,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serializes vocabulary state to dictionary format."""
        return {
            "special_tokens": {
                "pad_token": self.special_tokens.pad_token,
                "unk_token": self.special_tokens.unk_token,
                "bos_token": self.special_tokens.bos_token,
                "eos_token": self.special_tokens.eos_token,
            },
            "char2id": self.char2id,
            "counts": self.counts,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Vocabulary":
        """Deserializes vocabulary instance from dictionary data."""
        sp_data = data.get("special_tokens", {})
        special_tokens = SpecialTokens(
            pad_token=sp_data.get("pad_token", "<|pad|>"),
            unk_token=sp_data.get("unk_token", "<|unk|>"),
            bos_token=sp_data.get("bos_token", "<|startoftext|>"),
            eos_token=sp_data.get("eos_token", "<|endoftext|>"),
        )

        char2id: Dict[str, int] = data["char2id"]
        # Convert JSON string keys in id2char back to integers
        id2char: Dict[int, str] = {int(v): k for k, v in char2id.items()}
        counts: Dict[str, int] = data.get("counts", {})

        return cls(
            char2id=char2id,
            id2char=id2char,
            counts=counts,
            special_tokens=special_tokens,
        )

    def save(self, file_path: Path | str) -> Path:
        """Saves vocabulary as JSON file.

        Args:
            file_path: Destination JSON file path.

        Returns:
            Resolved Path to saved file.
        """
        path = Path(file_path).resolve()
        project_paths.ensure_dir(path.parent)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

        logger.info("Saved vocabulary JSON to %s", path)
        return path

    @classmethod
    def load(cls, file_path: Path | str) -> "Vocabulary":
        """Loads vocabulary instance from a JSON file.

        Args:
            file_path: Source JSON file path.

        Returns:
            Deserialized Vocabulary instance.

        Raises:
            VocabularyError: If file not found or corrupted.
        """
        path = Path(file_path).resolve()
        if not path.is_file():
            raise VocabularyError(f"Vocabulary file not found at: {path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            vocab = cls.from_dict(data)
            logger.info("Loaded vocabulary JSON from %s (size=%d)", path, len(vocab))
            return vocab
        except Exception as err:
            raise VocabularyError(f"Failed to load vocabulary from {path}: {err}") from err
