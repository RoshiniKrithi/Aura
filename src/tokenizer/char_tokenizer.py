"""Character-level tokenizer implementation for Aura.

WHY THIS FILE EXISTS:
    Implements a pure Python/PyTorch character-level tokenizer wrapping `Vocabulary`.
    Provides encoding, decoding, batch operations, sequence padding, truncation,
    BOS/EOS wrapping, and unknown token handling without third-party dependencies.

WHY THIS IMPLEMENTATION WAS CHOSEN:
    Character-level tokenization operates at single-character granularity. This guarantees
    near-zero out-of-vocabulary (OOV) rates while eliminating complex tokenization subword algorithms,
    making it ideal for validating foundational architecture components.

TIME COMPLEXITY:
    - encode(text): O(L) where L is string character length.
    - decode(token_ids): O(T) where T is token sequence length.
    - encode_batch(texts): O(B * L) where B is batch size and L is maximum sequence length.
    - decode_batch(token_ids_batch): O(B * T).

SPACE COMPLEXITY:
    O(L) auxiliary space to store token ID lists per sequence.

POSSIBLE IMPROVEMENTS:
    - Support vectorization returning PyTorch `torch.Tensor` objects directly with `pin_memory`.
"""

import logging
from pathlib import Path
from typing import Iterable, List, Optional

from src.tokenizer.exceptions import TokenizerError
from src.tokenizer.vocab import SpecialTokens, Vocabulary

logger = logging.getLogger(__name__)


class CharacterTokenizer:
    """Character-level tokenizer supporting single string and batch encode/decode."""

    def __init__(self, vocab: Vocabulary) -> None:
        """Initialize character tokenizer with a Vocabulary instance.

        Args:
            vocab: Instantiated Vocabulary object.
        """
        self.vocab = vocab

    @property
    def vocab_size(self) -> int:
        """Returns vocabulary size."""
        return len(self.vocab)

    @classmethod
    def from_corpus(
        cls,
        corpus: str | Iterable[str],
        special_tokens: Optional[SpecialTokens] = None,
    ) -> "CharacterTokenizer":
        """Constructs a CharacterTokenizer by building a new vocabulary from a text corpus.

        Args:
            corpus: Training text corpus string or iterable of lines.
            special_tokens: Optional custom SpecialTokens.

        Returns:
            Configured CharacterTokenizer instance.
        """
        vocab = Vocabulary.build_from_corpus(corpus=corpus, special_tokens=special_tokens)
        return cls(vocab=vocab)

    @classmethod
    def from_file(cls, vocab_file_path: Path | str) -> "CharacterTokenizer":
        """Loads a CharacterTokenizer from a saved JSON vocabulary file.

        Args:
            vocab_file_path: Path to vocabulary JSON file.

        Returns:
            Configured CharacterTokenizer instance.
        """
        vocab = Vocabulary.load(vocab_file_path)
        return cls(vocab=vocab)

    def save_vocab(self, vocab_file_path: Path | str) -> Path:
        """Saves vocabulary JSON to disk.

        Args:
            vocab_file_path: Output file path.

        Returns:
            Resolved Path.
        """
        return self.vocab.save(vocab_file_path)

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = False,
    ) -> List[int]:
        """Encodes a text string into a list of integer token IDs.

        Args:
            text: Input string to tokenize.
            add_special_tokens: If True, prepends BOS token and appends EOS token.
            max_length: Optional sequence length bound.
            padding: If True and max_length is specified, pads sequence with PAD token IDs.
            truncation: If True and max_length is specified, truncates sequence if it exceeds max_length.

        Returns:
            List of integer token IDs.

        Raises:
            TokenizerError: If sequence length exceeds max_length when truncation is False.
        """
        token_ids: List[int] = [self.vocab.get_id(char) for char in text]

        if add_special_tokens:
            token_ids = [self.vocab.bos_id] + token_ids + [self.vocab.eos_id]

        if max_length is not None:
            if len(token_ids) > max_length:
                if truncation:
                    token_ids = token_ids[:max_length]
                else:
                    raise TokenizerError(
                        f"Sequence length ({len(token_ids)}) exceeds max_length ({max_length}) without truncation enabled."
                    )

            if padding and len(token_ids) < max_length:
                padding_length = max_length - len(token_ids)
                token_ids = token_ids + [self.vocab.pad_id] * padding_length

        return token_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = False) -> str:
        """Decodes a list of integer token IDs back into a string.

        Args:
            token_ids: List of integer token IDs.
            skip_special_tokens: If True, filters out PAD, UNK, BOS, and EOS tokens from output string.

        Returns:
            Reconstructed string text.
        """
        special_ids = {
            self.vocab.pad_id,
            self.vocab.unk_id,
            self.vocab.bos_id,
            self.vocab.eos_id,
        }

        chars: List[str] = []
        for tid in token_ids:
            if skip_special_tokens and tid in special_ids:
                continue
            chars.append(self.vocab.get_char(tid))

        return "".join(chars)

    def encode_batch(
        self,
        texts: List[str],
        add_special_tokens: bool = False,
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = False,
    ) -> List[List[int]]:
        """Batch encodes a list of text strings into lists of token IDs.

        If padding is True and max_length is None, pads dynamically to the longest sequence in the batch.

        Args:
            texts: List of text strings to tokenize.
            add_special_tokens: If True, wraps each string with BOS and EOS tokens.
            max_length: Optional sequence length cap.
            padding: If True, pads sequences to max_length (or longest in batch).
            truncation: If True, truncates sequences exceeding max_length.

        Returns:
            2D List of token IDs (batch_size, sequence_length).
        """
        encoded_batch: List[List[int]] = [
            self.encode(
                text=text,
                add_special_tokens=add_special_tokens,
                max_length=max_length if truncation else None,
                padding=False,
                truncation=truncation,
            )
            for text in texts
        ]

        if padding:
            target_length = max_length
            if target_length is None:
                target_length = max(len(seq) for seq in encoded_batch) if encoded_batch else 0

            padded_batch: List[List[int]] = []
            for seq in encoded_batch:
                if len(seq) < target_length:
                    seq = seq + [self.vocab.pad_id] * (target_length - len(seq))
                elif len(seq) > target_length and truncation:
                    seq = seq[:target_length]
                padded_batch.append(seq)

            return padded_batch

        return encoded_batch

    def decode_batch(
        self, token_ids_batch: List[List[int]], skip_special_tokens: bool = False
    ) -> List[str]:
        """Batch decodes a 2D list of token IDs into text strings.

        Args:
            token_ids_batch: 2D list of integer token IDs.
            skip_special_tokens: If True, filters special control tokens.

        Returns:
            List of decoded text strings.
        """
        return [
            self.decode(token_ids, skip_special_tokens=skip_special_tokens)
            for token_ids in token_ids_batch
        ]
