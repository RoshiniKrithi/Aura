"""Byte-Pair Encoding (BPE) Tokenizer implementation for Aura.

WHY THIS FILE EXISTS:
    Implements a subword BPE tokenizer wrapping `BPEVocab`. Converts raw text into subword token IDs
    and decodes token IDs back into text, supporting compression analysis, dynamic batch padding,
    truncation, and BOS/EOS wrapping.

WHY THIS IMPLEMENTATION WAS CHOSEN:
    Subword tokenization compresses common programming keywords and DSA identifier patterns (e.g. `binary_search`)
    into single token IDs, dramatically shortening sequence lengths fed into the Transformer model.

TIME COMPLEXITY:
    - encode(text): O(L * M) where L is input length and M is total merge rank count.
    - decode(token_ids): O(T) where T is token sequence length.
    - get_compression_ratio(text): O(L * M).

SPACE COMPLEXITY:
    O(L) auxiliary space to store token lists.

POSSIBLE IMPROVEMENTS:
    - Pre-computed subword trie for sub-millisecond per-word performance.
"""

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from src.tokenizer.bpe_vocab import BPESpecialTokens, BPEVocab
from src.tokenizer.exceptions import TokenizerError

logger = logging.getLogger(__name__)


class BPETokenizer:
    """Byte-Pair Encoding subword tokenizer supporting single and batch encode/decode."""

    def __init__(self, vocab: BPEVocab) -> None:
        """Initialize BPE Tokenizer.

        Args:
            vocab: Instantiated BPEVocab object.
        """
        self.vocab = vocab

    @property
    def vocab_size(self) -> int:
        """Returns vocabulary size."""
        return len(self.vocab)

    @classmethod
    def from_files(
        cls, vocab_file: Path | str, merges_file: Path | str
    ) -> "BPETokenizer":
        """Loads a BPETokenizer from saved vocabulary JSON and merges text file.

        Args:
            vocab_file: Path to vocabulary JSON.
            merges_file: Path to merges text file.

        Returns:
            Configured BPETokenizer instance.
        """
        vocab = BPEVocab.load(vocab_file=vocab_file, merges_file=merges_file)
        return cls(vocab=vocab)

    def save(self, vocab_file: Path | str, merges_file: Path | str) -> Tuple[Path, Path]:
        """Saves vocabulary JSON and merges file to disk."""
        return self.vocab.save(vocab_file=vocab_file, merges_file=merges_file)

    def _encode_sequence(self, text: str) -> List[str]:
        """Encodes a text string into BPE subword tokens by applying merge ranks in priority order."""
        if not text:
            return []

        tokens: List[str] = list(text)
        if len(tokens) <= 1:
            return tokens

        while True:
            # Find adjacent pairs in current tokens list
            pairs: Set[Tuple[str, str]] = {
                (tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)
            }
            valid_pairs = {p: self.vocab.merges[p] for p in pairs if p in self.vocab.merges}

            if not valid_pairs:
                break

            # Pick pair with lowest rank index (highest priority)
            best_pair = min(valid_pairs, key=valid_pairs.get)  # type: ignore[arg-type]

            # Merge pair in tokens list
            new_tokens: List[str] = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == best_pair[0] and tokens[i + 1] == best_pair[1]:
                    new_tokens.append(best_pair[0] + best_pair[1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return tokens

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = False,
    ) -> List[int]:
        """Encodes text string into a list of subword token IDs.

        Args:
            text: Input text string.
            add_special_tokens: If True, prepends BOS and appends EOS token.
            max_length: Optional sequence length bound.
            padding: If True, pads sequence to max_length with PAD token ID.
            truncation: If True, truncates sequence exceeding max_length.

        Returns:
            List of integer token IDs.
        """
        subwords = self._encode_sequence(text)
        token_ids: List[int] = [self.vocab.get_id(subword) for subword in subwords]

        if add_special_tokens:
            token_ids = [self.vocab.bos_id] + token_ids + [self.vocab.eos_id]

        if max_length is not None:
            if len(token_ids) > max_length:
                if truncation:
                    token_ids = token_ids[:max_length]
                else:
                    raise TokenizerError(
                        f"Sequence length ({len(token_ids)}) exceeds max_length ({max_length}) without truncation."
                    )

            if padding and len(token_ids) < max_length:
                token_ids = token_ids + [self.vocab.pad_id] * (max_length - len(token_ids))

        return token_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = False) -> str:
        """Decodes integer token IDs back into string text.

        Args:
            token_ids: List of integer token IDs.
            skip_special_tokens: If True, filters out PAD, UNK, BOS, and EOS tokens.

        Returns:
            Reconstructed text string.
        """
        special_ids = {
            self.vocab.pad_id,
            self.vocab.unk_id,
            self.vocab.bos_id,
            self.vocab.eos_id,
        }

        subwords: List[str] = []
        for tid in token_ids:
            if skip_special_tokens and tid in special_ids:
                continue
            subwords.append(self.vocab.get_token(tid))

        return "".join(subwords)

    def encode_batch(
        self,
        texts: List[str],
        add_special_tokens: bool = False,
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = False,
    ) -> List[List[int]]:
        """Batch encodes list of text strings into padded token ID lists."""
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
        """Batch decodes list of token ID lists into strings."""
        return [
            self.decode(token_ids, skip_special_tokens=skip_special_tokens)
            for token_ids in token_ids_batch
        ]

    def get_compression_ratio(self, text: str) -> float:
        """Calculates text compression ratio: character_count / token_count.

        A higher ratio indicates better subword sequence compression.

        Args:
            text: Input text string.

        Returns:
            Compression ratio float.
        """
        if not text:
            return 1.0
        encoded = self.encode(text)
        if not encoded:
            return 1.0
        return len(text) / len(encoded)
