"""Byte-Pair Encoding (BPE) trainer engine for Aura.

WHY THIS FILE EXISTS:
    Executes the BPE merge learning algorithm over a training text corpus. Iteratively counts
    adjacent character/subword pair frequencies and learns optimal merge rules up to target vocabulary size.

WHY THIS IMPLEMENTATION WAS CHOSEN:
    Operating directly on character sequences (preserving spaces, tabs, and newlines) guarantees
    100% loss-free encoding and decoding round-trips for programming source code and indentation.

TIME COMPLEXITY:
    O(M * N) where M is the target number of merges, and N is the total character count in the corpus.

SPACE COMPLEXITY:
    O(N + V) auxiliary space to hold sequence token lists and pair frequency counters.

POSSIBLE IMPROVEMENTS:
    - Multi-threading pair frequency calculations across large multi-gigabyte code corpora.
"""

from collections import Counter
import logging
from typing import Dict, List, Optional, Tuple

from src.tokenizer.bpe_vocab import BPESpecialTokens, BPEVocab

logger = logging.getLogger(__name__)


class BPETrainer:
    """Trains a BPE vocabulary and merge rank table from text corpora."""

    def __init__(
        self,
        vocab_size: int = 500,
        special_tokens: Optional[BPESpecialTokens] = None,
    ) -> None:
        """Initialize BPE Trainer.

        Args:
            vocab_size: Target vocabulary size (including special tokens and base characters).
            special_tokens: Container for special control tokens.
        """
        self.target_vocab_size = vocab_size
        self.special_tokens = special_tokens or BPESpecialTokens()

    def _get_stats(self, sequences: List[List[str]]) -> Dict[Tuple[str, str], int]:
        """Counts frequencies of adjacent token pairs across all text sequences in corpus."""
        counts: Counter[Tuple[str, str]] = Counter()
        for seq in sequences:
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                counts[pair] += 1
        return counts

    def _merge_sequence(self, sequence: List[str], pair: Tuple[str, str], merged_token: str) -> List[str]:
        """Merges all occurrences of pair (pair[0], pair[1]) into merged_token in a token list."""
        new_seq: List[str] = []
        i = 0
        while i < len(sequence):
            if i < len(sequence) - 1 and sequence[i] == pair[0] and sequence[i + 1] == pair[1]:
                new_seq.append(merged_token)
                i += 2
            else:
                new_seq.append(sequence[i])
                i += 1
        return new_seq

    def train(self, corpus: str | List[str]) -> BPEVocab:
        """Trains BPE vocabulary from text corpus until target_vocab_size is reached.

        Args:
            corpus: Raw string text or list of text strings.

        Returns:
            Trained BPEVocab instance.
        """
        if isinstance(corpus, str):
            corpus_lines = [corpus]
        else:
            corpus_lines = corpus

        # 1. Initialize BPE Vocab container with special tokens
        vocab = BPEVocab(special_tokens=self.special_tokens)

        # 2. Convert corpus lines into lists of individual characters (preserving spaces/newlines)
        sequences: List[List[str]] = [list(line) for line in corpus_lines if line]

        if not sequences:
            logger.warning("Empty corpus provided for BPE training.")
            return vocab

        # 3. Register all base characters into vocabulary
        unique_chars = set(char for seq in sequences for char in seq)
        for char in sorted(unique_chars):
            vocab.add_token(char)

        # 4. Iteratively find most frequent pair and merge
        num_merges = self.target_vocab_size - len(vocab)
        logger.info(
            "Starting BPE training: Base Vocab Size=%d, Target Size=%d, Merges to learn=%d",
            len(vocab),
            self.target_vocab_size,
            max(0, num_merges),
        )

        for rank in range(max(0, num_merges)):
            pair_counts = self._get_stats(sequences)
            if not pair_counts:
                logger.info("No more adjacent pairs to merge. Stopping at vocab size %d.", len(vocab))
                break

            # Find pair with highest frequency (break ties deterministically)
            best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
            if pair_counts[best_pair] < 1:
                break

            merged_token = vocab.add_merge(best_pair, rank)

            # Apply merge to all sequences in corpus
            sequences = [
                self._merge_sequence(seq, best_pair, merged_token) for seq in sequences
            ]

        logger.info("Completed BPE training: Final Vocab Size=%d, Total Merges=%d", len(vocab), len(vocab.merges))
        return vocab
