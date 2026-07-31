"""Dataset Statistics Extractor for Aura LLM Pipeline.

Computes total characters, lines, words, byte sizes, character frequencies,
and distribution analytics for raw corpora and token sequences.
"""

from collections import Counter
from dataclasses import dataclass, field
import math
from typing import Dict, List, Sequence, Union


@dataclass
class CorpusStats:
    """Summary container holding quantitative corpus statistics."""

    total_characters: int = 0
    total_lines: int = 0
    total_words: int = 0
    total_bytes: int = 0
    unique_characters: int = 0
    char_frequencies: Dict[str, int] = field(default_factory=dict)
    avg_line_length: float = 0.0
    avg_word_length: float = 0.0
    entropy: float = 0.0


class DatasetStatistics:
    """Computes mathematical and statistical distributions over text corpora and token arrays.

    Time Complexity:
        O(N) character scan over corpus text.

    Space Complexity:
        O(U) where U is the unique character set size.
    """

    @staticmethod
    def compute_text_stats(text: str, top_k_chars: int = 20) -> CorpusStats:
        """Calculates detailed metrics for input text string.

        Args:
            text: Raw or cleaned text corpus string.
            top_k_chars: Number of most frequent characters to include in summary.

        Returns:
            CorpusStats summary object.
        """
        if not text:
            return CorpusStats()

        total_bytes = len(text.encode("utf-8"))
        total_chars = len(text)
        lines = text.splitlines()
        total_lines = len(lines)

        words = text.split()
        total_words = len(words)

        counts = Counter(text)
        unique_chars = len(counts)

        # Shannon Entropy Calculation: H = -sum(p_i * log2(p_i))
        entropy = 0.0
        for count in counts.values():
            prob = count / total_chars
            entropy -= prob * math.log2(prob)

        avg_line_len = total_chars / max(1, total_lines)
        avg_word_len = (
            sum(len(w) for w in words) / max(1, total_words) if words else 0.0
        )

        top_freqs = dict(counts.most_common(top_k_chars))

        return CorpusStats(
            total_characters=total_chars,
            total_lines=total_lines,
            total_words=total_words,
            total_bytes=total_bytes,
            unique_characters=unique_chars,
            char_frequencies=top_freqs,
            avg_line_length=round(avg_line_len, 2),
            avg_word_length=round(avg_word_len, 2),
            entropy=round(entropy, 4),
        )

    @staticmethod
    def compute_token_stats(
        token_ids: Sequence[int], vocab_size: int
    ) -> Dict[str, Union[int, float]]:
        """Calculates token sequence stats.

        Args:
            token_ids: List or array of token integers.
            vocab_size: Total vocabulary size of tokenizer.

        Returns:
            Dictionary containing token sequence statistics.
        """
        if not token_ids:
            return {
                "total_tokens": 0,
                "unique_tokens": 0,
                "vocab_utilization": 0.0,
                "min_token_id": 0,
                "max_token_id": 0,
            }

        total_tokens = len(token_ids)
        counts = Counter(token_ids)
        unique_tokens = len(counts)
        vocab_util = (unique_tokens / max(1, vocab_size)) * 100.0
        min_id = min(token_ids)
        max_id = max(token_ids)

        return {
            "total_tokens": total_tokens,
            "unique_tokens": unique_tokens,
            "vocab_utilization": round(vocab_util, 2),
            "min_token_id": min_id,
            "max_token_id": max_id,
        }
