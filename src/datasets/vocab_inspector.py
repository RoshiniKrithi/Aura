"""Vocabulary Inspector for Aura LLM Pipeline.

Inspects tokenized sequences against tokenizer vocabulary bounds, UNK token density,
special token distribution, and invalid out-of-bounds indices.
"""

from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Sequence

logger = logging.getLogger(__name__)


@dataclass
class InspectionReport:
    """Diagnostic report output from vocabulary inspection."""

    is_valid: bool
    total_tokens: int = 0
    unk_count: int = 0
    unk_ratio: float = 0.0
    out_of_bounds_count: int = 0
    out_of_bounds_tokens: List[int] = field(default_factory=list)
    special_token_counts: Dict[str, int] = field(default_factory=dict)
    vocab_coverage_ratio: float = 0.0
    warnings: List[str] = field(default_factory=list)


class VocabularyInspector:
    """Inspects tokenized integer sequences against tokenizer vocabulary specifications.

    Time Complexity:
        O(N) single-pass token array scan.

    Space Complexity:
        O(V) where V is vocabulary size for frequency counting.
    """

    def __init__(self, tokenizer: Any) -> None:
        """Initializes inspector with target tokenizer instance.

        Args:
            tokenizer: Instantiated tokenizer object (BPETokenizer, CharacterTokenizer, etc.).
        """
        self.tokenizer = tokenizer
        self.vocab_size = getattr(tokenizer, "vocab_size", None) or len(
            getattr(tokenizer, "vocab", {})
        )
        self.unk_id = getattr(tokenizer, "unk_id", None)

    def inspect(self, token_ids: Sequence[int]) -> InspectionReport:
        """Performs comprehensive boundary and UNK audit on token IDs sequence.

        Args:
            token_ids: Sequence of token IDs.

        Returns:
            InspectionReport summary container.
        """
        report = InspectionReport(is_valid=True, total_tokens=len(token_ids))

        if not token_ids:
            report.is_valid = True
            report.warnings.append("Empty token sequence provided.")
            return report

        oob_list: List[int] = []
        unk_count = 0
        unique_tokens = set()

        for tid in token_ids:
            unique_tokens.add(tid)

            # Boundary Check: 0 <= tid < vocab_size
            if tid < 0 or tid >= self.vocab_size:
                oob_list.append(tid)

            # UNK Check
            if self.unk_id is not None and tid == self.unk_id:
                unk_count += 1

        report.unk_count = unk_count
        report.unk_ratio = unk_count / len(token_ids)
        report.out_of_bounds_count = len(oob_list)
        report.out_of_bounds_tokens = oob_list[:10]  # Sample first 10
        report.vocab_coverage_ratio = len(unique_tokens) / max(1, self.vocab_size)

        if report.out_of_bounds_count > 0:
            report.is_valid = False
            report.warnings.append(
                f"Detected {report.out_of_bounds_count} token IDs outside valid vocab range [0, {self.vocab_size - 1}]."
            )

        if report.unk_ratio > 0.05:  # Warning threshold if >5% UNK tokens
            report.warnings.append(
                f"High UNK token ratio detected ({report.unk_ratio:.2%})."
            )

        logger.info(
            "Vocab Inspection complete: Total=%d, VocabCoverage=%.2f%%, Valid=%s",
            report.total_tokens,
            report.vocab_coverage_ratio * 100.0,
            report.is_valid,
        )
        return report
