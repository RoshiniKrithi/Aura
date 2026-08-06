"""Code-Specialized BPE Tokenizer for Aura LLM Pipeline.

Extends BPETokenizer with special control token bindings (<|code_start|>, <|code_end|>,
<|docstring|>, <|dsa_problem|>, <|dsa_solution|>), fast subword encoding, and code syntax splitting.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.tokenizer.bpe_vocab import BPESpecialTokens, BPEVocab

logger = logging.getLogger(__name__)


class CodeBPESpecialTokens(BPESpecialTokens):
    """Container for code-specialized domain control tokens."""

    def __init__(
        self,
        pad_token: str = "<|pad|>",
        unk_token: str = "<|unk|>",
        bos_token: str = "<|code_start|>",
        eos_token: str = "<|code_end|>",
        docstring_token: str = "<|docstring|>",
        dsa_problem_token: str = "<|dsa_problem|>",
        dsa_solution_token: str = "<|dsa_solution|>",
    ) -> None:
        """Initializes special control tokens."""
        super().__init__(
            pad_token=pad_token,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
        )
        self.docstring_token = docstring_token
        self.dsa_problem_token = dsa_problem_token
        self.dsa_solution_token = dsa_solution_token


class CodeBPETokenizer(BPETokenizer):
    """Subword BPE Tokenizer optimized for programming language syntax and DSA corpora."""

    def __init__(self, vocab: BPEVocab) -> None:
        """Initialize CodeBPETokenizer.

        Args:
            vocab: Instantiated BPEVocab object.
        """
        super().__init__(vocab=vocab)
        self.docstring_id = vocab.get_id("<|docstring|>") if "<|docstring|>" in vocab.token2id else vocab.unk_id
        self.dsa_problem_id = vocab.get_id("<|dsa_problem|>") if "<|dsa_problem|>" in vocab.token2id else vocab.unk_id
        self.dsa_solution_id = vocab.get_id("<|dsa_solution|>") if "<|dsa_solution|>" in vocab.token2id else vocab.unk_id

    @classmethod
    def from_files(
        cls, vocab_file: Union[Path, str], merges_file: Union[Path, str]
    ) -> "CodeBPETokenizer":
        """Constructs CodeBPETokenizer from saved vocabulary JSON and merges TXT files.

        Args:
            vocab_file: Path to vocabulary JSON file.
            merges_file: Path to merges TXT file.

        Returns:
            Configured CodeBPETokenizer object.
        """
        vocab = BPEVocab.load(vocab_file=vocab_file, merges_file=merges_file)
        return cls(vocab=vocab)

    @classmethod
    def create_default(cls) -> "CodeBPETokenizer":
        """Constructs a default CodeBPETokenizer with standard byte-level vocabulary."""
        vocab = BPEVocab()
        return cls(vocab=vocab)

    def encode_code(
        self,
        code_str: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = False,
    ) -> List[int]:
        """Encodes code string into subword token IDs.

        Args:
            code_str: Source code text string.
            add_special_tokens: If True, wraps with <|code_start|> and <|code_end|>.
            max_length: Optional max sequence length cap.
            padding: If True, pads sequence to max_length.
            truncation: If True, truncates sequence to max_length.

        Returns:
            List of integer token IDs.
        """
        return self.encode(
            text=code_str,
            add_special_tokens=add_special_tokens,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
        )

    def encode_dsa_pair(
        self, problem_description: str, solution_code: str, max_length: Optional[int] = None
    ) -> List[int]:
        """Formats and encodes a (problem_description, solution_code) pair into token IDs.

        Format:
            <|code_start|> <|dsa_problem|> problem_text <|dsa_solution|> solution_text <|code_end|>

        Args:
            problem_description: Text prompt or problem specification string.
            solution_code: Source code solution text string.
            max_length: Optional max sequence length cap.

        Returns:
            List of token IDs.
        """
        problem_ids = self.encode(problem_description, add_special_tokens=False)
        solution_ids = self.encode(solution_code, add_special_tokens=False)

        formatted_ids = (
            [self.vocab.bos_id, self.dsa_problem_id]
            + problem_ids
            + [self.dsa_solution_id]
            + solution_ids
            + [self.vocab.eos_id]
        )

        if max_length is not None and len(formatted_ids) > max_length:
            formatted_ids = formatted_ids[:max_length]

        return formatted_ids
