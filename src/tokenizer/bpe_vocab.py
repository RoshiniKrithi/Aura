"""Byte-Pair Encoding (BPE) vocabulary and merge table management for Aura.

WHY THIS FILE EXISTS:
    Encapsulates subword token-to-ID mappings, special control tokens, BPE merge rank tables
    ((token_a, token_b) -> rank), and robust JSON serialization.

WHY THIS IMPLEMENTATION WAS CHOSEN:
    Storing BPE merges in a rank hash dictionary allows O(1) lookup during sequence encoding.
    Serializing merges to JSON guarantees 100% lossless preservation of spaces, tabs, and newlines.

TIME COMPLEXITY:
    - Merge Rank Lookup: O(1) average time complexity.
    - Vocab & Merges Save / Load: O(V + M) where V is vocabulary size and M is total merge count.

SPACE COMPLEXITY:
    O(V + M) auxiliary space to store token dictionaries and merge rank mappings.

POSSIBLE IMPROVEMENTS:
    - Binary format serialization for massive 100k+ vocabulary files.
"""

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.tokenizer.exceptions import VocabularyError
from src.utils.paths import project_paths

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BPESpecialTokens:
    """Dataclass holding special control tokens for BPE tokenizer."""

    pad_token: str = "<|pad|>"
    unk_token: str = "<|unk|>"
    bos_token: str = "<|startoftext|>"
    eos_token: str = "<|endoftext|>"

    def to_list(self) -> List[str]:
        """Returns list of special token strings."""
        return [self.pad_token, self.unk_token, self.bos_token, self.eos_token]


class BPEVocab:
    """Manages BPE token-to-ID mappings, merge rank tables, and JSON/text persistence."""

    def __init__(
        self,
        token2id: Optional[Dict[str, int]] = None,
        id2token: Optional[Dict[int, str]] = None,
        merges: Optional[Dict[Tuple[str, str], int]] = None,
        special_tokens: Optional[BPESpecialTokens] = None,
    ) -> None:
        """Initialize BPE Vocabulary instance.

        Args:
            token2id: Map from token string to integer ID.
            id2token: Map from integer ID to token string.
            merges: Map from adjacent token tuple (pair_a, pair_b) to merge rank index.
            special_tokens: BPESpecialTokens container.
        """
        self.special_tokens = special_tokens or BPESpecialTokens()
        self.token2id: Dict[str, int] = token2id or {}
        self.id2token: Dict[int, str] = id2token or {}
        self.merges: Dict[Tuple[str, str], int] = merges or {}

        if not self.token2id:
            self._register_special_tokens()

    def _register_special_tokens(self) -> None:
        """Register special control tokens at starting indices."""
        for token in self.special_tokens.to_list():
            if token not in self.token2id:
                token_id = len(self.token2id)
                self.token2id[token] = token_id
                self.id2token[token_id] = token

    @property
    def pad_id(self) -> int:
        """Returns PAD token integer ID."""
        return self.token2id[self.special_tokens.pad_token]

    @property
    def unk_id(self) -> int:
        """Returns UNK token integer ID."""
        return self.token2id[self.special_tokens.unk_token]

    @property
    def bos_id(self) -> int:
        """Returns BOS token integer ID."""
        return self.token2id[self.special_tokens.bos_token]

    @property
    def eos_id(self) -> int:
        """Returns EOS token integer ID."""
        return self.token2id[self.special_tokens.eos_token]

    def __len__(self) -> int:
        """Returns total vocabulary size."""
        return len(self.token2id)

    def __contains__(self, token: str) -> bool:
        """Checks if token exists in vocabulary."""
        return token in self.token2id

    def get_id(self, token: str) -> int:
        """Returns integer ID for token, defaulting to UNK ID if missing."""
        return self.token2id.get(token, self.unk_id)

    def get_token(self, token_id: int) -> str:
        """Returns token string for integer ID."""
        if token_id not in self.id2token:
            raise VocabularyError(f"Token ID '{token_id}' not found in BPE vocabulary.")
        return self.id2token[token_id]

    def add_token(self, token: str) -> int:
        """Adds a new subword token to vocabulary if not already present."""
        if token not in self.token2id:
            token_id = len(self.token2id)
            self.token2id[token] = token_id
            self.id2token[token_id] = token
            return token_id
        return self.token2id[token]

    def add_merge(self, pair: Tuple[str, str], rank: int) -> str:
        """Registers a merge pair (pair_a, pair_b) and constructs merged token string."""
        self.merges[pair] = rank
        merged_token = pair[0] + pair[1]
        self.add_token(merged_token)
        return merged_token

    def save(self, vocab_file: Path | str, merges_file: Path | str) -> Tuple[Path, Path]:
        """Saves vocabulary JSON and merges file to disk.

        Args:
            vocab_file: Path to output JSON vocabulary file.
            merges_file: Path to output merges JSON file.

        Returns:
            Tuple of resolved (vocab_path, merges_path).
        """
        v_path = Path(vocab_file).resolve()
        m_path = Path(merges_file).resolve()

        project_paths.ensure_dir(v_path.parent)
        project_paths.ensure_dir(m_path.parent)

        # Save vocab JSON
        data = {
            "special_tokens": {
                "pad_token": self.special_tokens.pad_token,
                "unk_token": self.special_tokens.unk_token,
                "bos_token": self.special_tokens.bos_token,
                "eos_token": self.special_tokens.eos_token,
            },
            "token2id": self.token2id,
        }
        with open(v_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # Save merges JSON structure (sorted by rank)
        sorted_merges = sorted(self.merges.items(), key=lambda x: x[1])
        merges_data = [[pair[0], pair[1]] for pair, _ in sorted_merges]
        with open(m_path, "w", encoding="utf-8") as f:
            json.dump(merges_data, f, indent=2, ensure_ascii=False)

        logger.info("Saved BPE Vocabulary to %s and Merges to %s", v_path, m_path)
        return v_path, m_path

    @classmethod
    def load(cls, vocab_file: Path | str, merges_file: Path | str) -> "BPEVocab":
        """Loads BPEVocab instance from JSON vocabulary file and merges file.

        Args:
            vocab_file: Source JSON vocabulary file path.
            merges_file: Source merges file path.

        Returns:
            Deserialized BPEVocab instance.
        """
        v_path = Path(vocab_file).resolve()
        m_path = Path(merges_file).resolve()

        if not v_path.is_file():
            raise VocabularyError(f"Vocab file not found at {v_path}")
        if not m_path.is_file():
            raise VocabularyError(f"Merges file not found at {m_path}")

        # Load vocab JSON
        with open(v_path, "r", encoding="utf-8") as f:
            v_data = json.load(f)

        sp_data = v_data.get("special_tokens", {})
        special_tokens = BPESpecialTokens(
            pad_token=sp_data.get("pad_token", "<|pad|>"),
            unk_token=sp_data.get("unk_token", "<|unk|>"),
            bos_token=sp_data.get("bos_token", "<|startoftext|>"),
            eos_token=sp_data.get("eos_token", "<|endoftext|>"),
        )
        token2id: Dict[str, int] = v_data["token2id"]
        id2token: Dict[int, str] = {int(v): k for k, v in token2id.items()}

        # Load merges file
        merges: Dict[Tuple[str, str], int] = {}
        with open(m_path, "r", encoding="utf-8") as f:
            raw_merges = json.load(f)
            for rank, pair in enumerate(raw_merges):
                if len(pair) == 2:
                    merges[(pair[0], pair[1])] = rank

        vocab = cls(
            token2id=token2id,
            id2token=id2token,
            merges=merges,
            special_tokens=special_tokens,
        )
        logger.info("Loaded BPEVocab (size=%d, merges=%d)", len(vocab), len(merges))
        return vocab
