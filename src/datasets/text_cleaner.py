"""Production-grade text cleaning and normalization utility for Aura LLM data pipelines.

Handles Unicode normalization, control character stripping, tab expansion,
and line ending standardization across plain text and source code datasets.
"""

import logging
import re
import unicodedata

logger = logging.getLogger(__name__)


class TextCleaner:
    """Pure functional text cleaning engine for corpus normalization.

    Design Decisions:
        - Stateless implementation to avoid side-effects across parallel processing threads.
        - Preserves standard code whitespace (newlines, spaces, indentation) crucial for
          programming datasets (Python, C++, Java, etc.).
        - Configurable control character filtering to strip null bytes and corrupted escape codes.

    Time Complexity:
        O(N) where N is the length of the string in characters.

    Space Complexity:
        O(N) for string reconstruction buffers.
    """

    # Unprintable control characters excluding '\n' (0x0A), '\t' (0x09), and '\r' (0x0D)
    _CONTROL_CHAR_REGEX = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]")

    def __init__(
        self,
        normalize_unicode: str = "NFC",
        remove_unprintable: bool = True,
        fix_line_endings: bool = True,
        expand_tabs: bool = False,
        tab_size: int = 4,
    ) -> None:
        """Initializes cleaner configuration options.

        Args:
            normalize_unicode: Form for unicodedata.normalize ('NFC', 'NFD', 'NFKC', 'NFKD', or None).
            remove_unprintable: If True, strips non-standard ASCII control characters.
            fix_line_endings: If True, standardizes \r\n and \r to \n.
            expand_tabs: If True, replaces tab characters with tab_size spaces.
            tab_size: Number of spaces for tab expansion.
        """
        self.normalize_unicode = normalize_unicode
        self.remove_unprintable = remove_unprintable
        self.fix_line_endings = fix_line_endings
        self.expand_tabs = expand_tabs
        self.tab_size = tab_size

    def clean(self, text: str) -> str:
        """Applies configuration-based cleaning transformations on input string.

        Args:
            text: Raw input string.

        Returns:
            Normalized and cleaned output text string.
        """
        if not text:
            return ""

        # 1. Line ending normalization
        if self.fix_line_endings:
            text = text.replace("\r\n", "\n").replace("\r", "\n")

        # 2. Unicode normalization
        if self.normalize_unicode:
            text = unicodedata.normalize(self.normalize_unicode, text)

        # 3. Control character filtering
        if self.remove_unprintable:
            text = self._CONTROL_CHAR_REGEX.sub("", text)

        # 4. Tab expansion (optional, useful for certain code indentation settings)
        if self.expand_tabs:
            text = text.expandtabs(self.tab_size)

        return text

    @classmethod
    def clean_text_static(
        cls,
        text: str,
        normalize_unicode: str = "NFC",
        remove_unprintable: bool = True,
        fix_line_endings: bool = True,
    ) -> str:
        """Static convenience helper for quick text cleaning without instantiating cleaner."""
        cleaner = cls(
            normalize_unicode=normalize_unicode,
            remove_unprintable=remove_unprintable,
            fix_line_endings=fix_line_endings,
        )
        return cleaner.clean(text)
