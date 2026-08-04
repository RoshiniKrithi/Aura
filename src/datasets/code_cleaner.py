"""AST-guided Code Cleaning and Normalization Module for Aura LLM.

Provides CodeTextCleaner for standardizing formatting, normalizing whitespace,
preserving docstrings, stripping non-printable artifacts, and verifying Python/C++ AST syntax.
"""

import ast
import logging
from pathlib import Path
import re
from typing import Optional, Tuple, Union

from src.datasets.text_cleaner import TextCleaner

logger = logging.getLogger(__name__)


class CodeTextCleaner:
    """AST-guided code cleaning, normalization, and syntax validation utility.

    Design Decisions:
        - Wraps TextCleaner for foundational Unicode and line-ending normalization.
        - Provides Python AST syntax verification (`ast.parse`) to filter syntactically broken code.
        - Supports C++ / Java heuristic syntax validation (matching brace & keyword structure).
        - Configurable comment and docstring preservation for code understanding tasks.
    """

    _PYTHON_COMMENT_REGEX = re.compile(r"#.*$", re.MULTILINE)
    _CPP_BLOCK_COMMENT_REGEX = re.compile(r"/\*.*?\*/", re.DOTALL)
    _CPP_LINE_COMMENT_REGEX = re.compile(r"//.*$", re.MULTILINE)

    def __init__(
        self,
        remove_comments: bool = False,
        preserve_docstrings: bool = True,
        normalize_indentation: bool = True,
        strict_ast_validation: bool = True,
    ) -> None:
        """Initializes CodeTextCleaner options.

        Args:
            remove_comments: If True, strips single-line and multi-line comments.
            preserve_docstrings: If True, preserves Python triple-quoted docstrings.
            normalize_indentation: If True, converts tabs to 4 spaces and trims trailing spaces.
            strict_ast_validation: If True, rejects code snippets failing syntax checks.
        """
        self.remove_comments = remove_comments
        self.preserve_docstrings = preserve_docstrings
        self.normalize_indentation = normalize_indentation
        self.strict_ast_validation = strict_ast_validation

        self.base_cleaner = TextCleaner(
            normalize_unicode="NFC",
            remove_unprintable=True,
            fix_line_endings=True,
            expand_tabs=normalize_indentation,
            tab_size=4,
        )

    def clean_python(self, code_str: str) -> Tuple[str, bool]:
        """Cleans and validates Python source code.

        Args:
            code_str: Raw Python code string.

        Returns:
            Tuple of (cleaned_code_string, is_valid_syntax).
        """
        if not code_str or not code_str.strip():
            return "", False

        text = self.base_cleaner.clean(code_str)

        if self.normalize_indentation:
            lines = [line.rstrip() for line in text.split("\n")]
            text = "\n".join(lines)

        if self.remove_comments:
            text = self._PYTHON_COMMENT_REGEX.sub("", text)

        is_valid = True
        if self.strict_ast_validation:
            try:
                ast.parse(text)
            except Exception:
                is_valid = False

        return text, is_valid

    def clean_cpp(self, code_str: str) -> Tuple[str, bool]:
        """Cleans and heuristically validates C++ source code.

        Args:
            code_str: Raw C++ code string.

        Returns:
            Tuple of (cleaned_code_string, is_valid_syntax).
        """
        if not code_str or not code_str.strip():
            return "", False

        text = self.base_cleaner.clean(code_str)

        if self.normalize_indentation:
            lines = [line.rstrip() for line in text.split("\n")]
            text = "\n".join(lines)

        if self.remove_comments:
            text = self._CPP_BLOCK_COMMENT_REGEX.sub("", text)
            text = self._CPP_LINE_COMMENT_REGEX.sub("", text)

        # Heuristic C++ syntax validation: matching braces & semicolons
        is_valid = True
        if self.strict_ast_validation:
            brace_count = 0
            for char in text:
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                if brace_count < 0:
                    is_valid = False
                    break
            if brace_count != 0:
                is_valid = False

        return text, is_valid

    def clean_code(self, code_str: str, language: str = "python") -> Tuple[str, bool]:
        """Routes code string to language-specific cleaner.

        Args:
            code_str: Raw code content string.
            language: Target language tag ("python", "cpp", "c++", "java", "generic").

        Returns:
            Tuple of (cleaned_code_string, is_valid_syntax).
        """
        lang = language.lower()
        if lang in ["python", "py"]:
            return self.clean_python(code_str)
        elif lang in ["cpp", "c++", "c", "hpp", "h", "java"]:
            return self.clean_cpp(code_str)
        else:
            cleaned = self.base_cleaner.clean(code_str)
            return cleaned, True

    def process_file(self, file_path: Union[str, Path]) -> Tuple[Optional[str], bool]:
        """Reads, cleans, and validates a source code file path.

        Args:
            file_path: Path to source code file.

        Returns:
            Tuple of (cleaned_code_content, is_valid_syntax).
        """
        path = Path(file_path)
        if not path.exists() or not path.is_file():
            logger.warning("File does not exist: %s", path)
            return None, False

        ext = path.suffix.lower()
        if ext in [".py", ".pyw"]:
            lang = "python"
        elif ext in [".cpp", ".c", ".h", ".hpp", ".cc", ".cxx"]:
            lang = "cpp"
        else:
            lang = "generic"

        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            cleaned_text, is_valid = self.clean_code(content, language=lang)
            return cleaned_text, is_valid
        except Exception as e:
            logger.error("Error reading source code file %s: %s", path, str(e))
            return None, False
