"""Comprehensive Dataset Validator for Aura LLM Pipeline.

Validates dataset sources against corruption, empty files, missing paths,
invalid UTF-8 encodings, high non-printable byte density, and duplicate content.
"""

from dataclasses import dataclass, field
import hashlib
import logging
from pathlib import Path
import re
from typing import Dict, List, Union

from src.datasets.exceptions import DatasetValidationError
from src.utils.config import ValidationConfig

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Detailed metadata output resulting from dataset validation checks."""

    is_valid: bool
    total_files: int = 0
    valid_files: int = 0
    total_bytes: int = 0
    total_characters: int = 0
    duplicate_files: List[Path] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    file_hashes: Dict[str, Path] = field(default_factory=dict)


class DatasetValidator:
    """Validates data files and directories before tokenization and training.

    Design Decisions:
        - Strict UTF-8 verification to catch corrupted binaries before downstream tokenization crashes.
        - Content hash-based duplicate detection (SHA-256) to eliminate data contamination.
        - Non-printable byte density checks to reject misnamed binary or object files.

    Time Complexity:
        O(F * N) where F is the number of files and N is the average file byte length.

    Space Complexity:
        O(F) to store file paths and content hashes in memory.
    """

    _UNPRINTABLE_PATTERN = re.compile(rb"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]")

    def __init__(self, config: ValidationConfig | None = None) -> None:
        """Initializes validator with validation criteria configuration."""
        self.config = config or ValidationConfig()

    def validate_file(self, file_path: Union[Path, str]) -> ValidationResult:
        """Validates a single target file.

        Args:
            file_path: Path to the target file.

        Returns:
            ValidationResult summary container.
        """
        return self.validate_files([file_path])

    def validate_files(
        self, file_paths: List[Union[Path, str]]
    ) -> ValidationResult:
        """Validates a list of input file paths.

        Args:
            file_paths: List of file paths to inspect.

        Returns:
            ValidationResult summary container.
        """
        result = ValidationResult(is_valid=True, total_files=len(file_paths))

        if not file_paths:
            result.is_valid = False
            result.errors.append("No file paths provided for validation.")
            return result

        seen_hashes: Dict[str, Path] = {}

        for fp in file_paths:
            path = Path(fp).resolve()

            # 1. Existence Check
            if not path.exists():
                result.is_valid = False
                result.errors.append(f"Missing file: {path}")
                continue

            if not path.is_file():
                result.is_valid = False
                result.errors.append(f"Target is not a regular file: {path}")
                continue

            # 2. File Size & Empty Check
            file_bytes = path.stat().st_size
            result.total_bytes += file_bytes

            if file_bytes == 0:
                result.is_valid = False
                result.errors.append(f"Empty (0-byte) file detected: {path}")
                continue

            # 3. File Read & Encoding Check (UTF-8)
            try:
                with open(path, "rb") as f:
                    raw_bytes = f.read()
            except Exception as e:
                result.is_valid = False
                result.errors.append(
                    f"Corrupted or unreadable file {path}: {str(e)}"
                )
                continue

            # 4. Duplicate File Check via SHA-256
            if self.config.check_duplicates:
                file_hash = hashlib.sha256(raw_bytes).hexdigest()
                if file_hash in seen_hashes:
                    result.duplicate_files.append(path)
                    result.warnings.append(
                        f"Duplicate content found: '{path}' matches '{seen_hashes[file_hash]}'"
                    )
                else:
                    seen_hashes[file_hash] = path
                    result.file_hashes[file_hash] = path

            # 5. UTF-8 Byte Validation
            if self.config.check_utf8:
                try:
                    text_str = raw_bytes.decode("utf-8")
                except UnicodeDecodeError as e:
                    result.is_valid = False
                    result.errors.append(
                        f"Invalid UTF-8 encoding in file {path}: {str(e)}"
                    )
                    continue

                char_count = len(text_str)
                result.total_characters += char_count

                # 6. Minimum Character Length Check
                if char_count < self.config.min_char_count:
                    result.is_valid = False
                    result.errors.append(
                        f"File {path} char count ({char_count}) is below minimum threshold ({self.config.min_char_count})."
                    )
                    continue

            # 7. Non-printable Byte Ratio Check
            if file_bytes > 0:
                unprintable_bytes = len(
                    self._UNPRINTABLE_PATTERN.findall(raw_bytes)
                )
                unprintable_ratio = unprintable_bytes / file_bytes
                if unprintable_ratio > self.config.max_non_printable_ratio:
                    result.is_valid = False
                    result.errors.append(
                        f"File {path} contains excessive non-printable bytes ({unprintable_ratio:.2%})."
                    )
                    continue

            result.valid_files += 1

        if result.errors:
            result.is_valid = False

        logger.info(
            "Validation finished: %d/%d valid files. Valid=%s",
            result.valid_files,
            result.total_files,
            result.is_valid,
        )
        return result

    def validate_directory(
        self, dir_path: Union[Path, str], pattern: str = "*.txt"
    ) -> ValidationResult:
        """Glob-validates all matching files within a directory.

        Args:
            dir_path: Directory path to scan.
            pattern: Glob pattern to filter files.

        Returns:
            ValidationResult summary container.
        """
        path = Path(dir_path).resolve()
        if not path.exists() or not path.is_dir():
            result = ValidationResult(is_valid=False)
            result.errors.append(
                f"Directory does not exist or is not a directory: {path}"
            )
            return result

        matching_files = sorted(list(path.glob(pattern)))
        if not matching_files:
            result = ValidationResult(is_valid=False)
            result.errors.append(
                f"No files matching pattern '{pattern}' found in directory: {path}"
            )
            return result

        return self.validate_files(matching_files)
