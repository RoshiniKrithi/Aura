"""Dataset Pipeline Factory for Aura LLM Pipeline.

Provides high-level factory API for orchestrating reader selection, validation,
text cleaning, tokenization, sequence building, caching, and dataset instantiation.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from src.datasets.batch_builder import BatchBuilder
from src.datasets.cache import DatasetCache
from src.datasets.exceptions import DatasetError, DatasetValidationError
from src.datasets.reader import FolderDatasetReader, StreamingReader, TextFileReader
from src.datasets.sequence_builder import SequenceBuilder
from src.datasets.splitter import DatasetSplitter
from src.datasets.text_cleaner import TextCleaner
from src.datasets.text_dataset import AuraTextDataset
from src.datasets.validator import DatasetValidator
from src.utils.config import DatasetConfig

logger = logging.getLogger(__name__)


class DatasetFactory:
    """Central factory for constructing and executing Aura data pipelines.

    Design Decisions:
        - Decouples component creation (Reader, Validator, Cleaner, Tokenizer, Cache, Splitter).
        - Fully integrated with DatasetConfig parameters.
    """

    @classmethod
    def create_dataset_from_text(
        cls,
        text: str,
        tokenizer: Any,
        config: DatasetConfig | None = None,
        source_name: str = "custom_text",
    ) -> AuraTextDataset:
        """Constructs an AuraTextDataset directly from a raw or cleaned text string.

        Args:
            text: Raw input corpus string.
            tokenizer: Instantiated tokenizer object.
            config: DatasetConfig container.
            source_name: String label for caching and identification.

        Returns:
            Instantiated AuraTextDataset object.
        """
        cfg = config or DatasetConfig()

        cleaner = TextCleaner(
            normalize_unicode="NFC",
            remove_unprintable=True,
            fix_line_endings=True,
        )
        cleaned_text = cleaner.clean(text)

        cache = DatasetCache(
            cache_dir=cfg.cache.cache_dir, enabled=cfg.cache.enabled
        )
        cache_config = {
            "window_size": cfg.sequence.window_size,
            "stride": cfg.sequence.stride,
            "vocab_size": getattr(tokenizer, "vocab_size", None),
        }

        # Check Cache
        cached_tensors = cache.get(source_name, cache_config)
        if cached_tensors is not None:
            x_t, y_t = cached_tensors
            return AuraTextDataset(
                x_tensor=x_t,
                y_tensor=y_t,
                window_size=cfg.sequence.window_size,
                name=source_name,
            )

        # Tokenize
        token_ids = tokenizer.encode(cleaned_text)

        # Slice Sequences
        builder = SequenceBuilder(
            window_size=cfg.sequence.window_size,
            stride=cfg.sequence.stride,
            drop_last=True,
        )
        x_t, y_t = builder.build_sequences(token_ids)

        # Save Cache
        if cfg.cache.enabled and x_t.size(0) > 0:
            cache.save(source_name, cache_config, x_t, y_t)

        return AuraTextDataset(
            x_tensor=x_t,
            y_tensor=y_t,
            window_size=cfg.sequence.window_size,
            name=source_name,
        )

    @classmethod
    def create_dataset_from_file(
        cls,
        file_path: Union[Path, str],
        tokenizer: Any,
        config: DatasetConfig | None = None,
    ) -> AuraTextDataset:
        """Constructs an AuraTextDataset from a target text file.

        Args:
            file_path: Path to target file.
            tokenizer: Tokenizer instance.
            config: DatasetConfig container.

        Returns:
            Instantiated AuraTextDataset.
        """
        cfg = config or DatasetConfig()
        path = Path(file_path).resolve()

        # Validate
        validator = DatasetValidator(cfg.validation)
        val_result = validator.validate_file(path)
        if not val_result.is_valid:
            raise DatasetValidationError(
                f"File validation failed for {path}: {val_result.errors}"
            )

        # Read
        reader = TextFileReader(path, encoding=cfg.encoding)
        raw_text = reader.read_all()

        return cls.create_dataset_from_text(
            text=raw_text,
            tokenizer=tokenizer,
            config=cfg,
            source_name=path.name,
        )

    @classmethod
    def build_pipeline(
        cls,
        source_path: Union[Path, str],
        tokenizer: Any,
        config: DatasetConfig | None = None,
    ) -> Tuple[AuraTextDataset, AuraTextDataset, AuraTextDataset]:
        """Runs complete end-to-end dataset pipeline.

        Reads source -> Validates -> Cleans -> Tokenizes -> Sequence Slices ->
        Caches -> Splits into (Train, Val, Test) datasets.

        Args:
            source_path: Path to text file or folder.
            tokenizer: Tokenizer instance.
            config: DatasetConfig container.

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset).
        """
        cfg = config or DatasetConfig()
        path = Path(source_path).resolve()

        if path.is_file():
            full_ds = cls.create_dataset_from_file(
                path, tokenizer=tokenizer, config=cfg
            )
        elif path.is_dir():
            reader = FolderDatasetReader(
                path, pattern=cfg.file_pattern, encoding=cfg.encoding
            )
            val_result = DatasetValidator(cfg.validation).validate_files(
                reader.file_paths
            )
            if not val_result.is_valid:
                raise DatasetValidationError(
                    f"Folder dataset validation failed: {val_result.errors}"
                )
            raw_text = reader.read_all()
            full_ds = cls.create_dataset_from_text(
                raw_text, tokenizer=tokenizer, config=cfg, source_name=path.name
            )
        else:
            raise DatasetError(f"Invalid dataset source path: {path}")

        # Split
        splitter = DatasetSplitter(cfg.split)
        return splitter.split_dataset(full_ds)
