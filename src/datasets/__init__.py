"""Aura Dataset Pipeline Module.

WHY THIS FILE EXISTS:
    Root entry point for dataset loading, validation, cleaning, token sequence generation,
    caching, train/val/test splitting, batch building, and dynamic collation.

WHY THIS IMPLEMENTATION WAS CHOSEN:
    Centralized module exports enable downstream model training loops and evaluation engines
    to import dataset abstractions directly (`from src.datasets import DatasetFactory, AuraTextDataset`).
"""

from src.datasets.base_dataset import BaseDataset
from src.datasets.batch_builder import BatchBuilder
from src.datasets.cache import DatasetCache
from src.datasets.collate import BatchStatistics, CollateFunction
from src.datasets.exceptions import (
    DatasetCacheError,
    DatasetConfigError,
    DatasetError,
    DatasetReadError,
    DatasetValidationError,
    SequenceGenerationError,
)
from src.datasets.factory import DatasetFactory
from src.datasets.reader import (
    DatasetReader,
    FolderDatasetReader,
    StreamingReader,
    TextFileReader,
)
from src.datasets.sequence_builder import SequenceBuilder
from src.datasets.splitter import DatasetSplitter
from src.datasets.statistics import CorpusStats, DatasetStatistics
from src.datasets.streaming_dataset import AuraStreamingDataset
from src.datasets.text_cleaner import TextCleaner
from src.datasets.text_dataset import AuraTextDataset
from src.datasets.validator import DatasetValidator, ValidationResult
from src.datasets.vocab_inspector import InspectionReport, VocabularyInspector

__all__ = [
    "BaseDataset",
    "AuraTextDataset",
    "AuraStreamingDataset",
    "DatasetReader",
    "TextFileReader",
    "FolderDatasetReader",
    "StreamingReader",
    "TextCleaner",
    "DatasetValidator",
    "ValidationResult",
    "DatasetStatistics",
    "CorpusStats",
    "VocabularyInspector",
    "InspectionReport",
    "SequenceBuilder",
    "DatasetCache",
    "DatasetSplitter",
    "BatchBuilder",
    "CollateFunction",
    "BatchStatistics",
    "DatasetFactory",
    "DatasetError",
    "DatasetValidationError",
    "DatasetReadError",
    "DatasetCacheError",
    "DatasetConfigError",
    "SequenceGenerationError",
]
