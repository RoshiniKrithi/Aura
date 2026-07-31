"""Deterministic Dataset Splitter for Aura LLM Pipeline.

Splits dataset tensors, token arrays, or file paths into Train / Validation / Test
subsets using fixed random seeds to ensure strict experiment reproducibility.
"""

import logging
from pathlib import Path
import random
from typing import List, Sequence, Tuple, Union
import torch

from src.datasets.base_dataset import BaseDataset
from src.datasets.exceptions import DatasetConfigError
from src.datasets.text_dataset import AuraTextDataset
from src.utils.config import SplitConfig

logger = logging.getLogger(__name__)


class DatasetSplitter:
    """Splits dataset sequences or file paths deterministically into train, val, and test subsets.

    Design Decisions:
        - Strict random seed isolation for reproducible dataset splits.
        - Supports both dataset sequence splitting and file-level splitting to prevent data leakage.

    Time Complexity:
        O(N) index shuffle and slicing.

    Space Complexity:
        O(N) memory space for sliced sub-tensors or file lists.
    """

    def __init__(self, config: SplitConfig | None = None) -> None:
        """Initializes splitter options.

        Args:
            config: SplitConfig specifying train_ratio, val_ratio, test_ratio, seed, shuffle.
        """
        self.config = config or SplitConfig()

        total = (
            self.config.train_ratio
            + self.config.val_ratio
            + self.config.test_ratio
        )
        if not (0.99 <= total <= 1.01):
            raise DatasetConfigError(
                f"Split ratios must sum to 1.0. Got {self.config.train_ratio} + {self.config.val_ratio} + {self.config.test_ratio} = {total}"
            )

    def split_dataset(
        self, dataset: AuraTextDataset
    ) -> Tuple[AuraTextDataset, AuraTextDataset, AuraTextDataset]:
        """Splits an in-memory AuraTextDataset into (train, val, test) datasets.

        Args:
            dataset: Parent AuraTextDataset.

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset).
        """
        n = len(dataset)
        indices = list(range(n))

        if self.config.shuffle:
            rng = random.Random(self.config.seed)
            rng.shuffle(indices)

        train_end = int(n * self.config.train_ratio)
        val_end = train_end + int(n * self.config.val_ratio)

        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]

        train_x, train_y = dataset.x_tensor[train_idx], dataset.y_tensor[train_idx]
        val_x, val_y = dataset.x_tensor[val_idx], dataset.y_tensor[val_idx]
        test_x, test_y = dataset.x_tensor[test_idx], dataset.y_tensor[test_idx]

        train_ds = AuraTextDataset(
            x_tensor=train_x,
            y_tensor=train_y,
            window_size=dataset.window_size,
            name=f"{dataset.name}_train",
        )
        val_ds = AuraTextDataset(
            x_tensor=val_x,
            y_tensor=val_y,
            window_size=dataset.window_size,
            name=f"{dataset.name}_val",
        )
        test_ds = AuraTextDataset(
            x_tensor=test_x,
            y_tensor=test_y,
            window_size=dataset.window_size,
            name=f"{dataset.name}_test",
        )

        logger.info(
            "Split dataset of length %d into Train: %d, Val: %d, Test: %d (Seed: %d)",
            n,
            len(train_ds),
            len(val_ds),
            len(test_ds),
            self.config.seed,
        )

        return train_ds, val_ds, test_ds

    def split_files(
        self, file_paths: List[Union[Path, str]]
    ) -> Tuple[List[Path], List[Path], List[Path]]:
        """Splits raw file paths into train, val, and test path lists (prevents data leakage).

        Args:
            file_paths: List of source file paths.

        Returns:
            Tuple of (train_paths, val_paths, test_paths).
        """
        paths = [Path(fp).resolve() for fp in file_paths]
        n = len(paths)

        if self.config.shuffle:
            rng = random.Random(self.config.seed)
            rng.shuffle(paths)

        train_end = int(n * self.config.train_ratio)
        val_end = train_end + int(n * self.config.val_ratio)

        train_paths = paths[:train_end]
        val_paths = paths[train_end:val_end]
        test_paths = paths[val_end:]

        logger.info(
            "Split %d files into Train files: %d, Val files: %d, Test files: %d",
            n,
            len(train_paths),
            len(val_paths),
            len(test_paths),
        )

        return train_paths, val_paths, test_paths
