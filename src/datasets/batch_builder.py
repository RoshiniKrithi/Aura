"""PyTorch DataLoader Factory and Mini-Batch Builder for Aura LLM Pipeline.

Instantiates PyTorch DataLoaders configured with custom collate functions,
pin_memory options, multi-process workers, and deterministic sampling.
"""

import logging
from typing import Union
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

from src.datasets.collate import CollateFunction
from src.utils.config import BatchConfig

logger = logging.getLogger(__name__)


class BatchBuilder:
    """Factory for instantiating PyTorch DataLoaders for Aura datasets.

    Design Decisions:
        - Integrates CollateFunction for dynamic sequence padding.
        - Supports pin_memory for fast GPU host-to-device transfers.
        - Handles both map-style Datasets and IterableDatasets.
    """

    def __init__(self, config: BatchConfig | None = None) -> None:
        """Initializes batch builder.

        Args:
            config: BatchConfig containing batch_size, shuffle, pin_memory, num_workers, drop_last.
        """
        self.config = config or BatchConfig()
        self.collate_fn = CollateFunction(pad_token_id=self.config.pad_token_id)

    def build_dataloader(
        self, dataset: Union[Dataset, IterableDataset]
    ) -> DataLoader:
        """Creates configured PyTorch DataLoader.

        Args:
            dataset: PyTorch Dataset or IterableDataset.

        Returns:
            Instantiated PyTorch DataLoader instance.
        """
        is_iterable = isinstance(dataset, IterableDataset)

        # IterableDataset does not support shuffle=True or sampler in DataLoader
        shuffle_flag = self.config.shuffle if not is_iterable else False

        loader = DataLoader(
            dataset=dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle_flag,
            drop_last=self.config.drop_last if not is_iterable else False,
            collate_fn=self.collate_fn,
            pin_memory=self.config.pin_memory,
            num_workers=self.config.num_workers,
        )

        logger.info(
            "Created DataLoader: BatchSize=%d, Shuffle=%s, PinMemory=%s, NumWorkers=%d",
            self.config.batch_size,
            shuffle_flag,
            self.config.pin_memory,
            self.config.num_workers,
        )

        return loader
