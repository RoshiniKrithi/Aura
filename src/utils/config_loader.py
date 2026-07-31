"""Configuration loading and validation system for Aura.

Parses YAML configurations, validates parameters against dataclasses,
and allows programmatic overrides.
"""

import logging
from pathlib import Path
from typing import Any, Dict
import yaml

from src.utils.config import (
    AppConfig,
    DatasetConfig,
    InferenceConfig,
    LoggingConfig,
    ModelConfig,
    SystemConfig,
    TokenizerConfig,
    TrainingConfig,
)
from src.utils.paths import project_paths

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Loads and validates YAML configuration files into strongly-typed dataclasses."""

    @staticmethod
    def load_yaml(file_path: Path | str) -> Dict[str, Any]:
        """Loads raw dictionary content from a YAML file.

        Args:
            file_path: Path to YAML configuration file.

        Returns:
            Dictionary representation of configuration.

        Raises:
            FileNotFoundError: If the config file does not exist.
            yaml.YAMLError: If parsing fails.
        """
        path = Path(file_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Configuration file not found at: {path}")

        with open(path, "r", encoding="utf-8") as f:
            content = yaml.safe_load(f) or {}

        logger.info("Configuration YAML successfully loaded from: %s", path)
        return content

    @classmethod
    def from_yaml(cls, file_path: Path | str | None = None) -> AppConfig:
        """Parses a YAML configuration file into a strongly-typed AppConfig object.

        Args:
            file_path: Optional path to YAML file. Defaults to configs/config.yaml.

        Returns:
            Instantiated AppConfig container.
        """
        if file_path is None:
            file_path = project_paths.default_config_path

        raw_config = cls.load_yaml(file_path)

        from src.utils.config import (
            BatchConfig,
            CacheConfig,
            SequenceConfig,
            SplitConfig,
            ValidationConfig,
        )

        dataset_dict = raw_config.get("dataset", {}).copy()
        validation_cfg = ValidationConfig(**dataset_dict.pop("validation", {}))
        split_cfg = SplitConfig(**dataset_dict.pop("split", {}))
        batch_cfg = BatchConfig(**dataset_dict.pop("batch", {}))
        sequence_cfg = SequenceConfig(**dataset_dict.pop("sequence", {}))
        cache_cfg = CacheConfig(**dataset_dict.pop("cache", {}))

        dataset_cfg = DatasetConfig(
            validation=validation_cfg,
            split=split_cfg,
            batch=batch_cfg,
            sequence=sequence_cfg,
            cache=cache_cfg,
            **dataset_dict,
        )

        from src.attention.config import AttentionConfig
        from src.embeddings.config import EmbeddingConfig
        from src.embeddings.position_config import PositionEmbeddingConfig
        from src.ffn.config import FeedForwardConfig
        from src.normalization.config import LayerNormConfig

        system_cfg = SystemConfig(**raw_config.get("system", {}))
        logging_cfg = LoggingConfig(**raw_config.get("logging", {}))
        model_cfg = ModelConfig(**raw_config.get("model", {}))
        tokenizer_cfg = TokenizerConfig(**raw_config.get("tokenizer", {}))
        embedding_cfg = EmbeddingConfig(**raw_config.get("embedding", {}))
        position_embedding_cfg = PositionEmbeddingConfig(
            **raw_config.get("position_embedding", {})
        )
        attention_cfg = AttentionConfig(**raw_config.get("attention", {}))
        ffn_cfg = FeedForwardConfig(**raw_config.get("ffn", {}))
        layernorm_cfg = LayerNormConfig(**raw_config.get("layernorm", {}))
        training_cfg = TrainingConfig(**raw_config.get("training", {}))
        inference_cfg = InferenceConfig(**raw_config.get("inference", {}))

        return AppConfig(
            system=system_cfg,
            logging=logging_cfg,
            model=model_cfg,
            tokenizer=tokenizer_cfg,
            embedding=embedding_cfg,
            position_embedding=position_embedding_cfg,
            attention=attention_cfg,
            ffn=ffn_cfg,
            layernorm=layernorm_cfg,
            dataset=dataset_cfg,
            training=training_cfg,
            inference=inference_cfg,
        )


def load_config(file_path: Path | str | None = None) -> AppConfig:
    """Convenience helper to load default or specified AppConfig.

    Args:
        file_path: Optional configuration file path.

    Returns:
        Validated AppConfig object.
    """
    return ConfigLoader.from_yaml(file_path)
