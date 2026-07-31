"""Strongly-typed configuration dataclasses for Aura architecture.

Provides structured validation and explicit typing for all hyperparameter,
system, hardware, logging, dataset, training, and inference settings.
"""

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass(frozen=True)
class SystemConfig:
    """System and hardware execution settings."""

    seed: int = 42
    device: str = "auto"
    num_workers: int = 4
    pin_memory: bool = True
    deterministic: bool = True


@dataclass(frozen=True)
class LoggingConfig:
    """Logging and experiment monitoring settings."""

    level: str = "INFO"
    file_logging: bool = True
    log_dir: str = "outputs/logs"


@dataclass(frozen=True)
class ModelConfig:
    """GPT-style Transformer architecture hyperparameters."""

    name: str = "aura-gpt-base"
    vocab_size: int = 50257
    max_sequence_length: int = 1024
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    d_ff: int = 3072
    dropout: float = 0.1
    bias: bool = False
    initializer_range: float = 0.02

    @property
    def head_dim(self) -> int:
        """Calculates dimension per attention head."""
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        return self.d_model // self.n_heads


@dataclass(frozen=True)
class TokenizerConfig:
    """Tokenizer configuration parameters."""

    vocab_size: int = 50257
    special_tokens: Dict[str, str] = field(
        default_factory=lambda: {
            "pad_token": "<|pad|>",
            "unk_token": "<|unk|>",
            "bos_token": "<|startoftext|>",
            "eos_token": "<|endoftext|>",
        }
    )


@dataclass(frozen=True)
class ValidationConfig:
    """Dataset validation criteria."""

    min_char_count: int = 10
    max_non_printable_ratio: float = 0.1
    check_utf8: bool = True
    check_duplicates: bool = True
    max_file_size_mb: float = 1024.0


@dataclass(frozen=True)
class SplitConfig:
    """Train/Val/Test dataset splitting hyperparameters."""

    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42
    shuffle: bool = True


@dataclass(frozen=True)
class BatchConfig:
    """DataLoader and mini-batching parameters."""

    batch_size: int = 16
    drop_last: bool = False
    pad_token_id: int = 0
    shuffle: bool = True
    pin_memory: bool = True
    num_workers: int = 0


@dataclass(frozen=True)
class SequenceConfig:
    """Sliding context window sequence generation parameters."""

    window_size: int = 64
    stride: int = 64
    add_bos: bool = True
    add_eos: bool = True


@dataclass(frozen=True)
class CacheConfig:
    """Dataset disk caching configuration."""

    enabled: bool = True
    cache_dir: str = "data/cache"
    compress: bool = False


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset split, directory, and pipeline configuration."""

    dataset_name: str = "tiny_shakespeare"
    dataset_dir: str = "data"
    source_type: str = "text"  # "text", "folder", "multi_file", "streaming"
    file_pattern: str = "*.txt"
    encoding: str = "utf-8"
    train_split: str = "train"
    val_split: str = "validation"
    test_split: str = "test"
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    batch: BatchConfig = field(default_factory=BatchConfig)
    sequence: SequenceConfig = field(default_factory=SequenceConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)


@dataclass(frozen=True)
class TrainingConfig:
    """Training optimization and loop hyperparameters."""

    batch_size: int = 16
    gradient_accumulation_steps: int = 4
    learning_rate: float = 6.0e-4
    min_learning_rate: float = 6.0e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_steps: int = 2000
    max_steps: int = 100000
    eval_interval: int = 500
    save_interval: int = 1000
    checkpoint_dir: str = "checkpoints"


@dataclass(frozen=True)
class InferenceConfig:
    """Inference and generation decoding options."""

    max_new_tokens: int = 512
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9


from src.attention.config import AttentionConfig
from src.embeddings.config import EmbeddingConfig
from src.embeddings.position_config import PositionEmbeddingConfig
from src.ffn.config import FeedForwardConfig
from src.normalization.config import LayerNormConfig


@dataclass(frozen=True)
class AppConfig:
    """Root configuration aggregating all Aura system modules."""

    system: SystemConfig = field(default_factory=SystemConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    position_embedding: PositionEmbeddingConfig = field(default_factory=PositionEmbeddingConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    ffn: FeedForwardConfig = field(default_factory=FeedForwardConfig)
    layernorm: LayerNormConfig = field(default_factory=LayerNormConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)


